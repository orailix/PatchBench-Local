#!/usr/bin/env bash
# Data creation pipeline: RedBench export → LLM inference → WildGuard → ELO
# Usage:
#   ./run_data_pipeline.sh        # full pipeline via SLURM
#   ./run_data_pipeline.sh --test # test mode: 1 small model, 10 samples (local, no SLURM)

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }
TEST_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) TEST_MODE=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

RAW_DATA_FILE="../PatchBench/data_processing/raw_data/redbench_filtered.json"

# ── Test mode: run everything locally on one small model ──────────────────────
if [[ "$TEST_MODE" == true ]]; then
    TEST_JSON="../PatchBench/data_processing/raw_data/redbench_test.json"
    log "Extracting target unsafe prompts (103, 668, 856, 958)..."
    python -c '
import json
data = json.load(open("../PatchBench/data_processing/raw_data/redbench_filtered.json"))
target_ids = {103, 668, 856, 958}
test_data = data[:10] + [r for r in data if r.get("id") in target_ids]
json.dump(test_data, open("'"$TEST_JSON"'", "w"), indent=2)
print(f"Created {len(test_data)} test prompts in '"$TEST_JSON"'")
'
    log "Running inference on target test prompts..."
    python -m src.data.llm_inference --model "Qwen/Qwen2.5-3B-Instruct" --batch-size 4 --in-json "$TEST_JSON"
    python -m src.data.wildguard_label
    python -m src.data.elo_ranking
    log "Pipeline complete (test mode)."
    exit 0
fi

# ── Export RedBench filtered data (full pipeline) ─────────────────────────────
if [[ ! -f "$RAW_DATA_FILE" ]]; then
    log "[1/4] Exporting RedBench filtered data..."
    python -m src.data.export_redbench_filtered
else
    log "[1/4] $RAW_DATA_FILE already exists, skipping export step."
fi

# ── LLM inference jobs via SLURM ───────────────────────────────
log "[2/4] Submitting LLM inference jobs..."
INFERENCE_JIDS=()
for slurm_file in slurm/llm_inference/*.slurm; do
    JID=$(sbatch --parsable "$slurm_file")
    INFERENCE_JIDS+=("$JID")
    log "  Submitted $(basename "$slurm_file") → job $JID"
done

if [[ ${#INFERENCE_JIDS[@]} -eq 0 ]]; then
    echo "No inference slurm scripts found in slurm/llm_inference/. Aborting." >&2
    exit 1
fi

INFERENCE_DEP="afterok:$(IFS=:; echo "${INFERENCE_JIDS[*]}")"

# ── WildGuard labeling  (depends on inference) ────────────────
log "[3/4] Submitting WildGuard labeling..."
WG_JID=$(sbatch --parsable --dependency="$INFERENCE_DEP" slurm/wildguard/wildguard.slurm)
log "  Submitted wildguard → job $WG_JID"

# ── ELO ranking (depends on wildguard) ────────────────────────────────
log "[4/4] Submitting ELO ranking..."
ELO_JID=$(sbatch --parsable --dependency="afterok:${WG_JID}" slurm/elo/elo_ranking.slurm)
log "  Submitted elo → job $ELO_JID"

log "Pipeline submitted. Track with: squeue -u \$USER"
log "  Inference jobs : ${INFERENCE_JIDS[*]}"
log "  WildGuard job  : $WG_JID"
log "  ELO job        : $ELO_JID"
