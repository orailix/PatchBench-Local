#!/usr/bin/env bash

set -euo pipefail

# ─── Models ───────────────────────────────────────────────────────────────────
MODELS=(
    "google/gemma-3-4b-it"
    "Qwen/Qwen2.5-3B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "meta-llama/Llama-3.1-8B-Instruct"
)

METHODS=("alphasteer" "adasteer")

declare -A METHOD_CONFIG
METHOD_CONFIG["alphasteer"]="benchmark_configs/alphasteer.yaml"
METHOD_CONFIG["adasteer"]="benchmark_configs/adasteer.yaml"

declare -A TRAIN_MODULE
TRAIN_MODULE["alphasteer"]="src.patching.train.train_alphasteer"
TRAIN_MODULE["adasteer"]="src.patching.train.train_adasteer"

UNSTEERED_CONFIG="benchmark_configs/unsteered.yaml"
MMLU_OUTPUT_DIR="./mmlu_steered_output"
MMLU_BATCH_SIZE=4

# ─── Parse flags ──────────────────────────────────────────────────────────────
MMLU_ONLY=false
SINGLE_MODEL=""
SINGLE_METHOD=""
SKIP_UNSTEERED=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mmlu-only)      MMLU_ONLY=true; shift ;;
        --model)          SINGLE_MODEL="$2"; shift 2 ;;
        --method)         SINGLE_METHOD="$2"; shift 2 ;;
        --skip-unsteered) SKIP_UNSTEERED=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

[[ -n "$SINGLE_MODEL"  ]] && MODELS=("$SINGLE_MODEL")
[[ -n "$SINGLE_METHOD" ]] && METHODS=("$SINGLE_METHOD")

# ─── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

safe_name() { echo "$1" | sed 's#/#--#g'; }

unsteered_done() {
    local f="$MMLU_OUTPUT_DIR/$(safe_name "$1")_unsteered_mmlu_5shot_logprob.json"
    [[ -f "$f" ]]
}

method_done() {
    local f="$MMLU_OUTPUT_DIR/$(safe_name "$1")_${2}_mmlu_5shot_logprob.json"
    [[ -f "$f" ]]
}

training_done() {
    local model="$1" method="$2"
    case "$method" in
        alphasteer) [[ -f "vectors/alphasteer/${model}/steering_matrix.pt" ]] ;;
        adasteer)   [[ -f "vectors/adasteer/${model}/refusal.pt" ]]         ;;
        *)          return 1 ;;
    esac
}
validation_done() {
    local model_safe dir f
    model_safe=$(echo "$1" | sed 's#/#__#g')  
    case "$2" in
        alphasteer) dir="AlphaSteer" ;;
        adasteer)   dir="AdaSteer"   ;;
        *)          return 1 ;;
    esac
    f="analysis/validation/${dir}/${model_safe}__${2}__validation_report.json"
    [[ -f "$f" ]]
}

# ─── Main loop ────────────────────────────────────────────────────────────────
mkdir -p "$MMLU_OUTPUT_DIR"

for MODEL in "${MODELS[@]}"; do
    log "========================================================"
    log "Model: $MODEL"
    log "========================================================"

    # ── Unsteered baseline ────────────────────────────────────────────────────
    if [[ "$SKIP_UNSTEERED" == true ]]; then
        log "[unsteered] Skipped via --skip-unsteered"
    else
        if unsteered_done "$MODEL"; then
            log "[unsteered] Skipping (already done)"
        else
            log "[unsteered] Running MMLU baseline"
            python -m src.patching.inference.logprob_mmlu_eval \
                --model      "$MODEL" \
                --config     "$UNSTEERED_CONFIG" \
                --output-dir "$MMLU_OUTPUT_DIR" \
                --batch-size "$MMLU_BATCH_SIZE"
        fi
    fi

    # ── AlphaSteer / AdaSteer ─────────────────────────────────────────────────
    for METHOD in "${METHODS[@]}"; do
        log "--------------------------------------------------------"
        log "Method: $METHOD"
        log "--------------------------------------------------------"

        if method_done "$MODEL" "$METHOD"; then
            log "[$METHOD] Skipping (already done)"
            continue
        fi

        CONFIG="${METHOD_CONFIG[$METHOD]}"

        if [[ "$MMLU_ONLY" == false ]]; then
            # ── [1/4] Training ────────────────────────────────────────────────
            if training_done "$MODEL" "$METHOD"; then
                log "[1/4] Training skipped (artifacts already exist)"
            else
                log "[1/4] Training ($METHOD)"
                python -m "${TRAIN_MODULE[$METHOD]}" \
                    --model  "$MODEL" \
                    --config "$CONFIG"
            fi

            if validation_done "$MODEL" "$METHOD"; then
                log "[2/4] Validation skipped (report already exists)"
                log "[3/4] WildGuard labeling skipped"
            else
                log "[2/4] Validation sweep ($METHOD)"
                python -m src.patching.validation.run_validation \
                    --model  "$MODEL" \
                    --config "$CONFIG"

                log "[3/4] WildGuard labeling ($METHOD)"
                python -m src.patching.validation.label_validation_with_wildguard \
                    --model  "$MODEL" \
                    --config "$CONFIG"
            fi
        else
            log "[--mmlu-only] Skipping train/validation"
        fi

        # ── [4/4] MMLU eval ───────────────────────────────────────────────────
        log "[4/4] MMLU steered ($METHOD)"
        python -u -m src.patching.inference.logprob_mmlu_eval \
            --model      "$MODEL" \
            --config     "$CONFIG" \
            --output-dir "$MMLU_OUTPUT_DIR" \
            --batch-size "$MMLU_BATCH_SIZE"

        log "Done: $MODEL / $METHOD"
    done

    echo ""
done

# ─── Report ───────────────────────────────────────────────────────────────────
ALL_METHODS=("cast" "ast" "alphasteer" "adasteer")
REPORT_METHODS=()
for M in "${ALL_METHODS[@]}"; do
    for MDL in "${MODELS[@]}"; do
        if method_done "$MDL" "$M" 2>/dev/null; then
            REPORT_METHODS+=("$M")
            break
        fi
    done
done

log "Generating MMLU report (methods: ${REPORT_METHODS[*]})..."
python -m src.patching.inference.mmlu_report \
    --output-dir "$MMLU_OUTPUT_DIR" \
    --models     "${MODELS[@]}" \
    --methods    "${REPORT_METHODS[@]}"

log "Pipeline complete."
exit 0
