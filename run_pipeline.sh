#!/usr/bin/env bash
set -uo pipefail

METHOD=${1:-}
case "$METHOD" in
    unsteered|ast|cast|alphasteer|adasteer)
        ;;
    "")
        echo "Usage: $0 <unsteered|ast|cast|alphasteer|adasteer>" >&2
        exit 1
        ;;
    *)
        echo "Unknown method: $METHOD" >&2
        echo "Allowed methods: unsteered, ast, cast, alphasteer, adasteer" >&2
        exit 1
        ;;
esac

CONFIG="benchmark_configs/${METHOD}.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

case "$METHOD" in
    ast|cast)
        TRAIN_MODULE="src.patching.train.train_ast_cast"
        ;;
    unsteered)
        TRAIN_MODULE=""
        ;;
    *)
        TRAIN_MODULE="src.patching.train.train_${METHOD}"
        ;;
esac

MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "google/gemma-3-4b-it"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "meta-llama/Llama-3.1-8B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "google/gemma-3-12b-it"
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================================"
    echo "  MODEL: $MODEL"
    echo "  METHOD: $METHOD"
    echo "========================================================"

    failed=0

    run_step() {
        # Step label
        local step=$1; shift
        echo "[${step}] $*"
        if ! "$@"; then
            echo "[FAILED at step ${step}] $MODEL — skipping remaining steps for this model" >&2
            failed=1
        fi
    }

    if [[ "$METHOD" == "unsteered" ]]; then
        [ $failed -eq 0 ] && run_step "1/2 Benchmark inference" python -m src.patching.inference.llm_inference_steer_refusal          --model "$MODEL" --config "$CONFIG" --no-validation-selection
        [ $failed -eq 0 ] && run_step "2/2 Benchmark labeling"  python -m src.patching.inference.label_benchmark_with_wildguard       --model "$MODEL" --config "$CONFIG"
    else
        [ $failed -eq 0 ] && run_step "1/5 Training"             python -m "$TRAIN_MODULE"                                             --model "$MODEL" --config "$CONFIG"
        [ $failed -eq 0 ] && run_step "2/5 Validation generation" python -m src.patching.validation.run_validation                       --model "$MODEL" --config "$CONFIG"
        [ $failed -eq 0 ] && run_step "3/5 Validation labeling"   python -m src.patching.validation.label_validation_with_wildguard      --model "$MODEL" --config "$CONFIG"
        [ $failed -eq 0 ] && run_step "4/5 Benchmark inference"   python -m src.patching.inference.llm_inference_steer_refusal           --model "$MODEL" --config "$CONFIG"
        [ $failed -eq 0 ] && run_step "5/5 Benchmark labeling"    python -m src.patching.inference.label_benchmark_with_wildguard        --model "$MODEL" --config "$CONFIG"
    fi

    if [ $failed -eq 0 ]; then
        echo "[DONE] $MODEL"
    fi
done

echo ""
echo "All models complete."
