import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wildguard import load_wildguard

from src.patching.inference.benchmark_utils import (
    default_benchmark_output_path,
    print_category_metrics,
    save_category_metrics,
)
from src.patching.utils import load_benchmark_method_config_from_path


def build_parser() -> argparse.ArgumentParser:
    # This is the post-processing step for benchmark inference: it loads saved generations, labels them with WildGuard and writes benchmark metrics.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the benchmark YAML config.",
    )
    return parser


def _attach_wildguard_labels(outputs: list[dict]) -> list[dict]:
    # WildGuard is used after generation to assign refusal/compliance labels to each prompt/response pair in the saved benchmark outputs.

    print("Loading Wildguard model...")
    wildguard = load_wildguard()

    print("Processing items ...")
    # Convert the saved outputs to the sample format expected by WildGuard.
    samples = [
        {
            "id": item["id"],
            "prompt": item["prompt"],
            "response": item["answer"],
        }
        for item in outputs
    ]

    wildguard_results = wildguard.classify(samples)
    if len(wildguard_results) != len(outputs):
        raise ValueError(
            "WildGuard returned a different number of results than the number of generated outputs."
        )

    # Merge WildGuard labels back into the existing output rows while keeping the original prompt/answer content intact.
    # Override response_refusal for empty answers: the model emitted only an EOS token, which is unambiguously a refusal.
    for row, label in zip(outputs, wildguard_results):
        for key, value in label.items():
            if key in {"id", "prompt", "response"}:
                continue
            row[key] = value
        if row["answer"] == "":# or row["answer"] == ".":
            row["response_refusal"] = "refusal"
            row["response_harmfulness"] = "unharmful"
            
        if row["response_refusal"] == None:
            row["response_refusal"] = "refusal"
        
        if row["response_harmfulness"] == None:
            row["response_harmfulness"] = "unharmful"
            
    return outputs


def main() -> None:
    # Load the benchmark config only to determine which backend's saved outputs should be labeled.
    args = build_parser().parse_args()
    method_config = load_benchmark_method_config_from_path(args.config)
    backend = method_config.method_name

    # Benchmark inference must have been run already, because this script only labels and scores existing outputs.
    out_path = default_benchmark_output_path(
        model_name=args.model,
        backend=backend,
    )
    if not out_path.exists():
        raise FileNotFoundError(
            f"Cannot run WildGuard labeling because the benchmark output JSON does not exist: {out_path}"
        )

    print(f"[INFO] Benchmark config: {method_config.path.resolve()}")
    print(f"[INFO] Loading existing benchmark output for labeling: {out_path.resolve()}")
    outputs = json.loads(out_path.read_text(encoding="utf-8"))

    # Label the saved generations, then write to a temp file before replacing
    # the original so a mid-write crash cannot corrupt the raw generations.
    print("[INFO] Running WildGuard on generated answers...")
    outputs = _attach_wildguard_labels(outputs)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)
    print(f"[DONE] Saved WildGuard labels to: {out_path.resolve()}")

    # Print and persist the aggregate benchmark metrics computed from the new refusal/compliance labels.
    print_category_metrics(outputs)
    save_category_metrics(outputs, out_path)


if __name__ == "__main__":
    main()
