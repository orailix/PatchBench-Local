#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.utils import repo_to_latest_snapshot_dir, safe_name
from src.patching.steering import SteeringArtifacts, TrainingContext, get_steering_model
from src.patching.utils import (
    PromptTemplateSettings,
    load_benchmark_method_config_from_path,
)

ADA_VECTOR_DIR = REPO_ROOT / "vectors" / "alphasteer"

def build_parser() -> argparse.ArgumentParser:
    # Keep the CLI intentionally small: the benchmark YAML owns almost all training settings, and this entrypoint mainly selects the model and config.
    parser = argparse.ArgumentParser(
        description=(
            "Single AST/CAST training entrypoint driven by a benchmark YAML config. "
            "Training settings and optional path overrides come from the config."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the benchmark YAML config.",
    )
    return parser


def _build_training_context(
    model_name: str,
    method_name: str,
    prompt_settings: PromptTemplateSettings,
    dtype_name: str,
    snapshot_path: Path,
    training_defaults: dict[str, object],
) -> TrainingContext:
    method_kwargs = dict(training_defaults)
    method_kwargs["dtype"] = dtype_name
    
    # Prompt templating lives in `prompt_settings`, so drop duplicated values from `method_kwargs` before constructing the framework context object.
    for key in ("use_chat_template", "system_prompt", "add_generation_prompt"):
        method_kwargs.pop(key, None)

    return TrainingContext(
        method_name=method_name,
        model_name=model_name,
        snapshot_path=snapshot_path,
        artifacts_dir=ADA_VECTOR_DIR,
        prompt_settings=prompt_settings,
    )


def _print_artifact_summary(artifacts: SteeringArtifacts) -> None:
    # Print a compact deterministic summary after training completes so can easily see which files were produced.
    for name in sorted(artifacts.paths):
        print(f"[DONE] Saved {name} to: {artifacts.paths[name]}")


def main() -> None:
    # Load the benchmark config and reject non-AST/CAST methods early because this entrypoint is intentionally specialized to those two patching methods.
    args = build_parser().parse_args()
    method_config = load_benchmark_method_config_from_path(args.config)
    backend = method_config.method_name
    if backend not in {"alphasteer"}:
        raise ValueError(
            f"train_alphasteer.py only supports ALPHASTEER configs, but received method={backend!r}."
        )
    
    # Training defaults drive vectors creation for this entrypoint.
    training_defaults = method_config.fixed_params("training")
    dtype_name = str(training_defaults.get("dtype", "auto"))
    prompt_settings = PromptTemplateSettings(
        use_chat_template=bool(training_defaults.get("use_chat_template", True)),
        system_prompt=training_defaults.get("system_prompt", "You are a helpful assistant."),
        add_generation_prompt=bool(training_defaults.get("add_generation_prompt", False)),
    )

    snapshot_path = Path(repo_to_latest_snapshot_dir(args.model))
    
    # Convert the CLI/config state into the framework `TrainingContext`, then delegate the actual artifact computation to the selected steering backend.
    context = _build_training_context(
        model_name=args.model,
        method_name=backend,
        prompt_settings=prompt_settings,
        dtype_name=dtype_name,
        snapshot_path=snapshot_path,
        training_defaults=training_defaults,
    )

    steering = get_steering_model(backend)
    artifacts = steering.train(context)
    
    _print_artifact_summary(artifacts)

if __name__ == "__main__":
    main()
