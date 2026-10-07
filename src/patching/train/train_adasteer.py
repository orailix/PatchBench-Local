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
    get_adasteer_layers,
    load_benchmark_method_config_from_path,
)

ALPHA_VECTOR_DIR = REPO_ROOT / "vectors" / "alphasteer"

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
    alpha_layer: int = 9,
    calibration_lambdas: list[float] | None = None,
) -> TrainingContext:
    method_kwargs = dict(training_defaults)
    
    method_kwargs["dtype"] = dtype_name
    
    # Prompt templating lives in `prompt_settings`, so drop duplicated values from `method_kwargs` before constructing the framework context object.
    for key in ("use_chat_template", "system_prompt", "add_generation_prompt"):
        method_kwargs.pop(key, None)

    method_kwargs["alpha_layer"] = alpha_layer
    if calibration_lambdas is not None:
        method_kwargs["calibration_lambdas"] = calibration_lambdas

    return TrainingContext(
        method_name=method_name,
        model_name=model_name,
        snapshot_path=snapshot_path,
        artifacts_dir=ALPHA_VECTOR_DIR,
        prompt_settings=prompt_settings,
        method_kwargs=method_kwargs,
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
    if backend not in {"adasteer", "alphasteer"}:
        raise ValueError(
            f"train_adasteer.py only supports ADASTEER/ALPHASTEER configs, but received method={backend!r}."
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

    alpha_layer = get_adasteer_layers(args.model)[0]
    calibration_lambdas: list[float] = list(
        training_defaults.get(
            "calibration_lambdas",
            [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
    )

    # Convert the CLI/config state into the framework `TrainingContext`, then delegate the actual artifact computation to the selected steering backend.
    context = _build_training_context(
        model_name=args.model,
        method_name=backend,
        prompt_settings=prompt_settings,
        dtype_name=dtype_name,
        snapshot_path=snapshot_path,
        training_defaults=training_defaults,
        alpha_layer=alpha_layer,
        calibration_lambdas=calibration_lambdas,
    )

    steering = get_steering_model(backend)
    artifacts = steering.train(context)
    
    # Explicit when AdaSteer refusal training was intentionally skipped because a compatible vector already existed on disk.
    _print_artifact_summary(artifacts)


if __name__ == "__main__":
    main()
