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

CAST_TRAIN_DIR = REPO_ROOT / "benchmark_data" / "training_data" / "CAST"
CAST_REFUSAL_DIR = CAST_TRAIN_DIR / "refusal_vector"
CAST_CONDITION_DIR = CAST_TRAIN_DIR / "condition_vector"
CAST_VECTOR_DIR = REPO_ROOT / "vectors" / "CAST"
CAST_REFUSAL_VECTOR_DIR = CAST_VECTOR_DIR / "refusal_vectors"
CAST_CONDITION_VECTOR_DIR = CAST_VECTOR_DIR / "condition_vectors"
DEFAULT_MAX_QUESTIONS = 100
DEFAULT_MAX_SUFFIXES = 100


def _default_refusal_paths() -> tuple[Path, Path]:
    # Default AST/CAST refusal-training data 
    # benign prompts plus refusal/compliant suffix pairs.
    return (
        CAST_REFUSAL_DIR / "alpaca_for_refusal.json",
        CAST_REFUSAL_DIR / "behavior_refusal.json",
    )


def _default_refusal_vector_out(model: str, method: str) -> Path:
    # Build the default path for the refusal vector 
    model_version = model.split("/")[-1]
    return CAST_REFUSAL_VECTOR_DIR / f"{method}_{model_version}"


def _default_condition_vector_out(model: str, method: str) -> Path:
    # Build the default path for the condition vector 
    model_version = model.split("/")[-1]
    return CAST_CONDITION_VECTOR_DIR / f"{method}_{model_version}"


def _default_refusal_analysis_out(model: str, method: str) -> Path:
    model_version = model.split("/")[-1]
    return REPO_ROOT / "analysis" / f"refusal_PCA_{method}_{model_version}"


def _default_condition_analysis_out(model: str, method: str) -> Path:
    model_version = model.split("/")[-1]
    return REPO_ROOT / "analysis" / f"cast_condition_{method}_{model_version}"


def _as_saved_svec(path_prefix: Path) -> Path:
    if str(path_prefix).endswith(".svec"):
        return path_prefix
    return path_prefix.with_name(f"{path_prefix.name}.svec")


def _resolve_condition_json(model: str, explicit_path: str | None) -> Path:
    
    # Allow configs to point directly to a condition dataset when needed.
    if explicit_path is not None:
        return Path(explicit_path)

    # Otherwise infer the file from the model name
    model_slug = safe_name(model)
    candidates = sorted(CAST_CONDITION_DIR.glob(f"{model_slug}*.json"))

    if not candidates:
        raise FileNotFoundError(
            f"No CAST condition JSON found for model {model!r} in {CAST_CONDITION_DIR}."
        )
    if len(candidates) > 1:
        joined = ", ".join(str(path.name) for path in candidates)
        raise ValueError(
            f"Ambiguous CAST condition JSON for model {model!r}: {joined}. "
            "Specify condition_json in the benchmark config."
        )
    return candidates[0]


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
    alpaca_json: Path,
    behavior_refusal_json: Path,
    refusal_vector_out: Path,
    refusal_analysis_out_dir: Path,
    should_train_refusal: bool,
    condition_json: Path | None,
    condition_vector_out: Path | None,
    condition_analysis_out_dir: Path | None,
) -> TrainingContext:
    # All dataset paths that the steering wrappers may need. AST uses the
    # refusal data, while CAST additionally consumes a condition dataset.
    data_paths: dict[str, Path] = {
        "alpaca_json": alpaca_json,
        "behavior_refusal_json": behavior_refusal_json,
    }
    if condition_json is not None:
        data_paths["condition_json"] = condition_json

    # Training only needs the training fixed parameters here, so copy them  into `method_kwargs` before normalizing a few shared keys.
    method_kwargs = dict(training_defaults)
    
    # Normalize the common vector-method keys so AST and CAST can consume the same `TrainingContext` fields.
    vector_method = str(method_kwargs.get("vector_method", method_kwargs.get("method", "pca_pairwise")))
    method_kwargs["dtype"] = dtype_name
    method_kwargs["method"] = vector_method
    method_kwargs["vector_method"] = vector_method
    method_kwargs["train_behavior"] = should_train_refusal
    method_kwargs["vector_out"] = refusal_vector_out
    method_kwargs["refusal_analysis_out_dir"] = refusal_analysis_out_dir
    
    # CAST-specific condition outputs are optional because AST does not use them.
    if condition_vector_out is not None:
        method_kwargs["condition_vector_method"] = str(
            method_kwargs.get("condition_vector_method", vector_method)
        )
        method_kwargs["condition_vector_out"] = condition_vector_out
    if condition_analysis_out_dir is not None:
        method_kwargs["condition_analysis_out_dir"] = condition_analysis_out_dir
    
    # Prompt templating lives in `prompt_settings`, so drop duplicated values from `method_kwargs` before constructing the framework context object.
    for key in ("use_chat_template", "system_prompt", "add_generation_prompt"):
        method_kwargs.pop(key, None)

    return TrainingContext(
        method_name=method_name,
        model_name=model_name,
        snapshot_path=snapshot_path,
        artifacts_dir=CAST_VECTOR_DIR,
        prompt_settings=prompt_settings,
        data_paths=data_paths,
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
    if backend not in {"ast", "cast"}:
        raise ValueError(
            f"train_ast_cast.py only supports AST/CAST configs, but received method={backend!r}."
        )
    
    # Training defaults drive vectors creation for this entrypoint.
    training_defaults = method_config.fixed_params("training")
    dtype_name = str(training_defaults.get("dtype", "auto"))
    prompt_settings = PromptTemplateSettings(
        use_chat_template=bool(training_defaults.get("use_chat_template", True)),
        system_prompt=training_defaults.get("system_prompt", "You are a helpful assistant."),
        add_generation_prompt=bool(training_defaults.get("add_generation_prompt", False)),
    )

    # Validate a few shared numeric settings up front so the steering wrappers can assume they receive sensible values.
    batch_size = int(training_defaults.get("batch_size", 32))
    max_questions = training_defaults.get("max_questions", DEFAULT_MAX_QUESTIONS)
    max_suffixes = training_defaults.get("max_suffixes", DEFAULT_MAX_SUFFIXES)
    max_condition_examples = training_defaults.get("max_condition_examples")
    vector_method = str(training_defaults.get("vector_method", training_defaults.get("method", "pca_pairwise")))
    condition_vector_method = str(training_defaults.get("condition_vector_method", vector_method))

    if batch_size < 1:
        raise ValueError("training.fixed.batch_size must be >= 1.")
    if max_questions is not None and int(max_questions) < 1:
        raise ValueError("training.fixed.max_questions must be >= 1 when specified.")
    if max_suffixes is not None and int(max_suffixes) < 1:
        raise ValueError("training.fixed.max_suffixes must be >= 1 when specified.")
    if max_condition_examples is not None and int(max_condition_examples) < 1:
        raise ValueError("training.fixed.max_condition_examples must be >= 1 when specified.")

    # Resolve the AST/CAST refusal-training inputs and default output locations.
    alpaca_json, behavior_refusal_json = _default_refusal_paths()
    alpaca_json = Path(training_defaults.get("alpaca_json", alpaca_json))
    behavior_refusal_json = Path(
        training_defaults.get("behavior_refusal_json", behavior_refusal_json)
    )

    refusal_vector_out = (
        Path(training_defaults["vector_out"])
        if "vector_out" in training_defaults
        else _default_refusal_vector_out(model=args.model, method=vector_method)
    )
    refusal_analysis_out_dir = (
        Path(training_defaults["refusal_analysis_out_dir"])
        if "refusal_analysis_out_dir" in training_defaults
        else _default_refusal_analysis_out(model=args.model, method=vector_method)
    )
    # Skip refusal training when the target vector already exists unless the config explicitly forces retraining.
    refusal_vector_exists = _as_saved_svec(refusal_vector_out).exists()
    should_train_refusal = bool(training_defaults.get("force_refusal", False)) or not refusal_vector_exists

    # Emit the key resolved inputs/outputs before any expensive model loading.
    print(f"[INFO] Benchmark config: {method_config.path.resolve()}")
    print(f"[INFO] Backend: {backend}")
    print(f"[INFO] Refusal questions: {alpaca_json}")
    print(f"[INFO] Refusal suffixes: {behavior_refusal_json}")
    print(f"[INFO] Refusal vector target: {_as_saved_svec(refusal_vector_out)}")

    condition_json = None
    condition_vector_out = None
    condition_analysis_out_dir = None

    # CAST requires an additional condition dataset and output vector location.
    if backend == "cast":
        condition_json = _resolve_condition_json(
            model=args.model,
            explicit_path=training_defaults.get("condition_json"),
        )
        condition_vector_out = (
            Path(training_defaults["condition_vector_out"])
            if "condition_vector_out" in training_defaults
            else _default_condition_vector_out(model=args.model, method=condition_vector_method)
        )
        condition_analysis_out_dir = (
            Path(training_defaults["condition_analysis_out_dir"])
            if "condition_analysis_out_dir" in training_defaults
            else _default_condition_analysis_out(model=args.model, method=condition_vector_method)
        )

        print(f"[INFO] Condition pairs: {condition_json}")
        print(f"[INFO] Condition vector target: {_as_saved_svec(condition_vector_out)}")

    # Only resolve the full model snapshot when some training work actually needsit. 
    # Pure "reuse existing refusal vector" AST runs can avoid that lookup.
    needs_snapshot = should_train_refusal or backend == "cast"
    snapshot_path = (
        Path(repo_to_latest_snapshot_dir(args.model))
        if needs_snapshot
        else Path(".")
    )
    
    # Convert the CLI/config state into the framework `TrainingContext`, then delegate the actual artifact computation to the selected steering backend.
    context = _build_training_context(
        model_name=args.model,
        method_name=backend,
        prompt_settings=prompt_settings,
        dtype_name=dtype_name,
        snapshot_path=snapshot_path,
        training_defaults=training_defaults,
        alpaca_json=alpaca_json,
        behavior_refusal_json=behavior_refusal_json,
        refusal_vector_out=refusal_vector_out,
        refusal_analysis_out_dir=refusal_analysis_out_dir,
        should_train_refusal=should_train_refusal,
        condition_json=condition_json,
        condition_vector_out=condition_vector_out,
        condition_analysis_out_dir=condition_analysis_out_dir,
    )

    steering = get_steering_model(backend)
    artifacts = steering.train(context)
    
    
    # Explicit when AST refusal training was intentionally skipped because a compatible vector already existed on disk.
    if not should_train_refusal:
        print("[INFO] Refusal vector already exists. Skipping refusal training.")
    _print_artifact_summary(artifacts)


if __name__ == "__main__":
    main()
