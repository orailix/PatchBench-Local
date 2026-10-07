
import argparse
import gc
import json
import multiprocessing as mp
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from tqdm import tqdm
from transformers import GenerationConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
from wildguard import load_wildguard

from src.models.utils import add_padding_token, get_model, get_tokenizer, repo_to_latest_snapshot_dir
from src.patching.inference.benchmark_utils import (
    INFERENCE_RUNNER_KEYS,
    compute_category_metrics,
    default_validation_input_path,
)
from src.patching.steering import (
    BatchContext,
    InferenceContext,
    SteeringArtifacts,
    get_steering_model,
)
from src.patching.utils import PromptTemplateSettings, format_user_prompt, resolve_dtype

def build_common_parser(description: str) -> argparse.ArgumentParser:
    # Shared CLI used by validation entrypoints. Method-specific scripts keep their own logic, but they all need the same model/config/batching params.
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the benchmark YAML/JSON config.",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Optional cap on evaluated hyperparameter combinations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for inference batch size during validation.",
    )
    return parser


def attach_wildguard_labels(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # WildGuard is used after generation to label each prompt/response pair with safety judgments. 
    # The validation metrics are then computed from those labels rather than from raw generations alone.
   
    wildguard = load_wildguard()
    # Use the original plain user prompt rather than the chat-formatted `raw_prompt`, 
    # so WildGuard sees the same kind of input text it was designed to classify.
    samples = [
        {
            "id": item.get("id", i),
            "prompt": item["prompt"],
            "response": item["answer"],
        }
        for i, item in enumerate(outputs)
    ]
    wildguard_results = wildguard.classify(samples)

    # Merge the predicted labels back into the original output rows while keeping the existing generation fields untouched.
    for row, label in zip(outputs, wildguard_results):
        for key, value in label.items():
            if key not in {"id", "prompt", "response"}:
                row[key] = value
        if row["answer"] == "":
            row["response_refusal"] = "refusal"
            row["response_harmfulness"] = "unharmful"
            
        if row["response_refusal"] == None:
            row["response_refusal"] = "refusal"
        
        if row["response_harmfulness"] == None:
            row["response_harmfulness"] = "unharmful"

    return outputs


def build_method_kwargs(config_defaults: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config_defaults.items()
        if key not in INFERENCE_RUNNER_KEYS and value is not None
    }


def sample_param_sets(
    param_sets: list[dict[str, Any]],
    max_combos: int | None,
) -> tuple[list[dict[str, Any]], bool]:
    # Optionally subsample the full grid for faster experiments while keeping the selection deterministic across runs.
    if max_combos is None:
        return param_sets, False
    if max_combos < 1:
        raise ValueError("--max-combos must be >= 1 when provided.")
    if len(param_sets) <= max_combos:
        return param_sets, False

    rng = random.Random(42)
    sampled_indices = sorted(rng.sample(range(len(param_sets)), max_combos))
    return [param_sets[index] for index in sampled_indices], True


def strip_internal_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Validation rows carry `_combo_idx` internally so we can regroup outputs by hyperparameter setting. 
    cleaned_rows: list[dict[str, Any]] = []
    for row in rows:
        cleaned_row = dict(row)
        cleaned_row.pop("_combo_idx", None)
        cleaned_rows.append(cleaned_row)
    return cleaned_rows


def build_runtime_defaults(
    method_config,
    batch_size_override: int | None,
) -> tuple[dict[str, Any], PromptTemplateSettings, int, GenerationConfig, torch.dtype]:
    
    # Validation generation mostly uses inference defaults, but it may also need training defaults for fallback values such as dtype.
    training_defaults = method_config.fixed_params("training")
    inference_fixed = method_config.fixed_params("inference")
    validation_defaults = method_config.selected_params()

    # Start from inference-time settings, then let validation-selected defaults override them where the config specifies method hyperparameters to sweep.
    config_defaults = {**inference_fixed, **validation_defaults}
    batch_size = int(config_defaults.get("batch_size", 1))
    if batch_size_override is not None:
        batch_size = batch_size_override
    if batch_size < 1:
        raise ValueError("Validation batch size must be >= 1.")

    max_input_tokens = int(config_defaults.get("max_input_tokens", 4096))
    max_new_tokens = int(config_defaults.get("max_new_tokens", 250))
    dtype_name = str(config_defaults.get("dtype", training_defaults.get("dtype", "auto")))
    dtype = resolve_dtype(dtype_name)
    
    # Prompt formatting during validation should mirror inference so the tested hyperparameters see the same prompt structure as the final benchmark runs.
    prompt_settings = PromptTemplateSettings(
        use_chat_template=bool(config_defaults.get("use_chat_template", True)),
        system_prompt=config_defaults.get("system_prompt", "You are a helpful assistant."),
        add_generation_prompt=bool(config_defaults.get("add_generation_prompt", True)),
    )
    
    # Validation uses deterministic decoding so metric differences come from the  tested steering parameters rather than sampling randomness.
    generation_config = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens)
    return config_defaults, prompt_settings, max_input_tokens, generation_config, dtype


def load_validation_records(model_name: str) -> tuple[Path, list[dict[str, Any]]]:
    
    # Validation runs consume a per-model valdiation prompts JSON file prepared ahead oftime.
    # Fail early with a clear message if that input file is missing.
    in_path = default_validation_input_path(model_name)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Validation input JSON not found: {in_path}. "
            "Make sure the flattened validation file exists before running validation."
        )
    records = json.loads(in_path.read_text(encoding="utf-8"))
    return in_path, records


def _run_generations(
    args_dump,
    param_sets: list[dict[str, Any]],
    records: list[dict[str, Any]],
    method_config_dump,
    snapshot_path_dump: str,
    combo_index_offset: int = 0,
    total_combo_count: int | None = None,
) -> list[dict[str, Any]]:
    # This function runs inside the spawned child process created by `run_generation_search`.
    #
    # This is the place where validation-time inference actually happens:
    # the model is loaded here, prepared for each hyperparameter combination and then used to generate outputs batch by batch.
    #
    # Keeping the heavy model work in a subprocess makes cleanup more robust and avoids leaving CUDA/model state behind in the parent process.
    config_defaults, prompt_settings, max_input_tokens, generation_config, dtype = build_runtime_defaults(
        method_config_dump,
        args_dump.batch_size,
    )
    base_method_kwargs = build_method_kwargs(config_defaults)
    backend = method_config_dump.method_name

    tokenizer = get_tokenizer(args_dump.model, snapshot_path_dump)
    tokenizer.padding_side = "left"
    tokenizer = add_padding_token(tokenizer, args_dump.model)
    steering = get_steering_model(backend)
    reload_model_per_combo = steering.validation_requires_fresh_model_per_combo()
    shared_model = None
    if not reload_model_per_combo:
        shared_model = get_model(args_dump.model, snapshot_path_dump, dtype=dtype, device_map="auto")

    all_combo_outputs: list[dict[str, Any]] = []
    combo_count_for_logs = len(param_sets) if total_combo_count is None else int(total_combo_count)

    for combo_idx, combo_params in enumerate(param_sets):
        output_combo_idx = combo_index_offset + combo_idx
        combo_model = shared_model
        if reload_model_per_combo:
            combo_model = get_model(args_dump.model, snapshot_path_dump, dtype=dtype, device_map="auto")

        # Build one `InferenceContext` per hyperparameter combination so the
        # steering wrapper can prepare the model exactly as it would at
        # benchmark inference time.
        combo_method_kwargs = {**base_method_kwargs, **combo_params}
        inference_context = InferenceContext(
            method_name=backend,
            model_name=args_dump.model,
            snapshot_path=Path(snapshot_path_dump),
            dtype=dtype,
            prompt_settings=prompt_settings,
            artifacts=SteeringArtifacts(),
            method_kwargs=combo_method_kwargs,
        )
        combo_outputs: list[dict[str, Any]] = []
        prepared = None
        try:
            prepared = steering.prepare_model(model=combo_model, tokenizer=tokenizer, context=inference_context)
            for batch_start in tqdm(
                range(0, len(records), args_dump.runtime_batch_size),
                desc=f"Validating combo {output_combo_idx + 1}/{combo_count_for_logs}",
                unit="batch",
            ):
                batch = records[batch_start : batch_start + args_dump.runtime_batch_size]
                prompts_for_model = []
                ids = []

                # Format prompts exactly as the inference pipeline would, including chat templating and any system prompt from the config.
                for row_index, row in enumerate(batch):
                    prompt_for_model = row["prompt"]
                    chat_text = format_user_prompt(
                        tokenizer=prepared.tokenizer,
                        user_prompt=prompt_for_model,
                        settings=prompt_settings,
                    )
                    prompts_for_model.append(chat_text)
                    ids.append(row.get("id", batch_start + row_index))

                # Tokenize the batch and keep per-example prompt lengths so steering methods like AST/CAST can distinguish prompt tokens from generated continuation tokens at runtime.
                inputs = prepared.tokenizer(
                    prompts_for_model,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_tokens,
                )
                prompt_lengths = None
                if "attention_mask" in inputs:
                    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
                inputs = inputs.to(prepared.model.device)

                batch_context = BatchContext(
                    rows=batch,
                    prompt_texts=prompts_for_model,
                    inputs=inputs,
                    prompt_lengths=prompt_lengths,
                )

                # Let the steering backend manage any per-batch runtime state around
                # the actual generation call.
                #
                # The real inference happens on the next line:
                # `steering.generate_batch(...)` ultimately calls the model's
                # `generate(...)` method unless a steering method overrides it with
                # custom generation logic.
                steering.before_batch(prepared, batch_context)
                with torch.no_grad():
                    out = steering.generate_batch(prepared, batch_context, generation_config)
                steering.after_batch(prepared, batch_context, out)

                alpha_pos_list = None
                beta_pos_list = None
                if isinstance(out, tuple):
                    out, alpha_pos_list, beta_pos_list = out
                    alpha_pos_list = alpha_pos_list.detach().cpu().tolist()
                    beta_pos_list = beta_pos_list.detach().cpu().tolist()

                # Remove the prompt tokens and keep only the newly generated answer text when writing validation outputs.
                input_len = inputs["input_ids"].shape[1]

                for row_index in range(len(batch)):
                    answer = prepared.tokenizer.decode(
                        out[row_index][input_len:],
                        skip_special_tokens=True,
                    )
                    result = dict(batch[row_index])
                    result["id"] = ids[row_index]
                    result["raw_prompt"] = prompts_for_model[row_index]
                    result["answer"] = answer
                    result["_combo_idx"] = output_combo_idx
                    if alpha_pos_list is not None:
                        result["adasteer_alpha_pos"] = alpha_pos_list[row_index]
                    if beta_pos_list is not None:
                        result["adasteer_beta_pos"] = beta_pos_list[row_index]
                    combo_outputs.append(result)

        finally:
            # Some steering wrappers keep mutable layer state on the wrapped
            # model. Reset after each combo, and optionally release the whole
            # model when the method asked for a fresh load per combination.
            if prepared is not None:
                reset_fn = getattr(prepared.model, "reset_leash_to_default", None)
                if callable(reset_fn):
                    reset_fn()

            if reload_model_per_combo:
                if prepared is not None:
                    del prepared
                del combo_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        all_combo_outputs.extend(combo_outputs)

    return all_combo_outputs


def _run_generation_search_subprocess(
    args_dump,
    param_sets_dump: list[dict[str, Any]],
    records_dump: list[dict[str, Any]],
    method_config_dump,
    snapshot_path_dump: str,
    tmp_path: str,
    combo_index_offset: int = 0,
    total_combo_count: int | None = None,
) -> None:
    # Tiny subprocess entrypoint used by `multiprocessing`.
    #
    # It delegates all actual model loading and generation work to `_run_generations`, then serializes the raw outputs to a temporary JSON file so the parent process can reload them after the child exits.
    all_outputs = _run_generations(
        args_dump=args_dump,
        param_sets=param_sets_dump,
        records=records_dump,
        method_config_dump=method_config_dump,
        snapshot_path_dump=snapshot_path_dump,
        combo_index_offset=combo_index_offset,
        total_combo_count=total_combo_count,
    )
    tmp_output_path = Path(tmp_path)
    tmp_output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path.write_text(json.dumps(all_outputs), encoding="utf-8")


def run_generation_search(
    *,
    model_name: str,
    method_config,
    param_sets: list[dict[str, Any]],
    records: list[dict[str, Any]],
    batch_size_override: int | None,
    tmp_path: Path,
) -> tuple[list[dict[str, Any]], PromptTemplateSettings, torch.dtype]:
    # Parent-side orchestration for the validation generation pass.
    #
    # This function does NOT run the model forward itself. 
    # Instead, it prepares the validation settings, spawns a child process, and waits for that child to perform the actual inference work inside`_run_generations`.
    #
    # Spawning a child process helps isolate CUDA/model state and makes cleanup more robust after validation completes.
    config_defaults, prompt_settings, _, _, dtype = build_runtime_defaults(method_config, batch_size_override)
    runtime_batch_size = int(batch_size_override or config_defaults.get("batch_size", 1))
    if runtime_batch_size < 1:
        raise ValueError("Validation batch size must be >= 1.")

    args_dump = SimpleNamespace(
        model=model_name,
        batch_size=batch_size_override,
        runtime_batch_size=runtime_batch_size,
    )
    snapshot_path = str(repo_to_latest_snapshot_dir(model_name))
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    steering = get_steering_model(method_config.method_name)
    
    # Best-effort cleanup before loading the model in the subprocess.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ctx = mp.get_context("spawn")

    if steering.validation_requires_fresh_model_per_combo():
        # Stronger isolation for methods that mutate model internals in place:
        # each combo gets its own subprocess so CUDA memory is fully released
        # when that subprocess exits.
        all_combo_outputs: list[dict[str, Any]] = []
        total_combos = len(param_sets)
        for combo_idx, combo_params in enumerate(param_sets):
            combo_tmp_path = tmp_path.with_name(f"{tmp_path.stem}__combo_{combo_idx}{tmp_path.suffix}")
            process = ctx.Process(
                target=_run_generation_search_subprocess,
                args=(
                    args_dump,
                    [combo_params],
                    records,
                    method_config,
                    snapshot_path,
                    str(combo_tmp_path),
                    combo_idx,
                    total_combos,
                ),
            )
            process.start()
            process.join()

            if process.exitcode != 0:
                combo_tmp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Validation generation process failed for combo {combo_idx + 1}/{total_combos}."
                )

            combo_outputs = json.loads(combo_tmp_path.read_text(encoding="utf-8"))
            combo_tmp_path.unlink(missing_ok=True)
            all_combo_outputs.extend(combo_outputs)
    else:
        process = ctx.Process(
            target=_run_generation_search_subprocess,
            args=(
                args_dump,
                param_sets,
                records,
                method_config,
                snapshot_path,
                str(tmp_path),
                0,
                len(param_sets),
            ),
        )
        process.start()
        process.join()

        if process.exitcode != 0:
            raise RuntimeError("Validation generation process failed.")
        # Once the child process has finished all inference, reload the raw outputs
        # it wrote to disk, then remove the temporary handoff file.
        all_combo_outputs = json.loads(tmp_path.read_text(encoding="utf-8"))
        tmp_path.unlink(missing_ok=True)
    return all_combo_outputs, prompt_settings, dtype


def result_sort_key(result: dict[str, Any]) -> tuple[int, float]:
    # Sort valid scored results ahead of failures, then order successful runs by descending overall metric.
    overall_metric = result.get("metrics", {}).get("overall_metric")
    if isinstance(overall_metric, (int, float)):
        return (1, float(overall_metric))
    return (0, float("-inf"))


def score_generation_outputs(
    param_sets: list[dict[str, Any]],
    all_combo_outputs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    # First enrich the generations with WildGuard labels.
    # if that fails, keep a structured error payload so downstream reporting can explain why scoring was unavailable.
    wildguard_status: dict[str, Any] = {"status": "ok"}
    try:
        all_combo_outputs = attach_wildguard_labels(all_combo_outputs)
    except Exception as exc:
        wildguard_status = {"status": "error", "message": str(exc)}

    return compute_results_from_outputs(
        param_sets=param_sets,
        all_combo_outputs=all_combo_outputs,
        wildguard_status=wildguard_status,
    )


def compute_results_from_outputs(
    *,
    param_sets: list[dict[str, Any]],
    all_combo_outputs: list[dict[str, Any]],
    wildguard_status: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    
    # Group rows back by hyperparameter combination using the internal combo index attached during generation.
    outputs_by_combo = {index: [] for index in range(len(param_sets))}
    for row in all_combo_outputs:
        outputs_by_combo[row["_combo_idx"]].append(row)

    results: list[dict[str, Any]] = []
    for combo_idx, combo_params in enumerate(param_sets):
        combo_outputs = outputs_by_combo.get(combo_idx, [])
        result: dict[str, Any] = {
            "combo_index": combo_idx,
            "params": combo_params,
            "num_records": len(combo_outputs),
        }
        # If labeling failed globally, mark every combo as an error. Otherwise, compute per-category validation metrics from the labeled outputs.
        if wildguard_status["status"] != "ok":
            result["status"] = "error"
            result["error"] = wildguard_status["message"]
        else:
            try:
                result["metrics"] = compute_category_metrics(combo_outputs)
                result["status"] = "ok"
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)
        results.append(result)

    return all_combo_outputs, outputs_by_combo, results, wildguard_status


def sort_and_select_best(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    # Return both the full ranking and the best successful result.
    sorted_results = sorted(results, key=result_sort_key, reverse=True)
    best_result = next((result for result in sorted_results if result.get("status") == "ok"), None)
    return sorted_results, best_result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    # Shared JSON writer used by validation scripts when storing summaries and intermediate artifacts.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prompt_settings_dict(prompt_settings: PromptTemplateSettings) -> dict[str, Any]:
    # Convert the dataclass to plain JSON-serializable metadata for result files.
    return asdict(prompt_settings)


def collect_adasteer_thresholds(
    outputs: list[dict[str, Any]],
    param_sets: list[dict[str, Any]],
    pos_field: str,
    lambda_field: str,
    target_categories: set[str],
    target_is_refusal: bool = True,
    find_max: bool = False,
) -> list[tuple[float, float]]:
    """
    Collect (pos, threshold_lambda) pairs for prompts in target_categories.

    target_is_refusal=True,  find_max=False: min lambda where refused   (compliance→refusal)
    target_is_refusal=False, find_max=False: min lambda where compliant  (refusal→compliance)
    target_is_refusal=True,  find_max=True:  max lambda still refused    (last safe lambda before acceptance)

    Only includes prompts where the target is NOT already met at any λ≤0 (real transition).
    """
    from collections import defaultdict

    prompt_data: dict = defaultdict(lambda: {"pos": None, "by_lambda": {}})
    for row in outputs:
        if row.get("category") not in target_categories:
            continue
        combo_idx = row.get("_combo_idx")
        lam = param_sets[combo_idx].get(lambda_field) if combo_idx is not None else None
        if lam is not None:
            lam = float(lam)
        pos = row.get(pos_field)
        is_refused = row.get("response_refusal") == "refusal"
        target_met = is_refused if target_is_refusal else not is_refused
        pid = row.get("prompt")
        if pos is not None and prompt_data[pid]["pos"] is None:
            prompt_data[pid]["pos"] = pos
        if lam is not None:
            prompt_data[pid]["by_lambda"][lam] = target_met

    agg = max if find_max else min
    return [
        (data["pos"], agg(lam for lam, met in data["by_lambda"].items() if met))
        for data in prompt_data.values()
        if data["pos"] is not None
        and any(data["by_lambda"].values())
        and not any(met for lam, met in data["by_lambda"].items() if lam <= 0.0)
    ]


def fit_linear_thresholds(
    fit_points: list[tuple[float, float]],
) -> tuple[float, float, float, int]:
    """Fit threshold = w * (pos - b) from (pos, threshold) pairs. Returns (w, b, r2, n)."""
    import numpy as np

    if len(fit_points) < 2:
        return float("nan"), float("nan"), float("nan"), len(fit_points)

    x = np.array([p for p, _ in fit_points])
    y = np.array([lam for _, lam in fit_points])
    w, intercept = np.polyfit(x, y, 1)
    b = -intercept / w if abs(w) > 1e-12 else float("nan")
    y_pred = w * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(w), float(b), r2, len(fit_points)


def fit_adasteer_regression(
    outputs: list[dict[str, Any]],
    param_sets: list[dict[str, Any]],
    pos_field: str,
    lambda_field: str,
    target_categories: set[str],
    target_is_refusal: bool = True,
    find_max: bool = False,
) -> tuple[float, float, float, int]:
    """Convenience wrapper: collect thresholds then fit. Returns (w, b, r2, n_points)."""
    points = collect_adasteer_thresholds(
        outputs, param_sets, pos_field, lambda_field,
        target_categories, target_is_refusal, find_max,
    )
    print(points)
    return fit_linear_thresholds(points)
