#!/usr/bin/env python3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from src.models.utils import repo_to_latest_snapshot_dir
from src.patching.inference.benchmark_utils import default_validation_generations_path
from src.patching.steering import ValidationContext, get_steering_model
from src.patching.steering.base import build_param_grid
from src.patching.utils import load_benchmark_method_config_from_path
from src.patching.validation.common import (
    attach_wildguard_labels,
    build_common_parser,
    build_method_kwargs,
    build_runtime_defaults,
    collect_adasteer_thresholds,
    fit_adasteer_regression,
    fit_linear_thresholds,
    load_validation_records,
    run_generation_search,
    sample_param_sets,
    write_json,
)


# Message to precise that the validation is method-agnostic at the runner level.
# The config selects the steering method, and the steering class decides whether any method-specific offline validation is needed.
def build_parser():
    return build_common_parser(
        "Run validation with a shared runner and method-specific steering hooks.",
    )


def run_validation(args) -> None:
    # Load the method config, then use its `method` field to resolve which steering backend should drive validation behavior.
    method_config = load_benchmark_method_config_from_path(args.config, model_name=args.model)
    
    # Fix parameter values for models listed in fixed_by_model.
    selected = {}
    for name, spec in method_config.validation_selected.items():
        fixed_values = spec.get("fixed_by_model") if isinstance(spec, dict) else None
        if isinstance(fixed_values, dict) and args.model in fixed_values:
            value = fixed_values[args.model]
            spec = {**spec, "default": value, "values": [value]}
        selected[name] = spec

    method_config = replace(method_config, validation_selected=selected)
    backend = method_config.method_name

    # Collect the config sections needed for validation.
    # - `validation_fixed` configures the validation procedure itself
    # - `selected_defaults` provides fallback values when validation results do not exist yet or when a parameter has no explicit `values` list
    validation_fixed = method_config.fixed_params("validation")
    selected_defaults = method_config.selected_params()

    # Reuse the same runtime-default builder used elsewhere so validation uses the same prompt formatting, dtype and decoding assumptions as inference.
    config_defaults, prompt_settings, _, _, dtype = build_runtime_defaults(
        method_config,
        args.batch_size,
    )

    # Method kwargs are the steering-specific runtime arguments only.
    # Shared runner keys such as batch size or prompt formatting are stripped out so the steering class sees only the parameters it should own.
    runtime_method_kwargs = build_method_kwargs(config_defaults)

    # Validation records are the held-out prompts used to compare hyperparameter combinations before running the final benchmark.
    in_path, records = load_validation_records(args.model)
    records = records#[:100]

    # Resolve the model snapshot once here and pass it through the validation context so method-specific offline validation can load the same model if needed, for example CAST gate-search.
    snapshot_path = Path(repo_to_latest_snapshot_dir(args.model))
    steering = get_steering_model(backend)

    # `ValidationContext` is the shared "content" between the generic runner and the method-specific validation hooks living in `src/patching/steering/`.
    validation_context = ValidationContext(
        method_name=backend,
        model_name=args.model,
        snapshot_path=snapshot_path,
        dtype=dtype,
        prompt_settings=prompt_settings,
        validation_fixed=validation_fixed,
        validation_selected=method_config.validation_selected,
        selected_defaults=selected_defaults,
        method_kwargs=runtime_method_kwargs,
    )

    print(f"[INFO] Benchmark config: {method_config.path.resolve()}")
    print(f"[INFO] Backend: {backend}")
    print(f"[INFO] Validation input: {in_path.resolve()}")
    print(f"[INFO] Using dtype: {dtype}")

    generations_path = default_validation_generations_path(args.model, backend)
    temp_file = generations_path.with_suffix(generations_path.suffix + ".tmp")

    # Vérifier que les 4 paramètres de régression AdaSteer sont tous fixés pour ce modèle
    adasteer_params = ["w_r", "b_r", "w_c", "b_c"]
    has_fixed_adasteer = (
        backend == "adasteer"
        and all(
            isinstance(method_config.validation_selected.get(p), dict)
            and args.model in method_config.validation_selected[p].get("fixed_by_model", {})
            for p in adasteer_params
        )
    )

    if has_fixed_adasteer:
        selected["lambda_r"] = {"default": 0.0, "values": [0.0]}
        selected["lambda_c"] = {"default": 0.0, "values": [0.0]}
        method_config = replace(method_config, validation_selected=selected)  
        validation_context = replace(validation_context, validation_selected=selected)

    if backend == "adasteer" and not has_fixed_adasteer:
        generations_payload = _run_adasteer_two_pass(
            args=args,
            method_config=method_config,
            validation_context=validation_context,
            records=records,
            in_path=in_path,
            prompt_settings=prompt_settings,
            temp_file=temp_file,
        )
    elif backend == "adasteer" and has_fixed_adasteer :
        print(f"[INFO] AdaSteer parameters already fixed in config for {args.model}, skipping 2-pass sweep.")
        generations_payload = _run_standard_validation(
            args=args,
            method_config=method_config,
            validation_context=validation_context,
            steering=steering,
            records=records,
            in_path=in_path,
            prompt_settings=prompt_settings,
            temp_file=temp_file,
        )
    else:
        generations_payload = _run_standard_validation(
            args=args,
            method_config=method_config,
            validation_context=validation_context,
            steering=steering,
            records=records,
            in_path=in_path,
            prompt_settings=prompt_settings,
            temp_file=temp_file,
        )

    write_json(generations_path, generations_payload)
    print(f"[DONE] Saved validation generations to {generations_path.resolve()}")
    print(
        "[NEXT] Run validation labeling with WildGuard:\n"
        f"python -m src.patching.validation.label_validation_with_wildguard --model {args.model} --config {args.config}"
    )


def _run_standard_validation(
    args, method_config, validation_context, steering, records, in_path, prompt_settings, temp_file
) -> dict:
    offline_result = steering.offline_validation(
        context=validation_context,
        records=records,
        batch_size_override=args.batch_size,
    )

    # Phase 2: ask the steering method which hyperparameter combinations should be evaluated with real generation. AST uses the default full grid, while CAST injects the gate-search result and only sweeps the remaining params.
    plan = steering.build_validation_plan(
        context=validation_context,
        offline_result=offline_result,
    )
    all_param_sets = plan.param_sets

    # Optional sampling lets us cap the number of combinations for quicker experiments while keeping the selection deterministic across runs.
    param_sets, sampled = sample_param_sets(all_param_sets, args.max_combos)
    if not param_sets:
        raise ValueError("Validation produced no hyperparameter combinations to evaluate.")

    print(f"[INFO] Evaluating {len(param_sets)} validation combinations.")
    if sampled:
        print(f"[INFO] Sampled down from {len(all_param_sets)} to {len(param_sets)} combinations.")

    all_combo_outputs, _, _ = run_generation_search(
        model_name=args.model,
        method_config=method_config,
        param_sets=param_sets,
        records=records,
        batch_size_override=args.batch_size,
        tmp_path=temp_file,
    )
    return {
        "model_name": args.model,
        "backend": method_config.method_name,
        "config_path": str(method_config.path.resolve()),
        "validation_input_path": str(in_path.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_settings": {
            "use_chat_template": prompt_settings.use_chat_template,
            "system_prompt": prompt_settings.system_prompt,
            "add_generation_prompt": prompt_settings.add_generation_prompt,
        },
        "search": {
            "total_combinations": len(all_param_sets),
            "evaluated_combinations": len(param_sets),
            "sampled": sampled,
            "max_combos": args.max_combos,
        },
        "offline_validation": offline_result.report_payload or None,
        "validation_plan": plan.metadata or {},
        "param_sets": param_sets,
        "outputs": all_combo_outputs,
    }


def _adasteer_pass_param_sets(validation_selected: dict, overrides: dict) -> list[dict]:
    """Build a param grid from validation_selected with specific keys overridden to fixed values."""
    patched = {
        k: ({"default": overrides[k]} if k in overrides else v)
        for k, v in validation_selected.items()
    }
    # Ensure override keys not already in validation_selected are included
    for k, v in overrides.items():
        patched.setdefault(k, {"default": v})
    return build_param_grid(validation_selected=patched, selected_defaults={})


def _run_adasteer_two_pass(
    args, method_config, validation_context, records, in_path, prompt_settings, temp_file
) -> dict:
    vs = method_config.validation_selected

    # ── Pass 1: fix beta params to zero, sweep lambda_r ──────────────────────
    print("[AdaSteer] Pass 1: sweeping lambda_r (w_c=b_c=lambda_c=w_r=b_r=0)")
    pass1_params = _adasteer_pass_param_sets(vs, {
        "w_r": 0.0, "b_r": 0.0, "w_c": 0.0, "b_c": 0.0, "lambda_c": 0.0,
    })
    pass1_params, _ = sample_param_sets(pass1_params, args.max_combos)
    print(f"[AdaSteer] Pass 1: {len(pass1_params)} combinations")

    pass1_outputs, _, _ = run_generation_search(
        model_name=args.model,
        method_config=method_config,
        param_sets=pass1_params,
        records=records,
        batch_size_override=args.batch_size,
        tmp_path=temp_file,
    )
    print("[AdaSteer] Pass 1: labeling with WildGuard...")
    pass1_outputs = attach_wildguard_labels(pass1_outputs)

    w_r, b_r, r2_r, n_r = fit_adasteer_regression(
        pass1_outputs, pass1_params,
        pos_field="adasteer_alpha_pos",
        lambda_field="lambda_r",
        target_categories={"original", "close_harmful"},
    )
    print(f"[AdaSteer] Pass 1 fit: w_r={w_r:.6f}  b_r={b_r:.6f}  R²={r2_r:.4f}  n={n_r}")

    # ── Pass 2: use fitted w_r/b_r, sweep lambda_c ───────────────────────────
    # lambda_r is irrelevant when w_r≠0 (adaptive mode), so fix it to 0 to avoid
    # redundant combinations. Reuse lambda_r's value range for the lambda_c sweep.
    print("[AdaSteer] Pass 2: sweeping lambda_c (w_r/b_r fitted, w_c=b_c=0)")
    pass2_vs = {
        **vs,
        "w_r":      {"default": w_r},
        "b_r":      {"default": b_r},
        "w_c":      {"default": 0.0},
        "b_c":      {"default": 0.0},
        "lambda_r": {"default": 0.0},
        "lambda_c": vs.get("lambda_c") or vs.get("lambda_r", {"default": 0.1}),
    }
    pass2_params = build_param_grid(validation_selected=pass2_vs, selected_defaults={})
    pass2_params, _ = sample_param_sets(pass2_params, args.max_combos)
    # Offset combo indices so pass1 and pass2 outputs can coexist in a flat list.
    offset = len(pass1_params)
    print(f"[AdaSteer] Pass 2: {len(pass2_params)} combinations")

    pass2_outputs_raw, _, _ = run_generation_search(
        model_name=args.model,
        method_config=method_config,
        param_sets=pass2_params,
        records=records,
        batch_size_override=args.batch_size,
        tmp_path=temp_file,
    )

    # print("[AdaSteer] Pass 2: labeling with WildGuard...")
    pass2_outputs = attach_wildguard_labels(pass2_outputs_raw)

    # Apply offset before combining so all_param_sets lookups are correct for both passes.
    for row in pass2_outputs:
        if "_combo_idx" in row:
            row["_combo_idx"] += offset

    all_outputs = pass1_outputs + pass2_outputs
    all_param_sets = pass1_params + pass2_params

    # benign_points = collect_adasteer_thresholds(
    #     all_outputs, all_param_sets,
    #     pos_field="adasteer_beta_pos", lambda_field="lambda_c",
    #     target_categories={"benign_harmful_words", "benign_same_structure"},
    #     target_is_refusal=False, find_max=False,
    # )
    # w_c, b_c, r2_c, n_c = fit_linear_thresholds(benign_points)
    # print(f"[AdaSteer] Pass 2 fit: w_c={w_c:.6f}  b_c={b_c:.6f}  R²={r2_c:.4f}  n={n_c}")

    selected_params = {
        **{k: v.get("default") if isinstance(v, dict) else v for k, v in vs.items()},
        "w_r": w_r, "b_r": b_r,
        "w_c": 0.0, "b_c": 0.0,
        "lambda_r": 0.0,
        "lambda_c": 0.0,
    }

    return {
        "model_name": args.model,
        "backend": "adasteer",
        "config_path": str(method_config.path.resolve()),
        "validation_input_path": str(in_path.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_settings": {
            "use_chat_template": prompt_settings.use_chat_template,
            "system_prompt": prompt_settings.system_prompt,
            "add_generation_prompt": prompt_settings.add_generation_prompt,
        },
        "wildguard": {"status": "ok"},
        "param_sets": all_param_sets,
        "outputs": all_outputs,
        "selected_params": selected_params,
        "adasteer_calibration": {
            "pass1": {"w_r": w_r, "b_r": b_r, "r2": r2_r, "n": n_r},
            # "pass2": {"w_c": w_c, "b_c": b_c, "r2": r2_c, "n": n_c},
        },
    }


def main() -> None:
    # CLI entrypoint for the shared validation runner.
    args = build_parser().parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()
