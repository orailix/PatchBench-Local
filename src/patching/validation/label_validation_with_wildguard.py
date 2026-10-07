#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.patching.inference.benchmark_utils import (
    default_validation_best_outputs_path,
    default_validation_generations_path,
    default_validation_report_path,
)
from src.patching.utils import load_benchmark_method_config_from_path
from src.patching.validation.common import (
    attach_wildguard_labels,
    compute_results_from_outputs,
    sort_and_select_best,
    strip_internal_fields,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    # This is the last stage of validation: it takes already-generated validation outputs, labels them with WildGuard, and writes the final ranked report hyperparameters combinations.
    parser = argparse.ArgumentParser(
        description="Run WildGuard labeling and final scoring for saved validation generations.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the benchmark YAML config.",
    )
    return parser


def main() -> None:
    # Load the config only to resolve the patching method and the standard validation file locations associated with that method.
    args = build_parser().parse_args()
    method_config = load_benchmark_method_config_from_path(args.config)
    backend = method_config.method_name

    # Validation generation must have been run first, because this script only labels and scores saved outputs. It does not generate answers itself.
    generations_path = default_validation_generations_path(args.model, backend)
    if not generations_path.exists():
        raise FileNotFoundError(
            "Cannot run validation labeling because the saved validation generations do not exist: "
            f"{generations_path}"
        )

    report_path = default_validation_report_path(args.model, backend)
    best_outputs_path = default_validation_best_outputs_path(args.model, backend)

    # Load the saved validation generations. Method-specific offline validation metadata, when present, is already embedded in this payload.
    payload = json.loads(generations_path.read_text(encoding="utf-8"))
    offline_validation_payload = payload.get("offline_validation")
    outputs = payload.get("outputs")
    param_sets = payload.get("param_sets")
    if not isinstance(outputs, list):
        raise TypeError(f"Validation generations field 'outputs' must be a list: {generations_path}")
    if not isinstance(param_sets, list):
        raise TypeError(f"Validation generations field 'param_sets' must be a list: {generations_path}")

    pre_computed_selected_params = payload.get("selected_params")
    already_labeled = bool(outputs and "response_refusal" in outputs[0])

    print(f"[INFO] Benchmark config: {method_config.path.resolve()}")
    print(f"[INFO] Backend: {backend}")
    print(f"[INFO] Loading validation generations: {generations_path.resolve()}")

    wildguard_status: dict[str, object] = payload.get("wildguard") or {"status": "ok"}
    if already_labeled:
        print(f"[INFO] Outputs already labeled — skipping WildGuard.")
    else:
        print(f"[INFO] Running WildGuard on {len(outputs)} validation outputs...")
        try:
            outputs = attach_wildguard_labels(outputs)
        except Exception as exc:
            wildguard_status = {"status": "error", "message": str(exc)}
            print(f"[ERROR] WildGuard labeling failed: {exc}")

    outputs, outputs_by_combo, results, wildguard_status = compute_results_from_outputs(
        param_sets=param_sets,
        all_combo_outputs=outputs,
        wildguard_status=wildguard_status,
    )
    # Rank all evaluated combinations and keep the best successful result, if one exists.
    sorted_results, best_result = sort_and_select_best(results)

    # Save a compact artifact containing only the best combo labeled outputs.
    if best_result is not None:
        best_combo_outputs = strip_internal_fields(outputs_by_combo[best_result["combo_index"]])
        best_outputs_payload = {
            "model_name": payload["model_name"],
            "backend": payload["backend"],
            "config_path": payload["config_path"],
            "validation_input_path": payload["validation_input_path"],
            "selected_params": best_result["params"],
            "metrics": best_result["metrics"],
            "outputs": best_combo_outputs,
        }
    else:
        best_outputs_payload = {
            "model_name": payload["model_name"],
            "backend": payload["backend"],
            "config_path": payload["config_path"],
            "validation_input_path": payload["validation_input_path"],
            "selected_params": None,
            "metrics": None,
            "outputs": [],
            "error": "No valid validation result was produced.",
        }
    write_json(best_outputs_path, best_outputs_payload)

    # Also update the original generations file so it now contains the WildGuard labels and labeling status for every evaluated output.
    payload["outputs"] = outputs
    payload["wildguard"] = wildguard_status
    write_json(generations_path, payload)

    # Finally write the top-level validation report that will be used by inference when auto-loading the selected validation hyperparameters.
    report_payload = {
        "model_name": payload["model_name"],
        "backend": payload["backend"],
        "config_path": payload["config_path"],
        "validation_input_path": payload["validation_input_path"],
        "generated_at_utc": payload.get("generated_at_utc"),
        "prompt_settings": payload.get("prompt_settings"),
        "search": payload.get("search", {}),
        "wildguard": wildguard_status,
        "selected_params": pre_computed_selected_params or (None if best_result is None else best_result["params"]),
        "best_result": best_result,
        "results": sorted_results,
        "best_outputs_path": str(best_outputs_path.resolve()),
    }
    if payload.get("adasteer_calibration"):
        report_payload["adasteer_calibration"] = payload["adasteer_calibration"]
    
    if offline_validation_payload is not None:
        report_payload["offline_validation"] = offline_validation_payload
    write_json(report_path, report_payload)

    print(f"[DONE] Saved validation report to {report_path.resolve()}")
    print(f"[DONE] Saved best validation outputs to {best_outputs_path.resolve()}")

    if backend == "adasteer":
        from collections import defaultdict
        fitted_w_c = fitted_b_c = None

        # ── Beta / benign analysis ────────────────────────────────────────────
        BENIGN_CATEGORIES = {"benign_harmful_words", "benign_same_structure"}
        beta_data: dict = defaultdict(lambda: {"beta_pos": None, "by_lambda": {}})

        for row in outputs:
            if row.get("category") not in BENIGN_CATEGORIES:
                continue
            combo_idx = row.get("_combo_idx")
            lambda_c = param_sets[combo_idx].get("lambda_c") if combo_idx is not None else None
            beta_pos = row.get("adasteer_beta_pos")
            compliant = row.get("response_refusal") != "refusal"
            pid = row.get("prompt")

            if beta_pos is not None and beta_data[pid]["beta_pos"] is None:
                beta_data[pid]["beta_pos"] = beta_pos
            if lambda_c is not None:
                beta_data[pid]["by_lambda"][float(lambda_c)] = compliant

        beta_rows = []
        for pid, data in sorted(beta_data.items(), key=lambda x: str(x[0])):
            beta_pos = data["beta_pos"]
            # Only include prompts refused at some λ≤0 (real refusal→compliance transition)
            if not any(not compliant for lc, compliant in data["by_lambda"].items() if lc <= 0.0):
                continue
            compliant_lambdas = [lc for lc, compliant in data["by_lambda"].items() if compliant]
            print(compliant_lambdas)
            min_lambda = min(compliant_lambdas) if compliant_lambdas else None
            beta_rows.append((pid, beta_pos, min_lambda))

        col_w = (12, 14, 16)
        header = f"{'prompt_id':>{col_w[0]}} | {'beta_pos':>{col_w[1]}} | {'min_lambda_c (compliance)':>{col_w[2]}}"
        print(f"\n{header}")
        print("-" * (sum(col_w) + 6))
        for pid, beta_pos, min_lambda in beta_rows:
            beta_str = f"{beta_pos:.4f}" if beta_pos is not None else "N/A"
            lambda_str = f"{min_lambda}" if min_lambda is not None else "never refused"
            print(f"{str(pid[:50]):>{col_w[0]}} | {beta_str:>{col_w[1]}} | {lambda_str:>{col_w[2]}}")

        import numpy as np
        fit_rows_beta = [(pos, lam) for _, pos, lam in beta_rows if pos is not None and lam is not None]
        if len(fit_rows_beta) >= 2:
            x = np.array([pos for pos, _ in fit_rows_beta])
            y = np.array([lam for _, lam in fit_rows_beta])
            w_c, intercept = np.polyfit(x, y, 1)
            b_c = -intercept / w_c if abs(w_c) > 1e-12 else float("nan")
            fitted_w_c, fitted_b_c = w_c, b_c
            y_pred = w_c * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"\n[Regression]  lambda_min = w_c * (pos - b_c)")
            print(f"  w_c = {w_c:.6f}")
            print(f"  b_c = {b_c:.6f}")
            print(f"  R²  = {r2:.4f}  (n={len(fit_rows_beta)})")

            import matplotlib.pyplot as plt
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = w_c * x_line + intercept
            plt.figure()
            plt.scatter(x, y, label="data", zorder=3)
            plt.plot(x_line, y_line, label=f"fit: w_c={w_c:.4f}, b_c={b_c:.4f}, R²={r2:.3f}")
            plt.xlabel("adasteer_beta_pos")
            plt.ylabel("min lambda_c for refusal")
            plt.title("AdaSteer: min refusal lambda_c vs probe position (benign)")
            plt.legend()
            plt.tight_layout()
            plot_path = REPO_ROOT / "plot_beta.pdf"
            plt.savefig(plot_path)
            print(f"[DONE] Saved plot to {plot_path.resolve()}")
        else:
            print("\n[Regression] Not enough data points to fit.")

        if fitted_w_c is not None:
            updated_params = dict(report_payload.get("selected_params") or {})
            updated_params["w_c"] = fitted_w_c
            updated_params["b_c"] = fitted_b_c
            report_payload["selected_params"] = updated_params
            write_json(report_path, report_payload)
            print(f"[DONE] Updated selected_params in report with fitted w_c/b_c")


if __name__ == "__main__":
    main()
