"""MMLU comparison report: unsteered vs steered methods across models.

Reads the JSON files produced by logprob_mmlu_eval.py and prints a
summary table with one column per method, plus per-subject breakdowns.

Usage:
    python -m src.patching.inference.mmlu_report \
        --output-dir ./mmlu_steered_output
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def safe_name(model: str) -> str:
    return model.replace("/", "--")


def find_result(output_dir: Path, model: str, backend: str, num_few_shot: int = 5) -> Optional[Path]:
    path = output_dir / f"{safe_name(model)}_{backend}_mmlu_{num_few_shot}shot_logprob.json"
    return path if path.exists() else None


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_models_and_methods(output_dir: Path) -> tuple:
    """Scan output_dir and return (sorted models, sorted non-unsteered methods)."""
    models, methods = set(), set()
    for f in output_dir.glob("*_mmlu_*shot_logprob.json"):
        stem = f.stem  # e.g. Qwen--Qwen2.5-3B_ast_mmlu_5shot_logprob
        parts = stem.split("_mmlu_")[0].rsplit("_", 1)
        if len(parts) == 2:
            model_safe, method = parts
            model = model_safe.replace("--", "/", 1)
            models.add(model)
            if method != "unsteered":
                methods.add(method)
    return sorted(models), sorted(methods)


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def fmt_drop(drop: float) -> str:
    sign = "−" if drop < 0 else "+"
    return f"{sign}{abs(drop) * 100:.2f}pp"


def fmt_rel(rel: float) -> str:
    return f"{rel:+.1f}%"


def subject_delta_table(unst: dict, steered: dict, top_n: int = 10) -> str:
    u_map = {r["subject"]: r["accuracy"] for r in unst["per_subject"]}
    s_map = {r["subject"]: r["accuracy"] for r in steered["per_subject"]}
    deltas = sorted(
        [(subj, u_map[subj], s_map[subj], s_map[subj] - u_map[subj])
         for subj in u_map if subj in s_map],
        key=lambda x: x[3],
    )
    lines = [f"  {'Subject':<45} {'Unsteered':>10} {'Steered':>10} {'Drop':>10}",
             "  " + "─" * 78]
    for subj, u, s, d in deltas[:top_n]:
        lines.append(f"  {subj:<45} {fmt_pct(u):>10} {fmt_pct(s):>10} {fmt_drop(d):>10}")
    return "\n".join(lines)


def run_report(output_dir: Path, models: list, methods: list, num_few_shot: int = 5) -> None:
    col_w = 12  # width per method column

    # ── Overall summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  MMLU 5-shot Evaluation Report")
    print("=" * 100)

    # Header
    header = f"  {'Model':<40} {'Unsteered':>{col_w}}"
    for m in methods:
        header += f"  {m.upper():>{col_w}}"
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    # Per-model rows
    report_rows = []
    skipped = []

    for model in models:
        unst_path = find_result(output_dir, model, "unsteered", num_few_shot)
        if unst_path is None:
            skipped.append((model, "unsteered missing"))
            continue
        unst = load_result(unst_path)
        u = unst["overall_accuracy"]

        row_data = {"model": model, "unsteered": u, "methods": {}}
        line = f"  {model:<40} {fmt_pct(u):>{col_w}}"

        for method in methods:
            m_path = find_result(output_dir, model, method, num_few_shot)
            if m_path is None:
                line += f"  {'—':>{col_w}}"
                continue
            m_res = load_result(m_path)
            s    = m_res["overall_accuracy"]
            drop = s - u
            rel  = drop / u * 100 if u else 0.0
            line += f"  {fmt_pct(s):>{col_w}}"
            row_data["methods"][method] = {"accuracy": s, "drop": drop, "rel": rel, "result": m_res}

        print(line)
        report_rows.append((model, unst, row_data))

    # Averages
    if report_rows:
        print("  " + "─" * (len(header) - 2))
        avg_u = sum(r[2]["unsteered"] for r in report_rows) / len(report_rows)
        avg_line = f"  {'Average':<40} {fmt_pct(avg_u):>{col_w}}"
        avg_by_method = {}
        for method in methods:
            vals = [r[2]["methods"][method]["accuracy"]
                    for r in report_rows if method in r[2]["methods"]]
            if vals:
                avg_s = sum(vals) / len(vals)
                avg_drop = avg_s - avg_u
                avg_rel  = avg_drop / avg_u * 100 if avg_u else 0.0
                avg_line += f"  {fmt_pct(avg_s):>{col_w}}"
                avg_by_method[method] = avg_s
            else:
                avg_line += f"  {'—':>{col_w}}"
        # print(avg_line)

    # ── Per-model subject breakdowns ───────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  Per-model worst-hit subjects (top 10 drops per method)")
    print("=" * 100)

    for model, unst, row_data in report_rows:
        short = model.split("/")[-1]
        u = row_data["unsteered"]
        print(f"\n  ── {short}  (unsteered: {fmt_pct(u)}) " + "─" * 40)
        for method in methods:
            if method not in row_data["methods"]:
                print(f"\n  [{method.upper()}] result missing")
                continue
            md = row_data["methods"][method]
            print(f"\n  [{method.upper()}]  overall {fmt_pct(u)} → {fmt_pct(md['accuracy'])}  ({fmt_drop(md['drop'])})")
            print(subject_delta_table(unst, md["result"], top_n=10))

    # ── Subjects degraded consistently across models (all methods combined) ────
    if len(report_rows) > 1:
        print("\n" + "=" * 100)
        print("  Subjects with most consistent degradation (avg drop across all models × methods)")
        print("=" * 100)
        subject_drops: dict = {}
        total_pairs = 0
        for model, unst, row_data in report_rows:
            u_map = {r["subject"]: r["accuracy"] for r in unst["per_subject"]}
            for method, md in row_data["methods"].items():
                s_map = {r["subject"]: r["accuracy"] for r in md["result"]["per_subject"]}
                for subj in u_map:
                    if subj in s_map:
                        subject_drops.setdefault(subj, []).append(s_map[subj] - u_map[subj])
                total_pairs += 1

        avg_subj = {s: sum(v) / len(v) for s, v in subject_drops.items()}
        worst = sorted(avg_subj.items(), key=lambda x: x[1])[:15]
        print(f"\n  {'Subject':<45} {'Avg drop':>10}")
        print("  " + "─" * 58)
        for subj, d in worst:
            print(f"  {subj:<45} {fmt_drop(d):>10}")

    # ── Save JSON report ───────────────────────────────────────────────────────
    json_report = {
        "methods": methods,
        "models": [
            {
                "model": model,
                "unsteered_accuracy": row_data["unsteered"],
                "methods": {
                    method: {
                        "accuracy": md["accuracy"],
                        "drop_pp":  md["drop"],
                        "relative_drop_pct": md["rel"],
                    }
                    for method, md in row_data["methods"].items()
                },
            }
            for model, _, row_data in report_rows
        ],
    }
    report_path = output_dir / "mmlu_report.json"
    report_path.write_text(json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] Report saved to: {report_path.resolve()}")

    if skipped:
        print(f"[WARN] Skipped: {skipped}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MMLU multi-method comparison report")
    parser.add_argument("--output-dir", type=Path, default=Path("./mmlu_steered_output"))
    parser.add_argument("--models",  nargs="*", default=None,
                        help="Model names. Auto-discovered if omitted.")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Methods to include (e.g. ast cast). Auto-discovered if omitted.")
    parser.add_argument("--num-few-shot", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.models is None or args.methods is None:
        disc_models, disc_methods = discover_models_and_methods(args.output_dir)
        models  = args.models  or disc_models
        methods = args.methods or disc_methods
    else:
        models, methods = args.models, args.methods

    if not models:
        print(f"[ERROR] No result files found in {args.output_dir}")
        return

    run_report(args.output_dir, models, methods, args.num_few_shot)


if __name__ == "__main__":
    main()