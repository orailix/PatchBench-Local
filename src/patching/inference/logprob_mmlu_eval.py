import argparse
import json
import sys
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.utils import add_padding_token, get_model, get_tokenizer, repo_to_latest_snapshot_dir
from src.patching.steering import (
    InferenceContext,
    SteeringArtifacts,
    get_steering_model,
)
from src.patching.utils import (
    PromptTemplateSettings,
    load_benchmark_method_config_from_path,
    resolve_dtype,
)

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

ANSWER_LABELS = ["A", "B", "C", "D"]


def _subject_to_display(subject: str) -> str:
    return subject.replace("_", " ")


def _format_example(question: str, choices: list[str], answer_idx: int | None = None) -> str:
    lines = [question]
    for label, choice in zip(ANSWER_LABELS, choices):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    if answer_idx is not None:
        lines[-1] += f" {ANSWER_LABELS[answer_idx]}"
    return "\n".join(lines)


def _build_prompt(subject: str, few_shot_rows: list, test_row: dict) -> str:
    header = (
        f"The following are multiple choice questions (with answers) about "
        f"{_subject_to_display(subject)}.\n\n"
    )
    shots = [_format_example(row["question"], row["choices"], row["answer"]) for row in few_shot_rows]
    shots.append(_format_example(test_row["question"], test_row["choices"], answer_idx=None))
    return header + "\n\n".join(shots)


def _get_answer_token_ids(tokenizer) -> list[int]:
    ids = []
    for letter in ANSWER_LABELS:
        encoded = tokenizer.encode(f" {letter}", add_special_tokens=False)
        ids.append(encoded[-1])
    return ids


def _get_input_device(model: torch.nn.Module) -> torch.device:
    inner = getattr(model, "model", model)
    try:
        return next(inner.parameters()).device
    except StopIteration:
        pass
    for p in model.parameters():
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _score_answers_logprob(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    answer_token_ids: list[int],
    is_alphasteer: bool = False,
) -> list[int]:

    device = _get_input_device(model)

    # ── AlphaSteer Gemma: chat-template logprob (one-by-one, no padding) ─────
    if is_alphasteer:
        forward_model = (
            getattr(model, "model", model)
            if type(model).forward is nn.Module.forward
            else model
        )
        predictions = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            single_inputs = tokenizer(
                [formatted],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=4096,
                add_special_tokens=False,
            ).to(device)
            logits = forward_model(**single_inputs).logits
            last_logits = logits[0, -1, :]
            answer_scores = last_logits[answer_token_ids]
            predictions.append(answer_scores.argmax(dim=-1).item())
        return predictions

    # ── Tokenise raw-text prompts for all other methods ──────────────────────
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
    ).to(device)

    # ── CAST: reset LeashLayer ────────────────────────────────────────────────
    try:
        from collections import defaultdict
        from src.patching.CAST.activation_steering.leash_layer import LeashLayer
        LeashLayer.condition_met.clear()
        LeashLayer.forward_calls.clear()
        LeashLayer.condition_similarities = defaultdict(lambda: defaultdict(float))
        LeashLayer.prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    except ImportError:
        pass

    if type(model).forward is nn.Module.forward:
        forward_model = getattr(model, "model", model)
    else:
        forward_model = model

    # ── AdaSteer: 2-pass ──────────────────────────────────────────────────────
    if hasattr(model, "reset_alpha"):
        model.reset_alpha()
        forward_model(**inputs)

    logits = forward_model(**inputs).logits
    last_logits = logits[:, -1, :]
    answer_scores = last_logits[:, answer_token_ids]
    predictions = answer_scores.argmax(dim=-1).tolist()
    return predictions


def evaluate_subject(
    subject: str,
    model: torch.nn.Module,
    tokenizer,
    answer_token_ids: list[int],
    num_few_shot: int = 5,
    batch_size: int = 8,
    is_alphasteer: bool = False,
) -> dict:
    dev_data = list(load_dataset("cais/mmlu", subject, split="dev"))
    test_data = list(load_dataset("cais/mmlu", subject, split="test"))
    few_shot_rows = dev_data[:num_few_shot]

    all_prompts = [_build_prompt(subject, few_shot_rows, row) for row in test_data]
    all_labels = [row["answer"] for row in test_data]

    predictions = []
    for start in range(0, len(all_prompts), batch_size):
        batch_preds = _score_answers_logprob(
            model, tokenizer,
            all_prompts[start : start + batch_size],
            answer_token_ids,
            is_alphasteer=is_alphasteer,
        )
        predictions.extend(batch_preds)

    correct = sum(p == g for p, g in zip(predictions, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0
    return {
        "subject": subject,
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": len(all_labels),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MMLU 5-shot log-prob eval for hook-based steering (AlphaSteer, AdaSteer)"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config")
    parser.add_argument("--output-dir", default="./mmlu_steered_output")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-few-shot", type=int, default=5)
    parser.add_argument("--subjects", nargs="+", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    method_config = load_benchmark_method_config_from_path(args.config, model_name=args.model)
    backend = method_config.method_name
    config_defaults = {
        **method_config.fixed_params("inference"),
        **method_config.selected_params(),
    }

    from src.patching.inference.benchmark_utils import (
        load_selected_validation_params,
        default_validation_generations_path,
    )
    report_selected = load_selected_validation_params(args.model, backend)
    if report_selected:
        config_defaults.update(report_selected)
        print(f"[INFO] Loaded validation report params: {report_selected}")
    else:
        gen_path = default_validation_generations_path(args.model, backend)
        if gen_path.exists():
            gen_data = json.loads(gen_path.read_text())
            param_sets = gen_data.get("param_sets", [])
            if param_sets:
                config_defaults.update(param_sets[0])
                print(f"[INFO] Loaded params from validation generations: {param_sets[0]}")
            else:
                print("[INFO] No param_sets in generations file — using yaml defaults.")
        else:
            print("[INFO] No validation report found — using yaml defaults only.")

    dtype = resolve_dtype(str(config_defaults.get("dtype", "auto")))
    _runner_keys = {"batch_size", "max_input_tokens", "max_new_tokens", "dtype",
                    "use_chat_template", "system_prompt", "add_generation_prompt"}
    method_kwargs = {k: v for k, v in config_defaults.items() if k not in _runner_keys and v is not None}

    print(f"[INFO] Model:         {args.model}")
    print(f"[INFO] Backend:       {backend}")
    print(f"[INFO] Scoring:       log-prob (cais/mmlu)")
    print(f"[INFO] dtype:         {dtype}")
    print(f"[INFO] method_kwargs: {method_kwargs}")

    snapshot_path = Path(repo_to_latest_snapshot_dir(args.model))
    tokenizer = get_tokenizer(args.model, snapshot_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer = add_padding_token(tokenizer, args.model)

    model = get_model(args.model, snapshot_path, dtype=dtype, device_map="auto")

    prompt_settings = PromptTemplateSettings(
        use_chat_template=True,
        system_prompt=False,
        add_generation_prompt=False,
    )
    steering = get_steering_model(backend)
    inference_context = InferenceContext(
        method_name=backend,
        model_name=args.model,
        snapshot_path=snapshot_path,
        dtype=dtype,
        prompt_settings=prompt_settings,
        artifacts=SteeringArtifacts(),
        method_kwargs=method_kwargs,
    )
    prepared = steering.prepare_model(model=model, tokenizer=tokenizer, context=inference_context)
    tokenizer = prepared.tokenizer
    steered_model = prepared.model
    steered_model.eval()

    # ── Detect method and forward strategy ───────────────────────────────────
    uses_2pass    = hasattr(steered_model, "reset_alpha")
    is_alphasteer = "alphasteer" in backend.lower()

    model_type = steered_model.config.model_type.lower()                      
    alphasteer_needs_chat_template = is_alphasteer and "gemma" in model_type

    if uses_2pass:
        print("[INFO] Forward strategy: AdaSteer 2-pass (detection + scoring)")
    elif alphasteer_needs_chat_template:
        print(f"[INFO] Forward strategy: AlphaSteer logprob — Gemma: chat template, one-by-one")
    elif is_alphasteer:
        print(f"[INFO] Forward strategy: AlphaSteer logprob — {model_type}: raw text, batché")
    else:
        print("[INFO] Forward strategy: standard logprob (CAST/AST/unsteered)")

    answer_token_ids = _get_answer_token_ids(tokenizer)
    print(f"[INFO] Answer token ids: A={answer_token_ids[0]} B={answer_token_ids[1]} "
          f"C={answer_token_ids[2]} D={answer_token_ids[3]}")
    decoded = [tokenizer.decode([tid]) for tid in answer_token_ids]
    print(f"[INFO] Answer tokens decoded: {decoded}  (expect [' A',' B',' C',' D'] or similar)")

    subjects = args.subjects or MMLU_SUBJECTS
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for subject in tqdm(subjects, desc="Subjects"):
        r = evaluate_subject(
            subject, steered_model, tokenizer, answer_token_ids,
            args.num_few_shot, args.batch_size,
            is_alphasteer=alphasteer_needs_chat_template, 
        )
        results.append(r)
        print(f"  {subject}: {r['accuracy']:.4f} ({r['num_correct']}/{r['num_total']})")

    total_correct = sum(r["num_correct"] for r in results)
    total_questions = sum(r["num_total"] for r in results)
    overall_accuracy = total_correct / total_questions if total_questions else 0.0

    summary = {
        "model": args.model,
        "backend": backend,
        "config": args.config,
        "num_few_shot": args.num_few_shot,
        "prompt_format": "raw_text_no_chat_template",
        "dataset": "cais/mmlu",
        "scoring": "logprob_last_token_argmax",
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "per_subject": results,
    }

    out_file = out_dir / f"{args.model.replace('/', '--')}_{backend}_mmlu_{args.num_few_shot}shot_logprob.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[DONE] Overall MMLU accuracy: {overall_accuracy:.4f} ({total_correct}/{total_questions})")
    print(f"[DONE] Results saved to: {out_file.resolve()}")


if __name__ == "__main__":
    main()