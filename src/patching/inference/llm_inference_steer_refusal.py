import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.utils import logging

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.patching.inference.benchmark_utils import (
    INFERENCE_RUNNER_KEYS,
    default_benchmark_input_path,
    default_benchmark_output_path,
    default_validation_report_path,
    load_selected_validation_params,
)
from src.models.utils import add_padding_token, get_model, get_tokenizer, repo_to_latest_snapshot_dir
from src.patching.steering import (
    BatchContext,
    InferenceContext,
    SteeringArtifacts,
    get_steering_model,
)
from src.patching.utils import (
    PromptTemplateSettings,
    load_benchmark_method_config_from_path,
    format_user_prompt,
    resolve_dtype,
)

logging.set_verbosity_error()


def build_parser() -> argparse.ArgumentParser:
    # Main benchmark inference entrypoint: choose the model/config pair and optionally disable auto-loading of hyperparameters selected by validation.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the benchmark YAML config.",
    )
    parser.add_argument(
        "--no-validation-selection",
        action="store_true",
        help="Do not auto-load selected validation hyperparameters from the saved validation report.",
    )
    return parser


def _build_method_kwargs(config_defaults: dict[str, object]) -> dict[str, object]:
    # Separate method specific steering settings from generic runner settings such as batch size, dtype, and prompt-formatting options.
    return {
        key: value
        for key, value in config_defaults.items()
        if key not in INFERENCE_RUNNER_KEYS and value is not None
    }


def main() -> None:
    # Load the benchmark config, then optionally replace the config's default validation selected hyperparameters with the best ones saved by a previous validation run.
    args = build_parser().parse_args()
    method_config = load_benchmark_method_config_from_path(args.config, model_name=args.model)
    backend = method_config.method_name
    validation_selected = method_config.selected_params()
    validation_report_path = default_validation_report_path(args.model, backend)
    selected_params_source = "config defaults"
    if not args.no_validation_selection:
        report_selected = load_selected_validation_params(args.model, backend)
        if report_selected:
            validation_selected = report_selected
            selected_params_source = f"validation report: {validation_report_path.resolve()}"
        elif validation_report_path.exists():
            selected_params_source = (
                f"validation report exists but has no selected params: {validation_report_path.resolve()}"
            )
    
    # Inference defaults control the runner itself, while validation-selected values provide the steering hyperparameters to apply at test time.
    config_defaults = {
        **method_config.fixed_params("inference"),
        **validation_selected,
    }
    
    batch_size = int(config_defaults.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("inference_fixed.batch_size must be >= 1.")
    
    max_input_tokens = int(config_defaults.get("max_input_tokens", 4096))
    max_new_tokens = int(config_defaults.get("max_new_tokens", 250))
    dtype_name = str(config_defaults.get("dtype", "auto"))
    prompt_settings = PromptTemplateSettings(
        use_chat_template=bool(config_defaults.get("use_chat_template", True)),
        system_prompt=config_defaults.get("system_prompt", "You are a helpful assistant."),
        add_generation_prompt=bool(config_defaults.get("add_generation_prompt", True)),
    )

    # Load the benchmark prompts for this model and prepare the output path.
    in_path = default_benchmark_input_path(args.model)
    if not in_path.exists():
        raise FileNotFoundError(f"Benchmark input JSON not found: {in_path}")
    records = json.loads(in_path.read_text(encoding="utf-8"))

    out_path = default_benchmark_output_path(
        model_name=args.model,
        backend=backend,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve dtype and print the effective inference setup before loading the  model
    dtype = resolve_dtype(dtype_name)
    print("[INFO] Using dtype:", dtype)
    print(f"[INFO] Benchmark config: {method_config.path.resolve()}")
    print(f"[INFO] Benchmark input: {in_path.resolve()}")
    print(f"[INFO] Benchmark output: {out_path.resolve()}")
    print(f"[INFO] Hyperparameter source: {selected_params_source}")
    print(f"[INFO] Selected hyperparameters: {validation_selected}")

    # Load the base model/tokenizer, then let the selected steering class wrap them through the common `InferenceContext` interface.
    snapshot_path = Path(repo_to_latest_snapshot_dir(args.model))
    tokenizer = get_tokenizer(args.model, snapshot_path)
    tokenizer.padding_side = "left"
    tokenizer = add_padding_token(tokenizer, args.model)

    print(f"[INFO] Backend: {backend}")
    model = get_model(args.model, snapshot_path, dtype=dtype, device_map="auto")
    steering = get_steering_model(backend)
    inference_context = InferenceContext(
        method_name=backend,
        model_name=args.model,
        snapshot_path=snapshot_path,
        dtype=dtype,
        prompt_settings=prompt_settings,
        artifacts=SteeringArtifacts(),
        method_kwargs=_build_method_kwargs(config_defaults),
    )
    prepared = steering.prepare_model(
        model=model,
        tokenizer=tokenizer,
        context=inference_context,
    )
    tokenizer = prepared.tokenizer
    prepared_model = prepared.model

    # Log the final promptformatting choices because they materially affect the exact text sent to the model.
    print(
        "[INFO] Prompt formatting: "
        f"use_chat_template={prompt_settings.use_chat_template}, "
        f"system_prompt={prompt_settings.system_prompt!r}, "
        f"add_generation_prompt={prompt_settings.add_generation_prompt}"
    )

    # Benchmark inference is deterministic by default so results are comparable across methods and runs.
    gen_cfg = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    outputs = []

    for batch_start in tqdm(range(0, len(records), batch_size), desc="Running", unit="prompt"):
        batch = records[batch_start : batch_start + batch_size]
        prompts_for_model = []
        ids = []

        # Format each benchmark prompt exactly as it should be sent to the model,including chat templating and any system prompt.
        for i, row in enumerate(batch):
            prompt_for_model = row["prompt"]
            chat_text = format_user_prompt(
                tokenizer=tokenizer,
                user_prompt=prompt_for_model,
                settings=prompt_settings,
            )
            prompts_for_model.append(chat_text)
            ids.append(row.get("id", batch_start + i))

        # Tokenize the batch and keep prompt lengths so steering methods can distinguish prompt tokens from generated continuation tokens.
        inputs = tokenizer(
            prompts_for_model,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        prompt_lengths = None
        if "attention_mask" in inputs:
            prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        inputs = inputs.to(prepared_model.device)

        batch_context = BatchContext(
            rows=batch,
            prompt_texts=prompts_for_model,
            inputs=inputs,
            prompt_lengths=prompt_lengths,
        )
        # The steering backend owns any batch-local setup/cleanup around generation, which keeps this script method-agnostic.
        steering.before_batch(prepared, batch_context)
        with torch.no_grad():
            out = steering.generate_batch(prepared, batch_context, gen_cfg)
        steering.after_batch(prepared, batch_context, out)

        # Strip the prompt tokens from each sequence and store only the generated answer, while keeping both the original row and the exact formatted prompt that was sent to the model.
        input_len = inputs["input_ids"].shape[1]
        for k in range(len(batch)):
            gen_tokens = out[k][input_len:]
            answer = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            result = dict(batch[k])
            result["id"] = ids[k]
            result["raw_prompt"] = prompts_for_model[k]
            result["answer"] = answer
            outputs.append(result)

    # Some steering wrappers keep mutable state on the prepared model. Reset it once inference is finished.
    reset_fn = getattr(prepared_model, "reset_leash_to_default", None)
    if callable(reset_fn):
        reset_fn()

    # Save the raw benchmark generations. Labeling and metric computation happen in 'label_benchmark_with_wildguard.py'.
    out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved {len(outputs)} answers to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
