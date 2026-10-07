#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.models.download_llms import MODELS
from src.models.utils import repo_to_latest_snapshot_dir, get_tokenizer, add_padding_token

IN_JSON_DEFAULT = "../PatchBench/data_processing/raw_data/redbench_filtered.json"


def build_prompt_for_model(tokenizer, row):
    prompt_for_model = row["prompt"]
    choices = row.get("choices")
    if choices != "None":
        prompt_for_model = (
            f"{prompt_for_model}\n\n"
            "Choices:\n"
            f"{str(choices)}\n\n"
        )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_for_model},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-json", default=IN_JSON_DEFAULT)
    parser.add_argument(
        "--truncate-to",
        type=int,
        default=None,
        help="If set, tokenize with truncation=True and this max length.",
    )
    args = parser.parse_args()

    in_path = Path(args.in_json)
    records = json.loads(in_path.read_text(encoding="utf-8"))

    print(f"[INFO] Loaded {len(records)} rows from {in_path.resolve()}")
    print(f"[INFO] Models to evaluate: {len(MODELS)}")

    overall = {"model": None, "tokens": -1, "row_id": None}

    for model_name in MODELS:
        print(f"\n[INFO] Processing model: {model_name}")
        snapshot_path = repo_to_latest_snapshot_dir(model_name)
        tokenizer = get_tokenizer(model_name, snapshot_path)
        tokenizer = add_padding_token(tokenizer, model_name)

        model_max_tokens = -1
        model_max_row_id = None
        tok_kwargs = {
            "return_attention_mask": False,
            "add_special_tokens": True,
            "truncation": args.truncate_to is not None,
        }
        if args.truncate_to is not None:
            tok_kwargs["max_length"] = args.truncate_to

        for i, row in enumerate(tqdm(
            records,
            desc=f"Tokenizing {model_name}",
            unit="rows",
        )):
            prompt = build_prompt_for_model(tokenizer, row)
            encoded = tokenizer(prompt, **tok_kwargs)
            n_tokens = len(encoded["input_ids"])
            if n_tokens > model_max_tokens:
                model_max_tokens = n_tokens
                model_max_row_id = row.get("id", i)

        print(
            f"[RESULT] {model_name}: max_tokens={model_max_tokens} "
            f"(row_id={model_max_row_id})"
        )

        if model_max_tokens > overall["tokens"]:
            overall = {
                "model": model_name,
                "tokens": model_max_tokens,
                "row_id": model_max_row_id,
            }

    print(
        "\n[OVERALL] "
        f"model={overall['model']} "
        f"max_tokens={overall['tokens']} "
        f"row_id={overall['row_id']}"
    )


if __name__ == "__main__":
    main()
