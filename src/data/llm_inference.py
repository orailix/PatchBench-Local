#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

from src.models.utils import repo_to_latest_snapshot_dir, get_tokenizer, model_filename, get_model, add_padding_token

from transformers.utils import logging
logging.set_verbosity_error()

IN_JSON_DEFAULT = "../PatchBench/data_processing/raw_data/redbench_filtered.json"
OUT_DIR_DEFAULT = "../PatchBench/data_processing/model_inference"


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--max-input-tokens", type=int, default=4096)
parser.add_argument("--test", type=int, default=0)
parser.add_argument("--in-json", default=IN_JSON_DEFAULT)
parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
args = parser.parse_args()

in_path = Path(args.in_json)
records = json.loads(in_path.read_text(encoding="utf-8"))

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

stem = model_filename(args.model)
out_path = out_dir / f"{stem}.json"

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
print("[INFO] Using dtype:", dtype)

snapshot_path = repo_to_latest_snapshot_dir(args.model) 
tokenizer = get_tokenizer(args.model, snapshot_path)
tokenizer.padding_side = "left"

tokenizer = add_padding_token(tokenizer, args.model)

model = get_model(args.model, snapshot_path, dtype=dtype, device_map="auto")

gen_cfg = GenerationConfig(
    do_sample=False,
    max_new_tokens=250,
)

outputs = []

BATCH_SIZE = args.batch_size
MAX_INPUT_TOKENS = args.max_input_tokens

outputs = []

for batch_start in tqdm(range(0, len(records), BATCH_SIZE), desc="Running", unit="prompt"):
    if args.test != 0 and batch_start > 0:
        break
    
    batch = records[batch_start: batch_start + BATCH_SIZE]

    prompts_for_model = []
    raw_prompts = []
    ids = []

    for i, row in enumerate(batch):
        prompt_for_model = row["prompt"]
        choices = row.get("choices")
        if choices is not None:
            prompt_for_model = (
                f"{prompt_for_model}\n\n"
                "Choices:\n"
                f"{str(choices)}\n\n"
            )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_for_model},
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompts_for_model.append(chat_text)
        raw_prompts.append(row["prompt"])
        ids.append(f"{row.get('id', batch_start + i + 1)}_{stem}")

    inputs = tokenizer(
        prompts_for_model,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            generation_config=gen_cfg,
        )

    input_len = inputs["input_ids"].shape[1]
    for k in range(len(batch)):
        gen_tokens = out[k][input_len:]
        answer = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        outputs.append(
            {
                "id": ids[k],
                "prompt": raw_prompts[k],
                "answer": answer,
            }
        )

out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[DONE] Saved {len(outputs)} answers to: {out_path.resolve()}")
