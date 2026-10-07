#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


DEFAULT_SOURCE_JSON = "benchmark_data/training_data/CAST/condition_vector/condition_patching.json"
DEFAULT_PROMPTS_DIR = "benchmark_data/training_data/prompts-to-fix"

COMMON_SYSTEM_PREFIXES = (
    "You are a helpful assistant.\n\n",
    "You are a helpful assistant.\n",
)


def _strip_common_system_prefix(text: str) -> str:
    for prefix in COMMON_SYSTEM_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text.strip()


def _extract_chatml_user_prompt(text: str) -> str | None:
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, flags=re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def _extract_llama_user_prompt(text: str) -> str | None:
    match = re.search(
        r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    return match.group(1).strip()


def _extract_gemma_user_prompt(text: str) -> str | None:
    match = re.search(r"<start_of_turn>user\n(.*?)<end_of_turn>", text, flags=re.DOTALL)
    if match is None:
        return None
    return _strip_common_system_prefix(match.group(1))


def _extract_mistral_user_prompt(text: str) -> str | None:
    match = re.search(r"\[INST\]\s*(.*?)\s*\[/INST\]", text, flags=re.DOTALL)
    if match is None:
        return None
    return _strip_common_system_prefix(match.group(1))


def extract_user_prompt(text: str) -> str:
    for extractor in (
        _extract_chatml_user_prompt,
        _extract_llama_user_prompt,
        _extract_gemma_user_prompt,
        _extract_mistral_user_prompt,
    ):
        extracted = extractor(text)
        if extracted is not None:
            return extracted.strip()
    return text.strip()


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def build_unsafe_prompts(prompt_rows: list[dict], prompt_mode: str) -> list[str]:
    prompts: list[str] = []
    for index, row in enumerate(prompt_rows):
        if not isinstance(row, dict):
            raise TypeError(
                f"Expected an object at index {index}, got {type(row).__name__}."
            )
        prompt = row.get("prompt")
        if prompt is None:
            raise KeyError(f"Missing 'prompt' field at index {index}.")
        if prompt_mode == "raw":
            prompts.append(prompt.strip())
        else:
            prompts.append(extract_user_prompt(prompt))
    return prompts


def build_variant_rows(train_rows: list[dict], unsafe_prompts: list[str]) -> list[dict[str, str]]:
    row_count = min(len(train_rows), len(unsafe_prompts))
    variant_rows: list[dict[str, str]] = []
    for index in range(row_count):
        base_row = train_rows[index]
        if not isinstance(base_row, dict):
            raise TypeError(
                f"Expected a training object at index {index}, got {type(base_row).__name__}."
            )
        safe_text = base_row.get("harmless")
        if safe_text is None:
            raise KeyError(f"Missing 'harmless' field at training index {index}.")
        variant_rows.append(
            {
                "unsafe": unsafe_prompts[index],
                "safe": safe_text,
            }
        )
    return variant_rows


def generate_variants(
    source_json: Path,
    prompts_dir: Path,
    output_dir: Path,
    prompt_mode: str,
) -> list[tuple[str, int]]:
    source_payload = load_json(source_json)
    if not isinstance(source_payload, dict):
        raise TypeError("The source CAST JSON must be a top-level object.")

    train_rows = source_payload.get("train")
    if not isinstance(train_rows, list):
        raise TypeError("The source CAST JSON must contain a 'train' list.")

    prompt_files = sorted(prompts_dir.glob("*.json"))
    if not prompt_files:
        raise FileNotFoundError(f"No JSON files found in {prompts_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[tuple[str, int]] = []

    for prompt_file in prompt_files:
        prompt_payload = load_json(prompt_file)
        if not isinstance(prompt_payload, list):
            raise TypeError(f"{prompt_file} must contain a list of prompt rows.")

        unsafe_prompts = build_unsafe_prompts(prompt_payload, prompt_mode=prompt_mode)
        variant_rows = build_variant_rows(train_rows, unsafe_prompts)
        output_payload = {"train": variant_rows}

        output_path = output_dir / prompt_file.name
        output_path.write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries.append((prompt_file.name, len(variant_rows)))

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create CAST condition-vector JSON variants from prompts-to-fix files by "
            "dropping the test split, replacing harmful with unsafe prompts, and "
            "renaming harmless to safe."
        )
    )
    parser.add_argument("--source-json", default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--prompts-dir", default=DEFAULT_PROMPTS_DIR)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where generated JSON files are written. Defaults to the source JSON directory.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["user", "raw"],
        default="user",
        help=(
            "'user' extracts the underlying user request from chat-formatted prompts. "
            "'raw' keeps each prompt string exactly as stored in the prompts-to-fix file."
        ),
    )
    args = parser.parse_args()

    source_json = Path(args.source_json)
    prompts_dir = Path(args.prompts_dir)
    output_dir = Path(args.output_dir) if args.output_dir else source_json.parent

    summaries = generate_variants(
        source_json=source_json,
        prompts_dir=prompts_dir,
        output_dir=output_dir,
        prompt_mode=args.prompt_mode,
    )

    print(f"[INFO] Source train rows available: {len(load_json(source_json)['train'])}")
    print(f"[INFO] Output directory: {output_dir}")
    for filename, row_count in summaries:
        print(f"[DONE] {filename}: wrote {row_count} train pairs")


if __name__ == "__main__":
    main()
