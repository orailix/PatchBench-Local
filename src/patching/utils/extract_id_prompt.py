#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _extract_from_mapping(payload: dict) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for record_id, record in payload.items():
        if not isinstance(record, dict):
            raise TypeError(
                "Expected each top-level value to be an object containing a 'prompt' field."
            )
        prompt = record.get("prompt")
        if prompt is None:
            raise KeyError(f"Missing 'prompt' for id: {record_id}")
        records.append({"id": str(record_id), "prompt": prompt})
    return records


def _extract_from_list(payload: list) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise TypeError("Expected each list item to be an object with 'id' and 'prompt'.")
        record_id = record.get("id")
        prompt = record.get("prompt")
        if record_id is None:
            raise KeyError(f"Missing 'id' for list item at index {index}.")
        if prompt is None:
            raise KeyError(f"Missing 'prompt' for list item at index {index}.")
        records.append({"id": str(record_id), "prompt": prompt})
    return records


def extract_id_prompt_records(payload: dict | list) -> list[dict[str, str]]:
    if isinstance(payload, dict):
        return _extract_from_mapping(payload)
    if isinstance(payload, list):
        return _extract_from_list(payload)
    raise TypeError("Input JSON must be either an object or a list.")


def build_default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}-id-prompt.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a simplified JSON file containing only 'id' and 'prompt'."
    )
    parser.add_argument("--in-json", required=True, help="Path to the source JSON file.")
    parser.add_argument(
        "--out-json",
        default=None,
        help="Path to the output JSON file. Defaults to '<input>-id-prompt.json'.",
    )
    args = parser.parse_args()

    input_path = Path(args.in_json)
    output_path = Path(args.out_json) if args.out_json else build_default_output_path(input_path)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records = extract_id_prompt_records(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] Loaded {len(records)} records from: {input_path}")
    print(f"[DONE] Wrote id/prompt JSON to: {output_path}")


if __name__ == "__main__":
    main()
