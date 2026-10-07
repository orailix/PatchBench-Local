"""
Build the final dataset from selected-70-harmful and local_variation outputs.

For each model:
  - Reads ../PatchBench/data_processing/selected-70-harmful/{stem}.json (70 samples)
  - Reads ../PatchBench/data_processing/local_variation/{stem}.json (local prompts)
  - Splits 70 samples into 50 train / 20 validation (or adaptively in test mode)
  - Writes to ../PatchBench/final_dataset/{folder_name}/:
      row_data.json       — all samples (id, prompt, answer, label), excluding validation ids
      test.json           — local variations for the train samples
      validation.json     — local variations for the validation samples
"""

import json
from pathlib import Path

SELECTED_DIR   = Path("../PatchBench/data_processing/selected-70-harmful")
VARIATION_DIR  = Path("../PatchBench/data_processing/local_variation")
WILDGUARD_DIR  = Path("../PatchBench/data_processing/wildguard_inference")
OUTPUT_BASE    = Path("../PatchBench/final_dataset")

TRAIN_SIZE = 50
VAL_SIZE   = 20

STEM_TO_FOLDER = {
    "Qwen__Qwen2.5-3B-Instruct": "qwen2.5-3B-Instruct",
    "qwen__Qwen2.5-3B-Instruct": "qwen2.5-3B-Instruct",
    "Qwen__Qwen2.5-14B-Instruct": "qwen2.5-14B-Instruct",
    "qwen__Qwen2.5-14B-Instruct": "qwen2.5-14B-Instruct",
    "google__gemma-3-4b-it": "gemma-3-4b-it",
    "google__gemma-3-12b-it": "gemma-3-12b-it",
    "meta-llama__Llama-3.1-8B-Instruct": "llama-3.1-8B-Instruct",
    "meta-llama__Llama-4-Scout-17B-16E-Instruct": "llama4_scout-it",
    "llama4_scout_outputs-it": "llama4_scout-it",
    "mistralai__Ministral-3-14B-Instruct-2512": "ministral-3-14B-Instruct-2512",
    "mistralai__Mistral-7B-Instruct-v0.3": "mistral-7B-Instruct-v0.3",
}


def resolve_folder_name(stem: str) -> str:
    """Map the model stem to the canonical folder name expected by benchmark scripts."""
    if stem in STEM_TO_FOLDER:
        return STEM_TO_FOLDER[stem]
    for k, v in STEM_TO_FOLDER.items():
        if k.lower() == stem.lower():
            return v
    if "__" in stem:
        model_part = stem.split("__", 1)[1]
        return model_part[0].lower() + model_part[1:]
    return stem


def find_file(directory: Path, stem: str) -> Path | None:
    """Find a json file matching stem, with case-insensitive fallback."""
    p = directory / f"{stem}.json"
    if p.exists():
        return p
    if directory.exists():
        stem_lower = stem.lower()
        for f in directory.glob("*.json"):
            if f.stem.lower() == stem_lower:
                return f
    return None


def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_model(stem: str) -> None:
    selected_path = find_file(SELECTED_DIR, stem)
    variation_path = find_file(VARIATION_DIR, stem)
    wildguard_path = find_file(WILDGUARD_DIR, stem)

    if not selected_path:
        print(f"[SKIP] No selected file for {stem}")
        return

    samples = load_json(selected_path)

    # Adaptive split for small sample counts (e.g. test mode)
    if len(samples) < TRAIN_SIZE + VAL_SIZE:
        split_idx = max(1, len(samples) // 2)
        train_samples = samples[:split_idx]
        val_samples   = samples[split_idx:]
    else:
        train_samples = samples[:TRAIN_SIZE]
        val_samples   = samples[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

    val_ids = {str(s["id"]) for s in val_samples}
    if wildguard_path and wildguard_path.exists():
        wildguard_data = load_json(wildguard_path)
        row_data = [
            {"id": s["id"], "prompt": s["prompt"], "answer": s["answer"], "label": s["label"]}
            for s in wildguard_data
            if str(s["id"]) not in val_ids
        ]
    else:
        raise FileNotFoundError(f"No wildguard file for {stem} in {WILDGUARD_DIR}")

    # test.json / validation.json — local variations if available, raw split otherwise
    if variation_path and variation_path.exists():
        variations = load_json(variation_path)
        variation_by_id = {str(v["id"]): v for v in variations}

        def get_variations(split: list) -> list:
            result = []
            for s in split:
                sid = str(s["id"])
                if sid in variation_by_id:
                    result.append(variation_by_id[sid])
                else:
                    print(f"  [WARN] No local variation found for id {sid}")
            return result

        test_data = get_variations(train_samples)
        val_data  = get_variations(val_samples)
    else:
        raise FileNotFoundError(f"No local variation file for {stem} in {VARIATION_DIR}")

    folder_name = resolve_folder_name(stem)
    out_dir = OUTPUT_BASE / folder_name
    save_json(row_data,  out_dir / "row_data.json")
    save_json(test_data, out_dir / "test.json")
    save_json(val_data,  out_dir / "validation.json")
    print(f"[DONE] {stem} -> {folder_name}: {len(row_data)} row samples → {len(test_data)} test, {len(val_data)} validation")


if __name__ == "__main__":
    stems = [p.stem for p in sorted(SELECTED_DIR.glob("*.json"))]

    if not stems:
        print(f"No files found in {SELECTED_DIR}")
    else:
        print(f"Found {len(stems)} models: {stems}")
        for stem in stems:
            build_model(stem)
        print("\nAll models processed.")

