import json
import os
import shutil
import random
from pathlib import Path

folder_to_filename = {
    'gemma-3-12b-it': 'google__gemma-3-12b-it.json',
    'gemma-3-4b-it': 'google__gemma-3-4b-it.json',
    'llama-3.1-8B-Instruct': 'meta-llama__Llama-3.1-8B-Instruct.json',
    'llama4_scout-it': 'meta-llama__Llama-4-Scout-17B-16E-Instruct.json',
    'ministral-3-14B-Instruct-2512': 'mistralai__Ministral-3-14B-Instruct-2512.json',
    'mistral-7B-Instruct-v0.3': 'mistralai__Mistral-7B-Instruct-v0.3.json',
    'qwen2.5-14B-Instruct': 'Qwen__Qwen2.5-14B-Instruct.json',
    'qwen2.5-3B-Instruct': 'Qwen__Qwen2.5-3B-Instruct.json'
}

# Resolve paths relative to this script's location so it works anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

source_base = os.path.abspath(os.path.join(PROJECT_ROOT, "../PatchBench/final_dataset"))
target_test = os.path.join(PROJECT_ROOT, "benchmark_data/test_data")
target_val = os.path.join(PROJECT_ROOT, "benchmark_data/validation_data")

def flatten_entries(data):
    flattened = []
    for item in data:
        base_id = item['id']
        
        # Add original prompt
        if 'original_prompt' in item:
            flattened.append({
                "id": base_id,
                "prompt": item["original_prompt"],
                "category": "original",
                "type": "original"
            })
            
        groups = item.get('groups', {})
        for category, cat_data in groups.items():
            if isinstance(cat_data, dict):
                for group_type, items in cat_data.items():
                    if isinstance(items, list):
                        for subitem in items:
                            flattened.append({
                                "id": base_id,
                                "prompt": subitem["prompt"],
                                "category": category,
                                "type": subitem.get("type", group_type)
                            })
                    else:
                        print(f"Warning: Unexpected items type: {type(items)} in {base_id}")
            elif isinstance(cat_data, list):
                for subitem in cat_data:
                    flattened.append({
                        "id": base_id,
                        "prompt": subitem["prompt"],
                        "category": category,
                        "type": subitem.get("type", "unknown")
                    })
    return flattened

for folder, filename in folder_to_filename.items():
    source_dir = os.path.join(source_base, folder)
    if not os.path.exists(source_dir):
        alt_folder = filename.replace(".json", "")
        source_dir = os.path.join(source_base, alt_folder)
    test_src = os.path.join(source_dir, "test.json")
    val_src = os.path.join(source_dir, "validation.json")
    
    test_tgt = os.path.join(target_test, filename)
    val_tgt = os.path.join(target_val, filename)
    
    # Process test.json
    if os.path.exists(test_src):
        try:
            with open(test_src, 'r') as f:
                data = json.load(f)
            flat_data = flatten_entries(data)
            os.makedirs(os.path.dirname(test_tgt), exist_ok=True)
            with open(test_tgt, 'w') as f:
                json.dump(flat_data, f, indent=2)
            print(f"Flattened and copied test for {folder} to {test_tgt}")
        except json.JSONDecodeError:
            pass
        
    # Process validation.json
    if os.path.exists(val_src):
        try:
            with open(val_src, 'r') as f:
                data = json.load(f)
            flat_data = flatten_entries(data)
            os.makedirs(os.path.dirname(val_tgt), exist_ok=True)
            with open(val_tgt, 'w') as f:
                json.dump(flat_data, f, indent=2)
            print(f"Flattened and copied validation for {folder} to {val_tgt}")
        except json.JSONDecodeError:
            pass

# --- NEW CAST DATASET LOGIC ---

cast_base       = os.path.join(PROJECT_ROOT, "benchmark_data", "training_data", "CAST")
alphasteer_base = os.path.join(PROJECT_ROOT, "benchmark_data", "training_data", "AlphaSteer")
adasteer_base   = os.path.join(PROJECT_ROOT, "benchmark_data", "training_data", "AdaSteer")

# A) Create subfolder "refusal_vector" and copy files
refusal_dir = os.path.join(cast_base, "refusal_vector")
alphasteer_refusal_dir = alphasteer_base
adasteer_refusal_dir   = adasteer_base
os.makedirs(refusal_dir, exist_ok=True)
os.makedirs(alphasteer_refusal_dir, exist_ok=True)
os.makedirs(adasteer_refusal_dir, exist_ok=True)

alpaca_refusal_src = os.path.abspath(os.path.join(PROJECT_ROOT, "../PatchBench/alpaca/alpaca_for_refusal.json"))
behavior_refusal_src = os.path.abspath(os.path.join(PROJECT_ROOT, "../PatchBench/alpaca/behavior_refusal.json"))
alphasteer_refusal_src = os.path.abspath(os.path.join(PROJECT_ROOT, "../PatchBench/final_dataset/"))
adasteer_refusal_src = os.path.abspath(os.path.join(PROJECT_ROOT, "../PatchBench/final_dataset/"))

if os.path.exists(alpaca_refusal_src):
    shutil.copy(alpaca_refusal_src, os.path.join(refusal_dir, "alpaca_for_refusal.json"))
    print(f"Copied alpaca_for_refusal.json to {refusal_dir}")
else:
    print(f"Warning: missing source file {alpaca_refusal_src}")
if os.path.exists(behavior_refusal_src):
    shutil.copy(behavior_refusal_src, os.path.join(refusal_dir, "behavior_refusal.json"))
    print(f"Copied behavior_refusal.json to {refusal_dir}")
else:
    print(f"Warning: missing source file {behavior_refusal_src}")

# Copy raw_data from the dataset to alphasteer's training set
for root, _, files in os.walk(alphasteer_refusal_src):
    root = Path(root)
    if "row_data.json" in files:
        src_file = root / "row_data.json"
        rel_dir = root.relative_to(alphasteer_refusal_src)
        dst_dir = alphasteer_refusal_dir / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_dir / "row_data.json")
        print(f"Copied {src_file} to {dst_dir / 'row_data.json'}")
        
# Copy raw_data from the dataset to adasteer's training set
for root, _, files in os.walk(adasteer_refusal_src):
    root = Path(root)
    if "row_data.json" in files:
        src_file = root / "row_data.json"
        rel_dir = root.relative_to(adasteer_refusal_src)
        dst_dir = adasteer_refusal_dir / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_dir / "row_data.json")
        print(f"Copied {src_file} to {dst_dir / 'row_data.json'}")
        
# B) Create subfolder "condition_vector" and recreate per-model pair files
condition_dir = os.path.join(cast_base, "condition_vector")
os.makedirs(condition_dir, exist_ok=True)
CONDITION_VECTOR_PAIR_COUNT = 1500
CONDITION_VECTOR_RANDOM_SEED = 42

def select_safe_side_prompts(unique_safe_prompts, pair_count, rng):
    if len(unique_safe_prompts) >= pair_count:
        return rng.sample(unique_safe_prompts, pair_count)

    selected_safe = unique_safe_prompts[:]
    rng.shuffle(selected_safe)
    selected_safe.extend(rng.choices(unique_safe_prompts, k=pair_count - len(selected_safe)))
    return selected_safe

# Load safe prompts from alpaca_for_condition.json
alpaca_cond_src = os.path.abspath(os.path.join(PROJECT_ROOT, "../PatchBench/alpaca/alpaca_for_condition.json"))
alpaca_safe_prompts = []
if os.path.exists(alpaca_cond_src):
    with open(alpaca_cond_src, 'r', encoding='utf-8') as f:
        alpaca_cond_data = json.load(f)
        seen_safe_prompts = set()
        for item in alpaca_cond_data.get('train', []):
            prompt = item.get('safe')
            if not prompt or not prompt.strip() or prompt in seen_safe_prompts:
                continue
            seen_safe_prompts.add(prompt)
            alpaca_safe_prompts.append(prompt)

if alpaca_safe_prompts:
    condition_rng = random.Random(CONDITION_VECTOR_RANDOM_SEED)
    for folder, filename in folder_to_filename.items():
        row_data_src = os.path.join(source_base, folder, "row_data.json")
        if not os.path.exists(row_data_src):
            continue
            
        try:
            with open(row_data_src, 'r', encoding='utf-8') as f:
                row_data_content = json.load(f)
        except json.JSONDecodeError:
            # Could be Git LFS pointer
            print(f"Warning: Could not decode JSON for {row_data_src} (Git LFS pointer?)")
            continue
        
        # Extract prompts from row_data.json for the contrastive "unsafe" side.
        # Safe-labeled fallback prompts are still written to the "unsafe" field below. They are used to fill up to 1500 training pairs
        unsafe_prompts = []
        safe_row_prompts = []
        for item in row_data_content:
            prompt = item.get("prompt", item.get("original_prompt"))
            if not prompt:
                continue

            if item.get("label") == "unsafe":
                unsafe_prompts.append(prompt)
            elif item.get("label") == "safe":
                safe_row_prompts.append(prompt)

        selected_unsafe = unsafe_prompts[:CONDITION_VECTOR_PAIR_COUNT]
        missing_unsafe_count = CONDITION_VECTOR_PAIR_COUNT - len(selected_unsafe)
        if missing_unsafe_count > 0:
            if not safe_row_prompts:
                print(f"Warning: only found {len(selected_unsafe)} usable prompts in {row_data_src}; no safe fallback prompts available.")
                continue

            if len(safe_row_prompts) >= missing_unsafe_count:
                selected_unsafe.extend(condition_rng.sample(safe_row_prompts, missing_unsafe_count))
            else:
                selected_unsafe.extend(safe_row_prompts)
                selected_unsafe.extend(condition_rng.choices(safe_row_prompts, k=missing_unsafe_count - len(safe_row_prompts)))
        condition_rng.shuffle(selected_unsafe)

        if not selected_unsafe:
            print(f"Warning: no condition-vector unsafe-side prompts found for {folder}.")
            continue

        pair_count = len(selected_unsafe)
        selected_safe = select_safe_side_prompts(
            unique_safe_prompts=alpaca_safe_prompts,
            pair_count=pair_count,
            rng=condition_rng,
        )

        selected_unsafe = selected_unsafe[:pair_count]
        
        model_pairs = []
        for safe_p, unsafe_p in zip(selected_safe, selected_unsafe):
            model_pairs.append({
                "safe": safe_p,
                "unsafe": unsafe_p
            })
            
        out_data = {"train": model_pairs}
        # File path formatting uses the same filename as before but in condition_vector
        base_name, ext = os.path.splitext(filename)
        output_filename = f"{base_name}-prompts-to-fix{ext}"
        
        output_path = os.path.join(condition_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)
            
        print(f"Created {output_path} with {pair_count} condition pairs.")

print("Done!")
