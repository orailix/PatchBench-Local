from datasets import load_dataset, get_dataset_config_names
from pathlib import Path
import json

DATASET_ID = "knoveleng/redbench"

EXCLUDED_USER = {
    "Cyberattackassistance",
    "DeMET",
    "DiaSafety",
    "GEST",
    "GandalfignoreInstructions",
    "GandalfSummarization",
    "JADE",
    "LatentJailbreak",
    "MoralExceptQA",
    "ORBench",
    "PhysicalSafetyInstructions",
    "Qharm",
    "SGXSTest",
    "SafeText",
    "ToxiGen",
    "XSTest",
    "Xsafety",
}

OUT_JSON = "../PatchBench/data_processing/raw_data/redbench_filtered.json"
Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Listing configs for {DATASET_ID}...")
all_configs = get_dataset_config_names(DATASET_ID)
print(f"[INFO] Total configs found: {len(all_configs)}")

excluded_lower = {x.lower() for x in EXCLUDED_USER}
kept_configs = [c for c in all_configs if c.lower() not in excluded_lower]
print(f"[INFO] Configs kept: {len(kept_configs)}")

records = []
global_id = 1

for cfg in kept_configs:
    print(f"[INFO] Loading config: {cfg}")
    try:
        ds_dict = load_dataset(DATASET_ID, cfg)
        for split_name, ds in ds_dict.items():
            print(f"   Split '{split_name}' -> {len(ds)} rows")
            for row in ds:
                record = {"id": global_id, "prompt": row["prompt"]}
                choices = row.get("choices")
                if choices and choices != "None":
                    record["choices"] = choices
                records.append(record)
                global_id += 1
    except Exception as e:
        print(f"   [ERROR] {cfg}: {e}")

if not records:
    raise RuntimeError("No data loaded.")

print(f"[INFO] Total records: {len(records)}")
print(f"[INFO] Writing JSON -> {OUT_JSON}")

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"[DONE] Saved {len(records)} records to {OUT_JSON}")
