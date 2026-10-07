from wildguard import load_wildguard
import json
from pathlib import Path

def save_results(results):
    output_dir = Path("output") / "wildguard_results"
    with open(output_dir /"results.json", "w") as f:
        json.dump(results,f, indent=2)

def wildguard_moderate_samples(ds):
    # Load the model 
    print("Loading Wildguard model...")
    wildguard = load_wildguard()

    # Classify the items
    print("Processing items ...")
    samples = []

    for item in ds :
        sample = {
            "id" : item["id"],
            "prompt" : item["prompt"],
            "response" : item["answer"]
        }
        samples.append(sample)

    results = wildguard.classify(samples)

    return results

if __name__ == "__main__" :

    INPUT_PATH = Path("../PatchBench/data_processing/model_inference/")
    OUTPUT_PATH = Path("../PatchBench/data_processing/wildguard_inference")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    for file_path in INPUT_PATH.glob("*.json"):

        model_name = file_path.stem
        print(f"Processing {model_name}")

        with open(file_path, "r") as f:
            samples = json.load(f)

        results = wildguard_moderate_samples(samples)

        enriched = []

        for item, r in zip(samples, results):

            prompt_h = r["prompt_harmfulness"]
            response_h = r["response_harmfulness"]
            refusal = r["response_refusal"]

            label = "unsafe" if response_h == "harmful" else "safe"

            enriched.append({
                "model_name": model_name,
                "id": item["id"],
                "prompt": item["prompt"],
                "answer": item["answer"],
                "prompt_harmfulness": prompt_h,
                "response_harmfulness": response_h,
                "response_refusal": refusal,
                "label": label
            })

        out_file = OUTPUT_PATH / f"{model_name}.json"

        with open(out_file, "w") as f:
            json.dump(enriched, f, indent=2)