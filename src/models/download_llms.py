from huggingface_hub import snapshot_download
import os

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "allenai/wildguard",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-3-14B-Instruct-2512",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
]


def main() -> None:
    failures = 0
    for i, repo_id in enumerate(MODELS, start=1):
        print(f"\n[{i}/{len(MODELS)}] Downloading: {repo_id}")
        try:
            path = snapshot_download(
                repo_id=repo_id,
                resume_download=True,
                max_workers=1,
                token=True,
            )
            print(f"  ✅ Done. Snapshot path: {path}")
        except Exception as e:
            failures += 1
            print(f"  ❌ Failed: {repo_id}\n     {type(e).__name__}: {e}")

    print(f"\nFinished. Attempted: {len(MODELS)} | Failures: {failures}")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
