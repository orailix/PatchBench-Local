import json
from pathlib import Path
import argparse
import random
import torch
from loguru import logger
from filelock import FileLock
import numpy as np
from vllm import LLM, SamplingParams
from ..utils.elo_utils import EloGame
from ..utils.utils import generate_messages, parsing_and_saving_results

INPUT_PATH = Path("../PatchBench/data_processing/wildguard_inference/")
OUTPUT_PATH = Path("../PatchBench/data_processing/elo_ranking/")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
METCH_PER_SAMPLE = 50
DO_EVOLUTION_CHECK = False
MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"

def elo_eval_pair(ds: list[dict], match_per_sample: int, model_name: str, verbose: bool = False) -> None:
    """
    Compute ELO scores and ranking from pairwise evaluations

    Args:
        ds: list of dicts with keys id, prompt, answer, label
        match_per_sample: Number of pairwise matches per sample
        verbose: If True, print logs
    """

    #Seed
    np.random.seed(2026) 
    random.seed(2026)
    torch.manual_seed(2026)

    logger.info(
        f"Computing Elo with match_per_sample={match_per_sample}"
    ) if verbose else None

    #Pool
    all_configs = sorted(list(set(item["id"] for item in ds)))
    ranking = EloGame()
    for config in all_configs :
        ranking.addPlayer(config)

    logger.debug(
        f"Total number of samples: {len(all_configs)}"
    ) if verbose else None

    id_to_idx = {item["id"]: i for i, item in enumerate(ds)}
    all_matches = []

    # Determine the number of opponents to sample for each player.
    # Each match involves two players, so selecting half of match_per_sample
    num_opponents = min(match_per_sample // 2, len(all_configs) - 1)

    logger.debug(f"Number of opponent : {num_opponents}") if verbose else None
    total_expected_matches = len(all_configs) * num_opponents

    logger.debug(f"Total expected matches: {total_expected_matches}") if verbose else None

    # For each configuration, select a subset of opponents randomly
    for player_0 in all_configs:
        # Choosing opponents excluding the current player
        opponents = np.random.choice(
            [p for p in all_configs if p != player_0],
            size=min(num_opponents, len(all_configs)-1),
            replace=False # No duplicates
        )
        # Record each match as a tuple, sorted to avoid duplicate pairs
        for player_1 in opponents:
            all_matches.append((player_0, player_1))

    np.random.shuffle(all_matches)

    # Check if there are an history file for previous evaluations
    output_dir = Path("output") / "elo" / f"results_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_file = output_dir / f"matches_history_{model_name}_{match_per_sample}.jsonl"
    evaluated_matches = set()
    if history_file.exists():
        with open(history_file, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    p_a, p_b = record["p_a"], record["p_b"]
                    winner = record["winner"]
                    evaluated_matches.add((p_a, p_b))
                    ranking.recordMatch(p_a, p_b, winner=winner)

    print(f"Matches already evaluated in the history : {len(evaluated_matches)}")

    matches_to_evaluate = [m for m in all_matches if m not in evaluated_matches]

    print(f"Matches to send to vLLM for evaluation: {len(matches_to_evaluate)}")

    # Starting evaluation
    if len(matches_to_evaluate) > 0 :

        # Loadind LLM
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=4,
            dtype="bfloat16",
            max_model_len=4096
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=5,
        )

        tokenizer = llm.get_tokenizer()

        # Loading prompt
        prompts_messages = [generate_messages(ds, id_to_idx, p_a, p_b) for p_a, p_b in matches_to_evaluate]
        prompts_str = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in prompts_messages]

        logger.info("Starting vLLM inference") if verbose else None

        # LLM output
        outputs = llm.generate(prompts_str, sampling_params)

        scores_history = []
        capture_interval = max(1, len(all_configs) // 2)
        match_count = len(evaluated_matches)

        # writing results in a history file
        with open(history_file, "a") as f:
            for match, output in zip(matches_to_evaluate, outputs):
                p_a, p_b = match
                text = output.outputs[0].text.strip()

                if "0" in text and "1" not in text:
                    winner = p_a
                elif "1" in text and "0" not in text:
                    winner = p_b
                else:
                    winner = p_a if text.startswith("0") else p_b
                # Updating ELO score
                ranking.recordMatch(p_a, p_b, winner=winner)
                # saving match
                record = {"p_a": p_a, "p_b": p_b, "winner": winner}
                f.write(json.dumps(record) + "\n")

                match_count += 1
                if match_count % capture_interval == 0 or match_count == len(all_matches):
                    current_scores = {c: ranking.getPlayerRating(c) for c in all_configs}
                    scores_history.append({
                        "match_num": match_count,
                        "scores": current_scores
                    })
        if DO_EVOLUTION_CHECK : 
            evolution_file = output_dir / "elo_evolution.json"
            with evolution_file.open("w") as ef:
                json.dump(scores_history, ef, indent=2)

    # Sorting results by elo score
    result = {"elo_scores": {c: ranking.getPlayerRating(c) for c in all_configs}}
    result["elo_scores"] = dict(sorted(result["elo_scores"].items(), key=lambda item: item[1], reverse=True))
    # Saving results in a json output file
    output_path = OUTPUT_PATH / f"elo_{model_name}.json"
    with FileLock(output_path.with_suffix(".lock")):
        parsing_and_saving_results(ds, result, output_path)
    print("Évaluation terminée avec succès !")


def main(model_name: str):

    file_path = INPUT_PATH / f"{model_name}.json"

    print(f"\n=== ELO for model: {model_name} ===")


    # LOAD WILDGUARD OUTPUT
    with open(file_path, "r") as f:
        data = json.load(f)


    # FILTER UNSAFE
    ds_filtered_unsafe = [
        item for item in data
        if item.get("label") == "unsafe"
    ]

    print(f"Unsafe samples: {len(ds_filtered_unsafe)}")

    if len(ds_filtered_unsafe) == 0:
        print("Skipping (no unsafe samples)")
        return

    # RUN ELO
    elo_eval_pair(
        ds_filtered_unsafe,
        METCH_PER_SAMPLE,
        model_name,
        verbose=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    if args.model_name:
        main(args.model_name)
    else:
        for file_path in sorted(INPUT_PATH.glob("*.json")):
            main(file_path.stem)