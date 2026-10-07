# PatchBench

**PatchBench** is a curated bank of 400 high-confidence jailbreak failures across 8 open-source instruction-tuned models. Starting from 27,870 prompts aggregated from 37 public jailbreak datasets, we use WildGuard filtering, pairwise Elo ranking, and manual verification to retain only unsafe completions that genuinely answer harmful requests.

**PatchBench-Local** is an evaluation protocol that tests whether a patch is behaviourally precise. For each harmful source prompt it generates three families of local neighbours: harmful variants that preserve the malicious intent, benign prompts with matched structure, and benign prompts that reuse the key harmful terms. It evaluates both harmful-neighbour correction and benign-neighbour preservation, distinguishing selective repair from broader local suppression.


## PatchBench Data Pipeline

You can either use the PatchBench dataset directly from HuggingFace (`orailix/PatchBench`) or recreate it from scratch by following the pipeline below.

### Run LLM inference, WildGuard labeling and ELO ranking

The pipeline runs in four steps: export the source prompts, run LLM inference on all models, label outputs with WildGuard, and rank them with ELO. You can run it with :

```bash
./run_data_pipeline.sh
```

At this step, you will have intermediate folders in `../PatchBench/data_processing/`:

- `model_inference/` — raw model answers for each prompt
- `wildguard_inference/` — safety labels (safe / unsafe) from WildGuard
- `elo_ranking/` — pairwise ELO scores ranking the unsafe answers by severity

Then you have manual selection: from the ELO-ranked outputs, manually select the 70 most harmful answers per model and save them to:

```
../PatchBench/data_processing/selected-70-harmful/{model}.json
```

---

### Generate local prompt variations

Set your Mistral API key, then run:

```bash
export MISTRAL_API_KEY="your_key_here"
python -m src.eval.local_evaluation
```

Output: `../PatchBench/data_processing/local_variation/`

For each of the 70 selected prompts per model, this generates three families of neighbours: harmful paraphrases, benign prompts with the same structure, and benign prompts reusing the harmful terms.


### Build the final dataset

```bash
python -m src.data.build_final_dataset
```

For each model, this writes to `../PatchBench/final_dataset/{model}/`:

- `row_data.json` — all samples from WildGuard (safe + unsafe, id / prompt / answer / label), excluding the 20 validation ids
- `test.json` — local variations for the 50 most harmful samples (used for benchmark evaluation)
- `validation.json` — local variations for the 20 remaining samples (used for patching validation)


## Patching Benchmark Pipeline

Before running any benchmark or training scripts, you must prepare the raw data from the `PatchBench` repository. Ensure `git-lfs` is installed on your system.

1. Clone `PatchBench` alongside this repository and pull the large JSON files:
```bash
cd ..
git clone https://huggingface.co/datasets/orailix/PatchBench
cd PatchBench
git lfs install
git lfs pull
cd ../llm_patching_benchmark
```

2. Run the setup script to structure, flatten, and parse the local test/validation datasets and CAST vectors:
```bash
python -m src.data.set_up_benchmark_data
```

The patching pipeline is:

- `unsteered`: no training, no validation search, benchmark inference only
- `ast`: train refusal vector, run validation, label validation outputs, run benchmark inference, label benchmark outputs
- `cast`: train refusal + condition vectors, run CAST gate search + validation, label validation outputs, run benchmark inference, label benchmark outputs
- `adasteer`: train vectors, run validation, label validation outputs, run benchmark inference, label benchmark outputs
- `alphasteer`: train vectors, run validation, label validation outputs, run benchmark inference, label benchmark outputs

In order to train the models, run the validation and the inference, run the following command:
```sh
./run_pipeline.sh NAME_OF_THE_METHOD;
```
Results are then available in `analysis/validation` and `benchmark_results/`. 

## Evaluate General Capability with MMLU

To evaluate general capability of unsteered and steered model, PatchBench evaluates all methods with 5-shot MMLU. Two dedicated scripts run the full pipeline (training + validation + MMLU eval) and produce a comparison report.

For AST and CAST:
```bash
./run_mmlu_pipeline_cast_ast_pipeline.sh
```

For AlphaSteer and AdaSteer:
```bash
./run_mmlu_pipeline_alphasteer_adasteer.sh
```

Both scripts accept `--model` and `--method` flags to run a single configuration, and `--mmlu-only` to skip training and validation:

```bash
./run_mmlu_pipeline_cast_ast_pipeline.sh --model "Qwen/Qwen2.5-3B-Instruct" --method ast --mmlu-only
./run_mmlu_pipeline_alphasteer_adasteer.sh --model "Qwen/Qwen2.5-3B-Instruct" --method alphasteer --mmlu-only --skip-unsteered
```

Results and the comparison report are written to `mmlu_steered_output/`.

## Notes

- Validation is optional for inference, but if you skip it, inference falls back to the `default` values from `validation_selected` in the benchmark config.
- To ignore saved validation selections and force inference to use config defaults, add `--no-validation-selection` to `src.patching.inference.llm_inference_steer_refusal`.
- Validation reports are written under `analysis/validation/`.
- Benchmark outputs and benchmark metrics are written under `benchmark_results/`.
