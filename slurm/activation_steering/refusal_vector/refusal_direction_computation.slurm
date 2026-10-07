#!/bin/bash
#SBATCH --account=
#SBATCH --job-name=llmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --hint=nomultithread
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev

source ~/.bashrc
conda activate llm_patching_env

cd -- "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"

srun python -m src.patching.extract_save_refusal  --model Qwen/Qwen2.5-3B-Instruct --batch-size 32 --method pca_pairwise --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --force-recompute
srun python -m src.patching.extract_save_refusal  --model Qwen/Qwen2.5-3B-Instruct --batch-size 32 --method mean_diff --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --reuse-cache

srun python -m src.patching.extract_save_refusal  --model Qwen/Qwen2.5-14B-Instruct --batch-size 32 --method pca_pairwise --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --force-recompute
srun python -m src.patching.extract_save_refusal  --model Qwen/Qwen2.5-14B-Instruct --batch-size 32 --method mean_diff --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --reuse-cache

srun python -m src.patching.extract_save_refusal  --model google/gemma-3-12b-it --batch-size 32 --method pca_pairwise --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --force-recompute
srun python -m src.patching.extract_save_refusal  --model google/gemma-3-12b-it --batch-size 32 --method mean_diff --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --reuse-cache

srun python -m src.patching.extract_save_refusal  --model google/gemma-3-4b-it --batch-size 32 --method pca_pairwise --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --force-recompute
srun python -m src.patching.extract_save_refusal  --model google/gemma-3-4b-it --batch-size 32 --method mean_diff --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --reuse-cache

srun python -m src.patching.extract_save_refusal  --model meta-llama/Llama-3.1-8B-Instruct --batch-size 32 --method pca_pairwise --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --force-recompute
srun python -m src.patching.extract_save_refusal  --model meta-llama/Llama-3.1-8B-Instruct --batch-size 32 --method mean_diff --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --reuse-cache

srun python -m src.patching.extract_save_refusal  --model meta-llama/Llama-4-Scout-17B-16E-Instruct --batch-size 16 --method pca_pairwise --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --force-recompute
srun python -m src.patching.extract_save_refusal  --model meta-llama/Llama-4-Scout-17B-16E-Instruct --batch-size 16 --method mean_diff --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --reuse-cache

srun python -m src.patching.extract_save_refusal  --model mistralai/Mistral-7B-Instruct-v0.3 --batch-size 32 --method pca_pairwise --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --force-recompute
srun python -m src.patching.extract_save_refusal  --model mistralai/Mistral-7B-Instruct-v0.3 --batch-size 32 --method mean_diff --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --reuse-cache 

srun python -m src.patching.extract_save_refusal  --model mistralai/Ministral-3-14B-Instruct-2512 --batch-size 32 --method pca_pairwise --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --force-recompute
srun python -m src.patching.extract_save_refusal  --model mistralai/Ministral-3-14B-Instruct-2512 --batch-size 32 --method mean_diff --accumulate-last-x-tokens suffix-only --max-questions 100 --max-suffixes 100 --save-analysis --reuse-cache