#!/bin/bash
#SBATCH --job-name=elo_llama70b
#SBATCH --output=slurm/elo/logs/elo_eval_%j.out
#SBATCH --error=slurm/elo/logs/elo_eval_%j.err
#SBATCH -C a100
#SBATCH --partition=gpu_p5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread


mkdir -p slurm/elo/logs

source ~/.bashrc
conda activate patchbench_env

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

models=(
  'Qwen__Qwen2.5-3B-Instruct'
  'Qwen__Qwen2.5-14B-Instruct'
  'google__gemma-3-4b-it'
  'google__gemma-3-12b-it'
  'meta-llama__Llama-3.1-8B-Instruct'
  'meta-llama__Llama-4-Scout-17B-16E-Instruct'
  'mistralai__Ministral-3-14B-Instruct-2512'
  'mistralai__Mistral-7B-Instruct-v0.3'
)


for model in "${models[@]}"; do
	python3 -u -m src.data.elo_ranking --model_name "$model"
done
