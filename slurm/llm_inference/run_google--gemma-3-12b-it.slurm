#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH -C a100

source ~/.bashrc
conda activate patchbench_env

srun python3 -u -m src.data.llm_inference  --model google/gemma-3-12b-it --batch-size 64
