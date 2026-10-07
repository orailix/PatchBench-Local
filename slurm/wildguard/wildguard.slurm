#!/bin/bash
#SBATCH --job-name=wildguard_label
#SBATCH -C a100
#SBATCH --partition=gpu_p5
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread

mkdir -p slurm/wildguard/logs

source ~/.bashrc
conda activate patchbench_env

# Forcer le mode offline pour HF
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 -u -m src.data.wildguard_label
