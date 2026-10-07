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

srun python -u -m src.patching.llm_inference_steer_refusal --model Qwen/Qwen2.5-3B-Instruct --batch-size 128  --layers all --strength 2 --in-json test_data/alpaca/alpaca_test.json 

srun python -u -m src.patching.llm_inference_steer_refusal  --model Qwen/Qwen2.5-14B-Instruct --batch-size 128 --layers all --strength 2.5 --in-json test_data/alpaca/alpaca_test.json 

srun python -u -m src.patching.llm_inference_steer_refusal --model google/gemma-3-12b-it --batch-size 128 --layers all  --strength 200 --in-json test_data/alpaca/alpaca_test.json 

srun python -u -m src.patching.llm_inference_steer_refusal --model google/gemma-3-4b-it --batch-size 128 --layers all --strength 80 --in-json test_data/alpaca/alpaca_test.json 

srun python -u -m src.patching.llm_inference_steer_refusal --model meta-llama/Llama-3.1-8B-Instruct --batch-size 128 --layers all --strength 0.35 --in-json test_data/alpaca/alpaca_test.json 

srun python -u -m src.patching.llm_inference_steer_refusal --model meta-llama/Llama-4-Scout-17B-16E-Instruct --batch-size 32 --layers all --strength 1 --in-json test_data/alpaca/alpaca_test.json 

srun python -u -m src.patching.llm_inference_steer_refusal --model mistralai/Mistral-7B-Instruct-v0.3 --batch-size 128 --layers all --strength 0.08 --in-json test_data/alpaca/alpaca_test.json 

srun python -u -m src.patching.llm_inference_steer_refusal  --model mistralai/Ministral-3-14B-Instruct-2512 --batch-size 128 --layers all --strength 0.5 --in-json test_data/alpaca/alpaca_test.json 