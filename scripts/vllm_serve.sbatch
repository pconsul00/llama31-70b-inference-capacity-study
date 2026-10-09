#!/bin/bash
#SBATCH --job-name=vllm-tp4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/data/llm/logs/vllm_%j.log
##SBATCH --requeue

source /data/llm/vllm-env/bin/activate

vllm serve /data/llm/scratch/models/llama-3.1-70b-fp8 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 16384 \
  --port 8000 \
  --served-model-name llama-3.1-70b-fp8
