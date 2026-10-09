#!/bin/bash
#SBATCH --job-name=vllm-sweep
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/home/<user>/llm/inference/logs/sweep_%j.log

set -uo pipefail

VLLM_ENV=/home/<user>/llm/inference/vllm-env
MODEL_PATH=/data/llm/scratch/models/llama-3.1-70b-fp8
SWEEP_SCRIPT=/data/llm/run_sweep.sh
SERVER_LOG=/data/llm/logs/vllm_server_${SLURM_JOB_ID}.log

source "$VLLM_ENV/bin/activate"

echo "=== Starting vLLM server on $(hostname) at $(date) ==="
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if [ "$FREE_MB" -lt 80000 ]; then
  echo "ERROR: insufficient free GPU memory (${FREE_MB} MiB). Aborting."
  nvidia-smi
  exit 1
fi

# Launch server in the background, inside this same job's allocation
vllm serve "$MODEL_PATH" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 16384 \
  --port 8000 \
  --served-model-name llama-3.1-70b-fp8 \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for readiness rather than a fixed sleep -- more reliable across variable load times
echo "Waiting for server to become ready..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "Server ready after ${i}0s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server process died during startup. Check $SERVER_LOG"
    exit 1
  fi
  sleep 10
done

if ! curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
  echo "ERROR: server did not become ready within 10 minutes. Check $SERVER_LOG"
  kill "$SERVER_PID" 2>/dev/null
  exit 1
fi

echo "=== Running sweep at $(date) ==="
chmod +x "$SWEEP_SCRIPT"
"$SWEEP_SCRIPT"
SWEEP_RC=$?

echo "=== Sweep finished at $(date) with exit code $SWEEP_RC ==="

# Clean shutdown
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null

exit $SWEEP_RC
