#!/bin/bash
# vLLM parameter sweep: input length x output length x request rate
# Fixed request count per cell + GPU/VRAM sampling during each run.

set -uo pipefail

MODEL="llama-3.1-70b-fp8"
TOKENIZER="/data/llm/scratch/models/llama-3.1-70b-fp8"
BASE_URL="http://localhost:8000"
OUT_DIR="/data/llm/sweep_results"
MANIFEST="$OUT_DIR/manifest.csv"
NUM_PROMPTS=100          # fixed per cell — same load size everywhere
COOLDOWN_SECONDS=15      # pause between cells so one cell's tail doesn't bleed into the next
GPU_SAMPLE_INTERVAL=2    # seconds (nvidia-smi)
DCGM_SAMPLE_MS=2000      # milliseconds (dcgmi)
DCGM_FIELDS="1001,1002,1003,1004,1005,1009,1010"   # subgroup A only — see notes

mkdir -p "$OUT_DIR"
if [ ! -f "$MANIFEST" ]; then
  echo "input_len,output_len,rate,num_prompts,filename,status,elapsed_s,peak_gpu_util,avg_gpu_util,peak_mem_used_mib,avg_mem_used_mib,peak_power_draw_w,avg_power_draw_w,peak_sm_active,peak_tensor_active,peak_dram_active,start_time" > "$MANIFEST"
fi

# Set up a DCGM group scoped to this job's own GPUs once, reused across all cells.
# Use UUIDs, not cgroup-relative indices -- nv-hostengine runs outside the job's
# cgroup and its GPU numbering does not necessarily match CUDA_VISIBLE_DEVICES.
# nvidia-smi's UUID enumeration can bypass cgroup device restriction and list
# every GPU on the node -- so filter down to only the indices Slurm actually
# assigned this job, via CUDA_VISIBLE_DEVICES, rather than trusting the full list.
GPU_UUIDS=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | \
  awk -F', ' -v idx="$CUDA_VISIBLE_DEVICES" '
    BEGIN { n=split(idx,want,","); for(i=1;i<=n;i++) wantset[want[i]]=1 }
    ($1 in wantset) { print $2 }
  ' | paste -sd,)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job's GPU UUIDs (filtered): $GPU_UUIDS"

GPU_COUNT=$(echo "$GPU_UUIDS" | tr ',' '\n' | grep -c .)
if [ "$GPU_COUNT" -ne 4 ]; then
  echo "WARNING: expected 4 GPU UUIDs, got $GPU_COUNT. Check CUDA_VISIBLE_DEVICES indexing assumption below."
fi

# Clean up any stale sweep_group left over from a killed/interrupted prior run --
# DCGM refuses to create a second group with the same name, which fails silently
# and falls through to the warning below if a stale one is still hanging around.
for stale_id in $(env -u CUDA_VISIBLE_DEVICES dcgmi group -l | awk '
  /Group ID/{id=$NF}
  /Group Name/{if ($NF=="sweep_group") print id}
'); do
  echo "Removing stale DCGM group: $stale_id"
  env -u CUDA_VISIBLE_DEVICES dcgmi group -d "$stale_id" 2>/dev/null
done

DCGM_GROUP=$(env -u CUDA_VISIBLE_DEVICES dcgmi group -c sweep_group | grep -oP '(?<=group ID of )\d+')
if [ -n "$DCGM_GROUP" ] && [ "$GPU_COUNT" -eq 4 ]; then
  env -u CUDA_VISIBLE_DEVICES dcgmi group -g "$DCGM_GROUP" -a "$GPU_UUIDS"
  echo "DCGM group $DCGM_GROUP created with GPUs: $GPU_UUIDS"
else
  echo "WARNING: could not create DCGM group — DCGM logging will be skipped this run."
  DCGM_GROUP=""
fi

# Override any of these via environment variable to run a targeted batch, e.g.:
#   OUTPUT_LENS="10240" ./run_sweep.sh
#   OUTPUT_LENS="128 512 1024" ./run_sweep.sh
INPUT_LENS=(${INPUT_LENS:-128 256 384 512 1024})
OUTPUT_LENS=(${OUTPUT_LENS:-128 512 1024 2048 5120 10240})
RATES=(${RATES:-5 10 20 50 100})

for IL in "${INPUT_LENS[@]}"; do
  for OL in "${OUTPUT_LENS[@]}"; do
    for R in "${RATES[@]}"; do

      FNAME="bench_in${IL}_out${OL}_rate${R}.json"
      GPULOG="$OUT_DIR/gpu_in${IL}_out${OL}_rate${R}.csv"
      DCGMLOG="$OUT_DIR/dcgm_in${IL}_out${OL}_rate${R}.log"
      FPATH="$OUT_DIR/$FNAME"

      if grep -q ",${FNAME},completed," "$MANIFEST" 2>/dev/null; then
        echo "SKIP (already done): $FNAME"
        continue
      fi

      echo "=== in=${IL} out=${OL} rate=${R} num_prompts=${NUM_PROMPTS} start: $(date '+%F %T') ==="
      START=$(date +%s)
      START_ISO=$(date -Iseconds)

      # Start nvidia-smi sampling (VRAM + basic util) in the background
      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv -l "$GPU_SAMPLE_INTERVAL" -f "$GPULOG" &
      GPU_PID=$!

      # Start DCGM sampling (SM/tensor/DRAM/PCIe activity) in the background, if group setup succeeded
      DCGM_PID=""
      if [ -n "$DCGM_GROUP" ]; then
        env -u CUDA_VISIBLE_DEVICES dcgmi dmon -g "$DCGM_GROUP" -e "$DCGM_FIELDS" -d "$DCGM_SAMPLE_MS" > "$DCGMLOG" &
        DCGM_PID=$!
      fi

      vllm bench serve \
        --backend vllm \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER" \
        --dataset-name random \
        --random-input-len "$IL" \
        --random-output-len "$OL" \
        --random-range-ratio 0.3 \
        --seed 42 \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$R" \
        --save-result \
        --result-filename "$FPATH"

      RC=$?

      # Stop both samplers
      kill "$GPU_PID" 2>/dev/null; wait "$GPU_PID" 2>/dev/null
      if [ -n "$DCGM_PID" ]; then
        kill "$DCGM_PID" 2>/dev/null; wait "$DCGM_PID" 2>/dev/null
      fi

      END=$(date +%s)
      ELAPSED=$(( END - START ))

      if [ $RC -ne 0 ]; then
        STATUS="failed"
        PEAK_UTIL="NA"; AVG_UTIL="NA"; PEAK_MEM="NA"; AVG_MEM="NA"; PEAK_POWER="NA"; AVG_POWER="NA"; PEAK_SM="NA"; PEAK_TENSOR="NA"; PEAK_DRAM="NA"
      else
        STATUS="completed"
        # nvidia-smi log: peak utilization and memory (skip header)
        PEAK_UTIL=$(awk -F',' 'NR>1 && $3 ~ /[0-9]/ {gsub(/[^0-9]/,"",$3); if ($3+0>max) max=$3+0} END{print max+0}' "$GPULOG")
        PEAK_MEM=$(awk -F',' 'NR>1 && $4 ~ /[0-9]/ {gsub(/[^0-9]/,"",$4); if ($4+0>max) max=$4+0} END{print max+0}' "$GPULOG")
        AVG_UTIL=$(awk -F',' 'NR>1 && $3 ~ /[0-9]/ {gsub(/[^0-9]/,"",$3); sum+=$3+0; n++} END{if(n>0) printf "%.1f", sum/n; else print 0}' "$GPULOG")
        AVG_MEM=$(awk -F',' 'NR>1 && $4 ~ /[0-9]/ {gsub(/[^0-9]/,"",$4); sum+=$4+0; n++} END{if(n>0) printf "%.1f", sum/n; else print 0}' "$GPULOG")
        # power.draw is column 6, format like "412.31 W" -- strip everything but digits and the decimal point
        PEAK_POWER=$(awk -F',' 'NR>1 && $6 ~ /[0-9]/ {gsub(/[^0-9.]/,"",$6); if ($6+0>max) max=$6+0} END{printf "%.1f", max}' "$GPULOG")
        AVG_POWER=$(awk -F',' 'NR>1 && $6 ~ /[0-9]/ {gsub(/[^0-9.]/,"",$6); sum+=$6+0; n++} END{if(n>0) printf "%.1f", sum/n; else print 0}' "$GPULOG")

        # dcgmi dmon log: columns are GPU-ID, then one per requested field, in field order (1001..1010)
        # sm_active=1002 is column 3, tensor_active=1004 is column 5, dram_active=1005 is column 6 -- verify against your header row
        if [ -f "$DCGMLOG" ]; then
          PEAK_SM=$(awk 'NR>2 {gsub(/[^0-9.]/,"",$3); if ($3+0>max) max=$3+0} END{printf "%.2f", max}' "$DCGMLOG")
          PEAK_TENSOR=$(awk 'NR>2 {gsub(/[^0-9.]/,"",$5); if ($5+0>max) max=$5+0} END{printf "%.2f", max}' "$DCGMLOG")
          PEAK_DRAM=$(awk 'NR>2 {gsub(/[^0-9.]/,"",$6); if ($6+0>max) max=$6+0} END{printf "%.2f", max}' "$DCGMLOG")
        else
          PEAK_SM="NA"; PEAK_TENSOR="NA"; PEAK_DRAM="NA"
        fi
      fi

      echo "${IL},${OL},${R},${NUM_PROMPTS},${FNAME},${STATUS},${ELAPSED},${PEAK_UTIL},${AVG_UTIL},${PEAK_MEM},${AVG_MEM},${PEAK_POWER},${AVG_POWER},${PEAK_SM},${PEAK_TENSOR},${PEAK_DRAM},${START_ISO}" >> "$MANIFEST"
      echo "--- result: ${STATUS} in ${ELAPSED}s end: $(date '+%F %T') | VRAM avg/peak ${AVG_MEM}/${PEAK_MEM} MiB | util avg/peak ${AVG_UTIL}/${PEAK_UTIL}% | power avg/peak ${AVG_POWER}/${PEAK_POWER} W ---"

      # Cooldown before the next cell -- lets the queue fully drain and GPU
      # activity settle back to baseline, so one cell's tail doesn't bleed
      # into the next cell's numbers.
      echo "Cooling down for ${COOLDOWN_SECONDS}s before next cell..."
      sleep "$COOLDOWN_SECONDS"

    done
  done
done

echo "Sweep pass complete. See $MANIFEST for results."
