#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
run=${1:?Path to scored native rollout}
out=${2:?New output directory}
gpu_control=${CONTROL_GPU:?GPU UUID for control}
gpu_mismatch=${MISMATCH_GPU:?GPU UUID for mismatch}
[[ "$gpu_control" != "$gpu_mismatch" ]]
mkdir "$out"
export HF_HOME=~/hf_cache HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD:/home/robot_share/cosmos-framework${PYTHONPATH:+:$PYTHONPATH}"
model=$HF_HOME/models--nvidia--Cosmos3-Edge-Policy-DROID/snapshots/68b17b3c959ccd0999de9a972c5f7c8b57112f86
python=~/verl-omni/.venv/bin/python

nvidia-smi --query-gpu=timestamp,uuid,memory.used,utilization.gpu,utilization.memory,power.draw \
  --format=csv --loop-ms=500 >"$out/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
checkpoint_args=()
if [[ ${CHECKPOINT_BLOCKS:-0} == 0 ]]; then
  checkpoint_args=(--no-checkpoint-blocks)
fi
CUDA_VISIBLE_DEVICES="$gpu_control" "$python" -m training.native_update "$run" "$model" \
  --all-gen --tensorboard --rank "${LORA_RANK:-16}" "${checkpoint_args[@]}" --samples "${SAMPLES:-0}" --accumulate "${ACCUMULATE:-4}" --mismatch-weight 0 --output "$out/control" --port 18232 \
  >"$out/control.log" 2>&1 &
control=$!
CUDA_VISIBLE_DEVICES="$gpu_mismatch" "$python" -m training.native_update "$run" "$model" \
  --all-gen --tensorboard --rank "${LORA_RANK:-16}" "${checkpoint_args[@]}" --samples "${SAMPLES:-0}" --accumulate "${ACCUMULATE:-4}" --mismatch-weight .25 --output "$out/mismatch" --port 18233 \
  >"$out/mismatch.log" 2>&1 &
mismatch=$!
status=0
wait "$control" || status=1
wait "$mismatch" || status=1
kill "$monitor" 2>/dev/null || true
wait "$monitor" 2>/dev/null || true
trap - EXIT
"$python" -m training.monitor "$out/tensorboard/gpu" --gpu "$out/gpu.csv"
exit "$status"
