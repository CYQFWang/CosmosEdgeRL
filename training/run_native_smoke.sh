#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
workspace=$PWD
root=${OUT:?Set OUT to a fresh absolute directory}
model=~/hf_cache/models--nvidia--Cosmos3-Edge-Policy-DROID/snapshots/68b17b3c959ccd0999de9a972c5f7c8b57112f86
python=~/verl-omni/.venv/bin/python
export SIM_GPU POLICY_GPU
export NUM_ENVS=${NUM_ENVS:-2}
export COSMOS_CAPTURE_NFT=1
export DREAM_ENDPOINT_ONLY=1 DREAM_VIDEO_MODE=none
export HF_HOME=~/hf_cache HF_HUB_OFFLINE=1

mkdir "$root"
rollout_start=$SECONDS
OUT="$root/rollout" DREAM_MAX_STEPS=128 bash dream_eval/run_pilot.sh BananaInBowlTask
rollout_seconds=$((SECONDS - rollout_start))
judge_start=$SECONDS
CUDA_VISIBLE_DEVICES="$SIM_GPU" JUDGE_EAGER=1 JUDGE_GPU_FRACTION=.24 \
  bash dream_eval/score.sh "$root/rollout" >"$root/judge.log" 2>&1
judge_seconds=$((SECONDS - judge_start))
actor_start=$SECONDS
CUDA_VISIBLE_DEVICES="$POLICY_GPU" PYTHONPATH="$workspace:/home/robot_share/cosmos-framework" \
  "$python" -m training.native_update "$root/rollout" "$model" >"$root/actor.log" 2>&1
actor_seconds=$((SECONDS - actor_start))
reload_start=$SECONDS
OUT="$root/reload" DREAM_MAX_STEPS=32 COSMOS_HEAD_ADAPTER="$root/rollout/heads.pt" \
  bash dream_eval/run_pilot.sh BananaInBowlTask
reload_seconds=$((SECONDS - reload_start))
printf '{"rollout_wall_s":%d,"judge_wall_s":%d,"actor_wall_s":%d,"reload_rollout_wall_s":%d}\n' \
  "$rollout_seconds" "$judge_seconds" "$actor_seconds" "$reload_seconds" >"$root/timing.json"
