#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
out=${1:?New absolute output directory}
shift
if (( $# == 0 )); then
  set -- BananaInBowlTask YogurtInBowlTask
fi
mkdir "$out"
export COSMOS_UND_CACHE=8 COSMOS_CAPTURE_NFT=1 DREAM_ENDPOINT_ONLY=1 DREAM_VIDEO_MODE=none
export DREAM_MAX_STEPS=${DREAM_MAX_STEPS:-64}
export DREAM_INSTRUCTION=default

for layout in ${LAYOUTS:-1x1 1x2 1x4 2x1 2x2}; do
  [[ $layout =~ ^[1-9][0-9]*x[1-9][0-9]*$ ]]
  mkdir "$out/$layout"
  nvidia-smi --query-gpu=timestamp,uuid,memory.used,utilization.gpu,utilization.memory,power.draw \
    --format=csv --loop-ms=500 >"$out/$layout/gpu.csv" &
  monitor=$!
  trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
  started=$SECONDS
  DREAM_TASK_WORKERS=${layout%x*} NUM_ENVS=${layout#*x} OUT="$out/$layout" \
    bash dream_eval/run_pilot.sh "$@"
  printf '%s\n' "$((SECONDS - started))" >"$out/$layout/cold_wall_s"
  kill "$monitor" 2>/dev/null || true
  wait "$monitor" 2>/dev/null || true
  trap - EXIT
done
