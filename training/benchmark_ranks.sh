#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
run=${1:?Path to scored rollout}
out=${2:?New output directory}
mkdir "$out"
for rank in 8 16 32; do
  LORA_RANK=$rank CHECKPOINT_BLOCKS=0 bash training/run_option4_pair.sh "$run" "$out/rank_$rank"
done
LORA_RANK=16 CHECKPOINT_BLOCKS=1 bash training/run_option4_pair.sh "$run" "$out/rank_16_checkpoint"
