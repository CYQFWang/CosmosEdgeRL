#!/usr/bin/env bash
# Untrained-task evaluation of each v2 arm's epoch-3 checkpoint, paired with the stored screening base
# (same 10 tasks, seeds and layouts as eval_untrained_e02c2). One checkpoint at a time on both GPUs.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd ~/vis_rl
tasks="RubiksCubeTask CoffeePotInBinTask ClampInRightBinTask FoodPacking1BoxesTask FoodPacking1CansTask FruitsOnionToPlateTask PutBowlOnShelfTopTask BananaInBowlTask BananaOnPlateTask SauceBottlesCrateTask"
for arm in v2_T_01 v2_AV_01 v2_AA_01; do
  out="$here/$arm/eval_untrained_ep3"
  if [[ -f $out/summary.json ]]; then echo "$(date) $arm already evaluated"; continue; fi
  echo "$(date) evaluating $arm"
  ~/verl-omni/.venv/bin/python -u -m training.v1_eval "$out" --run "$here/v1_T_01" --pair-screening \
    --model "$here/$arm/checkpoints/e03c1" --tasks $tasks >"$out.log" 2>&1 || { echo "$(date) $arm eval failed; see $out.log"; exit 1; }
  echo "$(date) $arm done"
done
echo "$(date) all evaluations done"
