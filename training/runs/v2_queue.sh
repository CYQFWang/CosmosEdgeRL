#!/usr/bin/env bash
# Run the train_v2 arms in order: A-video, A-action, T'. Each stops after 3 epochs (epoch checkpoints kept).
# The queue stops at the first arm that does not finish with phase "done" (error, STOP, or AA without lambda).
# Resume: fix the cause, then rerun this script; finished arms are skipped, an interrupted arm resumes.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd ~/vis_rl
for arm in v2_AV_01 v2_AA_01 v2_T_01; do
  phase=$(python3 -c "import json,sys;print(json.load(open('$here/$arm/status.json')).get('phase',''))" 2>/dev/null || true)
  if [[ $phase == done ]]; then echo "$(date) $arm already done"; continue; fi
  echo "$(date) starting $arm"
  rm -f "$here/$arm/STOP"
  ~/verl-omni/.venv/bin/python -u -m training.v1_run "$here/$arm" >>"$here/$arm/supervisor.log" 2>&1
  phase=$(python3 -c "import json;print(json.load(open('$here/$arm/status.json')).get('phase',''))" 2>/dev/null || echo error)
  echo "$(date) $arm finished with phase=$phase"
  [[ $phase == done ]] || { echo "queue stopped at $arm"; exit 1; }
done
echo "$(date) all arms done"
