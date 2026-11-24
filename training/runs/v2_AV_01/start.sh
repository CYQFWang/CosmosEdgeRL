#!/usr/bin/env bash
# Start (or resume) the train_v1 run in the background. Logs: supervisor.log; live state: status.json.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd ~/vis_rl
nohup ~/verl-omni/.venv/bin/python -u -m training.v1_run "$here" "$@" >>"$here/supervisor.log" 2>&1 &
echo "started pid $!"
