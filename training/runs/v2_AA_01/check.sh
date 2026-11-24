#!/usr/bin/env bash
# Read-only progress report for this run. Usage: ./check.sh [--last N]
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd ~/vis_rl && exec ~/verl-omni/.venv/bin/python -m training.v1_status "$here" "$@"
