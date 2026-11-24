#!/usr/bin/env bash
# Graceful stop: finishes the running batches of the current collection, then exits. Resume with start.sh --resume.
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
touch "$here/STOP" && echo "stop requested"
