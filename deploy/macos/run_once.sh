#!/bin/bash
# Run the daily pipeline once, by hand, without touching the production schedule.
#
# `--trigger-type manual` means the run is given NO scheduled slot, so it can
# never satisfy or block the day's scheduled execution. That is the difference
# between this and what launchd fires.

set -euo pipefail

REPO="@REPO@"
PYTHON="@PYTHON@"
ENV_FILE="@ENV_FILE@"

cd "$REPO"
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
fi
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec "$PYTHON" -m bama_monitor.cli run-daily --trigger-type manual "$@"
