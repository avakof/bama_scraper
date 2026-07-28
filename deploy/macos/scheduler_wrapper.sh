#!/bin/bash
# launchd wrapper for the Bama scheduler daemon.
#
# launchd starts jobs with a near-empty environment: no shell profile is read, no
# virtualenv is activated, PATH is a bare minimum and the working directory is /.
# Everything this script needs is therefore absolute, and nothing is inherited.
#
# Placeholders (@REPO@, @PYTHON@, @ENV_FILE@) are substituted by install_launchd.sh.

set -euo pipefail

REPO="@REPO@"
PYTHON="@PYTHON@"
ENV_FILE="@ENV_FILE@"

cd "$REPO"

# Configuration and secrets come from a file, never from this script or the plist.
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON" ]]; then
  echo "FATAL: python interpreter not found at $PYTHON" >&2
  echo "the virtual environment may have been moved or deleted; reinstall with" >&2
  echo "  ./deploy/macos/install_launchd.sh" >&2
  exit 78   # EX_CONFIG
fi

echo "=== bama scheduler wrapper $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "repo   : $REPO"
echo "python : $PYTHON"
echo "db     : ${BAMA_MONITOR_DATABASE_URL:-<default sqlite>}"
echo "pid    : $$"

exec "$PYTHON" -m bama_monitor.scheduler_daemon "$@"
