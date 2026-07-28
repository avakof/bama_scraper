#!/bin/bash
# Install the Bama scheduler as a macOS LaunchAgent.
#
# Everything is resolved to an absolute path here, because launchd expands
# nothing: no ~, no $HOME, no PATH lookup, no activated virtualenv. A plist that
# works when you run it by hand and fails at login is almost always a relative
# path that the shell resolved and launchd did not.
#
# The script verifies rather than assumes: it lints the plist, boots the job,
# waits for the process, and prints launchctl's own view of the result. Writing
# a file is not installation.

set -euo pipefail

LABEL="com.bama.monitor.scheduler"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
HERE="$REPO/deploy/macos"
DOMAIN="gui/$(id -u)"
AGENTS="$HOME/Library/LaunchAgents"
PLIST="$AGENTS/${LABEL}.plist"
LOGS="$REPO/logs"
ENV_FILE="${BAMA_ENV_FILE:-$REPO/deploy/macos/monitor.env}"

echo "==> resolving paths"
PYTHON="$REPO/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "FATAL: no interpreter at $PYTHON" >&2
  echo "create it with:  python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi
PYTHON="$(cd "$(dirname "$PYTHON")" && pwd -P)/$(basename "$PYTHON")"
echo "    repo   : $REPO"
echo "    python : $PYTHON"
echo "    label  : $LABEL"
echo "    domain : $DOMAIN"

echo "==> macOS version"
sw_vers | sed 's/^/    /'
MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"

echo "==> environment file"
if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<ENVEOF
# Environment for the Bama scheduler LaunchAgent.
# Secrets live here, never in the plist and never in git.
# chmod 600 is applied by the installer.
BAMA_MONITOR_DATABASE_URL=sqlite:///$REPO/monitor_data/bama_monitor.sqlite
ENVEOF
  echo "    created $ENV_FILE"
else
  echo "    using existing $ENV_FILE"
fi
chmod 600 "$ENV_FILE"

echo "==> verifying the database is reachable"
set +e
DB_CHECK="$(cd "$REPO" && PYTHONPATH="$REPO/src" "$PYTHON" - <<'PYEOF' 2>&1
import os, sys
sys.path.insert(0, "src")
from bama_monitor.config import load_config
from bama_monitor.db import connect
cfg = load_config(None)
db = connect(cfg.database_url)
runs = db.scalar("SELECT COUNT(*) FROM monitoring_runs")
print(f"OK dialect={db.dialect} runs={runs}")
db.close()
PYEOF
)"
DB_STATUS=$?
set -e
echo "    $DB_CHECK"
if [[ $DB_STATUS -ne 0 ]]; then
  echo "FATAL: the database is not reachable; fix BAMA_MONITOR_DATABASE_URL first" >&2
  exit 1
fi

echo "==> log directory"
mkdir -p "$LOGS"
STDOUT="$LOGS/scheduler.out.log"
STDERR="$LOGS/scheduler.err.log"
touch "$STDOUT" "$STDERR"
if [[ ! -w "$STDOUT" || ! -w "$STDERR" ]]; then
  echo "FATAL: logs are not writable at $LOGS" >&2
  exit 1
fi
echo "    $STDOUT"
echo "    $STDERR"

echo "==> rendering scripts with absolute paths"
render() {
  sed -e "s|@REPO@|$REPO|g" -e "s|@PYTHON@|$PYTHON|g" -e "s|@ENV_FILE@|$ENV_FILE|g" \
      -e "s|@LABEL@|$LABEL|g" -e "s|@WRAPPER@|$HERE/scheduler_wrapper.rendered.sh|g" \
      -e "s|@STDOUT@|$STDOUT|g" -e "s|@STDERR@|$STDERR|g" "$1" > "$2"
}
render "$HERE/scheduler_wrapper.sh" "$HERE/scheduler_wrapper.rendered.sh"
render "$HERE/run_once.sh"          "$HERE/run_once.rendered.sh"
render "$HERE/status_launchd.sh"    "$HERE/status_launchd.rendered.sh"
render "$HERE/uninstall_launchd.sh" "$HERE/uninstall_launchd.rendered.sh"
chmod +x "$HERE"/*.rendered.sh
echo "    wrapper: $HERE/scheduler_wrapper.rendered.sh"

mkdir -p "$AGENTS"
render "$HERE/com.bama.monitor.scheduler.plist" "$PLIST"

echo "==> validating the plist"
plutil -lint "$PLIST"
if grep -q '@[A-Z_]*@' "$PLIST"; then
  echo "FATAL: unsubstituted placeholder left in $PLIST" >&2
  grep -n '@[A-Z_]*@' "$PLIST" >&2
  exit 1
fi
# Every path in the plist must be absolute, or launchd will fail at login with
# nothing useful in the log.
if grep -Eq '<string>[^/<][^<]*/' "$PLIST" && ! grep -q '<string>/' "$PLIST"; then
  echo "FATAL: a relative path reached the plist" >&2
  exit 1
fi
echo "    OK, all placeholders substituted"

echo "==> unloading any previous version"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null && echo "    previous job booted out" \
  || echo "    nothing previously loaded"

echo "==> bootstrapping (macOS $MACOS_MAJOR)"
launchctl bootstrap "$DOMAIN" "$PLIST"
launchctl enable "$DOMAIN/$LABEL"
launchctl kickstart -k "$DOMAIN/$LABEL"

echo "==> waiting for the scheduler process"
PID=""
for _ in $(seq 1 30); do
  PID="$(pgrep -f 'bama_monitor.scheduler_daemon' | head -1 || true)"
  [[ -n "$PID" ]] && break
  sleep 1
done

echo
echo "==> launchctl print"
launchctl print "$DOMAIN/$LABEL" 2>&1 | sed -n '1,20p'

echo
if [[ -n "$PID" ]]; then
  echo "SCHEDULER RUNNING, pid $PID"
  ps -p "$PID" -o pid,etime,command | tail -1
else
  echo "SCHEDULER NOT RUNNING — inspect the logs:" >&2
  echo "  tail -n 40 $STDERR" >&2
  tail -n 20 "$STDERR" >&2 || true
  exit 1
fi

echo
echo "==> next trigger"
cd "$REPO" && PYTHONPATH="$REPO/src" "$PYTHON" -m bama_monitor.scheduler_daemon --print-next

echo
echo "installed. Useful commands:"
echo "  ./deploy/macos/status_launchd.rendered.sh"
echo "  ./deploy/macos/uninstall_launchd.rendered.sh"
echo "  tail -f $STDOUT"
echo "  tail -f $STDERR"
