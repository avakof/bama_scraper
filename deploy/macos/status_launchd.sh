#!/bin/bash
# What is actually installed and running — not what we hope is.
set -uo pipefail

LABEL="@LABEL@"
REPO="@REPO@"
PYTHON="@PYTHON@"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
DOMAIN="gui/$(id -u)"

echo "=== plist ==="
if [[ -f "$PLIST" ]]; then
  echo "installed at $PLIST"
  plutil -lint "$PLIST"
else
  echo "NOT INSTALLED at $PLIST"
fi

echo
echo "=== launchctl print $DOMAIN/$LABEL ==="
launchctl print "$DOMAIN/$LABEL" 2>&1 | sed -n '1,32p' || echo "job not loaded"

echo
echo "=== process ==="
pgrep -fl "bama_monitor.scheduler_daemon" || echo "scheduler process NOT running"

echo
echo "=== next trigger ==="
cd "$REPO" 2>/dev/null && PYTHONPATH="$REPO/src" "$PYTHON" -m bama_monitor.scheduler_daemon \
  --print-next 2>/dev/null || echo "could not compute next trigger"

echo
echo "=== recent log tail ==="
tail -n 12 "$REPO/logs/scheduler.out.log" 2>/dev/null || echo "no stdout log yet"
