#!/bin/bash
# Remove the LaunchAgent. The database and all snapshots are left untouched.
set -uo pipefail

LABEL="@LABEL@"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
DOMAIN="gui/$(id -u)"

echo "==> stopping $LABEL"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null && echo "    booted out" \
  || echo "    not loaded (nothing to stop)"

if [[ -f "$PLIST" ]]; then
  rm -f "$PLIST"
  echo "==> removed $PLIST"
else
  echo "==> no plist at $PLIST"
fi

echo
echo "the scheduler is uninstalled. Data is untouched:"
echo "  database        : monitor_data/bama_monitor.sqlite (or \$BAMA_MONITOR_DATABASE_URL)"
echo "  daily snapshots : daily_snapshots/"
echo "  logs            : logs/"
