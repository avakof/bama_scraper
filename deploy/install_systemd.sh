#!/usr/bin/env bash
# Install the monitor as a systemd timer running at 13:00 Europe/Paris.
#
# Idempotent: safe to re-run after a code update.
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/bama-monitor}"
ETC_DIR="${ETC_DIR:-/etc/bama-monitor}"
SERVICE_USER="${SERVICE_USER:-bama}"
UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

if [[ $EUID -ne 0 ]]; then
  echo "must run as root (installs unit files into $UNIT_DIR)" >&2
  exit 1
fi
command -v systemctl >/dev/null || { echo "systemd not available on this host" >&2; exit 1; }

echo "==> service user"
id -u "$SERVICE_USER" &>/dev/null || useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"

echo "==> application directory $APP_DIR"
mkdir -p "$APP_DIR" "$APP_DIR/monitor_data" "$APP_DIR/reports" "$ETC_DIR"
# --delete keeps a redeploy clean, but never touches runtime data or reports.
rsync -a --delete \
  --exclude '.venv' --exclude 'monitor_data' --exclude 'reports' \
  --exclude '__pycache__' --exclude '.git' \
  "$REPO_ROOT/" "$APP_DIR/"

echo "==> virtualenv and dependencies"
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.venv/bin/pip" install --quiet -e "$APP_DIR"
"$APP_DIR/.venv/bin/pip" install --quiet "psycopg[binary]" apscheduler tzdata

echo "==> configuration"
if [[ ! -f "$ETC_DIR/config.yaml" ]]; then
  cp "$APP_DIR/deploy/config.example.yaml" "$ETC_DIR/config.yaml"
  echo "    wrote $ETC_DIR/config.yaml (review it)"
fi
if [[ ! -f "$ETC_DIR/monitor.env" ]]; then
  cat > "$ETC_DIR/monitor.env" <<'ENVEOF'
# Credentials live here, never in version control. chmod 600.
BAMA_MONITOR_DATABASE_URL=postgresql://bama:CHANGE_ME@localhost:5432/bama_monitor
# Optional alert sinks:
# BAMA_MONITOR_WEBHOOK_URL=
# BAMA_MONITOR_SMTP_URL=
# BAMA_MONITOR_ALERT_EMAIL=
ENVEOF
  echo "    wrote $ETC_DIR/monitor.env - EDIT THE DATABASE PASSWORD"
fi
chmod 600 "$ETC_DIR/monitor.env"
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR" "$ETC_DIR"

echo "==> database migrations"
sudo -u "$SERVICE_USER" env "$(grep -v '^#' "$ETC_DIR/monitor.env" | xargs)" \
  "$APP_DIR/.venv/bin/python" -m bama_monitor migrate --config "$ETC_DIR/config.yaml" || {
    echo "    migrations failed - check BAMA_MONITOR_DATABASE_URL" >&2; exit 1; }

echo "==> backup directory"
install -d -m 750 -o "$SERVICE_USER" -g "$SERVICE_USER" /var/backups/bama-monitor
install -m 750 "$HERE/backup.sh" "$APP_DIR/deploy/backup.sh"

echo "==> unit files"
for unit in bama-monitor.service bama-monitor.timer \
            bama-monitor-backup.service bama-monitor-backup.timer; do
  install -m 644 "$HERE/$unit" "$UNIT_DIR/$unit"
done
systemctl daemon-reload

echo "==> unit syntax"
# Verify before enabling: a unit that fails to parse would otherwise be discovered
# at 13:00, by which point a day of history is already missing.
systemd-analyze verify "$UNIT_DIR/bama-monitor.service" "$UNIT_DIR/bama-monitor.timer" \
  "$UNIT_DIR/bama-monitor-backup.service" "$UNIT_DIR/bama-monitor-backup.timer" || {
    echo "    unit verification failed" >&2; exit 1; }

systemctl enable --now bama-monitor.timer
systemctl enable --now bama-monitor-backup.timer

echo
echo "==> verification"
systemctl list-timers bama-monitor.timer bama-monitor-backup.timer --all --no-pager || true
echo
systemd-analyze calendar --iterations=3 --timezone=Europe/Paris '*-*-* 13:00:00' || true
echo
echo "==> restore drill (proves the backup is usable, not just present)"
sudo -u "$SERVICE_USER" env "$(grep -v '^#' "$ETC_DIR/monitor.env" | xargs)" \
  "$APP_DIR/deploy/backup.sh" drill || {
    echo "    RESTORE DRILL FAILED - do not consider this deployment complete" >&2; exit 1; }

echo
echo "Installed. Useful commands:"
echo "  systemctl start bama-monitor.service      # run once now"
echo "  journalctl -u bama-monitor.service -f     # follow logs"
echo "  systemctl list-timers bama-monitor.timer  # confirm next firing"
echo "  deploy/backup.sh drill                    # dump -> restore -> compare"
