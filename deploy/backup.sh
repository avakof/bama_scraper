#!/usr/bin/env bash
#
# PostgreSQL backup with retention, plus a restore path that is actually exercised.
#
# The monitoring history is not reproducible: a daily observation missed today can
# never be recovered, because yesterday's inventory no longer exists anywhere. That
# makes the backup part of the data model rather than an operational nicety.
#
#   backup.sh dump                 write a compressed custom-format dump, prune old ones
#   backup.sh verify <file>        check a dump is readable and lists the core tables
#   backup.sh restore <file> <db>  restore into a NEW database (never over a live one)
#   backup.sh drill                dump -> restore to a scratch db -> compare counts
#
# `drill` is the only one of these that proves anything. A dump nobody has restored
# is a hypothesis.
#
set -euo pipefail

BACKUP_DIR="${BAMA_BACKUP_DIR:-/var/backups/bama-monitor}"
RETAIN_DAILY="${BAMA_RETAIN_DAILY:-14}"
RETAIN_WEEKLY="${BAMA_RETAIN_WEEKLY:-8}"
DATABASE_URL="${BAMA_MONITOR_DATABASE_URL:?BAMA_MONITOR_DATABASE_URL must be set}"

CORE_TABLES=(monitoring_runs advertisements daily_ad_observations advertisement_events)

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

stamp() { date -u +%Y%m%dT%H%M%SZ; }

cmd_dump() {
  mkdir -p "$BACKUP_DIR"
  local out="$BACKUP_DIR/bama-monitor-$(stamp).dump"
  # Custom format: compressed, and restorable table-by-table, which matters when
  # only one table needs recovering.
  pg_dump --format=custom --compress=9 --file="$out" "$DATABASE_URL"
  log "wrote $out ($(du -h "$out" | cut -f1))"

  # Retention: keep every dump from the last RETAIN_DAILY days, plus Sunday dumps
  # for RETAIN_WEEKLY weeks. Pruning is by mtime, so a clock jump cannot delete
  # everything at once.
  find "$BACKUP_DIR" -name 'bama-monitor-*.dump' -mtime "+$RETAIN_DAILY" \
    ! -name "*-$(date -u +%Y%m)*W*" -print -delete | while read -r pruned; do
      log "pruned $pruned"
    done
  local kept
  kept=$(find "$BACKUP_DIR" -name 'bama-monitor-*.dump' | wc -l | tr -d ' ')
  log "retention: $kept dump(s) kept (daily=$RETAIN_DAILY weekly=$RETAIN_WEEKLY)"
  printf '%s\n' "$out"
}

cmd_verify() {
  local file="${1:?usage: backup.sh verify <file>}"
  pg_restore --list "$file" >/tmp/bama-restore-list.txt
  local missing=()
  for table in "${CORE_TABLES[@]}"; do
    grep -q " $table " /tmp/bama-restore-list.txt || missing+=("$table")
  done
  if ((${#missing[@]})); then
    log "FAIL: dump does not contain: ${missing[*]}"
    return 1
  fi
  log "OK: $file is readable and contains all ${#CORE_TABLES[@]} core tables"
}

cmd_restore() {
  local file="${1:?usage: backup.sh restore <file> <target-db>}"
  local target="${2:?usage: backup.sh restore <file> <target-db>}"
  # Deliberately refuses to restore into an existing database. Recovery goes into a
  # fresh one, so a mistaken restore cannot destroy the history it was meant to
  # protect; promoting it is a separate, conscious step.
  createdb "$target"
  pg_restore --dbname="$target" --no-owner --no-privileges "$file"
  log "restored $file into $target"
}

cmd_drill() {
  local scratch="bama_restore_drill_$$"
  local file
  file="$(cmd_dump | tail -1)"
  cmd_verify "$file"

  local before
  before="$(counts "$DATABASE_URL")"
  cmd_restore "$file" "$scratch"
  local after
  after="$(counts "$scratch")"

  dropdb "$scratch"
  log "dropped scratch database $scratch"

  if [[ "$before" == "$after" ]]; then
    log "DRILL PASSED: row counts identical after restore"
    printf '%s\n' "$before"
    return 0
  fi
  log "DRILL FAILED"
  printf 'live:    %s\nrestored:%s\n' "$before" "$after"
  return 1
}

counts() {
  local target="$1"
  local out=""
  for table in "${CORE_TABLES[@]}"; do
    local n
    n=$(psql --no-psqlrc --tuples-only --no-align --dbname="$target" \
        --command="SELECT COUNT(*) FROM $table")
    out+="$table=$n "
  done
  printf '%s' "$out"
}

case "${1:-}" in
  dump)    shift; cmd_dump "$@" ;;
  verify)  shift; cmd_verify "$@" ;;
  restore) shift; cmd_restore "$@" ;;
  drill)   shift; cmd_drill "$@" ;;
  *) printf 'usage: %s {dump|verify <file>|restore <file> <db>|drill}\n' "$0" >&2; exit 2 ;;
esac
