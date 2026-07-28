# Container image for the monitoring run.
#
# The container is a *task*, not a daemon: it performs one run and exits, so the
# host's systemd timer remains the scheduler. That keeps a single source of truth
# for "did today's run happen?" instead of splitting it between systemd and a
# scheduler inside the image.

FROM python:3.12-slim AS base

# tzdata is required for Europe/Paris; the rest is for psycopg and Playwright's
# optional browser path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata ca-certificates libpq5 \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Paris \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Unprivileged runtime user.
RUN useradd --create-home --shell /usr/sbin/nologin bama
WORKDIR /app

# Dependency layer first so code changes do not invalidate it.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install "psycopg[binary]" apscheduler tzdata

# Application code and migrations.
COPY migrations ./migrations
COPY deep_scraper ./deep_scraper
COPY deploy/config.example.yaml /etc/bama-monitor/config.yaml

RUN mkdir -p /data/monitor_data /data/reports && chown -R bama:bama /app /data
USER bama

VOLUME ["/data"]

# Fails the healthcheck when the database is unreachable or the schedule cannot
# be resolved, which is what an orchestrator should react to.
HEALTHCHECK --interval=5m --timeout=30s --start-period=20s --retries=3 \
    CMD python -m bama_monitor validate --config /etc/bama-monitor/config.yaml >/dev/null || exit 1

ENTRYPOINT ["python", "-m", "bama_monitor"]
CMD ["run-daily", "--config", "/etc/bama-monitor/config.yaml"]
