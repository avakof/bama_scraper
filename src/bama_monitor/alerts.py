"""Alerting.

Structured logging is always available and is the source of truth. Webhook and
email are optional adapters behind a common interface, and their destinations come
from the environment — nothing here reads or stores a credential.

Alerts are persisted before they are dispatched, so a delivery failure never loses
the fact that the condition occurred.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from bama_scraper.logging_config import get_logger

from .config import AlertConfig
from .models import AlertSeverity
from .repository import Repository

log = get_logger("bama_monitor.alerts")

_SEVERITY_ORDER = {AlertSeverity.INFO: 0, AlertSeverity.WARNING: 1, AlertSeverity.CRITICAL: 2}


@dataclass(slots=True)
class Alert:
    alert_type: str
    severity: AlertSeverity
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertSink(ABC):
    """One delivery channel."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Deliver the alert. Return False on failure rather than raising."""


class LogSink(AlertSink):
    """Structured log output. Always enabled."""

    def send(self, alert: Alert) -> bool:
        logger = {
            AlertSeverity.INFO: log.info,
            AlertSeverity.WARNING: log.warning,
            AlertSeverity.CRITICAL: log.error,
        }[alert.severity]
        logger(
            "monitor.alert", alert_type=alert.alert_type, message=alert.message, **alert.metadata
        )
        return True


class WebhookSink(AlertSink):
    """Generic JSON webhook, URL supplied through the environment."""

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self.url = url
        self.timeout = timeout

    def send(self, alert: Alert) -> bool:
        payload = json.dumps(
            {
                "type": alert.alert_type,
                "severity": str(alert.severity),
                "message": alert.message,
                "metadata": alert.metadata,
            },
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError) as exc:
            log.warning("monitor.alert.webhook_failed", error=repr(exc))
            return False


class EmailSink(AlertSink):
    """SMTP email. Both the SMTP URL and recipient come from the environment.

    Implemented against ``smtplib`` so no credentials are ever written to config;
    an unset environment variable simply disables the sink.
    """

    def __init__(self, smtp_url: str, recipient: str, timeout: float = 15.0) -> None:
        self.smtp_url = smtp_url
        self.recipient = recipient
        self.timeout = timeout

    def send(self, alert: Alert) -> bool:
        import smtplib
        from email.message import EmailMessage
        from urllib.parse import urlparse

        parsed = urlparse(self.smtp_url)
        message = EmailMessage()
        message["Subject"] = f"[bama-monitor:{alert.severity}] {alert.alert_type}"
        message["From"] = parsed.username or "bama-monitor@localhost"
        message["To"] = self.recipient
        message.set_content(
            f"{alert.message}\n\n"
            + json.dumps(alert.metadata, ensure_ascii=False, indent=2, default=str)
        )
        try:
            cls = smtplib.SMTP_SSL if parsed.scheme == "smtps" else smtplib.SMTP
            with cls(
                parsed.hostname or "localhost", parsed.port or 25, timeout=self.timeout
            ) as smtp:
                if parsed.scheme == "smtp":
                    with contextlib_suppress():
                        smtp.starttls()
                if parsed.username and parsed.password:
                    smtp.login(parsed.username, parsed.password)
                smtp.send_message(message)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("monitor.alert.email_failed", error=repr(exc))
            return False


def contextlib_suppress():  # small helper to keep the SMTP path readable
    import contextlib

    return contextlib.suppress(Exception)


class AlertManager:
    """Persists alerts, then fans them out to the configured sinks."""

    def __init__(
        self, repo: Repository, cfg: AlertConfig, *, extra_sinks: list[AlertSink] | None = None
    ):
        self.repo = repo
        self.cfg = cfg
        self.sinks: list[AlertSink] = [LogSink()]
        self.dispatched: list[Alert] = []

        webhook = os.environ.get(cfg.webhook_url_env)
        if webhook:
            self.sinks.append(WebhookSink(webhook))
        smtp_url = os.environ.get(cfg.smtp_url_env)
        recipient = os.environ.get(cfg.email_to_env)
        if smtp_url and recipient:
            self.sinks.append(EmailSink(smtp_url, recipient))
        if extra_sinks:
            self.sinks.extend(extra_sinks)

    def _meets_threshold(self, severity: AlertSeverity) -> bool:
        minimum = AlertSeverity(self.cfg.min_severity)
        return _SEVERITY_ORDER[severity] >= _SEVERITY_ORDER[minimum]

    def raise_alert(
        self,
        alert_type: str,
        message: str,
        *,
        severity: AlertSeverity | str = AlertSeverity.WARNING,
        run_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record and dispatch one alert.

        Persistence happens first and unconditionally: an alert that fails to
        deliver must still be visible in the database afterwards.
        """
        severity = AlertSeverity(str(severity))
        alert = Alert(
            alert_type=alert_type, severity=severity, message=message, metadata=metadata or {}
        )
        self.repo.record_alert(
            run_id=run_id,
            severity=str(severity),
            alert_type=alert_type,
            message=message,
            metadata=metadata,
        )
        self.dispatched.append(alert)
        if not self.cfg.enabled or not self._meets_threshold(severity):
            return
        for sink in self.sinks:
            sink.send(alert)

    def raise_many(self, payloads: list[dict[str, Any]], *, run_id: int | None = None) -> None:
        for payload in payloads:
            self.raise_alert(
                payload["alert_type"],
                payload["message"],
                severity=payload.get("severity", AlertSeverity.WARNING),
                run_id=run_id,
                metadata=payload.get("metadata"),
            )


#: Conditions the runner checks explicitly, so the set is documented in one place.
ALERT_TYPES = (
    "daily_run_failed",
    "access_blocked",
    "discovered_count_collapsed",
    "scraper_selector_changed",
    "no_advertisements_found",
    "detail_failure_rate_exceeded",
    "overlapping_run_skipped",
    "unusually_high_removal_rate",
    "database_unavailable",
)
