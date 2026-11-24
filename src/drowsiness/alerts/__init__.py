"""Alerts: present logic's driver state to the driver (banner, sound, log, webhook).

Alerts never decide: they only present what logic decided. See docs/architecture.md.
"""

from __future__ import annotations

from drowsiness.alerts.manager import AlertDecision, AlertManager, AlertSink, SoundCue
from drowsiness.alerts.sinks import LogSink, WebhookSink, log_event
from drowsiness.config import Config

__all__ = [
    "AlertDecision", "AlertManager", "AlertSink", "LogSink", "SoundCue", "WebhookSink",
    "build_alerts", "log_event",
]  # fmt: skip


# สร้างระบบแจ้งเตือนตาม config: log เปิดเสมอ, เสียงถ้าไม่ mute, webhook ถ้ามี URL
def build_alerts(cfg: Config) -> AlertManager:
    """AlertManager with the configured sinks; use as a context manager (stops sound)."""
    a = cfg.alerts
    sinks: list[AlertSink] = [LogSink()]
    if a.sound:
        # import ตรงนี้ (lazy): ถ้า mute จะไม่โหลดไลบรารีเสียงเลย
        from drowsiness.alerts.sound import SoundSink

        sinks.append(SoundSink(a))
    if a.webhook_url is not None:
        sinks.append(WebhookSink(a.webhook_url, a.webhook_timeout_s))
    return AlertManager(a, sinks)
