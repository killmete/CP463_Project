"""Alert sinks: the event log and an optional webhook. Sound is in sound.py.

Every `emit` returns immediately; slow work runs on the sink's own thread.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.request

from drowsiness.alerts.manager import AlertDecision
from drowsiness.types import AlertEvent

log = logging.getLogger("drowsiness")


def log_event(event: AlertEvent) -> None:
    details = " ".join(
        f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in event.details.items()
    )
    log.info("t=%.2f %s -> %s (%s) %s", event.timestamp, event.level.name, event.state.value,
             event.reason, details)  # fmt: skip


class LogSink:
    """Logs logic's events (INFO) and what was sounded or escalated (DEBUG)."""

    def emit(self, decision: AlertDecision) -> None:
        for event in decision.events:
            log_event(event)
        if decision.sounded:
            log.debug("t=%.2f sound %s for %s%s", decision.timestamp, decision.sound.value,
                      decision.reason, " (escalated)" if decision.escalated else "")  # fmt: skip
        elif decision.suppressed:
            log.debug("t=%.2f %s not sounded (cooldown)", decision.timestamp, decision.reason)

    def close(self) -> None:
        pass


class WebhookSink:
    """POSTs {timestamp, state, reason, level, escalated} JSON for each sounded alert.

    Off by default. Never sends images or landmarks. Failures are logged and dropped.
    """

    def __init__(self, url: str, timeout_s: float, maxsize: int = 16) -> None:
        self.url, self.timeout_s = url, timeout_s
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize)
        self._thread = threading.Thread(target=self._run, name="alert-webhook", daemon=True)
        self._thread.start()

    def emit(self, decision: AlertDecision) -> None:
        if not decision.sounded:
            return
        payload = {
            "timestamp": decision.timestamp,
            "state": decision.state.value,
            "reason": decision.reason,
            "level": decision.level.name,
            "escalated": decision.escalated,
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            log.warning("webhook queue full; alert dropped")

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            return  # daemon thread: it ends with the process
        self._thread.join(timeout=self.timeout_s + 0.5)

    def _run(self) -> None:
        while (payload := self._queue.get()) is not None:
            self.post(payload)

    def post(self, payload: dict) -> None:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s):
                pass
        except Exception as exc:  # network errors must never reach the frame loop
            log.warning("webhook POST failed: %s", exc)
