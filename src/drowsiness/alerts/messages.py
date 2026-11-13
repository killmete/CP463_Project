"""Driver-facing text for each reason (ASCII only: OpenCV Hershey fonts; <= 32 chars).

Short and actionable: say what was seen and what to do.
"""

from __future__ import annotations

from drowsiness.types import AlertLevel

MAX_CHARS = 32

MESSAGES: dict[str, str] = {
    # DROWSY_ALARM
    "sleep": "WAKE UP - pull over safely",
    "perclos_alarm": "Very drowsy - pull over and rest",
    # NO_FACE / DISTRACTED
    "no_face": "Face not visible - check camera",
    "long_distraction": "Look at the road",
    "short_distraction": "Keep your eyes on the road",
    # DROWSY_WARNING
    "microsleep": "Eyes closing - take a break",
    "perclos_warning": "Drowsy - take a break soon",
    "head_nod": "Head nodding - take a break",
    "yawning": "Frequent yawns - take a break",
    # INFO (shown briefly, never sounded)
    "calibrated": "Calibrated - monitoring",
    "calibration_fallback": "Calibration failed - defaults",
    "recalibrating": "Recalibrating - look ahead",
    "recalibration_failed": "Recalibration failed - kept old",
}

ESCALATED = "Drowsy again - stop and rest"
CALIBRATING = "Calibrating - look ahead"

_GENERIC = {
    AlertLevel.ALARM: "DANGER - pull over safely",
    AlertLevel.WARNING: "Attention - watch the road",
    AlertLevel.INFO: "",
}


def message_for(reason: str | None, level: AlertLevel) -> str:
    """Banner text for `reason`; an unknown reason gets a generic text for its level."""
    if reason is not None and reason in MESSAGES:
        return MESSAGES[reason]
    return _GENERIC[level]
