"""Availability checks for historical prediction snapshots."""

from datetime import datetime, timedelta


def require_utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")


def require_known_by(available_at: datetime, prediction_timestamp: datetime) -> None:
    """Reject input that was not available at the prediction cutoff."""
    require_utc(available_at, "available_at")
    require_utc(prediction_timestamp, "prediction_timestamp")
    if available_at > prediction_timestamp:
        raise ValueError("input became available after prediction_timestamp")
