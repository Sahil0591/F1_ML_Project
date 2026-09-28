"""Typed weather forecast values separated from their target and availability times."""

import math
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.sources.open_meteo import HOURLY_VARIABLES, ForecastSnapshot
from f1_ml_predictor.time import require_utc

FORECAST_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("weather_kind", pa.string(), nullable=False),
        pa.field("valid_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("available_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("captured_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("run_initialized_at", pa.timestamp("us", tz="UTC")),
        pa.field("availability_evidence", pa.string()),
        *[pa.field(name, pa.float64()) for name in HOURLY_VARIABLES],
    ]
)


def normalize_forecast(event: EventId, snapshot: ForecastSnapshot) -> pa.Table:
    require_utc(snapshot.available_at, "available_at")
    require_utc(snapshot.captured_at, "captured_at")
    if snapshot.available_at > snapshot.captured_at:
        raise ValueError("forecast availability cannot follow its capture")
    if snapshot.run_initialized_at is not None:
        require_utc(snapshot.run_initialized_at, "run_initialized_at")
        if snapshot.available_at < snapshot.run_initialized_at:
            raise ValueError("forecast cannot be available before initialization")
        if not snapshot.availability_evidence or not snapshot.availability_evidence.strip():
            raise ValueError("archived forecast availability evidence is required")
    elif snapshot.available_at != snapshot.captured_at:
        raise ValueError("live forecast availability must equal its capture time")
    if snapshot.payload.get("utc_offset_seconds", 0) != 0:
        raise ValueError("forecast response must use UTC")
    hourly = snapshot.payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ValueError("forecast hourly.time must be a list")
    times = hourly["time"]
    for variable in HOURLY_VARIABLES:
        if not isinstance(hourly.get(variable), list) or len(hourly[variable]) != len(times):
            raise ValueError(f"forecast {variable} has inconsistent length")
    rows: list[dict[str, Any]] = []
    seen: set[datetime] = set()
    for index, value in enumerate(times):
        if not isinstance(value, str):
            raise ValueError("forecast time must be an ISO timestamp")
        valid_at = datetime.fromisoformat(value)
        if valid_at.tzinfo is None:
            valid_at = valid_at.replace(tzinfo=UTC)
        require_utc(valid_at, "valid_at")
        if valid_at in seen:
            raise ValueError("forecast contains duplicate valid times")
        seen.add(valid_at)
        row: dict[str, Any] = {
            "event_id": event.partition(),
            "weather_kind": "forecast",
            "valid_at": valid_at,
            "available_at": snapshot.available_at,
            "captured_at": snapshot.captured_at,
            "run_initialized_at": snapshot.run_initialized_at,
            "availability_evidence": snapshot.availability_evidence,
        }
        for variable in HOURLY_VARIABLES:
            number = hourly[variable][index]
            if number is not None and (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
            ):
                raise ValueError(f"forecast {variable} must be finite or missing")
            row[variable] = number
        probability = row["precipitation_probability"]
        if probability is not None and not 0 <= probability <= 100:
            raise ValueError("precipitation_probability must be between 0 and 100")
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=FORECAST_SCHEMA)
