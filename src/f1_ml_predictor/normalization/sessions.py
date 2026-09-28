"""Driver lap summaries with capture-time availability and explicit source identity."""

import math
from datetime import datetime
from statistics import median
from typing import Any

import pyarrow as pa

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.time import require_known_by, require_utc

SESSION_CODES = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Qualifying": "Q"}
SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("session_code", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("quality_policy", pa.string(), nullable=False),
        pa.field("compound", pa.string()),
        pa.field("lap_count", pa.int32(), nullable=False),
        pa.field("median_lap_seconds", pa.float64(), nullable=False),
        pa.field("best_lap_seconds", pa.float64(), nullable=False),
        pa.field("median_tyre_age", pa.float64()),
        pa.field("available_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def normalize_summaries(rows: list[dict[str, Any]], source: str) -> pa.Table:
    normalized = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for row in rows:
        EntityId(EntityKind.DRIVER, row["driver_id"])
        if row["session_code"] not in SESSION_CODES.values():
            raise ValueError("summary must be from practice or qualifying")
        require_utc(row["available_at"], "available_at")
        key = (row["event_id"], row["driver_id"], row["session_code"], row["compound"])
        if key in seen:
            raise ValueError("duplicate session summary")
        seen.add(key)
        if (
            isinstance(row["lap_count"], bool)
            or not isinstance(row["lap_count"], int)
            or row["lap_count"] < 1
        ):
            raise ValueError("summary must contain a lap")
        for name in ("median_lap_seconds", "best_lap_seconds"):
            if not math.isfinite(row[name]) or row[name] <= 0:
                raise ValueError("summary lap times must be finite and positive")
        if row["best_lap_seconds"] > row["median_lap_seconds"]:
            raise ValueError("best lap cannot be slower than the median")
        age = row["median_tyre_age"]
        if age is not None and (
            isinstance(age, bool)
            or not isinstance(age, (int, float))
            or not math.isfinite(age)
            or age < 0
        ):
            raise ValueError("tyre age must be finite and nonnegative or missing")
        policy = (
            "accurate_undeleted_nonpit_generated_flag_filter"
            if source == "fastf1"
            else "positive_quicklap_nonpit_deleted_flag_unavailable"
        )
        normalized.append({**row, "source": source, "quality_policy": policy})
    return pa.Table.from_pylist(normalized, schema=SUMMARY_SCHEMA)


def summarize_openf1(
    event: EventId,
    session: dict[str, Any],
    laps: list[dict[str, Any]],
    stints: list[dict[str, Any]],
    pits: list[dict[str, Any]],
    driver_ids: dict[str, str],
    captured_at: datetime,
) -> list[dict[str, Any]]:
    require_utc(captured_at, "captured_at")
    if session.get("session_name") not in SESSION_CODES:
        raise ValueError("race session data is not allowed in pre-race summaries")
    if not isinstance(session.get("date_end"), str):
        raise ValueError("session end timestamp is required")
    end = datetime.fromisoformat(session["date_end"])
    require_known_by(end, captured_at)
    start = datetime.fromisoformat(session["date_start"]) if session.get("date_start") else None
    if start is not None:
        require_known_by(start, end)
    session_key = session.get("session_key")
    for collection in (laps, stints, pits):
        for record in collection:
            if record.get("session_key") is not None and record["session_key"] != session_key:
                raise ValueError("record does not belong to the selected session")
    code = SESSION_CODES[session["session_name"]]
    pit_laps = {(str(pit["driver_number"]), pit["lap_number"]) for pit in pits}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for lap in laps:
        number = str(lap["driver_number"])
        if number not in driver_ids:
            raise ValueError(f"missing canonical driver mapping for number {number}")
        if (
            isinstance(lap.get("lap_number"), bool)
            or not isinstance(lap.get("lap_number"), int)
            or lap["lap_number"] < 1
        ):
            raise ValueError("lap number must be positive")
        duration = lap.get("lap_duration")
        if duration is None or isinstance(duration, bool) or not isinstance(duration, (int, float)):
            continue
        if not math.isfinite(duration) or duration <= 0 or lap.get("is_pit_out_lap"):
            continue
        if (number, lap["lap_number"]) in pit_laps:
            continue
        if lap.get("date_start"):
            lap_start = datetime.fromisoformat(lap["date_start"])
            require_known_by(lap_start, end)
            if start is not None and lap_start < start:
                raise ValueError("lap precedes the selected session")
        candidates.setdefault(number, []).append(lap)
    groups: dict[tuple[str, str | None], list[tuple[float, float | None]]] = {}
    for number, driver_laps in candidates.items():
        threshold = min(lap["lap_duration"] for lap in driver_laps) * 1.07
        for lap in driver_laps:
            if lap["lap_duration"] > threshold:
                continue
            matches = [
                stint
                for stint in stints
                if str(stint.get("driver_number")) == number
                and isinstance(stint.get("lap_start"), int)
                and isinstance(stint.get("lap_end"), int)
                and stint["lap_start"] <= lap["lap_number"] <= stint["lap_end"]
            ]
            if len(matches) > 1:
                raise ValueError("overlapping stints cannot be assigned unambiguously")
            stint = matches[0] if matches else None
            compound = stint.get("compound") if stint else None
            tyre_age = None
            if stint and stint.get("tyre_age_at_start") is not None:
                age = stint["tyre_age_at_start"]
                if (
                    isinstance(age, bool)
                    or not isinstance(age, (int, float))
                    or not math.isfinite(age)
                    or age < 0
                ):
                    raise ValueError("stint tyre age must be finite and nonnegative")
                tyre_age = float(age + lap["lap_number"] - stint["lap_start"])
            groups.setdefault((driver_ids[number], compound), []).append(
                (lap["lap_duration"], tyre_age)
            )
    rows = []
    for (driver_id, compound), observations in sorted(
        groups.items(), key=lambda item: str(item[0])
    ):
        times = [duration for duration, _ in observations]
        ages = [age for _, age in observations if age is not None]
        rows.append(
            {
                "event_id": event.partition(),
                "driver_id": driver_id,
                "session_code": code,
                "compound": compound,
                "lap_count": len(times),
                "median_lap_seconds": median(times),
                "best_lap_seconds": min(times),
                "median_tyre_age": median(ages) if ages else None,
                "available_at": captured_at,
            }
        )
    return rows
