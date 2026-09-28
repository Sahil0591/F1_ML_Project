"""Optional FastF1 lap summaries without telemetry ingestion."""

import importlib
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.time import require_utc

SESSION_CODES = frozenset({"FP1", "FP2", "FP3", "Q"})


def _validate_session(session_code: str) -> None:
    if session_code not in SESSION_CODES:
        raise ValueError("session_code must be FP1, FP2, FP3, or Q")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except TypeError:
        # pandas.NA has no truth value; NaT and NaN compare unequal to themselves.
        return True


def summarize_laps(
    records: list[dict[str, Any]],
    event: EventId,
    session_code: str,
    driver_ids: dict[str, str],
    captured_at: datetime,
) -> list[dict[str, Any]]:
    """Aggregate known accurate, undeleted, non-pit laps by driver and compound."""
    _validate_session(session_code)
    require_utc(captured_at, "captured_at")
    groups: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for record in records:
        number = record.get("DriverNumber")
        if not isinstance(number, str) or number not in driver_ids:
            raise ValueError(f"No canonical driver ID mapping for FastF1 number {number!r}")
        if record.get("IsAccurate") is not True or record.get("Deleted") is not False:
            continue
        if record.get("FastF1Generated") is True:
            continue
        if not _missing(record.get("PitInTime")) or not _missing(record.get("PitOutTime")):
            continue
        lap_time = record.get("LapTime")
        if not isinstance(lap_time, timedelta):
            continue
        seconds = lap_time.total_seconds()
        if not math.isfinite(seconds) or seconds <= 0:
            continue
        compound = record.get("Compound")
        if not isinstance(compound, str) or not compound:
            continue
        times, ages = groups.setdefault((driver_ids[number], compound), ([], []))
        times.append(seconds)
        age = record.get("TyreLife")
        if isinstance(age, (int, float)) and not isinstance(age, bool):
            if math.isfinite(age) and age >= 0:
                ages.append(float(age))

    return [
        {
            "event_id": event.partition(),
            "driver_id": driver_id,
            "session_code": session_code,
            "compound": compound,
            "lap_count": len(times),
            "median_lap_seconds": float(median(times)),
            "best_lap_seconds": min(times),
            "median_tyre_age": float(median(ages)) if ages else None,
            "available_at": captured_at,
        }
        for (driver_id, compound), (times, ages) in sorted(groups.items())
    ]


def load_session_summary(
    event: EventId,
    session_code: str,
    driver_ids: dict[str, str],
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """Load lap timing using the optional FastF1 dependency and its local cache."""
    _validate_session(session_code)
    try:
        fastf1 = importlib.import_module("fastf1")
    except ModuleNotFoundError as exc:
        if exc.name != "fastf1":
            raise
        raise RuntimeError(
            "FastF1 session ingestion requires the optional dependency: "
            "install f1-ml-predictor[fastf1]"
        ) from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    session = fastf1.get_session(event.season, event.round, session_code)
    # Race control messages supply deletion flags required for valid lap filtering.
    session.load(laps=True, telemetry=False, weather=False, messages=True)
    captured_at = datetime.now(UTC)
    records = session.laps.to_dict(orient="records")
    resolved_ids = driver_ids.copy()
    results = getattr(session, "results", None)
    if results is not None:
        for result in results.to_dict(orient="records"):
            number = result.get("DriverNumber")
            source_id = result.get("DriverId")
            if not isinstance(number, str) or not isinstance(source_id, str) or not source_id:
                continue
            canonical_id = EntityId(EntityKind.DRIVER, source_id).value
            if number in resolved_ids and resolved_ids[number] != canonical_id:
                raise ValueError(f"Conflicting canonical driver IDs for FastF1 number {number!r}")
            resolved_ids[number] = canonical_id
    return summarize_laps(records, event, session_code, resolved_ids, captured_at)
