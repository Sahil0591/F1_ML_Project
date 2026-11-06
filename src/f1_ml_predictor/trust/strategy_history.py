"""Audit OpenF1 tyre, pit and long-run evidence for Gold races from 2023.

OpenF1 serves completed sessions free from 30 minutes after they end, but does
not certify when each record was published. A race's stints, pit stops and laps
therefore count from RACE_MARGIN after its scheduled end, and a practice
session's from PRACTICE_MARGIN after its end. Response bytes are retained and
the per-session summaries are replayed from them when the audit loads.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from f1_ml_predictor.benchmarks.builder import _safe_file, file_sha256
from f1_ml_predictor.sources.http import JsonSourceClient
from f1_ml_predictor.trust.historical import _immutable, _json
from f1_ml_predictor.trust.winter import DRIVER_ALIASES

BASE_URL = "https://api.openf1.org/v1"
FIRST_SEASON = 2023
RACE_MARGIN = timedelta(hours=24)
PRACTICE_MARGIN = timedelta(hours=2)
RACE_ENDPOINTS = ("drivers", "stints", "pit", "laps", "weather")
PRACTICE_ENDPOINTS = ("drivers", "stints", "laps")
DRY = ("SOFT", "MEDIUM", "HARD")
WET = ("INTERMEDIATE", "WET")
# A clean lap is within this factor of its stint's median (drops safety car laps).
CLEAN_FACTOR = 1.05
MIN_STINT_LAPS = 6
MIN_RUN_LAPS = 5
LANE_DELTA_BOUNDS = (-5.0, 10.0)
AUDIT_VERSION = "strategy-audit-v2"
# Bump when race_summary or practice_summary changes, to invalidate cached replays.
SUMMARY_VERSION = "strategy-summary-v2"
_RAW = Path("data/raw/openf1_strategy")
_AUDITS = Path("data/benchmarks/historical_audit")
_LIMITS = ((3, 1.0), (28, 60.0))


def _plain(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()


def driver_map(drivers: list[dict[str, Any]], roster: set[str]) -> dict[int, str]:
    """OpenF1 car numbers to Gold driver IDs in this roster; reserves are left out."""
    result: dict[int, str] = {}
    fallback: dict[int, str] = {}
    for item in drivers:
        number = item.get("driver_number")
        if not isinstance(number, int):
            continue
        known = DRIVER_ALIASES.get(str(item.get("full_name") or ""))
        if known in roster:
            result[number] = known
            continue
        last = _plain(str(item.get("last_name") or "")).replace(" ", "_")
        matches = [d for d in roster if last and (d == last or d.endswith(f"_{last}"))]
        if len(matches) == 1:
            fallback[number] = matches[0]
    if len(set(result.values())) != len(result):
        raise ValueError("two OpenF1 car numbers map to one Gold driver")
    # A surname match only fills a driver no exact name claimed, and only when one
    # number claims it: a reserve such as Arthur Leclerc never becomes Charles.
    claimed = set(result.values())
    for number, driver in fallback.items():
        if driver not in claimed and list(fallback.values()).count(driver) == 1:
            result[number] = driver
    return result


def _clean_laps(
    laps: list[dict[str, Any]], stints: list[dict[str, Any]], pit_laps: set[tuple[int, int]]
) -> list[tuple[int, str, list[tuple[float, float]]]]:
    """Per stint: driver, compound and (tyre age, lap time) for clean laps."""
    by_driver: dict[int, dict[int, float]] = {}
    for lap in laps:
        number, index, duration = (
            lap.get("driver_number"),
            lap.get("lap_number"),
            lap.get("lap_duration"),
        )
        if (
            isinstance(number, int)
            and isinstance(index, int)
            and isinstance(duration, (int, float))
            and duration > 0
            and not lap.get("is_pit_out_lap")
            and (number, index) not in pit_laps
        ):
            by_driver.setdefault(number, {})[index] = float(duration)
    result = []
    for stint in stints:
        number = stint.get("driver_number")
        start, end = stint.get("lap_start"), stint.get("lap_end")
        compound = str(stint.get("compound") or "").upper()
        age = stint.get("tyre_age_at_start") or 0
        if not isinstance(number, int) or not isinstance(start, int) or not isinstance(end, int):
            continue
        times = by_driver.get(number, {})
        points = [
            (float(age + lap - start), times[lap])
            for lap in range(start + 1, end + 1)
            if lap in times
        ]
        if not points:
            continue
        median = float(np.median([time for _, time in points]))
        result.append((number, compound, [(a, t) for a, t in points if t <= median * CLEAN_FACTOR]))
    return result


def _slope(points: list[tuple[float, float]]) -> float | None:
    ages = np.asarray([a for a, _ in points])
    if len(points) < MIN_STINT_LAPS or np.ptp(ages) == 0:
        return None
    return float(np.polyfit(ages, [t for _, t in points], 1)[0])


def _typical(pits: list[dict[str, Any]], field: str, spread: float) -> float | None:
    """Median of the normal-stop cluster: values within ``spread`` times the fastest."""
    values = [
        float(item[field])
        for item in pits
        if isinstance(item.get(field), (int, float)) and item[field] > 0
    ]
    if not values:
        return None
    fastest = min(values)
    return float(np.median([value for value in values if value <= fastest * spread]))


def race_summary(documents: dict[str, list[dict[str, Any]]], roster: set[str]) -> dict[str, Any]:
    """Stops, pit-lane deltas, stationary times and dry tyre degradation per driver."""
    numbers = driver_map(documents["drivers"], roster)
    pits = [item for item in documents["pit"] if item.get("driver_number") in numbers]
    lane_median = _typical(pits, "pit_duration", 1.3)
    stop_median = _typical(pits, "stop_duration", 2.0)
    drivers: dict[str, dict[str, Any]] = {
        driver: {"stints": 0, "lane_deltas": [], "stop_deltas": [], "degradation": []}
        for driver in numbers.values()
    }
    for item in pits:
        entry = drivers[numbers[item["driver_number"]]]
        lane, stop = item.get("pit_duration"), item.get("stop_duration")
        # Red-flag pit-lane waits and repairs are not crew performance, so they are
        # dropped; a slow but ordinary stop is kept.
        if lane_median is not None and isinstance(lane, (int, float)):
            if lane - lane_median <= LANE_DELTA_BOUNDS[1]:
                entry["lane_deltas"].append(max(lane - lane_median, LANE_DELTA_BOUNDS[0]))
        if stop_median is not None and isinstance(stop, (int, float)) and stop > 0:
            if stop - stop_median <= 8.0:
                entry["stop_deltas"].append(min(stop - stop_median, 5.0))
    compounds = {str(item.get("compound") or "").upper() for item in documents["stints"]}
    for stint in documents["stints"]:
        if stint.get("driver_number") in numbers:
            drivers[numbers[stint["driver_number"]]]["stints"] += 1
    pit_laps = {
        (item["driver_number"], item["lap_number"])
        for item in pits
        if isinstance(item.get("lap_number"), int)
    }
    slopes: list[tuple[str, str, float, int]] = []
    for number, compound, points in _clean_laps(documents["laps"], documents["stints"], pit_laps):
        if number in numbers and compound in DRY:
            slope = _slope(points)
            if slope is not None:
                slopes.append((numbers[number], compound, slope, len(points)))
    for compound in DRY:
        field = [slope for _, c, slope, _ in slopes if c == compound]
        if len(field) < 3:
            continue
        median = float(np.median(field))
        for driver, c, slope, laps in slopes:
            if c == compound:
                drivers[driver]["degradation"].append([slope - median, laps])
    stints = [entry["stints"] for entry in drivers.values() if entry["stints"]]
    return {
        "wet": bool(compounds & set(WET)),
        "lane_median": lane_median,
        "median_stops": float(np.median(stints)) - 1 if stints else None,
        "drivers": drivers,
        **track_weather(documents.get("weather", [])),
    }


def track_weather(readings: list[dict[str, Any]]) -> dict[str, float | None]:
    """The circuit weather station over the race session: share of readings with
    rain falling, and median air and track temperature. An independent measure
    of the conditions, to check forecasts and the fitted-tyre wet label against.
    """
    rain = [item["rainfall"] for item in readings if isinstance(item.get("rainfall"), (int, float))]
    air = [
        item["air_temperature"]
        for item in readings
        if isinstance(item.get("air_temperature"), (int, float))
    ]
    track = [
        item["track_temperature"]
        for item in readings
        if isinstance(item.get("track_temperature"), (int, float))
    ]
    return {
        "track_rain_fraction": float(np.mean([value > 0 for value in rain])) if rain else None,
        "track_air_temperature": float(np.median(air)) if air else None,
        "track_surface_temperature": float(np.median(track)) if track else None,
    }


def practice_summary(
    documents: dict[str, list[dict[str, Any]]], roster: set[str]
) -> dict[str, dict[str, float]]:
    """Long-run pace and degradation per driver against the field on each dry compound."""
    numbers = driver_map(documents["drivers"], roster)
    runs: list[tuple[str, str, float, float | None]] = []
    for number, compound, points in _clean_laps(documents["laps"], documents["stints"], set()):
        if number in numbers and compound in DRY and len(points) >= MIN_RUN_LAPS:
            runs.append(
                (
                    numbers[number],
                    compound,
                    float(np.median([t for _, t in points])),
                    _slope(points),
                )
            )
    gaps: dict[str, list[float]] = {}
    degradation: dict[str, list[float]] = {}
    for compound in DRY:
        field = [run for run in runs if run[1] == compound]
        if len({run[0] for run in field}) < 3:
            continue
        pace = float(np.median([run[2] for run in field]))
        slopes = [run[3] for run in field if run[3] is not None]
        slope_median = float(np.median(slopes)) if len(slopes) >= 3 else None
        for driver, _, median, slope in field:
            gaps.setdefault(driver, []).append(median - pace)
            if slope is not None and slope_median is not None:
                degradation.setdefault(driver, []).append(slope - slope_median)
    return {
        driver: {
            "long_run_gap": float(np.mean(gaps[driver])),
            **(
                {"long_run_degradation": float(np.mean(degradation[driver]))}
                if driver in degradation
                else {}
            ),
        }
        for driver in gaps
    }


class _Fetcher:
    """Throttled OpenF1 requests, each retained once and reused on later runs."""

    def __init__(self, root: Path, client: JsonSourceClient) -> None:
        self.root = root
        self.client = client

    def get(self, endpoint: str, params: dict[str, Any]) -> tuple[Any, dict[str, str]]:
        url = f"{BASE_URL}/{endpoint}"
        key = hashlib.sha256(_json({"url": url, "params": params})).hexdigest()
        index = self.root / _RAW / "requests" / f"{key}.json"
        if index.exists():
            record = json.loads(index.read_text(encoding="utf-8"))
            return _payload(self.root, record), record
        # OpenF1 answers 404 when a session has no records of a kind.
        response = self.client.get_response(url, params, accept=(404,))
        digest = hashlib.sha256(response.content).hexdigest()
        path = self.root / _RAW / "objects" / f"{digest}.json"
        _immutable(path, response.content)
        record = {
            "path": path.relative_to(self.root).as_posix(),
            "sha256": digest,
            "url": str(response.url),
            "status": response.status_code,
        }
        payload = _payload(self.root, record)
        _immutable(index, _json(record))
        return payload, record


def _payload(root: Path, record: dict[str, Any]) -> list[Any]:
    """The retained list, or an empty list for a retained "no results" 404."""
    content = _safe_file(root, record["path"], record["sha256"]).read_bytes()
    if record.get("status", 200) == 404:
        return []
    payload = json.loads(content)
    if not isinstance(payload, list):
        raise ValueError("OpenF1 response is not a list")
    return payload


def _end(session: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(session["date_end"])


def audit_strategy_history(
    root: Path,
    events: dict[str, tuple[datetime, set[str]]],
    *,
    client: JsonSourceClient | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Freeze race and practice evidence for Gold events from 2023.

    `events` maps each Gold event to its audited race start and roster.
    """
    owned = client is None
    source = client or JsonSourceClient(_LIMITS)
    fetcher = _Fetcher(root, source)
    sessions: dict[int, list[dict[str, Any]]] = {}
    records = []
    try:
        for event_id in sorted(events):
            season = int(event_id[7:11])
            race_start, roster = events[event_id]
            record: dict[str, Any] = {"event_id": event_id, "race_start": race_start.isoformat()}
            if season < FIRST_SEASON:
                records.append({**record, "status": "before_openf1"})
                continue
            if season not in sessions:
                sessions[season], _ = fetcher.get("sessions", {"year": season})
            races = [
                item
                for item in sessions[season]
                if item.get("session_name") == "Race"
                and abs(datetime.fromisoformat(item["date_start"]) - race_start)
                <= timedelta(hours=3)
            ]
            if len(races) != 1:
                records.append({**record, "status": "no_matching_race_session"})
                continue
            race = races[0]
            practice = sorted(
                (
                    item
                    for item in sessions[season]
                    if item.get("meeting_key") == race["meeting_key"]
                    and item.get("session_type") == "Practice"
                    and not item.get("is_cancelled")
                ),
                key=lambda item: item["date_start"],
            )
            entry: dict[str, Any] = {
                **record,
                "status": "audited",
                "race": {
                    "session_key": race["session_key"],
                    "date_end": race["date_end"],
                    "available_at": (_end(race) + RACE_MARGIN).isoformat(),
                    "documents": {},
                },
                "practice": [],
            }
            for endpoint in RACE_ENDPOINTS:
                _, document = fetcher.get(endpoint, {"session_key": race["session_key"]})
                entry["race"]["documents"][endpoint] = document
            for session in practice:
                item = {
                    "session_key": session["session_key"],
                    "session_name": session["session_name"],
                    "date_end": session["date_end"],
                    "available_at": (_end(session) + PRACTICE_MARGIN).isoformat(),
                    "documents": {},
                }
                for endpoint in PRACTICE_ENDPOINTS:
                    _, document = fetcher.get(endpoint, {"session_key": session["session_key"]})
                    item["documents"][endpoint] = document
                entry["practice"].append(item)
            records.append(entry)
            if progress is not None:
                progress(f"{event_id} race {race['session_key']} practice {len(practice)}")
    finally:
        if owned:
            source.close()
    audit = {
        "version": AUDIT_VERSION,
        "race_margin_hours": RACE_MARGIN.total_seconds() / 3600,
        "practice_margin_hours": PRACTICE_MARGIN.total_seconds() / 3600,
        "races": records,
    }
    content = _json(audit)
    digest = hashlib.sha256(content).hexdigest()
    path = root / _AUDITS / f"strategy-audit-{digest}.json"
    _immutable(path, content)
    counts: dict[str, int] = {}
    for item in records:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {"path": path.relative_to(root).as_posix(), "races": len(records), "status": counts}


def _documents(root: Path, documents: dict[str, dict[str, str]]) -> dict[str, list[Any]]:
    return {endpoint: _payload(root, item) for endpoint, item in documents.items()}


def load_strategy_history(
    root: Path, rosters: dict[str, set[str]]
) -> tuple[dict[str, dict[str, Any]], str] | None:
    """Verified strategy audit for exactly these Gold races, summarised, or None.

    Every summary is recomputed from the retained OpenF1 bytes for the Gold roster.
    """
    matches = []
    for path in (root / _AUDITS).glob("strategy-audit-*.json"):
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("version") == AUDIT_VERSION and {
            record["event_id"] for record in audit["races"]
        } == set(rosters):
            matches.append((path.stat().st_mtime, path.name, path, audit))
    if not matches:
        return None
    _, _, path, audit = max(matches)
    digest = file_sha256(path)
    if path.name != f"strategy-audit-{digest}.json":
        raise ValueError("strategy audit name does not match its content hash")
    # Replaying every retained laps file is slow, so the replay is cached per audit,
    # roster set and summary code version; deleting the cache forces a full replay.
    key = hashlib.sha256(
        _json({"rosters": {k: sorted(v) for k, v in rosters.items()}, "code": SUMMARY_VERSION})
    ).hexdigest()
    cache = root / _AUDITS / "strategy-summaries" / f"{digest}-{key}.json"
    if cache.exists():
        return _revive(json.loads(cache.read_text(encoding="utf-8"))), digest
    result = _replay(root, audit, rosters)
    _immutable(cache, _json(_freeze(result)))
    return result, digest


def _freeze(result: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        event: {
            **item,
            "race_start": item["race_start"].isoformat(),
            "available_at": item["available_at"].isoformat(),
            "practice": [
                {**session, "available_at": session["available_at"].isoformat()}
                for session in item["practice"]
            ],
        }
        for event, item in result.items()
    }


def _revive(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        event: {
            **item,
            "race_start": datetime.fromisoformat(item["race_start"]),
            "available_at": datetime.fromisoformat(item["available_at"]),
            "practice": [
                {**session, "available_at": datetime.fromisoformat(session["available_at"])}
                for session in item["practice"]
            ],
        }
        for event, item in frozen.items()
    }


def _replay(
    root: Path, audit: dict[str, Any], rosters: dict[str, set[str]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in audit["races"]:
        if record["status"] != "audited":
            continue
        roster = rosters[record["event_id"]]
        race = record["race"]
        available = datetime.fromisoformat(race["date_end"]) + RACE_MARGIN
        if available.isoformat() != race["available_at"]:
            raise ValueError("strategy audit race availability differs from the margin rule")
        practice = []
        for session in record["practice"]:
            session_available = datetime.fromisoformat(session["date_end"]) + PRACTICE_MARGIN
            if session_available.isoformat() != session["available_at"]:
                raise ValueError("strategy audit practice availability differs from the rule")
            practice.append(
                {
                    "session_name": session["session_name"],
                    "available_at": session_available,
                    "drivers": practice_summary(_documents(root, session["documents"]), roster),
                }
            )
        result[record["event_id"]] = {
            "race_start": datetime.fromisoformat(record["race_start"]),
            "available_at": available,
            "race": race_summary(_documents(root, race["documents"]), roster),
            "practice": practice,
        }
    return result
