"""Audit archived race-time weather forecasts against frozen Gold cutoffs.

Each race uses one ECMWF IFS HRES run from the Open-Meteo Single Runs archive:
the latest synoptic run whose initialisation is at least RELEASE_MARGIN before
the cutoff. Open-Meteo documents global model output as public 4-6 hours after
initialisation, so the margin bounds publication without a per-run certificate.
Races before the archive start keep missing weather.
"""

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from f1_ml_predictor.benchmarks.builder import _safe_file, file_sha256
from f1_ml_predictor.trust.historical import _immutable, _json

MODEL = "ecmwf_ifs"
ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_START = datetime(2024, 3, 14, tzinfo=UTC)
RELEASE_MARGIN = timedelta(hours=12)
# The 06Z and 18Z runs are not archived for the whole period; 00Z and 12Z are.
RUN_HOURS = (0, 12)
VARIABLES = ("temperature_2m", "precipitation", "wind_speed_10m")
RACE_HOURS = 2
# Rain in the hours before the start leaves the track wet even if it stops.
PRE_RACE_HOURS = 6
AVAILABILITY_EVIDENCE = (
    "Open-Meteo Single Runs API: global model output is generally available 4-6 hours "
    "after initialisation; a 12 hour margin bounds publication"
)
AUDIT_VERSION = "weather-forecast-audit-v2"
_RAW = Path("data/raw/open_meteo_runs")
_AUDITS = Path("data/benchmarks/historical_audit")


def forecast_run(cutoff: datetime) -> datetime:
    """Latest synoptic initialisation at least RELEASE_MARGIN before the cutoff."""
    latest = (cutoff - RELEASE_MARGIN).astimezone(UTC)
    hour = max(hour for hour in RUN_HOURS if hour <= latest.hour)
    return latest.replace(hour=hour, minute=0, second=0, microsecond=0)


def race_weather(payload: dict[str, Any], race_start: datetime) -> dict[str, float | None]:
    """Temperature and wind at the first hour from the start; rain before and during.

    Open-Meteo precipitation at a valid time is the sum over the preceding hour,
    so the race window is the RACE_HOURS valid times after the start and the
    pre-race window the PRE_RACE_HOURS valid times up to the start.
    """
    if payload.get("utc_offset_seconds", 0) != 0:
        raise ValueError("forecast response must use UTC")
    hourly = payload["hourly"]
    times = [datetime.fromisoformat(value).replace(tzinfo=UTC) for value in hourly["time"]]
    for name in VARIABLES:
        if len(hourly.get(name) or []) != len(times):
            raise ValueError(f"forecast {name} has inconsistent length")

    def value(name: str, index: int) -> float | None:
        number = hourly[name][index]
        if number is None:
            return None
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"forecast {name} must be numeric or missing")
        if not math.isfinite(number):
            raise ValueError(f"forecast {name} must be finite")
        return float(number)

    start = next(
        (
            index
            for index, at in enumerate(times)
            if race_start <= at < race_start + timedelta(hours=1)
        ),
        None,
    )
    window = [
        index
        for index, at in enumerate(times)
        if race_start < at <= race_start + timedelta(hours=RACE_HOURS)
    ]
    rain = [value("precipitation", index) for index in window]
    before = [
        value("precipitation", index)
        for index, at in enumerate(times)
        if race_start - timedelta(hours=PRE_RACE_HOURS) < at <= race_start
    ]
    return {
        "forecast_temperature_2m": None if start is None else value("temperature_2m", start),
        "forecast_wind_speed_10m": None if start is None else value("wind_speed_10m", start),
        "forecast_race_precipitation_mm": (
            None
            if len(window) != RACE_HOURS or any(item is None for item in rain)
            else float(sum(rain))  # type: ignore[arg-type]
        ),
        "forecast_pre_race_precipitation_mm": (
            None
            if len(before) != PRE_RACE_HOURS or any(item is None for item in before)
            else float(sum(before))  # type: ignore[arg-type]
        ),
    }


def _parameters(latitude: float, longitude: float, run: datetime) -> dict[str, str]:
    return {
        "latitude": f"{latitude}",
        "longitude": f"{longitude}",
        "hourly": ",".join(VARIABLES),
        "models": MODEL,
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "forecast_days": "3",
    }


def _retain(root: Path, response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    if len(response.content) > 5 * 1024 * 1024:
        raise ValueError("forecast response exceeds the bounded artifact size")
    digest = hashlib.sha256(response.content).hexdigest()
    path = root / _RAW / "objects" / f"{digest}.json"
    _immutable(path, response.content)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "url": str(response.url),
        "captured_at_utc": datetime.now(UTC).isoformat(),
    }


def circuit_locations(root: Path, seasons: list[int]) -> dict[str, tuple[float, float]]:
    """Circuit coordinates by Gold event from the retained season schedules."""
    result = {}
    for season in seasons:
        record = json.loads(
            (root / f"data/raw/historical_schedules/season={season}.json").read_text("utf-8")
        )
        for race in record["payload"]["MRData"]["RaceTable"]["Races"]:
            location = race["Circuit"]["Location"]
            event_id = f"season={int(race['season'])}/round={int(race['round']):02d}"
            result[event_id] = (float(location["lat"]), float(location["long"]))
    return result


def audit_historical_weather(
    root: Path,
    events: dict[str, tuple[datetime, datetime]],
    *,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Freeze one archived forecast per Gold race; `events` maps cutoff and race start."""
    locations = circuit_locations(root, sorted({int(event[7:11]) for event in events}))
    races = []
    owned = http_client is None
    client = http_client or httpx.Client(timeout=60.0)
    try:
        for event_id in sorted(events):
            cutoff, race_start = events[event_id]
            if not cutoff < race_start:
                raise ValueError(f"{event_id} cutoff is not before the race start")
            latitude, longitude = locations[event_id]
            run = forecast_run(cutoff)
            record: dict[str, Any] = {
                "event_id": event_id,
                "cutoff": cutoff.isoformat(),
                "race_start": race_start.isoformat(),
                "latitude": latitude,
                "longitude": longitude,
                "model": MODEL,
                "run_initialized_at": run.isoformat(),
                "available_at": (run + RELEASE_MARGIN).isoformat(),
            }
            if run < ARCHIVE_START:
                races.append({**record, "status": "before_archive"})
                continue
            response = client.get(ENDPOINT, params=_parameters(latitude, longitude, run))
            document = _retain(root, response)
            values = race_weather(json.loads(response.content), race_start)
            status = "audited" if all(v is not None for v in values.values()) else "incomplete"
            races.append({**record, "status": status, "document": document, "values": values})
    finally:
        if owned:
            client.close()
    audit = {
        "version": AUDIT_VERSION,
        "model": MODEL,
        "release_margin_hours": RELEASE_MARGIN.total_seconds() / 3600,
        "availability_evidence": AVAILABILITY_EVIDENCE,
        "races": races,
    }
    content = _json(audit)
    digest = hashlib.sha256(content).hexdigest()
    path = root / _AUDITS / f"weather-forecast-audit-{digest}.json"
    _immutable(path, content)
    counts: dict[str, int] = {}
    for race in races:
        counts[race["status"]] = counts.get(race["status"], 0) + 1
    return {"path": path.relative_to(root).as_posix(), "races": len(races), "status": counts}


def load_weather_forecasts(
    root: Path, cutoffs: dict[str, datetime]
) -> tuple[dict[str, dict[str, Any]], str] | None:
    """Verified weather audit for exactly these Gold races, or None.

    `cutoffs` maps each Gold event to its post-qualifying cutoff. The newest
    matching audit is replayed: run choice, release bound and every value must
    agree with the retained response bytes. The race start was bound from the
    audited direct catalog when the audit was frozen.
    """
    matches = []
    for path in (root / _AUDITS).glob("weather-forecast-audit-*.json"):
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("version") == AUDIT_VERSION and {
            record["event_id"] for record in audit["races"]
        } == set(cutoffs):
            matches.append((path.stat().st_mtime, path.name, path, audit))
    if not matches:
        return None
    _, _, path, audit = max(matches)
    digest = file_sha256(path)
    if path.name != f"weather-forecast-audit-{digest}.json":
        raise ValueError("weather audit name does not match its content hash")
    records = {}
    for record in audit["races"]:
        cutoff = cutoffs[record["event_id"]]
        race_start = datetime.fromisoformat(record["race_start"])
        if record["cutoff"] != cutoff.isoformat() or not cutoff < race_start:
            raise ValueError("weather audit targets another cutoff")
        run = forecast_run(cutoff)
        available = run + RELEASE_MARGIN
        if record["run_initialized_at"] != run.isoformat() or available > cutoff:
            raise ValueError("weather audit run choice differs from the release rule")
        if (record["status"] == "before_archive") != (run < ARCHIVE_START):
            raise ValueError("weather audit archive status differs from the run time")
        item = {**record, "available_at": available}
        if "document" in record:
            document = _safe_file(root, record["document"]["path"], record["document"]["sha256"])
            if race_weather(json.loads(document.read_bytes()), race_start) != record["values"]:
                raise ValueError("weather audit values differ from the retained response")
        records[record["event_id"]] = item
    return records, digest
