"""Retained season schedules that define historical weekend cutoff times.

Schedules are present-day observations of published session times. They only
place a historical cutoff (for example first practice); no schedule value is a
predictive feature, so retrieval time does not need to precede the cutoff.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.sources.http import JsonSourceClient
from f1_ml_predictor.sources.jolpica import BASE_URL
from f1_ml_predictor.time import require_utc

_DIRECTORY = Path("data/raw/historical_schedules")
_LIMITS = ((4, 1.0), (500, 3600.0))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class Weekend:
    event: EventId
    race_name: str
    circuit_id: str
    circuit_name: str
    first_practice: datetime | None
    qualifying: datetime | None
    sprint: datetime | None
    race: datetime | None
    sprint_qualifying: datetime | None = None


def _start(session: Any) -> datetime | None:
    if not isinstance(session, dict) or not session.get("date") or not session.get("time"):
        return None
    value = datetime.fromisoformat(f"{session['date']}T{session['time']}")
    require_utc(value, "session start")
    return value


def weekends(payload: dict[str, Any]) -> dict[EventId, Weekend]:
    """Parse a Jolpica season schedule; the circuit comes only from ``circuitId``."""
    table = payload["MRData"]["RaceTable"]
    result = {}
    for race in table["Races"]:
        event = EventId(int(race["season"]), int(race["round"]))
        result[event] = Weekend(
            event,
            race["raceName"],
            race["Circuit"]["circuitId"],
            race["Circuit"].get("circuitName", race["Circuit"]["circuitId"]),
            _start(race.get("FirstPractice")),
            _start(race.get("Qualifying")),
            _start(race.get("Sprint")),
            _start({"date": race.get("date"), "time": race.get("time")}),
            # 2023 called the sprint grid session the Sprint Shootout.
            _start(race.get("SprintQualifying") or race.get("SprintShootout")),
        )
    return result


def retained_schedules(
    root: Path, seasons: list[int], *, http_client: httpx.Client | None = None
) -> tuple[dict[EventId, Weekend], dict[str, dict[str, str]]]:
    """Fetch each season once, then verify the retained bytes on every later use."""
    directory = root / _DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    result: dict[EventId, Weekend] = {}
    sources: dict[str, dict[str, str]] = {}
    missing = [season for season in seasons if not (directory / f"season={season}.json").exists()]
    if missing:
        with JsonSourceClient(_LIMITS, http_client) as client:
            for season in missing:
                url = f"{BASE_URL}/{season}/"
                payload = client.get_json(url, {"limit": 100, "offset": 0})
                record = {
                    "version": 1,
                    "url": url,
                    "params": {"limit": 100, "offset": 0},
                    "retrieved_at_utc": datetime.now(UTC).isoformat(),
                    "classification": "current_state_only",
                    "use": "historical cutoff placement only",
                    "payload_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
                    "payload": payload,
                }
                with (directory / f"season={season}.json").open("xb") as handle:
                    handle.write(_canonical(record))
    for season in seasons:
        path = directory / f"season={season}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if hashlib.sha256(_canonical(record["payload"])).hexdigest() != record["payload_sha256"]:
            raise ValueError(f"retained {season} schedule payload hash mismatch")
        result.update(weekends(record["payload"]))
        sources[str(season)] = {
            "path": path.relative_to(root).as_posix(),
            "payload_sha256": record["payload_sha256"],
            "retrieved_at_utc": record["retrieved_at_utc"],
        }
    return result, sources
