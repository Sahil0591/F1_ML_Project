"""Live sprint captures: sprint qualifying (OpenF1) and the sprint result (Jolpica).

Jolpica has no sprint qualifying endpoint, so the sprint grid comes from OpenF1's
session result, mapped to canonical drivers through the session's own driver list
and the Jolpica season codes. Each capture freezes fresh payloads, their hashes and
the observation clock in an immutable bundle; the observation clock is the
availability time. A schedule time never proves a session finished: a capture needs
a nonempty, complete response observed after the scheduled start.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.normalization.jolpica import normalize_sprint
from f1_ml_predictor.prediction.schedules import Weekend
from f1_ml_predictor.prediction.sprint import (
    OPENF1,
    SPRINT_SESSION_NAMES,
    SprintGrid,
    crosswalk,
    openf1_grid,
)
from f1_ml_predictor.sources.http import JsonSourceClient
from f1_ml_predictor.sources.jolpica import BASE_URL
from f1_ml_predictor.time import require_utc

_DIRECTORY = Path("data/raw/prospective_sprint")
_LIMITS = ((3, 1.0), (30, 60.0))
SPRINT_QUALIFYING = "sprint_qualifying"
SPRINT_RESULT = "sprint_result"
# Jolpica publishes classifications some time after the chequered flag.
RESULT_POLL_DELAY = timedelta(minutes=45)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _source(client: JsonSourceClient, url: str, params: dict[str, Any]) -> dict[str, Any]:
    payload = client.get_json(url, params)
    return {
        "url": url,
        "params": params,
        "payload_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "payload": payload,
    }


def _freeze(root: Path, event: EventId, kind: str, record: dict[str, Any]) -> dict[str, Any]:
    data = _canonical(record)
    digest = hashlib.sha256(data).hexdigest()
    path = root / _DIRECTORY / event.partition() / f"{kind}-{digest}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
    return {
        "kind": kind,
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "captured_at": record["captured_at"],
    }


def _verified(root: Path, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    digest = path.stem.split("-", 1)[1] if "-" in path.stem else ""
    record: dict[str, Any] = json.loads(data)
    if hashlib.sha256(_canonical(record)).hexdigest() != digest:
        raise ValueError(f"sprint capture {path.name} changed since it was frozen")
    for source in record["sources"].values():
        if hashlib.sha256(_canonical(source["payload"])).hexdigest() != source["payload_sha256"]:
            raise ValueError(f"sprint capture {path.name} payload hash mismatch")
    record["bundle"] = {"path": path.relative_to(root).as_posix(), "sha256": digest}
    return record


def latest_capture(root: Path, event: EventId, kind: str, clock: datetime) -> dict[str, Any] | None:
    """The latest verified capture of one kind observed by the clock, if any."""
    directory = root / _DIRECTORY / event.partition()
    records = [
        _verified(root, path) for path in sorted(directory.glob(f"{kind}-*.json")) if path.is_file()
    ]
    known = [item for item in records if datetime.fromisoformat(item["captured_at"]) <= clock]
    return max(known, key=lambda item: item["captured_at"]) if known else None


def capture_grid(record: dict[str, Any]) -> SprintGrid:
    """Rebuild the sprint grid from a verified sprint qualifying capture."""
    sources = record["sources"]
    identities = crosswalk(sources["drivers_season"]["payload"])
    return openf1_grid(
        sources["session_result"]["payload"],
        sources["session_drivers"]["payload"],
        identities.codes,
        datetime.fromisoformat(record["captured_at"]),
        f"captured_live:{record['bundle']['sha256']}",
    )


def capture_points(record: dict[str, Any]) -> dict[str, float]:
    """Sprint points per canonical driver from a verified sprint result capture."""
    event = EventId(record["event"]["season"], record["event"]["round"])
    payload = record["sources"]["sprint"]["payload"]
    identities = crosswalk(record["sources"]["drivers_season"]["payload"])
    table = normalize_sprint(payload["MRData"]["RaceTable"]["Races"], event)
    return {
        identities.driver(row["driver_id"]): float(row["points"] or 0.0)
        for row in table.to_pylist()
    }


def captured_sprint_values(
    grid_record: dict[str, Any] | None, result_record: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    """Live race-contract sprint values from verified captures, at their capture clocks.

    Mirrors the Gold values: the sprint qualifying position, and from the sprint
    classification the classified position and whether the driver was classified.
    """
    values: dict[str, dict[str, Any]] = {}
    if grid_record is not None:
        grid = capture_grid(grid_record)
        for driver, position in grid.positions.items():
            values.setdefault(driver, {}).update(
                {
                    "sprint_qualifying_position": position,
                    "sprint_qualifying_position_available_at": grid.available_at,
                }
            )
    if result_record is not None:
        event = EventId(result_record["event"]["season"], result_record["event"]["round"])
        identities = crosswalk(result_record["sources"]["drivers_season"]["payload"])
        captured = datetime.fromisoformat(result_record["captured_at"])
        payload = result_record["sources"]["sprint"]["payload"]
        for row in normalize_sprint(payload["MRData"]["RaceTable"]["Races"], event).to_pylist():
            classified = str(row["position_text"] or "").isdecimal()
            values.setdefault(identities.driver(row["driver_id"]), {}).update(
                {
                    "sprint_position": row["position"] if classified else None,
                    "sprint_position_available_at": captured,
                    "sprint_classified": float(classified),
                    "sprint_classified_available_at": captured,
                }
            )
    return values


def sprint_tick(
    root: Path,
    event: EventId,
    weekend: Weekend,
    *,
    now: Callable[[], datetime] | None = None,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """One bounded sprint step for the active event; returns its status record."""
    clock = (now or (lambda: datetime.now(UTC)))()
    require_utc(clock, "sprint capture clock")
    if weekend.sprint is None or weekend.sprint_qualifying is None:
        return {"status": "not_a_sprint_weekend"}
    status: dict[str, Any] = {
        "sprint_qualifying_scheduled_start": weekend.sprint_qualifying.isoformat(),
        "sprint_scheduled_start": weekend.sprint.isoformat(),
    }
    grid = latest_capture(root, event, SPRINT_QUALIFYING, clock)
    result = latest_capture(root, event, SPRINT_RESULT, clock)
    if grid is not None:
        status["sprint_qualifying_capture"] = grid["bundle"]
    if result is not None:
        status["sprint_result_capture"] = result["bundle"]
    if clock < weekend.sprint_qualifying:
        status["status"] = "waiting_for_sprint_qualifying"
        return status
    with JsonSourceClient(_LIMITS, http_client) as client:
        if grid is None:
            if clock >= weekend.sprint:
                status["status"] = "sprint_qualifying_missed"
                return status
            captured = _capture_sprint_qualifying(root, client, event, weekend, clock)
            if captured is None:
                status["status"] = "waiting_for_sprint_qualifying_results"
                return status
            status["sprint_qualifying_capture"] = captured
        if result is None:
            if clock < weekend.sprint + RESULT_POLL_DELAY:
                status["status"] = "sprint_qualifying_captured"
                return status
            captured = _capture_sprint_result(root, client, event, clock)
            if captured is None:
                status["status"] = "waiting_for_sprint_results"
                return status
            status["sprint_result_capture"] = captured
    status["status"] = "sprint_captured"
    return status


def _capture_sprint_qualifying(
    root: Path, client: JsonSourceClient, event: EventId, weekend: Weekend, clock: datetime
) -> dict[str, Any] | None:
    assert weekend.sprint_qualifying is not None and weekend.sprint is not None
    sessions = client.get_json(f"{OPENF1}/sessions", {"year": event.season})
    low = weekend.sprint_qualifying - timedelta(hours=2)
    matches = [
        item
        for item in sessions
        if item.get("session_name") in SPRINT_SESSION_NAMES
        and low <= datetime.fromisoformat(item["date_start"]) < weekend.sprint
    ]
    if len(matches) != 1:
        raise ValueError("OpenF1 sprint qualifying session is missing or ambiguous")
    session = matches[0]
    if datetime.fromisoformat(session["date_end"]) > clock:
        return None
    key = int(session["session_key"])
    result = _source(client, f"{OPENF1}/session_result", {"session_key": key})
    if not result["payload"]:
        return None
    drivers = _source(client, f"{OPENF1}/drivers", {"session_key": key})
    season = _source(client, f"{BASE_URL}/{event.season}/drivers/", {"limit": 100, "offset": 0})
    record = {
        "version": 1,
        "kind": SPRINT_QUALIFYING,
        "event": {"season": event.season, "round": event.round},
        "captured_at": clock.isoformat(),
        "evidence_class": "captured_live",
        "decision_basis": "fresh_nonempty_openf1_sprint_qualifying_result_observed",
        "session": {
            "session_key": key,
            "session_name": session["session_name"],
            "date_start": session["date_start"],
            "date_end": session["date_end"],
        },
        "sources": {
            "sessions": {
                "url": f"{OPENF1}/sessions",
                "params": {"year": event.season},
                "payload_sha256": hashlib.sha256(_canonical(sessions)).hexdigest(),
                "payload": sessions,
            },
            "session_result": result,
            "session_drivers": drivers,
            "drivers_season": season,
        },
    }
    # Validate the mapping before freezing, so an unusable capture is never stored.
    openf1_grid(
        result["payload"],
        drivers["payload"],
        crosswalk(season["payload"]).codes,
        clock,
        "validation",
    )
    return _freeze(root, event, SPRINT_QUALIFYING, record)


def _capture_sprint_result(
    root: Path, client: JsonSourceClient, event: EventId, clock: datetime
) -> dict[str, Any] | None:
    sprint = _source(
        client, f"{BASE_URL}/{event.season}/{event.round}/sprint/", {"limit": 100, "offset": 0}
    )
    races = sprint["payload"]["MRData"]["RaceTable"]["Races"]
    table = normalize_sprint(races, event)
    total = int(sprint["payload"]["MRData"]["total"])
    if not table.num_rows:
        return None
    if total != table.num_rows:
        raise ValueError("Jolpica sprint response is incomplete")
    season = _source(client, f"{BASE_URL}/{event.season}/drivers/", {"limit": 100, "offset": 0})
    record = {
        "version": 1,
        "kind": SPRINT_RESULT,
        "event": {"season": event.season, "round": event.round},
        "captured_at": clock.isoformat(),
        "evidence_class": "captured_live",
        "decision_basis": "fresh_nonempty_jolpica_sprint_classification_observed",
        "sources": {"sprint": sprint, "drivers_season": season},
    }
    return _freeze(root, event, SPRINT_RESULT, record)
