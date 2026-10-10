"""Live sprint captures: sprint qualifying (OpenF1) and the sprint result (Jolpica).

Jolpica has no sprint qualifying endpoint, so the sprint grid comes from OpenF1's
session result, mapped to canonical drivers through the session's own driver list
and the Jolpica season codes. When Jolpica is still empty ``FALLBACK_DELAY`` after
the sprint, the OpenF1 sprint result is frozen instead, until Jolpica supersedes it.
Each capture freezes fresh payloads, their hashes and the observation clock in an
immutable bundle; the observation clock is the availability time. A schedule time
never proves a session finished: a capture needs a nonempty, complete response
observed after the scheduled start.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.normalization.jolpica import normalize_sprint
from f1_ml_predictor.prediction.schedules import Weekend
from f1_ml_predictor.prediction.sprint import (
    OPENF1,
    OPENF1_TEAMS,
    SPRINT_SESSION_NAMES,
    SprintGrid,
    crosswalk,
    openf1_grid,
    openf1_numbers,
)
from f1_ml_predictor.sources.http import JsonSourceClient
from f1_ml_predictor.sources.jolpica import BASE_URL
from f1_ml_predictor.time import require_utc
from f1_ml_predictor.trust.collected_outcomes import (
    _ROOT_URL,
    _exact_option,
    _same_event_url,
    _selected_registry,
)
from f1_ml_predictor.trust.fia_tables import parse_final_text
from f1_ml_predictor.trust.historical import _official, _retain_response, inspect_pdf
from f1_ml_predictor.trust.sprint_gold import _FINAL, REVIEWED_LATER_RULINGS, _later_rulings
from f1_ml_predictor.trust.winter import (
    DRIVER_ALIASES,
    _publication,
    _version_key,
    constructor_aliases_for_season,
    registry_rows,
)

_DIRECTORY = Path("data/raw/prospective_sprint")
_LIMITS = ((3, 1.0), (30, 60.0))
SPRINT_QUALIFYING = "sprint_qualifying"
SPRINT_RESULT = "sprint_result"
SPRINT_FIA = "sprint_fia_classification"
# Jolpica publishes classifications some time after the chequered flag.
RESULT_POLL_DELAY = timedelta(minutes=45)
# Past this delay after a session's scheduled start with Jolpica still empty, the
# OpenF1 session result is frozen instead (labelled with its provider).
FALLBACK_DELAY = timedelta(hours=2)


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


def capture_teams(record: dict[str, Any]) -> dict[str, str]:
    """Canonical constructor per driver from a capture's own OpenF1 driver list.

    The session team is the seat raced that weekend, so it covers a mid-season
    change the latest audited roster cannot know. Unmapped team names are left out.
    """
    sources = record["sources"]
    codes = crosswalk(sources["drivers_season"]["payload"]).codes
    teams: dict[str, str] = {}
    for driver in sources["session_drivers"]["payload"]:
        team = OPENF1_TEAMS.get(str(driver.get("team_name") or ""))
        code = driver.get("name_acronym")
        if team is not None and code in codes:
            teams[codes[code]] = team
    return teams


def sprint_classification(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Classified position (None if unclassified) and points per canonical driver.

    A Jolpica capture is classified by a numeric position text. An OpenF1 capture is
    classified by a numeric position: its ``dnf`` flag also marks a driver who retired
    late but is still classified, so it never decides classification.
    """
    sources = record["sources"]
    identities = crosswalk(sources["drivers_season"]["payload"])
    if record.get("provider") == "openf1":
        by_number = openf1_numbers(sources["session_drivers"]["payload"], identities.codes)
        rows: dict[str, dict[str, Any]] = {}
        for row in sources["session_result"]["payload"]:
            number = int(row["driver_number"])
            if number not in by_number:
                raise ValueError(f"OpenF1 sprint car {number} is not in the driver list")
            position = row.get("position")
            rows[by_number[number]] = {
                "position": position if isinstance(position, int) else None,
                "points": float(row.get("points") or 0.0),
            }
        return rows
    event = EventId(record["event"]["season"], record["event"]["round"])
    payload = sources["sprint"]["payload"]
    return {
        identities.driver(row["driver_id"]): {
            "position": row["position"] if str(row["position_text"] or "").isdecimal() else None,
            "points": float(row["points"] or 0.0),
        }
        for row in normalize_sprint(payload["MRData"]["RaceTable"]["Races"], event).to_pylist()
    }


def capture_points(record: dict[str, Any]) -> dict[str, float]:
    """Sprint points per canonical driver from a verified sprint result capture."""
    return {driver: row["points"] for driver, row in sprint_classification(record).items()}


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
        captured = datetime.fromisoformat(result_record["captured_at"])
        for driver, row in sprint_classification(result_record).items():
            classified = row["position"] is not None
            values.setdefault(driver, {}).update(
                {
                    "sprint_position": row["position"],
                    "sprint_position_available_at": captured,
                    "sprint_classified": float(classified),
                    "sprint_classified_available_at": captured,
                }
            )
    return values


def capture_fia_positions(root: Path, record: dict[str, Any]) -> dict[str, int | None]:
    """FIA sprint positions from a verified capture, after rechecking its retained PDF."""
    document = record["document"]
    if file_sha256(root / document["path"]) != document["sha256"]:
        raise ValueError("retained FIA sprint classification PDF changed")
    return {row["driver_id"]: row["position"] for row in record["classification"]}


def _capture_fia_sprint(
    root: Path, client: httpx.Client, event: EventId, race_name: str, clock: datetime
) -> dict[str, Any] | None:
    """Freeze the latest FIA Final Sprint Classification once no later ruling is pending.

    The registry is reached the same way as the race outcome collector (root, exact
    season, exact event) and rechecked after the PDF is read. Returns None while the
    document is unpublished or a later sprint ruling could still amend it.
    """

    def fetch(url: str) -> tuple[dict[str, Any], httpx.Response]:
        _official(url)
        response = client.get(
            url, timeout=30, follow_redirects=False, headers={"User-Agent": "f1-ml-predictor/0.1.0"}
        )
        if str(response.url) != url:
            raise ValueError("FIA sprint capture cannot follow an unverified redirect")
        return _retain_response(root, response), response

    _, root_response = fetch(_ROOT_URL)
    season_url = _exact_option(root_response.text, f"SEASON {event.season}")
    _, season_response = fetch(season_url)
    registry_url = _exact_option(season_response.text, race_name)
    if not registry_url.startswith(season_url + "/event/"):
        raise ValueError("discovered FIA registry URL does not match the exact event")

    def select(text: str) -> dict[str, Any] | None:
        # Unlike the race collector, unnumbered rows (late "DOC n - ..." uploads such
        # as revised standings) are allowed; numbered documents must stay unique.
        _selected_registry(text, event.season, race_name)
        rows = registry_rows(text)
        numbered = [row["document_id"] for row in rows if row["document_id"] is not None]
        if len(set(numbered)) != len(numbered):
            raise ValueError("FIA registry has duplicate document numbers")
        live = [row for row in rows if row.get("url") and not row["recalled"]]
        finals = sorted((row for row in live if _FINAL.match(row["title"])), key=_version_key)
        if not finals:
            return None
        final = finals[-1]
        if final["document_id"] is None:
            raise ValueError("FIA final sprint classification has no document number")
        if not _same_event_url(final["url"], event.season, race_name):
            raise ValueError("FIA final sprint classification belongs to another event")
        if any(row["url"] not in REVIEWED_LATER_RULINGS for row in _later_rulings(live, final)):
            return None
        return final

    registry_artifact, registry_response = fetch(registry_url)
    final = select(registry_response.text)
    if final is None:
        return None
    pdf_artifact, pdf_response = fetch(final["url"])
    if not pdf_response.content.startswith(b"%PDF"):
        raise ValueError("FIA final sprint classification is not a PDF")
    pdf = inspect_pdf(root / pdf_artifact["path"])
    if final["document_id"] is not None and pdf["document_id"] not in {None, final["document_id"]}:
        raise ValueError("FIA sprint PDF document number contradicts the registry")
    parsed = parse_final_text(
        pdf["text"], event, DRIVER_ALIASES, constructor_aliases_for_season(event.season)
    ).to_pylist()
    _, recheck_response = fetch(registry_url)
    if select(recheck_response.text) != final:
        raise ValueError("FIA sprint registry changed while the classification was read")
    published = _publication(final) + timedelta(minutes=1)
    if published > clock:
        raise ValueError("FIA sprint classification publication is after the capture clock")
    record = {
        "version": 1,
        "kind": SPRINT_FIA,
        "event": {"season": event.season, "round": event.round},
        "captured_at": clock.isoformat(),
        "evidence_class": "captured_live",
        "decision_basis": "latest_nonrecalled_fia_final_sprint_classification_without_later_ruling",
        "document": {
            "title": final["title"],
            "document_id": final["document_id"],
            "url": final["url"],
            "publication_cet": final["publication_cet"],
            "path": pdf_artifact["path"],
            "sha256": pdf_artifact["sha256"],
        },
        "registry": {"path": registry_artifact["path"], "sha256": registry_artifact["sha256"]},
        "classification": [
            {
                "driver_id": row["driver_id"],
                "constructor_id": row["constructor_id"],
                "position": row["position"],
                "classified": row["classified"],
            }
            for row in parsed
        ],
        "sources": {},
    }
    return _freeze(root, event, SPRINT_FIA, record)


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
        if result is None and clock < weekend.sprint + RESULT_POLL_DELAY:
            status["status"] = "sprint_qualifying_captured"
            return status
        # Jolpica stays the preferred source: an OpenF1 fallback is superseded by a
        # later Jolpica capture, which becomes the latest one.
        if result is None or result.get("provider") == "openf1":
            captured = _capture_sprint_result(root, client, event, clock)
            if captured is None and result is None and clock >= weekend.sprint + FALLBACK_DELAY:
                captured = capture_openf1_session(
                    root,
                    client,
                    event,
                    clock,
                    kind=SPRINT_RESULT,
                    names=("Sprint",),
                    window=(
                        weekend.sprint - timedelta(hours=2),
                        weekend.sprint + timedelta(hours=12),
                    ),
                    decision_basis="openf1_sprint_result_after_jolpica_delay",
                )
            if captured is None and result is None:
                status["status"] = "waiting_for_sprint_results"
                return status
            if captured is not None:
                status["sprint_result_capture"] = captured
                result = latest_capture(root, event, SPRINT_RESULT, clock)
    assert result is not None
    status["sprint_result_provider"] = result.get("provider", "jolpica")
    status["status"] = "sprint_captured"
    fia = latest_capture(root, event, SPRINT_FIA, clock)
    if fia is not None:
        status["sprint_fia_capture"] = fia["bundle"]
        return status
    # The FIA document confirms the Jolpica points; its absence never blocks a run.
    owned = http_client is None
    fia_client = http_client or httpx.Client(timeout=30, follow_redirects=False)
    try:
        captured = _capture_fia_sprint(root, fia_client, event, weekend.race_name, clock)
    except (ValueError, KeyError, RuntimeError, OSError, httpx.HTTPError) as exc:
        status["fia_status"] = f"error: {exc}"
    else:
        if captured is None:
            status["fia_status"] = "waiting_for_fia_final_sprint_classification"
        else:
            status["sprint_fia_capture"] = captured
    finally:
        if owned:
            fia_client.close()
    return status


def _capture_sprint_qualifying(
    root: Path, client: JsonSourceClient, event: EventId, weekend: Weekend, clock: datetime
) -> dict[str, Any] | None:
    assert weekend.sprint_qualifying is not None and weekend.sprint is not None
    return capture_openf1_session(
        root,
        client,
        event,
        clock,
        kind=SPRINT_QUALIFYING,
        names=SPRINT_SESSION_NAMES,
        window=(weekend.sprint_qualifying - timedelta(hours=2), weekend.sprint),
        decision_basis="fresh_nonempty_openf1_sprint_qualifying_result_observed",
    )


def capture_openf1_session(
    root: Path,
    client: JsonSourceClient,
    event: EventId,
    clock: datetime,
    *,
    kind: str,
    names: tuple[str, ...],
    window: tuple[datetime, datetime],
    decision_basis: str,
) -> dict[str, Any] | None:
    """Freeze one finished OpenF1 session result with its driver list and season codes.

    Returns None until the single matching session has ended and has a result. The
    mapping to canonical drivers is validated before anything is stored.
    """
    sessions = client.get_json(f"{OPENF1}/sessions", {"year": event.season})
    low, high = window
    matches = [
        item
        for item in sessions
        if item.get("session_name") in names
        and low <= datetime.fromisoformat(item["date_start"]) < high
    ]
    if len(matches) != 1:
        raise ValueError(f"OpenF1 {kind} session is missing or ambiguous")
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
        "kind": kind,
        "provider": "openf1",
        "event": {"season": event.season, "round": event.round},
        "captured_at": clock.isoformat(),
        "evidence_class": "captured_live",
        "decision_basis": decision_basis,
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
    return _freeze(root, event, kind, record)


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
