"""Live FIA race starting grid captures for the active weekend.

The collector freezes the latest non-recalled FIA race grid (provisional, final or
revised), its PDF and the registry page. Availability is the registry publication
minute plus one minute, the bound the historical provisional-grid audit uses. The
registry is read again after the PDF and must not have changed. The grid is parsed
when a prediction uses it, against that prediction's roster, as the audit parses a
Gold race against its roster.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.prediction.schedules import Weekend
from f1_ml_predictor.time import require_utc
from f1_ml_predictor.trust.candidates import fia_event_name
from f1_ml_predictor.trust.collected_outcomes import (
    _ROOT_URL,
    _exact_option,
    _retained_context,
    _same_event_url,
    _selected_registry,
    capture_driver_aliases,
)
from f1_ml_predictor.trust.grid_history import (
    _plain,
    grid_drivers,
    parse_grid_pdf,
    race_grid_records,
)
from f1_ml_predictor.trust.historical import _official, _retain_response, inspect_pdf
from f1_ml_predictor.trust.sprint_capture import _DIRECTORY, _freeze, _verified, latest_capture
from f1_ml_predictor.trust.winter import _publication, registry_rows

RACE_GRID = "race_grid"


def _select(text: str, event: EventId, race_name: str) -> dict[str, Any] | None:
    _selected_registry(text, event.season, race_name)
    rows = [row for row in race_grid_records(registry_rows(text)) if row.get("url")]
    if not rows:
        return None
    row = max(rows, key=lambda item: (_publication(item), int(item["document_id"] or 0)))
    if row["recalled"]:
        return None
    if not _same_event_url(row["url"], event.season, race_name):
        raise ValueError("FIA starting grid belongs to another event")
    return row


def _capture(
    root: Path, client: httpx.Client, event: EventId, race_name: str, clock: datetime
) -> dict[str, Any] | None:
    def fetch(url: str) -> tuple[dict[str, Any], httpx.Response]:
        _official(url)
        response = client.get(
            url, timeout=30, follow_redirects=False, headers={"User-Agent": "f1-ml-predictor/0.1.0"}
        )
        if str(response.url) != url:
            raise ValueError("FIA grid capture cannot follow an unverified redirect")
        return _retain_response(root, response), response

    race_name = fia_event_name(event.season, event.round, race_name)
    _, root_response = fetch(_ROOT_URL)
    season_url = _exact_option(root_response.text, f"SEASON {event.season}")
    _, season_response = fetch(season_url)
    registry_url = _exact_option(season_response.text, race_name)
    if not registry_url.startswith(season_url + "/event/"):
        raise ValueError("discovered FIA registry URL does not match the exact event")
    registry_artifact, registry_response = fetch(registry_url)
    row = _select(registry_response.text, event, race_name)
    if row is None:
        return None
    previous = latest_capture(root, event, RACE_GRID, clock)
    if (
        previous is not None
        and previous["document"]["url"] == row["url"]
        and previous["document"]["document_id"] == row["document_id"]
        and previous["document"]["publication_cet"] == row["publication_cet"]
    ):
        return None
    pdf_artifact, pdf_response = fetch(row["url"])
    if not pdf_response.content.startswith(b"%PDF"):
        raise ValueError("FIA starting grid is not a PDF")
    pdf_path = root / pdf_artifact["path"]
    inspected = inspect_pdf(pdf_path)
    cover = _plain(inspected["cover_text"])
    if (
        inspected["document_id"] is None
        or row["document_id"] not in {None, inspected["document_id"]}
        or str(event.season) not in cover
        or "starting grid" not in cover
    ):
        raise ValueError("grid PDF cover contradicts the registry version")
    if len(grid_drivers(pdf_path)) < 15:
        raise ValueError("FIA starting grid does not yield a readable starting order")
    _, recheck = fetch(registry_url)
    if _select(recheck.text, event, race_name) != row:
        raise ValueError("FIA grid registry changed while the starting grid was read")
    available_at = _publication(row) + timedelta(minutes=1)
    if available_at > clock:
        raise ValueError("FIA grid publication is after the capture clock")
    record = {
        "version": 1,
        "kind": RACE_GRID,
        "event": {"season": event.season, "round": event.round},
        "captured_at": clock.isoformat(),
        "evidence_class": "captured_live",
        "decision_basis": "latest_nonrecalled_fia_race_starting_grid",
        "available_at": available_at.isoformat(),
        "grid_status": "final" if "final" in _plain(row["title"]).split() else "provisional",
        "document": {
            "title": row["title"],
            "document_id": inspected["document_id"],
            "url": row["url"],
            "publication_cet": row["publication_cet"],
            "path": pdf_artifact["path"],
            "sha256": pdf_artifact["sha256"],
        },
        "registry": {"path": registry_artifact["path"], "sha256": registry_artifact["sha256"]},
        "sources": {},
    }
    return _freeze(root, event, RACE_GRID, record)


def grid_tick(
    root: Path,
    event: EventId,
    weekend: Weekend,
    *,
    now: Callable[[], datetime] | None = None,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Freeze the latest FIA race starting grid once it is newer than the last."""
    clock = (now or (lambda: datetime.now(UTC)))()
    require_utc(clock, "grid capture clock")
    if weekend.qualifying is None or clock < weekend.qualifying:
        return {"status": "waiting_for_qualifying"}
    owned = http_client is None
    client = http_client or httpx.Client(timeout=30, follow_redirects=False)
    try:
        captured = _capture(root, client, event, weekend.race_name, clock)
    finally:
        if owned:
            client.close()
    latest = latest_capture(root, event, RACE_GRID, clock)
    status: dict[str, Any] = {
        "status": "grid_captured" if latest is not None else "waiting_for_fia_grid"
    }
    if latest is not None:
        status["grid_capture"] = latest["bundle"]
        status["document_id"] = latest["document"]["document_id"]
    if captured is not None:
        status["new_capture"] = True
    return status


def grid_by(root: Path, event: EventId, clock: datetime, cutoff: datetime) -> dict[str, Any] | None:
    """The latest grid captured by the clock whose publication preceded the cutoff.

    A capture taken after the cutoff still counts when its publication bound is
    earlier; a revision published after the cutoff never hides the earlier grid.
    """
    directory = root / _DIRECTORY / event.partition()
    records = [
        _verified(root, path)
        for path in sorted(directory.glob(f"{RACE_GRID}-*.json"))
        if path.is_file()
    ]
    known = [
        item
        for item in records
        if datetime.fromisoformat(item["captured_at"]) <= clock
        and datetime.fromisoformat(item["available_at"]) <= cutoff
    ]
    return (
        max(known, key=lambda item: (item["available_at"], item["captured_at"])) if known else None
    )


def captured_grid(
    root: Path, record: dict[str, Any] | None, qualifying_bundle: Path, cutoff: datetime
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]] | None:
    """The grid parsed for the qualifying capture's roster, if public by the cutoff.

    FIA names map to the capture's own Jolpica IDs, so a driver Gold has not seen
    yet, or knows under another ID, still gets a grid slot.
    """
    if record is None:
        return None
    available_at = datetime.fromisoformat(record["available_at"])
    if available_at > cutoff:
        return None
    document = record["document"]
    path = root / document["path"]
    if file_sha256(path) != document["sha256"]:
        raise ValueError("retained FIA starting grid PDF changed")
    _, _, entries, constructors = _retained_context(qualifying_bundle)
    aliases = capture_driver_aliases(entries, set(constructors))
    grid = parse_grid_pdf(path, set(constructors), aliases=aliases)
    provenance = {
        "bundle": record["bundle"],
        "document_id": document["document_id"],
        "title": document["title"],
        "url": document["url"],
        "grid_status": record["grid_status"],
        "available_at": available_at.isoformat(),
    }
    return grid, provenance
