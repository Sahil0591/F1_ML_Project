"""Live FIA practice classification captures for the active weekend.

The collector freezes the latest non-recalled FIA practice classification, its PDF,
the registry page and the parsed timing rows. Availability is the registry
publication minute plus one minute, the same bound the historical practice audit
uses, so a capture taken after a later cutoff still proves what was public by then.
The registry is read again after the PDF and must not have changed.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.benchmarks.enrichment import _PRACTICE_FEATURES, practice_features
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.prediction.schedules import Weekend
from f1_ml_predictor.time import require_utc
from f1_ml_predictor.trust.collected_outcomes import (
    _ROOT_URL,
    _exact_option,
    _same_event_url,
    _selected_registry,
)
from f1_ml_predictor.trust.grid_history import _plain
from f1_ml_predictor.trust.historical import _official, _retain_response, inspect_pdf
from f1_ml_predictor.trust.practice_history import _session, parse_practice_pdf
from f1_ml_predictor.trust.sprint_capture import _freeze, latest_capture
from f1_ml_predictor.trust.winter import DRIVER_ALIASES, _publication, registry_rows

PRACTICE = "practice_classification"


def _select(text: str, event: EventId, race_name: str) -> tuple[dict[str, Any], int] | None:
    _selected_registry(text, event.season, race_name)
    rows = [
        (row, session)
        for row in registry_rows(text)
        if row.get("url")
        and not row["recalled"]
        and (session := _session(row["title"])) in {1, 2, 3}
    ]
    if not rows:
        return None
    row, session = max(
        rows,
        key=lambda pair: (pair[1], _publication(pair[0]), int(pair[0]["document_id"] or 0)),
    )
    if not _same_event_url(row["url"], event.season, race_name):
        raise ValueError("FIA practice classification belongs to another event")
    return row, session


def _capture(
    root: Path, client: httpx.Client, event: EventId, race_name: str, clock: datetime
) -> dict[str, Any] | None:
    def fetch(url: str) -> tuple[dict[str, Any], httpx.Response]:
        _official(url)
        response = client.get(
            url, timeout=30, follow_redirects=False, headers={"User-Agent": "f1-ml-predictor/0.1.0"}
        )
        if str(response.url) != url:
            raise ValueError("FIA practice capture cannot follow an unverified redirect")
        return _retain_response(root, response), response

    _, root_response = fetch(_ROOT_URL)
    season_url = _exact_option(root_response.text, f"SEASON {event.season}")
    _, season_response = fetch(season_url)
    registry_url = _exact_option(season_response.text, race_name)
    if not registry_url.startswith(season_url + "/event/"):
        raise ValueError("discovered FIA registry URL does not match the exact event")
    registry_artifact, registry_response = fetch(registry_url)
    selected = _select(registry_response.text, event, race_name)
    if selected is None:
        return None
    row, session = selected
    previous = latest_capture(root, event, PRACTICE, clock)
    if (
        previous is not None
        and previous["document"]["url"] == row["url"]
        and previous["document"]["document_id"] == row["document_id"]
        and previous["document"]["publication_cet"] == row["publication_cet"]
    ):
        return None
    pdf_artifact, pdf_response = fetch(row["url"])
    if not pdf_response.content.startswith(b"%PDF"):
        raise ValueError("FIA practice classification is not a PDF")
    pdf_path = root / pdf_artifact["path"]
    inspected = inspect_pdf(pdf_path)
    cover = _plain(inspected["cover_text"])
    if (
        inspected["document_id"] is None
        or row["document_id"] not in {None, inspected["document_id"]}
        or str(event.season) not in cover
        or "classification" not in cover
    ):
        raise ValueError("practice PDF cover contradicts the registry version")
    parsed, field_size = parse_practice_pdf(pdf_path, set(DRIVER_ALIASES.values()))
    _, recheck = fetch(registry_url)
    if _select(recheck.text, event, race_name) != selected:
        raise ValueError("FIA practice registry changed while the classification was read")
    available_at = _publication(row) + timedelta(minutes=1)
    if available_at > clock:
        raise ValueError("FIA practice publication is after the capture clock")
    record = {
        "version": 1,
        "kind": PRACTICE,
        "event": {"season": event.season, "round": event.round},
        "captured_at": clock.isoformat(),
        "evidence_class": "captured_live",
        "decision_basis": "latest_nonrecalled_fia_practice_classification",
        "available_at": available_at.isoformat(),
        "session": session,
        "field_size": field_size,
        "document": {
            "title": row["title"],
            "document_id": inspected["document_id"],
            "url": row["url"],
            "publication_cet": row["publication_cet"],
            "path": pdf_artifact["path"],
            "sha256": pdf_artifact["sha256"],
        },
        "registry": {"path": registry_artifact["path"], "sha256": registry_artifact["sha256"]},
        "practice": parsed,
        "sources": {},
    }
    return _freeze(root, event, PRACTICE, record)


def practice_tick(
    root: Path,
    event: EventId,
    weekend: Weekend,
    *,
    now: Callable[[], datetime] | None = None,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Freeze the latest FIA practice classification once it is newer than the last."""
    clock = (now or (lambda: datetime.now(UTC)))()
    require_utc(clock, "practice capture clock")
    if weekend.first_practice is None or clock < weekend.first_practice:
        return {"status": "waiting_for_first_practice"}
    owned = http_client is None
    client = http_client or httpx.Client(timeout=30, follow_redirects=False)
    try:
        captured = _capture(root, client, event, weekend.race_name, clock)
    finally:
        if owned:
            client.close()
    latest = latest_capture(root, event, PRACTICE, clock)
    status: dict[str, Any] = {
        "status": "practice_captured" if latest is not None else "waiting_for_fia_practice"
    }
    if latest is not None:
        status["practice_capture"] = latest["bundle"]
        status["session"] = latest["session"]
    if captured is not None:
        status["new_capture"] = True
    return status


def captured_practice_values(
    root: Path, record: dict[str, Any] | None, roster: dict[str, str], cutoff: datetime
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Weekend practice values for build_rows, if the classification was public by cutoff.

    The retained PDF is parsed again for the prediction roster, exactly as the
    historical audit parses it for a Gold race roster.
    """
    if record is None:
        return {}, None
    available_at = datetime.fromisoformat(record["available_at"])
    if available_at > cutoff:
        return {}, None
    document = record["document"]
    path = root / document["path"]
    if file_sha256(path) != document["sha256"]:
        raise ValueError("retained FIA practice classification PDF changed")
    parsed, field_size = parse_practice_pdf(path, set(roster))
    if field_size != record["field_size"]:
        raise ValueError("FIA practice field size differs from its capture")
    values: dict[str, dict[str, Any]] = {}
    for driver, constructor in roster.items():
        teammates = {
            other for other, team in roster.items() if team == constructor and other != driver
        }
        features, _ = practice_features(parsed, field_size, driver, teammates)
        values[driver] = {
            **features,
            **{f"{name}_available_at": available_at for name in _PRACTICE_FEATURES},
        }
    provenance = {
        "bundle": record["bundle"],
        "session": record["session"],
        "document_id": document["document_id"],
        "url": document["url"],
        "available_at": available_at.isoformat(),
        "drivers_parsed": len(parsed),
    }
    return values, provenance
