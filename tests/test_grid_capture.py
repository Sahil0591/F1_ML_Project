import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.prediction.shadow import render_shadows
from f1_ml_predictor.trust import grid_capture
from f1_ml_predictor.trust.grid_capture import RACE_GRID, captured_grid, grid_by
from f1_ml_predictor.trust.sprint_capture import _freeze

EVENT = EventId(2026, 20)
URL = "https://www.fia.com/system/files/decision-document/2026_sao_paulo_grand_prix_-_{}.pdf"
CUTOFF = datetime(2026, 11, 8, 16, 5, tzinfo=UTC)


def _row(document_id: str, published: str, title: str, *, recalled: bool = False) -> dict:
    slug = title.lower().replace(" ", "_")
    return {
        "document_id": document_id,
        "publication_cet": published,
        "title": title,
        "url": URL.format(slug),
        "recalled": recalled,
    }


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    monkeypatch.setattr(grid_capture, "_selected_registry", lambda *args: None)
    monkeypatch.setattr(grid_capture, "registry_rows", lambda text: rows)
    return rows


def test_selects_the_latest_race_grid_and_ignores_other_grids(registry: list) -> None:
    registry += [
        _row("50", "07.11.26 23:56", "Final Qualifying Classification"),
        _row("51", "08.11.26 00:01", "Provisional Starting Grid"),
        _row("40", "07.11.26 17:00", "Sprint Starting Grid"),
        _row("45", "07.11.26 22:00", "Infringement - Starting Grid"),
    ]
    assert grid_capture._select("", EVENT, "São Paulo Grand Prix")["document_id"] == "51"
    registry.append(_row("58", "08.11.26 15:40", "Final Starting Grid"))
    assert grid_capture._select("", EVENT, "São Paulo Grand Prix")["document_id"] == "58"


def test_recalled_latest_grid_or_other_event_is_not_used(registry: list) -> None:
    registry.append(_row("51", "08.11.26 00:01", "Provisional Starting Grid", recalled=True))
    assert grid_capture._select("", EVENT, "São Paulo Grand Prix") is None
    registry[0] = {**registry[0], "recalled": False}
    with pytest.raises(ValueError, match="another event"):
        grid_capture._select("", EVENT, "Mexico City Grand Prix")


def _freeze_grid(root: Path, document_id: str, available_at: str, captured_at: str) -> None:
    record = {
        "version": 1,
        "kind": RACE_GRID,
        "event": {"season": EVENT.season, "round": EVENT.round},
        "captured_at": captured_at,
        "available_at": available_at,
        "grid_status": "provisional",
        "document": {"document_id": document_id},
        "sources": {},
    }
    _freeze(root, EVENT, RACE_GRID, record)


def test_a_revision_published_after_the_cutoff_never_hides_the_earlier_grid(
    tmp_path: Path,
) -> None:
    _freeze_grid(tmp_path, "51", "2026-11-07T23:02:00+00:00", "2026-11-08T15:23:00+00:00")
    _freeze_grid(tmp_path, "58", "2026-11-08T16:30:00+00:00", "2026-11-08T16:31:00+00:00")
    later = datetime(2026, 11, 8, 17, tzinfo=UTC)
    selected = grid_by(tmp_path, EVENT, later, CUTOFF)
    assert selected is not None and selected["document"]["document_id"] == "51"
    assert grid_by(tmp_path, EVENT, later, later)["document"]["document_id"] == "58"
    early = datetime(2026, 11, 8, 15, tzinfo=UTC)
    assert grid_by(tmp_path, EVENT, early, CUTOFF) is None


def _record(tmp_path: Path, available_at: str) -> dict:
    (tmp_path / "grid.pdf").write_bytes(b"%PDF grid")
    return {
        "available_at": available_at,
        "grid_status": "provisional",
        "document": {
            "document_id": "51",
            "title": "Provisional Starting Grid",
            "url": URL.format("provisional_starting_grid"),
            "path": "grid.pdf",
            "sha256": hashlib.sha256(b"%PDF grid").hexdigest(),
        },
        "bundle": {"path": "grid.json", "sha256": "x"},
    }


def test_grid_is_parsed_with_the_capture_roster_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [
        {
            "Driver": {"driverId": "rafael_camara", "givenName": "Rafael", "familyName": "Camara"},
            "Constructor": {"constructorId": "haas"},
        }
    ]
    monkeypatch.setattr(
        grid_capture,
        "_retained_context",
        lambda bundle: ({}, {}, entries, {"rafael_camara": "haas"}),
    )
    seen: dict[str, Any] = {}

    def parse(path: Path, expected: set[str], *, aliases: dict[str, str]) -> dict:
        seen.update(expected=expected, aliases=aliases)
        return {"rafael_camara": {"grid_position": 11, "pit_lane_start": False}}

    monkeypatch.setattr(grid_capture, "parse_grid_pdf", parse)
    result = captured_grid(
        tmp_path, _record(tmp_path, "2026-11-07T23:02:00+00:00"), tmp_path, CUTOFF
    )
    assert result is not None
    grid, source = result
    assert grid["rafael_camara"]["grid_position"] == 11
    assert source["document_id"] == "51"
    assert seen["expected"] == {"rafael_camara"}
    assert seen["aliases"]["Rafael CAMARA"] == "rafael_camara"


def test_grid_after_the_cutoff_or_changed_on_disk_is_not_used(tmp_path: Path) -> None:
    assert captured_grid(tmp_path, None, tmp_path, CUTOFF) is None
    late = _record(tmp_path, "2026-11-08T16:06:00+00:00")
    assert captured_grid(tmp_path, late, tmp_path, CUTOFF) is None
    record = _record(tmp_path, "2026-11-07T23:02:00+00:00")
    (tmp_path / "grid.pdf").write_bytes(b"%PDF changed")
    with pytest.raises(ValueError, match="changed"):
        captured_grid(tmp_path, record, tmp_path, CUTOFF)


def test_shadow_files_frozen_before_the_grid_shadow_still_render(tmp_path: Path) -> None:
    drivers = {"norris": {"winner": 0.4, "podium": 0.8, "expected": 3.0, "dnf": 0.1}}
    payload = {
        "forecast": {"status": "unavailable", "reason": "offline"},
        "variants": {
            "official": {"drivers": drivers},
            "two_stage": {"drivers": drivers},
            "weather": {"status": "skipped", "reason": "no complete forecast"},
            "wet_skill": {"status": "skipped", "reason": "no weather audit"},
            "strategy": {"status": "skipped", "reason": "no audit"},
        },
    }
    (tmp_path / "shadow.json").write_text(json.dumps(payload), encoding="utf-8")
    text = render_shadows(tmp_path, {"path": "shadow.json"})
    assert "provisional_grid" not in text
    payload["variants"]["provisional_grid"] = {"drivers": drivers}
    (tmp_path / "shadow.json").write_text(json.dumps(payload), encoding="utf-8")
    assert "provisional_grid win" in render_shadows(tmp_path, {"path": "shadow.json"})
