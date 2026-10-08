"""Offline checks for the FIA sprint audit rules, Gold sprint loading and v3 features."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.prediction.contracts import (
    SPRINT_NUMERIC,
    numeric_features,
    sprint_weekend_values,
)
from f1_ml_predictor.prediction.sprint import MASKED
from f1_ml_predictor.trust.sprint_gold import (
    REVIEWED_LATER_RULINGS,
    SCHEMA,
    grid_kind,
    load_gold_sprints,
    select_documents,
)


def _row(number: int, title: str, cet: str, *, recalled: bool = False) -> dict:
    return {
        "document_id": str(number),
        "title": title,
        "publication_cet": cet,
        "url": f"https://www.fia.com/doc{number}.pdf",
        "recalled": recalled,
    }


def _registry(*extra: dict) -> list[dict]:
    return [
        _row(10, "Provisional Sprint Qualifying Classification", "21.08.26 17:40"),
        _row(11, "Final Sprint Qualifying Classification", "21.08.26 19:56"),
        _row(12, "Provisional Sprint Starting Grid", "21.08.26 20:01"),
        _row(20, "Provisional Sprint Classification", "22.08.26 12:53"),
        _row(21, "Final Sprint Classification", "22.08.26 14:55"),
        *extra,
    ]


def test_grid_document_follows_the_season_format() -> None:
    assert (grid_kind(2022), grid_kind(2023), grid_kind(2026)) == (
        "qualifying",
        "sprint_shootout",
        "sprint_qualifying",
    )


def test_the_first_grid_document_and_latest_final_classification_are_selected() -> None:
    grid, starting, final = select_documents(_registry(), 2026)
    assert grid["document_id"] == "10"
    assert starting["document_id"] == "12"
    assert final["document_id"] == "21"


def test_a_recalled_grid_document_is_never_used() -> None:
    rows = _registry()
    rows[0]["recalled"] = True
    grid, _, _ = select_documents(rows, 2026)
    assert grid["document_id"] == "11"


def test_a_later_sprint_ruling_excludes_the_sprint() -> None:
    ruling = _row(
        30, "Corrected - Sprint Infringement - Car 30 - Causing a Collision", "22.08.26 16:00"
    )
    with pytest.raises(ValueError, match="later_sprint_ruling_requires_review"):
        select_documents(_registry(ruling), 2026)


def test_a_reviewed_later_ruling_does_not_exclude_the_sprint() -> None:
    ruling = _row(
        30, "Corrected - Sprint Infringement - Car 30 - Causing a Collision", "22.08.26 16:00"
    )
    ruling["url"] = next(iter(REVIEWED_LATER_RULINGS))
    _, _, final = select_documents(_registry(ruling), 2026)
    assert final["document_id"] == "21"


def test_late_grid_uploads_and_points_are_not_rulings() -> None:
    late = [
        _row(31, "Final Sprint Qualifying Classification", "30.08.26 13:05"),
        _row(32, "Final Sprint Starting Grid", "30.08.26 14:13"),
        _row(33, "Championship Points after Sprint", "30.08.26 14:13"),
    ]
    _, _, final = select_documents(_registry(*late), 2026)
    assert final["document_id"] == "21"


def test_the_grid_must_be_published_before_the_starting_grid() -> None:
    rows = [row for row in _registry() if "Starting Grid" not in row["title"]]
    rows.append(_row(12, "Provisional Sprint Starting Grid", "21.08.26 17:40"))
    with pytest.raises(ValueError, match="not_published_before_the_starting_grid"):
        select_documents(rows, 2026)
    with pytest.raises(ValueError, match="starting_grid_missing"):
        select_documents([row for row in _registry() if "Starting" not in row["title"]], 2026)


def test_v3_race_contracts_add_sprint_values_after_qualifying_only() -> None:
    for contract in ("post_qualifying", "pre_race"):
        assert set(SPRINT_NUMERIC) <= set(numeric_features(contract))
    for contract in ("pre_weekend", "post_practice"):
        assert not set(SPRINT_NUMERIC) & set(numeric_features(contract))
    # The sprint model never sees values describing the sprint it predicts.
    assert set(SPRINT_NUMERIC) <= set(MASKED)


def test_gold_sprint_values_carry_their_own_clocks() -> None:
    grid_at = datetime(2026, 8, 21, 18, 41, tzinfo=UTC)
    label_at = grid_at + timedelta(hours=20)
    values = sprint_weekend_values(
        {
            "sprint_qualifying_position": 4,
            "grid_available_at": grid_at,
            "label_position": None,
            "label_classified": False,
            "label_available_at": label_at,
        }
    )
    assert values["sprint_qualifying_position_available_at"] == grid_at
    assert values["sprint_position"] is None
    assert values["sprint_classified"] == 0.0
    assert values["sprint_position_available_at"] == label_at
    assert sprint_weekend_values(None) == {}


def test_gold_sprint_loading_verifies_hashes(tmp_path: Path) -> None:
    assert load_gold_sprints(tmp_path) is None
    base = tmp_path / "data/benchmarks/gold_sprint_core_v1"
    moment = datetime(2026, 8, 22, tzinfo=UTC)
    table = pa.Table.from_pylist(
        [
            {
                "event_id": "season=2026/round=12",
                "driver_id": "russell",
                "constructor_id": "mercedes",
                "sprint_qualifying_position": 1,
                "sprint_qualifying_last_seconds": 71.567,
                "grid_available_at": moment,
                "label_position": 1,
                "label_classified": True,
                "label_dnf": False,
                "label_dnf_reason": "three_source_finished_agreement",
                "label_available_at": moment,
            }
        ],
        schema=SCHEMA,
    )
    staging = tmp_path / "sprints.parquet"
    pq.write_table(table, staging)
    data_sha = hashlib.sha256(staging.read_bytes()).hexdigest()
    manifest = json.dumps(
        {"dataset": {"sha256": data_sha, "path": "sprints.parquet"}, "excluded": []},
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    (base / digest).mkdir(parents=True)
    staging.replace(base / digest / "sprints.parquet")
    (base / digest / "manifest.json").write_bytes(manifest)
    (base / "current.json").write_text(json.dumps({"manifest_sha256": digest}), encoding="utf-8")
    rows, _, loaded = load_gold_sprints(tmp_path) or ([], {}, "")
    assert loaded == digest and rows[0]["driver_id"] == "russell"
    pq.write_table(table.slice(0, 0), base / digest / "sprints.parquet")
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        load_gold_sprints(tmp_path)
