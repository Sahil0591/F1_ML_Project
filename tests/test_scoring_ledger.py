"""Synthetic scoring fixtures exercise audit and point-in-time behavior."""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.benchmarks.enrichment import ENRICHMENT_VERSION
from f1_ml_predictor.benchmarks.scoring import (
    _POINT_FEATURES,
    build_gold_scoring,
    verify_scoring_manifest,
)
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.scoring.ledger import _digest, load_scoring_ledger

SOURCE = {"reference": "synthetic:fixture", "sha256": "a" * 64}


def _stamp(day: int) -> str:
    return f"2021-01-{day:02d}T18:00:00+00:00"


def _hashed(data: dict) -> dict:
    data["evidence_hash"] = _digest(data)
    return data


def _rule() -> dict:
    return _hashed(
        {
            "season": 2021,
            "first_round": 1,
            "last_round": 22,
            "race_points": [25, 18, 15],
            "sprint_points": [3, 2, 1],
            "reduced_race_points": [[12.5, 9, 7.5]],
            "fastest_lap_points": 1,
            "fastest_lap_eligible_through": 10,
            "constructor_scoring": "sum_awarded_entries",
            "source_evidence": [SOURCE],
            "revision_status": "audited",
        }
    )


def _entry(
    round_number: int,
    effective: str,
    driver: str | None,
    team: str | None,
    race: float,
    sprint: float = 0,
    bonus: float = 0,
    adjustment: float = 0,
    status: str = "audited",
) -> dict:
    return _hashed(
        {
            "season": 2021,
            "event_id": EventId(2021, round_number).partition(),
            "driver_id": driver,
            "constructor_id": team,
            "race_points": race,
            "sprint_points": sprint,
            "bonus_points": bonus,
            "adjustment_points": adjustment,
            "total_points": race + sprint + bonus + adjustment,
            "effective_at": effective,
            "source_evidence": [SOURCE],
            "revision_status": status,
        }
    )


def _event(
    round_number: int,
    effective: str,
    entries: list[dict],
    status: str = "audited",
    complete: bool = True,
) -> dict:
    return _hashed(
        {
            "season": 2021,
            "event_id": EventId(2021, round_number).partition(),
            "completed_at": _stamp(round_number * 4),
            "effective_at": effective,
            "complete": complete,
            "revision_status": status,
            "expected_driver_ids": ["alpha", "beta"],
            "entries": entries,
        }
    )


def _documents() -> tuple[dict, dict]:
    first = _event(
        1,
        _stamp(5),
        [
            _entry(1, _stamp(5), "alpha", "red", 25, bonus=1),
            _entry(1, _stamp(5), "beta", "blue", 18),
        ],
    )
    second = _event(
        2,
        _stamp(9),
        [
            _entry(2, _stamp(9), "alpha", "blue", 18, sprint=3),
            _entry(2, _stamp(9), "beta", "blue", 25),
            _entry(2, _stamp(9), None, "blue", 0, adjustment=-10),
        ],
    )
    revision = _event(
        2,
        _stamp(15),
        [
            _entry(2, _stamp(15), "alpha", "blue", 0, status="revised"),
            _entry(2, _stamp(15), "beta", "blue", 25, status="revised"),
            _entry(2, _stamp(15), None, "blue", 0, adjustment=-10, status="revised"),
        ],
        status="revised",
    )
    return {"schema_version": 1, "rules": [_rule()]}, {
        "schema_version": 1,
        "events": [first, second, revision],
    }


def _load(tmp_path: Path, rules: dict, events: dict):
    rules_path = tmp_path / "scoring_rules.json"
    events_path = tmp_path / "event_points_evidence.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    events_path.write_text(json.dumps(events), encoding="utf-8")
    return load_scoring_ledger(rules_path, events_path)


def test_sprint_bonus_constructor_penalty_transfer_and_revision(tmp_path: Path) -> None:
    rules, events = _documents()
    ledger = _load(tmp_path, rules, events)
    roster = {"alpha": "blue", "beta": "blue"}
    early = ledger.standings_before(EventId(2021, 3), datetime(2021, 1, 12, tzinfo=UTC), roster)
    assert early["alpha"]["driver_points_before_race"] == 47
    assert early["beta"]["driver_points_before_race"] == 43
    assert early["alpha"]["constructor_points_before_race"] == 54
    assert early["alpha"]["driver_championship_position"] == 1
    assert early["alpha"]["driver_points_last_3"] is None
    late = ledger.standings_before(EventId(2021, 3), datetime(2021, 1, 16, tzinfo=UTC), roster)
    assert late["alpha"]["driver_points_before_race"] == 26
    assert late["alpha"]["constructor_points_before_race"] == 33
    assert late["beta"]["driver_championship_position"] == 1
    assert ledger.sha256 == _load(tmp_path, rules, events).sha256


def test_reduced_points_and_missing_evidence(tmp_path: Path) -> None:
    rules, events = _documents()
    events["events"][0]["entries"][0] = _entry(1, _stamp(5), "alpha", "red", 12.5)
    events["events"][0] = _hashed(
        {key: value for key, value in events["events"][0].items() if key != "evidence_hash"}
    )
    ledger = _load(tmp_path, rules, events)
    before_second = ledger.standings_before(
        EventId(2021, 2), datetime(2021, 1, 8, tzinfo=UTC), {"alpha": "blue"}
    )
    assert before_second["alpha"]["driver_points_before_race"] == 12.5
    events["events"][0]["complete"] = False
    events["events"][0] = _hashed(
        {key: value for key, value in events["events"][0].items() if key != "evidence_hash"}
    )
    ledger = _load(tmp_path, rules, events)
    missing = ledger.standings_before(
        EventId(2021, 2), datetime(2021, 1, 8, tzinfo=UTC), {"alpha": "blue"}
    )
    assert missing["alpha"]["driver_points_before_race"] is None
    assert missing["alpha"]["missing_reason"] == "audited_prior_event_missing_or_uncertain"


@pytest.mark.parametrize("change", ["bad_total", "bad_hash", "missing_driver", "bad_sprint"])
def test_evidence_schema_rejects_incomplete_or_malformed(tmp_path: Path, change: str) -> None:
    rules, events = _documents()
    broken = copy.deepcopy(events["events"][0])
    if change == "bad_total":
        broken["entries"][0]["total_points"] = 99
    elif change == "bad_hash":
        broken["entries"][0]["evidence_hash"] = "0" * 64
    elif change == "missing_driver":
        broken["entries"].pop()
    else:
        broken["entries"][0] = _entry(1, _stamp(5), "alpha", "red", 25, sprint=8)
    events["events"][0] = _hashed(
        {key: value for key, value in broken.items() if key != "evidence_hash"}
    )
    with pytest.raises(ValueError):
        _load(tmp_path, rules, events)


def test_gold_scoring_waits_for_audit_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_gold_scoring(tmp_path, tmp_path / "source")


def test_midseason_rule_change_and_uncertain_revision(tmp_path: Path) -> None:
    rules, events = _documents()
    first_rule = rules["rules"][0]
    first_rule["last_round"] = 1
    rules["rules"][0] = _hashed(
        {key: value for key, value in first_rule.items() if key != "evidence_hash"}
    )
    second_rule = copy.deepcopy(first_rule)
    second_rule["first_round"] = 2
    second_rule["last_round"] = 22
    rules["rules"].append(
        _hashed({key: value for key, value in second_rule.items() if key != "evidence_hash"})
    )
    ledger = _load(tmp_path, rules, events)
    assert ledger.rule_for(EventId(2021, 1)).first_round == 1
    assert ledger.rule_for(EventId(2021, 2)).first_round == 2
    events["events"][2]["revision_status"] = "disputed"
    for entry in events["events"][2]["entries"]:
        entry["revision_status"] = "disputed"
        entry.update(
            _hashed({key: value for key, value in entry.items() if key != "evidence_hash"})
        )
    events["events"][2] = _hashed(
        {key: value for key, value in events["events"][2].items() if key != "evidence_hash"}
    )
    ledger = _load(tmp_path, rules, events)
    missing = ledger.standings_before(
        EventId(2021, 3), datetime(2021, 1, 16, tzinfo=UTC), {"alpha": "blue"}
    )
    assert missing["alpha"]["driver_points_before_race"] is None
    assert missing["alpha"]["missing_reason"] == "audited_prior_event_missing_or_uncertain"


def test_tied_standings_position_stays_missing_without_countback(tmp_path: Path) -> None:
    rules, events = _documents()
    events["events"][0]["entries"][0] = _entry(1, _stamp(5), "alpha", "red", 18)
    events["events"][0] = _hashed(
        {key: value for key, value in events["events"][0].items() if key != "evidence_hash"}
    )
    ledger = _load(tmp_path, rules, events)
    row = ledger.standings_before(
        EventId(2021, 2),
        datetime(2021, 1, 8, tzinfo=UTC),
        {"alpha": "red", "beta": "blue"},
    )["alpha"]
    assert row["driver_points_before_race"] == 18
    assert row["driver_championship_position"] is None
    assert row["driver_points_gap_to_leader"] == 0


def test_season_without_constructor_championship_keeps_team_points_missing(
    tmp_path: Path,
) -> None:
    rules, events = _documents()
    rule = rules["rules"][0]
    rule["constructor_scoring"] = "not_contested"
    rules["rules"][0] = _hashed(
        {key: value for key, value in rule.items() if key != "evidence_hash"}
    )
    ledger = _load(tmp_path, rules, events)
    row = ledger.standings_before(
        EventId(2021, 2), datetime(2021, 1, 8, tzinfo=UTC), {"alpha": "red"}
    )["alpha"]
    assert row["driver_points_before_race"] == 26
    assert row["constructor_points_before_race"] is None
    assert row["constructor_missing_reason"] == "constructor_championship_not_contested"


def test_new_gold_version_binds_scoring_hash_and_preserves_source(tmp_path: Path) -> None:
    rules, events = _documents()
    rules_path, evidence_path = tmp_path / "rules.json", tmp_path / "points.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    evidence_path.write_text(json.dumps(events), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for number in (1, 2):
        for driver, constructor in (("alpha", "red"), ("beta", "blue")):
            row = {
                "event_id": EventId(2021, number).partition(),
                "driver_id": driver,
                "constructor_id": constructor,
                "benchmark_tier": "Gold",
                "cutoff_kind": "post_qualifying",
                "prediction_timestamp": datetime(2021, 1, number * 4 + 2, tzinfo=UTC),
                "feature_timestamp": datetime(2021, 1, 4, tzinfo=UTC),
            }
            row.update(
                {name: 999.0 for name in _POINT_FEATURES if "driver_points_last" not in name}
            )
            row.update({f"{name}_missing": False for name in row if name in _POINT_FEATURES})
            rows.append(row)
    table = pa.Table.from_pylist(rows)
    datasets = {}
    for tier in ("Gold", "Silver", "Development"):
        path = source / f"{tier.lower()}.parquet"
        pq.write_table(table if tier == "Gold" else table.slice(0, 0), path)
        datasets[tier] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "rows": table.num_rows if tier == "Gold" else 0,
            "events": [EventId(2021, number).partition() for number in (1, 2)]
            if tier == "Gold"
            else [],
        }
    (source / "coverage.json").write_text("{}", encoding="utf-8")
    (source / "feature_provenance.json").write_text("[]", encoding="utf-8")
    manifest = {
        "version": 3,
        "enrichment_version": ENRICHMENT_VERSION,
        "coverage_sha256": file_sha256(source / "coverage.json"),
        "feature_provenance_sha256": file_sha256(source / "feature_provenance.json"),
        "feature_columns": [],
        "datasets": datasets,
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = build_gold_scoring(tmp_path, source, rules_path, evidence_path)
    output = tmp_path / report["benchmark_dir"]
    frozen = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert frozen["version"] == 4
    assert frozen["scoring_ledger_sha256"] == report["scoring_ledger_sha256"]
    assert frozen["source_manifest_sha256"] == file_sha256(source / "manifest.json")
    assert file_sha256(source / "gold.parquet") == datasets["Gold"]["sha256"]
    values = pq.read_table(output / "gold.parquet").to_pylist()
    assert values[0]["driver_points_before_race"] == 0
    assert values[2]["driver_points_before_race"] == 26
    assert values[2]["feature_timestamp"] == datetime(2021, 1, 5, 18, tzinfo=UTC)
    assert values[2]["driver_points_last_3_missing"] is True
    assert build_gold_scoring(tmp_path, source, rules_path, evidence_path) == report
    verify_scoring_manifest(output, frozen)
    (output / "scoring_provenance.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance hash mismatch"):
        verify_scoring_manifest(output, frozen)
