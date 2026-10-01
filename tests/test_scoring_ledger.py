"""Synthetic scoring fixtures exercise audit and point-in-time behavior."""

import copy
import hashlib
import json
from dataclasses import replace
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
from f1_ml_predictor.benchmarks.versioning import archive_benchmark
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
    race_schedule: str = "standard",
) -> dict:
    return _hashed(
        {
            "season": 2021,
            "event_id": EventId(2021, round_number).partition(),
            "completed_at": _stamp(round_number * 4),
            "effective_at": effective,
            "race_schedule": race_schedule,
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
    events["events"][0]["entries"][1] = _entry(1, _stamp(5), "beta", "blue", 9)
    events["events"][0]["race_schedule"] = "reduced:0"
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


@pytest.mark.parametrize(
    "change", ["bad_total", "bad_hash", "missing_driver", "bad_sprint", "bad_reduced"]
)
def test_evidence_schema_rejects_incomplete_or_malformed(tmp_path: Path, change: str) -> None:
    rules, events = _documents()
    broken = copy.deepcopy(events["events"][0])
    if change == "bad_total":
        broken["entries"][0]["total_points"] = 99
    elif change == "bad_hash":
        broken["entries"][0]["evidence_hash"] = "0" * 64
    elif change == "missing_driver":
        broken["entries"].pop()
    elif change == "bad_reduced":
        broken["entries"][0] = _entry(1, _stamp(5), "alpha", "red", 12.5)
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
    first = ledger.events[0]
    audited = replace(
        first,
        entries=tuple(
            replace(entry, race_position=position, race_position_audited=True)
            for entry, position in zip(first.entries, (1, 2), strict=True)
        ),
    )
    ledger = replace(ledger, events=(audited, *ledger.events[1:]))
    ordered = ledger.standings_before(
        EventId(2021, 2),
        datetime(2021, 1, 8, tzinfo=UTC),
        {"alpha": "red", "beta": "blue"},
    )
    assert ordered["alpha"]["driver_championship_position"] == 1
    assert ordered["beta"]["driver_championship_position"] == 2


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
            if number == 2 and driver == "alpha":
                constructor = "blue"
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
    (source / "coverage.json").write_text('{"coverage":[]}', encoding="utf-8")
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
    archived = archive_benchmark(output)
    assert archived is not None
    verify_scoring_manifest(archived["path"], frozen)
    assert (archived["path"] / "feature_provenance.json").is_file()
    (output / "scoring_provenance.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance hash mismatch"):
        verify_scoring_manifest(output, frozen)


def _native_fixture(tmp_path: Path) -> tuple[Path, Path]:
    audit = tmp_path / "data/audit"
    objects = tmp_path / "data/raw/fia_audit/objects"
    collection_dir = tmp_path / "data/raw/scoring_audit"
    audit.mkdir(parents=True)
    objects.mkdir(parents=True)
    collection_dir.mkdir(parents=True)

    def retained(contents: bytes) -> str:
        digest = hashlib.sha256(contents).hexdigest()
        (objects / f"{digest}.pdf").write_bytes(contents)
        return digest

    regulation = retained(b"fixture regulation")
    first = retained(b"fixture first points")
    revised = retained(b"fixture revised points")
    collection_bytes = b"fixture collection index"
    collection_hash = hashlib.sha256(collection_bytes).hexdigest()
    collection_path = collection_dir / "index.json"
    collection_path.write_bytes(collection_bytes)
    regulation_source = {"sha256": regulation, "url": "https://fia.example/regulation"}
    season_rule = {
        "season": 2022,
        "constructor_scoring": "sum of both cars",
        "fastest_lap_rule": "No fastest-lap point",
        "evidence_source": {
            "first_issue_in_force": regulation_source,
            "last_issue_checked": regulation_source,
        },
    }
    schedules = {
        "race": (25, 18),
        "sprint": (8,),
        "race_reduced_col1_2laps_to_lt25pct": (6,),
        "race_reduced_col2_25_to_lt50pct": (13,),
        "race_reduced_col3_50_to_lt75pct": (19,),
    }
    rules = {
        "schema_version": "scoring-evidence-v1",
        "audit_as_of_utc": "2022-01-20T00:00:00+00:00",
        "season_rules": [season_rule],
        "rules": [
            {"season": 2022, "event_type": kind, "position": position, "points": value}
            for kind, values in schedules.items()
            for position, value in enumerate(values, 1)
        ],
    }
    event_id = EventId(2022, 1).partition()
    first_at, revised_at = "2022-01-05T18:00:00+00:00", "2022-01-08T18:00:00+00:00"
    documents = [
        {
            "document_key": digest,
            "sha256": digest,
            "recalled": False,
            "url": f"https://fia.example/{digest}",
            "published_at_utc_upper_bound": when,
        }
        for digest, when in ((first, first_at), (revised, revised_at))
    ]
    records = []
    constructors = []
    for driver, constructor, initial, final, old_place, new_place in (
        ("alpha", "red", 25, 18, "1", "2"),
        ("beta", "blue", 18, 25, "2", "1"),
    ):
        records.append(
            {
                "season": 2022,
                "round": 1,
                "event_id_if_known": event_id,
                "driver": driver,
                "constructor": constructor,
                "race_points": final,
                "sprint_points": 0,
                "bonus_points": 0,
                "total_event_points": final,
                "checks": {"race_points_match_full_scale": True},
                "evidence_quality": "fia_official_formula1_confirmed",
                "revision_status": "revised",
                "first_published_at": first_at,
                "published_at": revised_at,
                "points_timeline": [
                    {
                        "document": first,
                        "published_at_utc_upper_bound": first_at,
                        "points": initial,
                        "position_token": old_place,
                    },
                    {
                        "document": revised,
                        "published_at_utc_upper_bound": revised_at,
                        "points": final,
                        "position_token": new_place,
                    },
                ],
            }
        )
        constructors.append(
            {
                "event_id_if_known": event_id,
                "constructor": constructor,
                "agrees": True,
                "fia_entrant_event_points": final,
            }
        )
    evidence = {
        "schema_version": "scoring-evidence-v1",
        "audit_as_of_utc": rules["audit_as_of_utc"],
        "collection_index": {
            "path": "data/raw/scoring_audit/index.json",
            "sha256": collection_hash,
        },
        "unresolved_events": [],
        "event_summaries": [
            {
                "season": 2022,
                "round": 1,
                "event_id": event_id,
                "status": "resolved",
                "drivers": 2,
            }
        ],
        "documents": documents,
        "records": records,
        "constructor_event_points": constructors,
    }
    rules_path, evidence_path = audit / "scoring_rules.json", audit / "event_points_evidence.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return rules_path, evidence_path


def test_native_fia_timeline_uses_only_prior_revisions(tmp_path: Path) -> None:
    rules_path, evidence_path = _native_fixture(tmp_path)
    ledger = load_scoring_ledger(rules_path, evidence_path)
    assert len(ledger.events) == 2
    ledger = replace(ledger, rules={2022: (replace(ledger.rules[2022][0], last_round=2),)})
    roster = {"alpha": "red", "beta": "blue"}
    early = ledger.standings_before(EventId(2022, 2), datetime(2022, 1, 7, tzinfo=UTC), roster)
    late = ledger.standings_before(EventId(2022, 2), datetime(2022, 1, 9, tzinfo=UTC), roster)
    assert early["alpha"]["driver_points_before_race"] == 25
    assert early["alpha"]["driver_championship_position"] == 1
    assert late["alpha"]["driver_points_before_race"] == 18
    assert late["beta"]["driver_championship_position"] == 1
    assert late["alpha"]["constructor_points_before_race"] == 18
    assert ledger.sha256 == load_scoring_ledger(rules_path, evidence_path).sha256


@pytest.mark.parametrize(
    "damage", ["document_hash", "collection_hash", "missing_driver", "disputed"]
)
def test_native_audit_rejects_damage_and_keeps_disputes_missing(
    tmp_path: Path, damage: str
) -> None:
    rules_path, evidence_path = _native_fixture(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if damage == "document_hash":
        evidence["documents"][0]["sha256"] = "0" * 64
    elif damage == "collection_hash":
        evidence["collection_index"]["sha256"] = "0" * 64
    elif damage == "missing_driver":
        evidence["records"].pop()
    else:
        evidence["records"][0]["revision_status"] = "pending_appeal"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    if damage == "disputed":
        ledger = load_scoring_ledger(rules_path, evidence_path)
        ledger = replace(ledger, rules={2022: (replace(ledger.rules[2022][0], last_round=2),)})
        row = ledger.standings_before(
            EventId(2022, 2), datetime(2022, 1, 9, tzinfo=UTC), {"alpha": "red"}
        )["alpha"]
        assert row["driver_points_before_race"] is None
        assert row["missing_reason"] == "audited_prior_event_missing_or_uncertain"
    else:
        with pytest.raises(ValueError):
            load_scoring_ledger(rules_path, evidence_path)
