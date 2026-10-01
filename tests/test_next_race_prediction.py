"""Offline fixtures exercise development next-race predictions end to end."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.benchmarks.builder import BENCHMARK_FEATURE_COLUMNS, file_sha256
from f1_ml_predictor.benchmarks.enrichment import _COUNTS, ENRICHMENT_FEATURE_COLUMNS
from f1_ml_predictor.benchmarks.rolling import ROLLING_FEATURE_COLUMNS, rolling_driver_features
from f1_ml_predictor.benchmarks.scoring import NEW_FEATURE_COLUMNS, SCORING_VERSION
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.models.boosting import BACKENDS
from f1_ml_predictor.models.development import _check_race
from f1_ml_predictor.models.protocol import PROTOCOL
from f1_ml_predictor.prediction.history import (
    choose_dnf_model,
    choose_position_backend,
    load_gold_version,
)
from f1_ml_predictor.prediction.live_features import (
    PRE_QUALIFYING,
    ScheduledEvent,
    apply_mask,
    availability_mask,
    build_live_rows,
    freeze_snapshot,
    load_schedule,
)
from f1_ml_predictor.prediction.pipeline import predict_next_race
from f1_ml_predictor.prediction.race import fit_composed_model
from f1_ml_predictor.prediction.season import (
    points_rules,
    published_standings,
    remaining_sessions,
)
from f1_ml_predictor.scoring.ledger import _digest, load_scoring_ledger
from f1_ml_predictor.simulation import (
    CurrentStandings,
    EventSimulation,
    PointsRules,
    simulate_championship,
)

SEASON = 2026
ROSTER = {"alpha": "red", "beta": "red", "gamma": "blue", "delta": "blue"}
ORDERS = (
    ("alpha", "gamma", "beta", "delta"),
    ("alpha", "beta", "gamma", "delta"),
    ("gamma", "alpha", "beta", "delta"),
    ("alpha", "gamma", "delta", "beta"),
    ("beta", "alpha", "gamma", "delta"),
    ("alpha", "gamma", "beta", "delta"),
)
RETIRED = {(2, "delta"), (4, "beta"), (6, "gamma")}
POINTS = (10, 6, 4, 3)
SOURCE = {"reference": "synthetic:fixture", "sha256": "a" * 64}
GOLD_COLUMNS = (
    *BENCHMARK_FEATURE_COLUMNS,
    *ROLLING_FEATURE_COLUMNS,
    *ENRICHMENT_FEATURE_COLUMNS,
    *NEW_FEATURE_COLUMNS,
)
INTEGER_COLUMNS = {*(f"history_count_{window}" for window in (3, 5, 10)), *_COUNTS}
CLOCK = datetime(2026, 7, 1, 12, tzinfo=UTC)


def _race_start(round_number: int) -> datetime:
    return datetime(2026, 3, 1, 14, tzinfo=UTC) + timedelta(days=14 * (round_number - 1))


def _schema(columns: tuple[str, ...]) -> pa.Schema:
    timestamp = pa.timestamp("us", tz="UTC")
    fields = [
        pa.field("event_id", pa.string()),
        pa.field("driver_id", pa.string()),
        pa.field("constructor_id", pa.string()),
        pa.field("circuit_id", pa.string()),
        pa.field("prediction_timestamp", timestamp),
        pa.field("feature_timestamp", timestamp),
        pa.field("benchmark_tier", pa.string()),
        pa.field("cutoff_kind", pa.string()),
        pa.field("qualifying_status", pa.string()),
        pa.field("start_type", pa.string()),
        pa.field("pit_lane_start", pa.bool_()),
        pa.field("grid_status", pa.string()),
    ]
    for name in columns:
        kind = (
            pa.bool_()
            if name.endswith("_missing")
            else pa.int32()
            if name in INTEGER_COLUMNS
            else pa.float64()
        )
        fields.append(pa.field(name, kind))
    fields += [
        pa.field("label_position", pa.int64()),
        pa.field("label_winner", pa.bool_()),
        pa.field("label_podium", pa.bool_()),
        pa.field("label_dnf", pa.bool_()),
        pa.field("label_available_at", timestamp),
        pa.field("label_final_audited", pa.bool_()),
        pa.field("label_audit_reference", pa.string()),
    ]
    return pa.schema(fields)


def _rows(columns: tuple[str, ...], *, dnf_labels: bool) -> list[dict]:
    rows = []
    for round_number, order in enumerate(ORDERS, 1):
        cutoff = _race_start(round_number) - timedelta(hours=20)
        for position, driver in enumerate(order, 1):
            row = {
                name: True if name.endswith("_missing") else 0 if name in INTEGER_COLUMNS else None
                for name in columns
            }
            qualifying = float(position if round_number % 2 else 5 - position)
            row.update(
                {
                    "event_id": EventId(SEASON, round_number).partition(),
                    "driver_id": driver,
                    "constructor_id": ROSTER[driver],
                    "circuit_id": f"circuit_{round_number}",
                    "prediction_timestamp": cutoff,
                    "feature_timestamp": cutoff - timedelta(hours=1),
                    "benchmark_tier": "Gold",
                    "cutoff_kind": "post_qualifying",
                    "qualifying_status": "completed",
                    "start_type": "grid",
                    "pit_lane_start": False,
                    "grid_status": "unknown",
                    "qualifying_position": qualifying,
                    "qualifying_position_missing": False,
                    "label_position": position,
                    "label_winner": position == 1,
                    "label_podium": position <= 3,
                    "label_dnf": (round_number, driver) in RETIRED if dnf_labels else None,
                    "label_available_at": _race_start(round_number) + timedelta(hours=6),
                    "label_final_audited": True,
                    "label_audit_reference": "synthetic-final",
                }
            )
            rows.append(row)
    return rows


def _write_version(directory: Path, rows: list[dict], schema: pa.Schema, extra: dict) -> Path:
    directory.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), directory / "gold.parquet")
    empty = pa.Table.from_pylist([], schema=schema)
    pq.write_table(empty, directory / "silver.parquet")
    pq.write_table(empty, directory / "development.parquet")
    (directory / "coverage.json").write_text("{}", encoding="utf-8")
    events = sorted({row["event_id"] for row in rows})
    manifest = {
        "coverage_sha256": file_sha256(directory / "coverage.json"),
        "datasets": {
            "Gold": {
                "path": "gold.parquet",
                "sha256": file_sha256(directory / "gold.parquet"),
                "rows": len(rows),
                "events": events,
            },
            "Silver": {
                "path": "silver.parquet",
                "sha256": file_sha256(directory / "silver.parquet"),
            },
            "Development": {
                "path": "development.parquet",
                "sha256": file_sha256(directory / "development.parquet"),
            },
        },
        **extra,
    }
    if manifest.get("version") == 4:
        for name, key in (
            ("feature_provenance.json", "feature_provenance_sha256"),
            ("scoring_provenance.json", "scoring_provenance_sha256"),
        ):
            (directory / name).write_text("[]", encoding="utf-8")
            manifest[key] = file_sha256(directory / name)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    final = directory.with_name(f"dataset-{file_sha256(directory / 'manifest.json')}")
    directory.rename(final)
    return final


def _hashed(data: dict) -> dict:
    data["evidence_hash"] = _digest(data)
    return data


def _write_ledger(root: Path) -> None:
    audit = root / "data/audit"
    audit.mkdir(parents=True)
    rule = _hashed(
        {
            "season": SEASON,
            "first_round": 1,
            "last_round": 6,
            "race_points": list(POINTS),
            "sprint_points": [3, 2, 1],
            "reduced_race_points": [],
            "fastest_lap_points": 0,
            "fastest_lap_eligible_through": None,
            "constructor_scoring": "sum_awarded_entries",
            "source_evidence": [SOURCE],
            "revision_status": "audited",
        }
    )
    events = []
    for round_number, order in enumerate(ORDERS, 1):
        effective = (_race_start(round_number) + timedelta(hours=5)).isoformat()
        event_id = EventId(SEASON, round_number).partition()
        status = "disputed" if round_number == 5 else "audited"
        entries = [
            _hashed(
                {
                    "season": SEASON,
                    "event_id": event_id,
                    "driver_id": driver,
                    "constructor_id": ROSTER[driver],
                    "race_points": POINTS[position],
                    "sprint_points": 0,
                    "bonus_points": 0,
                    "adjustment_points": 0,
                    "total_points": POINTS[position],
                    "effective_at": effective,
                    "source_evidence": [SOURCE],
                    "revision_status": status,
                }
            )
            for position, driver in enumerate(order)
        ]
        events.append(
            _hashed(
                {
                    "season": SEASON,
                    "event_id": event_id,
                    "completed_at": (_race_start(round_number) + timedelta(hours=2)).isoformat(),
                    "effective_at": effective,
                    "race_schedule": "standard",
                    "complete": True,
                    "revision_status": status,
                    "expected_driver_ids": sorted(ROSTER),
                    "entries": entries,
                }
            )
        )
    (audit / "scoring_rules.json").write_text(
        json.dumps({"schema_version": 1, "rules": [rule]}), encoding="utf-8"
    )
    (audit / "event_points_evidence.json").write_text(
        json.dumps({"schema_version": 1, "events": events}), encoding="utf-8"
    )


def _schedule_payload() -> dict:
    races = []
    for round_number in range(1, 10):
        start = (
            _race_start(round_number)
            if round_number < 7
            else (CLOCK + timedelta(days=3 + 14 * (round_number - 7)))
        )
        race = {
            "season": str(SEASON),
            "round": str(round_number),
            "raceName": f"Grand Prix {round_number}",
            "Circuit": {"circuitId": f"circuit_{round_number}"},
            "date": start.date().isoformat(),
            "time": start.strftime("%H:%M:%SZ"),
        }
        qualifying = start - timedelta(days=1)
        race["Qualifying"] = {
            "date": qualifying.date().isoformat(),
            "time": qualifying.strftime("%H:%M:%SZ"),
        }
        if round_number == 8:
            sprint = start - timedelta(hours=26)
            race["Sprint"] = {
                "date": sprint.date().isoformat(),
                "time": sprint.strftime("%H:%M:%SZ"),
            }
        races.append(race)
    return {"MRData": {"RaceTable": {"season": str(SEASON), "Races": races}}}


def _write_schedule(root: Path, observed: datetime) -> str:
    payload = _schedule_payload()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    relative = "data/raw/prospective_scheduler/observations/schedule-fixture.json"
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "provider": "jolpica",
                "url": "https://api.jolpi.ca/ergast/f1/2026/",
                "role": "schedule",
                "response_captured_at": observed.isoformat(),
                "payload_sha256": hashlib.sha256(canonical).hexdigest(),
                "payload": payload,
            }
        ),
        encoding="utf-8",
    )
    (root / "data/raw/prospective_scheduler/status.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "waiting_for_qualifying",
                "events": {},
                "evaluation_eligible": False,
                "schedule_observation": relative,
            }
        ),
        encoding="utf-8",
    )
    return relative


def _reference(root: Path, version_dir: Path, *, dnf: bool) -> None:
    metrics = {
        "hist": (1.10, 0.15, 1.10),
        "catboost": (1.20, 0.16, 1.00),
        "lightgbm": (1.30, 0.17, 1.30),
        "xgboost": (1.40, 0.18, 1.40),
    }

    def task(values: tuple[float, float, float]) -> dict:
        return {
            "winner": {"status": "evaluated", "log_loss": values[0]},
            "podium": {"status": "evaluated", "brier_score": values[1]},
            "finishing_position": {"status": "evaluated", "mean_absolute_error": values[2]},
            "dnf": {"status": "evaluated", "brier_score": 0.12},
        }

    baselines = {"heuristic": task((1.7, 0.2, 1.5)), "logistic": task((1.0, 0.14, 0.9))}
    no_selection = {"status": "no_selection", "selected_backend": None}
    report = {
        "benchmark_manifest_sha256": file_sha256(version_dir / "manifest.json"),
        "evaluation_protocol": PROTOCOL,
        "run_metadata": {
            "run_id": "dnf-run" if dnf else "position-run",
            "created_at": "2026-06-01T00:00:00+00:00",
            "calibration_method": "sigmoid",
            "model_parameters": {"backends": list(BACKENDS), "calibration_events": 1},
        },
        "comparisons": {
            "post_qualifying": {
                backend: {"metrics": task(values), "baselines": baselines}
                for backend, values in metrics.items()
            }
        },
        "task_selection": {
            "post_qualifying": {
                "winner": no_selection,
                "podium": no_selection,
                "finishing_position": no_selection,
                "dnf": {
                    **no_selection,
                    "observed_lowest_loss": {
                        "model": "logistic",
                        "value": 0.11,
                        "metric": "brier_score",
                    },
                },
            }
        },
    }
    path = root / "models/experiments/gold" / version_dir.name / "runs/fixture/comparison.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    root = tmp_path
    _write_ledger(root)
    ledger = load_scoring_ledger(
        root / "data/audit/scoring_rules.json", root / "data/audit/event_points_evidence.json"
    )
    outcomes_dir = root / "data/audit_outcomes"
    outcomes_dir.mkdir(parents=True)
    races = []
    for round_number in range(1, 7):
        path = outcomes_dir / f"round-{round_number}.parquet"
        path.write_bytes(f"outcome {round_number}".encode())
        races.append(
            {
                "event_id": EventId(SEASON, round_number).partition(),
                "outcomes": {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                },
            }
        )
    registry = root / "data/benchmarks/gold_core_registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"races": races}), encoding="utf-8")
    rows = _rows(GOLD_COLUMNS, dnf_labels=False)
    events: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        events.setdefault((SEASON, int(row["event_id"][-2:])), []).append(row)
    outcome_hashes = {key: "b" * 64 for key in events}
    for row in rows:
        values, _ = rolling_driver_features(
            events,
            outcome_hashes,
            SEASON,
            int(row["event_id"][-2:]),
            row["driver_id"],
            row["prediction_timestamp"],
        )
        row.update(values)
        for window in (3, 5, 10):
            row[f"recent_finish_mean_{window}_missing"] = (
                values[f"recent_finish_mean_{window}"] is None
            )
    gold = _write_version(
        root / "data/benchmarks/gold_championship_scoring_v3/source/ledger/versions/pending",
        rows,
        _schema(GOLD_COLUMNS),
        {
            "version": 4,
            "scoring_version": SCORING_VERSION,
            "feature_columns": list(GOLD_COLUMNS),
            "catalog_sha256": file_sha256(registry),
            "scoring_ledger_sha256": ledger.sha256,
            "scoring_rules_sha256": "c" * 64,
            "event_points_evidence_sha256": "d" * 64,
            "source_manifest_sha256": "e" * 64,
        },
    )
    dnf = _write_version(
        root / "data/benchmarks/gold_core_binary_dnf_v1/source/benchmark/versions/pending",
        _rows(BENCHMARK_FEATURE_COLUMNS, dnf_labels=True),
        _schema(BENCHMARK_FEATURE_COLUMNS),
        {"version": 1, "feature_columns": list(BENCHMARK_FEATURE_COLUMNS)},
    )
    _reference(root, gold, dnf=False)
    _reference(root, dnf, dnf=True)
    _write_schedule(root, CLOCK - timedelta(hours=1))
    return {"root": root, "gold": gold, "dnf": dnf, "ledger": ledger}


def test_end_to_end_development_prediction_is_coherent_and_labelled(workspace: dict) -> None:
    root = workspace["root"]
    result = predict_next_race(
        root,
        simulations=300,
        draws=512,
        championship_orders=128,
        device="cpu",
        gold_dir=workspace["gold"],
        dnf_dir=workspace["dnf"],
        now=lambda: CLOCK,
    )
    assert result["status"] == "development_only"
    assert result["event_id"] == "season=2026/round=07"
    assert result["cutoff_kind"] == PRE_QUALIFYING
    run_dir = root / result["run_dir"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_status"] == "development_only"
    assert manifest["validated_forecast"] is False
    assert all(manifest["validation_checks"].values())
    assert manifest["model_choice"]["position"]["backend"] == "hist"
    assert manifest["model_choice"]["dnf"]["model"] == "logistic"
    assert manifest["execution"]["devices"] == {PRE_QUALIFYING: "cpu"}
    masked = set(manifest["models"][PRE_QUALIFYING]["masked_for_training"])
    assert "qualifying_position" in masked
    assert any("round 5" in note for note in manifest["models"][PRE_QUALIFYING]["scoring_gate"])

    table = pq.read_table(run_dir / "predictions.parquet").to_pylist()
    assert {row["validation_status"] for row in table} == {"development_only"}
    assert not any(name.startswith("label_") for name in table[0])
    race = json.loads((run_dir / "race_distribution.json").read_text(encoding="utf-8"))
    _check_race(race["drivers"])
    assert sum(row["winner_probability"] for row in race["drivers"]) == pytest.approx(1)

    championship = json.loads((run_dir / "championship.json").read_text(encoding="utf-8"))
    assert championship["validation_status"] == "development_only"
    assert championship["simulator"]["status"] == "engineering_only"
    sessions = [(item["event_id"], item["session"]) for item in championship["sessions"]]
    assert sessions[0] == ("season=2026/round=07", "race")
    assert ("season=2026/round=08", "sprint") in sessions
    assert championship["starting_points"]["driver_points"]["alpha"] == pytest.approx(52)
    assert any("round 5" in note for note in championship["standings_source"]["notes"])
    wdc = championship["simulator"]["wdc"]
    assert sum(wdc["title_probability"].values()) + wdc["unresolved_tie_probability"] == (
        pytest.approx(1)
    )
    for places in wdc["final_position_probability"].values():
        assert sum(places) == pytest.approx(1)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "DEVELOPMENT ONLY" in report
    assert chr(0x2014) not in report

    snapshot = pq.read_table(root / manifest["snapshot"][PRE_QUALIFYING]["path"]).to_pylist()
    assert max(row["feature_timestamp"] for row in snapshot) <= CLOCK
    assert all(row["qualifying_position"] is None for row in snapshot)


def test_live_rows_use_only_audited_history_published_before_cutoff(workspace: dict) -> None:
    version = load_gold_version(workspace["gold"])
    ledger = workspace["ledger"]
    target = ScheduledEvent(
        EventId(SEASON, 7), "Grand Prix 7", "circuit_7", CLOCK + timedelta(days=3), None, None
    )
    outcomes = {key: "f" * 64 for key in version.events}
    rows, provenance = build_live_rows(
        version,
        outcomes,
        ledger,
        target=target,
        cutoff=CLOCK,
        cutoff_kind=PRE_QUALIFYING,
        roster=ROSTER,
        opened_at=CLOCK - timedelta(hours=1),
    )
    alpha = next(row for row in rows if row["driver_id"] == "alpha")
    assert alpha["recent_finish_mean_3"] == pytest.approx((2 + 1 + 1) / 3)
    assert alpha["qualifying_position"] is None and alpha["qualifying_position_missing"]
    assert alpha["driver_points_before_race"] is None
    assert not any(name.startswith("label_") for name in alpha)
    assert all(row["feature_timestamp"] <= CLOCK for row in rows)
    reasons = next(item for item in provenance if item["driver_id"] == "alpha")["missing_reasons"]
    assert reasons["qualifying_position"] == "qualifying_not_held_before_cutoff"

    early = _race_start(6) + timedelta(hours=1)
    rows, _ = build_live_rows(
        version,
        outcomes,
        ledger,
        target=target,
        cutoff=early,
        cutoff_kind=PRE_QUALIFYING,
        roster=ROSTER,
        opened_at=early - timedelta(minutes=5),
    )
    assert all(row["recent_finish_mean_3"] is None for row in rows)

    masked = availability_mask(version, rows)
    assert "qualifying_position" in masked
    hidden = apply_mask(version.rows, masked)
    assert all(row["qualifying_position"] is None for row in hidden)
    assert all(row["qualifying_position_missing"] for row in hidden)
    first = freeze_snapshot(workspace["root"], version, rows, {"fixture": True})
    assert freeze_snapshot(workspace["root"], version, rows, {"fixture": True}) == first


def test_composed_dnf_is_independent_of_pace(workspace: dict) -> None:
    version = load_gold_version(workspace["gold"])
    dnf_version = load_gold_version(workspace["dnf"])
    target = ScheduledEvent(
        EventId(SEASON, 7), "Grand Prix 7", "circuit_7", CLOCK + timedelta(days=3), None, None
    )
    rows, _ = build_live_rows(
        version,
        {key: "f" * 64 for key in version.events},
        workspace["ledger"],
        target=target,
        cutoff=CLOCK,
        cutoff_kind=PRE_QUALIFYING,
        roster=ROSTER,
        opened_at=CLOCK,
    )
    for position, row in enumerate(rows, 1):
        row["qualifying_position"] = float(position)
        row["qualifying_position_missing"] = False
    masked = availability_mask(version, rows)
    assert "qualifying_position" not in masked
    model = fit_composed_model(
        apply_mask(version.rows, masked),
        rows,
        apply_mask(dnf_version.rows, tuple(n for n in masked if n in dnf_version.feature_columns)),
        backend="hist",
        dnf_name="logistic",
        cutoff=CLOCK,
        cutoff_kind=PRE_QUALIFYING,
        seed=7,
        device="cpu",
        hardware=None,
    )
    distribution = model.distribution(rows, draws=1024, seed=7)
    _check_race(distribution)
    assert len({row["position_score"] for row in distribution}) > 1
    ordered = sorted(rows, key=lambda row: row["driver_id"])
    expected = model.dnf.predict(ordered)
    assert [row["dnf_model_probability"] for row in distribution] == pytest.approx(expected)
    faster = [{**row, "recent_finish_mean_3": 1.0, "recent_finish_mean_5": 1.0} for row in rows]
    changed = model.distribution(faster, draws=1024, seed=7)
    assert [row["dnf_model_probability"] for row in changed] == pytest.approx(expected)
    assert model.metadata["dnf_model"]["training_labels"] == 24
    orders = model.orders(rows, draws=256, seed=7)
    assert all(sorted(order) == sorted(ROSTER) for order in orders)


def test_model_choice_respects_selection_and_development_ranking() -> None:
    def comparison(winner: float, podium: float, mae: float) -> dict:
        return {
            "metrics": {
                "winner": {"log_loss": winner},
                "podium": {"brier_score": podium},
                "finishing_position": {"mean_absolute_error": mae},
            },
            "baselines": {
                "logistic": {
                    "winner": {"log_loss": 1.0},
                    "podium": {"brier_score": 0.1},
                    "finishing_position": {"mean_absolute_error": 2.0},
                }
            },
        }

    selections = {
        task: {"status": "no_selection", "selected_backend": None}
        for task in ("winner", "podium", "finishing_position", "dnf")
    }
    report = {
        "comparisons": {
            "post_qualifying": {
                "lightgbm": comparison(1.30, 0.081, 3.07),
                "catboost": comparison(1.40, 0.080, 3.02),
            }
        },
        "task_selection": {"post_qualifying": selections},
    }
    choice = choose_position_backend(report, "post_qualifying")
    assert choice["backend"] == "catboost"
    assert choice["status"] == "development_candidate"
    selections["winner"] = {"status": "provisional", "selected_backend": "lightgbm"}
    assert choose_position_backend(report, "post_qualifying")["backend"] == "lightgbm"
    selections["dnf"] = {
        "status": "no_selection",
        "observed_lowest_loss": {"model": "heuristic", "value": 0.1},
    }
    with pytest.raises(ValueError, match="heuristic"):
        choose_dnf_model(report, "post_qualifying")


def test_season_inputs_use_published_points_and_explicit_rules(workspace: dict) -> None:
    ledger = workspace["ledger"]
    standings, notes, used = published_standings(ledger, EventId(SEASON, 7), CLOCK, ROSTER)
    assert standings.driver_points["alpha"] == 52
    assert standings.constructor_points["red"] + standings.constructor_points["blue"] == 138
    assert notes == ["round 5 uses published points whose status is disputed"]
    assert len(used) == 6
    with pytest.raises(ValueError, match="round 6"):
        published_standings(ledger, EventId(SEASON, 7), _race_start(6), ROSTER)
    rules, status = points_rules(ledger, EventId(SEASON, 8), "sprint")
    assert rules.points_by_position == (3, 2, 1)
    assert status == "continuation_of_audited_rounds_1_6"

    def scheduled(round_number: int, sprint: datetime | None = None) -> ScheduledEvent:
        start = CLOCK + timedelta(days=round_number)
        return ScheduledEvent(
            EventId(SEASON, round_number), "race", "c", start, start - timedelta(days=1), sprint
        )

    target = scheduled(7)
    later = scheduled(8, CLOCK + timedelta(days=7, hours=12))
    sessions = remaining_sessions([target, later], target, CLOCK)
    assert [kind for _, kind, _ in sessions] == ["race", "sprint", "race"]
    started = scheduled(7, CLOCK - timedelta(hours=1))
    with pytest.raises(ValueError, match="sprint has run"):
        remaining_sessions([started], started, CLOCK)


def test_schedule_observation_hash_is_verified(workspace: dict) -> None:
    root = workspace["root"]
    relative = "data/raw/prospective_scheduler/observations/schedule-fixture.json"
    schedule, metadata = load_schedule(root, relative)
    assert schedule[6].event == EventId(SEASON, 7)
    assert schedule[7].sprint_start is not None
    path = root / relative
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["MRData"]["RaceTable"]["Races"][0]["raceName"] = "Changed"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_schedule(root, relative)
    assert metadata["captured_at"] == (CLOCK - timedelta(hours=1)).isoformat()


def test_championship_position_distribution_splits_ties_evenly() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    standings = CurrentStandings(
        2026, cutoff, {"a": 10, "b": 10, "c": 0}, {"red": 10, "blue": 10}, "a" * 64
    )
    event = EventSimulation(
        event_id=EventId(2026, 2),
        scheduled_at=cutoff + timedelta(days=1),
        available_at=cutoff,
        driver_constructors={"a": "red", "b": "blue", "c": "blue"},
        sampled_orders=(("c", "a", "b"),),
        points_eligible_samples=(("a", "b", "c"),),
        rules=PointsRules(2026, "race", (0,), "zero", "synthetic", "no_points"),
        source_hash="b" * 64,
    )
    result = simulate_championship(
        standings, [event], prediction_timestamp=cutoff, simulations=10, seed=1
    )
    places = result.wdc.final_position_probability
    assert places["a"] == pytest.approx((0.5, 0.5, 0.0))
    assert places["c"] == pytest.approx((0.0, 0.0, 1.0))
    matrix = np.asarray(list(places.values()))
    assert np.allclose(matrix.sum(axis=0), 1) and np.allclose(matrix.sum(axis=1), 1)
    assert result.to_dict()["wcc"]["final_position_probability"]["red"] == [0.5, 0.5]
