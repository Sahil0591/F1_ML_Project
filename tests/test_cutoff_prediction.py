"""Offline fixtures for cutoff-specific contracts, calibration, OOD and season worlds."""

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.benchmarks.builder import BENCHMARK_FEATURE_COLUMNS, file_sha256
from f1_ml_predictor.benchmarks.enrichment import _COUNTS, ENRICHMENT_FEATURE_COLUMNS
from f1_ml_predictor.benchmarks.rolling import ROLLING_FEATURE_COLUMNS
from f1_ml_predictor.benchmarks.scoring import NEW_FEATURE_COLUMNS, SCORING_VERSION
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.models.development import _check_race
from f1_ml_predictor.prediction.candidates import PlackettLuceRegression, fit_strength
from f1_ml_predictor.prediction.contracts import build_rows, feature_columns
from f1_ml_predictor.prediction.evaluation import _icc, training_rows
from f1_ml_predictor.prediction.joint import (
    average,
    expected_calibration_error,
    marginals,
    mix,
    sample_mixture,
    sample_orders,
    sharpness,
)
from f1_ml_predictor.prediction.ood import ood_report
from f1_ml_predictor.prediction.pipeline import predict_next_race
from f1_ml_predictor.prediction.schedules import weekends
from f1_ml_predictor.prediction.workspace import load_audited_history
from f1_ml_predictor.scoring.ledger import _digest
from f1_ml_predictor.simulation import (
    CurrentStandings,
    EventSimulation,
    PointsRules,
    simulate_championship,
)
from f1_ml_predictor.trust.binary_dnf import audit_binary_status

SEASON = 2026
ROSTER = {"a1": "red", "a2": "red", "b1": "blue", "b2": "blue", "c1": "green", "c2": "green"}
STRENGTH = {"a1": 3.0, "a2": 2.2, "b1": 1.8, "b2": 1.0, "c1": 0.5, "c2": 0.0}
COMPLETED = 14
POINTS = (10, 6, 4, 3, 2, 1)
SOURCE = {"reference": "synthetic:fixture", "sha256": "a" * 64}
GOLD_COLUMNS = (
    *BENCHMARK_FEATURE_COLUMNS,
    *ROLLING_FEATURE_COLUMNS,
    *ENRICHMENT_FEATURE_COLUMNS,
    *NEW_FEATURE_COLUMNS,
)
INTEGERS = {*(f"history_count_{window}" for window in (3, 5, 10)), *_COUNTS}
CLOCK = datetime(2026, 3, 1, 14, tzinfo=UTC) + timedelta(days=14 * (COMPLETED - 1) + 3)


def _start(round_number: int) -> datetime:
    if round_number <= COMPLETED:
        return datetime(2026, 3, 1, 14, tzinfo=UTC) + timedelta(days=14 * (round_number - 1))
    return CLOCK + timedelta(days=4 + 14 * (round_number - COMPLETED - 1))


def _circuit(round_number: int) -> str:
    return "sepang" if round_number == COMPLETED + 1 else f"circuit_{(round_number - 1) % 7}"


def _results(round_number: int) -> tuple[list[str], str | None]:
    rng = np.random.default_rng(round_number)
    noisy = {driver: value + rng.gumbel() for driver, value in STRENGTH.items()}
    order = sorted(noisy, key=lambda driver: -noisy[driver])
    retired = order[-1] if round_number % 3 == 0 else None
    return [driver for driver in order if driver != retired] + ([retired] if retired else []), (
        retired
    )


def _schema(columns: tuple[str, ...]) -> pa.Schema:
    timestamp = pa.timestamp("us", tz="UTC")
    fields = [
        pa.field(name, kind)
        for name, kind in (
            ("event_id", pa.string()),
            ("driver_id", pa.string()),
            ("constructor_id", pa.string()),
            ("circuit_id", pa.string()),
            ("prediction_timestamp", timestamp),
            ("feature_timestamp", timestamp),
            ("benchmark_tier", pa.string()),
            ("cutoff_kind", pa.string()),
            ("qualifying_status", pa.string()),
            ("start_type", pa.string()),
            ("pit_lane_start", pa.bool_()),
            ("grid_status", pa.string()),
        )
    ]
    for name in columns:
        kind = (
            pa.bool_()
            if name.endswith("_missing")
            else pa.int32()
            if name in INTEGERS
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


def _rows(columns: tuple[str, ...], *, dnf: bool) -> list[dict]:
    rows = []
    for round_number in range(1, COMPLETED + 1):
        order, retired = _results(round_number)
        cutoff = _start(round_number) - timedelta(hours=20)
        for place, driver in enumerate(order, 1):
            row = {
                name: True if name.endswith("_missing") else 0 if name in INTEGERS else None
                for name in columns
            }
            known = driver != retired
            row.update(
                {
                    "event_id": EventId(SEASON, round_number).partition(),
                    "driver_id": driver,
                    "constructor_id": ROSTER[driver],
                    "circuit_id": _circuit(round_number),
                    "prediction_timestamp": cutoff,
                    "feature_timestamp": cutoff - timedelta(hours=1),
                    "benchmark_tier": "Gold",
                    "cutoff_kind": "post_qualifying",
                    "qualifying_status": "completed",
                    "start_type": "grid",
                    "pit_lane_start": False,
                    "grid_status": "unknown",
                    "qualifying_position": float(place),
                    "qualifying_position_missing": False,
                    "label_position": place if known else None,
                    "label_winner": place == 1,
                    "label_podium": place <= 3 and known,
                    "label_dnf": (driver == retired) if dnf else None,
                    "label_available_at": _start(round_number) + timedelta(hours=6),
                    "label_final_audited": True,
                    "label_audit_reference": "synthetic-final",
                }
            )
            rows.append(row)
    return rows


def _version(directory: Path, rows: list[dict], schema: pa.Schema, extra: dict) -> Path:
    directory.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), directory / "gold.parquet")
    empty = pa.Table.from_pylist([], schema=schema)
    for name in ("silver", "development"):
        pq.write_table(empty, directory / f"{name}.parquet")
    (directory / "coverage.json").write_text("{}", encoding="utf-8")
    manifest = {
        "coverage_sha256": file_sha256(directory / "coverage.json"),
        "datasets": {
            "Gold": {
                "path": "gold.parquet",
                "sha256": file_sha256(directory / "gold.parquet"),
                "rows": len(rows),
                "events": sorted({row["event_id"] for row in rows}),
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


def _ledger(root: Path) -> None:
    audit = root / "data/audit"
    audit.mkdir(parents=True)
    rule = _hashed(
        {
            "season": SEASON,
            "first_round": 1,
            "last_round": COMPLETED,
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
    for round_number in range(1, COMPLETED + 1):
        order, retired = _results(round_number)
        effective = (_start(round_number) + timedelta(hours=5)).isoformat()
        event_id = EventId(SEASON, round_number).partition()
        entries = [
            _hashed(
                {
                    "season": SEASON,
                    "event_id": event_id,
                    "driver_id": driver,
                    "constructor_id": ROSTER[driver],
                    "race_points": 0 if driver == retired else POINTS[place],
                    "sprint_points": 0,
                    "bonus_points": 0,
                    "adjustment_points": 0,
                    "total_points": 0 if driver == retired else POINTS[place],
                    "effective_at": effective,
                    "source_evidence": [SOURCE],
                    "revision_status": "audited",
                }
            )
            for place, driver in enumerate(order)
        ]
        events.append(
            _hashed(
                {
                    "season": SEASON,
                    "event_id": event_id,
                    "completed_at": (_start(round_number) + timedelta(hours=2)).isoformat(),
                    "effective_at": effective,
                    "race_schedule": "standard",
                    "complete": True,
                    "revision_status": "audited",
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


def _session(moment: datetime) -> dict:
    return {"date": moment.date().isoformat(), "time": moment.strftime("%H:%M:%SZ")}


def _schedule() -> dict:
    races = []
    for round_number in range(1, COMPLETED + 4):
        start = _start(round_number)
        race = {
            "season": str(SEASON),
            "round": str(round_number),
            "raceName": "Bahrain Grand Prix in Malaysia"
            if round_number == COMPLETED + 1
            else f"Grand Prix {round_number}",
            "Circuit": {
                "circuitId": _circuit(round_number),
                "circuitName": "Sepang International Circuit"
                if round_number == COMPLETED + 1
                else f"Circuit {round_number}",
            },
            **_session(start),
            "FirstPractice": _session(start - timedelta(days=2)),
            "Qualifying": _session(start - timedelta(days=1)),
        }
        if round_number == COMPLETED + 2:
            race["Sprint"] = _session(start - timedelta(hours=26))
        races.append(race)
    return {"MRData": {"RaceTable": {"season": str(SEASON), "Races": races}}}


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    root = tmp_path
    _ledger(root)
    payload = _schedule()
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    retained = root / "data/raw/historical_schedules"
    retained.mkdir(parents=True)
    (retained / f"season={SEASON}.json").write_bytes(
        _canonical(
            {
                "version": 1,
                "url": "fixture",
                "params": {},
                "retrieved_at_utc": CLOCK.isoformat(),
                "classification": "current_state_only",
                "use": "historical cutoff placement only",
                "payload_sha256": digest,
                "payload": payload,
            }
        )
    )
    observation = "data/raw/prospective_scheduler/observations/schedule-fixture.json"
    (root / observation).parent.mkdir(parents=True)
    (root / observation).write_text(
        json.dumps(
            {
                "version": 1,
                "provider": "jolpica",
                "url": "fixture",
                "role": "schedule",
                "response_captured_at": (CLOCK - timedelta(minutes=5)).isoformat(),
                "payload_sha256": digest,
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
                "schedule_observation": observation,
            }
        ),
        encoding="utf-8",
    )
    races = []
    outcomes = root / "data/outcomes"
    outcomes.mkdir(parents=True)
    for round_number in range(1, COMPLETED + 1):
        path = outcomes / f"{round_number}.parquet"
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
    from f1_ml_predictor.scoring.ledger import load_scoring_ledger

    ledger = load_scoring_ledger(
        root / "data/audit/scoring_rules.json", root / "data/audit/event_points_evidence.json"
    )
    gold = _version(
        root / "data/benchmarks/gold/versions/pending",
        _rows(GOLD_COLUMNS, dnf=False),
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
    dnf = _version(
        root / "data/benchmarks/dnf/versions/pending",
        _rows(BENCHMARK_FEATURE_COLUMNS, dnf=True),
        _schema(BENCHMARK_FEATURE_COLUMNS),
        {"version": 1, "feature_columns": list(BENCHMARK_FEATURE_COLUMNS)},
    )
    return {"root": root, "gold": gold, "dnf": dnf}


def test_pre_weekend_pipeline_is_coherent_ood_aware_and_development_only(workspace: dict) -> None:
    root = workspace["root"]
    result = predict_next_race(
        root,
        simulations=300,
        draws=1024,
        worlds=20,
        orders_per_world=4,
        device="cpu",
        gold_dir=workspace["gold"],
        dnf_dir=workspace["dnf"],
        now=lambda: CLOCK,
        candidates=("logistic_pl", "ridge_pl", "hist"),
    )
    assert result["cutoff_kind"] == "pre_weekend"
    assert result["event_id"] == "season=2026/round=15"
    run = root / result["run_dir"]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["event"]["circuit_id"] == "sepang"
    assert manifest["event"]["circuit_name"] == "Sepang International Circuit"
    assert manifest["ood"]["circuit"]["unseen_in_training"] is True
    assert manifest["ood"]["status"] == "out_of_distribution"
    assert all(manifest["validation_checks"].values())
    assert manifest["development_quality"] in {"pass", "warn", "fail"}
    assert "driver_points_before_race" in manifest["evaluation"]["masked_for_training"]
    race = json.loads((run / "race_distribution.json").read_text(encoding="utf-8"))
    _check_race(race["drivers"])
    assert race["validation_status"] == "development_only"
    drivers = race["drivers"]
    # Clean-race positions are a permutation average with no retirements trailing.
    assert sum(item["clean_expected_position"] for item in drivers) == pytest.approx(
        sum(range(1, len(drivers) + 1))
    )
    favourite = max(drivers, key=lambda item: item["winner_probability"])
    assert favourite["clean_expected_position"] < favourite["expected_position"]
    order = sorted(drivers, key=lambda item: item["predicted_position"])
    assert [item["predicted_position"] for item in order] == list(range(1, len(drivers) + 1))
    assert [item["clean_expected_position"] for item in order] == sorted(
        item["clean_expected_position"] for item in drivers
    )
    championship = json.loads((run / "championship.json").read_text(encoding="utf-8"))
    assert championship["validation_status"] == "development_only"
    sessions = [(item["event_id"], item["session"]) for item in championship["sessions"]]
    assert ("season=2026/round=16", "sprint") in sessions
    assert championship["sessions"][0]["circuit_seen_before"] is False
    assert championship["uncertainty"]["worlds"] == 20
    for places in championship["simulator"]["wdc"]["final_position_probability"].values():
        assert sum(places) == pytest.approx(1)
    table = pq.read_table(run / "predictions.parquet").to_pylist()
    assert {row["validation_status"] for row in table} == {"development_only"}
    assert not any(key.startswith("label_") for key in table[0])
    report = (run / "report.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in report
    assert "Out-of-distribution status" in report
    assert re.search(r"(?<![0-9])0\.0%", report) is None


def test_contract_rows_use_only_published_history_and_weekend_values(workspace: dict) -> None:
    history, _, _ = load_audited_history(
        workspace["root"], gold_dir=workspace["gold"], dnf_dir=workspace["dnf"]
    )
    event = EventId(SEASON, 10)
    first_practice = history.weekends[event].first_practice
    assert first_practice is not None
    rows, reasons = build_rows(
        history,
        "pre_weekend",
        event=event,
        circuit_id=_circuit(10),
        cutoff=first_practice,
        roster=ROSTER,
    )
    assert "qualifying_position" not in rows[0]
    assert all(row["feature_timestamp"] <= first_practice for row in rows)
    a1 = next(row for row in rows if row["driver_id"] == "a1")
    finishes = []
    for round_number in (7, 8, 9):
        order, retired = _results(round_number)
        if retired != "a1":
            finishes.append(order.index("a1") + 1)
    assert a1["recent_finish_mean_3"] == pytest.approx(sum(finishes) / len(finishes))
    assert a1["circuit_seen_before"] == 1
    assert a1["driver_dnf_observations_any_10"] == 9
    # Strength features: a1 is the strongest synthetic driver in the strongest car.
    assert a1["driver_elo_events"] == 9
    assert a1["constructor_elo"] == max(row["constructor_elo"] for row in rows)
    assert a1["driver_teammate_qualifying_h2h_10"] is not None
    assert a1["driver_similar_circuit_delta"] is None
    assert reasons["a1"]["missing_reasons"]["driver_similar_circuit_delta"] == (
        "circuit_unprofiled"
    )
    weekend = {
        driver: {
            "practice_position": 1.0,
            "practice_position_available_at": first_practice + timedelta(hours=30),
        }
        for driver in ROSTER
    }
    practice, practice_reasons = build_rows(
        history,
        "post_practice",
        event=event,
        circuit_id=_circuit(10),
        cutoff=first_practice + timedelta(hours=24),
        roster=ROSTER,
        weekend=weekend,
    )
    assert all(row["practice_position"] is None for row in practice)
    assert (
        practice_reasons["a1"]["missing_reasons"]["practice_position"] == "published_after_cutoff"
    )
    early = _start(9) + timedelta(hours=1)
    rows, _ = build_rows(
        history, "pre_weekend", event=event, circuit_id=_circuit(10), cutoff=early, roster=ROSTER
    )
    assert all(row["recent_finish_mean_3"] is None for row in rows)
    assert set(feature_columns("pre_weekend")) < set(feature_columns("pre_race"))


def test_schedule_circuit_comes_from_identifier_not_title() -> None:
    parsed = weekends(_schedule())
    sepang = parsed[EventId(SEASON, COMPLETED + 1)]
    assert sepang.race_name == "Bahrain Grand Prix in Malaysia"
    assert sepang.circuit_id == "sepang"
    assert sepang.circuit_id != "bahrain"
    assert parsed[EventId(SEASON, COMPLETED + 2)].sprint is not None


def test_joint_layer_is_coherent_and_calibration_mixture_is_exact() -> None:
    rng = np.random.default_rng(1)
    utility = np.asarray([3.0, 1.0, 0.0, -1.0])
    dnf = np.full(4, 0.1)
    order, retired = sample_orders(utility, dnf, temperature=1.0, draws=4000, rng=rng)
    model = marginals(order, retired)
    prior = marginals(*sample_orders(np.zeros(4), dnf, temperature=1.0, draws=4000, rng=rng))
    mixed = mix(model, prior, 0.2)
    assert mixed["winner"].sum() == pytest.approx(1)
    assert np.allclose(mixed["finish"].sum(axis=0), 1) and np.allclose(
        mixed["finish"].sum(axis=1), 1
    )
    assert mixed["winner"].min() > 0
    blended = average([model, prior])
    assert blended["podium"].sum() == pytest.approx(3)
    sharp = sharpness(model["winner"], model["podium"])
    assert 1 <= sharp["effective_win_contenders"] <= 4
    flat = marginals(*sample_orders(utility, dnf, temperature=4.0, draws=4000, rng=rng))
    assert (
        sharpness(flat["winner"], flat["podium"])["effective_win_contenders"]
        > sharp["effective_win_contenders"]
    )
    components = [(utility, 1.0, 0.0), (utility[::-1].copy(), 1.0, 0.0)]
    order, retired = sample_mixture(components, dnf, draws=2000, rng=rng)
    assert sorted(order[0].tolist()) == [0, 1, 2, 3]
    error, curve = expected_calibration_error([0.1, 0.1, 0.9, 0.9], [False, False, True, True])
    assert error == pytest.approx(0.1) and len(curve) == 2


def test_candidates_recover_strength_and_logistic_pl_matches_logistic_baseline() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for race in range(30):
        x = rng.normal(size=8)
        order = np.argsort(-(1.5 * x + rng.gumbel(size=8)))
        for place, index in enumerate(order, 1):
            rows.append(
                {
                    "event_id": f"e{race}",
                    "driver_id": f"d{index}",
                    "f": float(x[index]),
                    "label_position": place if place <= 6 else None,
                    "label_winner": place == 1,
                }
            )
    model = PlackettLuceRegression(penalty=0.01).fit(rows, ("f",))
    assert model.coef_[0] > 1.0
    logistic = fit_strength("logistic_pl", rows, ("f",), seed=0)
    race = [row for row in rows if row["event_id"] == "e0"]
    utility = logistic.utility(race)
    winner = np.exp(utility) / np.exp(utility).sum()
    raw = logistic.model.predict_proba(np.asarray([[row["f"]] for row in race]))[:, 1]
    assert winner == pytest.approx(raw / raw.sum())


def test_training_gate_respects_label_and_dnf_clocks() -> None:
    cutoff = datetime(2026, 1, 10, tzinfo=UTC)
    rows = [
        {
            "event_id": "early",
            "prediction_timestamp": cutoff - timedelta(days=5),
            "label_available_at": cutoff - timedelta(days=4),
            "label_dnf": True,
            "label_dnf_available_at": cutoff + timedelta(days=1),
        },
        {
            "event_id": "late",
            "prediction_timestamp": cutoff - timedelta(days=1),
            "label_available_at": cutoff + timedelta(hours=1),
            "label_dnf": False,
            "label_dnf_available_at": cutoff - timedelta(hours=1),
        },
    ]
    selected, events = training_rows(rows, cutoff)
    assert events == 1
    assert [row["event_id"] for row in selected] == ["early"]
    assert selected[0]["label_dnf"] is None


def test_icc_detects_persistent_groups() -> None:
    rng = np.random.default_rng(3)
    independent = [(f"g{group}", float(rng.normal())) for group in range(40) for _ in range(8)]
    persistent = [
        (f"g{group}", offset + float(rng.normal()))
        for group in range(40)
        for offset in [float(rng.normal(0, 2))]
        for _ in range(8)
    ]
    assert abs(_icc(independent)) < 0.1
    assert _icc(persistent) > 0.5


def test_ood_flags_unseen_circuit_and_range() -> None:
    training = [
        {
            "event_id": f"e{race}",
            "driver_id": f"d{driver}",
            "circuit_id": "monza",
            "x": float(driver),
            "x_missing": False,
        }
        for race in range(5)
        for driver in range(4)
    ]
    live = [
        {
            "event_id": "live",
            "driver_id": "d0",
            "circuit_id": "sepang",
            "x": 40.0,
            "x_missing": False,
        },
        {
            "event_id": "live",
            "driver_id": "d1",
            "circuit_id": "sepang",
            "x": None,
            "x_missing": True,
        },
    ]
    report = ood_report(
        training, live, ("x", "x_missing"), circuit_id="sepang", circuit_name="Sepang"
    )
    assert report["circuit"]["unseen_in_training"] is True
    assert report["feature_range_violations"][0]["driver_id"] == "d0"
    assert report["novel_missing_pattern_drivers"] == ["d1"]
    assert "d0" in report["distance"]["beyond_99th_percentile"]


def test_scenario_groups_share_one_world_across_events() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    standings = CurrentStandings(2026, cutoff, {"a": 0, "b": 0}, {"red": 0, "blue": 0}, "a" * 64)
    rules = PointsRules(2026, "race", (10, 0), "fixture", "synthetic")

    def event(round_number: int, groups: tuple[int, ...] | None) -> EventSimulation:
        return EventSimulation(
            event_id=EventId(2026, round_number),
            scheduled_at=cutoff + timedelta(days=round_number),
            available_at=cutoff,
            driver_constructors={"a": "red", "b": "blue"},
            sampled_orders=(("a", "b"), ("b", "a")),
            points_eligible_samples=(("a", "b"), ("a", "b")),
            rules=rules,
            source_hash="b" * 64,
            sample_groups=groups,
        )

    grouped = simulate_championship(
        standings,
        [event(1, (0, 1)), event(2, (0, 1))],
        prediction_timestamp=cutoff,
        simulations=2000,
        seed=4,
    )
    assert grouped.wdc.unresolved_tie_probability == 0
    assert grouped.wdc.title_probability["a"] == pytest.approx(0.5, abs=0.05)
    independent = simulate_championship(
        standings,
        [event(1, None), event(2, None)],
        prediction_timestamp=cutoff,
        simulations=2000,
        seed=4,
    )
    assert independent.wdc.unresolved_tie_probability == pytest.approx(0.5, abs=0.05)
    with pytest.raises(ValueError, match="every remaining event or none"):
        simulate_championship(
            standings, [event(1, (0, 1)), event(2, None)], prediction_timestamp=cutoff
        )
    with pytest.raises(ValueError, match="share the same sample groups"):
        simulate_championship(
            standings, [event(1, (0, 1)), event(2, (0, 2))], prediction_timestamp=cutoff
        )
    assert "sample_groups" not in event(1, None).to_dict()


def test_binary_dnf_audit_agreement_rules_are_unchanged() -> None:
    fia = {"dnf_category": "unknown", "raw_status": "DNF", "classified": False}
    category, reason = audit_binary_status(
        fia, {"status": "Retired"}, {"dnf": True, "dns": False, "dsq": False}
    )
    assert reason == "three_source_retired_agreement" and category is not None
    assert audit_binary_status(fia, None, None)[1] == "source_missing"
