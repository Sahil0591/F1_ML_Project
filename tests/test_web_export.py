"""Offline fixtures for the web export adapter, plus a check against real local runs."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.prediction.web_export import (
    RUNS,
    SCHEMA_VERSION,
    ExportError,
    build_export,
    write_export,
)

DRIVERS = [("alpha_one", "alpha"), ("alpha_two", "alpha"), ("bravo_one", "bravo")]
NOW = datetime(2026, 10, 4, 12, tzinfo=UTC)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
    return file_sha256(path)


def _schedule(root: Path) -> str:
    payload = {
        "MRData": {
            "RaceTable": {
                "season": "2026",
                "Races": [
                    {
                        "season": "2026",
                        "round": str(number),
                        "raceName": f"Race {number}",
                        "date": f"2026-10-{number:02d}",
                        "time": "07:00:00Z",
                        "Circuit": {
                            "circuitId": f"circuit_{number}",
                            "circuitName": f"Circuit {number}",
                            "Location": {"locality": f"Town {number}", "country": "Testland"},
                        },
                    }
                    for number in (1, 2)
                ],
            }
        }
    }
    relative = "data/raw/prospective_scheduler/observations/schedule-test.json"
    _write(
        root / relative,
        {
            "role": "schedule",
            "provider": "jolpica",
            "payload": payload,
            "payload_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
            "response_captured_at": "2026-09-30T00:00:00+00:00",
            "url": "https://api.jolpi.ca/ergast/f1/2026/",
        },
    )
    return relative


def _finish(rank: int) -> list[float]:
    rows = {0: [0.6, 0.3, 0.1], 1: [0.3, 0.5, 0.2], 2: [0.1, 0.2, 0.7]}
    return rows[rank]


def make_run(
    root: Path,
    *,
    run_id: str = "run1",
    cutoff: str = "pre_weekend",
    created_at: str = "2026-10-01T10:00:00+00:00",
    with_order: bool = True,
    duplicate_order: bool = False,
) -> Path:
    directory = root / RUNS / "season=2026" / "round=02" / cutoff / run_id
    directory.mkdir(parents=True)
    identity = {
        "validation_status": "development_only",
        "methodology": "cutoff-specific-v3",
        "model_run_id": run_id,
        "cutoff_kind": cutoff,
        "prediction_timestamp_utc": "2026-10-01T09:00:00+00:00",
    }
    race = []
    for rank, (driver, team) in enumerate(DRIVERS):
        finish = _finish(rank)
        item: dict[str, Any] = {
            "driver_id": driver,
            "constructor_id": team,
            "winner_probability": finish[0],
            "podium_probability": sum(finish),
            "dnf_probability": 0.1,
            "dnf_model_probability": 0.125,
            "finish_distribution": finish,
            "expected_position": 1.5 + rank * 0.5,
            "most_likely_position": rank + 1,
            "position_interval_80": [1, 3],
            "winner_draws": int(finish[0] * 1000),
        }
        if with_order:
            item["clean_expected_position"] = 1.2 + rank * 0.6
            item["clean_most_likely_position"] = rank + 1
            item["predicted_position"] = 1 if duplicate_order else rank + 1
        race.append(item)
    rows = [
        {**identity, **{k: v for k, v in i.items() if k != "position_interval_80"}} for i in race
    ]
    pq.write_table(pa.Table.from_pylist(rows), directory / "predictions.parquet")
    race_sha = _write(
        directory / "race_distribution.json",
        {
            **identity,
            "draws": 1000,
            "seed": 42,
            "monte_carlo_resolution": 0.001,
            "primary_model": "ensemble_all",
            "primary_members": ["logistic_pl"],
            "calibration": {"logistic_pl": {"temperature": 1.4, "shrink": 0.0, "source": "shared"}},
            "drivers": race,
        },
    )
    places_driver = {"alpha_one": [0.7, 0.2, 0.1], "alpha_two": [0.2, 0.7, 0.1]}
    places_driver["bravo_one"] = [0.1, 0.1, 0.8]
    title = {"alpha_one": 0.7, "alpha_two": 0.2, "bravo_one": 0.1}
    simulator = {
        "season": 2026,
        "status": "engineering_only",
        "simulations": 1000,
        "limitations": ["fixture"],
        "wdc": {
            "title_probability": title,
            "tied_for_title_probability": {name: 0.0 for name in title},
            "unresolved_tie_probability": 0.0,
            "mean_final_points": {"alpha_one": 60.5, "alpha_two": 45.25, "bravo_one": 30.0},
            "final_position_probability": places_driver,
        },
        "wcc": {
            "title_probability": {"alpha": 0.9, "bravo": 0.1},
            "tied_for_title_probability": {"alpha": 0.0, "bravo": 0.0},
            "unresolved_tie_probability": 0.0,
            "mean_final_points": {"alpha": 105.75, "bravo": 30.0},
            "final_position_probability": {"alpha": [0.9, 0.1], "bravo": [0.1, 0.9]},
        },
    }
    championship_sha = _write(
        directory / "championship.json",
        {
            **identity,
            "validated_forecast": False,
            "race_distribution_sha256": race_sha,
            "starting_points": {
                "available_at": "2026-09-30T00:00:00+00:00",
                "driver_points": {"alpha_one": 25.0, "alpha_two": 18.0, "bravo_one": 15.0},
                "constructor_points": {"alpha": 43.0, "bravo": 15.0},
            },
            "standings_source": {"notes": []},
            "sessions": [
                {
                    "event_id": "season=2026/round=02",
                    "race_name": "Race 2",
                    "session": "race",
                    "scheduled_at": "2026-10-02T07:00:00+00:00",
                }
            ],
            "uncertainty": {
                "worlds": 100,
                "orders_per_world": 16,
                "monte_carlo_standard_error_max": 0.05,
                "sensitivity": {
                    "fixed_strength_no_persistent_uncertainty": {
                        "wdc": {"alpha_one": 0.8, "alpha_two": 0.15, "bravo_one": 0.05},
                        "wcc": {"alpha": 0.95, "bravo": 0.05},
                    }
                },
            },
            "assumptions": ["fixture assumption"],
            "simulator": simulator,
        },
    )
    _write(
        directory / "manifest.json",
        {
            **identity,
            "warning": "DEVELOPMENT ONLY.",
            "validated_forecast": False,
            "created_at": created_at,
            "git_commit": "65c35d642585f8e566324a66a063db785f7cae63",
            "notes": [],
            "event": {
                "event_id": "season=2026/round=02",
                "race_name": "Race 2",
                "circuit_id": "circuit_2",
                "circuit_name": "Circuit 2",
                "race_start": "2026-10-02T07:00:00+00:00",
                "first_practice": None,
                "qualifying_start": None,
                "sprint_weekend": False,
            },
            "schedule_source": {"path": _schedule(root)},
            "dataset": {
                "gold": "dataset-abc",
                "gold_manifest_sha256": "a" * 64,
                "dnf": "dataset-def",
                "dnf_known_labels": 10,
            },
            "protocol": {"version": "gold-cutoff-specific-v3", "sha256": "b" * 64},
            "feature_contract": "cutoff-contracts-v2",
            "evaluation": {
                "live_masked_evaluation": "models/missing/evaluation.json",
                "primary": "ensemble_all",
                "experimental_model": "lightgbm",
                "masked_for_training": [],
                "dnf": {"model": "prior", "candidates": {}},
            },
            "execution": {"devices": ["cpu"]},
            "sharpness": {"live": {"maximum_win_probability": 0.6}},
            "ood": {
                "status": "out_of_distribution",
                "reasons": ["circuit circuit_2 is absent"],
                "circuit": {"unseen_in_training": True, "training_circuits": 5},
            },
            "development_quality": "pass",
            "validation_checks": {"race_probabilities_coherent": True},
            "artifacts": {
                "predictions": {"sha256": file_sha256(directory / "predictions.parquet")},
                "race_distribution": {"sha256": race_sha},
                "championship": {"sha256": championship_sha},
            },
        },
    )
    return directory


def test_export_copies_race_and_title_values_exactly(tmp_path: Path) -> None:
    directory = make_run(tmp_path)
    outputs = build_export(tmp_path, now=NOW)
    snapshot = outputs["2026/round-02/pre_weekend.json"]
    assert snapshot["schema_version"] == SCHEMA_VERSION
    assert snapshot["identity"]["development_only"] is True
    assert snapshot["identity"]["generated_after_race_start"] is False
    assert snapshot["event"]["locality"] == "Town 2"
    race = json.loads((directory / "race_distribution.json").read_text(encoding="utf-8"))
    source = {item["driver_id"]: item for item in race["drivers"]}
    exported = snapshot["race"]["drivers"]
    assert [item["predicted_position"] for item in exported] == [1, 2, 3]
    for item in exported:
        for key, value in item.items():
            if key in source[item["driver_id"]]:
                assert value == source[item["driver_id"]][key], key
    championship = json.loads((directory / "championship.json").read_text(encoding="utf-8"))
    for kind in ("wdc", "wcc"):
        simulator = championship["simulator"][kind]
        for entry in snapshot["championship"][kind]["entries"]:
            assert entry["title_probability"] == simulator["title_probability"][entry["id"]]
            assert entry["expected_final_points"] == simulator["mean_final_points"][entry["id"]]
            assert (
                entry["final_position_probability"]
                == simulator["final_position_probability"][entry["id"]]
            )
    wdc = snapshot["championship"]["wdc"]["entries"]
    assert [entry["projected_position"] for entry in wdc] == [1, 2, 3]
    assert wdc[0]["top3_probability"] == pytest.approx(1.0)
    assert wdc[0]["most_likely_position"] == 1
    assert wdc[0]["constructor_id"] == "alpha"
    assert snapshot["actual_result"] is None


def test_index_lists_every_cutoff_and_never_invents_missing_ones(tmp_path: Path) -> None:
    make_run(tmp_path)
    index = build_export(tmp_path, now=NOW)["index.json"]
    race = index["seasons"][0]["races"][0]
    assert [entry["cutoff"] for entry in race["cutoffs"]] == [
        "pre_weekend",
        "post_practice",
        "post_sprint_qualifying",
        "post_qualifying",
        "pre_race",
    ]
    sprint = next(entry for entry in race["cutoffs"] if entry["session"] == "sprint")
    assert sprint["cutoff"] == "post_sprint_qualifying" and sprint["available"] is False
    assert [entry["available"] for entry in race["cutoffs"]] == [True, False, False, False, False]
    assert race["cutoffs"][1]["path"] is None
    assert index["latest"]["path"] == "2026/round-02/pre_weekend.json"
    assert race["actual_result_available"] is False


def test_latest_rerun_supersedes_earlier_run_at_the_same_cutoff(tmp_path: Path) -> None:
    make_run(tmp_path, run_id="old", created_at="2026-10-01T10:00:00+00:00", with_order=False)
    make_run(tmp_path, run_id="new", created_at="2026-10-03T10:00:00+00:00")
    outputs = build_export(tmp_path, now=NOW)
    snapshot = outputs["2026/round-02/pre_weekend.json"]
    assert snapshot["identity"]["run_id"] == "new"
    assert snapshot["identity"]["generated_after_race_start"] is True
    entry = outputs["index.json"]["seasons"][0]["races"][0]["cutoffs"][0]
    assert entry["superseded_run_ids"] == ["old"]


def test_runs_without_the_predicted_order_are_marked_not_reconstructed(tmp_path: Path) -> None:
    make_run(tmp_path, with_order=False)
    snapshot = build_export(tmp_path, now=NOW)["2026/round-02/pre_weekend.json"]
    assert snapshot["race"]["predicted_order_available"] is False
    assert all(item["predicted_position"] is None for item in snapshot["race"]["drivers"])


def test_tampered_or_invalid_runs_are_excluded(tmp_path: Path) -> None:
    good = make_run(tmp_path, run_id="good")
    tampered = make_run(tmp_path, run_id="tampered", cutoff="post_qualifying")
    duplicate = make_run(tmp_path, run_id="dup", cutoff="pre_race", duplicate_order=True)
    with (tampered / "championship.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    legacy = tmp_path / RUNS / "season=2026" / "round=02" / "pre_qualifying" / "legacy"
    _write(legacy / "manifest.json", {"validation_status": "development_only"})
    index = build_export(tmp_path, now=NOW)["index.json"]
    reasons = {Path(item["path"]).name: item["reason"] for item in index["excluded_runs"]}
    assert "hash mismatch" in reasons["tampered"]
    assert "permutation" in reasons["dup"]
    assert "legacy" in reasons["legacy"]
    assert good.name not in reasons
    assert duplicate.name in reasons


def test_actual_result_comparison_uses_ingested_classification(tmp_path: Path) -> None:
    make_run(tmp_path)
    results = tmp_path / "data/normalized/season=2026/round=02/results.parquet"
    results.parent.mkdir(parents=True)
    rows = [
        ("alpha_two", 1, "1", "Finished", 25.0),
        ("alpha_one", 2, "2", "Finished", 18.0),
        ("bravo_one", 3, "R", "Engine", 0.0),
    ]
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "event_id": "season=2026/round=02",
                    "driver_id": driver,
                    "constructor_id": dict(DRIVERS)[driver],
                    "position": position,
                    "position_text": text,
                    "grid": position,
                    "laps": 50,
                    "points": points,
                    "status": status,
                    "available_at": None,
                }
                for driver, position, text, status, points in rows
            ]
        ),
        results,
    )
    snapshot = build_export(tmp_path, now=NOW)["2026/round-02/pre_weekend.json"]
    actual = snapshot["actual_result"]
    assert actual is not None
    winner = actual["classification"][0]
    assert winner["driver_id"] == "alpha_two"
    assert winner["predicted_position"] == 2
    assert winner["position_delta"] == 1
    assert actual["summary"]["winner_probability"] == 0.3
    assert actual["summary"]["predicted_winner_id"] == "alpha_one"
    assert actual["summary"]["retirements"] == [
        {"driver_id": "bravo_one", "status": "Engine", "dnf_model_probability": 0.125}
    ]
    assert actual["summary"]["predicted_position_mae"] == 1.0
    assert actual["summary"]["classified_finishers"] == 2


def test_write_export_is_deterministic_and_refuses_foreign_directories(tmp_path: Path) -> None:
    make_run(tmp_path)
    output = tmp_path / "web/public/data"
    write_export(tmp_path, output, now=NOW)
    first = {p.relative_to(output): p.read_bytes() for p in output.rglob("*.json")}
    write_export(tmp_path, output, now=NOW)
    second = {p.relative_to(output): p.read_bytes() for p in output.rglob("*.json")}
    assert first == second
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(ExportError):
        write_export(tmp_path, foreign, now=NOW)
    assert (foreign / "notes.txt").exists()


def test_no_supported_runs_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ExportError):
        build_export(tmp_path, now=NOW)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not any((ROOT / RUNS).glob("season=*/round=*/*/*/manifest.json")),
    reason="no local development prediction runs",
)
def test_real_runs_export_with_exact_championship_values() -> None:
    outputs = build_export(ROOT)
    index = outputs["index.json"]
    for season in index["seasons"]:
        for race in season["races"]:
            for entry in race["cutoffs"]:
                if not entry["available"]:
                    continue
                snapshot = outputs[entry["path"]]
                source = ROOT / snapshot["identity"]["source_directory"]
                championship = json.loads((source / "championship.json").read_text("utf-8"))
                for kind in ("wdc", "wcc"):
                    simulator = championship["simulator"][kind]
                    entries = snapshot["championship"][kind]["entries"]
                    assert {item["id"] for item in entries} == set(simulator["title_probability"])
                    for item in entries:
                        assert (
                            item["title_probability"] == simulator["title_probability"][item["id"]]
                        )
                        assert (
                            item["expected_final_points"]
                            == simulator["mean_final_points"][item["id"]]
                        )
                drivers = snapshot["race"]["drivers"]
                if snapshot["race"]["predicted_order_available"]:
                    positions = sorted(item["predicted_position"] for item in drivers)
                    assert positions == list(range(1, len(drivers) + 1))
