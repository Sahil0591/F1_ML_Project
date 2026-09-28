import json
from datetime import UTC, datetime, timedelta

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_backtest import benchmark_table

from f1_ml_predictor.benchmarks.builder import (
    BENCHMARK_FEATURE_COLUMNS,
    build_benchmarks,
    file_sha256,
)
from f1_ml_predictor.models import probabilistic
from f1_ml_predictor.models.backtest import rolling_folds
from f1_ml_predictor.models.distributions import race_distribution
from f1_ml_predictor.models.hardware import choose_device, library_versions
from f1_ml_predictor.models.probabilistic import (
    calibration_split,
    fit_race_model,
    run_probabilistic_backtest,
    run_probabilistic_files,
    select_candidate,
)
from f1_ml_predictor.trust.evidence import BenchmarkTier


def development_table(count=6):
    rows = benchmark_table(count).to_pylist()
    for row in rows:
        row["benchmark_tier"] = BenchmarkTier.DEVELOPMENT.value
    return pa.Table.from_pylist(rows)


def test_joint_distribution_is_doubly_stochastic_and_links_all_marginals():
    result = race_distribution([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 0.2], seed=18)
    matrix = np.asarray([row["finish_distribution"] for row in result])
    assert matrix.sum(axis=0) == pytest.approx(np.ones(4))
    assert matrix.sum(axis=1) == pytest.approx(np.ones(4))
    assert sum(row["winner_probability"] for row in result) == pytest.approx(1.0)
    assert sum(row["podium_probability"] for row in result) == pytest.approx(3.0)
    assert result[2]["winner_probability"] == 0.0
    assert result[2]["dnf_probability"] == 1.0
    for row in result:
        assert row["winner_probability"] == row["finish_distribution"][0]
        assert row["podium_probability"] == pytest.approx(sum(row["finish_distribution"][:3]))
    for count in (1, 2):
        small = race_distribution([1.0] * count, [1.0] * count)
        assert all(row["podium_probability"] == 1.0 for row in small)
        assert sum(row["winner_probability"] for row in small) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "scores,dnf,settings",
    [
        ([], [], {}),
        ([float("nan")], [0.0], {}),
        ([1.0], [float("inf")], {}),
        ([1.0], [-0.1], {}),
        ([1.0], [0.0], {"temperature": 0}),
        ([1.0], [0.0], {"draws": True}),
        ([1.0], [0.0], {"draws": 1}),
    ],
)
def test_distribution_rejects_invalid_inputs(scores, dnf, settings):
    with pytest.raises(ValueError):
        race_distribution(scores, dnf, **settings)


def test_heldout_calibration_uses_only_earlier_complete_labelled_events():
    rows = development_table().to_pylist()
    folds, _ = rolling_folds(rows, min_train_events=3)
    fold = folds[0]
    train, calibration = calibration_split(rows, fold, 2)
    assert set(train).isdisjoint(calibration)
    assert set(train + calibration).isdisjoint(fold.test_indices)
    calibration_cutoff = rows[calibration[0]]["prediction_timestamp"]
    assert all(rows[i]["label_available_at"] <= calibration_cutoff for i in train)
    assert all(rows[i]["label_available_at"] <= fold.prediction_timestamp for i in calibration)
    # One delayed driver excludes its entire race both in the outer and inner folds.
    rows[0]["label_available_at"] = fold.prediction_timestamp + timedelta(days=1)
    new_folds, _ = rolling_folds(rows, min_train_events=2)
    for candidate in new_folds:
        event_indices = {i for i, row in enumerate(rows) if row["event_id"] == rows[0]["event_id"]}
        present = event_indices.intersection(candidate.train_indices)
        assert not present or present == event_indices
    assert calibration_split(rows, fold, 2) == ((), ())


def test_predictions_ignore_test_labels_and_are_order_invariant():
    rows = development_table().to_pylist()
    folds, _ = rolling_folds(rows, min_train_events=3)
    model = fit_race_model(rows, folds[0], "hist", seed=31)
    assert model is not None
    test_rows = [rows[i] for i in folds[0].test_indices]
    original = model.predict(test_rows)
    for row in test_rows:
        row["label_winner"] = not row["label_winner"]
        row["label_position"] = 99
        row["label_dnf"] = True
    assert model.predict(list(reversed(test_rows))) == original
    with pytest.raises(ValueError, match="used to fit"):
        model.predict([rows[i] for i in model.metadata["calibration_indices"]])
    changed_cutoff = [{**row, "cutoff_kind": "pre_race"} for row in test_rows]
    with pytest.raises(ValueError, match="cohorts differ"):
        model.predict(changed_cutoff)


def test_cpu_backtest_is_deterministic_and_records_paired_baseline_regressions():
    table = development_table()
    first = run_probabilistic_backtest(table, BenchmarkTier.DEVELOPMENT, backends=("hist",))
    second = run_probabilistic_backtest(
        table.take(list(reversed(range(table.num_rows)))),
        BenchmarkTier.DEVELOPMENT,
        backends=("hist",),
    )
    assert first["status"] == "evaluated"
    assert first["primary_accuracy_claim_allowed"] is False
    assert first["predictions"] == second["predictions"]
    assert first["comparisons"] == second["comparisons"]
    comparison = first["comparisons"]["post_qualifying"]["hist"]
    assert comparison["paired_cohorts"] == 3
    assert comparison["metrics"]["finishing_position"]["ranked_probability_score"] >= 0
    assert comparison["regressions"]
    assert first["selection"]["post_qualifying"]["status"] == "deferred"
    assert all(row["training_device"] == "cpu" for row in first["predictions"])


def test_missing_dnf_labels_keep_prior_and_do_not_invent_metrics():
    rows = development_table(4).to_pylist()
    for row in rows:
        row["label_dnf"] = None
    result = run_probabilistic_backtest(
        pa.Table.from_pylist(rows),
        BenchmarkTier.DEVELOPMENT,
        backends=("hist",),
    )
    assert result["folds"][0]["dnf_training_status"] == "smoothed_prior"
    assert result["folds"][0]["dnf_training_labels"] == 0
    assert result["comparisons"]["post_qualifying"]["hist"]["metrics"]["dnf"] == {
        "status": "insufficient_data",
        "n": 0,
    }


def test_empty_gold_files_are_honest_and_manifest_tampering_is_rejected(tmp_path):
    benchmark_dir = tmp_path / "benchmarks"
    build_benchmarks(tmp_path, benchmark_dir)
    report = tmp_path / "models" / "comparison.json"
    prediction = tmp_path / "predictions.parquet"
    result = run_probabilistic_files(
        benchmark_dir / "gold.parquet",
        BenchmarkTier.GOLD,
        report,
        prediction,
        backends=("hist",),
    )
    assert result["status"] == "insufficient_data"
    assert result["prediction_rows"] == 0
    assert result["primary_accuracy_claim_allowed"] is False
    assert result["comparisons"] == {}
    assert pq.read_table(prediction).num_rows == 0
    assert json.loads(report.read_text())["selection"]["selected_backend"] is None
    with (benchmark_dir / "gold.parquet").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="manifest"):
        run_probabilistic_files(
            benchmark_dir / "gold.parquet",
            BenchmarkTier.GOLD,
            report,
            prediction,
        )


def test_fitted_artifact_roundtrip_predicts_without_labels(tmp_path):
    table = development_table(4)
    result = run_probabilistic_backtest(
        table,
        BenchmarkTier.DEVELOPMENT,
        backends=("hist",),
        model_dir=tmp_path,
    )
    model = joblib.load(tmp_path / result["folds"][0]["model_artifact"])
    rows = [row for row in table.to_pylist() if row["event_id"] == result["folds"][0]["event_id"]]
    rows = [
        {key: value for key, value in row.items() if not key.startswith("label_")} for row in rows
    ]
    predictions = model.predict(rows)
    actual = [
        {key: value for key, value in row.items() if not key.startswith("label_")}
        for row in result["predictions"]
    ]
    assert predictions == actual


def test_nonempty_file_flow_binds_predictions_and_fitted_model_hashes(tmp_path):
    table = development_table(4)
    dataset = tmp_path / "development.parquet"
    pq.write_table(table, dataset)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(json.dumps({"fixture": True}))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "feature_columns": list(BENCHMARK_FEATURE_COLUMNS),
                "coverage_sha256": file_sha256(coverage),
                "datasets": {
                    "Development": {
                        "path": dataset.name,
                        "sha256": file_sha256(dataset),
                        "rows": table.num_rows,
                    }
                },
            }
        )
    )
    prediction = tmp_path / "predictions.parquet"
    result = run_probabilistic_files(
        dataset,
        BenchmarkTier.DEVELOPMENT,
        tmp_path / "models" / "report.json",
        prediction,
        backends=("hist",),
    )
    assert result["prediction_rows"] == 4
    assert result["prediction_sha256"] == file_sha256(prediction)
    assert result["benchmark_dataset_sha256"] == file_sha256(dataset)
    fold = result["folds"][0]
    assert fold["model_sha256"] == file_sha256(
        tmp_path / "models" / "fitted" / fold["model_artifact"]
    )
    assert result["source_sha256"]["probabilistic.py"]


def test_device_policy_requires_measured_benefit_and_cpu_fallback(monkeypatch):
    hardware = {
        "captured_at": datetime.now(UTC).isoformat(),
        "libraries": library_versions(),
        "workload": {"rows": 8000},
        "backends": {
            "xgboost": {
                "devices": {
                    "cpu": {"status": "usable", "median_seconds": 2},
                    "cuda": {"status": "usable", "median_seconds": 1},
                }
            },
        },
    }
    assert choose_device("xgboost", 100, "auto", hardware)[0] == "cpu"
    assert choose_device("xgboost", 8000, "auto", hardware)[0] == "cuda"
    assert choose_device("xgboost", 8000, "cuda", None)[0] == "cpu"
    hardware["captured_at"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    assert choose_device("xgboost", 8000, "auto", hardware)[0] == "cpu"
    original = probabilistic.estimator
    calls = []

    def fail_gpu(backend, task, seed, device):
        calls.append(device)
        if device != "cpu":
            raise RuntimeError("GPU memory exhausted")
        return original("hist", task, seed)

    monkeypatch.setattr(probabilistic, "estimator", fail_gpu)
    monkeypatch.setattr(probabilistic, "choose_device", lambda *args: ("cuda", "probe usable"))
    rows = development_table(4).to_pylist()
    folds, _ = rolling_folds(rows, min_train_events=3)
    model = fit_race_model(rows, folds[0], "hist", device="cuda")
    assert model.device == "cpu"
    assert "CPU fallback" in model.metadata["device_reason"]
    assert calls[0] == "cuda" and "cpu" in calls


def test_optional_libraries_are_not_required(monkeypatch):
    versions = library_versions()
    versions.update({name: None for name in ("xgboost", "lightgbm", "catboost")})
    monkeypatch.setattr(probabilistic, "library_versions", lambda: versions)
    result = run_probabilistic_backtest(development_table(4), BenchmarkTier.DEVELOPMENT)
    assert result["backends"]["xgboost"]["status"] == "not_installed"
    assert {row["backend"] for row in result["predictions"]} == {"hist"}


def test_selection_requires_five_events_and_no_regressions():
    metrics = {
        task: {"status": "evaluated", "n": 5, "log_loss": 0.3}
        for task in (
            "winner",
            "podium",
            "dnf",
            "finishing_position",
        )
    }
    comparison = {"hist": {"metrics": metrics, "regressions": []}}
    assert select_candidate(comparison)["selected_backend"] == "hist"
    comparison["hist"]["regressions"] = [{"task": "dnf"}]
    assert select_candidate(comparison)["selected_backend"] is None


def test_cutoff_kinds_are_evaluated_and_selected_separately():
    rows = development_table(4).to_pylist()
    duplicate = [{**row, "cutoff_kind": "pre_race", "grid_position": 10.0} for row in rows]
    result = run_probabilistic_backtest(
        pa.Table.from_pylist(rows + duplicate),
        BenchmarkTier.DEVELOPMENT,
        backends=("hist",),
    )
    assert set(result["comparisons"]) == {"post_qualifying", "pre_race"}
    assert set(result["selection"]) == {"post_qualifying", "pre_race"}
    ordered = sorted(
        rows + duplicate,
        key=lambda row: (
            row["event_id"],
            row["prediction_timestamp"],
            row["cutoff_kind"],
            row["driver_id"],
        ),
    )
    for report in result["folds"]:
        assert all(
            ordered[i]["cutoff_kind"] == report["cutoff_kind"]
            for i in (report["fit_indices"] + report["calibration_indices"])
        )
        assert report["calibration_label_availability_max"] <= report["prediction_timestamp"]


def test_future_features_duplicate_rosters_and_tier_mismatches_are_rejected():
    rows = development_table(4).to_pylist()
    with pytest.raises(ValueError, match="duplicate"):
        run_probabilistic_backtest(pa.Table.from_pylist(rows + rows[:1]), BenchmarkTier.DEVELOPMENT)
    rows[0]["feature_timestamp"] = rows[0]["prediction_timestamp"] + timedelta(seconds=1)
    with pytest.raises(ValueError):
        run_probabilistic_backtest(pa.Table.from_pylist(rows), BenchmarkTier.DEVELOPMENT)
