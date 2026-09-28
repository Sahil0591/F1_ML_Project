import pytest
from test_probabilistic import development_table

from f1_ml_predictor.models.backtest import rolling_folds
from f1_ml_predictor.models.probabilistic import fit_race_model, select_candidate
from f1_ml_predictor.trust.evidence import BenchmarkTier


def test_isotonic_calibration_skips_small_history_without_using_test_labels():
    rows = development_table(7).to_pylist()
    folds, _ = rolling_folds(rows, min_train_events=3)
    model = fit_race_model(
        rows, folds[-1], "hist", calibration_method="isotonic", calibration_event_count=3
    )
    assert model is not None
    assert len(model.metadata["calibration_events"]) == 3
    assert set(model.metadata["fit_indices"]).isdisjoint(model.metadata["calibration_indices"])
    assert set(model.metadata["calibration_indices"]).isdisjoint(folds[-1].test_indices)
    assert model.metadata["dnf_calibration"]["status"] == "skipped"
    earlier = model.metadata["calibration_prediction_timestamp"]
    assert model.metadata["fit_label_availability_max"] <= earlier
    assert model.calibrator is None


def test_repeated_prediction_timestamps_do_not_count_as_independent_selection_races():
    metrics = {
        task: {"status": "evaluated", "n": 100, "log_loss": 0.3}
        for task in ("winner", "podium", "dnf", "finishing_position")
    }
    comparisons = {
        "hist": {
            "metrics": metrics,
            "regressions": [],
            "paired_event_ids": ["season=2025/round=01"] * 100,
        }
    }
    assert (
        select_candidate(comparisons, tier=BenchmarkTier.GOLD, eligible_events=8)[
            "selected_backend"
        ]
        is None
    )
    with pytest.raises(ValueError, match="integer"):
        select_candidate(comparisons, tier=BenchmarkTier.GOLD, eligible_events=float("nan"))
