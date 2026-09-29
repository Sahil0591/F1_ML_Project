import math

import pytest
from test_probabilistic import development_table

from f1_ml_predictor.models.backtest import rolling_folds
from f1_ml_predictor.models.probabilistic import fit_race_model, select_candidate
from f1_ml_predictor.models.uncertainty import paired_loss_intervals
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
        select_candidate(comparisons, tier=BenchmarkTier.GOLD, eligible_events=25)[
            "selected_backend"
        ]
        is None
    )
    with pytest.raises(ValueError, match="integer"):
        select_candidate(comparisons, tier=BenchmarkTier.GOLD, eligible_events=float("nan"))


def test_paired_uncertainty_resamples_races_and_checks_exact_observations():
    candidate = []
    baseline = []
    for event in ("race-1", "race-2"):
        for driver, winner in (("a", True), ("b", False)):
            row = {
                "event_id": event,
                "driver_id": driver,
                "prediction_timestamp": "cutoff",
                "cutoff_kind": "post_qualifying",
                "label_winner": winner,
                "label_podium": winner,
                "label_dnf": False,
                "label_position": 1 if winner else 2,
                "winner_probability": 0.8 if winner else 0.2,
                "podium_probability": 0.8 if winner else 0.2,
                "dnf_probability": 0.1,
                "expected_position": 1 if winner else 2,
            }
            candidate.append(row)
            baseline.append(
                {
                    **row,
                    "heuristic_winner_probability": 0.5,
                    "heuristic_podium_probability": 0.5,
                    "heuristic_dnf_probability": 0.2,
                    "grid_finish_position": 2 if winner else 1,
                }
            )
    result = paired_loss_intervals(candidate, baseline, "heuristic", seed=42)
    winner = result["winner_log_loss"]
    assert winner["paired_events"] == 2
    assert winner["driver_race_observations"] == 2
    assert winner["mean_loss_delta"] == pytest.approx(math.log(0.5 / 0.8))
    assert winner["bootstrap_95_percent_interval"] == pytest.approx([winner["mean_loss_delta"]] * 2)
    assert result["dnf_brier"]["driver_race_observations"] == 4
    assert paired_loss_intervals(candidate, baseline, "heuristic", seed=42) == result
    with pytest.raises(ValueError, match="identical driver-race"):
        paired_loss_intervals(candidate[:-1], baseline, "heuristic", seed=42)
