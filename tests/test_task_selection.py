"""Task gates cannot be vetoed by an unrelated prediction target."""

from f1_ml_predictor.models.probabilistic import select_task_candidates
from f1_ml_predictor.trust.evidence import BenchmarkTier


def test_task_specific_gate_keeps_winner_separate_from_podium_regression():
    interval = {
        "status": "estimated",
        "bootstrap_95_percent_interval": [-0.2, -0.01],
    }
    comparison = {
        "paired_event_ids": [f"season=2025/round={number:02d}" for number in range(1, 27)],
        "metrics": {
            "winner": {"status": "evaluated", "log_loss": 1.1},
            "podium": {"status": "evaluated", "brier_score": 0.2},
            "dnf": {"status": "insufficient_data"},
            "finishing_position": {"status": "evaluated", "mean_absolute_error": 2.5},
        },
        "baselines": {
            baseline: {
                "winner": {"status": "evaluated", "log_loss": 1.3},
                "podium": {"status": "evaluated", "brier_score": 0.19},
                "dnf": {"status": "insufficient_data"},
                "finishing_position": {"status": "evaluated", "mean_absolute_error": 2.8},
            }
            for baseline in ("heuristic", "logistic")
        },
        "regressions": [{"task": "podium", "baseline": "logistic", "metric": "brier_score"}],
        "paired_uncertainty": {
            baseline: {
                task: interval
                for task in ("winner_log_loss", "podium_brier", "dnf_brier", "position_mae")
            }
            for baseline in ("heuristic", "logistic")
        },
    }
    result = select_task_candidates(
        {"catboost": comparison}, tier=BenchmarkTier.GOLD, eligible_events=29
    )
    assert result["winner"]["selected_backend"] == "catboost"
    assert result["podium"]["status"] == "no_selection"
    assert result["dnf"]["status"] == "no_selection"
