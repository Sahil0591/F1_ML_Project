from datetime import UTC, datetime, timedelta

import pytest

from f1_ml_predictor.benchmarks.enrichment import _derive


def _row(
    round_number: int,
    driver: str,
    constructor: str,
    position: int | None,
    *,
    available_days: int = 1,
) -> dict:
    cutoff = datetime(2025, 3, round_number * 2, 12, tzinfo=UTC)
    return {
        "event_id": f"season=2025/round={round_number:02d}",
        "driver_id": driver,
        "constructor_id": constructor,
        "prediction_timestamp": cutoff,
        "feature_timestamp": cutoff - timedelta(minutes=1),
        "benchmark_tier": "Gold",
        "cutoff_kind": "post_qualifying",
        "label_final_audited": True,
        "label_audit_reference": f"fia-final-{round_number}",
        "label_available_at": cutoff + timedelta(days=available_days),
        "label_position": position,
        "label_dnf": False if position is not None else None,
        "qualifying_position": float(position) if position is not None else None,
    }


def test_constructor_history_uses_prior_team_and_current_teammate() -> None:
    events = {
        (2025, number): [
            _row(number, "driver_a", "team_x" if number < 5 else "team_y", number),
            _row(number, "driver_b", "team_y", number + 1),
        ]
        for number in range(1, 7)
    }
    target = events[(2025, 6)][0]
    outcomes = {(2025, number): f"{number:064x}" for number in range(1, 7)}
    result, proof = _derive(target, events[(2025, 6)], events, outcomes)
    assert result["constructor_average_finish_last_3"] == 5.0
    assert result["constructor_finish_observations_last_3"] == 4
    assert result["constructor_teammate_aggregated_form"] == 4.0
    assert result["teammate_finish_observations_last_5"] == 5
    assert result["constructor_points_last_3"] is None
    assert (
        proof["reasons"]["constructor_points_last_3"] == "audited_scoring_and_sprint_ledger_absent"
    )
    assert "round=06" not in str(proof["source_events"])


def test_constructor_window_rejects_missing_or_late_prior_race() -> None:
    events = {
        (2025, number): [_row(number, "driver_a", "team_x", number)] for number in range(1, 7)
    }
    target = events[(2025, 6)][0]
    outcomes = {(2025, number): f"{number:064x}" for number in range(1, 7)}
    events[(2025, 4)][0]["label_available_at"] = target["prediction_timestamp"] + timedelta(days=1)
    result, proof = _derive(target, events[(2025, 6)], events, outcomes)
    assert result["constructor_average_finish_last_3"] is None
    assert (
        proof["reasons"]["constructor_average_finish_last_3"]
        == "prior_final_result_not_available_at_cutoff"
    )
    del events[(2025, 4)]
    result, proof = _derive(target, events[(2025, 6)], events, outcomes)
    assert result["constructor_average_finish_last_3"] is None
    assert proof["reasons"]["constructor_average_finish_last_3"] == "unaudited_prior_race_in_window"


def test_fia_practice_and_grid_respect_cutoff_and_roster() -> None:
    target = _row(6, "driver_a", "team_x", 1)
    teammate = _row(6, "driver_b", "team_x", 2)
    before = target["prediction_timestamp"] - timedelta(minutes=2)
    practice = {
        "practice": {
            "driver_a": {"position": 2, "best_lap_seconds": 81.2},
            "driver_b": {"position": 1, "best_lap_seconds": 80.7},
        },
        "field_size": 20,
        "available_at": before,
        "document_id": "7",
        "document": {"sha256": "practice-pdf"},
        "registry": {"sha256": "practice-registry"},
        "session": 3,
    }
    grid = {
        "grid": {"driver_a": {"grid_position": 3, "pit_lane_start": False}},
        "grid_status": "provisional",
        "available_at": before + timedelta(minutes=1),
        "document_id": "8",
        "document": {"sha256": "grid-pdf"},
        "registry": {"sha256": "grid-registry"},
    }
    result, proof = _derive(target, [target, teammate], {}, {}, grid, practice)
    assert result["practice_position"] == 2
    assert result["best_lap_gap_to_fastest"] == pytest.approx(0.5)
    assert result["teammate_practice_delta"] == pytest.approx(0.5)
    assert result["session_relative_rank"] == pytest.approx(1 / 19)
    assert result["grid_position"] == 3
    assert result["grid_status"] == "provisional"
    assert result["feature_timestamp"] == before + timedelta(minutes=1)
    assert proof["grid"]["document"]["sha256"] == "grid-pdf"
    assert proof["practice"]["document"]["sha256"] == "practice-pdf"

    grid["available_at"] = target["prediction_timestamp"] + timedelta(minutes=1)
    with pytest.raises(ValueError, match="available after prediction_timestamp"):
        _derive(target, [target, teammate], {}, {}, grid, practice)
