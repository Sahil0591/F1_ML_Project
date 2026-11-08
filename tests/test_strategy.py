"""OpenF1 tyre and pit summaries and cutoff-bounded strategy predictors."""

from datetime import UTC, datetime, timedelta

import pytest

from f1_ml_predictor.prediction.strategy_features import strategy_values
from f1_ml_predictor.trust.strategy_history import driver_map, practice_summary, race_summary

ROSTER = {"max_verstappen", "norris", "hulkenberg"}
DRIVERS = [
    {"driver_number": 1, "full_name": "Max VERSTAPPEN", "last_name": "Verstappen"},
    {"driver_number": 4, "full_name": "Lando NORRIS", "last_name": "Norris"},
    {"driver_number": 27, "full_name": "Nico HULKENBERG", "last_name": "Hülkenberg"},
    {"driver_number": 99, "full_name": "Reserve DRIVER", "last_name": "Driver"},
]


def _laps(slopes: dict[int, float], count: int = 12) -> list[dict]:
    laps = []
    for number, slope in slopes.items():
        for lap in range(1, count + 1):
            laps.append(
                {
                    "driver_number": number,
                    "lap_number": lap,
                    "lap_duration": 90.0 + slope * lap,
                    "is_pit_out_lap": lap == 1,
                }
            )
    return laps


def _stints(compound: str, count: int = 12) -> list[dict]:
    return [
        {
            "driver_number": number,
            "lap_start": 1,
            "lap_end": count,
            "compound": compound,
            "tyre_age_at_start": 0,
        }
        for number in (1, 4, 27)
    ]


def test_driver_map_uses_aliases_then_unique_surnames() -> None:
    assert driver_map(DRIVERS, ROSTER) == {1: "max_verstappen", 4: "norris", 27: "hulkenberg"}


def test_race_summary_degradation_pit_deltas_and_wet_flag() -> None:
    pits = [
        {"driver_number": 1, "lap_number": 20, "pit_duration": 22.0, "stop_duration": 2.4},
        {"driver_number": 4, "lap_number": 20, "pit_duration": 23.0, "stop_duration": 3.4},
        # A red-flag wait is not crew performance and is dropped.
        {"driver_number": 27, "lap_number": 20, "pit_duration": 60.0, "stop_duration": None},
    ]
    summary = race_summary(
        {
            "drivers": DRIVERS,
            "stints": _stints("MEDIUM"),
            "pit": pits,
            "laps": _laps({1: 0.02, 4: 0.05, 27: 0.08}),
        },
        ROSTER,
    )
    assert summary["wet"] is False
    assert summary["lane_median"] == 22.5
    assert summary["drivers"]["hulkenberg"]["lane_deltas"] == []
    assert summary["drivers"]["max_verstappen"]["lane_deltas"] == [-0.5]
    slope, laps = summary["drivers"]["max_verstappen"]["degradation"][0]
    assert slope == pytest.approx(-0.03)
    assert laps == 11
    wet = race_summary(
        {"drivers": DRIVERS, "stints": _stints("INTERMEDIATE"), "pit": [], "laps": []}, ROSTER
    )
    assert wet["wet"] is True
    assert wet["wet_share"] == 1.0


def _mixed(wet_laps: int, total: int = 20) -> list[dict]:
    return [
        stint
        for number in (1, 4, 27, 99)
        for stint in (
            {
                "driver_number": number,
                "lap_start": 1,
                "lap_end": wet_laps,
                "compound": "INTERMEDIATE",
            },
            {
                "driver_number": number,
                "lap_start": wet_laps + 1,
                "lap_end": total,
                "compound": "SOFT",
            },
        )
    ]


def test_wet_label_needs_a_fifth_of_the_laps_on_wet_tyres() -> None:
    def summary(wet_laps: int) -> dict:
        documents = {"drivers": DRIVERS, "stints": _mixed(wet_laps), "pit": [], "laps": []}
        return race_summary(documents, ROSTER)

    # A wet start that dries after two laps stays dry.
    assert summary(2)["wet"] is False
    assert summary(2)["wet_share"] == pytest.approx(0.1)
    assert summary(4)["wet"] is True
    assert summary(4)["wet_share"] == pytest.approx(0.2)


def test_practice_long_runs_against_the_field() -> None:
    laps = _laps({1: 0.0, 4: 0.0, 27: 0.0}, count=8)
    for lap in laps:
        lap["lap_duration"] += {1: -0.5, 4: 0.0, 27: 0.4}[lap["driver_number"]]
    summary = practice_summary(
        {"drivers": DRIVERS, "stints": _stints("HARD", 8), "laps": laps}, ROSTER
    )
    assert summary["max_verstappen"]["long_run_gap"] == pytest.approx(-0.5)
    assert summary["hulkenberg"]["long_run_gap"] == pytest.approx(0.4)


def _event(start: datetime, wet: bool, lane: float, degradation: float) -> dict:
    return {
        "race_start": start,
        "available_at": start + timedelta(hours=26),
        "race": {
            "wet": wet,
            "lane_median": lane,
            "median_stops": 1.0,
            "drivers": {
                "norris": {"stints": 2, "lane_deltas": [0.3], "degradation": [[degradation, 10]]},
            },
        },
        "practice": [],
    }


def test_strategy_values_ignore_races_not_yet_available() -> None:
    start = datetime(2025, 5, 4, 20, 0, tzinfo=UTC)
    strategy = {
        "season=2025/round=01": _event(start - timedelta(days=14), False, 22.0, 0.01),
        "season=2025/round=02": _event(start - timedelta(hours=10), False, 30.0, 0.09),
    }
    values = strategy_values(
        strategy,
        {key: {"norris": "mclaren"} for key in strategy},
        {key: "miami" for key in strategy},
        event_id="season=2025/round=03",
        circuit_id="miami",
        cutoff=start - timedelta(hours=2),
        roster={"norris": "mclaren"},
    )["norris"]
    # Round 2 ended ten hours before the start, inside the 24 hour margin.
    assert values["driver_tyre_degradation_10"] == pytest.approx(0.01)
    assert values["constructor_pit_lane_delta_10"] == pytest.approx(0.3)
    assert values["circuit_pit_lane_seconds"] == 22.0
    assert values["driver_tyre_degradation_10_available_at"] <= start - timedelta(hours=2)
    assert "practice_long_run_gap" not in values


def test_track_weather_from_station_readings() -> None:
    from f1_ml_predictor.trust.strategy_history import track_weather

    readings = [
        {"rainfall": 1, "air_temperature": 18.0, "track_temperature": 22.0},
        {"rainfall": 0, "air_temperature": 19.0, "track_temperature": 24.0},
        {"rainfall": 1, "air_temperature": 20.0, "track_temperature": 26.0},
        {"rainfall": None, "air_temperature": None, "track_temperature": None},
    ]
    assert track_weather(readings) == {
        "track_rain_fraction": pytest.approx(2 / 3),
        "track_air_temperature": 19.0,
        "track_surface_temperature": 24.0,
    }
    assert track_weather([]) == {
        "track_rain_fraction": None,
        "track_air_temperature": None,
        "track_surface_temperature": None,
    }
