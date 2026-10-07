from datetime import date

import pytest

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.normalization.jolpica import (
    ENTRY_SCHEMA,
    EVENT_SCHEMA,
    QUALIFYING_SCHEMA,
    RESULT_SCHEMA,
    SPRINT_SCHEMA,
    normalize_entries,
    normalize_qualifying,
    normalize_results,
    normalize_schedule,
    normalize_sprint,
)


def _race() -> dict:
    return {
        "season": "2024",
        "round": "1",
        "raceName": "Bahrain Grand Prix",
        "Circuit": {"circuitId": "bahrain"},
        "date": "2024-03-02",
        "time": "15:00:00Z",
    }


def _result(driver_id: str = "verstappen") -> dict:
    return {
        "Driver": {"driverId": driver_id},
        "Constructor": {"constructorId": "red_bull"},
        "position": "1",
        "positionText": "1",
        "grid": "1",
        "laps": "57",
        "points": "25",
        "status": "Finished",
    }


def test_typed_tables_preserve_ids_and_unknown_availability() -> None:
    event = EventId(2024, 1)
    schedule = normalize_schedule([_race()], 2024)
    qualifying = normalize_qualifying(
        [{**_race(), "QualifyingResults": [{**_result(), "Q1": "1:30.123"}]}], event
    )
    results = normalize_results([{**_race(), "Results": [_result()]}], event)
    entries = normalize_entries(qualifying, results)

    assert schedule.schema == EVENT_SCHEMA
    assert qualifying.schema == QUALIFYING_SCHEMA
    assert results.schema == RESULT_SCHEMA
    assert entries.schema == ENTRY_SCHEMA
    assert schedule["race_date"][0].as_py() == date(2024, 3, 2)
    assert qualifying["q1_seconds"][0].as_py() == pytest.approx(90.123)
    assert entries["constructor_id"][0].as_py() == "red_bull"
    for table in (schedule, qualifying, results, entries):
        assert table["available_at"].null_count == table.num_rows


def test_missing_session_data_yields_typed_empty_tables() -> None:
    event = EventId(2024, 1)
    qualifying = normalize_qualifying([], event)
    results = normalize_results([{**_race()}], event)
    entries = normalize_entries(qualifying, results)
    assert qualifying.schema == QUALIFYING_SCHEMA
    assert results.schema == RESULT_SCHEMA
    assert entries.schema == ENTRY_SCHEMA
    assert all(table.num_rows == 0 for table in (qualifying, results, entries))


def test_empty_qualifying_times_are_missing_values() -> None:
    qualifying = normalize_qualifying(
        [{**_race(), "QualifyingResults": [{**_result(), "Q1": "", "Q2": ""}]}],
        EventId(2024, 1),
    )
    assert qualifying["q1_seconds"][0].as_py() is None
    assert qualifying["q2_seconds"][0].as_py() is None


def test_wrong_event_and_duplicate_driver_are_rejected() -> None:
    event = EventId(2024, 1)
    with pytest.raises(ValueError, match="different event"):
        normalize_results([{**_race(), "round": "2", "Results": [_result()]}], event)
    with pytest.raises(ValueError, match="duplicate driver"):
        normalize_results([{**_race(), "Results": [_result(), _result()]}], event)


def test_malformed_race_and_lap_time_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid date"):
        normalize_schedule([{**_race(), "date": "not-a-date"}], 2024)
    with pytest.raises(ValueError, match="mm:ss.sss"):
        normalize_qualifying(
            [{**_race(), "QualifyingResults": [{**_result(), "Q1": "DNF"}]}], EventId(2024, 1)
        )


def test_constructor_conflict_is_not_silently_joined() -> None:
    event = EventId(2024, 1)
    qualifying = normalize_qualifying([{**_race(), "QualifyingResults": [_result()]}], event)
    altered = {**_result(), "Constructor": {"constructorId": "ferrari"}}
    results = normalize_results([{**_race(), "Results": [altered]}], event)
    with pytest.raises(ValueError, match="conflicting constructors"):
        normalize_entries(qualifying, results)


def test_sprint_results_keep_car_number_grid_and_unknown_availability() -> None:
    event = EventId(2024, 1)
    sprint = normalize_sprint(
        [{**_race(), "SprintResults": [{**_result(), "number": "1", "points": "8", "grid": "2"}]}],
        event,
    )
    assert sprint.schema == SPRINT_SCHEMA
    row = sprint.to_pylist()[0]
    assert (row["number"], row["grid"], row["points"]) == (1, 2, 8.0)
    assert row["available_at"] is None
    assert normalize_sprint([{**_race()}], event).num_rows == 0
