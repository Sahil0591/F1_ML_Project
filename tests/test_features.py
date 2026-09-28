from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.features.contracts import (
    FeatureInputs,
    PreRaceEvent,
    PublishedTable,
    ResultVersion,
)
from f1_ml_predictor.features.snapshot import build_snapshot
from f1_ml_predictor.features.storage import persist_snapshot
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.paths import StoragePaths


def dt(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2025, 3, day, hour, minute, tzinfo=UTC)


CUTOFF = dt(15, 10)
TARGET = EventId(2025, 3)
ROSTER_AVAILABLE = dt(15, 9, 30)
QUALIFYING_AVAILABLE = dt(15, 9, 20)


def publication(
    rows: list[dict[str, object]], available: datetime | None, ref: str
) -> PublishedTable:
    return PublishedTable(pa.Table.from_pylist(rows), available, ref)


def event_spec() -> PreRaceEvent:
    return PreRaceEvent(TARGET, "silverstone", dt(15, 14), dt(15, 9), dt(15, 8, 30), "event-proof")


def roster(
    rows: list[dict[str, object]] | None = None, available: datetime = ROSTER_AVAILABLE
) -> PublishedTable:
    return publication(
        rows
        or [
            {
                "event_id": TARGET.partition(),
                "driver_id": "driver_b",
                "constructor_id": "team_x",
                "grid_position": 4,
            },
            {
                "event_id": TARGET.partition(),
                "driver_id": "driver_a",
                "constructor_id": "team_x",
                "grid_position": 2,
            },
            {
                "event_id": TARGET.partition(),
                "driver_id": "driver_c",
                "constructor_id": "team_y",
                "grid_position": 8,
            },
        ],
        available,
        "roster-proof",
    )


def qualifying(
    rows: list[dict[str, object]] | None = None, available: datetime | None = QUALIFYING_AVAILABLE
) -> PublishedTable:
    return publication(
        rows
        or [
            {
                "event_id": TARGET.partition(),
                "driver_id": "driver_c",
                "constructor_id": "team_y",
                "position": 8,
                "q3_seconds": 92.0,
            },
            {
                "event_id": TARGET.partition(),
                "driver_id": "driver_b",
                "constructor_id": "team_x",
                "position": 4,
                "q3_seconds": 91.0,
            },
            {
                "event_id": TARGET.partition(),
                "driver_id": "driver_a",
                "constructor_id": "team_x",
                "position": 2,
                "q3_seconds": 90.0,
            },
        ],
        available,
        "qualifying-proof",
    )


def result(
    event: EventId, driver: str, position: int, available: datetime, *, dnf: bool = False
) -> ResultVersion:
    table = publication(
        [
            {
                "event_id": event.partition(),
                "driver_id": driver,
                "constructor_id": "team_x",
                "position": position,
                "dnf": dnf,
                "pit_stop_seconds": 24.0,
            }
        ],
        available,
        f"{event.partition()}-{position}",
    )
    return ResultVersion(event, dt(1 + event.round, 15), table)


def race_results(
    event: EventId,
    entries: list[tuple[str, int, bool]],
    available: datetime,
    reference: str,
) -> ResultVersion:
    rows = [
        {
            "event_id": event.partition(),
            "driver_id": driver,
            "constructor_id": "team_x",
            "position": position,
            "dnf": dnf,
            "pit_stop_seconds": 24.0,
        }
        for driver, position, dnf in entries
    ]
    return ResultVersion(event, dt(1 + event.round, 15), publication(rows, available, reference))


def base_inputs(**changes: object) -> FeatureInputs:
    values: dict[str, object] = {
        "event": event_spec(),
        "rosters": (roster(),),
        "qualifying": (qualifying(),),
    }
    values.update(changes)
    return FeatureInputs(**values)  # type: ignore[arg-type]


def test_roster_order_is_deterministic_and_lagged_form_and_teammate_features_are_correct() -> None:
    history = (
        race_results(
            EventId(2025, 1), [("driver_a", 3, False), ("driver_b", 7, False)], dt(10), "round-1"
        ),
        race_results(
            EventId(2025, 2), [("driver_a", 5, True), ("driver_b", 9, False)], dt(14), "round-2"
        ),
    )
    table = build_snapshot(base_inputs(history=history), CUTOFF)
    assert table.column("driver_id").to_pylist() == ["driver_a", "driver_b", "driver_c"]

    driver_a = table.to_pylist()[0]
    assert driver_a["recent_finish_mean"] == 4.0
    assert driver_a["recent_dnf_rate"] == 0.5
    assert driver_a["recent_pit_stop_seconds"] == 24.0
    assert driver_a["history_count"] == 2
    assert driver_a["teammate_qualifying_position_delta"] == -2.0
    assert driver_a["recent_finish_mean_missing"] is False
    assert table.to_pylist()[2]["recent_finish_mean_missing"] is True
    assert table.to_pylist()[2]["practice_best_seconds_missing"] is True


def test_latest_late_correction_is_selected_but_target_and_future_races_are_excluded() -> None:
    earlier = race_results(EventId(2025, 1), [("driver_a", 3, False)], dt(10), "initial-result")
    correction = race_results(
        EventId(2025, 1), [("driver_a", 11, False)], dt(14, 13), "corrected-result"
    )
    target_race = ResultVersion(
        TARGET,
        dt(15, 15),
        publication(
            [
                {
                    "event_id": TARGET.partition(),
                    "driver_id": "driver_a",
                    "constructor_id": "team_x",
                    "position": 1,
                    "dnf": False,
                }
            ],
            dt(15, 15, 30),
            "target-result",
        ),
    )
    future = ResultVersion(
        EventId(2025, 4),
        dt(20, 15),
        publication(
            [
                {
                    "event_id": "season=2025/round=04",
                    "driver_id": "driver_a",
                    "constructor_id": "team_x",
                    "position": 2,
                    "dnf": False,
                }
            ],
            dt(21),
            "future-result",
        ),
    )
    table = build_snapshot(base_inputs(history=(earlier, correction, target_race, future)), CUTOFF)
    row = table.to_pylist()[0]
    assert row["recent_finish_mean"] == 11.0
    assert row["history_count"] == 1
    assert "target-result" not in row["provenance"]


def test_missing_and_late_roster_or_qualifying_are_rejected() -> None:
    with pytest.raises(ValueError, match="roster and qualifying"):
        build_snapshot(base_inputs(rosters=()), CUTOFF)
    with pytest.raises(ValueError, match="roster and qualifying"):
        build_snapshot(base_inputs(qualifying=(qualifying(available=dt(15, 10, 1)),)), CUTOFF)
    with pytest.raises(ValueError, match="roster and qualifying"):
        build_snapshot(base_inputs(qualifying=(qualifying(available=None),)), CUTOFF)


def test_unknown_normalized_publication_cannot_be_backdated_or_used() -> None:
    normalized = pa.table({"event_id": [TARGET.partition()], "driver_id": ["driver_a"]})
    unknown = PublishedTable(normalized, None, "retrospective-fetch-without-release-proof")
    with pytest.raises(ValueError, match="roster and qualifying"):
        build_snapshot(base_inputs(rosters=(unknown,)), CUTOFF)

    with pytest.raises(ValueError, match="became available"):
        PublishedTable(
            pa.table(
                {
                    "event_id": [TARGET.partition()],
                    "available_at": [dt(15, 11)],
                }
            ),
            dt(15, 9),
            "incorrect-backdate",
        )


def test_rejects_cutoff_before_qualifying_or_at_race_start() -> None:
    with pytest.raises(ValueError, match="after qualifying"):
        build_snapshot(base_inputs(), dt(15, 8, 59))
    with pytest.raises(ValueError, match="after qualifying"):
        build_snapshot(base_inputs(), dt(15, 14))


@pytest.mark.parametrize(
    "row",
    [
        {
            "event_id": TARGET.partition(),
            "session_code": "R",
            "source": "fastf1",
            "driver_id": "driver_a",
        },
        {
            "event_id": TARGET.partition(),
            "valid_at": dt(15, 14),
            "captured_at": dt(15, 9),
            "temperature_2m": 19,
        },
    ],
)
def test_rejects_race_session_or_observed_weather(row: dict[str, object]) -> None:
    if row.get("session_code"):
        inputs = base_inputs(sessions=(publication([row], dt(15, 9, 40), "race-session"),))
    else:
        inputs = base_inputs(forecasts=(publication([row], dt(15, 9, 40), "observed-weather"),))
    with pytest.raises(ValueError):
        build_snapshot(inputs, CUTOFF)


def test_pre_race_practice_and_captured_forecast_are_used_with_availability() -> None:
    session = publication(
        [
            {
                "event_id": TARGET.partition(),
                "session_code": "FP2",
                "source": "fastf1",
                "driver_id": "driver_a",
                "best_lap_seconds": 89.0,
                "median_lap_seconds": 90.0,
                "median_tyre_age": 3.0,
                "compound": "SOFT",
            }
        ],
        dt(15, 9, 45),
        "practice-proof",
    )
    forecast = publication(
        [
            {
                "event_id": TARGET.partition(),
                "valid_at": dt(15, 14),
                "captured_at": dt(15, 9, 50),
                "available_at": dt(15, 9, 50),
                "temperature_2m": 18.0,
                "precipitation_probability": 20,
                "wind_speed_10m": 4.0,
                "run_initialized_at": None,
                "availability_evidence": None,
            }
        ],
        dt(15, 9, 55),
        "weather-capture-proof",
    )
    table = build_snapshot(base_inputs(sessions=(session,), forecasts=(forecast,)), CUTOFF)
    row = table.to_pylist()[0]
    assert row["practice_best_seconds"] == 89.0
    assert row["forecast_temperature_2m"] == 18.0
    assert row["forecast_temperature_2m_missing"] is False
    assert row["feature_timestamp"] == dt(15, 9, 55)


def test_live_forecast_cannot_claim_earlier_availability_than_capture() -> None:
    forecast = publication(
        [
            {
                "event_id": TARGET.partition(),
                "valid_at": dt(15, 14),
                "captured_at": dt(15, 9, 50),
                "available_at": dt(15, 9, 40),
                "temperature_2m": 18.0,
                "run_initialized_at": None,
            }
        ],
        dt(15, 9, 55),
        "backdated-live-forecast",
    )
    with pytest.raises(ValueError, match="cannot be backdated"):
        build_snapshot(base_inputs(forecasts=(forecast,)), CUTOFF)


def test_persisted_snapshot_is_content_addressed_and_reusable(tmp_path) -> None:
    table = build_snapshot(base_inputs(), CUTOFF)
    paths = StoragePaths(tmp_path)
    first = persist_snapshot(paths, table)
    second = persist_snapshot(paths, table)
    assert first == second
    assert first.is_file()
    assert pq.read_table(first).equals(table)


def test_reordering_inputs_preserves_features_and_provenance() -> None:
    original = base_inputs()
    reversed_roster = replace(original.rosters[0], table=original.rosters[0].table.take([2, 1, 0]))
    reversed_qualifying = replace(
        original.qualifying[0], table=original.qualifying[0].table.take([2, 1, 0])
    )
    reordered = replace(original, rosters=(reversed_roster,), qualifying=(reversed_qualifying,))
    assert build_snapshot(original, CUTOFF).equals(build_snapshot(reordered, CUTOFF))


def test_empty_qualifying_publication_is_rejected() -> None:
    empty = replace(qualifying(), table=qualifying().table.slice(0, 0))
    with pytest.raises(ValueError, match="qualifying publication must contain drivers"):
        build_snapshot(base_inputs(qualifying=(empty,)), CUTOFF)


def test_late_prior_result_correction_does_not_change_earlier_features() -> None:
    old = race_results(EventId(2025, 1), [("driver_a", 3, False)], dt(10), "initial-result")
    late = race_results(EventId(2025, 1), [("driver_a", 20, True)], dt(16), "late-correction")
    baseline = build_snapshot(base_inputs(history=(old,)), CUTOFF)
    assert baseline.equals(build_snapshot(base_inputs(history=(old, late)), CUTOFF))


def test_missing_compounds_are_not_encoded_as_zero() -> None:
    practice = publication(
        [
            {
                "event_id": TARGET.partition(),
                "driver_id": "driver_a",
                "session_code": "FP1",
                "source": "fastf1",
                "compound": None,
            }
        ],
        dt(15, 9, 40),
        "practice-proof",
    )
    row = build_snapshot(base_inputs(sessions=(practice,)), CUTOFF).to_pylist()[0]
    assert row["tyre_compound_count"] is None
    assert row["tyre_compound_count_missing"]


@pytest.mark.parametrize("rounds", [(1, 2), (3, 3)])
def test_mixed_or_target_round_standings_are_rejected(rounds: tuple[int, int]) -> None:
    standings = publication(
        [
            {"season": 2025, "round": round_number, "driver_id": driver}
            for round_number, driver in zip(rounds, ["driver_a", "driver_b"], strict=True)
        ],
        dt(15, 9),
        "standings-proof",
    )
    with pytest.raises(ValueError, match="standings"):
        build_snapshot(base_inputs(standings=standings), CUTOFF)


def test_current_and_future_event_ids_are_excluded_even_with_earlier_timestamps() -> None:
    earlier = race_results(EventId(2025, 1), [("driver_a", 3, False)], dt(10), "past")
    wrong_events = tuple(
        ResultVersion(
            event,
            dt(10),
            publication(
                [
                    {
                        "event_id": event.partition(),
                        "driver_id": "driver_a",
                        "constructor_id": "team_x",
                        "position": 1,
                        "dnf": False,
                    }
                ],
                dt(11),
                "wrong-event",
            ),
        )
        for event in (TARGET, EventId(2025, 4), EventId(2026, 1))
    )
    baseline = build_snapshot(base_inputs(history=(earlier,)), CUTOFF)
    assert baseline.equals(build_snapshot(base_inputs(history=(earlier, *wrong_events)), CUTOFF))


@pytest.mark.parametrize("evidence", ["archived-release-proof", "", " "])
def test_archived_forecast_requires_release_evidence_not_capture_before_cutoff(
    evidence: str,
) -> None:
    forecast = publication(
        [
            {
                "event_id": TARGET.partition(),
                "valid_at": dt(15, 14),
                "available_at": dt(15, 9, 45),
                "captured_at": dt(16),
                "run_initialized_at": dt(15, 6),
                "availability_evidence": evidence,
                "temperature_2m": -2.0,
                "precipitation_probability": 10,
                "wind_speed_10m": 2,
            }
        ],
        dt(15, 9, 45),
        "exact-archive-run",
    )
    if not evidence.strip():
        with pytest.raises(ValueError, match="release evidence"):
            build_snapshot(base_inputs(forecasts=(forecast,)), CUTOFF)
    else:
        row = build_snapshot(base_inputs(forecasts=(forecast,)), CUTOFF).to_pylist()[0]
        assert row["forecast_temperature_2m"] == -2.0
        assert row["feature_timestamp"] <= CUTOFF


def test_session_source_preference_does_not_average_sources() -> None:
    def practice(source: str, seconds: float) -> PublishedTable:
        return publication(
            [
                {
                    "event_id": TARGET.partition(),
                    "driver_id": "driver_a",
                    "session_code": "FP2",
                    "source": source,
                    "compound": "SOFT",
                    "best_lap_seconds": seconds,
                }
            ],
            dt(15, 9, 40),
            source,
        )

    inputs = base_inputs(sessions=(practice("fastf1", 90.0), practice("openf1", 85.0)))
    assert build_snapshot(inputs, CUTOFF).to_pylist()[0]["practice_best_seconds"] == 90.0
    assert (
        build_snapshot(inputs, CUTOFF, session_source="openf1").to_pylist()[0][
            "practice_best_seconds"
        ]
        == 85.0
    )
