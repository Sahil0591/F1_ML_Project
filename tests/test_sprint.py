"""Offline checks for sprint sources, captures, standings and export sessions."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.normalization.jolpica import SPRINT_SCHEMA
from f1_ml_predictor.prediction.schedules import Weekend, weekends
from f1_ml_predictor.prediction.sprint import (
    SprintGrid,
    _cross_check,
    _labels,
    _sprint_prior_rows,
    crosswalk,
    openf1_grid,
    weekend_values,
)
from f1_ml_predictor.prediction.web_export import EXPORT_CUTOFFS, session_of
from f1_ml_predictor.trust.qualifying_fallback import QUALIFYING_OPENF1, qualifying_fallback_tick
from f1_ml_predictor.trust.sprint_capture import (
    FALLBACK_DELAY,
    SPRINT_QUALIFYING,
    SPRINT_RESULT,
    capture_grid,
    capture_points,
    capture_teams,
    captured_sprint_values,
    latest_capture,
    sprint_tick,
)

EVENT = EventId(2026, 17)
NOW = datetime(2026, 10, 9, 14, tzinfo=UTC)


def _drivers_payload() -> dict:
    drivers = [
        ("max_verstappen", "Max", "Verstappen", "VER"),
        ("kevin_magnussen", "Kevin", "Magnussen", "MAG"),
        ("norris", "Lando", "Norris", "NOR"),
        ("piastri", "Oscar", "Piastri", "PIA"),
    ]
    return {
        "MRData": {
            "total": str(len(drivers)),
            "DriverTable": {
                "Drivers": [
                    {"driverId": key, "givenName": given, "familyName": family, "code": code}
                    for key, given, family, code in drivers
                ]
            },
        }
    }


def _session_drivers() -> list[dict]:
    return [
        {"driver_number": 3, "name_acronym": "VER", "team_name": "Red Bull Racing"},
        {"driver_number": 20, "name_acronym": "MAG", "team_name": "Haas F1 Team"},
        {"driver_number": 1, "name_acronym": "NOR", "team_name": "McLaren"},
        {"driver_number": 81, "name_acronym": "PIA", "team_name": "Unknown Racing"},
    ]


def _session_result() -> list[dict]:
    return [
        {"driver_number": 1, "position": 1, "duration": [91.2, 90.9, 90.5]},
        {"driver_number": 81, "position": 2, "duration": [91.3, 91.0, None]},
        {"driver_number": 3, "position": 3, "duration": [91.4, None, None]},
    ]


def test_crosswalk_uses_canonical_gold_identifiers() -> None:
    identities = crosswalk(_drivers_payload())
    assert identities.driver("kevin_magnussen") == "magnussen"
    assert identities.codes["MAG"] == "magnussen"
    assert identities.constructor("alphatauri") == "alpha_tauri"
    assert identities.constructor("red_bull") == "red_bull"


def test_openf1_grid_keeps_entered_drivers_without_a_result() -> None:
    grid = openf1_grid(
        _session_result(), _session_drivers(), crosswalk(_drivers_payload()).codes, NOW, "test"
    )
    assert grid.positions == {"norris": 1, "piastri": 2, "max_verstappen": 3, "magnussen": None}
    # The last stage reached is the latest non-missing lap.
    assert grid.last_seconds["norris"] == 90.5
    assert grid.last_seconds["max_verstappen"] == 91.4
    assert grid.last_seconds["magnussen"] is None


def test_openf1_grid_rejects_unmapped_and_empty_results() -> None:
    codes = crosswalk(_drivers_payload()).codes
    with pytest.raises(ValueError, match="no Jolpica code"):
        openf1_grid([], [{"driver_number": 9, "name_acronym": "XXX"}], codes, NOW, "test")
    with pytest.raises(ValueError, match="not in the driver list"):
        openf1_grid([{"driver_number": 99, "position": 1}], _session_drivers(), codes, NOW, "t")
    with pytest.raises(ValueError, match="empty"):
        openf1_grid([], _session_drivers(), codes, NOW, "test")


def test_weekend_values_follow_the_race_contract_conventions() -> None:
    grid = SprintGrid(
        {"a1": 1, "a2": 4, "b1": 2, "b2": None},
        {"a1": 90.0, "a2": 91.0, "b1": 90.5, "b2": None},
        NOW,
        "test",
    )
    values = weekend_values(grid, {"a1": "red", "a2": "red", "b1": "blue", "b2": "blue"})
    assert values["a1"]["teammate_qualifying_position_delta"] == -3
    assert values["a2"]["teammate_qualifying_position_delta"] == 3
    assert values["b1"]["teammate_qualifying_position_delta"] is None
    assert values["a1"]["qualifying_position_available_at"] == NOW


def _sprint_table(path: Path, rows: list[tuple]) -> Path:
    table = pa.Table.from_pylist(
        [
            {
                "event_id": EVENT.partition(),
                "driver_id": driver,
                "constructor_id": team,
                "number": None,
                "position": position,
                "position_text": text,
                "grid": None,
                "laps": None,
                "points": points,
                "status": status,
                "available_at": None,
            }
            for driver, team, position, text, points, status in rows
        ],
        schema=SPRINT_SCHEMA,
    )
    pq.write_table(table, path)
    return path


def test_sprint_labels_classify_retirements_and_leave_dns_unlabelled(tmp_path: Path) -> None:
    path = _sprint_table(
        tmp_path / "sprint.parquet",
        [
            ("norris", "mclaren", 1, "1", 8.0, "Finished"),
            ("piastri", "mclaren", 2, "2", 7.0, "+1 Lap"),
            ("max_verstappen", "red_bull", 3, "R", 0.0, "Retired"),
            ("kevin_magnussen", "haas", 4, "W", 0.0, "Did not start"),
        ],
    )
    roster, labels = _labels(path, NOW, crosswalk(_drivers_payload()))
    assert roster["magnussen"] == "haas"
    assert labels["norris"]["label_winner"] and labels["piastri"]["label_podium"]
    assert labels["piastri"]["label_dnf"] is False
    assert labels["max_verstappen"]["label_position"] is None
    assert labels["max_verstappen"]["label_dnf"] is True
    assert labels["magnussen"]["label_dnf"] is None
    assert labels["magnussen"]["label_dnf_available_at"] is None


def test_cross_check_excludes_a_sprint_whose_sources_disagree() -> None:
    labels = {
        "norris": {"points": 8.0, "label_position": 1},
        "piastri": {"points": 7.0, "label_position": 2},
    }
    evidence = {
        (2026, 17, "norris"): {"sprint_points": 8.0, "sprint_position": "1"},
        (2026, 17, "piastri"): {"sprint_points": 7.0, "sprint_position": "2"},
    }
    assert _cross_check(EVENT, labels, evidence) is None
    evidence[(2026, 17, "piastri")]["sprint_position"] = "3"
    assert _cross_check(EVENT, labels, evidence) == "sprint_position_disagree:piastri"
    evidence[(2026, 17, "piastri")]["sprint_points"] = 6.0
    assert _cross_check(EVENT, labels, evidence) == "sprint_points_disagree:piastri"
    del evidence[(2026, 17, "norris")]
    assert _cross_check(EVENT, labels, evidence) == "no_fia_sprint_record:norris"


def test_sprint_retirement_prior_uses_only_sprints_known_by_the_cutoff() -> None:
    def sprint(start: datetime, dnf: bool) -> object:
        class Item:
            sprint_start = start
            labels = {"x": {"label_dnf": dnf, "label_dnf_available_at": start + timedelta(hours=2)}}

        return Item()

    earlier = sprint(NOW - timedelta(days=30), True)
    later = sprint(NOW + timedelta(days=1), False)
    rows = _sprint_prior_rows([earlier, later], NOW)  # type: ignore[list-item]
    assert rows == [{"label_dnf": True}]


def _weekend(sprint_qualifying: datetime, sprint: datetime) -> Weekend:
    return Weekend(
        EVENT,
        "Singapore Grand Prix",
        "marina_bay",
        "Marina Bay",
        sprint_qualifying - timedelta(hours=4),
        sprint + timedelta(hours=4),
        sprint,
        sprint + timedelta(days=1),
        sprint_qualifying,
    )


def _transport(result: list[dict], *, end: datetime) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/v1/sessions"):
            return httpx.Response(
                200,
                json=[
                    {
                        "session_key": 11379,
                        "session_name": "Sprint Qualifying",
                        "date_start": (end - timedelta(minutes=44)).isoformat(),
                        "date_end": end.isoformat(),
                    }
                ],
            )
        if path.endswith("/v1/session_result"):
            return httpx.Response(200, json=result)
        if path.endswith("/v1/drivers"):
            return httpx.Response(200, json=_session_drivers())
        if path.endswith("/drivers/"):
            return httpx.Response(200, json=_drivers_payload())
        raise AssertionError(f"unexpected request {request.url}")

    return httpx.MockTransport(respond)


def test_sprint_tick_waits_then_freezes_a_verified_capture(tmp_path: Path) -> None:
    start = NOW - timedelta(hours=2)
    weekend = _weekend(start, NOW + timedelta(hours=20))
    end = start + timedelta(minutes=44)
    early = sprint_tick(tmp_path, EVENT, weekend, now=lambda: start - timedelta(minutes=1))
    assert early["status"] == "waiting_for_sprint_qualifying"
    with httpx.Client(transport=_transport([], end=end)) as client:
        waiting = sprint_tick(tmp_path, EVENT, weekend, now=lambda: NOW, http_client=client)
    assert waiting["status"] == "waiting_for_sprint_qualifying_results"
    with httpx.Client(transport=_transport(_session_result(), end=end)) as client:
        captured = sprint_tick(tmp_path, EVENT, weekend, now=lambda: NOW, http_client=client)
    assert captured["status"] == "sprint_qualifying_captured"
    record = latest_capture(tmp_path, EVENT, SPRINT_QUALIFYING, NOW)
    assert record is not None
    grid = capture_grid(record)
    assert grid.positions["norris"] == 1 and grid.positions["magnussen"] is None
    assert grid.available_at == NOW
    # A capture is invisible to an earlier clock.
    assert latest_capture(tmp_path, EVENT, SPRINT_QUALIFYING, NOW - timedelta(seconds=1)) is None


def test_capture_teams_read_the_session_seat_and_skip_unmapped_names(tmp_path: Path) -> None:
    start = NOW - timedelta(hours=2)
    weekend = _weekend(start, NOW + timedelta(hours=20))
    with httpx.Client(
        transport=_transport(_session_result(), end=start + timedelta(minutes=44))
    ) as client:
        sprint_tick(tmp_path, EVENT, weekend, now=lambda: NOW, http_client=client)
    record = latest_capture(tmp_path, EVENT, SPRINT_QUALIFYING, NOW)
    assert record is not None
    # Piastri's team name is unmapped, so the latest audited roster decides it.
    assert capture_teams(record) == {
        "max_verstappen": "red_bull",
        "magnussen": "haas",
        "norris": "mclaren",
    }


def test_tampered_sprint_capture_is_rejected(tmp_path: Path) -> None:
    start = NOW - timedelta(hours=2)
    weekend = _weekend(start, NOW + timedelta(hours=20))
    with httpx.Client(
        transport=_transport(_session_result(), end=start + timedelta(minutes=44))
    ) as client:
        sprint_tick(tmp_path, EVENT, weekend, now=lambda: NOW, http_client=client)
    path = next((tmp_path / "data/raw/prospective_sprint").rglob("sprint_qualifying-*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["captured_at"] = (NOW - timedelta(days=1)).isoformat()
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="changed since it was frozen"):
        latest_capture(tmp_path, EVENT, SPRINT_QUALIFYING, NOW)


def test_sprint_qualifying_is_missed_once_the_sprint_starts(tmp_path: Path) -> None:
    weekend = _weekend(NOW - timedelta(days=1), NOW - timedelta(hours=1))
    status = sprint_tick(tmp_path, EVENT, weekend, now=lambda: NOW)
    assert status["status"] == "sprint_qualifying_missed"


def test_schedule_reads_sprint_qualifying_and_2023_sprint_shootout() -> None:
    def race(round_number: int, key: str) -> dict:
        return {
            "season": "2026",
            "round": str(round_number),
            "raceName": "Grand Prix",
            "Circuit": {"circuitId": "c"},
            "date": "2026-10-11",
            "time": "12:00:00Z",
            key: {"date": "2026-10-09", "time": "12:30:00Z"},
        }

    parsed = weekends(
        {
            "MRData": {
                "RaceTable": {"Races": [race(1, "SprintQualifying"), race(2, "SprintShootout")]}
            }
        }
    )
    expected = datetime(2026, 10, 9, 12, 30, tzinfo=UTC)
    assert parsed[EventId(2026, 1)].sprint_qualifying == expected
    assert parsed[EventId(2026, 2)].sprint_qualifying == expected


def test_export_orders_the_sprint_snapshot_between_practice_and_qualifying() -> None:
    assert EXPORT_CUTOFFS == (
        "pre_weekend",
        "post_practice",
        "post_sprint_qualifying",
        "post_qualifying",
        "pre_race",
    )
    assert session_of("post_sprint_qualifying") == "sprint"
    assert session_of("post_qualifying") == "race"


def test_openf1_grid_ranks_a_non_numeric_position_behind_the_classified() -> None:
    result = [*_session_result(), {"driver_number": 20, "position": "RT", "duration": [None]}]
    grid = openf1_grid(result, _session_drivers(), crosswalk(_drivers_payload()).codes, NOW, "t")
    assert grid.positions["magnussen"] == 4
    assert grid.last_seconds["magnussen"] is None


def _fallback_transport(
    sessions: list[dict], results: dict[int, list[dict]], jolpica_sprint: dict | None = None
) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "api.openf1.org":
            if path.endswith("/v1/sessions"):
                return httpx.Response(200, json=sessions)
            key = int(request.url.params["session_key"])
            if path.endswith("/v1/session_result"):
                return httpx.Response(200, json=results.get(key, []))
            if path.endswith("/v1/drivers"):
                return httpx.Response(200, json=_session_drivers())
        if request.url.host == "api.jolpi.ca":
            if path.endswith("/drivers/"):
                return httpx.Response(200, json=_drivers_payload())
            if path.endswith("/sprint/"):
                empty = {"MRData": {"total": "0", "RaceTable": {"Races": []}}}
                return httpx.Response(200, json=jolpica_sprint or empty)
        # FIA documents are not part of these checks.
        return httpx.Response(404, text="")

    return httpx.MockTransport(respond)


def _session(key: int, name: str, start: datetime, minutes: int) -> dict:
    return {
        "session_key": key,
        "session_name": name,
        "date_start": start.isoformat(),
        "date_end": (start + timedelta(minutes=minutes)).isoformat(),
    }


def test_sprint_result_falls_back_to_openf1_then_jolpica_supersedes(tmp_path: Path) -> None:
    sprint_start = NOW - timedelta(hours=1)
    weekend = _weekend(sprint_start - timedelta(hours=20), sprint_start)
    sessions = [
        _session(11379, "Sprint Qualifying", weekend.sprint_qualifying, 44),  # type: ignore[arg-type]
        _session(11383, "Sprint", sprint_start, 40),
    ]
    sprint_result = [
        {"driver_number": 1, "position": 1, "points": 8.0, "dnf": False},
        # Retired on the last lap but still classified: dnf is not classification.
        {"driver_number": 81, "position": 2, "points": 7.0, "dnf": True},
        {"driver_number": 3, "position": None, "points": 0.0, "dnf": True},
    ]
    transport = _fallback_transport(sessions, {11379: _session_result(), 11383: sprint_result})
    with httpx.Client(transport=transport) as client:
        sprint_tick(
            tmp_path, EVENT, weekend, now=lambda: NOW - timedelta(hours=10), http_client=client
        )
        waiting = sprint_tick(tmp_path, EVENT, weekend, now=lambda: NOW, http_client=client)
        assert waiting["status"] == "waiting_for_sprint_results"
        later = NOW + FALLBACK_DELAY
        fallback = sprint_tick(tmp_path, EVENT, weekend, now=lambda: later, http_client=client)
    assert fallback["status"] == "sprint_captured"
    assert fallback["sprint_result_provider"] == "openf1"
    record = latest_capture(tmp_path, EVENT, SPRINT_RESULT, later)
    assert record is not None
    assert capture_points(record) == {
        "norris": 8.0,
        "piastri": 7.0,
        "max_verstappen": 0.0,
    }
    values = captured_sprint_values(None, record)
    assert values["piastri"]["sprint_position"] == 2
    assert values["piastri"]["sprint_classified"] == 1.0
    assert values["max_verstappen"]["sprint_position"] is None
    assert values["max_verstappen"]["sprint_classified"] == 0.0

    jolpica = {
        "MRData": {
            "total": "1",
            "RaceTable": {
                "Races": [
                    {
                        "season": "2026",
                        "round": "17",
                        "SprintResults": [
                            {
                                "number": "1",
                                "position": "1",
                                "positionText": "1",
                                "points": "8",
                                "grid": "1",
                                "laps": "20",
                                "status": "Finished",
                                "Driver": {"driverId": "norris"},
                                "Constructor": {"constructorId": "mclaren"},
                            }
                        ],
                    }
                ]
            },
        }
    }
    final = later + timedelta(hours=1)
    transport = _fallback_transport(sessions, {11379: _session_result()}, jolpica)
    with httpx.Client(transport=transport) as client:
        superseded = sprint_tick(tmp_path, EVENT, weekend, now=lambda: final, http_client=client)
    assert superseded["sprint_result_provider"] == "jolpica"
    # A cutoff before the Jolpica capture still sees the OpenF1 fallback.
    earlier = latest_capture(tmp_path, EVENT, SPRINT_RESULT, later)
    assert earlier is not None and earlier["provider"] == "openf1"


def test_qualifying_fallback_waits_for_the_delay_then_freezes_openf1(tmp_path: Path) -> None:
    qualifying = NOW - timedelta(hours=1)
    weekend = Weekend(
        EVENT,
        "Singapore Grand Prix",
        "marina_bay",
        "Marina Bay",
        qualifying - timedelta(days=1),
        qualifying,
        None,
        qualifying + timedelta(hours=22),
    )
    sessions = [_session(11384, "Qualifying", qualifying + timedelta(minutes=30), 60)]
    transport = _fallback_transport(sessions, {11384: _session_result()})
    with httpx.Client(transport=transport) as client:
        early = qualifying_fallback_tick(
            tmp_path, EVENT, weekend, now=lambda: NOW, http_client=client
        )
        assert early["status"] == "waiting_for_jolpica"
        later = qualifying + FALLBACK_DELAY
        captured = qualifying_fallback_tick(
            tmp_path, EVENT, weekend, now=lambda: later, http_client=client
        )
    assert captured["status"] == "openf1_captured"
    record = latest_capture(tmp_path, EVENT, QUALIFYING_OPENF1, later)
    assert record is not None and record["provider"] == "openf1"
    assert capture_grid(record).positions["norris"] == 1
    after_race = weekend.race + timedelta(minutes=1)  # type: ignore[operator]
    assert (
        qualifying_fallback_tick(tmp_path / "other", EVENT, weekend, now=lambda: after_race)[
            "status"
        ]
        == "missed"
    )
