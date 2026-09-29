import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.trust import collector, prospective
from f1_ml_predictor.trust.locking import advisory_lock
from f1_ml_predictor.trust.outcomes import DNF_TAXONOMY_VERSION, OUTCOME_SCHEMA
from f1_ml_predictor.trust.prospective import load_bundle
from f1_ml_predictor.trust.scheduler import (
    import_scheduler_outcomes,
    scheduler_status,
    scheduler_tick,
)


class Clock:
    def __init__(self) -> None:
        self.current = datetime.now(UTC).replace(microsecond=0)

    def now(self) -> datetime:
        return self.current

    def advance(self, **kwargs: Any) -> None:
        self.current += timedelta(**kwargs)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    controlled = Clock()

    class ControlledDatetime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return cls.fromtimestamp(controlled.current.timestamp(), tz)

    monkeypatch.setattr(collector, "datetime", ControlledDatetime)
    monkeypatch.setattr(prospective, "datetime", ControlledDatetime)
    return controlled


def response_payload(races: list[dict[str, Any]], *, total: int | None = None) -> dict[str, Any]:
    return {
        "MRData": {
            "limit": "100",
            "offset": "0",
            "total": str(len(races) if total is None else total),
            "RaceTable": {"Races": races},
        }
    }


def race(clock: Clock, *, offset: timedelta = timedelta(days=1), round_: int = 1) -> dict:
    start = clock.current + offset
    qualifying = start - timedelta(days=1, hours=2)
    return {
        "season": str(start.year),
        "round": str(round_),
        "raceName": "Prospective Grand Prix",
        "Circuit": {"circuitId": "test_circuit"},
        "date": start.date().isoformat(),
        "time": start.time().isoformat() + "Z",
        "Qualifying": {
            "date": qualifying.date().isoformat(),
            "time": qualifying.time().isoformat() + "Z",
        },
    }


def qualifying_payload(raw_race: dict) -> dict:
    return response_payload(
        [
            {
                **raw_race,
                "QualifyingResults": [
                    {
                        "position": str(position),
                        "Driver": {"driverId": driver},
                        "Constructor": {"constructorId": "team_a"},
                        "Q1": "1:30.1",
                    }
                    for position, driver in enumerate(("driver_a", "driver_b"), start=1)
                ],
            }
        ],
        total=2,
    )


def mock_client(
    raw_race: dict,
    *,
    qualifying: object | None = None,
    seen: list[httpx.Request] | None = None,
    schedule: list[dict] | None = None,
) -> httpx.Client:
    def response(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        assert request.url.host in {"api.jolpi.ca", "api.open-meteo.com"}
        assert "results" not in request.url.path
        if request.url.host == "api.open-meteo.com":
            return httpx.Response(200, json={"hourly": {"time": ["future"], "rain": [1]}})
        payload = (
            (qualifying_payload(raw_race) if qualifying is None else qualifying)
            if "qualifying" in request.url.path
            else response_payload([raw_race] if schedule is None else schedule)
        )
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(response))


def tick(tmp_path: Path, clock: Clock, client: httpx.Client, **kwargs: Any) -> dict:
    return scheduler_tick(
        tmp_path, now=clock.now, http_client=client, min_interval_seconds=60, **kwargs
    )


def active_entry(state: dict) -> dict:
    return state["events"][state["active_event"]]


def test_capture_uses_observed_decision_and_preserves_raw_evidence(
    tmp_path: Path, clock: Clock
) -> None:
    raw = race(clock)
    seen: list[httpx.Request] = []
    with mock_client(raw, seen=seen) as client:
        state = tick(tmp_path, clock, client)
    assert state["status"] == "captured"
    entry = active_entry(state)
    assert entry["qualifying_decision_at"] == clock.now().isoformat()
    assert entry["qualifying_decision_at"] != entry["qualifying_scheduled_start"]
    capture = entry["captures"][0]
    assert capture["capture_version"] == 1
    assert capture["normalization_required"] is False
    assert capture["gold_snapshot"] is True
    assert capture["evaluation_eligible"] is False
    manifest, tables = load_bundle(tmp_path / capture["bundle"])
    assert json.loads(tables["qualifying"]["payload_json"][0].as_py()) == qualifying_payload(raw)
    assert manifest["cutoff"] == manifest["captured_at"] == clock.now().isoformat()
    assert manifest["request_metadata"]["source_requests"]["qualifying"]["params"] == {
        "limit": 100,
        "offset": 0,
    }
    observation = json.loads((tmp_path / entry["qualifying_observation"]).read_text())
    canonical = json.dumps(observation["payload"], sort_keys=True, separators=(",", ":")).encode()
    assert observation["payload_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert observation["provider"] == "jolpica"
    assert len(seen) == 4
    assert scheduler_status(tmp_path) == state


def test_repeated_ticks_are_idempotent_and_explicit_versions_do_not_overwrite(
    tmp_path: Path, clock: Clock
) -> None:
    seen: list[httpx.Request] = []
    raw = race(clock)
    with mock_client(raw, seen=seen) as client:
        first = tick(tmp_path, clock, client)
        original = tmp_path / active_entry(first)["captures"][0]["bundle"] / "manifest.json"
        contents = original.read_bytes()
        assert tick(tmp_path, clock, client) == first
        assert len(seen) == 4
        clock.advance(seconds=61)
        assert len(active_entry(tick(tmp_path, clock, client))["captures"]) == 1
        assert len(seen) == 5
        second = tick(tmp_path, clock, client, new_capture=True)
    captures = active_entry(second)["captures"]
    assert [capture["capture_version"] for capture in captures] == [1, 2]
    assert len({capture["bundle"] for capture in captures}) == 2
    assert original.read_bytes() == contents
    assert len(list((tmp_path / "data/raw/prospective_scheduler/plans").rglob("*.json"))) == 2


def test_scheduled_start_without_results_does_not_claim_completion(
    tmp_path: Path, clock: Clock
) -> None:
    raw = race(clock)
    with mock_client(raw, qualifying=response_payload([])) as client:
        state = tick(tmp_path, clock, client)
    assert state["status"] == "waiting_for_qualifying_results"
    assert not active_entry(state)["captures"]
    assert "qualifying_decision_at" not in active_entry(state)
    assert not list(tmp_path.rglob("manifest.json"))


def test_no_qualifying_requests_before_scheduled_start(tmp_path: Path, clock: Clock) -> None:
    raw = race(clock, offset=timedelta(days=3))
    seen: list[httpx.Request] = []
    with mock_client(raw, seen=seen) as client:
        state = tick(tmp_path, clock, client)
    assert state["status"] == "waiting_for_qualifying"
    assert len(seen) == 1


def test_missed_race_is_recorded_and_next_race_discovered_without_backfill(
    tmp_path: Path, clock: Clock
) -> None:
    missed = race(clock, offset=timedelta(hours=1))
    with mock_client(missed, qualifying=response_payload([])) as client:
        first = tick(tmp_path, clock, client)
    first_name = first["active_event"]
    clock.advance(hours=2)
    upcoming = race(clock, offset=timedelta(days=3), round_=2)
    with mock_client(upcoming, schedule=[missed, upcoming]) as client:
        second = tick(tmp_path, clock, client)
    assert second["events"][first_name]["status"] == "missed"
    assert active_entry(second)["round"] == 2
    assert not list(tmp_path.rglob("manifest.json"))


@pytest.mark.parametrize("change", ["wrong_event", "duplicates", "truncated"])
def test_invalid_qualifying_cannot_trigger_capture(
    tmp_path: Path, clock: Clock, change: str
) -> None:
    raw = race(clock)
    payload = qualifying_payload(raw)
    if change == "wrong_event":
        payload["MRData"]["RaceTable"]["Races"][0]["round"] = "2"
    elif change == "duplicates":
        results = payload["MRData"]["RaceTable"]["Races"][0]["QualifyingResults"]
        results[1] = results[0]
    else:
        payload["MRData"]["total"] = "3"
    with mock_client(raw, qualifying=payload) as client:
        state = tick(tmp_path, clock, client)
    assert state["status"] == "error"
    assert not list(tmp_path.rglob("manifest.json"))


def test_pre_race_capture_requires_opt_in(tmp_path: Path, clock: Clock) -> None:
    raw = race(clock, offset=timedelta(minutes=30))
    with mock_client(raw) as client:
        first = tick(tmp_path, clock, client)
        clock.advance(seconds=61)
        default = tick(tmp_path, clock, client)
        assert len(active_entry(default)["captures"]) == 1
        clock.advance(seconds=61)
        opted_in = tick(tmp_path, clock, client, collect_pre_race=True)
    assert active_entry(first)["captures"][0]["cutoff_kind"] == "post_qualifying"
    assert [capture["cutoff_kind"] for capture in active_entry(opted_in)["captures"]] == [
        "post_qualifying",
        "pre_race",
    ]


def test_weather_is_optional_and_metadata_survives(tmp_path: Path, clock: Clock) -> None:
    request = {
        "name": "forecast",
        "role": "forecast",
        "url": "https://api.open-meteo.com/v1/forecast",
        "params": {"latitude": 51.5, "longitude": -0.1, "hourly": "rain"},
    }
    with mock_client(race(clock)) as client:
        state = tick(tmp_path, clock, client, forecast_request=request)
    capture = active_entry(state)["captures"][0]
    manifest, tables = load_bundle(tmp_path / capture["bundle"])
    metadata = manifest["request_metadata"]["source_requests"]["forecast"]
    assert metadata["params"] == request["params"]
    assert metadata["response_captured_at"] == clock.now().isoformat()
    assert json.loads(tables["forecast"]["payload_json"][0].as_py())["hourly"]["time"] == ["future"]


def test_failure_is_persistent_and_next_tick_can_retry(tmp_path: Path, clock: Clock) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(400))) as client:
        failure = tick(tmp_path, clock, client)
    assert failure["status"] == "error"
    assert scheduler_status(tmp_path)["error"]
    clock.advance(seconds=61)
    with mock_client(race(clock)) as client:
        success = tick(tmp_path, clock, client)
    assert success["status"] == "captured"
    assert "error" not in success


def test_unseen_race_during_outage_is_marked_missed_without_backfill(
    tmp_path: Path, clock: Clock
) -> None:
    missed = race(clock, offset=timedelta(days=2))
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(400))) as client:
        first = tick(tmp_path, clock, client)
    assert first["status"] == "error"
    clock.advance(days=3)
    upcoming = race(clock, offset=timedelta(days=3), round_=2)
    with mock_client(upcoming, schedule=[missed, upcoming]) as client:
        recovered = tick(tmp_path, clock, client)
    missed_id = f"season={clock.current.year}/round=01"
    assert recovered["events"][missed_id]["status"] == "missed"
    assert recovered["events"][missed_id]["captures"] == []
    assert recovered["missed_event_ids"] == [missed_id]
    assert not list(tmp_path.rglob("manifest.json"))


def test_error_retry_backoff_is_bounded_and_resets(tmp_path: Path, clock: Clock) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(400))) as client:
        first = tick(tmp_path, clock, client)
        assert first["consecutive_failures"] == 1
        assert datetime.fromisoformat(first["next_check_at"]) == clock.now() + timedelta(minutes=1)
        clock.advance(seconds=61)
        second = tick(tmp_path, clock, client)
        assert second["consecutive_failures"] == 2
        assert datetime.fromisoformat(second["next_check_at"]) == clock.now() + timedelta(minutes=2)
    clock.advance(minutes=2)
    with mock_client(race(clock)) as client:
        success = tick(tmp_path, clock, client)
    assert success["consecutive_failures"] == 0


def test_orphan_capture_is_recovered_without_another_capture(tmp_path: Path, clock: Clock) -> None:
    raw = race(clock)
    with mock_client(raw) as client:
        first = tick(tmp_path, clock, client)
    active_entry(first)["captures"] = []
    status_path = tmp_path / "data/raw/prospective_scheduler/status.json"
    status_path.write_text(json.dumps(first))
    clock.advance(seconds=61)
    seen: list[httpx.Request] = []
    with mock_client(raw, seen=seen) as client:
        recovered = tick(tmp_path, clock, client)
    assert recovered["status"] == "captured"
    assert len(active_entry(recovered)["captures"]) == 1
    assert len(seen) == 1


def outcome_file(tmp_path: Path, state: dict, clock: Clock) -> tuple[Path, str]:
    entry = active_entry(state)
    rows = []
    for position, driver in enumerate(("driver_a", "driver_b"), start=1):
        rows.append(
            {
                "season": entry["season"],
                "round": entry["round"],
                "driver_id": driver,
                "position": position,
                "classified": True,
                "winner": position == 1,
                "podium": True,
                "dnf": False,
                "dnf_category": "finished",
                "raw_status": "Finished",
                "taxonomy_version": DNF_TAXONOMY_VERSION,
                "final_audited": True,
                "label_available_at": clock.now(),
                "audit_reference": "fia:final-classification",
            }
        )
    path = tmp_path / "outcomes.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=OUTCOME_SCHEMA), path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def captured_state(tmp_path: Path, clock: Clock) -> dict:
    with mock_client(race(clock)) as client:
        return tick(tmp_path, clock, client)


def test_outcomes_are_hash_bound_complete_and_keep_eligibility_conservative(
    tmp_path: Path, clock: Clock
) -> None:
    state = captured_state(tmp_path, clock)
    clock.advance(days=2)
    path, digest = outcome_file(tmp_path, state, clock)
    record = import_scheduler_outcomes(tmp_path, path, expected_sha256=digest, now=clock.now)
    assert record["sha256"] == digest
    assert record["capture_manifest_sha256"] == [
        active_entry(state)["captures"][0]["manifest_sha256"]
    ]
    assert record["evaluation_eligible"] is True
    assert (tmp_path / record["path"]).read_bytes() == path.read_bytes()
    assert (
        import_scheduler_outcomes(tmp_path, path, expected_sha256=digest, now=clock.now) == record
    )
    assert len(active_entry(scheduler_status(tmp_path))["outcome_versions"]) == 1


@pytest.mark.parametrize("invalid", ["hash", "roster", "incomplete", "unaudited", "future"])
def test_outcome_import_rejects_hash_roster_and_audit_failures(
    tmp_path: Path, clock: Clock, invalid: str
) -> None:
    state = captured_state(tmp_path, clock)
    clock.advance(days=2)
    path, digest = outcome_file(tmp_path, state, clock)
    kwargs: dict[str, Any] = {}
    if invalid == "hash":
        digest = "0" * 64
    elif invalid == "roster":
        kwargs["field_roster"] = ["driver_a"]
    else:
        table = pq.read_table(path)
        rows = table.to_pylist()
        if invalid == "incomplete":
            rows.pop()
        elif invalid == "unaudited":
            rows[0]["final_audited"] = False
        else:
            rows[0]["label_available_at"] = clock.now() + timedelta(hours=1)
        pq.write_table(pa.Table.from_pylist(rows, schema=OUTCOME_SCHEMA), path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        import_scheduler_outcomes(tmp_path, path, expected_sha256=digest, now=clock.now, **kwargs)
    assert "outcome_versions" not in active_entry(scheduler_status(tmp_path))


def test_invalid_historical_season_is_rejected_before_network(tmp_path: Path, clock: Clock) -> None:
    def unexpected(_: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected request")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(ValueError, match="current or next season"):
            tick(tmp_path, clock, client, season=clock.current.year - 1)


def test_concurrent_tick_is_rejected(tmp_path: Path, clock: Clock) -> None:
    lock = tmp_path / "data/raw/prospective_scheduler/.tick.lock"
    with advisory_lock(lock):
        with pytest.raises(ValueError, match="locked"):
            scheduler_tick(tmp_path, now=clock.now)


def test_missing_schedule_time_stays_conservative(tmp_path: Path, clock: Clock) -> None:
    raw = race(clock)
    del raw["Qualifying"]["time"]
    with mock_client(raw) as client:
        state = tick(tmp_path, clock, client)
    assert state["status"] == "qualifying_schedule_missing"
    assert not active_entry(state)["captures"]


def test_malformed_qualifying_schedule_is_persisted_as_error(tmp_path: Path, clock: Clock) -> None:
    raw = race(clock)
    del raw["Qualifying"]["date"]
    with mock_client(raw) as client:
        state = tick(tmp_path, clock, client)
    assert state["status"] == "error"
    assert scheduler_status(tmp_path)["error"]
    assert not list(tmp_path.rglob("manifest.json"))


def test_normalization_failure_keeps_raw_capture_without_false_gold(
    tmp_path: Path, clock: Clock
) -> None:
    raw = race(clock)
    payload = qualifying_payload(raw)
    del payload["MRData"]["RaceTable"]["Races"][0]["QualifyingResults"][0]["Constructor"]
    with mock_client(raw, qualifying=payload) as client:
        state = tick(tmp_path, clock, client)
    assert state["status"] == "error"
    assert not active_entry(state)["captures"]
    assert len(list(tmp_path.rglob("manifest.json"))) == 1
    assert not (tmp_path / "data/benchmarks/prospective_registry.json").exists()
    clock.advance(seconds=61)
    seen: list[httpx.Request] = []
    with mock_client(raw, seen=seen) as client:
        retry = tick(tmp_path, clock, client)
    assert retry["status"] == "error"
    assert not active_entry(retry)["captures"]
    assert not seen
