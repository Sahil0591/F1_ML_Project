import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from f1_ml_predictor.trust.collector import collect_weekend
from f1_ml_predictor.trust.prospective import load_bundle


def plan_file(
    tmp_path: Path,
    *,
    url: str = "https://api.open-meteo.com/v1/forecast",
    role: str = "forecast",
    race_offset: int = 1,
) -> Path:
    now = datetime.now(UTC)
    race = now + timedelta(days=race_offset)
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "season": race.year,
                "round": 1,
                "race_start": race.isoformat(),
                "qualifying_decision_at": (now - timedelta(hours=2)).isoformat(),
                "cutoff_kind": "post_qualifying",
                "requests": [
                    {
                        "name": "forecast",
                        "role": role,
                        "url": url,
                        "params": {"latitude": 0, "longitude": 0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_fresh_mocked_response_is_frozen_without_historical_backdating(tmp_path: Path) -> None:
    payload = {"hourly": {"time": ["future-target"], "temperature_2m": [20]}}
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        bundle = collect_weekend(plan_file(tmp_path), tmp_path, http_client=client)
    manifest, tables = load_bundle(bundle)
    assert json.loads(tables["forecast"]["payload_json"][0].as_py()) == payload
    assert manifest["cutoff"] == manifest["captured_at"]
    assert manifest["inputs"]["forecast"]["evidence"]["class"] == "captured_live"
    assert manifest["request_metadata"]["source_requests"]["forecast"]["role"] == "forecast"
    assert "window" in manifest["request_metadata"]


@pytest.mark.parametrize(
    "url,role",
    [
        ("https://example.com/api", "forecast"),
        ("https://api.openf1.org/v1/car_data", "session_laps"),
        ("https://api.openf1.org/v1/weather", "forecast"),
        ("https://api.jolpi.ca/ergast/f1/2025/1/results/", "qualifying"),
        ("https://secret@api.open-meteo.com/v1/forecast", "forecast"),
    ],
)
def test_unsafe_or_outcome_endpoints_are_rejected_before_request(
    tmp_path: Path, url: str, role: str
) -> None:
    def unexpected(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected external request")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(ValueError):
            collect_weekend(plan_file(tmp_path, url=url, role=role), tmp_path, http_client=client)


def test_missed_past_race_cannot_be_captured_as_prospective(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="before race start"):
        collect_weekend(plan_file(tmp_path, race_offset=-1), tmp_path)


def replace_requests(path: Path, requests: object) -> Path:
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["requests"] = requests
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


@pytest.mark.parametrize("requests", [None, {}, [], [None], [[]], ["request"], [3]])
def test_malformed_request_lists_fail_before_network(tmp_path: Path, requests: object) -> None:
    path = replace_requests(plan_file(tmp_path), requests)

    def unexpected(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected request")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(ValueError):
            collect_weekend(path, tmp_path, http_client=client)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": None},
        {"name": []},
        {"name": ""},
        {"name": "window"},
        {"name": "source_requests"},
        {"params": None},
        {"params": []},
        {"params": {"latitude": [0]}},
        {"params": {"latitude": {"value": 0}}},
        {"params": {"api_key": "secret"}},
        {"params": {"Authorization": "secret"}},
        {"params": {"latitude": float("nan")}},
        {"role": []},
        {"url": None},
    ],
)
def test_malformed_request_fields_fail_before_network(tmp_path: Path, changes: dict) -> None:
    path = plan_file(tmp_path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["requests"][0].update(changes)
    replace_requests(path, plan["requests"])

    def unexpected(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected request")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(ValueError):
            collect_weekend(path, tmp_path, http_client=client)


def test_request_named_like_session_cannot_authorize_race_laps(tmp_path: Path) -> None:
    path = plan_file(tmp_path)
    season = json.loads(path.read_text(encoding="utf-8"))["season"]
    replace_requests(
        path,
        [
            {
                "name": "session_123",
                "role": "event",
                "url": f"https://api.jolpi.ca/ergast/f1/{season}",
            },
            {
                "name": "race_laps",
                "role": "session_laps",
                "url": "https://api.openf1.org/v1/laps",
                "params": {"session_key": 123},
            },
        ],
    )
    seen = []

    def response(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        assert request.url.host == "api.jolpi.ca", "unvalidated race laps were requested"
        return httpx.Response(200, json={"MRData": {}})

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(ValueError, match="session metadata"):
            collect_weekend(path, tmp_path, http_client=client)
    assert seen == ["api.jolpi.ca"]
    assert not list(tmp_path.rglob("manifest.json"))


@pytest.mark.parametrize(
    "date_end", [None, [], "bad-date", "2026-09-28T01:00:00", "2026-09-28T01:00:00+01:00"]
)
def test_session_end_must_be_a_valid_aware_utc_timestamp(tmp_path: Path, date_end: object) -> None:
    path = replace_requests(
        plan_file(tmp_path),
        [
            {
                "name": "session",
                "role": "session_metadata",
                "url": "https://api.openf1.org/v1/sessions",
                "params": {"session_key": 123},
            }
        ],
    )
    payload = [{"session_key": 123, "session_name": "Qualifying", "date_end": date_end}]
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(ValueError):
            collect_weekend(path, tmp_path, http_client=client)


@pytest.mark.parametrize("payload", [None, [], 3, "wrong"])
def test_malformed_forecast_response_fails_as_value_error(tmp_path: Path, payload: object) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=json.dumps(payload))
        )
    ) as client:
        with pytest.raises(ValueError):
            collect_weekend(plan_file(tmp_path), tmp_path, http_client=client)


def test_validated_metadata_allows_dependents_and_is_not_request_state(tmp_path: Path) -> None:
    path = plan_file(tmp_path)
    ended = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    replace_requests(
        path,
        [
            {
                "name": "laps",
                "role": "session_laps",
                "url": "https://api.openf1.org/v1/laps",
                "params": {"session_key": "123"},
            },
            {
                "name": "session_123",
                "role": "session_metadata",
                "url": "https://api.openf1.org/v1/sessions",
                "params": {"session_key": 123},
            },
        ],
    )

    def response(request: httpx.Request) -> httpx.Response:
        rows = (
            [{"session_key": 123, "session_name": "Qualifying", "date_end": ended}]
            if request.url.path == "/v1/sessions"
            else [{"session_key": 123, "lap_number": 1}]
        )
        return httpx.Response(200, json=rows)

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        bundle = collect_weekend(path, tmp_path, http_client=client)
    manifest, tables = load_bundle(bundle)
    assert set(tables) == {"session_123", "laps"}
    assert set(manifest["request_metadata"]) == {"source_requests", "window"}
    assert set(manifest["request_metadata"]["source_requests"]) == {"session_123", "laps"}
