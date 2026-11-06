"""Archived forecast run choice, race-window values and the weather audit replay."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from f1_ml_predictor.prediction.contracts import WEATHER_NUMERIC, with_weather
from f1_ml_predictor.trust.weather_history import (
    RELEASE_MARGIN,
    audit_historical_weather,
    forecast_run,
    load_weather_forecasts,
    race_weather,
)

START = datetime(2025, 3, 16, 4, 0, tzinfo=UTC)


def _payload(start: datetime, rain: list[float | None]) -> dict:
    """Twelve hourly values from eight hours before the start.

    ``rain`` covers the valid times from two hours before to three hours after.
    """
    times = [start - timedelta(hours=8) + timedelta(hours=hour) for hour in range(12)]
    return {
        "utc_offset_seconds": 0,
        "hourly": {
            "time": [at.strftime("%Y-%m-%dT%H:%M") for at in times],
            "temperature_2m": [10.0] * 8 + [17.0, 18.0, 19.0, 20.0],
            "precipitation": [0.0] * 6 + rain,
            "wind_speed_10m": [1.0] * 8 + [7.5, 4.0, 5.0, 6.0],
        },
    }


def test_run_is_the_latest_main_run_released_with_margin() -> None:
    cutoff = datetime(2025, 3, 16, 1, 5, tzinfo=UTC)
    run = forecast_run(cutoff)
    assert run == datetime(2025, 3, 15, 12, 0, tzinfo=UTC)
    assert run + RELEASE_MARGIN <= cutoff
    assert forecast_run(datetime(2025, 3, 16, 12, 0, tzinfo=UTC)) == datetime(
        2025, 3, 16, 0, 0, tzinfo=UTC
    )


def test_race_window_sums_the_hours_after_the_start() -> None:
    values = race_weather(_payload(START, [9.0, 9.0, 0.5, 1.5, 2.0, 9.0]), START)
    # Valid times start+1h and start+2h hold the rain that fell during the race;
    # the six valid times up to the start hold the rain that soaked the track.
    assert values == {
        "forecast_temperature_2m": 17.0,
        "forecast_wind_speed_10m": 7.5,
        "forecast_race_precipitation_mm": 3.5,
        "forecast_pre_race_precipitation_mm": 18.5,
    }
    gap = race_weather(_payload(START, [0.0, 0.0, 0.0, None, 0.0, 0.0]), START)
    assert gap["forecast_race_precipitation_mm"] is None
    assert gap["forecast_pre_race_precipitation_mm"] == 0.0


def test_audit_replays_retained_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schedules = tmp_path / "data/raw/historical_schedules"
    schedules.mkdir(parents=True)
    races = [
        {"season": season, "round": "1", "Circuit": {"Location": {"lat": "1.0", "long": "2.0"}}}
        for season in ("2023", "2025")
    ]
    for season, race in zip((2023, 2025), races, strict=True):
        record = {"payload": {"MRData": {"RaceTable": {"Races": [race]}}}}
        (schedules / f"season={season}.json").write_text(json.dumps(record), encoding="utf-8")
    old_start = datetime(2023, 3, 5, 15, 0, tzinfo=UTC)
    events = {
        "season=2023/round=01": (old_start - timedelta(hours=20), old_start),
        "season=2025/round=01": (START - timedelta(hours=20), START),
    }
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["run"])
        return httpx.Response(200, json=_payload(START, [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = audit_historical_weather(tmp_path, events, http_client=client)
    assert report["status"] == {"before_archive": 1, "audited": 1}
    assert seen == ["2025-03-14T12:00"]
    loaded = load_weather_forecasts(tmp_path, {key: value[0] for key, value in events.items()})
    assert loaded is not None
    records, _ = loaded
    assert records["season=2025/round=01"]["values"]["forecast_race_precipitation_mm"] == 2.0
    assert records["season=2025/round=01"]["available_at"] <= events["season=2025/round=01"][0]
    with pytest.raises(ValueError, match="another cutoff"):
        load_weather_forecasts(
            tmp_path, {key: value[0] - timedelta(hours=1) for key, value in events.items()}
        )


def test_weather_reaches_every_driver_only_when_audited() -> None:
    weekend = {"a": {"qualifying_position": 1.0}, "b": {"qualifying_position": 2.0}}
    available = datetime(2025, 3, 15, 0, 0, tzinfo=UTC)
    record = {
        "status": "audited",
        "available_at": available,
        "values": dict.fromkeys(WEATHER_NUMERIC, 1.0),
    }
    merged = with_weather(weekend, record)
    for driver in weekend:
        for name in WEATHER_NUMERIC:
            assert merged[driver][name] == 1.0
            assert merged[driver][f"{name}_available_at"] == available
    assert with_weather(weekend, {"status": "before_archive"}) is weekend
