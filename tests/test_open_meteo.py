from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.normalization.weather import normalize_forecast
from f1_ml_predictor.sources.http import SourceError
from f1_ml_predictor.sources.open_meteo import HOURLY_VARIABLES, OpenMeteoClient


def _payload() -> dict[str, object]:
    return {
        "hourly": {
            "time": ["2030-07-01T12:00", "2030-07-01T13:00"],
            "temperature_2m": [21.5, None],
            "precipitation_probability": [10, None],
            "wind_speed_10m": [3.2, 4.0],
        }
    }


def test_live_capture_requests_expected_endpoint_parameters_and_availability() -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_payload())

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        snapshot = OpenMeteoClient(http).capture_forecast(51.5, -0.12)

    assert len(seen) == 1
    request = seen[0]
    assert request.url.scheme == "https"
    assert request.url.host == "api.open-meteo.com"
    assert request.url.path == "/v1/forecast"
    assert dict(request.url.params) == {
        "latitude": "51.5",
        "longitude": "-0.12",
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "forecast_days": "7",
    }
    assert snapshot.request_path == str(request.url)
    assert snapshot.captured_at.tzinfo is UTC
    assert snapshot.available_at == snapshot.captured_at
    assert snapshot.run_initialized_at is None
    assert snapshot.availability_evidence is None


def test_historical_run_records_required_availability_proof_and_request() -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_payload())

    initialized = datetime(2030, 7, 1, 6, tzinfo=UTC)
    available = datetime(2030, 7, 1, 7, tzinfo=UTC)
    prediction = datetime(2030, 7, 1, 8, tzinfo=UTC)
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        snapshot = OpenMeteoClient(http).historical_run(
            51.5,
            -0.12,
            "ecmwf_ifs025",
            initialized,
            available,
            "archived bulletin",
            prediction,
        )

    request = seen[0]
    assert request.url.host == "single-runs-api.open-meteo.com"
    assert request.url.path == "/v1/forecast"
    assert request.url.params["models"] == "ecmwf_ifs025"
    assert request.url.params["run"] == "2030-07-01T06:00"
    assert snapshot.available_at == available
    assert snapshot.run_initialized_at == initialized
    assert snapshot.availability_evidence == "archived bulletin"


@pytest.mark.parametrize(
    ("run", "available", "evidence", "prediction"),
    [
        (
            datetime(2030, 7, 1, 6, tzinfo=UTC),
            datetime(2030, 7, 1, 7, tzinfo=UTC),
            "proof",
            datetime(2030, 7, 1, 6, tzinfo=UTC),
        ),
        (
            datetime(2030, 7, 1, 6, tzinfo=UTC),
            datetime(2030, 7, 1, 5, tzinfo=UTC),
            "proof",
            datetime(2030, 7, 1, 8, tzinfo=UTC),
        ),
        (
            datetime(2030, 7, 1, 6, tzinfo=UTC),
            datetime(2030, 7, 1, 7, tzinfo=UTC),
            "",
            datetime(2030, 7, 1, 8, tzinfo=UTC),
        ),
        (
            datetime(2030, 7, 1, 6, 0, 1, tzinfo=UTC),
            datetime(2030, 7, 1, 7, tzinfo=UTC),
            "proof",
            datetime(2030, 7, 1, 8, tzinfo=UTC),
        ),
    ],
)
def test_historical_run_rejects_cutoff_or_missing_proof(
    run: datetime,
    available: datetime,
    evidence: str,
    prediction: datetime,
) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=_payload()))
    ) as http:
        client = OpenMeteoClient(http)
        with pytest.raises(ValueError):
            client.historical_run(51.5, -0.12, "ecmwf", run, available, evidence, prediction)


def test_normalization_keeps_future_targets_and_missing_numeric_values() -> None:
    captured = datetime(2030, 7, 1, 8, 1, tzinfo=UTC)
    from f1_ml_predictor.sources.open_meteo import ForecastSnapshot

    snapshot = ForecastSnapshot(_payload(), captured, captured, "mock")
    table = normalize_forecast(EventId(2030, 4), snapshot)
    assert table.num_rows == 2
    assert table.column("event_id").to_pylist() == ["season=2030/round=04"] * 2
    assert table.column("valid_at")[0].value > int(captured.timestamp() * 1_000_000)
    assert table.column("temperature_2m").to_pylist() == [21.5, None]
    assert table.column("precipitation_probability").to_pylist() == [10.0, None]


def test_live_forecast_cannot_be_backdated_without_archived_run_evidence() -> None:
    from f1_ml_predictor.sources.open_meteo import ForecastSnapshot

    captured = datetime(2030, 7, 1, 8, tzinfo=UTC)
    available = datetime(2029, 7, 1, 8, tzinfo=UTC)
    with pytest.raises(ValueError, match="live forecast availability"):
        normalize_forecast(
            EventId(2030, 4), ForecastSnapshot(_payload(), captured, available, "mock")
        )


@pytest.mark.parametrize(
    "hourly",
    [
        {
            "time": ["2030-07-01T12:00"],
            "temperature_2m": [],
            "precipitation_probability": [0],
            "wind_speed_10m": [1],
        },
        {
            "time": ["2030-07-01T12:00"],
            "temperature_2m": [True],
            "precipitation_probability": [0],
            "wind_speed_10m": [1],
        },
    ],
)
def test_normalization_rejects_wrong_array_lengths_and_boolean_values(
    hourly: dict[str, list[object]],
) -> None:
    from f1_ml_predictor.sources.open_meteo import ForecastSnapshot

    payload = {"hourly": hourly}
    captured = datetime(2030, 7, 1, 8, tzinfo=UTC)
    snapshot = ForecastSnapshot(payload, captured, captured, "mock")
    with pytest.raises(ValueError):
        normalize_forecast(EventId(2030, 4), snapshot)


def test_missing_hourly_object_raises_source_error() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={}))
    ) as http:
        with pytest.raises(SourceError, match="missing hourly"):
            OpenMeteoClient(http).capture_forecast(51.5, -0.12)
