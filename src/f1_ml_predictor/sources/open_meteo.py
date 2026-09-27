"""Forecast snapshots with explicit availability, including archived model runs."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from f1_ml_predictor.sources.http import JsonSourceClient, SourceError
from f1_ml_predictor.time import require_known_by, require_utc

HOURLY_VARIABLES = ("temperature_2m", "precipitation_probability", "wind_speed_10m")


@dataclass(frozen=True, slots=True)
class ForecastSnapshot:
    payload: dict[str, Any]
    captured_at: datetime
    available_at: datetime
    request_path: str
    run_initialized_at: datetime | None = None
    availability_evidence: str | None = None


class OpenMeteoClient(JsonSourceClient):
    def __init__(self, client: httpx.Client | None = None) -> None:
        super().__init__(((600, 60.0), (5000, 3600.0), (10000, 86400.0)), client)

    @staticmethod
    def _parameters(latitude: float, longitude: float) -> dict[str, str | int | float]:
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("invalid forecast coordinates")
        return {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            "forecast_days": 7,
        }

    def _fetch(
        self, url: str, parameters: dict[str, str | int | float]
    ) -> tuple[dict[str, Any], datetime, str]:
        payload = self.get_json(url, parameters)
        if not isinstance(payload, dict) or not isinstance(payload.get("hourly"), dict):
            raise SourceError("forecast response is missing hourly data")
        return payload, datetime.now(UTC), str(httpx.URL(url, params=parameters))

    def capture_forecast(self, latitude: float, longitude: float) -> ForecastSnapshot:
        payload, captured_at, path = self._fetch(
            "https://api.open-meteo.com/v1/forecast", self._parameters(latitude, longitude)
        )
        return ForecastSnapshot(payload, captured_at, captured_at, path)

    def historical_run(
        self,
        latitude: float,
        longitude: float,
        model: str,
        run_initialized_at: datetime,
        known_available_at: datetime,
        availability_evidence: str,
        prediction_timestamp: datetime,
    ) -> ForecastSnapshot:
        require_utc(run_initialized_at, "run_initialized_at")
        require_known_by(known_available_at, prediction_timestamp)
        if known_available_at < run_initialized_at:
            raise ValueError("forecast cannot be available before model initialization")
        if run_initialized_at.second or run_initialized_at.microsecond:
            raise ValueError("model initialization must have minute precision")
        if not model.strip() or not availability_evidence.strip():
            raise ValueError("model and availability evidence are required")
        parameters = self._parameters(latitude, longitude)
        parameters.update({"models": model, "run": run_initialized_at.strftime("%Y-%m-%dT%H:%M")})
        payload, captured_at, path = self._fetch(
            "https://single-runs-api.open-meteo.com/v1/forecast", parameters
        )
        return ForecastSnapshot(
            payload,
            captured_at,
            known_available_at,
            path,
            run_initialized_at,
            availability_evidence,
        )
