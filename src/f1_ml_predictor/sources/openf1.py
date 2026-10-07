"""Bounded historical OpenF1 session access without car telemetry."""

from typing import Any

import httpx

from f1_ml_predictor.sources.http import JsonSourceClient, SourceError


class OpenF1Client(JsonSourceClient):
    def __init__(self, client: httpx.Client | None = None) -> None:
        super().__init__(((3, 1.0), (30, 60.0)), client)

    def sessions(self, season: int) -> list[dict[str, Any]]:
        if isinstance(season, bool) or not isinstance(season, int) or season < 2023:
            raise ValueError("OpenF1 historical coverage starts in 2023")
        payload = self.get_json("https://api.openf1.org/v1/sessions", {"year": season})
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise SourceError("OpenF1 response must be a list of objects")
        return payload

    def collection(self, endpoint: str, session_key: int) -> list[dict[str, Any]]:
        if endpoint not in {
            "sessions",
            "laps",
            "stints",
            "pit",
            "weather",
            "session_result",
            "drivers",
        }:
            raise ValueError("unsupported lightweight OpenF1 endpoint")
        if isinstance(session_key, bool) or not isinstance(session_key, int) or session_key < 1:
            raise ValueError("session_key must be positive")
        payload = self.get_json(
            f"https://api.openf1.org/v1/{endpoint}", {"session_key": session_key}
        )
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise SourceError("OpenF1 response must be a list of objects")
        return payload
