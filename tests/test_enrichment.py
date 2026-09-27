from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.ingestion.cache import RawCache
from f1_ml_predictor.ingestion.enrichment import (
    driver_crosswalk,
    ingest_openf1_session,
    persist_forecast,
    record_fia_evidence,
)
from f1_ml_predictor.normalization.sessions import normalize_summaries, summarize_openf1
from f1_ml_predictor.paths import StoragePaths
from f1_ml_predictor.sources.http import SourceError
from f1_ml_predictor.sources.open_meteo import ForecastSnapshot
from f1_ml_predictor.sources.openf1 import OpenF1Client

EVENT = EventId(2025, 1)
CAPTURE = datetime(2026, 9, 28, tzinfo=UTC)


def seed_event(paths: StoragePaths) -> None:
    cache = RawCache(paths.raw / "jolpica")
    cache.save("season=2025/schedule", [{"round": "1", "date": "2025-03-16"}], CAPTURE, "2025/")
    cache.save(
        f"{EVENT.partition()}/qualifying",
        [{"QualifyingResults": [{"number": "1", "Driver": {"driverId": "max_verstappen"}}]}],
        CAPTURE,
        "2025/1/qualifying/",
    )


class SessionClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def collection(self, endpoint: str, session_key: int) -> list[dict]:
        self.calls.append(endpoint)
        if endpoint == "sessions":
            return [
                {
                    "session_key": session_key,
                    "session_name": "Qualifying",
                    "date_end": "2025-03-15T06:00:00+00:00",
                }
            ]
        if endpoint == "laps":
            return [
                {
                    "session_key": session_key,
                    "driver_number": 1,
                    "lap_number": 1,
                    "lap_duration": 90.0,
                },
                {
                    "session_key": session_key,
                    "driver_number": 1,
                    "lap_number": 2,
                    "lap_duration": 91.0,
                },
                {
                    "session_key": session_key,
                    "driver_number": 1,
                    "lap_number": 3,
                    "lap_duration": 92.0,
                },
                {
                    "session_key": session_key,
                    "driver_number": 99,
                    "lap_number": 1,
                    "lap_duration": 100.0,
                },
            ]
        if endpoint == "stints":
            return [
                {
                    "session_key": session_key,
                    "driver_number": 1,
                    "lap_start": 1,
                    "lap_end": 3,
                    "compound": "SOFT",
                    "tyre_age_at_start": 0,
                }
            ]
        return [{"session_key": session_key, "driver_number": 1, "lap_number": 3}]


def test_session_ingestion_maps_ids_preserves_capture_and_reuses_cache(tmp_path: Path) -> None:
    paths = StoragePaths(tmp_path)
    seed_event(paths)
    client = SessionClient()
    first = ingest_openf1_session(paths, EVENT, 100, client)
    assert first.rows == 1
    assert first.unmapped_laps == 1
    table = pq.ParquetFile(first.path).read()
    assert table["driver_id"][0].as_py() == "max_verstappen"
    assert table["lap_count"][0].as_py() == 2
    assert table["median_lap_seconds"][0].as_py() == 90.5
    assert table["available_at"][0].as_py().year == 2026
    second = ingest_openf1_session(paths, EVENT, 100, client)
    assert not second.written
    assert second.path == first.path
    assert client.calls == ["sessions", "laps", "stints", "pit"]


def test_source_mapping_requires_qualifying_crosswalk(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="driver numbers"):
        driver_crosswalk(StoragePaths(tmp_path), EVENT)


def test_openf1_rejects_telemetry_endpoint_and_malformed_data() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))
    ) as http_client:
        client = OpenF1Client(http_client)
        with pytest.raises(ValueError, match="lightweight"):
            client.collection("car_data", 100)
        with pytest.raises(SourceError, match="list of objects"):
            client.collection("laps", 100)


def test_session_discovery_uses_year_filter() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.params["year"] == "2025"
        return httpx.Response(200, json=[{"session_key": 100}])

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        assert OpenF1Client(http_client).sessions(2025) == [{"session_key": 100}]


@pytest.mark.parametrize(
    "name,end",
    [
        ("Race", "2025-03-16T06:00:00+00:00"),
        ("Qualifying", "2099-03-15T06:00:00+00:00"),
    ],
)
def test_race_or_future_session_is_rejected(name: str, end: str) -> None:
    with pytest.raises(ValueError):
        summarize_openf1(EVENT, {"session_name": name, "date_end": end}, [], [], [], {}, CAPTURE)


def test_missing_stint_range_does_not_invent_tyre_values() -> None:
    rows = summarize_openf1(
        EVENT,
        {"session_name": "Practice 1", "date_end": "2025-03-14T06:00:00+00:00"},
        [{"driver_number": 1, "lap_number": 1, "lap_duration": 90.0}],
        [{"driver_number": 1, "lap_start": 1, "lap_end": None, "compound": "SOFT"}],
        [],
        {"1": "max_verstappen"},
        CAPTURE,
    )
    assert rows[0]["compound"] is None
    assert rows[0]["median_tyre_age"] is None
    assert normalize_summaries(rows, "openf1").num_rows == 1


@pytest.mark.parametrize("collection", ["laps", "stints", "pits"])
def test_mixed_session_records_are_rejected(collection: str) -> None:
    inputs = {"laps": [], "stints": [], "pits": []}
    inputs[collection] = [{"session_key": 101}]
    with pytest.raises(ValueError, match="selected session"):
        summarize_openf1(
            EVENT,
            {
                "session_key": 100,
                "session_name": "Qualifying",
                "date_end": "2025-03-15T06:00:00+00:00",
            },
            inputs["laps"],
            inputs["stints"],
            inputs["pits"],
            {},
            CAPTURE,
        )


def test_lap_after_session_end_is_rejected_even_when_captured_later() -> None:
    with pytest.raises(ValueError, match="after prediction_timestamp"):
        summarize_openf1(
            EVENT,
            {"session_name": "Qualifying", "date_end": "2025-03-15T06:00:00+00:00"},
            [
                {
                    "driver_number": 1,
                    "lap_number": 1,
                    "lap_duration": 90.0,
                    "date_start": "2025-03-16T06:00:00+00:00",
                }
            ],
            [],
            [],
            {"1": "max_verstappen"},
            CAPTURE,
        )


@pytest.mark.parametrize("age", [float("nan"), float("inf"), -1, True])
def test_invalid_tyre_age_is_rejected(age: float) -> None:
    with pytest.raises(ValueError, match="tyre age"):
        summarize_openf1(
            EVENT,
            {"session_name": "Qualifying", "date_end": "2025-03-15T06:00:00+00:00"},
            [{"driver_number": 1, "lap_number": 1, "lap_duration": 90.0}],
            [
                {
                    "driver_number": 1,
                    "lap_start": 1,
                    "lap_end": 1,
                    "compound": "SOFT",
                    "tyre_age_at_start": age,
                }
            ],
            [],
            {"1": "max_verstappen"},
            CAPTURE,
        )


def test_forecast_persistence_keeps_first_availability_for_unchanged_payload(
    tmp_path: Path,
) -> None:
    payload = {
        "hourly": {
            "time": ["2026-09-29T12:00"],
            "temperature_2m": [20.0],
            "precipitation_probability": [25],
            "wind_speed_10m": [3.0],
        }
    }
    first = ForecastSnapshot(payload, CAPTURE, CAPTURE, "https://api.open-meteo.com/v1/forecast")
    later = ForecastSnapshot(
        payload, CAPTURE + timedelta(hours=1), CAPTURE + timedelta(hours=1), first.request_path
    )
    paths = StoragePaths(tmp_path)
    report = persist_forecast(paths, EventId(2026, 18), first)
    repeated = persist_forecast(paths, EventId(2026, 18), later)
    assert not repeated.written
    assert repeated.path == report.path
    assert pq.ParquetFile(report.path).read()["available_at"][0].as_py() == CAPTURE


def test_fia_evidence_requires_official_link_and_nonfuture_publication(tmp_path: Path) -> None:
    paths = StoragePaths(tmp_path)
    with pytest.raises(ValueError, match="official"):
        record_fia_evidence(
            paths, EVENT, "https://example.com/grid.pdf", "grid", datetime(2025, 3, 15, tzinfo=UTC)
        )
    with pytest.raises(ValueError, match="after prediction_timestamp"):
        record_fia_evidence(
            paths, EVENT, "https://www.fia.com/grid.pdf", "grid", datetime(2099, 1, 1, tzinfo=UTC)
        )
    record_fia_evidence(
        paths, EVENT, "https://www.fia.com/grid.pdf", "grid", datetime(2025, 3, 15, tzinfo=UTC)
    )
    stored = RawCache(paths.raw / "fia").load(f"{EVENT.partition()}/grid")
    assert stored is not None
    assert stored.items[0]["published_at"].startswith("2025-03-15")
