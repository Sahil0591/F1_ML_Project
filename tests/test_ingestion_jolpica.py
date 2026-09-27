from datetime import UTC, datetime
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest

from f1_ml_predictor.ingestion.jolpica import JolpicaSeasonIngestor
from f1_ml_predictor.paths import StoragePaths
from f1_ml_predictor.sources.jolpica import JolpicaClient


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.points = "25"

    def fetch_collection(self, path: str, table_key: str, collection_key: str) -> list[dict]:
        self.calls.append(path)
        race = {
            "season": "2024",
            "round": "1",
            "raceName": "Bahrain Grand Prix",
            "Circuit": {"circuitId": "bahrain"},
            "date": "2024-03-02",
            "time": "15:00:00Z",
        }
        if path.endswith("/qualifying/"):
            race["QualifyingResults"] = [
                {
                    "Driver": {"driverId": "verstappen"},
                    "Constructor": {"constructorId": "red_bull"},
                    "position": "1",
                }
            ]
        elif path.endswith("/results/"):
            race["Results"] = [
                {
                    "Driver": {"driverId": "verstappen"},
                    "Constructor": {"constructorId": "red_bull"},
                    "position": "1",
                    "points": self.points,
                }
            ]
        return [race]


def test_historical_season_resumes_from_cache_and_rewrites_only_changes(tmp_path: Path) -> None:
    client = FakeClient()
    paths = StoragePaths(tmp_path)
    ingest = JolpicaSeasonIngestor(paths, client, now=lambda: datetime(2026, 9, 27, tzinfo=UTC))

    first = ingest.ingest_season(2024)
    assert first.events == 1
    assert first.fetched_collections == 3
    assert first.written_partitions == 4
    assert len(client.calls) == 3

    second = ingest.ingest_season(2024)
    assert second.fetched_collections == 0
    assert second.cache_hits == 3
    assert second.skipped_partitions == 4
    assert len(client.calls) == 3

    client.points = "18"
    third = ingest.ingest_season(2024, refresh=True)
    assert third.fetched_collections == 3
    assert third.written_partitions == 2
    assert third.skipped_partitions == 2
    result_path = paths.normalized / "season=2024" / "round=01" / "results.parquet"
    assert pq.ParquetFile(result_path).read()["points"][0].as_py() == 18.0
    result_cache = paths.raw / "jolpica" / "season=2024" / "round=01" / "results"
    raw_versions = list(result_cache.glob("*.json"))
    assert len(raw_versions) == 3  # two versions plus latest manifest


def test_interrupted_ingestion_reuses_completed_requests(tmp_path: Path) -> None:
    class InterruptedClient(FakeClient):
        interrupted = False

        def fetch_collection(self, path: str, table_key: str, collection_key: str) -> list[dict]:
            if path.endswith("/results/") and not self.interrupted:
                self.interrupted = True
                raise RuntimeError("interrupted")
            return super().fetch_collection(path, table_key, collection_key)

    client = InterruptedClient()
    ingest = JolpicaSeasonIngestor(
        StoragePaths(tmp_path), client, now=lambda: datetime(2026, 9, 27, tzinfo=UTC)
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        ingest.ingest_season(2024)
    report = ingest.ingest_season(2024)
    assert report.cache_hits == 2
    assert report.fetched_collections == 1
    assert len(client.calls) == 3


def test_mocked_http_response_reaches_parquet_with_correct_endpoint(tmp_path: Path) -> None:
    fixture = FakeClient()

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path.startswith("/ergast/f1/2024/")
        path = request.url.path.removeprefix("/ergast/f1/")
        races = fixture.fetch_collection(path, "RaceTable", "Races")
        return httpx.Response(
            200,
            json={
                "MRData": {
                    "limit": "100",
                    "offset": "0",
                    "total": "1",
                    "RaceTable": {"Races": races},
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = JolpicaClient(http_client)
        report = JolpicaSeasonIngestor(
            StoragePaths(tmp_path), client, now=lambda: datetime(2026, 9, 27, tzinfo=UTC)
        ).ingest_season(2024)
    assert report.written_partitions == 4
    assert fixture.calls == ["2024/", "2024/1/qualifying/", "2024/1/results/"]
