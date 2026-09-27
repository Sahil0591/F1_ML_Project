"""Resumable Jolpica season ingestion into raw and normalized zones."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.ingestion.cache import CachedCollection, RawCache
from f1_ml_predictor.normalization.jolpica import (
    normalize_entries,
    normalize_qualifying,
    normalize_results,
    normalize_schedule,
)
from f1_ml_predictor.paths import StoragePaths
from f1_ml_predictor.sources.jolpica import JolpicaClient

SCHEMA_VERSION = "1"


@dataclass(slots=True)
class IngestReport:
    season: int
    events: int = 0
    fetched_collections: int = 0
    cache_hits: int = 0
    written_partitions: int = 0
    skipped_partitions: int = 0


class JolpicaSeasonIngestor:
    def __init__(
        self,
        paths: StoragePaths,
        client: JolpicaClient,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.paths = paths
        self.client = client
        self.cache = RawCache(paths.raw / "jolpica")
        self.now = now

    def _collection(
        self, key: str, request_path: str, refresh: bool, report: IngestReport
    ) -> CachedCollection:
        cached = self.cache.load(key)
        if cached is not None and not refresh:
            report.cache_hits += 1
            return cached
        items = self.client.fetch_collection(request_path, "RaceTable", "Races")
        report.fetched_collections += 1
        return self.cache.save(key, items, self.now(), request_path)

    def ingest_season(self, season: int, *, refresh: bool = False) -> IngestReport:
        EventId(season, 1)
        report = IngestReport(season)
        current_or_future = season >= self.now().year
        should_refresh = refresh or current_or_future
        prefix = f"season={season}"
        schedule = self._collection(f"{prefix}/schedule", f"{season}/", should_refresh, report)
        events = normalize_schedule(schedule.items, season)
        report.events = events.num_rows
        self._write_partition(
            events,
            self.paths.normalized / "events" / prefix / "data.parquet",
            schedule.sha256,
            report,
        )
        for row in events.select(["round"]).to_pylist():
            event = EventId(season, row["round"])
            event_prefix = f"{prefix}/round={event.round:02d}"
            qualifying = self._collection(
                f"{event_prefix}/qualifying",
                f"{season}/{event.round}/qualifying/",
                should_refresh,
                report,
            )
            results = self._collection(
                f"{event_prefix}/results",
                f"{season}/{event.round}/results/",
                should_refresh,
                report,
            )
            qualifying_table = normalize_qualifying(qualifying.items, event)
            results_table = normalize_results(results.items, event)
            entries_table = normalize_entries(qualifying_table, results_table)
            base = self.paths.normalized / event_prefix
            self._write_partition(
                qualifying_table, base / "qualifying.parquet", qualifying.sha256, report
            )
            self._write_partition(results_table, base / "results.parquet", results.sha256, report)
            combined_hash = hashlib.sha256(
                f"{qualifying.sha256}:{results.sha256}".encode("ascii")
            ).hexdigest()
            self._write_partition(entries_table, base / "entries.parquet", combined_hash, report)
        return report

    @staticmethod
    def _write_partition(
        table: pa.Table, path: Path, source_hash: str, report: IngestReport
    ) -> None:
        metadata = {
            b"schema_version": SCHEMA_VERSION.encode(),
            b"source_sha256": source_hash.encode(),
        }
        if path.exists() and pq.read_schema(path).metadata == metadata:
            report.skipped_partitions += 1
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".pending-", suffix=".parquet", dir=path.parent)
        os.close(fd)
        try:
            pq.write_table(table.replace_schema_metadata(metadata), temp_name)
            os.replace(temp_name, path)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        report.written_partitions += 1
