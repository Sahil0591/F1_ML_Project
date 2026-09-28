"""Content-addressed storage for reproducible feature snapshots."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa

from f1_ml_predictor.features.snapshot import FEATURE_SCHEMA, LEGACY_FEATURE_SCHEMA
from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.ingestion.parquet import write_partition
from f1_ml_predictor.paths import StoragePaths
from f1_ml_predictor.time import require_known_by


def persist_snapshot(paths: StoragePaths, table: pa.Table) -> Path:
    if (
        not any(
            table.schema.remove_metadata().equals(schema)
            for schema in (FEATURE_SCHEMA, LEGACY_FEATURE_SCHEMA)
        )
        or not table.num_rows
    ):
        raise ValueError("nonempty feature table with the current schema is required")
    rows = table.to_pylist()
    events = {row["event_id"] for row in rows}
    cutoffs = {row["prediction_timestamp"] for row in rows}
    if len(events) != 1 or len(cutoffs) != 1:
        raise ValueError("feature snapshot must contain one event and cutoff")
    for row in rows:
        require_known_by(row["feature_timestamp"], row["prediction_timestamp"])
        if row["feature_version"] not in {"1", "2"}:
            raise ValueError("unsupported feature version")
        EntityId(EntityKind.DRIVER, row["driver_id"])
        EntityId(EntityKind.CONSTRUCTOR, row["constructor_id"])
        EntityId(EntityKind.CIRCUIT, row["circuit_id"])
    if len({row["driver_id"] for row in rows}) != len(rows):
        raise ValueError("feature snapshot has duplicate drivers")
    event_text = next(iter(events))
    season, round_number = (int(part.split("=")[1]) for part in event_text.split("/"))
    event = EventId(season, round_number)
    if event.partition() != event_text:
        raise ValueError("noncanonical event identifier")
    rows.sort(key=lambda row: row["driver_id"])
    table = pa.Table.from_pylist(rows, schema=table.schema.remove_metadata())
    cutoff: datetime = next(iter(cutoffs))
    digest = hashlib.sha256(
        json.dumps(
            rows, sort_keys=True, default=lambda value: value.isoformat(), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    path = (
        paths.features
        / event.partition()
        / f"{cutoff.strftime('%Y%m%dT%H%M%S%fZ')}-{digest}.parquet"
    )
    write_partition(table, path, digest, rows[0]["feature_version"])
    return path
