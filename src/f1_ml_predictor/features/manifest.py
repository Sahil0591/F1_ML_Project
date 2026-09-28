"""Load explicit as-of input evidence bound to exact local Parquet files."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from f1_ml_predictor.features.contracts import (
    FeatureInputs,
    PreRaceEvent,
    PublishedTable,
    ResultVersion,
)
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.time import require_utc


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("manifest timestamps must be ISO strings")
    parsed = datetime.fromisoformat(value)
    require_utc(parsed, "manifest timestamp")
    return parsed


def load_feature_request(manifest_path: Path, root: Path) -> tuple[FeatureInputs, datetime]:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("feature manifest must be a version 1 object")
    root = root.resolve()

    def publication(value: dict[str, Any], kind: str) -> PublishedTable:
        if not isinstance(value, dict):
            raise ValueError("publication manifest must be an object")
        if value.get("kind") != kind:
            raise ValueError(f"publication must declare kind {kind}")
        relative = Path(value["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("publication path must remain within the workspace root")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
            if digest.hexdigest() != value["sha256"]:
                raise ValueError("publication file hash does not match its evidence manifest")
            handle.seek(0)
            table = pq.ParquetFile(handle).read()
        available = value.get("available_at")
        return PublishedTable(
            table,
            _timestamp(available) if available is not None else None,
            value["evidence_reference"],
        )

    raw_event = manifest["event"]
    event = PreRaceEvent(
        EventId(raw_event["season"], raw_event["round"]),
        raw_event["circuit_id"],
        _timestamp(raw_event["race_start"]),
        _timestamp(raw_event["qualifying_completed_at"]),
        _timestamp(raw_event["available_at"]),
        raw_event["evidence_reference"],
    )
    inputs = FeatureInputs(
        event=event,
        rosters=tuple(publication(value, "roster") for value in manifest["rosters"]),
        qualifying=tuple(publication(value, "qualifying") for value in manifest["qualifying"]),
        history=tuple(
            ResultVersion(
                EventId(value["season"], value["round"]),
                _timestamp(value["race_completed_at"]),
                publication(value["publication"], "race_results"),
            )
            for value in manifest.get("history", [])
        ),
        sessions=tuple(
            publication(value, "session_summary") for value in manifest.get("sessions", [])
        ),
        forecasts=tuple(publication(value, "forecast") for value in manifest.get("forecasts", [])),
        standings=publication(manifest["standings"], "standings")
        if manifest.get("standings")
        else None,
        circuit=publication(manifest["circuit"], "circuit") if manifest.get("circuit") else None,
    )
    return inputs, _timestamp(manifest["prediction_timestamp"])
