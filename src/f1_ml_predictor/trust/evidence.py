"""Evidence classes do not silently promote historical current-state values."""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

import pyarrow as pa

from f1_ml_predictor.time import require_known_by, require_utc


class EvidenceClass(StrEnum):
    CAPTURED_LIVE = "captured_live"
    SOURCE_PUBLISHED_TIMESTAMP = "source_published_timestamp"
    VERSIONED_ARCHIVE = "versioned_archive"
    CONSERVATIVE_RECONSTRUCTION = "conservative_reconstruction"
    CURRENT_STATE_ONLY = "current_state_only"


class BenchmarkTier(StrEnum):
    GOLD = "Gold"
    SILVER = "Silver"
    DEVELOPMENT = "Development"


def table_hash(table: pa.Table) -> str:
    rows = sorted(
        json.dumps(row, sort_keys=True, default=lambda value: value.isoformat(), allow_nan=False)
        for row in table.to_pylist()
    )
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AvailabilityEvidence:
    kind: EvidenceClass
    reference: str
    available_at: datetime | None = None
    captured_at: datetime | None = None
    artifact_sha256: str | None = None
    source_published_at: datetime | None = None
    archive_version: str | None = None
    reconstruction_method: str | None = None
    audited: bool = False
    publication_precision_seconds: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceClass) or not isinstance(self.audited, bool):
            raise ValueError("invalid evidence class or audit flag")
        if type(
            self.publication_precision_seconds
        ) is not int or self.publication_precision_seconds not in {0, 60}:
            raise ValueError("publication precision must be exact or one minute")
        if (
            self.publication_precision_seconds
            and self.kind != EvidenceClass.SOURCE_PUBLISHED_TIMESTAMP
        ):
            raise ValueError("publication precision applies only to direct publication timestamps")
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("evidence reference is required")
        for name in ("available_at", "captured_at", "source_published_at"):
            value = getattr(self, name)
            if value is not None:
                require_utc(value, name)
        if self.artifact_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.artifact_sha256
        ):
            raise ValueError("evidence artifact must have a SHA-256 hash")
        if self.kind != EvidenceClass.CURRENT_STATE_ONLY:
            if self.available_at is None or self.artifact_sha256 is None:
                raise ValueError("temporal evidence requires availability and exact artifact hash")
        if self.kind == EvidenceClass.CAPTURED_LIVE:
            if self.captured_at is None or self.available_at != self.captured_at:
                raise ValueError("captured-live availability must equal capture time")
        elif self.kind == EvidenceClass.SOURCE_PUBLISHED_TIMESTAMP:
            if self.source_published_at is None or self.available_at != (
                self.source_published_at + timedelta(seconds=self.publication_precision_seconds)
            ):
                raise ValueError(
                    "source publication timestamp plus explicit precision must equal availability"
                )
        elif self.kind == EvidenceClass.VERSIONED_ARCHIVE:
            if not self.archive_version or not self.archive_version.strip():
                raise ValueError("archive version identity is required")
        elif self.kind == EvidenceClass.CONSERVATIVE_RECONSTRUCTION:
            if not self.reconstruction_method or not self.reconstruction_method.strip():
                raise ValueError("conservative reconstruction method is required")
        if self.captured_at is not None and self.available_at is not None:
            require_known_by(self.available_at, self.captured_at)
        if self.source_published_at is not None and self.available_at is not None:
            require_known_by(self.source_published_at, self.available_at)

    @property
    def tier(self) -> BenchmarkTier:
        if self.kind == EvidenceClass.CAPTURED_LIVE:
            return BenchmarkTier.GOLD
        if not self.audited or self.kind == EvidenceClass.CURRENT_STATE_ONLY:
            return BenchmarkTier.DEVELOPMENT
        if self.kind == EvidenceClass.CONSERVATIVE_RECONSTRUCTION:
            return BenchmarkTier.SILVER
        return BenchmarkTier.GOLD

    def validate_table(self, table: pa.Table, available_at: datetime | None) -> None:
        if self.available_at != available_at:
            raise ValueError("evidence and publication availability disagree")
        if self.artifact_sha256 is not None and self.artifact_sha256 != table_hash(table):
            raise ValueError("evidence does not bind this exact table version")
        if self.kind == EvidenceClass.CAPTURED_LIVE and "captured_at" in table.column_names:
            assert self.captured_at is not None
            for captured in table["captured_at"].to_pylist():
                if captured is not None:
                    require_known_by(captured, self.captured_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.kind.value,
            "reference": self.reference,
            "available_at": self.available_at.isoformat() if self.available_at else None,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "artifact_sha256": self.artifact_sha256,
            "source_published_at": self.source_published_at.isoformat()
            if self.source_published_at
            else None,
            "archive_version": self.archive_version,
            "reconstruction_method": self.reconstruction_method,
            "audited": self.audited,
            "publication_precision_seconds": self.publication_precision_seconds,
            "tier": self.tier.value,
        }


def evidence_from_dict(value: dict[str, Any]) -> AvailabilityEvidence:
    def timestamp(name: str) -> datetime | None:
        raw = value.get(name)
        return datetime.fromisoformat(raw) if raw is not None else None

    return AvailabilityEvidence(
        EvidenceClass(value["class"]),
        value["reference"],
        timestamp("available_at"),
        timestamp("captured_at"),
        value.get("artifact_sha256"),
        timestamp("source_published_at"),
        value.get("archive_version"),
        value.get("reconstruction_method"),
        value.get("audited", False),
        value.get("publication_precision_seconds", 0),
    )


def weakest_tier(tiers: list[BenchmarkTier]) -> BenchmarkTier:
    order = {BenchmarkTier.GOLD: 0, BenchmarkTier.SILVER: 1, BenchmarkTier.DEVELOPMENT: 2}
    return max(tiers, key=order.__getitem__) if tiers else BenchmarkTier.DEVELOPMENT
