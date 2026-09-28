"""Explicit publication contracts for externally verified input versions."""

from dataclasses import dataclass
from datetime import datetime

import pyarrow as pa

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.time import require_known_by, require_utc
from f1_ml_predictor.trust.evidence import AvailabilityEvidence, BenchmarkTier


@dataclass(frozen=True, slots=True)
class PublishedTable:
    table: pa.Table
    available_at: datetime | None
    evidence_reference: str
    evidence: AvailabilityEvidence | None = None

    def __post_init__(self) -> None:
        if self.available_at is not None:
            require_utc(self.available_at, "available_at")
            if not self.evidence_reference.strip():
                raise ValueError("publication evidence reference is required")
            if "available_at" in self.table.column_names:
                for value in self.table["available_at"].to_pylist():
                    if value is not None:
                        require_known_by(value, self.available_at)
        if self.evidence is not None:
            self.evidence.validate_table(self.table, self.available_at)

    @property
    def tier(self) -> BenchmarkTier:
        return self.evidence.tier if self.evidence is not None else BenchmarkTier.DEVELOPMENT


@dataclass(frozen=True, slots=True)
class PreRaceEvent:
    event: EventId
    circuit_id: str
    race_start: datetime
    qualifying_completed_at: datetime | None
    available_at: datetime
    evidence_reference: str
    evidence: AvailabilityEvidence | None = None
    qualifying_status: str = "completed"
    qualifying_cancelled_at: datetime | None = None

    def __post_init__(self) -> None:
        EntityId(EntityKind.CIRCUIT, self.circuit_id)
        require_utc(self.race_start, "race_start")
        if self.qualifying_status not in {"completed", "cancelled"}:
            raise ValueError("qualifying status must be completed or cancelled")
        decision = (
            self.qualifying_completed_at
            if self.qualifying_status == "completed"
            else self.qualifying_cancelled_at
        )
        if decision is None:
            raise ValueError("qualifying completion or cancellation timestamp is required")
        require_utc(decision, "qualifying decision")
        require_utc(self.available_at, "available_at")
        if self.qualifying_status == "cancelled":
            require_known_by(decision, self.available_at)
        if decision >= self.race_start:
            raise ValueError("qualifying must finish before the race")
        if not self.evidence_reference.strip():
            raise ValueError("event publication evidence is required")
        if self.evidence is not None and self.evidence.available_at != self.available_at:
            raise ValueError("event evidence and availability disagree")
        if self.evidence is not None:
            require_known_by(decision, self.available_at)
            self.evidence.validate_table(self.as_table(), self.available_at)

    def as_table(self) -> pa.Table:
        return pa.Table.from_pylist(
            [
                {
                    "event_id": self.event.partition(),
                    "circuit_id": self.circuit_id,
                    "race_start": self.race_start,
                    "qualifying_completed_at": self.qualifying_completed_at,
                    "qualifying_status": self.qualifying_status,
                    "qualifying_cancelled_at": self.qualifying_cancelled_at,
                }
            ]
        )

    @property
    def tier(self) -> BenchmarkTier:
        return self.evidence.tier if self.evidence is not None else BenchmarkTier.DEVELOPMENT


@dataclass(frozen=True, slots=True)
class ResultVersion:
    event: EventId
    race_completed_at: datetime
    publication: PublishedTable

    def __post_init__(self) -> None:
        require_utc(self.race_completed_at, "race_completed_at")
        if self.publication.available_at is not None:
            require_known_by(self.race_completed_at, self.publication.available_at)


@dataclass(frozen=True, slots=True)
class FeatureInputs:
    event: PreRaceEvent
    rosters: tuple[PublishedTable, ...]
    qualifying: tuple[PublishedTable, ...]
    history: tuple[ResultVersion, ...] = ()
    sessions: tuple[PublishedTable, ...] = ()
    forecasts: tuple[PublishedTable, ...] = ()
    standings: PublishedTable | None = None
    circuit: PublishedTable | None = None
