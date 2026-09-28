"""Explicit publication contracts for externally verified input versions."""

from dataclasses import dataclass
from datetime import datetime

import pyarrow as pa

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.time import require_known_by, require_utc


@dataclass(frozen=True, slots=True)
class PublishedTable:
    table: pa.Table
    available_at: datetime | None
    evidence_reference: str

    def __post_init__(self) -> None:
        if self.available_at is not None:
            require_utc(self.available_at, "available_at")
            if not self.evidence_reference.strip():
                raise ValueError("publication evidence reference is required")
            if "available_at" in self.table.column_names:
                for value in self.table["available_at"].to_pylist():
                    if value is not None:
                        require_known_by(value, self.available_at)


@dataclass(frozen=True, slots=True)
class PreRaceEvent:
    event: EventId
    circuit_id: str
    race_start: datetime
    qualifying_completed_at: datetime
    available_at: datetime
    evidence_reference: str

    def __post_init__(self) -> None:
        EntityId(EntityKind.CIRCUIT, self.circuit_id)
        require_utc(self.race_start, "race_start")
        require_utc(self.qualifying_completed_at, "qualifying_completed_at")
        require_utc(self.available_at, "available_at")
        if self.qualifying_completed_at >= self.race_start:
            raise ValueError("qualifying must finish before the race")
        if not self.evidence_reference.strip():
            raise ValueError("event publication evidence is required")


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
