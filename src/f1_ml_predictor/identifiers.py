"""Stable identifiers shared across source and model layers."""

import re
from dataclasses import dataclass
from enum import StrEnum

_JOLPICA_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


class EntityKind(StrEnum):
    DRIVER = "driver"
    CONSTRUCTOR = "constructor"
    CIRCUIT = "circuit"


@dataclass(frozen=True, slots=True)
class EntityId:
    kind: EntityKind
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EntityKind):
            raise ValueError("kind must be a supported entity kind")
        if not isinstance(self.value, str) or not _JOLPICA_ID.fullmatch(self.value):
            raise ValueError("value must be a canonical Jolpica identifier")


@dataclass(frozen=True, order=True, slots=True)
class EventId:
    season: int
    round: int

    def __post_init__(self) -> None:
        if isinstance(self.season, bool) or not isinstance(self.season, int) or self.season < 1950:
            raise ValueError("season must be an integer from 1950 onward")
        if isinstance(self.round, bool) or not isinstance(self.round, int) or self.round < 1:
            raise ValueError("round must be a positive integer")

    def partition(self) -> str:
        return f"season={self.season}/round={self.round:02d}"
