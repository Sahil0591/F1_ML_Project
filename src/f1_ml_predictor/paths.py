"""Local paths for separate data and model artifact zones."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StoragePaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def normalized(self) -> Path:
        return self.root / "data" / "normalized"

    @property
    def features(self) -> Path:
        return self.root / "data" / "features"

    @property
    def benchmarks(self) -> Path:
        return self.root / "data" / "benchmarks"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def predictions(self) -> Path:
        return self.root / "data" / "predictions"
