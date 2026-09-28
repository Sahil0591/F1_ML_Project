"""Named prediction windows are distinct evaluation cohorts."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from f1_ml_predictor.time import require_utc


class CutoffKind(StrEnum):
    POST_QUALIFYING = "post_qualifying"
    PROVISIONAL_GRID = "provisional_grid"
    PRE_RACE = "pre_race"


@dataclass(frozen=True, slots=True)
class PredictionCutoff:
    kind: CutoffKind
    timestamp: datetime
    pre_race_minutes: int = 60

    def validate(self, race_start: datetime, qualifying_decision_at: datetime) -> None:
        require_utc(self.timestamp, "prediction cutoff")
        require_utc(race_start, "race_start")
        require_utc(qualifying_decision_at, "qualifying decision")
        if not isinstance(self.kind, CutoffKind):
            raise ValueError("invalid cutoff kind")
        if not qualifying_decision_at <= self.timestamp < race_start:
            raise ValueError("cutoff must be after qualifying and before race start")
        if (
            isinstance(self.pre_race_minutes, bool)
            or not isinstance(self.pre_race_minutes, int)
            or self.pre_race_minutes < 1
        ):
            raise ValueError("pre-race window must be a positive number of minutes")
        if self.kind == CutoffKind.PRE_RACE and self.timestamp < race_start - timedelta(
            minutes=self.pre_race_minutes
        ):
            raise ValueError("cutoff is outside the declared pre-race window")
