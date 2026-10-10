"""Provisional OpenF1 qualifying when Jolpica has not published before the race.

The certified post-qualifying bundle needs Jolpica qualifying. If Jolpica is still
empty ``FALLBACK_DELAY`` after scheduled qualifying, the finished OpenF1 Qualifying
session result is frozen as a separate capture. Predictions may use it as a
provisional post-qualifying input; it never enters a certified bundle, Gold or
evaluation, and a later certified Jolpica capture always takes precedence.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.prediction.schedules import Weekend
from f1_ml_predictor.sources.http import JsonSourceClient
from f1_ml_predictor.time import require_utc
from f1_ml_predictor.trust.sprint_capture import (
    _LIMITS,
    FALLBACK_DELAY,
    capture_openf1_session,
    latest_capture,
)

QUALIFYING_OPENF1 = "qualifying_openf1"


def qualifying_fallback_tick(
    root: Path,
    event: EventId,
    weekend: Weekend,
    *,
    now: Callable[[], datetime] | None = None,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """One bounded step; call only while Jolpica qualifying is still empty."""
    clock = (now or (lambda: datetime.now(UTC)))()
    require_utc(clock, "qualifying fallback clock")
    if weekend.qualifying is None or weekend.race is None:
        return {"status": "qualifying_schedule_missing"}
    status: dict[str, Any] = {"available_after": (weekend.qualifying + FALLBACK_DELAY).isoformat()}
    existing = latest_capture(root, event, QUALIFYING_OPENF1, clock)
    if existing is not None:
        return {**status, "status": "openf1_captured", "capture": existing["bundle"]}
    if clock >= weekend.race:
        return {**status, "status": "missed"}
    if clock < weekend.qualifying + FALLBACK_DELAY:
        return {**status, "status": "waiting_for_jolpica"}
    with JsonSourceClient(_LIMITS, http_client) as client:
        captured = capture_openf1_session(
            root,
            client,
            event,
            clock,
            kind=QUALIFYING_OPENF1,
            names=("Qualifying",),
            window=(weekend.qualifying - timedelta(hours=2), weekend.race),
            decision_basis="openf1_qualifying_result_after_jolpica_delay",
        )
    if captured is None:
        return {**status, "status": "waiting_for_openf1_qualifying"}
    return {**status, "status": "openf1_captured", "capture": captured}
