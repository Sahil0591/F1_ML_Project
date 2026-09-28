"""Select a provider per session; disagreement is visible, never averaged."""

import math
from datetime import datetime
from typing import Any

from f1_ml_predictor.features.contracts import PublishedTable
from f1_ml_predictor.trust.evidence import BenchmarkTier


def arbitrate_sessions(
    publications: tuple[PublishedTable, ...],
    event_id: str,
    cutoff: datetime,
    preferred: str,
    *,
    disagreement_fraction: float = 0.02,
) -> tuple[list[PublishedTable], list[dict[str, Any]]]:
    if preferred not in {"fastf1", "openf1"}:
        raise ValueError("unsupported session provider")
    groups: dict[str, dict[str, PublishedTable]] = {}
    for publication in publications:
        if publication.available_at is None or publication.available_at > cutoff:
            continue
        rows = publication.table.to_pylist()
        if not rows:
            continue
        if any(row.get("event_id") != event_id for row in rows):
            raise ValueError("session publication contains a different event")
        codes = {row.get("session_code") for row in rows}
        provider_names = {row.get("source") for row in rows}
        if len(codes) != 1 or not codes <= {"FP1", "FP2", "FP3", "Q"}:
            raise ValueError("race sessions cannot enter pre-race features")
        if len(provider_names) != 1 or not provider_names <= {"fastf1", "openf1"}:
            raise ValueError("session publication has ambiguous provider identity")
        code, provider = next(iter(codes)), next(iter(provider_names))
        existing = groups.setdefault(code, {}).get(provider)
        if existing is not None and existing.available_at == publication.available_at:
            raise ValueError("ambiguous session versions")
        if existing is None:
            groups[code][provider] = publication
            continue
        assert existing.available_at is not None
        if publication.available_at > existing.available_at:
            groups[code][provider] = publication
    selected, reports = [], []
    quality_order = {BenchmarkTier.GOLD: 0, BenchmarkTier.SILVER: 1, BenchmarkTier.DEVELOPMENT: 2}
    for code, providers in sorted(groups.items()):
        chosen = min(
            providers,
            key=lambda name: (quality_order[providers[name].tier], name != preferred, name),
        )
        disagreement = False
        if len(providers) == 2:

            def best(publication: PublishedTable) -> dict[str, float]:
                result: dict[str, float] = {}
                for row in publication.table.to_pylist():
                    value = row.get("best_lap_seconds")
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and value > 0
                    ):
                        result[row["driver_id"]] = min(result.get(row["driver_id"], value), value)
                return result

            left, right = best(providers["fastf1"]), best(providers["openf1"])
            disagreement = any(
                abs(left[driver] - right[driver]) / min(left[driver], right[driver])
                > disagreement_fraction
                for driver in left.keys() & right.keys()
            )
        reports.append(
            {
                "session": code,
                "selected": chosen,
                "candidates": sorted(providers),
                "disagreement": disagreement,
                "policy": "evidence_then_preference_no_average",
            }
        )
        selected.append(providers[chosen])
    return selected, reports
