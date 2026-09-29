"""Prepare a bounded research catalog from retained, unaudited discovery hints.

The resulting entries are candidates only. audit-winter must bind exact FIA PDFs,
the registry, official schedule table and final outcomes before Gold inclusion.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from f1_ml_predictor.trust.f1_schedule import SCHEDULE_URLS

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY = ROOT / (
    "data/benchmarks/historical_audit/"
    "candidates-aeeddc9643120b943464e7bcc6135c9b4453becf1b50a33765af2e51a970d5d2.json"
)
BASE = ROOT / "docs/HISTORICAL_EXPANSION_CANDIDATES.json"
OUTPUT = ROOT / "docs/HISTORICAL_2026_CANDIDATES.json"
ROUNDS = {9, 10, 11, 12, 13, 15}


def upper_utc(raw_cet: str) -> datetime:
    """Use the later possible UTC bound when FIA's summer CET is ambiguous."""
    local = datetime.strptime(raw_cet, "%d.%m.%y %H:%M")
    return local.replace(tzinfo=timezone(timedelta(hours=1))).astimezone(UTC)


def main() -> None:
    discovery_bytes = DISCOVERY.read_bytes()
    discovery = json.loads(discovery_bytes)
    previous = json.loads(BASE.read_text(encoding="utf-8"))["candidates"]
    initial = previous
    if len(initial) != 27 or {item["round"] for item in initial if item["season"] == 2026} != {
        1,
        2,
        3,
    }:
        raise ValueError("unexpected frozen starting cohort")
    schedule = next(item["schedule_reference"] for item in initial if item["season"] == 2026)
    if schedule["url"] != SCHEDULE_URLS[2026]:
        raise ValueError("unexpected official 2026 schedule reference")
    additions = []
    for source in discovery["candidates"]:
        if source["season"] != 2026 or source["round"] not in ROUNDS:
            continue
        if source["status"] != "audit_required" or not source["final_race_records"]:
            raise ValueError(f"missing official candidate records for round {source['round']}")
        race_start = datetime.fromisoformat(source["discovery_race_start_hint_utc"])
        cutoff = race_start - timedelta(hours=3)
        qualifying = [
            row
            for row in source["qualifying_versions"]
            if not row["recalled"] and upper_utc(row["publication_cet"]) < cutoff
        ]
        if not qualifying:
            raise ValueError(f"no qualifying candidate before round {source['round']} cutoff")
        selected = max(
            qualifying,
            key=lambda row: (upper_utc(row["publication_cet"]), int(row["document_id"])),
        )
        final = max(
            (row for row in source["final_race_records"] if not row["recalled"]),
            key=lambda row: (upper_utc(row["publication_cet"]), int(row["document_id"])),
        )
        additions.append(
            {
                "season": 2026,
                "round": source["round"],
                "circuit_id": source["circuit_id"],
                "event_name": source["event_name"],
                "index_url": source["index_url"],
                "qualifying_url": selected["url"],
                "document_id": selected["document_id"],
                "published_at_utc": upper_utc(selected["publication_cet"]).isoformat(),
                "publication_clock_policy": (
                    "FIA printed CET; UTC+1 is the latest UTC bound if CET or CEST was meant"
                ),
                "prediction_timestamp_utc": cutoff.isoformat(),
                "expected_roster_size": 22,
                "race_start": race_start.isoformat(),
                "final_race_label": {
                    "url": final["url"],
                    "document_id": final["document_id"],
                },
                "schedule_reference": {
                    "url": schedule["url"],
                    "published_at_utc": schedule["published_at_utc"],
                },
                "research_registry_artifact": next(
                    artifact
                    for artifact in discovery["source_artifacts"]
                    if artifact["url"] == source["index_url"]
                ),
                "audit_status": "candidate_only_exact_version_audit_required",
            }
        )
    if {item["round"] for item in additions} != ROUNDS:
        raise ValueError("discovery did not contain all selected 2026 rounds")
    catalog = {
        "version": 1,
        "source_discovery": {
            "path": DISCOVERY.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(discovery_bytes).hexdigest(),
            "classification": "current_state_only",
        },
        "candidates": initial + sorted(additions, key=lambda item: -item["round"]),
    }
    OUTPUT.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {len(additions)} candidate-only 2026 races in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
