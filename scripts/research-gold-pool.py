"""Research 2025 FIA candidate versions against the published F1 race timetable.

This writes candidate hypotheses only. The audit-winter command downloads and
verifies exact documents before any race can enter Gold Core.
"""

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from f1_ml_predictor.trust.f1_schedule import SCHEDULE_URLS, validate_f1_schedule
from f1_ml_predictor.trust.historical import _retain_response, inspect_pdf
from f1_ml_predictor.trust.winter import _publication, registry_rows

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/HISTORICAL_EXPANSION_CANDIDATES.json"


def artifact_identity(artifact: dict[str, object]) -> dict[str, object]:
    return {key: artifact[key] for key in ("path", "sha256", "url")}


def main() -> None:
    original = json.loads((ROOT / "docs/HISTORICAL_WINTER_CANDIDATES.json").read_text())
    seeds = json.loads((ROOT / "docs/HISTORICAL_CANDIDATES.json").read_text())["candidates"]
    for round_number, event_name in (
        (6, "Miami Grand Prix"),
        (17, "Azerbaijan Grand Prix"),
        (19, "United States Grand Prix"),
    ):
        seeds.append(
            {
                "season": 2025,
                "round": round_number,
                "index_url": (
                    "https://www.fia.com/documents/championships/"
                    "fia-formula-one-world-championship-14/season/season-2025-2071/event/"
                    + event_name.replace(" ", "%20")
                ),
            }
        )
    schedule_path = ROOT / "data/normalized/events/season=2025/data.parquet"
    events = {
        (row["season"], row["round"]): row for row in pq.read_table(schedule_path).to_pylist()
    }
    known = {(row["season"], row["round"]) for row in original["candidates"]}
    candidates = list(original["candidates"])
    excluded = []
    with httpx.Client(
        timeout=30, follow_redirects=False, headers={"User-Agent": "f1-ml-predictor/0.1.0"}
    ) as client:
        schedule_response = client.get(SCHEDULE_URLS[2025])
        schedule_response.raise_for_status()
        schedule_artifact = _retain_response(ROOT, schedule_response)
        schedule_html = schedule_response.text
        for seed in seeds:
            key = seed["season"], seed["round"]
            if key in known:
                continue
            event = events.get(key)
            if event is None or event["race_start_utc"] is None:
                excluded.append({"season": key[0], "round": key[1], "reason": "schedule_missing"})
                continue
            cutoff = event["race_start_utc"] - timedelta(hours=3)
            try:
                verified = validate_f1_schedule(
                    schedule_html,
                    season=event["season"],
                    round_number=event["round"],
                    event_name=event["race_name"],
                    circuit_id=event["circuit_id"],
                    claimed_publication=datetime(2025, 2, 3, 17, 2, tzinfo=UTC),
                    claimed_race_start=event["race_start_utc"],
                    source_url=SCHEDULE_URLS[2025],
                    prediction_timestamp=cutoff,
                )
                time.sleep(1)
                response = client.get(seed["index_url"])
                response.raise_for_status()
                registry = _retain_response(ROOT, response)
                rows = registry_rows(response.text)
                qualifying = [
                    row
                    for row in rows
                    if "qualifying classification" in row["title"].lower()
                    and "sprint" not in row["title"].lower()
                    and not row["recalled"]
                    and row["url"]
                    and _publication(row) + timedelta(minutes=1) <= cutoff
                ]
                finals = [
                    row
                    for row in rows
                    if "final race classification" in row["title"].lower()
                    and "sprint" not in row["title"].lower()
                    and not row["recalled"]
                    and row["url"]
                ]
                if not qualifying or not finals:
                    raise ValueError("matching qualifying or final publication not found")
                latest_qualifying = max(
                    qualifying, key=lambda row: (_publication(row), int(row["document_id"]))
                )
                latest_final = max(
                    finals, key=lambda row: (_publication(row), int(row["document_id"]))
                )
                candidate = {
                    "season": event["season"],
                    "round": event["round"],
                    "circuit_id": event["circuit_id"],
                    "event_name": event["race_name"],
                    "index_url": seed["index_url"],
                    "qualifying_url": latest_qualifying["url"],
                    "document_id": latest_qualifying["document_id"],
                    "published_at_utc": _publication(latest_qualifying).isoformat(),
                    "publication_clock_policy": (
                        "FIA printed CET; UTC+1 is the latest UTC bound if CET or CEST was meant"
                    ),
                    "prediction_timestamp_utc": cutoff.isoformat(),
                    "expected_roster_size": 20,
                    "race_start": verified.race_start.isoformat(),
                    "final_race_label": {
                        "url": latest_final["url"],
                        "document_id": latest_final["document_id"],
                    },
                    "schedule_reference": {
                        "url": SCHEDULE_URLS[2025],
                        "published_at_utc": verified.publication_at.isoformat(),
                        "retained_artifact": artifact_identity(schedule_artifact),
                    },
                    "research_registry_artifact": artifact_identity(registry),
                    "audit_status": "candidate_only_exact_version_audit_required",
                }
                if event["round"] == 15:
                    later = [
                        row
                        for row in rows
                        if row["url"]
                        and not row["recalled"]
                        and _publication(row) > _publication(latest_final)
                        and row["title"] != "Championship Points"
                    ]
                    expected = {
                        "Summons - Williams - Right of Review",
                        "Summons - Racing Bulls - Williams Right of Review",
                        "Decision - Williams Petition for Right of Review",
                    }
                    if {row["title"] for row in later} != expected:
                        raise ValueError("unreviewed documents follow Dutch final classification")
                    decision = next(row for row in later if row["title"].startswith("Decision"))
                    time.sleep(1)
                    decision_response = client.get(decision["url"])
                    decision_response.raise_for_status()
                    decision_artifact = _retain_response(ROOT, decision_response)
                    decision_text = " ".join(
                        inspect_pdf(ROOT / decision_artifact["path"])["text"].lower().split()
                    )
                    conclusion = (
                        "no power to remedy that served time penalty by amending "
                        "the classifications"
                    )
                    if conclusion not in decision_text:
                        raise ValueError("right of review may amend the final classification")
                    candidate["post_final_review"] = {
                        "later_documents": [
                            {
                                "document_id": row["document_id"],
                                "title": row["title"],
                                "url": row["url"],
                            }
                            for row in later
                        ],
                        "decision_artifact": artifact_identity(decision_artifact),
                        "decision_document_id": decision["document_id"],
                        "conclusion": "classification_cannot_be_amended",
                    }
                candidates.append(candidate)
            except (ValueError, KeyError, OSError, httpx.HTTPError) as exc:
                excluded.append({"season": key[0], "round": key[1], "reason": str(exc)})
    OUTPUT.write_text(
        json.dumps(
            {
                "version": 1,
                "candidates": candidates,
                "research_exclusions": excluded,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"candidate races: {len(candidates)}")
    print(f"research exclusions: {len(excluded)}")
    print(f"catalog: {OUTPUT}")


if __name__ == "__main__":
    main()
