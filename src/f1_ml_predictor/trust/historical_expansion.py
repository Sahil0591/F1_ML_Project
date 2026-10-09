"""Exhaustive season-window audit orchestration and honest coverage accounting."""

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import httpx
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import BENCHMARK_FEATURE_COLUMNS, file_sha256
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.trust.f1_schedule import (
    RACE_TIME_ARTICLES,
    SCHEDULE_URLS,
    _ScheduleHTML,
    canonical_event_name,
    validate_event_timetable,
)
from f1_ml_predictor.trust.historical import _immutable, _json, _retain_response
from f1_ml_predictor.trust.winter import _publication


def event_id(row: dict[str, Any]) -> str:
    return EventId(row["season"], row["round"]).partition()


def discover_historical_timetables(
    root: Path, discovery: dict[str, Any], http_client: httpx.Client | None = None
) -> dict[str, dict[str, Any]]:
    """Resolve 2023-24 event schedules through the official article sitemap."""
    own = http_client is None
    client = http_client or httpx.Client(timeout=30, follow_redirects=False)
    candidates = [
        row
        for row in discovery["candidates"]
        if row["season"] in {2023, 2024} and row["status"] == "audit_required"
    ]
    artifacts = []
    resolved: dict[str, list[dict[str, Any]]] = {event_id(row): [] for row in candidates}

    def xml_links(url: str) -> list[str]:
        response = client.get(url)
        artifacts.append(_retain_response(root, response))
        if response.url.host != "www.formula1.com" or len(response.content) > 4 * 1024 * 1024:
            raise ValueError("unexpected Formula 1 sitemap response")
        document = ElementTree.fromstring(response.content)
        return [node.text for node in document.iter() if node.tag.endswith("loc") and node.text]

    try:
        sections = xml_links("https://www.formula1.com/en/latest/article/sitemap.xml")
        if not 1 <= len(sections) <= 50:
            raise ValueError("Formula 1 sitemap section count is unsupported")
        urls = []
        for section in sections:
            if not re.fullmatch(
                r"https://www\.formula1\.com/en/latest/articles/sitemap-\d+\.xml", section
            ):
                raise ValueError("Formula 1 sitemap contains an unexpected section")
            urls.extend(xml_links(section))
        timetable_urls = sorted(
            {
                url
                for url in urls
                if re.fullmatch(
                    r"https://www\.formula1\.com/en/latest/article/[a-z0-9-]+-202[34]-timetable\.[A-Za-z0-9]+",
                    url,
                )
            }
        )
        if not 40 <= len(timetable_urls) <= 70:
            raise ValueError("historical timetable sitemap coverage is incomplete")
        for url in timetable_urls:
            response = client.get(url)
            artifact = _retain_response(root, response)
            artifacts.append(artifact)
            if response.url.host != "www.formula1.com" or response.content.startswith(b"%PDF"):
                continue
            parser = _ScheduleHTML()
            parser.feed(response.text)
            try:
                published = next(
                    item["datePublished"]
                    for script in parser.scripts
                    for item in [json.loads(script)]
                    if item.get("@type") == "NewsArticle" and item.get("url") == url
                )
            except (StopIteration, KeyError, json.JSONDecodeError):
                continue
            for candidate in candidates:
                if f"-{candidate['season']}-timetable." not in url:
                    continue
                selected = next(
                    row
                    for row in candidate["qualifying_versions"]
                    if row["url"] == candidate["qualifying_url"]
                )
                cutoff = _publication(selected) + timedelta(minutes=2)
                try:
                    verified = validate_event_timetable(
                        response.text,
                        season=candidate["season"],
                        round_number=candidate["round"],
                        event_name=(
                            canonical_event_name(candidate["season"], candidate["circuit_id"])
                            if "\ufffd" in candidate["event_name"]
                            else candidate["event_name"]
                        ),
                        circuit_id=candidate["circuit_id"],
                        claimed_publication=datetime.fromisoformat(
                            published.replace("Z", "+00:00")
                        ),
                        claimed_race_start=None,
                        source_url=url,
                        prediction_timestamp=cutoff,
                    )
                except ValueError:
                    continue
                resolved[event_id(candidate)].append(
                    {
                        "kind": "event_timetable",
                        "url": url,
                        "published_at_utc": datetime.fromisoformat(
                            published.replace("Z", "+00:00")
                        ).isoformat(),
                        "source_path": artifact["path"],
                        "source_sha256": artifact["sha256"],
                        "race_start_utc": verified.race_start.isoformat(),
                    }
                )
    finally:
        if own:
            client.close()
    candidate_by_id = {event_id(row): row for row in candidates}
    for key, matches in resolved.items():
        if len(matches) > 1:
            clock_matches = [
                row
                for row in matches
                if row["race_start_utc"] == candidate_by_id[key]["discovery_race_start_hint_utc"]
            ]
            if len(clock_matches) == 1:
                resolved[key] = clock_matches
    ambiguous = [key for key, matches in resolved.items() if len(matches) > 1]
    if ambiguous:
        raise ValueError(f"multiple official timetable articles match: {ambiguous}")
    mapping = {key: matches[0] for key, matches in resolved.items() if matches}
    report = {
        "version": 1,
        "method": "official-formula1-sitemap-event-timetable",
        "matched": mapping,
        "unmatched": [key for key in resolved if key not in mapping],
        "source_artifacts": artifacts,
    }
    _immutable(
        root
        / "data/benchmarks/historical_audit"
        / f"timetable-resolution-{hashlib.sha256(_json(report)).hexdigest()}.json",
        _json(report),
    )
    return mapping


def prepare_audit_catalog(
    root: Path,
    discovery: dict[str, Any],
    reviewed_path: Path,
    timetables: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Extend reviewed candidates from registry metadata without asserting Gold.

    The 2022 season uses a published Formula 1 start-time table. Seasons with
    no validated historical schedule remain explicit direct-audit exclusions.
    """
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))["candidates"]
    by_event = {event_id(item): item for item in reviewed}
    if len(by_event) != len(reviewed):
        raise ValueError("reviewed catalog repeats a race")
    schedule_reference = {
        year: next(item["schedule_reference"] for item in reviewed if item["season"] == year)
        for year in SCHEDULE_URLS
        if year in {2025, 2026}
    }
    schedule_reference[2022] = {
        "url": SCHEDULE_URLS[2022],
        "published_at_utc": "2022-02-11T17:13:44.277+00:00",
    }
    for item in discovery["candidates"]:
        key = event_id(item)
        if key in by_event or item["status"] != "audit_required":
            continue
        year = item["season"]
        finals = item.get("final_race_records", [])
        final = max(finals, key=lambda row: (_publication(row), row["url"])) if finals else None
        qualifying = next(
            (
                row
                for row in item["qualifying_versions"]
                if row["url"] == item["qualifying_url"]
                and row["document_id"] == item["document_id"]
            ),
            None,
        )
        if qualifying is None:
            continue
        cutoff = _publication(qualifying) + timedelta(minutes=2)
        timetable = (timetables or {}).get(key)
        reference = schedule_reference.get(year) or timetable
        race_time_article = RACE_TIME_ARTICLES.get((year, item["round"]))
        if race_time_article is not None:
            reference = {
                "kind": "race_time_article",
                "url": race_time_article["url"],
                "published_at_utc": race_time_article["published_at_utc"],
            }
        if (
            timetable is not None
            and timetable["race_start_utc"] != item["discovery_race_start_hint_utc"]
        ):
            amendments = [
                row
                for row in item.get("registry_documents", [])
                if row.get("url")
                and not row["recalled"]
                and "change to timetable" in row["title"].lower()
                and _publication(row) + timedelta(minutes=1) <= cutoff
            ]
            if amendments:
                amendment = max(amendments, key=lambda row: (_publication(row), row["url"]))
                reference = {
                    "kind": "fia_timetable_amendment",
                    "url": amendment["url"],
                    "document_id": amendment["document_id"],
                    "published_at_utc": _publication(amendment).isoformat(),
                    "superseded_race_start_hint_utc": item["discovery_race_start_hint_utc"],
                    "supporting_event_timetable": timetable,
                }
        by_event[key] = {
            "season": year,
            "round": item["round"],
            "circuit_id": item["circuit_id"],
            "event_name": (
                canonical_event_name(year, item["circuit_id"])
                if "\ufffd" in item["event_name"]
                else item["event_name"]
            ),
            "fia_event_name": item.get("fia_event_name", item["event_name"]),
            "index_url": item["index_url"],
            "qualifying_url": qualifying["url"],
            "document_id": qualifying["document_id"],
            "published_at_utc": _publication(qualifying).isoformat(),
            "prediction_timestamp_utc": cutoff.isoformat(),
            # The qualifying classification may itself prove the full roster.
            # The direct audit checks its row count, later entry-list state,
            # and exact final-outcome membership before Gold admission.
            "roster_entry_list": None,
            "expected_roster_size": 22 if year == 2026 else 20,
            "race_start": (
                timetable["race_start_utc"] if timetable else item["discovery_race_start_hint_utc"]
            ),
            "schedule_reference": reference,
            "final_race_label": (
                {"url": final["url"], "document_id": final["document_id"]}
                if final is not None
                else None
            ),
            "candidate_origin": "automatic_registry_metadata_requires_direct_audit",
        }
    review_paths = [
        root / "docs/HISTORICAL_2026_LATER_REVIEWS.json",
        root / "docs/HISTORICAL_2024_LATER_REVIEWS.json",
    ]
    for review_path in review_paths:
        reviews = json.loads(review_path.read_text(encoding="utf-8"))["reviews"]
        for key, review in reviews.items():
            if key not in by_event or "post_final_review" in by_event[key]:
                raise ValueError("later-document review does not bind one unreviewed candidate")
            by_event[key]["post_final_review"] = review
    catalog = {
        "version": 1,
        "method": "five-year-exhaustive-direct-audit-v1",
        "discovery_sha256": hashlib.sha256(_json(discovery)).hexdigest(),
        "reviewed_catalog_sha256": file_sha256(reviewed_path),
        "later_reviews_sha256": [file_sha256(path) for path in review_paths],
        "candidates": list(by_event.values()),
    }
    path = (
        root
        / "data/benchmarks/historical_audit"
        / (f"window-candidates-{hashlib.sha256(_json(catalog)).hexdigest()}.json")
    )
    _immutable(path, _json(catalog))
    return path, catalog


def _verified_benchmark(root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    directory = root / "data/benchmarks/gold_core"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = root / "data/benchmarks/gold_core_registry.json"
    if file_sha256(registry) != manifest["catalog_sha256"]:
        raise ValueError("Gold registry and benchmark manifest differ")
    coverage = directory / "coverage.json"
    if file_sha256(coverage) != manifest["coverage_sha256"]:
        raise ValueError("Gold coverage report differs from its manifest")
    rows: dict[str, list[dict[str, Any]]] = {}
    for tier in ("Gold", "Silver", "Development"):
        spec = manifest["datasets"][tier]
        path = directory / spec["path"]
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"{tier} benchmark bytes differ from their manifest")
        data = pq.read_table(path).to_pylist()
        if len(data) != spec["rows"]:
            raise ValueError(f"{tier} benchmark row count differs from its manifest")
        rows[tier] = data
    return manifest, rows


def build_window_report(
    root: Path,
    discovery: dict[str, Any],
    discovery_path: Path,
    direct_audit: dict[str, Any],
    direct_catalog_path: Path,
    rolling: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Account for each discovered race and distinguish metadata from audited data."""
    root = root.resolve()
    manifest, datasets = _verified_benchmark(root)
    if direct_audit["catalog_sha256"] != file_sha256(direct_catalog_path):
        raise ValueError("direct audit is for another candidate catalog")
    if direct_audit["unattempted_races"]:
        raise ValueError("direct audit did not attempt its entire catalog")
    if discovery["source_errors"]:
        raise ValueError("a season source failed, so discovery is not exhaustive")
    direct = {row["event_id"]: row for row in direct_audit["races"]}
    if len(direct) != len(direct_audit["races"]):
        raise ValueError("direct audit repeats a race")
    direct_items = {
        event_id(row): row
        for row in json.loads(direct_catalog_path.read_text(encoding="utf-8"))["candidates"]
    }
    if set(direct_items) != set(direct):
        raise ValueError("direct audit and catalog race identities differ")
    tier_events = {tier: {row["event_id"] for row in rows} for tier, rows in datasets.items()}
    if any(tier_events[a] & tier_events[b] for a in tier_events for b in tier_events if a < b):
        raise ValueError("benchmark assigns multiple tiers to one race")
    candidate_ids = {event_id(row) for row in discovery["candidates"]}
    if len(candidate_ids) != len(discovery["candidates"]):
        raise ValueError("discovery repeats a completed race")
    if not set().union(*tier_events.values()) <= candidate_ids:
        raise ValueError("benchmark contains races outside the discovered window")
    if any(
        item["reason"] != "race_not_completed_as_of_discovery"
        for item in discovery["excluded_events"]
    ):
        raise ValueError("discovery omitted a completed or uncertain schedule row")

    by_year: dict[str, dict[str, Any]] = {}
    for year in sorted(discovery["seasons"]):
        rows = [row for row in datasets["Gold"] if row["event_id"].startswith(f"season={year}/")]
        by_year[str(year)] = {
            "candidate_races": 0,
            "gold_races": len({row["event_id"] for row in rows}),
            "gold_driver_rows": len(rows),
            "feature_nonnull_driver_rows": {
                name: sum(row[name] is not None for row in rows)
                for name in BENCHMARK_FEATURE_COLUMNS
                if not name.endswith("_missing")
            },
        }
    if rolling is not None:
        if rolling["source_gold_sha256"] != manifest["datasets"]["Gold"]["sha256"]:
            raise ValueError("rolling features are for another Gold dataset")
        rolling_path = root / rolling["feature_path"]
        if file_sha256(rolling_path) != rolling["feature_sha256"]:
            raise ValueError("rolling feature bytes differ from their report")
        rolling_rows = pq.read_table(rolling_path).to_pylist()
        if len(rolling_rows) != len(datasets["Gold"]):
            raise ValueError("rolling rows do not cover the Gold benchmark")
        for year in discovery["seasons"]:
            annual = [row for row in rolling_rows if row["event_id"].startswith(f"season={year}/")]
            by_year[str(year)]["prior_race_window_complete_driver_rows"] = {
                str(window): sum(
                    json.loads(row["provenance"])[str(window)]["complete"] for row in annual
                )
                for window in (3, 5, 10)
            }
            by_year[str(year)]["rolling_feature_nonnull_driver_rows"] = {
                name: sum(row[name] is not None for row in annual)
                for name in (
                    "recent_finish_mean_3",
                    "recent_finish_mean_5",
                    "recent_finish_mean_10",
                    "recent_dnf_rate_3",
                    "recent_dnf_rate_5",
                    "recent_dnf_rate_10",
                )
            }
    source: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    races = []
    for candidate in sorted(discovery["candidates"], key=lambda row: (row["season"], row["round"])):
        key = event_id(candidate)
        if (
            candidate["status"] == "audit_required"
            and candidate["season"] in SCHEDULE_URLS
            and key not in direct
        ):
            raise ValueError(f"direct audit omitted a supported candidate: {key}")
        by_year[str(candidate["season"])]["candidate_races"] += 1
        source["jolpica_schedule_identity_current_state"] += 1
        registry_found = candidate["status"] == "audit_required"
        if registry_found:
            source["fia_exact_event_registry_metadata"] += 1
        if candidate.get("qualifying_url"):
            source["fia_qualifying_publication_metadata"] += 1
        if candidate.get("final_race_records"):
            source["fia_final_publication_metadata"] += 1
        documents = candidate.get("registry_documents", [])
        if any("grid" in row["title"].lower() for row in documents):
            source["fia_grid_publication_metadata"] += 1
        if any("penalty" in row["title"].lower() for row in documents):
            source["fia_penalty_publication_metadata"] += 1
        cutoff_docs = []
        if key in direct_items:
            cutoff = datetime.fromisoformat(direct_items[key]["prediction_timestamp_utc"])
            cutoff_docs = [
                row
                for row in documents
                if row.get("url")
                and not row["recalled"]
                and _publication(row) + timedelta(minutes=1) <= cutoff
            ]
        grid_docs = [
            row
            for row in cutoff_docs
            if "starting grid" in row["title"].lower() and "sprint" not in row["title"].lower()
        ]
        decision_docs = [
            row
            for row in cutoff_docs
            if row["title"].lower().startswith(("decision", "infringement"))
            or "penalty" in row["title"].lower()
        ]
        if grid_docs:
            source["fia_race_grid_publication_metadata_by_cutoff"] += 1
        if decision_docs:
            source["fia_decision_or_infringement_metadata_by_cutoff"] += 1
        tier = next((name for name, events in tier_events.items() if key in events), None)
        audited = direct.get(key)
        if tier is not None:
            if tier == "Gold" and (audited is None or audited["status"] != "included"):
                raise ValueError(f"frozen Gold race failed the direct audit: {key}")
            status = tier
            reasons: list[str] = []
            if tier == "Gold":
                source["fia_qualifying_table_direct_audited"] += 1
                source["fia_final_outcome_direct_audited"] += 1
                source["formula1_schedule_direct_audited"] += 1
        else:
            status = "excluded"
            reasons = (
                list(audited["reasons"])
                if audited is not None
                else list(candidate.get("exclusion_reasons", []))
            )
            if not reasons:
                reasons = ["exact_direct_gold_audit_not_completed"]
            for reason in set(reasons):
                reason_counts[reason] += 1
        races.append(
            {
                "event_id": key,
                "event_name": candidate["event_name"],
                "status": status,
                "reasons": reasons,
                "direct_audit_attempted": audited is not None,
                "qualifying_publication_metadata": bool(candidate.get("qualifying_url")),
                "grid_publication_metadata": any(
                    "grid" in row["title"].lower() for row in documents
                ),
                "penalty_publication_metadata": any(
                    "penalty" in row["title"].lower() for row in documents
                ),
                "race_grid_document_ids_by_cutoff": [row["document_id"] for row in grid_docs],
                "decision_or_infringement_document_ids_by_cutoff": [
                    row["document_id"] for row in decision_docs
                ],
            }
        )
    counts = Counter(row["status"] for row in races)
    report = {
        "version": 1,
        "method": "five-year-exhaustive-gold-core-v1",
        "as_of_utc": discovery["as_of_utc"],
        "seasons": discovery["seasons"],
        "total_candidate_races": len(races),
        "gold_eligible_races": counts["Gold"],
        "silver_eligible_races": counts["Silver"],
        "development_only_races": counts["Development"],
        "excluded_races": counts["excluded"],
        "feature_coverage_by_year": by_year,
        "evidence_coverage_by_source": dict(sorted(source.items())),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "races": races,
        "discovery_catalog": discovery_path.relative_to(root).as_posix(),
        "discovery_sha256": file_sha256(discovery_path),
        "direct_catalog": direct_catalog_path.relative_to(root).as_posix(),
        "direct_catalog_sha256": file_sha256(direct_catalog_path),
        "gold_manifest_sha256": file_sha256(root / "data/benchmarks/gold_core/manifest.json"),
        "gold_dataset_sha256": manifest["datasets"]["Gold"]["sha256"],
        "rolling_audit": rolling,
        "interpretation": (
            "Only an exact direct-evidence build grants Gold. Registry titles and current "
            "schedule identities are discovery metadata. Missing optional weather, grid, "
            "and rolling values stay missing and never relax the Gold contract."
        ),
    }
    path = (
        root
        / "data/benchmarks/historical_audit"
        / (f"five-year-coverage-{hashlib.sha256(_json(report)).hexdigest()}.json")
    )
    _immutable(path, _json(report))
    return path, report
