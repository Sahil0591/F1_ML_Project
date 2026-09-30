"""Audit every completed race across five seasons against unchanged Gold Core."""

import argparse
import json
from pathlib import Path

from f1_ml_predictor.benchmarks.builder import build_benchmarks
from f1_ml_predictor.benchmarks.rolling import build_gold_rolling
from f1_ml_predictor.trust.candidates import discover_candidates
from f1_ml_predictor.trust.historical_expansion import (
    build_window_report,
    discover_historical_timetables,
    prepare_audit_catalog,
)
from f1_ml_predictor.trust.winter import audit_winter_pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--discovery", type=Path, help="Reuse an immutable discovery catalog")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.discovery:
        discovery_path = args.discovery.resolve()
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    else:
        result = discover_candidates(root, seasons=(2022, 2023, 2024, 2025, 2026), limit=None)
        discovery_path = root / result["catalog_path"]
        discovery = result["catalog"]
    if discovery["seasons"] != [2022, 2023, 2024, 2025, 2026]:
        raise ValueError("discovery is not the intended five-season window")
    timetables = discover_historical_timetables(root, discovery)
    direct_catalog_path, direct_catalog = prepare_audit_catalog(
        root, discovery, root / "docs/HISTORICAL_2026_CANDIDATES.json", timetables
    )
    direct = audit_winter_pool(
        direct_catalog_path, root, minimum_races=8, reuse_retained=True, exhaustive=True
    )
    build_benchmarks(
        root,
        root / "data/benchmarks/gold_core",
        root / "data/benchmarks/gold_core_registry.json",
    )
    rolling = build_gold_rolling(root)
    report_path, report = build_window_report(
        root, discovery, discovery_path, direct, direct_catalog_path, rolling
    )
    summary = {
        key: report[key]
        for key in (
            "total_candidate_races",
            "gold_eligible_races",
            "silver_eligible_races",
            "development_only_races",
            "excluded_races",
            "feature_coverage_by_year",
            "evidence_coverage_by_source",
            "exclusion_reason_counts",
        )
    }
    summary["report"] = report_path.relative_to(root).as_posix()
    summary["direct_audit_candidates"] = len(direct_catalog["candidates"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
