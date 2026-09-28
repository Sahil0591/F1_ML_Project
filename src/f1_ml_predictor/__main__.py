"""Small command-line entry point for bounded ingestion jobs."""

import argparse
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

from f1_ml_predictor.benchmarks.builder import build_benchmarks
from f1_ml_predictor.features.manifest import load_feature_request
from f1_ml_predictor.features.snapshot import build_snapshot
from f1_ml_predictor.features.storage import persist_snapshot
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.ingestion.enrichment import (
    EnrichmentReport,
    event_race,
    ingest_fastf1_session,
    ingest_openf1_session,
    persist_forecast,
)
from f1_ml_predictor.ingestion.jolpica import IngestReport, JolpicaSeasonIngestor
from f1_ml_predictor.paths import StoragePaths
from f1_ml_predictor.sources.jolpica import JolpicaClient
from f1_ml_predictor.sources.open_meteo import OpenMeteoClient
from f1_ml_predictor.sources.openf1 import OpenF1Client
from f1_ml_predictor.trust.collector import collect_weekend
from f1_ml_predictor.trust.cutoffs import CutoffKind
from f1_ml_predictor.trust.prospective import verify_bundle


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m f1_ml_predictor")
    subcommands = parser.add_subparsers(dest="command", required=True)
    ingest = subcommands.add_parser("ingest-season", help="Ingest one Jolpica season")
    ingest.add_argument("season", type=int)
    ingest.add_argument("--refresh", action="store_true")
    openf1 = subcommands.add_parser("ingest-openf1-session", help="Ingest one completed session")
    openf1.add_argument("season", type=int)
    openf1.add_argument("round", type=int)
    openf1.add_argument("session_key", type=int)
    openf1.add_argument("--refresh", action="store_true")
    fastf1 = subcommands.add_parser(
        "ingest-fastf1-session", help="Ingest lightweight lap summaries"
    )
    fastf1.add_argument("season", type=int)
    fastf1.add_argument("round", type=int)
    fastf1.add_argument("session_code", choices=["FP1", "FP2", "FP3", "Q"])
    forecast = subcommands.add_parser(
        "capture-forecast", help="Capture a forecast for an upcoming event"
    )
    forecast.add_argument("season", type=int)
    forecast.add_argument("round", type=int)
    discovery = subcommands.add_parser("list-openf1-sessions", help="Find historical session keys")
    discovery.add_argument("season", type=int)
    features = subcommands.add_parser(
        "build-snapshot", help="Build a verified pre-race feature snapshot"
    )
    features.add_argument("manifest", type=Path)
    features.add_argument("--form-window", type=int, default=5)
    features.add_argument("--session-source", choices=["fastf1", "openf1"], default="fastf1")
    features.add_argument(
        "--cutoff-kind", choices=list(CutoffKind), default=CutoffKind.POST_QUALIFYING
    )
    features.add_argument("--pre-race-minutes", type=int, default=60)
    features.add_argument("--certified-only", action="store_true")
    capture = subcommands.add_parser("capture-weekend", help="Freeze fresh prospective API inputs")
    capture.add_argument("plan", type=Path)
    verify = subcommands.add_parser("verify-capture", help="Verify an immutable prospective bundle")
    verify.add_argument("bundle", type=Path)
    benchmarks = subcommands.add_parser(
        "build-benchmarks", help="Build evidence-tiered benchmark datasets and coverage"
    )
    benchmarks.add_argument("--catalog", type=Path)
    for command in (ingest, openf1, fastf1, forecast, features, capture):
        command.add_argument("--root", type=Path, default=Path.cwd())
    benchmarks.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    paths = StoragePaths(getattr(args, "root", Path.cwd()))
    report: IngestReport | EnrichmentReport
    try:
        if args.command == "capture-weekend":
            print(f"path: {collect_weekend(args.plan, paths.root)}")
            return
        if args.command == "verify-capture":
            manifest = verify_bundle(args.bundle)
            print(f"verified: {manifest['manifest_sha256']}")
            return
        if args.command == "build-benchmarks":
            benchmark_report = build_benchmarks(paths.root, paths.benchmarks, args.catalog)
            print(f"included_races: {benchmark_report['included_races']}")
            print(f"excluded_races: {benchmark_report['excluded_races']}")
            print(f"path: {paths.benchmarks}")
            return
        if args.command == "build-snapshot":
            inputs, cutoff = load_feature_request(args.manifest, paths.root)
            table = build_snapshot(
                inputs,
                cutoff,
                form_window=args.form_window,
                session_source=args.session_source,
                cutoff_kind=CutoffKind(args.cutoff_kind),
                pre_race_minutes=args.pre_race_minutes,
                certified_only=args.certified_only,
            )
            print(f"rows: {table.num_rows}")
            print(f"path: {persist_snapshot(paths, table)}")
            return
        if args.command == "list-openf1-sessions":
            with OpenF1Client() as openf1_client:
                sessions = openf1_client.sessions(args.season)
            for session in sessions:
                if session.get("session_name") in {
                    "Practice 1",
                    "Practice 2",
                    "Practice 3",
                    "Qualifying",
                }:
                    print(
                        f"{session['session_key']} {session['date_start']} "
                        f"{session['session_name']} {session['circuit_short_name']}"
                    )
            return
        if args.command == "ingest-season":
            with JolpicaClient() as client:
                report = JolpicaSeasonIngestor(paths, client).ingest_season(
                    args.season, refresh=args.refresh
                )
        elif args.command == "ingest-openf1-session":
            with OpenF1Client() as openf1_client:
                report = ingest_openf1_session(
                    paths,
                    EventId(args.season, args.round),
                    args.session_key,
                    openf1_client,
                    refresh=args.refresh,
                )
        elif args.command == "ingest-fastf1-session":
            report = ingest_fastf1_session(
                paths, EventId(args.season, args.round), args.session_code
            )
        else:
            event = EventId(args.season, args.round)
            race = event_race(paths, event)
            days_until = (date.fromisoformat(race["date"]) - datetime.now(UTC).date()).days
            if not 0 <= days_until <= 6:
                raise ValueError(
                    "live forecast capture requires an event within the next seven days"
                )
            location = race["Circuit"]["Location"]
            with OpenMeteoClient() as weather_client:
                snapshot = weather_client.capture_forecast(
                    float(location["lat"]), float(location["long"])
                )
            report = persist_forecast(paths, event, snapshot)
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Ingestion failed: {exc}\n")
    for key, value in asdict(report).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
