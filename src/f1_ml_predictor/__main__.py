"""Small command-line entry point for bounded ingestion jobs."""

import argparse
import json
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

from f1_ml_predictor.benchmarks.builder import build_benchmarks
from f1_ml_predictor.benchmarks.enrichment import build_gold_enrichment
from f1_ml_predictor.benchmarks.rolling import build_gold_rolling
from f1_ml_predictor.benchmarks.scoring import build_gold_scoring
from f1_ml_predictor.benchmarks.versioning import archive_benchmark
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
from f1_ml_predictor.models.backtest import run_backtest_files
from f1_ml_predictor.models.boosting import BACKENDS
from f1_ml_predictor.models.development import publish_development_fold
from f1_ml_predictor.models.hardware import inspect_hardware
from f1_ml_predictor.models.probabilistic import run_probabilistic_files
from f1_ml_predictor.paths import StoragePaths
from f1_ml_predictor.prediction.pipeline import predict_next_race
from f1_ml_predictor.sources.jolpica import JolpicaClient
from f1_ml_predictor.sources.open_meteo import OpenMeteoClient
from f1_ml_predictor.sources.openf1 import OpenF1Client
from f1_ml_predictor.trust.candidates import discover_candidates
from f1_ml_predictor.trust.collector import collect_weekend
from f1_ml_predictor.trust.cutoffs import CutoffKind
from f1_ml_predictor.trust.evidence import BenchmarkTier
from f1_ml_predictor.trust.historical import discover_auditability, reconstruct_gold_core
from f1_ml_predictor.trust.prospective import verify_bundle
from f1_ml_predictor.trust.scheduler import (
    import_scheduler_outcomes,
    scheduler_status,
    scheduler_tick,
)
from f1_ml_predictor.trust.winter import audit_winter_pool


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
    candidates = subcommands.add_parser(
        "discover-gold-candidates", help="Discover and rank a bounded official registry pool"
    )
    candidates.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[datetime.now(UTC).year, datetime.now(UTC).year - 1],
    )
    candidates.add_argument("--limit", type=int, default=17)
    candidates.add_argument("--all", action="store_true", help="Inspect every completed race")
    candidates.add_argument("--root", type=Path, default=Path.cwd())
    historical = subcommands.add_parser("audit-candidates", help="Rank a bounded FIA audit pool")
    historical.add_argument("catalog", type=Path)
    historical.add_argument("--limit", type=int, default=17)
    historical.add_argument("--root", type=Path, default=Path.cwd())
    core = subcommands.add_parser(
        "build-gold-core", help="Freeze an audited historical Core request"
    )
    core.add_argument("manifest", type=Path)
    core.add_argument("--root", type=Path, default=Path.cwd())
    rolling = subcommands.add_parser(
        "build-gold-rolling", help="Freeze audited prior-race rolling features"
    )
    rolling.add_argument("--root", type=Path, default=Path.cwd())
    rolling.add_argument("--benchmark-dir", type=Path)
    rolling.add_argument("--catalog", type=Path)
    enrichment = subcommands.add_parser(
        "build-gold-enrichment", help="Freeze evidence-bounded historical Gold features"
    )
    enrichment.add_argument("--root", type=Path, default=Path.cwd())
    enrichment.add_argument("--benchmark-dir", type=Path)
    enrichment.add_argument("--catalog", type=Path)
    enrichment.add_argument("--grid-audit", type=Path)
    enrichment.add_argument("--practice-audit", type=Path)
    scoring = subcommands.add_parser(
        "build-gold-scoring", help="Freeze audited point-in-time Gold scoring features"
    )
    scoring.add_argument("--root", type=Path, default=Path.cwd())
    scoring.add_argument("--benchmark-dir", type=Path, required=True)
    scoring.add_argument("--rules", type=Path)
    scoring.add_argument("--event-points", type=Path)
    winter = subcommands.add_parser("audit-winter", help="Audit a bounded direct-publication pool")
    winter.add_argument("catalog", type=Path)
    winter.add_argument("--root", type=Path, default=Path.cwd())
    winter.add_argument("--reuse-retained", action="store_true")
    winter.add_argument("--exhaustive", action="store_true", help="Attempt every catalog race")
    winter.add_argument("--minimum-races", type=int, default=8)
    scheduled = subcommands.add_parser("collect-next-race", help="Run one bounded prospective tick")
    scheduled.add_argument("--season", type=int)
    scheduled.add_argument("--new-capture", action="store_true")
    scheduled.add_argument("--pre-race", action="store_true")
    scheduled.add_argument("--root", type=Path, default=Path.cwd())
    status = subcommands.add_parser("collection-status", help="Read prospective collection status")
    status.add_argument("--root", type=Path, default=Path.cwd())
    outcomes = subcommands.add_parser(
        "register-collected-outcomes", help="Link audited final labels"
    )
    outcomes.add_argument("outcomes", type=Path)
    outcomes.add_argument("--sha256", required=True)
    outcomes.add_argument("--root", type=Path, default=Path.cwd())
    backtest = subcommands.add_parser(
        "backtest", help="Run a tier-labelled chronological baseline evaluation"
    )
    backtest.add_argument("--tier", choices=[tier.value for tier in BenchmarkTier], default="Gold")
    backtest.add_argument("--min-train-events", type=int, default=2)
    backtest.add_argument("--seed", type=int, default=42)
    backtest.add_argument("--root", type=Path, default=Path.cwd())
    backtest.add_argument("--benchmark-dir", type=Path)
    hardware = subcommands.add_parser(
        "model-hardware", help="Probe and benchmark installed CPU/GPU backends"
    )
    hardware.add_argument("--workload-rows", type=int, default=8000)
    hardware.add_argument("--root", type=Path, default=Path.cwd())
    stronger = subcommands.add_parser(
        "compare-models", help="Compare calibrated chronological race models"
    )
    stronger.add_argument("--tier", choices=[tier.value for tier in BenchmarkTier], default="Gold")
    stronger.add_argument("--backends", choices=BACKENDS, nargs="+", default=list(BACKENDS))
    stronger.add_argument("--min-train-events", type=int, default=2)
    stronger.add_argument("--seed", type=int, default=42)
    stronger.add_argument("--draws", type=int, default=4096)
    stronger.add_argument(
        "--calibration", choices=["sigmoid", "isotonic", "identity"], default="sigmoid"
    )
    stronger.add_argument("--calibration-events", type=int, default=1)
    stronger.add_argument("--device", choices=["cpu", "auto", "cuda"], default="auto")
    stronger.add_argument("--root", type=Path, default=Path.cwd())
    stronger.add_argument("--benchmark-dir", type=Path)
    development = subcommands.add_parser(
        "publish-development", help="Publish one label-free historical outer-fold prediction"
    )
    development.add_argument("report", type=Path)
    development.add_argument("predictions", type=Path)
    development.add_argument("benchmark_dir", type=Path)
    development.add_argument("--backend", choices=BACKENDS, required=True)
    development.add_argument("--event-id")
    development.add_argument("--output", type=Path)
    development.add_argument("--root", type=Path, default=Path.cwd())
    nextrace = subcommands.add_parser(
        "predict-next-race", help="Publish development-only next race and title predictions"
    )
    nextrace.add_argument("--season", type=int)
    nextrace.add_argument("--simulations", type=int, default=100000)
    nextrace.add_argument("--seed", type=int, default=42)
    nextrace.add_argument("--draws", type=int, default=65536)
    nextrace.add_argument("--championship-orders", type=int, default=8192)
    nextrace.add_argument("--device", choices=["cpu", "auto", "cuda"], default="auto")
    nextrace.add_argument(
        "--no-collect", action="store_true", help="Use the retained schedule without a tick"
    )
    nextrace.add_argument("--benchmark-dir", type=Path)
    nextrace.add_argument("--dnf-benchmark-dir", type=Path)
    nextrace.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    paths = StoragePaths(getattr(args, "root", Path.cwd()))
    report: IngestReport | EnrichmentReport
    try:
        if args.command == "discover-gold-candidates":
            result = discover_candidates(
                paths.root, seasons=tuple(args.seasons), limit=None if args.all else args.limit
            )
            print(f"status: {result['auditability_report']['status']}")
            print(f"candidates: {len(result['catalog']['candidates'])}")
            print(f"eligible_gold_races: {result['catalog']['eligible_gold_races']}")
            print(f"catalog: {result['catalog_path']}")
            print(f"report: {result['report_path']}")
            return
        if args.command == "audit-candidates":
            result = discover_auditability(args.catalog, paths.root, limit=args.limit)
            print(f"status: {result['status']}")
            print(f"candidates: {len(result['candidates'])}")
            print(f"eligible_gold_races: {result['eligible_gold_races']}")
            return
        if args.command == "build-gold-core":
            print(json.dumps(reconstruct_gold_core(args.manifest, paths.root), indent=2))
            return
        if args.command == "build-gold-rolling":
            print(
                json.dumps(
                    build_gold_rolling(paths.root, args.benchmark_dir, args.catalog), indent=2
                )
            )
            return
        if args.command == "build-gold-enrichment":
            print(
                json.dumps(
                    build_gold_enrichment(
                        paths.root,
                        args.benchmark_dir,
                        args.catalog,
                        args.grid_audit,
                        args.practice_audit,
                    ),
                    indent=2,
                )
            )
            return
        if args.command == "build-gold-scoring":
            print(
                json.dumps(
                    build_gold_scoring(
                        paths.root, args.benchmark_dir, args.rules, args.event_points
                    ),
                    indent=2,
                )
            )
            return
        if args.command == "audit-winter":
            print(
                json.dumps(
                    audit_winter_pool(
                        args.catalog,
                        paths.root,
                        minimum_races=args.minimum_races,
                        reuse_retained=args.reuse_retained,
                        exhaustive=args.exhaustive,
                    ),
                    indent=2,
                )
            )
            return
        if args.command == "collect-next-race":
            result = scheduler_tick(
                paths.root,
                season=args.season,
                new_capture=args.new_capture,
                collect_pre_race=args.pre_race,
            )
            print(json.dumps(result, indent=2))
            if result.get("status") == "error":
                raise SystemExit(1)
            return
        if args.command == "predict-next-race":
            if not args.no_collect:
                tick = scheduler_tick(paths.root, season=args.season)
                print(f"collector: {tick.get('status')} {tick.get('error', '')}".rstrip())
            prediction = predict_next_race(
                paths.root,
                season=args.season,
                simulations=args.simulations,
                seed=args.seed,
                draws=args.draws,
                championship_orders=args.championship_orders,
                device=args.device,
                gold_dir=args.benchmark_dir,
                dnf_dir=args.dnf_benchmark_dir,
            )
            print(prediction["report"])
            print(f"status: {prediction['status']}")
            print(f"event: {prediction['event_id']}")
            print(f"cutoff: {prediction['cutoff_kind']} {prediction['prediction_timestamp_utc']}")
            print(f"path: {prediction['run_dir']}")
            return
        if args.command == "collection-status":
            print(json.dumps(scheduler_status(paths.root), indent=2))
            return
        if args.command == "register-collected-outcomes":
            print(
                json.dumps(
                    import_scheduler_outcomes(
                        paths.root, args.outcomes, expected_sha256=args.sha256
                    ),
                    indent=2,
                )
            )
            return
        if args.command == "model-hardware":
            output = paths.models / "experiments" / "hardware.json"
            result = inspect_hardware(output, workload_rows=args.workload_rows)
            print(f"backends: {list(result['backends'])}")
            print(f"path: {output}")
            return
        if args.command == "publish-development":
            metadata = json.loads(args.report.read_text(encoding="utf-8"))["run_metadata"]
            output = args.output or (
                paths.predictions
                / "development"
                / metadata["dataset_version"]
                / metadata["run_id"]
                / f"{args.backend}.parquet"
            )
            publication = publish_development_fold(
                args.report,
                args.predictions,
                args.benchmark_dir,
                output,
                backend=args.backend,
                event_id=args.event_id,
            )
            print(f"status: {publication['status']}")
            print(f"event: {publication['historical_heldout_event']}")
            print(f"path: {output}")
            return
        if args.command == "compare-models":
            tier = BenchmarkTier(args.tier)
            benchmark_dir = args.benchmark_dir or paths.benchmarks
            benchmark_snapshot = archive_benchmark(benchmark_dir)
            if benchmark_snapshot is None:
                raise ValueError("benchmark manifest is missing")
            benchmark_dir = benchmark_snapshot["path"]
            cohort = Path(benchmark_snapshot["dataset_version"])
            run_id = uuid.uuid4().hex
            hardware_path = paths.models / "experiments" / "hardware.json"
            hardware_report = (
                json.loads(hardware_path.read_text(encoding="utf-8"))
                if (args.device != "cpu" and hardware_path.exists())
                else None
            )
            report_path = (
                paths.models
                / "experiments"
                / tier.value.lower()
                / cohort
                / "runs"
                / run_id
                / "comparison.json"
            )
            result = run_probabilistic_files(
                benchmark_dir / f"{tier.value.lower()}.parquet",
                tier,
                report_path,
                paths.predictions
                / "probabilistic"
                / cohort
                / "runs"
                / run_id
                / f"{tier.value.lower()}.parquet",
                run_id=run_id,
                backends=tuple(args.backends),
                min_train_events=args.min_train_events,
                seed=args.seed,
                draws=args.draws,
                device=args.device,
                hardware=hardware_report,
                calibration_method=args.calibration,
                calibration_event_count=args.calibration_events,
            )
            print(f"status: {result['status']}")
            print(f"prediction_rows: {result['prediction_rows']}")
            print(f"selection: {result['selection']}")
            print(f"path: {report_path}")
            return
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
        if args.command == "backtest":
            tier = BenchmarkTier(args.tier)
            benchmark_dir = args.benchmark_dir or paths.benchmarks
            benchmark_snapshot = archive_benchmark(benchmark_dir)
            if benchmark_snapshot is None:
                raise ValueError("benchmark manifest is missing")
            benchmark_dir = benchmark_snapshot["path"]
            cohort = Path(benchmark_snapshot["dataset_version"])
            run_id = uuid.uuid4().hex
            baseline_report = (
                paths.models / "backtests" / cohort / "runs" / run_id / f"{tier.value.lower()}.json"
            )
            baseline_predictions = (
                paths.predictions
                / "backtests"
                / cohort
                / "runs"
                / run_id
                / f"{tier.value.lower()}.parquet"
            )
            result = run_backtest_files(
                benchmark_dir / f"{tier.value.lower()}.parquet",
                tier,
                baseline_report,
                baseline_predictions,
                min_train_events=args.min_train_events,
                seed=args.seed,
                run_id=run_id,
            )
            print(f"status: {result['status']}")
            print(f"prediction_rows: {result['prediction_rows']}")
            print(f"metrics: {baseline_report}")
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
