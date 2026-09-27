"""Small command-line entry point for bounded ingestion jobs."""

import argparse
from dataclasses import asdict
from pathlib import Path

from f1_ml_predictor.ingestion.jolpica import JolpicaSeasonIngestor
from f1_ml_predictor.paths import StoragePaths
from f1_ml_predictor.sources.jolpica import JolpicaClient, JolpicaError


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m f1_ml_predictor")
    subcommands = parser.add_subparsers(dest="command", required=True)
    ingest = subcommands.add_parser("ingest-season", help="Ingest one Jolpica season")
    ingest.add_argument("season", type=int)
    ingest.add_argument("--root", type=Path, default=Path.cwd())
    ingest.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.command == "ingest-season":
        try:
            with JolpicaClient() as client:
                report = JolpicaSeasonIngestor(StoragePaths(args.root), client).ingest_season(
                    args.season, refresh=args.refresh
                )
        except (JolpicaError, ValueError, OSError) as exc:
            parser.exit(1, f"Ingestion failed: {exc}\n")
        for key, value in asdict(report).items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
