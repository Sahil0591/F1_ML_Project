"""Freeze cutoff-valid FIA practice classification inputs."""

import argparse
import json
from pathlib import Path

from f1_ml_predictor.trust.practice_history import audit_historical_practice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_report", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(audit_historical_practice(args.root, args.coverage_report), indent=2))


if __name__ == "__main__":
    main()
