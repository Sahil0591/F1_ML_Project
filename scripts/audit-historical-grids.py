"""Freeze cutoff-valid FIA grid PDFs for the current Gold coverage report.

--cohort provisional_grid audits the first published race grid at its own cutoff.
"""

import argparse
import json
from pathlib import Path

from f1_ml_predictor.trust.grid_history import COHORTS, audit_historical_grids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_report", type=Path)
    parser.add_argument("--cohort", choices=COHORTS, default="post_qualifying")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(audit_historical_grids(args.root, args.coverage_report, args.cohort), indent=2)
    )


if __name__ == "__main__":
    main()
