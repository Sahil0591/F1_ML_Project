"""Freeze a separate audited binary DNF benchmark from retained source bytes."""

import argparse
import json
from pathlib import Path

from f1_ml_predictor.trust.binary_dnf import build_binary_dnf_benchmark

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT
    / "data/raw/dnf_audit_v1"
    / "capture-4a809a21c40d5f65c86d1bd6bc420471119a33578228fa597379c0725e604a89.json"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", type=Path, default=CAPTURE)
    args = parser.parse_args()
    print(json.dumps(build_binary_dnf_benchmark(ROOT, args.capture), indent=2))
