"""Out-of-distribution diagnostics for a live roster against its training rows."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from f1_ml_predictor.prediction.candidates import matrix


def _nearest(rows: np.ndarray[Any, Any], reference: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    distances = np.sqrt(((rows[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2))
    return np.asarray(distances.min(axis=1))


def ood_report(
    training: list[dict[str, Any]],
    live: list[dict[str, Any]],
    columns: tuple[str, ...],
    *,
    circuit_id: str,
    circuit_name: str,
) -> dict[str, Any]:
    """Unseen circuit, value ranges, missingness patterns and nearest-row distances."""
    seen = sorted({row["circuit_id"] for row in training})
    numeric = [name for name in columns if not name.endswith("_missing")]
    flags = [name for name in columns if name.endswith("_missing")]
    ranges = []
    for name in numeric:
        values = [float(row[name]) for row in training if row.get(name) is not None]
        if not values:
            continue
        low, high = min(values), max(values)
        for row in live:
            value = row.get(name)
            if value is not None and not low <= float(value) <= high:
                ranges.append(
                    {
                        "driver_id": row["driver_id"],
                        "feature": name,
                        "value": float(value),
                        "training_min": low,
                        "training_max": high,
                    }
                )
    patterns = Counter(tuple(bool(row[name]) for name in flags) for row in training)
    novel = [
        row["driver_id"] for row in live if patterns[tuple(bool(row[name]) for name in flags)] == 0
    ]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    reference = scaler.fit_transform(imputer.fit_transform(matrix(training, columns)))
    events = np.asarray([row["event_id"] for row in training])
    baseline = []
    for event in sorted(set(events)):
        inside = events == event
        baseline.extend(_nearest(reference[inside], reference[~inside]).tolist())
    live_matrix = scaler.transform(imputer.transform(matrix(live, columns)))
    live_distance = _nearest(live_matrix, reference)
    threshold = float(np.quantile(baseline, 0.99))
    percentiles = [float((np.asarray(baseline) <= value).mean()) for value in live_distance]
    far = [
        row["driver_id"]
        for row, value in zip(live, live_distance, strict=True)
        if value > threshold
    ]
    unseen = circuit_id not in seen
    reasons = []
    if unseen:
        reasons.append(f"circuit {circuit_id} is absent from the Gold training races")
    if far:
        reasons.append(f"{len(far)} drivers beyond the 99th percentile nearest-row distance")
    if novel:
        reasons.append(f"{len(novel)} drivers with a missing-value pattern never seen in training")
    return {
        "status": "out_of_distribution" if reasons else "in_distribution",
        "reasons": reasons,
        "circuit": {
            "circuit_id": circuit_id,
            "circuit_name": circuit_name,
            "unseen_in_training": unseen,
            "training_circuits": len(seen),
        },
        "feature_range_violations": ranges,
        "novel_missing_pattern_drivers": novel,
        "distance": {
            "method": "Euclidean distance to the nearest training row after median imputation "
            "and standardisation; reference is leave-one-race-out training distances",
            "reference_99th_percentile": threshold,
            "live_percentiles": {
                row["driver_id"]: value for row, value in zip(live, percentiles, strict=True)
            },
            "beyond_99th_percentile": far,
        },
    }
