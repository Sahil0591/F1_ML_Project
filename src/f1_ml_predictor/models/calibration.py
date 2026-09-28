"""Optional binary calibration fitted only on caller-supplied held-out history.

The caller owns the chronological train/calibration/test split. This helper does
not choose rows, refit a predictor, or use final evaluation labels automatically.
"""

from __future__ import annotations

import math
from typing import Any

_METHODS = {"identity", "sigmoid", "isotonic"}
_LOGIT_EPSILON = 1e-6


def _validated(probabilities: list[float]) -> list[float]:
    values = []
    for probability in probabilities:
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("calibration probabilities must be finite numbers between 0 and 1")
        value = float(probability)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("calibration probabilities must be finite numbers between 0 and 1")
        values.append(value)
    return values


def _method(method: str) -> None:
    if method not in _METHODS:
        raise ValueError("calibration method must be identity, sigmoid, or isotonic")


def _logits(probabilities: list[float]) -> list[list[float]]:
    result = []
    for probability in probabilities:
        bounded = min(1 - _LOGIT_EPSILON, max(_LOGIT_EPSILON, probability))
        result.append([math.log(bounded / (1 - bounded))])
    return result


def fit_binary_calibrator(
    probabilities: list[float],
    labels: list[bool | None],
    *,
    method: str = "sigmoid",
    seed: int = 42,
) -> tuple[Any | None, dict[str, Any]]:
    """Fit an optional mapping, or report why the known labels are insufficient.

    Sigmoid needs at least ten known labels with two of each class. Isotonic
    needs at least one hundred known labels, ten of each class, and three distinct
    probability scores. Unknown labels are excluded without inventing negatives.
    """
    _method(method)
    values = _validated(probabilities)
    if len(values) != len(labels):
        raise ValueError("calibration probabilities and labels must have equal lengths")
    if type(seed) is not int:
        raise ValueError("calibration seed must be an integer")
    if any(label is not None and type(label) is not bool for label in labels):
        raise ValueError("calibration labels must be boolean or unknown")
    known = [
        (value, label) for value, label in zip(values, labels, strict=True) if label is not None
    ]
    scores = [value for value, _ in known]
    outcomes = [int(label) for _, label in known]
    positives = sum(outcomes)
    negatives = len(known) - positives
    metadata: dict[str, Any] = {
        "method": method,
        "status": "identity" if method == "identity" else "skipped",
        "provided_samples": len(values),
        "samples": len(known),
        "positives": positives,
        "negatives": negatives,
        "unknown_labels": len(values) - len(known),
        "distinct_scores": len(set(scores)),
    }
    if method == "identity":
        return None, metadata
    minimum_samples, minimum_class = (100, 10) if method == "isotonic" else (10, 2)
    metadata.update({"minimum_samples": minimum_samples, "minimum_class_count": minimum_class})
    if len(known) < minimum_samples or min(positives, negatives) < minimum_class:
        metadata["reason"] = "insufficient_samples_or_class_counts"
        return None, metadata
    if method == "isotonic" and len(set(scores)) < 3:
        metadata["reason"] = "insufficient_distinct_scores"
        return None, metadata
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression(random_state=seed, max_iter=1000)
        calibrator.fit(_logits(scores), outcomes)
    else:
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(scores, outcomes)
    metadata["status"] = "fitted"
    return calibrator, metadata


def calibrated_probabilities(
    calibrator: Any | None, probabilities: list[float], method: str
) -> list[float]:
    """Apply the chosen mapping; a skipped/identity fit retains input probabilities."""
    _method(method)
    values = _validated(probabilities)
    if calibrator is None or method == "identity" or not values:
        return values
    if method == "sigmoid":
        predictions = [float(row[1]) for row in calibrator.predict_proba(_logits(values))]
    else:
        predictions = [float(value) for value in calibrator.predict(values)]
    if len(predictions) != len(values):
        raise ValueError("calibrator returned the wrong number of probabilities")
    return _validated(predictions)
