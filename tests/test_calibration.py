import math
from typing import Any

import pytest

from f1_ml_predictor.models.calibration import (
    calibrated_probabilities,
    fit_binary_calibrator,
)


def test_sigmoid_uses_logits_and_softens_overconfident_scores() -> None:
    probabilities = [0.05] * 20 + [0.95] * 20
    labels = [True] * 5 + [False] * 15 + [True] * 15 + [False] * 5
    calibrator, metadata = fit_binary_calibrator(probabilities, labels)
    assert metadata["status"] == "fitted"
    assert metadata["method"] == "sigmoid"
    assert metadata["samples"] == 40
    assert metadata["positives"] == metadata["negatives"] == 20
    mapped = calibrated_probabilities(calibrator, [0.0, 0.05, 0.5, 0.95, 1.0], "sigmoid")
    assert all(math.isfinite(probability) and 0 <= probability <= 1 for probability in mapped)
    assert mapped == sorted(mapped)
    assert 0.05 < mapped[1] < 0.5 < mapped[3] < 0.95
    assert mapped[2] == pytest.approx(0.5, abs=1e-4)


def test_isotonic_maps_empirical_rates_and_clips_out_of_training_range() -> None:
    probabilities = [0.1] * 50 + [0.5] * 50 + [0.9] * 50
    labels = [True] * 10 + [False] * 40 + [True] * 25 + [False] * 25 + [True] * 40 + [False] * 10
    calibrator, metadata = fit_binary_calibrator(probabilities, labels, method="isotonic")
    assert metadata["status"] == "fitted"
    assert metadata["distinct_scores"] == 3
    assert calibrated_probabilities(calibrator, [0.0, 0.1, 0.5, 0.9, 1.0], "isotonic") == (
        pytest.approx([0.2, 0.2, 0.5, 0.8, 0.8])
    )


def test_identity_and_empty_predictions_are_safe() -> None:
    calibrator, metadata = fit_binary_calibrator(
        [0.0, 0.7, 1.0], [False, None, True], method="identity"
    )
    assert calibrator is None
    assert metadata["status"] == "identity"
    assert metadata["unknown_labels"] == 1
    assert calibrated_probabilities(calibrator, [0.0, 0.7, 1.0], "identity") == [0.0, 0.7, 1.0]
    assert calibrated_probabilities(calibrator, [], "sigmoid") == []


def test_unknown_labels_are_excluded_instead_of_becoming_negatives() -> None:
    probabilities = [0.1] * 5 + [0.9] * 5 + [0.99] * 20
    labels = [False] * 5 + [True] * 5 + [None] * 20
    calibrator, metadata = fit_binary_calibrator(probabilities, labels)
    assert calibrator is not None
    assert metadata["samples"] == 10
    assert metadata["negatives"] == metadata["positives"] == 5
    assert metadata["unknown_labels"] == 20


@pytest.mark.parametrize(
    "probabilities,labels,method",
    [
        ([0.2] * 4 + [0.8] * 4, [False] * 4 + [True] * 4, "sigmoid"),
        ([0.2] * 9 + [0.8], [False] * 9 + [True], "sigmoid"),
        ([0.2] * 20, [False] * 20, "sigmoid"),
        ([0.2] * 50 + [0.8] * 49, [False] * 50 + [True] * 49, "isotonic"),
        ([0.2] * 91 + [0.5] * 5 + [0.8] * 4, [False] * 91 + [True] * 9, "isotonic"),
        ([0.2] * 50 + [0.8] * 50, [False] * 50 + [True] * 50, "isotonic"),
    ],
)
def test_small_or_uninformative_history_is_skipped(
    probabilities: list[float], labels: list[bool | None], method: str
) -> None:
    calibrator, metadata = fit_binary_calibrator(probabilities, labels, method=method)
    assert calibrator is None
    assert metadata["status"] == "skipped"
    assert metadata["reason"].startswith("insufficient_")
    assert calibrated_probabilities(calibrator, probabilities, method) == probabilities


@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1, True, "0.2"]
)
def test_invalid_probabilities_are_rejected_for_fit_and_prediction(invalid: Any) -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        fit_binary_calibrator([invalid], [True], method="identity")
    with pytest.raises(ValueError, match="finite numbers"):
        calibrated_probabilities(None, [invalid], "identity")


@pytest.mark.parametrize("method", ["other", "platt", ""])
def test_unknown_method_is_rejected(method: str) -> None:
    with pytest.raises(ValueError, match="method"):
        fit_binary_calibrator([0.5], [True], method=method)
    with pytest.raises(ValueError, match="method"):
        calibrated_probabilities(None, [0.5], method)


def test_mismatched_rows_and_nonboolean_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        fit_binary_calibrator([0.5], [])
    with pytest.raises(ValueError, match="boolean or unknown"):
        fit_binary_calibrator([0.5], [1])  # type: ignore[list-item]


def test_mapping_rejects_invalid_model_outputs() -> None:
    class InvalidCalibrator:
        def predict(self, values: list[float]) -> list[float]:
            return [float("nan")] * len(values)

    with pytest.raises(ValueError, match="finite numbers"):
        calibrated_probabilities(InvalidCalibrator(), [0.5], "isotonic")


def test_sigmoid_seed_repeats_the_same_fit() -> None:
    probabilities = [0.1] * 10 + [0.9] * 10
    labels = [False] * 8 + [True] * 2 + [False] * 2 + [True] * 8
    first, _ = fit_binary_calibrator(probabilities, labels, seed=17)
    second, _ = fit_binary_calibrator(probabilities, labels, seed=17)
    assert calibrated_probabilities(first, [0.2, 0.5, 0.8], "sigmoid") == (
        calibrated_probabilities(second, [0.2, 0.5, 0.8], "sigmoid")
    )
