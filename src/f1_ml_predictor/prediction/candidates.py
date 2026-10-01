"""Strength, baseline and DNF candidates for cutoff-specific race models.

Every joint candidate returns latent utilities (higher is stronger) for one
race roster, so baseline families can drive the same coherent sampler as the
boosting backends. Unknown finishing positions are never invented: regressors
skip them and the Plackett-Luce regression ranks them after every known finisher.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from f1_ml_predictor.models.backtest import _at_most_k
from f1_ml_predictor.models.boosting import BACKENDS, estimator
from f1_ml_predictor.models.hardware import choose_device

PL_PENALTY = 1.0
HEURISTIC_KEYS = (
    "grid_position",
    "qualifying_position",
    "driver_finish_mean_any_5",
    "recent_finish_mean_3",
    "constructor_average_finish_last_5",
)


def matrix(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> np.ndarray[Any, Any]:
    return np.asarray(
        [
            [float(row[name]) if row.get(name) is not None else np.nan for name in columns]
            for row in rows
        ],
        dtype=float,
    )


def _linear(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def _races(rows: list[dict[str, Any]]) -> list[list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["event_id"]].append(index)
    return list(grouped.values())


def _ranking(rows: list[dict[str, Any]], indices: list[int]) -> tuple[list[int], int]:
    """Known finishers by position, then unknown positions; returns stage count."""
    known = sorted(
        (index for index in indices if rows[index].get("label_position") is not None),
        key=lambda index: rows[index]["label_position"],
    )
    unknown = [index for index in indices if rows[index].get("label_position") is None]
    stages = min(len(known), len(indices) - 1)
    return known + unknown, stages


class PlackettLuceRegression:
    """Rank-ordered logit with an L2 penalty on standardized predictors."""

    def __init__(self, penalty: float = PL_PENALTY) -> None:
        self.penalty = penalty
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scale = StandardScaler()
        self.coef_: np.ndarray[Any, Any] = np.zeros(0)

    def fit(self, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> PlackettLuceRegression:
        features = self.scale.fit_transform(self.imputer.fit_transform(matrix(rows, columns)))
        races = [_ranking(rows, indices) for indices in _races(rows)]
        orders = [(np.asarray(order), stages) for order, stages in races if stages > 0]

        def loss(beta: np.ndarray[Any, Any]) -> tuple[float, np.ndarray[Any, Any]]:
            total = 0.5 * self.penalty * float(beta @ beta)
            gradient = self.penalty * beta.copy()
            for order, stages in orders:
                x = features[order]
                u = x @ beta
                shift = u.max()
                e = np.exp(u - shift)
                suffix = np.cumsum(e[::-1])[::-1]
                total -= float(u[:stages].sum() - (np.log(suffix[:stages]) + shift).sum())
                inverse = np.cumsum(1.0 / suffix[:stages])
                weight = e * inverse[np.minimum(np.arange(len(e)), stages - 1)]
                gradient -= x[:stages].sum(axis=0) - weight @ x
            return total, gradient

        result = minimize(
            loss,
            np.zeros(features.shape[1]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        self.coef_ = np.asarray(result.x)
        return self

    def utility(self, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> np.ndarray[Any, Any]:
        features = self.scale.transform(self.imputer.transform(matrix(rows, columns)))
        return np.asarray(features @ self.coef_, dtype=float)


@dataclass
class StrengthModel:
    name: str
    columns: tuple[str, ...]
    model: Any
    device: str = "cpu"
    device_reason: str = "linear model"
    metadata: dict[str, Any] = field(default_factory=dict)

    def utility(self, rows: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
        features = matrix(rows, self.columns)
        with threadpool_limits(limits=1):
            if self.name == "logistic_pl":
                positive = list(self.model.named_steps["model"].classes_).index(1)
                probability = self.model.predict_proba(features)[:, positive]
                return np.asarray(np.log(np.clip(probability, 1e-12, 1.0)), dtype=float)
            if self.name == "ridge_pl":
                return -np.asarray(self.model.predict(features), dtype=float)
            if self.name == "pl_regression":
                return np.asarray(self.model.utility(rows, self.columns))
            imputer, regressor = self.model
            fraction = np.asarray(regressor.predict(imputer.transform(features)), dtype=float)
            return -fraction * len(rows)


def fit_strength(
    name: str,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    *,
    seed: int,
    device: str = "cpu",
    hardware: dict[str, Any] | None = None,
) -> StrengthModel:
    """Fit one joint candidate on complete earlier races."""
    known = [row for row in rows if row.get("label_position") is not None]
    with threadpool_limits(limits=1):
        if name == "logistic_pl":
            model = _linear(LogisticRegression(l1_ratio=0.0, max_iter=2000, random_state=seed))
            model.fit(matrix(rows, columns), np.asarray([int(row["label_winner"]) for row in rows]))
            return StrengthModel(name, columns, model)
        if name == "ridge_pl":
            model = _linear(Ridge(alpha=1.0))
            model.fit(matrix(known, columns), np.asarray([row["label_position"] for row in known]))
            return StrengthModel(name, columns, model)
        if name == "pl_regression":
            return StrengthModel(name, columns, PlackettLuceRegression().fit(rows, columns))
        if name not in BACKENDS:
            raise ValueError(f"unknown strength candidate {name}")
        chosen, reason = choose_device(name, len(rows), device, hardware)
        sizes: dict[str, int] = defaultdict(int)
        for row in rows:
            sizes[row["event_id"]] += 1
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        features = imputer.fit_transform(matrix(known, columns))
        target = np.asarray([row["label_position"] / sizes[row["event_id"]] for row in known])
        try:
            regressor = estimator(name, "position", seed, chosen)
            regressor.fit(features, target)
        except Exception as exc:
            if chosen == "cpu":
                raise
            reason = f"GPU training failed; CPU fallback: {type(exc).__name__}"
            chosen = "cpu"
            regressor = estimator(name, "position", seed, chosen)
            regressor.fit(features, target)
        return StrengthModel(name, columns, (imputer, regressor), chosen, reason)


def baseline_marginals(
    train: list[dict[str, Any]], test: list[dict[str, Any]], columns: tuple[str, ...], seed: int
) -> dict[str, dict[str, np.ndarray[Any, Any]]]:
    """Heuristic and logistic/Ridge marginal baselines for one test roster."""
    count = len(test)
    keys = [key for key in HEURISTIC_KEYS if key in columns]
    position = np.asarray(
        [
            next(
                (float(row[key]) for key in keys if row.get(key) is not None and row[key] > 0),
                float(count + 1),
            )
            for row in test
        ]
    )
    inverse = 1.0 / position
    order = np.lexsort((np.asarray([row["driver_id"] for row in test]), position))
    rank = np.empty(count)
    rank[order] = np.arange(1, count + 1)
    heuristic = {
        "winner": inverse / inverse.sum(),
        "podium": (rank <= 3).astype(float),
        "expected": rank,
    }
    features = matrix(train, columns)
    test_features = matrix(test, columns)
    result = {"heuristic": heuristic}
    with threadpool_limits(limits=1):
        logistic: dict[str, np.ndarray[Any, Any]] = {}
        for task, label in (("winner", "label_winner"), ("podium", "label_podium")):
            model = _linear(LogisticRegression(l1_ratio=0.0, max_iter=2000, random_state=seed))
            model.fit(features, np.asarray([int(row[label]) for row in train]))
            positive = list(model.named_steps["model"].classes_).index(1)
            raw = model.predict_proba(test_features)[:, positive]
            logistic[task] = (
                raw / raw.sum() if task == "winner" else np.asarray(_at_most_k(raw.tolist(), 3))
            )
        known = [row for row in train if row.get("label_position") is not None]
        ridge = _linear(Ridge(alpha=1.0))
        ridge.fit(matrix(known, columns), np.asarray([row["label_position"] for row in known]))
        predicted = np.asarray(ridge.predict(test_features), dtype=float)
        order = np.lexsort((np.asarray([row["driver_id"] for row in test]), predicted))
        linear_rank = np.empty(count)
        linear_rank[order] = np.arange(1, count + 1)
        logistic["expected"] = linear_rank
    result["logistic"] = logistic
    return result


@dataclass
class DnfModel:
    name: str
    columns: tuple[str, ...]
    model: Any
    prior: float
    labels: int
    positives: int

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
        if self.model is None:
            return np.full(len(rows), self.prior)
        with threadpool_limits(limits=1):
            probabilities = self.model.predict_proba(matrix(rows, self.columns))
        positive = list(self.model.classes_ if hasattr(self.model, "classes_") else [0, 1])
        return np.asarray(probabilities[:, positive.index(1)], dtype=float)


def fit_dnf(name: str, rows: list[dict[str, Any]], columns: tuple[str, ...], seed: int) -> DnfModel:
    """Fit on audited binary DNF labels only; DNS, DSQ and unknown rows are absent."""
    labelled = [row for row in rows if row.get("label_dnf") is not None]
    labels = np.asarray([int(row["label_dnf"]) for row in labelled])
    prior = float((1 + labels.sum()) / (2 + len(labels)))
    if name == "prior" or len(set(labels.tolist())) < 2:
        return DnfModel(name, columns, None, prior, len(labels), int(labels.sum()))
    with threadpool_limits(limits=1):
        if name == "logistic":
            model: Any = _linear(LogisticRegression(l1_ratio=0.0, max_iter=2000, random_state=seed))
        elif name == "hist":
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=80,
                            max_leaf_nodes=7,
                            min_samples_leaf=20,
                            learning_rate=0.05,
                            l2_regularization=1.0,
                            early_stopping=False,
                            random_state=seed,
                        ),
                    ),
                ]
            )
        else:
            raise ValueError(f"unknown DNF candidate {name}")
        model.fit(matrix(labelled, columns), labels)
    return DnfModel(name, columns, model, prior, len(labels), int(labels.sum()))
