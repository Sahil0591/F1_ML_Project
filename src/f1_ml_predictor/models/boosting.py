"""Small fixed boosting configurations; optional backends are imported on demand."""

import importlib
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

BACKENDS = ("hist", "xgboost", "lightgbm", "catboost")


def estimator(backend: str, task: str, seed: int, device: str = "cpu") -> Any:
    if backend not in BACKENDS or task not in {"position", "dnf"}:
        raise ValueError("unknown boosting backend or task")
    if device not in {"cpu", "cuda", "gpu"}:
        raise ValueError("unknown training device")
    classifier = task == "dnf"
    if backend == "hist":
        if device != "cpu":
            raise ValueError("scikit-learn histogram boosting uses CPU")
        cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
        return cls(
            max_iter=80,
            max_leaf_nodes=7,
            min_samples_leaf=5,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    module = importlib.import_module(backend)
    if backend == "xgboost":
        if device == "gpu":
            raise ValueError("XGBoost CUDA device must be cuda")
        cls = module.XGBClassifier if classifier else module.XGBRegressor
        return cls(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            reg_lambda=1.0,
            tree_method="hist",
            device=device,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    if backend == "lightgbm":
        cls = module.LGBMClassifier if classifier else module.LGBMRegressor
        return cls(
            n_estimators=80,
            num_leaves=7,
            max_depth=3,
            learning_rate=0.05,
            min_child_samples=5,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            device_type=device,
            deterministic=device == "cpu",
            force_col_wise=True,
            verbosity=-1,
        )
    if device == "gpu":
        raise ValueError("CatBoost CUDA device must be cuda")
    cls = module.CatBoostClassifier if classifier else module.CatBoostRegressor
    settings = dict(
        iterations=80,
        depth=3,
        learning_rate=0.05,
        l2_leaf_reg=1.0,
        random_seed=seed,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        task_type="GPU" if device == "cuda" else "CPU",
    )
    if device == "cuda":
        settings["devices"] = "0"
    return cls(**settings)
