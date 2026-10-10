"""Group-disjoint supervised pipelines. No model is fitted on import."""

from __future__ import annotations
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def validate_table(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Require a finite feature table and aligned nonmissing patient identifiers."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if x.ndim != 2 or x.shape[1] == 0 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite feature table")
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(x) or len(groups) != len(x):
        raise ValueError("Labels and groups must align with rows")
    if any(g is None or str(g) in {"", "nan"} for g in groups):
        raise ValueError("Missing patient identity")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Expected binary 0/1 labels")
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two independent patient groups required")
    return x, y, groups


def patient_split(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, test_size: float = 0.25, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Split entire patient groups, before fitting any feature transform."""
    x, y, groups = validate_table(x, y, groups)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    return next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(x, y, groups))


def make_pipeline(model: str = "svm", seed: int = 7) -> Pipeline:
    """Construct an unfitted pipeline; class balancing is enabled."""
    estimators = {
        "svm": lambda: SVC(class_weight="balanced"),
        "logistic": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
        "forest": lambda: RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=1
        ),
    }
    if model not in estimators:
        raise ValueError("Choose svm, logistic or forest")
    return Pipeline([("scale", StandardScaler()), ("model", estimators[model]())])


def fit_training_partition(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: str = "svm",
    test_size: float = 0.25,
    seed: int = 7,
) -> tuple[Pipeline, np.ndarray]:
    """Optional training entry point. Fit only training rows and return untouched test indices."""
    x, y, groups = validate_table(x, y, groups)
    train, test = patient_split(x, y, groups, test_size, seed)
    if len(np.unique(y[train])) != 2:
        raise ValueError("Training partition needs both classes; revise the group split")
    pipeline = make_pipeline(model, seed)
    pipeline.fit(x[train], y[train])
    return pipeline, test


def cross_validation_indices(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5, seed: int = 7
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified group folds; class balance is approximate at patient level."""
    x, y, groups = validate_table(x, y, groups)
    if not 2 <= folds <= len(np.unique(groups)):
        raise ValueError("Invalid number of folds")
    return list(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(x, y, groups))
