"""L1 logistic regression on binary feature matrix + evaluation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    train_metrics: Dict[str, float]
    eval_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    feature_names: List[str]
    coefficients: np.ndarray
    active_features: List[Tuple[str, float]]  # (name, coefficient) for nonzero
    model: Any


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics from true labels and predicted probabilities."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = 0.5
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["pr_auc"] = 0.0
    try:
        metrics["pearson_r"] = float(np.corrcoef(y_true, y_prob)[0, 1])
    except (ValueError, IndexError):
        metrics["pearson_r"] = 0.0
    return metrics


def fit_and_evaluate(
    train_X: np.ndarray,
    train_y: np.ndarray,
    eval_X: np.ndarray,
    eval_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    feature_names: List[str],
) -> PredictionResult:
    """Fit L1 logistic regression and evaluate on all splits.

    Uses sklearn LogisticRegressionCV with elastic net (l1_ratio=1.0).
    """
    logger.info(
        "Fitting L1 logistic regression: train=%d, eval=%d, test=%d, features=%d",
        len(train_y), len(eval_y), len(test_y), train_X.shape[1],
    )

    model = LogisticRegressionCV(
        penalty="elasticnet",
        l1_ratios=[1.0],
        solver="saga",
        Cs=10,
        cv=5,
        scoring="roc_auc",
        max_iter=5000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(train_X, train_y)

    # Get coefficients
    coefs = model.coef_.flatten()
    active = [
        (name, float(coef))
        for name, coef in zip(feature_names, coefs)
        if abs(coef) > 1e-6
    ]
    active.sort(key=lambda x: abs(x[1]), reverse=True)

    logger.info(
        "Model: %d/%d active features, C=%.4f",
        len(active), len(feature_names), model.C_[0],
    )
    for name, coef in active[:10]:
        logger.info("  %+.4f  %s", coef, name)

    # Evaluate
    train_prob = model.predict_proba(train_X)[:, 1]
    eval_prob = model.predict_proba(eval_X)[:, 1]
    test_prob = model.predict_proba(test_X)[:, 1]

    train_metrics = _compute_metrics(train_y, train_prob)
    eval_metrics = _compute_metrics(eval_y, eval_prob)
    test_metrics = _compute_metrics(test_y, test_prob)

    logger.info("Train: %s", {k: f"{v:.4f}" for k, v in train_metrics.items()})
    logger.info("Eval:  %s", {k: f"{v:.4f}" for k, v in eval_metrics.items()})
    logger.info("Test:  %s", {k: f"{v:.4f}" for k, v in test_metrics.items()})

    return PredictionResult(
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        test_metrics=test_metrics,
        feature_names=feature_names,
        coefficients=coefs,
        active_features=active,
        model=model,
    )
