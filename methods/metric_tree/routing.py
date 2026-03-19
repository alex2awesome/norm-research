"""Routing strategies: confidence threshold tuning and learned routing."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from .config import TreeConfig

logger = logging.getLogger("metric_tree.routing")


def tune_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    low: float = 0.5,
    high: float = 0.95,
    n_steps: int = 20,
) -> Tuple[float, float, float]:
    """Sweep confidence thresholds to optimize accuracy * resolution_rate.

    Only classifies examples where max(prob, 1-prob) >= threshold.

    Returns (best_threshold, best_accuracy, best_resolution_rate).
    """
    best_threshold = low
    best_score = 0.0
    best_acc = 0.0
    best_rate = 0.0

    for threshold in np.linspace(low, high, n_steps):
        confidence = np.maximum(probs, 1 - probs)
        resolved = confidence >= threshold
        n_resolved = resolved.sum()

        if n_resolved == 0:
            continue

        preds = (probs[resolved] >= 0.5).astype(int)
        accuracy = (preds == labels[resolved]).mean()
        resolution_rate = n_resolved / len(probs)
        score = accuracy * resolution_rate

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_acc = float(accuracy)
            best_rate = float(resolution_rate)

    logger.info(
        "Threshold tuning: best=%.3f acc=%.3f resolution=%.3f",
        best_threshold, best_acc, best_rate,
    )
    return best_threshold, best_acc, best_rate


def build_learned_router(
    scores: np.ndarray,
    correct_mask: np.ndarray,
    config: TreeConfig,
) -> CalibratedClassifierCV:
    """Fit a calibrated classifier to predict whether the parent will misclassify.

    Input: feature scores from the parent classifier.
    Target: 1 = parent is correct, 0 = parent is wrong.

    Returns a CalibratedClassifierCV that outputs P(correct | features).
    """
    y = correct_mask.astype(int)

    base_clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=config.random_seed,
    )

    router = CalibratedClassifierCV(
        estimator=base_clf,
        cv=min(config.cv_folds, max(2, int(y.sum()), int((1 - y).sum()))),
        method="sigmoid",
    )
    router.fit(scores, y)

    train_acc = (router.predict(scores) == y).mean()
    logger.info("Learned router train accuracy: %.3f", train_acc)

    return router
