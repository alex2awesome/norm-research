"""Mahalanobis distance leaf model for binary feature vectors."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("metric_tree.mahalanobis")


def fit_mahalanobis(
    scores: np.ndarray,
    labels: np.ndarray,
    regularization: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit Mahalanobis distance model on binary feature vectors.

    Parameters
    ----------
    scores : ndarray of shape (n, d) — binary 0/1 feature vectors
    labels : ndarray of shape (n,) — binary 0/1 labels
    regularization : ridge regularization added to covariance diagonal

    Returns
    -------
    (positive_centroid, negative_centroid, cov_inv, base_rate)
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    base_rate = n_pos / max(n_pos + n_neg, 1)

    if n_pos == 0 or n_neg == 0:
        d = scores.shape[1] if scores.ndim == 2 else 1
        logger.warning("Degenerate split: %d pos, %d neg — using uniform centroid", n_pos, n_neg)
        centroid = scores.mean(axis=0) if len(scores) > 0 else np.zeros(d)
        return centroid, centroid, np.eye(d), base_rate

    pos_centroid = scores[pos_mask].mean(axis=0)
    neg_centroid = scores[neg_mask].mean(axis=0)

    # Pooled covariance from both classes
    cov = np.cov(scores.T) if scores.shape[0] > 1 else np.eye(scores.shape[1])
    # Handle 1D case where np.cov returns a scalar
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    elif cov.ndim == 1:
        cov = cov.reshape(1, 1)

    # Regularize
    cov += regularization * np.eye(cov.shape[0])

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix singular, using pseudo-inverse")
        cov_inv = np.linalg.pinv(cov)

    return pos_centroid, neg_centroid, cov_inv, base_rate


def predict_mahalanobis(
    scores: np.ndarray,
    positive_centroid: np.ndarray,
    negative_centroid: np.ndarray,
    cov_inv: np.ndarray,
    base_rate: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using Mahalanobis distance to class centroids.

    Parameters
    ----------
    scores : ndarray of shape (n, d)
    positive_centroid, negative_centroid : ndarray of shape (d,)
    cov_inv : ndarray of shape (d, d)
    base_rate : prior P(positive)

    Returns
    -------
    (predictions, probabilities) where predictions are 0/1 and
    probabilities are P(positive).
    """
    if len(scores) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    # Mahalanobis distance: d(x, mu) = sqrt((x - mu)^T Sigma^{-1} (x - mu))
    diff_pos = scores - positive_centroid
    diff_neg = scores - negative_centroid

    # Quadratic form: (x - mu)^T Sigma^{-1} (x - mu)
    dist_pos_sq = np.sum(diff_pos @ cov_inv * diff_pos, axis=1)
    dist_neg_sq = np.sum(diff_neg @ cov_inv * diff_neg, axis=1)

    # Convert to probabilities via softmax-like scheme with base rate prior
    # log P(pos | x) ∝ -0.5 * d_pos^2 + log(base_rate)
    # log P(neg | x) ∝ -0.5 * d_neg^2 + log(1 - base_rate)
    log_prior_pos = np.log(max(base_rate, 1e-10))
    log_prior_neg = np.log(max(1 - base_rate, 1e-10))

    log_score_pos = -0.5 * dist_pos_sq + log_prior_pos
    log_score_neg = -0.5 * dist_neg_sq + log_prior_neg

    # Numerically stable softmax
    max_log = np.maximum(log_score_pos, log_score_neg)
    prob_pos = np.exp(log_score_pos - max_log) / (
        np.exp(log_score_pos - max_log) + np.exp(log_score_neg - max_log)
    )

    predictions = (prob_pos >= 0.5).astype(int)
    return predictions, prob_pos
