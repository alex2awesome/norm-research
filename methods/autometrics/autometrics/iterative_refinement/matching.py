from __future__ import annotations

from bisect import bisect_left
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def normalize_pair_id(id_a: str, id_b: str) -> Tuple[str, str]:
    a = str(id_a)
    b = str(id_b)
    return (a, b) if a <= b else (b, a)


def exact_match(
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    feature_columns: List[str],
    positive_label: int,
    k_pairs: int,
    seen_pairs: set[Tuple[str, str]],
    max_feature_dist: Optional[float] = None,
    exact_only: bool = False,
) -> List[Tuple[str, str, float]]:
    """Find pos/neg pairs with identical (or nearest) metric score profiles.

    Two-pass strategy:
    1. **Exact**: Round each metric score to the nearest integer (scores are
       1-5 Likert) and group by the resulting score tuple.  Within each group
       that contains both a positive and a negative example, emit pairs with
       distance 0.
    2. **Near-exact**: If fewer than *k_pairs* exact matches are found,
       fall back to brute-force L1 nearest-neighbour search over the raw
       (unrounded) feature vectors.  An optional *max_feature_dist* caps
       the per-feature average distance.

    If *exact_only* is True, skip pass 2 and return only exact matches.

    Returns a list of ``(pos_id, neg_id, distance)`` sorted by distance,
    truncated to *k_pairs*.
    """
    if not feature_columns:
        return []

    positives = df[df[label_column] == positive_label].copy()
    negatives = df[df[label_column] != positive_label].copy()
    if positives.empty or negatives.empty:
        return []

    n_features = len(feature_columns)

    # ── Pass 1: exact matches on rounded scores ──
    def _score_key_from_vals(vals: list) -> tuple:
        return tuple(round(float(v)) for v in vals)

    pos_ids_list = positives[id_column].astype(str).tolist()
    pos_feat_vals = positives[feature_columns].values
    pos_keys = {pos_ids_list[i]: _score_key_from_vals(pos_feat_vals[i]) for i in range(len(pos_ids_list))}

    neg_ids_list = negatives[id_column].astype(str).tolist()
    neg_feat_vals = negatives[feature_columns].values
    neg_keys = {neg_ids_list[i]: _score_key_from_vals(neg_feat_vals[i]) for i in range(len(neg_ids_list))}

    # Invert: key -> list of IDs
    from collections import defaultdict
    pos_by_key: dict[tuple, List[str]] = defaultdict(list)
    neg_by_key: dict[tuple, List[str]] = defaultdict(list)
    for pid, key in pos_keys.items():
        pos_by_key[key].append(pid)
    for nid, key in neg_keys.items():
        neg_by_key[key].append(nid)

    candidates: List[Tuple[str, str, float]] = []
    used_pos: set[str] = set()
    used_neg: set[str] = set()

    for key in pos_by_key:
        if key not in neg_by_key:
            continue
        for pid in pos_by_key[key]:
            if pid in used_pos:
                continue
            for nid in neg_by_key[key]:
                if nid in used_neg:
                    continue
                pair_key = normalize_pair_id(pid, nid)
                if pair_key in seen_pairs:
                    continue
                candidates.append((pid, nid, 0.0))
                used_pos.add(pid)
                used_neg.add(nid)
                break  # one match per positive
            if len(candidates) >= k_pairs:
                break
        if len(candidates) >= k_pairs:
            break

    if len(candidates) >= k_pairs:
        return candidates[:k_pairs]

    if exact_only:
        return candidates

    # ── Pass 2: near-exact (L1 nearest neighbour) ──
    pos_ids = positives[id_column].astype(str).tolist()
    neg_ids = negatives[id_column].astype(str).tolist()
    pos_feats = positives[feature_columns].values.astype(float)
    neg_feats = negatives[feature_columns].values.astype(float)

    # Pairwise L1 distance matrix: (n_pos, n_neg), normalised per feature
    # so the distance is the mean absolute difference per metric.
    dist_matrix = np.abs(pos_feats[:, None, :] - neg_feats[None, :, :]).mean(axis=2)

    # Flatten and sort by distance
    flat_indices = np.argsort(dist_matrix, axis=None)
    already_seen = {(c[0], c[1]) for c in candidates}

    for flat_idx in flat_indices:
        if len(candidates) >= k_pairs:
            break
        pi, ni = divmod(int(flat_idx), len(neg_ids))
        pid = pos_ids[pi]
        nid = neg_ids[ni]
        if (pid, nid) in already_seen:
            continue
        pair_key = normalize_pair_id(pid, nid)
        if pair_key in seen_pairs:
            continue
        dist = float(dist_matrix[pi, ni])
        if max_feature_dist is not None and dist > max_feature_dist:
            break  # sorted, so all remaining are worse
        candidates.append((pid, nid, dist))
        already_seen.add((pid, nid))

    candidates.sort(key=lambda x: x[2])
    return candidates[:k_pairs]


def mahalanobis_match(
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    feature_columns: List[str],
    positive_label: int,
    k_pairs: int,
    seen_pairs: set[Tuple[str, str]],
    prob_column: Optional[str] = None,
    propensity_caliper: Optional[float] = None,
    max_mahal_dist: Optional[float] = None,
) -> List[Tuple[str, str, float]]:
    """Find pos/neg pairs closest in Mahalanobis distance over metric scores.

    Mahalanobis distance accounts for correlations between features, so pairs
    that differ on correlated metrics are penalised less than pairs that differ
    on independent metrics.  This provides a smooth transition between exact
    matching (identical scores) and propensity matching (single scalar).

    If *prob_column* and *propensity_caliper* are both provided, only consider
    pairs within the caliper on propensity score first, then rank by
    Mahalanobis distance within that caliper.  This implements the Rubin-
    recommended "caliper + Mahalanobis" approach.

    Falls back to standardised Euclidean distance (diagonal covariance) if
    the full covariance matrix is singular.

    Returns a list of ``(pos_id, neg_id, mahalanobis_distance)`` sorted by
    distance, truncated to *k_pairs*.
    """
    if not feature_columns:
        return []

    positives = df[df[label_column] == positive_label].copy()
    negatives = df[df[label_column] != positive_label].copy()
    if positives.empty or negatives.empty:
        return []

    pos_ids = positives[id_column].astype(str).tolist()
    neg_ids = negatives[id_column].astype(str).tolist()
    pos_feats = positives[feature_columns].values.astype(float)
    neg_feats = negatives[feature_columns].values.astype(float)

    # Compute pooled covariance from both classes
    all_feats = np.vstack([pos_feats, neg_feats])
    n_features = all_feats.shape[1]

    # Replace NaNs with column means for covariance estimation
    col_means = np.nanmean(all_feats, axis=0)
    nan_mask = np.isnan(all_feats)
    if nan_mask.any():
        for j in range(n_features):
            all_feats[nan_mask[:, j], j] = col_means[j]
        # Also fix pos/neg arrays
        pos_feats = all_feats[:len(pos_ids)]
        neg_feats = all_feats[len(pos_ids):]

    cov = np.cov(all_feats, rowvar=False)

    # Regularise: add small diagonal to handle near-singular covariance
    # (common when features are highly correlated or n_features > n_samples)
    reg = 1e-4 * np.eye(n_features)
    cov_reg = cov + reg

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        # Fallback: standardised Euclidean (diagonal only)
        stds = np.std(all_feats, axis=0)
        stds[stds < 1e-8] = 1.0
        cov_inv = np.diag(1.0 / (stds ** 2))

    # Compute pairwise Mahalanobis distances: (n_pos, n_neg)
    # d(x,y) = sqrt((x-y)^T @ cov_inv @ (x-y))
    diffs = pos_feats[:, None, :] - neg_feats[None, :, :]  # (n_pos, n_neg, n_feat)
    # Efficient: (diffs @ cov_inv) * diffs summed over features
    mahal_sq = np.einsum('ijk,kl,ijl->ij', diffs, cov_inv, diffs)
    mahal_sq = np.maximum(mahal_sq, 0.0)  # numerical safety
    dist_matrix = np.sqrt(mahal_sq)

    # Optional propensity caliper: mask pairs outside caliper
    if prob_column is not None and propensity_caliper is not None:
        if prob_column in positives.columns and prob_column in negatives.columns:
            pos_probs = positives[prob_column].values.astype(float)
            neg_probs = negatives[prob_column].values.astype(float)
            prop_diff = np.abs(pos_probs[:, None] - neg_probs[None, :])
            dist_matrix = np.where(prop_diff <= propensity_caliper, dist_matrix, np.inf)

    # Greedily pick closest pairs
    flat_indices = np.argsort(dist_matrix, axis=None)
    candidates: List[Tuple[str, str, float]] = []
    used_pos: set[str] = set()
    used_neg: set[str] = set()

    for flat_idx in flat_indices:
        if len(candidates) >= k_pairs:
            break
        pi, ni = divmod(int(flat_idx), len(neg_ids))
        pid = pos_ids[pi]
        nid = neg_ids[ni]
        dist = float(dist_matrix[pi, ni])
        if np.isinf(dist):
            break  # all remaining are outside caliper
        if max_mahal_dist is not None and dist > max_mahal_dist:
            break
        if pid in used_pos or nid in used_neg:
            continue
        pair_key = normalize_pair_id(pid, nid)
        if pair_key in seen_pairs:
            continue
        candidates.append((pid, nid, dist))
        used_pos.add(pid)
        used_neg.add(nid)

    return candidates


def propensity_match(
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    prob_column: str,
    positive_label: int,
    k_pairs: int,
    caliper: float,
    seen_pairs: set[Tuple[str, str]],
) -> List[Tuple[str, str, float]]:
    positives = df[df[label_column] == positive_label].copy()
    negatives = df[df[label_column] != positive_label].copy()
    if positives.empty or negatives.empty:
        return []

    neg_sorted = negatives[[id_column, prob_column]].copy()
    neg_sorted[id_column] = neg_sorted[id_column].astype(str)
    neg_sorted = neg_sorted.sort_values(by=prob_column).reset_index(drop=True)
    neg_probs = neg_sorted[prob_column].tolist()

    pos_id_strs = positives[id_column].astype(str).tolist()
    pos_probs = positives[prob_column].tolist()

    candidates: List[Tuple[str, str, float]] = []
    for pos_id, pos_prob in zip(pos_id_strs, pos_probs):
        pos_prob = float(pos_prob)
        idx = bisect_left(neg_probs, pos_prob)
        neighbor_indices = [idx - 1, idx]
        best = None
        best_diff = None
        for n_idx in neighbor_indices:
            if n_idx < 0 or n_idx >= len(neg_sorted):
                continue
            neg_id = neg_sorted.iloc[n_idx][id_column]
            pair_key = normalize_pair_id(pos_id, neg_id)
            if pair_key in seen_pairs:
                continue
            diff = abs(pos_prob - float(neg_sorted.iloc[n_idx][prob_column]))
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best = (pos_id, neg_id, diff)
        if best and best_diff is not None and best_diff <= caliper:
            candidates.append(best)

    candidates.sort(key=lambda x: x[2])
    return candidates[:k_pairs]


def residual_select(
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    prob_column: str,
    positive_label: int,
    k_pairs: int,
) -> Tuple[List[str], List[str]]:
    positives = df[df[label_column] == positive_label].copy()
    negatives = df[df[label_column] != positive_label].copy()
    if positives.empty or negatives.empty:
        return [], []

    hard_pos = positives.sort_values(by=prob_column, ascending=True).head(k_pairs)[id_column].astype(str).tolist()
    hard_neg = negatives.sort_values(by=prob_column, ascending=False).head(k_pairs)[id_column].astype(str).tolist()
    return hard_pos, hard_neg
