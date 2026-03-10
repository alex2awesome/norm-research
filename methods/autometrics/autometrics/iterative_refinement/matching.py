from __future__ import annotations

from bisect import bisect_left
from typing import Iterable, List, Tuple

import pandas as pd


def normalize_pair_id(id_a: str, id_b: str) -> Tuple[str, str]:
    a = str(id_a)
    b = str(id_b)
    return (a, b) if a <= b else (b, a)


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

    neg_sorted = negatives[[id_column, prob_column]].sort_values(by=prob_column).reset_index(drop=True)
    neg_probs = neg_sorted[prob_column].tolist()

    candidates: List[Tuple[str, str, float]] = []
    for _, pos_row in positives.iterrows():
        pos_prob = float(pos_row[prob_column])
        pos_id = str(pos_row[id_column])
        idx = bisect_left(neg_probs, pos_prob)
        neighbor_indices = [idx - 1, idx]
        best = None
        best_diff = None
        for n_idx in neighbor_indices:
            if n_idx < 0 or n_idx >= len(neg_sorted):
                continue
            neg_id = str(neg_sorted.loc[n_idx, id_column])
            pair_key = normalize_pair_id(pos_id, neg_id)
            if pair_key in seen_pairs:
                continue
            diff = abs(pos_prob - float(neg_sorted.loc[n_idx, prob_column]))
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
