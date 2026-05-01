"""Approach B: STaR-based feature weighting from prediction correctness."""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from .config import LocalExplanationConfig
from .parsing import StarLocalResult

logger = logging.getLogger(__name__)


def compute_star_weights(
    results: Dict[str, StarLocalResult],
    true_labels: Dict[str, int],
    positive_label: str,
    config: LocalExplanationConfig,
) -> List[Tuple[str, float]]:
    """Compute weighted features across all examples based on prediction correctness.

    For each feature in each example, assigns a weight from the 2x2 matrix:

    |                      | Correct prediction | Incorrect prediction |
    |----------------------|--------------------|-----------------------|
    | Winning-side feature | +1.0               | -1.0                  |
    | Losing-side feature  | +0.3               | +0.5                  |

    "Winning side" = the side the model's prediction aligned with:
      - features_for if model predicted positive
      - features_against if model predicted negative

    Returns: flat list of (feature_text, weight) for all features across all examples.
    """
    w_cw = config.weight_correct_winning
    w_cl = config.weight_correct_losing
    w_iw = config.weight_incorrect_winning
    w_il = config.weight_incorrect_losing

    all_weighted: List[Tuple[str, float]] = []

    n_correct = 0
    n_incorrect = 0
    n_skipped = 0

    for doc_id, result in results.items():
        true_label = true_labels.get(doc_id)
        if true_label is None:
            n_skipped += 1
            continue

        # Determine if prediction was correct
        if result.prediction == "positive":
            predicted_positive = True
        elif result.prediction == "negative":
            predicted_positive = False
        else:
            # Couldn't parse prediction — skip
            n_skipped += 1
            continue

        true_positive = (true_label == 1)
        correct = (predicted_positive == true_positive)

        if correct:
            n_correct += 1
        else:
            n_incorrect += 1

        # Determine winning/losing sides
        if predicted_positive:
            winning_features = result.features_for
            losing_features = result.features_against
        else:
            winning_features = result.features_against
            losing_features = result.features_for

        # Assign weights
        if correct:
            for f in winning_features:
                all_weighted.append((f, w_cw))
            for f in losing_features:
                all_weighted.append((f, w_cl))
        else:
            for f in winning_features:
                all_weighted.append((f, w_iw))
            for f in losing_features:
                all_weighted.append((f, w_il))

    total = n_correct + n_incorrect
    if total > 0:
        logger.info(
            "STaR weights: %d correct (%.1f%%), %d incorrect, %d skipped",
            n_correct, 100 * n_correct / total, n_incorrect, n_skipped,
        )
    else:
        logger.warning("STaR weights: no valid predictions to score")

    return all_weighted


def aggregate_feature_scores(
    weighted_features: List[Tuple[str, float]],
) -> Dict[str, float]:
    """Aggregate per-occurrence weights into per-feature scores.

    Returns: dict mapping feature_text -> total aggregated weight.
    """
    scores = defaultdict(float)
    counts = defaultdict(int)
    for feat, weight in weighted_features:
        scores[feat] += weight
        counts[feat] += 1

    logger.info(
        "Aggregated %d occurrences into %d unique features",
        len(weighted_features), len(scores),
    )
    return dict(scores)
