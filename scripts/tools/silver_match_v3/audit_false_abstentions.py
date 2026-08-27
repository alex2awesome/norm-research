#!/usr/bin/env python3
"""Estimate false-abstention risk with an exact one-sided confidence bound.

The key claim is conditional: among independently labeled rows on which the
system abstained, how often did a human establish that an exact current-bank
metric existed?  A point estimate alone is insufficient; ``claim_supported``
is true only when the exact 95% binomial upper bound is below the requested
threshold (5% by default).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import normalize_space, read_jsonl, sha256_file


def binomial_cdf(k: int, n: int, p: float) -> float:
    if not 0 <= k <= n or not 0.0 <= p <= 1.0:
        raise ValueError("invalid binomial arguments")
    if k == n or p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    logs = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * math.log(p)
        + (n - i) * math.log1p(-p)
        for i in range(k + 1)
    ]
    largest = max(logs)
    return math.exp(largest) * sum(math.exp(value - largest) for value in logs)


def clopper_pearson_upper(k: int, n: int, *, alpha: float = 0.05) -> float | None:
    """Return the exact one-sided ``1-alpha`` upper binomial bound."""
    if n == 0:
        return None
    if not 0 <= k <= n or not 0.0 < alpha < 1.0:
        raise ValueError("invalid count or alpha")
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    low, high = k / n, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        # P(X <= k | p) decreases as p increases.  The exact upper bound
        # solves this CDF == alpha.
        if binomial_cdf(k, n, middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def clopper_pearson_lower(k: int, n: int, *, alpha: float = 0.05) -> float | None:
    """Return the exact one-sided ``1-alpha`` lower binomial bound."""
    if n == 0:
        return None
    if not 0 <= k <= n or not 0.0 < alpha < 1.0:
        raise ValueError("invalid count or alpha")
    upper_failures = clopper_pearson_upper(n - k, n, alpha=alpha)
    assert upper_failures is not None
    return 1.0 - upper_failures


def _unique(paths: Iterable[Path], kind: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            uid = normalize_space(row.get("norm_uid"))
            if not uid:
                raise ValueError(f"{kind} row missing norm_uid in {path}")
            if uid in output:
                raise ValueError(f"duplicate {kind} norm_uid: {uid}")
            output[uid] = row
    return output


def _risk_summary(
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    alpha: float,
    target: float,
    precision_target: float,
) -> dict[str, Any]:
    predicted_abstentions = [
        pair for pair in pairs if pair[1].get("decision") != "MATCH"
    ]
    false_abstentions = [
        pair for pair in predicted_abstentions if pair[0].get("decision") == "MATCH"
    ]
    typed_abstention_exact = [
        pair
        for pair in predicted_abstentions
        if pair[0].get("decision") != "MATCH"
        and pair[0].get("decision") == pair[1].get("decision")
    ]
    gold_matches = [pair for pair in pairs if pair[0].get("decision") == "MATCH"]
    gold_match_abstentions = [
        pair for pair in gold_matches if pair[1].get("decision") != "MATCH"
    ]
    exact_correct = [
        pair
        for pair in gold_matches
        if pair[1].get("decision") == "MATCH"
        and str(pair[1].get("metric_id")) == str(pair[0].get("metric_id"))
    ]
    wrong_match = [
        pair
        for pair in gold_matches
        if pair[1].get("decision") == "MATCH"
        and str(pair[1].get("metric_id")) != str(pair[0].get("metric_id"))
    ]
    predicted_matches = [pair for pair in pairs if pair[1].get("decision") == "MATCH"]
    predicted_exact_correct = [
        pair
        for pair in predicted_matches
        if pair[0].get("decision") == "MATCH"
        and str(pair[1].get("metric_id")) == str(pair[0].get("metric_id"))
    ]
    predicted_wrong_leaf = [
        pair
        for pair in predicted_matches
        if pair[0].get("decision") == "MATCH"
        and str(pair[1].get("metric_id")) != str(pair[0].get("metric_id"))
    ]
    predicted_false_positive = [
        pair for pair in predicted_matches if pair[0].get("decision") != "MATCH"
    ]
    predicted_match_n = len(predicted_matches)
    predicted_match_k = len(predicted_exact_correct)
    precision_lower = clopper_pearson_lower(
        predicted_match_k, predicted_match_n, alpha=alpha
    )
    conditional_n = len(predicted_abstentions)
    conditional_k = len(false_abstentions)
    upper = clopper_pearson_upper(conditional_k, conditional_n, alpha=alpha)
    typed_abstention_k = len(typed_abstention_exact)
    typed_abstention_lower = clopper_pearson_lower(
        typed_abstention_k, conditional_n, alpha=alpha
    )
    confusion = Counter(
        f"{gold.get('decision')}->{prediction.get('decision')}"
        for gold, prediction in pairs
    )
    return {
        "audited_rows": len(pairs),
        "predicted_abstentions": conditional_n,
        "false_abstentions": conditional_k,
        "false_abstention_probability": (
            conditional_k / conditional_n if conditional_n else None
        ),
        "false_abstention_upper_bound": upper,
        "typed_abstention_exact_correct": typed_abstention_k,
        "typed_abstention_exact_accuracy": (
            typed_abstention_k / conditional_n if conditional_n else None
        ),
        "typed_abstention_exact_accuracy_lower_bound": typed_abstention_lower,
        "confidence_level_one_sided": 1.0 - alpha,
        "target_upper_bound": target,
        "claim_supported": upper is not None and upper < target,
        "gold_exact_matches": len(gold_matches),
        "gold_match_abstentions": len(gold_match_abstentions),
        "gold_match_abstention_rate": (
            len(gold_match_abstentions) / len(gold_matches) if gold_matches else None
        ),
        "gold_match_exact_correct": len(exact_correct),
        "gold_match_exact_accuracy": (
            len(exact_correct) / len(gold_matches) if gold_matches else None
        ),
        "gold_match_wrong_metric": len(wrong_match),
        "predicted_matches": predicted_match_n,
        "predicted_match_exact_correct": predicted_match_k,
        "predicted_match_exact_precision": (
            predicted_match_k / predicted_match_n if predicted_match_n else None
        ),
        "predicted_match_wrong_leaf": len(predicted_wrong_leaf),
        "predicted_match_false_positive": len(predicted_false_positive),
        "predicted_match_wrong_leaf_rate": (
            len(predicted_wrong_leaf) / predicted_match_n if predicted_match_n else None
        ),
        "predicted_match_false_positive_rate": (
            len(predicted_false_positive) / predicted_match_n
            if predicted_match_n
            else None
        ),
        "predicted_match_exact_precision_lower_bound": precision_lower,
        "predicted_match_precision_target": precision_target,
        "predicted_match_precision_claim_supported": (
            precision_lower is not None and precision_lower > precision_target
        ),
        "confusion_counts": dict(sorted(confusion.items())),
    }


def audit_false_abstentions(
    gold_paths: list[Path],
    prediction_paths: list[Path],
    *,
    alpha: float = 0.05,
    target: float = 0.05,
    precision_target: float = 0.90,
    require_complete: bool = True,
    analysis_exclusion_paths: list[Path] | None = None,
) -> dict[str, Any]:
    gold = _unique(gold_paths, "gold")
    predictions = _unique(prediction_paths, "prediction")
    analysis_exclusion_paths = analysis_exclusion_paths or []
    exclusions = (
        _unique(analysis_exclusion_paths, "analysis exclusion")
        if analysis_exclusion_paths
        else {}
    )
    overlap = set(gold) & set(exclusions)
    if overlap:
        raise ValueError(
            f"blind audit gold overlaps analysis exclusions: {sorted(overlap)[:3]}"
        )
    missing = set(gold) - set(predictions)
    if require_complete and missing:
        raise ValueError(
            f"predictions miss {len(missing)} gold UIDs; first={sorted(missing)[:3]}"
        )
    pairs = [(row, predictions[uid]) for uid, row in gold.items() if uid in predictions]
    if not pairs:
        raise ValueError("no joined audit rows")
    for gold_row, prediction in pairs:
        if normalize_space(gold_row.get("task")) != normalize_space(
            prediction.get("task")
        ):
            raise ValueError(f"task mismatch for {gold_row.get('norm_uid')}")
        if normalize_space(gold_row.get("corpus")) != normalize_space(
            prediction.get("corpus")
        ):
            raise ValueError(f"corpus mismatch for {gold_row.get('norm_uid')}")
    by_task: dict[str, list] = defaultdict(list)
    by_corpus: dict[str, list] = defaultdict(list)
    for pair in pairs:
        by_task[normalize_space(pair[0].get("task"))].append(pair)
        by_corpus[normalize_space(pair[0].get("corpus"))].append(pair)
    return {
        "schema_version": "silver-match-v3-false-abstention-audit-v1",
        "gold_inputs": {str(path): sha256_file(path) for path in gold_paths},
        "prediction_inputs": {
            str(path): sha256_file(path) for path in prediction_paths
        },
        "analysis_exclusions": {
            "inputs": {
                str(path): sha256_file(path) for path in analysis_exclusion_paths
            },
            "count": len(exclusions),
        },
        "gold_rows": len(gold),
        "joined_rows": len(pairs),
        "missing_prediction_rows": len(missing),
        "overall": _risk_summary(
            pairs,
            alpha=alpha,
            target=target,
            precision_target=precision_target,
        ),
        "by_task": {
            task: _risk_summary(
                rows,
                alpha=alpha,
                target=target,
                precision_target=precision_target,
            )
            for task, rows in sorted(by_task.items())
        },
        "by_corpus": {
            corpus: _risk_summary(
                rows,
                alpha=alpha,
                target=target,
                precision_target=precision_target,
            )
            for corpus, rows in sorted(by_corpus.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", action="append", required=True)
    parser.add_argument("--predictions", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target", type=float, default=0.05)
    parser.add_argument("--precision-target", type=float, default=0.90)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--analysis-exclusion", action="append", default=[])
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit_false_abstentions(
        [Path(path).resolve() for path in args.gold],
        [Path(path).resolve() for path in args.predictions],
        alpha=args.alpha,
        target=args.target,
        precision_target=args.precision_target,
        require_complete=not args.allow_incomplete,
        analysis_exclusion_paths=[
            Path(path).resolve() for path in args.analysis_exclusion
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "joined_rows": report["joined_rows"],
                "false_abstention_probability": report["overall"][
                    "false_abstention_probability"
                ],
                "false_abstention_upper_bound": report["overall"][
                    "false_abstention_upper_bound"
                ],
                "claim_supported": report["overall"]["claim_supported"],
                "predicted_match_exact_precision": report["overall"][
                    "predicted_match_exact_precision"
                ],
                "predicted_match_exact_precision_lower_bound": report["overall"][
                    "predicted_match_exact_precision_lower_bound"
                ],
                "predicted_match_precision_claim_supported": report["overall"][
                    "predicted_match_precision_claim_supported"
                ],
                "output": str(output),
                "output_sha256": sha256_file(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
