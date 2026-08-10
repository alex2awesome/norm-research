#!/usr/bin/env python3
"""Score one frozen paired base/LoRA Gemma run on Humor fresh select truth.

The paired inference artifacts consumed here are deliberately truth blind.  This
module is the only part of the pipeline that reads the final 293-row consensus.
The primary comparison treats an invalid or order-unstable prediction as
incorrect; per-order results are also retained so that the conservative score
is fully decomposable.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .adjudicate_gemma import DECISIONS, ordered_candidates
from .common import read_jsonl, sha256_file, write_jsonl


SYSTEMS = ("base", "lora")
ORDERS = ("original", "hashed")
INVALID = "INVALID_OUTPUT"
DECISION_ORDER = (
    "MATCH",
    "MATCH_FAMILY_ONLY",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _index(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    values = list(rows)
    indexed = {str(row.get("norm_uid") or ""): row for row in values}
    if not values or "" in indexed or len(indexed) != len(values):
        raise ValueError(f"{label} has empty, missing, or duplicate norm_uid values")
    return indexed


def _prediction_key(prediction: Mapping[str, Any]) -> tuple[str, str | None]:
    return str(prediction.get("decision") or ""), prediction.get("metric_id")


def _is_valid_prediction(prediction: Mapping[str, Any]) -> bool:
    decision, metric_id = _prediction_key(prediction)
    if decision == INVALID:
        # INVALID_OUTPUT is the runner's well-formed representation of a
        # failed model parse, not a valid task prediction.
        return False
    if decision not in DECISIONS:
        return False
    return (decision == "MATCH" and isinstance(metric_id, str) and bool(metric_id)) or (
        decision != "MATCH" and metric_id is None
    )


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta using a stable continued fraction."""

    if not 0.0 <= x <= 1.0 or a <= 0.0 or b <= 0.0:
        raise ValueError("invalid incomplete-beta arguments")
    if x in (0.0, 1.0):
        return x

    def fraction(aa: float, bb: float, xx: float) -> float:
        qab = aa + bb
        qap = aa + 1.0
        qam = aa - 1.0
        c = 1.0
        d = 1.0 - qab * xx / qap
        floor = 1e-300
        if abs(d) < floor:
            d = floor
        d = 1.0 / d
        result = d
        for iteration in range(1, 401):
            twice = 2 * iteration
            term = iteration * (bb - iteration) * xx / (
                (qam + twice) * (aa + twice)
            )
            d = 1.0 + term * d
            if abs(d) < floor:
                d = floor
            c = 1.0 + term / c
            if abs(c) < floor:
                c = floor
            d = 1.0 / d
            result *= d * c
            term = -(aa + iteration) * (qab + iteration) * xx / (
                (aa + twice) * (qap + twice)
            )
            d = 1.0 + term * d
            if abs(d) < floor:
                d = floor
            c = 1.0 + term / c
            if abs(c) < floor:
                c = floor
            d = 1.0 / d
            delta = d * c
            result *= delta
            if abs(delta - 1.0) <= 3e-14:
                return result
        raise ArithmeticError("incomplete-beta continued fraction did not converge")

    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * fraction(a, b, x) / a
    return 1.0 - front * fraction(b, a, 1.0 - x) / b


def _beta_quantile(probability: float, a: float, b: float) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if _regularized_beta(middle, a, b) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> list[float]:
    """Two-sided exact Clopper-Pearson interval for a binomial proportion."""

    if trials < 0 or not 0 <= successes <= trials or not 0.0 < alpha < 1.0:
        raise ValueError("invalid exact-binomial interval inputs")
    if trials == 0:
        return [0.0, 1.0]
    lower = (
        0.0
        if successes == 0
        else _beta_quantile(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = (
        1.0
        if successes == trials
        else _beta_quantile(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return [lower, upper]


def paired_exact_change(
    before_correct: Sequence[bool],
    after_correct: Sequence[bool],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired gain with an exact interval conditional on discordant pairs.

    Given the observed number of discordant pairs, the LoRA-win share has a
    Clopper-Pearson interval.  Multiplying ``2*q-1`` by the observed discordant
    fraction produces the conditional exact interval for the paired net gain.
    """

    if len(before_correct) != len(after_correct) or not before_correct:
        raise ValueError("paired vectors must be nonempty and equal length")
    both = base_only = lora_only = neither = 0
    for before, after in zip(before_correct, after_correct):
        if before and after:
            both += 1
        elif before:
            base_only += 1
        elif after:
            lora_only += 1
        else:
            neither += 1
    count = len(before_correct)
    discordant = base_only + lora_only
    q_interval = clopper_pearson(lora_only, discordant, alpha)
    discordant_fraction = discordant / count
    interval = [
        discordant_fraction * (2.0 * bound - 1.0) for bound in q_interval
    ]
    if discordant:
        # Exact one-sided McNemar/binomial test of LoRA wins > base wins.
        p_value = sum(
            math.comb(discordant, value)
            for value in range(lora_only, discordant + 1)
        ) / (2**discordant)
    else:
        p_value = 1.0
    return {
        "n": count,
        "both_correct": both,
        "base_only_correct": base_only,
        "lora_only_correct": lora_only,
        "neither_correct": neither,
        "discordant_count": discordant,
        "base_accuracy": sum(before_correct) / count,
        "lora_accuracy": sum(after_correct) / count,
        "gain": (lora_only - base_only) / count,
        "conditional_exact_interval_level": 1.0 - alpha,
        "conditional_exact_interval_method": (
            "Clopper-Pearson interval for LoRA wins among observed discordant "
            "pairs, transformed to paired net gain"
        ),
        "conditional_exact_gain_interval": interval,
        "exact_one_sided_mcnemar_p_lora_better": p_value,
    }


def _mean_paired_change_across_orders(
    truth: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    uids: Sequence[str],
    *,
    alpha: float,
) -> dict[str, Any]:
    """Mean exact accuracy over frozen orders with an exact simultaneous CI.

    Each order gets a conditional Clopper-Pearson paired-gain interval at
    ``1-alpha/2``.  The Bonferroni intersection has coverage of at least
    ``1-alpha``; averaging its endpoints gives a conservative simultaneous
    interval for the predeclared mean-over-orders gain without treating two
    predictions of one norm as independent observations.
    """

    if not uids:
        raise ValueError("mean paired comparison requires at least one UID")
    per_order: dict[str, dict[str, Any]] = {}
    for order in ORDERS:
        per_order[order] = paired_exact_change(
            [
                _is_valid_prediction(predictions["base"][order][uid])
                and _prediction_key(predictions["base"][order][uid])
                == _prediction_key(truth[uid])
                for uid in uids
            ],
            [
                _is_valid_prediction(predictions["lora"][order][uid])
                and _prediction_key(predictions["lora"][order][uid])
                == _prediction_key(truth[uid])
                for uid in uids
            ],
            alpha=alpha / len(ORDERS),
        )
    return {
        "n_norms": len(uids),
        "orders": list(ORDERS),
        "primary_estimand": "mean exact decision-and-leaf accuracy across two frozen orders",
        "base_mean_accuracy": sum(
            per_order[order]["base_accuracy"] for order in ORDERS
        )
        / len(ORDERS),
        "lora_mean_accuracy": sum(
            per_order[order]["lora_accuracy"] for order in ORDERS
        )
        / len(ORDERS),
        "gain": sum(per_order[order]["gain"] for order in ORDERS) / len(ORDERS),
        "simultaneous_conditional_exact_interval_level_at_least": 1.0 - alpha,
        "simultaneous_conditional_exact_gain_interval": [
            sum(
                per_order[order]["conditional_exact_gain_interval"][endpoint]
                for order in ORDERS
            )
            / len(ORDERS)
            for endpoint in (0, 1)
        ],
        "interval_method": (
            "mean endpoints of two Bonferroni-adjusted conditional exact "
            "Clopper-Pearson paired intervals; does not assume order-level independence"
        ),
        "per_order_paired_exact": per_order,
    }


def _classification_metrics(
    truth: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    candidate_present: set[str],
) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = {
        decision: Counter() for decision in DECISION_ORDER
    }
    exact = invalid = predicted_match = exact_leaf = binary_match_tp = 0
    truth_match = sum(row["decision"] == "MATCH" for row in truth.values())
    conditional_exact = 0
    for uid, gold in truth.items():
        prediction = predictions[uid]
        valid = _is_valid_prediction(prediction)
        predicted_decision = str(prediction.get("decision") or "") if valid else INVALID
        confusion[str(gold["decision"])][predicted_decision] += 1
        correct = valid and _prediction_key(prediction) == _prediction_key(gold)
        exact += correct
        invalid += not valid
        if predicted_decision == "MATCH":
            predicted_match += 1
            binary_match_tp += gold["decision"] == "MATCH"
            exact_leaf += correct
        if uid in candidate_present:
            conditional_exact += correct

    per_decision: dict[str, dict[str, Any]] = {}
    for decision in DECISION_ORDER:
        tp = confusion[decision][decision]
        support = sum(confusion[decision].values())
        predicted = sum(row[decision] for row in confusion.values())
        precision = _rate(tp, predicted)
        recall = _rate(tp, support)
        f1 = (
            None
            if precision is None or recall is None or precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        # A class with no truth or prediction support is explicitly zero in
        # the fixed-label macro average rather than silently disappearing.
        per_decision[decision] = {
            "support": support,
            "predicted": predicted,
            "true_positive": tp,
            "precision": precision,
            "recall": recall,
            "f1": 0.0 if f1 is None else f1,
        }
    return {
        "n": len(truth),
        "exact_decision_and_leaf_correct": exact,
        "exact_decision_and_leaf_accuracy": exact / len(truth),
        "invalid_count": invalid,
        "invalid_rate": invalid / len(truth),
        "typed_decision_macro_f1_all_7": sum(
            per_decision[value]["f1"] for value in DECISION_ORDER
        )
        / len(DECISION_ORDER),
        "typed_decision_macro_f1_non_noise_6": sum(
            per_decision[value]["f1"]
            for value in DECISION_ORDER
            if value != "NOISE"
        )
        / (len(DECISION_ORDER) - 1),
        "per_typed_decision": per_decision,
        "typed_decision_confusion": {
            gold: {
                predicted: counts.get(predicted, 0)
                for predicted in (*DECISION_ORDER, INVALID)
            }
            for gold, counts in confusion.items()
        },
        "match": {
            "truth_count": truth_match,
            "predicted_count": predicted_match,
            "binary_decision_true_positive": binary_match_tp,
            "binary_decision_precision": _rate(binary_match_tp, predicted_match),
            "binary_decision_recall": _rate(binary_match_tp, truth_match),
            "exact_leaf_correct": exact_leaf,
            "exact_leaf_precision_among_predicted_match": _rate(
                exact_leaf, predicted_match
            ),
            "exact_leaf_recall_of_gold_match": _rate(exact_leaf, truth_match),
        },
        "gold_match_candidate_present": {
            "support": len(candidate_present),
            "exact_leaf_correct": conditional_exact,
            "conditional_exact_leaf_accuracy": _rate(
                conditional_exact, len(candidate_present)
            ),
        },
    }


def _validate_and_unpack_predictions(
    truth: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    rows_by_order: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[
    dict[str, dict[str, dict[str, dict[str, Any]]]], set[str], int, str, str
]:
    truth_uids = set(truth)
    prediction_uids = set(rows_by_order[ORDERS[0]])
    if any(set(rows_by_order[order]) != prediction_uids for order in ORDERS):
        raise ValueError("paired order outputs have different inference universes")
    if not truth_uids <= prediction_uids or prediction_uids != set(candidates):
        raise ValueError(
            "predictions must exactly cover candidates and contain all resolved truth rows"
        )

    unpacked = {
        system: {order: {} for order in ORDERS} for system in SYSTEMS
    }
    candidate_present: set[str] = set()
    max_candidates: int | None = None
    prompt_hashes: set[str] = set()
    bank_hashes: set[str] = set()
    for uid in sorted(prediction_uids):
        source = candidates[uid]
        source_cards = list(source.get("candidates") or [])
        if not source_cards:
            raise ValueError(f"candidate source has an empty slate: {uid}")
        observed_sets: list[set[str]] = []
        for order in ORDERS:
            row = rows_by_order[order][uid]
            ids = [str(value) for value in row.get("candidate_ids") or []]
            if max_candidates is None:
                max_candidates = len(ids)
            if (
                row.get("order_mode") != order
                or len(ids) != max_candidates
                or len(ids) != len(set(ids))
            ):
                raise ValueError(f"invalid paired order/candidate contract: {order}:{uid}")
            expected = [
                str(value["metric_id"])
                for value in ordered_candidates(
                    source_cards[:max_candidates], order, uid
                )
            ]
            if ids != expected:
                raise ValueError(f"paired inference slate drift: {order}:{uid}")
            observed_sets.append(set(ids))
            prompt_hashes.add(str(row.get("prompt_sha256") or ""))
            bank_hashes.add(str(row.get("candidate_bank_source_sha256") or ""))
            for system in SYSTEMS:
                prediction = row.get(system)
                if not isinstance(prediction, dict):
                    raise ValueError(f"missing {system} prediction: {order}:{uid}")
                unpacked[system][order][uid] = prediction
            # The prompt and candidate sequence must be byte-identical across
            # the base and LoRA arms; one shared row is the structural proof.
            if row.get("base_item_prompt_sha256") != row.get(
                "lora_item_prompt_sha256"
            ):
                raise ValueError(f"base/LoRA prompt mismatch: {order}:{uid}")
        if observed_sets[0] != observed_sets[1]:
            raise ValueError(f"candidate sets differ across orders: {uid}")
        if uid in truth and truth[uid]["decision"] == "MATCH":
            if str(truth[uid].get("metric_id") or "") in observed_sets[0]:
                candidate_present.add(uid)
    if max_candidates is None or max_candidates < 1:
        raise ValueError("empty paired inference")
    if len(prompt_hashes) != 1 or "" in prompt_hashes:
        raise ValueError("paired inference does not bind one frozen prompt")
    if len(bank_hashes) != 1 or "" in bank_hashes:
        raise ValueError("paired inference does not bind one bank")
    return unpacked, candidate_present, max_candidates, next(iter(prompt_hashes)), next(
        iter(bank_hashes)
    )


def score_rows(
    truth_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    original_rows: Sequence[dict[str, Any]],
    hashed_rows: Sequence[dict[str, Any]],
    *,
    minimum_exact_gain: float = 0.03,
    minimum_stability: float = 0.90,
    maximum_invalid_rate: float = 0.01,
    alpha: float = 0.05,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth = _index(truth_rows, "truth")
    candidates = _index(candidate_rows, "candidates")
    rows_by_order = {
        "original": _index(original_rows, "original paired predictions"),
        "hashed": _index(hashed_rows, "hashed paired predictions"),
    }
    if any(
        row.get("task") != "humor"
        or row.get("gepa_role") != "select"
        or row.get("evaluation_only") is not True
        or row.get("training_eligible") is not False
        for row in truth.values()
    ):
        raise ValueError("truth is not wholly Humor select-only evaluation evidence")
    if set(str(row["decision"]) for row in truth.values()) - DECISIONS:
        raise ValueError("truth contains an unknown typed decision")
    for uid, row in truth.items():
        if (row["decision"] == "MATCH") != bool(row.get("metric_id")):
            raise ValueError(f"truth decision/leaf contract is invalid: {uid}")

    unpacked, candidate_present, candidate_k, prompt_hash, bank_hash = (
        _validate_and_unpack_predictions(truth, candidates, rows_by_order)
    )
    truth_match_uids = {
        uid for uid, row in truth.items() if row["decision"] == "MATCH"
    }
    system_reports: dict[str, Any] = {}
    strict_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    stability_vectors: dict[str, dict[str, bool]] = {}
    for system in SYSTEMS:
        order_reports = {
            order: _classification_metrics(
                truth, unpacked[system][order], candidate_present
            )
            for order in ORDERS
        }
        strict: dict[str, dict[str, Any]] = {}
        stable: dict[str, bool] = {}
        for uid in truth:
            left = unpacked[system]["original"][uid]
            right = unpacked[system]["hashed"][uid]
            is_stable = (
                _is_valid_prediction(left)
                and _is_valid_prediction(right)
                and _prediction_key(left) == _prediction_key(right)
            )
            stable[uid] = is_stable
            strict[uid] = (
                dict(left)
                if is_stable
                else {"decision": INVALID, "metric_id": None}
            )
        strict_metrics = _classification_metrics(truth, strict, candidate_present)
        system_reports[system] = {
            "orders": order_reports,
            "two_order": {
                "valid_exact_output_stable_count": sum(stable.values()),
                "valid_exact_output_stability": sum(stable.values()) / len(truth),
                "order_disagreement_or_invalid_count": len(truth) - sum(stable.values()),
                "strict_stable_prediction_metrics": strict_metrics,
            },
        }
        strict_predictions[system] = strict
        stability_vectors[system] = stable

    paired_primary = _mean_paired_change_across_orders(
        truth, unpacked, list(truth), alpha=alpha
    )
    match_uids = [uid for uid in truth if uid in truth_match_uids]
    paired_match_recall = _mean_paired_change_across_orders(
        truth, unpacked, match_uids, alpha=alpha
    )

    def pooled_exact_match_precision(system: str) -> float | None:
        exact = sum(
            system_reports[system]["orders"][order]["match"]["exact_leaf_correct"]
            for order in ORDERS
        )
        predicted = sum(
            system_reports[system]["orders"][order]["match"]["predicted_count"]
            for order in ORDERS
        )
        return _rate(exact, predicted)

    base_precision = pooled_exact_match_precision("base")
    lora_precision = pooled_exact_match_precision("lora")

    def nondecrease(after: float | None, before: float | None) -> bool:
        if after is None:
            return before is None
        return before is None or after >= before

    base_stability = system_reports["base"]["two_order"][
        "valid_exact_output_stability"
    ]
    lora_stability = system_reports["lora"]["two_order"][
        "valid_exact_output_stability"
    ]
    lora_max_invalid = max(
        system_reports["lora"]["orders"][order]["invalid_rate"]
        for order in ORDERS
    )
    checks = {
        "primary_exact_decision_and_leaf_gain_at_least_minimum": paired_primary[
            "gain"
        ]
        >= minimum_exact_gain,
        "exact_match_precision_non_decrease": nondecrease(
            lora_precision, base_precision
        ),
        "two_order_stability_non_decrease": lora_stability >= base_stability,
        "two_order_stability_at_least_minimum": lora_stability
        >= minimum_stability,
        "maximum_per_order_invalid_rate_at_most_limit": lora_max_invalid
        <= maximum_invalid_rate,
    }

    audit_rows: list[dict[str, Any]] = []
    for uid, gold in truth.items():
        values: dict[str, Any] = {}
        for system in SYSTEMS:
            values[system] = {
                order: {
                    "decision": unpacked[system][order][uid].get("decision"),
                    "metric_id": unpacked[system][order][uid].get("metric_id"),
                    "valid": _is_valid_prediction(unpacked[system][order][uid]),
                }
                for order in ORDERS
            }
            values[system]["two_order_stable"] = stability_vectors[system][uid]
            values[system]["exact_correct_order_count"] = sum(
                _is_valid_prediction(unpacked[system][order][uid])
                and _prediction_key(unpacked[system][order][uid])
                == _prediction_key(gold)
                for order in ORDERS
            )
        audit_rows.append(
            {
                "schema_version": "silver-match-v3-humor-gemma4-lora-select-row-audit-v1",
                "norm_uid": uid,
                "corpus": gold.get("corpus"),
                "truth": {
                    "decision": gold["decision"],
                    "metric_id": gold.get("metric_id"),
                },
                "gold_metric_candidate_present": uid in candidate_present,
                **values,
                "paired_exact_order_count_transition": (
                    f"{values['base']['exact_correct_order_count']}->"
                    f"{values['lora']['exact_correct_order_count']}"
                ),
            }
        )

    decision_counts = Counter(str(row["decision"]) for row in truth.values())
    report = {
        "schema_version": "silver-match-v3-humor-gemma4-lora-fresh-select-score-v1",
        "status": "PROMOTABLE" if all(checks.values()) else "QUARANTINE_ADAPTER",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "humor",
        "role": "fresh_select_model_evaluation_only",
        "truth": {
            "resolved_count": len(truth),
            "decision_counts": dict(sorted(decision_counts.items())),
            "gold_match_count": len(truth_match_uids),
        },
        "candidate_retrieval": {
            "candidate_k": candidate_k,
            "gold_match_candidate_present_count": len(candidate_present),
            "gold_match_candidate_absent_count": len(truth_match_uids)
            - len(candidate_present),
            "candidate_recall_of_gold_match": len(candidate_present)
            / len(truth_match_uids),
        },
        "systems": system_reports,
        "paired_primary_exact_decision_and_leaf": paired_primary,
        "paired_exact_match_recall": paired_match_recall,
        "promotion_gate": {
            "primary_unit": (
                "mean exact decision-and-leaf accuracy across the two pre-frozen "
                "orders on all resolved norms; each order is reported separately"
            ),
            "observed_pooled_exact_match_precision": {
                "base": base_precision,
                "lora": lora_precision,
            },
            "thresholds": {
                "minimum_exact_decision_and_leaf_accuracy_gain": minimum_exact_gain,
                "exact_match_precision_may_decrease": False,
                "minimum_two_order_exact_output_stability": minimum_stability,
                "two_order_exact_output_stability_may_decrease": False,
                "maximum_invalid_output_rate_each_order": maximum_invalid_rate,
            },
            "checks": checks,
            "passed": all(checks.values()),
        },
        "inference_contract": {
            "truth_read_by_inference": False,
            "base_and_lora_share_each_rendered_prompt_and_candidate_order": True,
            "prompt_sha256": prompt_hash,
            "candidate_bank_source_sha256": bank_hash,
            "no_hyperparameter_or_seed_search": True,
        },
    }
    return report, audit_rows


def _validate_final_truth_release(
    truth_path: Path,
    report_path: Path,
    unresolved_path: Path,
) -> None:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    truth = list(read_jsonl(truth_path))
    unresolved = list(read_jsonl(unresolved_path))
    truth_uids = {str(row.get("norm_uid") or "") for row in truth}
    unresolved_uids = {str(row.get("norm_uid") or "") for row in unresolved}
    if (
        payload.get("schema_version")
        != "silver-match-v3-exact-multi-pass-truth-report-v1"
        or payload.get("task") != "humor"
        or payload.get("gepa_role") != "select"
        or payload.get("complete") is not False
        or int(payload.get("resolved_count", -1)) != 293
        or int(payload.get("unresolved_count", -1)) != 7
        or int(payload.get("source_count", -1)) != 300
        or len(truth) != 293
        or len(unresolved) != 7
        or truth_uids & unresolved_uids
        or ((payload.get("outputs") or {}).get("resolved") or {}).get("sha256")
        != sha256_file(truth_path)
        or ((payload.get("outputs") or {}).get("unresolved") or {}).get("sha256")
        != sha256_file(unresolved_path)
    ):
        raise ValueError("inputs are not the final 293+7 Humor select truth release")


def _validate_truth_blind_inference(
    freeze_path: Path,
    meta_path: Path,
    original_path: Path,
    hashed_path: Path,
    candidates_path: Path,
) -> None:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    firewall = freeze.get("truth_firewall") or {}
    frozen_candidates = (freeze.get("inputs") or {}).get("candidates") or {}
    if (
        freeze.get("schema_version")
        != "silver-match-v3-paired-gemma4-lora-inference-freeze-v1"
        or freeze.get("status") != "FROZEN_BEFORE_PAIRED_MODEL_INFERENCE"
        or freeze.get("task") != "humor"
        or freeze.get("backend") != "direct_batch_vllm_not_openai_server"
        or firewall.get("truth_read") is not False
        or firewall.get("truth_path_argument_exists") is not False
        or firewall.get("resolved_or_unresolved_label_artifacts_read") is not False
        or firewall.get("scoring_in_separate_process_after_predictions") is not True
        or frozen_candidates.get("sha256") != sha256_file(candidates_path)
    ):
        raise ValueError("inference freeze does not prove a truth-blind paired run")
    outputs = meta.get("outputs") or {}
    if (
        meta.get("schema_version")
        != "silver-match-v3-paired-gemma4-lora-inference-meta-v1"
        or meta.get("status") != "COMPLETE_TRUTH_BLIND_PAIRED_INFERENCE"
        or meta.get("task") != "humor"
        or meta.get("truth_read") is not False
        or meta.get("backend") != "direct_batch_vllm_not_openai_server"
        or meta.get("same_loaded_base_model_instance_for_both_arms") is not True
        or (meta.get("inference_freeze") or {}).get("sha256")
        != sha256_file(freeze_path)
        or (outputs.get("original") or {}).get("sha256")
        != sha256_file(original_path)
        or (outputs.get("hashed") or {}).get("sha256") != sha256_file(hashed_path)
    ):
        raise ValueError("paired inference metadata is incomplete or hash-drifted")
    expected_freeze_hash = sha256_file(freeze_path)
    for order, path in (("original", original_path), ("hashed", hashed_path)):
        rows = list(read_jsonl(path))
        if (
            int((outputs.get(order) or {}).get("count", -1)) != len(rows)
            or any(
                row.get("order_mode") != order
                or row.get("inference_freeze_sha256") != expected_freeze_hash
                for row in rows
            )
        ):
            raise ValueError(f"{order} rows do not bind the truth-blind inference freeze")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--truth-consensus-report", required=True)
    parser.add_argument("--unresolved-exclusions", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--paired-original", required=True)
    parser.add_argument("--paired-hashed", required=True)
    parser.add_argument("--inference-freeze", required=True)
    parser.add_argument("--inference-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--row-audit-output", required=True)
    parser.add_argument("--minimum-exact-gain", type=float, default=0.03)
    parser.add_argument("--minimum-stability", type=float, default=0.90)
    parser.add_argument("--maximum-invalid-rate", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    paths = {
        "truth": Path(args.truth).resolve(),
        "truth_consensus_report": Path(args.truth_consensus_report).resolve(),
        "unresolved_exclusions": Path(args.unresolved_exclusions).resolve(),
        "candidates": Path(args.candidates).resolve(),
        "paired_original": Path(args.paired_original).resolve(),
        "paired_hashed": Path(args.paired_hashed).resolve(),
        "inference_freeze": Path(args.inference_freeze).resolve(),
        "inference_meta": Path(args.inference_meta).resolve(),
    }
    output = Path(args.output).resolve()
    audit_output = Path(args.row_audit_output).resolve()
    if output.exists() or audit_output.exists():
        raise FileExistsError("refusing to overwrite fresh-select score artifacts")
    if not 0.0 <= args.minimum_exact_gain <= 1.0:
        parser.error("--minimum-exact-gain must be in [0, 1]")
    if not 0.0 <= args.minimum_stability <= 1.0:
        parser.error("--minimum-stability must be in [0, 1]")
    if not 0.0 <= args.maximum_invalid_rate <= 1.0:
        parser.error("--maximum-invalid-rate must be in [0, 1]")
    _validate_final_truth_release(
        paths["truth"], paths["truth_consensus_report"], paths["unresolved_exclusions"]
    )
    _validate_truth_blind_inference(
        paths["inference_freeze"],
        paths["inference_meta"],
        paths["paired_original"],
        paths["paired_hashed"],
        paths["candidates"],
    )
    report, audit_rows = score_rows(
        list(read_jsonl(paths["truth"])),
        list(read_jsonl(paths["candidates"])),
        list(read_jsonl(paths["paired_original"])),
        list(read_jsonl(paths["paired_hashed"])),
        minimum_exact_gain=args.minimum_exact_gain,
        minimum_stability=args.minimum_stability,
        maximum_invalid_rate=args.maximum_invalid_rate,
        alpha=args.alpha,
    )
    write_jsonl(audit_output, audit_rows)
    report["row_audit"] = {**_artifact(audit_output), "count": len(audit_rows)}
    report["inputs"] = {
        name: _artifact(path) for name, path in sorted(paths.items())
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
