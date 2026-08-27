#!/usr/bin/env python3
"""Train the append-only metric-balanced silver-match v4 cross-encoder.

This module deliberately reuses the frozen v3 loader, model, and calibration
code while replacing the pair sampler that caused metric-card prior collapse.
It caps over-represented positive metrics, mixes lexical hard negatives with
globally balanced negatives, and fails before model initialization unless
every bank metric satisfies frozen positive/negative exposure gates.

Development evaluation remains grouped by norm and ranks the complete bank.
The v4 development panel is adaptive (the aggregate v3 failure was observed);
the blind production audit remains sealed and is never consumed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import train_cross_encoder as base
from .common import metric_card, sha256_file


BALANCED_OBJECTIVE_REVISION = "cross-encoder-metric-balanced-v4"
BALANCED_SCHEMA = "silver-match-v4-metric-balanced-pair-sampling-v1"

_ORIGINAL_VALIDATE = base.validate_frozen_policy
_ORIGINAL_GATE_REPORT = base.gate_report
_CONFIG: dict[str, Any] | None = None
_PAIR_AUDIT: dict[str, Any] | None = None
_EVIDENCE_STATUS: dict[str, str] | None = None
_AUDIT_ONLY = False


class PairAuditComplete(RuntimeError):
    """Internal successful stop used by --audit-pairs-only."""


def _stable_int(seed: int, *values: str) -> int:
    payload = "\x1f".join((str(seed), *values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _load_balanced_policy(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    global _CONFIG, _EVIDENCE_STATUS
    binding = _ORIGINAL_VALIDATE(args)
    if binding is None:
        raise ValueError("balanced CE requires a frozen policy")
    policy_path = Path(args.policy).resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("balanced_objective_revision") != BALANCED_OBJECTIVE_REVISION:
        raise ValueError("not the frozen metric-balanced v4 policy")
    if args.task not in (policy.get("scope") or []):
        raise ValueError("task is outside the balanced policy scope")
    config = policy.get("balanced_training") or {}
    if config.get("schema_version") != BALANCED_SCHEMA:
        raise ValueError("unsupported balanced-training contract")

    implementation = policy.get("implementation") or {}
    expected_path = "scripts/tools/silver_match_v3/train_cross_encoder_balanced.py"
    if implementation.get("balanced_train_cross_encoder_path") != expected_path:
        raise ValueError("balanced trainer path differs from policy")
    if sha256_file(Path(__file__).resolve()) != implementation.get(
        "balanced_train_cross_encoder_sha256"
    ):
        raise ValueError("balanced trainer hash differs from policy")
    base_path = Path(__file__).resolve().with_name("train_cross_encoder.py")
    if sha256_file(base_path) != implementation.get(
        "base_train_cross_encoder_sha256"
    ):
        raise ValueError("underlying v3 trainer hash differs from policy")

    if int(args.negatives_per_positive) != (
        int(config["hard_negatives_per_match"])
        + int(config["global_balanced_negatives_per_match"])
    ):
        raise ValueError("match negative count differs from balanced policy")
    if int(args.negatives_per_abstain) != (
        int(config["hard_negatives_per_abstain"])
        + int(config["global_balanced_negatives_per_abstain"])
    ):
        raise ValueError("abstention negative count differs from balanced policy")
    if policy.get("role_contract", {}).get("blind") != (
        "sealed uniform final-production match and false-abstention audits only"
    ):
        raise ValueError("blind role is not sealed by the balanced policy")
    development_status = str(policy.get("development_evidence_status") or "")
    blind_status = str(policy.get("blind_status") or "")
    if not development_status:
        raise ValueError("balanced policy omits development evidence status")
    if blind_status != "SEALED_UNCONSUMED":
        raise ValueError("balanced policy has consumed the blind role")
    _CONFIG = config
    _EVIDENCE_STATUS = {
        "development": development_status,
        "blind": blind_status,
    }
    return policy, {
        **binding,
        "balanced_training": config,
        "balanced_trainer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "development_independence": development_status,
        "blind_status": blind_status,
    }


def _ordered_candidates(
    label: base.CELabel,
    lexical_row: np.ndarray,
    bank_ids: list[str],
    bank_index: dict[str, int],
    candidate_ids: dict[str, list[str]],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for metric_id in candidate_ids.get(label.norm_uid, []):
        if metric_id in bank_index and metric_id not in seen:
            ordered.append(metric_id)
            seen.add(metric_id)
    for index in lexical_row:
        metric_id = bank_ids[int(index)]
        if metric_id not in seen:
            ordered.append(metric_id)
            seen.add(metric_id)
    return ordered


def build_balanced_training_pairs(
    labels: Sequence[base.CELabel],
    bank: list[dict[str, Any]],
    candidate_ids: dict[str, list[str]],
    *,
    negatives_per_positive: int,
    negatives_per_abstain: int,
    strong_positive_repeats: int,
) -> list[dict[str, Any]]:
    """Build deterministic pairs and fail closed on metric exposure."""

    global _PAIR_AUDIT
    if _CONFIG is None:
        raise RuntimeError("balanced policy was not validated before pair construction")
    cfg = _CONFIG
    seed = int(cfg["sampling_seed"])
    cap = int(cfg["max_unique_positive_uids_per_metric"])
    hard_match = int(cfg["hard_negatives_per_match"])
    global_match = int(cfg["global_balanced_negatives_per_match"])
    hard_abstain = int(cfg["hard_negatives_per_abstain"])
    global_abstain = int(cfg["global_balanced_negatives_per_abstain"])
    if hard_match + global_match != negatives_per_positive:
        raise ValueError("balanced match slots disagree with command")
    if hard_abstain + global_abstain != negatives_per_abstain:
        raise ValueError("balanced abstention slots disagree with command")

    bank_ids = [str(metric["metric_id"]) for metric in bank]
    bank_index = {metric_id: index for index, metric_id in enumerate(bank_ids)}
    cards = [metric_card(metric) for metric in bank]
    lexical = base.lexical_rankings(labels, cards)

    raw_positive: dict[str, list[tuple[int, base.CELabel]]] = defaultdict(list)
    nonmatch_indices: list[int] = []
    for index, label in enumerate(labels):
        if label.split != "train" or label.decision == "MATCH_FAMILY_ONLY":
            continue
        if label.decision == "MATCH":
            if label.metric_id is None:
                raise ValueError("MATCH label without metric ID")
            raw_positive[label.metric_id].append((index, label))
        else:
            nonmatch_indices.append(index)

    selected_positive: list[tuple[int, base.CELabel]] = []
    dropped_by_metric: dict[str, int] = {}
    for metric_id, values in sorted(raw_positive.items()):
        ranked = sorted(
            values,
            key=lambda value: (
                _stable_int(seed, "positive-cap", metric_id, value[1].norm_uid),
                value[1].norm_uid,
            ),
        )
        selected_positive.extend(ranked[:cap])
        if len(ranked) > cap:
            dropped_by_metric[metric_id] = len(ranked) - cap

    selected: list[tuple[int, base.CELabel, int]] = [
        (index, label, global_match) for index, label in selected_positive
    ] + [
        (index, labels[index], global_abstain) for index in nonmatch_indices
    ]
    selected.sort(
        key=lambda value: (
            _stable_int(seed, "query-order", value[1].norm_uid),
            value[1].norm_uid,
        )
    )

    rows: list[dict[str, Any]] = []
    negative_counts: Counter[str] = Counter()
    positive_pair_counts: Counter[str] = Counter()
    unique_positive_counts: Counter[str] = Counter()
    used_by_uid: dict[str, set[str]] = defaultdict(set)
    global_slots: list[tuple[int, base.CELabel, int]] = []

    for index, label, n_global in selected:
        ordered = _ordered_candidates(
            label, lexical[index], bank_ids, bank_index, candidate_ids
        )
        forbidden = set(label.acceptable_metric_ids)
        if label.decision == "MATCH":
            assert label.metric_id is not None
            unique_positive_counts[label.metric_id] += 1
            repeats = (
                strong_positive_repeats
                if label.supervision_strength == "strong"
                else 1
            )
            for repeat in range(repeats):
                rows.append(
                    {
                        "norm_uid": label.norm_uid,
                        "split": "train",
                        "query": label.query,
                        "metric_id": label.metric_id,
                        "metric_card": cards[bank_index[label.metric_id]],
                        "label": 1.0,
                        "kind": "positive_capped",
                        "repeat": repeat,
                        "supervision_strength": label.supervision_strength,
                    }
                )
                positive_pair_counts[label.metric_id] += 1
            n_hard = hard_match
        else:
            n_hard = hard_abstain

        hard_ids = [metric_id for metric_id in ordered if metric_id not in forbidden][
            :n_hard
        ]
        used_by_uid[label.norm_uid].update(hard_ids)
        for rank, metric_id in enumerate(hard_ids, 1):
            rows.append(
                {
                    "norm_uid": label.norm_uid,
                    "split": "train",
                    "query": label.query,
                    "metric_id": metric_id,
                    "metric_card": cards[bank_index[metric_id]],
                    "label": 0.0,
                    "kind": "hard_negative_balanced_v4",
                    "negative_rank": rank,
                    "gold_decision": label.decision,
                    "supervision_strength": label.supervision_strength,
                }
            )
            negative_counts[metric_id] += 1
        global_slots.append((index, label, n_global))

    minimum_negative = int(cfg["min_negative_exposure_per_bank_metric"])
    ratio = float(cfg["target_negative_to_positive_pair_ratio"])
    targets = {
        metric_id: max(
            minimum_negative,
            int(math.ceil(ratio * positive_pair_counts[metric_id])),
        )
        for metric_id in bank_ids
    }

    for _, label, n_global in global_slots:
        forbidden = set(label.acceptable_metric_ids) | used_by_uid[label.norm_uid]
        for slot in range(n_global):
            eligible = [metric_id for metric_id in bank_ids if metric_id not in forbidden]
            if not eligible:
                raise ValueError(f"no global negative available for {label.norm_uid}")

            def priority(metric_id: str) -> tuple[float, int, int]:
                target = targets[metric_id]
                remaining = max(target - negative_counts[metric_id], 0)
                normalized = remaining / target if target else 0.0
                tie = _stable_int(
                    seed,
                    "balanced-negative",
                    label.norm_uid,
                    str(slot),
                    metric_id,
                )
                return normalized, remaining, -tie

            metric_id = max(eligible, key=priority)
            forbidden.add(metric_id)
            used_by_uid[label.norm_uid].add(metric_id)
            rows.append(
                {
                    "norm_uid": label.norm_uid,
                    "split": "train",
                    "query": label.query,
                    "metric_id": metric_id,
                    "metric_card": cards[bank_index[metric_id]],
                    "label": 0.0,
                    "kind": "global_metric_balanced_negative_v4",
                    "negative_rank": slot + 1,
                    "gold_decision": label.decision,
                    "supervision_strength": label.supervision_strength,
                }
            )
            negative_counts[metric_id] += 1

    minimum_ratio = float(cfg["minimum_negative_to_positive_pair_ratio"])
    maximum_positive_fraction = float(
        cfg["maximum_positive_pair_fraction_per_metric"]
    )
    exposure_rows = []
    failures = []
    for metric_id in bank_ids:
        positive = int(positive_pair_counts[metric_id])
        negative = int(negative_counts[metric_id])
        total = positive + negative
        observed_ratio = negative / positive if positive else None
        positive_fraction = positive / total if total else 0.0
        reasons = []
        if negative < minimum_negative:
            reasons.append("minimum_negative_exposure")
        if positive and observed_ratio is not None and observed_ratio < minimum_ratio:
            reasons.append("negative_to_positive_ratio")
        if positive_fraction > maximum_positive_fraction:
            reasons.append("positive_fraction")
        if unique_positive_counts[metric_id] > cap:
            reasons.append("positive_uid_cap")
        row = {
            "metric_id": metric_id,
            "unique_positive_uids": int(unique_positive_counts[metric_id]),
            "positive_pairs": positive,
            "negative_pairs": negative,
            "negative_to_positive_pair_ratio": observed_ratio,
            "positive_pair_fraction": positive_fraction,
            "target_negative_pairs": int(targets[metric_id]),
            "gate_failures": reasons,
        }
        exposure_rows.append(row)
        if reasons:
            failures.append(row)

    _PAIR_AUDIT = {
        "schema_version": BALANCED_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "sampling_seed": seed,
        "raw_positive_uid_count": sum(len(values) for values in raw_positive.values()),
        "selected_positive_uid_count": len(selected_positive),
        "dropped_positive_uid_count": sum(dropped_by_metric.values()),
        "dropped_positive_uids_by_metric": dropped_by_metric,
        "typed_nonmatch_uid_count": len(nonmatch_indices),
        "pair_kind_counts": dict(sorted(Counter(row["kind"] for row in rows).items())),
        "gates": {
            "max_unique_positive_uids_per_metric": cap,
            "min_negative_exposure_per_bank_metric": minimum_negative,
            "minimum_negative_to_positive_pair_ratio": minimum_ratio,
            "maximum_positive_pair_fraction_per_metric": maximum_positive_fraction,
        },
        "failed_metric_count": len(failures),
        "failed_metrics": failures,
        "per_metric_exposure": exposure_rows,
    }
    if failures:
        raise ValueError(
            "metric exposure gates failed before training: "
            + json.dumps(failures[:10], sort_keys=True)
        )
    if _AUDIT_ONLY:
        raise PairAuditComplete(json.dumps(_PAIR_AUDIT, sort_keys=True))
    if not rows:
        raise ValueError("no balanced cross-encoder training pairs")
    return rows


def grouped_listwise_gate_report(
    labels: Sequence[base.CELabel],
    bank_ids: Sequence[str],
    scores: np.ndarray,
    score_threshold: float,
    margin_threshold: float,
    *,
    beta: float = 0.5,
) -> dict[str, Any]:
    """Add macro metric and prediction-concentration checks to full-bank eval."""

    report = _ORIGINAL_GATE_REPORT(
        labels,
        bank_ids,
        scores,
        score_threshold,
        margin_threshold,
        beta=beta,
    )
    if not labels:
        return report
    order = np.argsort(-scores, axis=1, kind="stable")
    top = order[:, 0]
    top_scores = scores[np.arange(len(labels)), top]
    second = scores[np.arange(len(labels)), order[:, 1]]
    retained = (top_scores >= score_threshold) & (
        (top_scores - second) >= margin_threshold
    )
    gold_ranks: dict[str, list[int]] = defaultdict(list)
    bank_position = {metric_id: index for index, metric_id in enumerate(bank_ids)}
    for row_index, label in enumerate(labels):
        if label.decision != "MATCH" or label.metric_id is None:
            continue
        rank = int(
            np.where(order[row_index] == bank_position[label.metric_id])[0][0]
        ) + 1
        gold_ranks[label.metric_id].append(rank)
    macro = {
        f"macro_metric_recall_at_{depth}": (
            float(
                np.mean(
                    [
                        np.mean(np.asarray(ranks) <= min(depth, len(bank_ids)))
                        for ranks in gold_ranks.values()
                    ]
                )
            )
            if gold_ranks
            else None
        )
        for depth in (1, 5, 10, 16, 30, 50)
    }
    predicted_ids = [bank_ids[int(top[index])] for index in np.where(retained)[0]]
    predicted_counts = Counter(predicted_ids)
    report["grouped_listwise"] = {
        "evaluation_unit": "one norm against the complete frozen bank",
        "gold_metric_count": len(gold_ranks),
        **macro,
        "retained_unique_predicted_metric_count": len(predicted_counts),
        "retained_max_metric_share": (
            max(predicted_counts.values()) / len(predicted_ids)
            if predicted_ids
            else None
        ),
        "retained_top_metric_counts": predicted_counts.most_common(10),
    }
    return report


def main() -> None:
    global _AUDIT_ONLY
    if "--audit-pairs-only" in sys.argv:
        sys.argv.remove("--audit-pairs-only")
        _AUDIT_ONLY = True
    base.validate_frozen_policy = _load_balanced_policy
    base.build_training_pairs = build_balanced_training_pairs
    base.gate_report = grouped_listwise_gate_report
    args = base.parse_args()
    try:
        report = base.train(args)
    except PairAuditComplete as exc:
        print(
            json.dumps(
                {
                    "status": "PAIR_EXPOSURE_AUDIT_PASS_NO_TRAINING",
                    "audit": json.loads(str(exc)),
                },
                sort_keys=True,
            )
        )
        return
    if _PAIR_AUDIT is None or _EVIDENCE_STATUS is None:
        raise RuntimeError("balanced pair audit missing after training")
    report["balanced_training_audit"] = _PAIR_AUDIT
    report["grouped_listwise_evaluation_contract"] = {
        "unit": "norm_uid",
        "candidate_universe": "complete frozen task bank",
        "selection": "top-1 score plus score/margin abstention gate",
        "pair_level_random_split_metrics_used": False,
        "development_status": _EVIDENCE_STATUS["development"],
        "blind_status": _EVIDENCE_STATUS["blind"],
    }
    output = Path(args.output_root).resolve() / args.task / "training_report.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "training_report": str(output),
                "training_report_sha256": sha256_file(output),
                "balanced_exposure_status": _PAIR_AUDIT["status"],
                "blind_status": _EVIDENCE_STATUS["blind"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
