#!/usr/bin/env python3
"""Select a conservative dev-only CE + two-order typed exact-leaf gate.

Acceptance requires the typed model to emit the same MATCH leaf under both
candidate orders and the pairwise CE to rank that leaf first.  CE probability
and top-two margin are selected on dev only, with the precision/Wilson contract
taking precedence over coverage.  This script must never receive test/blind
rows and fails closed if any input advertises such a split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping


CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def one_by_uid(rows: Iterable[Mapping[str, Any]], role: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in result:
            raise ValueError(f"missing/duplicate {role} norm_uid: {uid}")
        split = str(row.get("split") or "dev").lower()
        if split != "dev":
            raise ValueError(f"non-dev row in {role}: {split}")
        result[uid] = row
    return result


def wilson(successes: int, total: int, confidence: float = 0.95) -> list[float] | None:
    if total <= 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.0


def gold_ids(row: Mapping[str, Any]) -> set[str]:
    if row.get("decision") != "MATCH":
        return set()
    values = row.get("acceptable_metric_ids") or [row.get("metric_id")]
    return {str(value) for value in values if value not in (None, "")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--typed-original", required=True)
    parser.add_argument("--typed-reordered", required=True)
    parser.add_argument("--ce-scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--minimum-wilson-lower", type=float, default=0.85)
    parser.add_argument("--minimum-predictions", type=int, default=100)
    args = parser.parse_args()

    paths = {
        "truth": Path(args.truth).resolve(),
        "typed_original": Path(args.typed_original).resolve(),
        "typed_reordered": Path(args.typed_reordered).resolve(),
        "ce_scores": Path(args.ce_scores).resolve(),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    truth = one_by_uid(read_jsonl(paths["truth"]), "truth")
    original = one_by_uid(read_jsonl(paths["typed_original"]), "typed-original")
    reordered = one_by_uid(read_jsonl(paths["typed_reordered"]), "typed-reordered")
    if set(original) != set(truth) or set(reordered) != set(truth):
        raise ValueError("typed prediction coverage differs from exact dev truth")

    ce_groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    ce_pairs: set[tuple[str, str]] = set()
    for row in read_jsonl(paths["ce_scores"]):
        split = str(row.get("split") or "").lower()
        if split != "dev":
            raise ValueError(f"non-dev CE score row: {split}")
        uid, metric_id = str(row.get("norm_uid") or ""), str(row.get("metric_id") or "")
        identity = (uid, metric_id)
        if not all(identity) or identity in ce_pairs:
            raise ValueError(f"missing/duplicate CE pair: {identity}")
        ce_pairs.add(identity)
        probabilities = row.get("probabilities") or {}
        if set(probabilities) != {"EXACT", "REJECT"}:
            raise ValueError(f"invalid binary CE probabilities: {identity}")
        exact = float(probabilities["EXACT"])
        reject = float(probabilities["REJECT"])
        if not (0 <= exact <= 1 and 0 <= reject <= 1 and abs(exact + reject - 1) <= 1e-4):
            raise ValueError(f"invalid CE probability vector: {identity}")
        ce_groups[uid].append((metric_id, exact))

    overlap = sorted(set(truth) & set(ce_groups))
    if not overlap:
        raise ValueError("CE scores have no overlap with typed dev truth")
    stable: dict[str, tuple[str, str]] = {}
    for uid in overlap:
        left, right = original[uid], reordered[uid]
        if (
            left.get("decision") == right.get("decision") == "MATCH"
            and left.get("metric_id") not in (None, "")
            and left.get("metric_id") == right.get("metric_id")
        ):
            lconf = str(left.get("confidence") or "").lower()
            rconf = str(right.get("confidence") or "").lower()
            if lconf not in CONFIDENCE_RANK or rconf not in CONFIDENCE_RANK:
                raise ValueError(f"invalid typed confidence: {uid}")
            stable[uid] = (str(left["metric_id"]), min((lconf, rconf), key=CONFIDENCE_RANK.get))

    probabilities = sorted(
        {score for uid in stable for metric, score in ce_groups[uid] if metric == stable[uid][0]}
    )
    fixed_thresholds = {0.0, 0.90, 0.95, 0.98, 0.99, 0.995, 0.9960545301437378, 0.997, 0.999, 0.9995, 0.9999, 1.0}
    if probabilities:
        for index in range(0, 101):
            fixed_thresholds.add(probabilities[round(index * (len(probabilities) - 1) / 100)])
    score_thresholds = sorted(fixed_thresholds)
    margin_thresholds = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
    gold_match_count = sum(bool(gold_ids(truth[uid])) for uid in overlap)
    surfaces = []
    for confidence in ("high", "medium", "low"):
        minimum_rank = CONFIDENCE_RANK[confidence]
        for score_threshold in score_thresholds:
            for margin_threshold in margin_thresholds:
                accepted: list[tuple[str, str]] = []
                for uid, (metric_id, typed_confidence) in stable.items():
                    if CONFIDENCE_RANK[typed_confidence] < minimum_rank:
                        continue
                    ranked = sorted(ce_groups[uid], key=lambda value: (-value[1], value[0]))
                    if not ranked or ranked[0][0] != metric_id:
                        continue
                    top = ranked[0][1]
                    second = ranked[1][1] if len(ranked) > 1 else 0.0
                    if top >= score_threshold and top - second >= margin_threshold:
                        accepted.append((uid, metric_id))
                correct = sum(metric_id in gold_ids(truth[uid]) for uid, metric_id in accepted)
                support = len(accepted)
                precision = correct / support if support else 0.0
                recall = correct / gold_match_count if gold_match_count else 0.0
                interval = wilson(correct, support)
                lower = interval[0] if interval else None
                passed = bool(
                    support >= args.minimum_predictions
                    and precision >= args.minimum_precision
                    and lower is not None
                    and lower >= args.minimum_wilson_lower
                )
                surfaces.append(
                    {
                        "minimum_typed_confidence": confidence,
                        "minimum_ce_exact_probability": score_threshold,
                        "minimum_ce_top_margin": margin_threshold,
                        "accepted": support,
                        "correct": correct,
                        "precision": precision,
                        "precision_wilson_95": interval,
                        "recall_of_overlap_gold_matches": recall,
                        "f_beta_0_5": f_beta(precision, recall),
                        "gate_passed": passed,
                    }
                )

    feasible = [row for row in surfaces if row["gate_passed"]]
    if feasible:
        chosen = max(
            feasible,
            key=lambda row: (
                row["accepted"], row["precision_wilson_95"][0], row["precision"], row["f_beta_0_5"]
            ),
        )
        status = "PASS_VALIDATED_DEV_HYBRID_GATE"
    else:
        chosen = max(
            surfaces,
            key=lambda row: (
                row["precision_wilson_95"][0] if row["precision_wilson_95"] else -1.0,
                row["precision"], row["accepted"], row["f_beta_0_5"],
            ),
        )
        status = "FAIL_NO_DEV_HYBRID_GATE_MEETS_CONTRACT"
    report = {
        "schema_version": "silver-match-v3-humor-ce-typed-dev-hybrid-gate-v1",
        "status": status,
        "role": "dev_selection_only",
        "test_or_blind_rows_read": 0,
        "policy": {
            "acceptance": "typed exact leaf stable under both orders AND CE top-1 same leaf",
            "selection_priority": "meet precision and Wilson contract before coverage",
            "minimum_precision": args.minimum_precision,
            "minimum_wilson_lower": args.minimum_wilson_lower,
            "minimum_predictions": args.minimum_predictions,
        },
        "coverage": {
            "truth_dev_rows": len(truth),
            "ce_dev_norms": len(ce_groups),
            "paired_overlap_rows": len(overlap),
            "overlap_gold_match_rows": gold_match_count,
            "typed_two_order_stable_match_rows": len(stable),
        },
        "selected_gate": chosen,
        "feasible_gate_count": len(feasible),
        "surface_count": len(surfaces),
        "surface": surfaces,
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({key: report[key] for key in ("status", "coverage", "selected_gate")}, sort_keys=True))


if __name__ == "__main__":
    main()
