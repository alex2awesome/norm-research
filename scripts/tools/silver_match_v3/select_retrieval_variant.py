#!/usr/bin/env python3
"""Freeze a task's encoder/adapter plus fusion choice using external dev only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import sha256_file


KINDS = {"bge_base", "nemotron_base", "adapter"}
ADAPTER_POLICIES = {"legacy_k50_gain", "saturated_r50_noninferiority_depth_gain"}
SATURATED_POLICY_TASKS = {"peer-review", "legal-outcome-prediction"}


def paired_bootstrap_evidence(
    reference_items: Sequence[Mapping[str, Any]],
    candidate_items: Sequence[Mapping[str, Any]],
    *,
    noninferiority_margin: float,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    alpha: float = 0.05,
    minimum_n: int = 25,
) -> dict[str, Any]:
    if not 0 <= noninferiority_margin < 1:
        raise ValueError("noninferiority margin must be in [0, 1)")
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if minimum_n < 1:
        raise ValueError("minimum paired n must be positive")
    reference = {str(row["norm_uid"]): row for row in reference_items}
    candidate = {str(row["norm_uid"]): row for row in candidate_items}
    if len(reference) != len(reference_items) or len(candidate) != len(candidate_items):
        raise ValueError("duplicate UID in paired retrieval items")
    if reference.keys() != candidate.keys():
        raise ValueError("paired retrieval variants cover different UIDs")
    uids = sorted(reference)
    if any(
        str(reference[uid].get("metric_id")) != str(candidate[uid].get("metric_id"))
        for uid in uids
    ):
        raise ValueError("paired retrieval variants disagree on gold metric identity")
    if not uids:
        raise ValueError("paired retrieval evidence is missing item-level ranks")
    before = np.asarray(
        [int(reference[uid]["exact_rank"]) for uid in uids], dtype=float
    )
    after = np.asarray([int(candidate[uid]["exact_rank"]) for uid in uids], dtype=float)
    if np.any(before < 1) or np.any(after < 1):
        raise ValueError("paired exact ranks must be positive")
    deltas = {
        "recall_at_50": (after <= 50).astype(float) - (before <= 50).astype(float),
        "recall_at_80": (after <= 80).astype(float) - (before <= 80).astype(float),
        "mrr": 1.0 / after - 1.0 / before,
        "recall_at_16": (after <= 16).astype(float) - (before <= 16).astype(float),
        "recall_at_30": (after <= 30).astype(float) - (before <= 30).astype(float),
    }
    n = len(uids)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, max(n, 1), size=(bootstrap_repetitions, max(n, 1)))
    estimates = {}
    for name, values in deltas.items():
        samples = np.mean(values[indices], axis=1) if n else np.asarray([])
        estimates[name] = {
            "paired_point_delta": float(np.mean(values)) if n else None,
            "one_sided_lower_95": (
                float(np.quantile(samples, alpha)) if len(samples) else None
            ),
        }
    checks = {
        "minimum_paired_support": n >= minimum_n,
        "r50_observed_nonloss": estimates["recall_at_50"]["paired_point_delta"] >= 0,
        "r50_lower_bound_above_margin": estimates["recall_at_50"]["one_sided_lower_95"]
        >= -noninferiority_margin,
        "r80_observed_nonloss": estimates["recall_at_80"]["paired_point_delta"] >= 0,
        "r80_lower_bound_above_margin": estimates["recall_at_80"]["one_sided_lower_95"]
        >= -noninferiority_margin,
        "mrr_supported_positive_gain": estimates["mrr"]["paired_point_delta"] > 0
        and estimates["mrr"]["one_sided_lower_95"] > 0,
        "r16_or_r30_supported_positive_gain": any(
            estimates[name]["paired_point_delta"] > 0
            and estimates[name]["one_sided_lower_95"] > 0
            for name in ("recall_at_16", "recall_at_30")
        ),
    }
    return {
        "passed": all(checks.values()),
        "n": n,
        "checks": checks,
        "estimates": estimates,
        "rank_direction_counts": {
            "improved": int(np.sum(after < before)),
            "worsened": int(np.sum(after > before)),
            "tied": int(np.sum(after == before)),
        },
        "noninferiority_margin": noninferiority_margin,
        "bootstrap": {
            "method": "paired_percentile_one_sided",
            "repetitions": bootstrap_repetitions,
            "seed": bootstrap_seed,
            "alpha": alpha,
        },
        "tiny_dev_caveat": (
            "The paired percentile bound is conditional on this source-disjoint dev "
            "panel; small n limits power and any failed check retains the frozen base/fusion."
        ),
    }


def choose_variant(
    variants: Sequence[Mapping[str, Any]],
    min_adapter_gain: float = 0.03,
    *,
    adapter_policy: str = "legacy_k50_gain",
    noninferiority_margin: float = 0.05,
    bootstrap_repetitions: int = 20_000,
    bootstrap_seed: int = 947_311,
    minimum_paired_n: int = 25,
) -> dict[str, Any]:
    if not variants:
        raise ValueError("no retrieval variants")
    names = [str(row["name"]) for row in variants]
    if len(names) != len(set(names)):
        raise ValueError("duplicate retrieval variant name")
    references = [row for row in variants if row["kind"] == "nemotron_base"]
    if len(references) != 1:
        raise ValueError("exactly one unadapted nemotron_base reference is required")
    reference = references[0]
    ref_metrics = reference["dev_metrics"]
    if adapter_policy not in ADAPTER_POLICIES:
        raise ValueError(f"unknown adapter policy: {adapter_policy}")
    decisions = []
    eligible = []
    for row in variants:
        if row["kind"] not in KINDS:
            raise ValueError(f"unknown variant kind: {row['kind']}")
        metrics = row["dev_metrics"]
        if row["kind"] == "adapter":
            gain = metrics["recall_at_50"] - ref_metrics["recall_at_50"]
            wide_ok = metrics["recall_at_80"] >= ref_metrics["recall_at_80"]
            if adapter_policy == "legacy_k50_gain":
                paired = None
                passed = gain >= min_adapter_gain and wide_ok
                reason = (
                    "adapter clears +K50/no-K80-loss gate"
                    if passed
                    else "adapter rejected by +K50/no-K80-loss gate"
                )
            else:
                paired = paired_bootstrap_evidence(
                    reference.get("items") or [],
                    row.get("items") or [],
                    noninferiority_margin=noninferiority_margin,
                    bootstrap_repetitions=bootstrap_repetitions,
                    bootstrap_seed=bootstrap_seed,
                    minimum_n=minimum_paired_n,
                )
                passed = paired["passed"]
                reason = (
                    "adapter clears saturated-R50 paired noninferiority/depth-gain gate"
                    if passed
                    else "adapter rejected by saturated-R50 paired noninferiority/depth-gain gate"
                )
        else:
            gain, wide_ok, paired, passed = None, None, None, True
            reason = "unadapted baseline is always selection-eligible"
        decision = {
            "name": row["name"],
            "kind": row["kind"],
            "eligible": passed,
            "reason": reason,
            "adapter_recall_at_50_gain_vs_unadapted_nemotron": gain,
            "adapter_no_recall_at_80_loss": wide_ok,
            "paired_external_dev_evidence": paired,
            "dev_metrics": metrics,
        }
        decisions.append(decision)
        if passed:
            eligible.append(row)
    chosen = max(
        eligible,
        key=lambda row: (
            row["dev_metrics"]["recall_at_50"]
            + row["dev_metrics"]["macro_recall_at_50"],
            row["dev_metrics"]["recall_at_80"]
            + row["dev_metrics"]["macro_recall_at_80"],
            row["dev_metrics"]["mrr"],
            row["kind"] != "adapter",  # conservative deterministic tie-break
            row["name"],
        ),
    )
    return {
        "chosen_name": chosen["name"],
        "chosen_kind": chosen["kind"],
        "minimum_adapter_recall_at_50_gain": min_adapter_gain,
        "adapter_policy": adapter_policy,
        "noninferiority_margin": (
            noninferiority_margin
            if adapter_policy == "saturated_r50_noninferiority_depth_gain"
            else None
        ),
        "adapter_reference": reference["name"],
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME:KIND:/absolute/dev-fusion-report.json",
    )
    parser.add_argument("--min-adapter-gain", type=float, default=0.03)
    parser.add_argument(
        "--adapter-policy", choices=sorted(ADAPTER_POLICIES), default="legacy_k50_gain"
    )
    parser.add_argument("--noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=947_311)
    parser.add_argument("--minimum-paired-n", type=int, default=25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dev selection: {output}")
    if (
        args.adapter_policy == "saturated_r50_noninferiority_depth_gain"
        and args.task not in SATURATED_POLICY_TASKS
    ):
        raise ValueError(
            "saturated_r50_noninferiority_depth_gain is task-local to Peer/Legal"
        )

    variants = []
    label_hashes = None
    for spec in args.variant:
        try:
            name, kind, raw_path = spec.split(":", 2)
        except ValueError as exc:
            raise ValueError(f"invalid --variant {spec!r}") from exc
        if kind not in KINDS:
            raise ValueError(f"invalid variant kind: {kind}")
        path = Path(raw_path).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("task") != args.task or report.get("selection_split") != "dev":
            raise ValueError(f"not a dev fusion report for {args.task}: {path}")
        if any(
            split != "dev" and count
            for split, count in (report.get("split_counts") or {}).items()
        ):
            raise ValueError(f"fusion selection consumed a non-dev split: {path}")
        current_labels = report.get("label_inputs") or {}
        if label_hashes is None:
            label_hashes = current_labels
        elif current_labels != label_hashes:
            raise ValueError("variants were selected on different dev label artifacts")
        variants.append(
            {
                "name": name,
                "kind": kind,
                "fusion_report": str(path),
                "fusion_report_sha256": sha256_file(path),
                "candidate_inputs": report["candidate_inputs"],
                "dev_metrics": report["metrics"]["dev"],
                "items": report.get("items") or [],
            }
        )
    selection = choose_variant(
        variants,
        args.min_adapter_gain,
        adapter_policy=args.adapter_policy,
        noninferiority_margin=args.noninferiority_margin,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
        minimum_paired_n=args.minimum_paired_n,
    )
    chosen = next(row for row in variants if row["name"] == selection["chosen_name"])
    payload = {
        "schema_version": "silver-match-v3-retrieval-selection-v2",
        "task": args.task,
        "selection_split": "external_dev_only",
        "frozen_test_consumed": False,
        "selection": selection,
        "chosen": chosen,
        "variants": variants,
        "label_inputs": label_hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
