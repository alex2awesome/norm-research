#!/usr/bin/env python3
"""Freeze the balanced press-release verifier-dev proposal/truth pairs.

This is the first permitted join of the independently resolved verifier-dev
truth and the truth-hidden three-order R4 proposals.  It fails closed unless
both inputs remain bound to the original 300 frozen identities.  Unresolved
truth rows are retained in the audit accounting but never enter scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl, sha256_file, write_jsonl


POLICY_SCHEMA = "silver-match-v3-press-releases-verifier-dev-policy-amendment-v3"
BASE_POLICY_SCHEMA = "silver-match-v3-press-releases-verifier-dev-policy-v2"


def _index(path: Path, *, allow_empty: bool = False) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    values = {str(row.get("norm_uid") or ""): row for row in rows}
    if (not rows and not allow_empty) or "" in values or len(values) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid values: {path}")
    return values


def _stable_key(seed: int, target: str, metric_id: str, uid: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{target}\0{metric_id}\0{uid}".encode("utf-8")
    ).hexdigest()


def round_robin_metrics(
    rows: Iterable[dict[str, Any]], *, limit: int, seed: int, target: str
) -> list[dict[str, Any]]:
    """Select deterministically while cycling proposal leaves before refill."""

    groups: dict[str, deque[dict[str, Any]]] = {}
    staged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        staged[str(row["proposal_metric_id"])].append(row)
    for metric_id, values in staged.items():
        groups[metric_id] = deque(
            sorted(
                values,
                key=lambda row: _stable_key(
                    seed, target, metric_id, str(row["norm_uid"])
                ),
            )
        )
    selected: list[dict[str, Any]] = []
    metric_ids = sorted(groups)
    while len(selected) < limit and metric_ids:
        remaining: list[str] = []
        for metric_id in metric_ids:
            if len(selected) >= limit:
                break
            selected.append(groups[metric_id].popleft())
            if groups[metric_id]:
                remaining.append(metric_id)
        metric_ids = remaining
    return selected


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def build_pair_universe(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(getattr(args, name.replace("-", "_"))).resolve()
        for name in (
            "policy",
            "base-policy",
            "identity-freeze",
            "identities",
            "source-pack-validation",
            "source-items",
            "truth-report",
            "truth",
            "unresolved",
            "proposals",
            "proposal-report",
            "r4-output-freeze",
            "candidates",
        )
    }
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)

    policy = json.loads(paths["policy"].read_text(encoding="utf-8"))
    base = json.loads(paths["base-policy"].read_text(encoding="utf-8"))
    if (
        policy.get("schema_version") != POLICY_SCHEMA
        or base.get("schema_version") != BASE_POLICY_SCHEMA
        or policy.get("status")
        != "FROZEN_BEFORE_FRESH_DEV_PREDICTIONS_LABELS_OR_TRUTH"
        or base.get("status")
        != "FROZEN_BEFORE_FRESH_DEV_PREDICTIONS_LABELS_OR_TRUTH"
        or policy.get("task") != args.task
        or base.get("task") != args.task
        or (policy.get("base_policy") or {}).get("sha256")
        != sha256_file(paths["base-policy"])
    ):
        raise ValueError("unsupported or drifted verifier-dev policy chain")
    protocol = base.get("proposal_and_balance_protocol") or {}
    if (
        protocol.get("pair_universe")
        != "Intersection of the fixed 300 identities, independently frozen truth, and strict R4 proposals."
        or int(protocol.get("balance_seed", -1)) != args.seed
        or int(protocol.get("maximum_pairs", -1)) != args.maximum_pairs
    ):
        raise ValueError("pair-universe seed/cap differs from frozen policy")

    identity_freeze = json.loads(paths["identity-freeze"].read_text(encoding="utf-8"))
    identities = _index(paths["identities"])
    if (
        identity_freeze.get("schema_version")
        != "silver-match-v3-clean-gepa-panel-freeze-v1"
        or identity_freeze.get("status")
        != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or identity_freeze.get("role") != "verifier_dev"
        or identity_freeze.get("task") != args.task
        or int(identity_freeze.get("selected_count", -1)) != 300
        or (identity_freeze.get("outputs") or {}).get("identities", {}).get(
            "sha256"
        )
        != sha256_file(paths["identities"])
        or len(identities) != 300
    ):
        raise ValueError("fixed verifier-dev identity freeze is invalid")
    canonical = base.get("canonical_inputs") or {}
    if (
        (canonical.get("fresh_dev_identities") or {}).get("sha256")
        != sha256_file(paths["identities"])
        or (canonical.get("fresh_dev_identity_freeze") or {}).get("sha256")
        != sha256_file(paths["identity-freeze"])
    ):
        raise ValueError("identity files differ from frozen base policy")

    source_validation = json.loads(
        paths["source-pack-validation"].read_text(encoding="utf-8")
    )
    source_items = _index(paths["source-items"])
    if (
        source_validation.get("truth_hidden") is not True
        or source_validation.get("role") != "verifier_dev"
        or source_validation.get("task") != args.task
        or int(source_validation.get("count", -1)) != 300
        or (source_validation.get("inputs") or {}).get("identities", {}).get(
            "sha256"
        )
        != sha256_file(paths["identities"])
        or (source_validation.get("outputs") or {}).get("items", {}).get("sha256")
        != sha256_file(paths["source-items"])
        or set(source_items) != set(identities)
    ):
        raise ValueError("truth-hidden source pack is not the fixed identity panel")

    truth_report = json.loads(paths["truth-report"].read_text(encoding="utf-8"))
    truth = _index(paths["truth"])
    unresolved = _index(paths["unresolved"], allow_empty=True)
    if (
        truth_report.get("schema_version")
        != "silver-match-v3-exact-multi-pass-truth-report-v1"
        or truth_report.get("task") != args.task
        or int(truth_report.get("source_count", -1)) != 300
        or int(truth_report.get("resolved_count", -1)) != len(truth)
        or int(truth_report.get("unresolved_count", -1)) != len(unresolved)
        or (truth_report.get("outputs") or {}).get("resolved", {}).get("sha256")
        != sha256_file(paths["truth"])
        or (truth_report.get("outputs") or {}).get("unresolved", {}).get("sha256")
        != sha256_file(paths["unresolved"])
        or set(truth) & set(unresolved)
        or set(truth) | set(unresolved) != set(identities)
    ):
        raise ValueError("resolved/unresolved truth does not partition fixed identities")
    max_resolvers = int(
        (base.get("fresh_truth_protocol") or {}).get("maximum_resolver_passes", -1)
    )
    rounds = truth_report.get("rounds") or []
    if len(rounds) != 2 + max_resolvers:
        raise ValueError("truth release does not use the frozen resolver-pass cap")

    proposal_report = json.loads(paths["proposal-report"].read_text(encoding="utf-8"))
    output_freeze = json.loads(paths["r4-output-freeze"].read_text(encoding="utf-8"))
    proposals = _index(paths["proposals"])
    candidates = _index(paths["candidates"])
    if (
        proposal_report.get("schema_version")
        != "silver-match-v3-three-order-consensus-proposals-v1"
        or proposal_report.get("task") != args.task
        or int(proposal_report.get("input_count", -1)) != 300
        or int(proposal_report.get("consensus_match_count", -1)) != len(proposals)
        or (proposal_report.get("output") or {}).get("sha256")
        != sha256_file(paths["proposals"])
        or (proposal_report.get("output_freeze") or {}).get("sha256")
        != sha256_file(paths["r4-output-freeze"])
        or output_freeze.get("status") != "FROZEN_COMPLETE_BEFORE_TRUTH_JOIN"
        or output_freeze.get("truth_content_read_by_freezer") is not False
        or set(candidates) != set(identities)
        or not set(proposals) <= set(identities)
    ):
        raise ValueError("R4 strict proposals/candidates are not frozen identity-bound inputs")

    eligible: list[dict[str, Any]] = []
    for uid in sorted(set(truth) & set(proposals)):
        gold, proposal, candidate = truth[uid], proposals[uid], candidates[uid]
        proposal_id = str(proposal.get("metric_id") or "")
        candidate_ids = {
            str(row.get("metric_id") or "") for row in candidate.get("candidates") or []
        }
        if (
            proposal.get("decision") != "MATCH"
            or proposal.get("task") != args.task
            or proposal_id not in candidate_ids
        ):
            raise ValueError(f"invalid strict proposal/candidate binding: {uid}")
        target = (
            "CONFIRM_MATCH"
            if gold.get("decision") == "MATCH"
            and str(gold.get("metric_id") or "") == proposal_id
            else "REJECT"
        )
        eligible.append(
            {
                "norm_uid": uid,
                "target": target,
                "proposal_metric_id": proposal_id,
                "source_group": str(
                    gold.get("source_group")
                    or source_items[uid].get("source_group")
                    or source_items[uid].get("split_group")
                    or ""
                ),
            }
        )
    by_target = {
        target: [row for row in eligible if row["target"] == target]
        for target in ("CONFIRM_MATCH", "REJECT")
    }
    if not all(by_target.values()):
        raise ValueError("verifier-dev pair universe lacks one binary target class")
    per_class = min(
        len(by_target["CONFIRM_MATCH"]),
        len(by_target["REJECT"]),
        args.maximum_pairs // 2,
    )
    selected = []
    for target in ("CONFIRM_MATCH", "REJECT"):
        selected.extend(
            round_robin_metrics(
                by_target[target], limit=per_class, seed=args.seed, target=target
            )
        )
    selected.sort(
        key=lambda row: _stable_key(
            args.seed,
            str(row["target"]),
            str(row["proposal_metric_id"]),
            str(row["norm_uid"]),
        )
    )
    selected_uids = [str(row["norm_uid"]) for row in selected]
    if len(selected_uids) != len(set(selected_uids)) or any(
        not row["source_group"] for row in selected
    ):
        raise ValueError("selected pair identities/source groups are invalid")
    selected_groups = [row["source_group"] for row in selected]
    if len(selected_groups) != len(set(selected_groups)):
        raise ValueError("selected pair universe repeats a source group")

    output_root.mkdir(parents=True)
    output_paths = {
        "truth": output_root / "truth.jsonl",
        "primary": output_root / "primary.jsonl",
        "candidates": output_root / "candidates.top50.jsonl",
        "targets": output_root / "targets.jsonl",
    }
    write_jsonl(output_paths["truth"], [truth[uid] for uid in selected_uids])
    write_jsonl(output_paths["primary"], [proposals[uid] for uid in selected_uids])
    write_jsonl(output_paths["candidates"], [candidates[uid] for uid in selected_uids])
    write_jsonl(output_paths["targets"], selected)

    eligible_counts = Counter(row["target"] for row in eligible)
    selected_counts = Counter(row["target"] for row in selected)
    report = {
        "schema_version": "silver-match-v3-pr-verifier-dev-pair-universe-v1",
        "status": "FROZEN_BALANCED_BEFORE_VERIFIER_INFERENCE",
        "task": args.task,
        "role": "verifier_dev",
        "policy": {
            "pair_universe": protocol["pair_universe"],
            "balancing": protocol["balancing"],
            "seed": args.seed,
            "maximum_pairs": args.maximum_pairs,
            "round_robin_key": "proposal_metric_id_then_stable_sha256",
        },
        "fixed_identity_count": len(identities),
        "resolved_truth_count": len(truth),
        "unresolved_truth_count": len(unresolved),
        "strict_proposal_count": len(proposals),
        "eligible_intersection_count": len(eligible),
        "eligible_target_counts": dict(sorted(eligible_counts.items())),
        "selected_count": len(selected),
        "selected_target_counts": dict(sorted(selected_counts.items())),
        "selected_source_groups": len(set(selected_groups)),
        "selected_proposal_metric_counts": dict(
            sorted(Counter(row["proposal_metric_id"] for row in selected).items())
        ),
        "excluded": {
            "unresolved_with_strict_proposal": len(set(unresolved) & set(proposals)),
            "resolved_without_strict_proposal": len(set(truth) - set(proposals)),
            "eligible_dropped_for_balance": len(eligible) - len(selected),
        },
        "contracts": {
            "truth_join_happened_only_after_r4_output_freeze": True,
            "unresolved_truth_excluded": True,
            "identity_substitution": False,
            "fresh_dev_may_not_mutate_prompt": True,
            "fresh_dev_excluded_from_retriever_or_ce_gradients": True,
            "fresh_dev_excluded_from_mi_or_outcome_estimation": True,
            "fresh_dev_is_not_final_blind_audit": True,
        },
        "inputs": {name: _ref(path) for name, path in paths.items()},
        "outputs": {name: _ref(path) for name, path in output_paths.items()},
    }
    report_path = output_root / "FREEZE.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**report, "freeze_sha256": sha256_file(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="press-releases")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--base-policy", required=True)
    parser.add_argument("--identity-freeze", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--source-pack-validation", required=True)
    parser.add_argument("--source-items", required=True)
    parser.add_argument("--truth-report", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--unresolved", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--proposal-report", required=True)
    parser.add_argument("--r4-output-freeze", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--seed", type=int, default=2026071305)
    parser.add_argument("--maximum-pairs", type=int, default=300)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(build_pair_universe(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
