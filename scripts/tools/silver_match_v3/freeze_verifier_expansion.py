#!/usr/bin/env python3
"""Freeze a leakage-safe, source-disjoint verifier calibration expansion.

Selection uses only extracted norms, retriever candidates, and the identities of
previously labeled rows.  It must be run before any adjudicator or labeler sees
the selected rows.  One row is selected per source group, so verifier-train,
verifier-dev, and permanent blind-audit roles are source-group disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_group_for


def stable_key(seed: int, *parts: object) -> str:
    value = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def candidate_signal(row: dict[str, Any]) -> dict[str, Any]:
    candidates = list(row.get("candidates") or [])
    if len(candidates) < 2:
        raise ValueError(f"candidate row has fewer than two metrics: {row.get('norm_uid')}")
    metric_ids = [str(value.get("metric_id") or "") for value in candidates]
    if any(not value for value in metric_ids) or len(metric_ids) != len(set(metric_ids)):
        raise ValueError(f"missing/duplicate metric IDs: {row.get('norm_uid')}")
    top, second = candidates[:2]
    first_score = float(top.get("rrf_score") or 0.0)
    second_score = float(second.get("rrf_score") or 0.0)

    def winner(key: str) -> str:
        eligible = [value for value in candidates if value.get(key) is not None]
        return str(min(eligible, key=lambda value: (int(value[key]), str(value["metric_id"])))["metric_id"])

    dense_winner, word_winner, char_winner = (
        winner("dense_rank"),
        winner("word_rank"),
        winner("char_rank"),
    )
    return {
        "top_metric_id": metric_ids[0],
        "rrf_margin": first_score - second_score,
        "top_rrf_score": first_score,
        "dense_winner": dense_winner,
        "word_winner": word_winner,
        "char_winner": char_winner,
        "dense_lexical_agree": dense_winner == word_winner,
    }


def balanced_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin boundary strata and top leaves, enforcing unique groups."""
    buckets: dict[str, dict[str, deque[dict[str, Any]]]] = defaultdict(dict)
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[(row["boundary_stratum"], row["top_metric_id"])].append(row)
    for (stratum, metric_id), values in by_pair.items():
        values.sort(key=lambda row: stable_key(seed, "row", row["norm_uid"]))
        buckets[stratum][metric_id] = deque(values)
    strata = sorted(buckets, key=lambda value: stable_key(seed, "stratum", value))
    metric_orders = {
        stratum: sorted(
            buckets[stratum], key=lambda value: stable_key(seed, "metric", stratum, value)
        )
        for stratum in strata
    }
    metric_cursor = Counter()
    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            metrics = metric_orders[stratum]
            for _ in metrics:
                metric = metrics[metric_cursor[stratum] % len(metrics)]
                metric_cursor[stratum] += 1
                queue = buckets[stratum][metric]
                while queue and queue[0]["source_group"] in used_groups:
                    queue.popleft()
                if not queue:
                    continue
                row = queue.popleft()
                used_groups.add(row["source_group"])
                selected.append(row)
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"could select only {len(selected)} of {count} source-disjoint rows")
    return selected


def assign_roles(
    selected: list[dict[str, Any]], role_counts: dict[str, int], seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Assign immutable roles while approximately balancing every stratum."""
    output = {role: [] for role in role_counts}
    total = len(selected)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_stratum[row["boundary_stratum"]].append(row)
    for stratum, values in sorted(by_stratum.items()):
        values.sort(key=lambda row: stable_key(seed, "role-order", row["norm_uid"]))
        for row in values:
            eligible = [role for role, target in role_counts.items() if len(output[role]) < target]
            role = max(
                eligible,
                key=lambda value: (
                    role_counts[value] / total
                    - sum(item["boundary_stratum"] == stratum for item in output[value])
                    / len(values),
                    role_counts[value] - len(output[value]),
                    stable_key(seed, "role", stratum, row["norm_uid"], value),
                ),
            )
            output[role].append(row)
    actual = {role: len(rows) for role, rows in output.items()}
    if actual != role_counts:
        # The proportional pass can strand a small global deficit.  Rebalance
        # deterministically without changing the selected UID/group universe.
        over = [role for role in output if len(output[role]) > role_counts[role]]
        under = [role for role in output if len(output[role]) < role_counts[role]]
        for target in under:
            while len(output[target]) < role_counts[target]:
                donor = max(over, key=lambda value: len(output[value]) - role_counts[value])
                moved = output[donor].pop()
                output[target].append(moved)
                if len(output[donor]) == role_counts[donor]:
                    over.remove(donor)
    if {role: len(rows) for role, rows in output.items()} != role_counts:
        raise AssertionError("failed to allocate exact verifier expansion roles")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--norms", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--exclude-panel", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dev-count", type=int, default=400)
    parser.add_argument("--train-count", type=int, default=140)
    parser.add_argument("--audit-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    paths = {
        "norms": Path(args.norms).resolve(),
        "candidates": Path(args.candidates).resolve(),
    }
    exclude_paths = [Path(value).resolve() for value in args.exclude_panel]
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite frozen expansion: {output_root}")

    norm_rows = list(read_jsonl(paths["norms"]))
    if any(row.get("task") != args.task for row in norm_rows):
        raise ValueError("norm input contains rows outside the requested task")
    norms = {str(row["norm_uid"]): row for row in norm_rows}
    if not norms:
        raise ValueError("empty norms input")
    candidates = {str(row["norm_uid"]): row for row in read_jsonl(paths["candidates"])}
    if len(candidates) != sum(1 for _ in read_jsonl(paths["candidates"])):
        raise ValueError("duplicate candidate UID")
    if not set(candidates).issubset(norms):
        raise ValueError("candidate UIDs absent from norm universe")
    if any(row.get("task") != args.task for row in candidates.values()):
        raise ValueError("candidate input contains rows outside the requested task")

    excluded_uids: set[str] = set()
    for path in exclude_paths:
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if uid not in norms:
                raise ValueError(f"excluded panel UID absent from norms: {uid}")
            excluded_uids.add(uid)
    excluded_groups = {split_group_for(norms[uid]) for uid in excluded_uids}

    eligible: list[dict[str, Any]] = []
    margins: list[float] = []
    for uid, candidate in candidates.items():
        group = split_group_for(norms[uid])
        if group in excluded_groups:
            continue
        signal = candidate_signal(candidate)
        margins.append(float(signal["rrf_margin"]))
        eligible.append(
            {
                **candidate,
                "source_group": group,
                **signal,
            }
        )
    low, high = quantile(margins, 1 / 3), quantile(margins, 2 / 3)
    for row in eligible:
        margin = float(row["rrf_margin"])
        margin_band = "tight" if margin <= low else "wide" if margin >= high else "middle"
        channel = "agree" if row["dense_lexical_agree"] else "disagree"
        row["boundary_stratum"] = f"{margin_band}_{channel}"

    role_counts = {
        "verifier_dev": args.dev_count,
        "verifier_train": args.train_count,
        "permanent_blind_audit": args.audit_count,
    }
    selected = balanced_rows(eligible, sum(role_counts.values()), args.seed)
    roles = assign_roles(selected, role_counts, args.seed)

    output_root.mkdir(parents=True, exist_ok=False)
    output_paths: dict[str, Path] = {}
    for role, rows in roles.items():
        path = output_root / f"{role}.items.jsonl"
        output_paths[role] = path
        write_jsonl(
            path,
            (
                {
                    **row,
                    "verifier_expansion_role": role,
                    "selection_seed": args.seed,
                    "selection_uses_adjudicator_outputs": False,
                    "selection_uses_labels": False,
                    "permanently_excluded_from_gradients": role == "permanent_blind_audit",
                }
                for row in rows
            ),
        )

    group_sets = {role: {row["source_group"] for row in rows} for role, rows in roles.items()}
    if any(group_sets[left] & group_sets[right] for left in roles for right in roles if left < right):
        raise AssertionError("verifier expansion roles overlap by source group")
    if set().union(*group_sets.values()) & excluded_groups:
        raise AssertionError("verifier expansion overlaps a prior labeled source group")

    report = {
        "schema_version": "silver-match-v3-verifier-expansion-freeze-v1",
        "status": "FROZEN_BEFORE_PREDICTIONS_OR_LABELS",
        "task": args.task,
        "selection_seed": args.seed,
        "selection_inputs": {
            "norms": {"path": str(paths["norms"]), "sha256": sha256_file(paths["norms"])},
            "retriever_candidates": {
                "path": str(paths["candidates"]),
                "sha256": sha256_file(paths["candidates"]),
            },
            "excluded_panels_identity_only": {
                str(path): sha256_file(path) for path in exclude_paths
            },
            "adjudicator_outputs": None,
            "labels_for_selected_rows": None,
        },
        "design": {
            "requested_counts": role_counts,
            "source_group_policy": "one selected row per group; all roles mutually disjoint",
            "boundary_strata": "retrieval RRF-margin tercile x dense/lexical winner agreement",
            "coverage_policy": "round-robin boundary strata and top retrieved metric IDs",
            "rrf_margin_cutpoints": [low, high],
        },
        "exclusions": {
            "labeled_uids": len(excluded_uids),
            "labeled_source_groups": len(excluded_groups),
            "eligible_rows": len(eligible),
            "eligible_source_groups": len({row["source_group"] for row in eligible}),
        },
        "roles": {
            role: {
                "path": str(output_paths[role]),
                "sha256": sha256_file(output_paths[role]),
                "count": len(rows),
                "source_groups": len(group_sets[role]),
                "boundary_strata": dict(sorted(Counter(row["boundary_stratum"] for row in rows).items())),
                "top_metric_coverage": len({row["top_metric_id"] for row in rows}),
            }
            for role, rows in roles.items()
        },
        "usage_contract": {
            "may_select_verifier_prompt": ["verifier_dev"],
            "may_mutate_verifier_prompt": ["verifier_train"],
            "blind_audit_role_hidden_until_final": "permanent_blind_audit",
            "may_reopen_retriever_selection": False,
            "may_reopen_adjudicator_selection": False,
            "may_use_as_retriever_or_adjudicator_test": False,
        },
    }
    report_path = output_root / "FREEZE.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "freeze_sha256": sha256_file(report_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
