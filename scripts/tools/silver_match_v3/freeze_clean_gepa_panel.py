#!/usr/bin/env python3
"""Freeze an identity-only, leakage-safe GEPA panel before any labeling.

The selector deliberately cannot read truth, model predictions, metric IDs, or
downstream outcomes.  It chooses one canonical norm per source group from a
predeclared upstream split, after excluding every supplied prior panel by both
UID and source group.  The resulting identities can subsequently be rendered
for independent full-bank labeling and task-local adjudicator/verifier GEPA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for


def _stable(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{namespace}\x1f{value}".encode()).hexdigest()


def _resolve(path: str, anchor: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def _read_unique(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if not rows or "" in uids or len(uids) != len(set(uids)):
        raise ValueError(f"empty, missing, or duplicate norm_uid values: {path}")
    return rows


def _equivalent_group(value: str, canonical: str, task: str) -> bool:
    """Accept the historical ``task\x1fcorpus\x1fkind\x1fid`` namespace safely."""
    if value == canonical:
        return True
    parts = value.split("\x1f")
    return len(parts) >= 4 and parts[0] == task and ":".join(parts[1:]) == canonical


def _allocate(
    by_corpus: dict[str, list[tuple[str, dict[str, Any]]]],
    *,
    count: int,
    min_per_corpus: int,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    if count < 1 or min_per_corpus < 0:
        raise ValueError("count must be positive and min-per-corpus nonnegative")
    if sum(len(values) for values in by_corpus.values()) < count:
        raise ValueError("not enough eligible source groups for requested panel")
    selected: list[tuple[str, dict[str, Any]]] = []
    remaining: list[tuple[str, dict[str, Any]]] = []
    for corpus, values in sorted(by_corpus.items()):
        ordered = sorted(
            values,
            key=lambda value: (_stable(seed, f"group:{corpus}", value[0]), value[0]),
        )
        take = min_per_corpus
        if len(ordered) < take:
            raise ValueError(
                f"corpus {corpus} has {len(ordered)} eligible groups; "
                f"cannot satisfy --min-per-corpus {take}"
            )
        selected.extend(ordered[:take])
        remaining.extend(ordered[take:])
    if len(selected) > count:
        raise ValueError("count is smaller than the cross-corpus minimum allocation")
    remaining.sort(
        key=lambda value: (_stable(seed, "global-group", value[0]), value[0])
    )
    selected.extend(remaining[: count - len(selected)])
    return sorted(
        selected,
        key=lambda value: (_stable(seed, "output", value[0]), value[0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--role",
        choices=("optimize", "select", "adjudicator_dev", "verifier_dev", "blind_audit"),
        required=True,
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--min-per-corpus", type=int, default=0)
    parser.add_argument(
        "--required-upstream-split",
        choices=("train", "dev", "test"),
        default="train",
    )
    parser.add_argument(
        "--eligible-reference",
        help=(
            "optional identity/candidate JSONL limiting selectable UIDs; only norm_uid "
            "and source_group are read"
        ),
    )
    parser.add_argument(
        "--upstream-role-reference",
        help=(
            "optional authoritative UID-to-role JSONL (for example audited teacher "
            "labels); when supplied, its role field replaces the older calibration "
            "hash split for eligibility"
        ),
    )
    parser.add_argument("--upstream-role-field", default="split")
    parser.add_argument("--exclude-panel", action="append", default=[])
    parser.add_argument(
        "--exclude-uid-file",
        action="append",
        default=[],
        help="newline-delimited canonical norm UIDs to exclude by UID and source_group",
    )
    args = parser.parse_args()
    if not args.exclude_panel and not args.exclude_uid_file:
        parser.error("at least one --exclude-panel or --exclude-uid-file is required")

    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    norms: dict[str, dict[str, Any]] = {}
    task_corpora: set[str] = set()
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != args.task:
            continue
        task_corpora.add(str(corpus))
        path = _resolve(str(meta["path"]), manifest_path)
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in norms:
                raise ValueError(f"missing/duplicate canonical norm UID: {uid!r}")
            if str(row.get("task")) != args.task or str(row.get("corpus")) != corpus:
                raise ValueError(f"canonical task/corpus mismatch for {uid}")
            norms[uid] = row
    if not norms:
        raise ValueError(f"manifest has no canonical norms for task {args.task}")

    eligible_path = Path(args.eligible_reference).resolve() if args.eligible_reference else None
    eligible_uids = set(norms)
    eligible_reference_legacy_group_rows = 0
    if eligible_path is not None:
        eligible_rows = _read_unique(eligible_path)
        eligible_uids = {str(row["norm_uid"]) for row in eligible_rows}
        unknown = sorted(eligible_uids - set(norms))
        if unknown:
            raise ValueError(f"eligible reference contains noncanonical task UIDs: {unknown[:3]}")
        for row in eligible_rows:
            uid = str(row["norm_uid"])
            group = split_group_for(norms[uid])
            supplied = {
                str(row.get(field))
                for field in ("source_group", "split_group", "gepa_split_group")
                if row.get(field)
            }
            if supplied and not any(
                _equivalent_group(value, group, args.task) for value in supplied
            ):
                raise ValueError(f"eligible source_group mismatch for {uid}")
            if supplied and group not in supplied:
                eligible_reference_legacy_group_rows += 1

    upstream_role_path = (
        Path(args.upstream_role_reference).resolve()
        if args.upstream_role_reference
        else None
    )
    role_by_uid: dict[str, str] = {}
    role_by_group: dict[str, str] = {}
    role_counts: Counter[str] = Counter()
    if upstream_role_path is not None:
        role_rows = _read_unique(upstream_role_path)
        for row in role_rows:
            if row.get("task") != args.task:
                raise ValueError("upstream role reference contains another task")
            uid = str(row["norm_uid"])
            if uid not in norms:
                raise ValueError(f"upstream role reference has noncanonical UID: {uid}")
            role = str(row.get(args.upstream_role_field) or "")
            if role not in {"train", "dev", "test"}:
                raise ValueError(f"missing/invalid upstream role for {uid}: {role!r}")
            group = split_group_for(norms[uid])
            prior = role_by_group.setdefault(group, role)
            if prior != role:
                raise ValueError(
                    f"upstream role reference splits canonical source group {group}: "
                    f"{prior}/{role}"
                )
            role_by_uid[uid] = role
            role_counts[role] += 1
        if not eligible_uids <= set(role_by_uid):
            missing = sorted(eligible_uids - set(role_by_uid))
            raise ValueError(
                f"upstream role reference does not cover eligible UIDs: {missing[:3]}"
            )

    exclusion_paths = [Path(value).resolve() for value in args.exclude_panel]
    exclusion_uid_paths = [Path(value).resolve() for value in args.exclude_uid_file]
    excluded_uids: set[str] = set()
    excluded_groups: set[str] = set()
    exclusion_report: dict[str, Any] = {}
    for path in exclusion_paths:
        rows = _read_unique(path)
        panel_uids: set[str] = set()
        panel_groups: set[str] = set()
        for row in rows:
            uid = str(row["norm_uid"])
            supplied = str(
                row.get("source_group")
                or row.get("split_group")
                or row.get("gepa_split_group")
                or ""
            )
            canonical = split_group_for(norms[uid]) if uid in norms else ""
            if supplied and canonical and supplied != canonical:
                raise ValueError(f"excluded source_group mismatch for {uid} in {path}")
            group = canonical or supplied
            if not group:
                raise ValueError(f"cannot resolve excluded source group for {uid} in {path}")
            panel_uids.add(uid)
            panel_groups.add(group)
        excluded_uids.update(panel_uids)
        excluded_groups.update(panel_groups)
        exclusion_report[str(path)] = {
            "format": "jsonl",
            "sha256": sha256_file(path),
            "uids": len(panel_uids),
            "source_groups": len(panel_groups),
        }
    for path in exclusion_uid_paths:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"empty or duplicate newline-delimited UIDs: {path}")
        unknown = sorted(set(values) - set(norms))
        if unknown:
            raise ValueError(f"UID exclusion contains noncanonical task UIDs: {unknown[:3]}")
        panel_uids = set(values)
        panel_groups = {split_group_for(norms[uid]) for uid in values}
        excluded_uids.update(panel_uids)
        excluded_groups.update(panel_groups)
        exclusion_report[str(path)] = {
            "format": "newline_delimited_uids",
            "sha256": sha256_file(path),
            "uids": len(panel_uids),
            "source_groups": len(panel_groups),
        }

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for uid in eligible_uids:
        norm = norms[uid]
        group = split_group_for(norm)
        upstream_role = (
            role_by_uid[uid] if upstream_role_path is not None else split_for(group)
        )
        if upstream_role != args.required_upstream_split:
            continue
        if uid in excluded_uids or group in excluded_groups:
            continue
        by_group[group].append(norm)
    by_corpus: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for group, values in by_group.items():
        chosen = min(
            values,
            key=lambda row: (
                _stable(args.seed, "uid-within-group", str(row["norm_uid"])),
                str(row["norm_uid"]),
            ),
        )
        by_corpus[str(chosen["corpus"])].append((group, chosen))
    # Preserve empty task corpora so --min-per-corpus is a real all-corpus
    # requirement rather than silently applying only where eligible rows remain.
    for corpus in task_corpora:
        by_corpus.setdefault(corpus, [])
    chosen = _allocate(
        by_corpus,
        count=args.count,
        min_per_corpus=args.min_per_corpus,
        seed=args.seed,
    )

    rows = [
        {
            "schema_version": "silver-match-v3-clean-gepa-panel-identity-v1",
            "norm_uid": str(norm["norm_uid"]),
            "task": args.task,
            "corpus": str(norm["corpus"]),
            "source_group": group,
            "upstream_split": args.required_upstream_split,
            "gepa_role": args.role,
            "permanently_excluded_from_retriever_gradients": True,
            "permanently_excluded_from_mi_and_outcome_estimation": True,
        }
        for group, norm in chosen
    ]
    selected_uids = {str(row["norm_uid"]) for row in rows}
    selected_groups = {str(row["source_group"]) for row in rows}
    if selected_uids & excluded_uids or selected_groups & excluded_groups:
        raise AssertionError("selected panel overlaps an exclusion")

    output_root.mkdir(parents=True, exist_ok=False)
    identities = output_root / "identities.jsonl"
    write_jsonl(identities, rows)
    freeze = {
        "schema_version": "silver-match-v3-clean-gepa-panel-freeze-v1",
        "status": "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES",
        "task": args.task,
        "role": args.role,
        "selection_seed": args.seed,
        "required_upstream_split": args.required_upstream_split,
        "requested_count": args.count,
        "selected_count": len(rows),
        "selected_source_groups": len(selected_groups),
        "selected_by_corpus": dict(sorted(Counter(row["corpus"] for row in rows).items())),
        "min_per_corpus": args.min_per_corpus,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "eligible_reference": (
                {
                    "path": str(eligible_path),
                    "sha256": sha256_file(eligible_path),
                    "canonical_source_group_recomputed": True,
                    "legacy_namespaced_source_group_rows": (
                        eligible_reference_legacy_group_rows
                    ),
                }
                if eligible_path is not None
                else None
            ),
            "upstream_role_reference": (
                {
                    "path": str(upstream_role_path),
                    "sha256": sha256_file(upstream_role_path),
                    "field": args.upstream_role_field,
                    "role_counts": dict(sorted(role_counts.items())),
                    "authoritative": True,
                }
                if upstream_role_path is not None
                else {
                    "path": None,
                    "field": None,
                    "authoritative": False,
                    "fallback": "make_calibration.split_for(canonical_source_group)",
                }
            ),
            "exclusions": exclusion_report,
        },
        "exclusion_union": {
            "uids": len(excluded_uids),
            "source_groups": len(excluded_groups),
            "selected_uid_overlap": 0,
            "selected_source_group_overlap": 0,
        },
        "outputs": {
            "identities": {
                "path": str(identities),
                "sha256": sha256_file(identities),
            }
        },
        "content_contract": {
            "truth_fields_read": False,
            "model_prediction_fields_read": False,
            "metric_ids_read": False,
            "downstream_outcomes_read": False,
            "selection_uses_identity_and_source_group_only": True,
        },
    }
    freeze_path = output_root / "FREEZE.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {**freeze, "freeze_sha256": sha256_file(freeze_path)}, sort_keys=True
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
