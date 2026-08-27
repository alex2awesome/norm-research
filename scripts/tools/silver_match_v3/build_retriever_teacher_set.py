#!/usr/bin/env python3
"""Build immutable, source-disjoint retriever teachers and frozen evaluations.

Trusted high-volume model labels are useful distillation data, but they must
not leak a document represented in an independently labeled dev/test panel.
This utility blocks every such source group, keeps only exact MATCH labels as
positive training supervision, lets independent human labels override a
trusted-model disagreement, and writes a complete provenance audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .train_nemotron_lora import source_group_key


def _resolve_artifact(path: str | Path, manifest_path: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else manifest_path.parent / value


def load_task_norms(manifest_path: Path, task: str) -> tuple[dict[str, dict], str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if task not in manifest.get("banks", {}):
        raise KeyError(f"task absent from manifest: {task}")
    bank_sha = normalize_space(manifest["banks"][task].get("source_sha256"))
    if not bank_sha:
        raise ValueError(f"bank source hash missing for {task}")
    norms: dict[str, dict] = {}
    for corpus, meta in sorted(manifest.get("corpora", {}).items()):
        if meta.get("task") != task:
            continue
        for row in read_jsonl(_resolve_artifact(meta["path"], manifest_path)):
            uid = normalize_space(row.get("norm_uid"))
            if not uid or uid in norms:
                raise ValueError(f"missing/duplicate canonical UID: {uid!r}")
            if row.get("task") != task or row.get("corpus") != corpus:
                raise ValueError(f"canonical routing mismatch for {uid}")
            norms[uid] = row
    if not norms:
        raise ValueError(f"no canonical norms for {task}")
    return norms, bank_sha


def _canonicalize_teacher(
    row: Mapping[str, Any], norm: Mapping[str, Any], bank_sha: str
) -> dict[str, Any]:
    output = dict(row)
    output.update(
        {
            "norm_uid": norm["norm_uid"],
            "corpus": norm["corpus"],
            "task": norm["task"],
            "row": norm["row"],
            "decision": "MATCH",
            "current_bank_source_sha256": bank_sha,
        }
    )
    return output


def _is_weak_forced(row: Mapping[str, Any]) -> bool:
    return (
        normalize_space(row.get("supervision_strength"))
        in {"weak_forced_positive", "weak_forced_top3"}
        or normalize_space(row.get("label_source")) == "sonnet_forced_top3"
    )


def _acceptable_ids(row: Mapping[str, Any]) -> set[str]:
    values = {normalize_space(row.get("metric_id"))}
    for key in ("acceptable_metric_ids", "equivalent_metric_ids", "metric_ids"):
        raw = row.get(key)
        if isinstance(raw, str):
            values.add(normalize_space(raw))
        elif isinstance(raw, (list, tuple)):
            values.update(normalize_space(value) for value in raw)
    values.discard("")
    return values


def build_teacher_set(
    *,
    manifest_path: Path,
    task: str,
    trusted_paths: Sequence[Path],
    human_paths: Sequence[Path],
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    norms, bank_sha = load_task_norms(manifest_path, task)
    external: list[dict] = []
    human_train: list[tuple[str, dict]] = []
    seen_panel_uids: set[str] = set()
    panel_counts: Counter[str] = Counter()
    frozen_groups: set[str] = set()

    for path in human_paths:
        for source_row in read_jsonl(path):
            if source_row.get("task") != task:
                panel_counts["other_task"] += 1
                continue
            uid = normalize_space(source_row.get("norm_uid"))
            if uid not in norms:
                raise ValueError(f"human-panel UID absent from canonical manifest: {uid}")
            if uid in seen_panel_uids:
                raise ValueError(f"duplicate human-panel UID: {uid}")
            seen_panel_uids.add(uid)
            norm = norms[uid]
            split = normalize_space(source_row.get("split"))
            if split not in {"train", "dev", "test"}:
                raise ValueError(f"invalid human-panel split for {uid}: {split!r}")
            supplied_hash = normalize_space(source_row.get("current_bank_source_sha256"))
            if supplied_hash != bank_sha:
                raise ValueError(
                    f"human-panel bank hash mismatch for {uid}: {supplied_hash} != {bank_sha}"
                )
            row = dict(source_row)
            row.update(
                {
                    "norm_uid": uid,
                    "corpus": norm["corpus"],
                    "task": task,
                    "row": norm["row"],
                    "source_group": source_group_key(norm),
                    "human_panel": str(path),
                }
            )
            panel_counts[f"{split}:{row.get('decision')}"] += 1
            if split in {"dev", "test"}:
                external.append(row)
                frozen_groups.add(source_group_key(norm))
            elif row.get("decision") == "MATCH":
                human_train.append((str(path), _canonicalize_teacher(row, norm, bank_sha)))

    teacher_by_uid: dict[str, tuple[int, str, dict]] = {}
    audit_counts: Counter[str] = Counter()
    trusted_by_uid: dict[str, list[tuple[str, dict, dict]]] = defaultdict(list)

    # Trusted-model labels have lower precedence than an independent human
    # label for the same norm, but remain valuable for every unblocked source.
    for path in trusted_paths:
        for source_row in read_jsonl(path):
            if source_row.get("task") != task or source_row.get("decision") != "MATCH":
                audit_counts["trusted_nonmatch_or_other_task"] += 1
                continue
            uid = normalize_space(source_row.get("norm_uid"))
            norm = norms.get(uid)
            if norm is None:
                audit_counts["trusted_uid_absent"] += 1
                continue
            supplied_hash = normalize_space(source_row.get("current_bank_source_sha256"))
            if supplied_hash != bank_sha:
                audit_counts["trusted_bank_hash_mismatch"] += 1
                continue
            if source_group_key(norm) in frozen_groups:
                audit_counts["trusted_frozen_source_blocked"] += 1
                continue
            trusted_by_uid[uid].append((str(path), dict(source_row), norm))
            audit_counts["trusted_match_rows_admitted"] += 1

    # Forced top-3 labels are a *single weak supervision event*: rank 1 is the
    # positive and all returned metrics are acceptable alternatives that must
    # not be mined as negatives.  Collapsing by last row would silently train
    # on rank 3, while emitting all rows would let downstream merge ordering
    # decide the positive.  Canonicalize the event explicitly here.
    for uid, group in sorted(trusted_by_uid.items()):
        strong = [item for item in group if not _is_weak_forced(item[1])]
        selected = strong or group
        if strong:
            metric_ids = {
                normalize_space(row.get("metric_id")) for _, row, _ in selected
            }
            metric_ids.discard("")
            if len(metric_ids) != 1:
                audit_counts["trusted_strong_conflict_uids_excluded"] += 1
                continue
            source, row, norm = sorted(
                selected,
                key=lambda item: (
                    normalize_space(item[1].get("label_source")), item[0]
                ),
            )[0]
            rendered = _canonicalize_teacher(row, norm, bank_sha)
            acceptable = set()
            for _, candidate, _ in selected:
                acceptable.update(_acceptable_ids(candidate))
            if acceptable:
                rendered["acceptable_metric_ids"] = sorted(acceptable)
            audit_counts["trusted_strong_uids_selected"] += 1
        else:
            ranked = sorted(
                selected,
                key=lambda item: (
                    int(item[1].get("forced_rank") or 10**9),
                    normalize_space(item[1].get("metric_id")),
                    item[0],
                ),
            )
            source, row, norm = ranked[0]
            rendered = _canonicalize_teacher(row, norm, bank_sha)
            acceptable = set()
            for _, candidate, _ in ranked:
                acceptable.update(_acceptable_ids(candidate))
            rendered["acceptable_metric_ids"] = sorted(acceptable)
            rendered["supervision_strength"] = "weak_forced_positive"
            rendered["forced_group_rows"] = len(ranked)
            audit_counts["trusted_weak_forced_uids_selected"] += 1
            audit_counts["trusted_weak_forced_alternative_rows_merged"] += len(ranked) - 1
        teacher_by_uid[uid] = (0, source, rendered)
        audit_counts["trusted_selected_before_override"] += 1

    for path, row in human_train:
        uid = row["norm_uid"]
        norm = norms[uid]
        if source_group_key(norm) in frozen_groups:
            raise ValueError(f"human train row shares frozen source group: {uid}")
        existing = teacher_by_uid.get(uid)
        if existing is not None:
            if existing[2].get("metric_id") == row.get("metric_id"):
                audit_counts["human_trusted_agreement"] += 1
            else:
                audit_counts["human_overrode_trusted_conflict"] += 1
        teacher_by_uid[uid] = (1, path, row)
        audit_counts["human_train_match_selected"] += 1

    teachers = [value[2] for _, value in sorted(teacher_by_uid.items())]
    teacher_groups = {source_group_key(norms[row["norm_uid"]]) for row in teachers}
    overlap = teacher_groups & frozen_groups
    if overlap:
        raise ValueError(f"teacher/frozen source-group leakage: {sorted(overlap)[:5]}")
    external.sort(key=lambda row: (row["split"], row["corpus"], row["norm_uid"]))
    split_groups = {
        split: {
            row["source_group"] for row in external if row.get("split") == split
        }
        for split in ("dev", "test")
    }
    external_overlap = split_groups["dev"] & split_groups["test"]
    if external_overlap:
        raise ValueError(f"external dev/test source leakage: {sorted(external_overlap)[:5]}")
    if not teachers or not any(row.get("decision") == "MATCH" for row in external):
        raise ValueError("teacher/external MATCH sets must both be non-empty")

    report = {
        "task": task,
        "canonical_norms": len(norms),
        "bank_source_sha256": bank_sha,
        "teachers": len(teachers),
        "teacher_source_groups": len(teacher_groups),
        "teacher_metrics": len({row.get("metric_id") for row in teachers}),
        "external": len(external),
        "external_by_split_decision": dict(
            sorted(Counter(f"{r['split']}:{r.get('decision')}" for r in external).items())
        ),
        "external_source_groups": {
            split: len(groups) for split, groups in split_groups.items()
        },
        "source_group_overlap": {
            "teacher_external": len(overlap),
            "dev_test": len(external_overlap),
        },
        "panel_counts": dict(sorted(panel_counts.items())),
        "selection_counts": dict(sorted(audit_counts.items())),
    }
    return teachers, external, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--trusted", action="append", default=[])
    parser.add_argument("--human-panel", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    trusted_paths = tuple(Path(path).resolve() for path in args.trusted)
    human_paths = tuple(Path(path).resolve() for path in args.human_panel)
    output = Path(args.output_root).resolve() / args.task
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty teacher directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    teachers, external, report = build_teacher_set(
        manifest_path=manifest_path,
        task=args.task,
        trusted_paths=trusted_paths,
        human_paths=human_paths,
    )
    teacher_path = output / "teacher_train.jsonl"
    external_path = output / "external_dev_test.jsonl"
    write_jsonl(teacher_path, teachers)
    write_jsonl(external_path, external)
    report.update(
        {
            "inputs": {
                "manifest": {str(manifest_path): sha256_file(manifest_path)},
                "trusted": {str(path): sha256_file(path) for path in trusted_paths},
                "human_panels": {str(path): sha256_file(path) for path in human_paths},
            },
            "outputs": {
                str(teacher_path): sha256_file(teacher_path),
                str(external_path): sha256_file(external_path),
            },
        }
    )
    report_path = output / "teacher_build_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
