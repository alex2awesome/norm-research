#!/usr/bin/env python3
"""Freeze a fresh train-only teacher pack enriched for audited bank boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for
from .prepare_independent_teacher_pack import resolve


PEER_PATTERNS = [
    ("methods_transparency_rigor", r"method|experiment|analysis|\bdata\b|implement|detail|reproduc|clarif"),
    ("evidence_theory_correctness", r"support|evidence|theor|claim|correct|sound|proof|deriv"),
    ("novelty_prior_work", r"novel|incremental|prior work|related work|reference|citation|baseline|compar"),
    ("generalization_sampling", r"dataset|generaliz|generalis|real.?world|sampl|represent|external valid|scope"),
    ("question_significance_motivation", r"interesting|challenging|important|problem|question|motivat|signific"),
    ("presentation_conclusion_clarity", r"conclusion|clear|clarity|writing|readab|figure|table|organization"),
    ("generic_performance_no_fit", r"performance|good|better|insight|concern|effective|improv|robust"),
]

LEGAL_PATTERNS = [
    ("bare_holding_or_substantive_rule", r"lawful|unlawful|legal obligation|as a matter of law|must prove|burden|preponderance|claim|statut|regulat"),
    ("legal_analysis_application", r"reason|analysis|apply|application|erroneous|persuad|convinc|conclusion"),
    ("accuracy_evidence_record", r"true|accur|evidence|record|support|basis|fact|misstat|ground"),
    ("authority_precedent", r"precedent|appellate|decision|authority|citation|cite|case law|supreme court"),
    ("terminology_definition", r"mean|definition|terminolog|category|term|jargon|information"),
    ("forum_genre_jurisdiction", r"jurisdiction|forum|court|judge|docket|\buk\b|canada|form|brief|complaint"),
    ("issue_analogy_preservation", r"issue|standing|similar theor|analog|distinguish|waiv|sub.?issue|overlook"),
    ("candor_uncertainty_restraint", r"i don.?t know|uncertain|restraint|overstat|speculat|candid|honest"),
]


def peer_stratum(row: dict[str, Any]) -> str:
    # The extracted statement and aspect identify the evaluated criterion.  Long
    # review context mentions methods/experiments almost everywhere and would
    # otherwise collapse most rows into the methods stratum.
    primary = normalize_space(
        " ".join(str(row.get(key) or "") for key in ("aspect", "norm"))
    ).lower()
    for name, pattern in PEER_PATTERNS:
        if re.search(pattern, primary):
            return name
    context = normalize_space(str(row.get("context") or "")).lower()[:3000]
    for name, pattern in PEER_PATTERNS:
        if re.search(pattern, context):
            return name
    return "other"


def legal_stratum(row: dict[str, Any]) -> str:
    primary = normalize_space(
        " ".join(str(row.get(key) or "") for key in ("aspect", "norm"))
    ).lower()
    for name, pattern in LEGAL_PATTERNS:
        if re.search(pattern, primary):
            return name
    context = normalize_space(str(row.get("context") or "")).lower()[:3000]
    for name, pattern in LEGAL_PATTERNS:
        if re.search(pattern, context):
            return name
    return "other"


def deterministic_order(rows: Sequence[dict[str, Any]], namespace: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{namespace}\0{row['norm_uid']}".encode()).hexdigest(),
            row["norm_uid"],
        ),
    )


def round_robin_strata(rows: Sequence[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row["boundary_stratum"])].append(row)
    for name in by_stratum:
        by_stratum[name] = deterministic_order(by_stratum[name], f"boundary-enriched-v1:{name}")
    cursors = {name: 0 for name in by_stratum}
    selected = []
    while len(selected) < total:
        advanced = False
        for name in sorted(by_stratum):
            index = cursors[name]
            if index >= len(by_stratum[name]):
                continue
            selected.append(by_stratum[name][index])
            cursors[name] += 1
            advanced = True
            if len(selected) == total:
                break
        if not advanced:
            break
    if len(selected) != total:
        raise ValueError(f"requested {total}, selected only {len(selected)}")
    return deterministic_order(selected, "boundary-enriched-final-order-v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-meta")
    parser.add_argument("--external-label", action="append", default=[])
    parser.add_argument("--exclude-item", action="append", default=[])
    parser.add_argument("--total", type=int, default=600)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.task not in {"peer-review", "legal-outcome-prediction"}:
        raise ValueError("no audited boundary profile for this task")
    if args.total < 1 or args.chunk_size < 1:
        parser.error("--total and --chunk-size must be positive")

    manifest_path = Path(args.manifest).resolve()
    candidate_path = Path(args.candidates).resolve()
    candidate_meta_path = (
        Path(args.candidate_meta).resolve()
        if args.candidate_meta
        else candidate_path.with_suffix(candidate_path.suffix + ".meta.json")
    )
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite boundary pack: {output_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_meta = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
    if candidate_meta.get("output_sha256") != sha256_file(candidate_path):
        raise ValueError("candidate metadata hash mismatch")
    if candidate_meta.get("task") != args.task:
        raise ValueError("candidate metadata task mismatch")
    if (candidate_meta.get("input_hashes") or {}).get("manifest") != sha256_file(
        manifest_path
    ):
        raise ValueError("candidate slate manifest mismatch")

    bank_meta = manifest["banks"][args.task]
    bank_path = resolve(bank_meta["path"], manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_ids = [str(row["metric_id"]) for row in bank["metrics"]]
    bank_hash = str(bank_meta["source_sha256"])
    candidates = {}
    for row in read_jsonl(candidate_path):
        uid = str(row["norm_uid"])
        ids = [str(value["metric_id"]) for value in row.get("candidates") or []]
        if uid in candidates or ids != bank_ids:
            raise ValueError(f"candidate row is duplicate or not exact full bank: {uid}")
        candidates[uid] = row

    norms = {}
    for _, meta in manifest["corpora"].items():
        if meta.get("task") != args.task:
            continue
        for row in read_jsonl(resolve(meta["path"], manifest_path)):
            uid = str(row["norm_uid"])
            if uid in norms:
                raise ValueError(f"duplicate canonical norm UID: {uid}")
            norms[uid] = row
    if not set(candidates).issubset(norms):
        raise KeyError("candidate slate contains UIDs absent from canonical task norms")

    excluded_uids: set[str] = set()
    excluded_groups: set[str] = set()
    exclusion_paths = [
        *[Path(value).resolve() for value in args.external_label],
        *[Path(value).resolve() for value in args.exclude_item],
    ]
    for path in exclusion_paths:
        for row in read_jsonl(path):
            if row.get("task") != args.task:
                continue
            uid = str(row["norm_uid"])
            if uid not in norms:
                raise KeyError(f"exclusion UID absent from canonical norms: {uid}")
            excluded_uids.add(uid)
            excluded_groups.add(split_group_for(norms[uid]))

    eligible = []
    excluded = Counter()
    for uid in candidates:
        norm = norms[uid]
        group = split_group_for(norm)
        if split_for(group) != "train":
            excluded["not_train"] += 1
            continue
        if uid in excluded_uids or group in excluded_groups:
            excluded["prior_or_external_overlap"] += 1
            continue
        stratum = peer_stratum(norm) if args.task == "peer-review" else legal_stratum(norm)
        eligible.append(
            {
                **norm,
                "split_group": group,
                "split": "train",
                "boundary_stratum": stratum,
            }
        )
    selected = round_robin_strata(eligible, args.total)
    if len({row["split_group"] for row in selected}) != len(selected):
        raise ValueError("selected pack contains repeated source groups")

    output_root.mkdir(parents=True, exist_ok=True)
    items_path, candidates_path, bank_output = (
        output_root / "items.jsonl",
        output_root / "candidates.full-bank.jsonl",
        output_root / "bank.json",
    )
    write_jsonl(items_path, selected)
    write_jsonl(candidates_path, [candidates[row["norm_uid"]] for row in selected])
    bank_output.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    chunks = []
    for start in range(0, len(selected), args.chunk_size):
        path = output_root / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(path, selected[start : start + args.chunk_size])
        chunks.append(path)
    report = {
        "schema_version": "silver-match-v3-boundary-enriched-teacher-pack-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "count": len(selected),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "selected_by_stratum": dict(
            sorted(Counter(row["boundary_stratum"] for row in selected).items())
        ),
        "eligible_by_stratum": dict(
            sorted(Counter(row["boundary_stratum"] for row in eligible).items())
        ),
        "selected_source_groups": len({row["split_group"] for row in selected}),
        "train_split_count": len(selected),
        "bank_metric_count": len(bank_ids),
        "bank_source_sha256": bank_hash,
        "excluded": dict(sorted(excluded.items())),
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "candidate_slate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "candidate_meta": {"path": str(candidate_meta_path), "sha256": sha256_file(candidate_meta_path)},
            "exclusions": {str(path): sha256_file(path) for path in exclusion_paths},
        },
        "outputs": {
            "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            "candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
            "bank": {"path": str(bank_output), "sha256": sha256_file(bank_output)},
            "chunks": {str(path): sha256_file(path) for path in chunks},
        },
    }
    (output_root / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
