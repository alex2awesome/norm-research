#!/usr/bin/env python3
"""Build a strict three-pass MATCH pool and a truth-hidden stratified audit pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate UIDs: {path}")
    return output


def _decision_key(row: dict[str, Any]) -> tuple[str, str | None]:
    decision = str(row.get("decision") or "")
    metric_id = str(row["metric_id"]) if decision == "MATCH" else None
    return decision, metric_id


def _stratified_sample(
    rows: list[dict[str, Any]], *, audit_size: int, seed: int
) -> list[dict[str, Any]]:
    """Balance boundary strata, then round-robin metric leaves within each stratum."""
    if audit_size > len(rows):
        raise ValueError(
            f"audit size {audit_size} exceeds strict consensus pool {len(rows)}"
        )
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row.get("boundary_stratum") or "unstratified")].append(row)
    strata = sorted(by_stratum)
    quotas = {stratum: audit_size // len(strata) for stratum in strata}
    for stratum in sorted(
        strata,
        key=lambda value: (_key(seed, "quota", value), value),
    )[: audit_size % len(strata)]:
        quotas[stratum] += 1

    # Saturated strata return capacity to the others deterministically.
    while True:
        deficit = sum(max(0, quotas[s] - len(by_stratum[s])) for s in strata)
        if not deficit:
            break
        for stratum in strata:
            quotas[stratum] = min(quotas[stratum], len(by_stratum[stratum]))
        recipients = [s for s in strata if quotas[s] < len(by_stratum[s])]
        if not recipients:
            raise ValueError("could not allocate requested stratified audit size")
        for index in range(deficit):
            available = [s for s in recipients if quotas[s] < len(by_stratum[s])]
            chosen = sorted(
                available,
                key=lambda value: (
                    quotas[value] / len(by_stratum[value]),
                    _key(seed + index, "redistribute", value),
                    value,
                ),
            )[0]
            quotas[chosen] += 1

    selected: list[dict[str, Any]] = []
    for stratum in strata:
        by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_stratum[stratum]:
            by_metric[str(row["metric_id"])].append(row)
        for metric_id, values in by_metric.items():
            values.sort(
                key=lambda row: (
                    _key(seed, f"row:{stratum}:{metric_id}", str(row["norm_uid"])),
                    row["norm_uid"],
                )
            )
        metric_ids = sorted(
            by_metric,
            key=lambda metric_id: (
                _key(seed, f"metric:{stratum}", metric_id),
                metric_id,
            ),
        )
        stratum_selected: list[dict[str, Any]] = []
        depth = 0
        while len(stratum_selected) < quotas[stratum]:
            added = False
            for metric_id in metric_ids:
                values = by_metric[metric_id]
                if depth < len(values):
                    stratum_selected.append(values[depth])
                    added = True
                    if len(stratum_selected) == quotas[stratum]:
                        break
            if not added:
                raise AssertionError("stratified sampling exhausted unexpectedly")
            depth += 1
        selected.extend(stratum_selected)
    if (
        len(selected) != audit_size
        or len({row["norm_uid"] for row in selected}) != audit_size
    ):
        raise AssertionError("stratified audit selection is not unique and complete")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--third", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--audit-size", type=int, default=60)
    parser.add_argument("--min-high-confidence-passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--chunk-size", type=int, default=20)
    args = parser.parse_args()
    if args.audit_size < 60:
        parser.error("--audit-size must be at least 60")
    if not 0 <= args.min_high_confidence_passes <= 3:
        parser.error("--min-high-confidence-passes must be in [0, 3]")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    pack_root = Path(args.pack_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite three-pass output: {output_root}")
    paths = {
        "first": Path(args.first).resolve(),
        "second": Path(args.second).resolve(),
        "third": Path(args.third).resolve(),
    }
    validation_path = pack_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    source_items_path, source_bank_path = (
        pack_root / "items.jsonl",
        pack_root / "bank.json",
    )
    if sha256_file(source_items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("source items hash mismatch")
    if sha256_file(source_bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError("source bank hash mismatch")
    items = list(read_jsonl(source_items_path))
    item_by_uid = {str(row["norm_uid"]): row for row in items}
    if len(item_by_uid) != len(items):
        raise ValueError("source pack has duplicate UIDs")
    labels = {name: _index(path) for name, path in paths.items()}
    expected_uids = set(item_by_uid)
    for name, index in labels.items():
        if set(index) != expected_uids:
            raise ValueError(f"{name} does not cover the source pack exactly")

    task = str(validation["task"])
    bank_hash = str(validation["bank_source_sha256"])
    decisions: Counter[str] = Counter()
    strict: list[dict[str, Any]] = []
    for uid in sorted(expected_uids):
        pass_rows = [labels[name][uid] for name in ("first", "second", "third")]
        if any(row.get("task") != task for row in pass_rows):
            raise ValueError(f"task mismatch: {uid}")
        if any(row.get("current_bank_source_sha256") != bank_hash for row in pass_rows):
            raise ValueError(f"bank mismatch: {uid}")
        item = item_by_uid[uid]
        for field in ("corpus", "row", "split_group", "split"):
            if any(row.get(field) != item.get(field) for row in pass_rows):
                raise ValueError(f"source provenance mismatch for {field}: {uid}")
        keys = [_decision_key(row) for row in pass_rows]
        decisions[f"pattern:{'|'.join(key[0] for key in keys)}"] += 1
        if len(set(keys)) != 1:
            decisions["three_way_disagreement"] += 1
            continue
        decisions[f"three_way_exact:{keys[0][0]}"] += 1
        if keys[0][0] != "MATCH":
            decisions["three_way_exact_abstention"] += 1
            continue
        high_count = sum(row.get("confidence") == "high" for row in pass_rows)
        decisions[f"three_way_match_high_count:{high_count}"] += 1
        if high_count < args.min_high_confidence_passes:
            decisions["three_way_match_below_confidence_policy"] += 1
            continue
        first = pass_rows[0]
        strict.append(
            {
                **first,
                "boundary_stratum": item.get("boundary_stratum"),
                "label_source": "independent_codex_three_pass_full_bank_consensus",
                "training_eligible": False,
                "training_blocked_pending_blind_audit": True,
                "consensus_policy": {
                    "passes": 3,
                    "exact_decision_and_metric": True,
                    "minimum_high_confidence_passes": args.min_high_confidence_passes,
                },
                "pass_confidences": [row.get("confidence") for row in pass_rows],
                "pass_reasons": [row.get("reason") for row in pass_rows],
                "pass_label_sha256": [
                    sha256_file(paths[name]) for name in ("first", "second", "third")
                ],
            }
        )
        decisions["strict_match_retained"] += 1

    audit = _stratified_sample(strict, audit_size=args.audit_size, seed=args.seed)
    stratum_population = Counter(
        str(row.get("boundary_stratum") or "unstratified") for row in strict
    )
    stratum_sample = Counter(
        str(row.get("boundary_stratum") or "unstratified") for row in audit
    )
    audit = [
        {
            **row,
            "audit_stratum": str(row.get("boundary_stratum") or "unstratified"),
            "audit_stratum_population_n": stratum_population[
                str(row.get("boundary_stratum") or "unstratified")
            ],
            "audit_stratum_sample_n": stratum_sample[
                str(row.get("boundary_stratum") or "unstratified")
            ],
            "audit_design_weight": stratum_population[
                str(row.get("boundary_stratum") or "unstratified")
            ]
            / stratum_sample[str(row.get("boundary_stratum") or "unstratified")],
        }
        for row in audit
    ]
    audit_uids = {str(row["norm_uid"]) for row in audit}
    train = [row for row in strict if str(row["norm_uid"]) not in audit_uids]
    audit_groups = {str(row["split_group"]) for row in audit}
    train_groups = {str(row["split_group"]) for row in train}
    if audit_groups & train_groups:
        raise ValueError("audit and training candidates overlap by source group")

    output_root.mkdir(parents=True, exist_ok=True)
    all_path = output_root / "three_pass_consensus.all.jsonl"
    audit_proposals_path = output_root / "blind_audit.proposals.hidden.jsonl"
    train_path = output_root / "training_candidates.pending_audit.jsonl"
    write_jsonl(all_path, strict)
    write_jsonl(audit_proposals_path, audit)
    write_jsonl(train_path, train)

    audit_pack = output_root / "blind_audit_pack"
    audit_pack.mkdir(parents=True, exist_ok=True)
    audit_items = [item_by_uid[uid] for uid in audit_uids]
    audit_items.sort(
        key=lambda row: (
            _key(args.seed, "audit-item", str(row["norm_uid"])),
            row["norm_uid"],
        )
    )
    bank = json.loads(source_bank_path.read_text(encoding="utf-8"))
    metrics = list(bank["metrics"])
    metrics.sort(
        key=lambda row: (
            _key(args.seed, "audit-metric", str(row["metric_id"])),
            row["metric_id"],
        )
    )
    audit_items_path, audit_bank_path = (
        audit_pack / "items.jsonl",
        audit_pack / "bank.json",
    )
    write_jsonl(audit_items_path, audit_items)
    audit_bank_path.write_text(
        json.dumps(
            {**bank, "metrics": metrics}, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    chunks = []
    for start in range(0, len(audit_items), args.chunk_size):
        chunk_path = (
            audit_pack / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        )
        write_jsonl(chunk_path, audit_items[start : start + args.chunk_size])
        chunks.append(chunk_path)
    audit_validation = {
        "schema_version": "silver-match-v3-three-pass-blind-audit-pack-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "count": len(audit_items),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "bank_metric_count": len(metrics),
        "bank_source_sha256": bank_hash,
        "truth_hidden": True,
        "consensus_metric_ids_hidden": True,
        "permanently_excluded_from_gradients": True,
        "source_group_disjoint_from_training_candidates": True,
        "outputs": {
            "items": {
                "path": str(audit_items_path),
                "sha256": sha256_file(audit_items_path),
            },
            "bank": {
                "path": str(audit_bank_path),
                "sha256": sha256_file(audit_bank_path),
            },
            "chunks": {str(path): sha256_file(path) for path in chunks},
        },
    }
    (audit_pack / "validation.json").write_text(
        json.dumps(audit_validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "silver-match-v3-three-pass-consensus-audit-split-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "source_count": len(items),
        "decision_counts": dict(sorted(decisions.items())),
        "strict_match_count": len(strict),
        "audit_count": len(audit),
        "training_candidate_count": len(train),
        "audit_by_boundary_stratum": dict(
            sorted(Counter(str(row.get("boundary_stratum")) for row in audit).items())
        ),
        "audit_unique_metric_count": len({str(row["metric_id"]) for row in audit}),
        "audit_source_groups": len(audit_groups),
        "training_source_groups": len(train_groups),
        "audit_permanently_excluded_from_gradients": True,
        "audit_sampling_design": (
            "balanced boundary-stratum allocation with deterministic metric-leaf diversity; "
            "inverse stratum sampling weights recorded for design-weighted scoring"
        ),
        "inputs": {
            "pack_validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
            **{
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
        "outputs": {
            "all_consensus": {"path": str(all_path), "sha256": sha256_file(all_path)},
            "audit_proposals": {
                "path": str(audit_proposals_path),
                "sha256": sha256_file(audit_proposals_path),
            },
            "training_candidates": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
            },
            "audit_pack_validation": {
                "path": str(audit_pack / "validation.json"),
                "sha256": sha256_file(audit_pack / "validation.json"),
            },
        },
    }
    report_path = output_root / "consensus.report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
