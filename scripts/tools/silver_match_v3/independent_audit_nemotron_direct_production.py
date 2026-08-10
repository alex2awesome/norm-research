#!/usr/bin/env python3
"""Independent streaming audit of a direct Nemotron production candidate file."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL at line {number}")
            yield value


def resolve(raw: str, anchor: Path) -> Path:
    value = Path(raw)
    return value if value.is_absolute() else (anchor.parent / value).resolve()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    candidate_path = Path(args.candidates).resolve()
    meta_path = candidate_path.with_suffix(candidate_path.suffix + ".meta.json")
    audit_path = Path(args.upstream_audit).resolve()
    selection_path = Path(args.selection).resolve()
    queue_path = Path(args.queue).resolve()
    run_record_path = Path(args.run_record).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_meta = manifest["corpora"][args.corpus]
    task = str(corpus_meta["task"])
    bank_meta = manifest["banks"][task]
    corpus_path = resolve(str(corpus_meta["path"]), manifest_path)
    bank_path = resolve(str(bank_meta["path"]), manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))["metrics"]
    bank_ids = [str(value["metric_id"]) for value in bank]
    bank_index = {metric_id: index for index, metric_id in enumerate(bank_ids)}
    bank_id_set = set(bank_ids)
    if len(bank_index) != len(bank_ids):
        raise ValueError("duplicate bank metric IDs")
    canonical = {
        str(value["norm_uid"]): int(value["row"]) for value in rows(corpus_path)
    }
    if len(canonical) != int(corpus_meta["count"]):
        raise ValueError("canonical UID/count mismatch")

    seen: set[str] = set()
    depth = Counter()
    score_order_violations = 0
    tie_order_violations = 0
    nonfinite_scores = 0
    for value in rows(candidate_path):
        uid = str(value.get("norm_uid") or "")
        if not uid or uid in seen or uid not in canonical:
            raise ValueError(f"missing, duplicate, or foreign candidate UID: {uid!r}")
        if (
            value.get("task") != task
            or value.get("corpus") != args.corpus
            or int(value.get("row", -1)) != canonical[uid]
            or str(value.get("bank_source_sha256"))
            != str(bank_meta["source_sha256"])
        ):
            raise ValueError(f"routing/provenance mismatch: {uid}")
        candidates = value.get("candidates") or []
        depth[len(candidates)] += 1
        if len(candidates) != args.expected_k:
            raise ValueError(f"candidate depth mismatch: {uid}")
        ids = [str(item.get("metric_id") or "") for item in candidates]
        if len(set(ids)) != args.expected_k or not set(ids) <= bank_id_set:
            raise ValueError(f"duplicate or foreign metric ID: {uid}")
        if [int(item.get("rank", -1)) for item in candidates] != list(
            range(1, args.expected_k + 1)
        ):
            raise ValueError(f"non-contiguous ranks: {uid}")
        scores = [float(item["dense_score"]) for item in candidates]
        nonfinite_scores += sum(not math.isfinite(score) for score in scores)
        score_order_violations += sum(
            scores[index] < scores[index + 1]
            for index in range(len(scores) - 1)
        )
        tie_order_violations += sum(
            scores[index] == scores[index + 1]
            and bank_index[ids[index]] > bank_index[ids[index + 1]]
            for index in range(len(scores) - 1)
        )
        seen.add(uid)
    if seen != set(canonical):
        raise ValueError(f"candidate coverage mismatch: {len(seen)}/{len(canonical)}")
    if nonfinite_scores or score_order_violations or tie_order_violations:
        raise ValueError("dense scores or deterministic score ordering are invalid")

    candidate_sha = sha256_file(candidate_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    upstream = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
    adapter_path = Path(str((selection.get("chosen") or {}).get("adapter", {}).get("path") or ""))
    adapter_hashes = {
        child.name: sha256_file(child)
        for child in sorted(adapter_path.iterdir())
        if child.is_file()
    }
    failures = []
    if (
        meta.get("output_sha256") != candidate_sha
        or int(meta.get("observed_count", -1)) != len(canonical)
        or int(meta.get("output_k", -1)) != args.expected_k
        or meta.get("external_test_consumed") is not False
        or meta.get("external_labels_opened") is not False
    ):
        failures.append("candidate metadata")
    if (
        upstream.get("complete") is not True
        or int(upstream.get("observed_count", -1)) != len(canonical)
        or int(upstream.get("materialized_k", -1)) != args.expected_k
    ):
        failures.append("upstream candidate audit")
    if (
        selection.get("status") != "SELECTED_FOR_PRODUCTION_RETRIEVAL"
        or selection.get("frozen_external_test_consumed") is not False
        or (selection.get("chosen") or {}).get("adapter", {}).get("files")
        != adapter_hashes
    ):
        failures.append("promoted selection or adapter")
    if (
        queue.get("status") != "FROZEN_READY_NOT_LAUNCHED"
        or int(queue.get("expected_rows", -1)) != len(canonical)
        or int(queue.get("expected_k", -1)) != args.expected_k
        or (queue.get("safety") or {}).get("external_test_consumed") is not False
    ):
        failures.append("frozen queue")
    if (
        run_record.get("status") != "COMPLETED_EXACT_K50"
        or run_record.get("candidate_sha256") != candidate_sha
        or run_record.get("queue_sha256") != sha256_file(queue_path)
        or run_record.get("audit_sha256") != sha256_file(audit_path)
        or run_record.get("external_test_consumed") is not False
        or run_record.get("external_labels_opened") is not False
    ):
        failures.append("sealed run record")
    if failures:
        raise ValueError(f"cross-artifact provenance failed: {failures}")
    return {
        "schema_version": "silver-match-v3-independent-nemotron-production-audit-v1",
        "status": "PASS_EXACT_K50_INDEPENDENT_STREAMING_AUDIT",
        "task": task,
        "corpus": args.corpus,
        "expected_rows": len(canonical),
        "observed_rows": len(seen),
        "expected_k": args.expected_k,
        "candidate_depth_distribution": {
            str(key): value for key, value in sorted(depth.items())
        },
        "bank_metrics": len(bank_ids),
        "unique_norm_uids": len(seen),
        "duplicate_norm_uids": 0,
        "foreign_norm_uids": 0,
        "duplicate_candidate_metric_ids": 0,
        "foreign_candidate_metric_ids": 0,
        "nonfinite_dense_scores": nonfinite_scores,
        "score_order_violations": score_order_violations,
        "tie_order_violations": tie_order_violations,
        "external_labels_opened": False,
        "external_test_consumed": False,
        "artifacts": {
            "implementation": artifact(Path(__file__).resolve()),
            "manifest": artifact(manifest_path),
            "canonical_corpus": artifact(corpus_path),
            "bank": artifact(bank_path),
            "candidate": artifact(candidate_path),
            "candidate_meta": artifact(meta_path),
            "upstream_audit": artifact(audit_path),
            "selection": artifact(selection_path),
            "queue": artifact(queue_path),
            "run_record": artifact(run_record_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--upstream-audit", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--run-record", required=True)
    parser.add_argument("--expected-k", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output": artifact(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
