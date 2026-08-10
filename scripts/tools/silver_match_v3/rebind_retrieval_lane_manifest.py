#!/usr/bin/env python3
"""Rebind an immutable retrieval lane from a path-only manifest mirror.

The candidate bytes are never rewritten.  This utility is for a lane produced
against a host-local manifest whose scientific bank/corpus identities are the
same as the canonical manifest but whose artifact paths differ.  It verifies
both manifests, the relocated bank/corpus bytes, and every candidate row before
creating an append-only hard link plus canonical metadata.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

from .common import sha256_file


def _resolve(value: str, anchor: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row: {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row: {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_number}")
            yield row


def _bank_ids(path: Path, expected_count: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("metrics") or []
    metric_ids = [str(row.get("metric_id") or "") for row in rows]
    if (
        len(metric_ids) != expected_count
        or "" in metric_ids
        or len(set(metric_ids)) != len(metric_ids)
    ):
        raise ValueError("target bank has an invalid metric universe")
    return metric_ids


def _route(manifest: dict[str, Any], *, corpus: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    corpus_meta = (manifest.get("corpora") or {}).get(corpus)
    if not isinstance(corpus_meta, dict):
        raise KeyError(f"corpus absent from manifest: {corpus}")
    task = str(corpus_meta.get("task") or "")
    bank_meta = (manifest.get("banks") or {}).get(task)
    if not task or not isinstance(bank_meta, dict):
        raise KeyError(f"task bank absent from manifest: {task!r}")
    return task, corpus_meta, bank_meta


def _scientific_route(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in meta.items()
        if key not in {"path", "source_path", "source_paths"}
    }


def rebind_lane(
    *,
    source_candidate: Path,
    source_meta: Path,
    source_manifest: Path,
    target_manifest: Path,
    corpus: str,
    output_candidate: Path,
    runtime_fusion: Path | None = None,
    runtime_adapter: Path | None = None,
    source_artifact_inventory: Path | None = None,
    source_audit: Path | None = None,
    source_audit_sha256: str | None = None,
) -> dict[str, Any]:
    source_candidate = source_candidate.resolve()
    source_meta = source_meta.resolve()
    source_manifest = source_manifest.resolve()
    target_manifest = target_manifest.resolve()
    output_candidate = output_candidate.resolve()
    output_meta = output_candidate.with_suffix(output_candidate.suffix + ".meta.json")
    if output_candidate.exists() or output_meta.exists():
        raise FileExistsError(f"refusing to overwrite rebound lane: {output_candidate}")
    for path in (source_candidate, source_meta, source_manifest, target_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    meta = json.loads(source_meta.read_text(encoding="utf-8"))
    source_manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    target_manifest_payload = json.loads(target_manifest.read_text(encoding="utf-8"))
    source_manifest_sha = sha256_file(source_manifest)
    target_manifest_sha = sha256_file(target_manifest)
    candidate_sha = sha256_file(source_candidate)
    source_audit_ref = None
    if meta.get("manifest_sha256") != source_manifest_sha:
        if meta.get("manifest_sha256") or source_audit is None or not source_audit_sha256:
            raise ValueError("source lane is not bound to the supplied source manifest")
        source_audit = source_audit.resolve()
        if sha256_file(source_audit) != source_audit_sha256:
            raise ValueError("source audit hash differs from its frozen identity")
        audit = json.loads(source_audit.read_text(encoding="utf-8"))
        audit_inputs = audit.get("candidate_inputs") or {}
        audited = next(
            (
                value
                for path, value in audit_inputs.items()
                if Path(path).name == source_candidate.name
            ),
            None,
        )
        if (
            audit.get("complete") is not True
            or audit.get("manifest_sha256") != source_manifest_sha
            or audit.get("corpus") != corpus
            or audited is None
            or audited.get("sha256") != candidate_sha
            or audited.get("meta_sha256") != sha256_file(source_meta)
        ):
            raise ValueError("legacy source audit does not bind this lane and manifest")
        source_audit_ref = {
            "path": str(source_audit),
            "sha256": source_audit_sha256,
            "canonical_manifest_sha256": source_manifest_sha,
        }
    if meta.get("output_sha256") != candidate_sha:
        raise ValueError("source candidate hash differs from source metadata")
    if str(meta.get("corpus")) != corpus:
        raise ValueError("source candidate metadata has the wrong corpus")

    runtime_fusion_ref = None
    source_fusion_sha = str(meta.get("fusion_weights_sha256") or "")
    if source_fusion_sha:
        if runtime_fusion is None:
            source_fusion = Path(str(meta.get("fusion_weights") or ""))
            if not source_fusion.is_file():
                raise FileNotFoundError("source fusion is unavailable; supply runtime_fusion")
            runtime_fusion = source_fusion
        runtime_fusion = runtime_fusion.resolve()
        if not runtime_fusion.is_file() or sha256_file(runtime_fusion) != source_fusion_sha:
            raise ValueError("runtime fusion bytes differ from source lane")
        runtime_fusion_ref = {
            "path": str(runtime_fusion),
            "sha256": source_fusion_sha,
            "bytes_changed": False,
        }
    elif runtime_fusion is not None:
        raise ValueError("runtime fusion supplied for a lane without fusion provenance")

    runtime_adapter_ref = None
    source_adapter_hashes = meta.get("adapter_hashes") or {}
    if source_adapter_hashes:
        if runtime_adapter is None:
            source_adapter = Path(str(meta.get("adapter") or ""))
            if not source_adapter.is_dir():
                raise FileNotFoundError("source adapter is unavailable; supply runtime_adapter")
            runtime_adapter = source_adapter
        runtime_adapter = runtime_adapter.resolve()
        observed_adapter_hashes = {
            path.name: sha256_file(path)
            for path in sorted(runtime_adapter.iterdir())
            if path.is_file()
        }
        if observed_adapter_hashes != source_adapter_hashes:
            raise ValueError("runtime adapter bytes differ from source lane")
        runtime_adapter_ref = {
            "path": str(runtime_adapter),
            "file_sha256": observed_adapter_hashes,
            "bytes_changed": False,
        }
    elif runtime_adapter is not None:
        raise ValueError("runtime adapter supplied for a lane without adapter provenance")

    source_task, source_corpus, source_bank = _route(source_manifest_payload, corpus=corpus)
    target_task, target_corpus, target_bank = _route(target_manifest_payload, corpus=corpus)
    if source_task != target_task or str(meta.get("task")) != target_task:
        raise ValueError("source/target task routing differs")
    if _scientific_route(source_corpus) != _scientific_route(target_corpus):
        raise ValueError("source/target corpus metadata differs beyond artifact paths")
    if _scientific_route(source_bank) != _scientific_route(target_bank):
        raise ValueError("source/target bank metadata differs beyond artifact paths")

    target_corpus_path = _resolve(str(target_corpus["path"]), target_manifest)
    target_bank_path = _resolve(str(target_bank["path"]), target_manifest)
    corpus_binding = meta.get("canonical_corpus") or {}
    bank_binding = meta.get("bank_artifact") or {}
    source_inventory_ref = None
    if (not corpus_binding.get("sha256") or not bank_binding.get("sha256")) and source_artifact_inventory is not None:
        source_artifact_inventory = source_artifact_inventory.resolve()
        inventory = json.loads(source_artifact_inventory.read_text(encoding="utf-8"))
        if inventory.get("source_manifest_sha256") != source_manifest_sha:
            raise ValueError("source artifact inventory is bound to another manifest")
        rows = {
            (str(row["section"]), str(row["name"])): row
            for row in inventory.get("artifacts") or []
        }
        corpus_binding = rows.get(("corpora", corpus)) or {}
        bank_binding = rows.get(("banks", source_task)) or {}
        source_inventory_ref = {
            "path": str(source_artifact_inventory),
            "sha256": sha256_file(source_artifact_inventory),
            "source_manifest_sha256": source_manifest_sha,
        }
    if not corpus_binding.get("sha256") or not bank_binding.get("sha256"):
        raise ValueError("source metadata lacks exact bank/corpus byte bindings")
    target_corpus_sha = sha256_file(target_corpus_path)
    target_bank_sha = sha256_file(target_bank_path)
    if target_corpus_sha != corpus_binding["sha256"]:
        raise ValueError("target corpus bytes differ from the source lane binding")
    if target_bank_sha != bank_binding["sha256"]:
        raise ValueError("target bank bytes differ from the source lane binding")

    expected_count = int(target_corpus.get("count", -1))
    output_k = int(meta.get("output_k", -1))
    bank_ids = _bank_ids(target_bank_path, int(target_bank.get("count", -1)))
    if expected_count < 0 or not 1 <= output_k <= len(bank_ids):
        raise ValueError("invalid canonical count or lane depth")
    bank_set = set(bank_ids)
    sentinel = object()
    observed = 0
    seen: set[str] = set()
    for canonical, candidate in zip_longest(
        _iter_jsonl(target_corpus_path), _iter_jsonl(source_candidate), fillvalue=sentinel
    ):
        if canonical is sentinel or candidate is sentinel:
            raise ValueError("candidate/canonical row counts differ")
        uid = str(canonical.get("norm_uid") or "")
        if not uid or uid in seen:
            raise ValueError(f"canonical corpus has missing/duplicate norm_uid: {uid!r}")
        seen.add(uid)
        if (
            str(candidate.get("norm_uid")) != uid
            or str(candidate.get("corpus")) != corpus
            or str(candidate.get("task")) != target_task
            or int(candidate.get("row", -1)) != int(canonical.get("row", -1))
            or str(candidate.get("bank_source_sha256"))
            != str(target_bank.get("source_sha256"))
        ):
            raise ValueError(f"candidate route/order differs at norm: {uid}")
        candidates = candidate.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != output_k:
            raise ValueError(f"candidate depth differs at norm: {uid}")
        metric_ids = [str(row.get("metric_id") or "") for row in candidates]
        ranks = [int(row.get("rank", -1)) for row in candidates]
        if (
            len(set(metric_ids)) != output_k
            or not set(metric_ids) <= bank_set
            or ranks != list(range(1, output_k + 1))
        ):
            raise ValueError(f"candidate metric universe/ranks invalid at norm: {uid}")
        observed += 1
    if observed != expected_count or int(meta.get("input_count", -1)) != expected_count:
        raise ValueError("candidate count differs from canonical manifest")

    rebound = copy.deepcopy(meta)
    rebound["manifest"] = str(target_manifest)
    rebound["manifest_sha256"] = target_manifest_sha
    rebound["output_path"] = str(output_candidate)
    rebound["output_sha256"] = candidate_sha
    if runtime_fusion_ref is not None:
        rebound["fusion_weights"] = runtime_fusion_ref["path"]
    if runtime_adapter_ref is not None:
        rebound["adapter"] = runtime_adapter_ref["path"]
    rebound["canonical_corpus"] = {
        "path": str(target_corpus_path),
        "sha256": target_corpus_sha,
        "size_bytes": target_corpus_path.stat().st_size,
    }
    rebound["bank_artifact"] = {
        "path": str(target_bank_path),
        "sha256": target_bank_sha,
        "size_bytes": target_bank_path.stat().st_size,
    }
    rebound["runtime_relocation"] = {
        "schema_version": "silver-match-v3-candidate-manifest-rebind-v1",
        "path_only": True,
        "candidate_bytes_changed": False,
        "labels_or_ranks_changed": False,
        "source_candidate": str(source_candidate),
        "source_candidate_sha256": candidate_sha,
        "source_meta": str(source_meta),
        "source_meta_sha256": sha256_file(source_meta),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_manifest_sha,
        "target_manifest": str(target_manifest),
        "target_manifest_sha256": target_manifest_sha,
        "canonical_rows_verified": observed,
        "bank_metrics_verified": len(bank_ids),
        "runtime_fusion": runtime_fusion_ref,
        "runtime_adapter": runtime_adapter_ref,
        "source_artifact_inventory": source_inventory_ref,
        "source_audit": source_audit_ref,
    }

    output_candidate.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_candidate, output_candidate)
        output_meta.write_text(
            json.dumps(rebound, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        output_meta.unlink(missing_ok=True)
        output_candidate.unlink(missing_ok=True)
        raise
    return {
        "schema_version": "silver-match-v3-candidate-manifest-rebind-report-v1",
        "status": "CANONICAL_MANIFEST_REBOUND",
        "candidate": {
            "path": str(output_candidate),
            "sha256": sha256_file(output_candidate),
            "rows": observed,
            "output_k": output_k,
        },
        "metadata": {"path": str(output_meta), "sha256": sha256_file(output_meta)},
        "candidate_bytes_changed": False,
        "labels_or_ranks_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-candidate", required=True)
    parser.add_argument("--source-meta", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-candidate", required=True)
    parser.add_argument("--runtime-fusion")
    parser.add_argument("--runtime-adapter")
    parser.add_argument("--source-artifact-inventory")
    parser.add_argument("--source-audit")
    parser.add_argument("--source-audit-sha256")
    parser.add_argument("--output-report")
    args = parser.parse_args()
    report = rebind_lane(
        source_candidate=Path(args.source_candidate),
        source_meta=Path(args.source_meta),
        source_manifest=Path(args.source_manifest),
        target_manifest=Path(args.target_manifest),
        corpus=args.corpus,
        output_candidate=Path(args.output_candidate),
        runtime_fusion=Path(args.runtime_fusion) if args.runtime_fusion else None,
        runtime_adapter=Path(args.runtime_adapter) if args.runtime_adapter else None,
        source_artifact_inventory=(
            Path(args.source_artifact_inventory)
            if args.source_artifact_inventory
            else None
        ),
        source_audit=Path(args.source_audit) if args.source_audit else None,
        source_audit_sha256=args.source_audit_sha256,
    )
    if args.output_report:
        report_path = Path(args.output_report).resolve()
        if report_path.exists():
            raise FileExistsError(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["report"] = {"path": str(report_path), "sha256": sha256_file(report_path)}
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
