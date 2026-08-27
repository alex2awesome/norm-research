#!/usr/bin/env python3
"""Freeze a task's audited production candidate union and dev retriever choice.

This plan deliberately stops at retrieval.  It does not authorize adjudication
until a separate task K50 adjudicator/verifier policy is dev-supported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def freeze(
    *,
    manifest_path: Path,
    task: str,
    candidate_union_path: Path,
    audit_paths: list[Path],
    selection_path: Path,
    expected_k: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest_path)
    if task not in manifest.get("banks", {}):
        raise KeyError(task)
    corpora = {
        corpus for corpus, meta in manifest["corpora"].items() if meta["task"] == task
    }
    expected_count = sum(int(manifest["corpora"][corpus]["count"]) for corpus in corpora)
    bank_sha = str(manifest["banks"][task]["source_sha256"])

    audited_inputs: dict[str, str] = {}
    audited_corpora: set[str] = set()
    observed = 0
    audits = {}
    fusion_hashes: set[str] = set()
    adapters: set[str] = set()
    retrieval_signatures: set[tuple[str, str, str, bool]] = set()
    for path in audit_paths:
        audit = json.loads(path.read_text(encoding="utf-8"))
        corpus = str(audit.get("corpus") or "")
        if (
            audit.get("complete") is not True
            or audit.get("task") != task
            or audit.get("manifest_sha256") != manifest_sha
            or str(audit.get("bank_source_sha256")) != bank_sha
            or int(audit.get("expected_k", -1)) != expected_k
            or corpus in audited_corpora
        ):
            raise ValueError(f"invalid/duplicate candidate audit: {path}")
        audited_corpora.add(corpus)
        observed += int(audit["observed_count"])
        fusion_hashes.update(str(value) for value in audit.get("fusion_hashes") or [])
        adapters.update(str(value) for value in audit.get("adapters") or [])
        for signature in audit.get("retrieval_signatures") or []:
            retrieval_signatures.add(
                (
                    str(signature.get("encoder") or ""),
                    str(signature.get("query_format") or ""),
                    str(signature.get("query_views") or ""),
                    bool(signature.get("dense_query_instruction")),
                )
            )
        for raw, meta in (audit.get("candidate_inputs") or {}).items():
            resolved = str(Path(raw).resolve())
            if resolved in audited_inputs:
                raise ValueError(f"duplicate candidate input across audits: {resolved}")
            audited_inputs[resolved] = str(meta["sha256"])
        audits[str(path)] = sha256_file(path)
    if audited_corpora != corpora or observed != expected_count:
        raise ValueError(
            f"task audits incomplete: corpora={audited_corpora ^ corpora}, "
            f"count={observed}/{expected_count}"
        )

    union_meta_path = candidate_union_path.with_suffix(
        candidate_union_path.suffix + ".meta.json"
    )
    union_meta = json.loads(union_meta_path.read_text(encoding="utf-8"))
    union_sha = sha256_file(candidate_union_path)
    recorded_union_sha = union_meta.get("sha256") or union_meta.get("output_sha256")
    if recorded_union_sha != union_sha or int(union_meta.get("count", -1)) != expected_count:
        raise ValueError("candidate union metadata hash/count mismatch")
    combined_inputs = {
        str(Path(path).resolve()): str(value["sha256"])
        for path, value in (union_meta.get("inputs") or {}).items()
    }
    if combined_inputs != audited_inputs:
        raise ValueError("candidate union is not exactly the audited corpus union")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("task") != task
        or selection.get("selection_split") != "external_dev_only"
        or selection.get("frozen_test_consumed") is not False
    ):
        raise ValueError("retriever selection is not task-matched external-dev-only")
    chosen = selection.get("chosen") or {}
    fusion_path = Path(str(chosen.get("fusion_report") or "")).resolve()
    fusion_sha = sha256_file(fusion_path)
    if chosen.get("fusion_report_sha256") != fusion_sha or fusion_hashes != {fusion_sha}:
        raise ValueError("candidate fusion differs from selected fusion")
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    if fusion.get("selection_split") != "dev" or fusion.get("task") != task:
        raise ValueError("selected fusion is not task-matched dev selection")
    chosen_kind = str(chosen.get("kind") or "")
    selected_signatures: set[tuple[str, str, str, bool]] = set()
    selected_adapters: set[str] = set()
    selected_label_inputs = {
        str(Path(path).resolve()): str(digest)
        for path, digest in (selection.get("label_inputs") or {}).items()
    }
    if not selected_label_inputs:
        raise ValueError("retriever selection has no hash-pinned external dev labels")
    selected_label_uids: set[str] = set()
    for raw_path, recorded_hash in selected_label_inputs.items():
        label_path = Path(raw_path)
        if not label_path.exists() or sha256_file(label_path) != recorded_hash:
            raise ValueError("selected external dev label hash mismatch")
        for row in read_jsonl(label_path):
            if row.get("task") != task or row.get("split") != "dev":
                raise ValueError("retriever selection labels are not task-specific dev")
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in selected_label_uids:
                raise ValueError("missing/duplicate UID in selected external dev labels")
            selected_label_uids.add(uid)
    selected_candidate_inputs = chosen.get("candidate_inputs") or {}
    if not selected_candidate_inputs:
        raise ValueError("retriever selection has no hash-pinned dev candidate inputs")
    fusion_candidate_inputs = {
        str(Path(path).resolve()): str(digest)
        for path, digest in (fusion.get("candidate_inputs") or {}).items()
    }
    resolved_selected_candidates = {
        str(Path(path).resolve()): str(digest)
        for path, digest in selected_candidate_inputs.items()
    }
    fusion_label_inputs = {
        str(Path(path).resolve()): str(digest)
        for path, digest in (fusion.get("label_inputs") or {}).items()
    }
    if (
        fusion_candidate_inputs != resolved_selected_candidates
        or fusion_label_inputs != selected_label_inputs
    ):
        raise ValueError("retriever selection and selected fusion inputs differ")
    for raw_path, recorded_hash in selected_candidate_inputs.items():
        dev_path = Path(str(raw_path)).resolve()
        if not dev_path.exists() or sha256_file(dev_path) != str(recorded_hash):
            raise ValueError("selected dev candidate input hash mismatch")
        dev_meta_path = dev_path.with_suffix(dev_path.suffix + ".meta.json")
        if not dev_meta_path.exists():
            raise ValueError("selected dev candidate metadata missing")
        dev_meta = json.loads(dev_meta_path.read_text(encoding="utf-8"))
        if dev_meta.get("output_sha256") != str(recorded_hash):
            raise ValueError("selected dev candidate metadata hash mismatch")
        dev_items = Path(str(dev_meta.get("items") or "")).resolve()
        if (
            dev_meta.get("manifest_sha256") != manifest_sha
            or not dev_items.exists()
            or dev_meta.get("items_sha256") != sha256_file(dev_items)
        ):
            raise ValueError("selected dev candidate source is not hash-pinned")
        if any(row.get("split") != "dev" for row in read_jsonl(dev_items)):
            raise ValueError("selected dev candidate source includes a non-dev row")
        candidate_uid_rows = [
            str(row.get("norm_uid") or "")
            for row in read_jsonl(dev_path)
            if row.get("task") == task
        ]
        candidate_uids = set(candidate_uid_rows)
        if (
            "" in candidate_uids
            or len(candidate_uid_rows) != len(candidate_uids)
            or candidate_uids != selected_label_uids
        ):
            raise ValueError("selected candidate task rows differ from selected dev labels")
        selected_signatures.add(
            (
                str(dev_meta.get("encoder") or ""),
                str(dev_meta.get("query_format") or ""),
                str(dev_meta.get("query_views") or ""),
                bool(dev_meta.get("dense_query_instruction")),
            )
        )
        selected_adapter = str(dev_meta.get("adapter") or "")
        if selected_adapter:
            selected_adapter_path = Path(selected_adapter).resolve()
            actual_hashes = {
                path.name: sha256_file(path)
                for path in sorted(selected_adapter_path.iterdir())
                if path.is_file()
            }
            if actual_hashes != (dev_meta.get("adapter_hashes") or {}):
                raise ValueError("selected dev adapter hash mismatch")
            selected_adapters.add(str(selected_adapter_path))
    if len(retrieval_signatures) != 1 or retrieval_signatures != selected_signatures:
        raise ValueError("production retrieval signature differs from selected dev retriever")
    if chosen_kind == "adapter":
        resolved_adapters = {str(Path(path).resolve()) for path in adapters}
        if len(resolved_adapters) != 1 or resolved_adapters != selected_adapters:
            raise ValueError("selected adapter production candidates lack one adapter")
        adapter_path = Path(next(iter(resolved_adapters)))
        adapter_files = {
            path.name: sha256_file(path)
            for path in sorted(adapter_path.iterdir())
            if path.is_file()
        }
    else:
        if adapters or selected_adapters:
            raise ValueError("base selection unexpectedly used an adapter")
        adapter_path, adapter_files = None, None

    return {
        "schema_version": "silver-match-v3-retrieval-production-plan-v1",
        "status": "FROZEN_RETRIEVAL_READY_FOR_K50_POLICY",
        "task": task,
        "expected_k": expected_k,
        "expected_count": expected_count,
        "corpora": sorted(corpora),
        "manifest": artifact(manifest_path),
        "bank_source_sha256": bank_sha,
        "candidate_union": artifact(candidate_union_path),
        "candidate_union_meta": artifact(union_meta_path),
        "candidate_audits": audits,
        "retriever_selection": artifact(selection_path),
        "chosen": {
            "name": chosen.get("name"),
            "kind": chosen_kind,
            "fusion": artifact(fusion_path),
            "retrieval_signature": {
                "encoder": next(iter(retrieval_signatures))[0],
                "query_format": next(iter(retrieval_signatures))[1],
                "query_views": next(iter(retrieval_signatures))[2],
                "dense_query_instruction": next(iter(retrieval_signatures))[3],
            },
            "adapter": (
                {"path": str(adapter_path), "files": adapter_files}
                if adapter_path
                else None
            ),
        },
        "frozen_test": {
            "status": "SEALED_UNCONSUMED",
            "metrics_reported": False,
        },
        "authorization": {
            "retrieval_complete": True,
            "adjudication_authorized": False,
            "requirement": (
                "freeze a task-matched K50 adjudicator and verifier policy on dev before "
                "production adjudication"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate-union", required=True)
    parser.add_argument("--candidate-audit", action="append", required=True)
    parser.add_argument("--retriever-selection", required=True)
    parser.add_argument("--expected-k", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = freeze(
        manifest_path=Path(args.manifest).resolve(),
        task=args.task,
        candidate_union_path=Path(args.candidate_union).resolve(),
        audit_paths=[Path(path).resolve() for path in args.candidate_audit],
        selection_path=Path(args.retriever_selection).resolve(),
        expected_k=args.expected_k,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**plan, "output": artifact(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
