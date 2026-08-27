#!/usr/bin/env python3
"""Seal two independently permuted, mutually prediction-hidden label views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


FORBIDDEN_ITEM_FIELDS = {
    "decision",
    "metric_id",
    "acceptable_metric_ids",
    "reason",
    "label",
    "prediction",
    "raw_response",
    "candidate_ids",
}


def _load(root: Path, *, allow_label_artifacts: bool = False) -> dict[str, Any]:
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    items_path, bank_path = root / "items.jsonl", root / "bank.json"
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError(f"item hash mismatch: {root}")
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError(f"bank hash mismatch: {root}")
    chunks = sorted((root / "chunks").glob("part-*.jsonl"))
    recorded_chunks = validation["outputs"]["chunks"]
    if len(chunks) != len(recorded_chunks):
        raise ValueError(f"chunk count mismatch: {root}")
    recorded_by_name = {
        Path(recorded_path).name: recorded_sha256
        for recorded_path, recorded_sha256 in recorded_chunks.items()
    }
    if len(recorded_by_name) != len(recorded_chunks):
        raise ValueError(f"recorded chunk basenames are not unique: {root}")
    for path in chunks:
        # Independent label packs are deliberately copied into isolated
        # workspaces before either pass starts.  The immutable validation file
        # therefore records the source location, while the staged chunk has a
        # different absolute prefix.  Bind relocation by unique basename and
        # content hash rather than weakening the content check.
        if sha256_file(path) != recorded_by_name.get(path.name):
            raise ValueError(f"chunk hash mismatch: {path}")
    items = list(read_jsonl(items_path))
    uids = [str(row.get("norm_uid") or "") for row in items]
    if not items or "" in uids or len(uids) != len(set(uids)):
        raise ValueError(f"empty or duplicate item UIDs: {root}")
    leaked = [
        uid for uid, row in zip(uids, items) if FORBIDDEN_ITEM_FIELDS & set(row)
    ]
    if leaked:
        raise ValueError(f"view contains forbidden label/proposal fields: {leaked[:3]}")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    metric_ids = [str(row.get("metric_id") or "") for row in bank.get("metrics") or []]
    if not metric_ids or "" in metric_ids or len(metric_ids) != len(set(metric_ids)):
        raise ValueError(f"view bank has missing/duplicate metric IDs: {root}")
    label_artifacts = [
        path
        for name in ("raw_labels", "labels.validated.jsonl", "predictions.jsonl")
        if (path := root / name).exists()
    ]
    if label_artifacts and not allow_label_artifacts:
        raise ValueError(f"view already contains labels/predictions: {label_artifacts}")
    candidate_artifacts = sorted(root.glob("*candidate*"))
    if candidate_artifacts:
        raise ValueError(f"independent full-bank view exposes candidate proposals: {root}")
    return {
        "root": root,
        "validation_path": validation_path,
        "validation": validation,
        "items_path": items_path,
        "items": items,
        "uids": uids,
        "bank_path": bank_path,
        "bank": bank,
        "metric_ids": metric_ids,
        "chunks": chunks,
        "label_artifacts": label_artifacts,
    }


def _validate_transcript_audit(path: Path, view: dict[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    audit = json.loads(path.read_text(encoding="utf-8"))
    rows = {str(row.get("chunk") or ""): row for row in audit.get("chunks") or []}
    chunks = {chunk.stem: chunk for chunk in view["chunks"]}
    schema = audit.get("schema_version")
    if (
        schema
        not in {
            "silver-match-v3-isolated-labeler-transcript-audit-v1",
            "silver-match-v3-claude-labeler-transcript-audit-v1",
        }
        or audit.get("status") != "PASS"
        or audit.get("complete") is not True
        or audit.get("violations") != []
        or str((audit.get("bank") or {}).get("sha256") or "")
        != sha256_file(view["bank_path"])
        or set(rows) != set(chunks)
    ):
        raise ValueError(f"invalid transcript-isolation audit: {path}")
    for chunk_id, chunk in chunks.items():
        row = rows[chunk_id]
        if schema == "silver-match-v3-isolated-labeler-transcript-audit-v1":
            raw = view["root"] / "raw_labels" / f"{chunk_id}.json"
            log = view["root"] / "logs" / f"{chunk_id}.log"
            valid = (
                raw.is_file()
                and log.is_file()
                and row.get("chunk_sha256") == sha256_file(chunk)
                and row.get("raw_label_sha256") == sha256_file(raw)
                and row.get("log_sha256") == sha256_file(log)
            )
        else:
            raw = Path(str(row.get("raw_label_path") or "")).resolve()
            transcript = Path(str(row.get("transcript_path") or "")).resolve()
            stderr = Path(str(row.get("stderr_path") or "")).resolve()
            namespace = view["root"] / str(audit.get("output_namespace") or "")
            valid = (
                raw == (namespace / "raw_labels" / f"{chunk_id}.json").resolve()
                and raw.is_file()
                and transcript.is_file()
                and stderr.is_file()
                and row.get("chunk_sha256") == sha256_file(chunk)
                and row.get("raw_label_sha256") == sha256_file(raw)
                and row.get("transcript_sha256") == sha256_file(transcript)
                and row.get("stderr_sha256") == sha256_file(stderr)
            )
        if not valid:
            raise ValueError(f"transcript audit artifact drift: {chunk_id}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": "PASS",
        "audited_chunks": len(rows),
        "event_count": int(
            audit.get("command_count")
            if schema == "silver-match-v3-isolated-labeler-transcript-audit-v1"
            else audit.get("event_count")
            or 0
        ),
        "schema_version": schema,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-a", required=True)
    parser.add_argument("--pass-b", required=True)
    parser.add_argument("--transcript-audit-a")
    parser.add_argument("--transcript-audit-b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    post_label = bool(args.transcript_audit_a or args.transcript_audit_b)
    if post_label and not (args.transcript_audit_a and args.transcript_audit_b):
        parser.error("post-label audit requires both transcript audits")
    pass_a = _load(Path(args.pass_a).resolve(), allow_label_artifacts=post_label)
    pass_b = _load(Path(args.pass_b).resolve(), allow_label_artifacts=post_label)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    validation_a, validation_b = pass_a["validation"], pass_b["validation"]
    if validation_a.get("task") != validation_b.get("task"):
        raise ValueError("independent views have different tasks")
    if validation_a.get("bank_source_sha256") != validation_b.get("bank_source_sha256"):
        raise ValueError("independent views have different bank identities")
    if validation_a.get("source_pack") != validation_b.get("source_pack"):
        raise ValueError("independent views do not derive from the same frozen source pack")
    if validation_a.get("seed") == validation_b.get("seed"):
        raise ValueError("independent views must use distinct permutation seeds")
    if set(pass_a["uids"]) != set(pass_b["uids"]):
        raise ValueError("independent views cover different UID sets")
    by_uid_a = {str(row["norm_uid"]): row for row in pass_a["items"]}
    by_uid_b = {str(row["norm_uid"]): row for row in pass_b["items"]}
    if by_uid_a != by_uid_b:
        raise ValueError("independent views alter canonical item content")
    if pass_a["uids"] == pass_b["uids"]:
        raise ValueError("independent views have identical item order")
    if set(pass_a["metric_ids"]) != set(pass_b["metric_ids"]):
        raise ValueError("independent views contain different bank leaves")
    if pass_a["metric_ids"] == pass_b["metric_ids"]:
        raise ValueError("independent views have identical bank order")

    item_fixed = sum(
        left == right for left, right in zip(pass_a["uids"], pass_b["uids"])
    )
    metric_fixed = sum(
        left == right
        for left, right in zip(pass_a["metric_ids"], pass_b["metric_ids"])
    )
    transcript_audits = None
    if post_label:
        transcript_audits = {
            "A": _validate_transcript_audit(
                Path(args.transcript_audit_a), pass_a
            ),
            "B": _validate_transcript_audit(
                Path(args.transcript_audit_b), pass_b
            ),
        }
    report = {
        "schema_version": "silver-match-v3-independent-pack-view-audit-v1",
        "status": (
            "AUDITED_POSTLABEL_FROM_PRELABEL_FREEZES_AND_ISOLATED_TRANSCRIPTS"
            if post_label
            else "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING"
        ),
        "task": validation_a["task"],
        "count": len(pass_a["uids"]),
        "bank_metric_count": len(pass_a["metric_ids"]),
        "same_frozen_source_pack": True,
        "same_uid_set": True,
        "same_canonical_item_content_by_uid": True,
        "same_bank_leaf_set": True,
        "distinct_seeds": True,
        "distinct_item_order": True,
        "distinct_bank_order": True,
        "item_order_fixed_positions": item_fixed,
        "bank_order_fixed_positions": metric_fixed,
        "candidate_proposals_exposed_to_either_pass": False,
        "prior_truth_or_predictions_exposed_to_either_pass": False,
        "pass_predictions_mutually_visible": False,
        "post_label_artifacts_present": post_label,
        "transcript_isolation_audits": transcript_audits,
        "passes": {
            "A": {
                "root": str(pass_a["root"]),
                "seed": validation_a["seed"],
                "validation_sha256": sha256_file(pass_a["validation_path"]),
                "items_sha256": sha256_file(pass_a["items_path"]),
                "bank_sha256": sha256_file(pass_a["bank_path"]),
            },
            "B": {
                "root": str(pass_b["root"]),
                "seed": validation_b["seed"],
                "validation_sha256": sha256_file(pass_b["validation_path"]),
                "items_sha256": sha256_file(pass_b["items_path"]),
                "bank_sha256": sha256_file(pass_b["bank_path"]),
            },
        },
        "usage_contract": {
            "run_passes_in_separate_processes": True,
            "do_not_supply_other_pass_outputs_in_prompt_or_files": True,
            "validate_each_pass_before_consensus": True,
            "consensus_may_be_built_only_after_both_passes_are_complete": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "audit_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
