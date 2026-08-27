#!/usr/bin/env python3
"""Freeze the exact missing chunk frontier for a direct-batch GPU continuation.

Selection is based only on completed/missing artifact presence from a prior
non-promoted interruption freeze.  The script never reads partial decisions or
reasons, and it preserves each frozen pass's item and full-bank order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


PARTIAL_SCHEMA = "silver-match-v3-partial-openrouter-label-freeze-v1"
SCHEMA = "silver-match-v3-missing-openrouter-gpu-continuation-v1"


def _ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _index(path: Path, label: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"{label} has missing or duplicate UIDs")
    return rows, indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial-freeze", required=True)
    parser.add_argument("--pass-name", required=True)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--full-candidates", required=True)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite GPU continuation artifacts")
    partial_path = Path(args.partial_freeze).resolve()
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    pass_meta = (partial.get("passes") or {}).get(args.pass_name) or {}
    if (
        partial.get("schema_version") != PARTIAL_SCHEMA
        or partial.get("status")
        != "FROZEN_PARTIAL_TRANSCRIPT_AUDITED_INTERRUPTED_NO_PROMOTION"
        or pass_meta.get("promoted") is not False
    ):
        raise ValueError("invalid or promoted partial interruption freeze")

    pack_root = Path(args.pack_root).resolve()
    validation_path = pack_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        Path(str(pass_meta.get("root") or "")).resolve() != pack_root
        or (pass_meta.get("pack_validation") or {}).get("sha256")
        != sha256_file(validation_path)
        or validation.get("truth_hidden") is not True
    ):
        raise ValueError("partial freeze does not bind the supplied truth-hidden pack")
    chunks = {path.stem: path for path in sorted((pack_root / "chunks").glob("part-*.jsonl"))}
    missing = list(pass_meta.get("missing_chunks") or [])
    completed = {
        str(row.get("chunk") or "") for row in pass_meta.get("completed_chunks") or []
    }
    if (
        not missing
        or len(missing) != len(set(missing))
        or set(missing) & completed
        or set(missing) | completed != set(chunks)
    ):
        raise ValueError("completed/missing chunk partition is not exact")

    full_candidates = Path(args.full_candidates).resolve()
    candidate_freeze_path = Path(args.candidate_freeze).resolve()
    candidate_freeze = json.loads(candidate_freeze_path.read_text(encoding="utf-8"))
    full_rows, candidate_by_uid = _index(full_candidates, "full candidates")
    item_rows, item_by_uid = _index(pack_root / "items.jsonl", "pack items")
    if (
        candidate_freeze.get("schema_version")
        != "silver-match-v3-full-bank-blind-candidates-freeze-v1"
        or candidate_freeze.get("status") != "FROZEN_BEFORE_INFERENCE"
        or candidate_freeze.get("truth_hidden") is not True
        or candidate_freeze.get(
            "prior_decisions_metric_ids_predictions_and_proposals_read"
        )
        is not False
        or ((candidate_freeze.get("inputs") or {}).get("pack_validation") or {}).get(
            "sha256"
        )
        != sha256_file(validation_path)
        or (candidate_freeze.get("output") or {}).get("sha256")
        != sha256_file(full_candidates)
        or set(candidate_by_uid) != set(item_by_uid)
        or [str(row["norm_uid"]) for row in full_rows]
        != [str(row["norm_uid"]) for row in item_rows]
    ):
        raise ValueError("full candidates are not an exact pack-order projection")

    bank = json.loads((pack_root / "bank.json").read_text(encoding="utf-8"))
    bank_ids = [str(row["metric_id"]) for row in bank.get("metrics") or []]
    if len(bank_ids) != int(candidate_freeze.get("candidate_depth", -1)):
        raise ValueError("candidate depth differs from the frozen bank")
    for uid, row in candidate_by_uid.items():
        observed = [str(card["metric_id"]) for card in row.get("candidates") or []]
        if observed != bank_ids or row.get("truth_hidden") is not True:
            raise ValueError(f"full-bank order/provenance drift: {uid}")

    missing_uid_order: list[str] = []
    missing_chunk_refs: list[dict[str, Any]] = []
    for chunk in sorted(missing):
        path = chunks[chunk]
        rows = list(read_jsonl(path))
        missing_uid_order.extend(str(row["norm_uid"]) for row in rows)
        missing_chunk_refs.append(
            {"chunk": chunk, "count": len(rows), "input": _ref(path)}
        )
    if len(missing_uid_order) != len(set(missing_uid_order)):
        raise ValueError("missing chunks overlap in UID space")
    selected = [candidate_by_uid[uid] for uid in missing_uid_order]
    write_jsonl(output, selected)
    report = {
        "schema_version": SCHEMA,
        "status": "FROZEN_EXACT_MISSING_FRONTIER_BEFORE_DIRECT_BATCH_VLLM",
        "task": validation.get("task"),
        "pass_name": args.pass_name,
        "count": len(selected),
        "missing_chunk_count": len(missing),
        "candidate_depth": len(bank_ids),
        "bank_source_sha256": validation.get("bank_source_sha256"),
        "inputs": {
            "partial_interruption_freeze": _ref(partial_path),
            "pack_validation": _ref(validation_path),
            "pack_items": _ref(pack_root / "items.jsonl"),
            "pack_bank": _ref(pack_root / "bank.json"),
            "full_candidates": _ref(full_candidates),
            "full_candidate_freeze": _ref(candidate_freeze_path),
            "missing_chunks": missing_chunk_refs,
        },
        "output": _ref(output),
        "contracts": {
            "selection_uses_only_missing_chunk_artifact_presence": True,
            "partial_label_decisions_metric_ids_confidences_and_reasons_read": False,
            "all_and_only_missing_uids_selected": True,
            "frozen_item_order_preserved": True,
            "frozen_full_bank_order_preserved_per_row": True,
            "truth_keys_system_predictions_mi_and_outcomes_absent": True,
            "no_prompt_or_scientific_setting_retuning_from_partial_votes": True,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "pass_name": args.pass_name,
                "count": len(selected),
                "output": str(output),
                "output_sha256": sha256_file(output),
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
