"""Execute notice/comment relations on compiler-train before independent audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from methods.metric_seam.notice_comment_relations_v1 import RELATION_DEPTHS, analyze


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
DEFAULT_PROPOSAL = BASE / "notice_comment_relations_static_proposal_v1.json"
DEFAULT_MANIFEST = BASE / "items_v2/notice-and-comment/manifest.json"
DEFAULT_TRAIN = BASE / "items_v2/notice-and-comment/compiler_train.json"
DEFAULT_OUTPUT = BASE / "notice_comment_relations_compiler_train_v1.json"
PROGRAM_SOURCE = ROOT / "methods/metric_seam/notice_comment_relations_v1.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(proposal: Mapping, manifest: Mapping, rows: object) -> list[dict]:
    if (
        proposal.get("schema") != "metric-seam.notice-comment-static-proposal.v1"
        or proposal.get("status")
        != "author_proposal_complete_pending_independent_construct_audit"
        or proposal.get("summary", {}).get("execution_eligible_before_independent_audit")
        != 0
        or proposal.get("program", {}).get("source_sha256") != _sha256(PROGRAM_SOURCE)
    ):
        raise ValueError("notice/comment pre-audit proposal or source freeze drifted")
    if (
        manifest.get("schema") != "metric-seam.hierarchy-shared-items.v1"
        or manifest.get("task") != "notice-and-comment"
        or manifest.get("selection", {}).get("outcome_or_reference_values_used") is not False
        or manifest.get("policy", {}).get("outcome_columns_emitted") is not False
        or manifest.get("policy", {}).get("external_supervision_used") is not False
    ):
        raise ValueError("notice/comment item manifest is not label-free")
    if not isinstance(rows, list) or len(rows) != 150:
        raise ValueError("expected 150 notice/comment compiler-train rows")
    checked = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"item_key", "ctext"}:
            raise ValueError("notice/comment rows must contain only item_key and ctext")
        if (
            not isinstance(row["item_key"], str)
            or not row["item_key"].startswith("train_")
            or not isinstance(row["ctext"], str)
            or len(row["ctext"]) > 4000
        ):
            raise ValueError("notice/comment compiler-train split or representation drifted")
        checked.append(row)
    return checked


def _summarize(rows: list[dict], relation: str) -> dict:
    outputs = [row["relations"][relation] for row in rows]
    statuses = Counter(output["status"] for output in outputs)
    finite = [
        float(output["score"])
        for output in outputs
        if output["status"] == "measured"
        and isinstance(output["score"], (int, float))
        and math.isfinite(float(output["score"]))
    ]
    unique = sorted(set(finite))
    return {
        "depth": RELATION_DEPTHS[relation],
        "status_counts": dict(sorted(statuses.items())),
        "measured": len(finite),
        "positive_items": sum(value > 0 for value in finite),
        "unique_finite_scores": len(unique),
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "nondegenerate": len(unique) >= 2,
    }


def run(proposal: Mapping, manifest: Mapping, rows: object) -> dict:
    checked = _validate(proposal, manifest, rows)
    item_rows = []
    for row in checked:
        output = analyze(row["ctext"])
        item_rows.append(
            {
                "item_key": row["item_key"],
                "input_characters": len(row["ctext"]),
                "relations": output["relations"],
            }
        )
    by_relation = {
        relation: _summarize(item_rows, relation) for relation in sorted(RELATION_DEPTHS)
    }
    nondegenerate = [
        relation for relation, summary in by_relation.items() if summary["nondegenerate"]
    ]
    return {
        "schema": "metric-seam.notice-comment-train-execution.v1",
        "status": "compiler_train_exploratory_complete_pending_independent_construct_audit",
        "phase": "compiler_train",
        "program": {
            "source": "methods/metric_seam/notice_comment_relations_v1.py",
            "source_sha256": _sha256(PROGRAM_SOURCE),
            "local_parser_model": "en_core_web_sm",
        },
        "blindness": {
            "input_fields_passed_to_program": ["ctext"],
            "outcome_fields_passed_to_program": False,
            "reference_fields_passed_to_program": False,
            "heldout_items_or_outputs_loaded": False,
            "external_authority_or_docket_loaded": False,
            "remote_model_or_api_used": False,
            "accelerator_used": False,
        },
        "summary": {
            "items": len(item_rows),
            "relations_executed": len(by_relation),
            "nondegenerate_relations": len(nondegenerate),
            "nondegenerate_relation_ids": nondegenerate,
            "hierarchy_mappings_promoted": 0,
            "heldout_execution_authorized": False,
            "prompt_articulability_measurements": 0,
            "reconstruction_measurements": 0,
            "isomorphism_measurements": 0,
        },
        "by_relation": by_relation,
        "rows": item_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--compiler-train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        json.loads(args.proposal.read_text(encoding="utf-8")),
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.compiler_train.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

