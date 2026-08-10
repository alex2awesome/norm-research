#!/usr/bin/env python3
"""Freeze complete PR verifier-dev R4 predictions before any truth join."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adjudicate_gemma import DECISIONS
from .common import read_jsonl, sha256_file


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}


def freeze(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    schema = plan.get("schema_version")
    status = plan.get("status")
    supported_plan = (
        schema == "silver-match-v3-pr-verifier-dev-r4-proposal-plan-v1"
        and status == "FROZEN_BEFORE_DEV_PROPOSAL_INFERENCE"
    ) or (
        schema == "silver-match-v3-pr-verifier-dev-r4-proposal-plan-v2"
        and status == "FROZEN_BEFORE_REPAIRED_DEV_PROPOSAL_INFERENCE"
        and int((plan.get("rendering") or {}).get("seed") or -1) == 2026071321
        and int((plan.get("rendering") or {}).get("max_tokens") or -1) == 512
        and (plan.get("contracts") or {}).get("quarantined_v1_rows_reused")
        is False
        and (plan.get("contracts") or {}).get(
            "only_interface_runtime_seed_and_output_path_changed"
        )
        is True
    )
    if (
        not supported_plan
        or plan.get("task") != "press-releases"
        or plan.get("orders") != ["original", "hashed", "reverse"]
        or int(plan.get("row_count") or -1) != 300
        or int(plan.get("candidate_depth") or -1) != 50
        or plan.get("contracts", {}).get("truth_labels_or_predictions_read_by_plan")
        is not False
    ):
        raise ValueError("unsupported or weakened verifier-dev proposal plan")
    for ref in (plan.get("inputs") or {}).values():
        path = Path(str(ref.get("path") or "")).resolve()
        if not path.is_file() or sha256_file(path) != str(ref.get("sha256") or ""):
            raise ValueError(f"plan input drift: {path}")
    candidates_path = Path(plan["inputs"]["candidate_subset"]["path"])
    expected_uids = {str(row["norm_uid"]) for row in read_jsonl(candidates_path)}
    prompt_sha = plan["inputs"]["adjudicator_prompt"]["sha256"]
    candidate_sha = plan["inputs"]["candidate_subset"]["sha256"]

    outputs: dict[str, Any] = {}
    for order in plan["orders"]:
        path = Path(plan["outputs"][order]).resolve()
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        rows = list(read_jsonl(path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        uids = [str(row.get("norm_uid") or "") for row in rows]
        if (
            len(rows) != 300
            or len(set(uids)) != 300
            or set(uids) != expected_uids
            or meta.get("input_candidates_sha256") != candidate_sha
            or meta.get("prompt_sha256") != prompt_sha
            or meta.get("model") != plan["model"]
            or meta.get("order_mode") != order
            or int(meta.get("eligible_count") or -1) != 300
            or int(meta.get("new_count") or -1) != 300
            or int(meta.get("invalid_count", -1)) != 0
            or meta.get("output_sha256") != sha256_file(path)
        ):
            raise ValueError(f"incomplete or drifted verifier-dev output: {order}")
        for row in rows:
            decision = str(row.get("decision") or "")
            metric_id = row.get("metric_id")
            candidate_ids = [str(value) for value in row.get("candidate_ids") or []]
            if (
                row.get("task") != "press-releases"
                or row.get("order_mode") != order
                or row.get("model") != plan["model"]
                or row.get("prompt_sha256") != prompt_sha
                or row.get("parse_error") is not None
                or not str(row.get("raw_response") or "").strip()
                or decision not in DECISIONS
                or len(candidate_ids) != 50
                or len(set(candidate_ids)) != 50
                or (decision == "MATCH" and str(metric_id) not in candidate_ids)
                or (decision != "MATCH" and metric_id is not None)
            ):
                raise ValueError(f"invalid prediction row: {order}/{row.get('norm_uid')}")
        outputs[order] = {
            "predictions": _artifact(path),
            "meta": _artifact(meta_path),
            "count": len(rows),
            "invalid_count": 0,
        }
    return {
        "schema_version": "silver-match-v3-pr-verifier-dev-r4-proposal-output-freeze-v1",
        "status": "FROZEN_COMPLETE_BEFORE_TRUTH_JOIN",
        "task": "press-releases",
        "role": "verifier_dev_truth_hidden_proposals",
        "row_count": 300,
        "orders": plan["orders"],
        "prompt_sha256": prompt_sha,
        "model": plan["model"],
        "plan": _artifact(plan_path),
        "plan_schema_version": schema,
        "outputs": outputs,
        "truth_content_read_by_freezer": False,
        "contracts": plan["contracts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = freeze(Path(args.plan))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "freeze_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
