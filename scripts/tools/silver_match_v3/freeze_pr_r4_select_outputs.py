#!/usr/bin/env python3
"""Freeze complete PR R4 select predictions before joining select truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adjudicate_gemma import DECISIONS
from .common import read_jsonl, sha256_file


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite R4 output freeze")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != "silver-match-v3-pr-r4-select-openrouter-plan-v2"
        or plan.get("status") != "FROZEN_EXECUTION_SUPERSESSION_BEFORE_ANY_SELECT_SCORE"
        or plan.get("variant_count") != 1
        or plan.get("orders") != ["original", "hashed", "reverse"]
        or plan.get("contracts", {}).get("join_truth_only_after_both_outputs_frozen")
        is not True
        or plan.get("contracts", {}).get("score_requires_original_hashed_reverse_frozen")
        is not True
    ):
        raise ValueError("unsupported or unfrozen R4 select plan")
    for value in plan["inputs"].values():
        path = Path(value["path"])
        if sha256_file(path) != value["sha256"]:
            raise ValueError(f"plan input hash drift: {path}")
    expected_uids = {
        str(row["norm_uid"]) for row in read_jsonl(Path(plan["inputs"]["pack_items"]["path"]))
    }
    if len(expected_uids) != int(plan["row_count"]):
        raise ValueError("frozen R4 select identity count mismatch")

    frozen_outputs: dict[str, Any] = {}
    total_requests = 0
    for order in plan["orders"]:
        prediction_path = Path(plan["outputs"][order])
        meta_path = prediction_path.with_suffix(prediction_path.suffix + ".meta.json")
        rows = list(read_jsonl(prediction_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        observed_uids = [str(row.get("norm_uid") or "") for row in rows]
        if (
            len(rows) != int(plan["row_count"])
            or len(set(observed_uids)) != len(observed_uids)
            or set(observed_uids) != expected_uids
            or meta.get("selection_role") != "dev"
            or meta.get("order_mode") != order
            or meta.get("model") != plan["model"]
            or int(meta.get("eligible_count", -1)) != int(plan["row_count"])
            or int(meta.get("new_count", -1)) != int(plan["row_count"])
            or int(meta.get("invalid_count", -1)) != 0
            or int(meta.get("api_request_count", 10**9))
            > int(plan["max_api_requests_per_order"])
            or meta.get("input_candidates_sha256")
            != plan["inputs"]["candidates"]["sha256"]
            or meta.get("prompt_sha256") != plan["inputs"]["prompt"]["sha256"]
        ):
            raise ValueError(f"incomplete or drifted R4 select output: {order}")
        for row in rows:
            decision = str(row.get("decision") or "")
            metric_id = row.get("metric_id")
            candidate_ids = [str(value) for value in row.get("candidate_ids") or []]
            if (
                row.get("task") != "press-releases"
                or row.get("order_mode") != order
                or row.get("model") != plan["model"]
                or row.get("prompt_sha256") != plan["inputs"]["prompt"]["sha256"]
                or row.get("parse_error") is not None
                or not str(row.get("raw_response") or "").strip()
                or decision not in DECISIONS
                or len(candidate_ids) != int(plan["candidate_depth"])
                or len(set(candidate_ids)) != len(candidate_ids)
                or (decision == "MATCH" and str(metric_id) not in candidate_ids)
                or (decision != "MATCH" and metric_id is not None)
            ):
                raise ValueError(f"invalid frozen prediction row: {order}/{row.get('norm_uid')}")
        requests = int(meta["api_request_count"])
        total_requests += requests
        frozen_outputs[order] = {
            "predictions": _artifact(prediction_path),
            "meta": _artifact(meta_path),
            "count": len(rows),
            "api_request_count": requests,
            "retry_prompt_inferences": int(meta.get("retry_prompt_inferences", 0)),
            "invalid_count": 0,
        }
    if total_requests > int(plan["max_total_api_requests"]):
        raise ValueError("R4 select total request cap exceeded")
    freeze = {
        "schema_version": "silver-match-v3-pr-r4-select-output-freeze-v2",
        "status": "FROZEN_COMPLETE_BEFORE_TRUTH_JOIN",
        "task": "press-releases",
        "role": "select",
        "variant_count": 1,
        "orders": plan["orders"],
        "row_count": plan["row_count"],
        "prompt_sha256": plan["inputs"]["prompt"]["sha256"],
        "model": plan["model"],
        "total_api_requests": total_requests,
        "plan": _artifact(plan_path),
        "outputs": frozen_outputs,
        "truth_content_read_by_freezer": False,
        "contracts": plan["contracts"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "freeze_sha256": sha256_file(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
