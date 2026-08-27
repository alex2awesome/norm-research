#!/usr/bin/env python3
"""Freeze complete two-order PR verifier-dev predictions before scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .verify_gemma import DECISIONS


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    values = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in values or len(values) != len(rows):
        raise ValueError(f"invalid UID coverage: {path}")
    return values


def freeze(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version")
        != "silver-match-v3-pr-verifier-dev-inference-plan-v1"
        or plan.get("status")
        != "FROZEN_BEFORE_FIRST_FRESH_DEV_VERIFIER_INFERENCE"
        or plan.get("task") != "press-releases"
        or plan.get("role") != "verifier_dev_selection"
        or plan.get("orders") != ["original", "hashed"]
        or int(plan.get("variant_count", -1)) != 1
    ):
        raise ValueError("unsupported verifier-dev inference plan")
    for ref in (plan.get("inputs") or {}).values():
        path = Path(str(ref.get("path") or "")).resolve()
        if not path.is_file() or sha256_file(path) != str(ref.get("sha256") or ""):
            raise ValueError(f"plan input drift: {path}")
    pair_refs = plan.get("pair_outputs") or {}
    for ref in pair_refs.values():
        path = Path(str(ref.get("path") or "")).resolve()
        if not path.is_file() or sha256_file(path) != str(ref.get("sha256") or ""):
            raise ValueError(f"pair output drift: {path}")
    candidates = _index(Path(pair_refs["candidates"]["path"]))
    primary = _index(Path(pair_refs["primary"]["path"]))
    expected_uids = set(primary)
    if set(candidates) != expected_uids or len(expected_uids) != int(
        plan["selected_pair_count"]
    ):
        raise ValueError("pair candidates/primary coverage drift")
    prompt_sha = plan["inputs"]["prompt"]["sha256"]
    rendering = plan["rendering"]

    outputs: dict[str, Any] = {}
    for order in plan["orders"]:
        path = Path(plan["outputs"][order]).resolve()
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        rows = _index(path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            set(rows) != expected_uids
            or meta.get("input_candidates_sha256")
            != pair_refs["candidates"]["sha256"]
            or meta.get("primary_sha256") != pair_refs["primary"]["sha256"]
            or meta.get("prompt_sha256") != prompt_sha
            or meta.get("model") != plan["model"]
            or meta.get("order_mode") != order
            or int(meta.get("eligible_count", -1)) != len(expected_uids)
            or int(meta.get("new_count", -1)) != len(expected_uids)
            or int(meta.get("invalid_count", -1)) != 0
            or meta.get("output_sha256") != sha256_file(path)
            or int(meta.get("max_alternatives", -1))
            != int(rendering["max_alternatives"])
            or int(meta.get("max_tokens", -1)) != int(rendering["max_tokens"])
            or int(meta.get("seed", -1)) != int(rendering["seed"])
        ):
            raise ValueError(f"incomplete or drifted verifier-dev output: {order}")
        for uid, row in rows.items():
            proposal_id = str(primary[uid].get("metric_id") or "")
            candidate_ids = {
                str(value.get("metric_id") or "")
                for value in candidates[uid].get("candidates") or []
            }
            alternatives = [str(value) for value in row.get("alternative_ids") or []]
            decision, metric_id = str(row.get("decision") or ""), row.get("metric_id")
            if (
                row.get("task") != "press-releases"
                or row.get("order_mode") != order
                or row.get("model") != plan["model"]
                or row.get("prompt_sha256") != prompt_sha
                or row.get("primary_metric_id") != proposal_id
                or row.get("parse_error") is not None
                or not str(row.get("raw_response") or "").strip()
                or decision not in DECISIONS
                or len(alternatives) != int(rendering["max_alternatives"])
                or len(set(alternatives)) != len(alternatives)
                or proposal_id in alternatives
                or not set(alternatives) <= candidate_ids
                or (decision == "CONFIRM_MATCH" and str(metric_id) != proposal_id)
                or (
                    decision == "BETTER_CANDIDATE"
                    and str(metric_id) not in set(alternatives)
                )
                or (
                    decision not in {"CONFIRM_MATCH", "BETTER_CANDIDATE"}
                    and metric_id is not None
                )
            ):
                raise ValueError(f"invalid prediction row: {order}/{uid}")
        outputs[order] = {
            "predictions": _artifact(path),
            "meta": _artifact(meta_path),
            "count": len(rows),
            "invalid_count": 0,
        }
    return {
        "schema_version": "silver-match-v3-pr-verifier-dev-output-freeze-v1",
        "status": "FROZEN_COMPLETE_BEFORE_FRESH_DEV_SCORING",
        "task": "press-releases",
        "role": "verifier_dev_selection",
        "row_count": len(expected_uids),
        "orders": plan["orders"],
        "prompt_sha256": prompt_sha,
        "model": plan["model"],
        "plan": _artifact(plan_path),
        "outputs": outputs,
        "truth_or_targets_read_by_freezer": False,
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
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({**payload, "freeze_sha256": sha256_file(output)}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
