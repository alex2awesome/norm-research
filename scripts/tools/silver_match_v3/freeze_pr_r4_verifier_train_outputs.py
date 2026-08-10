#!/usr/bin/env python3
"""Freeze PR R4 optimize predictions before any verifier author sees them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adjudicate_gemma import DECISIONS
from .common import read_jsonl, sha256_file


SCHEMA = "silver-match-v3-pr-r4-verifier-train-output-freeze-v1"
LOCK_SCHEMA = "silver-match-v3-pr-r4-verifier-train-inference-lock-v1"


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def freeze(lock_path: Path) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("status") != "FROZEN_BEFORE_R4_OPTIMIZE_INFERENCE"
        or lock.get("task") != "press-releases"
        or lock.get("role") != "optimize_only_for_future_verifier_training"
        or lock.get("adjudicator_select_already_consumed") is not True
        or lock.get("adjudicator_prompt_frozen_before_select") is not True
        or lock.get("verifier_selection_requires_a_new_source-disjoint_panel") is not True
    ):
        raise ValueError("unsupported or weakened PR R4 verifier-train lock")

    repo_root = Path.cwd().resolve()
    inputs: dict[str, Path] = {}
    for name, value in (lock.get("inputs") or {}).items():
        path = Path(str(value.get("path") or ""))
        path = path if path.is_absolute() else repo_root / path
        path = path.resolve()
        if not path.is_file() or sha256_file(path) != value.get("sha256"):
            raise ValueError(f"frozen input changed: {name}={path}")
        inputs[name] = path

    candidates = list(read_jsonl(inputs["candidates"]))
    expected_uids = [str(row.get("norm_uid") or "") for row in candidates]
    expected_count = int(lock["inputs"]["candidates"]["count"])
    expected_k = int(lock["inputs"]["candidates"]["k"])
    if (
        len(candidates) != expected_count
        or "" in expected_uids
        or len(set(expected_uids)) != expected_count
        or any(len(row.get("candidates") or []) != expected_k for row in candidates)
    ):
        raise ValueError("frozen candidate universe is incomplete or malformed")
    expected_uid_set = set(expected_uids)

    inference = lock["inference"]
    orders = list(inference["orders"])
    if orders != ["original", "hashed", "reverse"]:
        raise ValueError("R4 verifier-train orders are not the frozen three-order set")
    prompt_sha = lock["inputs"]["frozen_r4_prompt"]["sha256"]
    candidate_sha = lock["inputs"]["candidates"]["sha256"]
    outputs: dict[str, Any] = {}
    total_requests = 0
    for order in orders:
        raw_path = Path(str(lock["outputs"][order]))
        raw_path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        raw_path = raw_path.resolve()
        meta_path = raw_path.with_suffix(raw_path.suffix + ".meta.json")
        if not raw_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"R4 verifier-train output incomplete: {order}")
        rows = list(read_jsonl(raw_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        uids = [str(row.get("norm_uid") or "") for row in rows]
        if (
            len(rows) != expected_count
            or len(set(uids)) != expected_count
            or set(uids) != expected_uid_set
            or meta.get("selection_role") != "train"
            or meta.get("order_mode") != order
            or meta.get("model") != inference["model"]
            or int(meta.get("eligible_count", -1)) != expected_count
            or int(meta.get("new_count", -1)) != expected_count
            or int(meta.get("invalid_count", -1)) != 0
            or int(meta.get("api_request_count", 10**9))
            > int(inference["maximum_requests_per_order"])
            or meta.get("input_candidates_sha256") != candidate_sha
            or meta.get("prompt_sha256") != prompt_sha
            or meta.get("output_sha256") != sha256_file(raw_path)
        ):
            raise ValueError(f"R4 verifier-train metadata drift/incompleteness: {order}")
        for row in rows:
            decision = str(row.get("decision") or "")
            metric_id = row.get("metric_id")
            candidate_ids = [str(value) for value in row.get("candidate_ids") or []]
            if (
                row.get("task") != "press-releases"
                or row.get("order_mode") != order
                or row.get("model") != inference["model"]
                or row.get("prompt_sha256") != prompt_sha
                or row.get("parse_error") is not None
                or not str(row.get("raw_response") or "").strip()
                or decision not in DECISIONS
                or len(candidate_ids) != expected_k
                or len(set(candidate_ids)) != expected_k
                or (decision == "MATCH" and str(metric_id) not in candidate_ids)
                or (decision != "MATCH" and metric_id is not None)
            ):
                raise ValueError(f"invalid R4 verifier-train row: {order}/{row.get('norm_uid')}")
        requests = int(meta["api_request_count"])
        total_requests += requests
        outputs[order] = {
            "predictions": _artifact(raw_path),
            "meta": _artifact(meta_path),
            "count": len(rows),
            "api_request_count": requests,
            "retry_prompt_inferences": int(meta.get("retry_prompt_inferences", 0)),
            "invalid_count": 0,
        }
    if total_requests > int(inference["maximum_total_requests"]):
        raise ValueError("R4 verifier-train total request cap exceeded")
    return {
        "schema_version": SCHEMA,
        "status": "FROZEN_COMPLETE_BEFORE_VERIFIER_AUTHORING",
        "task": "press-releases",
        "role": "optimize_only_for_future_verifier_training",
        "row_count": expected_count,
        "orders": orders,
        "prompt_sha256": prompt_sha,
        "model": inference["model"],
        "total_api_requests": total_requests,
        "lock": _artifact(lock_path),
        "outputs": outputs,
        "truth_content_read_by_freezer": False,
        "adjudicator_select_content_available_to_verifier_author": False,
        "verifier_selection_requires_new_source_disjoint_panel": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = freeze(Path(args.lock))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "freeze_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
