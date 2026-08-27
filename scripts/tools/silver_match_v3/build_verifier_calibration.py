#!/usr/bin/env python3
"""Build leakage-safe human calibration slates for proposal verification.

Each slate contains the model proposal, the independent human label, and the
strongest base-retrieval alternatives.  Typed human abstentions are retained:
they are necessary to measure whether the verifier rejects false MATCH claims.
The hidden human label never changes candidate ordering.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .common import read_jsonl, sha256_file, write_jsonl


def load_unique(paths: Sequence[Path], *, key: str, task: str | None = None) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for path in paths:
        for row in read_jsonl(path):
            if task is not None and row.get("task") != task:
                continue
            value = str(row[key])
            if value in output:
                raise ValueError(f"duplicate {key}: {value}")
            output[value] = row
    return output


def compact_slate(
    candidate_row: dict[str, Any],
    required_ids: Sequence[str],
    limit: int,
    *,
    reorder_required: bool = True,
) -> dict[str, Any]:
    if limit < len(set(required_ids)):
        raise ValueError("candidate limit is smaller than required metric set")
    by_id = {
        str(row["metric_id"]): row for row in candidate_row.get("candidates") or []
    }
    missing = [metric_id for metric_id in required_ids if metric_id not in by_id]
    if missing:
        raise ValueError(f"required metrics absent from full candidate bank: {missing}")
    if not reorder_required:
        ordered = list(candidate_row.get("candidates") or [])[:limit]
        visible = {str(row["metric_id"]) for row in ordered}
        hidden = [metric_id for metric_id in required_ids if metric_id not in visible]
        if hidden:
            raise ValueError(f"required metrics absent from compact slate: {hidden}")
        return {**candidate_row, "candidates": ordered}
    ordered = []
    seen = set()
    for metric_id in [*required_ids, *by_id]:
        if metric_id in seen:
            continue
        ordered.append(by_id[metric_id])
        seen.add(metric_id)
        if len(ordered) == limit:
            break
    return {**candidate_row, "candidates": ordered}


def build(
    *,
    task: str,
    proposal_rows: Sequence[dict],
    human_rows: Sequence[dict],
    candidates: dict[str, dict],
    split: str,
    candidate_limit: int,
) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    proposals = {
        str(row["norm_uid"]): row
        for row in proposal_rows
        if row.get("task") == task and row.get("decision") == "MATCH"
    }
    truth = [
        row
        for row in human_rows
        if row.get("task") == task
        and row.get("split") == split
        and str(row["norm_uid"]) in proposals
    ]
    truth_uids = [str(row["norm_uid"]) for row in truth]
    if len(truth_uids) != len(set(truth_uids)):
        raise ValueError(f"duplicate human truth UID in {split}")
    absent = sorted(set(truth_uids) - set(candidates))
    if absent:
        raise KeyError(f"human calibration UIDs lack candidate rows: {absent[:5]}")
    slates, primary = [], []
    agreement = Counter()
    for row in sorted(truth, key=lambda value: str(value["norm_uid"])):
        uid = str(row["norm_uid"])
        proposal = proposals[uid]
        proposal_id = str(proposal["metric_id"])
        truth_id = None if row.get("metric_id") is None else str(row["metric_id"])
        if row.get("decision") != "MATCH":
            agreement[f"typed_truth:{row.get('decision')}"] += 1
        else:
            agreement[
                "exact_agreement" if proposal_id == truth_id else "exact_conflict"
            ] += 1
        # Preserve the deployed retriever order.  Putting the hidden human gold
        # first would leak the answer to a verifier whose alternatives are read
        # positionally.  Only the proposal is required to be visible; whether
        # gold is naturally captured is recorded for calibration diagnostics.
        slates.append(
            compact_slate(
                candidates[uid],
                (proposal_id,),
                candidate_limit,
                reorder_required=False,
            )
        )
        primary.append(proposal)
    if not truth:
        raise ValueError(f"no overlapping proposal/human MATCH rows for {task}/{split}")
    return slates, primary, sorted(truth, key=lambda value: str(value["norm_uid"])), {
        "task": task,
        "split": split,
        "count": len(truth),
        "candidate_limit": candidate_limit,
        "agreement": dict(sorted(agreement.items())),
        "truth_decisions": dict(sorted(Counter(str(row["decision"]) for row in truth).items())),
        "gold_naturally_present": sum(
            row.get("decision") == "MATCH"
            and str(row["metric_id"])
            in {
                str(value["metric_id"])
                for value in candidates[str(row["norm_uid"])].get("candidates", [])[
                    :candidate_limit
                ]
            }
            for row in truth
        ),
        "gold_match_count": sum(row.get("decision") == "MATCH" for row in truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--human-panel", action="append", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "dev"), default=("train", "dev")
    )
    args = parser.parse_args()
    proposal_path = Path(args.proposals).resolve()
    human_paths = tuple(Path(path).resolve() for path in args.human_panel)
    candidate_paths = tuple(Path(path).resolve() for path in args.candidates)
    proposal_rows = list(read_jsonl(proposal_path))
    human_rows = [row for path in human_paths for row in read_jsonl(path)]
    candidates = load_unique(candidate_paths, key="norm_uid", task=args.task)
    output = Path(args.output_root).resolve() / args.task
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite calibration directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"task": args.task, "splits": {}}
    for split in args.splits:
        slates, primary, truth, split_report = build(
            task=args.task,
            proposal_rows=proposal_rows,
            human_rows=human_rows,
            candidates=candidates,
            split=split,
            candidate_limit=args.candidate_limit,
        )
        paths = {
            "candidates": output / f"{split}.candidates.jsonl",
            "primary": output / f"{split}.primary.jsonl",
            "truth": output / f"{split}.truth.jsonl",
        }
        write_jsonl(paths["candidates"], slates)
        write_jsonl(paths["primary"], primary)
        write_jsonl(paths["truth"], truth)
        split_report["output_hashes"] = {
            key: sha256_file(path) for key, path in paths.items()
        }
        report["splits"][split] = split_report
    report["input_hashes"] = {
        "proposals": {str(proposal_path): sha256_file(proposal_path)},
        "human_panels": {str(path): sha256_file(path) for path in human_paths},
        "candidates": {str(path): sha256_file(path) for path in candidate_paths},
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
