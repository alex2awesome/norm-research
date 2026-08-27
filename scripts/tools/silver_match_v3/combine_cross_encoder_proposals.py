#!/usr/bin/env python3
"""Combine two selected CE runs into exact-agreement provisional proposals."""

from __future__ import annotations

import argparse
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

from .common import read_jsonl, sha256_file


SCHEMA_VERSION = "silver-match-v3.cross-encoder-consensus.1"


def _meta(path: Path) -> tuple[Path, dict[str, Any]]:
    meta_path = Path(str(path) + ".meta.json")
    value = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "COMPLETE"
        or value.get("output_sha256") != sha256_file(path)
        or int(value.get("written", -1)) != int((value.get("pins") or {}).get("input_rows", -2))
    ):
        raise ValueError(f"CE proposal artifact is not sealed: {path}")
    return meta_path, value


def _selected_reports(selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if selection.get("status") != "TWO_VARIANT_CE_PROPOSAL_PATH_SELECTED":
        raise ValueError("selection did not authorize two-variant CE consensus")
    chosen = selection.get("chosen") or []
    if len(chosen) != 2:
        raise ValueError("selection must contain exactly two chosen variants")
    return {str(row["name"]): row for row in chosen}


def combine_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2:
        raise ValueError("exactly two CE rows are required")
    uid = str(rows[0]["norm_uid"])
    if any(str(row.get("norm_uid")) != uid for row in rows):
        raise ValueError("CE proposal streams are not UID-aligned")
    base_keys = ("task", "corpus", "row", "bank_source_sha256")
    if any(
        any(row.get(key) != rows[0].get(key) for key in base_keys)
        for row in rows[1:]
    ):
        raise ValueError(f"CE proposal identity mismatch for {uid}")
    candidate_sets = [
        {str(candidate["metric_id"]) for candidate in row.get("candidates") or []}
        for row in rows
    ]
    if not candidate_sets[0] or candidate_sets[0] != candidate_sets[1]:
        raise ValueError(f"CE runs used different candidate universes for {uid}")
    proposals = [row["ce_proposal"] for row in rows]
    metric_ids = [
        str(value["metric_id"])
        for value in proposals
        if value.get("decision") == "PROVISIONAL_MATCH" and value.get("metric_id")
    ]
    agreed = len(metric_ids) == 2 and metric_ids[0] == metric_ids[1]
    consensus = {
        "decision": "PROVISIONAL_MATCH" if agreed else "PROVISIONAL_ABSTAIN",
        "metric_id": metric_ids[0] if agreed else None,
        "agreement_count": 2 if agreed else 0,
        "variant_proposals": proposals,
    }
    first = rows[0]
    return {
        **{key: value for key, value in first.items() if key not in {"candidates", "ce_proposal"}},
        "schema_version": SCHEMA_VERSION,
        "candidates": first["candidates"],
        "ce_consensus": consensus,
    }


def _iter_combined(paths: list[Path]) -> Iterator[dict[str, Any]]:
    streams = [read_jsonl(path) for path in paths]
    for values in zip_longest(*streams):
        if any(value is None for value in values):
            raise ValueError("CE proposal streams have different row counts")
        yield combine_rows(list(values))


def run(selection_path: Path, inputs: list[str], output: Path) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = _selected_reports(selection)
    named_paths: dict[str, Path] = {}
    for value in inputs:
        if "=" not in value:
            raise ValueError("--input must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        if name in named_paths or name not in selected:
            raise ValueError(f"unknown or duplicate selected variant: {name}")
        named_paths[name] = Path(raw_path).resolve()
    if set(named_paths) != set(selected):
        raise ValueError("proposal inputs must exactly match selected CE variants")
    ordered_names = [str(row["name"]) for row in selection["chosen"]]
    paths = [named_paths[name] for name in ordered_names]
    metas = []
    pins = []
    for name, path in zip(ordered_names, paths):
        meta_path, meta = _meta(path)
        if (
            meta["pins"]["training_report_sha256"] != selected[name]["sha256"]
            or meta["pins"]["task"] != selection["task"]
            or meta["pins"]["bank_source_sha256"] != selection["bank_source_sha256"]
        ):
            raise ValueError(f"CE proposal pins differ from selection: {name}")
        metas.append(
            {
                "name": name,
                "path": str(meta_path),
                "sha256": sha256_file(meta_path),
                "output": str(path),
                "output_sha256": sha256_file(path),
            }
        )
        pins.append(meta["pins"])
    comparable = (
        "manifest_sha256",
        "bank_source_sha256",
        "candidate_inputs",
        "input_rows",
        "corpus_counts",
        "output_k",
    )
    if any(
        any(value.get(key) != pins[0].get(key) for key in comparable)
        for value in pins[1:]
    ):
        raise ValueError("selected CE runs did not score identical frozen inputs")
    output = output.resolve()
    report_path = Path(str(output) + ".report.json")
    if output.exists() or report_path.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"PROVISIONAL_MATCH": 0, "PROVISIONAL_ABSTAIN": 0}
    written = 0
    with output.open("x", encoding="utf-8") as handle:
        for row in _iter_combined(paths):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts[row["ce_consensus"]["decision"]] += 1
            written += 1
    if written != int(pins[0]["input_rows"]):
        raise ValueError("CE consensus output has incomplete row coverage")
    report = {
        "schema_version": "silver-match-v3-cross-encoder-consensus-report-v1",
        "status": "COMPLETE_PROVISIONAL_ONLY",
        "task": selection["task"],
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "inputs": metas,
        "input_rows": written,
        "decision_counts": counts,
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "final_silver_decisions_emitted": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(Path(args.selection), args.input, Path(args.output))
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
