#!/usr/bin/env python3
"""Freeze a fresh train-only Math pack enriched for recurring leaf boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for


PATTERNS = [
    ("metacognitive_checking", r"oops|wrong|error|mistake|check|verify|sanity|plausib|fallac|misprint"),
    ("tool_method_choice", r"use |using |theorem|algorithm|method|technique|lemma|crt|wlog|without loss|pre.?calcul|data structure"),
    ("proof_gap_rigor", r"why|justify|gap|missing|assum|premise|case|step|inference|rigor|proof"),
    ("special_case_counterexample", r"counterexample|special case|example|normalize|reformulat|auxiliary|simpler case|limit"),
    ("exposition_audience", r"clear|clarity|readab|ambig|explain|understand|audience|accessible|palatable|notation|terminolog"),
    ("publication_typesetting", r"latex|format|proofread|typo|typeset|layout|label|cross.?refer|publication|english"),
    ("benchmark_question", r"contest|problem|question|difficulty|too hard|too easy|misplaced|benchmark|test suite"),
    ("formal_code", r"mathlib|lean|code|function|variable|identifier|implementation|api|rename"),
    ("limits_complexity", r"complexity|time limit|diverge|cannot|can't|does not work|limit|constraint|performance"),
]


def stratum(row: dict) -> str:
    text = normalize_space(
        " ".join(str(row.get(key) or "") for key in ("aspect", "norm", "context"))
    ).lower()[:4000]
    for name, pattern in PATTERNS:
        if re.search(pattern, text):
            return name
    return "other"


def order_key(namespace: str, uid: str) -> tuple[str, str]:
    return hashlib.sha256(f"math-boundary-v1\0{namespace}\0{uid}".encode()).hexdigest(), uid


def sample(rows: list[dict], total: int) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["boundary_stratum"]), str(row["corpus"]))].append(row)
    for key, values in buckets.items():
        values.sort(key=lambda row: order_key("/".join(key), str(row["norm_uid"])))
    cursors = {key: 0 for key in buckets}
    output = []
    for _ in range(total):
        available = [key for key in sorted(buckets) if cursors[key] < len(buckets[key])]
        if not available:
            break
        key = available[len(output) % len(available)]
        output.append(buckets[key][cursors[key]])
        cursors[key] += 1
    if len(output) != total:
        raise ValueError(f"requested {total}, selected {len(output)}")
    return sorted(output, key=lambda row: order_key("final", str(row["norm_uid"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--total", type=int, default=600)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    task = "math-stackexchange"
    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    norms: dict[str, dict] = {}
    for corpus, meta in manifest["corpora"].items():
        if meta.get("task") != task:
            continue
        path = Path(meta["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        for row in read_jsonl(path):
            uid = str(row["norm_uid"])
            if uid in norms:
                raise ValueError(f"duplicate canonical norm UID: {uid}")
            norms[uid] = row

    excluded_uids: set[str] = set()
    excluded_groups: set[str] = set()
    exclusion_paths = [Path(path).resolve() for path in args.exclude_label]
    for path in exclusion_paths:
        for row in read_jsonl(path):
            if row.get("task") != task:
                continue
            uid = str(row["norm_uid"])
            if uid not in norms:
                raise KeyError(f"excluded UID outside canonical Math norms: {uid}")
            excluded_uids.add(uid)
            excluded_groups.add(split_group_for(norms[uid]))

    by_group: dict[str, list[dict]] = defaultdict(list)
    excluded = Counter()
    for uid, row in norms.items():
        group = split_group_for(row)
        if split_for(group) != "train":
            excluded["not_train"] += 1
            continue
        if uid in excluded_uids or group in excluded_groups:
            excluded["prior_or_external_overlap"] += 1
            continue
        by_group[group].append(row)
    eligible = []
    for group, values in by_group.items():
        chosen = min(values, key=lambda row: order_key("group", str(row["norm_uid"])))
        eligible.append(
            {
                **chosen,
                "split": "train",
                "split_group": group,
                "boundary_stratum": stratum(chosen),
            }
        )
    selected = sample(eligible, args.total)
    selected_groups = {str(row["split_group"]) for row in selected}
    if len(selected_groups) != len(selected):
        raise AssertionError("selected Math boundary pack repeats a source group")
    if selected_groups & excluded_groups:
        raise AssertionError("selected Math boundary pack leaks an excluded source group")

    bank_path = Path(manifest["banks"][task]["path"])
    if not bank_path.is_absolute():
        bank_path = manifest_path.parent / bank_path
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    items_path = output_root / "items.jsonl"
    bank_output = output_root / "bank.json"
    write_jsonl(items_path, selected)
    bank_output.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    chunks = []
    for start in range(0, len(selected), args.chunk_size):
        path = output_root / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(path, selected[start : start + args.chunk_size])
        chunks.append(path)
    report = {
        "schema_version": "silver-match-v3-math-boundary-pack-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "count": len(selected),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "selected_source_groups": len(selected_groups),
        "selected_by_corpus": dict(sorted(Counter(row["corpus"] for row in selected).items())),
        "selected_by_stratum": dict(
            sorted(Counter(row["boundary_stratum"] for row in selected).items())
        ),
        "excluded": dict(sorted(excluded.items())),
        "overlap": {
            "uid_with_exclusions": len({row["norm_uid"] for row in selected} & excluded_uids),
            "source_group_with_exclusions": len(selected_groups & excluded_groups),
        },
        "bank_source_sha256": manifest["banks"][task]["source_sha256"],
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
            "exclusions": {str(path): sha256_file(path) for path in exclusion_paths},
        },
        "outputs": {
            "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            "bank": {"path": str(bank_output), "sha256": sha256_file(bank_output)},
            "chunks": {str(path): sha256_file(path) for path in chunks},
        },
    }
    report_path = output_root / "validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
