#!/usr/bin/env python3
"""Build a frozen, class-balanced Sonnet teacher calibration slate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .common import normalize_space, read_jsonl, write_jsonl
from .config import DEFAULT_OUTPUT_ROOT


def hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_for(uid: str) -> str:
    bucket = int(hash_key(uid)[:8], 16) % 10
    return "train" if bucket < 6 else "dev" if bucket < 8 else "test"


def split_group_for(norm: dict) -> str:
    """Keep all norms from one human feedback unit in one split.

    ``paper_id`` is the strongest grouping when present; otherwise the source
    review/comment/thread ID is used.  Falling back to the norm UID is explicit
    and recorded in the frozen sample so leakage audits can count it.
    """
    corpus = normalize_space(norm.get("corpus"))
    paper_id = normalize_space(norm.get("paper_id"))
    source_id = normalize_space(norm.get("source_id"))
    if paper_id:
        return f"{corpus}:paper:{paper_id}"
    if source_id:
        return f"{corpus}:source:{source_id}"
    return f"{corpus}:uid:{norm['norm_uid']}"


def load_norm_index(manifest_path: Path) -> dict[str, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    norms: dict[str, dict] = {}
    for meta in (manifest.get("corpora") or {}).values():
        for row in read_jsonl(Path(meta["path"])):
            uid = str(row["norm_uid"])
            if uid in norms:
                raise ValueError(f"duplicate norm_uid in manifest: {uid}")
            norms[uid] = row
    return norms


def diverse_matches(rows: list[dict], limit: int) -> list[dict]:
    """Round-robin metric groups so generic high-mass metrics cannot dominate."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["metric_id"])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: hash_key(row["norm_uid"]))
    ordered_groups = sorted(groups, key=hash_key)
    selected = []
    cursor = 0
    while len(selected) < limit and ordered_groups:
        next_groups = []
        for metric_id in ordered_groups:
            values = groups[metric_id]
            if cursor < len(values):
                selected.append(values[cursor])
                next_groups.append(metric_id)
                if len(selected) >= limit:
                    break
        ordered_groups = next_groups
        cursor += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teachers", default=str(DEFAULT_OUTPUT_ROOT / "teachers" / "sonnet.jsonl")
    )
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT / "calibration"))
    parser.add_argument("--matches-per-corpus", type=int, default=500)
    args = parser.parse_args()
    teachers = list(read_jsonl(Path(args.teachers)))
    norm_index = load_norm_index(Path(args.manifest))
    missing = [row["norm_uid"] for row in teachers if row["norm_uid"] not in norm_index]
    if missing:
        raise ValueError(f"{len(missing)} teachers are absent from manifest; first={missing[0]}")
    by_corpus: dict[str, list[dict]] = defaultdict(list)
    for row in teachers:
        by_corpus[row["corpus"]].append(row)

    selected: list[dict] = []
    for corpus, rows in sorted(by_corpus.items()):
        matches = [row for row in rows if row["decision"] == "MATCH"]
        abstentions = [row for row in rows if row["decision"] != "MATCH"]
        selected.extend(diverse_matches(matches, args.matches_per_corpus))
        selected.extend(abstentions)
    selected = sorted(
        (
            {
                **row,
                "split_group": split_group_for(norm_index[row["norm_uid"]]),
                "split": split_for(split_group_for(norm_index[row["norm_uid"]])),
            }
            for row in selected
        ),
        key=lambda row: (row["corpus"], row["row"]),
    )
    output_root = Path(args.output_root)
    write_jsonl(output_root / "teacher_sample.jsonl", selected)
    uids: dict[str, list[str]] = defaultdict(list)
    for row in selected:
        uids[row["corpus"]].append(row["norm_uid"])
    for corpus, values in sorted(uids.items()):
        path = output_root / "uids" / f"{corpus}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{uid}\n" for uid in sorted(values)), encoding="utf-8")
    summary = {
        "total": len(selected),
        "by_corpus": {c: len(v) for c, v in uids.items()},
        "by_decision": {
            decision: sum(row["decision"] == decision for row in selected)
            for decision in sorted({row["decision"] for row in selected})
        },
        "by_split": {
            split: sum(row["split"] == split for row in selected)
            for split in ("train", "dev", "test")
        },
        "split_group_count": len({row["split_group"] for row in selected}),
        "uid_fallback_groups": sum(":uid:" in row["split_group"] for row in selected),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
