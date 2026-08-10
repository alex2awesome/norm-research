#!/usr/bin/env python3
"""Validate and promote strict three-pass labels for a frozen verifier panel."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def index(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate UID in {path}")
    return output


def decision_key(row: dict) -> tuple[str, str | None]:
    decision = str(row.get("decision") or "")
    metric_id = str(row.get("metric_id")) if decision == "MATCH" else None
    return decision, metric_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-items", required=True)
    parser.add_argument("--dev-items", required=True)
    parser.add_argument("--blind-audit-items", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--third", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    paths = {
        key: Path(value).resolve()
        for key, value in {
            "train_items": args.train_items,
            "dev_items": args.dev_items,
            "blind_audit_items": args.blind_audit_items,
            "first": args.first,
            "second": args.second,
            "third": args.third,
        }.items()
    }
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)

    train_items, dev_items, audit_items = (
        index(paths["train_items"]), index(paths["dev_items"]), index(paths["blind_audit_items"])
    )
    items = {**train_items, **dev_items}
    if len(items) != len(train_items) + len(dev_items):
        raise ValueError("verifier train/dev overlap by UID")
    if set(items) & set(audit_items):
        raise ValueError("permanent blind audit was exposed to the labeling universe")
    item_groups = {str(row["source_group"]) for row in items.values()}
    audit_groups = {str(row["source_group"]) for row in audit_items.values()}
    if len(item_groups) != len(items) or len(audit_groups) != len(audit_items):
        raise ValueError("a verifier role contains repeated source groups")
    if item_groups & audit_groups:
        raise ValueError("permanent blind audit overlaps labeling by source group")

    labels = {name: index(paths[name]) for name in ("first", "second", "third")}
    if any(set(values) != set(items) for values in labels.values()):
        raise ValueError("each independent pass must cover verifier train+dev exactly")
    meta = {}
    for name in labels:
        meta_path = paths[name].with_suffix(paths[name].suffix + ".meta.json")
        meta[name] = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta[name].get("output_sha256") != sha256_file(paths[name]):
            raise ValueError(f"label output/meta hash mismatch: {name}")
    pass_identities = {
        (meta[name].get("model"), meta[name].get("prompt_sha256"), meta[name].get("order_mode"), meta[name].get("seed"))
        for name in labels
    }
    if len(pass_identities) != 3:
        raise ValueError("the three label passes are not independently configured")

    counts: Counter[str] = Counter()
    promoted = {"train": [], "dev": []}
    for uid in sorted(items):
        item = items[uid]
        rows = [labels[name][uid] for name in ("first", "second", "third")]
        if any(row.get("task") != "humor" for row in rows):
            raise ValueError(f"task mismatch: {uid}")
        if any(
            row.get("candidate_bank_source_sha256") != item.get("bank_source_sha256")
            for row in rows
        ):
            raise ValueError(f"bank mismatch: {uid}")
        keys = [decision_key(row) for row in rows]
        if len(set(keys)) != 1:
            counts["excluded_exact_disagreement"] += 1
            continue
        counts[f"exact_consensus:{keys[0][0]}"] += 1
        if any(row.get("confidence") != "high" for row in rows):
            counts["excluded_not_all_high"] += 1
            continue
        decision, metric_id = keys[0]
        role = str(item["verifier_expansion_role"])
        split = "train" if role == "verifier_train" else "dev" if role == "verifier_dev" else None
        if split is None:
            raise ValueError(f"unexpected labeled role: {role}")
        promoted[split].append(
            {
                "schema_version": "silver-match-v3.0",
                "norm_uid": uid,
                "corpus": item.get("corpus"),
                "task": "humor",
                "row": item.get("row"),
                "source_group": item.get("source_group"),
                "split_group": item.get("source_group"),
                "split": split,
                "decision": decision,
                "metric_id": metric_id,
                "confidence": "high",
                "reason": "strict exact high-confidence consensus of three independently configured strong-model passes",
                "label_source": "strict_three_model_exact_high_consensus",
                "current_bank_source_sha256": item.get("bank_source_sha256"),
                "boundary_stratum": item.get("boundary_stratum"),
                "verifier_expansion_role": role,
                "pass_models": [meta[name].get("model") for name in ("first", "second", "third")],
                "pass_model_snapshots": [meta[name].get("model_snapshot") for name in ("first", "second", "third")],
                "pass_prompt_sha256": [meta[name].get("prompt_sha256") for name in ("first", "second", "third")],
                "pass_order_modes": [meta[name].get("order_mode") for name in ("first", "second", "third")],
                "pass_output_sha256": [sha256_file(paths[name]) for name in ("first", "second", "third")],
                "pass_reasons": [row.get("reason") for row in rows],
            }
        )
        counts[f"promoted:{split}:{decision}"] += 1

    output_root.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for split in ("train", "dev"):
        path = output_root / f"{split}.strict-consensus.labels.jsonl"
        write_jsonl(path, promoted[split])
        outputs[split] = {"path": str(path), "sha256": sha256_file(path), "count": len(promoted[split])}
    report = {
        "schema_version": "silver-match-v3-verifier-expansion-three-pass-promotion-v1",
        "task": "humor",
        "policy": "exact decision+metric agreement and high confidence in all three independently configured passes",
        "source_count": len(items),
        "permanent_blind_audit_count": len(audit_items),
        "permanent_blind_audit_labeled": False,
        "counts": dict(sorted(counts.items())),
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "pass_identities": [list(value) for value in sorted(pass_identities, key=str)],
        "outputs": outputs,
    }
    report_path = output_root / "PROMOTION.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
