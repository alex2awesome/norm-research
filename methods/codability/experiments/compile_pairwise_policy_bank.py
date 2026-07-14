#!/usr/bin/env python
"""Compile a small frozen bank for within-Llama pairwise policy elicitation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import sha256_file, text_sha256


CELL_ID = "N_humor_49"


def _canonical(arm: dict) -> str:
    return next(form["prompt"] for form in arm["forms"] if form["id"] == "canonical")


def _copy(arm_id: str, source_arm: dict, *, source_family: str,
          source_shard_root: str, source_job: str) -> dict:
    content = _canonical(source_arm)
    return {
        "id": arm_id,
        "content": content,
        "content_sha256": text_sha256(content),
        "channel": source_arm["channel"],
        "provenance": source_arm["provenance"],
        "source_partition": source_arm.get("source_partition"),
        "semantic_content_word_count": source_arm["semantic_content_word_count"],
        "base_policy_source": {
            "family": source_family,
            "shard_root": source_shard_root,
            "model_job": source_job,
            "arm_id": source_arm["id"],
        },
    }


def compile_bank(*, source_bank_path: str, rule_bank_path: str, rank_bank_path: str,
                 source_shard_root: str, rule_shard_root: str,
                 rank_shard_root: str) -> dict:
    source_bank = json.loads(Path(source_bank_path).read_text())
    rule_bank = json.loads(Path(rule_bank_path).read_text())
    rank_bank = json.loads(Path(rank_bank_path).read_text())
    source = {arm["id"]: arm for cell in source_bank["cells"] if cell["id"] == CELL_ID
              for arm in cell["arms"]}
    rules = {arm["id"]: arm for cell in rule_bank["cells"] if cell["id"] == CELL_ID
             for arm in cell["arms"]}
    ranks = {arm["id"]: arm for cell in rank_bank["cells"] if cell["id"] == CELL_ID
             for arm in cell["arms"]}
    selected = [
        _copy("name", source["name"], source_family="source",
              source_shard_root=source_shard_root, source_job="llama3_small"),
        _copy("source_explanation", source["source_explanation"], source_family="source",
              source_shard_root=source_shard_root, source_job="llama3_small"),
        _copy("self_contrastive", rules["rule_contrastive_v0_from_self"],
              source_family="target_rules", source_shard_root=rule_shard_root,
              source_job="llama3_target_policy_rules"),
    ]
    for origin in ("prompt_selection", "unit_certification"):
        selected.append(_copy(
            f"behavior_contrastive_from_{origin}",
            rules[f"rule_contrastive_v1_from_{origin}"],
            source_family="target_rules", source_shard_root=rule_shard_root,
            source_job="llama3_target_policy_rules"))
        selected.append(_copy(
            f"rank_patch_from_{origin}",
            ranks[f"rank_rank_patch_v1_parent-behavior_standalone_from_{origin}"],
            source_family="rank_contrasts", source_shard_root=rank_shard_root,
            source_job="llama3_rank_contrasts"))
    payload = {
        "schema": "pairwise_policy_bank/v1",
        "status": "frozen-before-pairwise-small-executor-scoring",
        "objective": "recover 8B name-only item order with 3B comparative elicitation",
        "model_family": "Llama only: Llama-3.2-3B executor, Llama-3.1-8B target",
        "anchor_policy": "fixed 8B name-only behavior; no external labels or evaluator",
        "scope": "public residual folds only; no lockbox authorization",
        "source_artifacts": {
            "source_bank": {"path": source_bank_path, "sha256": sha256_file(source_bank_path)},
            "rule_bank": {"path": rule_bank_path, "sha256": sha256_file(rule_bank_path)},
            "rank_bank": {"path": rank_bank_path, "sha256": sha256_file(rank_bank_path)},
        },
        "cell": {
            "id": CELL_ID,
            "domain": "humor",
            "construct": "Wordplay quality and clarity",
            "arms": selected,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bank_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--rule-bank", required=True)
    parser.add_argument("--rank-bank", required=True)
    parser.add_argument("--source-shard-root", required=True)
    parser.add_argument("--rule-shard-root", required=True)
    parser.add_argument("--rank-shard-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    bank = compile_bank(
        source_bank_path=args.source_bank, rule_bank_path=args.rule_bank,
        rank_bank_path=args.rank_bank, source_shard_root=args.source_shard_root,
        rule_shard_root=args.rule_shard_root, rank_shard_root=args.rank_shard_root)
    out = Path(args.out)
    out.write_text(json.dumps(bank, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256_file(out),
                      "n_arms": len(bank["cell"]["arms"])}, indent=1))


if __name__ == "__main__":
    main()
