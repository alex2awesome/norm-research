#!/usr/bin/env python3
"""Evaluate a task-specific Nemotron adapter on one frozen external split.

Run dev first.  Run test only after the dev promotion decision is frozen; the
separate output directories and input hashes make accidental repeated test
inspection visible.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import metric_card, norm_query, read_jsonl, sha256_file
from .train_nemotron_lora import (
    DEFAULT_NEMOTRON,
    _encode,
    format_query,
    load_nemotron_adapter,
    normalize_name,
    retrieval_metrics,
    stable_rank,
)


def _resolve(path: str | Path, manifest_path: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else manifest_path.parent / value


def load_eval(
    manifest_path: Path, label_path: Path, task: str, split: str
) -> tuple[list[dict], list[dict], list[dict], str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_meta = manifest["banks"][task]
    bank = json.loads(_resolve(bank_meta["path"], manifest_path).read_text(encoding="utf-8"))[
        "metrics"
    ]
    bank_sha = str(bank_meta["source_sha256"])
    # Fail closed on role-mixed files.  Filtering a combined dev/test artifact
    # after opening it still exposes the test labels to the selection process,
    # even if the later scoring loop ignores those rows.  Callers must provide
    # a mechanically frozen, split-specific projection instead.
    all_labels = list(read_jsonl(label_path))
    if not all_labels:
        raise ValueError("external label artifact is empty")
    foreign_roles = sorted(
        {
            (str(row.get("task") or ""), str(row.get("split") or ""))
            for row in all_labels
            if row.get("task") != task or row.get("split") != split
        }
    )
    if foreign_roles:
        raise ValueError(
            "external label artifact is not task/split-isolated; "
            f"requested={task}/{split}, foreign_roles={foreign_roles[:5]}"
        )
    labels = [row for row in all_labels if row.get("decision") == "MATCH"]
    if not labels:
        raise ValueError(f"no external MATCH labels for {task}/{split}")
    uids = [str(row["norm_uid"]) for row in labels]
    if len(uids) != len(set(uids)):
        raise ValueError("duplicate external MATCH UID")
    for row in labels:
        if row.get("current_bank_source_sha256") != bank_sha:
            raise ValueError(f"external bank hash mismatch: {row['norm_uid']}")
    wanted = set(uids)
    norms: dict[str, dict] = {}
    for corpus, meta in manifest["corpora"].items():
        if meta.get("task") != task:
            continue
        for row in read_jsonl(_resolve(meta["path"], manifest_path)):
            uid = str(row["norm_uid"])
            if uid in wanted:
                norms[uid] = row
    missing = sorted(wanted - set(norms))
    if missing:
        raise KeyError(f"external UIDs absent from manifest: {missing[:5]}")
    ordered_norms = [norms[uid] for uid in uids]
    bank_ids = {str(metric["metric_id"]) for metric in bank}
    invalid = [(row["norm_uid"], row.get("metric_id")) for row in labels if row.get("metric_id") not in bank_ids]
    if invalid:
        raise ValueError(f"external metric absent from bank: {invalid[:5]}")
    return labels, ordered_norms, bank, bank_sha


def score_model(
    model: Any,
    labels: Sequence[Mapping[str, Any]],
    norms: Sequence[Mapping[str, Any]],
    bank: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    queries = [format_query(norm_query(dict(norm))) for norm in norms]
    cards = [metric_card(dict(metric)) for metric in bank]
    bank_embeddings = _encode(model, cards, batch_size)
    query_embeddings = _encode(model, queries, batch_size)
    scores = np.asarray(query_embeddings) @ np.asarray(bank_embeddings).T
    bank_ids = [str(metric["metric_id"]) for metric in bank]
    family = {str(metric["metric_id"]): normalize_name(metric["name"]) for metric in bank}
    gold_ids = [str(row["metric_id"]) for row in labels]
    result = retrieval_metrics(
        scores,
        gold_ids,
        bank_ids,
        family,
        ks=(1, 3, 5, 10, 16, 30, 50, 80, 120, 180),
    )
    bank_index = {metric_id: idx for idx, metric_id in enumerate(bank_ids)}
    result["items"] = [
        {
            "norm_uid": str(label["norm_uid"]),
            "corpus": str(label.get("corpus") or ""),
            "human_panel": str(label.get("human_panel") or "UNSPECIFIED"),
            "metric_id": gold_id,
            "exact_rank": stable_rank(scores[index]).index(bank_index[gold_id]) + 1,
        }
        for index, (label, gold_id) in enumerate(zip(labels, gold_ids))
    ]
    for field, output_key in (("human_panel", "by_human_panel"), ("corpus", "by_corpus")):
        groups = sorted({str(row.get(field) or "UNSPECIFIED") for row in labels})
        result[output_key] = {}
        for group in groups:
            indices = [
                index
                for index, row in enumerate(labels)
                if str(row.get(field) or "UNSPECIFIED") == group
            ]
            result[output_key][group] = retrieval_metrics(
                scores[indices],
                [gold_ids[index] for index in indices],
                bank_ids,
                family,
                ks=(1, 3, 5, 10, 16, 30, 50, 80, 120, 180),
            )
    return result


def paired_comparison(
    before: Mapping[str, Any], after: Mapping[str, Any], ks: Sequence[int]
) -> dict[str, Any]:
    before_items = {row["norm_uid"]: row for row in before["items"]}
    after_items = {row["norm_uid"]: row for row in after["items"]}
    if before_items.keys() != after_items.keys():
        raise ValueError("base and adapter item sets differ")
    output: dict[str, Any] = {}
    for k in ks:
        both = before_only = after_only = neither = 0
        rank_improved = rank_worsened = rank_tied = 0
        for uid in sorted(before_items):
            base_rank = int(before_items[uid]["exact_rank"])
            adapter_rank = int(after_items[uid]["exact_rank"])
            base_hit, adapter_hit = base_rank <= k, adapter_rank <= k
            both += int(base_hit and adapter_hit)
            before_only += int(base_hit and not adapter_hit)
            after_only += int(adapter_hit and not base_hit)
            neither += int(not base_hit and not adapter_hit)
            rank_improved += int(adapter_rank < base_rank)
            rank_worsened += int(adapter_rank > base_rank)
            rank_tied += int(adapter_rank == base_rank)
        output[f"at_{k}"] = {
            "both_hit": both,
            "base_only_hit": before_only,
            "adapter_only_hit": after_only,
            "neither_hit": neither,
            "net_additional_hits": after_only - before_only,
            "rank_improved": rank_improved,
            "rank_worsened": rank_worsened,
            "rank_tied": rank_tied,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default=DEFAULT_NEMOTRON)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("auto", "eager", "sdpa"), default="eager")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-dev-recall-gain", type=float, default=0.03)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    label_path = Path(args.labels).resolve()
    adapter_path = Path(args.adapter).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen evaluation: {output_path}")
    labels, norms, bank, bank_sha = load_eval(
        manifest_path, label_path, args.task, args.split
    )
    os.environ.setdefault(
        "HF_MODULES_CACHE", "/lfs/skampere3/0/alexspan/.cache/huggingface/modules"
    )
    import torch
    from sentence_transformers import SentenceTransformer

    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if args.attention != "auto":
        model_kwargs["attn_implementation"] = args.attention
    base = SentenceTransformer(
        args.model,
        device=args.device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
    )
    base.max_seq_length = args.max_seq_length
    before = score_model(base, labels, norms, bank, args.batch_size)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    trained = load_nemotron_adapter(
        args.model,
        adapter_path,
        device=args.device,
        attention=args.attention,
        max_seq_length=args.max_seq_length,
    )
    after = score_model(trained, labels, norms, bank, args.batch_size)
    paired = paired_comparison(before, after, (10, 16, 30, 50, 80))
    dev_gain = after["exact"]["recall_at_50"] - before["exact"]["recall_at_50"]
    no_wide_loss = after["exact"]["recall_at_80"] >= before["exact"]["recall_at_80"]
    dev_gate = (
        {
            "primary_metric": "exact.recall_at_50",
            "minimum_gain": args.min_dev_recall_gain,
            "actual_gain": dev_gain,
            "secondary_requirement": "exact.recall_at_80 must not decrease",
            "secondary_passed": no_wide_loss,
            "passed": dev_gain >= args.min_dev_recall_gain and no_wide_loss,
        }
        if args.split == "dev"
        else None
    )
    report = {
        "task": args.task,
        "split": args.split,
        "selection_role": "promotion_dev" if args.split == "dev" else "frozen_test_once",
        "n_match_labels": len(labels),
        "bank_metrics": len(bank),
        "bank_source_sha256": bank_sha,
        "before": before,
        "after": after,
        "paired": paired,
        "promotion_gate": dev_gate,
        "delta": {
            key: after["exact"].get(key) - before["exact"].get(key)
            for key in (
                "recall_at_10",
                "recall_at_16",
                "recall_at_30",
                "recall_at_50",
                "recall_at_80",
                "mrr",
            )
        },
        "input_hashes": {
            "manifest": sha256_file(manifest_path),
            "labels": sha256_file(label_path),
            "adapter": {
                path.name: sha256_file(path)
                for path in sorted(adapter_path.iterdir())
                if path.is_file()
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
