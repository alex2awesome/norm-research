#!/usr/bin/env python3
"""Distill clean Sonnet MATCH teachers into a task-aware BGE retriever."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from .common import metric_card, norm_query, read_jsonl, sha256_file
from .config import BGE_ENCODER, DEFAULT_OUTPUT_ROOT
from .make_calibration import split_for, split_group_for


def uid_int(uid: str) -> int:
    return int(hashlib.sha256(uid.encode()).hexdigest()[:16], 16)


def recall_at_k(model, examples, banks, ks=(1, 5, 10, 30, 50)):
    by_task: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for query, task, metric_id in examples:
        by_task[task].append((query, metric_id))
    result = {}
    for task, values in sorted(by_task.items()):
        metrics = banks[task]
        cards = [metric_card(metric) for metric in metrics]
        card_emb = model.encode(cards, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        query_emb = model.encode(
            [query for query, _ in values],
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = np.asarray(query_emb) @ np.asarray(card_emb).T
        order = np.argsort(-scores, axis=1)
        gold = [metric_id for _, metric_id in values]
        ids = [metric["metric_id"] for metric in metrics]
        result[task] = {
            "n": len(values),
            **{
                f"recall_at_{k}": float(
                    np.mean([gold[i] in [ids[j] for j in order[i, :k]] for i in range(len(gold))])
                )
                for k in ks
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument("--teachers", default=str(DEFAULT_OUTPUT_ROOT / "teachers/sonnet.jsonl"))
    parser.add_argument("--model", default=BGE_ENCODER)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--hard-negative-pool", type=int, default=8)
    args = parser.parse_args()

    from sentence_transformers import InputExample, SentenceTransformer, losses

    manifest_path, teachers_path = Path(args.manifest), Path(args.teachers)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    norms = {}
    for corpus, meta in manifest["corpora"].items():
        for row in read_jsonl(Path(meta["path"])):
            norms[row["norm_uid"]] = row
    banks = {}
    bank_by_id = {}
    for task, meta in manifest["banks"].items():
        values = json.loads(Path(meta["path"]).read_text(encoding="utf-8"))["metrics"]
        banks[task] = values
        bank_by_id[task] = {metric["metric_id"]: metric for metric in values}

    teachers = [
        row for row in read_jsonl(teachers_path)
        if row["decision"] == "MATCH" and row["norm_uid"] in norms
    ]
    model = SentenceTransformer(args.model)

    # Base-model sibling neighborhoods provide deterministic hard negatives.
    siblings = {}
    for task, metrics in banks.items():
        embeddings = model.encode(
            [metric_card(metric) for metric in metrics],
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = np.asarray(embeddings) @ np.asarray(embeddings).T
        ids = [metric["metric_id"] for metric in metrics]
        for i, metric_id in enumerate(ids):
            order = [j for j in np.argsort(-scores[i]) if j != i]
            siblings[(task, metric_id)] = [ids[j] for j in order[: args.hard_negative_pool]]

    train_examples = []
    eval_examples: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for teacher in teachers:
        uid = teacher["norm_uid"]
        norm = norms[uid]
        task, positive_id = norm["task"], teacher["metric_id"]
        if positive_id not in bank_by_id[task]:
            continue
        query = norm_query(norm)
        # A reviewer/comment/thread may yield several near-duplicate norms.
        # Splitting by the feedback unit prevents those siblings from leaking
        # across train and evaluation.
        split = split_for(split_group_for(norm))
        eval_examples[split].append((query, task, positive_id))
        if split != "train":
            continue
        negatives = siblings[(task, positive_id)]
        negative_id = negatives[uid_int(uid) % len(negatives)]
        example = InputExample(
            texts=[
                query,
                metric_card(bank_by_id[task][positive_id]),
                metric_card(bank_by_id[task][negative_id]),
            ]
        )
        # Give the independently corrected audit/rescue labels more influence.
        repeats = 3 if teacher["label_source"] in {"sonnet_audit", "sonnet_rescue"} else 1
        train_examples.extend([example] * repeats)

    before = {
        split: recall_at_k(model, values, banks)
        for split, values in eval_examples.items()
        if split in {"dev", "test"}
    }
    loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=args.margin,
    )
    warmup = math.ceil(len(loader) * args.epochs * 0.1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(output),
        show_progress_bar=True,
        checkpoint_save_steps=0,
    )
    trained = SentenceTransformer(str(output))
    after = {
        split: recall_at_k(trained, values, banks)
        for split, values in eval_examples.items()
        if split in {"dev", "test"}
    }
    report = {
        "base_model": args.model,
        "output": str(output),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "teachers": str(teachers_path),
        "teachers_sha256": sha256_file(teachers_path),
        "n_match_teachers": len(teachers),
        "n_train_examples_after_weighting": len(train_examples),
        "split_counts": {key: len(value) for key, value in eval_examples.items()},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "before": before,
        "after": after,
    }
    (output / "silver_match_training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
