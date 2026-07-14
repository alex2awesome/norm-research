#!/usr/bin/env python
"""Freeze cross-fitted item-adaptive demonstrations of the larger Llama policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.policy_data import (
    _align_orbit,
    _average_repetitions,
    _orbits,
    load_public_index,
)


CELL_ID = "N_humor_49"
PARTITIONS = ("residual_prompt_selection", "residual_unit_certification")
RETRIEVAL_RECIPES = (
    ("word_policy_k2", "word", "policy", 2),
    ("char_policy_k2", "char", "policy", 2),
    ("hybrid_policy_k1", "hybrid", "policy", 1),
    ("hybrid_policy_k2", "hybrid", "policy", 2),
    ("hybrid_policy_k4", "hybrid", "policy", 4),
    ("hybrid_residual_k1", "hybrid", "residual", 1),
    ("hybrid_residual_k2", "hybrid", "residual", 2),
    ("hybrid_residual_k4", "hybrid", "residual", 4),
)
PARENTS = ("demos_only", "source_explanation", "self_contrastive",
           "behavior_contrastive")


def _canonical(arm: dict) -> str:
    return next(form["prompt"] for form in arm["forms"] if form["id"] == "canonical")


def _items(packet_root: str | Path, partition: str) -> list[dict]:
    path = Path(packet_root) / "humor" / "items" / f"{partition}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _truncate(text: str, words: int = 80) -> str:
    tokens = str(text).split()
    return str(text) if len(tokens) <= words else " ".join(tokens[:words]) + " …"


def similarity_matrices(train_texts: list[str], test_texts: list[str]) -> dict[str, np.ndarray]:
    """Training-vocabulary-only lexical retrieval; test scores/labels are never inputs."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    word = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=1, max_features=50_000,
        sublinear_tf=True, norm="l2")
    char = TfidfVectorizer(
        analyzer="char_wb", lowercase=True, ngram_range=(3, 5), min_df=1,
        max_features=75_000, sublinear_tf=True, norm="l2")
    word_train = word.fit_transform(train_texts)
    word_test = word.transform(test_texts)
    char_train = char.fit_transform(train_texts)
    char_test = char.transform(test_texts)
    word_similarity = (word_test @ word_train.T).toarray()
    char_similarity = (char_test @ char_train.T).toarray()
    return {
        "word": word_similarity,
        "char": char_similarity,
        "hybrid": 0.5 * word_similarity + 0.5 * char_similarity,
    }


def retrieve_assignments(*, train_hashes: list[str], test_hashes: list[str],
                         similarity: np.ndarray, target: np.ndarray,
                         sparse: np.ndarray, pool: str, k: int) -> dict[str, dict]:
    target = np.asarray(target, float)
    sparse = np.asarray(sparse, float)
    if similarity.shape != (len(test_hashes), len(train_hashes)):
        raise ValueError("similarity matrix does not align with item hashes")
    positive = np.flatnonzero(target >= 0.5)
    negative = np.flatnonzero(target < 0.5)
    if min(len(positive), len(negative)) < k:
        raise ValueError("insufficient positive or negative teaching examples")
    if pool == "policy":
        positive_priority = np.abs(target - 0.5) * 2.0
        negative_priority = positive_priority
    elif pool == "residual":
        positive_priority = np.maximum(target - sparse, 0.0)
        negative_priority = np.maximum(sparse - target, 0.0)
    else:
        raise ValueError(f"unknown teaching pool {pool}")

    def choose(row: np.ndarray, indexes: np.ndarray, priority: np.ndarray) -> list[str]:
        # Semantic proximity dominates; confidence/residual strength breaks weak-neighbor ties.
        score = 0.85 * row + 0.15 * priority
        ranked = sorted(indexes, key=lambda index: (
            -float(score[index]), train_hashes[index]))
        return [train_hashes[index] for index in ranked[:k]]

    return {
        test_hash: {
            "positive": choose(similarity[test_index], positive, positive_priority),
            "negative": choose(similarity[test_index], negative, negative_priority),
        }
        for test_index, test_hash in enumerate(test_hashes)
    }


def compile_bank(*, shard_root: str, packet_root: str, source_bank_path: str,
                 rule_bank_path: str) -> dict:
    source_bank = json.loads(Path(source_bank_path).read_text())
    rule_bank = json.loads(Path(rule_bank_path).read_text())
    source_cell = next(cell for cell in source_bank["cells"] if cell["id"] == CELL_ID)
    rule_cell = next(cell for cell in rule_bank["cells"] if cell["id"] == CELL_ID)
    source_arms = {arm["id"]: arm for arm in source_cell["arms"]}
    rule_arms = {arm["id"]: arm for arm in rule_cell["arms"]}
    indexes = {partition: load_public_index(shard_root, partition)
               for partition in PARTITIONS}
    evaluations = []
    all_arm_specs = {}
    for evaluation_partition in PARTITIONS:
        teaching_partition = next(
            partition for partition in PARTITIONS if partition != evaluation_partition)
        teaching_rows = _items(packet_root, teaching_partition)
        evaluation_rows = _items(packet_root, evaluation_partition)
        teaching_by_hash = {row["text_sha256"]: row for row in teaching_rows}
        teaching_data_big = _average_repetitions(
            indexes[teaching_partition][("llama8_big_sparse", "humor")])
        teaching_data_small = _average_repetitions(
            indexes[teaching_partition][("llama3_small", "humor")])
        big_orbit = _orbits(teaching_data_big["scores"], teaching_data_big["meta"],
                            cell_id=CELL_ID)["name"]
        small_orbit = _align_orbit(
            _orbits(teaching_data_small["scores"], teaching_data_small["meta"],
                    cell_id=CELL_ID)["name"],
            teaching_data_small["hashes"], teaching_data_big["hashes"])
        teaching_hashes = teaching_data_big["hashes"]
        target = np.mean(np.stack(list(big_orbit.values())), axis=0)
        sparse = np.mean(np.stack(list(small_orbit.values())), axis=0)
        if not set(teaching_hashes) <= set(teaching_by_hash):
            raise ValueError("teaching scores and text packet differ")
        teaching_texts = [teaching_by_hash[value]["text"] for value in teaching_hashes]
        evaluation_hashes = [row["text_sha256"] for row in evaluation_rows]
        evaluation_texts = [row["text"] for row in evaluation_rows]
        if set(teaching_hashes) & set(evaluation_hashes):
            raise ValueError("teaching and evaluation partitions overlap")
        similarities = similarity_matrices(teaching_texts, evaluation_texts)
        teaching_examples = {
            item_hash: {
                "text": _truncate(teaching_by_hash[item_hash]["text"]),
                "target_score": float(target[index]),
                "small_name_score": float(sparse[index]),
            }
            for index, item_hash in enumerate(teaching_hashes)
        }
        retrievals = []
        for retrieval_id, method, pool, k in RETRIEVAL_RECIPES:
            retrievals.append({
                "id": retrieval_id,
                "method": method,
                "pool": pool,
                "k_per_polarity": k,
                "assignments": retrieve_assignments(
                    train_hashes=teaching_hashes, test_hashes=evaluation_hashes,
                    similarity=similarities[method], target=target, sparse=sparse,
                    pool=pool, k=k),
            })
        parent_texts = {
            "demos_only": "",
            "source_explanation": _canonical(source_arms["source_explanation"]),
            "self_contrastive": _canonical(rule_arms["rule_contrastive_v0_from_self"]),
            "behavior_contrastive": _canonical(
                rule_arms[
                    f"rule_contrastive_v1_from_{teaching_partition.removeprefix('residual_')}"]),
        }
        arms = []
        teaching_suffix = teaching_partition.removeprefix("residual_")
        average_words = float(np.mean([len(value["text"].split())
                                      for value in teaching_examples.values()]))
        for parent_id in PARENTS:
            for retrieval_id, _method, _pool, k in RETRIEVAL_RECIPES:
                arm_id = f"{parent_id}_plus_{retrieval_id}_from_{teaching_suffix}"
                estimated_words = int(round(
                    len(parent_texts[parent_id].split()) + 2 * k * average_words + 55))
                spec = {
                    "id": arm_id,
                    "channel": "item_adaptive_ostensive",
                    "provenance": "ostensive_teaching",
                    "source_partition": teaching_partition,
                    "parent_id": parent_id,
                    "retrieval_id": retrieval_id,
                    "semantic_content_word_count": estimated_words,
                    "forms": [{"id": form, "prompt": (
                        f"Dynamic {form} template: {parent_id} + {retrieval_id}")}
                              for form in ("canonical", "question", "boilerplate")],
                }
                arms.append(spec)
                all_arm_specs[arm_id] = spec
        evaluations.append({
            "evaluation_partition": evaluation_partition,
            "teaching_partition": teaching_partition,
            "evaluation_item_sha256": evaluation_hashes,
            "teaching_examples": teaching_examples,
            "retrievals": retrievals,
            "parent_texts": parent_texts,
            "arms": arms,
            "teaching_target_shards": teaching_data_big["shard_sha256"],
            "teaching_small_shards": teaching_data_small["shard_sha256"],
        })
    payload = {
        "schema": "adaptive_ostensive_bank/v1",
        "status": "frozen-before-adaptive-small-executor-scoring",
        "objective": "maximize exact 3B reconstruction of fixed 8B name-only policy",
        "model_family": "Llama only: Llama-3.2-3B executor, Llama-3.1-8B target",
        "anchor_policy": "8B name-only soft policy; no human or external label",
        "leakage_boundary": ("Each evaluation prompt uses only its item text plus examples and "
                             "scores from the opposite public fold; evaluation target scores are "
                             "not read during compilation or scoring."),
        "retrieval_fit": ("TF-IDF vocabulary and teaching priorities use the opposite fold only; "
                          "evaluation item text is transformed without fitting."),
        "source_artifacts": {
            "source_bank": {"path": source_bank_path,
                            "sha256": sha256_file(source_bank_path)},
            "rule_bank": {"path": rule_bank_path, "sha256": sha256_file(rule_bank_path)},
        },
        "evaluations": evaluations,
        "cells": [{
            "id": CELL_ID,
            "domain": "humor",
            "gi": 49,
            "construct": "Wordplay quality and clarity",
            "arms": sorted(all_arm_specs.values(), key=lambda row: row["id"]),
        }],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bank_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--rule-bank", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compile_bank(shard_root=args.shard_root, packet_root=args.packet_root,
                          source_bank_path=args.source_bank, rule_bank_path=args.rule_bank)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256_file(out),
                      "bank_content_sha256": result["bank_content_sha256"],
                      "n_evaluations": len(result["evaluations"]),
                      "n_arms_per_evaluation": [len(row["arms"])
                                                for row in result["evaluations"]]}, indent=1))


if __name__ == "__main__":
    main()
