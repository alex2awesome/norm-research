#!/usr/bin/env python
"""Score public items comparatively with Llama-3B and recover pairwise policy orbits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.policy_data import (
    PUBLIC_DEVELOPMENT_PARTITIONS,
    require_partition,
)
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend


PAIR_FORMS = ("canonical", "question", "boilerplate")


def circulant_edges(n_items: int, *, degree: int, seed: int) -> np.ndarray:
    """A deterministic random-permuted regular comparison graph."""
    if degree <= 0 or degree % 2 or degree >= n_items:
        raise ValueError("degree must be positive, even, and smaller than n_items")
    permutation = np.random.default_rng(seed).permutation(n_items)
    edges = set()
    for offset in range(1, degree // 2 + 1):
        for position, left in enumerate(permutation):
            right = permutation[(position + offset) % n_items]
            edges.add(tuple(sorted((int(left), int(right)))))
    result = np.asarray(sorted(edges), dtype=int)
    counts = np.bincount(result.ravel(), minlength=n_items)
    if not np.all(counts == degree):
        raise AssertionError("comparison graph is not regular")
    return result


def pair_prompt(content: str, left: str, right: str, *, form: str) -> str:
    if form == "canonical":
        framing = f"Criterion:\n{content}"
    elif form == "question":
        framing = f"Which item better satisfies the following criterion?\n{content}"
    elif form == "boilerplate":
        framing = f"You are an expert evaluator. Compare strictly on this criterion:\n{content}"
    else:
        raise ValueError(f"unknown pair form {form}")
    return (
        f"{framing}\n\nItem A:\n{left}\n\nItem B:\n{right}\n\n"
        "Which item better satisfies the criterion? Answer with exactly one letter: A or B."
    )


def score_ab(backend, prompts: list[str], *, seed: int) -> np.ndarray:
    """Return normalized P(A), P(B), including older cluster-backend compatibility."""
    if hasattr(backend, "score_choices"):
        return np.asarray(backend.score_choices(prompts, ["A", "B"], seed=seed), float)
    probability_a = np.asarray(
        backend.score_binary(prompts, pos="A", neg="B", seed=seed), float)
    return np.column_stack([probability_a, 1.0 - probability_a])


def recover_pairwise_scores(edges: np.ndarray, forward: np.ndarray, reverse: np.ndarray,
                            *, n_items: int, bt_ridge: float = 0.1,
                            bt_max_iter: int = 100, bt_tolerance: float = 1e-10) -> dict:
    """Order-symmetrize comparisons and recover Borda plus Bradley–Terry coordinates."""
    forward = np.asarray(forward, float)
    reverse = np.asarray(reverse, float)
    if forward.shape != (len(edges), 2) or reverse.shape != forward.shape:
        raise ValueError("choice probabilities do not align with edges")
    # Forward: left is A. Reverse: left is B. Both terms below are P(left wins).
    left_forward = forward[:, 0]
    left_reverse = reverse[:, 1]
    valid = np.isfinite(left_forward) & np.isfinite(left_reverse)
    probability = np.where(valid, 0.5 * (left_forward + left_reverse), 0.5)
    order_disagreement = np.abs(left_forward[valid] - left_reverse[valid])

    wins = np.zeros(n_items, float)
    comparisons = np.zeros(n_items, float)
    for (left, right), value in zip(edges, probability):
        wins[left] += value
        wins[right] += 1.0 - value
        comparisons[left] += 1.0
        comparisons[right] += 1.0
    borda = np.divide(wins, comparisons, out=np.full(n_items, 0.5), where=comparisons > 0)

    # Maximize the Bradley--Terry likelihood with fractional outcomes equal to the
    # order-symmetrized choice probabilities.  Ridge fixes the additive location and
    # keeps nearly separated comparison graphs numerically well behaved.
    latent = np.zeros(n_items, float)
    left_index, right_index = edges[:, 0], edges[:, 1]
    for _iteration in range(bt_max_iter):
        difference = np.clip(latent[left_index] - latent[right_index], -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-difference))
        residual = probability - fitted
        gradient = np.zeros(n_items, float)
        np.add.at(gradient, left_index, residual)
        np.add.at(gradient, right_index, -residual)
        gradient -= bt_ridge * latent
        weight = fitted * (1.0 - fitted)
        information = bt_ridge * np.eye(n_items)
        np.add.at(information, (left_index, left_index), weight)
        np.add.at(information, (right_index, right_index), weight)
        np.add.at(information, (left_index, right_index), -weight)
        np.add.at(information, (right_index, left_index), -weight)
        step = np.linalg.solve(information, gradient)
        latent += step
        latent -= np.mean(latent)
        if float(np.max(np.abs(step))) <= bt_tolerance:
            break
    latent -= np.mean(latent)
    bt_score = 1.0 / (1.0 + np.exp(-np.clip(latent, -20.0, 20.0)))
    return {
        "borda": borda,
        "bradley_terry": bt_score,
        "edge_left_probability": probability,
        "nan_pair_rate": float(1.0 - np.mean(valid)),
        "mean_order_disagreement": (float(np.mean(order_disagreement))
                                    if len(order_disagreement) else None),
    }


def _load_items(packet_root: str | Path, partition: str) -> list[dict]:
    path = Path(packet_root) / "humor" / "items" / f"{partition}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_partition(*, backend, bank: dict, packet_root: str, partition: str,
                    degree: int, seed: int, max_item_chars: int) -> dict:
    require_partition(
        partition,
        allowed=PUBLIC_DEVELOPMENT_PARTITIONS,
        operation="pairwise policy scoring",
    )
    items = _load_items(packet_root, partition)
    hashes = [row["text_sha256"] for row in items]
    texts = [row["text"][:max_item_chars] for row in items]
    edges = circulant_edges(len(items), degree=degree, seed=seed)
    borda_rows, bt_rows, meta, diagnostics = [], [], [], []
    for arm_index, arm in enumerate(bank["cell"]["arms"]):
        for form_index, form in enumerate(PAIR_FORMS):
            forward_prompts, reverse_prompts = [], []
            for left, right in edges:
                forward_prompts.append(pair_prompt(
                    arm["content"], texts[left], texts[right], form=form))
                reverse_prompts.append(pair_prompt(
                    arm["content"], texts[right], texts[left], form=form))
            score_seed = seed + 1009 * arm_index + form_index
            forward = score_ab(backend, forward_prompts, seed=score_seed)
            reverse = score_ab(backend, reverse_prompts, seed=score_seed)
            recovered = recover_pairwise_scores(
                edges, forward, reverse, n_items=len(items))
            borda_rows.append(recovered["borda"])
            bt_rows.append(recovered["bradley_terry"])
            meta.append({
                "cell_id": bank["cell"]["id"], "domain": bank["cell"]["domain"],
                "construct": bank["cell"]["construct"], "arm_id": arm["id"],
                "form": form, "source_partition": arm["source_partition"],
                "channel": arm["channel"], "provenance": arm["provenance"],
                "content_sha256": arm["content_sha256"],
            })
            diagnostics.append({
                "arm_id": arm["id"], "form": form,
                "nan_pair_rate": recovered["nan_pair_rate"],
                "mean_order_disagreement": recovered["mean_order_disagreement"],
            })
    return {
        "partition": partition,
        "hashes": hashes,
        "edges": edges,
        "borda": np.asarray(borda_rows),
        "bradley_terry": np.asarray(bt_rows),
        "meta": meta,
        "diagnostics": diagnostics,
    }


def run(*, bank_path: str, packet_root: str, partitions: list[str], out_dir: str,
        model: str = "meta-llama/Llama-3.2-3B-Instruct", degree: int = 12,
        seed: int = 20260723, fake: bool = False) -> dict:
    bank = json.loads(Path(bank_path).read_text())
    if bank.get("status") != "frozen-before-pairwise-small-executor-scoring":
        raise ValueError("pairwise bank is not frozen")
    if len(partitions) != len(set(partitions)):
        raise ValueError("pairwise partitions must be unique")
    for partition in partitions:
        require_partition(
            partition,
            allowed=PUBLIC_DEVELOPMENT_PARTITIONS,
            operation="pairwise policy scoring",
        )
    config = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
    if fake:
        config.vllm_fake = True
    backend = make_judge_backend(model, config, temperature=None)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for partition_index, partition in enumerate(partitions):
        result = score_partition(
            backend=backend, bank=bank, packet_root=packet_root, partition=partition,
            degree=degree, seed=seed + 100_003 * partition_index,
            max_item_chars=max(100, config.max_text_chars // 2))
        out = out_root / f"pairwise_{partition}.npz"
        np.savez_compressed(
            out,
            borda_scores=result["borda"],
            bradley_terry_scores=result["bradley_terry"],
            meta=np.asarray([json.dumps(row, sort_keys=True) for row in result["meta"]],
                            dtype=object),
            probe_sha256=np.asarray(result["hashes"]),
            edges=result["edges"],
            model=model,
            partition=partition,
            degree=degree,
            seed=seed + 100_003 * partition_index,
            bank_sha256=sha256_file(bank_path),
        )
        sidecar = out.with_suffix(".json")
        sidecar.write_text(json.dumps({
            "schema": "pairwise_policy_scores/v1",
            "status": "public-development-only",
            "partition": partition,
            "model": model,
            "model_family": "Llama",
            "n_items": len(result["hashes"]),
            "n_edges": len(result["edges"]),
            "degree": degree,
            "n_score_rows": len(result["meta"]),
            "bank_sha256": sha256_file(bank_path),
            "diagnostics": result["diagnostics"],
        }, indent=1))
        outputs.append({"partition": partition, "npz": str(out), "report": str(sidecar),
                        "n_items": len(result["hashes"]), "n_edges": len(result["edges"])})
    return {"schema": "pairwise_policy_execution/v1", "bank_sha256": sha256_file(bank_path),
            "model": model, "degree": degree, "outputs": outputs,
            "lockbox_status": "not authorized; public residual folds only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--partition", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    result = run(bank_path=args.bank, packet_root=args.packet_root,
                 partitions=args.partition, out_dir=args.out_dir, model=args.model,
                 degree=args.degree, seed=args.seed, fake=args.fake)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
