#!/usr/bin/env python
"""Cross-fit full-text revisions against residual errors of the fixed larger sparse policy.

The writer is the same 8B model whose name-only soft policy is being reconstructed.  Each revision
uses only one public fold's target/executor disagreement and is evaluated only on the other fold.
No human/corpus label, external judge, or third-model target is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file, text_sha256
from methods.codability.experiments.compile_residual_isomorphism_bank import (
    BEST_SOURCE,
    DOMAIN_DIR,
    MAX_EXAMPLE_WORDS,
    PARTITIONS,
    _eligible_teaching_item,
)
from methods.codability.experiments.policy_data import (
    _align_orbit,
    _average_repetitions,
    _orbits,
    load_public_index,
)
from methods.codability.experiments.policy_isomorphism import _orbit_point
from methods.codability.experiments.synthesize_target_policy_rules import (
    _diverse_take,
    _truncate,
    policy_calibration,
)
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend


REVISION_VIEWS = {
    "minimal_repair": (
        "Preserve every useful distinction in the current specification, but repair the smallest "
        "set of rules needed to correct the systematic over- and under-judgments."
    ),
    "rank_repair": (
        "Prioritize item ordering: articulate why items the target scores higher should outrank "
        "items the executor currently favors. Encode interactions and tie-breakers, not examples."
    ),
    "gestalt_repair": (
        "Infer the tacit holistic gestalt behind the residual pattern, including subtle social or "
        "normative cues, compensating strengths, vetoes, and boundary conditions."
    ),
}
N_VARIANTS = 2


def identity_loss(point: dict) -> float:
    candidate = point["candidate_robust"]
    target = point["target_self_robust"]
    return float(
        max(candidate["mae_tvd"] - target["mae_tvd"], 0.0) / 0.02
        + max(target["spearman"] - candidate["spearman"], 0.0) / 0.05
        + max(candidate["binary_flip_rate"] - target["binary_flip_rate"], 0.0) / 0.02
        + max(candidate["absolute_bias"] - target["absolute_bias"], 0.0) / 0.02
    )


def choose_parents(rows: list[dict], *, incumbent_id: str, max_parents: int = 3) -> list[dict]:
    """Keep the intact incumbent plus MAE- and rank-leading textual parents."""
    if max_parents < 1:
        raise ValueError("max_parents must be positive")
    incumbent = next(row for row in rows if row["arm_id"] == incumbent_id)
    min_mae = min(rows, key=lambda row: (row["point"]["candidate_robust"]["mae_tvd"],
                                         row["arm_id"]))
    mae_cut = incumbent["point"]["candidate_robust"]["mae_tvd"] + 0.03
    eligible_rank = [row for row in rows
                     if row["point"]["candidate_robust"]["mae_tvd"] <= mae_cut]
    max_rank = max(eligible_rank, key=lambda row: (
        row["point"]["candidate_robust"]["spearman"], -identity_loss(row["point"]),
        row["arm_id"]))
    selected = []
    for row in (incumbent, min_mae, max_rank):
        if row["arm_id"] not in {old["arm_id"] for old in selected}:
            selected.append(row)
        if len(selected) >= max_parents:
            break
    # If priority roles collapse to the same arm, fill the remaining slots explicitly by joint
    # identity loss.  This replaces the old fourth tuple element, which was silently skipped
    # whenever the three priority roles were distinct and could not fill max_parents > 3.
    if len(selected) < max_parents:
        for row in sorted(rows, key=lambda value: (
                identity_loss(value["point"]), value["arm_id"])):
            if row["arm_id"] not in {old["arm_id"] for old in selected}:
                selected.append(row)
            if len(selected) >= max_parents:
                break
    return selected


def residual_panel(rows: list[dict], *, n_direction: int = 3,
                   n_pairs: int = 2) -> dict:
    """Select directional residuals and severe rank inversions, deterministically."""
    hashes: set[str] = set()
    positive = [{**row, "priority": max(row["target"] - row["executor"], 0.0)}
                for row in rows if row["target"] > row["executor"]]
    negative = [{**row, "priority": max(row["executor"] - row["target"], 0.0)}
                for row in rows if row["executor"] > row["target"]]
    under = _diverse_take(positive, n_direction, hashes)
    over = _diverse_take(negative, n_direction, hashes)
    inversions = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            high, low = (left, right) if left["target"] >= right["target"] else (right, left)
            target_gap = high["target"] - low["target"]
            executor_reversal = low["executor"] - high["executor"]
            if target_gap >= 0.15 and executor_reversal > 0:
                inversions.append({"high": high, "low": low,
                                   "priority": float(target_gap * executor_reversal)})
    inversions.sort(key=lambda pair: (-pair["priority"], pair["high"]["text_sha256"],
                                      pair["low"]["text_sha256"]))
    pairs, used = [], set(hashes)
    for pair in inversions:
        pair_hashes = {pair["high"]["text_sha256"], pair["low"]["text_sha256"]}
        if pair_hashes & used:
            continue
        pairs.append(pair)
        used |= pair_hashes
        if len(pairs) == n_pairs:
            break
    return {"underpredicted": under, "overpredicted": over, "rank_inversions": pairs}


def revision_prompt(name: str, parent_text: str, instruction: str,
                    panel: dict, calibration: dict) -> str:
    parts = [
        "You are revising an explicit criterion so a smaller evaluator reproduces your own fixed "
        "name-only item policy. There is no external ground truth; the target scores below are "
        "the entire reconstruction target.",
        f"Criterion name: {name}",
        f"Current specification:\n---\n{parent_text}\n---",
        instruction,
        (f"Policy strictness on this varied teaching panel: mean YES propensity "
         f"{calibration['mean_p_yes']:.2f}; clear-YES rate "
         f"{100 * calibration['binary_positive_rate']:.0f}%. This describes strictness, not a "
         "quota."),
        "Cases the current executor scores too LOW:",
    ]
    for index, row in enumerate(panel["underpredicted"], 1):
        parts.append(f"[U{index}: target {row['target']:.2f}; executor "
                     f"{row['executor']:.2f}]\n{row['text']}")
    parts.append("Cases the current executor scores too HIGH:")
    for index, row in enumerate(panel["overpredicted"], 1):
        parts.append(f"[O{index}: target {row['target']:.2f}; executor "
                     f"{row['executor']:.2f}]\n{row['text']}")
    parts.append("Rank reversals to repair:")
    for index, pair in enumerate(panel["rank_inversions"], 1):
        high, low = pair["high"], pair["low"]
        parts.append(
            f"[R{index} HIGH: target {high['target']:.2f}; executor {high['executor']:.2f}]\n"
            f"{high['text']}\n[R{index} LOW: target {low['target']:.2f}; executor "
            f"{low['executor']:.2f}]\n{low['text']}")
    parts.append(
        "Return only a standalone revised criterion specification of 180–300 words. Generalize "
        "the missing distinctions; do not mention examples, scores, models, datasets, revisions, "
        "or quotas, and do not copy distinctive phrases from the cases."
    )
    return "\n\n".join(parts)


def _load_items(packet_root: str | Path, domain: str, partition: str) -> dict[str, dict]:
    path = Path(packet_root) / DOMAIN_DIR[domain] / "items" / f"{partition}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["text_sha256"]: row for row in rows}


def build_requests(*, target_shard_root: str, source_shard_root: str,
                   rule_shard_root: str, source_bank_path: str, rule_bank_path: str,
                   packet_root: str, target_job: str = "llama8_big_sparse",
                   source_job: str = "llama3_small",
                   rule_job: str = "llama3_target_policy_rules") -> tuple[list[dict], dict]:
    source_bank = json.loads(Path(source_bank_path).read_text())
    rule_bank = json.loads(Path(rule_bank_path).read_text())
    source_cells = {cell["id"]: cell for cell in source_bank["cells"]}
    rule_cells = {cell["id"]: cell for cell in rule_bank["cells"]}
    requests, selections = [], {}
    for partition in PARTITIONS:
        target_index = load_public_index(target_shard_root, partition)
        source_index = load_public_index(source_shard_root, partition)
        rule_index = load_public_index(rule_shard_root, partition)
        for cell_id, source_cell in source_cells.items():
            domain, name = source_cell["domain"], source_cell["construct"]
            source_specs = {arm["id"]: arm for arm in source_cell["arms"]}
            rule_specs = {arm["id"]: arm for arm in rule_cells[cell_id]["arms"]}
            target_data = _average_repetitions(target_index[(target_job, domain)])
            source_data = _average_repetitions(source_index[(source_job, domain)])
            rule_data = _average_repetitions(rule_index[(rule_job, domain)])
            target_orbits = _orbits(target_data["scores"], target_data["meta"], cell_id=cell_id)
            source_orbits = _orbits(source_data["scores"], source_data["meta"], cell_id=cell_id)
            rule_orbits = _orbits(rule_data["scores"], rule_data["meta"], cell_id=cell_id)
            hashes = target_data["hashes"]
            target = target_orbits["name"]
            q = np.mean(np.stack(list(target.values())), axis=0)
            parent_rows = []
            incumbent_id = BEST_SOURCE[cell_id]
            incumbent = _align_orbit(source_orbits[incumbent_id], source_data["hashes"], hashes)
            parent_rows.append({"arm_id": incumbent_id,
                                "parent_provenance": "intact_source",
                                "content": next(form["prompt"] for form in
                                                source_specs[incumbent_id]["forms"]
                                                if form["id"] == "canonical"),
                                "orbit": incumbent,
                                "point": _orbit_point(target, incumbent)})
            for arm_id, orbit in rule_orbits.items():
                if arm_id == "name":
                    continue
                spec = rule_specs[arm_id]
                # A parent may be fold-independent or derived from this teaching fold, never from
                # the opposite (future evaluation) fold.
                if spec.get("source_partition") not in (None, partition):
                    continue
                aligned = _align_orbit(orbit, rule_data["hashes"], hashes)
                parent_rows.append({
                    "arm_id": arm_id, "parent_provenance": spec["provenance"],
                    "content": next(form["prompt"] for form in spec["forms"]
                                    if form["id"] == "canonical"),
                    "orbit": aligned, "point": _orbit_point(target, aligned),
                })
            parents = choose_parents(parent_rows, incumbent_id=incumbent_id)
            items = _load_items(packet_root, domain, partition)
            calibration = policy_calibration(q)
            selection_key = f"{cell_id}:{partition}"
            selections[selection_key] = []
            for parent_slot, parent in enumerate(parents):
                p = np.mean(np.stack(list(parent["orbit"].values())), axis=0)
                rows = []
                for item_hash, target_score, executor_score in zip(hashes, q, p):
                    text = items[item_hash]["text"]
                    if _eligible_teaching_item(domain, text):
                        rows.append({
                            "text_sha256": item_hash,
                            "text": _truncate(text, MAX_EXAMPLE_WORDS[domain]),
                            "target": float(target_score), "executor": float(executor_score),
                        })
                panel = residual_panel(rows)
                panel_hashes = [row["text_sha256"] for kind in
                                ("underpredicted", "overpredicted") for row in panel[kind]]
                panel_hashes += [row[side]["text_sha256"]
                                 for row in panel["rank_inversions"]
                                 for side in ("high", "low")]
                selections[selection_key].append({
                    "parent_slot": parent_slot,
                    "parent_arm_id": parent["arm_id"],
                    "parent_provenance": parent["parent_provenance"],
                    "parent_identity_loss": identity_loss(parent["point"]),
                    "parent_robust": parent["point"]["candidate_robust"],
                    "teaching_item_sha256": panel_hashes,
                })
                for view_id, instruction in REVISION_VIEWS.items():
                    for variant in range(N_VARIANTS):
                        requests.append({
                            "cell_id": cell_id, "domain": domain, "construct": name,
                            "source_partition": partition,
                            "parent_slot": parent_slot,
                            "parent_arm_id": parent["arm_id"],
                            "parent_provenance": parent["parent_provenance"],
                            "view": view_id, "variant": variant,
                            "prompt": revision_prompt(name, parent["content"], instruction,
                                                      panel, calibration),
                            "teaching_item_sha256": panel_hashes,
                            "calibration": calibration,
                        })
    return requests, selections


def synthesize(*, backend, requests: list[dict], writer_model: str,
               seed: int = 20260717) -> list[dict]:
    seeds = [seed + 1009 * index + 7919 * row["variant"]
             for index, row in enumerate(requests)]
    outputs = backend.generate_batch(
        [row["prompt"] for row in requests], max_tokens=460, temperature=0.7, seed=seeds,
        validate=lambda value: 50 <= len(str(value).split()) <= 420,
    )
    result = []
    for request, request_seed, output in zip(requests, seeds, outputs):
        value = str(output).strip()
        result.append({**request, "prompt_sha256": text_sha256(request["prompt"]),
                       "seed": request_seed, "writer_model": writer_model,
                       "articulation": value, "articulation_sha256": text_sha256(value),
                       "articulation_word_count": len(value.split())})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-shard-root", required=True)
    parser.add_argument("--source-shard-root", required=True)
    parser.add_argument("--rule-shard-root", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--rule-bank", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--writer-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    requests, selections = build_requests(
        target_shard_root=args.target_shard_root, source_shard_root=args.source_shard_root,
        rule_shard_root=args.rule_shard_root, source_bank_path=args.source_bank,
        rule_bank_path=args.rule_bank, packet_root=args.packet_root)
    config = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
    if args.fake:
        config.vllm_fake = True
    backend = make_judge_backend(args.writer_model, config, temperature=0.7)
    rows = synthesize(backend=backend, requests=requests, writer_model=args.writer_model,
                      seed=args.seed)
    payload = {
        "schema": "residual_policy_revision/v1",
        "status": "generated-before-revised-small-executor-scoring",
        "objective": "cross-fitted full-text evolution toward direct larger-policy isomorphism",
        "anchor_policy": "fixed 8B name-only behavior; no external label or evaluator",
        "writer_model": args.writer_model,
        "source_bank": {"path": args.source_bank, "sha256": sha256_file(args.source_bank)},
        "rule_bank": {"path": args.rule_bank, "sha256": sha256_file(args.rule_bank)},
        "packet_manifest": {"path": args.packet_manifest,
                            "sha256": sha256_file(args.packet_manifest)},
        "fold_policy": "revision from one public fold is evaluated only on the opposite fold",
        "parent_selections": selections,
        "rows": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256_file(out),
                      "n_requests": len(requests), "n_rows": len(rows)}, indent=1))


if __name__ == "__main__":
    main()
