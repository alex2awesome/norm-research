#!/usr/bin/env python
"""Cross-fit target-explained rank contrasts into compact policy articulations.

The current frontier repairs probability level but loses item order.  This generation mines severe
target/executor rank reversals on one public teaching fold, asks the fixed 8B target to articulate
each local contrast, and compresses the resulting micro-rules into item-independent decision text.
Every fold-indexed articulation is executed only on the opposite public fold.  No external target,
human label, corpus label, lockbox item, or third-model judgment is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file, text_sha256
from methods.codability.experiments.compile_residual_isomorphism_bank import (
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
from methods.codability.experiments.synthesize_target_policy_rules import (
    _jaccard,
    _truncate,
    calibration_text,
    policy_calibration,
)
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend


TARGET_CELL_IDS = ("N_humor_49",)
N_PAIRS = 16
N_VARIANTS = 2
SYNTHESIS_MODES = {
    "rank_lexicon": (
        "Write a compact lexicon of the latent comparison axes. For each axis give its high and "
        "low pole, what observable evidence activates it, and how it interacts with other axes."
    ),
    "rank_decision_list": (
        "Write an ordered conditional decision list for relative judgment. State vetoes, "
        "compensating strengths, boundary conditions, and explicit tie-breakers that determine "
        "which of two plausible items should rank higher."
    ),
    "rank_patch": (
        "Write only the smallest reusable patch needed to repair relative ordering. Preserve the "
        "parent criterion's useful meaning and calibration; add missing distinctions rather than "
        "replacing its evaluative object."
    ),
}


def select_rank_contrasts(rows: list[dict], *, n_pairs: int = N_PAIRS,
                          min_target_gap: float = 0.10) -> list[dict]:
    """Choose disjoint, lexically diverse severe target/executor reversals."""
    candidates = []
    ordered = sorted(rows, key=lambda row: (row["target"], row["text_sha256"]))
    for low_index, low in enumerate(ordered):
        for high in ordered[low_index + 1:]:
            target_gap = high["target"] - low["target"]
            executor_reversal = low["executor"] - high["executor"]
            if target_gap < min_target_gap or executor_reversal <= 0.0:
                continue
            candidates.append({
                "high": high,
                "low": low,
                "target_gap": float(target_gap),
                "executor_reversal": float(executor_reversal),
                "priority": float(target_gap * executor_reversal),
            })
    candidates.sort(key=lambda pair: (
        -pair["priority"], pair["high"]["text_sha256"], pair["low"]["text_sha256"]))

    selected: list[dict] = []
    used: set[str] = set()
    # Greedy selection preserves severe errors while preventing one lexical motif from consuming
    # the whole curriculum. Candidate search is restricted to a deterministic top pool.
    pool = candidates[: min(len(candidates), 240)]
    while pool and len(selected) < n_pairs:
        eligible = [pair for pair in pool
                    if pair["high"]["text_sha256"] not in used
                    and pair["low"]["text_sha256"] not in used]
        if not eligible:
            break
        if not selected:
            choice = eligible[0]
        else:
            choice = max(eligible, key=lambda pair: (
                0.8 * pair["priority"]
                + 0.2 * min(
                    1.0 - max(
                        _jaccard(pair[side]["text"], prior[prior_side]["text"])
                        for prior_side in ("high", "low"))
                    for prior in selected for side in ("high", "low")
                ),
                pair["target_gap"],
                pair["high"]["text_sha256"],
            ))
        selected.append(choice)
        used.add(choice["high"]["text_sha256"])
        used.add(choice["low"]["text_sha256"])
        pool.remove(choice)
    return selected


def contrast_prompt(construct: str, pair: dict) -> str:
    high, low = pair["high"], pair["low"]
    return (
        "You are making one local distinction in your own fixed name-only evaluation policy "
        "explicit. There is no external ground truth: the stated ordering is the target.\n\n"
        f"Criterion: {construct}\n\n"
        f"HIGHER item (target YES propensity {high['target']:.2f}):\n{high['text']}\n\n"
        f"LOWER item (target YES propensity {low['target']:.2f}):\n{low['text']}\n\n"
        "Explain the reusable criterion-specific reason the higher item should outrank the lower. "
        "Name the decisive latent dimension, counter-cue, interaction, exception, or tie-breaker. "
        "Do not substitute general quality, harmlessness, or professionalism unless it is genuinely "
        "part of this named criterion. Return only a 45–100 word micro-rule. Do not mention scores, "
        "models, datasets, examples, or this request, and do not copy distinctive phrases."
    )


def synthesis_prompt(construct: str, parent_text: str, micro_rules: list[str], mode: str,
                     calibration: dict) -> str:
    rules = "\n".join(f"- {value}" for value in micro_rules)
    length = "110–180" if mode == "rank_patch" else "220–340"
    return (
        "You are compressing repeated local rank distinctions from your own fixed name-only policy "
        "into explicit content for a smaller evaluator. The goal is unseen-item order, not merely "
        "mean calibration. No external ground truth is involved.\n\n"
        f"Criterion: {construct}\n\nParent criterion:\n---\n{parent_text}\n---\n\n"
        f"{calibration_text(calibration)}\n\n"
        f"Micro-rules inferred from disjoint rank reversals:\n{rules}\n\n"
        f"{SYNTHESIS_MODES[mode]}\n\n"
        f"Return only a standalone {length} word criterion specification. Preserve the ordinary "
        "meaning indexed by the criterion name. Generalize across the micro-rules, remove duplicate "
        "or spurious cues, and do not mention rules, examples, rankings, scores, models, datasets, "
        "or quotas."
    )


def _load_items(packet_root: str | Path, domain: str, partition: str) -> dict[str, dict]:
    path = Path(packet_root) / DOMAIN_DIR[domain] / "items" / f"{partition}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["text_sha256"]: row for row in rows}


def build_contrast_requests(*, source_shard_root: str, rule_shard_root: str,
                            source_bank_path: str, rule_bank_path: str,
                            packet_root: str, target_job: str = "llama8_big_sparse",
                            source_job: str = "llama3_small",
                            rule_job: str = "llama3_target_policy_rules") -> tuple[list[dict], dict]:
    source_bank = json.loads(Path(source_bank_path).read_text())
    rule_bank = json.loads(Path(rule_bank_path).read_text())
    source_cells = {cell["id"]: cell for cell in source_bank["cells"]}
    rule_cells = {cell["id"]: cell for cell in rule_bank["cells"]}
    requests, contexts = [], {}
    for partition in PARTITIONS:
        source_index = load_public_index(source_shard_root, partition)
        rule_index = load_public_index(rule_shard_root, partition)
        for cell_id in TARGET_CELL_IDS:
            cell = source_cells[cell_id]
            domain, construct = cell["domain"], cell["construct"]
            source_data = _average_repetitions(source_index[(source_job, domain)])
            target_data = _average_repetitions(source_index[(target_job, domain)])
            rule_data = _average_repetitions(rule_index[(rule_job, domain)])
            source_orbits = _orbits(source_data["scores"], source_data["meta"], cell_id=cell_id)
            target_orbits = _orbits(target_data["scores"], target_data["meta"], cell_id=cell_id)
            rule_orbits = _orbits(rule_data["scores"], rule_data["meta"], cell_id=cell_id)
            target = target_orbits["name"]
            q = np.mean(np.stack(list(target.values())), axis=0)
            hashes = target_data["hashes"]
            items = _load_items(packet_root, domain, partition)
            origin = partition.removeprefix("residual_")
            behavior_id = f"rule_contrastive_v1_from_{origin}"
            source_specs = {arm["id"]: arm for arm in cell["arms"]}
            rule_specs = {arm["id"]: arm for arm in rule_cells[cell_id]["arms"]}
            parents = [
                ("name", source_specs["name"], source_orbits["name"]),
                ("behavior", rule_specs[behavior_id], rule_orbits[behavior_id]),
            ]
            for parent_id, parent_spec, parent_orbit in parents:
                aligned = _align_orbit(parent_orbit,
                                       (source_data["hashes"] if parent_id == "name"
                                        else rule_data["hashes"]), hashes)
                p = np.mean(np.stack(list(aligned.values())), axis=0)
                teaching_rows = []
                for item_hash, target_score, executor_score in zip(hashes, q, p):
                    text = items[item_hash]["text"]
                    if _eligible_teaching_item(domain, text):
                        teaching_rows.append({
                            "text_sha256": item_hash,
                            "text": _truncate(text, MAX_EXAMPLE_WORDS[domain]),
                            "target": float(target_score),
                            "executor": float(executor_score),
                        })
                pairs = select_rank_contrasts(teaching_rows)
                if len(pairs) < 4:
                    raise ValueError(f"too few rank reversals for {cell_id}:{partition}:{parent_id}")
                parent_text = next(form["prompt"] for form in parent_spec["forms"]
                                   if form["id"] == "canonical")
                context_key = f"{cell_id}:{partition}:{parent_id}"
                contexts[context_key] = {
                    "cell_id": cell_id, "domain": domain, "construct": construct,
                    "source_partition": partition, "parent_id": parent_id,
                    "parent_arm_id": parent_spec["id"], "parent_text": parent_text,
                    "parent_content_sha256": text_sha256(parent_text),
                    "calibration": policy_calibration(q), "pairs": pairs,
                }
                for pair_index, pair in enumerate(pairs):
                    requests.append({
                        "context_key": context_key, "cell_id": cell_id, "domain": domain,
                        "construct": construct, "source_partition": partition,
                        "parent_id": parent_id, "pair_index": pair_index,
                        "prompt": contrast_prompt(construct, pair),
                        "high_item_sha256": pair["high"]["text_sha256"],
                        "low_item_sha256": pair["low"]["text_sha256"],
                        "target_gap": pair["target_gap"],
                        "executor_reversal": pair["executor_reversal"],
                    })
    return requests, contexts


def synthesize(*, backend, contrast_requests: list[dict], contexts: dict,
               writer_model: str, seed: int = 20260722) -> tuple[list[dict], list[dict]]:
    contrast_seeds = [seed + 1009 * index for index in range(len(contrast_requests))]
    outputs = backend.generate_batch(
        [row["prompt"] for row in contrast_requests], max_tokens=190, temperature=0.7,
        seed=contrast_seeds, validate=lambda value: 25 <= len(str(value).split()) <= 160)
    contrasts, by_context = [], {}
    for request, request_seed, output in zip(contrast_requests, contrast_seeds, outputs):
        value = str(output).strip()
        row = {**request, "prompt_sha256": text_sha256(request["prompt"]),
               "seed": request_seed, "writer_model": writer_model, "micro_rule": value,
               "micro_rule_sha256": text_sha256(value),
               "micro_rule_word_count": len(value.split())}
        contrasts.append(row)
        by_context.setdefault(request["context_key"], []).append(row)

    synthesis_requests = []
    for context_key, context in contexts.items():
        micro_rules = [row["micro_rule"] for row in sorted(
            by_context[context_key], key=lambda value: value["pair_index"])]
        for mode in SYNTHESIS_MODES:
            for variant in range(N_VARIANTS):
                synthesis_requests.append({
                    "context_key": context_key, "cell_id": context["cell_id"],
                    "domain": context["domain"], "construct": context["construct"],
                    "source_partition": context["source_partition"],
                    "parent_id": context["parent_id"],
                    "parent_arm_id": context["parent_arm_id"], "mode": mode,
                    "variant": variant,
                    "prompt": synthesis_prompt(
                        context["construct"], context["parent_text"], micro_rules, mode,
                        context["calibration"]),
                    "teaching_item_sha256": [
                        value for pair in context["pairs"]
                        for value in (pair["high"]["text_sha256"],
                                      pair["low"]["text_sha256"])],
                    "micro_rule_sha256": [row["micro_rule_sha256"] for row in
                                          sorted(by_context[context_key],
                                                 key=lambda value: value["pair_index"])],
                })
    synthesis_seeds = [seed + 1_000_003 + 1009 * index + 7919 * row["variant"]
                       for index, row in enumerate(synthesis_requests)]
    outputs = backend.generate_batch(
        [row["prompt"] for row in synthesis_requests], max_tokens=560, temperature=0.7,
        seed=synthesis_seeds, validate=lambda value: 60 <= len(str(value).split()) <= 460)
    articulations = []
    for request, request_seed, output in zip(synthesis_requests, synthesis_seeds, outputs):
        value = str(output).strip()
        articulations.append({
            **request, "prompt_sha256": text_sha256(request["prompt"]),
            "seed": request_seed, "writer_model": writer_model,
            "articulation": value, "articulation_sha256": text_sha256(value),
            "articulation_word_count": len(value.split()),
        })
    return contrasts, articulations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-shard-root", required=True)
    parser.add_argument("--rule-shard-root", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--rule-bank", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--writer-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    requests, contexts = build_contrast_requests(
        source_shard_root=args.source_shard_root, rule_shard_root=args.rule_shard_root,
        source_bank_path=args.source_bank, rule_bank_path=args.rule_bank,
        packet_root=args.packet_root)
    config = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
    if args.fake:
        config.vllm_fake = True
    backend = make_judge_backend(args.writer_model, config, temperature=0.7)
    contrasts, rows = synthesize(
        backend=backend, contrast_requests=requests, contexts=contexts,
        writer_model=args.writer_model, seed=args.seed)
    payload = {
        "schema": "rank_contrast_articulation/v1",
        "status": "generated-before-small-executor-scoring",
        "objective": "repair item ordering in direct 3B reconstruction of 8B name-only policy",
        "anchor_policy": "fixed 8B name-only policy; no external ground truth or lockbox access",
        "writer_model": args.writer_model,
        "source_bank": {"path": args.source_bank, "sha256": sha256_file(args.source_bank)},
        "rule_bank": {"path": args.rule_bank, "sha256": sha256_file(args.rule_bank)},
        "packet_manifest": {"path": args.packet_manifest,
                            "sha256": sha256_file(args.packet_manifest)},
        "fold_policy": "all contrast-derived text is executed only on the opposite public fold",
        "contexts": contexts, "contrasts": contrasts, "rows": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256_file(out),
                      "n_contexts": len(contexts), "n_contrasts": len(contrasts),
                      "n_articulations": len(rows)}, indent=1))


if __name__ == "__main__":
    main()
