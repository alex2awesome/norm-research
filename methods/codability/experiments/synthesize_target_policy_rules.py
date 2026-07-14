#!/usr/bin/env python
"""Ask the fixed larger sparse-policy reader to make its own latent policy explicit.

This is an oracle-free reconstruction instrument: the only behavioral signal is the frozen 8B
name-only policy on a public teaching fold.  No corpus label, human score, external evaluator, or
third-model judgment enters selection or synthesis.  Fold-indexed rules must be executed only on
the opposite public fold; construct-only self-explications are independent of either fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

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
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend


RULE_VIEWS = {
    "gestalt": (
        "Make the holistic evaluative gestalt explicit. Describe subtle social or normative "
        "cues, interactions among cues, compensating strengths, veto-like failures, and boundary "
        "cases that a merely literal definition would miss."
    ),
    "contrastive": (
        "Infer the reusable distinctions separating high from low judgments. State observable "
        "positive and negative indicators, causal mechanisms, interactions, exceptions, and the "
        "features that should not be mistaken for the criterion."
    ),
    "procedure": (
        "Write an ordered decision procedure. Include what to inspect first, how to integrate "
        "conflicting evidence, how strict the threshold is, and how to resolve ambiguous or "
        "borderline cases."
    ),
}
N_VARIANTS = 2


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _truncate(text: str, words: int) -> str:
    values = str(text).split()
    return str(text) if len(values) <= words else " ".join(values[:words]) + " …"


def _diverse_take(rows: list[dict], n: int, selected_hashes: set[str]) -> list[dict]:
    """Take high-priority, lexically diverse rows without duplicating panel members."""
    pool = [row for row in sorted(rows, key=lambda r: (-r["priority"], r["text_sha256"]))
            if row["text_sha256"] not in selected_hashes][:60]
    chosen: list[dict] = []
    while pool and len(chosen) < n:
        prior = chosen
        if not prior:
            choice = pool[0]
        else:
            choice = max(
                pool,
                key=lambda row: (
                    0.8 * row["priority"]
                    + 0.2 * min(1.0 - _jaccard(row["text"], old["text"]) for old in prior),
                    -len(row["text"].split()),
                    row["text_sha256"],
                ),
            )
        chosen.append(choice)
        selected_hashes.add(choice["text_sha256"])
        pool.remove(choice)
    return chosen


def select_teaching_panel(rows: list[dict], *, per_slice: int = 2) -> list[dict]:
    """Stratify target-only prototypes/boundaries plus target-vs-small residuals.

    The small reader is used only to locate missing distinctions.  It is never presented as an
    authority to the writer, and its score is omitted from the synthesis prompt.
    """
    pools = [
        ("high_target", [{**r, "priority": r["target"]} for r in rows]),
        ("low_target", [{**r, "priority": 1.0 - r["target"]} for r in rows]),
        ("boundary", [{**r, "priority": 1.0 - 2.0 * abs(r["target"] - 0.5)}
                      for r in rows]),
        ("positive_residual", [{**r, "priority": max(r["target"] - r["small"], 0.0)}
                               for r in rows]),
        ("negative_residual", [{**r, "priority": max(r["small"] - r["target"], 0.0)}
                               for r in rows]),
    ]
    # Slice filters make the first two true poles and residual slices directional.
    filters = {
        "high_target": lambda r: r["target"] >= 0.5,
        "low_target": lambda r: r["target"] < 0.5,
        "boundary": lambda r: True,
        "positive_residual": lambda r: r["target"] > r["small"],
        "negative_residual": lambda r: r["small"] > r["target"],
    }
    selected: list[dict] = []
    hashes: set[str] = set()
    for slice_id, pool in pools:
        eligible = [row for row in pool if filters[slice_id](row)]
        for row in _diverse_take(eligible, per_slice, hashes):
            selected.append({**row, "slice": slice_id})
    return selected


def policy_calibration(q: np.ndarray) -> dict:
    q = np.asarray(q, float)
    return {
        "mean_p_yes": float(np.mean(q)),
        "binary_positive_rate": float(np.mean(q >= 0.5)),
        "p_yes_quantiles": {str(k): float(np.quantile(q, k / 100.0))
                            for k in (10, 25, 50, 75, 90)},
    }


def calibration_text(calibration: Mapping) -> str:
    pct = round(100.0 * float(calibration["binary_positive_rate"]))
    mean = float(calibration["mean_p_yes"])
    return (
        f"Strictness calibration: on a varied reference panel, this policy gave a clear YES to "
        f"about {pct}% of items and had mean YES propensity {mean:.2f}. Treat this as threshold "
        "calibration, never as a quota for a new batch."
    )


def construct_only_prompt(name: str, view_instruction: str) -> str:
    return (
        "You are making your own latent evaluative knowledge explicit so a smaller language model "
        "can reproduce the judgments you would make from a criterion name alone.\n\n"
        f"Criterion name: {name}\n\n{view_instruction}\n\n"
        "Return only a standalone, reusable criterion specification of 140–220 words. Do not "
        "mention models, this request, hidden knowledge, or any examples you were not given. Avoid "
        "generic evaluation advice; articulate the particular judgment policy indexed by the name."
    )


def behavior_prompt(name: str, view_instruction: str, panel: list[dict],
                    calibration: Mapping) -> str:
    examples = []
    for index, row in enumerate(panel, 1):
        examples.append(
            f"[Example {index}; YES propensity {row['target']:.2f}]\n{row['text']}"
        )
    return (
        "You are compressing a fixed evaluator's item-level policy into explicit natural-language "
        "rules so a smaller evaluator can reproduce it on unseen items. There is no external "
        "ground truth: the displayed judgments are the entire behavioral target.\n\n"
        f"Criterion name: {name}\n\n{calibration_text(calibration)}\n\n"
        f"{view_instruction}\n\nBehavioral examples:\n\n" + "\n\n".join(examples) +
        "\n\nInfer an item-independent policy that generalizes beyond these examples. Return only a "
        "standalone criterion specification of 170–260 words. Do not mention example numbers, "
        "probabilities, models, datasets, or quotas. Do not copy distinctive phrases from examples."
    )


def _load_items(packet_root: str | Path, domain: str, partition: str) -> dict[str, dict]:
    path = Path(packet_root) / DOMAIN_DIR[domain] / "items" / f"{partition}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["text_sha256"]: row for row in rows}


def build_requests(*, shard_root: str, source_bank_path: str, packet_root: str,
                   small_job: str, target_job: str) -> tuple[list[dict], dict]:
    bank = json.loads(Path(source_bank_path).read_text())
    indexes = {partition: load_public_index(shard_root, partition) for partition in PARTITIONS}
    requests: list[dict] = []
    panel_manifest: dict = {}
    for cell in bank["cells"]:
        cell_id, domain, name = cell["id"], cell["domain"], cell["construct"]
        for view_id, instruction in RULE_VIEWS.items():
            for variant in range(N_VARIANTS):
                requests.append({
                    "cell_id": cell_id, "domain": domain, "construct": name,
                    "source_partition": None, "view": view_id, "variant": variant,
                    "prompt": construct_only_prompt(name, instruction),
                    "teaching_item_sha256": [], "calibration": None,
                })
        for partition in PARTITIONS:
            index = indexes[partition]
            small = _average_repetitions(index[(small_job, domain)])
            target = _average_repetitions(index[(target_job, domain)])
            small_orbits = _orbits(small["scores"], small["meta"], cell_id=cell_id)
            target_orbits = _orbits(target["scores"], target["meta"], cell_id=cell_id)
            hashes = target["hashes"]
            small_name = _align_orbit(small_orbits["name"], small["hashes"], hashes)
            q = np.mean(np.stack(list(target_orbits["name"].values())), axis=0)
            sparse = np.mean(np.stack(list(small_name.values())), axis=0)
            items = _load_items(packet_root, domain, partition)
            rows = []
            for item_hash, target_score, small_score in zip(hashes, q, sparse):
                text = items[item_hash]["text"]
                if _eligible_teaching_item(domain, text):
                    rows.append({
                        "text_sha256": item_hash,
                        "text": _truncate(text, MAX_EXAMPLE_WORDS[domain]),
                        "target": float(target_score),
                        "small": float(small_score),
                    })
            panel = select_teaching_panel(rows)
            calibration = policy_calibration(q)
            panel_manifest[f"{cell_id}:{partition}"] = {
                "teaching_item_sha256": [row["text_sha256"] for row in panel],
                "teaching_slices": [row["slice"] for row in panel],
                "calibration": calibration,
            }
            for view_id, instruction in RULE_VIEWS.items():
                for variant in range(N_VARIANTS):
                    requests.append({
                        "cell_id": cell_id, "domain": domain, "construct": name,
                        "source_partition": partition, "view": view_id, "variant": variant,
                        "prompt": behavior_prompt(name, instruction, panel, calibration),
                        "teaching_item_sha256": [row["text_sha256"] for row in panel],
                        "calibration": calibration,
                    })
    return requests, panel_manifest


def synthesize(*, backend, requests: list[dict], writer_model: str,
               seed: int = 20260712) -> list[dict]:
    seeds = [seed + 1009 * i + 7919 * int(row["variant"])
             for i, row in enumerate(requests)]
    outputs = backend.generate_batch(
        [row["prompt"] for row in requests], max_tokens=420,
        temperature=0.7, seed=seeds,
        validate=lambda value: 40 <= len(str(value).split()) <= 380,
    )
    rows = []
    for request, request_seed, output in zip(requests, seeds, outputs):
        value = str(output).strip()
        rows.append({
            **request, "prompt_sha256": text_sha256(request["prompt"]),
            "seed": request_seed, "writer_model": writer_model,
            "articulation": value, "articulation_sha256": text_sha256(value),
            "articulation_word_count": len(value.split()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--writer-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--small-job", default="llama3_small")
    parser.add_argument("--target-job", default="llama8_big_sparse")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    requests, panels = build_requests(
        shard_root=args.shard_root, source_bank_path=args.source_bank,
        packet_root=args.packet_root, small_job=args.small_job, target_job=args.target_job)
    config = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
    if args.fake:
        config.vllm_fake = True
    backend = make_judge_backend(args.writer_model, config, temperature=0.7)
    rows = synthesize(backend=backend, requests=requests, writer_model=args.writer_model,
                      seed=args.seed)
    payload = {
        "schema": "target_policy_self_articulation/v1",
        "status": "generated-before-small-executor-scoring",
        "objective": "make the fixed larger name-only policy explicit without an external oracle",
        "writer_model": args.writer_model,
        "target_job": args.target_job,
        "small_job_used_for_residual_selection_only": args.small_job,
        "source_bank": {"path": args.source_bank, "sha256": sha256_file(args.source_bank)},
        "packet_manifest": {"path": args.packet_manifest,
                            "sha256": sha256_file(args.packet_manifest)},
        "shard_root": args.shard_root,
        "fold_policy": ("behavior-indexed rules derived from one public fold may be executed only "
                        "on the opposite public fold"),
        "anchor_policy": ("no external label or evaluator; only the 8B sparse soft policy enters "
                          "the synthesis prompts"),
        "panels": panels,
        "rows": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256_file(out),
                      "n_requests": len(requests), "n_rows": len(rows)}, indent=1))


if __name__ == "__main__":
    main()
