#!/usr/bin/env python
"""Hierarchically distill every item in a public fold into explicit target-policy textbooks.

Each 200-item teaching fold is compressed in eight independent memos, then merged by the fixed 8B
target model into standalone rules.  The rules are executed only on the opposite fold.  No lockbox,
external label, human target, or third-model judgment is read.
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
from methods.codability.experiments.synthesize_target_policy_rules import (
    _truncate,
    calibration_text,
    policy_calibration,
)
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend


N_CHUNKS = 8
TEXTBOOK_MODES = {
    "compressed": (
        "Write a compact 280–420 word decision specification. Preserve the most predictive "
        "distinctions, interactions, vetoes, and tie-breakers; omit exposition that does not change "
        "a judgment."
    ),
    "textbook": (
        "Write a comprehensive 650–900 word policy textbook. Integrate definition, mechanisms, "
        "positive and negative cues, interactions, compensating strengths, vetoes, boundary cases, "
        "an ordered procedure, and relative-ranking tie-breakers."
    ),
    "gestalt": (
        "Write a 450–650 word holistic specification of the latent evaluative gestalt. Emphasize "
        "configurations of cues, social or normative expectations, exceptions, and why superficially "
        "similar items should be ordered differently."
    ),
}
N_VARIANTS = 2


def interleaved_chunks(rows: list[dict], n_chunks: int = N_CHUNKS) -> list[list[dict]]:
    """Deterministically spread the full target-score range across every chunk."""
    ordered = sorted(rows, key=lambda row: (row["target"], row["text_sha256"]))
    chunks = [[] for _ in range(n_chunks)]
    for block_index, start in enumerate(range(0, len(ordered), n_chunks)):
        block = ordered[start:start + n_chunks]
        destinations = range(n_chunks) if block_index % 2 == 0 else reversed(range(n_chunks))
        for destination, row in zip(destinations, block):
            chunks[destination].append(row)
    return chunks


def memo_prompt(name: str, parent_text: str, chunk: list[dict], chunk_index: int) -> str:
    cases = []
    for index, row in enumerate(chunk, 1):
        cases.append(
            f"[{index}: target YES {row['target']:.2f}; current executor YES "
            f"{row['executor']:.2f}]\n{row['text']}"
        )
    return (
        "You are analyzing one stratified batch from a fixed evaluator's item policy. There is no "
        "external ground truth: target YES is the policy to reconstruct. Diagnose reusable semantic "
        "distinctions that explain both the target's ordering and the current smaller executor's "
        "residual errors.\n\n"
        f"Criterion: {name}\n\nCurrent explicit specification:\n---\n{parent_text}\n---\n\n"
        f"Batch {chunk_index + 1}:\n\n" + "\n\n".join(cases) +
        "\n\nReturn a 140–220 word technical memo of general rules, interactions, vetoes, boundary "
        "logic, and ranking tie-breakers. Do not quote cases or mention scores, models, datasets, "
        "batches, or quotas."
    )


def synthesis_prompt(name: str, parent_text: str, memos: list[str], mode: str,
                     calibration: dict) -> str:
    memo_block = "\n\n".join(f"[Policy memo {index}]\n{memo}"
                              for index, memo in enumerate(memos, 1))
    return (
        "You are consolidating independent analyses of a fixed evaluator's policy into explicit "
        "natural-language knowledge for a smaller evaluator. The memos summarize the entire public "
        "teaching fold, not a hand-picked example subset. Resolve contradictions by extracting "
        "general distinctions; do not average incompatible rules mechanically.\n\n"
        f"Criterion: {name}\n\nCurrent source specification:\n---\n{parent_text}\n---\n\n"
        f"{calibration_text(calibration)}\n\n{memo_block}\n\n{TEXTBOOK_MODES[mode]}\n\n"
        "Return only the standalone criterion specification. Never mention memos, examples, scores, "
        "models, folds, datasets, revisions, hidden knowledge, or quotas."
    )


def _load_items(packet_root: str | Path, domain: str, partition: str) -> dict[str, dict]:
    path = Path(packet_root) / DOMAIN_DIR[domain] / "items" / f"{partition}.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["text_sha256"]: row for row in values}


def build_memo_requests(*, target_shard_root: str, source_shard_root: str,
                        source_bank_path: str, packet_root: str,
                        target_job: str = "llama8_big_sparse",
                        source_job: str = "llama3_small") -> tuple[list[dict], dict]:
    bank = json.loads(Path(source_bank_path).read_text())
    requests, contexts = [], {}
    max_words = {**MAX_EXAMPLE_WORDS, "humor": 45, "cw": 60, "pr": 60}
    for partition in PARTITIONS:
        target_index = load_public_index(target_shard_root, partition)
        source_index = load_public_index(source_shard_root, partition)
        for cell in bank["cells"]:
            cell_id, domain, name = cell["id"], cell["domain"], cell["construct"]
            specs = {arm["id"]: arm for arm in cell["arms"]}
            source_id = BEST_SOURCE[cell_id]
            parent_text = next(form["prompt"] for form in specs[source_id]["forms"]
                               if form["id"] == "canonical")
            target_data = _average_repetitions(target_index[(target_job, domain)])
            source_data = _average_repetitions(source_index[(source_job, domain)])
            target_orbits = _orbits(target_data["scores"], target_data["meta"], cell_id=cell_id)
            source_orbits = _orbits(source_data["scores"], source_data["meta"], cell_id=cell_id)
            hashes = target_data["hashes"]
            q = np.mean(np.stack(list(target_orbits["name"].values())), axis=0)
            parent_orbit = _align_orbit(source_orbits[source_id], source_data["hashes"], hashes)
            p = np.mean(np.stack(list(parent_orbit.values())), axis=0)
            items = _load_items(packet_root, domain, partition)
            rows = []
            for item_hash, target_score, executor_score in zip(hashes, q, p):
                text = items[item_hash]["text"]
                if _eligible_teaching_item(domain, text):
                    rows.append({"text_sha256": item_hash,
                                 "text": _truncate(text, max_words[domain]),
                                 "target": float(target_score),
                                 "executor": float(executor_score)})
            chunks = interleaved_chunks(rows)
            key = f"{cell_id}:{partition}"
            contexts[key] = {"cell_id": cell_id, "domain": domain, "construct": name,
                             "source_partition": partition, "source_arm": source_id,
                             "parent_text": parent_text, "calibration": policy_calibration(q),
                             "n_eligible_items": len(rows),
                             "chunk_item_sha256": [[row["text_sha256"] for row in chunk]
                                                   for chunk in chunks]}
            for chunk_index, chunk in enumerate(chunks):
                requests.append({"context_key": key, "cell_id": cell_id, "domain": domain,
                                 "source_partition": partition, "chunk_index": chunk_index,
                                 "prompt": memo_prompt(name, parent_text, chunk, chunk_index)})
    return requests, contexts


def generate_textbooks(*, backend, memo_requests: list[dict], contexts: dict,
                       writer_model: str, seed: int = 20260720) -> tuple[list[dict], list[dict]]:
    memo_seeds = [seed + 1009 * index for index in range(len(memo_requests))]
    memo_outputs = backend.generate_batch(
        [row["prompt"] for row in memo_requests], max_tokens=340, temperature=0.7,
        seed=memo_seeds, validate=lambda value: 60 <= len(str(value).split()) <= 320)
    memo_rows = []
    by_context: dict[str, list[tuple[int, str]]] = {}
    for request, request_seed, output in zip(memo_requests, memo_seeds, memo_outputs):
        value = str(output).strip()
        memo_rows.append({**request, "prompt_sha256": text_sha256(request["prompt"]),
                          "seed": request_seed, "memo": value,
                          "memo_sha256": text_sha256(value),
                          "memo_word_count": len(value.split())})
        by_context.setdefault(request["context_key"], []).append(
            (request["chunk_index"], value))
    synthesis_requests = []
    for context_key, context in contexts.items():
        memos = [value for _, value in sorted(by_context[context_key])]
        for mode in TEXTBOOK_MODES:
            for variant in range(N_VARIANTS):
                synthesis_requests.append({
                    "context_key": context_key, "cell_id": context["cell_id"],
                    "domain": context["domain"], "construct": context["construct"],
                    "source_partition": context["source_partition"],
                    "source_arm": context["source_arm"], "mode": mode,
                    "variant": variant,
                    "prompt": synthesis_prompt(context["construct"], context["parent_text"],
                                               memos, mode, context["calibration"]),
                    "teaching_item_sha256": [value for chunk in
                                             context["chunk_item_sha256"] for value in chunk],
                })
    synthesis_seeds = [seed + 1_000_003 + 1009 * index + 7919 * row["variant"]
                       for index, row in enumerate(synthesis_requests)]
    outputs = backend.generate_batch(
        [row["prompt"] for row in synthesis_requests], max_tokens=1400, temperature=0.7,
        seed=synthesis_seeds, validate=lambda value: 140 <= len(str(value).split()) <= 1100)
    textbooks = []
    for request, request_seed, output in zip(synthesis_requests, synthesis_seeds, outputs):
        value = str(output).strip()
        textbooks.append({**request, "prompt_sha256": text_sha256(request["prompt"]),
                          "seed": request_seed, "writer_model": writer_model,
                          "articulation": value, "articulation_sha256": text_sha256(value),
                          "articulation_word_count": len(value.split())})
    return memo_rows, textbooks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-shard-root", required=True)
    parser.add_argument("--source-shard-root", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--writer-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    requests, contexts = build_memo_requests(
        target_shard_root=args.target_shard_root, source_shard_root=args.source_shard_root,
        source_bank_path=args.source_bank, packet_root=args.packet_root)
    config = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
    if args.fake:
        config.vllm_fake = True
    backend = make_judge_backend(args.writer_model, config, temperature=0.7)
    memos, rows = generate_textbooks(backend=backend, memo_requests=requests, contexts=contexts,
                                     writer_model=args.writer_model, seed=args.seed)
    payload = {
        "schema": "hierarchical_policy_textbook/v1",
        "status": "generated-before-small-executor-scoring",
        "objective": "full-fold explicit compression of the fixed 8B name-only policy",
        "anchor_policy": "no external ground truth; no lockbox access",
        "writer_model": args.writer_model,
        "source_bank": {"path": args.source_bank, "sha256": sha256_file(args.source_bank)},
        "packet_manifest": {"path": args.packet_manifest,
                            "sha256": sha256_file(args.packet_manifest)},
        "fold_policy": "textbook distilled from one public fold is executed only on the other",
        "contexts": contexts, "memos": memos, "rows": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256_file(out),
                      "n_memos": len(memos), "n_textbooks": len(rows)}, indent=1))


if __name__ == "__main__":
    main()
