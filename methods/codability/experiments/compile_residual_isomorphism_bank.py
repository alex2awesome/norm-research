#!/usr/bin/env python
"""Compile cross-fitted ostensive articulations from public 8B/3B policy disagreements.

Each teaching arm is derived from one 200-item public fold and evaluated only on the opposite fold.
The examples are explicit demonstrations of the target policy's missing distinctions, not generic
prompt content.  No lockbox item or score is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file, text_sha256
from methods.codability.experiments.policy_data import (
    _align_orbit,
    _average_repetitions,
    _orbits,
    load_public_index,
)


PARTITIONS = ["residual_prompt_selection", "residual_unit_certification"]
DOMAIN_DIR = {"humor": "humor", "cw": "cw", "pr": "pr"}
BEST_SOURCE = {
    "N_humor_23": "source_definition",
    "N_humor_49": "source_explanation",
    "N_cw_27": "source_dossier_v2",
    "N_pr_8": "source_definition",
}
MAX_EXAMPLE_WORDS = {"humor": 70, "cw": 110, "pr": 80}
PR_EXTRACTION_ARTIFACTS = (
    "does not contain a press release", "no identifiable main body text",
    "raw page content", "there is no main body text", "```",
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _truncate(text: str, words: int) -> str:
    values = str(text).split()
    return str(text) if len(values) <= words else " ".join(values[:words]) + " …"


def select_diverse_examples(rows: list[dict], *, n: int, max_words: int) -> list[dict]:
    """Greedy residual strength first, then lexical diversity; deterministic."""
    candidates = sorted(rows, key=lambda row: (-row["priority"], row["text_sha256"]))[:50]
    selected = []
    while candidates and len(selected) < n:
        if not selected:
            choice = candidates[0]
        else:
            choice = max(candidates, key=lambda row: (
                0.75 * row["priority"]
                + 0.25 * min(1.0 - _jaccard(row["text"], old["text"]) for old in selected),
                -len(row["text"].split()), row["text_sha256"],
            ))
        selected.append({**choice, "text": _truncate(choice["text"], max_words)})
        candidates.remove(choice)
    return selected


def _load_items(packet_root: str | Path, domain: str, partition: str) -> dict[str, dict]:
    path = Path(packet_root) / DOMAIN_DIR[domain] / "items" / f"{partition}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["text_sha256"]: row for row in rows}


def _matched_forms(content: str) -> list[dict]:
    values = [
        ("canonical", content),
        ("question", f"Does the item meet this criterion? {content} Answer YES or NO."),
        ("boilerplate", f"You are an expert evaluator. Evaluate strictly. {content}"),
    ]
    return [{"id": form_id, "prompt": prompt, "prompt_sha256": text_sha256(prompt),
             "total_word_count": len(prompt.split())} for form_id, prompt in values]


def _example_block(name: str, positives: list[dict], negatives: list[dict], *,
                   heading: str) -> str:
    parts = [f"Criterion: {name}.", heading,
             "Examples judged to SATISFY the criterion:"]
    parts.extend(f"[YES {index}] {row['text']}" for index, row in enumerate(positives, 1))
    parts.append("Examples judged NOT to satisfy the criterion:")
    parts.extend(f"[NO {index}] {row['text']}" for index, row in enumerate(negatives, 1))
    parts.append("Apply the same distinction to the new item, including boundary cases.")
    return "\n\n".join(parts)


def _arm(arm_id: str, content: str, *, source_partition: str, recipe: str,
         source_hashes: list[str], source_arm: str | None = None) -> dict:
    return {
        "id": arm_id, "channel": "ostensive", "provenance": "ostensive_teaching",
        "control_for": None, "source_partition": source_partition,
        "recipe": recipe, "source_arm": source_arm,
        "teaching_item_sha256": source_hashes,
        "semantic_content_word_count": len(content.split()),
        "content_sha256": text_sha256(content), "forms": _matched_forms(content),
    }


def _eligible_teaching_item(domain: str, text: str) -> bool:
    """Keep exact-policy search from laundering obvious corpus failures into construct knowledge."""
    value = str(text).lower()
    if domain == "pr" and any(pattern in value for pattern in PR_EXTRACTION_ARTIFACTS):
        return False
    return bool(value.strip())


def _candidate_rows(hashes: list[str], items: dict[str, dict], q: np.ndarray,
                    sparse: np.ndarray, *, domain: str) -> dict[str, list[dict]]:
    rows = []
    for item_hash, target, small in zip(hashes, q, sparse):
        item = items[item_hash]
        if _eligible_teaching_item(domain, item["text"]):
            rows.append({"text_sha256": item_hash, "text": item["text"],
                         "target": float(target), "small": float(small)})
    positive_miss = [{**row, "priority": float(max(row["target"] - row["small"], 0.0))}
                     for row in rows if row["target"] >= 0.5]
    negative_miss = [{**row, "priority": float(max(row["small"] - row["target"], 0.0))}
                     for row in rows if row["target"] < 0.5]
    positive_proto = [{**row, "priority": float(row["target"])}
                      for row in rows if row["target"] >= 0.5]
    negative_proto = [{**row, "priority": float(1.0 - row["target"])}
                      for row in rows if row["target"] < 0.5]
    return {"positive_miss": positive_miss, "negative_miss": negative_miss,
            "positive_proto": positive_proto, "negative_proto": negative_proto}


def compile_bank(*, executor_shard_root: str, source_bank_path: str,
                 packet_root: str, small_job: str = "llama3_small",
                 big_job: str = "llama8_big_sparse") -> dict:
    source_bank = json.loads(Path(source_bank_path).read_text())
    cells_by_id = {cell["id"]: cell for cell in source_bank["cells"]}
    indexes = {partition: load_public_index(executor_shard_root, partition)
               for partition in PARTITIONS}
    output_cells = []
    for cell_id, source_cell in cells_by_id.items():
        domain, name = source_cell["domain"], source_cell["construct"]
        arm_specs = {arm["id"]: arm for arm in source_cell["arms"]}
        sparse_forms = arm_specs["name"]["forms"]
        arms = [{"id": "name", "channel": "sparse", "provenance": "construct_name",
                 "control_for": None, "source_partition": None,
                 "semantic_content_word_count": len(name.split()),
                 "content_sha256": text_sha256(name), "forms": sparse_forms}]
        best_source_id = BEST_SOURCE[cell_id]
        best_source_text = next(form["prompt"] for form in arm_specs[best_source_id]["forms"]
                                if form["id"] == "canonical")
        for partition in PARTITIONS:
            index = indexes[partition]
            small = _average_repetitions(index[(small_job, domain)])
            big = _average_repetitions(index[(big_job, domain)])
            small_orbits = _orbits(small["scores"], small["meta"], cell_id=cell_id)
            big_orbits = _orbits(big["scores"], big["meta"], cell_id=cell_id)
            target_hashes = big["hashes"]
            small_name = _align_orbit(small_orbits["name"], small["hashes"], target_hashes)
            q = np.mean(np.stack(list(big_orbits["name"].values())), axis=0)
            sparse = np.mean(np.stack(list(small_name.values())), axis=0)
            items = _load_items(packet_root, domain, partition)
            pools = _candidate_rows(target_hashes, items, q, sparse, domain=domain)
            max_words = MAX_EXAMPLE_WORDS[domain]
            for count in (1, 2, 4):
                positives = select_diverse_examples(
                    pools["positive_miss"], n=count, max_words=max_words)
                negatives = select_diverse_examples(
                    pools["negative_miss"], n=count, max_words=max_words)
                block = _example_block(
                    name, positives, negatives,
                    heading="These contrasts expose distinctions missed by a smaller evaluator.")
                hashes = [row["text_sha256"] for row in positives + negatives]
                suffix = partition.removeprefix("residual_")
                arms.append(_arm(f"residual_{count}x{count}_from_{suffix}", block,
                                 source_partition=partition, recipe=f"residual_{count}x{count}",
                                 source_hashes=hashes))
                combined = f"{best_source_text}\n\n{block}"
                arms.append(_arm(f"source_plus_residual_{count}x{count}_from_{suffix}", combined,
                                 source_partition=partition,
                                 recipe=f"source_plus_residual_{count}x{count}",
                                 source_hashes=hashes, source_arm=best_source_id))
            positives = select_diverse_examples(
                pools["positive_proto"], n=2, max_words=max_words)
            negatives = select_diverse_examples(
                pools["negative_proto"], n=2, max_words=max_words)
            block = _example_block(name, positives, negatives,
                                   heading="These are high-confidence prototypes and counterexamples.")
            hashes = [row["text_sha256"] for row in positives + negatives]
            suffix = partition.removeprefix("residual_")
            arms.append(_arm(f"prototypes_2x2_from_{suffix}", block,
                             source_partition=partition, recipe="prototypes_2x2",
                             source_hashes=hashes))
            arms.append(_arm(f"source_plus_prototypes_2x2_from_{suffix}",
                             f"{best_source_text}\n\n{block}", source_partition=partition,
                             recipe="source_plus_prototypes_2x2", source_hashes=hashes,
                             source_arm=best_source_id))
        output_cells.append({"id": cell_id, "domain": domain, "gi": source_cell["gi"],
                             "construct": name, "arms": arms})
    payload = {
        "schema": "residual_policy_isomorphism_arm_bank/v1",
        "status": "crossfit-bank-frozen-before-any-residual-arm-executor-score",
        "objective": "maximize direct 3B-articulated replication of the 8B name-only policy",
        "source_bank": {"path": source_bank_path, "sha256": sha256_file(source_bank_path)},
        "executor_shard_root": executor_shard_root,
        "fold_policy": ("arms derived from residual_prompt_selection are evaluated only on "
                        "residual_unit_certification and vice versa"),
        "treatment": "development-disagreement knowledge made explicit as ostensive contrasts",
        "teaching_item_hygiene": ("obvious press-release extraction failures are excluded before "
                                  "example ranking; exact replication may not relabel corpus errors "
                                  "as construct knowledge"),
        "cells": output_cells,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bank_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def execution_manifest(bank_path: str | Path, packet_manifest: str | Path) -> dict:
    return {
        "schema": "residual_policy_isomorphism_execution_manifest/v1",
        "status": "frozen-before-residual-arm-executor-scoring",
        "arm_bank_sha256": sha256_file(bank_path),
        "packet_manifest_sha256": sha256_file(packet_manifest),
        "phases": {"crossfit": PARTITIONS},
        "model_jobs": [{"id": "llama3_residual_isomorphism",
                        "model": "meta-llama/Llama-3.2-3B-Instruct",
                        "role": "small", "arm_policy": "all"}],
        "primary_estimand": "opposite-fold direct policy isomorphism",
        "lockbox_status": "not authorized by this search manifest",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--executor-shard-root", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execution-manifest-out", required=True)
    args = parser.parse_args()
    bank = compile_bank(executor_shard_root=args.executor_shard_root,
                        source_bank_path=args.source_bank, packet_root=args.packet_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, indent=1))
    manifest = execution_manifest(out, args.packet_manifest)
    manifest_path = Path(args.execution_manifest_out)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256_file(out),
                      "execution_manifest": str(manifest_path),
                      "execution_manifest_sha256": sha256_file(manifest_path),
                      "n_cells": len(bank["cells"]),
                      "n_arms": sum(len(cell["arms"]) for cell in bank["cells"])}, indent=1))


if __name__ == "__main__":
    main()
