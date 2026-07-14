#!/usr/bin/env python
"""Compile cross-fitted full-text policy revisions into a frozen small-executor bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import sha256_file, text_sha256
from methods.codability.experiments.compile_residual_isomorphism_bank import (
    BEST_SOURCE,
    PARTITIONS,
)


def _matched_forms(content: str) -> list[dict]:
    values = [
        ("canonical", content),
        ("question", f"Does the item meet this criterion? {content} Answer YES or NO."),
        ("boilerplate", f"You are an expert evaluator. Evaluate strictly. {content}"),
    ]
    return [{"id": form_id, "prompt": prompt, "prompt_sha256": text_sha256(prompt),
             "total_word_count": len(prompt.split())} for form_id, prompt in values]


def _source_then_forms(source_arm: dict, addition: str) -> list[dict]:
    result = []
    for form in source_arm["forms"]:
        prompt = f"{form['prompt']}\n\n{addition}"
        result.append({"id": form["id"], "prompt": prompt,
                       "prompt_sha256": text_sha256(prompt),
                       "total_word_count": len(prompt.split())})
    return result


def _arm(arm_id: str, content: str, forms: list[dict], row: dict, *, recipe: str,
         source_arm: str | None) -> dict:
    return {
        "id": arm_id, "channel": "declarative",
        "provenance": "target_residual_revision", "control_for": None,
        "source_partition": row["source_partition"], "recipe": recipe,
        "parent_slot": row["parent_slot"], "parent_arm_id": row["parent_arm_id"],
        "parent_provenance": row["parent_provenance"], "source_arm": source_arm,
        "writer_model": row["writer_model"],
        "synthesis_prompt_sha256": row["prompt_sha256"],
        "articulation_sha256": row["articulation_sha256"],
        "teaching_item_sha256": row["teaching_item_sha256"],
        "semantic_content_word_count": len(content.split()),
        "content_sha256": text_sha256(content), "forms": forms,
    }


def compile_bank(*, revision_path: str | Path, source_bank_path: str | Path) -> dict:
    revisions = json.loads(Path(revision_path).read_text())
    source_bank = json.loads(Path(source_bank_path).read_text())
    by_cell: dict[str, list[dict]] = {}
    for row in revisions["rows"]:
        by_cell.setdefault(row["cell_id"], []).append(row)
    cells = []
    for cell in source_bank["cells"]:
        cell_id = cell["id"]
        specs = {arm["id"]: arm for arm in cell["arms"]}
        source_id = BEST_SOURCE[cell_id]
        source_arm = specs[source_id]
        source_text = next(form["prompt"] for form in source_arm["forms"]
                           if form["id"] == "canonical")
        arms = [{**specs["name"], "source_partition": None}]
        for row in sorted(by_cell[cell_id], key=lambda value: (
                value["source_partition"], value["parent_slot"], value["view"],
                value["variant"])):
            origin = row["source_partition"].removeprefix("residual_")
            stem = (f"{row['view']}_v{row['variant']}_slot{row['parent_slot']}_from_{origin}")
            rule = row["articulation"]
            arms.append(_arm(f"revision_{stem}", rule, _matched_forms(rule), row,
                             recipe="revision_only", source_arm=None))
            combined = f"{source_text}\n\n{rule}"
            arms.append(_arm(
                f"source_then_revision_{stem}", combined,
                _source_then_forms(source_arm, rule), row,
                recipe="source_then_revision", source_arm=source_id))
        cells.append({"id": cell_id, "domain": cell["domain"], "gi": cell["gi"],
                      "construct": cell["construct"], "arms": arms})
    payload = {
        "schema": "residual_policy_revision_arm_bank/v1",
        "status": "frozen-after-revision-before-small-executor-scoring",
        "objective": "maximize opposite-fold direct reconstruction of the 8B name-only policy",
        "anchor_policy": "no external ground truth; fixed target behavior is the sole objective",
        "revision_artifact": {"path": str(revision_path),
                              "sha256": sha256_file(revision_path)},
        "source_bank": {"path": str(source_bank_path),
                        "sha256": sha256_file(source_bank_path)},
        "fold_policy": "each revised text is evaluated only on the opposite public fold",
        "selection_priority": "isomorphism first; text cost and diversity subordinate",
        "cells": cells,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bank_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def execution_manifest(bank_path: str | Path, packet_manifest: str | Path) -> dict:
    return {
        "schema": "residual_policy_revision_execution_manifest/v1",
        "status": "frozen-before-small-executor-scoring",
        "arm_bank_sha256": sha256_file(bank_path),
        "packet_manifest_sha256": sha256_file(packet_manifest),
        "phases": {"crossfit": PARTITIONS},
        "model_jobs": [{"id": "llama3_policy_revision",
                        "model": "meta-llama/Llama-3.2-3B-Instruct",
                        "role": "small", "arm_policy": "all"}],
        "primary_estimand": "opposite-fold direct policy isomorphism",
        "lockbox_status": "not authorized by this search manifest",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execution-manifest-out", required=True)
    args = parser.parse_args()
    bank = compile_bank(revision_path=args.revision, source_bank_path=args.source_bank)
    out = Path(args.out)
    out.write_text(json.dumps(bank, indent=1))
    manifest = execution_manifest(out, args.packet_manifest)
    manifest_out = Path(args.execution_manifest_out)
    manifest_out.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256_file(out),
                      "execution_manifest": str(manifest_out),
                      "execution_manifest_sha256": sha256_file(manifest_out),
                      "n_cells": len(bank["cells"]),
                      "n_arms": sum(len(cell["arms"]) for cell in bank["cells"])}, indent=1))


if __name__ == "__main__":
    main()
