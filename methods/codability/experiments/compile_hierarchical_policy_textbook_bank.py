#!/usr/bin/env python
"""Compile full-fold hierarchical policy textbooks into a frozen cross-fit arm bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import sha256_file, text_sha256
from methods.codability.experiments.compile_residual_isomorphism_bank import BEST_SOURCE, PARTITIONS


def _matched_forms(content: str) -> list[dict]:
    values = [("canonical", content),
              ("question", f"Does the item meet this criterion? {content} Answer YES or NO."),
              ("boilerplate", f"You are an expert evaluator. Evaluate strictly. {content}")]
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


def _arm(arm_id: str, content: str, forms: list[dict], row: dict, *,
         recipe: str, source_arm: str | None) -> dict:
    return {"id": arm_id, "channel": "declarative",
            "provenance": "target_hierarchical_articulation", "control_for": None,
            "source_partition": row["source_partition"], "recipe": recipe,
            "mode": row["mode"], "variant": row["variant"], "source_arm": source_arm,
            "writer_model": row["writer_model"],
            "synthesis_prompt_sha256": row["prompt_sha256"],
            "articulation_sha256": row["articulation_sha256"],
            "teaching_item_sha256": row["teaching_item_sha256"],
            "semantic_content_word_count": len(content.split()),
            "content_sha256": text_sha256(content), "forms": forms}


def compile_bank(*, textbook_path: str | Path, source_bank_path: str | Path) -> dict:
    textbooks = json.loads(Path(textbook_path).read_text())
    source = json.loads(Path(source_bank_path).read_text())
    by_cell: dict[str, list[dict]] = {}
    for row in textbooks["rows"]:
        by_cell.setdefault(row["cell_id"], []).append(row)
    cells = []
    for cell in source["cells"]:
        cell_id = cell["id"]
        specs = {arm["id"]: arm for arm in cell["arms"]}
        source_id = BEST_SOURCE[cell_id]
        source_arm = specs[source_id]
        source_text = next(form["prompt"] for form in source_arm["forms"]
                           if form["id"] == "canonical")
        arms = [{**specs["name"], "source_partition": None}]
        for row in sorted(by_cell[cell_id], key=lambda value: (
                value["source_partition"], value["mode"], value["variant"])):
            origin = row["source_partition"].removeprefix("residual_")
            stem = f"{row['mode']}_v{row['variant']}_from_{origin}"
            text = row["articulation"]
            arms.append(_arm(f"textbook_{stem}", text, _matched_forms(text), row,
                             recipe="textbook_only", source_arm=None))
            combined = f"{source_text}\n\n{text}"
            arms.append(_arm(f"source_then_textbook_{stem}", combined,
                             _source_then_forms(source_arm, text), row,
                             recipe="source_then_textbook", source_arm=source_id))
        cells.append({"id": cell_id, "domain": cell["domain"], "gi": cell["gi"],
                      "construct": cell["construct"], "arms": arms})
    payload = {
        "schema": "hierarchical_policy_textbook_arm_bank/v1",
        "status": "frozen-after-textbook-generation-before-small-executor-scoring",
        "objective": "maximize opposite-fold direct 3B reconstruction of the 8B sparse policy",
        "anchor_policy": "no external ground truth; no lockbox access",
        "textbook_artifact": {"path": str(textbook_path),
                              "sha256": sha256_file(textbook_path)},
        "source_bank": {"path": str(source_bank_path),
                        "sha256": sha256_file(source_bank_path)},
        "fold_policy": "each full-fold textbook is evaluated only on the opposite public fold",
        "cells": cells,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bank_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def execution_manifest(bank_path: str | Path, packet_manifest: str | Path) -> dict:
    return {
        "schema": "hierarchical_policy_textbook_execution_manifest/v1",
        "status": "frozen-before-small-executor-scoring",
        "arm_bank_sha256": sha256_file(bank_path),
        "packet_manifest_sha256": sha256_file(packet_manifest),
        "phases": {"crossfit": PARTITIONS},
        "model_jobs": [{"id": "llama3_policy_textbook",
                        "model": "meta-llama/Llama-3.2-3B-Instruct",
                        "role": "small", "arm_policy": "all"}],
        "primary_estimand": "opposite-fold direct policy isomorphism",
        "lockbox_status": "not authorized by this search manifest",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--textbook", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execution-manifest-out", required=True)
    args = parser.parse_args()
    bank = compile_bank(textbook_path=args.textbook, source_bank_path=args.source_bank)
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
