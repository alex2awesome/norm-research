#!/usr/bin/env python
"""Compile frozen target-self-articulations into a cross-fitted executor arm bank."""
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
from methods.codability.experiments.synthesize_target_policy_rules import calibration_text


def _matched_forms(content: str) -> list[dict]:
    values = [
        ("canonical", content),
        ("question", f"Does the item meet this criterion? {content} Answer YES or NO."),
        ("boilerplate", f"You are an expert evaluator. Evaluate strictly. {content}"),
    ]
    return [{"id": form_id, "prompt": prompt, "prompt_sha256": text_sha256(prompt),
             "total_word_count": len(prompt.split())} for form_id, prompt in values]


def _source_then_forms(source_arm: dict, addition: str) -> list[dict]:
    forms = []
    for source_form in source_arm["forms"]:
        prompt = f"{source_form['prompt']}\n\n{addition}"
        forms.append({"id": source_form["id"], "prompt": prompt,
                      "prompt_sha256": text_sha256(prompt),
                      "total_word_count": len(prompt.split())})
    return forms


def _arm(arm_id: str, content: str, *, forms: list[dict], provenance: str,
         source_partition: str | None, synthesis: dict | None,
         source_arm: str | None = None, recipe: str) -> dict:
    return {
        "id": arm_id,
        "channel": "declarative",
        "provenance": provenance,
        "control_for": None,
        "source_partition": source_partition,
        "recipe": recipe,
        "source_arm": source_arm,
        "writer_model": synthesis.get("writer_model") if synthesis else None,
        "synthesis_prompt_sha256": synthesis.get("prompt_sha256") if synthesis else None,
        "articulation_sha256": synthesis.get("articulation_sha256") if synthesis else None,
        "teaching_item_sha256": synthesis.get("teaching_item_sha256", []) if synthesis else [],
        "semantic_content_word_count": len(content.split()),
        "content_sha256": text_sha256(content),
        "forms": forms,
    }


def compile_bank(*, synthesis_path: str | Path, source_bank_path: str | Path) -> dict:
    synthesis = json.loads(Path(synthesis_path).read_text())
    source_bank = json.loads(Path(source_bank_path).read_text())
    rows_by_cell: dict[str, list[dict]] = {}
    for row in synthesis["rows"]:
        rows_by_cell.setdefault(row["cell_id"], []).append(row)
    output_cells = []
    for cell in source_bank["cells"]:
        cell_id, name = cell["id"], cell["construct"]
        arm_specs = {arm["id"]: arm for arm in cell["arms"]}
        source_id = BEST_SOURCE[cell_id]
        source_arm = arm_specs[source_id]
        source_text = next(form["prompt"] for form in source_arm["forms"]
                           if form["id"] == "canonical")
        name_arm = arm_specs["name"]
        arms = [{**name_arm, "source_partition": None}]

        # Construct-only rules are independent of either teaching fold and therefore reusable on
        # both public folds.  Preserve source-first and rule-first wordings as distinct hypotheses.
        for row in sorted(rows_by_cell[cell_id], key=lambda r: (
                str(r["source_partition"]), r["view"], r["variant"])):
            source_partition = row["source_partition"]
            origin = "self" if source_partition is None else source_partition.removeprefix(
                "residual_")
            suffix = f"{row['view']}_v{row['variant']}_from_{origin}"
            rule = row["articulation"]
            provenance = ("target_self_articulation" if source_partition is None
                          else "target_behavior_articulation")
            arms.append(_arm(
                f"rule_{suffix}", rule, forms=_matched_forms(rule),
                provenance=provenance, source_partition=source_partition,
                synthesis=row, recipe="rule_only"))
            source_then = f"{source_text}\n\n{rule}"
            arms.append(_arm(
                f"source_then_rule_{suffix}", source_then,
                forms=_source_then_forms(source_arm, rule), provenance=provenance,
                source_partition=source_partition, synthesis=row, source_arm=source_id,
                recipe="source_then_rule"))
            rule_then = f"{rule}\n\n{source_text}"
            arms.append(_arm(
                f"rule_then_source_{suffix}", rule_then, forms=_matched_forms(rule_then),
                provenance=provenance, source_partition=source_partition,
                synthesis=row, source_arm=source_id, recipe="rule_then_source"))
            if source_partition is not None:
                calibrated = calibration_text(row["calibration"])
                joined = f"{source_text}\n\n{rule}\n\n{calibrated}"
                arms.append(_arm(
                    f"source_rule_calibrated_{suffix}", joined,
                    forms=_source_then_forms(source_arm, f"{rule}\n\n{calibrated}"),
                    provenance=provenance, source_partition=source_partition,
                    synthesis=row, source_arm=source_id,
                    recipe="source_then_rule_then_calibration"))

        # Isolate whether policy strictness, rather than semantic articulation, closes distance.
        for partition in PARTITIONS:
            calibration = synthesis["panels"][f"{cell_id}:{partition}"]["calibration"]
            statement = calibration_text(calibration)
            origin = partition.removeprefix("residual_")
            source_content = f"{source_text}\n\n{statement}"
            arms.append(_arm(
                f"source_calibrated_from_{origin}", source_content,
                forms=_source_then_forms(source_arm, statement),
                provenance="target_behavior_articulation", source_partition=partition,
                synthesis=None, source_arm=source_id, recipe="source_plus_calibration"))
            name_content = f"Criterion: {name}.\n\n{statement}"
            arms.append(_arm(
                f"name_calibrated_from_{origin}", name_content,
                forms=_matched_forms(name_content),
                provenance="target_behavior_articulation", source_partition=partition,
                synthesis=None, recipe="name_plus_calibration"))
        output_cells.append({"id": cell_id, "domain": cell["domain"], "gi": cell["gi"],
                             "construct": name, "arms": arms})
    payload = {
        "schema": "target_policy_rule_arm_bank/v1",
        "status": "frozen-after-target-self-articulation-before-small-executor-scoring",
        "objective": "maximize direct 3B textual reconstruction of the 8B name-only policy",
        "anchor_policy": ("no external ground truth; target self-explication and public-fold "
                          "target behavior are the only learned signals"),
        "synthesis": {"path": str(synthesis_path), "sha256": sha256_file(synthesis_path)},
        "source_bank": {"path": str(source_bank_path), "sha256": sha256_file(source_bank_path)},
        "fold_policy": ("fold-indexed rules and calibration are evaluated only on the opposite "
                        "public fold; construct-only rules are evaluated on both"),
        "selection_priority": "isomorphism first; prompt length and diversity are subordinate",
        "cells": output_cells,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bank_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def execution_manifest(bank_path: str | Path, packet_manifest: str | Path) -> dict:
    return {
        "schema": "target_policy_rule_execution_manifest/v1",
        "status": "frozen-before-small-executor-scoring",
        "arm_bank_sha256": sha256_file(bank_path),
        "packet_manifest_sha256": sha256_file(packet_manifest),
        "phases": {"crossfit": PARTITIONS},
        "model_jobs": [{"id": "llama3_target_policy_rules",
                        "model": "meta-llama/Llama-3.2-3B-Instruct",
                        "role": "small", "arm_policy": "all"}],
        "primary_estimand": "opposite-fold direct policy isomorphism",
        "lockbox_status": "not authorized by this search manifest",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthesis", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execution-manifest-out", required=True)
    args = parser.parse_args()
    bank = compile_bank(synthesis_path=args.synthesis, source_bank_path=args.source_bank)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, indent=1))
    manifest = execution_manifest(out, args.packet_manifest)
    manifest_out = Path(args.execution_manifest_out)
    manifest_out.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({
        "out": str(out), "sha256": sha256_file(out),
        "execution_manifest": str(manifest_out),
        "execution_manifest_sha256": sha256_file(manifest_out),
        "n_cells": len(bank["cells"]),
        "n_arms": sum(len(cell["arms"]) for cell in bank["cells"]),
    }, indent=1))


if __name__ == "__main__":
    main()
