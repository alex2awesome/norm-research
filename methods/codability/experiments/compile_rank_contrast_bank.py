#!/usr/bin/env python
"""Compile rank-contrast micro-rules, syntheses, and ostensive curricula into a frozen bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import sha256_file, text_sha256
from methods.codability.experiments.compile_residual_isomorphism_bank import PARTITIONS
from methods.codability.experiments.compile_target_policy_rule_bank import _matched_forms


def _parent_then_forms(parent: dict, addition: str) -> list[dict]:
    forms = []
    for form in parent["forms"]:
        prompt = f"{form['prompt']}\n\n{addition}"
        forms.append({"id": form["id"], "prompt": prompt,
                      "prompt_sha256": text_sha256(prompt),
                      "total_word_count": len(prompt.split())})
    return forms


def _arm(arm_id: str, content: str, forms: list[dict], *, channel: str,
         source_partition: str, parent_id: str, parent_arm_id: str,
         recipe: str, row: dict | None, teaching_hashes: list[str]) -> dict:
    return {
        "id": arm_id,
        "channel": channel,
        "provenance": "target_rank_articulation",
        "control_for": None,
        "source_partition": source_partition,
        "recipe": recipe,
        "parent_id": parent_id,
        "parent_arm_id": parent_arm_id,
        "writer_model": row.get("writer_model") if row else None,
        "synthesis_prompt_sha256": row.get("prompt_sha256") if row else None,
        "articulation_sha256": row.get("articulation_sha256") if row else None,
        "teaching_item_sha256": teaching_hashes,
        "semantic_content_word_count": len(content.split()),
        "content_sha256": text_sha256(content),
        "forms": forms,
    }


def _micro_rule_text(construct: str, rules: list[str]) -> str:
    values = "\n".join(f"- {value}" for value in rules)
    return (
        f"Criterion: {construct}\n\nRelative-order distinctions and tie-breakers:\n{values}\n\n"
        "Apply these only as criterion-specific refinements. Preserve the ordinary holistic meaning "
        "of the criterion and integrate conflicting cues rather than counting them mechanically."
    )


def _curriculum_text(construct: str, records: list[tuple[dict, str]]) -> str:
    parts = [
        f"Criterion: {construct}",
        ("Use the following explicit comparison anchors to learn criterion-specific boundaries. "
         "Generalize the stated distinctions; do not reward surface resemblance."),
    ]
    for index, (pair, micro_rule) in enumerate(records, 1):
        parts.append(
            f"[Contrast {index}: HIGHER]\n{pair['high']['text']}\n"
            f"[Contrast {index}: LOWER]\n{pair['low']['text']}\n"
            f"Reusable distinction: {micro_rule}"
        )
    return "\n\n".join(parts)


def compile_bank(*, synthesis_path: str | Path, source_bank_path: str | Path,
                 rule_bank_path: str | Path) -> dict:
    synthesis = json.loads(Path(synthesis_path).read_text())
    source_bank = json.loads(Path(source_bank_path).read_text())
    rule_bank = json.loads(Path(rule_bank_path).read_text())
    source_cells = {cell["id"]: cell for cell in source_bank["cells"]}
    rule_cells = {cell["id"]: cell for cell in rule_bank["cells"]}
    contexts = synthesis["contexts"]
    contrast_by_context = {}
    for row in synthesis["contrasts"]:
        contrast_by_context.setdefault(row["context_key"], []).append(row)
    articulation_by_context = {}
    for row in synthesis["rows"]:
        articulation_by_context.setdefault(row["context_key"], []).append(row)

    cells = []
    for cell_id, source_cell in source_cells.items():
        source_specs = {arm["id"]: arm for arm in source_cell["arms"]}
        rule_specs = {arm["id"]: arm for arm in rule_cells[cell_id]["arms"]}
        arms = [{**source_specs["name"], "source_partition": None}]
        cell_contexts = sorted(
            (value for value in contexts.values() if value["cell_id"] == cell_id),
            key=lambda value: (value["source_partition"], value["parent_id"]))
        for context in cell_contexts:
            context_key = (f"{cell_id}:{context['source_partition']}:"
                           f"{context['parent_id']}")
            if context_key not in contexts:
                raise ValueError(f"context-key mismatch: {context_key}")
            origin = context["source_partition"].removeprefix("residual_")
            parent_spec = (source_specs["name"] if context["parent_id"] == "name"
                           else rule_specs[context["parent_arm_id"]])
            parent_text = next(form["prompt"] for form in parent_spec["forms"]
                               if form["id"] == "canonical")
            contrast_rows = sorted(contrast_by_context[context_key],
                                   key=lambda value: value["pair_index"])
            micro_rules = [row["micro_rule"] for row in contrast_rows]
            teaching_hashes = [
                value for pair in context["pairs"]
                for value in (pair["high"]["text_sha256"], pair["low"]["text_sha256"])]

            micro_text = _micro_rule_text(context["construct"], micro_rules)
            for recipe, content, forms in (
                    ("micro_rules_only", micro_text, _matched_forms(micro_text)),
                    ("parent_then_micro_rules", f"{parent_text}\n\n{micro_text}",
                     _parent_then_forms(parent_spec, micro_text))):
                arm_id = (f"rank_micro_{recipe}_parent-{context['parent_id']}_from_{origin}")
                arms.append(_arm(
                    arm_id, content, forms, channel="declarative",
                    source_partition=context["source_partition"],
                    parent_id=context["parent_id"], parent_arm_id=context["parent_arm_id"],
                    recipe=recipe, row=None, teaching_hashes=teaching_hashes))

            for k in (2, 4):
                records = [(context["pairs"][row["pair_index"]], row["micro_rule"])
                           for row in contrast_rows[:k]]
                curriculum = _curriculum_text(context["construct"], records)
                content = f"{parent_text}\n\n{curriculum}"
                arm_id = f"rank_curriculum_k{k}_parent-{context['parent_id']}_from_{origin}"
                curriculum_hashes = [
                    value for pair, _ in records
                    for value in (pair["high"]["text_sha256"], pair["low"]["text_sha256"])]
                arms.append(_arm(
                    arm_id, content, _parent_then_forms(parent_spec, curriculum),
                    channel="ostensive", source_partition=context["source_partition"],
                    parent_id=context["parent_id"], parent_arm_id=context["parent_arm_id"],
                    recipe=f"parent_then_rank_curriculum_k{k}", row=None,
                    teaching_hashes=curriculum_hashes))

            for row in sorted(articulation_by_context[context_key],
                              key=lambda value: (value["mode"], value["variant"])):
                rule = row["articulation"]
                stem = (f"rank_{row['mode']}_v{row['variant']}_parent-"
                        f"{context['parent_id']}")
                arms.append(_arm(
                    f"{stem}_standalone_from_{origin}", rule, _matched_forms(rule),
                    channel="procedural",
                    source_partition=context["source_partition"],
                    parent_id=context["parent_id"], parent_arm_id=context["parent_arm_id"],
                    recipe=f"{row['mode']}_standalone", row=row,
                    teaching_hashes=teaching_hashes))
                combined = f"{parent_text}\n\n{rule}"
                arms.append(_arm(
                    f"{stem}_parent_then_from_{origin}", combined,
                    _parent_then_forms(parent_spec, rule),
                    channel="procedural", source_partition=context["source_partition"],
                    parent_id=context["parent_id"], parent_arm_id=context["parent_arm_id"],
                    recipe=f"parent_then_{row['mode']}", row=row,
                    teaching_hashes=teaching_hashes))
        cells.append({
            "id": cell_id, "domain": source_cell["domain"], "gi": source_cell["gi"],
            "construct": source_cell["construct"], "arms": arms,
        })
    payload = {
        "schema": "rank_contrast_arm_bank/v1",
        "status": "frozen-after-rank-articulation-before-small-executor-scoring",
        "objective": "maximize opposite-fold item-order and complete policy reconstruction",
        "anchor_policy": "fixed 8B name-only behavior; no external ground truth or lockbox access",
        "synthesis": {"path": str(synthesis_path), "sha256": sha256_file(synthesis_path)},
        "source_bank": {"path": str(source_bank_path),
                        "sha256": sha256_file(source_bank_path)},
        "rule_bank": {"path": str(rule_bank_path), "sha256": sha256_file(rule_bank_path)},
        "fold_policy": "rank content from one public fold is executed only on the opposite fold",
        "selection_priority": "complete policy isomorphism first; text cost subordinate",
        "cells": cells,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bank_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def execution_manifest(bank_path: str | Path, packet_manifest: str | Path) -> dict:
    return {
        "schema": "rank_contrast_execution_manifest/v1",
        "status": "frozen-before-small-executor-scoring",
        "arm_bank_sha256": sha256_file(bank_path),
        "packet_manifest_sha256": sha256_file(packet_manifest),
        "phases": {"crossfit": PARTITIONS},
        "model_jobs": [{
            "id": "llama3_rank_contrasts",
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "role": "small",
            "arm_policy": "all",
        }],
        "primary_estimand": "opposite-fold direct policy isomorphism with rank-bearing content",
        "lockbox_status": "not authorized by this development manifest",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthesis", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--rule-bank", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execution-manifest-out", required=True)
    args = parser.parse_args()
    bank = compile_bank(
        synthesis_path=args.synthesis, source_bank_path=args.source_bank,
        rule_bank_path=args.rule_bank)
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
