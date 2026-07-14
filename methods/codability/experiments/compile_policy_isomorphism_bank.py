#!/usr/bin/env python
"""Compile an isomorphism-first bank of name-anchored intact and composed articulations.

This search bank fixes two losses in the original source arms: explicit knowledge no longer replaces
the construct name, and every articulation uses the exact same outer form operators as the sparse
larger-reader target.  Candidate diversity is obtained through faithful source channels,
combinations, and orderings.  Cost is recorded but never used to discard a more isomorphic arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import sha256_file, text_sha256
from methods.codability.experiments.compile_fresh_name_arm_bank import (
    CELL_TARGETS,
    SOURCE_FILES,
    _corrected_rung,
)


COMPONENT_CHANNEL = {
    "definition": "declarative",
    "explanation": "explanatory",
    "full_rubric": "procedural",
    "exemplars_v2": "ostensive",
}

# Deliberately modest first generation: broad enough to expose interactions without spending the
# public search budget on arbitrary prompt tricks.  Order variants are kept because composition is
# part of the tacit-knowledge hypothesis.
RECIPES = [
    ("definition", ["definition"]),
    ("explanation", ["explanation"]),
    ("rubric", ["full_rubric"]),
    ("examples", ["exemplars_v2"]),
    ("definition_explanation", ["definition", "explanation"]),
    ("definition_rubric", ["definition", "full_rubric"]),
    ("explanation_rubric", ["explanation", "full_rubric"]),
    ("definition_explanation_rubric", ["definition", "explanation", "full_rubric"]),
    ("rubric_definition_explanation", ["full_rubric", "definition", "explanation"]),
    ("definition_rubric_examples", ["definition", "full_rubric", "exemplars_v2"]),
    ("rubric_examples_definition", ["full_rubric", "exemplars_v2", "definition"]),
    ("examples_definition_rubric", ["exemplars_v2", "definition", "full_rubric"]),
    ("full_stack", ["definition", "explanation", "full_rubric", "exemplars_v2"]),
    ("examples_first_full_stack", ["exemplars_v2", "definition", "explanation", "full_rubric"]),
]


def _words(text: str) -> int:
    return len(str(text).split())


def _matched_forms(content: str) -> list[dict]:
    values = [
        ("canonical", content),
        ("question", f"Does the item meet this criterion? {content} Answer YES or NO."),
        ("boilerplate", f"You are an expert evaluator. Evaluate strictly. {content}"),
    ]
    return [{"id": form_id, "prompt": prompt, "prompt_sha256": text_sha256(prompt),
             "total_word_count": _words(prompt)} for form_id, prompt in values]


def _content(name: str, components: dict[str, str], order: list[str]) -> str:
    labels = {"definition": "Definition", "explanation": "Explanation",
              "full_rubric": "Recognition rule", "exemplars_v2": "Boundary examples"}
    sections = [f"Criterion: {name}."]
    sections.extend(f"{labels[key]}: {components[key]}" for key in order)
    return "\n\n".join(sections)


def _arm(arm_id: str, content: str, components: list[str]) -> dict:
    channel = (COMPONENT_CHANNEL[components[0]] if len(components) == 1 else "composed")
    return {
        "id": arm_id, "channel": channel, "provenance": "source_composed",
        "control_for": None, "components": components,
        "composition_degree": len(components), "semantic_content_word_count": _words(content),
        "content_sha256": text_sha256(content), "forms": _matched_forms(content),
    }


def compile_bank(source_files: dict[str, Path] = SOURCE_FILES) -> dict:
    source_meta, cells = {}, []
    for domain, source_path in source_files.items():
        messages = json.loads(Path(source_path).read_text())
        source_meta[domain] = {"path": str(source_path), "sha256": sha256_file(source_path)}
        for (cell_domain, gi), _target_jobs in CELL_TARGETS.items():
            if cell_domain != domain:
                continue
            message = messages[str(gi)]
            name = message["name"]
            components = {key: _corrected_rung(message, key) for key in COMPONENT_CHANNEL}
            arms = [{"id": "name", "channel": "sparse", "provenance": "construct_name",
                     "control_for": None, "components": [], "composition_degree": 0,
                     "semantic_content_word_count": _words(name),
                     "content_sha256": text_sha256(name), "forms": _matched_forms(name)}]
            for recipe_id, order in RECIPES:
                arms.append(_arm(f"iso_{recipe_id}", _content(name, components, order), order))
            # The source dossier is a separately authored composition, not merely our section join.
            dossier = _corrected_rung(message, "dossier_v2")
            arms.append(_arm("iso_authored_dossier", f"Criterion: {name}.\n\n{dossier}",
                             ["definition", "explanation", "full_rubric", "exemplars_v2"]))
            cells.append({"id": f"N_{domain}_{gi}", "domain": domain, "gi": gi,
                          "construct": name, "arms": arms})
    payload = {
        "schema": "policy_isomorphism_arm_bank/v1",
        "status": "development-designed-before-any-v1-bank-executor-score",
        "objective": ("maximize direct 3B-articulated replication of the 8B name-only policy; "
                      "cost and unit count are subordinate"),
        "target": "Llama-3.1-8B-Instruct sparse construct-name form orbit",
        "source_messages": source_meta,
        "form_policy": "exact sparse-target outer wrappers around every candidate content",
        "treatment_boundary": ("construct name plus faithful intact source knowledge and frozen "
                               "compositions/orderings; no item-specific hints or generic optimizer"),
        "recipes": [{"id": recipe_id, "components": order} for recipe_id, order in RECIPES],
        "cells": cells,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bank_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_bank(bank: dict) -> list[str]:
    errors = []
    for cell in bank.get("cells", []):
        ids = [arm["id"] for arm in cell["arms"]]
        if len(ids) != len(set(ids)) or "name" not in ids:
            errors.append(f"{cell.get('id')}: duplicate arms or missing name")
        hashes = []
        for arm in cell["arms"]:
            if [form["id"] for form in arm["forms"]] != ["canonical", "question", "boilerplate"]:
                errors.append(f"{cell['id']}/{arm['id']}: invalid form orbit")
            hashes.extend(form["prompt_sha256"] for form in arm["forms"])
        if len(hashes) != len(set(hashes)):
            errors.append(f"{cell['id']}: duplicate prompts")
    return errors


def execution_manifest(bank_path: str | Path, packet_manifest: str | Path) -> dict:
    return {
        "schema": "policy_isomorphism_execution_manifest/v1",
        "status": "frozen-before-v1-bank-executor-scoring",
        "arm_bank_sha256": sha256_file(bank_path),
        "packet_manifest_sha256": sha256_file(packet_manifest),
        "phases": {
            "search": ["residual_prompt_selection"],
            "validation": ["residual_unit_certification"],
            "lockbox": ["residual_lockbox"],
        },
        "model_jobs": [{
            "id": "llama3_isomorphism", "model": "meta-llama/Llama-3.2-3B-Instruct",
            "role": "small", "arm_policy": "all",
        }],
        "primary_estimand": "direct item-level 3B-articulated to 8B-name policy isomorphism",
        "selection_priority": "isomorphism first; diversity only within the best isomorphism band",
        "lockbox_gate": ("selected diverse candidate set, matched controls, margins, and all hashes "
                         "must be frozen before smaller-reader lockbox execution"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--execution-manifest-out", required=True)
    args = parser.parse_args()
    bank = compile_bank()
    errors = validate_bank(bank)
    if errors:
        raise ValueError(errors)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, indent=1))
    manifest = execution_manifest(out, args.packet_manifest)
    manifest_out = Path(args.execution_manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256_file(out),
                      "execution_manifest": str(manifest_out),
                      "execution_manifest_sha256": sha256_file(manifest_out),
                      "n_cells": len(bank["cells"]),
                      "n_arms": sum(len(cell["arms"]) for cell in bank["cells"])}, indent=1))


if __name__ == "__main__":
    main()
