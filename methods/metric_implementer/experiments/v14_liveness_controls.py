"""Synthetic planted-positive and degenerate controls for the v14 sentinel."""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np

from ..recon_channel import mcq_logit_values_from_precomputed_behaviors
from .cr3_evidence_store import EvidenceCellStore
from .v14_behavioral_channel import (
    BEHAVIORAL_ARMS,
    canonical_template_sha256,
    corpus_token_counts,
    execute_rule_probe_cells,
    induce_requests,
    induction_prompt,
    blind_prompt,
    shuffled_state,
)
from .v14_value_bound import binary_entropy_bits, plugin_binary_mutual_information


CONTROL_SCHEMA = "cr3-v14-synthetic-liveness-controls-v1"


def synthetic_control_data() -> dict:
    demos = [
        "The record includes 17 blue shapes.",
        "The record includes several blue shapes.",
        "A card reports 42 green objects.",
        "A card reports many green objects.",
        "The note lists 8 orange pieces.",
        "The note lists some orange pieces.",
        "This line contains 305 red marks.",
        "This line contains numerous red marks.",
    ]
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    heldout = []
    heldout_labels = []
    words = ("violet", "silver", "indigo", "yellow", "maroon")
    for index in range(75):
        heldout.append(f"A {words[index % len(words)]} memo contains {1000 + index} tokens.")
        heldout_labels.append(1)
        heldout.append(f"A {words[index % len(words)]} memo contains several tokens.")
        heldout_labels.append(0)
    return {
        "demo_texts": demos, "demo_labels": labels,
        "heldout_texts": heldout,
        "heldout_labels": np.asarray(heldout_labels, dtype=np.uint8),
        "noun": "item",
    }


def run_liveness_constructor_controls(
    constructor, *, decoder_family: str, decoder_revision: str,
    templates: Mapping[str, object], store: EvidenceCellStore,
    query_batch_size: int = 1024,
) -> dict:
    del query_batch_size
    data = synthetic_control_data()
    panel_sha = hashlib.sha256(b"cr3-v14-synthetic-number-panel").hexdigest()
    canonical_state = int("".join(map(str, data["demo_labels"].tolist())), 2)
    shuffled = shuffled_state(canonical_state, 8, panel_sha)
    state_labels = {
        canonical_state: data["demo_labels"],
        shuffled: ((shuffled >> np.arange(7, -1, -1)) & 1).astype(np.uint8),
    }
    requests = []
    logical = {}
    for arm in BEHAVIORAL_ARMS:
        template = str(templates["behavioral"][arm])
        template_sha = canonical_template_sha256(template)
        for state, labels in state_labels.items():
            key = store.induction_key(
                template_sha256=template_sha, decoder_revision=decoder_revision,
                arm=arm, panel_sha256=panel_sha, state=state,
            )
            requests.append({
                "cache_key": key,
                "prompt": induction_prompt(
                    template=template, noun=data["noun"], texts=data["demo_texts"],
                    labels=labels.tolist(), max_chars=600, arm=arm,
                ),
                "arm": arm, "panel_sha256": panel_sha, "state": state,
                "template_sha256": template_sha, "example_texts": data["demo_texts"],
            })
            logical[(arm, state)] = key
        blind_sha = canonical_template_sha256("blind\x1f" + template)
        blind_key = store.induction_key(
            template_sha256=blind_sha, decoder_revision=decoder_revision,
            arm=arm, panel_sha256="synthetic-blind", state=-1,
        )
        requests.append({
            "cache_key": blind_key,
            "prompt": blind_prompt(template=template, noun=data["noun"], arm=arm),
            "arm": arm, "panel_sha256": "synthetic-blind", "state": -1,
            "template_sha256": blind_sha, "example_texts": [],
        })
        logical[(arm, -1)] = blind_key
    induced = induce_requests(
        constructor, requests=requests, store=store,
        corpus_counts=corpus_token_counts([*data["demo_texts"], *data["heldout_texts"]]),
    )
    voided = [key for key, row in induced.items() if row.get("void")]
    if voided:
        # Planted liveness criteria are constructed to be explainable without their
        # keywords; a void here is an instrument failure and must halt the gate.
        raise RuntimeError(f"liveness induction voided {len(voided)} planted cells")

    target_scores = np.zeros((1, len(data["demo_texts"])), dtype=float)
    target_scores[0] = data["demo_labels"]
    planted_distractors = [
        {
            "metric_id": "uppercase", "description": "Whether the item is entirely uppercase.",
            "scores": np.asarray([text.isupper() for text in data["demo_texts"]], dtype=float),
            "body": "Whether the item is entirely uppercase.",
        },
        {
            "metric_id": "question", "description": "Whether the item ends with a question mark.",
            "scores": np.asarray([text.endswith("?") for text in data["demo_texts"]], dtype=float),
            "body": "Whether the item ends with a question mark.",
        },
        {
            "metric_id": "short", "description": "Whether the item has fewer than four words.",
            "scores": np.asarray([len(text.split()) < 4 for text in data["demo_texts"]], dtype=float),
            "body": "Whether the item has fewer than four words.",
        },
    ]
    planted = mcq_logit_values_from_precomputed_behaviors(
        constructor, noun="item", candidate_prompt_texts=["synthetic planted state"],
        target_metric_id="numeral", target_description="Whether the item contains a numeral.",
        target_score_rows=target_scores, probe_texts=data["demo_texts"],
        distractors=planted_distractors, design_indices=np.arange(8),
        codebook_frozen_before_prompt_search=True, n_examples=8,
        n_reconstruction_draws=8, fixed_teaching_panel=True,
        mcq_prompt_template=str(templates["mcq"]),
    )[0]
    constant = np.zeros((1, 8), dtype=float)
    degenerate_distractors = [
        {
            "metric_id": f"constant-{index}",
            "description": f"Whether the item contains impossible marker {index}.",
            "scores": np.zeros(8),
            "body": f"Whether the item contains impossible marker {index}.",
        }
        for index in range(1, 4)
    ]
    degenerate = mcq_logit_values_from_precomputed_behaviors(
        constructor, noun="item", candidate_prompt_texts=["synthetic degenerate state"],
        target_metric_id="constant-0",
        target_description="Whether the item contains impossible marker zero.",
        target_score_rows=constant, probe_texts=data["demo_texts"],
        distractors=degenerate_distractors, design_indices=np.arange(8),
        codebook_frozen_before_prompt_search=True, n_examples=8,
        n_reconstruction_draws=8, fixed_teaching_panel=True,
        mcq_prompt_template=str(templates["mcq"]),
    )[0]
    return {
        "schema": CONTROL_SCHEMA, "stage": "constructor_complete",
        "decoder_family": str(decoder_family), "decoder_revision": str(decoder_revision),
        "panel_sha256": panel_sha, "canonical_state": canonical_state,
        "shuffled_state": shuffled, "logical_keys": {
            f"{arm}:{state}": key for (arm, state), key in logical.items()
        },
        "induced": induced,
        "mcq": {"planted": planted, "degenerate": degenerate},
    }


def finish_liveness_executor_controls(
    executor, *, constructor_result: Mapping[str, object], executor_revision: str,
    readout_id: str, store: EvidenceCellStore, query_batch_size: int = 2048,
) -> list[dict]:
    data = synthetic_control_data()
    induced = constructor_result["induced"]
    rules = {str(row["rule_sha256"]): str(row["rule"]) for row in induced.values()}
    executions = execute_rule_probe_cells(
        executor, rules=rules, probe_texts=data["heldout_texts"],
        executor_revision=executor_revision, readout_id=readout_id, store=store,
        max_chars=600, query_batch_size=query_batch_size,
    )
    logical = {}
    for label, key in constructor_result["logical_keys"].items():
        arm, state = label.rsplit(":", 1)
        logical[(arm, int(state))] = key
    target = data["heldout_labels"]

    def mi(key):
        rule_sha = str(induced[key]["rule_sha256"])
        prediction = [
            executions[(rule_sha, index)]["hard_prediction"]
            for index in range(len(data["heldout_texts"]))
        ]
        return plugin_binary_mutual_information(target, prediction)

    rows = []
    canonical = int(constructor_result["canonical_state"])
    shuffled = int(constructor_result["shuffled_state"])
    for arm in BEHAVIORAL_ARMS:
        annotated = mi(logical[(arm, canonical)])
        blind = mi(logical[(arm, -1)])
        shuffled_value = mi(logical[(arm, shuffled)])
        value = max(0.0, annotated - max(blind, shuffled_value))
        rows.append({
            "metric_key": "synthetic-numeral", "channel": "behavioral", "arm": arm,
            "decoder_family": constructor_result["decoder_family"],
            "structurally_valid": True, "planted_positive_value": float(value),
            "degenerate_control_value": 0.0,
            "blind_value": float(blind), "annotated_canonical_value": float(annotated),
            "cap": binary_entropy_bits(target),
        })
    for name in ("planted", "degenerate"):
        detail = constructor_result["mcq"][name]
        identification = detail["identification"]
        if name == "planted":
            rows.append({
                "metric_key": "synthetic-numeral", "channel": "mcq", "arm": None,
                "decoder_family": constructor_result["decoder_family"],
                "structurally_valid": True,
                "planted_positive_value": float(detail["value_mark"]),
                "degenerate_control_value": float(
                    constructor_result["mcq"]["degenerate"]["value_mark"]
                ),
                "blind_value": float(identification["no_demonstration_score"]),
                "annotated_canonical_value": float(detail["raw_target_option_probability"]),
                "cap": float(detail["value_cap"]),
            })
    return rows
