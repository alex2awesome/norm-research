"""Batched exhaustive v14 Reconstruction-MCQ state adapter."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from ..recon_channel import (
    _RECON_MCQ_LOGITS,
    mcq_logit_values_from_precomputed_behaviors,
    mcq_option_order_design,
)
from .cr3_evidence_store import EvidenceCellStore, evidence_cell_key
from .v14_value_bound import enumerate_states


STATE_TABLE_SCHEMA = "cr3-v14-mcq-state-tables-v1"
DEFAULT_MCQ_TEMPLATE = _RECON_MCQ_LOGITS


def _sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _cell_key(
    *, constructor_revision: str, template_sha256: str, menu_sha256: str,
    panel_sha256: str, state: int, permutation_sha256: str,
) -> str:
    return evidence_cell_key("v14_mcq_state", {
        "constructor_revision": str(constructor_revision),
        "template_sha256": str(template_sha256),
        "menu_sha256": str(menu_sha256),
        "panel_sha256": str(panel_sha256),
        "state": int(state),
        "permutation_sha256": str(permutation_sha256),
    })


def evaluate_mcq_state_tables_v14(
    reconstructor, *, design_manifest: Mapping[str, object], noun: str,
    target_metric_id: str, target_description: str, distractors: Sequence[object],
    probe_texts: Sequence[str], constructor_revision: str,
    store: EvidenceCellStore, template: str = DEFAULT_MCQ_TEMPLATE,
    max_chars: int = 600, n_reconstruction_draws: int = 8,
    query_batch_size: int = 512,
    state_indices_by_panel: Sequence[Sequence[int]] | None = None,
) -> dict:
    panels = list(design_manifest["panels"])
    panel_size = int(design_manifest["panel_size"])
    if panel_size not in {6, 8}:
        raise ValueError("v14 MCQ supports the frozen k=6 fan-out and k=8 sentinel designs")
    n_states = 1 << panel_size
    if int(n_reconstruction_draws) < 2:
        raise ValueError("MCQ needs a counterbalanced permutation block")
    option_descriptions = [str(target_description)] + [
        str(item.get("description") if isinstance(item, Mapping) else item.description)
        for item in distractors
    ]
    # The noun is part of every query, so it belongs in the cache namespace even
    # when two tasks happen to expose identical option text.
    menu_sha = _sha({"noun": str(noun), "option_descriptions": option_descriptions})
    template_sha = hashlib.sha256(str(template).encode("utf-8")).hexdigest()
    permutation = mcq_option_order_design(len(option_descriptions), int(n_reconstruction_draws))
    permutation_sha = _sha(permutation)
    blind_key = evidence_cell_key("v14_mcq_blind", {
        "constructor_revision": str(constructor_revision),
        "template_sha256": template_sha,
        "menu_sha256": menu_sha,
        "permutation_sha256": permutation_sha,
    })
    blind = store.get(blind_key)
    raw_lift = np.full((len(panels), n_states), np.nan, dtype=float)
    clipped = np.full_like(raw_lift, np.nan)
    normalized_lift = np.full_like(raw_lift, np.nan)
    main_probability = np.full_like(raw_lift, np.nan)
    shuffled_probability = np.full_like(raw_lift, np.nan)
    annotation_accuracy = np.full_like(raw_lift, np.nan)
    hits = misses = 0
    state_bits = enumerate_states(panel_size)
    for panel_position, panel in enumerate(panels):
        requested_states = (
            list(range(n_states)) if state_indices_by_panel is None
            else sorted(set(map(int, state_indices_by_panel[panel_position])))
        )
        if not requested_states or requested_states[0] < 0 or requested_states[-1] >= n_states:
            raise ValueError("MCQ requested state lies outside the frozen panel code space")
        panel_indices = np.asarray(panel["indices"], dtype=int)
        keys = {
            state:
            _cell_key(
                constructor_revision=constructor_revision, template_sha256=template_sha,
                menu_sha256=menu_sha, panel_sha256=str(panel["panel_sha256"]),
                state=state, permutation_sha256=permutation_sha,
            )
            for state in requested_states
        }
        payloads = {state: store.get(keys[state]) for state in requested_states}
        missing = [state for state in requested_states if payloads[state] is None]
        hits += len(requested_states) - len(missing)
        misses += len(missing)
        if missing:
            rows = np.zeros((len(missing), len(probe_texts)), dtype=float)
            rows[:, panel_indices] = state_bits[missing]
            details = mcq_logit_values_from_precomputed_behaviors(
                reconstructor,
                noun=str(noun),
                candidate_prompt_texts=[
                    f"v14 finite state {state:0{panel_size}b}" for state in missing
                ],
                target_metric_id=str(target_metric_id),
                target_description=str(target_description),
                target_score_rows=rows,
                probe_texts=list(map(str, probe_texts)),
                distractors=list(distractors),
                design_indices=panel_indices,
                codebook_frozen_before_prompt_search=True,
                n_examples=panel_size,
                n_reconstruction_draws=int(n_reconstruction_draws),
                max_chars=int(max_chars),
                query_batch_size=int(query_batch_size),
                fixed_no_demo_canonical_probabilities=(
                    None if blind is None else np.asarray(blind["canonical_probabilities"], dtype=float)
                ),
                fixed_teaching_panel=True,
                mcq_prompt_template=str(template),
            )
            for state, detail in zip(missing, details):
                identification = detail["identification"]
                if blind is None:
                    canonical = identification["conditions"]["no_demonstrations"][
                        "canonical_choice_probabilities"
                    ]
                    blind = store.put(blind_key, "v14_mcq_blind", {
                        "canonical_probabilities": canonical,
                        "target_probability": float(identification["no_demonstration_score"]),
                        "template_sha256": template_sha,
                        "menu_sha256": menu_sha,
                    })
                lift = float(identification["annotation_lift_over_strongest_control"])
                q0 = float(identification["no_demonstration_score"])
                denominator = max(1e-12, 1.0 - q0)
                payloads[state] = store.put(keys[state], "v14_mcq_state", {
                    "panel_sha256": str(panel["panel_sha256"]),
                    "state": int(state),
                    "raw_lift": lift,
                    "clipped_value": float(np.clip(lift, 0.0, 1.0 - q0)),
                    "normalized_lift": float(lift / denominator),
                    "raw_target_probability": float(detail["raw_target_option_probability"]),
                    "shuffled_target_probability": float(identification["shuffled_label_score"]),
                    "annotation_accuracy": float(identification["identification_acc"]),
                    "blind_target_probability": q0,
                })
        for state in requested_states:
            payload = payloads[state]
            if payload is None:
                raise RuntimeError("MCQ state cache is incomplete")
            raw_lift[panel_position, state] = float(payload["raw_lift"])
            clipped[panel_position, state] = float(payload["clipped_value"])
            normalized_lift[panel_position, state] = float(payload["normalized_lift"])
            main_probability[panel_position, state] = float(payload["raw_target_probability"])
            shuffled_probability[panel_position, state] = float(payload["shuffled_target_probability"])
            annotation_accuracy[panel_position, state] = float(payload["annotation_accuracy"])
    observed_mask = np.isfinite(raw_lift)
    if blind is None or not np.any(observed_mask) or np.any(~np.isfinite(raw_lift[observed_mask])):
        raise RuntimeError("MCQ state evaluation did not produce finite requested evidence")
    canonical_accuracy = [
        annotation_accuracy[position, int("".join(
            map(str, np.asarray(panel.get("target_state_bits", []), dtype=int).tolist())
        ), 2)]
        for position, panel in enumerate(panels)
        if len(panel.get("target_state_bits", [])) == panel_size
    ]
    return {
        "schema": STATE_TABLE_SCHEMA,
        "channel": "mcq",
        "panel_design_sha256": str(design_manifest["design_sha256"]),
        "constructor_revision": str(constructor_revision),
        "template_sha256": template_sha,
        "menu_sha256": menu_sha,
        "permutation_design_sha256": permutation_sha,
        "raw_lift": raw_lift,
        "clipped_value": clipped,
        "normalized_lift": normalized_lift,
        "raw_target_probability": main_probability,
        "shuffled_target_probability": shuffled_probability,
        "annotation_accuracy": annotation_accuracy,
        "observed_state_mask": observed_mask,
        "state_scope": "exhaustive" if state_indices_by_panel is None else "observed_only",
        "blind": blind,
        "best_explanation_rate": (
            float(np.mean(canonical_accuracy)) if canonical_accuracy else None
        ),
        "cache_hits": int(hits),
        "cache_misses": int(misses),
        "non_disclosure": {
            "candidate_prompt_text_passed_to_query_builder": False,
            "queries_depend_only_on_panel_texts_labels_and_frozen_menu": True,
        },
    }
