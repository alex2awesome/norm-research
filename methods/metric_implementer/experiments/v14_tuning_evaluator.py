"""Frozen sparse reference-set evaluator used by bounded v14 GEPA rounds."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..recon_channel import mcq_logit_values_from_precomputed_behaviors
from .cr3_evidence_store import EvidenceCellStore, evidence_cell_key
from .v14_behavioral_channel import (
    BEHAVIORAL_ARMS,
    blind_prompt,
    canonical_template_sha256,
    corpus_token_counts,
    execute_rule_probe_cells,
    induce_requests,
    induction_prompt,
    shuffled_state,
)
from .v14_decoder_tuning import stratified_reference_states
from .v14_value_bound import (
    binary_entropy_bits,
    plugin_binary_mutual_information,
    signatures_to_states,
)


REFERENCE_SCHEMA = "cr3-v14-gepa-reference-set-v1"
TRANSFER_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
_CPU_EMBEDDERS = {}


def _cpu_embedder(model_name: str):
    model = _CPU_EMBEDDERS.get(str(model_name))
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(model_name), device="cpu")
        _CPU_EMBEDDERS[str(model_name)] = model
    return model


def cached_probe_embeddings(
    texts: Sequence[str], *, cache_path: str | Path,
    model_name: str = TRANSFER_EMBEDDING_MODEL, batch_size: int = 64,
) -> np.ndarray:
    """Load or create the frozen BGE probe embedding matrix."""
    source_sha = hashlib.sha256(
        "\x1e".join(map(str, texts)).encode("utf-8")
    ).hexdigest()
    path = Path(cache_path)
    if path.is_file():
        with np.load(path, allow_pickle=False) as artifact:
            if str(artifact["model_name"]) != str(model_name) or str(
                artifact["text_sha256"]
            ) != source_sha:
                raise RuntimeError("probe embedding cache belongs to a different model or corpus")
            matrix = np.asarray(artifact["embeddings"], dtype=float)
    else:
        # Development-reference preparation is explicitly CPU-only; never let the
        # embedding helper auto-select an unapproved CUDA device on sk3.
        model = _cpu_embedder(str(model_name))
        matrix = np.asarray(model.encode(
            list(map(str, texts)), batch_size=int(batch_size),
            normalize_embeddings=True, show_progress_bar=False,
        ), dtype=float)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
        np.savez_compressed(
            temporary, embeddings=matrix, model_name=np.asarray(model_name),
            text_sha256=np.asarray(source_sha),
        )
        os.replace(temporary, path)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or np.any(~np.isfinite(matrix)):
        raise RuntimeError("invalid probe embedding matrix")
    return matrix


def cached_text_embeddings(
    texts: Sequence[str], *, store: EvidenceCellStore,
    model_name: str = TRANSFER_EMBEDDING_MODEL, batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Content-addressed CPU BGE embeddings for induced rules and demo texts."""
    unique = list(dict.fromkeys(map(str, texts)))
    output = {}
    missing = []
    keys = {}
    for text in unique:
        key = evidence_cell_key("v14_text_embedding", {
            "model": str(model_name),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
        keys[text] = key
        row = store.get(key)
        if row is None:
            missing.append(text)
        else:
            output[text] = np.asarray(row["embedding"], dtype=float)
    if missing:
        model = _cpu_embedder(str(model_name))
        matrix = np.asarray(model.encode(
            missing, batch_size=int(batch_size), normalize_embeddings=True,
            show_progress_bar=False,
        ), dtype=float)
        for text, vector in zip(missing, matrix):
            store.put(keys[text], "v14_text_embedding", {
                "model": str(model_name), "embedding": vector.astype(float).tolist(),
            })
            output[text] = vector
    return output


def _state(bits: Sequence[int]) -> int:
    vector = np.asarray(bits, dtype=int)
    return int(np.sum(vector * (1 << np.arange(len(vector) - 1, -1, -1))))


def freeze_reference_set(
    *, metric_key: str, panel_design: Mapping[str, object],
    candidate_signatures: np.ndarray, target_signature: Sequence[float],
    decoder_development_indices: Sequence[int], candidate_reference_values: Sequence[float],
    probe_embeddings: np.ndarray | None = None, n_trials: int = 4, n_states: int = 6,
) -> dict:
    """Freeze canonical and mined high/mid/low states before template mutation."""
    panels = list(panel_design["panels"])
    families = sorted({str(row["decoder_family"]) for row in panels})
    chosen = []
    for family in families:
        family_rows = [row for row in panels if str(row["decoder_family"]) == family]
        chosen.append(min(family_rows, key=lambda row: str(row["panel_sha256"])))
    remaining = sorted(
        (row for row in panels if row not in chosen), key=lambda row: str(row["panel_sha256"]),
    )
    chosen.extend(remaining[:max(0, int(n_trials) - len(chosen))])
    signatures = np.asarray(candidate_signatures, dtype=float)
    values = np.asarray(candidate_reference_values, dtype=float)
    target = np.asarray(target_signature, dtype=float) > 0.5
    if signatures.ndim != 2 or signatures.shape[0] != len(values):
        raise ValueError("reference signatures and values are not aligned")
    rows = []
    embeddings = None if probe_embeddings is None else np.asarray(probe_embeddings, dtype=float)
    if embeddings is not None and (
        embeddings.ndim != 2 or embeddings.shape[0] != signatures.shape[1]
    ):
        raise ValueError("probe embeddings do not align with signatures")
    ddec = np.asarray(decoder_development_indices, dtype=int)
    for panel in chosen[:int(n_trials)]:
        indices = list(map(int, panel["indices"]))
        realized = signatures_to_states(signatures, [indices])[:, 0].astype(int)
        canonical = _state(target[indices].astype(int))
        reference = stratified_reference_states(
            canonical_state=canonical, prompt_states=realized,
            prompt_values=values, metric_key=metric_key, trial=int(panel["trial"]),
            n_states=n_states,
        )
        transfer = {"available": False, "near_indices": [], "far_indices": []}
        if embeddings is not None:
            demo = embeddings[np.asarray(indices, dtype=int)]
            heldout = embeddings[ddec]
            demo = demo / np.linalg.norm(demo, axis=1, keepdims=True)
            heldout = heldout / np.linalg.norm(heldout, axis=1, keepdims=True)
            similarity = np.max(heldout @ demo.T, axis=1)
            order = np.argsort(-similarity, kind="stable")
            size = len(order) // 2
            transfer = {
                "available": True,
                "near_indices": ddec[order[:size]].astype(int).tolist(),
                "far_indices": ddec[order[-size:]].astype(int).tolist(),
                "embedding_model": "BAAI/bge-large-en-v1.5",
            }
        rows.append({
            "trial": int(panel["trial"]), "panel_sha256": str(panel["panel_sha256"]),
            "decoder_family": str(panel["decoder_family"]), "indices": indices,
            **reference, "near_far_transfer": transfer,
            "identification_ceiling_bits": float(panel["identification_mi_bits"]),
        })
    return {
        "schema": REFERENCE_SCHEMA,
        "metric_key": str(metric_key),
        "decoder_development_indices": list(map(int, decoder_development_indices)),
        "n_trials": len(rows), "n_states_per_trial": int(n_states), "trials": rows,
        "selection_values": "frozen candidate_reference_values",
    }


def induce_behavioral_reference_templates(
    constructor, *, templates: Sequence[str], arm: str,
    contexts: Sequence[Mapping[str, object]], decoder_family: str,
    decoder_revision: str, store: EvidenceCellStore, max_chars: int = 600,
) -> dict:
    if arm not in BEHAVIORAL_ARMS:
        raise ValueError("unknown behavioral tuning arm")
    output = {}
    for context in contexts:
        metric_key = str(context["metric_key"])
        texts = list(map(str, context["probe_texts"]))
        allowed = set(map(
            int, context["reference_set"]["decoder_development_indices"],
        ))
        allowed.update(
            int(index)
            for trial in context["reference_set"]["trials"]
            for index in trial["indices"]
        )
        counts = corpus_token_counts([texts[index] for index in sorted(allowed)])
        requests = []
        logical = {}
        for template in templates:
            template_sha = canonical_template_sha256(template)
            for trial in context["reference_set"]["trials"]:
                if str(trial["decoder_family"]) != str(decoder_family):
                    continue
                indices = list(map(int, trial["indices"]))
                demos = [texts[index] for index in indices]
                declared_states = set(trial["search_states"] + trial["heldout_prompt_states"])
                declared_states.update(
                    shuffled_state(state, len(indices), str(trial["panel_sha256"]))
                    for state in list(declared_states)
                )
                for state in sorted(declared_states):
                    labels = ((int(state) >> np.arange(len(indices) - 1, -1, -1)) & 1).tolist()
                    key = store.induction_key(
                        template_sha256=template_sha, decoder_revision=decoder_revision,
                        arm=arm, panel_sha256=str(trial["panel_sha256"]), state=int(state),
                    )
                    requests.append({
                        "cache_key": key,
                        "prompt": induction_prompt(
                            template=template, noun=str(context["noun"]), texts=demos,
                            labels=labels, max_chars=max_chars, arm=arm,
                        ),
                        "arm": arm, "panel_sha256": str(trial["panel_sha256"]),
                        "state": int(state), "template_sha256": template_sha,
                        "example_texts": demos,
                    })
                    logical[(template_sha, int(trial["trial"]), int(state))] = key
            blind_sha = canonical_template_sha256("blind\x1f" + template)
            blind_key = store.induction_key(
                template_sha256=blind_sha, decoder_revision=decoder_revision,
                arm=arm, panel_sha256=f"blind:{metric_key}", state=-1,
            )
            requests.append({
                "cache_key": blind_key,
                "prompt": blind_prompt(template=template, noun=str(context["noun"]), arm=arm),
                "arm": arm, "panel_sha256": f"blind:{metric_key}", "state": -1,
                "template_sha256": blind_sha, "example_texts": [],
            })
            logical[(template_sha, -1, -1)] = blind_key
        induced = induce_requests(
            constructor, requests=requests, store=store, corpus_counts=counts,
        )
        output[metric_key] = {"logical_keys": logical, "induced": induced}
    return output


def score_behavioral_reference_templates(
    executor, *, templates: Sequence[str], arm: str,
    contexts: Sequence[Mapping[str, object]], induction_rows: Mapping[str, object],
    executor_revision: str, readout_id: str, store: EvidenceCellStore,
    max_chars: int = 600, query_batch_size: int = 4096,
    decoder_family: str | None = None,
) -> list[dict]:
    output = []
    for context in contexts:
        metric_key = str(context["metric_key"])
        ddec = list(map(int, context["reference_set"]["decoder_development_indices"]))
        ddec_texts = [str(context["probe_texts"][index]) for index in ddec]
        target = (np.asarray(context["target_signature"], dtype=float)[ddec] > 0.5).astype(int)
        entropy = binary_entropy_bits(target)
        logical = induction_rows[metric_key]["logical_keys"]
        induced = induction_rows[metric_key]["induced"]
        rules = {str(row["rule_sha256"]): str(row["rule"]) for row in induced.values()}
        demo_texts = [
            str(context["probe_texts"][index])
            for trial in context["reference_set"]["trials"]
            for index in trial["indices"]
        ]
        embeddings = cached_text_embeddings(
            [*rules.values(), *demo_texts], store=store,
        )
        executions = execute_rule_probe_cells(
            executor, rules=rules, probe_texts=ddec_texts,
            executor_revision=executor_revision, readout_id=readout_id, store=store,
            max_chars=max_chars, query_batch_size=query_batch_size,
        )
        ddec_position = {int(index): position for position, index in enumerate(ddec)}

        def mi_for_key(key, positions=None):
            rule_sha = str(induced[key]["rule_sha256"])
            predicted = [executions[(rule_sha, index)]["hard_prediction"] for index in range(len(ddec))]
            if positions is None:
                return plugin_binary_mutual_information(target, predicted)
            local = [ddec_position[int(index)] for index in positions]
            return plugin_binary_mutual_information(target[local], np.asarray(predicted)[local])

        def copy_penalty(key, trial):
            rule = str(induced[key]["rule"])
            rule_vector = embeddings[rule]
            demo_vectors = np.vstack([
                embeddings[str(context["probe_texts"][index])] for index in trial["indices"]
            ])
            similarity = float(np.max(demo_vectors @ rule_vector))
            return 0.05 * max(0.0, similarity)

        for template in templates:
            template_sha = canonical_template_sha256(template)
            blind_mi = mi_for_key(logical[(template_sha, -1, -1)])
            for trial in context["reference_set"]["trials"]:
                if decoder_family is not None and str(trial["decoder_family"]) != str(
                    decoder_family
                ):
                    continue
                for split_name in ("search_states", "heldout_prompt_states"):
                    for state in trial[split_name]:
                        key = logical[(template_sha, int(trial["trial"]), int(state))]
                        shuffled = shuffled_state(
                            state, len(trial["indices"]), str(trial["panel_sha256"])
                        )
                        shuffled_key = logical[(template_sha, int(trial["trial"]), shuffled)]
                        transmission = mi_for_key(key)
                        shuffled_mi = mi_for_key(shuffled_key)
                        lift = transmission - max(blind_mi, shuffled_mi)
                        penalty = copy_penalty(key, trial)
                        transfer = trial["near_far_transfer"]
                        near_lift = far_lift = ratio = None
                        if transfer["available"]:
                            near = transfer["near_indices"]
                            far = transfer["far_indices"]
                            near_lift = mi_for_key(key, near) - max(
                                mi_for_key(logical[(template_sha, -1, -1)], near),
                                mi_for_key(shuffled_key, near),
                            )
                            far_lift = mi_for_key(key, far) - max(
                                mi_for_key(logical[(template_sha, -1, -1)], far),
                                mi_for_key(shuffled_key, far),
                            )
                            ratio = (
                                float(far_lift / near_lift) if near_lift > 1e-12 else None
                            )
                        output.append({
                            "template_sha256": template_sha, "metric_key": metric_key,
                            "decoder_family": str(trial["decoder_family"]),
                            "trial": int(trial["trial"]), "state": int(state),
                            "reference_split": (
                                "search" if split_name == "search_states" else "heldout_prompt"
                            ),
                            "transmission_bits": float(transmission),
                            "blind_bits": float(blind_mi), "shuffled_bits": float(shuffled_mi),
                            "raw_lift_bits": float(lift),
                            "normalized_fitness_before_copy_penalty": (
                                float(lift / entropy) if entropy > 0.0 else 0.0
                            ),
                            "embedding_copy_penalty": penalty,
                            "normalized_fitness": (
                                float(lift / entropy - penalty) if entropy > 0.0 else -penalty
                            ),
                            "target_entropy_bits": float(entropy),
                            "near_lift_bits": None if near_lift is None else float(near_lift),
                            "far_lift_bits": None if far_lift is None else float(far_lift),
                            "far_near_ratio": ratio,
                            "induced_rule": str(induced[key]["rule"]),
                        })
    return output


def score_mcq_reference_templates(
    reconstructor, *, templates: Sequence[str], contexts: Sequence[Mapping[str, object]],
    decoder_family: str, constructor_revision: str, store: EvidenceCellStore | None = None,
    query_batch_size: int = 1024,
) -> list[dict]:
    output = []
    for context in contexts:
        texts = list(map(str, context["probe_texts"]))
        for template in templates:
            template_sha = canonical_template_sha256(template)
            for trial in context["reference_set"]["trials"]:
                if str(trial["decoder_family"]) != str(decoder_family):
                    continue
                states = sorted(set(trial["search_states"] + trial["heldout_prompt_states"]))
                cache_key = evidence_cell_key("v14_mcq_reference_batch", {
                    "constructor_revision": str(constructor_revision),
                    "template_sha256": template_sha,
                    "metric_key": str(context["metric_key"]),
                    "panel_sha256": str(trial["panel_sha256"]),
                    "states": states,
                    "canonical_flag_version": 1,
                })
                cached = None if store is None else store.get(cache_key)
                if cached is not None:
                    output.extend(cached["rows"])
                    continue
                rows = np.zeros((len(states), len(texts)), dtype=float)
                indices = np.asarray(trial["indices"], dtype=int)
                for position, state in enumerate(states):
                    rows[position, indices] = (
                        (int(state) >> np.arange(len(indices) - 1, -1, -1)) & 1
                    )
                details = mcq_logit_values_from_precomputed_behaviors(
                    reconstructor, noun=str(context["noun"]),
                    candidate_prompt_texts=[f"reference-state-{state}" for state in states],
                    target_metric_id=str(context["metric_key"]),
                    target_description=str(context["target_description"]),
                    target_score_rows=rows, probe_texts=texts,
                    distractors=list(context["distractors"]), design_indices=indices,
                    codebook_frozen_before_prompt_search=True, n_examples=8,
                    n_reconstruction_draws=8, query_batch_size=query_batch_size,
                    fixed_teaching_panel=True, mcq_prompt_template=template,
                )
                search = set(map(int, trial["search_states"]))
                batch_rows = []
                for state, detail in zip(states, details):
                    identification = detail["identification"]
                    blind = float(identification["no_demonstration_score"])
                    lift = float(identification["annotation_lift_over_strongest_control"])
                    probabilities = np.mean(np.asarray(
                        identification["conditions"]["annotations"][
                            "canonical_choice_probabilities"
                        ], dtype=float,
                    ), axis=0)
                    predicted_position = int(np.argmax(probabilities))
                    predicted_metric = str(
                        detail["option_codebook"][predicted_position]["metric_id"]
                    )
                    batch_rows.append({
                        "template_sha256": template_sha,
                        "metric_key": str(context["metric_key"]),
                        "decoder_family": str(decoder_family),
                        "trial": int(trial["trial"]), "state": int(state),
                        "reference_split": "search" if state in search else "heldout_prompt",
                        "raw_lift": lift, "blind_probability": blind,
                        "normalized_fitness": float(lift / max(1e-12, 1.0 - blind)),
                        "target_metric_id": str(context["metric_key"]),
                        "predicted_metric_id": predicted_metric,
                        "identification_ceiling_bits": float(
                            trial["identification_ceiling_bits"]
                        ),
                        "is_canonical_state": int(state) == int(trial["canonical_state"]),
                    })
                if store is not None:
                    store.put(cache_key, "v14_mcq_reference_batch", {"rows": batch_rows})
                output.extend(batch_rows)
    return output


def aggregate_template_fitness(rows: Sequence[Mapping[str, object]]) -> dict[str, dict]:
    by_template: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_template.setdefault(str(row["template_sha256"]), []).append(row)
    reports = {}
    for template_sha, group in by_template.items():
        search = [float(row["normalized_fitness"]) for row in group if row["reference_split"] == "search"]
        transfer = [
            float(row["normalized_fitness"]) for row in group
            if row["reference_split"] == "heldout_prompt"
        ]
        family = {}
        for decoder_family in sorted({str(row["decoder_family"]) for row in group}):
            family_rows = [
                float(row["normalized_fitness"]) for row in group
                if str(row["decoder_family"]) == decoder_family and row["reference_split"] == "search"
            ]
            family[decoder_family] = float(np.mean(family_rows)) if family_rows else None
        search_mean = float(np.mean(search)) if search else float("-inf")
        transfer_mean = float(np.mean(transfer)) if transfer else float("-inf")
        near = [
            float(row["near_lift_bits"]) for row in group
            if row.get("near_lift_bits") is not None and row["reference_split"] == "search"
        ]
        far = [
            float(row["far_lift_bits"]) for row in group
            if row.get("far_lift_bits") is not None and row["reference_split"] == "search"
        ]
        near_mean = float(np.mean(near)) if near else None
        far_mean = float(np.mean(far)) if far else None
        far_near_ok = (
            True if near_mean is None or near_mean <= 0.01
            else bool(far_mean is not None and far_mean >= 0.5 * near_mean)
        )
        identification_mi = identification_ceiling = residual = None
        identification_rows = [
            row for row in group
            if row.get("target_metric_id") is not None
            and row["reference_split"] == "search"
            and bool(row.get("is_canonical_state", False))
        ]
        if identification_rows:
            target_labels = [str(row["target_metric_id"]) for row in identification_rows]
            predicted_labels = [str(row["predicted_metric_id"]) for row in identification_rows]
            identification_mi = _categorical_mutual_information(target_labels, predicted_labels)
            identification_ceiling = float(np.mean([
                float(row["identification_ceiling_bits"]) for row in identification_rows
            ]))
            residual = max(0.0, identification_ceiling - identification_mi)
        feedback = []
        search_rows = [row for row in group if row["reference_split"] == "search"]
        contrast_groups = {}
        for row in search_rows:
            contrast_groups.setdefault(
                (str(row["decoder_family"]), str(row["metric_key"]), int(row["trial"])), []
            ).append(row)
        for key in sorted(contrast_groups)[:12]:
            contrast = contrast_groups[key]
            best = max(contrast, key=lambda row: float(row["normalized_fitness"]))
            worst = min(contrast, key=lambda row: float(row["normalized_fitness"]))
            feedback.append({
                "decoder_family": key[0],
                "high": {
                    "state": int(best["state"]),
                    "fitness": float(best["normalized_fitness"]),
                    "induced_rule": best.get("induced_rule"),
                },
                "low": {
                    "state": int(worst["state"]),
                    "fitness": float(worst["normalized_fitness"]),
                    "induced_rule": worst.get("induced_rule"),
                },
            })
        reports[template_sha] = {
            "pooled_fitness": search_mean,
            "heldout_prompt_fitness": transfer_mean,
            "heldout_prompt_transfer_ok": bool(transfer and transfer_mean >= 0.0),
            "near_fitness_bits": near_mean, "far_fitness_bits": far_mean,
            "far_near_transfer_ok": far_near_ok,
            "per_family_fitness": family,
            "n_search_cells": len(search), "n_heldout_prompt_cells": len(transfer),
            "dev_identification_mi_bits": identification_mi,
            "dev_identification_ceiling_bits": identification_ceiling,
            "dev_identification_residual_bits": residual,
            "feedback": feedback,
        }
    return reports


def _categorical_mutual_information(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("categorical MI needs aligned nonempty labels")
    left_values = {value: index for index, value in enumerate(sorted(set(left)))}
    right_values = {value: index for index, value in enumerate(sorted(set(right)))}
    counts = np.zeros((len(left_values), len(right_values)), dtype=float)
    for lvalue, rvalue in zip(left, right):
        counts[left_values[lvalue], right_values[rvalue]] += 1.0
    joint = counts / counts.sum()
    product = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    keep = joint > 0.0
    return float(np.sum(joint[keep] * np.log2(joint[keep] / product[keep])))
