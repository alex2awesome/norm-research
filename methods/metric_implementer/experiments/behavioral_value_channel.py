"""Induce-and-execute behavioral value adapter for CR-3 v13.1.

For every finite six-bit teaching state, a deterministic constructor induces a rule.
Rules are content-addressed and executed once on the frozen held-out set H by the fixed
binary executor.  The state value is exact plug-in target MI minus the stronger blind or
shuffled-label control.  Execution degeneracy is diagnostic metadata, never a gate.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import numpy as np

from ..backends import parse_json_obj
from ..batch_scoring import _YESNO_TEMPLATE
from ..recon_channel import _feat_corr_table
from .cr3_reconstruction_values import _binary_state_rows, _bootstrap
from .cr3_sampled_value_certify import (
    BEHAVIORAL_PANEL_SIZE,
    VALUE_BOUND_STATE_SCHEMA,
    active_panel_design,
)
from .v13_value_cache import ValueCache, cache_key


BEHAVIORAL_ARMS = ("unconstrained", "no_verbatim_examples")
INDUCTION_TEMPLATE_ID = "cr3-v13.1-data-driven-six-example-rule-v1"
EXECUTION_TEMPLATE_ID = "rubric-first-constrained-binary-v3"
NO_VERBATIM_SHINGLE_WORDS = 12


def plugin_binary_mutual_information(target: Sequence[int], predicted: Sequence[int]) -> float:
    y = np.asarray(target, dtype=np.int64)
    z = np.asarray(predicted, dtype=np.int64)
    if y.shape != z.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("MI needs aligned nonempty binary vectors")
    if np.any((y != 0) & (y != 1)) or np.any((z != 0) & (z != 1)):
        raise ValueError("MI inputs must be binary")
    counts = np.zeros((2, 2), dtype=float)
    np.add.at(counts, (y, z), 1.0)
    joint = counts / len(y)
    product = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    keep = joint > 0.0
    return float(np.sum(joint[keep] * np.log2(joint[keep] / product[keep])))


def binary_entropy_bits(values: Sequence[int]) -> float:
    y = np.asarray(values, dtype=np.int64)
    if y.ndim != 1 or len(y) == 0 or np.any((y != 0) & (y != 1)):
        raise ValueError("entropy needs a nonempty binary vector")
    p = float(np.mean(y))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def balanced_agreement(target: Sequence[int], predicted: Sequence[int]) -> float:
    y = np.asarray(target, dtype=np.int64)
    z = np.asarray(predicted, dtype=np.int64)
    if y.shape != z.shape or y.ndim != 1:
        raise ValueError("balanced agreement needs aligned vectors")
    recalls = [float(np.mean(z[y == label] == label)) for label in (0, 1) if np.any(y == label)]
    return float(np.mean(recalls)) if recalls else 0.0


def _format_examples(texts: Sequence[str], labels: Sequence[int], max_chars: int) -> str:
    return "\n\n".join(
        f"[label={int(label)}]\n```\n{str(text)[:max_chars]}\n```"
        for text, label in zip(texts, labels)
    )


def _induction_prompt(
    *, noun: str, texts: Sequence[str], labels: Sequence[int], max_chars: int, arm: str,
) -> str:
    examples = _format_examples(texts, labels, max_chars)
    feature_table, _ = _feat_corr_table(list(texts), np.asarray(labels, dtype=float))
    no_verbatim = (
        "Do not quote, copy, paraphrase as a lookup table, or include any example text. "
        "State only a general criterion that can be applied to unseen texts.\n"
        if arm == "no_verbatim_examples" else ""
    )
    return (
        "You are reverse-engineering a hidden binary evaluation criterion applied to "
        f"{noun}. The only evidence is six labeled examples.\n\n"
        "Surface-feature correlations are supplied to make the induction data-driven; "
        "strong correlations should be named explicitly rather than replaced by a generic "
        "quality judgment.\n\n"
        f"FEATURE CORRELATIONS:\n{feature_table}\n\nEXAMPLES:\n{examples}\n\n"
        f"{no_verbatim}Now articulate the single rule that best explains the labels and lets a "
        "fresh evaluator classify new held-out texts. Reply with only the rule."
    )


def _blind_prompt(*, noun: str, arm: str) -> str:
    no_verbatim = (
        "Do not include examples or imagined example text. "
        if arm == "no_verbatim_examples" else ""
    )
    return (
        f"Infer a plausible binary evaluation criterion for {noun} without seeing any labeled "
        f"examples. {no_verbatim}Now articulate the single general rule only."
    )


def _normalized_rule(raw: str) -> str:
    text = str(raw or "").strip()
    obj = parse_json_obj(text)
    if obj:
        for field in ("rule", "rubric"):
            if str(obj.get(field) or "").strip():
                text = str(obj[field]).strip()
                break
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    if not text:
        raise RuntimeError("constructor returned an empty induced rule")
    return text


def contains_verbatim_example(
    rule: str, example_texts: Sequence[str], *, shingle_words: int = NO_VERBATIM_SHINGLE_WORDS,
) -> bool:
    def words(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", str(value).lower())

    rule_words = words(rule)
    if len(rule_words) < shingle_words:
        return False
    rule_shingles = {
        tuple(rule_words[start:start + shingle_words])
        for start in range(len(rule_words) - shingle_words + 1)
    }
    for text in example_texts:
        item_words = words(text)
        for start in range(len(item_words) - shingle_words + 1):
            if tuple(item_words[start:start + shingle_words]) in rule_shingles:
                return True
    return False


def _redact_verbatim_example_shingles(
    rule: str,
    example_texts: Sequence[str],
    *,
    shingle_words: int = NO_VERBATIM_SHINGLE_WORDS,
) -> str:
    """Remove copied spans after bounded model repair, preserving fail-closed disclosure."""
    example_shingles: set[tuple[str, ...]] = set()
    for text in example_texts:
        words = re.findall(r"[a-z0-9]+", str(text).lower())
        example_shingles.update(
            tuple(words[start:start + shingle_words])
            for start in range(len(words) - shingle_words + 1)
        )

    redacted = str(rule)
    while True:
        matches = list(re.finditer(r"[a-z0-9]+", redacted, flags=re.IGNORECASE))
        words = [match.group(0).lower() for match in matches]
        bad_ranges = [
            (matches[start].start(), matches[start + shingle_words - 1].end())
            for start in range(len(words) - shingle_words + 1)
            if tuple(words[start:start + shingle_words]) in example_shingles
        ]
        if not bad_ranges:
            cleaned = re.sub(r"\s+", " ", redacted).strip()
            return cleaned or (
                "Classify unseen texts using generalizable semantic and structural properties "
                "that distinguish the positive class from the negative class."
            )

        merged: list[list[int]] = []
        for start, end in bad_ranges:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in reversed(merged):
            redacted = redacted[:start] + " [example-specific phrase omitted] " + redacted[end:]


def _repair_prompt(rule: str, example_texts: Sequence[str]) -> str:
    hashes = [hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12] for text in example_texts]
    return (
        "Rewrite the candidate rule below as a general criterion. It failed the no-verbatim "
        "constraint by copying a labeled example. Remove every example-specific phrase, proper "
        "noun, quotation, and lookup behavior. Do not reproduce the examples; their content hashes "
        f"are {hashes}. Reply with only the repaired general rule.\n\nCANDIDATE RULE:\n{rule}"
    )


def _seed_from_key(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def _store_rule(cache: ValueCache, rule: str) -> str:
    rule_sha = hashlib.sha256(rule.encode("utf-8")).hexdigest()
    key = cache_key("behavioral_rule", {"rule_sha256": rule_sha})
    cache.put(key, "behavioral_rule", {"rule_sha256": rule_sha, "rule": rule})
    return rule_sha


def _load_rule(cache: ValueCache, rule_sha: str) -> str:
    key = cache_key("behavioral_rule", {"rule_sha256": str(rule_sha)})
    row = cache.get(key)
    if row is None or hashlib.sha256(str(row["rule"]).encode("utf-8")).hexdigest() != rule_sha:
        raise RuntimeError(f"missing or damaged content-addressed rule {rule_sha}")
    return str(row["rule"])


def _induce_missing(
    constructor, *, requests: Sequence[Mapping[str, object]], cache: ValueCache,
) -> dict[str, dict]:
    if not requests:
        return {}
    prompts = [str(request["prompt"]) for request in requests]
    seeds = [_seed_from_key(str(request["cache_key"])) for request in requests]
    raws = constructor.generate_batch(
        prompts, system=None, max_tokens=450, temperature=0.0, seed=seeds
    )
    if len(raws) != len(requests):
        raise RuntimeError("constructor returned an incomplete induction batch")
    rules = [_normalized_rule(raw) for raw in raws]
    fallback_redacted = [False] * len(rules)
    # Deterministic repair passes enforce the second declared prompt arm structurally.
    for _attempt in range(2):
        bad = [
            index for index, (request, rule) in enumerate(zip(requests, rules))
            if request["arm"] == "no_verbatim_examples"
            and contains_verbatim_example(rule, request["example_texts"])
        ]
        if not bad:
            break
        repair_prompts = [
            _repair_prompt(rules[index], requests[index]["example_texts"])
            for index in bad
        ]
        repaired = constructor.generate_batch(
            repair_prompts, system=None, max_tokens=450, temperature=0.0,
            seed=[seeds[index] + 1_000_003 * (_attempt + 1) for index in bad],
        )
        for index, raw in zip(bad, repaired):
            rules[index] = _normalized_rule(raw)
    for index, (request, rule) in enumerate(zip(requests, rules)):
        if (request["arm"] == "no_verbatim_examples"
                and contains_verbatim_example(rule, request["example_texts"])):
            rules[index] = _redact_verbatim_example_shingles(
                rule, request["example_texts"]
            )
            fallback_redacted[index] = True
    output = {}
    for index, (request, rule) in enumerate(zip(requests, rules)):
        if (request["arm"] == "no_verbatim_examples"
                and contains_verbatim_example(rule, request["example_texts"])):
            raise RuntimeError("no-verbatim induction still contains a demo-text shingle")
        rule_sha = _store_rule(cache, rule)
        payload = cache.put(str(request["cache_key"]), "behavioral_induction", {
            "arm": str(request["arm"]),
            "panel_sha256": str(request["panel_sha256"]),
            "state": int(request["state"]),
            "prompt_sha256": hashlib.sha256(
                str(request["prompt"]).encode("utf-8")
            ).hexdigest(),
            "rule_sha256": rule_sha,
            "no_verbatim_enforced": str(request["arm"]) == "no_verbatim_examples",
            "no_verbatim_fallback_redacted": fallback_redacted[index],
        })
        output[str(request["cache_key"])] = payload
    return output


def _execute_rules(
    executor, *, rule_hashes: Sequence[str], cache: ValueCache,
    heldout_texts: Sequence[str], heldout_sha256: str, executor_revision: str,
    executor_readout_id: str, max_chars: int, query_batch_size: int,
) -> dict[str, dict]:
    unique = list(dict.fromkeys(map(str, rule_hashes)))
    results = {}
    missing = []
    keys = {}
    for rule_sha in unique:
        key = cache_key("behavioral_execution", {
            "executor_revision": str(executor_revision),
            "executor_readout_id": str(executor_readout_id),
            "rule_sha256": rule_sha,
            "heldout_sha256": str(heldout_sha256),
            "execution_template_id": EXECUTION_TEMPLATE_ID,
        })
        keys[rule_sha] = key
        row = cache.get(key)
        if row is None:
            missing.append(rule_sha)
        else:
            results[rule_sha] = row
    heldout_n = len(heldout_texts)
    rules_per_batch = max(1, int(query_batch_size) // max(1, heldout_n))
    for start in range(0, len(missing), rules_per_batch):
        batch_hashes = missing[start:start + rules_per_batch]
        prompts = []
        for rule_sha in batch_hashes:
            rule = _load_rule(cache, rule_sha)
            prompts.extend([
                _YESNO_TEMPLATE.format(rubric=rule, text=str(text)[:max_chars])
                for text in heldout_texts
            ])
        scores = np.asarray(executor.score_binary_constrained(
            prompts, system=None, pos="YES", neg="NO", seed=0
        ), dtype=float)
        if scores.shape != (len(batch_hashes) * heldout_n,) or np.any(~np.isfinite(scores)):
            raise RuntimeError("executor returned incomplete/non-finite constrained scores")
        matrix = scores.reshape(len(batch_hashes), heldout_n)
        for position, rule_sha in enumerate(batch_hashes):
            row = cache.put(keys[rule_sha], "behavioral_execution", {
                "rule_sha256": rule_sha,
                "heldout_sha256": str(heldout_sha256),
                "p_yes": matrix[position].astype(float).tolist(),
                "hard_predictions": (matrix[position] > 0.5).astype(int).tolist(),
            })
            results[rule_sha] = row
    return results


def _shuffled_state(state: int, panel_size: int, panel_sha256: str) -> int:
    bits = _binary_state_rows(panel_size)[int(state)].astype(np.int64)
    if np.unique(bits).size < 2:
        return int(state)
    shift = 1 + int(panel_sha256[:8], 16) % (panel_size - 1)
    shuffled = np.roll(bits, shift)
    if np.array_equal(shuffled, bits):
        for offset in range(1, panel_size):
            candidate = np.roll(bits, offset)
            if not np.array_equal(candidate, bits):
                shuffled = candidate
                break
    return int("".join(map(str, shuffled.astype(int).tolist())), 2)


def evaluate_behavioral_state_tables(
    constructor, executor=None, *, codebook_manifest: Mapping[str, object],
    design_manifest: Mapping[str, object], target_metric_key: str, tier: str,
    constructor_revision: str, executor_revision: str, executor_readout_id: str,
    cache: ValueCache, query_batch_size: int = 2048, induction_only: bool = False,
) -> dict:
    """Fill both declared behavioral arms and return exact finite state tables."""
    active = active_panel_design(design_manifest, channel="behavioral", tier=tier)
    target = _bootstrap(codebook_manifest["metrics"][str(target_metric_key)]["bootstrap_path"])
    probe_texts = [str(text) for text in target["probe_texts"]]
    heldout = dict(design_manifest["heldout"])
    heldout_indices = np.asarray(heldout["indices"], dtype=int)
    pool_indices = {
        int(index) for pool in active["pools"] for index in pool["indices"]
    }
    if pool_indices.intersection(map(int, heldout_indices)):
        raise RuntimeError("behavioral teaching pools overlap the frozen held-out set H")
    heldout_texts = [probe_texts[index] for index in heldout_indices]
    heldout_target = np.asarray(heldout["target_scores"], dtype=np.int64)
    target_entropy = binary_entropy_bits(heldout_target)
    heldout_sha = hashlib.sha256(json.dumps({
        "probe_sha256": design_manifest["probe_sha256"],
        "indices": heldout_indices.astype(int).tolist(),
        "text_sha256": heldout["probe_text_sha256"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    induction_rows: dict[tuple[str, int, int], dict] = {}
    requests = []
    for arm in BEHAVIORAL_ARMS:
        for panel_position, panel in enumerate(active["panels"]):
            panel_indices = [int(index) for index in panel["fixed_teaching_indices"]]
            panel_texts = [probe_texts[index] for index in panel_indices]
            for state, labels in enumerate(_binary_state_rows(BEHAVIORAL_PANEL_SIZE)):
                prompt = _induction_prompt(
                    noun=str(codebook_manifest["reconstruction_noun"]),
                    texts=panel_texts,
                    labels=labels.astype(int).tolist(),
                    max_chars=int(codebook_manifest["reconstruction_max_chars"]),
                    arm=arm,
                )
                prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                key = cache_key("behavioral_induction", {
                    "constructor_revision": str(constructor_revision),
                    "panel_sha256": str(panel["panel_sha256"]),
                    "state": int(state),
                    "arm": arm,
                    "induction_template_id": INDUCTION_TEMPLATE_ID,
                    "prompt_sha256": prompt_sha,
                })
                payload = cache.get(key)
                if payload is None:
                    requests.append({
                        "cache_key": key,
                        "prompt": prompt,
                        "arm": arm,
                        "panel_sha256": panel["panel_sha256"],
                        "state": state,
                        "example_texts": panel_texts,
                    })
                else:
                    induction_rows[(arm, panel_position, state)] = payload
    generated = _induce_missing(constructor, requests=requests, cache=cache)
    for arm in BEHAVIORAL_ARMS:
        for panel_position, panel in enumerate(active["panels"]):
            panel_indices = [int(index) for index in panel["fixed_teaching_indices"]]
            panel_texts = [probe_texts[index] for index in panel_indices]
            for state, labels in enumerate(_binary_state_rows(BEHAVIORAL_PANEL_SIZE)):
                prompt = _induction_prompt(
                    noun=str(codebook_manifest["reconstruction_noun"]), texts=panel_texts,
                    labels=labels.astype(int).tolist(),
                    max_chars=int(codebook_manifest["reconstruction_max_chars"]), arm=arm,
                )
                key = cache_key("behavioral_induction", {
                    "constructor_revision": str(constructor_revision),
                    "panel_sha256": str(panel["panel_sha256"]), "state": int(state),
                    "arm": arm, "induction_template_id": INDUCTION_TEMPLATE_ID,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                })
                payload = cache.get(key) or generated.get(key)
                if payload is None:
                    raise RuntimeError("behavioral induction cache is incomplete")
                induction_rows[(arm, panel_position, state)] = payload

    blind_rows = {}
    blind_requests = []
    for arm in BEHAVIORAL_ARMS:
        prompt = _blind_prompt(noun=str(codebook_manifest["reconstruction_noun"]), arm=arm)
        key = cache_key("behavioral_induction", {
            "constructor_revision": str(constructor_revision),
            "panel_sha256": "blind-no-panel", "state": -1, "arm": arm,
            "induction_template_id": INDUCTION_TEMPLATE_ID,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })
        payload = cache.get(key)
        if payload is None:
            blind_requests.append({
                "cache_key": key, "prompt": prompt, "arm": arm,
                "panel_sha256": "blind-no-panel", "state": -1,
                "example_texts": [],
            })
        else:
            blind_rows[arm] = payload
    generated_blind = _induce_missing(constructor, requests=blind_requests, cache=cache)
    for request in blind_requests:
        blind_rows[str(request["arm"])] = generated_blind[str(request["cache_key"])]

    rule_hashes = [row["rule_sha256"] for row in induction_rows.values()]
    rule_hashes.extend(row["rule_sha256"] for row in blind_rows.values())
    if induction_only:
        return {
            "schema": VALUE_BOUND_STATE_SCHEMA,
            "channel": "behavioral",
            "stage": "induction_complete",
            "tier": str(tier).upper(),
            "n_induction_cells": int(len(induction_rows) + len(blind_rows)),
            "n_distinct_rules": int(len(set(rule_hashes))),
            "heldout_sha256": heldout_sha,
        }
    if executor is None:
        raise ValueError("executor is required after the induction-only stage")
    executions = _execute_rules(
        executor, rule_hashes=rule_hashes, cache=cache, heldout_texts=heldout_texts,
        heldout_sha256=heldout_sha, executor_revision=str(executor_revision),
        executor_readout_id=str(executor_readout_id),
        max_chars=int(codebook_manifest["reconstruction_max_chars"]),
        query_batch_size=int(query_batch_size),
    )

    arm_results = {}
    for arm in BEHAVIORAL_ARMS:
        state_values = np.empty((len(active["panels"]), 64), dtype=float)
        raw_mi = np.empty_like(state_values)
        shuffled_mi = np.empty_like(state_values)
        balanced = np.empty_like(state_values)
        blind_rule_sha = str(blind_rows[arm]["rule_sha256"])
        blind_predictions = executions[blind_rule_sha]["hard_predictions"]
        blind_mi = plugin_binary_mutual_information(heldout_target, blind_predictions)
        blind_balanced = balanced_agreement(heldout_target, blind_predictions)
        rule_sha_table = np.empty((len(active["panels"]), 64), dtype=object)
        shuffled_rule_sha_table = np.empty_like(rule_sha_table)
        for panel_position, panel in enumerate(active["panels"]):
            for state in range(64):
                row = induction_rows[(arm, panel_position, state)]
                rule_sha = str(row["rule_sha256"])
                shuffled_state = _shuffled_state(
                    state, BEHAVIORAL_PANEL_SIZE, str(panel["panel_sha256"])
                )
                shuffled_rule_sha = str(
                    induction_rows[(arm, panel_position, shuffled_state)]["rule_sha256"]
                )
                predictions = executions[rule_sha]["hard_predictions"]
                shuffled_predictions = executions[shuffled_rule_sha]["hard_predictions"]
                mi = plugin_binary_mutual_information(heldout_target, predictions)
                mi_shuffled = plugin_binary_mutual_information(
                    heldout_target, shuffled_predictions
                )
                state_values[panel_position, state] = float(np.clip(
                    mi - max(blind_mi, mi_shuffled), 0.0, target_entropy
                ))
                raw_mi[panel_position, state] = mi
                shuffled_mi[panel_position, state] = mi_shuffled
                balanced[panel_position, state] = balanced_agreement(
                    heldout_target, predictions
                )
                rule_sha_table[panel_position, state] = rule_sha
                shuffled_rule_sha_table[panel_position, state] = shuffled_rule_sha
        if np.any(~np.isfinite(state_values)) or np.any(state_values > target_entropy + 1e-12):
            raise RuntimeError("behavioral state table violates its entropy ceiling")
        hard_matrix = np.asarray([
            executions[rule_sha]["hard_predictions"] for rule_sha in dict.fromkeys(
                map(str, rule_sha_table.ravel())
            )
        ], dtype=float)
        arm_results[arm] = {
            "state_values": state_values,
            "raw_mutual_information_bits": raw_mi,
            "shuffled_mutual_information_bits": shuffled_mi,
            "balanced_agreement": balanced,
            "rule_sha256": rule_sha_table,
            "shuffled_rule_sha256": shuffled_rule_sha_table,
            "blind_rule_sha256": blind_rule_sha,
            "blind_mutual_information_bits": float(blind_mi),
            "blind_balanced_agreement": float(blind_balanced),
            "n_distinct_rules": int(len(set(map(str, rule_sha_table.ravel())))),
            "execution_degeneracy": {
                "n_distinct_heldout_verdict_vectors": int(len({
                    row.astype(np.uint8).tobytes() for row in hard_matrix
                })),
                "share_constant_heldout_verdict_vectors": float(np.mean([
                    np.unique(row).size == 1 for row in hard_matrix
                ])),
                "reported_not_gated": True,
            },
        }
    return {
        "schema": VALUE_BOUND_STATE_SCHEMA,
        "channel": "behavioral",
        "tier": str(tier).upper(),
        "active_design": active,
        "target_entropy_bits": float(target_entropy),
        "heldout_sha256": heldout_sha,
        "heldout_size": int(len(heldout_indices)),
        "arms": arm_results,
        "cache": {
            "n_induction_cells": int(len(induction_rows) + len(blind_rows)),
            "n_distinct_rules_executed": int(len(set(rule_hashes))),
        },
        "non_disclosure": {
            "candidate_prompt_text_passed_to_inducer": False,
            "candidate_prompt_text_passed_to_executor": False,
            "only_panel_texts_and_labels_enter_induction": True,
            "execution_uses_frozen_heldout_H_only": True,
            "no_verbatim_shingle_words": NO_VERBATIM_SHINGLE_WORDS,
        },
    }
