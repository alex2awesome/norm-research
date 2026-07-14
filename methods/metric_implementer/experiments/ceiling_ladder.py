"""Independent-reference C0/C1/C2 ceiling ladder and v13.1 robustness audit.

The module is deliberately phased. ``freeze`` and ``reference`` are CPU/API-only,
``constructor`` keeps Llama-3.3-70B resident for closed-menu identification and
eight-demo induction, ``executor`` keeps the frozen Llama-3.1-8B executor resident,
and ``aggregate`` is CPU-only.  No phase mines or mutates the v14 instrument.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import string
import subprocess
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from ..backends import parse_json_obj
from ..batch_scoring import _YESNO_TEMPLATE
from ..config import ImplementerConfig
from ..recon_channel import _feat_corr_table
from ..vllm_backend import (
    CR3_BINARY_READOUT_ID,
    _single_token_label_id,
    make_judge_backend,
    model_revision_id,
    release_resident_engines,
)
from .cr3_reconstruction_values import _binary_state_rows, _bootstrap
from .cr3_sampled_value_certify import (
    _payload_sha256,
    active_panel_design,
    build_value_bound_design_manifest,
    enumerate_exact_pool_values,
)
from .run_v13_value_campaign import (
    FIXED_EXECUTOR,
    _load_codebook_for_entry,
    _prepare_contexts,
    load_metrics_manifest,
    select_metric_entries,
)
from .behavioral_value_channel import (
    INDUCTION_TEMPLATE_ID as V13_INDUCTION_TEMPLATE_ID,
    _blind_prompt as v13_blind_prompt,
    _shuffled_state as v13_shuffled_state,
)
from .v13_value_cache import cache_key as v13_cache_key
from .v14_value_bound import (
    balanced_agreement, plugin_binary_mutual_information, signatures_to_states,
)
from .v14_probe_extension import load_extension


SCHEMA = "cr3-independent-ceiling-ladder-v1"
REFERENCE_SCHEMA = "cr3-independent-reference-v1"
CONSTRUCTOR_SCHEMA = "cr3-ceiling-ladder-constructor-v1"
EXECUTOR_SCHEMA = "cr3-ceiling-ladder-executor-v1"
DEFAULT_CONSTRUCTOR = "meta-llama/Llama-3.3-70B-Instruct"
N_MENU_PERMUTATIONS = 8
N_REFERENCE_PASSES = 3
N_PERMUTATIONS = 10_000
REFERENCE_RELIABILITY_BOOTSTRAPS = 10_000
DEFAULT_TEMPLATE = """You are reverse-engineering a hidden binary evaluation criterion applied to {noun}.
Infer one general rule from the labeled examples. Use the feature correlations as fallible clues,
compare hypotheses against every label, and state the most specific rule that generalizes.

FEATURE CORRELATIONS:
{feature_table}

LABELED EXAMPLES:
{examples}

Reply with only the criterion."""


def normalized_rule(raw: str) -> str:
    text_value = str(raw or "").strip()
    parsed = parse_json_obj(text_value)
    if parsed:
        for field in ("rule", "rubric", "criterion"):
            if str(parsed.get(field) or "").strip():
                text_value = str(parsed[field]).strip()
                break
    text_value = re.sub(
        r"^```(?:text)?\s*|\s*```$", "", text_value, flags=re.IGNORECASE
    ).strip()
    if not text_value:
        raise RuntimeError("decoder returned an empty induced rule")
    return re.sub(r"\s+", " ", text_value)


def induction_prompt(
    *, template: str, noun: str, texts: Sequence[str], labels: Sequence[int],
    max_chars: int, arm: str,
) -> str:
    if arm != "unconstrained":
        raise ValueError("the ceiling ladder declares only unconstrained C2 induction")
    feature_table, _ = _feat_corr_table(list(texts), np.asarray(labels, dtype=float))
    examples = "\n\n".join(
        f"[label={int(label)}]\n```\n{str(item)[:int(max_chars)]}\n```"
        for item, label in zip(texts, labels)
    )
    return str(template).format(
        noun=str(noun), feature_table=feature_table, examples=examples,
    )


def blind_prompt(*, template: str, noun: str, arm: str) -> str:
    if arm != "unconstrained":
        raise ValueError("the ceiling ladder declares only unconstrained C2 induction")
    return str(template).format(
        noun=str(noun),
        feature_table="(No labeled examples or feature correlations are provided.)",
        examples="(No labeled examples are provided.)",
    )


def quantized_induction_prompt(
    *, noun: str, texts: Sequence[str], levels: Sequence[int], max_chars: int,
) -> str:
    feature_table, _ = _feat_corr_table(list(texts), np.asarray(levels, dtype=float))
    examples = "\n\n".join(
        f"[P(YES) level={int(level)} of 3]\n```\n{str(item)[:int(max_chars)]}\n```"
        for item, level in zip(texts, levels)
    )
    return (
        "You are reverse-engineering a hidden binary evaluation criterion. Each example carries "
        "a four-level quantization of the source evaluator's P(YES): 0 means [0,.25), 1 means "
        "[.25,.50), 2 means [.50,.75), and 3 means [.75,1]. Use confidence as evidence, infer "
        f"one general rule for unseen {noun}, and reply with only the criterion.\n\n"
        f"FEATURE CORRELATIONS:\n{feature_table}\n\nLABELED EXAMPLES:\n{examples}"
    )


def shuffled_levels(levels: Sequence[int], panel_sha256: str) -> list[int]:
    values = np.asarray(levels, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 3):
        raise ValueError("C3 levels must lie in 0..3")
    if np.unique(values).size < 2:
        return values.astype(int).tolist()
    shift = 1 + int(str(panel_sha256)[:8], 16) % (len(values) - 1)
    result = np.roll(values, shift)
    if np.array_equal(values, result):
        for offset in range(1, len(values)):
            result = np.roll(values, offset)
            if not np.array_equal(values, result):
                break
    return result.astype(int).tolist()


def shuffled_state(state: int, panel_size: int, panel_sha256: str) -> int:
    bits = _binary_state_rows(int(panel_size))[int(state)].astype(np.int64)
    if np.unique(bits).size < 2:
        return int(state)
    shift = 1 + int(str(panel_sha256)[:8], 16) % (int(panel_size) - 1)
    shuffled = np.roll(bits, shift)
    if np.array_equal(shuffled, bits):
        for offset in range(1, int(panel_size)):
            candidate = np.roll(bits, offset)
            if not np.array_equal(candidate, bits):
                shuffled = candidate
                break
    return int("".join(map(str, shuffled.astype(int).tolist())), 2)


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _atomic_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(value)).strip("_")


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def miller_madow_mutual_information(
    target: Sequence[int], predicted: Sequence[int], *, clip: bool = False,
) -> float:
    """Observed-support Miller--Madow correction for a binary contingency table."""
    left = np.asarray(target, dtype=np.uint8)
    right = np.asarray(predicted, dtype=np.uint8)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("Miller-Madow MI needs aligned nonempty vectors")
    if np.any(left > 1) or np.any(right > 1):
        raise ValueError("Miller-Madow MI inputs must be binary")
    counts = np.zeros((2, 2), dtype=int)
    np.add.at(counts, (left, right), 1)
    k_left = int(np.sum(counts.sum(axis=1) > 0))
    k_right = int(np.sum(counts.sum(axis=0) > 0))
    k_joint = int(np.sum(counts > 0))
    correction = (k_left + k_right - k_joint - 1) / (2.0 * len(left) * math.log(2.0))
    result = float(plugin_binary_mutual_information(left, right) + correction)
    return max(0.0, result) if clip else result


def ordinary_accuracy(target: Sequence[int], predicted: Sequence[int]) -> float:
    left = np.asarray(target, dtype=np.uint8)
    right = np.asarray(predicted, dtype=np.uint8)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("accuracy needs aligned one-dimensional vectors")
    return float(np.mean(left == right))


def binary_auc(target: Sequence[int], scores: Sequence[float]) -> float | None:
    labels = np.asarray(target, dtype=np.uint8)
    values = np.asarray(scores, dtype=float)
    if labels.shape != values.shape or labels.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("AUC needs aligned finite scores")
    positive = values[labels == 1]
    negative = values[labels == 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))


def value_against_controls(
    target: Sequence[int], prediction: Sequence[int], blind: Sequence[int],
    shuffled: Sequence[int], *, corrected: bool,
) -> dict[str, float]:
    estimator = miller_madow_mutual_information if corrected else plugin_binary_mutual_information
    main = float(estimator(target, prediction))
    blind_value = float(estimator(target, blind))
    shuffled_value = float(estimator(target, shuffled))
    raw = main - max(blind_value, shuffled_value)
    return {
        "mi": main, "blind_mi": blind_value, "shuffled_mi": shuffled_value,
        "raw_lift": float(raw), "value": float(max(0.0, raw)),
        "accuracy": ordinary_accuracy(target, prediction),
        "balanced_accuracy": balanced_agreement(target, prediction),
    }


def permutation_pvalue(observed: float, null_values: Sequence[float]) -> dict[str, float]:
    null = np.asarray(null_values, dtype=float)
    if null.ndim != 1 or not len(null) or np.any(~np.isfinite(null)):
        raise ValueError("permutation null must be a finite nonempty vector")
    null_mean = float(np.mean(null))
    null_sd = float(np.std(null, ddof=0))
    return {
        "observed": float(observed),
        "null_median": float(np.median(null)),
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z_score": None if null_sd <= 0.0 else float((float(observed) - null_mean) / null_sd),
        "z_score_unavailable_reason": "zero_null_variance" if null_sd <= 0.0 else None,
        "percentile": float(np.mean(null <= float(observed))),
        "p_greater_equal": float((1 + np.sum(null >= float(observed))) / (len(null) + 1)),
    }


def fleiss_kappa(labels: np.ndarray) -> float:
    """Fleiss' kappa for an item-by-rater binary label matrix."""
    matrix = np.asarray(labels, dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("Fleiss kappa needs at least two raters and one item")
    if np.any(matrix > 1):
        raise ValueError("Fleiss kappa labels must be binary")
    n_raters = matrix.shape[1]
    positive = matrix.sum(axis=1).astype(float)
    negative = n_raters - positive
    observed = np.mean(
        (positive * (positive - 1.0) + negative * (negative - 1.0))
        / (n_raters * (n_raters - 1.0))
    )
    p_positive = float(np.mean(matrix))
    expected = p_positive ** 2 + (1.0 - p_positive) ** 2
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else 1.0


def reference_reliability_gate(
    pass_by_item: np.ndarray, *, seed: int,
    n_bootstraps: int = REFERENCE_RELIABILITY_BOOTSTRAPS,
) -> dict:
    """Require three-pass agreement to clear pooled-marginal chance."""
    matrix = np.asarray(pass_by_item, dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[0] != N_REFERENCE_PASSES or matrix.shape[1] < 2:
        raise ValueError("reference gate requires three complete pass vectors")
    if np.any((matrix != 0) & (matrix != 1)):
        raise ValueError("reference passes must be binary")
    pairs = [(left, right) for left in range(matrix.shape[0]) for right in range(left)]

    def statistics(sample: np.ndarray) -> tuple[float, float, float]:
        observed, chance = [], []
        for left, right in pairs:
            lhs, rhs = matrix[left, sample], matrix[right, sample]
            observed.append(float(np.mean(lhs == rhs)))
            p_left, p_right = float(np.mean(lhs)), float(np.mean(rhs))
            chance.append(p_left * p_right + (1.0 - p_left) * (1.0 - p_right))
        agreement, expected = float(np.mean(observed)), float(np.mean(chance))
        return agreement, expected, agreement - expected

    observed, chance, gap = statistics(np.arange(matrix.shape[1], dtype=int))
    rng = np.random.default_rng(int(seed))
    bootstrap_gaps = np.empty(int(n_bootstraps), dtype=float)
    for index in range(int(n_bootstraps)):
        sample = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        bootstrap_gaps[index] = statistics(sample)[2]
    lower = float(np.quantile(bootstrap_gaps, 0.05))
    return {
        "schema": "cr3-reference-reliability-gate-v1",
        "n_items": int(matrix.shape[1]), "n_passes": int(matrix.shape[0]),
        "pass_positive_rates": matrix.mean(axis=1).astype(float).tolist(),
        "mean_pairwise_agreement": observed,
        "pooled_marginal_chance_agreement": chance,
        "agreement_above_chance": gap,
        "bootstrap_replicates": int(n_bootstraps),
        "one_sided_95_lower_bound_above_chance": lower,
        "passed": bool(lower > 0.0),
    }


def dawid_skene_binary(
    labels: np.ndarray, *, max_iterations: int = 500, tolerance: float = 1e-10,
) -> dict[str, object]:
    """Small Dawid--Skene EM sensitivity model for the three blind judge passes.

    The three passes use the same model and therefore are not independent raters.  The
    result is reported only as an attenuation sensitivity analysis, never as ground
    truth or as a replacement for majority vote.
    """
    matrix = np.asarray(labels, dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("Dawid-Skene needs an item-by-rater label matrix")
    if np.any(matrix > 1):
        raise ValueError("Dawid-Skene labels must be binary")
    posterior = (matrix.mean(axis=1) * 0.8 + 0.1).clip(1e-6, 1.0 - 1e-6)
    sensitivity = np.full(matrix.shape[1], 0.9, dtype=float)
    specificity = np.full(matrix.shape[1], 0.9, dtype=float)
    prevalence = float(np.mean(posterior))
    for iteration in range(int(max_iterations)):
        previous = posterior.copy()
        total_positive = max(float(np.sum(posterior)), 1e-12)
        total_negative = max(float(np.sum(1.0 - posterior)), 1e-12)
        # Beta(2,2) smoothing keeps small/degenerate metric cells finite.
        sensitivity = (1.0 + np.sum(posterior[:, None] * matrix, axis=0)) / (
            2.0 + total_positive
        )
        specificity = (1.0 + np.sum((1.0 - posterior)[:, None] * (1 - matrix), axis=0)) / (
            2.0 + total_negative
        )
        prevalence = float((1.0 + np.sum(posterior)) / (2.0 + len(posterior)))
        log_one = np.full(len(matrix), math.log(prevalence), dtype=float)
        log_zero = np.full(len(matrix), math.log(1.0 - prevalence), dtype=float)
        for rater in range(matrix.shape[1]):
            observed = matrix[:, rater]
            log_one += np.where(
                observed == 1, np.log(sensitivity[rater]), np.log1p(-sensitivity[rater])
            )
            log_zero += np.where(
                observed == 0, np.log(specificity[rater]), np.log1p(-specificity[rater])
            )
        maximum = np.maximum(log_one, log_zero)
        probability_one = np.exp(log_one - maximum)
        probability_zero = np.exp(log_zero - maximum)
        posterior = probability_one / (probability_one + probability_zero)
        if float(np.max(np.abs(posterior - previous))) <= float(tolerance):
            break
    return {
        "posterior_probability": posterior,
        "prevalence": prevalence,
        "sensitivity": sensitivity.tolist(),
        "specificity": specificity.tolist(),
        "iterations": int(iteration + 1),
        "same_model_passes_not_independent": True,
    }


def soft_binary_mutual_information(
    target_probability: Sequence[float], predicted: Sequence[int],
) -> float:
    probability = np.asarray(target_probability, dtype=float)
    right = np.asarray(predicted, dtype=np.uint8)
    if (probability.ndim != 1 or probability.shape != right.shape or not len(right)
            or np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1))
            or np.any(right > 1)):
        raise ValueError("soft MI needs aligned probabilities and binary predictions")
    counts = np.zeros((2, 2), dtype=float)
    for predicted_label in (0, 1):
        selected = right == predicted_label
        counts[1, predicted_label] = float(np.sum(probability[selected]))
        counts[0, predicted_label] = float(np.sum(1.0 - probability[selected]))
    joint = counts / float(len(right))
    product = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    keep = joint > 0.0
    return float(np.sum(joint[keep] * np.log2(joint[keep] / product[keep])))


def _permuted_ladder_values(
    target: Sequence[int], predictions: Sequence[Sequence[int]],
    blind_predictions: Sequence[Sequence[int]] | None = None,
    shuffled_predictions: Sequence[Sequence[int]] | None = None, *,
    n_permutations: int = N_PERMUTATIONS, seed: int = 0,
) -> np.ndarray:
    """Permutation null preserving the complete panel/control selection rule."""
    target_array = np.asarray(target, dtype=np.uint8)
    predicted = np.asarray(predictions, dtype=np.uint8)
    if predicted.ndim == 1:
        predicted = predicted[None, :]
    if predicted.ndim != 2 or predicted.shape[1] != len(target_array):
        raise ValueError("prediction rows do not align with the target")
    if (blind_predictions is None) != (shuffled_predictions is None):
        raise ValueError("blind and shuffled controls must be supplied together")
    blind = shuffled = None
    if blind_predictions is not None:
        blind = np.asarray(blind_predictions, dtype=np.uint8)
        shuffled = np.asarray(shuffled_predictions, dtype=np.uint8)
        if blind.ndim == 1:
            blind = np.repeat(blind[None, :], len(predicted), axis=0)
        if shuffled.ndim == 1:
            shuffled = shuffled[None, :]
        if blind.shape != predicted.shape or shuffled.shape != predicted.shape:
            raise ValueError("control predictions do not align with panel predictions")
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(n_permutations), dtype=float)
    for index in range(int(n_permutations)):
        permuted = rng.permutation(target_array)
        values = []
        for panel in range(len(predicted)):
            main = plugin_binary_mutual_information(permuted, predicted[panel])
            if blind is None:
                values.append(main)
            else:
                baseline = max(
                    plugin_binary_mutual_information(permuted, blind[panel]),
                    plugin_binary_mutual_information(permuted, shuffled[panel]),
                )
                values.append(max(0.0, main - baseline))
        output[index] = float(np.mean(values))
    return output


PLANTED_CRITERIA = (
    {
        "id": "ascii_question_mark",
        "description": "Return YES if and only if the text contains the ASCII character ?.\n",
        "truth": lambda text: "?" in text,
    },
    {
        "id": "arabic_digit",
        "description": "Return YES if and only if the text contains an Arabic digit 0 through 9.\n",
        "truth": lambda text: bool(re.search(r"[0-9]", text)),
    },
    {
        "id": "terminal_exclamation",
        "description": "Return YES if and only if the stripped text ends with the ASCII character !.\n",
        "truth": lambda text: text.strip().endswith("!"),
    },
    {
        "id": "double_quoted_span",
        "description": "Return YES if and only if the text contains two ASCII double-quote characters.\n",
        "truth": lambda text: text.count('"') >= 2,
    },
    {
        "id": "first_person_token",
        "description": "Return YES if and only if the text contains the case-insensitive whole-word token I or me.\n",
        "truth": lambda text: bool(re.search(r"(?i)\b(?:i|me)\b", text)),
    },
    {
        "id": "at_least_twenty_words",
        "description": "Return YES if and only if the text has at least 20 whitespace-separated words.\n",
        "truth": lambda text: len(text.split()) >= 20,
    },
)


def planted_manifest(texts: Sequence[str]) -> dict:
    rows = []
    for criterion in PLANTED_CRITERIA:
        labels = [int(criterion["truth"](str(text))) for text in texts]
        rows.append({
            "id": criterion["id"], "description": criterion["description"],
            "labels": labels, "positive_rate": float(np.mean(labels)),
        })
    payload = {"schema": "cr3-mechanical-reference-v1", "rows": rows}
    return {**payload, "sha256": _sha(payload)}


def _planted_text(serial: int) -> str:
    """Synthetic text whose six mechanical labels are the low six bits of serial."""
    bits = [(int(serial) >> shift) & 1 for shift in range(5, -1, -1)]
    words = ["sample", f"token{chr(97 + serial % 26)}", "plain", "content"]
    if bits[3]:
        words.append('"quoted"')
    if bits[4]:
        words.append("I")
    if bits[1]:
        words.append("7")
    if bits[0]:
        words.append("?")
    if bits[5]:
        words.extend([f"neutral{chr(97 + index)}" for index in range(20 - len(words))])
    return " ".join(words) + ("!" if bits[2] else ".")


def freeze_planted_design(root: str | Path) -> dict:
    """Freeze a non-circular, mechanically labeled calibration through all rungs."""
    base = Path(root).resolve()
    texts = [_planted_text(index) for index in range(96)]
    criteria = []
    for criterion in PLANTED_CRITERIA:
        labels = np.asarray([int(criterion["truth"](text)) for text in texts], dtype=np.uint8)
        candidates = list(range(36))
        positives = [index for index in candidates if labels[index] == 1]
        negatives = [index for index in candidates if labels[index] == 0]
        if len(positives) < 4 or len(negatives) < 4:
            raise RuntimeError(f"planted teaching split is imbalanced for {criterion['id']}")
        selected = sorted(positives, key=lambda x: _sha((criterion["id"], x)))[:4]
        selected += sorted(negatives, key=lambda x: _sha((criterion["id"], x)))[:4]
        selected.sort(key=lambda x: _sha((criterion["id"], "order", x)))
        panel_core = {
            "indices": selected, "texts": [texts[index] for index in selected],
            "target_labels": labels[selected].astype(int).tolist(),
        }
        panel = {**panel_core, "panel_sha256": _sha(panel_core)}
        criteria.append({
            "id": criterion["id"], "description": criterion["description"],
            "panel": panel, "heldout_labels": labels[36:96].astype(int).tolist(),
        })
    payload = {
        "schema": "cr3-planted-ladder-v1", "noun": "text", "max_chars": 4096,
        "heldout_texts": texts[36:96], "criteria": criteria,
        "split_disjoint": True, "labels_mechanically_computed": True,
    }
    payload["sha256"] = _sha(payload)
    _atomic_json(base / "planted" / "planted_design.json", payload)
    return payload


def _bootstrap_forms(path: str | Path, fallback: str) -> list[str]:
    with np.load(path, allow_pickle=True) as artifact:
        if "target_form_texts" in artifact.files:
            forms = [str(value).strip() for value in artifact["target_form_texts"]]
        elif "metric_description" in artifact.files:
            forms = [str(artifact["metric_description"]).strip()]
        else:
            forms = [str(fallback).strip()]
    forms = list(dict.fromkeys(value for value in forms if value))
    if not forms:
        raise ValueError(f"bootstrap {path} has no source description payload")
    return forms


def _size11_keys(keys: Sequence[str], target: str, metric_key: str) -> list[str]:
    if target not in keys:
        raise ValueError("target is absent from its task codebook")
    others = sorted(
        (str(key) for key in keys if str(key) != str(target)),
        key=lambda key: (_sha({"metric": metric_key, "candidate": key, "arm": "size11"}), key),
    )
    return [str(target), *others[:10]]


def freeze_design(
    metrics_manifest: str | Path, out_root: str | Path, *,
    probe_extension_root: str | Path | None = None,
) -> dict:
    """Freeze append-only H=240, native eight-demo panels, and task-local menus."""
    root = Path(out_root).resolve()
    manifest, base = load_metrics_manifest(metrics_manifest)
    entries = select_metric_entries(manifest, base)
    if len(entries) != 35:
        raise ValueError(f"ceiling ladder requires exactly 35 Tier-B metrics, got {len(entries)}")
    index_rows = []
    for entry in entries:
        metric_key = str(entry["metric_key"])
        codebook = _load_codebook_for_entry(entry, base)
        design = build_value_bound_design_manifest(
            codebook, target_metric_key=metric_key, heldout_size=60,
        )
        active = active_panel_design(design, channel="mcq", tier="B")
        target = _bootstrap(codebook["metrics"][metric_key]["bootstrap_path"])
        probes = list(map(str, target["probe_texts"]))
        operational_scores = np.asarray(target["target"], dtype=float)
        if operational_scores.shape != (len(probes),) or np.any(~np.isfinite(operational_scores)):
            raise RuntimeError(f"{metric_key} has invalid frozen operational scores")
        if probe_extension_root is None:
            raise ValueError("v14.1 ladder freeze requires --probe-extension-root")
        extension_path = Path(probe_extension_root) / f"{entry['task']}.npz"
        extension = load_extension(extension_path)
        extension_keys = list(map(str, extension["metric_keys"]))
        if metric_key not in extension_keys:
            raise ValueError(f"probe extension lacks {metric_key}")
        probes = [*probes, *map(str, extension["texts"])]
        operational_scores = np.concatenate([
            operational_scores,
            np.asarray(extension["scores"], dtype=float)[extension_keys.index(metric_key)],
        ])
        quantized_scores = np.digitize(
            operational_scores, np.asarray([0.25, 0.50, 0.75]), right=False
        ).astype(np.uint8)
        design_set = set(map(int, codebook["design_indices"]))
        base_candidates = [index for index in range(300) if index not in design_set]
        heldout = sorted(
            base_candidates,
            key=lambda index: _sha((metric_key, "v14.1-heldout", index)),
        )[:150]
        heldout = sorted([*heldout, *range(300, 390)])
        menu_keys = sorted(map(str, codebook["metrics"]))
        forms_by_key = {
            key: _bootstrap_forms(
                codebook["metrics"][key]["bootstrap_path"],
                str(codebook["metrics"][key]["description"]),
            ) for key in menu_keys
        }
        # C1 must expose a literal member of the exact form orbit that generated each target.
        menu_descriptions = {key: forms_by_key[key][0] for key in menu_keys}
        if metric_key not in menu_keys:
            raise RuntimeError(f"{metric_key} absent from its own task-local menu")
        panels = []
        for panel in active["panels"]:
            indices = list(map(int, panel["fixed_teaching_indices"]))
            labels = [int(value) for value in panel["fixed_teaching_target_scores"]]
            if (len(indices) != 8 or len(set(indices)) != 8
                    or any(label not in (0, 1) for label in labels)):
                raise RuntimeError(f"{metric_key} has an invalid eight-demo panel")
            panels.append({
                "panel_sha256": str(panel["panel_sha256"]),
                "indices": indices, "texts": [probes[index] for index in indices],
                "target_labels": labels,
                "positive_count": int(sum(labels)),
                "target_quantized_labels": quantized_scores[indices].astype(int).tolist(),
            })
        payload = {
            "schema": SCHEMA, "metric_key": metric_key,
            "task": str(entry["task"]), "level": str(entry["level"]),
            "metric": str(entry["metric"]), "noun": str(codebook["reconstruction_noun"]),
            "max_chars": int(codebook["reconstruction_max_chars"]),
            "probe_sha256": str(design["probe_sha256"]),
            "heldout_indices": heldout,
            "heldout_texts": [probes[index] for index in heldout],
            "operational_target": (operational_scores[heldout] > 0.5).astype(int).tolist(),
            "reference_probe_texts": probes,
            "operational_reference_scores": operational_scores.astype(float).tolist(),
            "operational_reference_target": (operational_scores > 0.5).astype(int).tolist(),
            "panels": panels,
            "menu_keys": menu_keys,
            "size11_keys": _size11_keys(menu_keys, metric_key, metric_key),
            "menu_descriptions": menu_descriptions,
            "forms_by_key": forms_by_key,
            "target_description_payload_sha256": _sha(forms_by_key[metric_key]),
            "source_bootstrap_sha256": str(codebook["metrics"][metric_key]["bootstrap_sha256"]),
            "probe_extension_path": str(extension_path.resolve()),
            "probe_extension_sha256": extension["sha256"],
            "design_manifest_sha256": str(design["design_manifest_sha256"]),
            "design_scientific_sha256": _sha({
                key: value for key, value in design.items()
                if key not in {"design_manifest_sha256", "codebook_manifest_sha256"}
            }),
        }
        payload["freeze_sha256"] = _sha(payload)
        path = root / "designs" / _safe(metric_key) / "ladder_design.json"
        _atomic_json(path, payload)
        index_rows.append({
            "metric_key": metric_key, "task": entry["task"],
            "path": str(path.relative_to(root)),
            "freeze_sha256": payload["freeze_sha256"], "menu_size": len(menu_keys),
        })
    frame = pd.DataFrame(index_rows).sort_values(["task", "metric_key"])
    _atomic_parquet(root / "design_index.parquet", frame)
    campaign = {
        "schema": SCHEMA, "phase": "freeze", "n_metrics": len(frame),
        "metrics_manifest": str(Path(metrics_manifest).resolve()),
        "metrics_manifest_sha256": _file_sha(metrics_manifest),
        "design_index": str(root / "design_index.parquet"),
        "design_index_sha256": _file_sha(root / "design_index.parquet"),
        "heldout_n": 240, "reference_n": 390, "panel_size": 8,
        "no_cross_task_distractors": True,
        "constructor_scaling_claim": {
            "status": "WITHDRAWN_UNMATCHED_COMPARISON",
            "reported_8b_row_candidate_provenance": (
                "sk2:/lfs/skampere2/0/alexspan/cr3-v13.1/outputs/tier_b/lanes/"
                "llama31_8b/results.parquet"
            ),
            "reason": (
                "the previously quoted 8B mean used 35 metrics while the quoted 70B mean used "
                "29; no 13.7x scaling claim is permitted until identical metric rows are joined "
                "on design-scientific identity"
            ),
        },
    }
    campaign["freeze_sha256"] = _sha(campaign)
    _atomic_json(root / "freeze_manifest.json", campaign)
    freeze_planted_design(root)
    return campaign


def _load_designs(root: str | Path, metric_keys: Sequence[str] | None = None) -> list[dict]:
    base = Path(root).resolve()
    index = pd.read_parquet(base / "design_index.parquet")
    requested = None if not metric_keys else set(map(str, metric_keys))
    rows = []
    for path in index["path"]:
        candidate = Path(str(path))
        if not candidate.is_absolute():
            candidate = base / candidate
        elif not candidate.is_file():
            # Backward-compatible relocation of an already frozen development artifact.
            candidate = base / "designs" / candidate.parent.name / candidate.name
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        observed = str(payload.pop("freeze_sha256"))
        if _sha(payload) != observed:
            raise RuntimeError(f"damaged ladder design {candidate}")
        payload["freeze_sha256"] = observed
        if requested is None or str(payload["metric_key"]) in requested:
            rows.append(payload)
    if requested is not None and {row["metric_key"] for row in rows} != requested:
        raise ValueError("requested ladder metric is absent from frozen design")
    return rows


def _reference_schema(n_real: int, n_anchors: int) -> str:
    return json.dumps({
        "type": "object", "additionalProperties": False,
        "properties": {
            "labels": {"type": "array", "minItems": n_real + n_anchors,
                       "maxItems": n_real + n_anchors,
                       "items": {"type": "integer", "enum": [0, 1]}},
        }, "required": ["labels"],
    }, separators=(",", ":"))


def _extract_structured_claude(stdout: str) -> dict:
    outer = json.loads(stdout)
    if isinstance(outer, dict) and isinstance(outer.get("structured_output"), dict):
        return dict(outer["structured_output"])
    if isinstance(outer, dict) and isinstance(outer.get("result"), str):
        return json.loads(outer["result"])
    if isinstance(outer, dict) and "labels" in outer:
        return outer
    raise RuntimeError("Claude output lacked the declared structured payload")


def _sonnet_call(prompt: str, *, n_real: int, n_anchors: int) -> dict:
    command = [
        "claude", "-p", "--model", "sonnet", "--no-session-persistence",
        "--output-format", "json", "--json-schema", _reference_schema(n_real, n_anchors),
        "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch",
    ]
    failures = []
    for attempt in range(2):
        completed = subprocess.run(
            command, input=prompt, text=True, capture_output=True, check=False, timeout=600,
        )
        if not completed.returncode:
            return _extract_structured_claude(completed.stdout)
        failures.append({
            "returncode": int(completed.returncode),
            "stderr": completed.stderr[-1000:], "stdout": completed.stdout[-1000:],
        })
        if attempt == 0:
            time.sleep(5)
    raise RuntimeError(f"Claude reference transport failed twice: {failures}")


def _anchor_rows(pass_index: int, metric_key: str, batch_index: int = 0) -> list[dict]:
    base = [
        ("Return YES if and only if the entire text is exactly BLUE.", "BLUE", 1),
        ("Return YES if and only if the entire text is exactly BLUE.", "RED", 0),
        ("Return YES if and only if the entire text is exactly cat.", "cat", 1),
        ("Return YES if and only if the entire text is exactly cat.", "dog", 0),
        ("Return YES if and only if the entire text is exactly 7.", "7", 1),
        ("Return YES if and only if the entire text is exactly 7.", "8", 0),
    ]
    return [
        {"id": _sha((metric_key, pass_index, batch_index, index))[:12], "criterion": criterion,
         "text": text, "label": label}
        for index, (criterion, text, label) in enumerate(base)
    ]


def _reference_batches(
    order: Sequence[int], texts: Sequence[str], *, max_chars: int,
    max_items: int = 50, character_budget: int = 100_000,
) -> list[list[int]]:
    batches = []
    current = []
    used = 0
    for index in order:
        cost = len(str(texts[int(index)])[:int(max_chars)]) + 32
        if current and (len(current) >= int(max_items) or used + cost > int(character_budget)):
            batches.append(current)
            current = []
            used = 0
        current.append(int(index))
        used += cost
    if current:
        batches.append(current)
    return batches


def _build_reference_strict(
    out_root: str | Path, *, metric_keys: Sequence[str] | None = None,
    n_passes: int = N_REFERENCE_PASSES, assemble: bool = True,
) -> dict:
    """Create three isolated blind Sonnet passes with fail-closed arithmetic anchors."""
    root = Path(out_root).resolve()
    designs = _load_designs(root, metric_keys)
    rows = []
    for design in designs:
        metric_key = str(design["metric_key"])
        metric_directory = root / "reference" / _safe(metric_key)
        metric_labels_path = metric_directory / "labels.parquet"
        metric_report_path = metric_directory / "report.json"
        if metric_labels_path.is_file() and metric_report_path.is_file():
            cached = pd.read_parquet(metric_labels_path)
            expected_items = len(design["reference_probe_texts"])
            if (len(cached) != expected_items or cached.metric_key.nunique() != 1
                    or str(cached.metric_key.iloc[0]) != metric_key):
                raise RuntimeError(f"invalid resumable reference artifact for {metric_key}")
            rows.extend(cached.to_dict(orient="records"))
            continue
        metric_row_start = len(rows)
        forms = list(map(str, design["forms_by_key"][metric_key]))
        # The exact frozen orbit is exposed; Sonnet applies all forms and returns one verdict.
        criterion = "\n\n".join(
            f"FORM {index + 1}: {form}" for index, form in enumerate(forms)
        )
        texts = list(map(str, design["reference_probe_texts"]))
        pass_vectors = []
        batches_per_pass = []
        for pass_index in range(int(n_passes)):
            order = sorted(
                range(len(texts)),
                key=lambda index: _sha((metric_key, pass_index, index, "sonnet-order")),
            )
            restored = np.empty(len(texts), dtype=np.uint8)
            batches = _reference_batches(
                order, texts, max_chars=int(design["max_chars"]),
            )
            batches_per_pass.append(len(batches))
            for batch_index, batch in enumerate(batches):
                anchors = _anchor_rows(pass_index, metric_key, batch_index)
                records = [
                    {"kind": "real", "index": int(index), "criterion": criterion,
                     "text": texts[index][:int(design["max_chars"])]}
                    for index in batch
                ] + [
                    {"kind": "anchor", "index": int(index),
                     "criterion": row["criterion"], "text": row["text"]}
                    for index, row in enumerate(anchors)
                ]
                records.sort(key=lambda row: _sha((
                    metric_key, pass_index, batch_index, row["kind"], row["index"], "mix"
                )))
                cell_path = (
                    metric_directory / "cells" /
                    f"pass_{pass_index:02d}__batch_{batch_index:03d}.json"
                )
                if cell_path.is_file():
                    cell = json.loads(cell_path.read_text(encoding="utf-8"))
                    observed_cell_sha = str(cell.pop("sha256", ""))
                    if _sha(cell) != observed_cell_sha:
                        raise RuntimeError(f"reference cell checksum mismatch in {cell_path}")
                    expected_identity = {
                        "metric_key": metric_key, "pass_index": pass_index,
                        "batch_index": batch_index,
                        "real_indices": list(map(int, batch)),
                        "description_payload_sha256": design[
                            "target_description_payload_sha256"
                        ],
                    }
                    if any(cell.get(key) != value for key, value in expected_identity.items()):
                        raise RuntimeError(f"reference cell identity mismatch in {cell_path}")
                    cached_labels = list(map(int, cell["real_labels"]))
                    if len(cached_labels) != len(batch):
                        raise RuntimeError(f"reference cell is incomplete in {cell_path}")
                    restored[np.asarray(batch, dtype=int)] = np.asarray(
                        cached_labels, dtype=np.uint8
                    )
                    continue
                prompt = (
                    "Apply each item's supplied criterion to its text independently. Return 1 only "
                    "when that criterion applies and 0 otherwise. Some criteria contain multiple "
                    "frozen forms; apply each form and use their majority verdict, with ties as 0. "
                    "Do not infer labels from position or from other items. Return one labels array "
                    "in displayed order, with no explanation.\n\nITEMS:\n" + "\n\n".join(
                        f"[{position}]\nCRITERION:\n{row['criterion']}\nTEXT:\n{row['text']}"
                        for position, row in enumerate(records)
                    )
                )
                output = _sonnet_call(prompt, n_real=len(batch), n_anchors=len(anchors))
                displayed = list(map(int, output["labels"]))
                observed_anchors = [None] * len(anchors)
                for position, row in enumerate(records):
                    if row["kind"] == "anchor":
                        observed_anchors[int(row["index"])] = displayed[position]
                expected_anchors = [row["label"] for row in anchors]
                if observed_anchors != expected_anchors:
                    output = _sonnet_call(
                        prompt + "\n\nRETRY: the prior hidden-anchor check failed; apply every rule literally.",
                        n_real=len(batch), n_anchors=len(anchors),
                    )
                    displayed = list(map(int, output["labels"]))
                    observed_anchors = [None] * len(anchors)
                    for position, row in enumerate(records):
                        if row["kind"] == "anchor":
                            observed_anchors[int(row["index"])] = displayed[position]
                if observed_anchors != expected_anchors:
                    raise RuntimeError(
                        f"Sonnet anchors failed twice for {metric_key} pass {pass_index} "
                        f"batch {batch_index}: observed={observed_anchors}, "
                        f"expected={expected_anchors}"
                    )
                for position, row in enumerate(records):
                    if row["kind"] == "real":
                        restored[int(row["index"])] = int(displayed[position])
                cell_payload = {
                    "schema": "cr3-independent-reference-cell-v1",
                    "metric_key": metric_key, "pass_index": pass_index,
                    "batch_index": batch_index, "real_indices": list(map(int, batch)),
                    "real_labels": restored[np.asarray(batch, dtype=int)].astype(int).tolist(),
                    "description_payload_sha256": design[
                        "target_description_payload_sha256"
                    ],
                    "n_blinded_anchors": len(anchors), "anchor_check_passed": True,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
                cell_payload["sha256"] = _sha(cell_payload)
                _atomic_json(cell_path, cell_payload)
            pass_vectors.append(restored)
        matrix = np.vstack(pass_vectors)
        reliability = reference_reliability_gate(
            matrix, seed=_stable_seed(metric_key, "reference-reliability"),
        )
        _atomic_json(metric_directory / "reliability.json", reliability)
        if not reliability["passed"]:
            raise RuntimeError(
                f"reference reliability failed for {metric_key}: "
                f"lower_gap={reliability['one_sided_95_lower_bound_above_chance']:.6f}"
            )
        majority = (np.sum(matrix, axis=0) >= (len(matrix) // 2 + 1)).astype(np.uint8)
        latent = dawid_skene_binary(matrix.T)
        pairwise = [
            float(np.mean(matrix[left] == matrix[right]))
            for left in range(len(matrix)) for right in range(left)
        ]
        for index in range(len(texts)):
            rows.append({
                "metric_key": metric_key, "task": design["task"],
                "probe_index": index,
                **{f"pass_{p + 1}": int(matrix[p, index]) for p in range(len(matrix))},
                "majority_label": int(majority[index]),
                "latent_probability": float(latent["posterior_probability"][index]),
                "unanimous": bool(np.unique(matrix[:, index]).size == 1),
            })
        metric_report = {
            "schema": REFERENCE_SCHEMA, "metric_key": metric_key,
            "n_passes": len(matrix), "n_items": len(texts),
            "n_batches_per_pass": batches_per_pass,
            "mean_pairwise_agreement": float(np.mean(pairwise)),
            "fleiss_kappa": fleiss_kappa(matrix.T),
            "unanimity": float(np.mean(np.all(matrix == matrix[0], axis=0))),
            "majority_positive_rate": float(np.mean(majority)),
            "reference_reliability_gate": reliability,
            "description_payload_sha256": design["target_description_payload_sha256"],
            "attenuation_sensitivity": {
                key: value for key, value in latent.items()
                if key != "posterior_probability"
            },
        }
        metric_report["sha256"] = _sha(metric_report)
        _atomic_json(metric_report_path, metric_report)
        _atomic_parquet(metric_labels_path, pd.DataFrame(rows[metric_row_start:]))
        print(
            f"[reference] completed {metric_key}: kappa={metric_report['fleiss_kappa']:.4f}",
            flush=True,
        )
    if not assemble:
        return {
            "schema": REFERENCE_SCHEMA,
            "completed_metric_keys": [str(design["metric_key"]) for design in designs],
        }
    frame = pd.DataFrame(rows).sort_values(["task", "metric_key", "probe_index"])
    _atomic_parquet(root / "reference" / "sonnet_labels.parquet", frame)
    manifest = {
        "schema": REFERENCE_SCHEMA, "n_metrics": int(frame.metric_key.nunique()),
        "n_passes": int(n_passes), "n_items": len(frame),
        "labels_path": str(root / "reference" / "sonnet_labels.parquet"),
        "labels_sha256": _file_sha(root / "reference" / "sonnet_labels.parquet"),
        "reference_is_independent_of_executor_outputs": True,
        "anchor_policy": "six arithmetic anchors per batch; one retry then fail closed",
    }
    manifest["sha256"] = _sha(manifest)
    _atomic_json(root / "reference" / "reference_manifest.json", manifest)
    # A deterministic mechanical suite over the union of the first task's held-out texts.
    _atomic_json(root / "reference" / "planted_reference.json", planted_manifest(
        designs[0]["heldout_texts"]
    ))
    return manifest


def build_reference(
    out_root: str | Path, *, metric_keys: Sequence[str] | None = None,
    n_passes: int = N_REFERENCE_PASSES, void_on_failure: bool = False,
) -> dict:
    """Build references, optionally recording a bounded failure per metric.

    ``void_on_failure`` is intended for one-metric worker invocations.  It keeps a
    transport or anchor failure from aborting the remaining campaign, while making
    the failed metric structurally ineligible for the independent-reference result.
    """
    if not void_on_failure:
        return _build_reference_strict(
            out_root, metric_keys=metric_keys, n_passes=n_passes, assemble=True,
        )
    root = Path(out_root).resolve()
    designs = _load_designs(root, metric_keys)
    decisions = []
    for design in designs:
        metric_key = str(design["metric_key"])
        metric_directory = root / "reference" / _safe(metric_key)
        labels_path = metric_directory / "labels.parquet"
        report_path = metric_directory / "report.json"
        void_path = metric_directory / "void.json"
        if labels_path.is_file() and report_path.is_file():
            decisions.append({"metric_key": metric_key, "status": "valid_cached"})
            continue
        if void_path.is_file():
            decisions.append({"metric_key": metric_key, "status": "void_cached"})
            continue
        try:
            _build_reference_strict(
                root, metric_keys=[metric_key], n_passes=n_passes, assemble=False,
            )
            if void_path.exists():
                void_path.unlink()
            decisions.append({"metric_key": metric_key, "status": "valid"})
        except Exception as exc:  # fail closed and preserve accepted batch cells
            cells = sorted((metric_directory / "cells").glob("*.json"))
            payload = {
                "schema": "cr3-independent-reference-void-v1",
                "metric_key": metric_key,
                "task": str(design["task"]),
                "status": "void",
                "eligible_for_independent_reference": False,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "accepted_reference_cells": len(cells),
                "description_payload_sha256": design[
                    "target_description_payload_sha256"
                ],
                "anchor_policy": "six literal anchors per batch; one retry then void",
            }
            payload["sha256"] = _sha(payload)
            _atomic_json(void_path, payload)
            decisions.append({"metric_key": metric_key, "status": "void"})
            print(f"[reference] VOID {metric_key}: {exc}", flush=True)
    return {"schema": REFERENCE_SCHEMA, "decisions": decisions}


def assemble_reference(out_root: str | Path, *, require_decided: bool = True) -> dict:
    """Assemble valid per-metric labels and an explicit valid/void campaign manifest."""
    root = Path(out_root).resolve()
    designs = _load_designs(root)
    valid, void, undecided, frames = [], [], [], []
    for design in designs:
        metric_key = str(design["metric_key"])
        directory = root / "reference" / _safe(metric_key)
        labels_path = directory / "labels.parquet"
        report_path = directory / "report.json"
        void_path = directory / "void.json"
        has_valid = labels_path.is_file() and report_path.is_file()
        if has_valid and void_path.is_file():
            raise RuntimeError(f"reference is both valid and void for {metric_key}")
        if has_valid:
            frame = pd.read_parquet(labels_path)
            expected_items = len(design["reference_probe_texts"])
            if (len(frame) != expected_items or frame.metric_key.nunique() != 1
                    or str(frame.metric_key.iloc[0]) != metric_key
                    or sorted(frame.probe_index.astype(int).tolist()) != list(range(expected_items))):
                raise RuntimeError(f"invalid reference artifact for {metric_key}")
            frames.append(frame)
            valid.append(metric_key)
        elif void_path.is_file():
            payload = json.loads(void_path.read_text(encoding="utf-8"))
            observed_sha = str(payload.pop("sha256", ""))
            if _sha(payload) != observed_sha or payload.get("metric_key") != metric_key:
                raise RuntimeError(f"invalid void artifact for {metric_key}")
            void.append(metric_key)
        else:
            undecided.append(metric_key)
    if require_decided and undecided:
        raise RuntimeError(f"reference decisions incomplete: {undecided}")
    if not frames:
        raise RuntimeError("no independently referenced metrics survived calibration")
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["task", "metric_key", "probe_index"]
    )
    labels_path = root / "reference" / "sonnet_labels.parquet"
    _atomic_parquet(labels_path, combined)
    manifest = {
        "schema": REFERENCE_SCHEMA,
        "n_metrics": len(valid),
        "n_valid_metrics": len(valid),
        "n_void_metrics": len(void),
        "n_undecided_metrics": len(undecided),
        "n_decided_metrics": len(valid) + len(void),
        "n_passes": N_REFERENCE_PASSES,
        "n_items": len(combined),
        "valid_metric_keys": valid,
        "void_metric_keys": void,
        "undecided_metric_keys": undecided,
        "labels_path": str(labels_path),
        "labels_sha256": _file_sha(labels_path),
        "reference_is_independent_of_executor_outputs": True,
        "anchor_policy": "six literal anchors per batch; one retry then void",
        "voids_are_excluded_from_independent_reference_results": True,
    }
    manifest["sha256"] = _sha(manifest)
    _atomic_json(root / "reference" / "reference_manifest.json", manifest)
    _atomic_json(root / "reference" / "planted_reference.json", planted_manifest(
        designs[0]["heldout_texts"]
    ))
    return manifest


def _backend(model: str, *, fake: bool):
    cfg = ImplementerConfig()
    cfg.vllm_fake = bool(fake)
    cfg.vllm_gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90"))
    cfg.vllm_tp_size = 1
    cfg.vllm_max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
    if os.environ.get("METRIC_IMPLEMENTER_LFS_HOME"):
        cfg.vllm_lfs_home = os.environ["METRIC_IMPLEMENTER_LFS_HOME"]
    overrides = json.loads(os.environ.get("CEILING_MODEL_PATH_OVERRIDES_JSON", "{}"))
    runtime = str(overrides.get(str(model), str(model)))
    backend = make_judge_backend(runtime, cfg, 0.0)
    revision = str(model) if fake else model_revision_id(runtime)
    return backend, revision


def _choice_labels(backend, needed: int) -> list[str]:
    if hasattr(backend, "_engine"):
        tokenizer = backend._engine(backend.model, backend.cfg).get_tokenizer()
        candidates = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
        candidates += [value for value in string.punctuation if value not in "`\\\"'"]
        labels = []
        token_ids = set()
        for candidate in candidates:
            try:
                token_id = _single_token_label_id(tokenizer, candidate)
            except ValueError:
                continue
            if token_id not in token_ids:
                labels.append(candidate)
                token_ids.add(token_id)
            if len(labels) >= int(needed):
                break
    else:
        labels = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
        labels += list("!#$%&()*+,-./:;<=>?@[]^_{|}~")
    if len(labels) < int(needed):
        raise RuntimeError(
            f"decoder tokenizer exposes only {len(labels)} readable one-token labels; need {needed}"
        )
    return labels[:int(needed)]


def _menu_prompt(
    design: Mapping[str, object], keys_in_order: Sequence[str], labels: Sequence[str],
    *, panel: Mapping[str, object] | None, state_labels: Sequence[int] | None,
) -> str:
    if panel is None:
        examples = "(No labeled demonstrations are provided.)"
    else:
        examples = "\n\n".join(
            f"[label={int(label)}]\n```\n{str(text)[:int(design['max_chars'])]}\n```"
            for text, label in zip(panel["texts"], state_labels)
        )
    menu = "\n".join(
        f"{label}: {design['menu_descriptions'][key]}"
        for label, key in zip(labels, keys_in_order)
    )
    return (
        f"A hidden criterion was applied to {design['noun']}. Identify which exact criterion "
        "best explains the labels. Choose only one option label.\n\n"
        f"DEMONSTRATIONS:\n{examples}\n\nCLOSED TASK-LOCAL MENU:\n{menu}\n\nOPTION:"
    )


def _score_menu_condition(
    decoder, design: Mapping[str, object], menu_keys: Sequence[str], *,
    panel: Mapping[str, object] | None, labels_for_state: Sequence[int] | None,
    condition: str,
) -> dict:
    option_keys = list(map(str, menu_keys))
    labels = _choice_labels(decoder, len(option_keys))
    canonical = np.zeros(len(option_keys), dtype=float)
    for permutation in range(N_MENU_PERMUTATIONS):
        order = sorted(
            option_keys,
            key=lambda key: (_sha((design["metric_key"], condition, permutation, key)), key),
        )
        prompt = _menu_prompt(
            design, order, labels, panel=panel, state_labels=labels_for_state,
        )
        probabilities = np.asarray(decoder.score_choices(
            [prompt], labels, seed=_stable_seed(design["metric_key"], condition, permutation),
        )[0], dtype=float)
        if probabilities.shape != (len(order),) or not np.isclose(probabilities.sum(), 1.0):
            raise RuntimeError("closed-menu constrained posterior is invalid")
        by_key = dict(zip(order, probabilities))
        canonical += np.asarray([by_key[key] for key in option_keys])
    canonical /= N_MENU_PERMUTATIONS
    maximum = float(np.max(canonical))
    tied = [key for key, value in zip(option_keys, canonical) if np.isclose(value, maximum)]
    return {
        "menu_keys": option_keys, "posterior": canonical.tolist(),
        "picked_metric_key": sorted(tied)[0], "tie_size": len(tied),
        "n_permutations": N_MENU_PERMUTATIONS,
    }


def _score_menu_conditions_batch(
    decoder, design: Mapping[str, object], menu_keys: Sequence[str],
    conditions: Sequence[tuple[str, Mapping[str, object] | None, Sequence[int] | None]],
) -> dict[str, dict]:
    """Score every permutation for many panel/control conditions in one vLLM flush."""
    option_keys = list(map(str, menu_keys))
    labels = _choice_labels(decoder, len(option_keys))
    prompts = []
    seeds = []
    metadata = []
    for condition, panel, state_labels in conditions:
        for permutation in range(N_MENU_PERMUTATIONS):
            order = sorted(
                option_keys,
                key=lambda key: (_sha((design["metric_key"], condition, permutation, key)), key),
            )
            prompts.append(_menu_prompt(
                design, order, labels, panel=panel, state_labels=state_labels,
            ))
            seeds.append(_stable_seed(design["metric_key"], condition, permutation))
            metadata.append((str(condition), order))
    probabilities = decoder.score_choices(prompts, labels, seed=seeds)
    totals = {str(condition): np.zeros(len(option_keys), dtype=float)
              for condition, _panel, _state in conditions}
    for (condition, order), row in zip(metadata, probabilities):
        vector = np.asarray(row, dtype=float)
        if vector.shape != (len(order),) or not np.isclose(vector.sum(), 1.0):
            raise RuntimeError("closed-menu constrained posterior is invalid")
        by_key = dict(zip(order, vector))
        totals[condition] += np.asarray([by_key[key] for key in option_keys])
    output = {}
    for condition, total in totals.items():
        posterior = total / N_MENU_PERMUTATIONS
        maximum = float(np.max(posterior))
        tied = [key for key, value in zip(option_keys, posterior)
                if np.isclose(value, maximum)]
        output[condition] = {
            "menu_keys": option_keys, "posterior": posterior.tolist(),
            "picked_metric_key": sorted(tied)[0], "tie_size": len(tied),
            "n_permutations": N_MENU_PERMUTATIONS,
        }
    return output


def _run_planted_constructor(root: Path, decoder, *, model: str, revision: str) -> None:
    source = json.loads((root / "planted" / "planted_design.json").read_text())
    menu_keys = [str(row["id"]) for row in source["criteria"]]
    descriptions = {str(row["id"]): str(row["description"]) for row in source["criteria"]}
    output = []
    for criterion in source["criteria"]:
        target_key = str(criterion["id"])
        panel = dict(criterion["panel"])
        design = {
            "metric_key": f"planted::{target_key}", "noun": source["noun"],
            "max_chars": source["max_chars"], "menu_descriptions": descriptions,
        }
        canonical = list(map(int, panel["target_labels"]))
        state = int("".join(map(str, canonical)), 2)
        shuffled = _binary_state_rows(8)[shuffled_state(
            state, 8, str(panel["panel_sha256"])
        )].astype(int).tolist()
        batched = _score_menu_conditions_batch(decoder, design, menu_keys, [
            ("blind", None, None), ("canonical", panel, canonical),
            ("shuffled", panel, shuffled),
        ])
        c1 = {condition: batched[condition]
              for condition in ("blind", "canonical", "shuffled")}
        requests = [
            induction_prompt(
                template=DEFAULT_TEMPLATE, noun=str(source["noun"]), texts=panel["texts"],
                labels=labels, max_chars=int(source["max_chars"]), arm="unconstrained",
            ) for labels in (canonical, shuffled)
        ]
        requests.append(blind_prompt(
            template=DEFAULT_TEMPLATE, noun=str(source["noun"]), arm="unconstrained",
        ))
        raw_rules = decoder.generate_batch(
            requests, system=None, max_tokens=128, temperature=0.0,
            seed=[_stable_seed(target_key, condition, "planted-c2")
                  for condition in ("canonical", "shuffled", "blind")],
        )
        rules = {
            condition: normalized_rule(raw)
            for condition, raw in zip(("canonical", "shuffled", "blind"), raw_rules)
        }
        c0_scores = decoder.score_binary_constrained([
            _YESNO_TEMPLATE.format(
                rubric=criterion["description"], text=str(text)[:int(source["max_chars"])]
            ) for text in source["heldout_texts"]
        ], pos="YES", neg="NO", seed=0)
        output.append({
            "target_id": target_key, "c1": c1, "c2_rules": rules,
            "c0_70b_scores": list(map(float, c0_scores)),
        })
    payload = {
        "schema": "cr3-planted-constructor-v1", "model": model,
        "revision": revision, "rows": output, "design_sha256": source["sha256"],
    }
    payload["sha256"] = _sha(payload)
    _atomic_json(root / "planted" / "constructor.json", payload)


def run_constructor(
    out_root: str | Path, *, model: str = DEFAULT_CONSTRUCTOR,
    metric_keys: Sequence[str] | None = None, fake: bool = False,
) -> dict:
    root = Path(out_root).resolve()
    designs = _load_designs(root, metric_keys)
    decoder, revision = _backend(model, fake=fake)
    completed = []
    try:
        for design in designs:
            metric_key = str(design["metric_key"])
            target = root / "constructor" / _safe(metric_key) / "constructor.json"
            if target.is_file():
                completed.append(metric_key)
                continue
            c1 = {}
            for arm, menu_keys in (
                ("full_task_bank", design["menu_keys"]),
                ("size11", design["size11_keys"]),
            ):
                conditions = [(f"{arm}:blind", None, None)]
                panel_states = []
                for position, panel in enumerate(design["panels"]):
                    canonical = list(map(int, panel["target_labels"]))
                    state = int("".join(map(str, canonical)), 2)
                    shuffled = _binary_state_rows(8)[shuffled_state(
                        state, 8, str(panel["panel_sha256"])
                    )].astype(int).tolist()
                    canonical_name = f"{arm}:{position}:canonical"
                    shuffled_name = f"{arm}:{position}:shuffled"
                    conditions.extend([
                        (canonical_name, panel, canonical),
                        (shuffled_name, panel, shuffled),
                    ])
                    panel_states.append((panel, canonical_name, shuffled_name))
                batched = _score_menu_conditions_batch(
                    decoder, design, menu_keys, conditions,
                )
                arm_rows = {"blind": batched[f"{arm}:blind"], "panels": []}
                for panel, canonical_name, shuffled_name in panel_states:
                    arm_rows["panels"].append({
                        "panel_sha256": panel["panel_sha256"],
                        "canonical": batched[canonical_name],
                        "shuffled": batched[shuffled_name],
                    })
                c1[arm] = arm_rows
            prompts = []
            request_meta = []
            for position, panel in enumerate(design["panels"]):
                canonical = list(map(int, panel["target_labels"]))
                state = int("".join(map(str, canonical)), 2)
                shuffled = _binary_state_rows(8)[shuffled_state(
                    state, 8, str(panel["panel_sha256"])
                )].astype(int).tolist()
                for condition, state_labels in (("canonical", canonical), ("shuffled", shuffled)):
                    prompts.append(induction_prompt(
                        template=DEFAULT_TEMPLATE, noun=str(design["noun"]),
                        texts=panel["texts"], labels=state_labels,
                        max_chars=int(design["max_chars"]), arm="unconstrained",
                    ))
                    request_meta.append({"panel": position, "condition": condition})
            prompts.append(blind_prompt(
                template=DEFAULT_TEMPLATE, noun=str(design["noun"]), arm="unconstrained",
            ))
            request_meta.append({"panel": -1, "condition": "blind"})
            raws = decoder.generate_batch(
                prompts, system=None, max_tokens=128, temperature=0.0,
                seed=[_stable_seed(metric_key, row["panel"], row["condition"], "c2")
                      for row in request_meta],
            )
            rules = []
            for meta, raw in zip(request_meta, raws):
                rule = normalized_rule(raw)
                rules.append({**meta, "rule": rule, "rule_sha256": hashlib.sha256(
                    rule.encode("utf-8")
                ).hexdigest()})
            c3_prompts = []
            c3_meta = []
            for position, panel in enumerate(design["panels"]):
                canonical_levels = list(map(int, panel["target_quantized_labels"]))
                shuffled = shuffled_levels(canonical_levels, str(panel["panel_sha256"]))
                for condition, levels in (("canonical", canonical_levels), ("shuffled", shuffled)):
                    c3_prompts.append(quantized_induction_prompt(
                        noun=str(design["noun"]), texts=panel["texts"], levels=levels,
                        max_chars=int(design["max_chars"]),
                    ))
                    c3_meta.append({"panel": position, "condition": condition})
            c3_raw = decoder.generate_batch(
                c3_prompts, system=None, max_tokens=128, temperature=0.0,
                seed=[_stable_seed(metric_key, row["panel"], row["condition"], "c3")
                      for row in c3_meta],
            )
            c3_rules = []
            for meta, raw in zip(c3_meta, c3_raw):
                rule = normalized_rule(raw)
                c3_rules.append({**meta, "rule": rule, "rule_sha256": hashlib.sha256(
                    rule.encode("utf-8")
                ).hexdigest()})
            # Independent fidelity audit candidate: 70B executes the exact target orbit.
            form_scores = []
            reference_form_scores = []
            for form in design["forms_by_key"][metric_key]:
                form_scores.append(decoder.score_binary_constrained([
                    _YESNO_TEMPLATE.format(rubric=form, text=str(text)[:int(design["max_chars"])])
                    for text in design["heldout_texts"]
                ], pos="YES", neg="NO", seed=0))
                reference_form_scores.append(decoder.score_binary_constrained([
                    _YESNO_TEMPLATE.format(rubric=form, text=str(text)[:int(design["max_chars"])])
                    for text in design["reference_probe_texts"]
                ], pos="YES", neg="NO", seed=0))
            payload = {
                "schema": CONSTRUCTOR_SCHEMA, "metric_key": metric_key,
                "decoder_model": model, "decoder_revision": revision,
                "c1": c1, "c2_rules": rules,
                "c3_rules": c3_rules,
                "c3_blind_rule_reuses_c2_blind": True,
                "c3_exhaustive_state_cap_available": False,
                "c3_exhaustive_state_cap_reason": (
                    "binary v13 caches contain 64/256-state decoder outputs, not the 65,536 "
                    "quaternary decoder outputs required for exact lookup"
                ),
                "c0_70b_form_scores": form_scores,
                "c0_70b_reference_form_scores": reference_form_scores,
                "design_freeze_sha256": design["freeze_sha256"],
            }
            payload["sha256"] = _sha(payload)
            _atomic_json(target, payload)
            completed.append(metric_key)
        if metric_keys is None:
            _run_planted_constructor(root, decoder, model=model, revision=revision)
    finally:
        release_resident_engines()
    manifest = {
        "schema": CONSTRUCTOR_SCHEMA, "model": model, "revision": revision,
        "n_metrics": len(completed), "metric_keys": sorted(completed),
    }
    manifest["sha256"] = _sha(manifest)
    _atomic_json(root / "constructor" / "manifest.json", manifest)
    return manifest


def _criterion_scores(
    executor, criteria: Sequence[str], texts: Sequence[str], *, max_chars: int,
) -> dict[str, list[float]]:
    unique = list(dict.fromkeys(map(str, criteria)))
    output = {}
    heldout_n = len(texts)
    query_batch_size = int(os.environ.get("CEILING_QUERY_BATCH_SIZE", "2048"))
    criteria_per_batch = max(1, query_batch_size // max(1, heldout_n))
    for start in range(0, len(unique), criteria_per_batch):
        batch = unique[start:start + criteria_per_batch]
        prompts = [
            _YESNO_TEMPLATE.format(rubric=criterion, text=str(text)[:int(max_chars)])
            for criterion in batch for text in texts
        ]
        values = np.asarray(executor.score_binary_constrained(
            prompts, pos="YES", neg="NO", seed=0
        ), dtype=float)
        if values.shape != (len(batch) * heldout_n,) or np.any(~np.isfinite(values)):
            raise RuntimeError("executor returned incomplete criterion scores")
        matrix = values.reshape(len(batch), heldout_n)
        for criterion, row in zip(batch, matrix):
            output[hashlib.sha256(criterion.encode("utf-8")).hexdigest()] = list(map(float, row))
    return output


def _run_planted_executor(root: Path, executor, *, revision: str) -> None:
    design = json.loads((root / "planted" / "planted_design.json").read_text())
    constructor = json.loads((root / "planted" / "constructor.json").read_text())
    descriptions = {str(row["id"]): str(row["description"]) for row in design["criteria"]}
    criteria = list(descriptions.values())
    for row in constructor["rows"]:
        criteria.extend(row["c2_rules"].values())
        for condition in ("blind", "canonical", "shuffled"):
            criteria.append(descriptions[row["c1"][condition]["picked_metric_key"]])
    scores = _criterion_scores(
        executor, criteria, design["heldout_texts"], max_chars=int(design["max_chars"]),
    )
    payload = {
        "schema": "cr3-planted-executor-v1", "executor_model": FIXED_EXECUTOR,
        "executor_revision": revision, "criterion_scores": scores,
        "constructor_sha256": constructor["sha256"], "design_sha256": design["sha256"],
    }
    payload["sha256"] = _sha(payload)
    _atomic_json(root / "planted" / "executor.json", payload)


def run_executor(
    out_root: str | Path, *, metric_keys: Sequence[str] | None = None,
    fake: bool = False,
) -> dict:
    root = Path(out_root).resolve()
    designs = _load_designs(root, metric_keys)
    executor, revision = _backend(FIXED_EXECUTOR, fake=fake)
    completed = []
    try:
        for design in designs:
            metric_key = str(design["metric_key"])
            target = root / "executor" / _safe(metric_key) / "executor.json"
            if target.is_file():
                completed.append(metric_key)
                continue
            constructor_path = root / "constructor" / _safe(metric_key) / "constructor.json"
            constructor = json.loads(constructor_path.read_text(encoding="utf-8"))
            criteria = list(design["forms_by_key"][metric_key])
            for arm in constructor["c1"].values():
                picks = [arm["blind"]["picked_metric_key"]]
                for panel in arm["panels"]:
                    picks.extend([
                        panel["canonical"]["picked_metric_key"],
                        panel["shuffled"]["picked_metric_key"],
                    ])
                for picked in picks:
                    criteria.extend(design["forms_by_key"][picked])
            criteria.extend(row["rule"] for row in constructor["c2_rules"])
            criteria.extend(row["rule"] for row in constructor["c3_rules"])
            scores = _criterion_scores(
                executor, criteria, design["heldout_texts"], max_chars=int(design["max_chars"]),
            )
            reference_scores = _criterion_scores(
                executor, design["forms_by_key"][metric_key],
                design["reference_probe_texts"], max_chars=int(design["max_chars"]),
            )
            payload = {
                "schema": EXECUTOR_SCHEMA, "metric_key": metric_key,
                "executor_model": FIXED_EXECUTOR, "executor_revision": revision,
                "readout_id": CR3_BINARY_READOUT_ID, "criterion_scores": scores,
                "c0_reference_scores": reference_scores,
                "constructor_sha256": constructor["sha256"],
                "design_freeze_sha256": design["freeze_sha256"],
            }
            payload["sha256"] = _sha(payload)
            _atomic_json(target, payload)
            completed.append(metric_key)
        if metric_keys is None:
            _run_planted_executor(root, executor, revision=revision)
    finally:
        release_resident_engines()
    manifest = {
        "schema": EXECUTOR_SCHEMA, "model": FIXED_EXECUTOR, "revision": revision,
        "n_metrics": len(completed), "metric_keys": sorted(completed),
    }
    manifest["sha256"] = _sha(manifest)
    _atomic_json(root / "executor" / "manifest.json", manifest)
    return manifest


def _orbit_prediction(forms: Sequence[str], scores: Mapping[str, Sequence[float]]) -> np.ndarray:
    matrix = np.vstack([
        np.asarray(scores[hashlib.sha256(str(form).encode("utf-8")).hexdigest()], dtype=float)
        for form in forms
    ])
    return (np.mean(matrix, axis=0) > 0.5).astype(np.uint8)


def _rule_prediction(rule: str, scores: Mapping[str, Sequence[float]]) -> np.ndarray:
    key = hashlib.sha256(str(rule).encode("utf-8")).hexdigest()
    return (np.asarray(scores[key], dtype=float) > 0.5).astype(np.uint8)


def _metric_reference(
    frame: pd.DataFrame, metric_key: str, indices: Sequence[int] | None = None,
) -> np.ndarray:
    rows = frame[frame.metric_key == metric_key].sort_values("probe_index")
    if not len(rows) or rows.probe_index.astype(int).tolist() != list(range(len(rows))):
        raise RuntimeError(f"independent reference for {metric_key} is incomplete")
    result = rows.majority_label.to_numpy(dtype=np.uint8)
    return result if indices is None else result[np.asarray(indices, dtype=int)]


def aggregate_planted_ladder(root: Path) -> pd.DataFrame:
    design = json.loads((root / "planted" / "planted_design.json").read_text())
    constructor = json.loads((root / "planted" / "constructor.json").read_text())
    executor = json.loads((root / "planted" / "executor.json").read_text())
    by_id = {str(row["id"]): row for row in design["criteria"]}
    scores = executor["criterion_scores"]
    rows = []
    for evidence in constructor["rows"]:
        target_id = str(evidence["target_id"])
        target = np.asarray(by_id[target_id]["heldout_labels"], dtype=np.uint8)
        description = str(by_id[target_id]["description"])
        c0 = _rule_prediction(description, scores)
        c0_null = _permuted_ladder_values(
            target, [c0], seed=_stable_seed("planted", target_id, "C0"),
        )
        c0_permutation = permutation_pvalue(
            plugin_binary_mutual_information(target, c0), c0_null
        )
        rows.append({
            "criterion_id": target_id, "rung": "C0", "value_bits":
                plugin_binary_mutual_information(target, c0),
            "mm_value_bits": miller_madow_mutual_information(target, c0, clip=True),
            "accuracy": ordinary_accuracy(target, c0),
            "balanced_accuracy": balanced_agreement(target, c0),
            "permutation_percentile": c0_permutation["percentile"],
            "permutation_z_score": c0_permutation["z_score"],
            "permutation_p_greater_equal": c0_permutation["p_greater_equal"],
        })
        picked = evidence["c1"]["canonical"]["picked_metric_key"]
        blind_pick = evidence["c1"]["blind"]["picked_metric_key"]
        shuffled_pick = evidence["c1"]["shuffled"]["picked_metric_key"]
        c1 = value_against_controls(
            target, _rule_prediction(by_id[picked]["description"], scores),
            _rule_prediction(by_id[blind_pick]["description"], scores),
            _rule_prediction(by_id[shuffled_pick]["description"], scores), corrected=False,
        )
        c1_null = _permuted_ladder_values(
            target, [_rule_prediction(by_id[picked]["description"], scores)],
            [_rule_prediction(by_id[blind_pick]["description"], scores)],
            [_rule_prediction(by_id[shuffled_pick]["description"], scores)],
            seed=_stable_seed("planted", target_id, "C1"),
        )
        c1_permutation = permutation_pvalue(c1["value"], c1_null)
        rows.append({
            "criterion_id": target_id, "rung": "C1", "value_bits": c1["value"],
            "mm_value_bits": value_against_controls(
                target, _rule_prediction(by_id[picked]["description"], scores),
                _rule_prediction(by_id[blind_pick]["description"], scores),
                _rule_prediction(by_id[shuffled_pick]["description"], scores), corrected=True,
            )["value"],
            "accuracy": c1["accuracy"], "balanced_accuracy": c1["balanced_accuracy"],
            "identification_correct": picked == target_id,
            "permutation_percentile": c1_permutation["percentile"],
            "permutation_z_score": c1_permutation["z_score"],
            "permutation_p_greater_equal": c1_permutation["p_greater_equal"],
        })
        rules = evidence["c2_rules"]
        c2 = value_against_controls(
            target, _rule_prediction(rules["canonical"], scores),
            _rule_prediction(rules["blind"], scores),
            _rule_prediction(rules["shuffled"], scores), corrected=False,
        )
        c2_null = _permuted_ladder_values(
            target, [_rule_prediction(rules["canonical"], scores)],
            [_rule_prediction(rules["blind"], scores)],
            [_rule_prediction(rules["shuffled"], scores)],
            seed=_stable_seed("planted", target_id, "C2"),
        )
        c2_permutation = permutation_pvalue(c2["value"], c2_null)
        rows.append({
            "criterion_id": target_id, "rung": "C2", "value_bits": c2["value"],
            "mm_value_bits": value_against_controls(
                target, _rule_prediction(rules["canonical"], scores),
                _rule_prediction(rules["blind"], scores),
                _rule_prediction(rules["shuffled"], scores), corrected=True,
            )["value"],
            "accuracy": c2["accuracy"], "balanced_accuracy": c2["balanced_accuracy"],
            "permutation_percentile": c2_permutation["percentile"],
            "permutation_z_score": c2_permutation["z_score"],
            "permutation_p_greater_equal": c2_permutation["p_greater_equal"],
        })
    frame = pd.DataFrame(rows)
    _atomic_parquet(root / "planted" / "ceiling_ladder.parquet", frame)
    return frame


def aggregate_ladder(out_root: str | Path) -> pd.DataFrame:
    root = Path(out_root).resolve()
    references = pd.read_parquet(root / "reference" / "sonnet_labels.parquet")
    valid_metric_keys = set(map(str, references.metric_key.unique()))
    designs = [
        design for design in _load_designs(root)
        if str(design["metric_key"]) in valid_metric_keys
    ]
    reference_manifest = json.loads((
        root / "reference" / "reference_manifest.json"
    ).read_text(encoding="utf-8"))
    rows = []
    fidelity_rows = []
    executor_agreement_rows = []
    for design in designs:
        metric_key = str(design["metric_key"])
        reference_target = _metric_reference(references, metric_key)
        target = reference_target[np.asarray(design["heldout_indices"], dtype=int)]
        reference_rows = references[references.metric_key == metric_key].sort_values(
            "probe_index"
        )
        reference_latent = reference_rows.latent_probability.to_numpy(dtype=float)
        latent_target = reference_latent[
            np.asarray(design["heldout_indices"], dtype=int)
        ]
        constructor = json.loads((
            root / "constructor" / _safe(metric_key) / "constructor.json"
        ).read_text(encoding="utf-8"))
        executor = json.loads((
            root / "executor" / _safe(metric_key) / "executor.json"
        ).read_text(encoding="utf-8"))
        scores = executor["criterion_scores"]
        c0_8b = _orbit_prediction(design["forms_by_key"][metric_key], scores)
        c0_70b = (np.mean(np.asarray(constructor["c0_70b_form_scores"], dtype=float), axis=0)
                   > 0.5).astype(np.uint8)
        c0_8b_reference_probability = np.mean(np.vstack([
            np.asarray(executor["c0_reference_scores"][
                hashlib.sha256(str(form).encode("utf-8")).hexdigest()
            ], dtype=float) for form in design["forms_by_key"][metric_key]
        ]), axis=0)
        c0_70b_reference_probability = np.mean(np.asarray(
            constructor["c0_70b_reference_form_scores"], dtype=float
        ), axis=0)
        c0_8b_reference = (c0_8b_reference_probability > 0.5).astype(np.uint8)
        c0_70b_reference = (c0_70b_reference_probability > 0.5).astype(np.uint8)
        for model, prediction, probability in (
            (FIXED_EXECUTOR, c0_8b_reference, c0_8b_reference_probability),
            (DEFAULT_CONSTRUCTOR, c0_70b_reference, c0_70b_reference_probability),
        ):
            fidelity_rows.append({
                "metric_key": metric_key, "task": design["task"], "model": model,
                "accuracy": ordinary_accuracy(reference_target, prediction),
                "balanced_accuracy": balanced_agreement(reference_target, prediction),
                "mi_bits": plugin_binary_mutual_information(reference_target, prediction),
                "mm_mi_bits": miller_madow_mutual_information(reference_target, prediction),
                "positive_rate": float(np.mean(prediction)),
                "auc": binary_auc(reference_target, probability),
                "attenuation_sensitivity_mi_bits": soft_binary_mutual_information(
                    reference_latent, prediction
                ),
            })
        executor_agreement_rows.append({
            "metric_key": metric_key, "task": design["task"],
            "ordinary_agreement": ordinary_accuracy(c0_8b_reference, c0_70b_reference),
            "balanced_agreement_8b_as_reference": balanced_agreement(
                c0_8b_reference, c0_70b_reference
            ),
            "mi_bits": plugin_binary_mutual_information(
                c0_8b_reference, c0_70b_reference
            ),
        })
        c0 = {
            "mi": plugin_binary_mutual_information(target, c0_8b),
            "mm_mi": miller_madow_mutual_information(target, c0_8b),
            "accuracy": ordinary_accuracy(target, c0_8b),
            "balanced_accuracy": balanced_agreement(target, c0_8b),
            "attenuation_sensitivity_mi": soft_binary_mutual_information(
                latent_target, c0_8b
            ),
        }
        c0_null = _permuted_ladder_values(
            target, [c0_8b], n_permutations=N_PERMUTATIONS,
            seed=_stable_seed(metric_key, "C0", "permutation-null"),
        )
        c0_permutation = permutation_pvalue(c0["mi"], c0_null)
        rows.append({
            "metric_key": metric_key, "task": design["task"], "rung": "C0",
            "menu_arm": None, "raw_mi_bits": c0["mi"], "mm_mi_bits": c0["mm_mi"],
            "raw_lift_bits": c0["mi"], "mm_raw_lift_bits": c0["mm_mi"],
            "value_bits": max(0.0, c0["mi"]), "mm_value_bits": max(0.0, c0["mm_mi"]),
            "accuracy": c0["accuracy"], "balanced_accuracy": c0["balanced_accuracy"],
            "blind_mi_bits": None, "shuffled_mi_bits": None,
            "attenuation_sensitivity_mi_bits": c0["attenuation_sensitivity_mi"],
            "permutation_percentile": c0_permutation["percentile"],
            "permutation_z_score": c0_permutation["z_score"],
            "permutation_p_greater_equal": c0_permutation["p_greater_equal"],
            "permutation_null_median": c0_permutation["null_median"],
        })
        for menu_arm, c1 in constructor["c1"].items():
            blind_key = c1["blind"]["picked_metric_key"]
            blind_pred = _orbit_prediction(design["forms_by_key"][blind_key], scores)
            target_menu_position = c1["blind"]["menu_keys"].index(metric_key)
            blind_target_probability = float(
                c1["blind"]["posterior"][target_menu_position]
            )
            per_panel = []
            per_panel_mm = []
            accuracies = []
            balances = []
            predictions = []
            blind_predictions = []
            shuffled_predictions = []
            mcq_raw_lifts = []
            mcq_values = []
            mcq_target_probabilities = []
            mcq_shuffled_probabilities = []
            for panel in c1["panels"]:
                picked = panel["canonical"]["picked_metric_key"]
                shuffled_pick = panel["shuffled"]["picked_metric_key"]
                prediction = _orbit_prediction(design["forms_by_key"][picked], scores)
                shuffled_prediction = _orbit_prediction(
                    design["forms_by_key"][shuffled_pick], scores
                )
                target_probability = float(
                    panel["canonical"]["posterior"][target_menu_position]
                )
                shuffled_probability = float(
                    panel["shuffled"]["posterior"][target_menu_position]
                )
                mcq_raw = target_probability - max(
                    blind_target_probability, shuffled_probability
                )
                mcq_target_probabilities.append(target_probability)
                mcq_shuffled_probabilities.append(shuffled_probability)
                mcq_raw_lifts.append(mcq_raw)
                mcq_values.append(max(0.0, mcq_raw))
                predictions.append(prediction)
                blind_predictions.append(blind_pred)
                shuffled_predictions.append(shuffled_prediction)
                per_panel.append(value_against_controls(
                    target, prediction, blind_pred, shuffled_prediction, corrected=False,
                ))
                per_panel_mm.append(value_against_controls(
                    target, prediction, blind_pred, shuffled_prediction, corrected=True,
                ))
                accuracies.append(ordinary_accuracy(target, prediction))
                balances.append(balanced_agreement(target, prediction))
            c1_observed = float(np.mean([row["value"] for row in per_panel]))
            c1_null = _permuted_ladder_values(
                target, predictions, blind_predictions, shuffled_predictions,
                n_permutations=N_PERMUTATIONS,
                seed=_stable_seed(metric_key, "C1", menu_arm, "permutation-null"),
            )
            c1_permutation = permutation_pvalue(c1_observed, c1_null)
            rows.append({
                "metric_key": metric_key, "task": design["task"], "rung": "C1",
                "menu_arm": menu_arm,
                "raw_mi_bits": float(np.mean([row["mi"] for row in per_panel])),
                "mm_mi_bits": float(np.mean([row["mi"] for row in per_panel_mm])),
                "raw_lift_bits": float(np.mean([row["raw_lift"] for row in per_panel])),
                "mm_raw_lift_bits": float(np.mean([row["raw_lift"] for row in per_panel_mm])),
                "value_bits": c1_observed,
                "mm_value_bits": float(np.mean([row["value"] for row in per_panel_mm])),
                "accuracy": float(np.mean(accuracies)),
                "balanced_accuracy": float(np.mean(balances)),
                "blind_mi_bits": float(np.mean([row["blind_mi"] for row in per_panel])),
                "shuffled_mi_bits": float(np.mean([row["shuffled_mi"] for row in per_panel])),
                "identification_rate": float(np.mean([
                    panel["canonical"]["picked_metric_key"] == metric_key
                    for panel in c1["panels"]
                ])),
                "mcq_target_probability": float(np.mean(mcq_target_probabilities)),
                "mcq_blind_target_probability": blind_target_probability,
                "mcq_shuffled_target_probability": float(np.mean(
                    mcq_shuffled_probabilities
                )),
                "mcq_raw_lift": float(np.mean(mcq_raw_lifts)),
                "mcq_value": float(np.mean(mcq_values)),
                "attenuation_sensitivity_mi_bits": float(np.mean([
                    soft_binary_mutual_information(latent_target, prediction)
                    for prediction in predictions
                ])),
                "permutation_percentile": c1_permutation["percentile"],
                "permutation_z_score": c1_permutation["z_score"],
                "permutation_p_greater_equal": c1_permutation["p_greater_equal"],
                "permutation_null_median": c1_permutation["null_median"],
            })
        rules = constructor["c2_rules"]
        blind_rule = next(row["rule"] for row in rules if row["condition"] == "blind")
        blind_pred = _rule_prediction(blind_rule, scores)
        per_panel = []
        per_panel_mm = []
        predictions = []
        shuffled_predictions = []
        for position in range(len(design["panels"])):
            canonical_rule = next(
                row["rule"] for row in rules
                if row["panel"] == position and row["condition"] == "canonical"
            )
            shuffled_rule = next(
                row["rule"] for row in rules
                if row["panel"] == position and row["condition"] == "shuffled"
            )
            prediction = _rule_prediction(canonical_rule, scores)
            shuffled_prediction = _rule_prediction(shuffled_rule, scores)
            predictions.append(prediction)
            shuffled_predictions.append(shuffled_prediction)
            per_panel.append(value_against_controls(
                target, prediction, blind_pred, shuffled_prediction, corrected=False,
            ))
            per_panel_mm.append(value_against_controls(
                target, prediction, blind_pred, shuffled_prediction, corrected=True,
            ))
        c2_observed = float(np.mean([row["value"] for row in per_panel]))
        c2_null = _permuted_ladder_values(
            target, predictions, [blind_pred] * len(predictions), shuffled_predictions,
            n_permutations=N_PERMUTATIONS,
            seed=_stable_seed(metric_key, "C2", "permutation-null"),
        )
        c2_permutation = permutation_pvalue(c2_observed, c2_null)
        rows.append({
            "metric_key": metric_key, "task": design["task"], "rung": "C2",
            "menu_arm": None,
            "raw_mi_bits": float(np.mean([row["mi"] for row in per_panel])),
            "mm_mi_bits": float(np.mean([row["mi"] for row in per_panel_mm])),
            "raw_lift_bits": float(np.mean([row["raw_lift"] for row in per_panel])),
            "mm_raw_lift_bits": float(np.mean([row["raw_lift"] for row in per_panel_mm])),
            "value_bits": c2_observed,
            "mm_value_bits": float(np.mean([row["value"] for row in per_panel_mm])),
            "accuracy": float(np.mean([row["accuracy"] for row in per_panel])),
            "balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in per_panel])),
            "blind_mi_bits": float(np.mean([row["blind_mi"] for row in per_panel])),
            "shuffled_mi_bits": float(np.mean([row["shuffled_mi"] for row in per_panel])),
            "attenuation_sensitivity_mi_bits": float(np.mean([
                soft_binary_mutual_information(latent_target, prediction)
                for prediction in predictions
            ])),
            "permutation_percentile": c2_permutation["percentile"],
            "permutation_z_score": c2_permutation["z_score"],
            "permutation_p_greater_equal": c2_permutation["p_greater_equal"],
            "permutation_null_median": c2_permutation["null_median"],
        })
        c3_predictions = []
        c3_shuffled_predictions = []
        c3_panel = []
        c3_panel_mm = []
        for position in range(len(design["panels"])):
            canonical_rule = next(
                row["rule"] for row in constructor["c3_rules"]
                if row["panel"] == position and row["condition"] == "canonical"
            )
            shuffled_rule = next(
                row["rule"] for row in constructor["c3_rules"]
                if row["panel"] == position and row["condition"] == "shuffled"
            )
            prediction = _rule_prediction(canonical_rule, scores)
            shuffled_prediction = _rule_prediction(shuffled_rule, scores)
            c3_predictions.append(prediction)
            c3_shuffled_predictions.append(shuffled_prediction)
            c3_panel.append(value_against_controls(
                target, prediction, blind_pred, shuffled_prediction, corrected=False,
            ))
            c3_panel_mm.append(value_against_controls(
                target, prediction, blind_pred, shuffled_prediction, corrected=True,
            ))
        c3_observed = float(np.mean([row["value"] for row in c3_panel]))
        c3_null = _permuted_ladder_values(
            target, c3_predictions, [blind_pred] * len(c3_predictions),
            c3_shuffled_predictions, n_permutations=N_PERMUTATIONS,
            seed=_stable_seed(metric_key, "C3", "permutation-null"),
        )
        c3_permutation = permutation_pvalue(c3_observed, c3_null)
        rows.append({
            "metric_key": metric_key, "task": design["task"], "rung": "C3",
            "menu_arm": None,
            "raw_mi_bits": float(np.mean([row["mi"] for row in c3_panel])),
            "mm_mi_bits": float(np.mean([row["mi"] for row in c3_panel_mm])),
            "raw_lift_bits": float(np.mean([row["raw_lift"] for row in c3_panel])),
            "mm_raw_lift_bits": float(np.mean([row["raw_lift"] for row in c3_panel_mm])),
            "value_bits": c3_observed,
            "mm_value_bits": float(np.mean([row["value"] for row in c3_panel_mm])),
            "accuracy": float(np.mean([row["accuracy"] for row in c3_panel])),
            "balanced_accuracy": float(np.mean([
                row["balanced_accuracy"] for row in c3_panel
            ])),
            "blind_mi_bits": float(np.mean([row["blind_mi"] for row in c3_panel])),
            "shuffled_mi_bits": float(np.mean([
                row["shuffled_mi"] for row in c3_panel
            ])),
            "attenuation_sensitivity_mi_bits": float(np.mean([
                soft_binary_mutual_information(latent_target, prediction)
                for prediction in c3_predictions
            ])),
            "permutation_percentile": c3_permutation["percentile"],
            "permutation_z_score": c3_permutation["z_score"],
            "permutation_p_greater_equal": c3_permutation["p_greater_equal"],
            "permutation_null_median": c3_permutation["null_median"],
            "exact_structural_cap_bits": None,
            "exact_structural_cap_unavailable_reason": constructor[
                "c3_exhaustive_state_cap_reason"
            ],
        })
    frame = pd.DataFrame(rows)
    _atomic_parquet(root / "ceiling_ladder.parquet", frame)
    fidelity = pd.DataFrame(fidelity_rows)
    _atomic_parquet(root / "fidelity_audit_v2.parquet", fidelity)
    executor_agreement = pd.DataFrame(executor_agreement_rows)
    _atomic_parquet(root / "cross_executor_disagreement.parquet", executor_agreement)
    planted = aggregate_planted_ladder(root)
    comparison = fidelity.pivot(index="metric_key", columns="model", values="balanced_accuracy")
    ladder_tests = {
        f"{rung}:{arm if pd.notna(arm) else 'default'}": aggregate_raw_lift_tests(group)
        for (rung, arm), group in frame.groupby(["rung", "menu_arm"], dropna=False)
    }
    decision = bootstrap_ladder_decision(frame)
    report = {
        "schema": SCHEMA, "n_metrics": len(designs),
        "n_independent_reference_valid_metrics": len(designs),
        "n_independent_reference_void_metrics": int(
            reference_manifest.get("n_void_metrics", 0)
        ),
        "independent_reference_void_metric_keys": reference_manifest.get(
            "void_metric_keys", []
        ),
        "ladder_path": str(root / "ceiling_ladder.parquet"),
        "fidelity_path": str(root / "fidelity_audit_v2.parquet"),
        "cross_executor_disagreement_path": str(
            root / "cross_executor_disagreement.parquet"
        ),
        "planted_ladder_path": str(root / "planted" / "ceiling_ladder.parquet"),
        "ladder_means": frame.groupby(["rung", "menu_arm"], dropna=False)[
            ["value_bits", "mm_value_bits", "accuracy", "balanced_accuracy"]
        ].mean().reset_index().to_dict(orient="records"),
        "mcq_identification_means": frame[frame.rung == "C1"].groupby(
            "menu_arm"
        )[["mcq_target_probability", "mcq_raw_lift", "mcq_value"]].mean(
        ).reset_index().to_dict(orient="records"),
        "fidelity_means": fidelity.groupby("model")[[
            "auc", "accuracy", "balanced_accuracy", "mi_bits", "mm_mi_bits",
            "attenuation_sensitivity_mi_bits",
        ]].mean().reset_index().to_dict(orient="records"),
        "planted_ladder_means": planted.groupby("rung")[[
            "value_bits", "mm_value_bits", "accuracy", "balanced_accuracy"
        ]].mean().reset_index().to_dict(orient="records"),
        "unclipped_lift_tests": ladder_tests,
        "ceiling_ladder_decision": decision,
        "cross_executor_balanced_agreement_difference_mean": float(
            comparison.get(DEFAULT_CONSTRUCTOR, pd.Series(dtype=float)).mean()
            - comparison.get(FIXED_EXECUTOR, pd.Series(dtype=float)).mean()
        ),
        "cross_executor_agreement_means": executor_agreement[[
            "ordinary_agreement", "balanced_agreement_8b_as_reference", "mi_bits"
        ]].mean().to_dict(),
        "old_frozen_8b_reference_valid_for_executor_selection": False,
        "independent_reference": "three blind Sonnet passes with majority vote",
        "attenuation_analysis_is_sensitivity_only": True,
        "permutation_null_preserves_panel_and_control_aggregation": True,
        "permutation_count_per_metric_rung": N_PERMUTATIONS,
        "no_v14_tuning_launched": True,
    }
    report["sha256"] = _sha(report)
    _atomic_json(root / "report.json", report)
    return frame


def aggregate_raw_lift_tests(rows: pd.DataFrame) -> dict:
    values = np.asarray(rows["raw_lift_bits"], dtype=float)
    nonzero = values[~np.isclose(values, 0.0)]
    result = {
        "n": len(values), "n_positive": int(np.sum(values > 0)),
        "n_negative": int(np.sum(values < 0)),
        "sign_test_p_greater": (
            float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5,
                            alternative="greater").pvalue) if len(nonzero) else 1.0
        ),
    }
    if len(nonzero):
        result["wilcoxon_p_greater"] = float(wilcoxon(
            nonzero, alternative="greater", zero_method="wilcox"
        ).pvalue)
    else:
        result["wilcoxon_p_greater"] = 1.0
    return result


def bootstrap_ladder_decision(
    rows: pd.DataFrame, *, n_bootstrap: int = 20_000, seed: int = 14,
) -> dict[str, object]:
    """Paired metric bootstrap for the two pre-declared ladder interpretations."""
    c0 = rows[rows.rung == "C0"].set_index("metric_key")["value_bits"]
    c1 = rows[(rows.rung == "C1") & (rows.menu_arm == "full_task_bank")].set_index(
        "metric_key"
    )["value_bits"]
    c2 = rows[rows.rung == "C2"].set_index("metric_key")["value_bits"]
    c3 = rows[rows.rung == "C3"].set_index("metric_key")["value_bits"]
    common = sorted(set(c0.index) & set(c1.index) & set(c2.index) & set(c3.index))
    if not common:
        raise RuntimeError("ladder decision has no aligned metrics")
    matrix = np.column_stack([
        c0.loc[common].to_numpy(float), c1.loc[common].to_numpy(float),
        c2.loc[common].to_numpy(float), c3.loc[common].to_numpy(float),
    ])
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(0, len(matrix), size=(int(n_bootstrap), len(matrix)))
    means = matrix[samples].mean(axis=1)
    d01 = means[:, 0] - means[:, 1]
    d12 = means[:, 1] - means[:, 2]
    d32 = means[:, 3] - means[:, 2]

    def interval(values: np.ndarray) -> list[float]:
        return list(map(float, np.quantile(values, [0.025, 0.5, 0.975])))

    d01_ci = interval(d01)
    d12_ci = interval(d12)
    d32_ci = interval(d32)
    if d01_ci[0] <= 0.0 <= d01_ci[2] and d12_ci[0] > 0.0:
        classification = "ARTICULATION_BOTTLENECK_C0_APPROX_C1_ABOVE_C2"
        action = "v14 decoder articulation tuning is supported"
    elif d01_ci[0] > 0.0 and d12_ci[0] <= 0.0 <= d12_ci[2]:
        classification = "IDENTIFICATION_BOTTLENECK_C0_ABOVE_C1_APPROX_C2"
        action = "do not tune articulation; raise k or repair panels"
    else:
        classification = "MIXED_OR_UNRESOLVED"
        action = "report the ladder without launching tuning"
    return {
        "n_aligned_metrics": len(common),
        "means": {"C0": float(matrix[:, 0].mean()), "C1": float(matrix[:, 1].mean()),
                  "C2": float(matrix[:, 2].mean()), "C3": float(matrix[:, 3].mean())},
        "paired_bootstrap_C0_minus_C1_95pct": d01_ci,
        "paired_bootstrap_C1_minus_C2_95pct": d12_ci,
        "paired_bootstrap_C3_minus_C2_95pct": d32_ci,
        "c3_code_gain_supported": bool(d32_ci[0] > 0.0),
        "classification": classification, "action": action,
        "bootstrap_replicates": int(n_bootstrap),
    }


def _load_v13_cache_rows(paths: Sequence[str | Path]) -> tuple[dict[str, dict], dict[str, dict]]:
    inductions: dict[str, dict] = {}
    executions: dict[tuple[str, str], dict] = {}
    for path in map(Path, paths):
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            for key, kind, payload_json in connection.execute(
                "SELECT key, kind, payload_json FROM entries "
                "WHERE kind IN ('behavioral_induction','behavioral_execution')"
            ):
                payload = json.loads(payload_json)
                if kind == "behavioral_induction":
                    prior = inductions.setdefault(str(key), payload)
                else:
                    identity = (str(payload["rule_sha256"]), str(payload["heldout_sha256"]))
                    prior = executions.setdefault(identity, payload)
                if prior != payload:
                    raise RuntimeError(f"non-identical v13 cache evidence for {key}")
        finally:
            connection.close()
    return inductions, executions


def _binary_mi_matrix(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Plug-in binary MI for every target-row/prediction-row pair."""
    left = np.asarray(targets, dtype=np.uint8)
    right = np.asarray(predictions, dtype=np.uint8)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("MI matrix inputs are misaligned")
    n = float(left.shape[1])
    n11 = left.astype(float) @ right.astype(float).T
    n1x = left.sum(axis=1, dtype=float)[:, None]
    nx1 = right.sum(axis=1, dtype=float)[None, :]
    counts = (n - n1x - nx1 + n11, nx1 - n11, n1x - n11, n11)
    row_margins = (n - n1x, n1x)
    column_margins = (n - nx1, nx1)
    result = np.zeros_like(n11, dtype=float)
    for row in (0, 1):
        for column in (0, 1):
            count = counts[row * 2 + column]
            keep = count > 0.0
            term = np.zeros_like(count)
            denominator = row_margins[row] * column_margins[column]
            term[keep] = (count[keep] / n) * np.log2(
                (count[keep] * n) / denominator[keep]
            )
            result += term
    return result


def _native_v13_metric_audit(
    *, artifact_dir: Path, context: Mapping[str, object], target: np.ndarray,
    reference_name: str, inductions: Mapping[str, dict],
    executions: Mapping[tuple[str, str], dict], n_permutations: int,
) -> tuple[dict[str, object], np.ndarray | None]:
    certificate = json.loads((artifact_dir / "certificate.json").read_text())
    design = context["design"]
    active = active_panel_design(design, channel="behavioral", tier="B")
    with np.load(artifact_dir / "state_tables.npz", allow_pickle=True) as state:
        rule_sha = np.asarray(state["unconstrained__rule_sha256"], dtype=object).astype(str)
    if rule_sha.shape != (len(active["panels"]), 64):
        raise RuntimeError(f"incomplete v13 state rule table in {artifact_dir}")
    heldout = design["heldout"]
    heldout_sha = hashlib.sha256(json.dumps({
        "probe_sha256": design["probe_sha256"], "indices": heldout["indices"],
        "text_sha256": heldout["probe_text_sha256"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    blind_text = v13_blind_prompt(
        noun=str(context["codebook"]["reconstruction_noun"]), arm="unconstrained",
    )
    blind_key = v13_cache_key("behavioral_induction", {
        "constructor_revision": str(certificate["constructor"]["revision"]),
        "panel_sha256": "blind-no-panel", "state": -1, "arm": "unconstrained",
        "induction_template_id": V13_INDUCTION_TEMPLATE_ID,
        "prompt_sha256": hashlib.sha256(blind_text.encode()).hexdigest(),
    })
    if blind_key not in inductions:
        raise RuntimeError(f"missing v13 blind induction for {certificate['metric_key']}")
    blind_rule = str(inductions[blind_key]["rule_sha256"])
    all_rules = list(dict.fromkeys([*rule_sha.ravel().tolist(), blind_rule]))
    prediction_rows = []
    for rule in all_rules:
        key = (str(rule), heldout_sha)
        if key not in executions:
            raise RuntimeError(f"missing v13 execution {rule[:12]} for {certificate['metric_key']}")
        prediction_rows.append(executions[key]["hard_predictions"])
    predictions = np.asarray(prediction_rows, dtype=np.uint8)
    if predictions.shape != (len(all_rules), len(target)):
        raise RuntimeError("v13 execution vectors do not align with frozen H")
    rule_position = {rule: index for index, rule in enumerate(all_rules)}
    rule_indices = np.vectorize(rule_position.__getitem__)(rule_sha)
    shuffled_indices = np.empty_like(rule_indices)
    for panel_position, panel in enumerate(active["panels"]):
        for state in range(64):
            shuffled_indices[panel_position, state] = rule_indices[
                panel_position,
                v13_shuffled_state(state, 6, str(panel["panel_sha256"])),
            ]
    blind_index = rule_position[blind_rule]

    plugin_by_rule = _binary_mi_matrix(target[None, :], predictions)[0]
    mm_by_rule = np.asarray([
        miller_madow_mutual_information(target, row) for row in predictions
    ])
    plugin_raw = plugin_by_rule[rule_indices] - np.maximum(
        plugin_by_rule[blind_index], plugin_by_rule[shuffled_indices]
    )
    mm_raw = mm_by_rule[rule_indices] - np.maximum(
        mm_by_rule[blind_index], mm_by_rule[shuffled_indices]
    )
    plugin_values = np.maximum(plugin_raw, 0.0)
    mm_values = np.maximum(mm_raw, 0.0)
    accuracy_values = np.asarray([
        ordinary_accuracy(target, row) for row in predictions
    ])[rule_indices]
    balanced_values = np.asarray([
        balanced_agreement(target, row) for row in predictions
    ])[rule_indices]
    signatures = np.asarray(context["population"]["signatures"], dtype=float)
    plugin_aggregation = enumerate_exact_pool_values(
        design, channel="behavioral", tier="B", state_values=plugin_values,
        signatures=signatures,
    )
    mm_aggregation = enumerate_exact_pool_values(
        design, channel="behavioral", tier="B", state_values=mm_values,
        signatures=signatures,
    )
    accuracy_aggregation = enumerate_exact_pool_values(
        design, channel="behavioral", tier="B", state_values=accuracy_values,
        signatures=signatures,
    )
    balanced_aggregation = enumerate_exact_pool_values(
        design, channel="behavioral", tier="B", state_values=balanced_values,
        signatures=signatures,
    )
    panel_indices = [panel["fixed_teaching_indices"] for panel in active["panels"]]
    codes = signatures_to_states(signatures, panel_indices)
    achieved_index = int(plugin_aggregation["achieved_prompt_index"])
    selected_accuracy = float(np.mean([
        accuracy_values[panel, codes[achieved_index, panel]]
        for panel in range(len(active["panels"]))
    ]))
    selected_balanced = float(np.mean([
        balanced_values[panel, codes[achieved_index, panel]]
        for panel in range(len(active["panels"]))
    ]))

    null = None
    permutation = None
    if int(n_permutations) > 0:
        rng = np.random.default_rng(_stable_seed(
            certificate["metric_key"], reference_name, "native-v13-null"
        ))
        null = np.empty(int(n_permutations), dtype=float)
        cursor = 0
        while cursor < len(null):
            count = min(250, len(null) - cursor)
            permuted = np.vstack([rng.permutation(target) for _ in range(count)])
            mi = _binary_mi_matrix(permuted, predictions)
            state_values = np.maximum(
                mi[:, rule_indices] - np.maximum(
                    mi[:, blind_index, None, None], mi[:, shuffled_indices]
                ), 0.0,
            )
            prompt_values = np.zeros((count, len(codes)), dtype=float)
            for panel in range(len(active["panels"])):
                prompt_values += state_values[:, panel, :][:, codes[:, panel]]
            null[cursor:cursor + count] = np.max(
                prompt_values / len(active["panels"]), axis=1
            )
            cursor += count
        permutation = permutation_pvalue(plugin_aggregation["achieved_value"], null)
    result = {
        "metric_key": str(certificate["metric_key"]), "task": str(certificate["task"]),
        "reference": reference_name, "heldout_n": len(target),
        "achieved_value_bits": float(plugin_aggregation["achieved_value"]),
        "exact_structural_cap_bits": float(plugin_aggregation["exact_structural_cap"]),
        "exact_structural_gap_bits": float(plugin_aggregation["exact_structural_gap"]),
        "mm_achieved_value_bits": float(mm_aggregation["achieved_value"]),
        "mm_exact_structural_cap_bits": float(mm_aggregation["exact_structural_cap"]),
        "achieved_raw_lift_bits": float(np.mean([
            plugin_raw[panel, codes[achieved_index, panel]]
            for panel in range(len(active["panels"]))
        ])),
        "selected_prompt_accuracy": selected_accuracy,
        "selected_prompt_balanced_accuracy": selected_balanced,
        "exact_structural_accuracy_cap": float(accuracy_aggregation["exact_structural_cap"]),
        "exact_structural_balanced_accuracy_cap": float(
            balanced_aggregation["exact_structural_cap"]
        ),
        "majority_accuracy_baseline": float(max(np.mean(target), 1.0 - np.mean(target))),
        "permutation_percentile": None if permutation is None else permutation["percentile"],
        "permutation_z_score": None if permutation is None else permutation["z_score"],
        "permutation_p_greater_equal": (
            None if permutation is None else permutation["p_greater_equal"]
        ),
        "permutation_null_median": None if permutation is None else permutation["null_median"],
        "cap_is_exact_over_all_4096_patterns_per_pool": True,
        "achievement_is_selected_over_frozen_candidate_population": True,
    }
    return result, null


def audit_native_v13(
    *, metrics_manifest: str | Path, v13_root: str | Path, out_root: str | Path,
    cache_paths: Sequence[str | Path] | None = None,
    n_permutations: int = N_PERMUTATIONS,
) -> pd.DataFrame:
    """CPU-only re-evaluation of native six-demo v13 tables under independent labels."""
    if int(n_permutations) < 200:
        raise ValueError("Phase A requires at least 200 label permutations per reported value")
    ladder_root = Path(out_root).resolve()
    reference = pd.read_parquet(ladder_root / "reference" / "sonnet_labels.parquet")
    independently_valid = set(map(str, reference.metric_key.unique()))
    reference_manifest = json.loads((
        ladder_root / "reference" / "reference_manifest.json"
    ).read_text(encoding="utf-8"))
    manifest, base = load_metrics_manifest(metrics_manifest)
    entries = select_metric_entries(manifest, base)
    contexts = {
        str(row["entry"]["metric_key"]): row for row in _prepare_contexts(entries, base)
    }
    source_root = Path(v13_root).resolve()
    paths = list(map(Path, cache_paths or sorted(source_root.rglob("*.sqlite"))))
    if not paths:
        raise FileNotFoundError(f"no v13 SQLite caches found under {source_root}")
    inductions, executions = _load_v13_cache_rows(paths)
    artifacts = {}
    for certificate_path in source_root.rglob("certificate.json"):
        if certificate_path.parent.name != "behavioral":
            continue
        certificate = json.loads(certificate_path.read_text())
        if (certificate.get("tier") != "B" or certificate.get("constructor", {}).get("model")
                != DEFAULT_CONSTRUCTOR):
            continue
        key = str(certificate["metric_key"])
        prior = artifacts.setdefault(key, certificate_path.parent)
        if prior != certificate_path.parent:
            raise RuntimeError(f"duplicate native v13 behavioral artifact for {key}")
    expected = set(contexts)
    if set(artifacts) != expected:
        raise RuntimeError(
            f"native v13 artifacts incomplete: missing={sorted(expected-set(artifacts))}, "
            f"extra={sorted(set(artifacts)-expected)}"
        )
    rows = []
    null_arrays = {}
    for metric_key in sorted(expected):
        targets = []
        if metric_key in independently_valid:
            targets.append((
                "independent_sonnet_majority",
                _metric_reference(
                    reference, metric_key,
                    contexts[metric_key]["design"]["heldout"]["indices"],
                ),
                int(n_permutations),
            ))
        targets.append((
            "operational_frozen_8b",
            np.asarray(
                contexts[metric_key]["design"]["heldout"]["target_scores"],
                dtype=np.uint8,
            ),
            0,
        ))
        for reference_name, target, permutations in targets:
            row, null = _native_v13_metric_audit(
                artifact_dir=artifacts[metric_key], context=contexts[metric_key],
                target=target, reference_name=reference_name, inductions=inductions,
                executions=executions, n_permutations=permutations,
            )
            rows.append(row)
            if null is not None:
                null_arrays[_safe(metric_key)] = null
    frame = pd.DataFrame(rows).sort_values(["reference", "task", "metric_key"])
    _atomic_parquet(ladder_root / "native_v13_robustness.parquet", frame)
    np.savez_compressed(ladder_root / "native_v13_permutation_nulls.npz", **null_arrays)
    primary = frame[frame.reference == "independent_sonnet_majority"]
    report = {
        "schema": "cr3-native-v13-robustness-v1", "n_metrics": len(primary),
        "n_independent_reference_valid_metrics": len(primary),
        "n_independent_reference_void_metrics": int(
            reference_manifest.get("n_void_metrics", 0)
        ),
        "independent_reference_void_metric_keys": reference_manifest.get(
            "void_metric_keys", []
        ),
        "n_operational_sensitivity_metrics": int(
            (frame.reference == "operational_frozen_8b").sum()
        ),
        "headline_exact_cap_bits_mean": float(primary.exact_structural_cap_bits.mean()),
        "achieved_value_bits_mean": float(primary.achieved_value_bits.mean()),
        "mm_achieved_value_bits_mean": float(primary.mm_achieved_value_bits.mean()),
        "selected_prompt_accuracy_mean": float(primary.selected_prompt_accuracy.mean()),
        "majority_accuracy_baseline_mean": float(primary.majority_accuracy_baseline.mean()),
        "exact_structural_accuracy_cap_mean": float(
            primary.exact_structural_accuracy_cap.mean()
        ),
        "unclipped_lift_tests": aggregate_raw_lift_tests(primary.rename(
            columns={"achieved_raw_lift_bits": "raw_lift_bits"}
        )),
        "n_permutations_per_metric": int(n_permutations),
        "permutation_is_selection_preserving": True,
        "frozen_8b_reference_is_sensitivity_not_ground_truth": True,
    }
    report["sha256"] = _sha(report)
    _atomic_json(ladder_root / "native_v13_robustness_report.json", report)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=(
        "freeze", "reference", "reference-assemble", "constructor", "executor",
        "native-audit", "aggregate",
    ))
    parser.add_argument("--metrics-manifest")
    parser.add_argument("--probe-extension-root")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--metric-keys", nargs="+")
    parser.add_argument("--constructor-model", default=DEFAULT_CONSTRUCTOR)
    parser.add_argument("--v13-root")
    parser.add_argument("--cache-paths", nargs="+")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--fake-backends", action="store_true")
    parser.add_argument("--void-on-reference-failure", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "freeze":
        if not args.metrics_manifest:
            raise ValueError("freeze requires --metrics-manifest")
        freeze_design(
            args.metrics_manifest, args.out_root,
            probe_extension_root=args.probe_extension_root,
        )
    elif args.phase == "reference":
        build_reference(
            args.out_root, metric_keys=args.metric_keys,
            void_on_failure=args.void_on_reference_failure,
        )
    elif args.phase == "reference-assemble":
        assemble_reference(args.out_root)
    elif args.phase == "constructor":
        run_constructor(
            args.out_root, model=args.constructor_model, metric_keys=args.metric_keys,
            fake=args.fake_backends,
        )
    elif args.phase == "executor":
        run_executor(args.out_root, metric_keys=args.metric_keys, fake=args.fake_backends)
    elif args.phase == "native-audit":
        if not args.metrics_manifest or not args.v13_root:
            raise ValueError("native-audit requires --metrics-manifest and --v13-root")
        audit_native_v13(
            metrics_manifest=args.metrics_manifest, v13_root=args.v13_root,
            out_root=args.out_root, cache_paths=args.cache_paths,
            n_permutations=args.n_permutations,
        )
    elif args.phase == "aggregate":
        aggregate_ladder(args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
