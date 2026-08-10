#!/usr/bin/env python3
"""Train a bidirectional three-way or binary Nemotron cross-encoder with LoRA.

The input is pair-level JSONL.  Every row binds one human norm/evidence view to
one metric card and carries an ``EXACT``, ``FAMILY``, or ``REJECT`` label.  The
model reads both concatenation orders, mean-pools native hidden states with the
attention mask, averages the two 4096->3 classifier outputs, and trains only
q/k/v/o LoRA weights plus the separately persisted classifier head.

Training is deliberately exposure-budget based.  Each phase samples globally
at the exact deterministic 25/25/50 class target, works under one process or
``torchrun`` DDP, and freezes an append-only adapter/head checkpoint at the
requested cumulative budget.  A source-disjoint dev set tunes an exact-score
and top-candidate-margin gate using point precision and its Wilson lower bound.
The selected adapter and head are reloaded into a fresh base model and checked
against logits captured before saving.

Model libraries are imported only by the training/reload paths.  Pooling,
sampling, source splitting, and threshold selection are CPU-testable without
loading a Transformers model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .common import normalize_space, read_jsonl, sha256_file


DEFAULT_NEMOTRON = (
    "/lfs/skampere3/0/shared_hf_cache/"
    "models--nvidia--llama-embed-nemotron-8b/snapshots/"
    "aa3b43a495a9b280d1bdb716da37c54bb495d630"
)
HIDDEN_SIZE = 4096
MAX_SEQUENCE_LENGTH = 1024
CLASS_NAMES = ("EXACT", "FAMILY", "REJECT")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_SAMPLING_WEIGHTS = {"EXACT": 0.25, "FAMILY": 0.25, "REJECT": 0.50}
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
REPORT_SCHEMA = "silver-match-v3-nemotron-bidirectional-cross-encoder-v1"
BINARY_CLASS_NAMES = ("NON_EXACT", "EXACT")
BINARY_CLASS_TO_ID = {name: index for index, name in enumerate(BINARY_CLASS_NAMES)}


@dataclass(frozen=True)
class PairExample:
    norm_uid: str
    source_group: str
    metric_id: str
    norm_text: str
    evidence: str
    metric_card: str
    label: str

    @property
    def label_id(self) -> int:
        return CLASS_TO_ID[self.label]

    @property
    def norm_evidence(self) -> str:
        pieces = [f"Human evaluative statement: {self.norm_text}"]
        if self.evidence and self.evidence != self.norm_text:
            pieces.append(f"Evidence passage: {self.evidence}")
        return "\n".join(pieces)


def output_class_names(classification_mode: str) -> tuple[str, ...]:
    if classification_mode == "three_way":
        return CLASS_NAMES
    if classification_mode == "binary":
        return BINARY_CLASS_NAMES
    raise ValueError(f"unknown classification mode: {classification_mode!r}")


def output_label_id(label: str, classification_mode: str) -> int:
    normalized = normalize_class(label)
    if classification_mode == "three_way":
        return CLASS_TO_ID[normalized]
    if classification_mode == "binary":
        return BINARY_CLASS_TO_ID["EXACT" if normalized == "EXACT" else "NON_EXACT"]
    raise ValueError(f"unknown classification mode: {classification_mode!r}")


def sampling_label(label: str, classification_mode: str) -> str:
    names = output_class_names(classification_mode)
    return names[output_label_id(label, classification_mode)]


def sampling_weights(
    classification_mode: str, binary_positive_fraction: float = 0.5
) -> dict[str, float]:
    if classification_mode == "three_way":
        return dict(CLASS_SAMPLING_WEIGHTS)
    if classification_mode != "binary":
        raise ValueError(f"unknown classification mode: {classification_mode!r}")
    if binary_positive_fraction not in {0.25, 0.5}:
        raise ValueError("binary_positive_fraction must be 0.25 or 0.5")
    return {
        "NON_EXACT": 1.0 - binary_positive_fraction,
        "EXACT": binary_positive_fraction,
    }


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def normalize_class(value: Any) -> str:
    """Normalize explicit or numeric three-way labels."""

    if isinstance(value, bool):
        raise ValueError("boolean is not a valid three-way label")
    if isinstance(value, (int, np.integer)):
        if 0 <= int(value) < len(CLASS_NAMES):
            return CLASS_NAMES[int(value)]
    if isinstance(value, float) and value.is_integer():
        return normalize_class(int(value))
    text = normalize_space(value).upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "MATCH": "EXACT",
        "MATCH_EXACT": "EXACT",
        "MATCH_FAMILY": "FAMILY",
        "MATCH_FAMILY_ONLY": "FAMILY",
        "NONMATCH": "REJECT",
        "NO_MATCH": "REJECT",
        "ABSTAIN": "REJECT",
    }
    text = aliases.get(text, text)
    if text not in CLASS_TO_ID:
        raise ValueError(f"unknown three-way label: {value!r}")
    return text


def _metric_card_from_row(row: Mapping[str, Any]) -> str:
    direct = normalize_space(row.get("metric_card"))
    if direct:
        return direct
    metric = row.get("metric") if isinstance(row.get("metric"), Mapping) else row
    name = normalize_space(metric.get("name") or metric.get("metric_name"))
    description = normalize_space(
        metric.get("description")
        or metric.get("definition")
        or metric.get("metric_description")
    )
    examples = metric.get("examples") or []
    if isinstance(examples, str):
        examples = [examples]
    examples = [normalize_space(value) for value in examples if normalize_space(value)]
    if not name and not description:
        raise ValueError("pair row lacks metric_card or metric name/definition")
    card = name
    if description:
        card += (". " if card else "") + f"Definition: {description}"
    if examples:
        card += (". " if card else "") + "Examples: " + "; ".join(examples[:4])
    return normalize_space(card)


def pair_example_from_row(row: Mapping[str, Any], *, source: str = "<memory>") -> PairExample:
    uid = normalize_space(row.get("norm_uid") or row.get("uid"))
    group = normalize_space(row.get("source_group") or row.get("split_group"))
    metric_id = normalize_space(row.get("candidate_metric_id") or row.get("metric_id"))
    norm_text = normalize_space(
        row.get("norm")
        or row.get("statement")
        or row.get("human_statement")
        or row.get("query")
    )
    evidence = normalize_space(row.get("evidence") or row.get("context"))
    raw_label = next(
        (
            row[key]
            # ``build_nemotron_ce_pairs`` names the audited pair-level target
            # ``relation`` while retaining the original norm-level decision
            # for provenance.  Pair-level fields must always win; otherwise a
            # REJECT candidate attached to a MATCH norm is misread as MATCH.
            for key in ("relation", "ce_label", "target", "class_label", "label")
            if row.get(key) is not None
        ),
        None,
    )
    if raw_label is None and row.get("decision") is not None:
        raw_label = row.get("decision")
    missing = [
        name
        for name, value in (
            ("norm_uid", uid),
            ("source_group", group),
            ("metric_id", metric_id),
            ("norm/statement/query", norm_text),
        )
        if not value
    ]
    if missing or raw_label is None:
        raise ValueError(f"incomplete pair row in {source}: missing {missing or ['label']}")
    return PairExample(
        norm_uid=uid,
        source_group=group,
        metric_id=metric_id,
        norm_text=norm_text,
        evidence=evidence,
        metric_card=_metric_card_from_row(row),
        label=normalize_class(raw_label),
    )


def load_pair_examples(paths: Sequence[Path]) -> list[PairExample]:
    rows: list[PairExample] = []
    seen: set[tuple[str, str]] = set()
    uid_groups: dict[str, str] = {}
    for path in paths:
        for line_no, row in enumerate(read_jsonl(path), 1):
            value = pair_example_from_row(row, source=f"{path}:{line_no}")
            key = (value.norm_uid, value.metric_id)
            if key in seen:
                raise ValueError(f"duplicate norm/metric pair across inputs: {key}")
            previous = uid_groups.setdefault(value.norm_uid, value.source_group)
            if previous != value.source_group:
                raise ValueError(
                    f"norm UID crosses source groups: {value.norm_uid}: "
                    f"{previous!r} vs {value.source_group!r}"
                )
            seen.add(key)
            rows.append(value)
    if not rows:
        raise ValueError("pair inputs are empty")
    return rows


def source_group_is_dev(source_group: str, *, seed: int, dev_fraction: float) -> bool:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")
    bucket = _stable_seed(seed, "source-dev", source_group) / float(2**64 - 1)
    return bucket < dev_fraction


def deterministic_source_split(
    examples: Sequence[PairExample], *, seed: int, dev_fraction: float
) -> tuple[list[PairExample], list[PairExample], dict[str, Any]]:
    group_roles = {
        group: ("dev" if source_group_is_dev(group, seed=seed, dev_fraction=dev_fraction) else "train")
        for group in sorted({row.source_group for row in examples})
    }
    train = [row for row in examples if group_roles[row.source_group] == "train"]
    dev = [row for row in examples if group_roles[row.source_group] == "dev"]
    if not train or not dev:
        raise ValueError(
            "deterministic source split produced an empty role; provide explicit "
            "--dev-pairs or a larger source universe"
        )
    return train, dev, source_split_audit(train, dev, mode="deterministic_hash")


def source_split_audit(
    train: Sequence[PairExample], dev: Sequence[PairExample], *, mode: str
) -> dict[str, Any]:
    train_groups = {row.source_group for row in train}
    dev_groups = {row.source_group for row in dev}
    overlap = sorted(train_groups & dev_groups)
    if overlap:
        raise ValueError(f"train/dev source-group leakage: {overlap[:10]}")
    return {
        "complete": True,
        "mode": mode,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_norm_uids": len({row.norm_uid for row in train}),
        "dev_norm_uids": len({row.norm_uid for row in dev}),
        "train_source_groups": len(train_groups),
        "dev_source_groups": len(dev_groups),
        "source_group_overlap_count": 0,
        "train_class_counts": dict(sorted(Counter(row.label for row in train).items())),
        "dev_class_counts": dict(sorted(Counter(row.label for row in dev).items())),
    }


def class_quotas(
    num_samples: int,
    weights: Mapping[str, float] = CLASS_SAMPLING_WEIGHTS,
) -> dict[str, int]:
    """Allocate an exact sample budget by stable largest remainder."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    names = tuple(weights)
    if not names or set(names) not in (set(CLASS_NAMES), set(BINARY_CLASS_NAMES)):
        raise ValueError(
            f"weights must cover exactly {CLASS_NAMES} or {BINARY_CLASS_NAMES}"
        )
    total = sum(float(weights[name]) for name in names)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("class weights must sum to one")
    raw = {name: num_samples * float(weights[name]) for name in names}
    quotas = {name: int(math.floor(raw[name])) for name in names}
    remaining = num_samples - sum(quotas.values())
    priority = sorted(
        names,
        key=lambda name: (-(raw[name] - quotas[name]), names.index(name)),
    )
    for name in priority[:remaining]:
        quotas[name] += 1
    return quotas


def _draw_with_replacement(
    values: Sequence[int], count: int, *, seed: int, label: str
) -> list[int]:
    if count == 0:
        return []
    if not values:
        raise ValueError(f"cannot sample class {label}: no source rows")
    output: list[int] = []
    cycle = 0
    while len(output) < count:
        current = list(values)
        random.Random(_stable_seed(seed, label, cycle)).shuffle(current)
        output.extend(current[: count - len(output)])
        cycle += 1
    return output


def deterministic_weighted_indices(
    labels: Sequence[Any],
    *,
    num_samples: int,
    seed: int,
    epoch: int = 0,
    weights: Mapping[str, float] = CLASS_SAMPLING_WEIGHTS,
) -> list[int]:
    """Return a deterministic exact-quota global sample order."""

    names = tuple(weights)
    buckets: dict[str, list[int]] = {name: [] for name in names}
    for index, value in enumerate(labels):
        normalized = normalize_class(value) if set(names) == set(CLASS_NAMES) else str(value)
        if normalized not in buckets:
            raise ValueError(f"label {normalized!r} is not represented by sampler weights")
        buckets[normalized].append(index)
    quotas = class_quotas(num_samples, weights)
    phase_seed = _stable_seed(seed, "weighted-sampler", epoch)
    by_class = {
        name: _draw_with_replacement(
            buckets[name], quotas[name], seed=phase_seed, label=name
        )
        for name in names
    }
    class_order = [name for name in names for _ in range(quotas[name])]
    random.Random(_stable_seed(phase_seed, "class-order")).shuffle(class_order)
    offsets = Counter()
    output = []
    for name in class_order:
        output.append(by_class[name][offsets[name]])
        offsets[name] += 1
    return output


class DeterministicWeightedSampler(Sampler[int]):
    """Exact 25/25/50 sampler with deterministic torchrun rank sharding."""

    def __init__(
        self,
        labels: Sequence[Any],
        *,
        num_samples: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        weights: Mapping[str, float] = CLASS_SAMPLING_WEIGHTS,
    ) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid rank/world_size")
        if num_samples % world_size:
            raise ValueError("global sample budget must be divisible by world_size")
        names = tuple(weights)
        self.labels = tuple(
            normalize_class(value) if set(names) == set(CLASS_NAMES) else str(value)
            for value in labels
        )
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.weights = dict(weights)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        global_order = deterministic_weighted_indices(
            self.labels,
            num_samples=self.num_samples,
            seed=self.seed,
            epoch=self.epoch,
            weights=self.weights,
        )
        return iter(global_order[self.rank :: self.world_size])

    def __len__(self) -> int:
        return self.num_samples // self.world_size


def attention_mask_mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool native token states, excluding padding exactly by mask."""

    if last_hidden_state.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("expected hidden [batch, tokens, hidden] and mask [batch, tokens]")
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("hidden state and attention mask shapes do not align")
    mask = attention_mask.to(device=last_hidden_state.device, dtype=last_hidden_state.dtype)
    denominator = mask.sum(dim=1, keepdim=True)
    if bool(torch.any(denominator == 0).item()):
        raise ValueError("cannot pool an all-padding sequence")
    return (last_hidden_state * mask.unsqueeze(-1)).sum(dim=1) / denominator


class BidirectionalNemotronCrossEncoder(nn.Module):
    """Average logits from norm->card and card->norm concatenation orders."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int = HIDDEN_SIZE,
        num_classes: int = len(CLASS_NAMES),
    ) -> None:
        super().__init__()
        if hidden_size != HIDDEN_SIZE:
            raise ValueError(f"classifier input is frozen at {HIDDEN_SIZE}, got {hidden_size}")
        if num_classes not in {2, 3}:
            raise ValueError("classifier output must have two or three classes")
        self.backbone = backbone
        self.num_classes = num_classes
        self.head = nn.Linear(HIDDEN_SIZE, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.ndim != 3 or input_ids.shape[1] != 2:
            raise ValueError("input_ids must be [batch, 2 directions, tokens]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        batch, directions, tokens = input_ids.shape
        flat_ids = input_ids.reshape(batch * directions, tokens)
        flat_mask = attention_mask.reshape(batch * directions, tokens)
        outputs = self.backbone(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        if hidden.shape[-1] != HIDDEN_SIZE:
            raise RuntimeError(
                f"llama-embed-nemotron hidden size drift: {hidden.shape[-1]} != {HIDDEN_SIZE}"
            )
        pooled = attention_mask_mean_pool(hidden, flat_mask)
        logits = self.head(pooled.to(self.head.weight.dtype))
        return logits.reshape(batch, directions, self.num_classes).mean(dim=1).float()


class PairDataset(Dataset[PairExample]):
    def __init__(self, examples: Sequence[PairExample]) -> None:
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PairExample:
        return self.examples[index]


def bidirectional_collate(
    tokenizer: Any, *, max_length: int, classification_mode: str = "three_way"
):
    if not 1 <= max_length <= MAX_SEQUENCE_LENGTH:
        raise ValueError(f"max_length must be in [1, {MAX_SEQUENCE_LENGTH}]")

    def collate(examples: Sequence[PairExample]) -> dict[str, Any]:
        first: list[str] = []
        second: list[str] = []
        for row in examples:
            first.extend((row.norm_evidence, row.metric_card))
            second.extend((row.metric_card, row.norm_evidence))
        encoded = tokenizer(
            first,
            text_pair=second,
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
        batch = len(examples)
        return {
            "input_ids": encoded["input_ids"].reshape(batch, 2, -1),
            "attention_mask": encoded["attention_mask"].reshape(batch, 2, -1),
            "labels": torch.tensor(
                [output_label_id(row.label, classification_mode) for row in examples],
                dtype=torch.long,
            ),
        }

    return collate


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float] | None:
    if total <= 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if not len(values):
        return None
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.5, 0.9, 1.0)
    }


def grouped_exact_gate_report(
    examples: Sequence[PairExample],
    probabilities: np.ndarray,
    score_threshold: float,
    margin_threshold: float,
    *,
    classification_mode: str = "three_way",
) -> dict[str, Any]:
    """Score exact matches at norm level.

    Three-way mode retains at most one top metric. Binary mode independently
    thresholds every candidate, so a norm may yield zero, one, or many metrics.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    names = output_class_names(classification_mode)
    exact_index = names.index("EXACT")
    if probabilities.shape != (len(examples), len(names)):
        raise ValueError(f"probabilities must have shape [examples, {len(names)}]")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain non-finite values")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        groups[row.norm_uid].append(index)
    if not groups:
        raise ValueError("cannot score an empty dev set")

    tp = fp = fn = tn = 0
    top_scores: list[float] = []
    margins: list[float] = []
    predicted_count = 0
    exact_gold_groups = 0
    abstained_groups = 0
    for uid in sorted(groups):
        indices = groups[uid]
        gold_exact = [index for index in indices if examples[index].label == "EXACT"]
        if classification_mode == "three_way" and len(gold_exact) > 1:
            raise ValueError(f"dev norm has multiple EXACT candidates: {uid}")
        exact_gold_groups += bool(gold_exact)
        ranked = sorted(
            indices,
            key=lambda index: (
                -float(probabilities[index, exact_index]),
                examples[index].metric_id,
                index,
            ),
        )
        top = ranked[0]
        top_score = float(probabilities[top, exact_index])
        second_score = (
            float(probabilities[ranked[1], exact_index])
            if len(ranked) > 1
            else 0.0
        )
        margin = top_score - second_score
        top_scores.append(top_score)
        margins.append(margin)
        if classification_mode == "binary":
            predicted = {
                index
                for index in indices
                if float(probabilities[index, exact_index]) >= score_threshold
            }
            gold = set(gold_exact)
            current_tp = len(predicted & gold)
            current_fp = len(predicted - gold)
            current_fn = len(gold - predicted)
            current_tn = len(set(indices) - predicted - gold)
            tp += current_tp
            fp += current_fp
            fn += current_fn
            tn += current_tn
            predicted_count += len(predicted)
            abstained_groups += int(not predicted)
            continue
        top_class_is_exact = int(np.argmax(probabilities[top])) == exact_index
        predicts_exact = (
            top_class_is_exact
            and top_score >= score_threshold
            and margin >= margin_threshold
        )
        success = predicts_exact and examples[top].label == "EXACT"
        predicted_count += int(predicts_exact)
        tp += int(success)
        fp += int(predicts_exact and not success)
        fn += int(bool(gold_exact) and not success)
        tn += int(not gold_exact and not predicts_exact)
        abstained_groups += int(not predicts_exact)

    precision = tp / predicted_count if predicted_count else 1.0
    gold_exact_pairs = tp + fn
    recall = tp / gold_exact_pairs if gold_exact_pairs else 0.0
    beta2 = 0.25
    f_beta = (
        (1.0 + beta2) * precision * recall / (beta2 * precision + recall)
        if precision + recall
        else 0.0
    )
    row_predictions = np.argmax(probabilities, axis=1)
    row_gold = np.asarray(
        [output_label_id(row.label, classification_mode) for row in examples],
        dtype=np.int64,
    )
    interval = wilson_interval(tp, predicted_count)
    return {
        "norm_groups": len(groups),
        "gold_exact_groups": exact_gold_groups,
        "gold_exact_pairs": gold_exact_pairs,
        "score_threshold": float(score_threshold),
        "top_margin_threshold": float(margin_threshold),
        "predicted_exact_count": predicted_count,
        "abstained_norm_groups": abstained_groups,
        "abstention_rate": abstained_groups / len(groups),
        "set_valued_predictions": classification_mode == "binary",
        "exact_precision": precision,
        "exact_precision_wilson_95": interval,
        "exact_precision_wilson_95_lower": interval[0] if interval else None,
        "exact_recall": recall,
        "exact_f_beta_0_5": f_beta,
        "pair_classification_accuracy": float(np.mean(row_predictions == row_gold)),
        "pair_three_way_accuracy": (
            float(np.mean(row_predictions == row_gold))
            if classification_mode == "three_way"
            else None
        ),
        "classification_mode": classification_mode,
        "top_exact_score_quantiles": _quantiles(np.asarray(top_scores)),
        "top_candidate_margin_quantiles": _quantiles(np.asarray(margins)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _grid(values: np.ndarray, size: int, *, include_one: bool) -> list[float]:
    if size < 2:
        raise ValueError("threshold grid size must be at least two")
    anchors = [0.0, *np.quantile(values, np.linspace(0.0, 1.0, size)).tolist()]
    if include_one:
        anchors.append(1.0)
    return sorted({float(value) for value in anchors})


def tune_dev_thresholds(
    examples: Sequence[PairExample],
    probabilities: np.ndarray,
    *,
    min_exact_precision: float,
    min_wilson_lower: float,
    min_exact_predictions: int,
    score_grid_size: int = 41,
    margin_grid_size: int = 31,
    classification_mode: str = "three_way",
) -> dict[str, Any]:
    """Tune deterministic exact-score/top-margin thresholds on grouped dev pairs."""

    if not 0.0 <= min_exact_precision <= 1.0:
        raise ValueError("min_exact_precision must be in [0, 1]")
    if not 0.0 <= min_wilson_lower <= 1.0:
        raise ValueError("min_wilson_lower must be in [0, 1]")
    if min_exact_predictions < 0:
        raise ValueError("min_exact_predictions must be non-negative")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    names = output_class_names(classification_mode)
    exact_index = names.index("EXACT")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        groups[row.norm_uid].append(index)
    top_scores = []
    margins = []
    for uid in sorted(groups):
        values = sorted(
            (
                float(probabilities[index, exact_index]),
                examples[index].metric_id,
            )
            for index in groups[uid]
        )
        descending = sorted(values, key=lambda value: (-value[0], value[1]))
        if classification_mode == "binary":
            top_scores.extend(value[0] for value in descending)
        else:
            top_scores.append(descending[0][0])
        margins.append(descending[0][0] - (descending[1][0] if len(descending) > 1 else 0.0))
    margin_grid = (
        [0.0]
        if classification_mode == "binary"
        else _grid(np.asarray(margins), margin_grid_size, include_one=False)
    )
    reports = [
        grouped_exact_gate_report(
            examples,
            probabilities,
            score,
            margin,
            classification_mode=classification_mode,
        )
        for score in _grid(np.asarray(top_scores), score_grid_size, include_one=True)
        for margin in margin_grid
    ]
    feasible = [
        row
        for row in reports
        if row["predicted_exact_count"] >= min_exact_predictions
        and row["exact_precision"] >= min_exact_precision
        and row["exact_precision_wilson_95_lower"] is not None
        and row["exact_precision_wilson_95_lower"] >= min_wilson_lower
    ]

    def selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
        lower = row["exact_precision_wilson_95_lower"]
        return (
            float(row["exact_f_beta_0_5"]),
            float(row["exact_recall"]),
            float(lower if lower is not None else -1.0),
            float(row["exact_precision"]),
            float(row["predicted_exact_count"]),
            float(row["pair_classification_accuracy"]),
            -float(row["score_threshold"]),
            -float(row["top_margin_threshold"]),
        )

    if feasible:
        best = max(feasible, key=selection_key)
    else:
        best = max(
            reports,
            key=lambda row: (
                float(
                    row["exact_precision_wilson_95_lower"]
                    if row["exact_precision_wilson_95_lower"] is not None
                    else -1.0
                ),
                float(row["exact_precision"]),
                *selection_key(row),
            ),
        )
    return {
        **best,
        "precision_wilson_gate_met": bool(feasible),
        "minimum_exact_precision": float(min_exact_precision),
        "minimum_wilson_lower": float(min_wilson_lower),
        "minimum_exact_predictions": int(min_exact_predictions),
        "threshold_candidates_evaluated": len(reports),
    }


def checkpoint_selection_key(report: Mapping[str, Any]) -> tuple[float, ...]:
    lower = report.get("exact_precision_wilson_95_lower")
    return (
        float(bool(report.get("precision_wilson_gate_met"))),
        float(report.get("exact_f_beta_0_5", 0.0)),
        float(lower if lower is not None else -1.0),
        float(report.get("exact_precision", 0.0)),
        float(report.get("exact_recall", 0.0)),
        float(report.get("predicted_exact_count", 0)),
    )


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    if not torch.cuda.is_available():
        raise RuntimeError("Nemotron cross-encoder training requires CUDA bf16")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected CUDA device does not support bfloat16")
    return DistributedContext(rank, local_rank, world_size, world_size > 1, device)


def _barrier(context: DistributedContext) -> None:
    if context.distributed:
        import torch.distributed as dist

        dist.barrier()


def _set_determinism(seed: int, rank: int) -> None:
    derived = seed + rank
    random.seed(derived)
    np.random.seed(derived)
    torch.manual_seed(derived)
    torch.cuda.manual_seed_all(derived)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _hidden_size(config: Any) -> int:
    for value in (config, getattr(config, "text_config", None)):
        if value is None:
            continue
        for name in ("hidden_size", "d_model"):
            observed = getattr(value, name, None)
            if observed is not None:
                return int(observed)
    raise ValueError("base model config does not expose hidden_size")


def _load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Nemotron tokenizer has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_base_backbone(model_name: str, attention: str, device: torch.device) -> nn.Module:
    from transformers import AutoModel

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if attention != "auto":
        kwargs["attn_implementation"] = attention
    backbone = AutoModel.from_pretrained(model_name, **kwargs)
    if _hidden_size(backbone.config) != HIDDEN_SIZE:
        raise ValueError(
            f"expected llama-embed-nemotron hidden size {HIDDEN_SIZE}, "
            f"got {_hidden_size(backbone.config)}"
        )
    if hasattr(backbone, "config"):
        backbone.config.use_cache = False
    return backbone.to(device)


def _build_trainable_model(args: argparse.Namespace, device: torch.device):
    from peft import LoraConfig, TaskType, get_peft_model

    backbone = _load_base_backbone(args.model, args.attention, device)
    if args.gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        try:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(LORA_TARGETS),
        bias="none",
    )
    peft_backbone = get_peft_model(backbone, config)
    model = BidirectionalNemotronCrossEncoder(
        peft_backbone,
        num_classes=len(output_class_names(args.classification_mode)),
    ).to(device)
    backbone_non_lora = [
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    lora_parameters = [
        parameter
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    if backbone_non_lora or not lora_parameters:
        raise RuntimeError(
            f"LoRA isolation failed: non_lora={backbone_non_lora[:20]}, "
            f"lora_parameters={len(lora_parameters)}"
        )
    return model, lora_parameters, list(model.head.parameters())


def _windowed(loader: Iterable[Any], size: int) -> Iterator[list[Any]]:
    window: list[Any] = []
    for batch in loader:
        window.append(batch)
        if len(window) == size:
            yield window
            window = []
    if window:
        yield window


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _unwrap(model: nn.Module) -> BidirectionalNemotronCrossEncoder:
    return getattr(model, "module", model)


def predict_logits(
    model: nn.Module,
    examples: Sequence[PairExample],
    tokenizer: Any,
    *,
    device: torch.device,
    max_length: int,
    batch_size: int,
    classification_mode: str = "three_way",
) -> np.ndarray:
    loader = DataLoader(
        PairDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=bidirectional_collate(
            tokenizer,
            max_length=max_length,
            classification_mode=classification_mode,
        ),
    )
    core = _unwrap(model)
    core.eval()
    output = []
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = core(batch["input_ids"], batch["attention_mask"])
            output.append(logits.detach().float().cpu().numpy())
    return (
        np.concatenate(output, axis=0)
        if output
        else np.empty((0, len(output_class_names(classification_mode))), dtype=np.float32)
    )


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _exclusive_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, line.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _save_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    *,
    exposure_budget: int,
    dev_report: Mapping[str, Any],
    mean_loss: float,
    optimizer_updates: int,
    cumulative_class_exposures: Mapping[str, int],
    reference_examples: Sequence[PairExample],
    reference_logits: np.ndarray,
    classification_mode: str,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    core = _unwrap(model)
    adapter_dir = checkpoint_dir / "adapter"
    core.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    head_path = checkpoint_dir / "head.safetensors"
    save_file(
        {
            name: tensor.detach().float().cpu().contiguous()
            for name, tensor in core.head.state_dict().items()
        },
        head_path,
    )
    metadata = {
        "schema_version": "silver-match-v3-nemotron-ce-checkpoint-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exposure_budget": exposure_budget,
        "mean_cross_entropy": mean_loss,
        "optimizer_updates": optimizer_updates,
        "cumulative_class_exposures": dict(cumulative_class_exposures),
        "classification_mode": classification_mode,
        "labels": list(output_class_names(classification_mode)),
        "hidden_to_classes": [HIDDEN_SIZE, len(output_class_names(classification_mode))],
        "lora_targets": list(LORA_TARGETS),
        "dev": dict(dev_report),
        "reload_reference": [
            {
                "norm_uid": row.norm_uid,
                "metric_id": row.metric_id,
                "logits": [float(value) for value in reference_logits[index]],
            }
            for index, row in enumerate(reference_examples)
        ],
    }
    meta_path = checkpoint_dir / "checkpoint.json"
    _exclusive_json(meta_path, metadata)
    hashes = _file_hashes(checkpoint_dir)
    return {
        "path": str(checkpoint_dir),
        "exposure_budget": exposure_budget,
        "dev": dict(dev_report),
        "artifact_sha256": hashes,
        "checkpoint_metadata_sha256": sha256_file(meta_path),
    }


def _load_saved_model(
    args: argparse.Namespace, checkpoint_dir: Path, device: torch.device
) -> BidirectionalNemotronCrossEncoder:
    from peft import PeftModel
    from safetensors.torch import load_file

    backbone = _load_base_backbone(args.model, args.attention, device)
    backbone = PeftModel.from_pretrained(
        backbone, checkpoint_dir / "adapter", is_trainable=False
    )
    model = BidirectionalNemotronCrossEncoder(
        backbone,
        num_classes=len(output_class_names(args.classification_mode)),
    ).to(device)
    state = load_file(checkpoint_dir / "head.safetensors", device="cpu")
    missing, unexpected = model.head.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"head reload mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


def _linear_schedule(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup = int(math.ceil(total_steps * warmup_ratio))

    def scale(step: int) -> float:
        if warmup and step < warmup:
            return max(step, 1) / warmup
        remaining = total_steps - step
        denominator = max(total_steps - warmup, 1)
        return max(0.0, remaining / denominator)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale), warmup


def _phase_update_steps(
    delta: int, context: DistributedContext, batch_size: int, accumulation: int
) -> int:
    local_rows = delta // context.world_size
    batches = math.ceil(local_rows / batch_size)
    return math.ceil(batches / accumulation)


def _train_phase(
    model: nn.Module,
    train_examples: Sequence[PairExample],
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    context: DistributedContext,
    phase_samples: int,
    phase_index: int,
    seed: int,
    batch_size: int,
    accumulation: int,
    max_length: int,
    max_grad_norm: float,
    classification_mode: str,
    sampler_weights: Mapping[str, float],
) -> tuple[float, int]:
    sampler = DeterministicWeightedSampler(
        [sampling_label(row.label, classification_mode) for row in train_examples],
        num_samples=phase_samples,
        seed=seed,
        rank=context.rank,
        world_size=context.world_size,
        weights=sampler_weights,
    )
    sampler.set_epoch(phase_index)
    loader = DataLoader(
        PairDataset(train_examples),
        sampler=sampler,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=bidirectional_collate(
            tokenizer,
            max_length=max_length,
            classification_mode=classification_mode,
        ),
    )
    model.train()
    loss_sum = 0.0
    row_count = 0
    updates = 0
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for window in _windowed(loader, accumulation):
        optimizer.zero_grad(set_to_none=True)
        for ordinal, cpu_batch in enumerate(window):
            batch = _move_batch(cpu_batch, context.device)
            synchronize = ordinal == len(window) - 1
            sync_context = (
                nullcontext()
                if synchronize or not context.distributed
                else model.no_sync()  # type: ignore[attr-defined]
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(batch["input_ids"], batch["attention_mask"])
                    loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
                (loss / len(window)).backward()
            observed = int(batch["labels"].shape[0])
            loss_sum += float(loss.detach().cpu()) * observed
            row_count += observed
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        scheduler.step()
        updates += 1
    totals = torch.tensor([loss_sum, float(row_count)], device=context.device)
    if context.distributed:
        import torch.distributed as dist

        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    global_rows = int(totals[1].item())
    if global_rows != phase_samples:
        raise RuntimeError(
            f"exposure accounting drift: observed {global_rows}, expected {phase_samples}"
        )
    return float(totals[0].item() / max(global_rows, 1)), updates


def _parse_budgets(values: Sequence[int]) -> list[int]:
    budgets = sorted(set(int(value) for value in values))
    if not budgets or budgets[0] <= 0:
        raise ValueError("at least one positive exposure budget is required")
    return budgets


def train(args: argparse.Namespace) -> dict[str, Any] | None:
    os.environ.setdefault(
        "HF_MODULES_CACHE", str(Path.home() / ".cache" / "huggingface" / "modules")
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    context = initialize_distributed()
    _set_determinism(args.seed, context.rank)
    output = Path(args.output).resolve()
    event_path = output / "events.jsonl"
    try:
        train_paths = [Path(value).resolve() for value in args.train_pairs]
        dev_paths = [Path(value).resolve() for value in args.dev_pairs]
        all_train = load_pair_examples(train_paths)
        if dev_paths:
            train_examples = all_train
            dev_examples = load_pair_examples(dev_paths)
            split_audit = source_split_audit(
                train_examples, dev_examples, mode="explicit_files"
            )
        else:
            train_examples, dev_examples, split_audit = deterministic_source_split(
                all_train, seed=args.split_seed, dev_fraction=args.dev_fraction
            )
        class_names = output_class_names(args.classification_mode)
        sampler_weights = sampling_weights(
            args.classification_mode, args.binary_positive_fraction
        )
        missing_classes = set(class_names) - {
            sampling_label(row.label, args.classification_mode)
            for row in train_examples
        }
        if missing_classes:
            raise ValueError(
                f"weighted training requires all output classes; missing {sorted(missing_classes)}"
            )
        budgets = _parse_budgets(args.exposure_budget)
        if any(value % context.world_size for value in budgets):
            raise ValueError("every cumulative exposure budget must be divisible by world_size")
        output_conflict = bool(
            context.is_main and output.exists() and any(output.iterdir())
        )
        if context.distributed:
            import torch.distributed as dist

            value = [output_conflict]
            dist.broadcast_object_list(value, src=0)
            output_conflict = bool(value[0])
        if output_conflict:
            raise FileExistsError(f"refusing to reuse non-empty output: {output}")
        if context.is_main:
            output.mkdir(parents=True, exist_ok=True)
            split_path = output / "split_assignments.jsonl"
            _exclusive_jsonl(
                split_path,
                (
                    {
                        "norm_uid": row.norm_uid,
                        "source_group": row.source_group,
                        "metric_id": row.metric_id,
                        "label": row.label,
                        "split": split,
                    }
                    for split, values in (
                        ("train", train_examples),
                        ("dev", dev_examples),
                    )
                    for row in sorted(
                        values, key=lambda value: (value.norm_uid, value.metric_id)
                    )
                ),
            )
            run_config = {
                "schema_version": REPORT_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "train_pairs": {str(path): sha256_file(path) for path in train_paths},
                "dev_pairs": {str(path): sha256_file(path) for path in dev_paths},
                "split_seed": args.split_seed,
                "dev_fraction": args.dev_fraction,
                "seed": args.seed,
                "max_length": args.max_length,
                "exposure_budgets": budgets,
                "classification_mode": args.classification_mode,
                "sampler_weights": sampler_weights,
                "batch_size_per_rank": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "world_size": context.world_size,
                "lora_learning_rate": args.lora_learning_rate,
                "head_learning_rate": args.head_learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lora": {
                    "rank": args.lora_rank,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "targets": list(LORA_TARGETS),
                },
                "classifier": [HIDDEN_SIZE, len(class_names)],
                "labels": list(class_names),
                "attention": args.attention,
                "bf16_cuda": True,
                "bidirectional_concatenation": True,
                "pooling": "native_attention_mask_mean",
                "split_assignments_sha256": sha256_file(split_path),
                "dev_gate": {
                    "minimum_exact_precision": args.min_exact_precision,
                    "minimum_wilson_lower": args.min_wilson_lower,
                    "minimum_exact_predictions": args.min_exact_predictions,
                },
                "split_audit": split_audit,
            }
            _exclusive_json(output / "run_config.json", run_config)
            _append_event(
                event_path,
                {
                    "event": "RUN_STARTED",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "world_size": context.world_size,
                },
            )
        _barrier(context)

        tokenizer = _load_tokenizer(args.model)
        core, lora_parameters, head_parameters = _build_trainable_model(
            args, context.device
        )
        trainable_parameters = sum(
            parameter.numel() for parameter in [*lora_parameters, *head_parameters]
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": lora_parameters,
                    "lr": args.lora_learning_rate,
                    "name": "lora",
                },
                {
                    "params": head_parameters,
                    "lr": args.head_learning_rate,
                    "name": "head",
                },
            ],
            weight_decay=args.weight_decay,
        )
        deltas = [budgets[0], *(right - left for left, right in zip(budgets, budgets[1:]))]
        total_updates = sum(
            _phase_update_steps(
                delta,
                context,
                args.batch_size,
                args.gradient_accumulation_steps,
            )
            for delta in deltas
        )
        scheduler, warmup_steps = _linear_schedule(
            optimizer, total_updates, args.warmup_ratio
        )
        model: nn.Module = core
        if context.distributed:
            from torch.nn.parallel import DistributedDataParallel

            model = DistributedDataParallel(
                core,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                find_unused_parameters=False,
            )

        checkpoint_reports: list[dict[str, Any]] = []
        cumulative = Counter({name: 0 for name in class_names})
        optimizer_updates = 0
        reference_examples = sorted(
            dev_examples, key=lambda row: (row.norm_uid, row.metric_id)
        )[: args.reload_verify_examples]
        previous_budget = 0
        for phase_index, budget in enumerate(budgets):
            delta = budget - previous_budget
            mean_loss, phase_updates = _train_phase(
                model,
                train_examples,
                tokenizer,
                optimizer,
                scheduler,
                context=context,
                phase_samples=delta,
                phase_index=phase_index,
                seed=args.seed,
                batch_size=args.batch_size,
                accumulation=args.gradient_accumulation_steps,
                max_length=args.max_length,
                max_grad_norm=args.max_grad_norm,
                classification_mode=args.classification_mode,
                sampler_weights=sampler_weights,
            )
            optimizer_updates += phase_updates
            cumulative.update(class_quotas(delta, sampler_weights))
            _barrier(context)
            if context.is_main:
                logits = predict_logits(
                    model,
                    dev_examples,
                    tokenizer,
                    device=context.device,
                    max_length=args.max_length,
                    batch_size=args.eval_batch_size,
                    classification_mode=args.classification_mode,
                )
                probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
                dev_report = tune_dev_thresholds(
                    dev_examples,
                    probabilities,
                    min_exact_precision=args.min_exact_precision,
                    min_wilson_lower=args.min_wilson_lower,
                    min_exact_predictions=args.min_exact_predictions,
                    classification_mode=args.classification_mode,
                )
                reference_logits = predict_logits(
                    model,
                    reference_examples,
                    tokenizer,
                    device=context.device,
                    max_length=args.max_length,
                    batch_size=args.eval_batch_size,
                    classification_mode=args.classification_mode,
                )
                checkpoint = _save_checkpoint(
                    model,
                    output / "checkpoints" / f"exposure-{budget:012d}",
                    exposure_budget=budget,
                    dev_report=dev_report,
                    mean_loss=mean_loss,
                    optimizer_updates=optimizer_updates,
                    cumulative_class_exposures=cumulative,
                    reference_examples=reference_examples,
                    reference_logits=reference_logits,
                    classification_mode=args.classification_mode,
                )
                checkpoint_reports.append(checkpoint)
                _append_event(
                    event_path,
                    {
                        "event": "CHECKPOINT_SAVED",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "exposure_budget": budget,
                        "path": checkpoint["path"],
                        "checkpoint_metadata_sha256": checkpoint[
                            "checkpoint_metadata_sha256"
                        ],
                    },
                )
            _barrier(context)
            previous_budget = budget

        selected: dict[str, Any] | None = None
        if context.is_main:
            selected = max(
                checkpoint_reports,
                key=lambda row: (
                    *checkpoint_selection_key(row["dev"]),
                    -int(row["exposure_budget"]),
                ),
            )
        _barrier(context)
        # Release the training graph before the mandatory fresh-base reload.
        del model, core, optimizer, scheduler
        torch.cuda.empty_cache()
        _barrier(context)

        if context.is_main:
            assert selected is not None
            selected_dir = Path(selected["path"])
            metadata = json.loads(
                (selected_dir / "checkpoint.json").read_text(encoding="utf-8")
            )
            reloaded = _load_saved_model(args, selected_dir, context.device)
            reloaded_logits = predict_logits(
                reloaded,
                reference_examples,
                tokenizer,
                device=context.device,
                max_length=args.max_length,
                batch_size=args.eval_batch_size,
                classification_mode=args.classification_mode,
            )
            expected_logits = np.asarray(
                [row["logits"] for row in metadata["reload_reference"]],
                dtype=np.float32,
            )
            maximum_error = float(np.max(np.abs(reloaded_logits - expected_logits)))
            reload_passed = bool(
                np.allclose(
                    reloaded_logits,
                    expected_logits,
                    atol=args.reload_atol,
                    rtol=0.0,
                )
            )
            reload_report = {
                "status": "PASS" if reload_passed else "FAIL",
                "selected_checkpoint": str(selected_dir),
                "examples": len(reference_examples),
                "maximum_absolute_logit_error": maximum_error,
                "absolute_tolerance": args.reload_atol,
                "adapter_and_head_loaded_into_fresh_base": True,
                "selected_checkpoint_artifact_sha256": _file_hashes(selected_dir),
            }
            _exclusive_json(output / "reload_verification.json", reload_report)
            if not reload_passed:
                raise RuntimeError(
                    f"adapter/head reload verification failed: max error {maximum_error}"
                )
            report = {
                "schema_version": REPORT_SCHEMA,
                "status": "COMPLETE",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "host": socket.gethostname(),
                "model": args.model,
                "classification_mode": args.classification_mode,
                "labels": list(class_names),
                "hidden_to_classes": [HIDDEN_SIZE, len(class_names)],
                "max_sequence_length": args.max_length,
                "bf16_cuda": True,
                "world_size": context.world_size,
                "sampler": {
                    "weights": sampler_weights,
                    "deterministic": True,
                    "cumulative_class_exposures": dict(cumulative),
                    "total_exposures": sum(cumulative.values()),
                },
                "split_audit": split_audit,
                "separate_learning_rates": {
                    "lora": args.lora_learning_rate,
                    "head": args.head_learning_rate,
                },
                "trainable_parameters": trainable_parameters,
                "optimizer_updates": optimizer_updates,
                "warmup_steps": warmup_steps,
                "checkpoints": checkpoint_reports,
                "selected_checkpoint": selected,
                "reload_verification": reload_report,
                "input_sha256": {
                    "train_pairs": {
                        str(path): sha256_file(path) for path in train_paths
                    },
                    "dev_pairs": {str(path): sha256_file(path) for path in dev_paths},
                    "trainer": sha256_file(Path(__file__).resolve()),
                    "run_config": sha256_file(output / "run_config.json"),
                    "split_assignments": sha256_file(
                        output / "split_assignments.jsonl"
                    ),
                },
            }
            _exclusive_json(output / "training_report.json", report)
            _append_event(
                event_path,
                {
                    "event": "RUN_COMPLETE",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "selected_exposure_budget": selected["exposure_budget"],
                    "training_report_sha256": sha256_file(
                        output / "training_report.json"
                    ),
                },
            )
            print(json.dumps(report, sort_keys=True), flush=True)
            return report
        return None
    except Exception as exc:
        if context.is_main and output.exists():
            _append_event(
                event_path,
                {
                    "event": "RUN_FAILED",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    finally:
        if context.distributed:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", action="append", required=True)
    parser.add_argument(
        "--dev-pairs",
        action="append",
        default=[],
        help="Explicit source-disjoint dev pairs; otherwise split --train-pairs by source group.",
    )
    parser.add_argument("--model", default=DEFAULT_NEMOTRON)
    parser.add_argument(
        "--classification-mode",
        choices=("three_way", "binary"),
        default="three_way",
        help=(
            "Binary maps EXACT=1 and FAMILY/REJECT=0; default preserves "
            "three-way behavior."
        ),
    )
    parser.add_argument(
        "--binary-positive-fraction",
        type=float,
        choices=(0.25, 0.5),
        default=0.5,
        help=(
            "Deterministic train exposure fraction for EXACT in binary mode; "
            "dev remains natural."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exposure-budget",
        action="append",
        required=True,
        type=int,
        help="Cumulative example exposure; repeat for append-only checkpoints.",
    )
    parser.add_argument("--max-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--attention", choices=("auto", "eager", "sdpa"), default="eager")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--split-seed", type=int, default=71337)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--min-exact-precision", type=float, default=0.90)
    parser.add_argument("--min-wilson-lower", type=float, default=0.80)
    parser.add_argument("--min-exact-predictions", type=int, default=20)
    parser.add_argument("--reload-verify-examples", type=int, default=8)
    parser.add_argument("--reload-atol", type=float, default=2e-3)
    args = parser.parse_args(argv)
    positive = {
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "reload_verify_examples": args.reload_verify_examples,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        parser.error(f"arguments must be positive: {invalid}")
    if not 1 <= args.max_length <= MAX_SEQUENCE_LENGTH:
        parser.error(f"--max-length must be in [1, {MAX_SEQUENCE_LENGTH}]")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must be in [0, 1)")
    if not 0.0 < args.dev_fraction < 1.0:
        parser.error("--dev-fraction must be in (0, 1)")
    if args.reload_atol < 0:
        parser.error("--reload-atol must be non-negative")
    try:
        args.exposure_budget = _parse_budgets(args.exposure_budget)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
