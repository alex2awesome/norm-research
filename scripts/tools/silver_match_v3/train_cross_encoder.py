#!/usr/bin/env python3
"""Train a clean, task-specific candidate cross-encoder with an abstention gate.

Only frozen v3 teacher labels are consumed.  Historical v1/v2 CE weights are
never loaded.  Source/document groups are deterministically split before pair
construction, dev selects the epoch and score/margin gate, and the frozen test
split is evaluated once after selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .common import metric_card, norm_query, normalize_space, read_jsonl, sha256_file, write_jsonl
from .config import BGE_RERANKER, DEFAULT_OUTPUT_ROOT
from .retrieve import build_vectorizers, top_indices
from .score_verifier_calibration import wilson_interval
from .train_nemotron_lora import source_group_key, split_source_group


ALLTASK_POLICY_V1 = "silver-match-v3-cross-encoder-alltask-policy-v1"
PRESS_RELEASES_POLICY_V2 = (
    "silver-match-v3-cross-encoder-press-releases-policy-v2"
)


def _supported_policy_task(policy: dict[str, Any], task: str) -> bool:
    schema = policy.get("schema_version")
    scope = policy.get("scope") or []
    if schema == ALLTASK_POLICY_V1:
        return task in scope
    if schema == PRESS_RELEASES_POLICY_V2:
        return task == "press-releases" and scope == ["press-releases"]
    return False


@dataclass(frozen=True)
class CELabel:
    norm_uid: str
    corpus: str
    task: str
    source_group: str
    split: str
    query: str
    decision: str
    metric_id: str | None
    acceptable_metric_ids: tuple[str, ...]
    supervision_strength: str
    teacher_sources: tuple[str, ...]


def validate_frozen_policy(args: argparse.Namespace) -> dict[str, Any] | None:
    """Bind a training invocation to the predeclared all-task CE policy."""
    if not args.policy:
        if args.variant_name:
            raise ValueError("--variant-name requires --policy")
        return None
    policy_path = Path(args.policy).resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if not _supported_policy_task(policy, args.task):
        raise ValueError("unsupported policy or task outside policy scope")
    eligibility_path = policy_path.with_suffix(".ELIGIBILITY.json")
    eligibility = None
    if eligibility_path.exists():
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        if (
            eligibility.get("policy_sha256") != sha256_file(policy_path)
            or args.task not in eligibility.get("eligible_primary_tasks", [])
        ):
            raise ValueError("frozen policy eligibility registry restricts this task")
    elif policy.get("schema_version") == PRESS_RELEASES_POLICY_V2:
        raise ValueError("press-releases v2 policy requires an eligibility registry")
    if not args.variant_name:
        raise ValueError("policy-bound training requires --variant-name")
    variants = {
        str(value["name"]): value for value in policy.get("predeclared_variants") or []
    }
    if args.variant_name not in variants:
        raise ValueError("variant is not predeclared by the frozen policy")
    variant = variants[args.variant_name]
    if args.seed != int(variant["seed"]) or not math.isclose(
        args.learning_rate, float(variant["learning_rate"]), rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("seed/learning rate differ from the frozen variant")
    fixed = policy["fixed_training"]
    argument_map = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "warmup_ratio": args.warmup_ratio,
        "negatives_per_positive": args.negatives_per_positive,
        "negatives_per_abstain": args.negatives_per_abstain,
        "strong_positive_repeats": args.strong_positive_repeats,
    }
    for key, observed in argument_map.items():
        expected = fixed[key]
        if isinstance(expected, float):
            matches = math.isclose(float(observed), expected, rel_tol=0, abs_tol=1e-12)
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(f"training argument {key} differs from frozen policy")
    gate = policy["dev_gate"]
    expected_gates = {
        "min_dev_precision": gate["minimum_exact_match_precision"],
        "min_dev_precision_lower": gate[
            "minimum_exact_match_precision_wilson_95_lower"
        ],
        "min_dev_predictions": gate["minimum_retained_predictions"],
        "min_dev_gain": gate["minimum_exact_f_beta_0_5_gain_over_frozen_base"],
    }
    for name, expected in expected_gates.items():
        observed = getattr(args, name)
        if not math.isclose(float(observed), float(expected), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"gate {name} differs from frozen policy")
    if not args.dev_only:
        raise ValueError("all-task policy forbids opening a test split during CE selection")
    base = policy["base_model"]
    model_path = Path(args.model).resolve()
    if model_path != Path(base["path"]).resolve():
        raise ValueError("base model path differs from frozen policy")
    observed_hashes = {
        relative: sha256_file(model_path / relative)
        for relative in sorted(base["file_sha256"])
    }
    if observed_hashes != base["file_sha256"]:
        raise ValueError("base model files differ from frozen policy")
    implementation = policy.get("implementation") or {}
    expected_trainer_sha = implementation.get("train_cross_encoder_sha256")
    if expected_trainer_sha and sha256_file(Path(__file__).resolve()) != expected_trainer_sha:
        raise ValueError("CE trainer implementation differs from frozen policy")
    return {
        "path": str(policy_path),
        "sha256": sha256_file(policy_path),
        "schema_version": policy["schema_version"],
        "variant_name": args.variant_name,
        "variant": variant,
        "base_model_file_sha256": observed_hashes,
        "eligibility": (
            {
                "path": str(eligibility_path),
                "sha256": sha256_file(eligibility_path),
            }
            if eligibility is not None
            else None
        ),
        "policy_schema": policy["schema_version"],
    }


def _resolve(path: str, manifest_path: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else manifest_path.parent / value


def _uid_set(paths: Sequence[Path]) -> set[str]:
    values: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            uid = normalize_space(row.get("norm_uid"))
            if not uid:
                raise ValueError(f"row without norm_uid in {path}")
            values.add(uid)
    return values


def enforce_press_releases_v2_inputs(
    policy_path: Path,
    manifest_path: Path,
    bank_source_sha256: str,
    generic_teacher_paths: Sequence[Path],
    explicit_role_paths: dict[str, Sequence[Path]],
    candidate_paths: Sequence[Path],
) -> dict[str, Any] | None:
    """Fail closed on the independently frozen PR-only CE role boundary."""
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != PRESS_RELEASES_POLICY_V2:
        return None
    if generic_teacher_paths:
        raise ValueError("press-releases v2 requires explicit train/dev roles")
    if explicit_role_paths.get("test"):
        raise ValueError("press-releases v2 forbids consuming a test role")
    if not explicit_role_paths.get("train") or not explicit_role_paths.get("dev"):
        raise ValueError("press-releases v2 requires nonempty train and dev teachers")

    artifacts = policy["immutable_artifacts"]
    manifest = artifacts["manifest"]
    if (
        manifest_path != Path(manifest["path"]).resolve()
        or sha256_file(manifest_path) != manifest["sha256"]
    ):
        raise ValueError("manifest differs from press-releases v2 policy")
    if bank_source_sha256 != artifacts["bank"]["source_sha256"]:
        raise ValueError("bank source differs from press-releases v2 policy")

    identity_sets: dict[str, set[str]] = {}
    for role, key in (("train", "optimize_identity"), ("dev", "select_identity")):
        identity = artifacts[key]
        path = Path(identity["path"]).resolve()
        if sha256_file(path) != identity["sha256"]:
            raise ValueError(f"{role} identity artifact differs from policy")
        identity_sets[role] = _uid_set([path])

    role_uids = {
        role: _uid_set(paths)
        for role, paths in explicit_role_paths.items()
        if paths
    }
    if not identity_sets["train"].issubset(role_uids.get("train", set())):
        raise ValueError("optimize identities are not all present in CE train labels")
    if identity_sets["dev"] != role_uids.get("dev", set()):
        raise ValueError("CE dev labels must equal the frozen select identity panel")
    if identity_sets["train"] & identity_sets["dev"]:
        raise ValueError("frozen optimize/select identities overlap")

    expected_candidates = {
        str(Path(meta["path"]).resolve()): meta["sha256"]
        for meta in artifacts["candidate_inputs"]
    }
    observed_candidates = {str(path): sha256_file(path) for path in candidate_paths}
    if observed_candidates != expected_candidates:
        raise ValueError("candidate inputs differ from press-releases v2 policy")
    return {
        "manifest_sha256": manifest["sha256"],
        "bank_source_sha256": artifacts["bank"]["source_sha256"],
        "optimize_identity_count": len(identity_sets["train"]),
        "select_identity_count": len(identity_sets["dev"]),
        "candidate_inputs": observed_candidates,
        "test_role_consumed": False,
    }


def _is_weak(row: dict[str, Any]) -> bool:
    return (
        normalize_space(row.get("supervision_strength")) == "weak_forced_positive"
        or normalize_space(row.get("label_source")) == "sonnet_forced_top3"
    )


def merge_teacher_rows(
    rows: Iterable[tuple[str, dict[str, Any]]],
    norms: dict[str, dict[str, Any]],
    task: str,
    bank_ids: set[str],
    bank_sha: str,
    *,
    split_seed: int,
    split_by_uid: dict[str, str] | None = None,
) -> tuple[list[CELabel], dict[str, Any]]:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    for source, row in rows:
        uid = normalize_space(row.get("norm_uid"))
        if not uid or uid not in norms or norms[uid].get("task") != task:
            counters["outside_task_manifest"] += 1
            continue
        row_task = normalize_space(row.get("task"))
        if row_task and row_task != task:
            counters["teacher_task_mismatch"] += 1
            continue
        row_sha = normalize_space(
            row.get("current_bank_source_sha256") or row.get("bank_source_sha256")
        )
        if row_sha and row_sha != bank_sha:
            counters["stale_bank_hash"] += 1
            continue
        if split_by_uid is not None:
            explicit_role = split_by_uid.get(uid)
            gepa_role = normalize_space(row.get("gepa_role"))
            if gepa_role == "optimize" and (
                explicit_role != "train" or row.get("ce_training_eligible") is not True
            ):
                raise ValueError(
                    f"optimize truth lacks a policy-bound CE bridge or train role: {uid}"
                )
            if gepa_role == "select" and explicit_role != "dev":
                raise ValueError(f"select truth may appear only in the CE dev role: {uid}")
            if gepa_role in {"evaluation", "test", "blind"}:
                raise ValueError(f"ineligible GEPA/evaluation role in CE teachers: {uid}")
        grouped[uid].append((source, row))

    labels: list[CELabel] = []
    conflicts = []
    for uid, group in sorted(grouped.items()):
        norm = norms[uid]
        strong_matches = [pair for pair in group if pair[1].get("decision") == "MATCH" and not _is_weak(pair[1])]
        weak_matches = [pair for pair in group if pair[1].get("decision") == "MATCH" and _is_weak(pair[1])]
        nonmatches = [pair for pair in group if pair[1].get("decision") != "MATCH" and not _is_weak(pair[1])]
        metric_id: str | None = None
        acceptable: set[str] = set()
        strength = "strong"
        decision: str
        chosen: list[tuple[str, dict[str, Any]]]
        if strong_matches:
            ids = {normalize_space(row.get("metric_id")) for _, row in strong_matches}
            ids.discard("")
            if len(ids) != 1:
                conflicts.append({"norm_uid": uid, "metric_ids": sorted(ids)})
                continue
            metric_id = next(iter(ids))
            if metric_id not in bank_ids:
                counters["strong_metric_outside_bank"] += 1
                continue
            decision, chosen = "MATCH", strong_matches
            acceptable.add(metric_id)
        elif weak_matches:
            ranked = sorted(
                weak_matches,
                key=lambda pair: (
                    int(pair[1].get("forced_rank") or 10**9),
                    normalize_space(pair[1].get("metric_id")),
                ),
            )
            metric_id = normalize_space(ranked[0][1].get("metric_id"))
            acceptable = {
                normalize_space(row.get("metric_id"))
                for _, row in weak_matches
                if normalize_space(row.get("metric_id")) in bank_ids
            }
            if metric_id not in bank_ids:
                counters["weak_metric_outside_bank"] += 1
                continue
            decision, chosen, strength = "MATCH", weak_matches, "weak_forced_top3"
        elif nonmatches:
            decisions = Counter(normalize_space(row.get("decision")) for _, row in nonmatches)
            # Contradictory abstention subtypes remain useful as a no-exact-match
            # gate target; retain the majority subtype only for reporting.
            decision = sorted(decisions, key=lambda value: (-decisions[value], value))[0]
            chosen = nonmatches
        else:
            counters["no_usable_label"] += 1
            continue
        for _, row in chosen:
            for key in ("acceptable_metric_ids", "equivalent_metric_ids", "metric_ids"):
                values = row.get(key) or []
                if isinstance(values, str):
                    values = [values]
                acceptable.update(str(value) for value in values if str(value) in bank_ids)
        group_key = source_group_key(norm)
        split = (
            split_by_uid[uid]
            if split_by_uid is not None and uid in split_by_uid
            else split_source_group(group_key, split_seed)
        )
        labels.append(
            CELabel(
                norm_uid=uid,
                corpus=str(norm["corpus"]),
                task=task,
                source_group=group_key,
                split=split,
                query=norm_query(norm),
                decision=decision,
                metric_id=metric_id,
                acceptable_metric_ids=tuple(sorted(acceptable)),
                supervision_strength=strength,
                teacher_sources=tuple(sorted({Path(source).name for source, _ in chosen})),
            )
        )
    audit = {
        "selected": len(labels),
        "decision_counts": dict(sorted(Counter(label.decision for label in labels).items())),
        "split_counts": dict(sorted(Counter(label.split for label in labels).items())),
        "strength_counts": dict(sorted(Counter(label.supervision_strength for label in labels).items())),
        "rejections": dict(sorted(counters.items())),
        "conflicting_strong_matches": conflicts,
        "split_mode": "explicit_role" if split_by_uid is not None else "source_hash",
    }
    if conflicts:
        raise ValueError(f"conflicting strong teacher matches: {conflicts[:5]}")
    return labels, audit


def build_explicit_split_map(
    role_paths: dict[str, Sequence[Path]],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Map label UIDs to predeclared roles and reject cross-role reuse.

    The clean v3 panels are frozen as optimize/select/blind artifacts before
    labels exist.  CE training must preserve those roles rather than hashing
    the resulting labels into new splits.  Repeated independent annotations
    within one role are allowed; the same UID in two roles is not.
    """
    allowed = {"train", "dev", "test"}
    unknown = set(role_paths) - allowed
    if unknown:
        raise ValueError(f"unknown explicit split roles: {sorted(unknown)}")
    split_by_uid: dict[str, str] = {}
    provenance: dict[str, dict[str, str]] = {}
    for role in ("train", "dev", "test"):
        for path in role_paths.get(role, ()):
            provenance[str(path)] = {"role": role, "sha256": sha256_file(path)}
            for row in read_jsonl(path):
                uid = normalize_space(row.get("norm_uid"))
                if not uid:
                    raise ValueError(f"teacher row without norm_uid in {path}")
                previous = split_by_uid.get(uid)
                if previous is not None and previous != role:
                    raise ValueError(
                        f"teacher UID {uid} appears in explicit roles {previous} and {role}"
                    )
                split_by_uid[uid] = role
    return split_by_uid, provenance


def audit_source_group_splits(labels: Sequence[CELabel]) -> dict[str, Any]:
    """Prove that no canonical source family crosses a CE data role."""
    roles: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        roles[label.source_group].add(label.split)
    overlaps = {
        group: sorted(values)
        for group, values in roles.items()
        if len(values) > 1
    }
    if overlaps:
        raise ValueError(
            "source groups cross CE roles: "
            + json.dumps(dict(list(sorted(overlaps.items()))[:10]), sort_keys=True)
        )
    return {
        "complete": True,
        "unique_source_groups": len(roles),
        "source_group_counts": dict(
            sorted(Counter(next(iter(values)) for values in roles.values()).items())
        ),
        "cross_role_source_group_count": 0,
    }


def load_candidate_ids(
    paths: Sequence[Path], required_uids: set[str] | None = None
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for path in paths:
        for row in read_jsonl(path):
            uid = str(row["norm_uid"])
            if required_uids is not None and uid not in required_uids:
                continue
            if uid in output:
                raise ValueError(f"duplicate candidate UID across inputs: {uid}")
            output[uid] = [
                str(value.get("metric_id") if isinstance(value, dict) else value)
                for value in row.get("candidates") or []
            ]
    return output


def lexical_rankings(labels: Sequence[CELabel], cards: list[str]) -> np.ndarray:
    word, char, card_word, card_char = build_vectorizers(cards)
    queries = [label.query for label in labels]
    scores = (word.transform(queries) @ card_word.T).toarray()
    scores += (char.transform(queries) @ card_char.T).toarray()
    return top_indices(scores, len(cards))


def build_training_pairs(
    labels: Sequence[CELabel],
    bank: list[dict[str, Any]],
    candidate_ids: dict[str, list[str]],
    *,
    negatives_per_positive: int,
    negatives_per_abstain: int,
    strong_positive_repeats: int,
) -> list[dict[str, Any]]:
    bank_ids = [str(metric["metric_id"]) for metric in bank]
    bank_index = {metric_id: idx for idx, metric_id in enumerate(bank_ids)}
    cards = [metric_card(metric) for metric in bank]
    lexical = lexical_rankings(labels, cards)
    rows = []
    for i, label in enumerate(labels):
        if label.split != "train" or label.decision == "MATCH_FAMILY_ONLY":
            continue
        ordered = []
        seen = set()
        for metric_id in candidate_ids.get(label.norm_uid, []):
            if metric_id in bank_index and metric_id not in seen:
                ordered.append(metric_id)
                seen.add(metric_id)
        for idx in lexical[i]:
            metric_id = bank_ids[int(idx)]
            if metric_id not in seen:
                ordered.append(metric_id)
                seen.add(metric_id)
        if label.decision == "MATCH":
            assert label.metric_id is not None
            repeats = strong_positive_repeats if label.supervision_strength == "strong" else 1
            for repeat in range(repeats):
                rows.append(
                    {
                        "norm_uid": label.norm_uid,
                        "split": "train",
                        "query": label.query,
                        "metric_id": label.metric_id,
                        "metric_card": cards[bank_index[label.metric_id]],
                        "label": 1.0,
                        "kind": "positive",
                        "repeat": repeat,
                        "supervision_strength": label.supervision_strength,
                    }
                )
            negatives = [
                metric_id for metric_id in ordered
                if metric_id not in set(label.acceptable_metric_ids)
            ][:negatives_per_positive]
        else:
            negatives = ordered[:negatives_per_abstain]
        for rank, metric_id in enumerate(negatives, 1):
            rows.append(
                {
                    "norm_uid": label.norm_uid,
                    "split": "train",
                    "query": label.query,
                    "metric_id": metric_id,
                    "metric_card": cards[bank_index[metric_id]],
                    "label": 0.0,
                    "kind": "hard_negative" if label.decision == "MATCH" else "abstention_negative",
                    "negative_rank": rank,
                    "gold_decision": label.decision,
                    "supervision_strength": label.supervision_strength,
                }
            )
    if not rows:
        raise ValueError("no cross-encoder training pairs")
    return rows


def score_full_bank(model: Any, labels: Sequence[CELabel], bank: list[dict[str, Any]], batch_size: int) -> np.ndarray:
    import torch

    cards = [metric_card(metric) for metric in bank]
    pairs = [[label.query, card] for label in labels for card in cards]
    if not pairs:
        return np.empty((0, len(bank)), dtype=np.float32)
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        activation_fn=torch.nn.Sigmoid(),
    )
    return np.asarray(scores, dtype=np.float32).reshape(len(labels), len(bank))


def gate_report(
    labels: Sequence[CELabel],
    bank_ids: Sequence[str],
    scores: np.ndarray,
    score_threshold: float,
    margin_threshold: float,
    *,
    beta: float = 0.5,
) -> dict[str, Any]:
    if len(labels) != scores.shape[0]:
        raise ValueError("label/score row mismatch")
    if not labels:
        return {"n": 0}
    order = np.argsort(-scores, axis=1, kind="stable")
    top = order[:, 0]
    top_scores = scores[np.arange(len(labels)), top]
    second_scores = scores[np.arange(len(labels)), order[:, 1]] if len(bank_ids) > 1 else np.zeros(len(labels))
    margins = top_scores - second_scores
    predicted = (top_scores >= score_threshold) & (margins >= margin_threshold)
    gold_match = np.asarray([label.decision == "MATCH" for label in labels])
    exact = np.asarray([
        label.decision == "MATCH" and bank_ids[int(top[i])] == label.metric_id
        for i, label in enumerate(labels)
    ])
    tp = int(np.sum(predicted & exact))
    fp = int(np.sum(predicted & ~exact))
    fn = int(np.sum(gold_match & ~(predicted & exact)))
    tn = int(np.sum(~gold_match & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / int(np.sum(gold_match)) if np.sum(gold_match) else 0.0
    b2 = beta * beta
    f_beta = (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.0
    nonmatch_recall = tn / int(np.sum(~gold_match)) if np.sum(~gold_match) else None
    ranks = []
    for i, label in enumerate(labels):
        if label.decision == "MATCH":
            ranks.append(int(np.where(order[i] == bank_ids.index(label.metric_id))[0][0]) + 1)
    return {
        "n": len(labels),
        "n_gold_match": int(np.sum(gold_match)),
        "n_gold_nonmatch": int(np.sum(~gold_match)),
        "score_threshold": float(score_threshold),
        "margin_threshold": float(margin_threshold),
        "predicted_match_rate": float(np.mean(predicted)),
        "exact_match_precision": precision,
        "exact_match_precision_wilson_95": wilson_interval(tp, tp + fp),
        "exact_match_recall": recall,
        "predicted_match_count": tp + fp,
        "exact_f_beta_0_5": f_beta,
        "nonmatch_recall": nonmatch_recall,
        "ungated_exact_top1": float(np.mean(np.asarray(ranks) == 1)) if ranks else None,
        **{
            f"ungated_exact_recall_at_{k}": float(np.mean(np.asarray(ranks) <= min(k, len(bank_ids))))
            if ranks else None
            for k in (1, 5, 10, 16, 30, 50)
        },
        "top_score_quantiles": {
            str(q): float(np.quantile(top_scores, q)) for q in (0, 0.1, 0.5, 0.9, 1)
        },
        "margin_quantiles": {
            str(q): float(np.quantile(margins, q)) for q in (0, 0.1, 0.5, 0.9, 1)
        },
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def tune_gate(
    labels: Sequence[CELabel],
    bank_ids: Sequence[str],
    scores: np.ndarray,
    *,
    min_precision: float,
    min_predictions: int = 20,
    min_precision_lower: float = 0.8,
) -> dict[str, Any]:
    if not labels:
        raise ValueError("cannot tune gate without dev labels")
    order = np.argsort(-scores, axis=1, kind="stable")
    top_scores = scores[np.arange(len(labels)), order[:, 0]]
    second = scores[np.arange(len(labels)), order[:, 1]] if len(bank_ids) > 1 else np.zeros(len(labels))
    margins = top_scores - second
    score_grid = sorted(set([0.0, 1.0, *np.quantile(top_scores, np.linspace(0, 1, 41)).tolist()]))
    margin_grid = sorted(set([0.0, *np.quantile(margins, np.linspace(0, 1, 21)).tolist()]))
    reports = [
        gate_report(labels, bank_ids, scores, score, margin)
        for score in score_grid
        for margin in margin_grid
    ]
    feasible = [
        row
        for row in reports
        if row["exact_match_precision"] >= min_precision
        and row["predicted_match_count"] >= min_predictions
        and row["exact_match_precision_wilson_95"] is not None
        and row["exact_match_precision_wilson_95"][0] >= min_precision_lower
    ]
    pool = feasible or reports
    best = max(
        pool,
        key=lambda row: (
            row["exact_f_beta_0_5"],
            row["exact_match_precision"],
            row["exact_match_recall"],
            row["nonmatch_recall"] if row["nonmatch_recall"] is not None else 0.0,
            -row["score_threshold"],
            -row["margin_threshold"],
        ),
    )
    return {
        **best,
        "precision_constraint_met": bool(feasible),
        "min_precision": min_precision,
        "min_predictions": min_predictions,
        "min_precision_lower": min_precision_lower,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    policy_binding = validate_frozen_policy(args)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.task not in manifest.get("banks", {}):
        raise KeyError(args.task)
    bank_meta = manifest["banks"][args.task]
    bank_path = _resolve(bank_meta["path"], manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))["metrics"]
    bank_ids = [str(metric["metric_id"]) for metric in bank]
    norms = {}
    for corpus, meta in manifest["corpora"].items():
        if meta["task"] != args.task:
            continue
        for row in read_jsonl(_resolve(meta["path"], manifest_path)):
            norms[str(row["norm_uid"])] = row
    generic_teacher_paths = [Path(path).resolve() for path in args.teachers]
    explicit_role_paths = {
        "train": [Path(path).resolve() for path in args.train_teachers],
        "dev": [Path(path).resolve() for path in args.dev_teachers],
        "test": [Path(path).resolve() for path in args.test_teachers],
    }
    has_explicit_roles = any(explicit_role_paths.values())
    if generic_teacher_paths and has_explicit_roles:
        raise ValueError(
            "use either --teachers with deterministic source hashing or explicit "
            "--train-teachers/--dev-teachers/--test-teachers, not both"
        )
    if has_explicit_roles:
        teacher_paths = [
            path
            for role in ("train", "dev", "test")
            for path in explicit_role_paths[role]
        ]
        split_by_uid, explicit_role_provenance = build_explicit_split_map(
            explicit_role_paths
        )
    else:
        teacher_paths = generic_teacher_paths
        split_by_uid = None
        explicit_role_provenance = {}
    if not teacher_paths:
        raise ValueError("at least one teacher label file is required")
    rows = ((str(path), row) for path in teacher_paths for row in read_jsonl(path))
    labels, teacher_audit = merge_teacher_rows(
        rows,
        norms,
        args.task,
        set(bank_ids),
        str(bank_meta["source_sha256"]),
        split_seed=args.split_seed,
        split_by_uid=split_by_uid,
    )
    source_group_split_audit = audit_source_group_splits(labels)
    by_split = {split: [label for label in labels if label.split == split] for split in ("train", "dev", "test")}
    required_splits = ("train", "dev") if args.dev_only else ("train", "dev", "test")
    if any(not by_split[split] for split in required_splits):
        raise ValueError(
            f"empty required split: { {key: len(value) for key, value in by_split.items()} }"
        )
    candidate_paths = [Path(path).resolve() for path in args.candidates]
    pr_v2_input_audit = (
        enforce_press_releases_v2_inputs(
            Path(args.policy).resolve(),
            manifest_path,
            str(bank_meta["source_sha256"]),
            generic_teacher_paths,
            explicit_role_paths,
            candidate_paths,
        )
        if args.policy
        else None
    )
    candidate_ids = load_candidate_ids(
        candidate_paths, {label.norm_uid for label in labels}
    )
    pairs = build_training_pairs(
        labels,
        bank,
        candidate_ids,
        negatives_per_positive=args.negatives_per_positive,
        negatives_per_abstain=args.negatives_per_abstain,
        strong_positive_repeats=args.strong_positive_repeats,
    )
    output = Path(args.output_root).resolve() / args.task
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "labels.jsonl", [label.__dict__ for label in labels])
    write_jsonl(output / "training_pairs.jsonl", pairs)

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    examples = [InputExample(texts=[row["query"], row["metric_card"]], label=row["label"]) for row in pairs]
    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,
        generator=generator,
    )
    model = CrossEncoder(args.model, num_labels=1, max_length=args.max_length, device=args.device)
    base_dev_scores = score_full_bank(model, by_split["dev"], bank, args.eval_batch_size)
    base_dev = tune_gate(
        by_split["dev"],
        bank_ids,
        base_dev_scores,
        min_precision=args.min_dev_precision,
        min_predictions=args.min_dev_predictions,
        min_precision_lower=args.min_dev_precision_lower,
    )
    epoch_reports = []
    best = None
    total_steps = math.ceil(len(loader)) * args.epochs
    for epoch in range(1, args.epochs + 1):
        epoch_dir = output / f"epoch-{epoch}"
        model.fit(
            train_dataloader=loader,
            epochs=1,
            warmup_steps=max(1, math.ceil(total_steps * args.warmup_ratio / args.epochs)),
            optimizer_params={"lr": args.learning_rate},
            output_path=str(epoch_dir),
            save_best_model=False,
            show_progress_bar=True,
            use_amp=True,
        )
        # CrossEncoder.fit does not persist output_path without an evaluator
        # in sentence-transformers 5.x, so freeze every epoch explicitly.
        model.save(str(epoch_dir))
        dev_scores = score_full_bank(model, by_split["dev"], bank, args.eval_batch_size)
        dev = tune_gate(
            by_split["dev"],
            bank_ids,
            dev_scores,
            min_precision=args.min_dev_precision,
            min_predictions=args.min_dev_predictions,
            min_precision_lower=args.min_dev_precision_lower,
        )
        report = {"epoch": epoch, "dev": dev}
        epoch_reports.append(report)
        if best is None or (
            dev["precision_constraint_met"],
            dev["exact_f_beta_0_5"],
            dev["exact_match_precision"],
            dev["exact_match_recall"],
        ) > (
            best["dev"]["precision_constraint_met"],
            best["dev"]["exact_f_beta_0_5"],
            best["dev"]["exact_match_precision"],
            best["dev"]["exact_match_recall"],
        ):
            best = report
    assert best is not None
    selected_epoch = int(best["epoch"])
    model_dir = output / "model"
    shutil.copytree(output / f"epoch-{selected_epoch}", model_dir)
    dev_promotable = (
        best["dev"]["exact_f_beta_0_5"]
        >= base_dev["exact_f_beta_0_5"] + args.min_dev_gain
        and best["dev"]["precision_constraint_met"]
        and best["dev"]["exact_match_precision"] >= args.min_dev_precision
        and best["dev"]["exact_match_precision_wilson_95"] is not None
        and best["dev"]["exact_match_precision_wilson_95"][0]
        >= args.min_dev_precision_lower
        and best["dev"]["predicted_match_count"] >= args.min_dev_predictions
    )
    test = None
    promotable = False
    if not args.dev_only:
        selected = CrossEncoder(str(model_dir), device=args.device)
        test_scores = score_full_bank(
            selected, by_split["test"], bank, args.eval_batch_size
        )
        test = gate_report(
            by_split["test"],
            bank_ids,
            test_scores,
            best["dev"]["score_threshold"],
            best["dev"]["margin_threshold"],
        )
        promotable = (
            dev_promotable
            and test["predicted_match_count"] >= args.min_test_predictions
            and test["exact_match_precision"] >= args.min_test_precision
            and test["exact_match_precision_wilson_95"] is not None
            and test["exact_match_precision_wilson_95"][0]
            >= args.min_test_precision_lower
        )
    report = {
        "task": args.task,
        "status": (
            (
                "DEV_PROMOTABLE_PENDING_BLIND"
                if dev_promotable
                else "REJECTED_DEV_GATE"
            )
            if args.dev_only
            else ("PROMOTABLE" if promotable else "REJECTED_VALIDATION_GATE")
        ),
        "selected_epoch": selected_epoch,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "bank": str(bank_path),
        "bank_source_sha256": bank_meta["source_sha256"],
        "model_base": args.model,
        "frozen_policy": policy_binding,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "teacher_inputs": {str(path): sha256_file(path) for path in teacher_paths},
        "teacher_split_mode": "explicit_role" if has_explicit_roles else "source_hash",
        "explicit_role_inputs": explicit_role_provenance,
        "candidate_inputs": {str(path): sha256_file(path) for path in candidate_paths},
        "press_releases_v2_input_audit": pr_v2_input_audit,
        "teacher_audit": teacher_audit,
        "source_group_split_audit": source_group_split_audit,
        "training_pair_counts": dict(sorted(Counter(row["kind"] for row in pairs).items())),
        "base_dev": base_dev,
        "epochs": epoch_reports,
        "selected_dev": best["dev"],
        "dev_promotable": dev_promotable,
        "frozen_test": test,
        "frozen_test_consumed": not args.dev_only,
        "gates": {
            "min_dev_precision": args.min_dev_precision,
            "min_dev_precision_lower": args.min_dev_precision_lower,
            "min_dev_predictions": args.min_dev_predictions,
            "min_test_precision": args.min_test_precision,
            "min_test_precision_lower": args.min_test_precision_lower,
            "min_test_predictions": args.min_test_predictions,
            "min_dev_gain": args.min_dev_gain,
        },
        "model_dir": str(model_dir),
        "model_hashes": {
            str(path.relative_to(model_dir)): sha256_file(path)
            for path in sorted(model_dir.rglob("*")) if path.is_file()
        },
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--variant-name")
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument(
        "--teachers",
        action="append",
        default=[],
        help="teacher file; source groups are deterministically hash-split",
    )
    parser.add_argument(
        "--train-teachers",
        action="append",
        default=[],
        help="teacher file from a predeclared train/optimize role",
    )
    parser.add_argument(
        "--dev-teachers",
        action="append",
        default=[],
        help="teacher file from a predeclared dev/select role",
    )
    parser.add_argument(
        "--test-teachers",
        action="append",
        default=[],
        help="teacher file from a predeclared sealed test role",
    )
    parser.add_argument("--candidates", action="append", default=[])
    parser.add_argument("--model", default=BGE_RERANKER)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--negatives-per-abstain", type=int, default=4)
    parser.add_argument("--strong-positive-repeats", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=73129)
    parser.add_argument("--seed", type=int, default=94117)
    parser.add_argument("--min-dev-precision", type=float, default=0.85)
    parser.add_argument("--min-dev-precision-lower", type=float, default=0.80)
    parser.add_argument("--min-dev-predictions", type=int, default=20)
    parser.add_argument("--min-test-precision", type=float, default=0.75)
    parser.add_argument("--min-test-precision-lower", type=float, default=0.70)
    parser.add_argument("--min-test-predictions", type=int, default=20)
    parser.add_argument("--min-dev-gain", type=float, default=0.01)
    parser.add_argument(
        "--dev-only",
        action="store_true",
        help="select and save on dev without opening the held-out test split",
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
