#!/usr/bin/env python3
"""Freeze a generic, source-disjoint typed Gemma-4 adjudicator dataset.

The builder is intentionally task-agnostic.  It joins one or more canonical
truth files to the current bank and canonical norms, constructs a compact
truth-blind candidate slate, and emits two option-order views for train rows
but exactly one view for every held-out row.  Gold candidates may be inserted
only into train.  Dev, test, and blind examples therefore measure the complete
retriever-plus-adjudicator stack rather than an oracle-candidate condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adjudicate_gemma import build_item_prompt
from .build_nemotron_ce_pairs import (
    DECISIONS,
    SPLITS,
    _family_anchors,
    _load_bank,
    _load_canonical_norms,
    _load_families,
    _load_split_assignments,
    _load_truth,
    _relation,
    _resolve,
)
from .common import normalize_space, read_jsonl, sha256_file, write_jsonl


SCHEMA = "silver-match-v3-gemma4-typed-example-v2"
REPORT_SCHEMA = "silver-match-v3-gemma4-typed-dataset-report-v2"
CONFIDENCES = {"low", "medium", "high"}
OUTPUT_SPLITS = ("train", "dev", "test", "blind")


def _ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _stable_digest(seed: int, *parts: Any) -> str:
    payload = "\x1f".join((str(seed), *(normalize_space(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


def _parse_candidate_specs(
    values: Sequence[str], base: Path
) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        lane, separator, raw_path = value.partition("=")
        lane = normalize_space(lane)
        if not separator or not lane or not raw_path or lane in seen:
            raise ValueError(f"candidate inputs must be unique LANE=PATH values: {value!r}")
        seen.add(lane)
        output.append((lane, _resolve(raw_path, base)))
    if not output:
        raise ValueError("at least one candidate input is required")
    return output


def _score(candidate: Mapping[str, Any]) -> float | None:
    probabilities = candidate.get("probabilities")
    if isinstance(probabilities, Mapping) and probabilities.get("EXACT") is not None:
        raw = probabilities["EXACT"]
    else:
        raw = next(
            (
                candidate[key]
                for key in ("exact_probability", "exact_score", "score", "reranker_score")
                if candidate.get(key) is not None
            ),
            None,
        )
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate score is not numeric") from exc
    if not math.isfinite(value):
        raise ValueError("candidate score is not finite")
    return value


def _load_candidates(
    specs: Sequence[tuple[str, Path]],
    *,
    truth: Mapping[str, Mapping[str, Any]],
    task: str,
    bank_hash: str,
    bank_ids: set[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Load candidate lists without consulting truth labels or split outcomes."""

    best: dict[str, dict[str, tuple[tuple[Any, ...], dict[str, Any]]]] = defaultdict(dict)
    refs: dict[str, dict[str, Any]] = {}
    per_lane: dict[str, dict[str, int]] = {}
    for lane_index, (lane, path) in enumerate(specs):
        refs[lane] = _ref(path)
        seen_uids: set[str] = set()
        audit: Counter[str] = Counter()
        for row in read_jsonl(path):
            uid = normalize_space(row.get("norm_uid"))
            if not uid or uid in seen_uids:
                raise ValueError(f"candidate lane has missing/duplicate norm_uid: {lane}/{uid}")
            seen_uids.add(uid)
            supplied_hash = normalize_space(
                row.get("bank_source_sha256") or row.get("current_bank_source_sha256")
            )
            if row.get("task") != task or supplied_hash != bank_hash:
                raise ValueError(f"candidate task/bank mismatch: {lane}/{uid}")
            audit["input_rows"] += 1
            if uid not in truth:
                audit["outside_truth_rows_ignored"] += 1
                continue
            audit["joined_truth_rows"] += 1
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"candidate lane has no candidate list: {lane}/{uid}")
            seen_metrics: set[str] = set()
            for position, candidate in enumerate(candidates, 1):
                if not isinstance(candidate, Mapping):
                    raise ValueError(f"candidate is not an object: {lane}/{uid}")
                metric_id = normalize_space(candidate.get("metric_id"))
                if not metric_id or metric_id in seen_metrics or metric_id not in bank_ids:
                    raise ValueError(
                        f"candidate metric is duplicate/missing/out of bank: {lane}/{uid}/{metric_id}"
                    )
                seen_metrics.add(metric_id)
                try:
                    rank = int(candidate.get("rank", position))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"candidate rank is invalid: {lane}/{uid}") from exc
                if rank < 1:
                    raise ValueError(f"candidate rank is non-positive: {lane}/{uid}")
                score = _score(candidate)
                # Lane order and rank are frozen before labels.  Score is only
                # a tie-break within the same retrieval lane/rank contract.
                key = (lane_index, rank, -(score if score is not None else -math.inf), metric_id)
                provenance = {
                    "lane": lane,
                    "lane_artifact_sha256": refs[lane]["sha256"],
                    "rank": rank,
                    **({"score": score} if score is not None else {}),
                }
                prior = best[uid].get(metric_id)
                if prior is None or key < prior[0]:
                    best[uid][metric_id] = (key, provenance)
                audit["candidate_rows_validated"] += 1
        per_lane[lane] = dict(sorted(audit.items()))
    missing = sorted(set(truth) - set(best))
    if missing:
        raise ValueError(f"candidate inputs miss {len(missing)} truth UIDs: {missing[:5]}")
    ordered = {
        uid: [metric_id for metric_id, _ in sorted(values.items(), key=lambda item: item[1][0])]
        for uid, values in best.items()
    }
    provenance = {
        uid: {metric_id: value[1] for metric_id, value in values.items()}
        for uid, values in best.items()
    }
    return ordered, provenance, {
        "per_lane": per_lane,
        "truth_uid_coverage": len(ordered),
        "outside_truth_rows_ignored": sum(
            audit.get("outside_truth_rows_ignored", 0) for audit in per_lane.values()
        ),
        "ordering_contract": "candidate_lane_cli_order_then_rank_then_score_then_metric_id",
        "truth_labels_used_for_retrieved_candidate_order": False,
    }


def _confidence(row: Mapping[str, Any]) -> tuple[str, bool]:
    raw = row.get("confidence")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence is outside [0,1]: {row.get('norm_uid')}")
        return ("high" if value >= 0.8 else "medium" if value >= 0.5 else "low"), False
    value = normalize_space(raw).lower()
    if value in CONFIDENCES:
        return value, False
    if not value:
        return "medium", True
    raise ValueError(f"unknown confidence: {row.get('norm_uid')}/{raw!r}")


def _reason(row: Mapping[str, Any]) -> str:
    reason = normalize_space(row.get("reason") or row.get("rationale"))
    if not reason:
        predictions = row.get("source_predictions")
        if isinstance(predictions, Mapping):
            for source in row.get("agreement_sources") or predictions:
                prediction = predictions.get(source)
                if isinstance(prediction, Mapping):
                    reason = normalize_space(
                        prediction.get("reason") or prediction.get("rationale")
                    )
                    if reason:
                        break
    if not reason:
        raise ValueError(f"truth row lacks a training rationale: {row.get('norm_uid')}")
    return reason


def structured_target(
    *, decision: str, metric_id: str | None, confidence: str, reason: str
) -> tuple[str, dict[str, dict[str, int]]]:
    """Render canonical compact JSON and exact character spans for each field."""

    fields: list[tuple[str, Any]] = [
        ("decision", decision),
        ("metric_id", metric_id),
        ("confidence", confidence),
        ("reason", reason),
    ]
    parts = ["{"]
    spans: dict[str, dict[str, int]] = {}
    length = 1
    for index, (name, value) in enumerate(fields):
        if index:
            parts.append(",")
            length += 1
        segment = (
            json.dumps(name, ensure_ascii=False, separators=(",", ":"))
            + ":"
            + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
        spans[name] = {"start": length, "end": length + len(segment)}
        parts.append(segment)
        length += len(segment)
    parts.append("}")
    return "".join(parts), spans


def _target_for_slate(
    truth_row: Mapping[str, Any],
    candidate_ids: Sequence[str],
    families: Mapping[str, frozenset[str]],
) -> tuple[str, str, str | None]:
    truth_decision = normalize_space(truth_row.get("decision"))
    relations = {
        metric_id: _relation(truth_row, metric_id, families)
        for metric_id in candidate_ids
    }
    exact = [metric_id for metric_id in candidate_ids if relations[metric_id] == "EXACT"]
    family = [metric_id for metric_id in candidate_ids if relations[metric_id] == "FAMILY"]
    if exact:
        return "EXACT", "MATCH", exact[0]
    if truth_decision == "MATCH_FAMILY_ONLY" or family:
        return "FAMILY", "MATCH_FAMILY_ONLY", None
    if truth_decision == "MATCH":
        return "REJECT", "NO_CANDIDATE_FITS", None
    if truth_decision not in DECISIONS:
        raise ValueError(f"invalid truth decision: {truth_row.get('norm_uid')}")
    return "REJECT", truth_decision, None


def build(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest_path = Path(args.manifest).resolve()
    bank_path = Path(args.bank).resolve()
    hierarchy_path = Path(args.hierarchy).resolve()
    prompt_path = Path(args.prompt).resolve()
    truth_paths = tuple(Path(path).resolve() for path in args.truth)
    assignments_path = (
        Path(args.split_assignments).resolve()
        if getattr(args, "split_assignments", None)
        else None
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    metrics, metric_by_id, bank_hash = _load_bank(manifest, bank_path, args.task)
    assignments, assignment_audit = _load_split_assignments(
        assignments_path, task=args.task, bank_hash=bank_hash
    )
    truth_rows, truth, truth_audit = _load_truth(
        truth_paths,
        task=args.task,
        bank_hash=bank_hash,
        bank_ids=set(metric_by_id),
        split_assignments=assignments,
        allow_unanchored_family_only=True,
    )
    norms = _load_canonical_norms(manifest, manifest_path, args.task, truth)
    families = _load_families(
        hierarchy_path, task=args.task, metric_by_id=metric_by_id
    )
    specs = _parse_candidate_specs(args.candidates, manifest_path.parent)
    candidates, provenance, candidate_audit = _load_candidates(
        specs,
        truth=truth,
        task=args.task,
        bank_hash=bank_hash,
        bank_ids=set(metric_by_id),
    )
    prompt = prompt_path.read_text(encoding="utf-8").rstrip() + "\n"
    if not prompt.strip():
        raise ValueError("prompt is empty")

    buckets: dict[str, list[dict[str, Any]]] = {split: [] for split in OUTPUT_SPLITS}
    injections: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    default_confidences = 0
    for uid in sorted(truth):
        truth_row = truth[uid]
        norm = norms[uid]
        split = normalize_space(truth_row.get("split"))
        if split not in SPLITS:
            raise ValueError(f"unsupported split: {uid}/{split}")
        available = list(candidates[uid])
        if len(available) < args.max_candidates:
            raise ValueError(
                f"candidate union has fewer than K={args.max_candidates}: {uid}/{len(available)}"
            )
        selected = available[: args.max_candidates]
        if split == "train":
            truth_decision = normalize_space(truth_row.get("decision"))
            insertion: str | None = None
            if truth_decision == "MATCH":
                insertion = normalize_space(truth_row.get("metric_id"))
            elif truth_decision == "MATCH_FAMILY_ONLY":
                anchors = sorted(
                    _family_anchors(truth_row),
                    key=lambda metric_id: (
                        _stable_digest(args.order_seed, "family-anchor", uid, metric_id),
                        metric_id,
                    ),
                )
                insertion = anchors[0] if anchors else None
            if insertion and insertion not in selected:
                selected[-1] = insertion
                provenance[uid][insertion] = {
                    "lane": "train_gold_injection",
                    "lane_artifact_sha256": "derived-from-frozen-train-truth",
                    "rank": None,
                }
                injections[
                    "train_exact" if truth_decision == "MATCH" else "train_family_anchor"
                ] += 1
        if len(selected) != args.max_candidates or len(set(selected)) != len(selected):
            raise AssertionError(f"candidate slate is not exact unique K: {uid}")

        target_relation, target_decision, target_metric = _target_for_slate(
            truth_row, selected, families
        )
        confidence, defaulted = _confidence(truth_row)
        default_confidences += int(defaulted)
        assistant, spans = structured_target(
            decision=target_decision,
            metric_id=target_metric,
            confidence=confidence,
            reason=_reason(truth_row),
        )
        target_counts[target_relation] += 1
        orderings = [("retrieval_order", selected)]
        if split == "train":
            permutation = sorted(
                selected,
                key=lambda metric_id: (
                    _stable_digest(args.order_seed, "option-order", uid, metric_id),
                    metric_id,
                ),
            )
            if len(permutation) > 1 and permutation == selected:
                permutation = selected[1:] + selected[:1]
            orderings.append(("sha256_permutation", permutation))
        for view, ordered_ids in orderings:
            rendered = build_item_prompt(
                prompt,
                norm,
                [{"metric_id": metric_id} for metric_id in ordered_ids],
                metric_by_id,
                context_chars=args.context_chars,
                description_chars=args.description_chars,
                example_chars=args.example_chars,
                max_examples=args.max_examples,
            )
            buckets[split].append(
                {
                    "schema_version": SCHEMA,
                    "task": args.task,
                    "corpus": str(norm["corpus"]),
                    "norm_uid": uid,
                    "source_group": str(truth_row["source_group"]),
                    "split": split,
                    "view": view,
                    "gradient_eligible": split == "train"
                    and truth_row.get("gradient_eligible") is not False,
                    "truth_decision": str(truth_row["decision"]),
                    "target_relation": target_relation,
                    "decision": target_decision,
                    "metric_id": target_metric,
                    "candidate_metric_ids": ordered_ids,
                    "candidate_provenance": {
                        metric_id: provenance[uid][metric_id] for metric_id in ordered_ids
                    },
                    "current_bank_source_sha256": bank_hash,
                    "frozen_hierarchy_sha256": sha256_file(hierarchy_path),
                    "target_field_char_spans": spans,
                    "messages": [
                        {"role": "user", "content": rendered},
                        {"role": "assistant", "content": assistant},
                    ],
                }
            )

    group_splits: dict[str, set[str]] = defaultdict(set)
    uid_splits: dict[str, set[str]] = defaultdict(set)
    for split, rows in buckets.items():
        for row in rows:
            group_splits[str(row["source_group"])].add(split)
            uid_splits[str(row["norm_uid"])].add(split)
            if (split == "train") != bool(row["gradient_eligible"]):
                # Explicit train exclusions are allowed, but held-out gradients never are.
                if split != "train":
                    raise AssertionError("held-out example became gradient eligible")
    crossed_groups = {key: value for key, value in group_splits.items() if len(value) > 1}
    crossed_uids = {key: value for key, value in uid_splits.items() if len(value) > 1}
    if crossed_groups or crossed_uids:
        raise ValueError(
            "source-disjoint output audit failed: "
            f"groups={len(crossed_groups)} uids={len(crossed_uids)}"
        )
    train_views = Counter(row["norm_uid"] for row in buckets["train"])
    heldout_views = Counter(
        row["norm_uid"]
        for split in ("dev", "test", "blind")
        for row in buckets[split]
    )
    if any(value != 2 for value in train_views.values()) or any(
        value != 1 for value in heldout_views.values()
    ):
        raise AssertionError("train/held-out option-order view contract failed")
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "FROZEN_SOURCE_DISJOINT_TYPED_DATASET_READY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "bank_source_sha256": bank_hash,
        "bank_metric_count": len(metrics),
        "max_candidates": args.max_candidates,
        "truth_rows": len(truth_rows),
        "truth_split_counts": dict(sorted(Counter(row["split"] for row in truth_rows).items())),
        "truth_decision_counts": dict(
            sorted(Counter(str(row["decision"]) for row in truth_rows).items())
        ),
        "example_split_counts": {split: len(rows) for split, rows in buckets.items()},
        "target_relation_truth_row_counts": dict(sorted(target_counts.items())),
        "train_candidate_injections": dict(sorted(injections.items())),
        "candidate_injections_outside_train": 0,
        "confidence_defaults_to_medium": default_confidences,
        "train_views_per_norm": 2,
        "heldout_views_per_norm": 1,
        "source_groups_crossing_splits": 0,
        "norm_uids_crossing_splits": 0,
        "test_or_blind_gradient_eligible": 0,
        "test_or_blind_derived_candidates": 0,
        "heldout_candidate_order_uses_truth": False,
        "candidate_audit": candidate_audit,
        "split_assignment_audit": assignment_audit,
        "truth_join_audit": truth_audit,
        "inputs": {
            "manifest": _ref(manifest_path),
            "bank": _ref(bank_path),
            "hierarchy": _ref(hierarchy_path),
            "prompt": _ref(prompt_path),
            "truth": [_ref(path) for path in truth_paths],
            "split_assignments": _ref(assignments_path) if assignments_path else None,
            "candidate_lanes": {lane: _ref(path) for lane, path in specs},
        },
        "training_contract": {
            "gradient_split": "train",
            "selection_split": "dev",
            "held_out_splits": ["test", "blind"],
            "no_test_or_blind_derived_augmentation": True,
            "teacher_forced_fields": ["decision", "metric_id", "confidence", "reason"],
        },
    }
    return buckets, report


def write_release(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report).resolve()
    outputs = {split: output_dir / f"{split}.jsonl" for split in OUTPUT_SPLITS}
    if output_dir.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite a frozen Gemma typed release")
    buckets, report = build(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_refs: dict[str, dict[str, Any] | None] = {}
    for split, path in outputs.items():
        rows = buckets[split]
        if not rows:
            output_refs[split] = None
            continue
        write_jsonl(path, rows)
        output_refs[split] = {**_ref(path), "count": len(rows)}
    report["outputs"] = output_refs
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return {**report, "report": _ref(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--truth", action="append", required=True)
    parser.add_argument("--split-assignments")
    parser.add_argument("--candidates", action="append", required=True, metavar="LANE=PATH")
    parser.add_argument("--hierarchy", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--order-seed", type=int, default=2026071401)
    parser.add_argument("--context-chars", type=int, default=1400)
    parser.add_argument("--description-chars", type=int, default=520)
    parser.add_argument("--example-chars", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if min(
        args.max_candidates,
        args.context_chars,
        args.description_chars,
        args.example_chars,
        args.max_examples,
    ) <= 0:
        parser.error("candidate/prompt size fields must be positive")
    return args


def main() -> None:
    report = write_release(parse_args())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
