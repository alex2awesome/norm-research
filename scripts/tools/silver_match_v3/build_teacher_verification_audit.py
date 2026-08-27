#!/usr/bin/env python3
"""Build a deterministic, stratified packet for independent teacher audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .adjudicate_gemma import load_inputs
from .common import read_jsonl, sha256_file, write_jsonl
from .finalize_teacher_verifications import index_unique


def audit_metric_card(metric: dict[str, Any]) -> dict[str, Any]:
    description = str(metric.get("description") or "")
    examples = [str(value) for value in (metric.get("examples") or [])[:2]]
    return {
        "metric_id": metric["metric_id"],
        "name": metric.get("name"),
        "description": description[:700],
        "examples": [value[:240] for value in examples],
    }


def outcome(proposed: str, left: dict[str, Any], right: dict[str, Any]) -> str:
    confirms = [
        row.get("decision") == "CONFIRM_MATCH" and row.get("metric_id") == proposed
        for row in (left, right)
    ]
    confidences = {str(left.get("confidence")), str(right.get("confidence"))}
    if all(confirms) and confidences <= {"high", "medium"} and "high" in confidences:
        return "retained"
    if (
        left.get("decision") == right.get("decision") == "BETTER_CANDIDATE"
        and left.get("metric_id")
        and left.get("metric_id") == right.get("metric_id")
    ):
        return "stable_correction"
    return "other_rejection"


def stable_key(seed: int, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}\x1f{row['norm_uid']}".encode()).hexdigest()


def diverse_take(rows: Sequence[dict[str, Any]], count: int, seed: int) -> list[dict]:
    """Prefer unseen source buckets and metrics, then fill deterministically."""
    pending = sorted(rows, key=lambda row: stable_key(seed, row))
    selected: list[dict] = []
    seen_sources: Counter[str] = Counter()
    seen_metrics: Counter[str] = Counter()
    while pending and len(selected) < count:
        pending.sort(
            key=lambda row: (
                seen_sources[row["source_bucket"]],
                seen_metrics[row["proposed_metric_id"]],
                stable_key(seed, row),
            )
        )
        row = pending.pop(0)
        selected.append(row)
        seen_sources[row["source_bucket"]] += 1
        seen_metrics[row["proposed_metric_id"]] += 1
    return selected


def blind_packet(rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Remove machine outcomes/strata so the independent auditor stays blind."""
    allowed = (
        "norm_uid",
        "task",
        "corpus",
        "source_id",
        "source_bucket",
        "norm",
        "context",
        "proposed_metric_id",
        "proposed_metric",
        "retrieved_alternative_metrics",
        "manual_decision",
        "manual_metric_id",
        "manual_reason",
        "auditor",
    )
    output = [{key: row.get(key) for key in allowed} for row in rows]
    return sorted(output, key=lambda row: stable_key(seed + 1, row))


def build_packet(
    proposals: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    first: Sequence[dict[str, Any]],
    second: Sequence[dict[str, Any]],
    norms_by_corpus: dict[str, dict[str, Any]],
    banks: dict[str, dict[str, dict[str, Any]]],
    *,
    per_stratum: int,
    rare_max_count: int,
    seed: int,
    max_alternatives: int = 12,
    retained_per_stratum: int | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    proposal_by_uid = index_unique(proposals, "proposals")
    candidate_by_uid = index_unique(candidates, "candidates")
    first_by_uid = index_unique(first, "first verification")
    second_by_uid = index_unique(second, "second verification")
    keys = proposal_by_uid.keys()
    if keys != candidate_by_uid.keys() or keys != first_by_uid.keys() or keys != second_by_uid.keys():
        raise ValueError("audit inputs have different UID sets")
    metric_frequency = Counter(str(row["metric_id"]) for row in proposals)
    population: list[dict] = []
    for uid, proposal in proposal_by_uid.items():
        task, corpus = str(proposal["task"]), str(proposal["corpus"])
        norm = norms_by_corpus[corpus][uid]
        left, right = first_by_uid[uid], second_by_uid[uid]
        candidate_values = list(candidate_by_uid[uid].get("candidates") or [])
        if not candidate_values:
            raise ValueError(f"empty candidate slate: {uid}")
        injected = bool(
            candidate_by_uid[uid].get("primary_was_injected")
            or candidate_values[0].get("injected_primary")
        )
        proposed = str(proposal["metric_id"])
        correction_ids = sorted(
            {
                str(row["metric_id"])
                for row in (left, right)
                if row.get("decision") == "BETTER_CANDIDATE" and row.get("metric_id")
            }
        )
        alternative_ids = [
            str(value["metric_id"])
            for value in candidate_values
            if str(value["metric_id"]) != proposed
        ][:max_alternatives]
        source_id = str(norm.get("source_id") or corpus)
        source_bucket = source_id.split("__", 1)[0]
        status = outcome(proposed, left, right)
        rarity = "rare" if metric_frequency[proposed] <= rare_max_count else "common"
        population.append(
            {
                "norm_uid": uid,
                "task": task,
                "corpus": corpus,
                "source_id": source_id,
                "source_bucket": source_bucket,
                "norm": norm.get("norm"),
                "context": str(norm.get("context") or "")[:2400],
                "proposed_metric_id": proposed,
                "proposed_metric": audit_metric_card(banks[task][proposed]),
                "proposal_frequency": metric_frequency[proposed],
                "metric_rarity": rarity,
                "proposal_retrieval_status": (
                    "injected_for_verification" if injected else "natural_top_k"
                ),
                "gemma_outcome": status,
                "first_verification": left,
                "second_verification": right,
                "correction_metrics": [
                    audit_metric_card(banks[task][metric_id])
                    for metric_id in correction_ids
                ],
                "retrieved_alternative_metrics": [
                    audit_metric_card(banks[task][metric_id])
                    for metric_id in alternative_ids
                ],
                "manual_decision": None,
                "manual_metric_id": None,
                "manual_reason": None,
                "auditor": None,
            }
        )
    strata: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in population:
        strata[
            (
                row["gemma_outcome"],
                row["proposal_retrieval_status"],
                row["metric_rarity"],
            )
        ].append(row)
    selected = []
    for stratum, rows in sorted(strata.items()):
        quota = (
            retained_per_stratum
            if stratum[0] == "retained" and retained_per_stratum is not None
            else per_stratum
        )
        chosen = diverse_take(rows, quota, seed)
        for row in chosen:
            row["audit_stratum"] = list(stratum)
            row["audit_stratum_population_n"] = len(rows)
            row["audit_stratum_sample_n"] = len(chosen)
            row["audit_design_weight"] = len(rows) / len(chosen)
            selected.append(row)
    selected.sort(key=lambda row: (row["audit_stratum"], stable_key(seed, row)))
    report = {
        "population": len(population),
        "selected": len(selected),
        "per_stratum": per_stratum,
        "retained_per_stratum": retained_per_stratum,
        "rare_max_count": rare_max_count,
        "seed": seed,
        "max_alternatives": max_alternatives,
        "population_strata": {
            "|".join(key): len(rows) for key, rows in sorted(strata.items())
        },
        "selected_strata": dict(
            sorted(Counter("|".join(row["audit_stratum"]) for row in selected).items())
        ),
        "selected_source_buckets": dict(
            sorted(Counter(row["source_bucket"] for row in selected).items())
        ),
        "sampling_design": (
            "equal allocation over outcome x retrieval-status x metric-rarity; "
            "within-stratum purposive diversity over source bucket and metric"
        ),
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--blind-output",
        help="optional auditor-facing packet with machine outcomes and strata removed",
    )
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--retained-per-stratum", type=int, default=12)
    parser.add_argument("--rare-max-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=739391)
    parser.add_argument("--max-alternatives", type=int, default=12)
    args = parser.parse_args()
    paths = {
        key: Path(getattr(args, key)).resolve()
        for key in ("manifest", "proposals", "candidates", "first", "second")
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    _, _, norms_by_corpus, banks = load_inputs(paths["manifest"], paths["candidates"])
    rows, report = build_packet(
        list(read_jsonl(paths["proposals"])),
        list(read_jsonl(paths["candidates"])),
        list(read_jsonl(paths["first"])),
        list(read_jsonl(paths["second"])),
        norms_by_corpus,
        banks,
        per_stratum=args.per_stratum,
        rare_max_count=args.rare_max_count,
        seed=args.seed,
        max_alternatives=args.max_alternatives,
        retained_per_stratum=args.retained_per_stratum,
    )
    write_jsonl(output, rows)
    report["input_hashes"] = {key: sha256_file(path) for key, path in paths.items()}
    report["output_sha256"] = sha256_file(output)
    if args.blind_output:
        blind_output = Path(args.blind_output).resolve()
        if blind_output.exists():
            raise FileExistsError(f"refusing to overwrite {blind_output}")
        write_jsonl(blind_output, blind_packet(rows, args.seed))
        report["blind_output"] = str(blind_output)
        report["blind_output_sha256"] = sha256_file(blind_output)
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
