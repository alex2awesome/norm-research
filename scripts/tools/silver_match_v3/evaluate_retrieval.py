#!/usr/bin/env python3
"""Evaluate retrieval with exact, strict-equivalence, and R3-family recall.

Family recall is deliberately reported alongside—not instead of—exact recall.
It is a diagnostic upper-bound-style measure because R3 grandparent families
contain related but distinct metrics.  Strict equivalence receives credit only
from the independently qualified groups emitted by ``build_relations.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_jsonl
from .config import DEFAULT_OUTPUT_ROOT


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def load_unique(paths: list[Path], key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            value = str(row[key])
            if value in rows:
                raise ValueError(f"duplicate {key}={value} across candidate inputs")
            rows[value] = row
    return rows


def candidate_metric_ids(row: dict[str, Any]) -> list[str]:
    candidates = row.get("candidates") or []
    ids = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            value = candidate.get("metric_id")
        else:
            value = candidate
        if value is not None:
            ids.append(str(value))
    return ids


def first_rank(candidate_ids: list[str], accepted: set[str]) -> int | None:
    for rank, metric in enumerate(candidate_ids, 1):
        if metric in accepted:
            return rank
    return None


def _teacher_bank_sha(row: dict[str, Any]) -> str | None:
    value = row.get("current_bank_source_sha256", row.get("bank_source_sha256"))
    return str(value) if value else None


def validate_provenance(
    teacher: dict[str, Any],
    candidate: dict[str, Any],
    task_relations: dict[str, Any],
) -> None:
    expected = str(task_relations["bank_source_sha256"])
    candidate_sha = candidate.get("bank_source_sha256")
    if not candidate_sha:
        raise ValueError(f"candidate {teacher['norm_uid']} lacks bank_source_sha256")
    if str(candidate_sha) != expected:
        raise ValueError(
            f"candidate bank mismatch for {teacher['norm_uid']}: "
            f"{candidate_sha} != {expected}"
        )
    teacher_sha = _teacher_bank_sha(teacher)
    if not teacher_sha:
        raise ValueError(f"teacher {teacher['norm_uid']} lacks current bank provenance")
    if teacher_sha != expected:
        raise ValueError(
            f"teacher bank mismatch for {teacher['norm_uid']}: "
            f"{teacher_sha} != {expected}"
        )


def evaluate(
    teachers: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    relations: dict[str, Any],
    ks: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ks = sorted(set(ks))
    if not ks or ks[0] < 1:
        raise ValueError("ks must contain positive integers")
    task_specs = relations.get("tasks") or {}
    teacher_uids = [str(row["norm_uid"]) for row in teachers]
    if len(teacher_uids) != len(set(teacher_uids)):
        raise ValueError("duplicate norm_uid in teacher inputs")
    match_teachers = [row for row in teachers if row.get("decision") == "MATCH"]
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    errors = []

    for teacher in match_teachers:
        uid = str(teacher["norm_uid"])
        task = str(teacher.get("task") or "")
        corpus = str(teacher.get("corpus") or "")
        truth = str(teacher.get("metric_id") or "")
        if task not in task_specs:
            raise ValueError(f"teacher {uid} uses task absent from relations: {task}")
        task_spec = task_specs[task]
        metric_relations = task_spec.get("metric_relations") or {}
        if truth not in metric_relations:
            raise ValueError(f"teacher {uid} uses metric absent from relations: {task}/{truth}")

        candidate = candidates.get(uid)
        ids: list[str] = []
        if candidate is not None:
            candidate_task = str(candidate.get("task") or task)
            if candidate_task != task:
                raise ValueError(
                    f"task mismatch for {uid}: teacher={task}, candidate={candidate_task}"
                )
            validate_provenance(teacher, candidate, task_spec)
            ids = candidate_metric_ids(candidate)
            unknown = sorted(set(ids).difference(metric_relations))
            if unknown:
                raise ValueError(
                    f"candidate {uid} contains IDs outside {task} bank: {unknown[:5]}"
                )

        relation = metric_relations[truth]
        exact = {truth}
        equivalent = set(relation.get("equivalent_metric_ids") or [truth])
        family = set(relation.get("family_metric_ids") or [truth])
        exact_rank = first_rank(ids, exact)
        equivalence_rank = first_rank(ids, equivalent)
        family_rank = first_rank(ids, family)
        strata = ("overall", f"task:{task}", f"corpus:{corpus}")
        for stratum in strata:
            counter = counters[stratum]
            counter["denominator"] += 1
            counter["candidate_present"] += int(candidate is not None)
            counter["strict_equivalence_available"] += int(len(equivalent) > 1)
            counter["nontrivial_family_available"] += int(len(family) > 1)
            for k in ks:
                counter[f"exact@{k}"] += int(exact_rank is not None and exact_rank <= k)
                counter[f"equivalence@{k}"] += int(
                    equivalence_rank is not None and equivalence_rank <= k
                )
                counter[f"family@{k}"] += int(family_rank is not None and family_rank <= k)
        if exact_rank is None or exact_rank > max(ks):
            errors.append(
                {
                    "norm_uid": uid,
                    "corpus": corpus,
                    "task": task,
                    "teacher_metric_id": truth,
                    "exact_rank": exact_rank,
                    "equivalence_rank": equivalence_rank,
                    "family_rank": family_rank,
                    "equivalent_metric_ids": sorted(equivalent),
                    "family_metric_ids": sorted(family),
                    "candidate_ids": ids,
                }
            )

    def render(counter: dict[str, int]) -> dict[str, Any]:
        den = counter.get("denominator", 0)
        result: dict[str, Any] = {
            "n_match": den,
            "candidate_coverage": safe_rate(counter.get("candidate_present", 0), den),
            "strict_equivalence_available_rate": safe_rate(
                counter.get("strict_equivalence_available", 0), den
            ),
            "nontrivial_family_available_rate": safe_rate(
                counter.get("nontrivial_family_available", 0), den
            ),
        }
        for k in ks:
            exact = safe_rate(counter.get(f"exact@{k}", 0), den)
            equivalent = safe_rate(counter.get(f"equivalence@{k}", 0), den)
            family = safe_rate(counter.get(f"family@{k}", 0), den)
            result[f"exact_recall_at_{k}"] = exact
            result[f"equivalence_recall_at_{k}"] = equivalent
            result[f"family_recall_at_{k}"] = family
            result[f"equivalence_credit_gain_at_{k}"] = (
                None if exact is None or equivalent is None else equivalent - exact
            )
            result[f"family_credit_gain_at_{k}"] = (
                None if exact is None or family is None else family - exact
            )
        return result

    report = {
        "relation_schema_version": relations.get("relation_schema_version"),
        "pair_labels_sha256": relations.get("pair_labels_sha256"),
        "ks": ks,
        "n_teachers": len(teachers),
        "n_match_teachers": len(match_teachers),
        "overall": render(counters["overall"]),
        "by_task": {
            key.split(":", 1)[1]: render(value)
            for key, value in sorted(counters.items())
            if key.startswith("task:")
        },
        "by_corpus": {
            key.split(":", 1)[1]: render(value)
            for key, value in sorted(counters.items())
            if key.startswith("corpus:")
        },
        "interpretation": {
            "exact": "candidate ID equals teacher metric ID",
            "equivalence": "candidate is in a strict pair-label-validated R3 same-construct group",
            "family": "candidate shares an R3 merge or subsumption family; related metrics may remain distinct",
        },
    }
    return report, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument(
        "--relations", default=str(DEFAULT_OUTPUT_ROOT / "relations.json")
    )
    parser.add_argument("--split", choices=("all", "train", "dev", "test"), default="all")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 16])
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors-output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teachers = [
        row
        for row in read_jsonl(Path(args.teachers))
        if args.split == "all" or row.get("split") == args.split
    ]
    candidates = load_unique([Path(path) for path in args.candidates], "norm_uid")
    relations = json.loads(Path(args.relations).read_text(encoding="utf-8"))
    report, errors = evaluate(teachers, candidates, relations, args.ks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.errors_output:
        write_jsonl(Path(args.errors_output), errors)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
