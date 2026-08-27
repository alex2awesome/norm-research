#!/usr/bin/env python3
"""Stream-audit complete silver-match outputs and report every decision rate.

The production finalizer writes one canonical-order JSONL per corpus.  This
audit deliberately compares those files row-for-row with the frozen manifest,
so a matching total count cannot hide duplicated, omitted, or misrouted UIDs.
It also reports zero-valued categories: consumers should never have to infer
whether an absent decision was impossible, forgotten, or simply unobserved.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Mapping

from .adjudicate_gemma import CONFIDENCES
from .common import read_jsonl, sha256_file
from .finalize_adjudications import FINAL_DECISIONS


DECISION_ORDER = (
    "MATCH",
    "MATCH_FAMILY_ONLY",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
    "UNSTABLE_MATCH",
    "INVALID_OUTPUT",
)

if set(DECISION_ORDER) != FINAL_DECISIONS:  # fail loudly when taxonomy changes
    raise RuntimeError(
        f"audit taxonomy drift: missing={FINAL_DECISIONS - set(DECISION_ORDER)}, "
        f"extra={set(DECISION_ORDER) - FINAL_DECISIONS}"
    )


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else anchor.parent / value


def _peek(path: Path) -> dict[str, Any]:
    iterator = read_jsonl(path)
    try:
        return next(iterator)
    except StopIteration as exc:
        raise ValueError(f"empty final output: {path}") from exc


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _complete_counts(counter: Mapping[str, int]) -> dict[str, int]:
    return {decision: int(counter.get(decision, 0)) for decision in DECISION_ORDER}


def _summary(
    counts: Mapping[str, int],
    total: int,
    *,
    confidence: Mapping[str, int] | None = None,
    verification: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    completed = _complete_counts(counts)
    result: dict[str, Any] = {
        "count": total,
        "decision_counts": completed,
        "decision_rates": {
            decision: _rate(count, total) for decision, count in completed.items()
        },
        "rollup_rates": {
            "verified_exact_match": _rate(completed["MATCH"], total),
            "family_only": _rate(completed["MATCH_FAMILY_ONLY"], total),
            "no_bank_candidate_fits": _rate(completed["NO_CANDIDATE_FITS"], total),
            "noise": _rate(completed["NOISE"], total),
            "verification_failed": _rate(completed["UNSTABLE_MATCH"], total),
            "invalid_output": _rate(completed["INVALID_OUTPUT"], total),
            "not_verified_exact_match": _rate(total - completed["MATCH"], total),
        },
    }
    if confidence is not None:
        result["confidence_counts"] = {
            value: int(confidence.get(value, 0)) for value in sorted(CONFIDENCES)
        }
    if verification is not None:
        result["verification_status_counts"] = dict(sorted(verification.items()))
    return result


def _macro_summary(
    counts_by_group: Mapping[str, Mapping[str, int]],
    totals_by_group: Mapping[str, int],
) -> dict[str, Any]:
    groups = sorted(group for group, total in totals_by_group.items() if total)
    rates = {
        decision: [
            counts_by_group[group].get(decision, 0) / totals_by_group[group]
            for group in groups
        ]
        for decision in DECISION_ORDER
    }
    return {
        "groups": len(groups),
        "group_names": groups,
        "decision_rate_macro_mean": {
            decision: sum(values) / len(values) if values else None
            for decision, values in rates.items()
        },
        "decision_rate_range": {
            decision: {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for decision, values in rates.items()
        },
    }


def audit_outputs(
    manifest_path: Path,
    final_paths: Iterable[Path],
    *,
    tasks: set[str] | None = None,
    corpora: set[str] | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_corpus_meta = manifest.get("corpora", {})
    bank_meta = manifest.get("banks", {})
    if not all_corpus_meta or not bank_meta:
        raise ValueError("manifest lacks corpora or banks")
    unknown_tasks = (tasks or set()) - set(bank_meta)
    unknown_corpora = (corpora or set()) - set(all_corpus_meta)
    if unknown_tasks or unknown_corpora:
        raise ValueError(
            f"unknown audit scope: tasks={sorted(unknown_tasks)}, "
            f"corpora={sorted(unknown_corpora)}"
        )
    corpus_meta = {
        corpus: meta
        for corpus, meta in all_corpus_meta.items()
        if (not tasks or meta["task"] in tasks)
        and (not corpora or corpus in corpora)
    }
    if not corpus_meta:
        raise ValueError("audit scope selects no corpora")

    paths_by_corpus: dict[str, Path] = {}
    for path in final_paths:
        path = path.resolve()
        corpus = str(_peek(path).get("corpus") or "")
        if corpus not in corpus_meta:
            raise ValueError(f"unknown/empty corpus {corpus!r} in {path}")
        if corpus in paths_by_corpus:
            raise ValueError(
                f"multiple final files for corpus {corpus}: "
                f"{paths_by_corpus[corpus]} and {path}"
            )
        paths_by_corpus[corpus] = path
    missing_corpora = set(corpus_meta) - set(paths_by_corpus)
    if missing_corpora:
        raise ValueError(f"missing final corpora: {sorted(missing_corpora)}")

    selected_tasks = {str(meta["task"]) for meta in corpus_meta.values()}
    banks: dict[str, set[str]] = {}
    for task in sorted(selected_tasks):
        meta = bank_meta[task]
        payload = json.loads(_resolve(meta["path"], manifest_path).read_text(encoding="utf-8"))
        banks[task] = {str(row["metric_id"]) for row in payload["metrics"]}

    overall = Counter()
    overall_confidence = Counter()
    overall_verification = Counter()
    task_counts: dict[str, Counter] = defaultdict(Counter)
    task_confidence: dict[str, Counter] = defaultdict(Counter)
    task_verification: dict[str, Counter] = defaultdict(Counter)
    corpus_counts: dict[str, Counter] = defaultdict(Counter)
    corpus_confidence: dict[str, Counter] = defaultdict(Counter)
    corpus_verification: dict[str, Counter] = defaultdict(Counter)
    metric_counts: dict[str, Counter] = defaultdict(Counter)
    stratum_counts: dict[str, Counter] = defaultdict(Counter)
    stratum_totals = Counter()
    task_totals = Counter()
    corpus_totals = Counter()
    input_hashes: dict[str, str] = {}

    sentinel = object()
    for corpus in sorted(corpus_meta):
        meta = corpus_meta[corpus]
        task = str(meta["task"])
        expected_path = _resolve(meta["path"], manifest_path)
        final_path = paths_by_corpus[corpus]
        input_hashes[str(final_path)] = sha256_file(final_path)
        count = 0
        for position, pair in enumerate(
            zip_longest(read_jsonl(expected_path), read_jsonl(final_path), fillvalue=sentinel),
            1,
        ):
            expected, final = pair
            if expected is sentinel or final is sentinel:
                raise ValueError(
                    f"row-count mismatch for {corpus} at canonical position {position}"
                )
            uid = str(expected.get("norm_uid") or "")
            if str(final.get("norm_uid") or "") != uid:
                raise ValueError(
                    f"UID/order mismatch for {corpus} at {position}: "
                    f"{uid} != {final.get('norm_uid')}"
                )
            if final.get("corpus") != corpus or final.get("task") != task:
                raise ValueError(f"routing mismatch for {uid}")
            if final.get("row") != expected.get("row"):
                raise ValueError(f"canonical row mismatch for {uid}")
            if str(final.get("bank_source_sha256") or "") != str(
                bank_meta[task]["source_sha256"]
            ):
                raise ValueError(f"bank provenance mismatch for {uid}")
            decision = str(final.get("decision") or "")
            if decision not in FINAL_DECISIONS:
                raise ValueError(f"unknown decision {decision!r} for {uid}")
            metric_id = final.get("metric_id")
            if decision == "MATCH":
                if str(metric_id or "") not in banks[task]:
                    raise ValueError(f"MATCH metric outside {task} bank for {uid}: {metric_id}")
                metric_counts[task][str(metric_id)] += 1
            elif metric_id is not None:
                raise ValueError(f"non-MATCH decision carries metric_id for {uid}")
            confidence = str(final.get("confidence") or "")
            if confidence not in CONFIDENCES:
                raise ValueError(f"invalid confidence {confidence!r} for {uid}")
            verification_status = str(final.get("verification_status") or "")
            if not verification_status:
                raise ValueError(f"missing verification_status for {uid}")
            overall[decision] += 1
            overall_confidence[confidence] += 1
            overall_verification[verification_status] += 1
            task_counts[task][decision] += 1
            task_confidence[task][confidence] += 1
            task_verification[task][verification_status] += 1
            corpus_counts[corpus][decision] += 1
            corpus_confidence[corpus][confidence] += 1
            corpus_verification[corpus][verification_status] += 1
            for field in ("polarity", "kind", "extraction_valid"):
                value = expected.get(field)
                value = "UNKNOWN" if value is None or value == "" else str(value)
                stratum = f"{field}:{value}"
                stratum_counts[stratum][decision] += 1
                stratum_totals[stratum] += 1
            task_totals[task] += 1
            corpus_totals[corpus] += 1
            count += 1
        if count != int(meta["count"]):
            raise ValueError(f"manifest count mismatch for {corpus}: {count} != {meta['count']}")

    total = sum(corpus_totals.values())
    expected_total = sum(int(meta["count"]) for meta in corpus_meta.values())
    if total != expected_total:
        raise ValueError(f"global count mismatch: {total} != {expected_total}")
    return {
        "schema_version": "silver-match-v3-final-audit-v1",
        "complete": True,
        "global_complete": set(corpus_meta) == set(all_corpus_meta),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "expected_rows": expected_total,
        "audited_rows": total,
        "corpora_expected": len(corpus_meta),
        "global_corpora_in_manifest": len(all_corpus_meta),
        "corpora_audited": len(paths_by_corpus),
        "scope": {
            "tasks": sorted(tasks) if tasks else None,
            "corpora": sorted(corpora) if corpora else None,
        },
        "overall": _summary(
            overall,
            total,
            confidence=overall_confidence,
            verification=overall_verification,
        ),
        "by_task": {
            task: {
                **_summary(
                    task_counts[task],
                    task_totals[task],
                    confidence=task_confidence[task],
                    verification=task_verification[task],
                ),
                "matched_metric_coverage": {
                    "count": len(metric_counts[task]),
                    "bank_count": len(banks[task]),
                    "rate": _rate(len(metric_counts[task]), len(banks[task])),
                    "match_counts_by_metric": dict(sorted(metric_counts[task].items())),
                },
            }
            for task in sorted(task_totals)
        },
        "by_corpus": {
            corpus: _summary(
                corpus_counts[corpus],
                corpus_totals[corpus],
                confidence=corpus_confidence[corpus],
                verification=corpus_verification[corpus],
            )
            for corpus in sorted(corpus_totals)
        },
        "macro_over_tasks": _macro_summary(task_counts, task_totals),
        "macro_over_corpora": _macro_summary(corpus_counts, corpus_totals),
        "by_norm_stratum": {
            stratum: _summary(stratum_counts[stratum], stratum_totals[stratum])
            for stratum in sorted(stratum_totals)
        },
        "input_hashes": dict(sorted(input_hashes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--final", action="append", required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    report = audit_outputs(
        Path(args.manifest).resolve(),
        [Path(path) for path in args.final],
        tasks=set(args.task) or None,
        corpora=set(args.corpus) or None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "complete": report["complete"],
        "audited_rows": report["audited_rows"],
        "corpora_audited": report["corpora_audited"],
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
