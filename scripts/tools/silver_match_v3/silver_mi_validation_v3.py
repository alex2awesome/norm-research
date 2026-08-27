#!/usr/bin/env python3
"""Correlate frozen v3 silver salience with label-free per-metric MI scores.

Unlike the legacy top-k forced-match analysis, this consumes only independently
verified exact ``MATCH`` decisions from a task analysis release.  Typed
abstentions remain in the denominator and their blind-audit risk is carried
into the result.  The primary salience measure is unique source-group presence;
raw norm counts and equal-corpus shares are reported as sensitivities.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from .common import normalize_name, read_jsonl, sha256_file


METRIC_INDEX_RE = re.compile(r"(?:^|_)metric(\d+)(?:_|\.|$)")
SCORE_FIELDS = {
    "OPT": "opt_omega_bits",
    "T_HM": "H_M",
    "EPS_ADV": "eps_bits_adv",
}


def _check_artifact(value: dict[str, Any], label: str) -> Path:
    path = Path(str(value.get("path") or ""))
    if not path.exists() or sha256_file(path) != str(value.get("sha256") or ""):
        raise ValueError(f"released artifact changed: {label}={path}")
    return path


def _load_release(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    release = json.loads(path.read_text(encoding="utf-8"))
    if (
        release.get("status") != "TASK_FROZEN_ANALYSIS_READY"
        or not (release.get("analysis_firewall") or {}).get("task_matcher_is_immutable")
        or (release.get("analysis_firewall") or {}).get(
            "may_tune_this_or_other_task_matchers_from_results"
        )
        is not False
    ):
        raise ValueError("task is not frozen behind the analysis firewall")
    manifest_path = _check_artifact(release["manifest"], "manifest")
    for key in ("production_plan", "final_audit", "blind_risk_audit"):
        _check_artifact(release[key], key)
    for index, artifact in enumerate(release.get("final_outputs") or []):
        _check_artifact(artifact, f"final_outputs[{index}]")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return release, manifest, manifest_path


def _load_certificate(
    certificate_path: Path, bank_rows: list[dict[str, Any]], task: str
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    payload = json.loads(certificate_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if payload.get("schema_version") != "silver-match-v3-extracted-mi-certificate-v1":
            raise ValueError("MI certificate schema is not the frozen extracted format")
        declared_tasks = {
            normalize_name(value)
            for value in (payload.get("task"), payload.get("source_task"))
            if value
        }
        if declared_tasks and declared_tasks != {normalize_name(task)}:
            raise ValueError(
                f"MI certificate task differs from released task: {sorted(declared_tasks)}"
            )
    rows = payload if isinstance(payload, list) else payload.get("table", [])
    bank_by_index = {int(row["metric_index"]): row for row in bank_rows}
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bank_rows:
        by_name[normalize_name(row["name"])].append(row)
    scores: dict[str, dict[str, float]] = {}
    join_counts = Counter()
    unmatched = []
    for row in rows:
        if not isinstance(row, dict) or row.get("opt_omega_bits") is None:
            continue
        target = None
        match = METRIC_INDEX_RE.search(Path(str(row.get("file") or "")).name)
        if match:
            target = bank_by_index.get(int(match.group(1)))
            if target and normalize_name(target["name"]) != normalize_name(row.get("name")):
                raise ValueError(
                    f"certificate metric index/name mismatch: {row.get('file')}"
                )
            join_counts["metric_index"] += bool(target)
        if target is None:
            candidates = by_name.get(normalize_name(row.get("name")), [])
            if len(candidates) == 1:
                target = candidates[0]
                join_counts["unique_name"] += 1
        if target is None:
            unmatched.append({"file": row.get("file"), "name": row.get("name")})
            continue
        metric_id = str(target["metric_id"])
        if metric_id in scores:
            raise ValueError(f"duplicate certificate mapping for {metric_id}")
        current = {
            label: float(row[field])
            for label, field in SCORE_FIELDS.items()
            if row.get(field) is not None
        }
        gains = row.get("gains") or []
        if gains:
            current["G1"] = float(gains[0])
        scores[metric_id] = current
    if len(scores) < 4:
        raise ValueError("fewer than four bank metrics have MI certificates")
    return scores, {
        "certificate_rows": len(rows),
        "joined_metrics": len(scores),
        "bank_metrics": len(bank_rows),
        "bank_coverage": len(scores) / len(bank_rows),
        "join_counts": dict(sorted(join_counts.items())),
        "unmatched_count": len(unmatched),
        "unmatched_sample": unmatched[:10],
        "declared_task": task,
    }


def _load_events(
    release: dict[str, Any], manifest: dict[str, Any], manifest_path: Path
) -> tuple[
    list[dict[str, str]],
    Counter[str],
    Counter[str],
    Counter[str],
    dict[str, set[str]],
]:
    final_by_corpus: dict[str, Path] = {}
    for artifact in release["final_outputs"]:
        path = Path(artifact["path"])
        first = next(read_jsonl(path))
        corpus = str(first.get("corpus") or "")
        if corpus in final_by_corpus:
            raise ValueError(f"duplicate final corpus: {corpus}")
        final_by_corpus[corpus] = path
    events = []
    decisions: Counter[str] = Counter()
    analysis_decisions: Counter[str] = Counter()
    excluded_decisions: Counter[str] = Counter()
    excluded_uids = set((release.get("analysis_exclusions") or {}).get("norm_uids") or [])
    if len(excluded_uids) != int(
        (release.get("analysis_exclusions") or {}).get("count", -1)
    ):
        raise ValueError("released analysis exclusion count/UIDs differ")
    all_groups: dict[str, set[str]] = defaultdict(set)
    sentinel = object()
    for corpus in release["corpora"]:
        final_path = final_by_corpus.get(corpus)
        if final_path is None:
            raise ValueError(f"missing released final corpus: {corpus}")
        raw_norm_path = Path(manifest["corpora"][corpus]["path"])
        if not raw_norm_path.is_absolute():
            raw_norm_path = manifest_path.parent / raw_norm_path
        for norm, final in zip_longest(
            read_jsonl(raw_norm_path), read_jsonl(final_path), fillvalue=sentinel
        ):
            if norm is sentinel or final is sentinel:
                raise ValueError(f"released final coverage changed: {corpus}")
            if norm.get("norm_uid") != final.get("norm_uid"):
                raise ValueError(f"released UID order changed: {corpus}")
            decision = str(final.get("decision") or "")
            decisions[decision] += 1
            if str(norm["norm_uid"]) in excluded_uids:
                excluded_decisions[decision] += 1
                continue
            analysis_decisions[decision] += 1
            source_id = str(norm.get("source_id") or norm.get("paper_id") or norm["norm_uid"])
            group = f"{corpus}\x1f{source_id}"
            all_groups[corpus].add(group)
            if decision != "MATCH":
                continue
            events.append(
                {
                    "metric_id": str(final["metric_id"]),
                    "corpus": corpus,
                    "group": group,
                    "polarity": str(norm.get("polarity") or "UNKNOWN").lower(),
                    "kind": str(norm.get("kind") or "UNKNOWN"),
                }
            )
    if sum(decisions.values()) != int(release["expected_rows"]):
        raise ValueError("released decision count changed")
    return events, decisions, analysis_decisions, excluded_decisions, all_groups


def _salience_measures(
    events: list[dict[str, str]],
    metric_ids: list[str],
    corpora: list[str],
    all_groups: dict[str, set[str]],
) -> tuple[dict[str, np.ndarray], dict[str, set[str]]]:
    micro = Counter(event["metric_id"] for event in events)
    group_metrics: dict[str, set[str]] = defaultdict(set)
    for groups in all_groups.values():
        for group in groups:
            group_metrics[group]  # retain source groups with no exact MATCH
    corpus_counts: dict[str, Counter[str]] = defaultdict(Counter)
    polarity_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        group_metrics[event["group"]].add(event["metric_id"])
        corpus_counts[event["corpus"]][event["metric_id"]] += 1
        polarity_counts[event["polarity"]][event["metric_id"]] += 1
    presence = Counter()
    for values in group_metrics.values():
        presence.update(values)
    equal_corpus = Counter()
    for corpus in corpora:
        total = sum(corpus_counts[corpus].values())
        for metric_id in metric_ids:
            equal_corpus[metric_id] += (
                corpus_counts[corpus][metric_id] / total if total else 0.0
            ) / len(corpora)
    measures = {
        "source_presence": np.array([presence[mid] for mid in metric_ids], float),
        "micro_norm": np.array([micro[mid] for mid in metric_ids], float),
        "equal_corpus_share": np.array([equal_corpus[mid] for mid in metric_ids], float),
        "positive_micro_norm": np.array(
            [polarity_counts["positive"][mid] for mid in metric_ids], float
        ),
        "negative_micro_norm": np.array(
            [polarity_counts["negative"][mid] for mid in metric_ids], float
        ),
    }
    return measures, group_metrics


def _partial_spearman(y: np.ndarray, x: np.ndarray, covars: list[np.ndarray]) -> float | None:
    if np.all(y == y[0]) or np.all(x == x[0]):
        return None
    def residual(values: np.ndarray) -> np.ndarray:
        design = np.column_stack([rankdata(value) for value in covars] + [np.ones(len(values))])
        ranked = rankdata(values)
        beta, *_ = np.linalg.lstsq(design, ranked, rcond=None)
        return ranked - design @ beta
    return float(pearsonr(residual(y), residual(x)).statistic)


def _permutation_p(y: np.ndarray, x: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float | None, float | None]:
    if len(y) < 3 or np.all(y == y[0]) or np.all(x == x[0]):
        return None, None
    observed = spearmanr(y, x).statistic
    if not np.isfinite(observed):
        return None, None
    exceed = sum(
        abs(spearmanr(rng.permutation(y), x).statistic) >= abs(observed)
        for _ in range(n)
    )
    return float(observed), (exceed + 1) / (n + 1)


def _bootstrap_source_presence(
    group_metrics: dict[str, set[str]],
    metric_ids: list[str],
    score: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> list[float]:
    index = {metric_id: position for position, metric_id in enumerate(metric_ids)}
    by_corpus: dict[str, list[set[str]]] = defaultdict(list)
    for group, values in group_metrics.items():
        by_corpus[group.split("\x1f", 1)[0]].append(values)
    values = []
    for _ in range(n):
        salience = np.zeros(len(metric_ids), float)
        for groups in by_corpus.values():
            for sampled in rng.integers(0, len(groups), size=len(groups)):
                for metric_id in groups[int(sampled)]:
                    if metric_id in index:
                        salience[index[metric_id]] += 1
        if np.all(salience == salience[0]) or np.all(score == score[0]):
            continue
        rho = spearmanr(salience, score).statistic
        if np.isfinite(rho):
            values.append(float(rho))
    return values


def _split_half_source_reliability(
    group_metrics: dict[str, set[str]],
    metric_ids: list[str],
    n: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    index = {metric_id: position for position, metric_id in enumerate(metric_ids)}
    by_corpus: dict[str, list[set[str]]] = defaultdict(list)
    for group, values in group_metrics.items():
        by_corpus[group.split("\x1f", 1)[0]].append(values)
    corrected = []
    for _ in range(n):
        left = np.zeros(len(metric_ids), float)
        right = np.zeros(len(metric_ids), float)
        for groups in by_corpus.values():
            order = rng.permutation(len(groups))
            midpoint = len(order) // 2
            for target, positions in ((left, order[:midpoint]), (right, order[midpoint:])):
                for position in positions:
                    for metric_id in groups[int(position)]:
                        if metric_id in index:
                            target[index[metric_id]] += 1
        if np.all(left == left[0]) or np.all(right == right[0]):
            continue
        rho = spearmanr(left, right).statistic
        if np.isfinite(rho) and rho > -1:
            corrected.append(float(2 * rho / (1 + rho)))
    return {
        "replicates": len(corrected),
        "spearman_brown_median": float(np.median(corrected)) if corrected else None,
        "spearman_brown_95": (
            [float(np.quantile(corrected, .025)), float(np.quantile(corrected, .975))]
            if corrected
            else None
        ),
    }


def run_validation(
    *,
    release_path: Path,
    certificate_path: Path,
    n_permutations: int = 2000,
    n_bootstrap: int = 1000,
    seed: int = 1729,
) -> dict[str, Any]:
    release, manifest, manifest_path = _load_release(release_path)
    task = str(release["task"])
    bank_path = Path(manifest["banks"][task]["path"])
    if not bank_path.is_absolute():
        bank_path = manifest_path.parent / bank_path
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank_payload.get("source_sha256") != release["bank_source_sha256"]:
        raise ValueError("released bank source changed")
    bank_rows = bank_payload["metrics"]
    bank_by_id = {str(row["metric_id"]): row for row in bank_rows}
    certificate, join = _load_certificate(certificate_path, bank_rows, task)
    metric_ids = sorted(certificate, key=lambda value: int(value[1:]))
    (
        events,
        decisions,
        analysis_decisions,
        excluded_decisions,
        all_groups,
    ) = _load_events(release, manifest, manifest_path)
    measures, group_metrics = _salience_measures(
        events, metric_ids, release["corpora"], all_groups
    )
    total_matches = len(events)
    scored_match_mass = sum(event["metric_id"] in certificate for event in events)
    leaf_count = np.array([float(bank_by_id[mid].get("leaf_count") or 0) for mid in metric_ids])
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}
    score_names = sorted({name for values in certificate.values() for name in values})
    for measure_name, salience in measures.items():
        measure_result = {}
        for score_name in score_names:
            available = np.array([score_name in certificate[mid] for mid in metric_ids])
            mids = [mid for mid, keep in zip(metric_ids, available) if keep]
            y = salience[available]
            x = np.array([certificate[mid][score_name] for mid in mids], float)
            hm = np.array([certificate[mid].get("T_HM", np.nan) for mid in mids], float)
            leaves = leaf_count[available]
            rho, p = _permutation_p(y, x, n_permutations, rng)
            partial = None
            if score_name != "T_HM" and np.all(np.isfinite(hm)):
                partial = _partial_spearman(y, x, [np.log1p(leaves), hm])
            record: dict[str, Any] = {
                "n_metrics": len(mids),
                "spearman_rho": rho,
                "permutation_p_two_sided": p,
                "partial_rho_given_log_leaf_count_and_HM": partial,
                "nonzero_silver_metrics": int(np.count_nonzero(y)),
            }
            if measure_name == "source_presence" and rho is not None:
                boot = _bootstrap_source_presence(
                    group_metrics, mids, x, n_bootstrap, rng
                )
                record["source_group_bootstrap_n"] = len(boot)
                record["source_group_bootstrap_rho_95"] = (
                    [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
                    if boot
                    else None
                )
            measure_result[score_name] = record
        results[measure_name] = measure_result
    per_metric = []
    for position, metric_id in enumerate(metric_ids):
        bank_row = bank_by_id[metric_id]
        per_metric.append(
            {
                "metric_id": metric_id,
                "metric_index": int(bank_row["metric_index"]),
                "metric_name": str(bank_row["name"]),
                "leaf_count": float(bank_row.get("leaf_count") or 0),
                "mi_scores": dict(sorted(certificate[metric_id].items())),
                "silver_salience": {
                    name: float(values[position])
                    for name, values in sorted(measures.items())
                },
            }
        )
    return {
        "schema_version": "silver-match-v3-mi-validation-v1",
        "status": "TASK_FROZEN_ANALYSIS",
        "task": task,
        "release": {"path": str(release_path), "sha256": sha256_file(release_path)},
        "certificate": {
            "path": str(certificate_path),
            "sha256": sha256_file(certificate_path),
            **join,
        },
        "rows": int(release["expected_rows"]),
        "decision_counts": dict(sorted(decisions.items())),
        "analysis_excluded_rows": int(release["analysis_exclusions"]["count"]),
        "analysis_excluded_decision_counts": dict(sorted(excluded_decisions.items())),
        "analysis_eligible_rows": sum(analysis_decisions.values()),
        "analysis_decision_counts": dict(sorted(analysis_decisions.items())),
        "exact_matches": total_matches,
        "exact_match_rate": (
            total_matches / sum(analysis_decisions.values())
            if analysis_decisions
            else None
        ),
        "source_groups": len(group_metrics),
        "analysis_units": {
            "metric_unit_key": "bank metric_id",
            "source_group_key": (
                "corpus + U+001F + (source_id else paper_id else norm_uid)"
            ),
            "eligible_event": (
                "exact MATCH among rows outside the frozen analysis-exclusion UID set"
            ),
        },
        "silver_source_group_split_half_reliability": _split_half_source_reliability(
            group_metrics, metric_ids, 200, np.random.default_rng(seed + 1)
        ),
        "silver_match_mass_on_scored_metrics": scored_match_mass,
        "silver_match_mass_on_scored_metrics_rate": (
            scored_match_mass / total_matches if total_matches else None
        ),
        "partial_covariate_audit": {
            "leaf_count_nonzero_metrics": int(np.count_nonzero(leaf_count)),
            "leaf_count_unique_values": sorted(
                {float(value) for value in leaf_count.tolist()}
            ),
            "interpretation": (
                "If leaf_count is constant, the reported partial correlation effectively "
                "adjusts only for H_M plus an intercept."
            ),
        },
        "blind_risk": release["blind_risk"],
        "precision_claim_supported": release["precision_claim_supported"],
        "false_abstention_claim_supported": release[
            "false_abstention_claim_supported"
        ],
        "primary_estimand": "source_presence.OPT.spearman_rho",
        "per_metric": per_metric,
        "results": results,
        "randomization": {
            "seed": seed,
            "permutations": n_permutations,
            "source_group_bootstrap_replicates": n_bootstrap,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = run_validation(
        release_path=Path(args.release).resolve(),
        certificate_path=Path(args.certificate).resolve(),
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "task": report["task"],
        "exact_matches": report["exact_matches"],
        "primary": report["results"]["source_presence"]["OPT"],
        "output": str(output),
        "output_sha256": sha256_file(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
