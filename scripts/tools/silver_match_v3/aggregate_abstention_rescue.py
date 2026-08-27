#!/usr/bin/env python3
"""Validate exhaustive rescue trials and build final match/abstention audits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .adjudicate_gemma import DECISIONS
from .common import normalize_space, read_jsonl, sha256_file, write_jsonl


def _unique(paths: Iterable[Path], kind: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            uid = normalize_space(row.get("norm_uid"))
            if not uid or uid in output:
                raise ValueError(f"missing/duplicate {kind} norm_uid: {uid!r}")
            output[uid] = row
    return output


def _trial_key(row: dict[str, Any]) -> tuple[str, int]:
    uid = normalize_space(row.get("norm_uid"))
    try:
        trial = int(row.get("rescue_trial"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing/invalid rescue_trial for {uid}") from exc
    return uid, trial


def _trial_map(paths: Iterable[Path], kind: str) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for path in paths:
        for row in read_jsonl(path):
            key = _trial_key(row)
            if key in output:
                raise ValueError(f"duplicate {kind} trial: {key}")
            output[key] = row
    return output


def _confidence_order(value: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(value, 3)


def _consensus(primary: dict[str, Any], trials: list[dict[str, Any]]) -> tuple[str, float, dict[str, int]]:
    votes = Counter()
    primary_decision = normalize_space(primary.get("decision"))
    if primary_decision in DECISIONS and primary_decision != "MATCH":
        votes[primary_decision] += 2  # primary saw the strongest retrieval slate
    for row in trials:
        decision = normalize_space(row.get("decision"))
        if decision in DECISIONS and decision != "MATCH":
            votes[decision] += 1
    if not votes:
        return "INVALID_OUTPUT", 0.0, {}
    decision, count = sorted(votes.items(), key=lambda pair: (-pair[1], pair[0]))[0]
    total = sum(votes.values())
    return decision, count / total, dict(sorted(votes.items()))


def aggregate_rescue(
    *,
    manifest_path: Path,
    rescue_manifest_path: Path,
    primary_paths: list[Path],
    adjudication_paths: list[Path],
    output_root: Path,
    max_finalists: int = 16,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rescue_manifest = json.loads(rescue_manifest_path.read_text(encoding="utf-8"))
    if rescue_manifest.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("rescue manifest was built against a different canonical manifest")
    coverage_repeats = int(rescue_manifest.get("coverage_repeats", 1))
    reinclude_primary = bool(rescue_manifest.get("reinclude_primary", False))
    if coverage_repeats < 1:
        raise ValueError("invalid rescue coverage repeat count")
    expected_paths = [Path(path) for path in rescue_manifest.get("outputs", {})]
    for path in expected_paths:
        expected_hash = rescue_manifest["outputs"][str(path)]["sha256"]
        if sha256_file(path) != expected_hash:
            raise ValueError(f"rescue candidate hash changed: {path}")
    full_candidate_paths = [Path(path) for path in rescue_manifest["candidate_inputs"]]
    for path in full_candidate_paths:
        if sha256_file(path) != rescue_manifest["candidate_inputs"][str(path)]:
            raise ValueError(f"full candidate input hash changed: {path}")
    full_candidates: dict[str, dict[str, Any]] = {}
    for path in full_candidate_paths:
        for row in read_jsonl(path):
            uid = normalize_space(row.get("norm_uid"))
            if not uid:
                raise ValueError(f"full-candidate row lacks norm_uid in {path}")
            existing = full_candidates.get(uid)
            if existing is None:
                full_candidates[uid] = row
                continue
            existing_ids = {candidate["metric_id"] for candidate in existing.get("candidates") or []}
            new_ids = {candidate["metric_id"] for candidate in row.get("candidates") or []}
            if (
                existing.get("task") != row.get("task")
                or existing.get("corpus") != row.get("corpus")
                or existing.get("bank_source_sha256") != row.get("bank_source_sha256")
                or existing_ids != new_ids
            ):
                raise ValueError(f"full-candidate systems disagree on routing/bank for {uid}")
    primary = _unique(primary_paths, "primary")
    expected = _trial_map(expected_paths, "expected-candidate")
    observed = _trial_map(adjudication_paths, "adjudication")
    missing, extra = set(expected) - set(observed), set(observed) - set(expected)
    if missing or extra:
        raise ValueError(
            f"rescue adjudication coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    by_uid: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for (uid, trial), candidate_row in expected.items():
        result = observed[(uid, trial)]
        expected_ids = [row["metric_id"] for row in candidate_row["candidates"]]
        if result.get("candidate_ids") != expected_ids:
            raise ValueError(f"candidate/order mismatch in rescue result {(uid, trial)}")
        if normalize_space(result.get("candidate_bank_source_sha256")) != normalize_space(
            candidate_row.get("bank_source_sha256")
        ):
            raise ValueError(f"bank provenance mismatch in rescue result {(uid, trial)}")
        decision = normalize_space(result.get("decision"))
        if decision not in DECISIONS | {"INVALID_OUTPUT"}:
            raise ValueError(f"invalid rescue decision {decision!r} for {(uid, trial)}")
        if decision == "MATCH" and result.get("metric_id") not in expected_ids:
            raise ValueError(f"rescue MATCH outside its trial slate for {(uid, trial)}")
        by_uid[uid].append((trial, candidate_row, result))

    finalists, no_match = [], []
    status_counts = Counter()
    decision_counts = Counter()
    capture_pattern_counts = Counter()
    for uid, grouped in sorted(by_uid.items()):
        grouped.sort(key=lambda item: item[0])
        first = grouped[0][1]
        task = first["task"]
        bank_payload = json.loads(Path(manifest["banks"][task]["path"]).read_text(encoding="utf-8"))
        bank_ids = {row["metric_id"] for row in bank_payload["metrics"]}
        primary_row = primary.get(uid)
        if primary_row is None:
            raise ValueError(f"rescue UID absent from primary results: {uid}")
        primary_ids = set(
            primary_row.get("candidate_ids")
            or first.get("primary_candidate_ids")
            or []
        )
        exposures = Counter()
        trial_candidate_by_id = {}
        for _, candidate_row, _ in grouped:
            for candidate in candidate_row["candidates"]:
                metric_id = candidate["metric_id"]
                exposures[metric_id] += 1
                trial_candidate_by_id[metric_id] = candidate
        if set(exposures) - bank_ids:
            raise ValueError(
                f"rescue contains metrics outside the bank for {uid}: "
                f"{sorted(set(exposures)-bank_ids)[:3]}"
            )
        expected_exposures = {
            metric_id: (
                coverage_repeats
                if reinclude_primary or metric_id not in primary_ids
                else 0
            )
            for metric_id in bank_ids
        }
        mismatched = {
            metric_id: (exposures.get(metric_id, 0), expected)
            for metric_id, expected in expected_exposures.items()
            if exposures.get(metric_id, 0) != expected
        }
        if mismatched:
            raise ValueError(
                f"rescue repeated-coverage invariant failed for {uid}: "
                f"sample={list(sorted(mismatched.items()))[:3]}"
            )

        proposals = []
        for trial, candidate_row, result in grouped:
            if result["decision"] == "MATCH":
                proposals.append(
                    (
                        _confidence_order(str(result.get("confidence") or "")),
                        trial,
                        str(result["metric_id"]),
                        result,
                        int(candidate_row.get("rescue_capture", 0)),
                    )
                )
        detected_captures = {
            capture for _, _, _, _, capture in proposals
        }
        capture_pattern = "".join(
            "1" if capture in detected_captures else "0"
            for capture in range(coverage_repeats)
        )
        capture_pattern_counts[capture_pattern] += 1
        unique_proposals = []
        seen_proposals = set()
        proposal_captures: dict[str, set[int]] = defaultdict(set)
        for _, trial, metric_id, result, capture in sorted(proposals):
            proposal_captures[metric_id].add(capture)
            if metric_id not in seen_proposals:
                unique_proposals.append((trial, metric_id, result))
                seen_proposals.add(metric_id)
        if unique_proposals:
            selected_ids = [metric_id for _, metric_id, _ in unique_proposals]
            # Add high-ranked primary and local rescue rivals so the final judge
            # performs a real contrastive choice rather than rubber-stamping a
            # single block proposal.
            full = full_candidates[uid]["candidates"]
            for candidate in full:
                if len(selected_ids) >= max_finalists:
                    break
                metric_id = candidate["metric_id"]
                if metric_id not in selected_ids:
                    selected_ids.append(metric_id)
            full_by_id = {row["metric_id"]: row for row in full}
            candidate_cards = [
                trial_candidate_by_id.get(metric_id) or full_by_id[metric_id]
                for metric_id in selected_ids
            ]
            trial_summary = [
                {
                    "trial": trial,
                    "capture": candidate_row.get("rescue_capture"),
                    "decision": result["decision"],
                    "metric_id": result.get("metric_id"),
                    "confidence": result.get("confidence"),
                    "reason": result.get("reason"),
                }
                for trial, candidate_row, result in grouped
            ]
            rescue_context = (
                f"All {len(bank_ids)} frozen bank metrics were covered in "
                f"{coverage_repeats} independent rescue captures across {len(grouped)} "
                f"complementary trials (primary metrics re-included={reinclude_primary}). "
                f"Exact-match proposals were {sorted(seen_proposals)}. Independently assess "
                "the finalist cards; abstain if none is exact. Trial summaries: "
                + json.dumps(trial_summary, ensure_ascii=False, sort_keys=True)
            )
            finalists.append(
                {
                    "schema_version": manifest["schema_version"],
                    "norm_uid": uid,
                    "corpus": first["corpus"],
                    "task": task,
                    "row": first.get("row"),
                    "bank_source_sha256": first["bank_source_sha256"],
                    "rescue_context": rescue_context,
                    "rescue_exhaustive": True,
                    "rescue_bank_count": len(bank_ids),
                    "rescue_coverage_repeats": coverage_repeats,
                    "rescue_reincludes_primary": reinclude_primary,
                    "rescue_proposed_metric_ids": sorted(seen_proposals),
                    "rescue_proposal_captures_by_metric": {
                        metric_id: sorted(captures)
                        for metric_id, captures in sorted(proposal_captures.items())
                    },
                    "rescue_capture_pattern": capture_pattern,
                    "candidates": candidate_cards,
                }
            )
            status_counts["MATCH_FINALISTS"] += 1
        else:
            decision, fraction, votes = _consensus(
                primary_row, [result for _, _, result in grouped]
            )
            decision_counts[decision] += 1
            status = "EXHAUSTIVE_NO_MATCH_CONSENSUS" if fraction >= 2 / 3 else "EXHAUSTIVE_NO_MATCH_DISAGREEMENT"
            status_counts[status] += 1
            no_match.append(
                {
                    "schema_version": manifest["schema_version"],
                    "norm_uid": uid,
                    "corpus": first["corpus"],
                    "task": task,
                    "row": first.get("row"),
                    "bank_source_sha256": first["bank_source_sha256"],
                    "rescue_status": status,
                    "rescue_exhaustive": True,
                    "rescue_bank_count": len(bank_ids),
                    "rescue_coverage_repeats": coverage_repeats,
                    "rescue_reincludes_primary": reinclude_primary,
                    "provisional_decision": decision,
                    "consensus_fraction": fraction,
                    "vote_counts": votes,
                    "primary_decision": primary_row.get("decision"),
                    "rescue_capture_pattern": capture_pattern,
                    "trial_results": [
                        {
                            "trial": trial,
                            "capture": candidate_row.get("rescue_capture"),
                            "decision": result.get("decision"),
                            "confidence": result.get("confidence"),
                            "reason": result.get("reason"),
                        }
                        for trial, candidate_row, result in grouped
                    ],
                }
            )

    finalists_path = output_root / "match_finalists.jsonl"
    no_match_path = output_root / "no_match_provisional.jsonl"
    write_jsonl(finalists_path, finalists)
    write_jsonl(no_match_path, no_match)
    report = {
        "schema_version": "silver-match-v3-abstention-rescue-aggregate-v1",
        "manifest_sha256": sha256_file(manifest_path),
        "rescue_manifest_sha256": sha256_file(rescue_manifest_path),
        "primary_inputs": {str(path): sha256_file(path) for path in primary_paths},
        "adjudication_inputs": {str(path): sha256_file(path) for path in adjudication_paths},
        "expected_trial_rows": len(expected),
        "observed_trial_rows": len(observed),
        "rescued_uids": len(by_uid),
        "coverage_repeats": coverage_repeats,
        "reinclude_primary": reinclude_primary,
        "status_counts": dict(sorted(status_counts.items())),
        "no_match_provisional_decision_counts": dict(sorted(decision_counts.items())),
        "proposal_capture_pattern_counts": dict(sorted(capture_pattern_counts.items())),
        "outputs": {
            str(finalists_path): {"count": len(finalists), "sha256": sha256_file(finalists_path)},
            str(no_match_path): {"count": len(no_match), "sha256": sha256_file(no_match_path)},
        },
    }
    if coverage_repeats >= 2:
        n1 = sum(
            count for pattern, count in capture_pattern_counts.items() if pattern[0] == "1"
        )
        n2 = sum(
            count for pattern, count in capture_pattern_counts.items() if pattern[1] == "1"
        )
        overlap = sum(
            count
            for pattern, count in capture_pattern_counts.items()
            if pattern[:2] == "11"
        )
        chapman = ((n1 + 1) * (n2 + 1) / (overlap + 1)) - 1
        report["capture_recapture_diagnostic"] = {
            "captures": [0, 1],
            "capture_0_detected": n1,
            "capture_1_detected": n2,
            "overlap": overlap,
            "chapman_detectable_match_uid_estimate": chapman,
            "observed_union": n1 + n2 - overlap,
            "estimated_unobserved": max(0.0, chapman - (n1 + n2 - overlap)),
            "claim_scope": (
                "proposal-discovery diversity diagnostic only; final false-abstention "
                "probability comes from the blind exact-bank audit"
            ),
        }
    report_path = output_root / "aggregate_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rescue-manifest", required=True)
    parser.add_argument("--primary", action="append", required=True)
    parser.add_argument("--adjudication", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-finalists", type=int, default=16)
    args = parser.parse_args()
    report = aggregate_rescue(
        manifest_path=Path(args.manifest).resolve(),
        rescue_manifest_path=Path(args.rescue_manifest).resolve(),
        primary_paths=[Path(path).resolve() for path in args.primary],
        adjudication_paths=[Path(path).resolve() for path in args.adjudication],
        output_root=Path(args.output_root).resolve(),
        max_finalists=args.max_finalists,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
