#!/usr/bin/env python3
"""Freeze and re-verify a task's production final blind-risk release.

This is the fail-closed bridge between task final files and the all-task
release.  A passing artifact is not a caller-supplied status flag: the module
reconstructs the deterministic hidden samples, verifies transcript-audited
independent gold labels, and recomputes exact match precision, missed-bank
match risk, and typed-abstention accuracy from the bound final files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .adjudicate_gemma import DECISIONS
from .audit_false_abstentions import audit_false_abstentions
from .audit_final_outputs import audit_outputs
from .common import normalize_space, read_jsonl, sha256_file
from .freeze_final_risk_gold_consensus import verify_gold_consensus_release
from .prepare_false_abstention_audit import _rank


SCHEMA = "silver-match-v3-task-final-risk-release-v2"
SAMPLE_SCHEMA = "silver-match-v3-final-decision-sample-v2"
PACK_SCHEMA = "silver-match-v3-final-risk-label-pack-v1"


def _resolve(value: str | Path, anchor: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _bound_artifact(reference: dict[str, Any], anchor: Path, *, kind: str) -> Path:
    if (
        not isinstance(reference, dict)
        or not reference.get("path")
        or not reference.get("sha256")
    ):
        raise ValueError(f"{kind} lacks a path/hash binding")
    path = _resolve(str(reference["path"]), anchor)
    if not path.is_file() or sha256_file(path) != str(reference["sha256"]):
        raise ValueError(f"{kind} artifact changed: {path}")
    return path


def _normalized_inputs(values: dict[str, Any], anchor: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for raw_path, raw_identity in values.items():
        path = _resolve(raw_path, anchor)
        expected = (
            str(raw_identity.get("sha256") or "")
            if isinstance(raw_identity, dict)
            else str(raw_identity or "")
        )
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"bound input changed: {path}")
        output[str(path)] = expected
    return output


def _index_rows(path: Path, *, kind: str) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed = {normalize_space(row.get("norm_uid")): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"{kind} has missing/duplicate norm UIDs: {path}")
    return indexed


def _task_finals(
    manifest_path: Path,
    manifest: dict[str, Any],
    task: str,
    final_paths: list[Path],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    expected = {
        corpus
        for corpus, meta in (manifest.get("corpora") or {}).items()
        if str(meta.get("task") or "") == task
    }
    if not expected:
        raise ValueError(f"manifest has no corpora for task {task}")
    by_corpus: dict[str, Path] = {}
    rows_by_uid: dict[str, dict[str, Any]] = {}
    for raw_path in final_paths:
        path = raw_path.resolve()
        rows = list(read_jsonl(path))
        corpora = {str(row.get("corpus") or "") for row in rows}
        tasks = {str(row.get("task") or "") for row in rows}
        if len(corpora) != 1 or tasks != {task}:
            raise ValueError(f"final file is not one canonical {task} corpus: {path}")
        corpus = next(iter(corpora))
        if corpus in by_corpus:
            raise ValueError(f"duplicate final binding for {corpus}")
        by_corpus[corpus] = path
        for row in rows:
            uid = normalize_space(row.get("norm_uid"))
            if not uid or uid in rows_by_uid:
                raise ValueError(f"missing/duplicate task final UID: {uid!r}")
            rows_by_uid[uid] = row
    if set(by_corpus) != expected:
        raise ValueError(
            f"task finals do not cover exact corpus set; missing={sorted(expected - set(by_corpus))} "
            f"extra={sorted(set(by_corpus) - expected)}"
        )
    audit = audit_outputs(manifest_path, list(by_corpus.values()), corpora=expected)
    if int(audit.get("audited_rows", -1)) != sum(
        int(manifest["corpora"][corpus]["count"]) for corpus in expected
    ):
        raise ValueError("task final audit count mismatch")
    return by_corpus, rows_by_uid


def _verify_pack_validation(
    path: Path,
    *,
    task: str,
    scope: str,
    sample_uids: set[str],
    manifest_sha256: str,
    bank_source_sha256: str,
    blind_path: Path,
    key_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != PACK_SCHEMA
        or payload.get("task") != task
        or payload.get("sample_scope") != scope
        or payload.get("truth_hidden") is not True
        or payload.get("system_decisions_hidden") is not True
        or payload.get("system_key_excluded_from_label_pack") is not True
        or payload.get("permanently_excluded_from_gradients") is not True
        or int(payload.get("count", -1)) != len(sample_uids)
        or payload.get("bank_source_sha256") != bank_source_sha256
        or ((payload.get("inputs") or {}).get("manifest") or {}).get("sha256")
        != manifest_sha256
    ):
        raise ValueError(f"invalid final-risk label pack validation: {path}")
    blind_ref = (payload.get("inputs") or {}).get("blind_sample") or {}
    key_ref = (payload.get("inputs") or {}).get("system_key") or {}
    if (
        _resolve(str(blind_ref.get("path") or ""), path) != blind_path
        or blind_ref.get("sha256") != sha256_file(blind_path)
        or _resolve(str(key_ref.get("path") or ""), path) != key_path
        or key_ref.get("sha256") != sha256_file(key_path)
        or key_ref.get("excluded_from_pack") is not True
    ):
        raise ValueError("label pack is not bound to the blind sample/key")
    outputs = payload.get("outputs") or {}
    items = _bound_artifact(outputs.get("items") or {}, path, kind="label-pack items")
    bank = _bound_artifact(outputs.get("bank") or {}, path, kind="label-pack bank")
    chunk_refs = outputs.get("chunks") or {}
    chunk_uids: list[str] = []
    for raw_chunk, expected_sha in sorted(chunk_refs.items()):
        chunk = _resolve(raw_chunk, path)
        if not chunk.is_file() or sha256_file(chunk) != str(expected_sha):
            raise ValueError(f"label-pack chunk changed: {chunk}")
        chunk_uids.extend(str(row["norm_uid"]) for row in read_jsonl(chunk))
    item_uids = [str(row["norm_uid"]) for row in read_jsonl(items)]
    if (
        len(item_uids) != len(set(item_uids))
        or set(item_uids) != sample_uids
        or chunk_uids != item_uids
    ):
        raise ValueError("label-pack item/chunk coverage differs from sampled UIDs")
    bank_payload = json.loads(bank.read_text(encoding="utf-8"))
    if (
        bank_payload.get("task") != task
        or bank_payload.get("source_sha256") != bank_source_sha256
    ):
        raise ValueError("label-pack bank identity mismatch")
    return payload, items, bank


def _verify_sample(
    report_path: Path,
    *,
    sample_kind: str,
    task: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    final_artifacts: dict[str, str],
    final_rows: dict[str, dict[str, Any]],
    exclusion_artifacts: dict[str, str],
    excluded_uids: set[str],
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scope = f"task:{task}"
    if (
        report.get("schema_version") != SAMPLE_SCHEMA
        or report.get("sample_kind") != sample_kind
        or report.get("manifest_sha256") != manifest_sha256
        or report.get("sampling")
        != "uniform_without_replacement_by_lowest_stable_sha256_rank"
    ):
        raise ValueError(f"invalid {sample_kind} sample report: {report_path}")
    if (
        _normalized_inputs(report.get("final_inputs") or {}, report_path)
        != final_artifacts
    ):
        raise ValueError(f"{sample_kind} sample is not bound to exact task finals")
    reported_exclusions = _normalized_inputs(
        ((report.get("analysis_exclusions") or {}).get("inputs") or {}), report_path
    )
    if reported_exclusions != exclusion_artifacts:
        raise ValueError(f"{sample_kind} sample exclusion set differs")
    output = (report.get("outputs") or {}).get(scope)
    if not isinstance(output, dict):
        raise ValueError(f"{sample_kind} sample lacks task scope {scope}")
    blind = _bound_artifact(output.get("blind") or {}, report_path, kind="blind sample")
    key = _bound_artifact(output.get("key") or {}, report_path, kind="sample key")
    blind_rows = list(read_jsonl(blind))
    key_rows = list(read_jsonl(key))
    blind_uids = [str(row.get("norm_uid") or "") for row in blind_rows]
    key_uids = [str(row.get("norm_uid") or "") for row in key_rows]
    if (
        not blind_uids
        or len(blind_uids) != len(set(blind_uids))
        or blind_uids != key_uids
        or any(
            row.get("decision") is not None or row.get("metric_id") is not None
            for row in blind_rows
        )
        or any("system_decision" in row for row in blind_rows)
    ):
        raise ValueError(f"{sample_kind} task sample is not blind/unique/aligned")
    sample_n = int(output.get("sample_n", -1))
    if sample_n != len(blind_uids):
        raise ValueError(f"{sample_kind} sample count mismatch")
    eligible = [
        row
        for uid, row in final_rows.items()
        if uid not in excluded_uids
        and ((row.get("decision") == "MATCH") == (sample_kind == "match"))
    ]
    expected = sorted(
        eligible,
        key=lambda row: (
            _rank(str(report["seed"]), scope, str(row["norm_uid"])),
            str(row["norm_uid"]),
        ),
    )[:sample_n]
    expected_uids = [str(row["norm_uid"]) for row in expected]
    if expected_uids != blind_uids or int(
        output.get("population_decisions", -1)
    ) != len(eligible):
        raise ValueError(
            f"{sample_kind} sample is not the declared deterministic uniform sample"
        )
    for rank, (blind_row, key_row) in enumerate(
        zip(blind_rows, key_rows, strict=True), 1
    ):
        uid = blind_uids[rank - 1]
        final = final_rows[uid]
        if (
            key_row.get("sample_scope") != scope
            or int(key_row.get("sample_rank", -1)) != rank
            or key_row.get("system_decision") != final.get("decision")
            or key_row.get("system_metric_id") != final.get("metric_id")
            or blind_row.get("task") != task
            or blind_row.get("corpus") != final.get("corpus")
        ):
            raise ValueError(f"{sample_kind} sample key/content mismatch for {uid}")
    pack_ref = output.get("label_pack_validation") or {}
    pack_validation = _bound_artifact(
        pack_ref, report_path, kind="label-pack validation"
    )
    _, items, bank = _verify_pack_validation(
        pack_validation,
        task=task,
        scope=scope,
        sample_uids=set(blind_uids),
        manifest_sha256=manifest_sha256,
        bank_source_sha256=manifest["banks"][task]["source_sha256"],
        blind_path=blind,
        key_path=key,
    )
    return {
        "report": report,
        "report_path": report_path,
        "scope": scope,
        "uids": set(blind_uids),
        "ordered_uids": blind_uids,
        "blind": blind,
        "key": key,
        "pack_validation": pack_validation,
        "pack_items": items,
        "pack_bank": bank,
    }


def _verify_gold(
    gold_path: Path,
    validation_path: Path,
    *,
    sample: dict[str, Any],
    task: str,
    bank_source_sha256: str,
) -> None:
    consensus_release = verify_gold_consensus_release(
        validation_path,
        expected_sample_report=sample["report_path"],
        expected_truth=gold_path,
    )
    gold = _index_rows(gold_path, kind="blind gold")
    if set(gold) != sample["uids"]:
        raise ValueError("independent gold does not cover the exact sampled UIDs")
    bank_payload = json.loads(sample["pack_bank"].read_text(encoding="utf-8"))
    bank_ids = {str(row["metric_id"]) for row in bank_payload.get("metrics") or []}
    for uid, row in gold.items():
        decision = str(row.get("decision") or "")
        metric_id = row.get("metric_id")
        if (
            row.get("task") != task
            or decision not in DECISIONS
            or row.get("current_bank_source_sha256") != bank_source_sha256
            or (decision == "MATCH" and str(metric_id) not in bank_ids)
            or (decision != "MATCH" and metric_id is not None)
        ):
            raise ValueError(f"invalid independent gold row: {uid}")
    if (
        consensus_release.get("task") != task
        or consensus_release.get("sample_scope") != sample["scope"]
        or consensus_release.get("complete") is not True
        or int(consensus_release.get("count", -1)) != len(gold)
        or consensus_release.get("bank_source_sha256") != bank_source_sha256
        or consensus_release.get("contract", {}).get(
            "at_least_two_complete_independently_permuted_passes"
        )
        is not True
        or consensus_release.get("contract", {}).get("unique_exact_two_vote_consensus")
        is not True
        or consensus_release.get("contract", {}).get(
            "every_pass_strictly_transcript_audited"
        )
        is not True
        or consensus_release.get("contract", {}).get(
            "no_unresolved_sample_rows_dropped"
        )
        is not True
    ):
        raise ValueError("independent gold consensus release is incomplete or misbound")


def _evaluate(
    *,
    manifest_path: Path,
    task: str,
    final_paths: list[Path],
    match_sample_report: Path,
    match_gold: Path,
    match_gold_validation: Path,
    abstention_sample_report: Path,
    abstention_gold: Path,
    abstention_gold_validation: Path,
    analysis_exclusion_paths: list[Path],
    alpha: float,
    false_abstention_target: float,
    match_precision_target: float,
    typed_abstention_point_target: float,
    typed_abstention_lower_target: float,
    minimum_support: int,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if task not in (manifest.get("banks") or {}):
        raise ValueError(f"unknown task: {task}")
    manifest_sha = sha256_file(manifest_path)
    by_corpus, final_rows = _task_finals(manifest_path, manifest, task, final_paths)
    final_artifacts = {str(path): sha256_file(path) for path in by_corpus.values()}
    exclusion_artifacts = {
        str(path.resolve()): sha256_file(path.resolve())
        for path in analysis_exclusion_paths
    }
    excluded_uids: set[str] = set()
    for path in analysis_exclusion_paths:
        excluded_uids.update(_index_rows(path.resolve(), kind="analysis exclusion"))
    if excluded_uids & set(final_rows) and not exclusion_artifacts:
        raise AssertionError("unreachable exclusion binding state")

    match_sample = _verify_sample(
        match_sample_report.resolve(),
        sample_kind="match",
        task=task,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        final_artifacts=final_artifacts,
        final_rows=final_rows,
        exclusion_artifacts=exclusion_artifacts,
        excluded_uids=excluded_uids,
    )
    abstention_sample = _verify_sample(
        abstention_sample_report.resolve(),
        sample_kind="abstention",
        task=task,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        final_artifacts=final_artifacts,
        final_rows=final_rows,
        exclusion_artifacts=exclusion_artifacts,
        excluded_uids=excluded_uids,
    )
    if match_sample["uids"] & abstention_sample["uids"]:
        raise ValueError("match and abstention audit samples overlap")
    _verify_gold(
        match_gold.resolve(),
        match_gold_validation.resolve(),
        sample=match_sample,
        task=task,
        bank_source_sha256=manifest["banks"][task]["source_sha256"],
    )
    _verify_gold(
        abstention_gold.resolve(),
        abstention_gold_validation.resolve(),
        sample=abstention_sample,
        task=task,
        bank_source_sha256=manifest["banks"][task]["source_sha256"],
    )
    match_risk = audit_false_abstentions(
        [match_gold.resolve()],
        list(by_corpus.values()),
        alpha=alpha,
        target=false_abstention_target,
        precision_target=match_precision_target,
        analysis_exclusion_paths=analysis_exclusion_paths,
    )["overall"]
    abstention_risk = audit_false_abstentions(
        [abstention_gold.resolve()],
        list(by_corpus.values()),
        alpha=alpha,
        target=false_abstention_target,
        precision_target=match_precision_target,
        analysis_exclusion_paths=analysis_exclusion_paths,
    )["overall"]
    gates = {
        "match_sample_support": match_risk["audited_rows"] >= minimum_support,
        "match_sample_contains_only_system_matches": (
            match_risk["predicted_matches"] == match_risk["audited_rows"]
        ),
        "match_exact_precision_point": (
            match_risk["predicted_match_exact_precision"] is not None
            and match_risk["predicted_match_exact_precision"] >= match_precision_target
        ),
        "match_exact_precision_lower": (
            match_risk["predicted_match_exact_precision_lower_bound"] is not None
            and match_risk["predicted_match_exact_precision_lower_bound"]
            >= match_precision_target
        ),
        "abstention_sample_support": abstention_risk["audited_rows"] >= minimum_support,
        "abstention_sample_contains_only_system_abstentions": (
            abstention_risk["predicted_abstentions"] == abstention_risk["audited_rows"]
        ),
        "false_abstention_upper_bound": (
            abstention_risk["false_abstention_upper_bound"] is not None
            and abstention_risk["false_abstention_upper_bound"]
            < false_abstention_target
        ),
        "typed_abstention_point": (
            abstention_risk["typed_abstention_exact_accuracy"] is not None
            and abstention_risk["typed_abstention_exact_accuracy"]
            >= typed_abstention_point_target
        ),
        "typed_abstention_lower": (
            abstention_risk["typed_abstention_exact_accuracy_lower_bound"] is not None
            and abstention_risk["typed_abstention_exact_accuracy_lower_bound"]
            >= typed_abstention_lower_target
        ),
    }
    return {
        "schema_version": SCHEMA,
        "task": task,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "complete": all(gates.values()),
        "production_final_blind_audit": True,
        "gates": gates,
        "thresholds": {
            "alpha_one_sided": alpha,
            "minimum_support_per_sample": minimum_support,
            "false_abstention_upper_target": false_abstention_target,
            "match_exact_precision_point_target": match_precision_target,
            "match_exact_precision_lower_target": match_precision_target,
            "typed_abstention_exact_point_target": typed_abstention_point_target,
            "typed_abstention_exact_lower_target": typed_abstention_lower_target,
        },
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "finals": {
            corpus: _artifact(path) for corpus, path in sorted(by_corpus.items())
        },
        "analysis_exclusions": {
            str(path.resolve()): sha256_file(path.resolve())
            for path in analysis_exclusion_paths
        },
        "match_audit": {
            "sample_report": _artifact(match_sample_report.resolve()),
            "gold": _artifact(match_gold.resolve()),
            "gold_validation": _artifact(match_gold_validation.resolve()),
            "statistics": match_risk,
        },
        "abstention_audit": {
            "sample_report": _artifact(abstention_sample_report.resolve()),
            "gold": _artifact(abstention_gold.resolve()),
            "gold_validation": _artifact(abstention_gold_validation.resolve()),
            "statistics": abstention_risk,
        },
    }


def verify_task_final_risk_release(
    release_path: Path,
    *,
    expected_manifest_path: Path | None = None,
    expected_final_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    """Recompute a frozen release from its bound artifacts and thresholds."""

    release_path = release_path.resolve()
    frozen = json.loads(release_path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != SCHEMA:
        raise ValueError(f"unsupported task final-risk schema: {release_path}")
    manifest = _bound_artifact(
        frozen.get("manifest") or {}, release_path, kind="manifest"
    )
    if (
        expected_manifest_path is not None
        and manifest != expected_manifest_path.resolve()
    ):
        raise ValueError("task final-risk release binds another manifest")
    final_refs = frozen.get("finals") or {}
    final_paths = [
        _bound_artifact(reference, release_path, kind=f"final {corpus}")
        for corpus, reference in sorted(final_refs.items())
    ]
    if expected_final_paths is not None and {
        path.resolve() for path in expected_final_paths
    } != set(final_paths):
        raise ValueError("task final-risk release binds another set of final files")
    exclusions = [
        _resolve(raw_path, release_path)
        for raw_path, expected_sha in sorted(
            (frozen.get("analysis_exclusions") or {}).items()
        )
        if _resolve(raw_path, release_path).is_file()
        and sha256_file(_resolve(raw_path, release_path)) == expected_sha
    ]
    if len(exclusions) != len(frozen.get("analysis_exclusions") or {}):
        raise ValueError("task final-risk exclusion artifact changed")
    match = frozen.get("match_audit") or {}
    abstention = frozen.get("abstention_audit") or {}
    thresholds = frozen.get("thresholds") or {}
    recomputed = _evaluate(
        manifest_path=manifest,
        task=str(frozen.get("task") or ""),
        final_paths=final_paths,
        match_sample_report=_bound_artifact(
            match.get("sample_report") or {}, release_path, kind="match sample report"
        ),
        match_gold=_bound_artifact(
            match.get("gold") or {}, release_path, kind="match gold"
        ),
        match_gold_validation=_bound_artifact(
            match.get("gold_validation") or {},
            release_path,
            kind="match gold validation",
        ),
        abstention_sample_report=_bound_artifact(
            abstention.get("sample_report") or {},
            release_path,
            kind="abstention sample report",
        ),
        abstention_gold=_bound_artifact(
            abstention.get("gold") or {}, release_path, kind="abstention gold"
        ),
        abstention_gold_validation=_bound_artifact(
            abstention.get("gold_validation") or {},
            release_path,
            kind="abstention gold validation",
        ),
        analysis_exclusion_paths=exclusions,
        alpha=float(thresholds["alpha_one_sided"]),
        false_abstention_target=float(thresholds["false_abstention_upper_target"]),
        match_precision_target=float(thresholds["match_exact_precision_lower_target"]),
        typed_abstention_point_target=float(
            thresholds["typed_abstention_exact_point_target"]
        ),
        typed_abstention_lower_target=float(
            thresholds["typed_abstention_exact_lower_target"]
        ),
        minimum_support=int(thresholds["minimum_support_per_sample"]),
    )
    for key in ("task", "status", "complete", "production_final_blind_audit", "gates"):
        if frozen.get(key) != recomputed.get(key):
            raise ValueError(
                f"frozen task final-risk field differs from recomputation: {key}"
            )
    if frozen.get("thresholds") != recomputed.get("thresholds"):
        raise ValueError("frozen task final-risk thresholds differ from recomputation")
    if (
        frozen.get("match_audit", {}).get("statistics")
        != recomputed["match_audit"]["statistics"]
    ):
        raise ValueError("frozen match-risk statistics differ from recomputation")
    if (
        frozen.get("abstention_audit", {}).get("statistics")
        != recomputed["abstention_audit"]["statistics"]
    ):
        raise ValueError("frozen abstention-risk statistics differ from recomputation")
    return recomputed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--final", action="append", required=True)
    parser.add_argument("--match-sample-report", required=True)
    parser.add_argument("--match-gold", required=True)
    parser.add_argument("--match-gold-validation", required=True)
    parser.add_argument("--abstention-sample-report", required=True)
    parser.add_argument("--abstention-gold", required=True)
    parser.add_argument("--abstention-gold-validation", required=True)
    parser.add_argument("--analysis-exclusion", action="append", default=[])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--false-abstention-target", type=float, default=0.05)
    parser.add_argument("--match-precision-target", type=float, default=0.90)
    parser.add_argument("--typed-abstention-point-target", type=float, default=0.90)
    parser.add_argument("--typed-abstention-lower-target", type=float, default=0.80)
    parser.add_argument("--minimum-support", type=int, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = _evaluate(
        manifest_path=Path(args.manifest),
        task=args.task,
        final_paths=[Path(path) for path in args.final],
        match_sample_report=Path(args.match_sample_report),
        match_gold=Path(args.match_gold),
        match_gold_validation=Path(args.match_gold_validation),
        abstention_sample_report=Path(args.abstention_sample_report),
        abstention_gold=Path(args.abstention_gold),
        abstention_gold_validation=Path(args.abstention_gold_validation),
        analysis_exclusion_paths=[
            Path(path).resolve() for path in args.analysis_exclusion
        ],
        alpha=args.alpha,
        false_abstention_target=args.false_abstention_target,
        match_precision_target=args.match_precision_target,
        typed_abstention_point_target=args.typed_abstention_point_target,
        typed_abstention_lower_target=args.typed_abstention_lower_target,
        minimum_support=args.minimum_support,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "task": result["task"],
                "output": str(output),
                "output_sha256": sha256_file(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
