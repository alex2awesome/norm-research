import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.audit_isolated_labeler_transcripts import audit
from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.freeze_task_final_risk_release import (
    _evaluate,
    verify_task_final_risk_release,
)
from scripts.tools.silver_match_v3.freeze_final_risk_gold_consensus import (
    evaluate_gold_consensus,
)
from scripts.tools.silver_match_v3.prepare_false_abstention_audit import prepare


def _run_module(module: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _write_label_pass(
    *,
    tmp_path: Path,
    pack_root: Path,
    final_by_uid: dict[str, dict],
    name: str,
) -> tuple[Path, Path, Path]:
    pack_validation_path = pack_root / "validation.json"
    guide = tmp_path / "guide.md"
    if not guide.exists():
        guide.write_text("Use the frozen bank and assigned chunk only.\n")
    raw_root = pack_root / "raw_labels"
    log_root = pack_root / "logs"
    raw_root.mkdir()
    log_root.mkdir()
    raw_sha_by_uid = {}
    for chunk in sorted((pack_root / "chunks").glob("part-*.jsonl")):
        rows = list(read_jsonl(chunk))
        labels = []
        for item in rows:
            final = final_by_uid[item["norm_uid"]]
            labels.append(
                {
                    "norm_uid": item["norm_uid"],
                    "decision": final["decision"],
                    "metric_id": final["metric_id"],
                    "confidence": "high",
                    "reason": "independent exact full-bank decision",
                }
            )
        raw = raw_root / f"{chunk.stem}.json"
        raw.write_text(
            json.dumps({"task": "task", "chunk_id": chunk.stem, "labels": labels})
            + "\n"
        )
        raw_sha = sha256_file(raw)
        for item in rows:
            raw_sha_by_uid[item["norm_uid"]] = (raw, raw_sha)
        bank_rel = (pack_root / "bank.json").relative_to(tmp_path)
        chunk_rel = chunk.relative_to(tmp_path)
        guide_rel = guide.relative_to(tmp_path)
        (log_root / f"{chunk.stem}.log").write_text(
            "sandbox: read-only\n"
            "approval: never\n"
            "exec\n"
            f"/bin/zsh -lc \"sed -n '1,80p' {bank_rel}; "
            f"sed -n '1,80p' {chunk_rel}; sed -n '1,80p' {guide_rel}\"\n"
        )
    transcript_payload = audit(pack_root, [guide], tmp_path)
    assert transcript_payload["status"] == "PASS"
    transcript = pack_root / "transcript.audit.json"
    transcript.write_text(
        json.dumps(transcript_payload, indent=2, sort_keys=True) + "\n"
    )

    pack_items = list(read_jsonl(pack_root / "items.jsonl"))
    gold_rows = []
    for item in pack_items:
        uid = item["norm_uid"]
        final = final_by_uid[uid]
        raw, raw_sha = raw_sha_by_uid[uid]
        gold_rows.append(
            {
                "schema_version": "silver-match-v3.0",
                "norm_uid": uid,
                "corpus": "corpus",
                "task": "task",
                "row": item["row"],
                "split_group": item["split_group"],
                "split": item["split"],
                "decision": final["decision"],
                "metric_id": final["metric_id"],
                "current_bank_source_sha256": "bank-source",
                "confidence": "high",
                "reason": "independent exact full-bank decision",
                "label_source": "independent_codex_full_bank",
                "annotator": "test-independent-labeler",
                "retrieved_rank": None,
                "training_eligible_preverification": False,
                "raw_label_chunk": str(raw),
                "raw_label_chunk_sha256": raw_sha,
            }
        )
    gold = pack_root / "labels.validated.jsonl"
    write_jsonl(gold, gold_rows)
    validation = pack_root / "labels.validation.json"
    validation.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-independent-label-validation-v1",
                "task": "task",
                "complete": True,
                "count": len(gold_rows),
                "unique_uids": len(gold_rows),
                "bank_source_sha256": "bank-source",
                "pack_validation": {
                    "path": str(pack_validation_path),
                    "sha256": sha256_file(pack_validation_path),
                },
                "transcript_audit": {
                    "path": str(transcript),
                    "sha256": sha256_file(transcript),
                    "status": "PASS",
                    "audited_chunks": transcript_payload["audited_chunks"],
                },
                "output": {"path": str(gold), "sha256": sha256_file(gold)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return gold, validation, transcript


def _write_independent_gold(
    *,
    tmp_path: Path,
    sample_report: dict,
    sample_root: Path,
    final_by_uid: dict[str, dict],
    name: str,
) -> tuple[Path, Path]:
    scope = "task:task"
    pack_validation_ref = sample_report["outputs"][scope]["label_pack_validation"]
    source_validation = Path(pack_validation_ref["path"])
    source_root = source_validation.parent
    pass_a = sample_root / f"{name}.pass-a"
    pass_b = sample_root / f"{name}.pass-b"
    module_root = "scripts.tools.silver_match_v3"
    _run_module(
        f"{module_root}.permute_independent_teacher_pack",
        "--pack-root",
        str(source_root),
        "--output-root",
        str(pass_a),
        "--seed",
        "104729",
        "--chunk-size",
        "25",
    )
    _run_module(
        f"{module_root}.permute_independent_teacher_pack",
        "--pack-root",
        str(source_root),
        "--output-root",
        str(pass_b),
        "--seed",
        "130363",
        "--chunk-size",
        "25",
    )
    independence = sample_root / f"{name}.independence.prelabel.json"
    _run_module(
        f"{module_root}.audit_independent_pack_views",
        "--pass-a",
        str(pass_a),
        "--pass-b",
        str(pass_b),
        "--output",
        str(independence),
    )
    labels_a, validation_a, transcript_a = _write_label_pass(
        tmp_path=tmp_path,
        pack_root=pass_a,
        final_by_uid=final_by_uid,
        name=f"{name}-a",
    )
    labels_b, validation_b, transcript_b = _write_label_pass(
        tmp_path=tmp_path,
        pack_root=pass_b,
        final_by_uid=final_by_uid,
        name=f"{name}-b",
    )
    truth = sample_root / f"{name}.consensus.truth.jsonl"
    unresolved = sample_root / f"{name}.consensus.unresolved.jsonl"
    disagreements = sample_root / f"{name}.consensus.disagreements.jsonl"
    report = sample_root / f"{name}.consensus.report.json"
    _run_module(
        f"{module_root}.finalize_exact_multi_pass_truth",
        "--pack-root",
        str(source_root),
        "--label-pass",
        f"A={labels_a}",
        "--label-pass",
        f"B={labels_b}",
        "--pass-pack",
        f"A={pass_a}",
        "--pass-pack",
        f"B={pass_b}",
        "--output",
        str(truth),
        "--unresolved-output",
        str(unresolved),
        "--disagreements-output",
        str(disagreements),
        "--report",
        str(report),
        "--gepa-role",
        "evaluation",
    )
    release_payload = evaluate_gold_consensus(
        sample_report_path=sample_root / "sample_report.json",
        scope=scope,
        truth_path=truth,
        consensus_report_path=report,
        independence_audit_path=independence,
        label_validation_paths={"A": validation_a, "B": validation_b},
        transcript_audit_paths={"A": transcript_a, "B": transcript_b},
    )
    release = sample_root / f"{name}.gold.consensus.release.json"
    release.write_text(json.dumps(release_payload, indent=2, sort_keys=True) + "\n")
    return truth, release


def _fixture(tmp_path: Path):
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "task": "task",
                "metrics": [
                    {
                        "metric_id": f"a{index}",
                        "name": f"criterion {index}",
                        "description": f"exact criterion {index}",
                    }
                    for index in range(7)
                ],
            }
        )
        + "\n"
    )
    norms, finals = [], []
    for index in range(140):
        uid = f"u{index:04d}"
        norms.append(
            {
                "schema_version": "silver-match-v3.0",
                "norm_uid": uid,
                "row": index,
                "corpus": "corpus",
                "task": "task",
                "norm": f"explicit criterion {index}",
                "context": f"context {index}",
                "kind": "critique",
                "polarity": "negative",
            }
        )
        if index < 70:
            decision, metric_id = "MATCH", "a0"
        else:
            decision, metric_id = (
                ("NOISE", None) if index % 2 else ("NO_CANDIDATE_FITS", None)
            )
        finals.append(
            {
                "norm_uid": uid,
                "row": index,
                "corpus": "corpus",
                "task": "task",
                "bank_source_sha256": "bank-source",
                "decision": decision,
                "metric_id": metric_id,
                "confidence": "high",
                "verification_status": "independently_verified",
            }
        )
    norm_path = tmp_path / "norms.jsonl"
    final_path = tmp_path / "final.jsonl"
    write_jsonl(norm_path, norms)
    write_jsonl(final_path, finals)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3.0",
                "total_norms": 140,
                "total_corpora": 1,
                "total_tasks": 1,
                "corpora": {
                    "corpus": {
                        "task": "task",
                        "count": 140,
                        "path": str(norm_path),
                    }
                },
                "banks": {
                    "task": {
                        "count": 7,
                        "path": str(bank),
                        "source_sha256": "bank-source",
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    match_root = tmp_path / "match-audit"
    abstention_root = tmp_path / "abstention-audit"
    match_report = prepare(
        manifest_path=manifest,
        final_paths=[final_path],
        output_root=match_root,
        global_n=60,
        per_task_n=60,
        seed="match-seed",
        sample_kind="match",
    )
    abstention_report = prepare(
        manifest_path=manifest,
        final_paths=[final_path],
        output_root=abstention_root,
        global_n=60,
        per_task_n=60,
        seed="abstention-seed",
        sample_kind="abstention",
    )
    final_by_uid = {row["norm_uid"]: row for row in finals}
    match_gold, match_validation = _write_independent_gold(
        tmp_path=tmp_path,
        sample_report=match_report,
        sample_root=match_root,
        final_by_uid=final_by_uid,
        name="match",
    )
    abstention_gold, abstention_validation = _write_independent_gold(
        tmp_path=tmp_path,
        sample_report=abstention_report,
        sample_root=abstention_root,
        final_by_uid=final_by_uid,
        name="abstention",
    )
    return {
        "manifest": manifest,
        "final": final_path,
        "match_report": match_root / "sample_report.json",
        "match_gold": match_gold,
        "match_validation": match_validation,
        "abstention_report": abstention_root / "sample_report.json",
        "abstention_gold": abstention_gold,
        "abstention_validation": abstention_validation,
    }


def test_final_risk_release_recomputes_strict_blind_gates(tmp_path: Path):
    fixture = _fixture(tmp_path)
    result = _evaluate(
        manifest_path=fixture["manifest"],
        task="task",
        final_paths=[fixture["final"]],
        match_sample_report=fixture["match_report"],
        match_gold=fixture["match_gold"],
        match_gold_validation=fixture["match_validation"],
        abstention_sample_report=fixture["abstention_report"],
        abstention_gold=fixture["abstention_gold"],
        abstention_gold_validation=fixture["abstention_validation"],
        analysis_exclusion_paths=[],
        alpha=0.05,
        false_abstention_target=0.05,
        match_precision_target=0.90,
        typed_abstention_point_target=0.90,
        typed_abstention_lower_target=0.80,
        minimum_support=60,
    )
    assert result["status"] == "PASS"
    assert all(result["gates"].values())
    release = tmp_path / "risk.release.json"
    release.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    verified = verify_task_final_risk_release(
        release,
        expected_manifest_path=fixture["manifest"],
        expected_final_paths=[fixture["final"]],
    )
    assert verified["complete"] is True


def test_final_risk_release_fails_closed_on_gold_drift(tmp_path: Path):
    fixture = _fixture(tmp_path)
    result = _evaluate(
        manifest_path=fixture["manifest"],
        task="task",
        final_paths=[fixture["final"]],
        match_sample_report=fixture["match_report"],
        match_gold=fixture["match_gold"],
        match_gold_validation=fixture["match_validation"],
        abstention_sample_report=fixture["abstention_report"],
        abstention_gold=fixture["abstention_gold"],
        abstention_gold_validation=fixture["abstention_validation"],
        analysis_exclusion_paths=[],
        alpha=0.05,
        false_abstention_target=0.05,
        match_precision_target=0.90,
        typed_abstention_point_target=0.90,
        typed_abstention_lower_target=0.80,
        minimum_support=60,
    )
    release = tmp_path / "risk.release.json"
    release.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    fixture["match_gold"].write_text(fixture["match_gold"].read_text() + "{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_task_final_risk_release(release)
