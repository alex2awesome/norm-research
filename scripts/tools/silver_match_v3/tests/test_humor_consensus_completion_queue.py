import argparse
import json
import sys
from pathlib import Path

import pytest

import scripts.tools.silver_match_v3.freeze_humor_consensus_completion_queue as completion
from scripts.tools.silver_match_v3.common import sha256_file


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def _artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _completion_outputs(tmp_path: Path) -> tuple[dict, dict]:
    source_validation = _json(tmp_path / "source.validation.json", {"task": "humor"})
    truth = _jsonl(
        tmp_path / "truth" / "truth.all.jsonl",
        [
            {
                "task": "humor",
                "norm_uid": "u1",
                "source_group": "g1",
                "split": "train",
                "decision": "MATCH",
                "metric_id": "m1",
            }
        ],
    )
    manifest = _json(
        truth.parent / "MANIFEST.json",
        {
            "schema_version": "silver-match-v3-consensus-training-truth-manifest-v1",
            "status": "COMPLETE_EXACT_CONSENSUS_WITH_FROZEN_SPLITS",
            "task": "humor",
            "source_group_cross_split_count": 0,
            "blind_rows_training_eligible": 0,
            "inputs": {
                "pack_validation": {
                    "path": str(source_validation),
                    "sha256": sha256_file(source_validation),
                }
            },
            "outputs": {
                "all": {"path": str(truth), "sha256": sha256_file(truth), "count": 1}
            },
        },
    )
    eligible = _jsonl(tmp_path / "ce" / "eligible.jsonl", [json.loads(truth.read_text())])
    typed = _jsonl(tmp_path / "ce" / "typed.jsonl", [])
    ce_report = _json(
        tmp_path / "ce" / "REPORT.json",
        {
            "schema_version": "silver-match-v3-ce-eligible-truth-report-v1",
            "status": "PARTITIONED_WITHOUT_INFERRED_FAMILY_ANCHORS",
            "task": "humor",
            "source_groups_crossing_splits": 0,
            "input": {"path": str(truth), "sha256": sha256_file(truth), "count": 1},
            "outputs": {
                "eligible": {
                    "path": str(eligible),
                    "sha256": sha256_file(eligible),
                    "count": 1,
                },
                "typed_only": {
                    "path": str(typed),
                    "sha256": sha256_file(typed),
                    "count": 0,
                },
            },
        },
    )
    return (
        {
            "consensus_truth": str(truth),
            "consensus_truth_manifest": str(manifest),
            "ce_truth_report": str(ce_report),
            "poll_seconds": 1,
            "source_validation_sha256": sha256_file(source_validation),
        },
        _artifact(source_validation),
    )


def _plan(tmp_path: Path, *, ready: bool = True) -> tuple[dict, Path]:
    if ready:
        watch, source_ref = _completion_outputs(tmp_path)
    else:
        source = _json(tmp_path / "source.validation.json", {"task": "humor"})
        source_ref = _artifact(source)
        watch = {
            "consensus_truth": str(tmp_path / "missing" / "truth.all.jsonl"),
            "consensus_truth_manifest": str(tmp_path / "missing" / "MANIFEST.json"),
            "ce_truth_report": str(tmp_path / "missing" / "REPORT.json"),
            "poll_seconds": 1,
            "source_validation_sha256": sha256_file(source),
        }
    queue_path = _json(tmp_path / "queue.json", {"placeholder": True})
    final_root = tmp_path / "handoff"
    pairs = tmp_path / "production" / "pairs.jsonl"
    universe = tmp_path / "production" / "universe.jsonl"
    plan = {
        "schema_version": completion.SCHEMA,
        "status": completion.STATUS,
        "task": "humor",
        "bindings": {"consensus_source_validation": source_ref},
        "watch": watch,
        "commands": {
            "freeze_final_training_handoff": [
                sys.executable,
                "-u",
                "-m",
                "scripts.tools.silver_match_v3.freeze_humor_final_stack_handoff",
            ],
            "materialize_unlabeled_production_pairs": [
                sys.executable,
                "-u",
                "-m",
                "scripts.tools.silver_match_v3.materialize_nemotron_ce_production_pairs",
            ],
        },
        "outputs": {
            "final_handoff_root": str(final_root),
            "handoff_manifest": str(final_root / "HANDOFF_MANIFEST.json"),
            "final_stack_queue": str(final_root / "FINAL_STACK_QUEUE.json"),
            "production_pairs": str(pairs),
            "production_norm_universe": str(universe),
            "production_pair_report": str(pairs) + ".meta.json",
            "receipt_directory": str(tmp_path / "receipts"),
            "receipt_filename_rule": "<sha256-of-exact-json-bytes>.json",
            "cpu_log": str(tmp_path / "cpu.log"),
        },
        "execution": {
            "repo_root": str(tmp_path),
            "python": sys.executable,
            "only_permitted_modules": [
                "scripts.tools.silver_match_v3.freeze_humor_final_stack_handoff",
                "scripts.tools.silver_match_v3.materialize_nemotron_ce_production_pairs",
            ],
        },
        "safety": {
            "gpu_launches_permitted": False,
            "trainers_or_scorers_executed": False,
            "final_stack_freezer_reused_unchanged": True,
            "production_pair_labels_materialized": False,
            "primary_candidate_depth": 200,
            "fullbank_rescue_bound": True,
            "train_capture_is_diagnostic_only": True,
            "untouched_dev_capture_gate_passed": True,
            "progressive_scoring_policy_bound": True,
            "pre_ce_confidence_early_stopping_authorized": False,
            "release_ready": False,
        },
    }
    _json(queue_path, plan)
    return plan, queue_path


def _freeze_args(tmp_path: Path) -> argparse.Namespace:
    dummy = tmp_path / "dummy"
    candidate = _jsonl(tmp_path / "production-candidate.jsonl", [])
    values = dict(
        manifest=str(dummy),
        bank=str(dummy),
        hierarchy=str(dummy),
        existing_truth=str(dummy),
        existing_truth_report=str(dummy),
        consensus_truth=str(tmp_path / "future" / "truth.all.jsonl"),
        consensus_truth_manifest=str(tmp_path / "future" / "MANIFEST.json"),
        candidate_capture_freeze=str(dummy),
        pilot_selection=str(dummy),
        ce_model=str(tmp_path / "ce-model"),
        gemma_model=str(tmp_path / "gemma-model"),
        independent_labeling_guide=str(dummy),
        python=sys.executable,
        ce_trainer=str(dummy),
        ce_scorer=str(dummy),
        gemma_trainer=str(dummy),
        runtime_root=str(tmp_path / "runtime"),
        output_root=str(tmp_path / "handoff"),
        ce_seed=[11, 29],
        gepa_rule=[f"R{i}={dummy}" for i in range(1, 10)],
        gepa_train_only_audit=[f"R{i}={dummy}" for i in (7, 8, 9)],
        gemma_seed=31,
        pair_seed=37,
        maximum_pairs=400_000,
        global_negatives_per_norm=4,
        ce_context_chars=1600,
        gemma_max_candidates=8,
        gemma_order_seed=41,
        gemma_context_chars=1400,
        gemma_description_chars=520,
        gemma_example_chars=180,
        gemma_max_examples=2,
        consensus_source_validation=str(dummy),
        consensus_relocation_report=str(dummy),
        ce_truth_report=str(tmp_path / "future" / "CE_REPORT.json"),
        production_candidate=[f"jokes={candidate}"],
        production_candidate_audit=[f"jokes={dummy}"],
        production_rescue_candidate=[f"jokes={candidate}"],
        production_rescue_candidate_audit=[f"jokes={dummy}"],
        production_train_capture_diagnostic=str(dummy),
        production_dev_capture_gate=str(dummy),
        production_dev_policy_gate=str(dummy),
        production_pairs=str(tmp_path / "production" / "pairs.jsonl"),
        production_norm_universe=str(tmp_path / "production" / "universe.jsonl"),
        production_k=200,
        production_context_chars=1400,
        receipt_directory=str(tmp_path / "receipts"),
        repo_root=str(tmp_path),
        poll_seconds=1,
        queue_output=str(tmp_path / "completion.queue.json"),
        static_audit_receipt=None,
        static_audit_receipt_sha256=None,
    )
    return argparse.Namespace(**values)


def test_freezer_constructs_only_two_cpu_modules_and_no_training_launch(tmp_path, monkeypatch):
    args = _freeze_args(tmp_path)
    source = _json(tmp_path / "source.json", {})
    monkeypatch.setattr(
        completion,
        "_audit_static_inputs",
        lambda _args: {
            "bindings": {
                "source": _artifact(source),
                "consensus_source_validation": _artifact(source),
            },
            "candidate_audit": {},
            "pilot_audit": {},
            "pilot_recipe": {},
            "production_candidates": {
                "jokes": {"path": str(tmp_path / "production-candidate.jsonl")}
            },
            "production_candidate_audits": {"jokes": {"sha256": "a" * 64}},
            "production_rescue_candidates": {
                "jokes": {"path": str(tmp_path / "production-candidate.jsonl")}
            },
            "production_rescue_candidate_audits": {
                "jokes": {"sha256": "a" * 64}
            },
            "production_capture_evidence": {
                "train_diagnostic_only": {},
                "untouched_dev_promotion_gate": {},
            },
            "corpus_order": ["jokes"],
            "bank_source_sha256": "b" * 64,
        },
    )
    plan = completion.freeze_queue(args)
    completion.validate_queue(plan)
    modules = [command[3] for command in plan["commands"].values()]
    assert modules == [
        "scripts.tools.silver_match_v3.freeze_humor_final_stack_handoff",
        "scripts.tools.silver_match_v3.materialize_nemotron_ce_production_pairs",
    ]
    assert not any("train_nemotron" in module or "train_gemma" in module for module in modules)
    assert plan["safety"]["gpu_launches_permitted"] is False


def test_copied_remote_static_receipt_freezes_but_cannot_execute_without_bytes(
    tmp_path, monkeypatch
):
    args = _freeze_args(tmp_path)
    source = _json(tmp_path / "source.json", {})
    remote_candidate = tmp_path / "remote-host-only" / "candidate.jsonl"
    args.production_candidate = [f"jokes={remote_candidate}"]
    args.production_rescue_candidate = [f"jokes={remote_candidate}"]
    static = {
        "bindings": {
            "consensus_source_validation": _artifact(source),
            "remote_candidate": {
                "path": str(remote_candidate),
                "sha256": "c" * 64,
                "size_bytes": 999,
            },
            "production_candidate_jokes": {
                "path": str(remote_candidate),
                "sha256": "c" * 64,
                "size_bytes": 999,
            },
            "production_candidate_meta_jokes": {
                "path": str(remote_candidate) + ".meta.json",
                "sha256": "e" * 64,
                "size_bytes": 100,
            },
            "production_candidate_audit_jokes": {
                "path": str(remote_candidate) + ".audit.json",
                "sha256": "d" * 64,
                "size_bytes": 100,
            },
            "production_rescue_candidate_jokes": {
                "path": str(remote_candidate),
                "sha256": "c" * 64,
                "size_bytes": 999,
            },
            "production_rescue_candidate_meta_jokes": {
                "path": str(remote_candidate) + ".meta.json",
                "sha256": "e" * 64,
                "size_bytes": 100,
            },
            "production_rescue_candidate_audit_jokes": {
                "path": str(remote_candidate) + ".audit.json",
                "sha256": "d" * 64,
                "size_bytes": 100,
            },
        },
        "candidate_audit": {},
        "pilot_audit": {},
        "pilot_recipe": {},
        "production_candidates": {
            "jokes": {
                "path": str(remote_candidate),
                "sha256": "c" * 64,
                "meta": {"meta_sha256": "e" * 64},
            }
        },
        "production_candidate_audits": {
            "jokes": {
                "sha256": "d" * 64,
                "candidate_sha256": "c" * 64,
                "candidate_meta_sha256": "e" * 64,
            }
        },
        "production_rescue_candidates": {
            "jokes": {
                "path": str(remote_candidate),
                "sha256": "c" * 64,
                "meta": {"meta_sha256": "e" * 64},
            }
        },
        "production_rescue_candidate_audits": {
            "jokes": {
                "sha256": "d" * 64,
                "candidate_sha256": "c" * 64,
                "candidate_meta_sha256": "e" * 64,
            }
        },
        "production_capture_evidence": {
            "train_diagnostic_only": {},
            "untouched_dev_promotion_gate": {},
        },
        "corpus_order": ["jokes"],
        "bank_source_sha256": "b" * 64,
    }
    monkeypatch.setattr(completion, "_audit_static_inputs", lambda _args: static)
    receipt = completion.build_static_audit_receipt(args)
    receipt_path = _json(tmp_path / "copied-static-audit.json", receipt)
    args.static_audit_receipt = str(receipt_path)
    args.static_audit_receipt_sha256 = sha256_file(receipt_path)
    plan = completion.freeze_queue(args)
    assert plan["static_audit"]["provenance"]["receipt_sha256"] == sha256_file(
        receipt_path
    )
    with pytest.raises(ValueError, match="artifact changed"):
        completion.validate_queue(plan)


def test_copied_remote_static_receipt_rejects_command_contract_drift(
    tmp_path, monkeypatch
):
    args = _freeze_args(tmp_path)
    source = _json(tmp_path / "source.json", {})
    candidate = tmp_path / "production-candidate.jsonl"
    monkeypatch.setattr(
        completion,
        "_audit_static_inputs",
        lambda _args: {
            "bindings": {
                "consensus_source_validation": _artifact(source),
                "production_candidate_jokes": {
                    "path": str(candidate),
                    "sha256": "c" * 64,
                    "size_bytes": 0,
                },
                "production_candidate_meta_jokes": {
                    "path": str(candidate) + ".meta.json",
                    "sha256": "e" * 64,
                    "size_bytes": 0,
                },
                "production_candidate_audit_jokes": {
                    "path": str(candidate) + ".audit.json",
                    "sha256": "d" * 64,
                    "size_bytes": 0,
                },
                "production_rescue_candidate_jokes": {
                    "path": str(candidate),
                    "sha256": "c" * 64,
                    "size_bytes": 0,
                },
                "production_rescue_candidate_meta_jokes": {
                    "path": str(candidate) + ".meta.json",
                    "sha256": "e" * 64,
                    "size_bytes": 0,
                },
                "production_rescue_candidate_audit_jokes": {
                    "path": str(candidate) + ".audit.json",
                    "sha256": "d" * 64,
                    "size_bytes": 0,
                },
            },
            "candidate_audit": {},
            "pilot_audit": {},
            "pilot_recipe": {},
            "production_candidates": {
                "jokes": {
                    "path": str(candidate),
                    "sha256": "c" * 64,
                    "meta": {"meta_sha256": "e" * 64},
                }
            },
            "production_candidate_audits": {
                "jokes": {
                    "sha256": "d" * 64,
                    "candidate_sha256": "c" * 64,
                    "candidate_meta_sha256": "e" * 64,
                }
            },
            "production_rescue_candidates": {
                "jokes": {
                    "path": str(candidate),
                    "sha256": "c" * 64,
                    "meta": {"meta_sha256": "e" * 64},
                }
            },
            "production_rescue_candidate_audits": {
                "jokes": {
                    "sha256": "d" * 64,
                    "candidate_sha256": "c" * 64,
                    "candidate_meta_sha256": "e" * 64,
                }
            },
            "production_capture_evidence": {
                "train_diagnostic_only": {},
                "untouched_dev_promotion_gate": {},
            },
            "corpus_order": ["jokes"],
            "bank_source_sha256": "b" * 64,
        },
    )
    receipt_path = _json(
        tmp_path / "copied-static-audit.json",
        completion.build_static_audit_receipt(args),
    )
    args.static_audit_receipt = str(receipt_path)
    args.static_audit_receipt_sha256 = sha256_file(receipt_path)
    args.production_context_chars += 1
    with pytest.raises(ValueError, match="contract differs"):
        completion.freeze_queue(args)


def test_exact_consensus_and_ce_partition_must_be_hash_bound(tmp_path):
    plan, _ = _plan(tmp_path)
    result = completion.consensus_completion(plan)
    assert result["truth_count"] == 1
    manifest = Path(plan["watch"]["consensus_truth_manifest"])
    payload = json.loads(manifest.read_text())
    payload["outputs"]["all"]["sha256"] = "0" * 64
    _json(manifest, payload)
    with pytest.raises(ValueError, match="reference changed"):
        completion.consensus_completion(plan)


def test_coverage_candidate_audit_binds_exact_depth_candidate_and_meta(tmp_path):
    manifest = _json(tmp_path / "manifest.json", {"task": "humor"})
    candidate = _jsonl(tmp_path / "candidate.k200.jsonl", [{"norm_uid": "u1"}])
    meta = _json(candidate.with_suffix(candidate.suffix + ".meta.json"), {"k": 200})
    audit_path = _json(
        tmp_path / "candidate.k200.audit.json",
        {
            "schema_version": "silver-match-v3-production-candidate-audit-v1",
            "complete": True,
            "task": "humor",
            "corpus": "jokes",
            "expected_count": 1,
            "observed_count": 1,
            "expected_k": 200,
            "materialized_k": 200,
            "bank_count": 285,
            "bank_source_sha256": "b" * 64,
            "manifest_sha256": sha256_file(manifest),
            "candidate_count_distribution": {"200": 1},
            "candidate_inputs": {
                str(candidate.resolve()): {
                    "count": 1,
                    "sha256": sha256_file(candidate),
                    "meta_sha256": sha256_file(meta),
                }
            },
        },
    )
    result = completion._validate_production_candidate_audit(
        audit_path=audit_path,
        candidate_path=candidate,
        candidate_meta={"meta": str(meta)},
        manifest_path=manifest,
        corpus="jokes",
        bank_hash="b" * 64,
        bank_count=285,
        expected_count=1,
        expected_k=200,
    )
    assert result["candidate_sha256"] == sha256_file(candidate)
    candidate.write_text('{"norm_uid":"changed"}\n')
    with pytest.raises(ValueError, match="failed closed"):
        completion._validate_production_candidate_audit(
            audit_path=audit_path,
            candidate_path=candidate,
            candidate_meta={"meta": str(meta)},
            manifest_path=manifest,
            corpus="jokes",
            bank_hash="b" * 64,
            bank_count=285,
            expected_count=1,
            expected_k=200,
        )


def test_untouched_dev_capture_is_required_for_k200_promotion(tmp_path):
    candidate = _jsonl(tmp_path / "candidate.k200.jsonl", [{"norm_uid": "u1"}])
    labels = _jsonl(
        tmp_path / "dev.capture.labels.jsonl",
        [
            {
                "task": "humor",
                "split": "dev",
                "decision": "MATCH",
                "norm_uid": "u1",
                "metric_id": "m1",
                "current_bank_source_sha256": "b" * 64,
            }
        ],
    )
    group = {
        "gold_matches": 1,
        "confidence_level_one_sided": 0.95,
        "target_upper_bound": 0.05,
        "under_target_supported": True,
        "union_capture_rate": 1.0,
        "union_miss_upper_bound": 0.04,
        "unique_candidate_union_size": {"max": 200},
    }
    report = _json(
        tmp_path / "dev.capture.json",
        {
            "schema_version": "silver-match-v3-candidate-capture-v1",
            "k": 200,
            "candidate_inputs": {
                str(candidate.resolve()): sha256_file(candidate)
            },
            "label_inputs": {str(labels.resolve()): sha256_file(labels)},
            "groups": {"task_split:humor:dev": group},
        },
    )
    audit = completion._validate_capture_report(
        report,
        role="dev",
        candidate_path=candidate,
        bank_hash="b" * 64,
        require_gate=True,
    )
    assert audit["under_five_percent_supported"] is True
    payload = json.loads(report.read_text())
    payload["groups"]["task_split:humor:dev"]["union_miss_upper_bound"] = 0.051
    _json(report, payload)
    with pytest.raises(ValueError, match="does not support"):
        completion._validate_capture_report(
            report,
            role="dev",
            candidate_path=candidate,
            bank_hash="b" * 64,
            require_gate=True,
        )


def test_waiting_state_makes_no_mutation_or_subprocess_call(tmp_path, monkeypatch):
    plan, queue = _plan(tmp_path, ready=False)
    monkeypatch.setattr(
        completion,
        "_run_cpu",
        lambda *_args, **_kwargs: pytest.fail("waiting watcher launched a command"),
    )
    result = completion.run_once(plan, queue)
    assert result == {"status": "WAITING_FOR_EXACT_CONSENSUS", "mutations_performed": 0}


def test_completion_resumes_two_cpu_stages_and_seals_stable_content_receipt(
    tmp_path, monkeypatch
):
    plan, queue = _plan(tmp_path)
    state = {"handoff": False, "production": False}
    calls = []
    handoff = {"manifest": {"sha256": "a" * 64}, "queue": {"sha256": "b" * 64}}
    production = {"norm_count": 2, "pair_count": 100, "pairs": {"sha256": "c" * 64}}

    monkeypatch.setattr(
        completion,
        "_validate_handoff",
        lambda _plan: handoff if state["handoff"] else None,
    )
    monkeypatch.setattr(
        completion,
        "_validate_production",
        lambda _plan: production if state["production"] else None,
    )

    def fake_run(command, _plan):
        calls.append(command[3])
        if command[3].endswith("freeze_humor_final_stack_handoff"):
            state["handoff"] = True
        else:
            state["production"] = True

    monkeypatch.setattr(completion, "_run_cpu", fake_run)
    first = completion.run_once(plan, queue)
    second = completion.run_once(plan, queue)
    assert first["status"] == second["status"] == "COMPLETE_CPU_ONLY"
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert Path(first["receipt"]).name == first["receipt_sha256"] + ".json"
    assert calls == [
        "scripts.tools.silver_match_v3.freeze_humor_final_stack_handoff",
        "scripts.tools.silver_match_v3.materialize_nemotron_ce_production_pairs",
    ]
    assert first["gpu_processes_launched"] == 0


def test_partial_handoff_fails_closed_instead_of_overwriting(tmp_path):
    plan, _ = _plan(tmp_path)
    Path(plan["outputs"]["final_handoff_root"]).mkdir()
    with pytest.raises(ValueError, match="partial final-stack handoff"):
        completion._validate_handoff(plan)
