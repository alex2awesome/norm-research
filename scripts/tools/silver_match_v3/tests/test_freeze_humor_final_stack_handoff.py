import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file
from scripts.tools.silver_match_v3.freeze_humor_final_stack_handoff import (
    FINAL_EXPOSURES,
    _validate_train_only_prompt_audit,
    freeze,
    freeze_composite_gemma_prompt,
    join_truth,
    load_full_candidate_bundle,
)


def _json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _truth_fixture(tmp_path: Path):
    bank_hash = hashlib.sha256(b"current-humor-bank").hexdigest()
    bank = _json(
        tmp_path / "bank.json",
        {
            "task": "humor",
            "source_sha256": bank_hash,
            "metrics": [
                {
                    "task": "humor",
                    "metric_id": f"m{i}",
                    "name": f"metric {i}",
                    "description": f"definition {i}",
                    "examples": [],
                }
                for i in range(3)
            ],
        },
    )
    groups = {
        role: f"humor\x1fjokes\x1fsource\x1fs-{role}"
        for role in ("train", "dev", "test", "blind")
    }
    existing_rows = [
        {
            "task": "humor",
            "corpus": "jokes",
            "norm_uid": "u-train",
            "source_group": groups["train"],
            "split": "train",
            "decision": "MATCH",
            "metric_id": "m0",
            "acceptable_metric_ids": ["m0"],
            "confidence": "high",
            "trusted_ce_reason": "sonnet_high_unique_bridge_anchor_pass",
            "current_bank_source_sha256": bank_hash,
        },
        {
            "task": "humor",
            "corpus": "jokes",
            "norm_uid": "u-dev",
            "source_group": groups["dev"],
            "split": "dev",
            "decision": "NOISE",
            "metric_id": None,
            "confidence": "high",
            "reason": "garbled text",
            "current_bank_source_sha256": bank_hash,
        },
        {
            "task": "humor",
            "corpus": "jokes",
            "norm_uid": "u-test",
            "source_group": groups["test"],
            "split": "test",
            "decision": "MATCH",
            "metric_id": "m1",
            "acceptable_metric_ids": ["m1"],
            "confidence": "high",
            "reason": "exact test criterion",
            "current_bank_source_sha256": bank_hash,
        },
    ]
    consensus_rows = [
        {
            "task": "humor",
            "corpus": "jokes",
            "norm_uid": "u-blind",
            "source_group": groups["blind"],
            "split": "test",
            "collection_role": "blind",
            "decision": "NO_CANDIDATE_FITS",
            "metric_id": None,
            "confidence": "high",
            "reason": "explicit criterion is absent from this bank",
            "training_eligible": False,
            "dev_selection_eligible": False,
            "blind_evaluation_only": True,
            "current_bank_source_sha256": bank_hash,
        }
    ]
    existing = _jsonl(tmp_path / "existing.jsonl", existing_rows)
    consensus = _jsonl(tmp_path / "consensus.jsonl", consensus_rows)
    existing_report = _json(
        tmp_path / "existing.report.json",
        {
            "schema_version": "silver-match-v3-humor-ce-existing-truth-report-v1",
            "status": "CANONICAL_EXISTING_TRUTH_READY",
            "task": "humor",
            "bank_source_sha256": bank_hash,
            "output": {
                "path": str(existing),
                "sha256": sha256_file(existing),
                "count": len(existing_rows),
            },
        },
    )
    consensus_manifest = _json(
        tmp_path / "consensus.manifest.json",
        {
            "schema_version": "silver-match-v3-consensus-training-truth-manifest-v1",
            "status": "COMPLETE_EXACT_CONSENSUS_WITH_FROZEN_SPLITS",
            "task": "humor",
            "outputs": {
                "all": {
                    "path": str(consensus),
                    "sha256": sha256_file(consensus),
                    "count": len(consensus_rows),
                }
            },
        },
    )
    return {
        "bank_hash": bank_hash,
        "bank": bank,
        "groups": groups,
        "existing": existing,
        "existing_report": existing_report,
        "consensus": consensus,
        "consensus_manifest": consensus_manifest,
    }


def test_join_restores_blind_role_and_never_trains_test_or_blind(tmp_path: Path):
    fixture = _truth_fixture(tmp_path)
    rows, report = join_truth(
        existing_path=fixture["existing"],
        existing_report_path=fixture["existing_report"],
        consensus_path=fixture["consensus"],
        consensus_manifest_path=fixture["consensus_manifest"],
        bank_path=fixture["bank"],
    )
    by_uid = {row["norm_uid"]: row for row in rows}
    assert report["role_counts"] == {"train": 1, "dev": 1, "test": 1, "blind": 1}
    assert by_uid["u-blind"]["split"] == "blind"
    assert by_uid["u-blind"]["pre_handoff_frozen_split"] == "test"
    assert by_uid["u-blind"]["gradient_eligible"] is False
    assert by_uid["u-test"]["gradient_eligible"] is False
    assert by_uid["u-train"]["reason_is_provenance_only"] is True


def test_join_fails_on_cross_source_uid_or_source_role_leakage(tmp_path: Path):
    fixture = _truth_fixture(tmp_path)
    consensus_row = next(read_jsonl(fixture["consensus"]))
    consensus_row["norm_uid"] = "u-train"
    _jsonl(fixture["consensus"], [consensus_row])
    manifest = json.loads(fixture["consensus_manifest"].read_text())
    manifest["outputs"]["all"]["sha256"] = sha256_file(fixture["consensus"])
    _json(fixture["consensus_manifest"], manifest)
    with pytest.raises(ValueError, match="UID conflict"):
        join_truth(
            existing_path=fixture["existing"],
            existing_report_path=fixture["existing_report"],
            consensus_path=fixture["consensus"],
            consensus_manifest_path=fixture["consensus_manifest"],
            bank_path=fixture["bank"],
        )

    fixture = _truth_fixture(tmp_path / "groups")
    consensus_row = next(read_jsonl(fixture["consensus"]))
    consensus_row["source_group"] = fixture["groups"]["train"]
    _jsonl(fixture["consensus"], [consensus_row])
    manifest = json.loads(fixture["consensus_manifest"].read_text())
    manifest["outputs"]["all"]["sha256"] = sha256_file(fixture["consensus"])
    _json(fixture["consensus_manifest"], manifest)
    with pytest.raises(ValueError, match="source-disjoint roles"):
        join_truth(
            existing_path=fixture["existing"],
            existing_report_path=fixture["existing_report"],
            consensus_path=fixture["consensus"],
            consensus_manifest_path=fixture["consensus_manifest"],
            bank_path=fixture["bank"],
        )


def _candidate_freeze(tmp_path: Path, bank_hash: str, uids: list[str]) -> Path:
    paths = {}
    for lane_index, lane in enumerate(("dense", "lexical")):
        candidate = _jsonl(
            tmp_path / f"{lane}.jsonl",
            [
                {
                    "task": "humor",
                    "norm_uid": uid,
                    "bank_source_sha256": bank_hash,
                    "candidates": [
                        {"metric_id": metric, "rank": rank}
                        for rank, metric in enumerate(
                            (("m0", "m1", "m2") if lane_index == 0 else ("m2", "m1", "m0")),
                            1,
                        )
                    ],
                }
                for uid in uids
            ],
        )
        paths[lane] = {"path": str(candidate), "sha256": sha256_file(candidate)}
    return _json(
        tmp_path / "capture.json",
        {
            "schema_version": "silver-match-v3-candidate-capture-sequence-v1",
            "selection_split": "dev",
            "test_labels_used_for_selection": False,
            "candidate_inputs": paths,
            "available_lanes": ["dense:rank", "dense:word_rank", "lexical:rank"],
            "selected_sequence": ["dense:rank", "lexical:rank"],
        },
    )


def test_candidate_bundle_uses_every_hash_bound_retriever_input(tmp_path: Path):
    capture = _candidate_freeze(tmp_path, "bank", ["u"])
    specs, audit = load_full_candidate_bundle(capture, bank_hash="bank")
    assert [lane for lane, _ in specs] == ["dense", "lexical"]
    assert audit["all_frozen_candidate_inputs_used"] is True
    assert audit["selected_sequence_ignored_as_subset"] is True
    payload = json.loads(capture.read_text())
    payload["candidate_inputs"]["dense"]["sha256"] = "0" * 64
    _json(capture, payload)
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_full_candidate_bundle(capture, bank_hash="bank")


def test_train_only_prompt_audit_rejects_heldout_source_uid(tmp_path: Path):
    prompt = tmp_path / "r9.txt"
    prompt.write_text("rules only\n")
    items = _jsonl(
        tmp_path / "items.jsonl",
        [{"norm_uid": "heldout", "collection_role": "blind", "split": "blind"}],
    )
    audit = _json(
        tmp_path / "audit.json",
        {
            "schema_version": "silver-match-v3-humor-resolver-gepa-judge-audit-v1",
            "status": "FROZEN_TRAIN_ONLY_PROMPT_REFINEMENT_BEFORE_RESOLVER_LABELING",
            "task": "humor",
            "role_contract": {
                "allowed_role": "train",
                "dev_rows_read_for_rule_authorship": 0,
                "test_or_blind_rows_read_for_rule_authorship": 0,
                "resolver_votes_or_outcomes_read": 0,
                "rule_authorship_completed_before_resolver_labels": True,
            },
            "source_items": {"path": str(items), "sha256": sha256_file(items)},
            "prompt": {"path": str(prompt), "sha256": sha256_file(prompt)},
            "judged_train_disagreements": [{"norm_uid": "heldout"}],
        },
    )
    with pytest.raises(ValueError, match="non-train"):
        _validate_train_only_prompt_audit(audit, round_name="R9", prompt_path=prompt)


def test_composite_prompt_fails_on_any_component_hash_drift(tmp_path: Path):
    repo = Path(__file__).resolve().parents[4]
    prompt_root = repo / "scripts/tools/silver_match_v3/prompts"
    names = {
        "R1": "verify_humor_gepa_r1_precision.txt",
        "R2": "verify_humor_gepa_r2_precision.txt",
        "R3": "verify_humor_gepa_r3_exact_object.txt",
        "R4": "verify_humor_gepa_r4_speech_act_and_audio_owner.txt",
        "R5": "verify_humor_gepa_r5_criterion_nucleus.txt",
        "R6": "verify_humor_gepa_r6_falsification_and_abstention.txt",
        "R7": "verify_humor_gepa_r7_fullbank_resolver_train_only.txt",
        "R8": "verify_humor_gepa_r8_named_outcome_and_owner_train_only.txt",
        "R9": "verify_humor_gepa_r9_truth_structure_and_freshness_train_only.txt",
    }
    rounds = {name: prompt_root / filename for name, filename in names.items()}
    drifted = tmp_path / names["R8"]
    drifted.write_text(rounds["R8"].read_text() + "\ndrift\n", encoding="utf-8")
    rounds["R8"] = drifted
    audit_root = (
        repo
        / "outputs/silver_match_v3/humor/remediation_v3/model_improvement_v2/truth_consensus_v1"
    )
    with pytest.raises(ValueError, match="component hash drift: R8"):
        freeze_composite_gemma_prompt(
            guide_path=repo / "scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md",
            round_paths=rounds,
            train_only_audits={
                "R7": audit_root / "RESOLVER_R7_TRAIN_ONLY_JUDGE_AUDIT.json",
                "R8": audit_root / "RESOLVER_R8_TRAIN_ONLY_JUDGE_AUDIT.json",
                "R9": audit_root / "RESOLVER_R9_TRAIN_ONLY_JUDGE_AUDIT.json",
            },
            output_path=tmp_path / "composite.txt",
            manifest_path=tmp_path / "manifest.json",
        )


def test_full_handoff_freezes_two_seed_ce_and_weighted_gemma_without_readiness(
    tmp_path: Path,
):
    fixture = _truth_fixture(tmp_path)
    uids = ["u-train", "u-dev", "u-test", "u-blind"]
    norms = _jsonl(
        tmp_path / "norms.jsonl",
        [
            {
                "task": "humor",
                "corpus": "jokes",
                "norm_uid": uid,
                "source_id": f"s-{uid.removeprefix('u-')}",
                "row": index,
                "norm": f"human statement {uid}",
                "context": f"evidence {uid}",
            }
            for index, uid in enumerate(uids)
        ],
    )
    manifest = _json(
        tmp_path / "manifest.json",
        {
            "banks": {
                "humor": {
                    "path": str(fixture["bank"]),
                    "source_sha256": fixture["bank_hash"],
                    "count": 3,
                }
            },
            "corpora": {
                "jokes": {"task": "humor", "path": str(norms), "count": 4}
            },
        },
    )
    hierarchy = _json(
        tmp_path / "hierarchy.json",
        {
            "task": "humor",
            "n_r2_clusters_in": 3,
            "n_merged_groups": 1,
            "merged_groups": [{"metric_ids": ["m0", "m1", "m2"]}],
        },
    )
    capture = _candidate_freeze(tmp_path, fixture["bank_hash"], uids)

    ce_model = tmp_path / "ce-model"
    gemma_model = tmp_path / "gemma-model"
    ce_model.mkdir()
    gemma_model.mkdir()
    pilot_root = tmp_path / "pilot-winner"
    pilot_root.mkdir()
    run_config = _json(
        pilot_root / "run_config.json",
        {
            "model": str(ce_model),
            "dev_pairs": {"dev.jsonl": "abc"},
            "split_audit": {"source_group_overlap_count": 0},
            "max_length": 1024,
            "batch_size_per_rank": 8,
            "gradient_accumulation_steps": 4,
            "lora_learning_rate": 1e-4,
            "head_learning_rate": 1e-3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "lora": {"rank": 16, "alpha": 32, "dropout": 0.05},
            "attention": "eager",
            "dev_gate": {
                "minimum_exact_precision": 0.9,
                "minimum_wilson_lower": 0.8,
                "minimum_exact_predictions": 20,
            },
        },
    )
    base_manifest = _json(tmp_path / "base-manifest.json", {"status": "locked"})
    pilot = _json(
        tmp_path / "PILOT_SELECTION.json",
        {
            "schema_version": "silver-match-v3-humor-ce-pilot-selection-v1",
            "selection_data_role": "development_only",
            "test_opened_before_selection": False,
            "winner": "r16",
            "winner_record": {
                "root": str(pilot_root),
                "run_config_sha256": sha256_file(run_config),
            },
            "base_manifest": str(base_manifest),
            "base_manifest_sha256": sha256_file(base_manifest),
        },
    )
    repo = Path(__file__).resolve().parents[4]
    prompt_root = repo / "scripts/tools/silver_match_v3/prompts"
    gepa_rules = [
        f"R1={prompt_root / 'verify_humor_gepa_r1_precision.txt'}",
        f"R2={prompt_root / 'verify_humor_gepa_r2_precision.txt'}",
        f"R3={prompt_root / 'verify_humor_gepa_r3_exact_object.txt'}",
        f"R4={prompt_root / 'verify_humor_gepa_r4_speech_act_and_audio_owner.txt'}",
        f"R5={prompt_root / 'verify_humor_gepa_r5_criterion_nucleus.txt'}",
        f"R6={prompt_root / 'verify_humor_gepa_r6_falsification_and_abstention.txt'}",
        f"R7={prompt_root / 'verify_humor_gepa_r7_fullbank_resolver_train_only.txt'}",
        f"R8={prompt_root / 'verify_humor_gepa_r8_named_outcome_and_owner_train_only.txt'}",
        f"R9={prompt_root / 'verify_humor_gepa_r9_truth_structure_and_freshness_train_only.txt'}",
    ]
    audit_root = (
        repo
        / "outputs/silver_match_v3/humor/remediation_v3/model_improvement_v2/truth_consensus_v1"
    )
    output = tmp_path / "handoff"
    args = argparse.Namespace(
        manifest=str(manifest),
        bank=str(fixture["bank"]),
        hierarchy=str(hierarchy),
        existing_truth=str(fixture["existing"]),
        existing_truth_report=str(fixture["existing_report"]),
        consensus_truth=str(fixture["consensus"]),
        consensus_truth_manifest=str(fixture["consensus_manifest"]),
        candidate_capture_freeze=str(capture),
        pilot_selection=str(pilot),
        ce_model=str(ce_model),
        gemma_model=str(gemma_model),
        independent_labeling_guide=str(
            repo / "scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"
        ),
        gepa_rule=gepa_rules,
        gepa_train_only_audit=[
            f"R7={audit_root / 'RESOLVER_R7_TRAIN_ONLY_JUDGE_AUDIT.json'}",
            f"R8={audit_root / 'RESOLVER_R8_TRAIN_ONLY_JUDGE_AUDIT.json'}",
            f"R9={audit_root / 'RESOLVER_R9_TRAIN_ONLY_JUDGE_AUDIT.json'}",
        ],
        python=sys.executable,
        ce_trainer=str(repo / "scripts/tools/silver_match_v3/train_nemotron_cross_encoder.py"),
        ce_scorer=str(repo / "scripts/tools/silver_match_v3/run_nemotron_ce.py"),
        gemma_trainer=str(repo / "scripts/tools/silver_match_v3/train_gemma4_typed_lora.py"),
        runtime_root=str(tmp_path / "runtime"),
        output_root=str(output),
        ce_seed=[101, 202],
        gemma_seed=303,
        pair_seed=404,
        maximum_pairs=400_000,
        global_negatives_per_norm=0,
        ce_context_chars=100,
        gemma_max_candidates=3,
        gemma_order_seed=505,
        gemma_context_chars=100,
        gemma_description_chars=100,
        gemma_example_chars=100,
        gemma_max_examples=1,
    )
    result = freeze(args)
    queue = json.loads((output / "FINAL_STACK_QUEUE.json").read_text())
    assert result["status"] == "FROZEN_HANDOFF_NOT_PRODUCTION_OR_RELEASE_READY"
    assert queue["readiness"]["release_ready"] is False
    assert queue["readiness"]["production_ready"] is False
    prompt_manifest = json.loads((output / "prompts/MANIFEST.json").read_text())
    assert prompt_manifest["component_order"] == [
        "GUIDE", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"
    ]
    assert prompt_manifest["truth_examples_included"] is False
    assert prompt_manifest["example_uids_included"] is False
    assert [run["seed"] for run in queue["ce"]["runs"]] == [101, 202]
    for run in queue["ce"]["runs"]:
        command = run["command"]
        assert [
            int(command[index + 1])
            for index, value in enumerate(command)
            if value == "--exposure-budget"
        ] == list(FINAL_EXPOSURES)
        assert str(output / "ce/train.pairs.jsonl") in command
        assert str(output / "ce/dev.pairs.jsonl") in command
        assert str(output / "ce/test.pairs.jsonl") not in command
        assert str(output / "ce/blind.pairs.jsonl") not in command
    gemma_command = queue["gemma"]["command"]
    assert str(output / "gemma/dataset/train.jsonl") in gemma_command
    assert str(output / "gemma/dataset/dev.jsonl") in gemma_command
    assert str(output / "gemma/dataset/test.jsonl") not in gemma_command
    assert str(output / "gemma/dataset/blind.jsonl") not in gemma_command
    assert not list(tmp_path.glob(".handoff.staging-*"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze(args)
