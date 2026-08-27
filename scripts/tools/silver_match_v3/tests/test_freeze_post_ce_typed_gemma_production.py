from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.adjudicate_gemma import ordered_candidates
from scripts.tools.silver_match_v3.aggregate_nemotron_ce_seed_consensus import (
    CONSENSUS_REPORT_SCHEMA,
    CONSENSUS_SCHEMA,
)
from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.freeze_post_ce_typed_gemma_production import (
    INFERENCE_COMPLETE_STATUS,
    INFERENCE_META_SCHEMA,
    OUTPUT_SCHEMA,
    PAIRED_SCHEMA,
    _directory_ref,
    consolidate,
    freeze_queue,
)
from scripts.tools.silver_match_v3.freeze_task_final_stack_handoff import (
    PROMPT_MANIFEST_SCHEMA,
)
from scripts.tools.silver_match_v3.run_paired_gemma_lora_batch import (
    EXPECTED_MODEL_CONTENT_SHA256,
    EXPECTED_MODEL_FILE_COUNT,
)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _payload(decision: str, metric_id=None, *, reason="valid reason", parse_error=None):
    return {
        "decision": decision,
        "metric_id": metric_id,
        "confidence": "high",
        "reason": reason,
        "parse_error": parse_error,
        "raw_response": None,
    }


def _fixture(tmp_path: Path):
    task = "demo-task"
    bank_sha = "b" * 64
    metric_ids = ["m1", "m2", "m3"]
    bank = tmp_path / "bank.json"
    _json(
        bank,
        {
            "source_sha256": bank_sha,
            "metrics": [
                {"metric_id": metric, "name": metric, "description": f"criterion {metric}"}
                for metric in metric_ids
            ],
        },
    )
    corpus_uids = {"alpha": ["u1", "u2", "u3"], "beta": ["u4", "u5"]}
    norm_paths = {}
    all_norms = []
    for corpus, uids in corpus_uids.items():
        rows = [
            {
                "norm_uid": uid,
                "task": task,
                "corpus": corpus,
                "row": row,
                "norm": f"human criterion {uid}",
                "context": f"context {uid}",
                "source_group": f"group-{uid}",
            }
            for row, uid in enumerate(uids)
        ]
        path = tmp_path / f"{corpus}.jsonl"
        write_jsonl(path, rows)
        norm_paths[corpus] = path
        all_norms.extend(rows)
    manifest = tmp_path / "manifest.json"
    _json(
        manifest,
        {
            "banks": {
                task: {"path": str(bank), "count": 3, "source_sha256": bank_sha}
            },
            "corpora": {
                corpus: {"path": str(path), "count": len(corpus_uids[corpus]), "task": task}
                for corpus, path in norm_paths.items()
            },
        },
    )

    ce_rows = []
    for norm in all_norms:
        automatic = norm["norm_uid"] == "u1"
        states = {
            "seed-a": {
                "top_metric_id": "m1",
                "passes_frozen_gate": automatic,
            },
            "seed-b": {
                "top_metric_id": "m1" if automatic else "m2",
                "passes_frozen_gate": automatic,
            },
        }
        ce_rows.append(
            {
                "schema_version": CONSENSUS_SCHEMA,
                "norm_uid": norm["norm_uid"],
                "task": task,
                "corpus": norm["corpus"],
                "source_group": norm["source_group"],
                "split": "production",
                "decision": "MATCH" if automatic else "ROUTE_TO_ADJUDICATION",
                "routing_category": "MATCH" if automatic else "SEED_DISAGREEMENT",
                "automatic_match": automatic,
                "metric_id": "m1" if automatic else None,
                "candidate_count": 3,
                "seed_decisions": states,
                "candidates": [{"metric_id": metric} for metric in metric_ids],
                "provisional_routing_only": not automatic,
                "human_abstention_subtype_assigned": False,
            }
        )
    ce = tmp_path / "ce.jsonl"
    write_jsonl(ce, ce_rows)
    ce_report = tmp_path / "ce.report.json"
    _json(
        ce_report,
        {
            "schema_version": CONSENSUS_REPORT_SCHEMA,
            "status": "COMPLETE",
            "output_sha256": sha256_file(ce),
            "norm_count": 5,
            "validation": {
                "all_norms_preserved": True,
                "all_thresholds_from_checkpoint_dev": True,
                "seed_norm_candidate_source_split_universes_identical": True,
                "test_threshold_tuning_performed": False,
            },
        },
    )

    unions = {}
    for corpus, uids in corpus_uids.items():
        union = tmp_path / f"{corpus}.union.jsonl"
        write_jsonl(
            union,
            [
                {
                    "norm_uid": uid,
                    "task": task,
                    "corpus": corpus,
                    "row": row,
                    "bank_source_sha256": bank_sha,
                    "candidates": [
                        {"metric_id": metric, "rank": rank}
                        for rank, metric in enumerate(metric_ids, 1)
                    ],
                }
                for row, uid in enumerate(uids)
            ],
        )
        _json(
            union.with_suffix(".jsonl.meta.json"),
            {
                "manifest_sha256": sha256_file(manifest),
                "output_sha256": sha256_file(union),
                "task": task,
                "corpus": corpus,
                "bank_source_sha256": bank_sha,
                "input_count": len(uids),
                "output_k": 3,
                "union": {
                    "lanes": [
                        {"name": "nemotron", "kind": "complete-bank"},
                        {"name": "bge", "kind": "complete-bank"},
                    ]
                },
            },
        )
        unions[corpus] = union

    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (model / name).write_text("{}")
    model_inventory = tmp_path / "model.inventory.json"
    _json(
        model_inventory,
        {
            "status": "FROZEN_CONTENT_HASH_INVENTORY",
            "root": str(model),
            "file_count": EXPECTED_MODEL_FILE_COUNT,
            "content_inventory_sha256": EXPECTED_MODEL_CONTENT_SHA256,
        },
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}))
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return a typed decision.")
    prompt_manifest = tmp_path / "prompt.manifest.json"
    _json(
        prompt_manifest,
        {
            "schema_version": PROMPT_MANIFEST_SCHEMA,
            "status": "FROZEN_TASK_LOCAL_RULES_ONLY_COMPOSITE",
            "task": task,
            "truth_examples_included": False,
            "truth_labels_votes_or_outcomes_included": False,
            "example_uids_included": False,
            "output": {"path": str(prompt), "sha256": sha256_file(prompt)},
        },
    )
    train_dataset = tmp_path / "typed.train.jsonl"
    dev_dataset = tmp_path / "typed.dev.jsonl"
    write_jsonl(
        train_dataset,
        [
            {
                "task": task,
                "norm_uid": "train-1",
                "source_group": "train-group",
                "split": "train",
                "gradient_eligible": True,
            }
        ],
    )
    write_jsonl(
        dev_dataset,
        [
            {
                "task": task,
                "norm_uid": "dev-1",
                "source_group": "dev-group",
                "split": "dev",
                "gradient_eligible": False,
            }
        ],
    )
    training_report = tmp_path / "training.report.json"
    _json(
        training_report,
        {
            "schema_version": "silver-match-v3-gemma4-typed-lora-train-report-v2",
            "status": "COMPLETE_DEV_SELECTED_ADAPTER_FRESH_RELOAD_VERIFIED",
            "model": str(model),
            "dataset": {
                "path": str(train_dataset),
                "sha256": sha256_file(train_dataset),
            },
            "dev_dataset": {
                "path": str(dev_dataset),
                "sha256": sha256_file(dev_dataset),
            },
            "model_inventory": {
                "path": str(model_inventory),
                "sha256": sha256_file(model_inventory),
            },
            "selection": {
                "status": "SELECTED_ON_DEV_ONLY",
                "selection_split": "dev",
                "test_or_blind_data_read": False,
            },
            "adapter": {
                "directory": str(adapter),
                "config": {
                    "path": str(adapter / "adapter_config.json"),
                    "sha256": sha256_file(adapter / "adapter_config.json"),
                },
                "weights": {
                    "path": str(adapter / "adapter_model.safetensors"),
                    "sha256": sha256_file(adapter / "adapter_model.safetensors"),
                },
                "adapter_only": True,
                "inference_reload_verified": True,
                "fresh_base_reload_verified": True,
                "content": _directory_ref(adapter),
            },
            "source_disjoint_audit": {
                "status": "PASS_SOURCE_DISJOINT_HELDOUT_GRADIENT_EXCLUDED",
                "norm_uid_overlap_count": 0,
                "source_group_overlap_count": 0,
                "heldout_gradient_eligible_count": 0,
            },
        },
    )
    return {
        "task": task,
        "bank_sha": bank_sha,
        "manifest": manifest,
        "ce": ce,
        "ce_report": ce_report,
        "unions": unions,
        "training_report": training_report,
        "adapter": adapter,
        "prompt_manifest": prompt_manifest,
        "model": model,
        "model_inventory": model_inventory,
    }


def _freeze(fixture, tmp_path: Path):
    candidate_output = tmp_path / "unresolved.jsonl"
    candidate_report = tmp_path / "unresolved.report.json"
    queue = tmp_path / "queue.json"
    inference_root = tmp_path / "inference"
    payload = freeze_queue(
        manifest_path=fixture["manifest"],
        task=fixture["task"],
        ce_path=fixture["ce"],
        ce_report_path=fixture["ce_report"],
        candidate_paths=fixture["unions"],
        training_report_path=fixture["training_report"],
        adapter_path=fixture["adapter"],
        prompt_manifest_path=fixture["prompt_manifest"],
        model_path=fixture["model"],
        model_inventory_path=fixture["model_inventory"],
        python_path=Path(sys.executable),
        candidate_output_path=candidate_output,
        candidate_report_path=candidate_report,
        queue_output_path=queue,
        inference_output_root=inference_root,
        max_candidates=3,
    )
    return payload, queue, candidate_output, candidate_report, inference_root


def test_freeze_emits_every_and_only_ce_routed_norm(tmp_path: Path):
    fixture = _fixture(tmp_path)
    queue, _, candidates, report_path, _ = _freeze(fixture, tmp_path)
    rows = list(read_jsonl(candidates))
    assert [row["norm_uid"] for row in rows] == ["u2", "u3", "u4", "u5"]
    assert all(len(row["candidates"]) == 3 for row in rows)
    report = json.loads(report_path.read_text())
    assert report["canonical_count"] == 5
    assert report["ce_automatic_match_count"] == 1
    assert report["ce_routed_count"] == 4
    assert report["routing_audit"]["every_ce_routed_norm_emitted_once"] is True
    assert queue["backend"] == "direct_batch_vllm_not_openai_server"
    assert queue["orders"] == ["original", "hashed"]
    assert queue["production_policy"]["base_arm_used_for_consolidation"] is False
    assert any("run_paired_gemma_lora_batch" in value for value in queue["command"])


def _paired_row(candidate, order, lora, *, freeze_sha, model, adapter, prompt_sha):
    uid = candidate["norm_uid"]
    cards = ordered_candidates(candidate["candidates"], order, uid)
    return {
        "schema_version": PAIRED_SCHEMA,
        "norm_uid": uid,
        "task": candidate["task"],
        "corpus": candidate["corpus"],
        "row": candidate["row"],
        "order_mode": order,
        "candidate_ids": [row["metric_id"] for row in cards],
        "candidate_bank_source_sha256": candidate["bank_source_sha256"],
        "inference_freeze_sha256": freeze_sha,
        "model": model,
        "adapter": adapter,
        "prompt_sha256": prompt_sha,
        "base_item_prompt_sha256": f"item-{uid}",
        "lora_item_prompt_sha256": f"item-{uid}",
        # Deliberately contradictory base predictions prove they are ignored.
        "base": _payload("NOISE"),
        "lora": lora,
    }


def test_consolidation_promotes_only_valid_order_stable_lora(tmp_path: Path):
    fixture = _fixture(tmp_path)
    _, queue_path, candidates_path, _, inference_root = _freeze(fixture, tmp_path)
    queue = json.loads(queue_path.read_text())
    candidates = {row["norm_uid"]: row for row in read_jsonl(candidates_path)}
    left = {
        "u2": _payload("MATCH", "m1"),
        "u3": _payload("NO_CANDIDATE_FITS"),
        "u4": _payload("MATCH", "m1"),
        "u5": _payload("INVALID_OUTPUT", reason="parse failed", parse_error="no_json"),
    }
    right = {
        "u2": _payload("MATCH", "m1"),
        "u3": _payload("NO_CANDIDATE_FITS"),
        "u4": _payload("MATCH", "m2"),
        "u5": _payload("NOISE"),
    }
    inference_root.mkdir(parents=True, exist_ok=True)
    freeze = inference_root / "truth_blind_inference.freeze.json"
    _json(
        freeze,
        {
            "schema_version": "silver-match-v3-paired-gemma4-lora-inference-freeze-v1",
            "status": "FROZEN_BEFORE_PAIRED_MODEL_INFERENCE",
            "task": fixture["task"],
            "backend": "direct_batch_vllm_not_openai_server",
            "truth_firewall": {
                "truth_read": False,
                "truth_path_argument_exists": False,
                "scoring_in_separate_process_after_predictions": True,
            },
            "inputs": {
                "manifest": queue["inputs"]["manifest"],
                "candidates": queue["inputs"]["unresolved_candidates"],
                "prompt_components": [queue["inputs"]["prompt"]],
                "runner_script": queue["implementations"]["paired_runner"],
                "model_inventory": queue["inputs"]["model_inventory"],
                "model_identity": {"path": queue["inputs"]["model_path"]},
                "adapter_identity": {
                    "path": queue["inputs"]["adapter"]["path"],
                    "files": {
                        row["relative_path"]: {"sha256": row["sha256"]}
                        for row in queue["inputs"]["adapter"]["files"]
                    },
                },
            },
            "paired_contract": {
                "systems": ["base", "lora"],
                "orders": ["original", "hashed"],
                "same_model_instance": True,
                "same_candidate_set_and_rendered_prompt_within_each_pair": True,
                "no_hyperparameter_or_seed_search": True,
            },
        },
    )
    freeze_sha = sha256_file(freeze)
    prompt_sha = "p" * 64
    row_kwargs = {
        "freeze_sha": freeze_sha,
        "model": queue["inputs"]["model_path"],
        "adapter": queue["inputs"]["adapter"]["path"],
        "prompt_sha": prompt_sha,
    }
    original = inference_root / "paired.original.jsonl"
    hashed = inference_root / "paired.hashed.jsonl"
    write_jsonl(
        original,
        [
            _paired_row(candidates[uid], "original", left[uid], **row_kwargs)
            for uid in candidates
        ],
    )
    write_jsonl(
        hashed,
        [
            _paired_row(candidates[uid], "hashed", right[uid], **row_kwargs)
            for uid in candidates
        ],
    )
    meta = inference_root / "paired_inference.meta.json"
    _json(
        meta,
        {
            "schema_version": INFERENCE_META_SCHEMA,
            "status": INFERENCE_COMPLETE_STATUS,
            "task": fixture["task"],
            "truth_read": False,
            "backend": "direct_batch_vllm_not_openai_server",
            "same_loaded_base_model_instance_for_both_arms": True,
            "prompt_sha256": prompt_sha,
            "inference_freeze": {
                "path": str(freeze),
                "sha256": freeze_sha,
            },
            "outputs": {
                "original": {
                    "path": str(original),
                    "sha256": sha256_file(original),
                    "count": 4,
                },
                "hashed": {
                    "path": str(hashed),
                    "sha256": sha256_file(hashed),
                    "count": 4,
                },
            },
        },
    )
    output = tmp_path / "consolidated.jsonl"
    report_path = tmp_path / "consolidated.report.json"
    report = consolidate(
        queue_path=queue_path,
        paired_original_path=original,
        paired_hashed_path=hashed,
        inference_meta_path=meta,
        output_path=output,
        report_output_path=report_path,
    )
    rows = {row["norm_uid"]: row for row in read_jsonl(output)}
    assert len(rows) == 5
    assert all(row["schema_version"] == OUTPUT_SCHEMA for row in rows.values())
    assert rows["u1"]["production_route"] == "CE_AUTOMATIC_SAME_LEAF_TWO_GATE"
    assert (rows["u2"]["decision"], rows["u2"]["metric_id"]) == ("MATCH", "m1")
    assert rows["u2"]["requires_exhaustive_rescue"] is True
    assert rows["u3"]["decision"] == "NO_CANDIDATE_FITS"
    assert rows["u3"]["requires_exhaustive_rescue"] is True
    assert rows["u4"]["decision"] == "UNSTABLE_MATCH"
    assert rows["u4"]["requires_exhaustive_rescue"] is True
    assert rows["u5"]["decision"] == "INVALID_OUTPUT"
    assert rows["u5"]["requires_exhaustive_rescue"] is True
    assert all(row["base_arm_used_for_production"] is False for row in rows.values())
    assert report["counts"] == {
        "canonical": 5,
        "ce_automatic_match": 1,
        "ce_routed_to_lora": 4,
        "lora_stable_match": 1,
        "lora_stable_typed_abstention": 1,
        "lora_invalid_rescue": 1,
        "lora_disagreement_rescue": 1,
        "order_failure_rescue_required": 2,
        "exhaustive_rescue_required": 4,
    }
    assert report["routing_audit"]["base_arm_used_for_production"] is False


def test_freeze_fails_if_union_drops_a_canonical_norm(tmp_path: Path):
    fixture = _fixture(tmp_path)
    path = fixture["unions"]["beta"]
    write_jsonl(path, list(read_jsonl(path))[:1])
    meta_path = path.with_suffix(".jsonl.meta.json")
    meta = json.loads(meta_path.read_text())
    meta["output_sha256"] = sha256_file(path)
    _json(meta_path, meta)
    with pytest.raises(ValueError, match="length mismatch"):
        _freeze(fixture, tmp_path)


def test_freeze_rejects_non_dev_selected_adapter(tmp_path: Path):
    fixture = _fixture(tmp_path)
    report = json.loads(fixture["training_report"].read_text())
    report["selection"]["selection_split"] = "test"
    _json(fixture["training_report"], report)
    with pytest.raises(ValueError, match="complete dev-selected"):
        _freeze(fixture, tmp_path)
