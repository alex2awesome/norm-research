from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.tools.silver_match_v3.freeze_alltask_scaleout_handoff as handoff_module
from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_alltask_scaleout_handoff import (
    NONHUMOR_ORDER,
    PILOT_ORDER,
    TASK_ORDER,
    freeze_alltask_scaleout_handoff,
)
from scripts.tools.silver_match_v3.validate_alltask_scaleout_handoff import validate


CORPORA = {
    "humor": {"humor_multi": 77378},
    "code-review": {"crse": 39146, "github_code_review": 572639},
    "creative-writing": {
        "creative_writing": 4929,
        "litbench_rationales": 329514,
    },
    "legal-outcome-prediction": {
        "bva_opinions": 44678,
        "cavc_decisions": 28424,
        "courtlistener_opinions": 44026,
        "dol_arb": 3803,
        "law_se": 61872,
        "legaladvice_uk": 89770,
        "nlrb_decisions": 17037,
        "ptab_fwd": 5806,
        "reddit_supremecourt": 26154,
        "ttab_inter_partes": 4952,
    },
    "math-stackexchange": {
        "aops_forum": 14035,
        "competition_editorials": 32583,
        "math_se": 8238,
        "mathlib": 2069,
    },
    "peer-review": {"pr_review_feedback": 277420},
    "press-releases": {"press_releases": 26996},
    "notice-and-comment": {
        "nc_public_comments": 15870,
        "notice_and_comment": 5176,
    },
}
TRAIN_ROWS = {
    "code-review": 668,
    "creative-writing": 4866,
    "legal-outcome-prediction": 1567,
    "math-stackexchange": 1247,
    "peer-review": 1535,
    "press-releases": 2170,
    "notice-and-comment": 1219,
}
DEV_ROWS = {
    "code-review": 132,
    "creative-writing": 913,
    "legal-outcome-prediction": 350,
    "math-stackexchange": 237,
    "peer-review": 414,
    "press-releases": 629,
    "notice-and-comment": 209,
}
QUERIES = {
    "code-review": 2207,
    "creative-writing": 849,
    "legal-outcome-prediction": 1094,
    "math-stackexchange": 1304,
    "peer-review": 1000,
    "press-releases": 1102,
    "notice-and-comment": 992,
}
BANKS = {
    "humor": 285,
    "code-review": 133,
    "creative-writing": 371,
    "legal-outcome-prediction": 104,
    "math-stackexchange": 141,
    "peer-review": 88,
    "press-releases": 221,
    "notice-and-comment": 88,
}


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _binding(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _three_way(total: int) -> dict[str, int]:
    return {"EXACT": 1, "FAMILY": 1, "REJECT": total - 2}


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    corpus_rows = {}
    candidate_rows = {}
    for task, corpora in CORPORA.items():
        for corpus, count in corpora.items():
            corpus_rows[corpus] = {
                "task": task,
                "count": count,
                "path": f"/canonical/{corpus}.jsonl",
                "sha256": corpus.ljust(64, "0")[:64],
            }
            if task != "legal-outcome-prediction":
                candidate_rows[corpus] = {"expected_k": 50}
    coverage = _write(
        tmp_path / "coverage.json",
        {
            "manifest": {
                "path": "/canonical/manifest.json",
                "sha256": "a" * 64,
                "total_tasks": 8,
                "total_corpora": 23,
                "total_norms": 1732515,
            },
            "extractions": {"complete": True, "corpora": corpus_rows},
            "candidate_retrieval": {"corpora": candidate_rows},
            "retriever_selections": {
                "complete": True,
                "tasks": {task: {"chosen_name": "task-local"} for task in TASK_ORDER},
            },
        },
    )
    final = _write(
        tmp_path / "final.json",
        {
            "summary": {
                "expected_tasks": 8,
                "expected_corpora": 23,
                "expected_count": 1732515,
                "canonical_final_ready_tasks": 0,
                "canonical_final_ready_corpora": 0,
            },
            "tasks": {task: {"canonical_final_ready": False} for task in TASK_ORDER},
        },
    )

    truth_bindings = {}
    ce_records = []
    teacher_records = []
    for task in NONHUMOR_ORDER:
        role_path = tmp_path / "truth" / f"{task}.role_map.jsonl"
        role_path.parent.mkdir(parents=True, exist_ok=True)
        role_path.write_text("" if task == "notice-and-comment" else "{}\n", encoding="utf-8")
        queue = _write(
            tmp_path / "truth" / f"{task}.json",
            {
                "task": task,
                "leakage_failures": [],
                "missing_required_roles": (
                    ["norm_train", "norm_dev", "norm_blind"]
                    if task == "notice-and-comment"
                    else ["norm_blind"]
                ),
                "authoritative_role_map": {
                    **_binding(role_path),
                    "row_count": 0 if task == "notice-and-comment" else 1,
                },
            },
        )
        truth_bindings[task] = _binding(queue)

        report = _write(
            tmp_path / "ce" / task / "report.json",
            {
                "task": task,
                "contracts": {
                    "source_group_disjoint": True,
                    "cross_task_truth_borrowed": False,
                    "no_blind_or_test_rows_read": True,
                },
                "source_coverage": {"source_group_overlap": 0, "norm_uid_overlap": 0},
                "class_counts": {
                    "train": _three_way(TRAIN_ROWS[task]),
                    "dev": _three_way(DEV_ROWS[task]),
                },
            },
        )
        queue_path = _write(
            tmp_path / "ce" / task / "TRAIN_QUEUE.json",
            {"task": task, "status": "READY_FOR_TASK_LOCAL_TRAINING"},
        )
        ce_records.append(
            {
                "task": task,
                "train_rows": TRAIN_ROWS[task],
                "dev_rows": DEV_ROWS[task],
                "low_exact_dev_support": task in {"math-stackexchange", "notice-and-comment"},
                "class_counts": {
                    "train": _three_way(TRAIN_ROWS[task]),
                    "dev": _three_way(DEV_ROWS[task]),
                },
                "report": _binding(report),
                "train_queue": _binding(queue_path),
            }
        )

        teacher_report = _write(
            tmp_path / "teacher" / task / "report.json",
            {
                "task": task,
                "bank": {"metric_count": BANKS[task]},
                "contracts": {
                    "cross_task_borrowing": False,
                    "eval_heldout_blind_labels_used_as_truth": False,
                    "full_current_task_bank_presented": True,
                    "source_group_firewall": True,
                    "typed_abstention_enabled": True,
                },
            },
        )
        teacher_records.append(
            {
                "task": task,
                "eligible_unique_queries": QUERIES[task],
                "tier_query_counts": {"pilot": 256, "scale": QUERIES[task]},
                "report": _binding(teacher_report),
            }
        )

    truth = _write(
        tmp_path / "truth" / "FREEZE.json",
        {
            "task_order": list(NONHUMOR_ORDER),
            "executable_task_count": 0,
            "release_ready": False,
            "queues": truth_bindings,
        },
    )
    ce = _write(
        tmp_path / "ce" / "FREEZE.json",
        {
            "task_order": list(NONHUMOR_ORDER),
            "all_tasks_have_three_way_train_and_dev": True,
            "all_tasks_source_group_disjoint": True,
            "cross_task_truth_borrowed": False,
            "blind_or_test_rows_read": 0,
            "task_reports": ce_records,
        },
    )
    teacher = _write(
        tmp_path / "teacher" / "FREEZE.json",
        {
            "task_order": list(NONHUMOR_ORDER),
            "labels_collected": 0,
            "release_ready": False,
            "global_attrition_audit": {"unique_train_only_recovery_queries": 8548},
            "task_records": teacher_records,
        },
    )
    pilot = _write(
        tmp_path / "pilot.json",
        {
            "task_order": list(PILOT_ORDER),
            "tier": "pilot",
            "rows_per_task": 256,
            "notice_and_comment_launched": False,
            "core_or_scale_launched": False,
            "private_selection_ledger_staged": False,
            "teacher_visible_prior_labels_or_proposals": False,
        },
    )
    humor_truth = _write(
        tmp_path / "humor_truth.json",
        {
            "task": "humor",
            "total_count": 6600,
            "role_counts": {"blind": 1000, "dev": 600, "train": 5000},
            "bank_metric_count": 285,
        },
    )
    k200_meta = _write(
        tmp_path / "humor_k200.meta.json",
        {
            "task": "humor",
            "corpus": "humor_multi",
            "input_count": 77378,
            "new_count": 77378,
            "output_k": 200,
            "output_sha256": handoff_module.HUMOR_K200_CANDIDATE_SHA256,
            "manifest_sha256": "b614e345a07123f9fe79d9521351886107476d34cf2b09daa50efce71dc1356f",
        },
    )
    monkeypatch.setattr(
        handoff_module, "HUMOR_K200_META_SHA256", sha256_file(k200_meta)
    )
    k200_audit = _write(
        tmp_path / "humor_k200.audit.json",
        {
            "task": "humor",
            "corpus": "humor_multi",
            "complete": True,
            "observed_count": 77378,
            "materialized_k": 200,
            "bank_count": 285,
            "candidate_inputs": {
                "/remote/humor.k200.jsonl": {
                    "count": 77378,
                    "sha256": handoff_module.HUMOR_K200_CANDIDATE_SHA256,
                    "meta_sha256": handoff_module.HUMOR_K200_META_SHA256,
                }
            },
        },
    )
    monkeypatch.setattr(
        handoff_module, "HUMOR_K200_AUDIT_SHA256", sha256_file(k200_audit)
    )
    k200_capture = _write(
        tmp_path / "humor_k200.capture.json",
        {
            "candidate_inputs": {
                "/remote/humor.k200.jsonl": handoff_module.HUMOR_K200_CANDIDATE_SHA256
            },
            "overall": {
                "gold_matches": 549,
                "under_target_supported": True,
                "union_capture_rate": 0.9745,
                "union_miss_upper_bound": 0.0396,
            }
        },
    )
    monkeypatch.setattr(
        handoff_module, "HUMOR_K200_CAPTURE_SHA256", sha256_file(k200_capture)
    )
    k200_pairs = _write(
        tmp_path / "humor_k200.pairs.meta.json",
        {
            "task": "humor",
            "norm_count": 77378,
            "candidate_depth": 200,
            "pair_count": 15475600,
            "labels_present": False,
            "release_ready": False,
            "pairs": {"sha256": handoff_module.HUMOR_K200_PAIRS_SHA256},
            "norm_universe": {"sha256": handoff_module.HUMOR_UNIVERSE_SHA256},
            "corpora": {
                "humor_multi": {
                    "pair_count": 15475600,
                    "candidate_union": {
                        "sha256": handoff_module.HUMOR_K200_CANDIDATE_SHA256,
                        "meta_sha256": handoff_module.HUMOR_K200_META_SHA256,
                        "output_k": 200,
                        "complete_bank_lane_names": ["bge", "human"],
                    },
                }
            },
        },
    )
    monkeypatch.setattr(
        handoff_module, "HUMOR_K200_PAIRS_META_SHA256", sha256_file(k200_pairs)
    )
    full285_meta = _write(
        tmp_path / "humor_full285.meta.json",
        {
            "task": "humor",
            "corpus": "humor_multi",
            "input_count": 77378,
            "new_count": 77378,
            "output_k": 285,
            "output_sha256": handoff_module.HUMOR_FULL285_CANDIDATE_SHA256,
            "bank_source_sha256": "1b4a29d34b4ef4d999e0cb0b2d1125286372349ff6dfa21a6adc5bc8e76f0de9",
        },
    )
    monkeypatch.setattr(
        handoff_module, "HUMOR_FULL285_META_SHA256", sha256_file(full285_meta)
    )
    full285_audit = _write(
        tmp_path / "humor_full285.audit.json",
        {
            "task": "humor",
            "corpus": "humor_multi",
            "complete": True,
            "observed_count": 77378,
            "materialized_k": 285,
            "bank_count": 285,
            "candidate_inputs": {
                "/remote/humor.full285.jsonl": {
                    "count": 77378,
                    "sha256": handoff_module.HUMOR_FULL285_CANDIDATE_SHA256,
                    "meta_sha256": handoff_module.HUMOR_FULL285_META_SHA256,
                }
            },
        },
    )
    monkeypatch.setattr(
        handoff_module, "HUMOR_FULL285_AUDIT_SHA256", sha256_file(full285_audit)
    )
    humor_recipe = _write(
        tmp_path / "humor_recipe.json",
        {
            "schema_version": "silver-match-v3-nemotron-bidirectional-cross-encoder-v1",
            "model": "/models/nemotron-8b",
            "max_length": 1024,
            "bidirectional_concatenation": True,
            "pooling": "native_attention_mask_mean",
            "labels": ["EXACT", "FAMILY", "REJECT"],
            "lora": {
                "alpha": 64,
                "dropout": 0.05,
                "rank": 32,
                "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "lora_learning_rate": 0.00005,
            "head_learning_rate": 0.001,
            "sampler_weights": {"EXACT": 0.25, "FAMILY": 0.25, "REJECT": 0.5},
            "exposure_budgets": [10000, 25000, 50000],
        },
    )
    return {
        "coverage_path": coverage,
        "final_coverage_path": final,
        "truth_audit_freeze_path": truth,
        "ce_seed_freeze_path": ce,
        "teacher_pack_freeze_path": teacher,
        "pilot_freeze_path": pilot,
        "humor_truth_collection_path": humor_truth,
        "humor_k200_meta_path": k200_meta,
        "humor_k200_audit_path": k200_audit,
        "humor_k200_capture_path": k200_capture,
        "humor_k200_pairs_path": k200_pairs,
        "humor_full285_meta_path": full285_meta,
        "humor_full285_audit_path": full285_audit,
        "humor_recipe_path": humor_recipe,
    }


def test_freezes_exact_fail_closed_scaleout_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = freeze_alltask_scaleout_handoff(**_fixture(tmp_path, monkeypatch))

    assert result["task_order"] == list(TASK_ORDER)
    assert result["task_order"][0] == "humor"
    assert result["task_order"][-1] == "notice-and-comment"
    assert result["scope"]["norms"] == 1732515
    assert result["scope"]["reextraction_required"] is False
    assert result["launch_authorized"] is False
    assert result["summary"]["nonhumor_three_way_seed_train_rows"] == 13272
    assert result["summary"]["nonhumor_three_way_seed_dev_rows"] == 2884
    assert result["summary"]["nonhumor_train_only_teacher_queries"] == 8548
    assert result["tasks"]["humor"]["retrieval"]["required_primary_k"] == 200
    assert result["tasks"]["humor"]["retrieval"]["required_full_bank_rescue_k"] == 285
    assert result["tasks"]["humor"]["retrieval"]["primary_structurally_complete"] is True
    assert result["tasks"]["humor"]["retrieval"]["full_bank_rescue_structurally_complete"] is True
    assert result["tasks"]["humor"]["ce_training"]["staged_pair_count"] == 15475600
    assert result["tasks"]["legal-outcome-prediction"]["retrieval"]["legacy_missing_corpora"]
    assert result["tasks"]["notice-and-comment"]["truth"]["role_map_rows"] == 0
    assert result["recipe_seed"]["reuse_humor_weights_across_tasks"] is False
    assert all(not row["launch_authorized"] for row in result["tasks"].values())


def test_rejects_cross_task_ce_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    ce_path = paths["ce_seed_freeze_path"]
    payload = json.loads(ce_path.read_text(encoding="utf-8"))
    payload["cross_task_truth_borrowed"] = True
    _write(ce_path, payload)

    with pytest.raises(ValueError, match="CE seed isolation contracts failed"):
        freeze_alltask_scaleout_handoff(**paths)


def test_rejects_notice_and_comment_not_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    truth_path = paths["truth_audit_freeze_path"]
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    payload["task_order"][-2:] = reversed(payload["task_order"][-2:])
    _write(truth_path, payload)

    with pytest.raises(ValueError, match="truth queue order mismatch"):
        freeze_alltask_scaleout_handoff(**paths)


def test_independent_validator_rehashes_inputs_and_rejects_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    handoff = _write(
        tmp_path / "handoff.json", freeze_alltask_scaleout_handoff(**paths)
    )
    assert validate(handoff)["status"] == "PASS"

    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["tasks"]["humor"]["launch_authorized"] = True
    _write(handoff, payload)
    with pytest.raises(ValueError, match="task claims forbidden readiness: humor"):
        validate(handoff)
