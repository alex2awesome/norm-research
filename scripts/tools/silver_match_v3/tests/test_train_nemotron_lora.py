import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from scripts.tools.silver_match_v3.common import stable_uid
from scripts.tools.silver_match_v3.train_nemotron_lora import (
    LabeledNorm,
    audit_source_splits,
    build_triplets,
    epoch_promotion_passes,
    epoch_selection_key,
    format_query,
    load_universe,
    merge_match_teachers,
    retrieval_metrics,
    select_hard_negatives,
    source_group_key,
    split_source_group,
    validate_adapter_artifact,
)


def _label(uid="u0", split="train", metric_id="a0", acceptable=("a0",)):
    return LabeledNorm(
        norm_uid=uid,
        corpus="peer",
        task="peer-review",
        source_group="peer-review\x1fpeer\x1fpaper\x1fp0",
        split=split,
        query=format_query("Task: peer-review. Human evaluative statement: be clearer"),
        metric_id=metric_id,
        acceptable_metric_ids=acceptable,
        teacher_sources=("sonnet_audit",),
    )


def test_source_split_uses_paper_boundary_and_is_stable():
    first = source_group_key(
        {
            "task": "peer-review",
            "corpus": "peer",
            "paper_id": "paper-7",
            "source_id": "review-1",
            "norm_uid": "n1",
        }
    )
    second = source_group_key(
        {
            "task": "peer-review",
            "corpus": "peer",
            "paper_id": "paper-7",
            "source_id": "review-2",
            "norm_uid": "n2",
        }
    )
    assert first == second
    assert split_source_group(first) == split_source_group(second)


def test_split_audit_rejects_cross_split_source_leakage():
    left = _label(split="train")
    right = LabeledNorm(**{**left.__dict__, "norm_uid": "u1", "split": "dev"})
    with pytest.raises(ValueError, match="source-group split leakage"):
        audit_source_splits([left, right])


def test_merge_teachers_deduplicates_agreement_and_surfaces_conflict():
    common = {
        "decision": "MATCH",
        "norm_uid": "u0",
        "task": "humor",
        "current_bank_source_sha256": "bank",
    }
    merged, audit = merge_match_teachers(
        [
            ("one.jsonl", {**common, "metric_id": "a0", "label_source": "sonnet"}),
            ("two.jsonl", {**common, "metric_id": "a0", "label_source": "gemma"}),
            ("three.jsonl", {**common, "metric_id": "a1", "label_source": "audit"}),
            ("one.jsonl", {"decision": "BANK_GAP", "norm_uid": "u9"}),
        ]
    )
    assert not merged
    assert audit["conflicting_match_uids"] == 1
    assert audit["rows_by_decision"] == {"BANK_GAP": 1, "MATCH": 3}


def test_forced_top3_is_one_weak_primary_with_all_three_acceptable():
    rows = []
    for rank, metric_id in enumerate(("a2", "a7", "a4"), 1):
        rows.append(
            (
                "forced.jsonl",
                {
                    "decision": "MATCH",
                    "norm_uid": "u0",
                    "metric_id": metric_id,
                    "forced_rank": rank,
                    "label_source": "sonnet_forced_top3",
                    "supervision_strength": "weak_forced_positive",
                    "task": "code-review",
                },
            )
        )
    merged, audit = merge_match_teachers(rows)
    assert merged["u0"]["metric_id"] == "a2"
    assert merged["u0"]["acceptable_metric_ids"] == ["a2", "a4", "a7"]
    assert merged["u0"]["supervision_strength"] == "weak_forced_top3"
    assert audit["weak_forced_groups"] == 1
    assert audit["conflicting_match_uids"] == 0


def test_gradient_locked_teacher_is_rejected_by_trainer():
    with pytest.raises(ValueError, match="gradient-locked teacher"):
        merge_match_teachers(
            [
                (
                    "locked.jsonl",
                    {
                        "decision": "MATCH",
                        "norm_uid": "u0",
                        "metric_id": "a0",
                        "gradient_eligible": False,
                    },
                )
            ]
        )


def test_hard_negatives_exclude_gold_family_and_use_both_lanes():
    selected = select_hard_negatives(
        query_scores=[1.0, 0.95, 0.9, 0.8],
        sibling_scores=[1.0, 0.98, 0.5, 0.97],
        bank_ids=["a0", "a1", "a2", "a3"],
        excluded_ids={"a0", "a1"},
        pool_size=4,
        count=2,
    )
    assert [(row["metric_id"], row["negative_strategy"]) for row in selected] == [
        ("a2", "query_hard"),
        ("a3", "metric_sibling"),
    ]


def test_build_triplets_never_uses_acceptable_metric_as_negative():
    bank = [
        {"metric_id": "a0", "name": "Clarity", "description": "clear", "examples": []},
        {
            "metric_id": "a1",
            "name": "Clarity",
            "description": "also clear",
            "examples": [],
        },
        {"metric_id": "a2", "name": "Novelty", "description": "new", "examples": []},
    ]
    labels = [_label(acceptable=("a0", "a1"))]
    query_embeddings = np.asarray([[1.0, 0.0]])
    bank_embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    triplets = build_triplets(
        labels,
        bank,
        query_embeddings,
        bank_embeddings,
        pool_size=3,
        negatives_per_positive=1,
    )
    assert triplets[0]["negative_metric_id"] == "a2"
    assert "metric_id" not in triplets[0]


def test_retrieval_reports_exact_and_name_family_recall():
    metrics = retrieval_metrics(
        np.asarray([[0.8, 0.9, 0.1]]),
        gold_ids=["a0"],
        bank_ids=["a0", "a1", "a2"],
        family_by_id={"a0": "clarity", "a1": "clarity", "a2": "novelty"},
        ks=(1, 2),
    )
    assert metrics["exact"]["recall_at_1"] == 0.0
    assert metrics["exact"]["recall_at_2"] == 1.0
    assert metrics["name_family"]["recall_at_1"] == 1.0


def test_depth_selection_can_choose_rank_gain_when_recall_50_is_saturated():
    base = {
        "recall_at_1": 0.1,
        "recall_at_3": 0.2,
        "recall_at_5": 0.3,
        "recall_at_10": 0.4,
        "recall_at_16": 0.5,
        "recall_at_30": 0.8,
        "recall_at_50": 1.0,
        "mrr": 0.25,
    }
    trained = {**base, "recall_at_16": 0.6, "recall_at_30": 0.9, "mrr": 0.31}
    base_key = epoch_selection_key(base, 50, "depth_lexicographic")
    trained_key = epoch_selection_key(trained, 50, "depth_lexicographic")
    assert trained_key > base_key
    assert epoch_promotion_passes(
        base_key,
        trained_key,
        policy="depth_lexicographic",
        minimum_primary_gain=0.01,
    )


def test_depth_selection_never_trades_away_primary_recall():
    before = (1.0, 0.25, 0.8)
    after = (0.99, 0.9, 1.0)
    assert not epoch_promotion_passes(
        before,
        after,
        policy="depth_lexicographic",
        minimum_primary_gain=0.0,
    )


def test_legacy_single_k_gate_is_unchanged():
    assert epoch_promotion_passes(
        (0.8,), (0.81,), policy="single_k", minimum_primary_gain=0.01
    )
    assert not epoch_promotion_passes(
        (0.8,), (0.8,), policy="single_k", minimum_primary_gain=0.01
    )


def test_query_instruction_is_applied_only_once():
    value = format_query("  Task: humor.  Human evaluative statement: weak ending ")
    assert value.startswith("Instruct: Given a human evaluative statement")
    assert value.count("\nQuery: ") == 1
    assert value.endswith("Human evaluative statement: weak ending")


def test_adapter_validator_accepts_only_adapter_tensors(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        """{
          "peft_type": "LORA",
          "base_model_name_or_path": "/immutable/base",
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }""",
        encoding="utf-8",
    )
    save_file(
        {"base_model.layer.q_proj.lora_A.weight": np.ones((2, 3), dtype=np.float32)},
        tmp_path / "adapter_model.safetensors",
    )
    audit = validate_adapter_artifact(tmp_path)
    assert audit["tensor_count"] == 1
    assert audit["parameter_count_from_shapes"] == 6


def test_teacher_manifest_bridge_is_exact_unique_and_restores_document_groups(tmp_path):
    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    (current / "banks").mkdir(parents=True)
    (current / "norms").mkdir()
    (legacy / "norms").mkdir(parents=True)
    bank = {
        "task": "humor",
        "source_sha256": "bank-hash",
        "metrics": [
            {
                "metric_id": "a0",
                "name": "Timing",
                "description": "timing",
                "examples": [],
            },
            {
                "metric_id": "a1",
                "name": "Originality",
                "description": "novel",
                "examples": [],
            },
        ],
    }
    (current / "banks/humor.json").write_text(json.dumps(bank), encoding="utf-8")

    current_rows = []
    legacy_rows = []
    teachers = []
    for doc in range(100):
        for signal in range(2):
            text = f"document {doc} signal {signal} should be funnier"
            old_uid = stable_uid("humor", doc, signal, text)
            new_uid = stable_uid("humor_multi", doc, signal, text)
            legacy_rows.append(
                {
                    "norm_uid": old_uid,
                    "corpus": "humor",
                    "task": "humor",
                    "source_id": str(len(legacy_rows)),
                    "norm": text,
                }
            )
            current_rows.append(
                {
                    "norm_uid": new_uid,
                    "corpus": "humor_multi",
                    "task": "humor",
                    "source_id": f"doc-{doc}",
                    "norm": text,
                }
            )
            teachers.append(
                {
                    "norm_uid": old_uid,
                    "task": "humor",
                    "decision": "MATCH",
                    "metric_id": "a0",
                    "current_bank_source_sha256": "bank-hash",
                    "label_source": "sonnet",
                }
            )
    # An exact quote with two production occurrences is intentionally unsafe.
    old_ambiguous_uid = stable_uid("humor", "ambiguous quote")
    legacy_rows.append(
        {
            "norm_uid": old_ambiguous_uid,
            "corpus": "humor",
            "task": "humor",
            "source_id": "legacy-ambiguous",
            "norm": "same quote",
        }
    )
    teachers.append(
        {
            "norm_uid": old_ambiguous_uid,
            "task": "humor",
            "decision": "MATCH",
            "metric_id": "a0",
            "current_bank_source_sha256": "bank-hash",
        }
    )
    for suffix in ("x", "y"):
        current_rows.append(
            {
                "norm_uid": stable_uid("humor_multi", suffix, "same quote"),
                "corpus": "humor_multi",
                "task": "humor",
                "source_id": f"ambiguous-{suffix}",
                "norm": "same quote",
            }
        )

    def write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    write_jsonl(current / "norms/humor_multi.jsonl", current_rows)
    write_jsonl(legacy / "norms/humor.jsonl", legacy_rows)
    teacher_path = tmp_path / "teachers.jsonl"
    write_jsonl(teacher_path, teachers)
    current_manifest = current / "manifest.json"
    current_manifest.write_text(
        json.dumps(
            {
                "aliases": {"humor": "humor_multi"},
                "banks": {
                    "humor": {
                        "path": "banks/humor.json",
                        "source_sha256": "bank-hash",
                    }
                },
                "corpora": {
                    "humor_multi": {
                        "task": "humor",
                        "path": "norms/humor_multi.jsonl",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    legacy_manifest = legacy / "manifest.json"
    legacy_manifest.write_text(
        json.dumps(
            {"corpora": {"humor": {"task": "humor", "path": "norms/humor.jsonl"}}}
        ),
        encoding="utf-8",
    )
    universe = load_universe(
        current_manifest,
        [teacher_path],
        "humor",
        split_seed=73129,
        train_percent=80,
        dev_percent=10,
        require_bank_hash=True,
        teacher_manifest_path=legacy_manifest,
    )
    assert len(universe.labels) == 200
    assert universe.teacher_audit["bridge"] == {"unique_exact_quote": 200}
    assert universe.teacher_audit["rejections"]["exact_quote_bridge_ambiguous"] == 1
    assert sum(universe.split_audit["source_groups"].values()) == 100
    assert sum(universe.split_audit["rows"].values()) == 200


def _explicit_split_fixture(tmp_path, splits):
    (tmp_path / "banks").mkdir()
    (tmp_path / "norms").mkdir()
    bank = {
        "task": "humor",
        "source_sha256": "bank-hash",
        "metrics": [
            {"metric_id": "a0", "name": "Timing", "description": "timing", "examples": []}
        ],
    }
    (tmp_path / "banks/humor.json").write_text(json.dumps(bank), encoding="utf-8")
    norms = [
        {
            "norm_uid": f"u{i}",
            "corpus": "humor_multi",
            "task": "humor",
            "source_id": f"source-{i}",
            "norm": f"joke criterion {i}",
        }
        for i in range(len(splits))
    ]
    teachers = [
        {
            "norm_uid": f"u{i}",
            "task": "humor",
            "decision": "MATCH",
            "metric_id": "a0",
            "current_bank_source_sha256": "bank-hash",
            **({"split": split} if split is not None else {}),
        }
        for i, split in enumerate(splits)
    ]
    norms_path = tmp_path / "norms/humor_multi.jsonl"
    teacher_path = tmp_path / "teachers.jsonl"
    norms_path.write_text("".join(json.dumps(row) + "\n" for row in norms), encoding="utf-8")
    teacher_path.write_text(
        "".join(json.dumps(row) + "\n" for row in teachers), encoding="utf-8"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "banks": {
                    "humor": {
                        "path": "banks/humor.json",
                        "source_sha256": "bank-hash",
                    }
                },
                "corpora": {
                    "humor_multi": {
                        "task": "humor",
                        "path": "norms/humor_multi.jsonl",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, teacher_path


def test_explicit_teacher_splits_are_preserved(tmp_path):
    manifest, teachers = _explicit_split_fixture(tmp_path, ["train", "dev", "test"])
    universe = load_universe(
        manifest,
        [teachers],
        "humor",
        split_seed=73129,
        train_percent=80,
        dev_percent=10,
        require_bank_hash=True,
        respect_teacher_splits=True,
    )
    assert [row.split for row in universe.labels] == ["train", "dev", "test"]
    assert universe.teacher_audit["split_mode"] == "explicit_teacher_role"


def test_explicit_teacher_splits_fail_closed_when_missing(tmp_path):
    manifest, teachers = _explicit_split_fixture(tmp_path, ["train", "dev", None])
    with pytest.raises(ValueError, match="missing, conflicting, or invalid explicit split"):
        load_universe(
            manifest,
            [teachers],
            "humor",
            split_seed=73129,
            train_percent=80,
            dev_percent=10,
            require_bank_hash=True,
            respect_teacher_splits=True,
        )
