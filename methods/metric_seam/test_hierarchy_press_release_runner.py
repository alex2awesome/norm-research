from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from methods.metric_seam.hierarchy_press_release_runner import (
    CELL_RELATIONS,
    CELL_RELATION_FILTERS,
    DEFAULT_PANEL,
    DIRECT_RELATION_PAIRS,
    FREEZE_SCHEMA,
    NO_CANDIDATE_REASONS,
    PressReleaseHierarchyError,
    RELATION_SEEDS,
    SOURCE_MAP_SCHEMA,
    build_freeze_receipt,
    build_source_map,
    execute_split,
    main,
    validate_items,
    validate_manifest,
    verify_freeze,
)
from methods.metric_seam.press_release_relations_v1 import RELATION_SPECS


@pytest.fixture(scope="module")
def panel():
    return json.loads(DEFAULT_PANEL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source_map(panel):
    return build_source_map(panel, panel_path=DEFAULT_PANEL)


def _items(prefix="train"):
    return [
        {
            "item_key": f"{prefix}_0001",
            "ctext": (
                "ACME will host a summit in Boston on May 2, 2026. "
                "Register at https://example.org/register."
            ),
        }
    ]


def _manifest(task="press-releases"):
    return {
        "schema": "metric-seam.hierarchy-shared-items.v1",
        "task": task,
        "representation": {
            "field": "ctext",
            "max_chars": 4000,
            "same_bytes_required_for_prompt_and_code": True,
        },
        "selection": {"train_n": 1, "heldout_n": 1},
        "policy": {
            "outcome_columns_emitted": False,
            "external_supervision_used": False,
        },
    }


def test_source_map_exhaustively_adjudicates_all_90_cells(source_map):
    assert source_map["schema"] == SOURCE_MAP_SCHEMA
    assert len(CELL_RELATIONS) == 90
    assert len(source_map["rows"]) == 90
    assert source_map["summary"]["cells_by_level"] == {
        "R1": 30,
        "R2": 30,
        "R3": 30,
    }
    assert source_map["summary"]["decision_counts"] == {
        "bounded_non_discovery_in_frozen_program_class": 30,
        "relation_local_candidate": 60,
    }
    assert source_map["summary"]["candidate_applications"] == 149
    assert source_map["summary"]["unique_relation_programs"] == 17
    assert source_map["summary"]["direct_named_subrelation_applications"] == 30
    assert source_map["summary"]["whole_construct_matches_established"] == 0


def test_requested_vs_implemented_scope_is_explicit_per_cell(source_map):
    for row in source_map["rows"]:
        assert row["requested_construct"]
        assert row["requested_description"]
        assert len(row["requested_source_text_sha256"]) == 64
        assert row["whole_construct_match_established"] is False
        assert row["unimplemented_scope"]
        if row["implemented_candidates"]:
            assert row["bounded_non_discovery_reason"] is None
            for candidate in row["implemented_candidates"]:
                assert candidate["relation_id"] in RELATION_SPECS
                assert candidate["subrelation_match"] in {
                    "direct_named_subrelation",
                    "adjacent_partial_subrelation",
                }
                assert candidate["implemented_relation"]
                assert candidate["does_not_establish"]
                assert candidate["cell_local_applicability_filter"]
                assert candidate["program_relation_depth_ceiling"] in {2, 3}
                assert candidate["matched_relation_depth"] is None
                assert (
                    candidate["matched_depth_status"]
                    == "requires_item_local_runtime_witness"
                )
        else:
            assert row["decision"] == "bounded_non_discovery_in_frozen_program_class"
            assert row["bounded_non_discovery_reason"]


def test_mapping_and_non_discovery_ledgers_are_exhaustive():
    empty = {key for key, relations in CELL_RELATIONS.items() if not relations}
    assert empty == set(NO_CANDIDATE_REASONS)
    assert set(RELATION_SEEDS) == set(RELATION_SPECS)
    assert all(pair[1] in CELL_RELATIONS[pair[0]] for pair in DIRECT_RELATION_PAIRS)
    assert all(
        pair[0] in CELL_RELATIONS and pair[1] in CELL_RELATIONS[pair[0]]
        for pair in CELL_RELATION_FILTERS
    )


def test_historical_programs_are_source_inspected_not_executed(source_map):
    seeds = source_map["historical_seed_source_inspection"]
    assert len(seeds) == 21
    for seed in seeds.values():
        assert seed["sha256"]
        assert seed["ast_node_count"] > 0
        assert seed["imported_or_executed"] is False
    assert source_map["design_scope"]["items_loaded"] is False
    assert source_map["design_scope"]["programs_imported_or_executed"] is False


def test_item_and_manifest_contracts_reject_extra_or_supervised_fields():
    items = _items()
    validate_items(items, phase="compiler_train")
    manifest = _manifest()
    assert validate_manifest(
        manifest,
        items,
        task="press-releases",
        phase="compiler_train",
    ) == 4000

    extra = deepcopy(items)
    extra[0]["score"] = 1
    with pytest.raises(PressReleaseHierarchyError, match="exactly item_key and ctext"):
        validate_items(extra, phase="compiler_train")
    supervised = deepcopy(manifest)
    supervised["policy"]["external_supervision_used"] = True
    with pytest.raises(PressReleaseHierarchyError, match="outcome-blind"):
        validate_manifest(
            supervised,
            items,
            task="press-releases",
            phase="compiler_train",
        )


def test_split_execution_records_exact_ctext_and_no_criterion_scores():
    items = _items()
    result = execute_split(
        items,
        task="press-releases",
        phase="compiler_train",
        max_chars=4000,
        source_map_sha256="a" * 64,
    )
    row = result["rows"][0]
    assert row["ctext_sha256"] == hashlib.sha256(
        items[0]["ctext"].encode("utf-8")
    ).hexdigest()
    assert row["absence_inference_permitted"] is False
    assert result["design"]["outcomes_or_references_loaded"] is False
    assert result["design"]["api_or_llm_calls_used"] is False
    assert result["design"]["local_cpu_statistical_parser_used"] is True
    assert result["design"]["gpu_used"] is False
    assert result["design"]["criterion_scalar_scores_emitted"] is False
    assert result["design"]["whole_criterion_verdicts_emitted"] is False
    assert "score" not in row
    assert all("score" not in relation for relation in row["result"]["relations"].values())


def test_secondary_news_family_reuses_frozen_relations_without_cell_claims():
    result = execute_split(
        _items(),
        task="news-homepages",
        phase="compiler_train",
        max_chars=4000,
        source_map_sha256="a" * 64,
        freeze_sha256="b" * 64,
    )
    assert result["source_family_role"] == "secondary_news_homepage_source_family_check"
    assert result["bindings"]["train_freeze_sha256"] == "b" * 64
    assert "cells" not in result["summary"]


def test_freeze_binds_implementation_source_map_train_output_and_panel(
    tmp_path, source_map
):
    source_map_path = tmp_path / "source_map.json"
    source_map_path.write_text(json.dumps(source_map), encoding="utf-8")
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps({"phase": "compiler_train"}), encoding="utf-8")
    freeze_path = tmp_path / "freeze.json"
    freeze = build_freeze_receipt(
        source_map_path=source_map_path,
        train_output_path=train_path,
        panel_path=DEFAULT_PANEL,
    )
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    assert freeze["schema"] == FREEZE_SCHEMA
    freeze_sha = verify_freeze(
        freeze,
        freeze_path=freeze_path,
        source_map_path=source_map_path,
        panel_path=DEFAULT_PANEL,
    )
    assert len(freeze_sha) == 64

    source_map_path.write_text(json.dumps({"mutated": True}), encoding="utf-8")
    with pytest.raises(PressReleaseHierarchyError, match="source map drifted"):
        verify_freeze(
            freeze,
            freeze_path=freeze_path,
            source_map_path=source_map_path,
            panel_path=DEFAULT_PANEL,
        )


def test_invalid_freeze_fails_before_heldout_file_is_opened(tmp_path):
    freeze_path = tmp_path / "invalid_freeze.json"
    freeze_path.write_text("{}", encoding="utf-8")
    missing_items = tmp_path / "SEALED_ITEMS_MUST_NOT_OPEN.json"
    with pytest.raises(PressReleaseHierarchyError, match="unexpected train freeze"):
        main(
            [
                "--task",
                "press-releases",
                "--phase",
                "heldout_pre_reference",
                "--items",
                str(missing_items),
                "--manifest",
                str(tmp_path / "missing_manifest.json"),
                "--source-map",
                str(tmp_path / "missing_source_map.json"),
                "--output",
                str(tmp_path / "must_not_write.json"),
                "--freeze",
                str(freeze_path),
            ]
        )
    assert not missing_items.exists()
    assert not (tmp_path / "must_not_write.json").exists()
