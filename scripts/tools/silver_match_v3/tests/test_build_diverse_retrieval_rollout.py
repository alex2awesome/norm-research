import json
from pathlib import Path

from scripts.tools.silver_match_v3.build_diverse_retrieval_rollout import (
    TASK_ORDER,
    build,
)


def _inputs(repo: Path):
    return {
        "coverage_path": repo / "outputs/silver_match_v3/alltask_coverage_20260713_v2.json",
        "bank_manifest_path": repo
        / "outputs/silver_match_v3/humor/remediation_v3/production_adjudication_v1/input_evidence/manifest.sk2_ce_v1.json",
        "humor_selection_path": repo
        / "outputs/silver_match_v3/humor/remediation_v3/production_adjudication_v1/input_evidence/HUMOR_NEMOTRON_SELECTED_EXTERNAL_DEV_V2.json",
        "humor_capture_path": repo
        / "outputs/silver_match_v3/humor/remediation_v3/model_improvement_v2/capture_recapture_v1/capture_union.k50.v2.json",
    }


def test_diverse_rollout_covers_all_tasks_and_exact_humor_corpus():
    repo = Path(__file__).resolve().parents[4]
    inventory, humor = build(**_inputs(repo))
    assert inventory["task_order"] == list(TASK_ORDER)
    assert inventory["task_count"] == 8
    assert inventory["corpus_count"] == 23
    assert inventory["norm_count"] == 1_732_515
    assert inventory["bank_leaf_count"] == 1_431
    assert inventory["tasks"][-1]["task"] == "notice-and-comment"
    assert all(
        row["required_materialization"]["ce_union"][
            "generated_full_bank_lane_count"
        ]
        == 2
        for row in inventory["tasks"]
    )

    assert humor["norm_count"] == 77_378
    assert humor["bank"]["count"] == 285
    assert humor["selected_retriever"]["name"] == "nemotron-humor-sonnet-clean-v2"
    assert humor["queue_spec_template"]["union"]["output_k"] == 50
    assert len(humor["queue_spec_template"]["systems"]) == 2
    assert humor["expected_materialization"][
        "minimum_complete_bank_lane_identities_in_union_meta"
    ] == 2
    assert humor["corpora"]["humor_multi"]["count"] == 77_378
    assert humor["expected_materialization"]["union_rows"] == 77_378
    assert "diagnostic-only" in humor["diagnostic_capture_evidence"]["scope"]


def test_static_rollout_artifacts_match_builder():
    repo = Path(__file__).resolve().parents[4]
    expected_inventory, expected_humor = build(**_inputs(repo))
    inventory = json.loads(
        (repo / "outputs/silver_match_v3/ALLTASK_DIVERSE_RETRIEVAL_ROLLOUT_V1.json").read_text()
    )
    humor = json.loads(
        (
            repo
            / "outputs/silver_match_v3/humor/retrieval_queue_v2/HUMOR_FULL_CORPUS_DIVERSE_RETRIEVAL_TEMPLATE_V1.json"
        ).read_text()
    )
    assert inventory == expected_inventory
    assert humor == expected_humor
