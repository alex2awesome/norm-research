import json
from pathlib import Path

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_task_retrieval_queue_template import (
    IMPLEMENTATIONS,
    freeze_template,
)


def test_freezes_non_executable_exact_task_inventory(tmp_path):
    repo = tmp_path / "repo"
    implementation_root = repo / "scripts/tools/silver_match_v3"
    implementation_root.mkdir(parents=True)
    for name in IMPLEMENTATIONS:
        (implementation_root / name).write_text(name)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-alltask-release-coverage-audit-v1",
                "manifest": {"path": "/remote/manifest.json", "sha256": "a" * 64},
                "extractions": {
                    "complete": True,
                    "corpora": {
                        "c2": {
                            "task": "legal",
                            "count": 3,
                            "path": "/remote/c2.jsonl",
                            "sha256": "2" * 64,
                        },
                        "c1": {
                            "task": "legal",
                            "count": 2,
                            "path": "/remote/c1.jsonl",
                            "sha256": "1" * 64,
                        },
                        "other": {
                            "task": "other",
                            "count": 9,
                            "path": "/remote/other.jsonl",
                            "sha256": "3" * 64,
                        },
                    },
                },
                "candidate_retrieval": {"missing_corpora": ["c1", "c2"]},
                "retriever_selections": {
                    "tasks": {
                        "legal": {
                            "chosen_kind": "nemotron_base",
                            "chosen_name": "nemotron-base",
                            "path": "/remote/selection.json",
                            "sha256": "4" * 64,
                            "fusion_path": "/remote/fusion.json",
                            "fusion_sha256": "5" * 64,
                        }
                    }
                },
            }
        )
    )
    frozen = freeze_template(
        coverage_path=coverage,
        task="legal",
        bank_count=104,
        bank_source_sha256="b" * 64,
        output_root="/remote/production/legal",
        repo_root=repo,
    )
    assert frozen["status"] == "FROZEN_TEMPLATE_NOT_EXECUTABLE"
    assert frozen["release_ready"] is False
    assert list(frozen["corpora"]) == ["c1", "c2"]
    assert frozen["corpus_count"] == 2
    assert frozen["norm_count"] == 5
    assert frozen["queue_spec_template"]["full_k"] == 104
    assert frozen["queue_spec_template"]["selection"] == "/remote/selection.json"
    assert frozen["queue_spec_template"]["systems"][0]["fusion"] == "/remote/fusion.json"
    assert frozen["selected_retriever"]["sha256"] == "4" * 64
    assert frozen["expected_materialization"]["one_exact_row_per_norm_required"] is True
    assert frozen["inference_contract"]["openai_compatible_server"] is False


def test_frozen_legal_template_matches_v2_authoritative_inventory():
    repo = Path(__file__).resolve().parents[4]
    template_path = (
        repo
        / "outputs/silver_match_v3/legal-outcome-prediction/retrieval_queue_v1"
        / "LEGAL_TEN_CORPUS_QUEUE_TEMPLATE.json"
    )
    template = json.loads(template_path.read_text())
    coverage_path = repo / "outputs/silver_match_v3/alltask_coverage_20260713_v2.json"
    coverage = json.loads(coverage_path.read_text())
    legal = {
        name: meta
        for name, meta in coverage["extractions"]["corpora"].items()
        if meta["task"] == "legal-outcome-prediction"
    }
    assert template["status"] == "FROZEN_TEMPLATE_NOT_EXECUTABLE"
    assert template["release_ready"] is False
    assert template["coverage_audit"]["sha256"] == sha256_file(coverage_path)
    assert template["corpus_count"] == len(legal) == 10
    assert template["norm_count"] == sum(meta["count"] for meta in legal.values()) == 326522
    assert {
        name: {key: meta[key] for key in ("task", "count", "path", "sha256")}
        for name, meta in legal.items()
    } == template["corpora"]
    selected = coverage["retriever_selections"]["tasks"]["legal-outcome-prediction"]
    assert template["selected_retriever"] == selected
    for name, identity in template["implementations"].items():
        assert identity["sha256"] == sha256_file(
            repo / "scripts/tools/silver_match_v3" / name
        )
