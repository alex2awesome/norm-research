import json

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.freeze_retrieval_queue import freeze
from scripts.tools.silver_match_v3.run_frozen_retrieval_queue import (
    validate_plan,
)


def test_freezes_three_lane_full_bank_union_without_redundant_projection(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {"metrics": [{"metric_id": f"m{i}"} for i in range(3)]}
        )
    )
    norms = tmp_path / "norms.jsonl"
    write_jsonl(norms, [{"norm_uid": "a" * 64, "row": 0}])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "banks": {
                    "t": {"count": 3, "path": str(bank), "source_sha256": "source-bank"}
                },
                "corpora": {"c": {"task": "t", "count": 1, "path": str(norms)}},
            }
        )
    )
    primary_fusion = tmp_path / "primary.json"
    diverse_fusion = tmp_path / "diverse.json"
    nemotron_fusion = tmp_path / "nemotron-diverse.json"
    components = {
        "dense_rank": 1,
        "dense_statement_rank": 1,
        "word_rank": 1,
        "word_statement_rank": 1,
        "char_rank": 1,
        "char_statement_rank": 1,
    }
    for path in (primary_fusion, diverse_fusion, nemotron_fusion):
        path.write_text(
            json.dumps(
                {
                    "task": "t",
                    "selection_split": "dev",
                    "bank_size": 3,
                    "selected": {"component_weights": components},
                }
            )
        )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "task": "t",
                "selection_split": "external_dev_only",
                "frozen_test_consumed": False,
                "chosen": {
                    "kind": "nemotron_base",
                    "name": "nemotron-base",
                    "fusion_report": str(primary_fusion),
                    "fusion_report_sha256": sha256_file(primary_fusion),
                },
            }
        )
    )
    repo = tmp_path / "repo"
    tools = repo / "scripts/tools/silver_match_v3"
    tools.mkdir(parents=True)
    for name in (
        "retrieve.py",
        "audit_candidate_outputs.py",
        "truncate_candidate_depth.py",
        "materialize_retrieval_lane_union.py",
        "run_frozen_retrieval_queue.py",
    ):
        (tools / name).write_text(name)
    encoders = []
    for revision in ("a" * 40, "b" * 40, "c" * 40):
        encoder = tmp_path / revision
        encoder.mkdir()
        (encoder / "config.json").write_text(json.dumps({"revision": revision}))
        encoders.append(encoder)
    python_target = tmp_path / "python-target"
    python_target.write_text("python")
    python = tmp_path / "python"
    python.symlink_to(python_target)
    existing = tmp_path / "existing.jsonl"
    write_jsonl(existing, [{"norm_uid": "a" * 64, "candidates": []}])
    existing.with_suffix(".jsonl.meta.json").write_text("{}")
    existing_audit = tmp_path / "existing.audit.json"
    existing_audit.write_text("{}")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "task": "t",
                "manifest": str(manifest),
                "selection": str(selection),
                "output_root": str(tmp_path / "production"),
                "repo_root": str(repo),
                "python": str(python),
                "gpu_index": 2,
                "full_k": 3,
                "primary_k": 2,
                "systems": [
                    {
                        "name": "nemotron-base",
                        "role": "primary",
                        "encoder": str(encoders[0]),
                        "query_format": "nemotron",
                        "fusion": str(primary_fusion),
                    },
                    {
                        "name": "bge",
                        "role": "diverse",
                        "encoder": str(encoders[1]),
                        "query_format": "raw",
                        "fusion": str(diverse_fusion),
                    },
                    {
                        "name": "nemotron-diverse",
                        "role": "diverse",
                        "encoder": str(encoders[2]),
                        "query_format": "nemotron",
                        "fusion": str(nemotron_fusion),
                    },
                ],
                "union": {
                    "name": "diverse-fullbank-rrf",
                    "output_k": 3,
                    "rank_constant": 60,
                    "lane_weights": {
                        "nemotron-base": 1,
                        "bge": 1,
                        "nemotron-diverse": 1,
                        "existing-prefix": 1,
                    },
                    "preserve_k": 1,
                    "preserve_components": {
                        "nemotron-base": ["rank"],
                        "bge": ["rank"],
                        "nemotron-diverse": ["rank"],
                        "existing-prefix": ["rank"],
                    },
                },
                "existing_lanes": [
                    {
                        "name": "existing-prefix",
                        "expected_k": 2,
                        "candidates": {"c": str(existing)},
                        "audits": {"c": str(existing_audit)},
                    }
                ],
            }
        )
    )
    plan = freeze(spec)
    assert plan["status"] == "FROZEN_NOT_LAUNCHED"
    assert plan["release_ready"] is False
    assert [step["kind"] for step in plan["steps"]] == [
        "retrieve",
        "audit",
        "retrieve",
        "audit",
        "retrieve",
        "audit",
        "union",
        "audit",
    ]
    validate_plan(plan)
    assert plan["execution"]["python"] == str(python.absolute())
    assert plan["execution"]["gpu_count_gate_applied"] is False
    assert plan["execution"]["projected_owner_count_check_applied"] is False
    assert plan["execution"]["uses_batched_encoder_inference"] is True
    assert plan["execution"]["uses_openai_server"] is False
    assert (
        plan["union"]["algorithm"]
        == "coverage-preserving-component-prefix-rrf-v1"
    )
    assert plan["coverage_contract"] == {
        "scope": "all-manifest-corpora-for-task",
        "corpus_count": 1,
        "norm_count": 1,
        "one_exact_candidate_row_per_norm_required": True,
        "diagnostic_subset_reuse_forbidden": True,
    }
    union_step = next(step for step in plan["steps"] if step["kind"] == "union")
    assert len(union_step["source_candidates"]) == 4
    assert union_step["source_expected_k"][str(existing.resolve())] == 2
