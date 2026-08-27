import json

import pytest

from scripts.tools.silver_match_v3.audit_production_adjudications import audit
from scripts.tools.silver_match_v3.common import sha256_file


def _artifact(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def _fixture(tmp_path):
    files = {}
    for name in (
        "manifest",
        "candidate-meta",
        "retriever",
        "adjudicator-selection",
        "adjudicator-code",
        "adjudicator-prompt",
        "dev-original-meta",
        "dev-hashed-meta",
        "verifier-selection",
        "verifier-code",
        "verifier-prompt",
        "verifier-policy",
        "candidate-audit",
    ):
        path = tmp_path / name
        path.write_text(name)
        files[name] = path
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "norm_uid": "u",
                "corpus": "c",
                "task": "t",
                "bank_source_sha256": "b" * 64,
                "candidates": [{"metric_id": "m1"}, {"metric_id": "m2"}],
            }
        )
        + "\n"
    )
    prompt_sha = "p" * 64
    model = "/model/snapshot"
    rendering = {
        "context_chars": 10,
        "description_chars": 20,
        "example_chars": 5,
        "max_examples": 0,
    }
    plan = {
        "status": "FROZEN_READY_FOR_UNLABELED_PRODUCTION",
        "task": "t",
        "corpora": ["c"],
        "expected_count": 1,
        "bank_source_sha256": "b" * 64,
        "manifest": _artifact(files["manifest"]),
        "candidate_union": _artifact(candidates),
        "candidate_union_meta": _artifact(files["candidate-meta"]),
        "retriever_selection": _artifact(files["retriever"]),
        "candidate_audits": {
            str(files["candidate-audit"]): sha256_file(files["candidate-audit"])
        },
        "adjudicator": {
            "selection": _artifact(files["adjudicator-selection"]),
            "implementation": _artifact(files["adjudicator-code"]),
            "prompt_components": {
                str(files["adjudicator-prompt"]): {
                    "sha256": sha256_file(files["adjudicator-prompt"])
                }
            },
            "selected_dev_run_meta": {
                "original": _artifact(files["dev-original-meta"]),
                "hashed": _artifact(files["dev-hashed-meta"]),
            },
            "prompt_sha256": prompt_sha,
            "model": model,
            "prompt_rendering": rendering,
            "candidate_depth": 2,
        },
        "verifier": {
            "selection": _artifact(files["verifier-selection"]),
            "implementation": _artifact(files["verifier-code"]),
            "production_policy": _artifact(files["verifier-policy"]),
            "prompt_components": {
                str(files["verifier-prompt"]): {
                    "sha256": sha256_file(files["verifier-prompt"])
                }
            },
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    outputs = {}
    for order in ("original", "hashed"):
        path = tmp_path / f"{order}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "norm_uid": "u",
                    "corpus": "c",
                    "task": "t",
                    "decision": "MATCH",
                    "metric_id": "m1",
                    "candidate_ids": ["m1", "m2"],
                    "candidate_bank_source_sha256": "b" * 64,
                    "prompt_sha256": prompt_sha,
                    "model": model,
                    "order_mode": order,
                    "parse_error": None,
                }
            )
            + "\n"
        )
        meta = {
            "input_candidates_sha256": sha256_file(candidates),
            "output_sha256": sha256_file(path),
            "prompt_sha256": prompt_sha,
            "model": model,
            "order_mode": order,
            "max_candidates": 2,
            "prompt_rendering": rendering,
            "invalid_count": 0,
        }
        path.with_suffix(".jsonl.meta.json").write_text(json.dumps(meta))
        outputs[order] = path
    return plan_path, outputs


def test_audit_binds_two_order_outputs_to_frozen_plan(tmp_path):
    plan, outputs = _fixture(tmp_path)
    report = audit(
        plan_path=plan,
        original_path=outputs["original"],
        hashed_path=outputs["hashed"],
        output_path=tmp_path / "audit.json",
    )
    assert report["complete"] is True
    assert report["count"] == 1


def test_audit_rejects_tampered_row(tmp_path):
    plan, outputs = _fixture(tmp_path)
    row = json.loads(outputs["hashed"].read_text())
    row["candidate_bank_source_sha256"] = "wrong"
    outputs["hashed"].write_text(json.dumps(row) + "\n")
    meta_path = outputs["hashed"].with_suffix(".jsonl.meta.json")
    meta = json.loads(meta_path.read_text())
    meta["output_sha256"] = sha256_file(outputs["hashed"])
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="row provenance mismatch"):
        audit(
            plan_path=plan,
            original_path=outputs["original"],
            hashed_path=outputs["hashed"],
            output_path=tmp_path / "audit.json",
        )


def test_audit_routes_parser_failure_to_rescue_instead_of_force_parsing(tmp_path):
    plan, outputs = _fixture(tmp_path)
    row = json.loads(outputs["original"].read_text())
    row.update(
        {
            "decision": "INVALID_OUTPUT",
            "metric_id": None,
            "parse_error": "no_json",
            "raw_response": "truncated",
        }
    )
    outputs["original"].write_text(json.dumps(row) + "\n")
    meta_path = outputs["original"].with_suffix(".jsonl.meta.json")
    meta = json.loads(meta_path.read_text())
    meta["output_sha256"] = sha256_file(outputs["original"])
    meta["invalid_count"] = 1
    meta_path.write_text(json.dumps(meta))
    report = audit(
        plan_path=plan,
        original_path=outputs["original"],
        hashed_path=outputs["hashed"],
        output_path=tmp_path / "audit.json",
    )
    assert report["orders"]["original"]["invalid_count"] == 1
