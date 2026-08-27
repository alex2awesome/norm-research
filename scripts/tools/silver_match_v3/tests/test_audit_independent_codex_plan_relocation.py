import argparse
import json

from scripts.tools.silver_match_v3.audit_independent_codex_plan_relocation import audit


def _pack(root, seed):
    return {
        "root": str(root),
        "task": "demo",
        "count": 2,
        "seed": seed,
        "source_pack": {"validation_sha256": "source"},
        "validation": {"path": str(root / "validation.json"), "sha256": f"v{seed}"},
        "items": {"path": str(root / "items.jsonl"), "sha256": f"i{seed}"},
        "bank": {"path": str(root / "bank.json"), "sha256": f"b{seed}"},
        "chunks": [
            {"path": str(root / "chunks/part-000.jsonl"), "sha256": f"c{seed}"}
        ],
    }


def _plan(tmp_path, name, *, local, audit_path):
    workspace = tmp_path / name / "workspace"
    (workspace / "pack").mkdir(parents=True)
    python = "/local/python" if local else "/remote/python"
    impl = {
        "runner": {"path": f"/{name}/runner.py", "sha256": "runner"},
        "labeling_guide": {"path": f"/{name}/guide.md", "sha256": "guide"},
        "isolation_guide": {"path": f"/{name}/isolation.md", "sha256": "isolation"},
        "output_schema": {"path": f"/{name}/schema.json", "sha256": "schema"},
    }
    if local:
        impl["boundary_guides"] = [impl["isolation_guide"]]
    source_a, source_b = _pack(tmp_path / "source-a", 1), _pack(tmp_path / "source-b", 2)
    staged_a, staged_b = _pack(tmp_path / "staged-a", 1), _pack(tmp_path / "staged-b", 2)
    command = lambda suffix, staged: {
        "cwd": str(workspace),
        "environment": {"PYTHONPATH": f"/{name}/implementation"},
        "argv": [
            python,
            "-u",
            "-m",
            "scripts.tools.silver_match_v3.run_codex_pack_labels",
            "--pack-root",
            str(workspace / "pack"),
            "--task",
            "demo",
            "--pass-name",
            f"demo-pass-{suffix}",
            "--model",
            "model",
        ],
    }
    return {
        "schema_version": "silver-match-v3-independent-codex-label-execution-plan-v1",
        "status": "FROZEN_BEFORE_EITHER_INDEPENDENT_LABEL_PASS",
        "task": "demo",
        "row_count": 2,
        "pass_count": 2,
        "runtime": {
            "model": "model",
            "reasoning_effort": "high",
            "concurrency_per_pass": 2,
            "timeout_seconds": 900,
            "chunk_attempts": 1,
            "python": {"path": python, "sha256": "local-python" if local else "remote-python"},
        },
        "implementation": impl,
        "inputs": {
            "prelabel_independence_audit": {"path": str(audit_path), "sha256": "audit"},
            "source_pass_a": source_a,
            "source_pass_b": source_b,
            "staged_pass_a": staged_a,
            "staged_pass_b": staged_b,
            **({"external_policy": None} if local else {}),
        },
        "commands": {"A": command("a", staged_a), "B": command("b", staged_b)},
        "contracts": {"full_bank_required_for_every_item": True},
    }


def _prelabel(path, root):
    path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-independent-pack-view-audit-v1",
                "status": "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING",
                "task": "demo",
                "count": 2,
                "bank_metric_count": 3,
                "same_uid_set": True,
                "same_canonical_item_content_by_uid": True,
                "same_bank_leaf_set": True,
                "distinct_seeds": True,
                "distinct_item_order": True,
                "distinct_bank_order": True,
                "candidate_proposals_exposed_to_either_pass": False,
                "prior_truth_or_predictions_exposed_to_either_pass": False,
                "pass_predictions_mutually_visible": False,
                "passes": {
                    "A": {
                        "root": f"{root}/a",
                        "seed": 1,
                        "validation_sha256": "v1",
                        "items_sha256": "i1",
                        "bank_sha256": "b1",
                    },
                    "B": {
                        "root": f"{root}/b",
                        "seed": 2,
                        "validation_sha256": "v2",
                        "items_sha256": "i2",
                        "bank_sha256": "b2",
                    },
                },
                "usage_contract": {"run_passes_in_separate_processes": True},
            }
        )
    )


def test_freezes_host_only_relocation_with_backward_compatible_plan_fields(tmp_path):
    source_audit, target_audit = tmp_path / "source-audit.json", tmp_path / "target-audit.json"
    _prelabel(source_audit, "/remote")
    _prelabel(target_audit, "/local")
    source_plan = _plan(tmp_path, "remote", local=False, audit_path=source_audit)
    target_plan = _plan(tmp_path, "local", local=True, audit_path=target_audit)
    source_path, target_path = tmp_path / "source-plan.json", tmp_path / "target-plan.json"
    source_path.write_text(json.dumps(source_plan))
    target_path.write_text(json.dumps(target_plan))
    codex, auth = tmp_path / "codex", tmp_path / "auth.json"
    codex.write_text("#!/bin/sh\necho 'Logged in using ChatGPT'\n")
    codex.chmod(0o700)
    auth.write_text("{}")
    output = tmp_path / "relocation.json"
    result = audit(
        argparse.Namespace(
            source_plan=str(source_path),
            relocated_plan=str(target_path),
            source_prelabel_audit=str(source_audit),
            codex_bin=str(codex),
            auth_file=str(auth),
            output=str(output),
        )
    )
    assert result["status"] == "FROZEN_APPEND_ONLY_BYTE_IDENTICAL_EXECUTION_RELOCATION"
    assert all(result["equalities"].values())
