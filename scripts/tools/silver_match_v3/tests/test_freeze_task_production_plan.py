import json

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_task_production_plan import freeze_plan


def _dump(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    implementation_dir = repo / "scripts" / "tools" / "silver_match_v3"
    implementation_dir.mkdir(parents=True)
    (implementation_dir / "adjudicate_gemma.py").write_text(
        "# adjudicator\n", encoding="utf-8"
    )
    (implementation_dir / "verify_gemma.py").write_text(
        "# verifier\n", encoding="utf-8"
    )
    prompt = repo / "prompt.txt"
    verifier_prompt = repo / "verifier.txt"
    prompt.write_text("adjudicate", encoding="utf-8")
    verifier_prompt.write_text("verify", encoding="utf-8")

    source_a = tmp_path / "a.jsonl"
    source_b = tmp_path / "b.jsonl"
    source_a.write_text('{"norm_uid":"a"}\n', encoding="utf-8")
    source_b.write_text('{"norm_uid":"b"}\n', encoding="utf-8")
    candidates = tmp_path / "all.jsonl"
    candidates.write_text(source_a.read_text() + source_b.read_text(), encoding="utf-8")

    manifest = _dump(
        tmp_path / "manifest.json",
        {
            "banks": {"task": {"source_sha256": "b" * 64}},
            "corpora": {
                "a": {"task": "task", "count": 1},
                "b": {"task": "task", "count": 1},
            },
        },
    )
    manifest_sha = sha256_file(manifest)
    audits = []
    for corpus, source in (("a", source_a), ("b", source_b)):
        audits.append(
            _dump(
                tmp_path / f"{corpus}.audit.json",
                {
                    "complete": True,
                    "task": "task",
                    "corpus": corpus,
                    "manifest_sha256": manifest_sha,
                    "bank_source_sha256": "b" * 64,
                    "observed_count": 1,
                    "candidate_inputs": {
                        str(source): {"sha256": sha256_file(source)},
                    },
                },
            )
        )
    _dump(
        candidates.with_suffix(".jsonl.meta.json"),
        {
            "count": 2,
            "sha256": sha256_file(candidates),
            "inputs": {
                str(source_a): {"sha256": sha256_file(source_a)},
                str(source_b): {"sha256": sha256_file(source_b)},
            },
        },
    )

    retriever = _dump(
        tmp_path / "retriever.json",
        {"task": "task", "selection_split": "dev"},
    )
    adjudicator = _dump(
        tmp_path / "adjudicator.json",
        {
            "task": "task",
            "selection_split": "dev",
            "candidate_depth": 50,
            "chosen": {
                "prompt": "prompt.txt",
                "prompt_sha256": "a" * 64,
                "prompt_component_sha256": {"prompt.txt": sha256_file(prompt)},
                "inputs": {},
            },
        },
    )
    adjudicator_payload = json.loads(adjudicator.read_text())
    for order in ("original", "hashed"):
        output = tmp_path / f"adjudicator.{order}.jsonl"
        output.write_text(json.dumps({"decision": "MATCH"}) + "\n")
        output_sha = sha256_file(output)
        adjudicator_payload["chosen"]["inputs"][order] = {
            "path": str(output),
            "sha256": output_sha,
        }
        _dump(
            output.with_suffix(".jsonl.meta.json"),
            {
                "order_mode": order,
                "output_sha256": output_sha,
                "prompt_sha256": "a" * 64,
                "max_candidates": 50,
                "prompt_component_sha256": {"prompt.txt": sha256_file(prompt)},
                "model": "/models/gemma-snapshot",
                "prompt_rendering": {
                    "context_chars": 1200,
                    "description_chars": 260,
                    "example_chars": 80,
                    "max_examples": 0,
                },
            },
        )
    _dump(adjudicator, adjudicator_payload)
    verifier = _dump(
        tmp_path / "verifier.json",
        {
            "task": "task",
            "selection_split": "external_dev_only",
            "calibration_power_status": "supported",
            "chosen": {
                "statistically_supported": True,
                "prompt": "verifier.txt",
                "prompt_sha256": "c" * 64,
                "prompt_component_sha256": {
                    "verifier.txt": sha256_file(verifier_prompt)
                },
            },
        },
    )
    policy = _dump(
        tmp_path / "policy.json",
        {
            "task": "task",
            "selection_split": "dev",
            "inputs": {
                "selection": {"sha256": sha256_file(verifier)},
            },
            "dev_gate": {"cleared": True},
            "may_run_on_production_unlabeled_norms": True,
            "prompt": {"rendered_prompt_sha256": "c" * 64},
            "rendering": {"seed": 1},
            "order_policy": {"retain_only_if": "strict consensus"},
        },
    )
    return {
        "manifest_path": manifest,
        "task": "task",
        "candidate_path": candidates,
        "candidate_audit_paths": audits,
        "retriever_selection_path": retriever,
        "adjudicator_selection_path": adjudicator,
        "verifier_selection_path": verifier,
        "verifier_policy_path": policy,
        "repo_root": repo,
    }, prompt


def test_freeze_plan_requires_complete_hash_linked_task_chain(tmp_path):
    kwargs, _ = _fixture(tmp_path)
    plan = freeze_plan(**kwargs)
    assert plan["status"] == "FROZEN_READY_FOR_UNLABELED_PRODUCTION"
    assert plan["corpora"] == ["a", "b"]
    assert plan["expected_count"] == 2
    assert plan["adjudicator"]["candidate_depth"] == 50
    assert plan["verifier"]["blind_final_match_audit_required"] is True


def test_freeze_plan_rejects_changed_prompt_component(tmp_path):
    kwargs, prompt = _fixture(tmp_path)
    prompt.write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt component changed"):
        freeze_plan(**kwargs)
