import json
import hashlib
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import write_jsonl
from scripts.tools.silver_match_v3.freeze_task_gepa_api_plan import freeze
from scripts.tools.silver_match_v3.make_calibration import split_for, split_group_for


def _fixture(tmp_path: Path) -> Namespace:
    task = "code-review"
    norms = []
    i = 0
    while len(norms) < 6:
        source_id = f"source-{i}"
        i += 1
        row = {
            "norm_uid": f"u-{i}",
            "task": task,
            "corpus": "c",
            "source_id": source_id,
            "row": i,
            "norm": f"criterion {i}",
        }
        if split_for(split_group_for(row)) == "train":
            norms.append(row)
    norms_path = tmp_path / "norms.jsonl"
    write_jsonl(norms_path, norms)

    bank_path = tmp_path / "bank.json"
    bank_path.write_text(
        json.dumps(
            {
                "metrics": [
                    {"metric_id": "a0", "name": "zero", "description": "zero"},
                    {"metric_id": "a1", "name": "one", "description": "one"},
                    {"metric_id": "a2", "name": "two", "description": "two"},
                ]
            }
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "corpora": {"c": {"task": task, "path": str(norms_path)}},
                "banks": {task: {"path": str(bank_path)}},
            }
        )
    )

    panel = []
    for index, row in enumerate(norms[:4]):
        panel.append(
            {
                "norm_uid": row["norm_uid"],
                "task": task,
                "decision": "MATCH" if index != 3 else "NO_CANDIDATE_FITS",
                "metric_id": "a0" if index != 3 else None,
                "confidence": "high",
                "predeclared_split": "train",
                "split": "train" if index < 2 else "dev",
                "gepa_split_seed": 7,
                "gepa_dev_percent": 25,
            }
        )
    panel_path = tmp_path / "panel.jsonl"
    write_jsonl(panel_path, panel)
    candidate_path = tmp_path / "candidates.jsonl"
    write_jsonl(
        candidate_path,
        (
            {
                "norm_uid": row["norm_uid"],
                "task": task,
                "bank_source_sha256": "bank-hash",
                "candidates": [
                    {"metric_id": "a0", "rank": 1},
                    {"metric_id": "a1", "rank": 2},
                    {"metric_id": "a2", "rank": 3},
                ],
            }
            for row in norms[:4]
        ),
    )
    exclusion_path = tmp_path / "exclude.jsonl"
    write_jsonl(exclusion_path, [{"norm_uid": norms[4]["norm_uid"], "task": task}])
    adjudicator_prompt = tmp_path / "adjudicator.txt"
    adjudicator_prompt.write_text("adjudicator\n")
    verifier_prompt = tmp_path / "verifier.txt"
    verifier_prompt.write_text("verifier\n")
    adjudicator_hash = hashlib.sha256("adjudicator\n".encode()).hexdigest()
    verifier_hash = hashlib.sha256("verifier\n".encode()).hexdigest()
    predeclaration_path = tmp_path / "predeclaration.json"
    predeclaration_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-task-local-gepa-predeclaration-v1",
                "status": "PREDECLARED_PENDING_CANONICAL_PACKS_AND_COMPLETE_EXCLUSIONS",
                "candidate_k": 2,
                "split": {
                    "gepa_seed": 7,
                    "gepa_dev_percent": 25,
                    "minimum_prompt_train_rows": 2,
                    "minimum_prompt_dev_rows": 2,
                },
                "selection_gate": {
                    "minimum_point_precision": 0.9,
                    "minimum_wilson_95_lower": 0.8,
                    "minimum_retained_support": 2,
                },
                "api": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "model": "google/gemma-4-31b-it",
                    "maximum_total_logical_requests_per_task": 1_000,
                    "implicit_transport_retries": 0,
                },
                "direct_batch": {
                    "model": "local-gemma",
                    "batch_size": 16,
                    "gpu_memory_utilization": 0.8,
                },
                "tasks": {
                    task: {
                        "adjudicator_variants": [
                            {"name": "r0", "combined_prompt_sha256": adjudicator_hash}
                        ],
                        "verifier_variants": [
                            {"name": "r0", "combined_prompt_sha256": verifier_hash}
                        ],
                    }
                },
            }
        )
    )
    return Namespace(
        task=task,
        predeclaration=str(predeclaration_path),
        manifest=str(manifest_path),
        panel=str(panel_path),
        candidates=str(candidate_path),
        exclude_reference=[str(exclusion_path)],
        adjudicator_variant=[f"r0={adjudicator_prompt}"],
        verifier_variant=[f"r0={verifier_prompt}"],
        output_root=str(tmp_path / "freeze"),
        candidate_k=2,
        minimum_train=2,
        minimum_dev=2,
        minimum_point_precision=0.9,
        minimum_wilson_lower=0.8,
        minimum_retained=2,
        max_total_api_requests=100,
        api_base_url="https://openrouter.ai/api/v1",
        api_key_file="~/.openrouter-api-key.txt",
        model="google/gemma-4-31b-it",
        concurrency=2,
        direct_model="local-gemma",
        direct_batch_size=16,
        gpu_memory_utilization=0.8,
    )


def test_freeze_task_gepa_plan_is_train_only_excluded_and_bounded(tmp_path):
    args = _fixture(tmp_path)
    result = freeze(args)
    root = Path(args.output_root)
    plan = json.loads((root / "COMMAND_PLAN.json").read_text())
    assert result["status"] == "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE"
    assert result["maximum_total_api_requests"] == 40
    assert plan["exclusions"]["selected_source_group_overlap"] == 0
    assert plan["roles"]["train"]["count"] == 2
    assert plan["roles"]["dev"]["count"] == 2
    assert not plan["scientific_scope"]["test_or_blind_audit_consumed"]
    assert not plan["scientific_scope"]["production_consumed"]
    inference = [row for row in plan["commands"] if row["stage"] in {"adjudicator", "verifier"}]
    assert inference
    assert all(row["maximum_api_requests"] == 4 for row in inference)
    assert all("--max-api-requests" in row["command"]["argv"] for row in inference)
    assert all(
        row["command"]["argv"][row["command"]["argv"].index("--transport-retries") + 1]
        == "0"
        for row in inference
    )
    assert all("google/gemma-4-31b-it" in row["command"]["argv"] for row in inference)
    assert all("direct_batch_command" in row for row in inference)
    assert {
        row["direct_batch_command"]["module"] for row in inference
    } == {
        "scripts.tools.silver_match_v3.adjudicate_gemma",
        "scripts.tools.silver_match_v3.verify_gemma",
    }


def test_freeze_task_gepa_plan_rejects_excluded_group(tmp_path):
    args = _fixture(tmp_path)
    panel = list(
        json.loads(line)
        for line in Path(args.panel).read_text().splitlines()
        if line.strip()
    )
    excluded = json.loads(Path(args.exclude_reference[0]).read_text().splitlines()[0])
    excluded["norm_uid"] = panel[0]["norm_uid"]
    write_jsonl(Path(args.exclude_reference[0]), [excluded])
    with pytest.raises(ValueError, match="excluded source group"):
        freeze(args)


def test_freeze_task_gepa_plan_fails_closed_on_api_budget(tmp_path):
    args = _fixture(tmp_path)
    args.max_total_api_requests = 39
    with pytest.raises(ValueError, match="exceeds budget"):
        freeze(args)


def test_freeze_task_gepa_plan_counts_verifier_cross_product(tmp_path):
    args = _fixture(tmp_path)
    prompt = Path(args.adjudicator_variant[0].split("=", 1)[1])
    args.adjudicator_variant.append(f"r1={prompt}")
    lock = json.loads(Path(args.predeclaration).read_text())
    lock["tasks"][args.task]["adjudicator_variants"].append(
        {
            "name": "r1",
            "combined_prompt_sha256": hashlib.sha256(
                prompt.read_text().rstrip().encode() + b"\n"
            ).hexdigest(),
        }
    )
    Path(args.predeclaration).write_text(json.dumps(lock))
    args.max_total_api_requests = 1_000
    result = freeze(args)
    # Per adjudicator variant: 16 adjudicator + 24 verifier requests.
    assert result["maximum_total_api_requests"] == 80
