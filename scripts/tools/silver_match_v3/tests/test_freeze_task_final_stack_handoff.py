import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file
from scripts.tools.silver_match_v3.freeze_task_final_stack_handoff import freeze
from scripts.tools.silver_match_v3.train_nemotron_lora import source_group_key


def _json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _truth_report(
    path: Path,
    *,
    task: str,
    bank_hash: str,
    source_name: str,
    source_kind: str,
    truth_path: Path,
) -> Path:
    return _json(
        path,
        {
            "schema_version": "silver-match-v3-task-truth-source-report-v1",
            "status": "FROZEN_TRUSTED_TRUTH_SOURCE",
            "task": task,
            "source_name": source_name,
            "source_kind": source_kind,
            "bank_source_sha256": bank_hash,
            "output": {
                "path": str(truth_path),
                "sha256": sha256_file(truth_path),
                "count": sum(1 for _ in read_jsonl(truth_path)),
            },
        },
    )


def _fixture(tmp_path: Path, *, task: str, corpus_names: list[str]):
    bank_hash = hashlib.sha256(f"{task}:bank".encode()).hexdigest()
    bank = _json(
        tmp_path / "bank.json",
        {
            "task": task,
            "source_sha256": bank_hash,
            "metrics": [
                {
                    "task": task,
                    "metric_id": f"m{i}",
                    "name": f"{task} metric {i}",
                    "description": f"definition {i}",
                    "examples": [],
                }
                for i in range(3)
            ],
        },
    )
    per_corpus = 2 if len(corpus_names) == 2 else 1
    norms = []
    corpus_paths = {}
    corpus_meta = {}
    for corpus in corpus_names:
        rows = []
        for index in range(per_corpus):
            uid = hashlib.sha256(f"{task}:{corpus}:{index}".encode()).hexdigest()
            row = {
                "task": task,
                "corpus": corpus,
                "norm_uid": uid,
                "source_id": f"source-{corpus}-{index}",
                "row": index,
                "norm": f"human criterion {task} {corpus} {index}",
                "context": f"evidence {corpus} {index}",
            }
            rows.append(row)
            norms.append(row)
        path = _jsonl(tmp_path / "corpora" / f"{corpus}.jsonl", rows)
        corpus_paths[corpus] = path
        corpus_meta[corpus] = {
            "task": task,
            "path": str(path),
            "count": len(rows),
            "coverage_complete": True,
            "missing_optional_segments": [],
        }
    manifest = _json(
        tmp_path / "manifest.json",
        {
            "source_mode": "canonical",
            "banks": {
                task: {
                    "path": str(bank),
                    "source_sha256": bank_hash,
                    "count": 3,
                }
            },
            "corpora": corpus_meta,
            "routing": {corpus: task for corpus in corpus_names},
        },
    )
    hierarchy = _json(
        tmp_path / "hierarchy.json",
        {
            "task": task,
            "n_r2_clusters_in": 3,
            "n_merged_groups": 1,
            "merged_groups": [{"metric_ids": ["m0", "m1", "m2"]}],
        },
    )

    truth_norms = norms[:4]
    roles = ("train", "dev", "test", "blind")
    decisions = (
        ("MATCH", "m0", "exact training criterion"),
        ("NOISE", None, "garbled development extraction"),
        ("MATCH", "m1", "exact held-out criterion"),
        ("NO_CANDIDATE_FITS", None, "specific blind criterion absent from bank"),
    )
    truth_rows = []
    role_rows = []
    for norm, role, (decision, metric, reason) in zip(truth_norms, roles, decisions):
        group = source_group_key(norm)
        truth_rows.append(
            {
                "task": task,
                "corpus": norm["corpus"],
                "norm_uid": norm["norm_uid"],
                "source_group": group,
                "split": "test" if role == "blind" else role,
                "decision": decision,
                "metric_id": metric,
                "acceptable_metric_ids": [metric] if metric else [],
                "confidence": "high",
                "reason": reason,
                "current_bank_source_sha256": bank_hash,
            }
        )
        role_rows.append(
            {
                "schema_version": "silver-match-v3-task-truth-role-map-v1",
                "task": task,
                "corpus": norm["corpus"],
                "norm_uid": norm["norm_uid"],
                "source_group": group,
                "role": role,
                "permanent_blind": role == "blind",
                "current_bank_source_sha256": bank_hash,
            }
        )
    existing = _jsonl(tmp_path / "truth" / "existing.jsonl", truth_rows[:2])
    new = _jsonl(tmp_path / "truth" / "new.jsonl", truth_rows[2:])
    existing_report = _truth_report(
        tmp_path / "truth" / "existing.report.json",
        task=task,
        bank_hash=bank_hash,
        source_name="trusted",
        source_kind="existing",
        truth_path=existing,
    )
    new_report = _truth_report(
        tmp_path / "truth" / "new.report.json",
        task=task,
        bank_hash=bank_hash,
        source_name="consensus",
        source_kind="new",
        truth_path=new,
    )
    role_map = _jsonl(tmp_path / "truth" / "roles.jsonl", role_rows)

    lane_bindings = {}
    for lane_index, lane in enumerate(("dense", "lexical")):
        order = ("m0", "m1", "m2") if lane_index == 0 else ("m2", "m1", "m0")
        path = _jsonl(
            tmp_path / "candidates" / f"{lane}.jsonl",
            [
                {
                    "task": task,
                    "corpus": norm["corpus"],
                    "norm_uid": norm["norm_uid"],
                    "bank_source_sha256": bank_hash,
                    "candidates": [
                        {"metric_id": metric, "rank": rank}
                        for rank, metric in enumerate(order, 1)
                    ],
                }
                for norm in norms
            ],
        )
        lane_bindings[lane] = {"path": str(path), "sha256": sha256_file(path)}
    candidate_bundle = _json(
        tmp_path / "candidates" / "BUNDLE.json",
        {
            "schema_version": "silver-match-v3-task-full-corpus-candidate-bundle-v1",
            "status": "FROZEN_FULL_CORPUS_DIVERSE_CANDIDATE_LANES",
            "task": task,
            "bank_source_sha256": bank_hash,
            "selection_split": "dev",
            "test_or_blind_labels_used_for_selection": False,
            "corpora": {
                corpus: {
                    "count": corpus_meta[corpus]["count"],
                    "canonical_norm_sha256": sha256_file(corpus_paths[corpus]),
                }
                for corpus in corpus_names
            },
            "lanes": lane_bindings,
        },
    )

    guide = tmp_path / "prompts" / "guide.txt"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text("Label exact criteria and use typed abstentions.\n", encoding="utf-8")
    rule = tmp_path / "prompts" / "rule.txt"
    rule.write_text(f"{task}: prefer the literal task-local owner.\n", encoding="utf-8")
    audit = _json(
        tmp_path / "prompts" / "audit.json",
        {
            "schema_version": "silver-match-v3-task-gepa-judge-audit-v1",
            "status": "FROZEN_TRAIN_ONLY_PROMPT_REFINEMENT_BEFORE_LABELING",
            "task": task,
            "role_contract": {
                "allowed_role": "train",
                "dev_rows_read_for_rule_authorship": 0,
                "test_or_blind_rows_read_for_rule_authorship": 0,
                "resolver_votes_or_outcomes_read": 0,
                "rule_authorship_completed_before_resolver_labels": True,
            },
            "prompt": {"path": str(rule), "sha256": sha256_file(rule)},
            "judged_train_disagreements": [{"norm_uid": truth_norms[0]["norm_uid"]}],
        },
    )
    prompt_components = _json(
        tmp_path / "prompts" / "COMPONENTS.json",
        {
            "schema_version": "silver-match-v3-task-prompt-components-v1",
            "status": "FROZEN_TASK_LOCAL_RULE_COMPONENTS",
            "task": task,
            "guide": {"name": "GUIDE", "path": str(guide), "sha256": sha256_file(guide)},
            "rules": [{"name": "GEPA-R1", "path": str(rule), "sha256": sha256_file(rule)}],
            "train_only_judge_audits": [
                {
                    "name": "GEPA-R1-audit",
                    "component": "GEPA-R1",
                    "path": str(audit),
                    "sha256": sha256_file(audit),
                }
            ],
        },
    )

    ce_model = tmp_path / "ce-model"
    gemma_model = tmp_path / "gemma-model"
    ce_model.mkdir()
    gemma_model.mkdir()
    pilot_root = tmp_path / "pilot"
    pilot_root.mkdir()
    run_config = _json(
        pilot_root / "run_config.json",
        {
            "model": str(ce_model),
            "dev_pairs": {"dev.jsonl": "bound"},
            "split_audit": {"source_group_overlap_count": 0},
            "max_length": 1024,
            "batch_size_per_rank": 8,
            "gradient_accumulation_steps": 4,
            "lora_learning_rate": 1e-4,
            "head_learning_rate": 1e-3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "lora": {"rank": 16, "alpha": 32, "dropout": 0.05},
            "attention": "eager",
            "dev_gate": {
                "minimum_exact_precision": 0.9,
                "minimum_wilson_lower": 0.8,
                "minimum_exact_predictions": 20,
            },
        },
    )
    base_manifest = _json(tmp_path / "BASE_MODEL_MANIFEST.json", {"status": "locked"})
    pilot = _json(
        tmp_path / "PILOT_SELECTION.json",
        {
            "schema_version": "silver-match-v3-task-ce-pilot-selection-v1",
            "status": "FROZEN_DEV_ONLY_SELECTION",
            "task": task,
            "selection_data_role": "development_only",
            "test_opened_before_selection": False,
            "winner": "pilot-a",
            "winner_record": {
                "root": str(pilot_root),
                "run_config_sha256": sha256_file(run_config),
            },
            "base_manifest": str(base_manifest),
            "base_manifest_sha256": sha256_file(base_manifest),
        },
    )
    repo = Path(__file__).resolve().parents[4]
    output = tmp_path / "handoff"
    args = argparse.Namespace(
        task=task,
        manifest=str(manifest),
        hierarchy=str(hierarchy),
        role_map=str(role_map),
        candidate_bundle=str(candidate_bundle),
        prompt_components=str(prompt_components),
        pilot_selection=str(pilot),
        ce_model=str(ce_model),
        gemma_model=str(gemma_model),
        python=sys.executable,
        ce_trainer=str(repo / "scripts/tools/silver_match_v3/train_nemotron_cross_encoder.py"),
        ce_scorer=str(repo / "scripts/tools/silver_match_v3/run_nemotron_ce.py"),
        gemma_trainer=str(repo / "scripts/tools/silver_match_v3/train_gemma4_typed_lora.py"),
        runtime_root=str(tmp_path / "runtime"),
        output_root=str(output),
        existing_truth=[f"trusted={existing}"],
        existing_truth_report=[f"trusted={existing_report}"],
        new_truth=[f"consensus={new}"],
        new_truth_report=[f"consensus={new_report}"],
        ce_seed=[111, 222],
        gemma_seed=333,
        pair_seed=444,
        maximum_pairs=400_000,
        global_negatives_per_norm=0,
        ce_context_chars=120,
        gemma_max_candidates=3,
        gemma_order_seed=555,
        gemma_context_chars=120,
        gemma_description_chars=120,
        gemma_example_chars=120,
        gemma_max_examples=1,
    )
    return {
        "args": args,
        "output": output,
        "candidate_bundle": candidate_bundle,
        "role_map": role_map,
        "corpus_names": corpus_names,
        "norms": norms,
    }


@pytest.mark.parametrize(
    ("task", "corpus_names"),
    [
        ("code-review", ["code_comments", "pull_requests"]),
        (
            "legal-outcome-prediction",
            [f"legal_corpus_{index:02d}" for index in range(10)],
        ),
    ],
)
def test_generic_handoff_exact_multicorpus_scope_and_firewalls(
    tmp_path: Path, task: str, corpus_names: list[str]
):
    fixture = _fixture(tmp_path, task=task, corpus_names=corpus_names)
    result = freeze(fixture["args"])
    output = fixture["output"]
    scope = json.loads((output / "TASK_SCOPE.json").read_text())
    queue = json.loads((output / "FINAL_STACK_QUEUE.json").read_text())
    truth = list(read_jsonl(output / "truth/truth.joined.all.jsonl"))
    assert result["status"] == "FROZEN_HANDOFF_NOT_PRODUCTION_OR_RELEASE_READY"
    assert scope["corpus_count"] == len(corpus_names)
    assert scope["norm_count"] == len(fixture["norms"])
    assert set(scope["corpora"]) == set(corpus_names)
    for corpus in corpus_names:
        assert scope["corpora"][corpus]["sha256"] == sha256_file(
            Path(json.loads(Path(fixture["args"].manifest).read_text())["corpora"][corpus]["path"])
        )
    assert Counter(row["split"] for row in truth) == {
        "train": 1,
        "dev": 1,
        "test": 1,
        "blind": 1,
    }
    assert next(row for row in truth if row["split"] == "blind")["permanent_blind"] is True
    assert queue["task"] == task
    assert queue["readiness"]["release_ready"] is False
    assert queue["readiness"]["production_ready"] is False
    assert len(queue["ce"]["runs"]) == 2
    for run in queue["ce"]["runs"]:
        assert str(output / "ce/train.pairs.jsonl") in run["command"]
        assert str(output / "ce/dev.pairs.jsonl") in run["command"]
        assert str(output / "ce/test.pairs.jsonl") not in run["command"]
        assert str(output / "ce/blind.pairs.jsonl") not in run["command"]
    prompt_manifest = json.loads((output / "prompts/MANIFEST.json").read_text())
    assert prompt_manifest["example_uids_included"] is False
    assert prompt_manifest["train_only_judge_audits"][0][
        "train_only_authorship_validated"
    ] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze(fixture["args"])


@pytest.mark.parametrize("failure", ["foreign_task", "foreign_corpus"])
def test_generic_handoff_rejects_foreign_task_or_corpus(tmp_path: Path, failure: str):
    fixture = _fixture(
        tmp_path,
        task="code-review",
        corpus_names=["code_comments", "pull_requests"],
    )
    if failure == "foreign_task":
        bundle = json.loads(fixture["candidate_bundle"].read_text())
        lane_path = Path(bundle["lanes"]["dense"]["path"])
        rows = list(read_jsonl(lane_path))
        rows[0]["task"] = "peer-review"
        _jsonl(lane_path, rows)
        bundle["lanes"]["dense"]["sha256"] = sha256_file(lane_path)
        _json(fixture["candidate_bundle"], bundle)
        with pytest.raises(ValueError, match="foreign task/corpus/bank"):
            freeze(fixture["args"])
    else:
        roles = list(read_jsonl(fixture["role_map"]))
        roles[0]["corpus"] = "foreign-corpus"
        _jsonl(fixture["role_map"], roles)
        with pytest.raises(ValueError, match="foreign task/corpus norm"):
            freeze(fixture["args"])
