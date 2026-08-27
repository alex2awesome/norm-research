import argparse
import json
import platform
import sys
from pathlib import Path

import pytest

import scripts.tools.silver_match_v3.run_frozen_nemotron_ce_production as runner
from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_nemotron_ce_production_queue import (
    freeze,
)
from scripts.tools.silver_match_v3.materialize_nemotron_ce_production_pairs import (
    META_SCHEMA,
    PAIR_SCHEMA,
    UNIVERSE_SCHEMA,
)
from scripts.tools.silver_match_v3.run_frozen_nemotron_ce_production import (
    build_consensus_command,
    freeze_seed_manifest,
    validate_merged_scores,
    validate_queue,
    validate_score_shard,
)
from scripts.tools.silver_match_v3.run_nemotron_ce import (
    CHECKPOINT_SCHEMA,
    SCORE_META_SCHEMA,
    SCORE_SCHEMA,
    build_base_manifest,
    merge_score_shards,
    pair_shard,
)
from scripts.tools.silver_match_v3.train_nemotron_cross_encoder import (
    CLASS_NAMES,
    HIDDEN_SIZE,
    LORA_TARGETS,
    REPORT_SCHEMA,
)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _checkpoint_and_report(
    root: Path,
    *,
    seed: int,
    model: Path,
    train_pairs: Path,
    dev_pairs: Path,
) -> tuple[Path, Path]:
    run = root / f"train-seed-{seed}"
    checkpoint = run / "checkpoints" / "exposure-000000100000"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    _json(adapter / "adapter_config.json", {"base_model_name_or_path": str(model)})
    (adapter / "adapter_model.safetensors").write_bytes(f"adapter-{seed}".encode())
    (checkpoint / "head.safetensors").write_bytes(f"head-{seed}".encode())
    dev = {
        "score_threshold": 0.71 + seed / 1000,
        "top_margin_threshold": 0.12,
        "retained_exact_precision": 0.96,
    }
    _json(
        checkpoint / "checkpoint.json",
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "labels": list(CLASS_NAMES),
            "hidden_to_classes": [HIDDEN_SIZE, len(CLASS_NAMES)],
            "lora_targets": list(LORA_TARGETS),
            "dev": dev,
        },
    )
    selected = {
        "path": str(checkpoint),
        "exposure_budget": 100_000,
        "dev": dev,
        "artifact_sha256": _artifact_hashes(checkpoint),
        "checkpoint_metadata_sha256": sha256_file(checkpoint / "checkpoint.json"),
    }
    inputs = {
        "train_pairs": {str(train_pairs.resolve()): sha256_file(train_pairs)},
        "dev_pairs": {str(dev_pairs.resolve()): sha256_file(dev_pairs)},
    }
    run_config = {
        "schema_version": REPORT_SCHEMA,
        "model": str(model),
        "seed": seed,
        "max_length": 512,
        "train_pairs": inputs["train_pairs"],
        "dev_pairs": inputs["dev_pairs"],
    }
    _json(run / "run_config.json", run_config)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETE",
        "model": str(model),
        "labels": list(CLASS_NAMES),
        "hidden_to_classes": [HIDDEN_SIZE, len(CLASS_NAMES)],
        "max_sequence_length": 512,
        "selected_checkpoint": selected,
        "input_sha256": {
            **inputs,
            "run_config": sha256_file(run / "run_config.json"),
        },
    }
    _json(run / "training_report.json", report)
    return checkpoint, run / "training_report.json"


def _fixture(tmp_path: Path, *, leak_label: bool = False) -> argparse.Namespace:
    task = "humor"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    base_manifest = tmp_path / "base-manifest.json"
    build_base_manifest(model, base_manifest)

    bank_source = "b" * 64
    bank = tmp_path / "bank.json"
    _json(
        bank,
        {
            "task": task,
            "source_sha256": bank_source,
            "metrics": [
                {"metric_id": "m1", "name": "surprise"},
                {"metric_id": "m2", "name": "brevity"},
            ],
        },
    )
    canonical = tmp_path / "canonical.jsonl"
    _jsonl(
        canonical,
        [
            {"norm_uid": "n1", "task": task, "corpus": "jokes"},
            {"norm_uid": "n2", "task": task, "corpus": "jokes"},
        ],
    )
    manifest = tmp_path / "manifest.json"
    _json(
        manifest,
        {
            "banks": {
                task: {"path": str(bank), "source_sha256": bank_source, "count": 2}
            },
            "corpora": {
                "jokes": {"path": str(canonical), "task": task, "count": 2}
            },
        },
    )
    candidates = tmp_path / "candidates.jsonl"
    _jsonl(candidates, [{"norm_uid": "n1"}, {"norm_uid": "n2"}])
    candidate_meta = candidates.with_suffix(candidates.suffix + ".meta.json")
    _json(
        candidate_meta,
        {
            "task": task,
            "corpus": "jokes",
            "input_count": 2,
            "output_sha256": sha256_file(candidates),
            "manifest_sha256": sha256_file(manifest),
            "bank_source_sha256": bank_source,
            "output_k": 2,
            "union": {"lanes": [{"name": "dense"}, {"name": "lexical"}]},
        },
    )
    universe = tmp_path / "universe.jsonl"
    _jsonl(
        universe,
        [
            {
                "schema_version": UNIVERSE_SCHEMA,
                "task": task,
                "corpus": "jokes",
                "norm_uid": uid,
                "source_group": f"group-{uid}",
                "split": "production",
            }
            for uid in ("n1", "n2")
        ],
    )
    pairs = tmp_path / "pairs.jsonl"
    pair_rows = []
    for uid in ("n1", "n2"):
        for rank, metric_id in enumerate(("m1", "m2"), 1):
            row = {
                "schema_version": PAIR_SCHEMA,
                "task": task,
                "corpus": "jokes",
                "norm_uid": uid,
                "source_group": f"group-{uid}",
                "split": "production",
                "query": f"norm {uid}",
                "metric_id": metric_id,
                "metric_card": f"metric {metric_id}",
                "candidate_rank": rank,
                "current_bank_source_sha256": bank_source,
            }
            if leak_label and uid == "n1" and metric_id == "m1":
                row["label"] = "EXACT"
            pair_rows.append(row)
    _jsonl(pairs, pair_rows)
    pair_report = pairs.with_suffix(pairs.suffix + ".meta.json")
    _json(
        pair_report,
        {
            "schema_version": META_SCHEMA,
            "status": "FROZEN_COMPLETE_UNLABELED_PRODUCTION_PAIR_UNIVERSE",
            "task": task,
            "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "bank": {
                "path": str(bank),
                "sha256": sha256_file(bank),
                "source_sha256": bank_source,
                "metric_count": 2,
            },
            "corpus_order": ["jokes"],
            "corpora": {
                "jokes": {
                    "canonical": {
                        "path": str(canonical),
                        "sha256": sha256_file(canonical),
                        "count": 2,
                    },
                    "candidate_union": {
                        "path": str(candidates),
                        "sha256": sha256_file(candidates),
                        "meta": str(candidate_meta),
                        "meta_sha256": sha256_file(candidate_meta),
                        "lane_names": ["dense", "lexical"],
                        "output_k": 2,
                    },
                    "pair_count": 4,
                }
            },
            "norm_count": 2,
            "candidate_depth": 2,
            "pair_count": 4,
            "pairs": {"path": str(pairs), "sha256": sha256_file(pairs)},
            "norm_universe": {
                "path": str(universe),
                "sha256": sha256_file(universe),
                "count": 2,
            },
            "labels_present": False,
            "single_lane_candidates_accepted": False,
            "diagnostic_subset_accepted": False,
            "release_ready": False,
        },
    )
    train_pairs = tmp_path / "train.pairs.jsonl"
    dev_pairs = tmp_path / "dev.pairs.jsonl"
    _jsonl(train_pairs, [{"task": task, "norm_uid": "train-1"}])
    _jsonl(dev_pairs, [{"task": task, "norm_uid": "dev-1"}])
    checkpoint_a, report_a = _checkpoint_and_report(
        tmp_path, seed=11, model=model, train_pairs=train_pairs, dev_pairs=dev_pairs
    )
    checkpoint_b, report_b = _checkpoint_and_report(
        tmp_path, seed=29, model=model, train_pairs=train_pairs, dev_pairs=dev_pairs
    )
    repo_root = Path(__file__).resolve().parents[4]
    return argparse.Namespace(
        task=task,
        pair_report=str(pair_report),
        seed_id=["11", "29"],
        training_report=[str(report_a), str(report_b)],
        checkpoint=[str(checkpoint_a), str(checkpoint_b)],
        model=str(model),
        base_manifest=str(base_manifest),
        python=sys.executable,
        repo_root=str(repo_root),
        output_root=str(tmp_path / "runtime"),
        target_host=platform.node(),
        gpu_index=[0, 1],
        num_shards=2,
        batch_size=4,
        attention="eager",
        output=str(tmp_path / "queue.json"),
    )


def _write_synthetic_scores(plan: dict) -> None:
    pair_rows = [json.loads(line) for line in Path(plan["production_pairs"]["pairs"]["path"]).read_text().splitlines()]
    for seed_index, seed in enumerate(plan["seeds"]):
        for job in seed["shards"]:
            rows = []
            for pair in pair_rows:
                if pair_shard(pair["norm_uid"], job["num_shards"]) != job["shard_id"]:
                    continue
                rows.append(
                    {
                        "schema_version": SCORE_SCHEMA,
                        "norm_uid": pair["norm_uid"],
                        "metric_id": pair["metric_id"],
                        "source_group": pair["source_group"],
                        "split": "production",
                        "predicted_relation": "EXACT",
                        "probabilities": {"EXACT": 0.8 - seed_index * 0.01, "FAMILY": 0.1 + seed_index * 0.01, "REJECT": 0.1},
                    }
                )
            output = Path(job["output"])
            _jsonl(output, rows)
            _json(
                Path(job["meta"]),
                {
                    "schema_version": SCORE_META_SCHEMA,
                    "input_pairs": str(Path(plan["production_pairs"]["pairs"]["path"]).resolve()),
                    "input_pairs_sha256": plan["production_pairs"]["pairs"]["sha256"],
                    "output": str(output.resolve()),
                    "output_sha256": sha256_file(output),
                    "row_count": len(rows),
                    "norm_group_count": len({row["norm_uid"] for row in rows}),
                    "shard_id": job["shard_id"],
                    "num_shards": job["num_shards"],
                    "base_contract": plan["base_model"]["verified_contract"],
                    "checkpoint_contract": seed["checkpoint_contract"],
                    "labels": list(CLASS_NAMES),
                    "bidirectional_concatenation": True,
                    "pooling": "native_attention_mask_mean",
                    "max_length": seed["checkpoint_contract"]["max_sequence_length"],
                    "cuda_bf16": True,
                    "attention": "eager",
                },
            )
            assert validate_score_shard(plan, seed, job)
        merge_score_shards(
            [Path(job["output"]) for job in seed["shards"]],
            Path(seed["merged"]["scores"]),
        )
        assert validate_merged_scores(plan, seed)


def test_freezes_two_seed_task_local_deterministic_commands(tmp_path):
    plan = freeze(_fixture(tmp_path))
    assert plan["production_pairs"]["norm_count"] == 2
    assert plan["production_pairs"]["pair_count"] == 4
    assert len(plan["seeds"]) == 2
    assert sum(plan["production_pairs"]["shard_pair_counts"]) == 4
    commands = [job["command"] for seed in plan["seeds"] for job in seed["shards"]]
    assert all(command[command.index("--device") + 1] == "0" for command in commands)
    assert all("--score-threshold" not in command for command in commands)
    validate_queue(plan, hostname=platform.node())


def test_rejects_any_label_in_production_pairs(tmp_path):
    with pytest.raises(ValueError, match="leakage"):
        freeze(_fixture(tmp_path, leak_label=True))


def test_sk3_prohibited_gpu_fails_before_launch(tmp_path):
    args = _fixture(tmp_path)
    args.target_host = "sk3"
    args.gpu_index = [1]
    with pytest.raises(ValueError, match="sk3 GPU policy violation"):
        freeze(args)


def test_synthetic_merge_content_addressed_manifest_and_consensus_command(tmp_path):
    plan = freeze(_fixture(tmp_path))
    _write_synthetic_scores(plan)
    manifest, digest = freeze_seed_manifest(plan)
    assert manifest.name == f"{digest}.json"
    assert sha256_file(manifest) == digest
    command = build_consensus_command(plan, manifest, digest)
    assert command[command.index("--seed-manifest-sha256") + 1] == digest
    assert not any("threshold" in value for value in command)
    assert freeze_seed_manifest(plan) == (manifest, digest)


def test_partial_shard_is_not_resume_eligible(tmp_path):
    plan = freeze(_fixture(tmp_path))
    job = plan["seeds"][0]["shards"][0]
    Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["output"]).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="partial score shard"):
        validate_score_shard(plan, plan["seeds"][0], job)


def test_runner_resumes_only_missing_shards_in_unique_gpu_waves(tmp_path, monkeypatch):
    plan = freeze(_fixture(tmp_path))
    state = {
        (seed["seed_id"], job["shard_id"]): False
        for seed in plan["seeds"]
        for job in seed["shards"]
    }
    state[(plan["seeds"][0]["seed_id"], 0)] = True
    by_output = {
        str(Path(job["output"])): (seed["seed_id"], job["shard_id"])
        for seed in plan["seeds"]
        for job in seed["shards"]
    }
    launched = []
    guarded = []

    def fake_valid(_plan, seed, job):
        return state[(seed["seed_id"], job["shard_id"])]

    def fake_guard(gpus, *, hostname):
        guarded.append((tuple(gpus), hostname))
        return {"selected_gpu_indices": list(gpus), "host": hostname}

    class FakePopen:
        def __init__(self, command, **_kwargs):
            output = command[command.index("--output") + 1]
            self.key = by_output[output]
            launched.append(self.key)

        def wait(self):
            state[self.key] = True
            return 0

    monkeypatch.setattr(runner, "validate_score_shard", fake_valid)
    monkeypatch.setattr(runner, "validate_launch_gpus", fake_guard)
    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    guards = runner._run_missing_shards(
        plan, {"hostname": platform.node(), "gpus": (0, 1)}
    )
    assert len(launched) == 3
    assert (plan["seeds"][0]["seed_id"], 0) not in launched
    assert len(guards) == len(guarded) == 2
    assert all(len(set(gpus)) == len(gpus) for gpus, _ in guarded)


def test_runner_rejects_command_tampering_or_threshold_injection(tmp_path):
    plan = freeze(_fixture(tmp_path))
    plan["seeds"][0]["shards"][0]["command"].extend(
        ["--score-threshold", "0.0"]
    )
    with pytest.raises(ValueError, match="command"):
        validate_queue(plan, hostname=platform.node(), deep_inputs=False)
