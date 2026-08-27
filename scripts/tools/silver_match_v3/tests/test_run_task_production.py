from pathlib import Path

from scripts.tools.silver_match_v3.run_task_production import (
    _run_parallel_gpu,
    adjudicator_command,
    verifier_command,
)


def _prompt(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.write_text(name)
    return str(path)


def _plan(tmp_path: Path) -> dict:
    adj = _prompt(tmp_path, "adj.txt")
    adj_addon = _prompt(tmp_path, "adj-addon.txt")
    verify = _prompt(tmp_path, "verify.txt")
    verify_addon = _prompt(tmp_path, "verify-addon.txt")
    return {
        "manifest": {"path": "/data/manifest.json"},
        "candidate_union": {"path": "/data/candidates.jsonl"},
        "adjudicator": {
            "prompt": adj,
            "prompt_addons": [adj_addon],
            "model": "/models/gemma",
            "candidate_depth": 50,
            "prompt_rendering": {
                "context_chars": 1200,
                "description_chars": 260,
                "example_chars": 80,
                "max_examples": 0,
            },
            "production_sampling": {
                "max_model_len": 8192,
                "max_tokens": 160,
                "seed": 17,
            },
        },
        "verifier": {
            "prompt": verify,
            "prompt_addons": [verify_addon],
            "rendering": {
                "model": "/models/gemma",
                "max_alternatives": 49,
                "context_chars": 1200,
                "description_chars": 260,
                "example_chars": 180,
                "max_examples": 0,
                "max_model_len": 8192,
                "max_tokens": 180,
                "seed": 29,
            },
        },
    }


def test_adjudicator_command_is_direct_batch_and_plan_rendered(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    command = adjudicator_command(
        plan=plan,
        repo_root=tmp_path,
        gemma_python=Path("/env/gemma/bin/python"),
        output=Path("/out/original.jsonl"),
        order="original",
        batch_size=128,
        gpu_memory_utilization=0.88,
    )
    assert "scripts.tools.silver_match_v3.adjudicate_gemma" in command
    assert "--resume" in command
    assert command[command.index("--max-candidates") + 1] == "50"
    assert command[command.index("--context-chars") + 1] == "1200"
    assert command[command.index("--order-mode") + 1] == "original"
    assert "--prompt-addon" in command
    assert not any("server" in value.lower() for value in command)


def test_verifier_command_is_task_prompted_and_hashed(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    command = verifier_command(
        plan=plan,
        repo_root=tmp_path,
        gemma_python=Path("/env/gemma/bin/python"),
        primary=Path("/out/primary.jsonl"),
        output=Path("/out/verify.jsonl"),
        order="hashed",
        batch_size=64,
        gpu_memory_utilization=0.75,
    )
    assert "scripts.tools.silver_match_v3.verify_gemma" in command
    assert command[command.index("--primary") + 1] == "/out/primary.jsonl"
    assert command[command.index("--max-alternatives") + 1] == "49"
    assert command[command.index("--order-mode") + 1] == "hashed"
    assert command[command.index("--batch-size") + 1] == "64"
    assert "--prompt-addon" in command


def test_commands_bind_stable_shard_coordinates(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    adjudicator = adjudicator_command(
        plan=plan,
        repo_root=tmp_path,
        gemma_python=Path("/env/gemma/bin/python"),
        output=Path("/out/adj-shard.jsonl"),
        order="hashed",
        batch_size=128,
        gpu_memory_utilization=0.88,
        shard_id=2,
        num_shards=4,
    )
    verifier = verifier_command(
        plan=plan,
        repo_root=tmp_path,
        gemma_python=Path("/env/gemma/bin/python"),
        primary=Path("/out/primary.jsonl"),
        output=Path("/out/verifier-shard.jsonl"),
        order="original",
        batch_size=128,
        gpu_memory_utilization=0.88,
        shard_id=3,
        num_shards=4,
    )
    assert adjudicator[adjudicator.index("--shard-id") + 1] == "2"
    assert adjudicator[adjudicator.index("--num-shards") + 1] == "4"
    assert verifier[verifier.index("--shard-id") + 1] == "3"
    assert verifier[verifier.index("--num-shards") + 1] == "4"


def test_parallel_runner_allows_one_resumable_stage_with_two_gpu_pool(
    tmp_path: Path,
) -> None:
    log = tmp_path / "one.log"
    _run_parallel_gpu(
        [("one", ["/bin/sh", "-c", "printf done"], log)],
        gpus=[0, 4],
        cwd=tmp_path,
    )
    assert log.read_text().endswith("done")


def test_parallel_runner_executes_three_order_jobs_in_two_gpu_waves(
    tmp_path: Path,
) -> None:
    jobs = []
    for index in range(3):
        log = tmp_path / f"{index}.log"
        jobs.append((str(index), ["/bin/sh", "-c", f"printf order{index}"], log))
    _run_parallel_gpu(jobs, gpus=[0, 4], cwd=tmp_path)
    assert [
        (tmp_path / f"{index}.log").read_text().endswith(f"order{index}")
        for index in range(3)
    ] == [True, True, True]
