from pathlib import Path
import sys

import pytest

from scripts.tools.silver_match_v3.watch_and_launch_task_lora import (
    is_gpu_free,
    sha256_file,
    training_command,
    verify_queue_hashes,
    verify_training_python,
)


def test_gpu_free_gate_excludes_reserved_and_memory_occupied_devices() -> None:
    free = {
        "index": 2,
        "memory_free_mib": 170000,
        "memory_used_mib": 614,
        "utilization_percent": 0,
    }
    assert is_gpu_free(
        free,
        excluded={5},
        minimum_free_memory_mib=120000,
        maximum_used_memory_mib=2048,
        maximum_utilization_percent=5,
    )
    assert not is_gpu_free(
        {**free, "index": 5},
        excluded={5},
        minimum_free_memory_mib=120000,
        maximum_used_memory_mib=2048,
        maximum_utilization_percent=5,
    )
    assert not is_gpu_free(
        {**free, "memory_used_mib": 166000},
        excluded={5},
        minimum_free_memory_mib=120000,
        maximum_used_memory_mib=2048,
        maximum_utilization_percent=5,
    )


def queue_fixture(tmp_path: Path) -> dict:
    files = {}
    for name in (
        "manifest",
        "combined_teachers",
        "external_dev",
        "external_dev_test",
        "promotion_policy",
        "trainer",
    ):
        path = tmp_path / name
        path.write_text(name)
        files[name] = path
    config = {
        "learning_rate": 0.00002,
        "epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 512,
        "margin": 0.15,
        "hard_negative_pool": 16,
        "negatives_per_positive": 2,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "train_percent": 80,
        "dev_percent": 10,
        "selection_k": 50,
        "epoch_selection_policy": "depth_lexicographic",
        "seed": 1729,
        "split_seed": 73129,
    }
    return {
        "task": "peer-review",
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in files.items()
            if name != "trainer"
        },
        "trainer": {
            "path": str(files["trainer"]),
            "expected_sha256_before_launch": sha256_file(files["trainer"]),
            "output_root": str(tmp_path / "adapter"),
            "hyperparameters": config,
        },
    }


def test_queue_hashes_fail_closed_and_training_command_excludes_eval(
    tmp_path: Path,
) -> None:
    queue = queue_fixture(tmp_path)
    observed = verify_queue_hashes(queue)
    assert set(observed) == {
        "manifest",
        "combined_teachers",
        "external_dev",
        "external_dev_test",
        "promotion_policy",
        "trainer",
    }
    command = training_command(queue, Path("/env/python"))
    rendered = " ".join(command)
    assert queue["inputs"]["combined_teachers"]["path"] in rendered
    assert queue["inputs"]["external_dev"]["path"] not in rendered
    assert queue["inputs"]["external_dev_test"]["path"] not in rendered
    Path(queue["trainer"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_queue_hashes(queue)


def test_training_python_preflight_requires_lora_stack() -> None:
    libraries = verify_training_python(Path(sys.executable))
    assert {"peft", "sentence_transformers", "torch", "transformers"} <= set(libraries)
