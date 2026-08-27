from __future__ import annotations

import os
import sys

from methods.metric_seam import run_cpu_tests


def test_sanitized_environment_removes_credentials_and_masks_devices() -> None:
    source = {
        "PATH": "/bin",
        "OPENAI_API_KEY": "secret",
        "SOME_NEW_PROVIDER_ACCESS_TOKEN": "secret",
        "HF_TOKEN": "secret",
        "ZAI_KEY_FILE": "/secret/key/file",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "PYTHONPATH": "/existing",
    }
    env = run_cpu_tests.sanitized_environment(source)
    assert "OPENAI_API_KEY" not in env
    assert "SOME_NEW_PROVIDER_ACCESS_TOKEN" not in env
    assert "HF_TOKEN" not in env
    assert "ZAI_KEY_FILE" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["NVIDIA_VISIBLE_DEVICES"] == "none"
    assert env["HIP_VISIBLE_DEVICES"] == "-1"
    assert env["ROCR_VISIBLE_DEVICES"] == "-1"
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(run_cpu_tests.ROOT),
        "/existing",
    ]


def test_commands_use_current_interpreter_and_package_entrypoints() -> None:
    command_list = run_cpu_tests.commands(pytest_args=("-x",))
    assert all(command[0] == sys.executable for command in command_list)
    assert command_list[0][1:4] == ("-m", "pytest", "methods/metric_seam")
    assert "-x" in command_list[0]
    assert command_list[-1][1:3] == (
        "-m",
        "methods.metric_seam.hybrids.eval_ops_capability_v2",
    )
