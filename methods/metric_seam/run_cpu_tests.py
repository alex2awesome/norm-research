#!/usr/bin/env python
"""Run the canonical metric-seam CPU-only verification suite.

The runner deliberately uses the current interpreter and repository root for every
subprocess.  It removes API credentials and masks accelerator devices before launching
pytest or the historical standalone batteries.  It does not make model/API calls, use a
GPU, or rewrite scientific artifacts.

Usage: python -m methods.metric_seam.run_cpu_tests
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

# Provider-neutral patterns cover credentials added after this runner was written;
# explicit token names cover common credentials that do not contain ``API_KEY``.
_SECRET_NAME_FRAGMENTS = (
    "API_KEY",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "KEY_FILE",
    "CREDENTIAL",
)
_SECRET_NAMES = {
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
}
_DEVICE_MASK = {
    "CUDA_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "none",
    "HIP_VISIBLE_DEVICES": "-1",
    "ROCR_VISIBLE_DEVICES": "-1",
}


def sanitized_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a subprocess environment with credentials removed and GPUs hidden."""

    env = dict(os.environ if source is None else source)
    for name in list(env):
        upper = name.upper()
        if upper in _SECRET_NAMES or any(
            fragment in upper for fragment in _SECRET_NAME_FRAGMENTS
        ):
            env.pop(name, None)
    env.update(_DEVICE_MASK)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), current_pythonpath) if part
    )
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def commands(*, pytest_args: Sequence[str] = ()) -> tuple[tuple[str, ...], ...]:
    """Return the complete ordered CPU suite using this Python interpreter."""

    python = sys.executable
    return (
        (
            python,
            "-m",
            "pytest",
            "methods/metric_seam",
            "-q",
            *pytest_args,
        ),
        (python, "-m", "methods.metric_seam.tests_certificates"),
        (python, "-m", "methods.metric_seam.hybrids.test_ops_capability"),
        (
            python,
            "-m",
            "methods.metric_seam.hybrids.eval_ops_capability_v2",
            "--check",
        ),
    )


def run(command_list: Sequence[Sequence[str]], *, dry_run: bool = False) -> int:
    """Run commands in order, returning immediately on the first failure."""

    env = sanitized_environment()
    for command in command_list:
        rendered = shlex.join(command)
        print(f"[metric-seam CPU] {rendered}", flush=True)
        if dry_run:
            continue
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if result.returncode:
            print(
                f"[metric-seam CPU] FAILED ({result.returncode}): {rendered}",
                file=sys.stderr,
            )
            return result.returncode
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact CPU-only commands without executing them",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="append one argument to the pytest command (repeatable)",
    )
    args = parser.parse_args(argv)
    return run(commands(pytest_args=args.pytest_arg), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
