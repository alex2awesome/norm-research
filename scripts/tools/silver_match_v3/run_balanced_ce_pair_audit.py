#!/usr/bin/env python3
"""Run a frozen balanced-CE queue through pair construction only."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .common import sha256_file
from .launch_frozen_cross_encoder_queues import validate_queue


PAIR_FLAGS = (
    "--task",
    "--policy",
    "--manifest",
    "--train-teachers",
    "--dev-teachers",
    "--candidates",
    "--negatives-per-positive",
    "--negatives-per-abstain",
    "--strong-positive-repeats",
)


def _values(command: list[str], flag: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, token in enumerate(command):
        if token == flag:
            if index + 1 >= len(command):
                raise ValueError(f"missing value after {flag}")
            values.append(command[index + 1])
    if not values:
        raise ValueError(f"pair-affecting command flag is missing: {flag}")
    return tuple(values)


def _pair_contract(command: list[str]) -> dict[str, tuple[str, ...]]:
    return {flag: _values(command, flag) for flag in PAIR_FLAGS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    queue_path = Path(args.queue).resolve()
    queue, verified_artifacts = validate_queue(queue_path, set())
    commands = queue.get("commands") or []
    if not commands:
        raise ValueError("queue has no commands")
    contracts = [_pair_contract(list(entry["command"])) for entry in commands]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("queue variants differ on pair-affecting command arguments")
    policy_path = Path(contracts[0]["--policy"][0]).resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    sampling_seed = policy.get("balanced_training", {}).get("sampling_seed")
    if sampling_seed is None:
        raise ValueError("balanced policy omits policy-fixed sampling_seed")
    pair_contract = {
        "all_variant_count": len(commands),
        "all_variants_pair_equivalent": True,
        "audited_variants": [entry["variant"] for entry in commands],
        "normalized_pair_arguments": {
            flag: list(values) for flag, values in contracts[0].items()
        },
        "pair_sampling_seed_source": "frozen balanced policy, not variant seed",
        "policy_sampling_seed": int(sampling_seed),
        "verified_queue_artifact_count": len(verified_artifacts),
    }
    command = list(commands[0]["command"]) + ["--audit-pairs-only"]
    completed = subprocess.run(command, text=True, capture_output=True)

    record: dict[str, object]
    if completed.returncode == 0:
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise ValueError("pair audit emitted no JSON")
        emitted = json.loads(lines[-1])
        if emitted.get("status") != "PAIR_EXPOSURE_AUDIT_PASS_NO_TRAINING":
            raise ValueError(f"unexpected pair audit status: {emitted.get('status')}")
        record = {
            **emitted,
            "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
            "audited_variant": commands[0]["variant"],
            "queue_pair_contract": pair_contract,
            "training_started": False,
            "model_initialized": False,
        }
    else:
        record = {
            "status": "AUDIT_PROCESS_FAILURE",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
            "audited_variant": commands[0]["variant"],
            "queue_pair_contract": pair_contract,
            "training_started": False,
            "model_initialized": False,
        }

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "status": record["status"],
                "returncode": completed.returncode,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
