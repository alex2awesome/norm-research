#!/usr/bin/env python3
"""Freeze three policy-bound task-specific cross-encoder training commands."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import normalize_space, read_jsonl, sha256_file
from .score_verifier_calibration import wilson_interval
from .train_nemotron_lora import source_group_key


ALLTASK_POLICY_V1 = "silver-match-v3-cross-encoder-alltask-policy-v1"
PRESS_RELEASES_POLICY_V2 = "silver-match-v3-cross-encoder-press-releases-policy-v2"
TRAINER_RELATIVE_PATH = Path("scripts/tools/silver_match_v3/train_cross_encoder.py")
SELECTOR_RELATIVE_PATH = Path(
    "scripts/tools/silver_match_v3/select_cross_encoder_variants.py"
)


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _extra_artifact(value: str) -> dict[str, Any]:
    if "=" not in value:
        raise ValueError("--extra-binding must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise ValueError("empty extra binding name")
    return {"name": name.strip(), **_artifact(Path(raw_path).resolve())}


def _eligibility(policy_path: Path, policy: dict[str, Any], task: str) -> dict[str, Any] | None:
    path = policy_path.with_suffix(".ELIGIBILITY.json")
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("policy_sha256") != sha256_file(policy_path)
        or task not in value.get("eligible_primary_tasks", [])
    ):
        raise ValueError("policy eligibility registry restricts this task")
    return _artifact(path)


def _validate_policy_and_implementation(
    policy: dict[str, Any], task: str, repo_root: Path
) -> dict[str, Any]:
    schema = policy.get("schema_version")
    scope = policy.get("scope", [])
    if schema == ALLTASK_POLICY_V1:
        if task not in scope:
            raise ValueError("unsupported policy/task")
        return {"schema_version": schema, "policy_binding_verified": True}
    if not (
        schema == PRESS_RELEASES_POLICY_V2
        and task == "press-releases"
        and scope == ["press-releases"]
    ):
        raise ValueError("unsupported policy/task")

    implementation = policy.get("implementation", {})
    expected_paths = {
        "train_cross_encoder": TRAINER_RELATIVE_PATH,
        "select_cross_encoder_variants": SELECTOR_RELATIVE_PATH,
    }
    audit: dict[str, Any] = {
        "schema_version": schema,
        "policy_revision": policy.get("policy_revision"),
        "policy_binding_verified": True,
        "artifacts": {},
    }
    for key, relative_path in expected_paths.items():
        path_key = f"{key}_path"
        sha_key = f"{key}_sha256"
        if implementation.get(path_key) != str(relative_path):
            raise ValueError(f"unexpected policy implementation path: {path_key}")
        artifact = _artifact(repo_root / relative_path)
        if artifact["sha256"] != implementation.get(sha_key):
            raise ValueError(f"policy implementation hash mismatch: {key}")
        audit["artifacts"][key] = artifact
    return audit


def _audit_dev_gate_feasibility(
    *,
    dev_uids: set[str],
    possible_match_uids: set[str],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Prove that an oracle could satisfy the frozen dev support gate.

    This is deliberately an upper-bound calculation: every UID with any
    ``MATCH`` row is treated as an exactly recoverable true positive and every
    other row can be perfectly abstained.  If even that oracle cannot meet the
    minimum retained count, precision, and Wilson lower bound, running a model
    would be both uninformative and a waste of compute.
    """

    dev_count = len(dev_uids)
    gold_match_upper_bound = len(possible_match_uids & dev_uids)
    min_predictions = int(gate["minimum_retained_predictions"])
    min_precision = float(gate["minimum_exact_match_precision"])
    min_wilson_lower = float(
        gate["minimum_exact_match_precision_wilson_95_lower"]
    )
    feasible: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for predicted in range(min_predictions, dev_count + 1):
        true_positive = min(gold_match_upper_bound, predicted)
        precision = true_positive / predicted
        interval = wilson_interval(true_positive, predicted)
        row = {
            "predicted_match_count": predicted,
            "maximum_true_positive_count": true_positive,
            "maximum_precision": precision,
            "maximum_wilson_95_lower": interval[0] if interval else None,
        }
        if best is None or (
            row["maximum_wilson_95_lower"] or -1.0,
            row["maximum_precision"],
            row["maximum_true_positive_count"],
        ) > (
            best["maximum_wilson_95_lower"] or -1.0,
            best["maximum_precision"],
            best["maximum_true_positive_count"],
        ):
            best = row
        if (
            precision >= min_precision
            and interval is not None
            and interval[0] >= min_wilson_lower
        ):
            feasible.append(row)
    audit = {
        "complete": True,
        "feasible_under_oracle": bool(feasible),
        "dev_unique_uid_count": dev_count,
        "possible_gold_match_upper_bound": gold_match_upper_bound,
        "minimum_retained_predictions": min_predictions,
        "minimum_exact_match_precision": min_precision,
        "minimum_exact_match_precision_wilson_95_lower": min_wilson_lower,
        "best_oracle_case": best,
        "first_feasible_oracle_case": feasible[0] if feasible else None,
    }
    if not feasible:
        raise ValueError(
            "dev gate is mathematically infeasible even for an oracle: "
            + json.dumps(audit, sort_keys=True)
        )
    return audit


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    policy_path = Path(args.policy).resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    repo_root = Path(args.repo_root).resolve()
    implementation_audit = _validate_policy_and_implementation(
        policy, args.task, repo_root
    )
    eligibility = _eligibility(policy_path, policy, args.task)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.task not in manifest.get("banks", {}):
        raise ValueError("task absent from manifest")
    bank_meta = manifest["banks"][args.task]
    bank_path = Path(bank_meta["path"]).resolve()
    bank = json.loads(bank_path.read_text(encoding="utf-8"))["metrics"]
    bank_ids = {str(row["metric_id"]) for row in bank}

    role_paths = {
        "train": [Path(value).resolve() for value in args.train_teachers],
        "dev": [Path(value).resolve() for value in args.dev_teachers],
    }
    if not all(role_paths.values()):
        raise ValueError("both train and dev teacher inputs are required")
    uid_roles: dict[str, set[str]] = defaultdict(set)
    dev_uids: set[str] = set()
    dev_possible_match_uids: set[str] = set()
    label_counts: Counter[str] = Counter()
    teacher_artifacts: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for role, paths in role_paths.items():
        for path in paths:
            teacher_artifacts[role].append(_artifact(path))
            for row in read_jsonl(path):
                uid = normalize_space(row.get("norm_uid"))
                if not uid:
                    raise ValueError(f"teacher row without UID: {path}")
                uid_roles[uid].add(role)
                if role == "dev":
                    dev_uids.add(uid)
                    if row.get("decision") == "MATCH":
                        dev_possible_match_uids.add(uid)
                label_counts[role] += 1
                row_task = normalize_space(row.get("task"))
                if row_task and row_task != args.task:
                    raise ValueError(f"teacher task mismatch: {path}/{uid}")
                row_bank = normalize_space(
                    row.get("current_bank_source_sha256")
                    or row.get("bank_source_sha256")
                )
                if row_bank and row_bank != bank_meta["source_sha256"]:
                    raise ValueError(f"stale teacher bank: {path}/{uid}")
                if row.get("decision") == "MATCH" and str(row.get("metric_id")) not in bank_ids:
                    raise ValueError(f"teacher metric absent from bank: {path}/{uid}")
                gepa_role = normalize_space(row.get("gepa_role"))
                if gepa_role == "optimize" and (
                    role != "train" or row.get("ce_training_eligible") is not True
                ):
                    raise ValueError(
                        f"optimize truth lacks CE bridge authorization: {path}/{uid}"
                    )
                if gepa_role == "select" and role != "dev":
                    raise ValueError(f"select truth is outside CE dev: {path}/{uid}")
                if gepa_role in {"evaluation", "test", "blind"}:
                    raise ValueError(f"forbidden truth role in CE queue: {path}/{uid}")
    cross_uid = {uid: roles for uid, roles in uid_roles.items() if len(roles) > 1}
    if cross_uid:
        raise ValueError(f"teacher UIDs cross roles: {sorted(cross_uid)[:5]}")

    needed = set(uid_roles)
    norm_groups: dict[str, str] = {}
    for corpus, meta in manifest["corpora"].items():
        if meta["task"] != args.task:
            continue
        for row in read_jsonl(Path(meta["path"])):
            uid = str(row["norm_uid"])
            if uid in needed:
                norm_groups[uid] = source_group_key(row)
    missing = needed - set(norm_groups)
    if missing:
        raise ValueError(f"teacher UIDs absent from task manifest: {sorted(missing)[:5]}")
    group_roles: dict[str, set[str]] = defaultdict(set)
    for uid, roles in uid_roles.items():
        group_roles[norm_groups[uid]].update(roles)
    cross_groups = {
        group: roles for group, roles in group_roles.items() if len(roles) > 1
    }
    if cross_groups:
        raise ValueError(f"source groups cross CE roles: {sorted(cross_groups)[:5]}")

    candidate_paths = [Path(value).resolve() for value in args.candidates]
    if not candidate_paths:
        raise ValueError("candidate inputs are required")
    candidate_artifacts = [_artifact(path) for path in candidate_paths]
    extra_bindings = [_extra_artifact(value) for value in args.extra_binding or []]
    extra_names = [row["name"] for row in extra_bindings]
    if len(extra_names) != len(set(extra_names)):
        raise ValueError("duplicate extra binding name")
    # Preserve a virtual environment's requested interpreter entry point.
    # Resolving ``env/bin/python`` to the base binary makes the subsequently
    # launched process lose pyvenv.cfg and its frozen site-packages.
    python = Path(args.python).absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    output_root = Path(args.output_root).resolve()
    fixed = policy["fixed_training"]
    gate = policy["dev_gate"]
    dev_gate_feasibility = _audit_dev_gate_feasibility(
        dev_uids=dev_uids,
        possible_match_uids=dev_possible_match_uids,
        gate=gate,
    )
    commands = []
    for variant in policy["predeclared_variants"]:
        variant_root = output_root / str(variant["name"])
        command = [
            str(python),
            "-u",
            "-m",
            "scripts.tools.silver_match_v3.train_cross_encoder",
            "--task",
            args.task,
            "--policy",
            str(policy_path),
            "--variant-name",
            str(variant["name"]),
            "--manifest",
            str(manifest_path),
            "--model",
            str(policy["base_model"]["path"]),
            "--output-root",
            str(variant_root),
            "--device",
            "cuda",
            "--epochs",
            str(fixed["epochs"]),
            "--batch-size",
            str(fixed["batch_size"]),
            "--eval-batch-size",
            str(fixed["eval_batch_size"]),
            "--max-length",
            str(fixed["max_length"]),
            "--learning-rate",
            str(variant["learning_rate"]),
            "--warmup-ratio",
            str(fixed["warmup_ratio"]),
            "--negatives-per-positive",
            str(fixed["negatives_per_positive"]),
            "--negatives-per-abstain",
            str(fixed["negatives_per_abstain"]),
            "--strong-positive-repeats",
            str(fixed["strong_positive_repeats"]),
            "--seed",
            str(variant["seed"]),
            "--min-dev-precision",
            str(gate["minimum_exact_match_precision"]),
            "--min-dev-precision-lower",
            str(gate["minimum_exact_match_precision_wilson_95_lower"]),
            "--min-dev-predictions",
            str(gate["minimum_retained_predictions"]),
            "--min-dev-gain",
            str(gate["minimum_exact_f_beta_0_5_gain_over_frozen_base"]),
            "--dev-only",
        ]
        for path in role_paths["train"]:
            command.extend(("--train-teachers", str(path)))
        for path in role_paths["dev"]:
            command.extend(("--dev-teachers", str(path)))
        for path in candidate_paths:
            command.extend(("--candidates", str(path)))
        commands.append(
            {
                "variant": variant,
                "output_root": str(variant_root),
                "expected_report": str(variant_root / args.task / "training_report.json"),
                "command": command,
            }
        )
    implementation = repo_root / "scripts/tools/silver_match_v3/train_cross_encoder.py"
    return {
        "schema_version": "silver-match-v3-cross-encoder-training-queue-v1",
        "status": "FROZEN_NOT_LAUNCHED",
        "task": args.task,
        "policy": _artifact(policy_path),
        "policy_eligibility": eligibility,
        "manifest": _artifact(manifest_path),
        "bank": {
            **_artifact(bank_path),
            "source_sha256": bank_meta["source_sha256"],
            "count": len(bank),
        },
        "teacher_inputs": teacher_artifacts,
        "candidate_inputs": candidate_artifacts,
        "extra_bindings": extra_bindings,
        "role_audit": {
            "complete": True,
            "label_counts": dict(sorted(label_counts.items())),
            "unique_uids": len(uid_roles),
            "unique_source_groups": len(group_roles),
            "cross_role_uid_count": 0,
            "cross_role_source_group_count": 0,
        },
        "dev_gate_feasibility": dev_gate_feasibility,
        "implementation": _artifact(implementation),
        "implementation_audit": implementation_audit,
        "repo_root": str(repo_root),
        "commands": commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-teachers", action="append", required=True)
    parser.add_argument("--dev-teachers", action="append", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--extra-binding", action="append")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = freeze(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
