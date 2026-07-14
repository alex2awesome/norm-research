"""Frozen Phase-C/Phase-E design helpers added by the v14.1 roadmap."""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np

from .v14_panel_design import canonical_sha256


DECODER_SCALE_MODELS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "Qwen/Qwen2.5-32B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
)
OMEGA_SIZES = (1, 2, 3, 5, 8)
OMEGA_COMPILERS = ("conjunction", "weighted_sum")


def _rank(run_sha: str, purpose: str, key: str) -> str:
    return hashlib.sha256(f"{run_sha}\x1f{purpose}\x1f{key}".encode()).hexdigest()


def exchangeable_r2_class(
    rows: Sequence[Mapping[str, object]], *, clone_cap: float,
) -> dict[str, list[dict]]:
    """Apply one eligibility rule to every possible C1 gold and distractor."""
    output: dict[str, list[dict]] = {}
    for source in rows:
        row = dict(source)
        if str(row.get("level")) != "R2":
            continue
        rate = float(row["positive_rate_on_teaching"])
        if not 0.3 <= rate <= 0.7:
            continue
        row["entropy_bin"] = int(np.floor(rate / 0.05 + 1e-12))
        row["clone_cap"] = float(clone_cap)
        output.setdefault(str(row["task"]), []).append(row)
    for task in output:
        output[task].sort(key=lambda row: str(row["metric_key"]))
    return output


def build_exchangeable_c1_menus(
    rows: Sequence[Mapping[str, object]], *, run_sha: str, menu_size: int = 11,
    clone_cap: float = 0.95,
) -> dict:
    classes = exchangeable_r2_class(rows, clone_cap=clone_cap)
    tasks, infeasible = {}, []
    for task, eligible in sorted(classes.items()):
        menus = []
        for gold in sorted(eligible, key=lambda row: _rank(run_sha, "c1-gold", row["metric_key"])):
            candidates = []
            for candidate in eligible:
                if candidate["metric_key"] == gold["metric_key"]:
                    continue
                if (gold.get("r3_ancestor") and candidate.get("r3_ancestor")
                        and gold["r3_ancestor"] == candidate["r3_ancestor"]):
                    continue
                similarity = float((gold.get("clone_similarity") or {}).get(
                    str(candidate["metric_key"]), 0.0
                ))
                if similarity >= float(clone_cap):
                    continue
                candidates.append(candidate)
            same_bin = [row for row in candidates if row["entropy_bin"] == gold["entropy_bin"]]
            remainder = [row for row in candidates if row not in same_bin]
            ordered = [
                *sorted(same_bin, key=lambda row: _rank(run_sha, gold["metric_key"], row["metric_key"])),
                *sorted(remainder, key=lambda row: _rank(run_sha, gold["metric_key"], row["metric_key"])),
            ]
            if len(ordered) < int(menu_size) - 1:
                infeasible.append(str(gold["metric_key"]))
                continue
            keys = [str(gold["metric_key"]), *[str(row["metric_key"]) for row in ordered[:menu_size - 1]]]
            menus.append({
                "gold_metric_key": str(gold["metric_key"]), "menu_metric_keys": keys,
                "gold_entropy_bin": int(gold["entropy_bin"]),
                "strict_entropy_bin_matches": int(min(len(same_bin), menu_size - 1)),
            })
        tasks[task] = {"class_size": len(eligible), "menus": menus}
    payload = {
        "schema": "cr3-v14-c1-exchangeable-r2-v1", "run_sha": str(run_sha),
        "eligibility": "R2 and positive_rate_on_teaching in [0.3,0.7]",
        "menu_size": int(menu_size), "clone_cap": float(clone_cap),
        "tasks": tasks, "infeasible_gold_metric_keys": sorted(infeasible),
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def compile_omega(criteria: Sequence[str], compiler: str) -> str:
    values = [str(value).strip() for value in criteria if str(value).strip()]
    if not values or compiler not in OMEGA_COMPILERS:
        raise ValueError("Omega compiler requires criteria and a declared compiler")
    bullets = "\n".join(f"- {value}" for value in values)
    if compiler == "weighted_sum":
        return (
            "Score the item in [0,1] as the fraction of these checks it satisfies "
            f"(each worth 1/{len(values)}; sum clamped to 1.0):\n{bullets}"
        )
    return f"Judge whether the item meets every one of these standards:\n{bullets}"


def build_nested_omega_design(
    eligible_rows: Sequence[Mapping[str, object]], *, run_sha: str,
    target_key: str,
) -> dict:
    candidates = [dict(row) for row in eligible_rows if str(row["metric_key"]) != str(target_key)]
    ordered = sorted(candidates, key=lambda row: _rank(run_sha, target_key, row["metric_key"]))
    if len(ordered) < max(OMEGA_SIZES):
        raise ValueError(f"{target_key} lacks eight eligible Omega units")
    rows = []
    for size in OMEGA_SIZES:
        selected = ordered[:size]
        for compiler in OMEGA_COMPILERS:
            rows.append({
                "omega_size": size, "compiler": compiler,
                "unit_metric_keys": [str(row["metric_key"]) for row in selected],
                "compiled_description": compile_omega(
                    [str(row["description"]) for row in selected], compiler,
                ),
            })
    payload = {
        "schema": "cr3-v14-omega-scaling-design-v1", "run_sha": str(run_sha),
        "target_key": str(target_key), "rows": rows,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload
