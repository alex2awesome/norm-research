"""Shared-template utilities for the v14 decoder instrument.

The in-house bounded GEPA search loop that used to live here (``tune_shared_template``,
``tune_shared_template_batched``, ``propose_mutations`` + private helpers) was DEPRECATED
2026-07-19 (user decision): official github GEPA (gepa 0.1.4) is the only sanctioned optimizer
for reconstruction experiments. Verbatim copies live in
``archive/inhouse_gepa_deprecated.py``; the public names remain here as shims that raise
``RuntimeError``. The live ``--phase tune`` path now calls ``gepa.optimize`` (see
``run_v14_value_campaign.run_decoder_tuning`` and ``official_gepa_decoder_tune.py``).

``template_sha256``, ``validate_shared_template``, ``select_dev_metrics`` and
``stratified_reference_states`` remain live utilities used by the official-GEPA path,
the tuning evaluator, and the campaign driver.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Mapping, Sequence

import numpy as np


_DEPRECATED_INHOUSE_GEPA = (
    "in-house GEPA loop deprecated 2026-07-19 — use official gepa "
    "(see feedback_official_gepa_only memory / plan file D1)"
)


def template_sha256(template: str) -> str:
    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()


def validate_shared_template(
    template: str, *, forbidden_strings: Sequence[str], required_fields: Sequence[str],
) -> None:
    text = str(template).strip()
    if not text:
        raise ValueError("decoder template is empty")
    missing = [field for field in required_fields if "{" + field + "}" not in text]
    if missing:
        raise ValueError(f"decoder template lacks required fields: {missing}")
    lowered = re.sub(r"\s+", " ", text.lower())
    leaks = sorted({
        value for value in map(str, forbidden_strings)
        if len(value.strip()) >= 4 and re.sub(r"\s+", " ", value.lower().strip()) in lowered
    })
    if leaks:
        raise ValueError(f"decoder template leaks frozen metric/menu content: {leaks[:3]}")


def select_dev_metrics(
    metrics: Sequence[Mapping[str, object]], *, certified_metric_keys: Sequence[str],
    run_sha: str, n_dev: int | None = 8, min_tasks: int = 7,
) -> list[dict]:
    """Select task-spanning metrics plus deterministic additional metrics.

    ``min_tasks`` defaults to the frozen seven-task requirement. Passing a lower
    floor is a DECLARED DEVIATION for populations where whole tasks are
    teaching-balance-infeasible (e.g. news-homepages/legal under the fixed 8B
    executor); callers must record the deviation in the run artifacts.
    """
    certified = set(map(str, certified_metric_keys))
    eligible = [dict(row) for row in metrics if str(row["metric_key"]) not in certified]
    if n_dev is None:
        n_dev = len(eligible)
    if len(eligible) < int(n_dev):
        raise ValueError("not enough metrics remain outside the certification population")
    by_task: dict[str, list[dict]] = {}
    for row in eligible:
        by_task.setdefault(str(row["task"]), []).append(row)
    if len(by_task) < int(min_tasks):
        raise ValueError(
            f"v14 development metrics must span at least {int(min_tasks)} tasks"
        )

    def rank(row: Mapping[str, object], salt: str) -> tuple:
        entropy = float(row.get("target_entropy_bits", 0.0))
        quintile = int(row.get("target_entropy_quintile", min(4, max(0, int(entropy * 5)))))
        digest = hashlib.sha256(json.dumps({
            "run_sha": run_sha, "salt": salt, "metric": str(row["metric_key"]),
        }, sort_keys=True).encode("utf-8")).hexdigest()
        return quintile, digest, str(row["metric_key"])

    selected = []
    task_order = sorted(
        by_task,
        key=lambda task: hashlib.sha256(f"{run_sha}\x1f{task}".encode()).hexdigest(),
    )
    desired_quintiles = [0, 1, 2, 3, 4, 1, 3]
    for position, task in enumerate(task_order[:7]):
        desired = desired_quintiles[position]
        choice = min(
            by_task[task],
            key=lambda row: (
                abs(int(row.get("target_entropy_quintile", 2)) - desired),
                rank(row, f"task-{task}"),
            ),
        )
        selected.append(choice)
    while len(selected) < int(n_dev):
        remaining = [row for row in eligible if row not in selected]
        if not remaining:
            raise ValueError("not enough eligible metrics to fill the development pool")
        largest_task = min(
            by_task,
            key=lambda task: (-sum(row in remaining for row in by_task[task]), task),
        )
        additional = min(
            [row for row in remaining if str(row["task"]) == largest_task] or remaining,
            key=lambda row: rank(row, f"additional-{len(selected)}"),
        )
        selected.append(additional)
    return selected[:int(n_dev)]


def stratified_reference_states(
    *, canonical_state: int, prompt_states: Sequence[int], prompt_values: Sequence[float],
    metric_key: str, trial: int, n_states: int = 6,
) -> dict:
    states = np.asarray(prompt_states, dtype=int)
    values = np.asarray(prompt_values, dtype=float)
    if states.shape != values.shape or states.ndim != 1 or len(states) < 3:
        raise ValueError("reference state inputs must be aligned and nontrivial")
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, 3)
    selected = [int(canonical_state)]
    for label, group in zip(("low", "mid", "high"), groups):
        ranked = sorted(
            map(int, group),
            key=lambda index: hashlib.sha256(
                f"{metric_key}\x1f{trial}\x1f{label}\x1f{index}".encode()
            ).hexdigest(),
        )
        for index in ranked:
            if int(states[index]) not in selected:
                selected.append(int(states[index]))
                break
    remaining = sorted(
        set(map(int, states)) - set(selected),
        key=lambda state: hashlib.sha256(
            f"{metric_key}\x1f{trial}\x1ftransfer\x1f{state}".encode()
        ).hexdigest(),
    )
    selected.extend(remaining[:max(0, int(n_states) - len(selected))])
    return {
        "search_states": selected[:4],
        "heldout_prompt_states": selected[4:int(n_states)],
        "canonical_state": int(canonical_state),
    }


def propose_mutations(*args, **kwargs):
    """DEPRECATED shim. The in-house GEPA reflective-mutation proposer was retired
    2026-07-19; the verbatim implementation lives in
    ``archive/inhouse_gepa_deprecated.py``. Official gepa's ``reflection_lm`` now
    performs mutation (see ``run_v14_value_campaign.run_decoder_tuning``)."""
    raise RuntimeError(_DEPRECATED_INHOUSE_GEPA)


def tune_shared_template(*args, **kwargs):
    """DEPRECATED shim. The in-house bounded GEPA search was retired 2026-07-19;
    the verbatim implementation lives in ``archive/inhouse_gepa_deprecated.py``.
    Use official ``gepa.optimize`` via ``run_v14_value_campaign.run_decoder_tuning``."""
    raise RuntimeError(_DEPRECATED_INHOUSE_GEPA)


def tune_shared_template_batched(*args, **kwargs):
    """DEPRECATED shim. The model-resident in-house GEPA search was retired
    2026-07-19; the verbatim implementation lives in
    ``archive/inhouse_gepa_deprecated.py``. Use official ``gepa.optimize`` via
    ``run_v14_value_campaign.run_decoder_tuning``."""
    raise RuntimeError(_DEPRECATED_INHOUSE_GEPA)
