"""Bounded shared-template tuning for the v14 decoder instrument."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Callable, Mapping, Sequence

import numpy as np


TRACE_SCHEMA = "cr3-v14-gepa-trace-v1"
MAX_ROUNDS = 4
CANDIDATES_PER_ROUND = 8
BEAM_SIZE = 2


def template_sha256(template: str) -> str:
    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


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
    run_sha: str, n_dev: int = 8,
) -> list[dict]:
    """Select seven task-spanning metrics plus one deterministic additional metric."""
    certified = set(map(str, certified_metric_keys))
    eligible = [dict(row) for row in metrics if str(row["metric_key"]) not in certified]
    if len(eligible) < int(n_dev):
        raise ValueError("not enough metrics remain outside the certification population")
    by_task: dict[str, list[dict]] = {}
    for row in eligible:
        by_task.setdefault(str(row["task"]), []).append(row)
    if len(by_task) < 7:
        raise ValueError("v14 development metrics must span at least seven tasks")

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
    remaining = [row for row in eligible if row not in selected]
    largest_task = min(
        by_task,
        key=lambda task: (-sum(row in remaining for row in by_task[task]), task),
    )
    additional = min(
        [row for row in remaining if str(row["task"]) == largest_task] or remaining,
        key=lambda row: rank(row, "additional"),
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


def _mutation_prompt(
    *, channel: str, arm: str, incumbent: str, feedback: Sequence[Mapping[str, object]],
    mutation_index: int,
) -> str:
    compact_feedback = json.dumps(list(feedback), sort_keys=True, ensure_ascii=False)[:12000]
    return f"""Improve a shared decoder instruction for a measurement instrument.
Channel: {channel}
Arm: {arm}
Mutation index: {mutation_index}

The instruction must work across metrics, tasks, and Qwen/Llama/Mistral decoder families. It may
not mention any metric, answer, option, menu, or example content. Preserve every format placeholder
in the incumbent exactly. Improve evidence use and contrastive reasoning; do not add exemplars.

INCUMBENT:
{incumbent}

DEVELOPMENT FEEDBACK (paired high/low induced rules and scores from dev metrics only):
{compact_feedback}

Use the contrasts to improve evidence-routed decoding, but do not copy any induced rule or metric
content into the shared instruction. Return only the rewritten instruction template."""


def _propose_mutations(
    proposer, *, channel: str, arm: str, incumbent: str,
    feedback: Sequence[Mapping[str, object]], round_index: int, count: int,
) -> list[str]:
    prompts = [
        _mutation_prompt(
            channel=channel, arm=arm, incumbent=incumbent, feedback=feedback,
            mutation_index=index,
        )
        for index in range(int(count))
    ]
    outputs = proposer.generate_batch(
        prompts, system=None, max_tokens=900, temperature=0.7,
        seed=[stable_seed(channel, arm, round_index, index) for index in range(int(count))],
    )
    if len(outputs) != len(prompts):
        raise RuntimeError("GEPA proposer returned an incomplete mutation batch")
    return [str(value).strip() for value in outputs]


def propose_mutations(
    proposer, *, channel: str, arm: str, incumbent: str,
    feedback: Sequence[Mapping[str, object]], round_index: int, count: int = 7,
) -> list[str]:
    return _propose_mutations(
        proposer, channel=channel, arm=arm, incumbent=incumbent,
        feedback=feedback, round_index=round_index, count=count,
    )


def tune_shared_template(
    proposer, evaluator: Callable[[str], Mapping[str, object]], *,
    seed_template: str, channel: str, arm: str,
    forbidden_strings: Sequence[str], required_fields: Sequence[str],
    max_rounds: int = MAX_ROUNDS, candidates_per_round: int = CANDIDATES_PER_ROUND,
    beam_size: int = BEAM_SIZE, minimum_gain: float = 0.01,
    mcq_residual_threshold_bits: float = 0.02,
) -> dict:
    """Run the predeclared finite GEPA search and return one frozen winner."""
    if not 1 <= int(max_rounds) <= MAX_ROUNDS:
        raise ValueError("v14 GEPA is capped at four rounds")
    if int(candidates_per_round) != CANDIDATES_PER_ROUND or int(beam_size) != BEAM_SIZE:
        raise ValueError("v14 GEPA is frozen at eight candidates and beam size two")
    validate_shared_template(
        seed_template, forbidden_strings=forbidden_strings, required_fields=required_fields,
    )
    incumbent = str(seed_template)
    incumbent_report = dict(evaluator(incumbent))
    incumbent_fitness = float(incumbent_report["pooled_fitness"])
    incumbent_admissible = (
        np.isfinite(incumbent_fitness)
        and bool(incumbent_report.get("heldout_prompt_transfer_ok", False))
        and bool(incumbent_report.get("far_near_transfer_ok", True))
    )
    trace = [{
        "round": 0, "candidate": 0, "template_sha256": template_sha256(incumbent),
        "pooled_fitness": incumbent_fitness, "accepted": bool(incumbent_admissible),
        "report": incumbent_report,
    }]
    feedback = list(incumbent_report.get("feedback", []))
    stopping_reason = "maximum_rounds"
    for round_index in range(1, int(max_rounds) + 1):
        proposed = _propose_mutations(
            proposer, channel=channel, arm=arm, incumbent=incumbent,
            feedback=feedback, round_index=round_index,
            count=int(candidates_per_round) - 1,
        )
        candidates = [incumbent, *proposed]
        unique = []
        seen = set()
        for candidate in candidates:
            digest = template_sha256(candidate)
            if digest not in seen:
                unique.append(candidate)
                seen.add(digest)
        reports = []
        for candidate_index, candidate in enumerate(unique):
            try:
                validate_shared_template(
                    candidate, forbidden_strings=forbidden_strings,
                    required_fields=required_fields,
                )
                report = dict(evaluator(candidate))
                fitness = float(report["pooled_fitness"])
                transfer_ok = bool(report.get("heldout_prompt_transfer_ok", True))
                far_near_ok = bool(report.get("far_near_transfer_ok", True))
                admissible = np.isfinite(fitness) and transfer_ok and far_near_ok
                failure = None if admissible else "transfer_failure"
            except Exception as exc:
                report = {"error": str(exc)}
                fitness = float("-inf")
                admissible = False
                failure = "invalid_candidate"
            reports.append((fitness, template_sha256(candidate), candidate, report, admissible))
            trace.append({
                "round": round_index, "candidate": candidate_index,
                "template_sha256": template_sha256(candidate),
                "pooled_fitness": None if not np.isfinite(fitness) else fitness,
                "accepted": False, "failure": failure, "report": report,
            })
        admissible_rows = sorted(
            (row for row in reports if row[-1]), key=lambda row: (-row[0], row[1])
        )
        if not admissible_rows:
            stopping_reason = "transfer_failure"
            break
        beam = admissible_rows[:int(beam_size)]
        best_fitness, best_sha, best_template, best_report, _ = beam[0]
        gain = float(best_fitness - incumbent_fitness)
        for row in reversed(trace):
            if row["round"] == round_index and row["template_sha256"] == best_sha:
                row["accepted"] = True
                break
        if gain < float(minimum_gain) and incumbent_admissible:
            stopping_reason = "gain_below_0.01"
            break
        incumbent = best_template
        incumbent_report = best_report
        incumbent_fitness = float(best_fitness)
        incumbent_admissible = True
        feedback = list(best_report.get("feedback", []))
        if channel == "mcq":
            residual = best_report.get("dev_identification_residual_bits")
            if residual is not None and float(residual) < float(mcq_residual_threshold_bits):
                stopping_reason = "dev_identification_residual_below_0.02_bits"
                break
    if not incumbent_admissible:
        raise RuntimeError("no behavioral/MCQ decoder template passed held-out transfer")
    frozen = {
        "schema": TRACE_SCHEMA,
        "channel": str(channel),
        "arm": str(arm),
        "shared_across_decoder_families": True,
        "mechanical_model_specific_chat_formatting_allowed": True,
        "searched_per_family_variation_allowed": False,
        "seed_template": str(seed_template),
        "seed_template_sha256": template_sha256(seed_template),
        "winner_template": incumbent,
        "winner_template_sha256": template_sha256(incumbent),
        "winner_pooled_fitness": float(incumbent_fitness),
        "winner_report": incumbent_report,
        "stopping_reason": stopping_reason,
        "round_cap": int(max_rounds),
        "trace": trace,
    }
    frozen["freeze_sha256"] = hashlib.sha256(json.dumps(
        frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return frozen


def tune_shared_template_batched(
    propose_mutations: Callable[[str, Sequence[Mapping[str, object]], int, int], Sequence[str]],
    batch_evaluator: Callable[[Sequence[str]], Mapping[str, Mapping[str, object]]], *,
    seed_template: str, channel: str, arm: str, forbidden_strings: Sequence[str],
    required_fields: Sequence[str], max_rounds: int = MAX_ROUNDS,
    minimum_gain: float = 0.01, mcq_residual_threshold_bits: float = 0.02,
) -> dict:
    """Model-resident GEPA variant: all eight candidates are evaluated as one batch."""
    if not 1 <= int(max_rounds) <= MAX_ROUNDS:
        raise ValueError("v14 GEPA is capped at four rounds")
    validate_shared_template(
        seed_template, forbidden_strings=forbidden_strings, required_fields=required_fields,
    )
    initial = dict(batch_evaluator([str(seed_template)]))
    seed_sha = template_sha256(seed_template)
    if seed_sha not in initial:
        raise RuntimeError("batch evaluator omitted the seed template")
    incumbent = str(seed_template)
    incumbent_report = dict(initial[seed_sha])
    incumbent_fitness = float(incumbent_report["pooled_fitness"])
    incumbent_admissible = (
        np.isfinite(incumbent_fitness)
        and bool(incumbent_report.get("heldout_prompt_transfer_ok", False))
        and bool(incumbent_report.get("far_near_transfer_ok", True))
    )
    feedback = list(incumbent_report.get("feedback", []))
    trace = [{
        "round": 0, "candidate": 0, "template_sha256": seed_sha,
        "pooled_fitness": incumbent_fitness, "accepted": bool(incumbent_admissible),
        "report": incumbent_report,
    }]
    stopping_reason = "maximum_rounds"
    for round_index in range(1, int(max_rounds) + 1):
        mutations = list(propose_mutations(incumbent, feedback, round_index, 7))
        if len(mutations) != 7:
            raise RuntimeError("GEPA proposer must return exactly seven mutations")
        candidates = [incumbent, *map(str, mutations)]
        valid_candidates = []
        invalid_reasons = {}
        for candidate in candidates:
            try:
                validate_shared_template(
                    candidate, forbidden_strings=forbidden_strings,
                    required_fields=required_fields,
                )
                valid_candidates.append(candidate)
            except Exception as exc:
                invalid_reasons[template_sha256(candidate)] = str(exc)
        reports = dict(batch_evaluator(valid_candidates))
        scored = []
        for candidate_index, candidate in enumerate(candidates):
            digest = template_sha256(candidate)
            report = dict(reports.get(digest, {}))
            if digest in invalid_reasons:
                report = {"error": invalid_reasons[digest]}
            fitness = float(report.get("pooled_fitness", float("-inf")))
            admissible = (
                np.isfinite(fitness)
                and bool(report.get("heldout_prompt_transfer_ok", False))
                and bool(report.get("far_near_transfer_ok", True))
            )
            trace.append({
                "round": round_index, "candidate": candidate_index,
                "template_sha256": digest,
                "pooled_fitness": fitness if np.isfinite(fitness) else None,
                "accepted": False,
                "failure": None if admissible else "invalid_or_transfer_failure",
                "report": report,
            })
            if admissible:
                scored.append((fitness, digest, candidate, report))
        if not scored:
            stopping_reason = "transfer_failure"
            break
        # Top two are retained in the trace; the best is the next mutation parent.
        beam = sorted(scored, key=lambda row: (-row[0], row[1]))[:2]
        best_fitness, best_sha, best_template, best_report = beam[0]
        for row in reversed(trace):
            if row["round"] == round_index and row["template_sha256"] in {
                value[1] for value in beam
            }:
                row["accepted"] = True
        gain = float(best_fitness - incumbent_fitness)
        if gain < float(minimum_gain) and incumbent_admissible:
            stopping_reason = "gain_below_0.01"
            break
        incumbent = str(best_template)
        incumbent_report = dict(best_report)
        incumbent_fitness = float(best_fitness)
        incumbent_admissible = True
        feedback = list(best_report.get("feedback", []))
        if channel == "mcq":
            residual = best_report.get("dev_identification_residual_bits")
            if residual is not None and float(residual) < float(mcq_residual_threshold_bits):
                stopping_reason = "dev_identification_residual_below_0.02_bits"
                break
    if not incumbent_admissible:
        raise RuntimeError("no behavioral/MCQ decoder template passed held-out transfer")
    frozen = {
        "schema": TRACE_SCHEMA, "channel": str(channel), "arm": str(arm),
        "shared_across_decoder_families": True,
        "mechanical_model_specific_chat_formatting_allowed": True,
        "searched_per_family_variation_allowed": False,
        "seed_template": str(seed_template),
        "seed_template_sha256": seed_sha,
        "winner_template": incumbent,
        "winner_template_sha256": template_sha256(incumbent),
        "winner_pooled_fitness": incumbent_fitness,
        "winner_report": incumbent_report,
        "stopping_reason": stopping_reason,
        "round_cap": int(max_rounds), "trace": trace,
    }
    frozen["freeze_sha256"] = hashlib.sha256(json.dumps(
        frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return frozen
