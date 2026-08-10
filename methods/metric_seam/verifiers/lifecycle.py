"""Construct-valid, proposal-first lifecycle for metric-seam verifier units.

This module intentionally contains only the small amount of orchestration that
was missing from the verifier core.  It does not score metrics and it does not
compute implementation agreement.  Those operations are permitted only after
the proposal, corpus probe, and construct challenge recorded here have passed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
from typing import Mapping, Sequence

from .schema import Verdict


PROPOSAL_SCHEMA = "metric-seam.unit-proposal.v1"
BASE_RATE_SCHEMA = "metric-seam.pre-authoring-base-rate-probe.v1"
CHALLENGE_SCHEMA = "metric-seam.construct-challenge.v1"
SELECTION_SCHEMA = "metric-seam.construct-valid-node-selection.v1"
STATES = frozenset({"not_applicable", "satisfied", "violated"})
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")


@dataclass(frozen=True)
class UnitProposal:
    unit_id: str
    task: str
    criterion_id: str
    construct_text: str
    relation: str
    occasion: str
    satisfied_when: str
    violated_when: str
    required_context: str
    non_goals: tuple[str, ...]
    proxy_risks: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "unit_id", "task", "criterion_id", "construct_text", "relation",
            "occasion", "satisfied_when", "violated_when", "required_context",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not _SAFE_ID.fullmatch(self.unit_id):
            raise ValueError("unit_id must be a safe opaque identifier")
        for name in ("non_goals", "proxy_risks"):
            values = getattr(self, name)
            if not values or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain nonempty strings")

    def to_json_value(self) -> dict[str, object]:
        return {
            "schema": PROPOSAL_SCHEMA,
            "unit_id": self.unit_id,
            "task": self.task,
            "criterion_id": self.criterion_id,
            "construct_text": self.construct_text,
            "relation": self.relation,
            "occasion": self.occasion,
            "satisfied_when": self.satisfied_when,
            "violated_when": self.violated_when,
            "required_context": self.required_context,
            "non_goals": list(self.non_goals),
            "proxy_risks": list(self.proxy_risks),
            "authorship_constraints": {
                "detector_source_seen": False,
                "detector_outputs_seen": False,
                "heldout_seen": False,
            },
        }

    @classmethod
    def from_json_value(cls, value: Mapping[str, object]) -> "UnitProposal":
        if value.get("schema") != PROPOSAL_SCHEMA:
            raise ValueError("unsupported unit proposal schema")
        expected = {
            "schema", "unit_id", "task", "criterion_id", "construct_text",
            "relation", "occasion", "satisfied_when", "violated_when",
            "required_context", "non_goals", "proxy_risks", "authorship_constraints",
        }
        if set(value) != expected:
            raise ValueError("unit proposal fields drifted")
        constraints = value["authorship_constraints"]
        if constraints != {
            "detector_source_seen": False,
            "detector_outputs_seen": False,
            "heldout_seen": False,
        }:
            raise ValueError("proposal is not pre-authoring and TRAIN-only")
        return cls(
            unit_id=value["unit_id"], task=value["task"], criterion_id=value["criterion_id"],
            construct_text=value["construct_text"], relation=value["relation"],
            occasion=value["occasion"], satisfied_when=value["satisfied_when"],
            violated_when=value["violated_when"], required_context=value["required_context"],
            non_goals=tuple(value["non_goals"]), proxy_risks=tuple(value["proxy_risks"]),
        )


def stable_train_sample(
    rows: Sequence[Mapping[str, object]], *, salt: str, sample_size: int = 32
) -> tuple[Mapping[str, object], ...]:
    """Select a deterministic TRAIN sample without consulting outcomes."""

    if not salt or sample_size < 1:
        raise ValueError("a nonempty salt and positive sample size are required")
    keys: set[str] = set()
    ranked: list[tuple[str, Mapping[str, object]]] = []
    for row in rows:
        item_key = row.get("item_key")
        if not isinstance(item_key, str) or not item_key or item_key in keys:
            raise ValueError("TRAIN rows require unique nonempty item_key values")
        keys.add(item_key)
        digest = hashlib.sha256(f"{salt}\0{item_key}".encode()).hexdigest()
        ranked.append((digest, row))
    ranked.sort(key=lambda value: (value[0], str(value[1]["item_key"])))
    return tuple(row for _, row in ranked[:sample_size])


def evaluate_pre_authoring_probe(
    proposal: UnitProposal,
    sample_item_ids: Sequence[str],
    verdicts: Mapping[str, Verdict | None],
    *,
    min_applies: int = 6,
    min_each_polarity: int = 2,
    occurrence_rate_bounds: tuple[float, float] = (0.10, 0.90),
) -> dict[str, object]:
    """Apply the cheap corpus-support stop before a detector may be authored."""

    if len(sample_item_ids) != len(set(sample_item_ids)) or not sample_item_ids:
        raise ValueError("sample_item_ids must be nonempty and unique")
    if set(verdicts) != set(sample_item_ids):
        raise ValueError("probe verdicts must exactly cover the frozen sample")
    counts = Counter(
        "execution_error" if verdicts[item_id] is None else verdicts[item_id].state
        for item_id in sample_item_ids
    )
    applies = counts["satisfied"] + counts["violated"]
    occurrence_rate = counts["violated"] / len(sample_item_ids)
    lower, upper = occurrence_rate_bounds
    if not 0 <= lower < upper <= 1:
        raise ValueError("occurrence-rate bounds must satisfy 0 <= lower < upper <= 1")
    failures: list[str] = []
    if counts["execution_error"]:
        failures.append("probe_execution_error")
    if applies < min_applies:
        failures.append("too_few_applicable_items")
    if counts["satisfied"] < min_each_polarity:
        failures.append("too_few_satisfied_items")
    if counts["violated"] < min_each_polarity:
        failures.append("too_few_violated_items")
    if not lower <= occurrence_rate <= upper:
        failures.append("occurrence_rate_out_of_bounds")
    passed = not failures
    return {
        "schema": BASE_RATE_SCHEMA,
        "unit_id": proposal.unit_id,
        "status": "corpus_supported_pre_authorship" if passed else "corpus_unsupported_pre_authorship",
        "passed": passed,
        "sample_item_ids": list(sample_item_ids),
        "sample_size": len(sample_item_ids),
        "state_counts": {state: counts[state] for state in sorted(STATES)},
        "execution_errors": counts["execution_error"],
        "occurrence_rate": occurrence_rate,
        "thresholds": {"min_applies": min_applies, "min_each_polarity": min_each_polarity,
                       "occurrence_rate_lower": lower, "occurrence_rate_upper": upper},
        "failed_checks": failures,
        "detector_authorship_permitted": passed,
    }


@dataclass(frozen=True)
class ConstructControl:
    control_id: str
    ctext: str
    expected_state: str
    proxy_triggered: bool
    rationale: str

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.control_id):
            raise ValueError("control_id must be a safe opaque identifier")
        if not self.ctext or self.expected_state not in {"satisfied", "violated"}:
            raise ValueError("controls require text and an applicable expected state")
        if type(self.proxy_triggered) is not bool or not self.rationale:
            raise ValueError("controls require a boolean proxy flag and rationale")


def evaluate_construct_challenge(
    proposal: UnitProposal,
    controls: Sequence[ConstructControl],
    judge_states: Mapping[str, Sequence[str]],
    verifier_verdicts: Mapping[str, Verdict],
    *,
    minimum_per_critical_quadrant: int = 4,
) -> dict[str, object]:
    """Falsify a proxy with crossed construct/proxy controls.

    The two critical quadrants are proxy-on/construct-satisfied and
    proxy-off/construct-violated.  Two blinded construct judgments must agree
    with the predeclared state; the verifier must then classify that state.
    """

    ids = [control.control_id for control in controls]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("construct controls must be nonempty and unique")
    if set(judge_states) != set(ids) or set(verifier_verdicts) != set(ids):
        raise ValueError("judges and verifier must exactly cover the controls")
    proxy_on_satisfied = sum(c.proxy_triggered and c.expected_state == "satisfied" for c in controls)
    proxy_off_violated = sum((not c.proxy_triggered) and c.expected_state == "violated" for c in controls)
    coverage_passed = (
        proxy_on_satisfied >= minimum_per_critical_quadrant
        and proxy_off_violated >= minimum_per_critical_quadrant
    )
    rows: list[dict[str, object]] = []
    for control in controls:
        states = list(judge_states[control.control_id])
        judge_passed = len(states) == 2 and all(
            state == control.expected_state for state in states
        )
        verifier_state = verifier_verdicts[control.control_id].state
        verifier_passed = verifier_state == control.expected_state
        rows.append({
            "control_id": control.control_id,
            "expected_state": control.expected_state,
            "proxy_triggered": control.proxy_triggered,
            "judge_states": states,
            "judge_passed": judge_passed,
            "verifier_state": verifier_state,
            "verifier_passed": verifier_passed,
            "passed": judge_passed and verifier_passed,
        })
    passed = coverage_passed and all(row["passed"] for row in rows)
    return {
        "schema": CHALLENGE_SCHEMA,
        "unit_id": proposal.unit_id,
        "status": "construct_challenge_passed" if passed else "construct_challenge_failed",
        "passed": passed,
        "critical_quadrant_counts": {
            "proxy_on_construct_satisfied": proxy_on_satisfied,
            "proxy_off_construct_violated": proxy_off_violated,
        },
        "coverage_passed": coverage_passed,
        "controls": rows,
    }


def select_after_construct_gate(
    proposal: UnitProposal,
    *,
    base_rate_probe: Mapping[str, object],
    natural_train_gate: Mapping[str, object],
    construct_challenge: Mapping[str, object],
    authorship_rounds: int,
) -> dict[str, object]:
    """Make the only selection decision licensed before freezing/transcription."""

    if not 1 <= authorship_rounds <= 2:
        raise ValueError("authorship is capped at two rounds")
    checks = {
        "base_rate_probe": base_rate_probe.get("passed") is True,
        "natural_train_gate": natural_train_gate.get("passed") is True,
        "construct_challenge": construct_challenge.get("passed") is True,
    }
    selected = all(checks.values())
    return {
        "schema": SELECTION_SCHEMA,
        "unit_id": proposal.unit_id,
        "status": "selected_for_freeze" if selected else "rejected_before_freeze",
        "selected": selected,
        "checks": checks,
        "authorship_rounds": authorship_rounds,
        "agreement_computed": False,
        "prompt_transcription_permitted": selected,
    }


def select_validity_v2_before_freeze(
    proposal: UnitProposal,
    *,
    base_rate_probe: Mapping[str, object],
    natural_train_gate: Mapping[str, object],
    construct_challenge: Mapping[str, object],
    node_gate: Mapping[str, object],
    authorship_rounds: int,
) -> dict[str, object]:
    """Validity-v2 selection with a mandatory pre-freeze per-node gate.

    Historical callers retain ``select_after_construct_gate`` for exact replay;
    new pipelines must use this entry point and cannot omit the Shapley/applies
    decision produced before transcription.
    """
    legacy = select_after_construct_gate(
        proposal,
        base_rate_probe=base_rate_probe,
        natural_train_gate=natural_train_gate,
        construct_challenge=construct_challenge,
        authorship_rounds=authorship_rounds,
    )
    rows = node_gate.get("nodes")
    node_gate_valid = (
        isinstance(rows, list)
        and bool(rows)
        and all(row.get("decision") in {"KEEP", "DELETE"} for row in rows)
        and any(row.get("decision") == "KEEP" for row in rows)
    )
    checks = {**legacy["checks"], "per_node_gate": node_gate_valid}
    selected = all(checks.values())
    return {
        **legacy,
        "schema": "metric-seam.construct-valid-node-selection.v2",
        "status": "selected_for_freeze" if selected else "rejected_before_freeze",
        "selected": selected,
        "checks": checks,
        "per_node_gate_summary": {
            "n_nodes": len(rows) if isinstance(rows, list) else 0,
            "n_keep": (sum(row.get("decision") == "KEEP" for row in rows)
                       if isinstance(rows, list) else 0),
            "n_delete": (sum(row.get("decision") == "DELETE" for row in rows)
                         if isinstance(rows, list) else 0),
        },
        "prompt_transcription_permitted": selected,
    }
