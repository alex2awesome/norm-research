"""Small, executable primitives for the 2026-07-14 seam-validity roadmap.

This module deliberately does not author verifiers or choose constructs.  It
implements the two frozen selection rules around authorship and the node-type
capture--recapture readout.  Inputs and outputs are plain JSON-compatible data
so an experimental manifest can preserve the entire decision trail.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence


BASE_RATE_LO = 0.10
BASE_RATE_HI = 0.90
DEFAULT_PHI_EPSILON = 0.01
DEFAULT_MIN_APPLIES_RATE = 0.10


def base_rate_probe(
    occurrences: Iterable[bool],
    *,
    lower: float = BASE_RATE_LO,
    upper: float = BASE_RATE_HI,
) -> dict:
    """Decide whether a proposed detector is worth authoring.

    The probe observes only the proposed relation's occurrence on a frozen
    pre-authoring sample.  It never consumes criterion scores or held-out data.
    """
    observations = [bool(x) for x in occurrences]
    if not observations:
        raise ValueError("base-rate probe requires at least one observation")
    rate = sum(observations) / len(observations)
    passed = lower <= rate <= upper
    return {
        "n": len(observations),
        "n_occurs": sum(observations),
        "occurrence_rate": rate,
        "lower_inclusive": lower,
        "upper_inclusive": upper,
        "decision": "AUTHOR" if passed else "KILL_BEFORE_AUTHORING",
        "passed": passed,
    }


def per_node_gate(
    nodes: Iterable[Mapping],
    *,
    phi_epsilon: float = DEFAULT_PHI_EPSILON,
    min_applies_rate: float = DEFAULT_MIN_APPLIES_RATE,
) -> dict:
    """Apply the frozen post-authoring deletion rule to attributed nodes."""
    rows = []
    for node in nodes:
        phi = float(node["phi"])
        applies = float(node["applies_rate"])
        delete = abs(phi) < phi_epsilon and applies < min_applies_rate
        rows.append({
            **dict(node),
            "decision": "DELETE" if delete else "KEEP",
            "reason": (
                "low attribution and low natural applicability" if delete
                else "survives the conjunctive deletion rule"
            ),
        })
    return {
        "rule": "DELETE iff abs(phi) < phi_epsilon AND applies_rate < min_applies_rate",
        "phi_epsilon": phi_epsilon,
        "min_applies_rate": min_applies_rate,
        "n_nodes": len(rows),
        "n_delete": sum(r["decision"] == "DELETE" for r in rows),
        "nodes": rows,
    }


@dataclass(frozen=True, order=True)
class NodeType:
    """The roadmap's capture--recapture unit, not a source-code node id."""

    op_class: str
    witness_kind: str
    relation: str

    @classmethod
    def from_mapping(cls, row: Mapping) -> "NodeType":
        return cls(
            op_class=str(row["op_class"]).strip().lower(),
            witness_kind=str(row["witness_kind"]).strip().lower(),
            relation=" ".join(str(row["relation"]).strip().lower().split()),
        )

    def as_dict(self) -> dict:
        return {
            "op_class": self.op_class,
            "witness_kind": self.witness_kind,
            "relation": self.relation,
        }


def capture_recapture_node_types(fleets: Sequence[Iterable[Mapping]]) -> dict:
    """Estimate unseen node-type mass across K independently authored fleets.

    Uses the incidence-based, bias-corrected Chao2 estimator.  This is suitable
    for K>=2 presence/absence samples and does not pretend fleet node counts are
    independent draws.  The roadmap requires sign-off on K before any fleet is
    commissioned; this function therefore records K but does not choose it.
    """
    if len(fleets) < 2:
        raise ValueError("capture--recapture requires at least two blind fleets")
    normalized = [{NodeType.from_mapping(row) for row in fleet} for fleet in fleets]
    incidence = Counter(node for fleet in normalized for node in fleet)
    observed = len(incidence)
    q1 = sum(freq == 1 for freq in incidence.values())
    q2 = sum(freq == 2 for freq in incidence.values())
    k = len(normalized)
    # Bias-corrected Chao2 remains finite when Q2=0.
    unseen = ((k - 1) / k) * (q1 * (q1 - 1)) / (2 * (q2 + 1))
    estimated_total = observed + unseen
    return {
        "unit": "(op_class, witness_kind, relation)",
        "estimator": "bias-corrected incidence Chao2",
        "k_fleets": k,
        "fleet_sizes_unique_types": [len(fleet) for fleet in normalized],
        "observed_unique_types": observed,
        "q1_seen_in_exactly_one_fleet": q1,
        "q2_seen_in_exactly_two_fleets": q2,
        "estimated_unseen_types": unseen,
        "estimated_total_types": estimated_total,
        "estimated_coverage": (observed / estimated_total if estimated_total else 1.0),
        "incidence": [
            {**node.as_dict(), "n_fleets": incidence[node]}
            for node in sorted(incidence)
        ],
        "scope": "width of independently proposed node types; does not validate constructs or move levels",
    }


def validate_lifecycle(events: Sequence[Mapping]) -> dict:
    """Check the only licensed verifier-building order.

    The function is intentionally an audit readout rather than an orchestration
    framework: it reports the first missing/out-of-order stage and changes no
    files or models.
    """
    required = [
        "PROPOSE", "BASE_RATE_PROBE", "AUTHOR", "PER_NODE_GATE",
        "SELECT", "FREEZE", "TRANSCRIBE",
    ]
    observed = [str(e.get("stage")) for e in events]
    cursor = 0
    violations = []
    for stage in observed:
        if cursor < len(required) and stage == required[cursor]:
            cursor += 1
        elif stage in required[cursor + 1:]:
            violations.append({"stage": stage, "expected": required[cursor]})
    return {
        "required_order": required,
        "observed": observed,
        "complete": cursor == len(required) and not violations,
        "next_required": None if cursor == len(required) else required[cursor],
        "violations": violations,
    }
