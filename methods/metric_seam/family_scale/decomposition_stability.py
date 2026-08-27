"""Freeze and compare independent metric-text-only DAG decompositions.

The instrument consumes exactly one metric name/text pair and two or three
fleet outputs.  It never accepts item text, corpus paths, scores, labels, or
retrieval material.  Matching is deliberately mechanical: canonical exact
relation identity and a coarser operation/witness-type identity are reported
separately.  Capture-recapture quantities are descriptive diagnostics, not
population estimates licensed by an independence assumption.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Any, Iterable, Literal, Mapping, Sequence


SCHEMA = "metric-seam.metric-text-decomposition-submission.v1"
REPORT_SCHEMA = "metric-seam.metric-text-decomposition-stability.v1"
OP_CLASSES = frozenset({"computation", "evidence", "individuation"})
_HYPHENS = str.maketrans({
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "﹘": "-",
    "﹣": "-",
    "－": "-",
})


class DecompositionSchemaError(ValueError):
    """Raised when a metric-text-only submission violates its contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _exact_keys(value: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise DecompositionSchemaError(
            f"{path}: key mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _phrase(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise DecompositionSchemaError(f"{path}: expected string")
    normalized = unicodedata.normalize("NFKC", value).translate(_HYPHENS)
    normalized = " ".join(normalized.casefold().split()).strip(" .;:,")
    if not normalized:
        raise DecompositionSchemaError(f"{path}: phrase is empty after normalization")
    if any(ord(character) < 32 for character in normalized):
        raise DecompositionSchemaError(f"{path}: control characters are forbidden")
    return normalized


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DecompositionSchemaError(f"{path}: expected nonempty string")
    if any(ord(character) < 32 and character not in "\n\t" for character in value):
        raise DecompositionSchemaError(f"{path}: forbidden control character")
    return value


@dataclass(frozen=True, order=True)
class CanonicalRelation:
    """Canonical executable DAG relation type proposed from metric text."""

    op_class: Literal["computation", "evidence", "individuation"]
    witness_kind: str
    relation: str

    def __post_init__(self) -> None:
        if self.op_class not in OP_CLASSES:
            raise DecompositionSchemaError(
                f"op_class must be one of {sorted(OP_CLASSES)}"
            )
        if self.witness_kind != _phrase(self.witness_kind, "witness_kind"):
            raise DecompositionSchemaError("witness_kind is not canonical")
        if self.relation != _phrase(self.relation, "relation"):
            raise DecompositionSchemaError("relation is not canonical")

    @classmethod
    def from_value(cls, value: object, path: str = "relation") -> "CanonicalRelation":
        if not isinstance(value, dict):
            raise DecompositionSchemaError(f"{path}: expected object")
        _exact_keys(value, {"op_class", "witness_kind", "relation"}, path)
        op_class = value["op_class"]
        if not isinstance(op_class, str) or op_class not in OP_CLASSES:
            raise DecompositionSchemaError(
                f"{path}.op_class: expected one of {sorted(OP_CLASSES)}"
            )
        return cls(
            op_class=op_class,
            witness_kind=_phrase(value["witness_kind"], f"{path}.witness_kind"),
            relation=_phrase(value["relation"], f"{path}.relation"),
        )

    def to_value(self) -> dict[str, str]:
        return {
            "op_class": self.op_class,
            "witness_kind": self.witness_kind,
            "relation": self.relation,
        }

    @property
    def relation_id(self) -> str:
        return "rel_" + _sha256(self.to_value())[:20]

    @property
    def type_key(self) -> tuple[str, str]:
        return self.op_class, self.witness_kind


@dataclass(frozen=True)
class FleetDecomposition:
    fleet_id: str
    relations: tuple[CanonicalRelation, ...]
    submitted_order_sha256: str

    @classmethod
    def from_value(cls, value: object, path: str = "fleet") -> "FleetDecomposition":
        if not isinstance(value, dict):
            raise DecompositionSchemaError(f"{path}: expected object")
        _exact_keys(value, {"fleet_id", "relations"}, path)
        fleet_id = _phrase(value["fleet_id"], f"{path}.fleet_id")
        rows = value["relations"]
        if not isinstance(rows, list) or not rows:
            raise DecompositionSchemaError(f"{path}.relations: expected nonempty array")
        relations = tuple(
            CanonicalRelation.from_value(row, f"{path}.relations[{index}]")
            for index, row in enumerate(rows)
        )
        if len(set(relations)) != len(relations):
            raise DecompositionSchemaError(
                f"{path}.relations: duplicate after canonical normalization"
            )
        submitted = [relation.to_value() for relation in relations]
        return cls(fleet_id, relations, _sha256(submitted))

    @property
    def relation_set(self) -> frozenset[CanonicalRelation]:
        return frozenset(self.relations)

    @property
    def type_set(self) -> frozenset[tuple[str, str]]:
        return frozenset(relation.type_key for relation in self.relations)

    @property
    def set_sha256(self) -> str:
        return _sha256(
            [
                relation.to_value()
                for relation in sorted(self.relation_set)
            ]
        )


def _ratio(numerator: int, denominator: int) -> dict[str, object] | None:
    if denominator == 0:
        return None
    value = Fraction(numerator, denominator)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": round(float(value), 6),
    }


def _fraction_value(value: Fraction) -> dict[str, object]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": round(float(value), 6),
    }


def _pairwise(
    left: FleetDecomposition, right: FleetDecomposition
) -> dict[str, object]:
    left_set = left.relation_set
    right_set = right.relation_set
    exact_overlap = len(left_set & right_set)
    exact_union = len(left_set | right_set)
    left_types = left.type_set
    right_types = right.type_set
    type_overlap = len(left_types & right_types)
    type_union = len(left_types | right_types)
    # Chapman two-sample estimator.  It remains finite at zero overlap, but its
    # assumptions are explicitly not asserted by this descriptive instrument.
    chapman = (
        Fraction((len(left_set) + 1) * (len(right_set) + 1), exact_overlap + 1)
        - 1
    )
    return {
        "fleet_a": left.fleet_id,
        "fleet_b": right.fleet_id,
        "n_a": len(left_set),
        "n_b": len(right_set),
        "exact_overlap": exact_overlap,
        "exact_union": exact_union,
        "exact_jaccard": _ratio(exact_overlap, exact_union),
        "exact_overlap_coefficient": _ratio(
            exact_overlap, min(len(left_set), len(right_set))
        ),
        "type_overlap": type_overlap,
        "type_union": type_union,
        "type_jaccard": _ratio(type_overlap, type_union),
        "chapman_descriptive_total": _fraction_value(chapman),
    }


def _mean_ratio(rows: Sequence[dict[str, object]], field: str) -> dict[str, object] | None:
    ratios: list[Fraction] = []
    for row in rows:
        value = row[field]
        if isinstance(value, dict):
            ratios.append(Fraction(value["numerator"], value["denominator"]))
    if not ratios:
        return None
    return _fraction_value(sum(ratios, Fraction()) / len(ratios))


def build_stability_report(
    *,
    metric_name: str,
    metric_text: str,
    fleets: Sequence[FleetDecomposition],
    submission_sha256: str | None = None,
) -> dict[str, object]:
    """Build a deterministic descriptive stability/capture report."""

    metric_name = _nonempty_text(metric_name, "metric.name")
    metric_text = _nonempty_text(metric_text, "metric.text")
    if not 2 <= len(fleets) <= 3:
        raise DecompositionSchemaError("exactly two or three fleets are required")
    fleet_ids = [fleet.fleet_id for fleet in fleets]
    if len(set(fleet_ids)) != len(fleet_ids):
        raise DecompositionSchemaError("fleet_id values must be unique")
    if not all(isinstance(fleet, FleetDecomposition) for fleet in fleets):
        raise TypeError("fleets must contain FleetDecomposition values")

    ordered_fleets = tuple(sorted(fleets, key=lambda fleet: fleet.fleet_id))
    all_relations = sorted(
        set().union(*(fleet.relation_set for fleet in ordered_fleets))
    )
    captured = {
        relation: tuple(
            fleet.fleet_id
            for fleet in ordered_fleets
            if relation in fleet.relation_set
        )
        for relation in all_relations
    }
    frequency = {
        str(count): sum(len(fleet_ids) == count for fleet_ids in captured.values())
        for count in range(1, len(ordered_fleets) + 1)
    }
    all_intersection = set.intersection(
        *(set(fleet.relation_set) for fleet in ordered_fleets)
    )
    pairs = [
        _pairwise(ordered_fleets[left], ordered_fleets[right])
        for left in range(len(ordered_fleets))
        for right in range(left + 1, len(ordered_fleets))
    ]

    capture_recapture: dict[str, object]
    if len(ordered_fleets) == 2:
        capture_recapture = {
            "method": "two_sample_chapman_descriptive",
            "estimate": pairs[0]["chapman_descriptive_total"],
            "identified": True,
        }
    else:
        singleton_count = frequency["1"]
        doubleton_count = frequency["2"]
        if doubleton_count:
            # Incidence Chao2 with three sampling units:
            # S_obs + ((T-1)/T) * Q1^2/(2 Q2) = S_obs + Q1^2/(3 Q2).
            estimate = Fraction(len(all_relations), 1) + Fraction(
                singleton_count * singleton_count, 3 * doubleton_count
            )
            capture_recapture = {
                "method": "three_fleet_incidence_chao2_descriptive",
                "estimate": _fraction_value(estimate),
                "identified": True,
            }
        else:
            capture_recapture = {
                "method": "three_fleet_incidence_chao2_descriptive",
                "estimate": None,
                "identified": False,
                "reason": "no relation was captured by exactly two fleets",
            }

    metric_value = {"name": metric_name, "text": metric_text}
    freeze_core = {
        "metric_sha256": _sha256(metric_value),
        "fleets": [
            {
                "fleet_id": fleet.fleet_id,
                "submitted_order_sha256": fleet.submitted_order_sha256,
                "canonical_set_sha256": fleet.set_sha256,
            }
            for fleet in ordered_fleets
        ],
    }
    if submission_sha256 is not None:
        freeze_core["submission_sha256"] = submission_sha256
    freeze_id = _sha256(freeze_core)

    return {
        "schema": REPORT_SCHEMA,
        "status": "frozen_descriptive_comparison_complete",
        "input_scope": "metric_text_only",
        "metric": metric_value,
        "fleets": [
            {
                "fleet_id": fleet.fleet_id,
                "relation_count": len(fleet.relation_set),
                "submitted_order_sha256": fleet.submitted_order_sha256,
                "canonical_set_sha256": fleet.set_sha256,
                "relation_ids": sorted(
                    relation.relation_id for relation in fleet.relation_set
                ),
            }
            for fleet in ordered_fleets
        ],
        "canonical_relations": [
            {
                "relation_id": relation.relation_id,
                **relation.to_value(),
                "captured_by": list(captured[relation]),
                "capture_count": len(captured[relation]),
            }
            for relation in all_relations
        ],
        "stability": {
            "fleet_count": len(ordered_fleets),
            "observed_relation_union": len(all_relations),
            "all_fleet_exact_intersection": len(all_intersection),
            "capture_frequency": frequency,
            "recaptured_relation_count": sum(
                len(fleet_ids) >= 2 for fleet_ids in captured.values()
            ),
            "pairwise": pairs,
            "mean_pairwise_exact_jaccard": _mean_ratio(pairs, "exact_jaccard"),
            "mean_pairwise_type_jaccard": _mean_ratio(pairs, "type_jaccard"),
            "capture_recapture": capture_recapture,
        },
        "freeze": {
            "freeze_id": freeze_id,
            **freeze_core,
            "corpus_accessed": False,
            "scores_or_labels_accessed": False,
            "semantic_adjudication_performed": False,
        },
        "claim_limits": [
            "Canonical exact matching is lexical after declared normalization, not semantic equivalence.",
            "Operation/witness-type matching is coarser and does not establish relation identity.",
            "Capture-recapture estimates are descriptive because fleet independence and equal capture are not assumed.",
            "No result establishes executable correctness, corpus applicability, verifiability, or tacitness.",
        ],
    }


def load_submission(value: object) -> dict[str, object]:
    """Validate one closed-world submission and build its stability report."""

    if not isinstance(value, dict):
        raise DecompositionSchemaError("submission: expected object")
    _exact_keys(value, {"schema", "metric", "fleets"}, "submission")
    if value["schema"] != SCHEMA:
        raise DecompositionSchemaError(f"submission.schema: expected {SCHEMA!r}")
    metric = value["metric"]
    if not isinstance(metric, dict):
        raise DecompositionSchemaError("submission.metric: expected object")
    _exact_keys(metric, {"name", "text"}, "submission.metric")
    rows = value["fleets"]
    if not isinstance(rows, list):
        raise DecompositionSchemaError("submission.fleets: expected array")
    fleets = tuple(
        FleetDecomposition.from_value(row, f"submission.fleets[{index}]")
        for index, row in enumerate(rows)
    )
    return build_stability_report(
        metric_name=metric["name"],
        metric_text=metric["text"],
        fleets=fleets,
        submission_sha256=_sha256(value),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two or three metric-text-only decomposition fleets."
    )
    parser.add_argument("input", type=Path, help="closed-world submission JSON")
    parser.add_argument("--output", type=Path, help="write report JSON instead of stdout")
    args = parser.parse_args()
    submission = json.loads(args.input.read_text())
    report = load_submission(submission)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
