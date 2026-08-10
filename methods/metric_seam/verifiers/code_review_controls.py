"""Frozen same-corpus capability controls for the code-review verifier pilot.

These controls are co-run with real units.  They test whether the selected
verifier class can recover two deliberately observable structured relations and
decline a label that is not a function of the item.  They do not license an
unqualified tacitness claim for real units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .code_review_mutations import swallowed_error_go


@dataclass(frozen=True)
class ControlSpec:
    control_id: str
    construct: str
    relation: str
    expected: Literal["certify", "do_not_certify", "directional_only"]
    mutation_kind: str | None
    extension: str | None
    planted_source: tuple[str, ...]
    decision_rule: str


def _deep_nesting_go() -> tuple[str, ...]:
    return (
        "package internal",
        "",
        "func route(a, b, c bool) int {",
        "\tif a {",
        "\t\tif b {",
        "\t\t\tif c {",
        "\t\t\t\treturn 1",
        "\t\t\t}",
        "\t\t}",
        "\t}",
        "\treturn 0",
        "}",
    )


CONTROLS = (
    ControlSpec(
        control_id="pcr901",
        construct="Control-flow nesting",
        relation="An added function contains a conditional nested inside two other conditionals.",
        expected="certify",
        mutation_kind="three_deep_conditional",
        extension="go",
        planted_source=_deep_nesting_go(),
        decision_rule="structured syntax-tree ancestry over added code",
    ),
    ControlSpec(
        control_id="pcr902",
        construct="Swallowed error path",
        relation="An added error branch discards a non-nil error and returns success.",
        expected="certify",
        mutation_kind="swallowed_error",
        extension="go",
        planted_source=swallowed_error_go(),
        decision_rule="structured error-branch and return-value analysis over added code",
    ),
    ControlSpec(
        control_id="pcr906",
        construct="Concealed maintenance allocation",
        relation=(
            "The item has a positive allocation stored only in a sealed random registry "
            "that is not a function of the item text."
        ),
        expected="do_not_certify",
        mutation_kind=None,
        extension=None,
        planted_source=(),
        decision_rule="no input-only verifier can recover the sealed allocation",
    ),
    ControlSpec(
        control_id="pcr905",
        construct="Architectural elegance in project context",
        relation=(
            "The change is an elegant architectural choice given unstated project history, "
            "requirements, and alternatives."
        ),
        expected="directional_only",
        mutation_kind=None,
        extension=None,
        planted_source=(),
        decision_rule="directional holistic control; no hard non-certification claim",
    ),
)


def controls_by_id() -> dict[str, ControlSpec]:
    return {control.control_id: control for control in CONTROLS}
