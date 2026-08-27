"""Verifier-native adapter for the existing Math-a12 symbolic capability.

The measurement unit here is one *presented equality pair*, not a whole
document and not the parent rigor criterion.  The deterministic verifier
parses both sides as bounded rational expressions and returns one of the
shared three states:

* not applicable: the bounded parser cannot decide the pair;
* satisfied: the two expressions are exactly identical on the emitted
  denominator-nonzero domain; or
* violated: the two expressions are exactly non-identical.

``violated`` is therefore a relation-instance label.  It is not, by itself,
evidence that the surrounding document made a false universal claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Sequence

from methods.metric_seam.hybrids.ops_symbolic_steps_v1 import (
    MAX_PAIR_CANDIDATES,
    MathOps,
    _ANSWER_SPLIT_RE,
    _equation_rows,
)
from methods.metric_seam.hybrids.ops_symbolic_steps_v2 import (
    verify_expression_pair,
)

from .schema import Span, Verdict


RELATION_ID = "explicit_rational_equality_preservation"
SOURCE_PATH = "answer.md"
_SAFE_ITEM_KEY = re.compile(r"[A-Za-z0-9_.-]+\Z")


@dataclass(frozen=True)
class EqualityPair:
    """One bounded adjacent equality pair with its source-local witness."""

    item_key: str
    pair_id: str
    lhs: str
    rhs: str
    witness: Span

    def __post_init__(self) -> None:
        if not self.item_key or not _SAFE_ITEM_KEY.fullmatch(self.item_key):
            raise ValueError("item_key must be a safe opaque identifier")
        if not self.pair_id or not _SAFE_ITEM_KEY.fullmatch(self.pair_id):
            raise ValueError("pair_id must be a safe opaque identifier")
        if not self.lhs or not self.rhs:
            raise ValueError("equality sides must be nonempty")
        if self.witness.node_id != self.pair_id:
            raise ValueError("witness node_id must bind the pair_id")

    def to_request_value(self) -> dict[str, object]:
        """Return the no-float record shared with the independent LLM arm."""

        return {
            "item_key": self.item_key,
            "pair_id": self.pair_id,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "source_span": self.witness.to_json_value(),
        }


def _answer_with_offset(text: str) -> tuple[str, int]:
    """Return the judged answer segment and its character offset in ``text``."""

    matches = list(_ANSWER_SPLIT_RE.finditer(text or ""))
    if not matches:
        return text or "", 0
    match = matches[-1]
    return (text or "")[match.end() :], match.end()


def _line_span(text: str, start: int, end: int, *, pair_id: str) -> Span:
    start_line = text.count("\n", 0, max(start, 0)) + 1
    end_index = max(start, end - 1)
    end_line = text.count("\n", 0, max(end_index, 0)) + 1
    return Span(SOURCE_PATH, start_line, end_line, node_id=pair_id)


def extract_equality_pairs(
    text: str,
    *,
    item_key: str,
    max_pairs: int = MAX_PAIR_CANDIDATES,
) -> tuple[EqualityPair, ...]:
    """Extract adjacent equality pairs using the frozen MathOps parser seed.

    MathOps returns span contents without offsets.  We recover the occurrence
    monotonically in the answer segment and bind every adjacent pair to the
    containing source lines.  A missing occurrence is skipped rather than
    assigned an invented location.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not _SAFE_ITEM_KEY.fullmatch(item_key or ""):
        raise ValueError("item_key must be a safe opaque identifier")
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")

    answer, answer_offset = _answer_with_offset(text)
    search_cursor = 0
    pairs: list[EqualityPair] = []
    occurrence = 0
    for _kind, math_span in MathOps.extract_math_spans(answer):
        local_start = answer.find(math_span, search_cursor)
        if local_start < 0:
            local_start = answer.find(math_span)
        if local_start < 0:
            continue
        local_end = local_start + len(math_span)
        search_cursor = local_end
        absolute_start = answer_offset + local_start
        absolute_end = answer_offset + local_end
        for parts in _equation_rows(math_span):
            for lhs, rhs in zip(parts, parts[1:]):
                if len(pairs) >= max_pairs:
                    return tuple(pairs)
                occurrence += 1
                digest = hashlib.sha256(
                    (item_key + "\0" + str(occurrence) + "\0" + lhs + "\0" + rhs).encode(
                        "utf-8", errors="replace"
                    )
                ).hexdigest()[:20]
                pair_id = f"pair-{occurrence:03d}-{digest}"
                witness = _line_span(
                    text, absolute_start, absolute_end, pair_id=pair_id
                )
                pairs.append(
                    EqualityPair(
                        item_key=item_key,
                        pair_id=pair_id,
                        lhs=lhs,
                        rhs=rhs,
                        witness=witness,
                    )
                )
    return tuple(pairs)


def verify_pair(pair: EqualityPair) -> Verdict:
    """Apply the exact symbolic operation and emit the shared verdict type."""

    result = verify_expression_pair(pair.lhs, pair.rhs)
    status = result.get("status")
    if status == "verified_rational_identity":
        return Verdict(True, False, (pair.witness,))
    if status in ("exact_nonidentity_witness", "universal_identity_counterexample"):
        return Verdict(True, True, (pair.witness,))
    if status in ("parse_noncoverage", "symbolically_unresolved"):
        return Verdict(False, False)
    raise ValueError(f"unexpected symbolic verifier status: {status!r}")


def verify_document_pairs(text: str, *, item_key: str) -> tuple[tuple[EqualityPair, Verdict], ...]:
    return tuple(
        (pair, verify_pair(pair))
        for pair in extract_equality_pairs(text, item_key=item_key)
    )


def ablate_witness_lines(text: str, witnesses: Sequence[Span]) -> str:
    """Delete file-qualified answer lines for the certificate ablation check."""

    if any(span.path != SOURCE_PATH for span in witnesses):
        raise ValueError("math a12 ablation received a foreign witness path")
    deleted = {
        line
        for span in witnesses
        for path, line in span.lines()
        if path == SOURCE_PATH
    }
    return "\n".join(
        line for number, line in enumerate(text.split("\n"), 1) if number not in deleted
    )


def states(values: Iterable[Verdict]) -> tuple[str, ...]:
    """Small public helper used by diagnostics and tests."""

    return tuple(value.state for value in values)
