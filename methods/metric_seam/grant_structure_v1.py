"""Deterministic relation extractors for the grant-funding hierarchy lane.

The module deliberately scores narrow, inspectable relations rather than a
whole proposal.  It consumes only the frozen hierarchy ``ctext`` projection;
it does not load a solicitation, outcome, reference judgment, model, or
external evidence.  Absence is a measured zero for document-internal presence
relations.  Arithmetic abstains unless an itemized currency list and a stated
total are both visible in the same frozen projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Callable, Iterable, Sequence


SCHEMA = "metric-seam.grant-structure-program.v1"
PROGRAM_ID = "grant_structure_v1"
DISCOVERY_MODE = "manual_mock_decomposition_seed"
INPUT_REPRESENTATION = "items_v2/grant-funding exact ctext; at most 4000 characters"

RELATION_DEPTHS = {
    "aim_hypothesis_experiment_graph": 2,
    "budget_sum_consistency": 3,
    "citation_claim_link": 2,
    "dissemination_output_channel_graph": 2,
    "document_outline_structure": 1,
    "evaluation_measurement_chain": 2,
    "front_matter_coverage": 1,
    "partner_role_graph": 2,
    "quantified_need_gap": 2,
    "resource_use_graph": 2,
    "risk_mitigation_graph": 2,
    "role_responsibility_graph": 2,
    "schedule_dependency_graph": 2,
}

ABSTENTION_CONDITIONS = {
    "budget_sum_consistency": (
        "fewer than two visible itemized currency amounts or no visible stated total"
    ),
    "other_relations": "empty ctext only; absence in nonempty ctext is a measured zero",
}

_SPACE_RE = re.compile(r"[\t \f\v]+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_HEADING_NUMBER_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)])\s+(.+?)\s*$")
_MONEY_RE = re.compile(
    r"(?<![\w])(?:US\s*)?\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*([kKmMbB])?\b"
)
_PERCENT_OR_COUNT_RE = re.compile(
    r"(?<!\w)(?:\d+(?:\.\d+)?\s*%(?!\w)|(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s+"
    r"(?:participants?|students?|patients?|households?|sites?|schools?|years?|months?))\b)",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(
    r"(?:\([A-Z][A-Za-z'’.-]+(?:\s+et\s+al\.)?,?\s*(?:19|20)\d{2}[a-z]?\)"
    r"|\[[0-9,;\s-]+\]|\b(?:doi:\s*|https?://|www\.)\S+)",
    re.IGNORECASE,
)
_SCHEDULE_TIME_RE = re.compile(
    r"\b(?:year|month|week|quarter|phase)\s*\d+\b|\b\d+\s*(?:years?|months?|weeks?)\b"
    r"|\bQ[1-4]\b",
    re.IGNORECASE,
)
_SCHEDULE_YEAR_CONTEXT_RE = re.compile(
    r"\b(?:by|during|from|through(?:out)?|until|between)\s+(?:19|20)\d{2}"
    r"(?:\s*[-–]\s*\d{2,4})?\b|\b(?:start|beginning|end|due)[^.\n]{0,40}"
    r"\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_ROLE_RE = re.compile(
    r"\b(?:principal investigator|project director|co-investigator|coordinator|manager|"
    r"director|officer|consultant|staff|team)\b",
    re.IGNORECASE,
)
_RESPONSIBILITY_RE = re.compile(
    r"\b(?:will|responsible for|lead(?:s|ing)?|oversee(?:s|ing)?|manag(?:e|es|ed|ing)|"
    r"coordinat(?:e|es|ed|ing)|supervis(?:e|es|ed|ing)|role)\b",
    re.IGNORECASE,
)
_SCHEDULE_ACTION_RE = re.compile(
    r"\b(?:milestones?|deliverables?|aims?|activities?|phases?|complet(?:e|es|ed|ing)|"
    r"begin(?:s|ning)?|then|after|before|concurrent(?:ly)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RelationResult:
    score: float | None
    status: str
    certificate: dict

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "status": self.status,
            "certificate": self.certificate,
        }


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _sentences(text: str) -> list[str]:
    return [
        normalized
        for part in _SENTENCE_BOUNDARY_RE.split(text)
        if (normalized := _normalize(part))
    ]


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in terms)


def _bounded_count(count: int, saturation: int = 3) -> float:
    return round(min(max(count, 0) / saturation, 1.0), 6)


def _presence_result(
    matches: Sequence[dict], *, saturation: int = 3, relation: str
) -> RelationResult:
    return RelationResult(
        score=_bounded_count(len(matches), saturation),
        status="measured",
        certificate={
            "relation": relation,
            "match_count": len(matches),
            "matches": list(matches[:8]),
            "saturation": saturation,
        },
    )


def _window_pairs(
    sentences: Sequence[str],
    left: Callable[[str], bool],
    right: Callable[[str], bool],
    *,
    maximum_distance: int = 1,
) -> list[dict]:
    pairs: list[dict] = []
    for left_index, sentence in enumerate(sentences):
        if not left(sentence):
            continue
        start = max(0, left_index - maximum_distance)
        stop = min(len(sentences), left_index + maximum_distance + 1)
        for right_index in range(start, stop):
            if right(sentences[right_index]):
                pairs.append(
                    {
                        "left_sentence": left_index,
                        "right_sentence": right_index,
                        "distance": abs(left_index - right_index),
                        "left_text": sentence[:200],
                        "right_text": sentences[right_index][:200],
                    }
                )
                break
    return pairs


def _money_value(match: re.Match[str]) -> Decimal | None:
    try:
        value = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    suffix = (match.group(2) or "").casefold()
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[suffix]
    return value * multiplier


def _budget_sum_consistency(text: str) -> RelationResult:
    lines = [_normalize(line) for line in text.splitlines() if _normalize(line)]
    item_rows: list[tuple[int, Decimal, str]] = []
    total_rows: list[tuple[int, Decimal, str]] = []
    for index, line in enumerate(lines):
        values = [value for match in _MONEY_RE.finditer(line) if (value := _money_value(match))]
        if not values:
            continue
        is_total = _contains_any(
            line,
            (
                "total",
                "investment",
                "amount requested",
                "funding request",
                "overall budget",
                "project cost",
            ),
        )
        if is_total:
            total_rows.extend((index, value, line[:240]) for value in values)
        elif len(values) == 1 and (
            line.lstrip().startswith(("•", "-", "*"))
            or " for " in line.casefold()
            or "cost" in line.casefold()
            or "budget" in line.casefold()
        ):
            item_rows.append((index, values[0], line[:240]))

    candidates: list[dict] = []
    for total_index, declared, total_line in total_rows:
        preceding = [row for row in item_rows if 0 < total_index - row[0] <= 12]
        if len(preceding) < 2 or declared <= 0:
            continue
        item_sum = sum((row[1] for row in preceding), Decimal(0))
        relative_error = abs(item_sum - declared) / declared
        score = max(Decimal(0), Decimal(1) - min(relative_error, Decimal(1)))
        candidates.append(
            {
                "declared_total": float(declared),
                "item_sum": float(item_sum),
                "relative_error": round(float(relative_error), 8),
                "score": round(float(score), 6),
                "n_items": len(preceding),
                "total_line": total_line,
                "item_lines": [row[2] for row in preceding],
            }
        )
    if not candidates:
        return RelationResult(
            score=None,
            status="abstained",
            certificate={
                "relation": "budget_sum_consistency",
                "reason": ABSTENTION_CONDITIONS["budget_sum_consistency"],
                "visible_item_amounts": len(item_rows),
                "visible_total_amounts": len(total_rows),
            },
        )
    # Select the most recent visible declaration, not the best-fitting total.
    chosen = candidates[-1]
    return RelationResult(
        score=chosen["score"],
        status="measured",
        certificate={
            "relation": "budget_sum_consistency",
            **chosen,
            "selection_rule": "last checkable stated total in ctext order",
        },
    )


def _quantified_need_gap(sentences: Sequence[str]) -> RelationResult:
    matches = []
    for index, sentence in enumerate(sentences):
        if _PERCENT_OR_COUNT_RE.search(sentence) and _contains_any(
            sentence,
            (
                "compared",
                "versus",
                "vs.",
                "gap",
                "lack",
                "need",
                "shortage",
                "underserved",
                "barrier",
                "baseline",
            ),
        ):
            matches.append({"sentence": index, "text": sentence[:240]})
    return _presence_result(matches, relation="quantified_need_gap")


def _citation_claim_link(sentences: Sequence[str]) -> RelationResult:
    matches = []
    for index, sentence in enumerate(sentences):
        if _CITATION_RE.search(sentence) and (
            _PERCENT_OR_COUNT_RE.search(sentence)
            or _contains_any(
                sentence,
                (
                    "show",
                    "demonstrat",
                    "evidence",
                    "found",
                    "suggest",
                    "report",
                    "associated",
                    "increase",
                    "decrease",
                ),
            )
        ):
            matches.append({"sentence": index, "text": sentence[:240]})
    return _presence_result(matches, relation="citation_claim_link")


def _evaluation_measurement_chain(sentences: Sequence[str]) -> RelationResult:
    metric_terms = ("metric", "indicator", "measure", "outcome", "success criteria", "evaluate")
    method_terms = ("survey", "interview", "record", "data", "analysis", "assess", "collect")
    use_terms = ("report", "monitor", "decision", "adjust", "improve", "go/no-go", "milestone")
    matches = []
    for index, sentence in enumerate(sentences):
        window = " ".join(sentences[index : index + 2])
        dimensions = {
            "metric": _contains_any(window, metric_terms),
            "method": _contains_any(window, method_terms),
            "use": _contains_any(window, use_terms),
            "quantified": bool(_PERCENT_OR_COUNT_RE.search(window)),
        }
        if dimensions["metric"] and dimensions["method"] and sum(dimensions.values()) >= 3:
            matches.append({"sentence": index, "dimensions": dimensions, "text": window[:260]})
    return _presence_result(matches, relation="evaluation_measurement_chain")


def _aim_hypothesis_experiment_graph(sentences: Sequence[str]) -> RelationResult:
    def hypothesis(value: str) -> bool:
        return _contains_any(
            value, ("hypothes", "predict", "proposition", "research question")
        )

    def test(value: str) -> bool:
        return _contains_any(
            value,
            ("aim", "experiment", "study", "test", "evaluate", "analysis", "investigate"),
        )

    pairs = _window_pairs(sentences, hypothesis, test, maximum_distance=2)
    return _presence_result(pairs, relation="aim_hypothesis_experiment_graph")


def _role_responsibility_graph(sentences: Sequence[str]) -> RelationResult:
    matches = []
    for index, sentence in enumerate(sentences):
        role = _ROLE_RE.search(sentence)
        responsibility = _RESPONSIBILITY_RE.search(sentence)
        if (
            role
            and responsibility
            and role.end() <= responsibility.start()
            and responsibility.start() - role.end() <= 80
        ):
            matches.append({"sentence": index, "text": sentence[:240]})
    return _presence_result(matches, relation="role_responsibility_graph")


def _risk_mitigation_graph(sentences: Sequence[str]) -> RelationResult:
    def risk(value: str) -> bool:
        return _contains_any(
            value,
            ("risk", "pitfall", "limitation", "failure", "challenge", "barrier"),
        )

    def mitigation(value: str) -> bool:
        return _contains_any(
            value,
            (
                "mitigat",
                "alternative",
                "contingenc",
                "fallback",
                "if ",
                "address",
                "adapt",
                "otherwise",
            ),
        )

    pairs = _window_pairs(sentences, risk, mitigation, maximum_distance=1)
    return _presence_result(pairs, relation="risk_mitigation_graph")


def _schedule_dependency_graph(sentences: Sequence[str]) -> RelationResult:
    matches = []
    for index, sentence in enumerate(sentences):
        window = " ".join(sentences[index : index + 2])
        calendar_year_in_schedule_context = bool(_SCHEDULE_YEAR_CONTEXT_RE.search(window))
        if (
            _SCHEDULE_TIME_RE.search(window) or calendar_year_in_schedule_context
        ) and _SCHEDULE_ACTION_RE.search(window):
            matches.append({"sentence": index, "text": window[:260]})
    return _presence_result(matches, relation="schedule_dependency_graph")


def _partner_role_graph(sentences: Sequence[str]) -> RelationResult:
    def partner(value: str) -> bool:
        return _contains_any(
            value,
            (
                "partner",
                "collaborat",
                "stakeholder",
                "community advisory",
                "subaward",
                "consortium",
            ),
        )

    def role(value: str) -> bool:
        return _contains_any(
            value,
            (
                " will ",
                "responsible",
                "provide",
                "lead",
                "contribute",
                "commit",
                "role",
                "support",
            ),
        )

    pairs = _window_pairs(sentences, partner, role, maximum_distance=1)
    return _presence_result(pairs, relation="partner_role_graph")


def _dissemination_output_channel_graph(sentences: Sequence[str]) -> RelationResult:
    def output(value: str) -> bool:
        return _contains_any(
            value,
            ("result", "finding", "data", "resource", "tool", "report", "curriculum"),
        )

    def channel(value: str) -> bool:
        return _contains_any(
            value,
            (
                "publish",
                "conference",
                "website",
                "repository",
                "workshop",
                "briefing",
                "disseminat",
                "share",
                "audience",
            ),
        )

    pairs = _window_pairs(sentences, output, channel, maximum_distance=1)
    return _presence_result(pairs, relation="dissemination_output_channel_graph")


def _resource_use_graph(sentences: Sequence[str]) -> RelationResult:
    def resource(value: str) -> bool:
        return _contains_any(
            value,
            (
                "facility",
                "facilities",
                "equipment",
                "laboratory",
                "center",
                "infrastructure",
            ),
        )

    def use(value: str) -> bool:
        return _contains_any(
            value,
            (
                "use",
                "enable",
                "support",
                "provide",
                "available",
                "access",
                "conduct",
                "perform",
            ),
        )

    pairs = _window_pairs(sentences, resource, use, maximum_distance=1)
    return _presence_result(pairs, relation="resource_use_graph")


def _front_matter_coverage(text: str) -> RelationResult:
    front = _normalize(text[:1400])
    dimensions = {
        "problem_or_need": _contains_any(front, ("problem", "need", "challenge", "gap")),
        "approach_or_activity": _contains_any(
            front, ("propose", "project", "approach", "program", "study", "will ")
        ),
        "outcome_or_impact": _contains_any(
            front, ("outcome", "impact", "result", "benefit", "improve", "enable")
        ),
        "quantified_specificity": bool(_PERCENT_OR_COUNT_RE.search(front) or _MONEY_RE.search(front)),
    }
    return RelationResult(
        score=round(sum(dimensions.values()) / len(dimensions), 6),
        status="measured",
        certificate={
            "relation": "front_matter_coverage",
            "dimensions": dimensions,
            "front_characters": len(text[:1400]),
        },
    )


def _document_outline_structure(text: str) -> RelationResult:
    headings: list[dict] = []
    for index, raw_line in enumerate(text.splitlines()):
        line = _normalize(raw_line)
        if not line or len(line) > 100:
            continue
        numbered = _HEADING_NUMBER_RE.match(line)
        all_caps = len(line) >= 4 and line.upper() == line and any(char.isalpha() for char in line)
        known = _contains_any(
            line,
            (
                "executive summary",
                "specific aims",
                "project description",
                "budget",
                "expected outcomes",
                "method",
                "research strategy",
                "timeline",
                "evaluation",
            ),
        )
        if numbered or all_caps or known:
            headings.append({"line": index, "text": line[:100]})
    unique = {row["text"].casefold() for row in headings}
    return RelationResult(
        score=_bounded_count(len(unique), 6),
        status="measured",
        certificate={
            "relation": "document_outline_structure",
            "heading_count": len(headings),
            "unique_heading_count": len(unique),
            "headings": headings[:12],
            "saturation": 6,
        },
    )


def analyze(ctext: str) -> dict:
    """Return all frozen relation outputs for one exact grant ``ctext``."""

    if not isinstance(ctext, str):
        raise TypeError("ctext must be a string")
    sentences = _sentences(ctext)
    if not ctext.strip():
        relations = {
            relation: RelationResult(
                score=None,
                status="abstained",
                certificate={"relation": relation, "reason": "empty ctext"},
            ).as_dict()
            for relation in RELATION_DEPTHS
        }
    else:
        results = {
            "aim_hypothesis_experiment_graph": _aim_hypothesis_experiment_graph(sentences),
            "budget_sum_consistency": _budget_sum_consistency(ctext),
            "citation_claim_link": _citation_claim_link(sentences),
            "dissemination_output_channel_graph": _dissemination_output_channel_graph(
                sentences
            ),
            "document_outline_structure": _document_outline_structure(ctext),
            "evaluation_measurement_chain": _evaluation_measurement_chain(sentences),
            "front_matter_coverage": _front_matter_coverage(ctext),
            "partner_role_graph": _partner_role_graph(sentences),
            "quantified_need_gap": _quantified_need_gap(sentences),
            "resource_use_graph": _resource_use_graph(sentences),
            "risk_mitigation_graph": _risk_mitigation_graph(sentences),
            "role_responsibility_graph": _role_responsibility_graph(sentences),
            "schedule_dependency_graph": _schedule_dependency_graph(sentences),
        }
        relations = {name: result.as_dict() for name, result in results.items()}
    return {
        "schema": SCHEMA,
        "program_id": PROGRAM_ID,
        "discovery_mode": DISCOVERY_MODE,
        "input_characters": len(ctext),
        "sentence_count": len(sentences),
        "relations": relations,
    }
