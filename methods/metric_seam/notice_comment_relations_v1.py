"""Typed dependency/evidence relations for short public comments.

The hierarchy corpus is a 4,000-character cap, but the selected public comments
are much shorter (compiler-train median about 110 characters).  This module
therefore does not pretend to verify full rule-document or agency-compliance
constructs.  It exposes narrow relations that are actually present in a
comment: actionable requests, provision/authority targeting, causal support,
quantity-to-action links, distributional-impact links, and specialized
uncertainty, burden, cost-comparison, and time-value structures.

spaCy's local CPU parser supplies token/dependency structure.  No LLM, remote
model, outcome, reference score, docket corpus, or external authority is used.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
from typing import Iterable, Sequence

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token


SCHEMA = "metric-seam.notice-comment-relations.v1"
PROGRAM_ID = "notice_comment_relations_v1"
DISCOVERY_MODE = "manual_mock_decomposition_seed"
PARSER_MODEL = "en_core_web_sm"
INPUT_REPRESENTATION = "items_v2/notice-and-comment exact ctext; at most 4000 characters"

RELATION_DEPTHS = {
    "actionable_target_dependency": 2,
    "burden_breakdown_relation": 2,
    "causal_support_action_link": 2,
    "corrective_target_dependency": 2,
    "cost_comparison_relation": 2,
    "distributional_group_impact_link": 2,
    "identity_authenticity_action_link": 2,
    "legal_authority_action_link": 2,
    "pinpoint_provision_action_link": 2,
    "privacy_restriction_action_link": 2,
    "quantified_action_link": 2,
    "supported_actionable_target_graph": 3,
    "time_value_relation": 2,
    "uncertainty_bound_relation": 2,
}

_ACTION_LEMMAS = {
    "adopt",
    "allow",
    "amend",
    "clarify",
    "correct",
    "define",
    "eliminate",
    "exclude",
    "include",
    "oppose",
    "prohibit",
    "recognize",
    "recommend",
    "remove",
    "request",
    "require",
    "retain",
    "revise",
    "support",
    "urge",
    "withdraw",
}
_DIRECTIVE_LEMMAS = {"oppose", "recommend", "request", "support", "urge"}
_MODAL_ACTIONS = {"must", "shall", "should"}
_TARGET_DEPS = {
    "attr",
    "ccomp",
    "dative",
    "dobj",
    "obj",
    "obl",
    "oprd",
    "pobj",
    "xcomp",
}
_CAUSAL_MARKERS = {
    "as a result",
    "based on",
    "because",
    "due to",
    "evidence",
    "given that",
    "therefore",
    "thus",
}
_GROUP_TERMS = {
    "children",
    "community",
    "consumer",
    "disadvantaged",
    "elderly",
    "household",
    "industry",
    "low-income",
    "minority",
    "patient",
    "region",
    "rural",
    "small business",
    "state",
    "tribal",
    "worker",
}
_IMPACT_LEMMAS = {
    "affect",
    "benefit",
    "burden",
    "cost",
    "harm",
    "impact",
    "increase",
    "reduce",
}
_CORRECTIVE_ACTIONS = {"amend", "clarify", "correct", "remove", "revise", "withdraw"}
_PRIVACY_TERMS = {
    "confidential",
    "copyright",
    "personal information",
    "personally identifiable",
    "pii",
    "privacy",
    "restricted",
    "sensitive",
}
_IDENTITY_TERMS = {
    "authentic",
    "consent",
    "fabricat",
    "fake",
    "identity",
    "impersonat",
    "sponsor",
    "third-party",
}

_LEGAL_CITATION_RE = re.compile(
    r"\b(?:\d+\s+(?:U\.?S\.?C\.?|C\.?F\.?R\.?|Fed\.?\s*Reg\.?)\s*§?\s*"
    r"[\w.()\-]+|(?:section|§)\s*[\w.()\-]+|Executive\s+Order\s+\d+|EO\s*\d+)(?!\w)",
    re.IGNORECASE,
)
_PINPOINT_RE = re.compile(
    r"\b(?:section|subsection|paragraph|page|p\.|§)\s*[\w.()\-]+(?!\w)|"
    r"\b\d+\s+C\.?F\.?R\.?\s*§?\s*[\w.()\-]+(?!\w)",
    re.IGNORECASE,
)
# A bare digit is usually an identifier in these comments (rule year, form
# number, part number, or paragraph ordinal), not quantitative support.  Keep
# this deliberately narrower than a generic number recognizer: a quantity must
# carry currency or a substantive unit/magnitude.
_QUANTITY_RE = re.compile(
    r"(?<!\w)(?:"
    r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion))?"
    r"|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:-\s*)?"
    r"(?:%|percent|dollars?|flight\s+hours?|hours?|minutes?|days?|months?|years?|"
    r"responses?|respondents?|mg/L|million|billion|basis\s+points?|miles?|feet|"
    r"kilograms?|kg|tons?)"
    r")(?!\w)",
    re.IGNORECASE,
)
_UNCERTAINTY_RE = re.compile(
    r"\b(?:confidence interval|credible interval|distribution|expected value|high[- ]case|"
    r"low[- ]case|lower bound|upper bound|percentile|range|scenario|sensitivity)\b",
    re.IGNORECASE,
)
_BURDEN_ENTITY_RE = re.compile(
    r"\b(?:respondents?|responses?|frequency|hours?|minutes?|time per response|annual burden|"
    r"wage rate|opportunity cost)\b",
    re.IGNORECASE,
)
_COST_RE = re.compile(
    r"\b(?:costs?|savings?|benefits?|net benefits?|cost-effectiveness|in-house|contract(?:or|ing)?|"
    r"commercial bid|incremental|alternative)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(?:compared (?:with|to)|versus|vs\.?|more than|less than|higher than|lower than|"
    r"increase|decrease|difference|alternative)\b",
    re.IGNORECASE,
)
_TIME_VALUE_RE = re.compile(
    r"\b(?:discount rate|net present value|NPV|annualized value|inflation|real dollars?|"
    r"nominal dollars?|base year)\b",
    re.IGNORECASE,
)

_NLP: Language | None = None


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


def _nlp() -> Language:
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(PARSER_MODEL, disable=("ner",))
        required = {"parser", "lemmatizer"}
        if not required <= set(_NLP.pipe_names):
            raise RuntimeError(f"{PARSER_MODEL} lacks required pipes: {sorted(required)}")
    return _NLP


def _bounded_count(count: int, saturation: int = 2) -> float:
    return round(min(max(count, 0) / saturation, 1.0), 6)


def _result(relation: str, matches: Sequence[dict], *, saturation: int = 2) -> RelationResult:
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


def _sentence_for_char(doc: Doc, start: int, end: int) -> Span | None:
    overlapping = [
        sentence
        for sentence in doc.sents
        if sentence.start_char < end and start < sentence.end_char
    ]
    if not overlapping:
        return None
    return doc[overlapping[0].start : overlapping[-1].end]


def _action_nodes(sentence: Span) -> list[Token]:
    actions: dict[int, Token] = {}
    for token in sentence:
        lemma = token.lemma_.casefold()
        modal_child = any(child.lower_ in _MODAL_ACTIONS for child in token.children)
        imperative_root = bool(
            token.dep_ == "ROOT"
            and token.tag_ == "VB"
            and not any(child.dep_ in {"csubj", "nsubj", "nsubjpass"} for child in token.children)
        )
        if (
            lemma in _ACTION_LEMMAS
            and token.pos_ in {"AUX", "VERB"}
            and (lemma in _DIRECTIVE_LEMMAS or modal_child or imperative_root)
        ):
            actions[token.i] = token
        if token.lower_ in _MODAL_ACTIONS and token.dep_ in {"aux", "auxpass"}:
            head = token.head
            if head.pos_ in {"AUX", "VERB"}:
                actions[head.i] = head
    return [actions[index] for index in sorted(actions)]


def _target_tokens(action: Token, sentence: Span) -> list[Token]:
    targets = [
        child
        for child in action.children
        if child.dep_ in _TARGET_DEPS and child.i in range(sentence.start, sentence.end)
    ]
    if targets:
        return targets
    return [
        token
        for token in sentence
        if token.i > action.i
        and token.pos_ in {"NOUN", "PROPN"}
        and not token.is_stop
    ][:1]


def _dependency_distance(left: Token, right: Token, sentence: Span) -> int | None:
    if left.i == right.i:
        return 0
    allowed = set(range(sentence.start, sentence.end))
    queue: deque[tuple[Token, int]] = deque([(left, 0)])
    visited = {left.i}
    while queue:
        token, distance = queue.popleft()
        neighbors = [token.head, *token.children]
        for neighbor in neighbors:
            if neighbor.i not in allowed or neighbor.i in visited:
                continue
            if neighbor.i == right.i:
                return distance + 1
            visited.add(neighbor.i)
            queue.append((neighbor, distance + 1))
    return None


def _actionable_targets(doc: Doc) -> list[dict]:
    matches = []
    for sentence_index, sentence in enumerate(doc.sents):
        for action in _action_nodes(sentence):
            targets = _target_tokens(action, sentence)
            if not targets:
                continue
            target = targets[0]
            matches.append(
                {
                    "sentence": sentence_index,
                    "action": action.lemma_.casefold(),
                    "action_text": action.text,
                    "target": " ".join(token.text for token in target.subtree)[:180],
                    "target_head": target.lemma_.casefold(),
                    "dependency_distance": _dependency_distance(action, target, sentence),
                    "sentence_text": sentence.text[:260],
                }
            )
    return matches


def _span_action_links(
    doc: Doc,
    pattern: re.Pattern[str],
    *,
    require_pinpoint: bool = False,
    excluded_ranges: Sequence[tuple[int, int]] = (),
) -> list[dict]:
    links = []
    for match in pattern.finditer(doc.text):
        if any(start < match.end() and match.start() < end for start, end in excluded_ranges):
            continue
        sentence = _sentence_for_char(doc, match.start(), match.end())
        if sentence is None:
            continue
        actions = _action_nodes(sentence)
        if not actions:
            continue
        span = doc.char_span(match.start(), match.end(), alignment_mode="expand")
        if span is None:
            continue
        anchor = span.root
        action = min(
            actions,
            key=lambda node: (
                _dependency_distance(anchor, node, sentence) is None,
                _dependency_distance(anchor, node, sentence) or abs(anchor.i - node.i),
            ),
        )
        distance = _dependency_distance(anchor, action, sentence)
        links.append(
            {
                "span": match.group(0),
                "action": action.lemma_.casefold(),
                "dependency_distance": distance,
                "local_token_distance": abs(anchor.i - action.i),
                "dependency_path_connected": distance is not None,
                "pinpoint": bool(_PINPOINT_RE.search(match.group(0))),
                "sentence_text": sentence.text[:260],
            }
        )
    if require_pinpoint:
        return [row for row in links if row["pinpoint"]]
    return links


def _causal_support_links(doc: Doc) -> list[dict]:
    matches = []
    sentences = list(doc.sents)
    for sentence_index, sentence in enumerate(sentences):
        windows = [sentence]
        if sentence_index + 1 < len(sentences):
            windows.append(doc[sentence.start : sentences[sentence_index + 1].end])
        for window in windows:
            lowered = window.text.casefold()
            markers = sorted(marker for marker in _CAUSAL_MARKERS if marker in lowered)
            actions = _action_nodes(window)
            if not markers or not actions:
                continue
            matches.append(
                {
                    "sentence": sentence_index,
                    "markers": markers,
                    "actions": [token.lemma_.casefold() for token in actions],
                    "sentence_text": window.text[:260],
                }
            )
            break
    return matches


def _distributional_links(doc: Doc) -> list[dict]:
    matches = []
    for sentence_index, sentence in enumerate(doc.sents):
        lowered = sentence.text.casefold()
        groups = sorted(term for term in _GROUP_TERMS if term in lowered)
        impacts = sorted(
            {token.lemma_.casefold() for token in sentence if token.lemma_.casefold() in _IMPACT_LEMMAS}
        )
        if groups and impacts:
            matches.append(
                {
                    "sentence": sentence_index,
                    "groups": groups,
                    "impact_predicates": impacts,
                    "sentence_text": sentence.text[:260],
                }
            )
    return matches


def _filtered_actionable(
    actionable: Sequence[dict],
    relation: str,
    *,
    action_lemmas: set[str] | None = None,
    text_terms: set[str] | None = None,
) -> RelationResult:
    matches = []
    for row in actionable:
        text = f"{row['target']} {row['sentence_text']}".casefold()
        if action_lemmas is not None and row["action"] not in action_lemmas:
            continue
        terms = sorted(term for term in (text_terms or set()) if term in text)
        if text_terms is not None and not terms:
            continue
        matches.append({**row, "matched_terms": terms})
    return _result(relation, matches)


def _quantity_spans(text: str) -> list[re.Match[str]]:
    """Return unit-bearing quantities outside legal/pinpoint identifiers."""
    excluded = [
        *(match.span() for match in _LEGAL_CITATION_RE.finditer(text)),
        *(match.span() for match in _PINPOINT_RE.finditer(text)),
    ]
    return [
        match
        for match in _QUANTITY_RE.finditer(text)
        if not any(start < match.end() and match.start() < end for start, end in excluded)
    ]


def _supported_actionable_graph(actionable: Sequence[dict]) -> RelationResult:
    matches = []
    for row in actionable:
        text = row["sentence_text"]
        support_types = []
        if _LEGAL_CITATION_RE.search(text):
            support_types.append("legal_authority")
        if _PINPOINT_RE.search(text):
            support_types.append("pinpoint_provision")
        if _quantity_spans(text):
            support_types.append("quantity")
        if any(marker in text.casefold() for marker in _CAUSAL_MARKERS):
            support_types.append("causal_marker")
        if support_types:
            matches.append({**row, "support_types": sorted(set(support_types))})
    return _result("supported_actionable_target_graph", matches)


def _specialized_relation(
    doc: Doc,
    relation: str,
    required_patterns: Iterable[re.Pattern[str]],
    *,
    minimum_patterns: int,
    require_action: bool = False,
) -> RelationResult:
    matches = []
    patterns = list(required_patterns)
    for sentence_index, sentence in enumerate(doc.sents):
        found = [pattern.pattern for pattern in patterns if pattern.search(sentence.text)]
        actions = _action_nodes(sentence)
        if len(found) >= minimum_patterns and (not require_action or actions):
            matches.append(
                {
                    "sentence": sentence_index,
                    "matched_pattern_families": len(found),
                    "actions": [token.lemma_.casefold() for token in actions],
                    "sentence_text": sentence.text[:260],
                }
            )
    return _result(relation, matches)


def analyze(ctext: str) -> dict:
    if not isinstance(ctext, str):
        raise TypeError("ctext must be a string")
    if not ctext.strip():
        relations = {
            relation: RelationResult(
                score=None,
                status="abstained",
                certificate={"relation": relation, "reason": "empty ctext"},
            ).as_dict()
            for relation in RELATION_DEPTHS
        }
        return {
            "schema": SCHEMA,
            "program_id": PROGRAM_ID,
            "parser_model": PARSER_MODEL,
            "relations": relations,
        }

    doc = _nlp()(ctext)
    actionable = _actionable_targets(doc)
    legal = _span_action_links(doc, _LEGAL_CITATION_RE)
    pinpoint = _span_action_links(doc, _PINPOINT_RE, require_pinpoint=True)
    excluded_numeric_ranges = [
        *(match.span() for match in _LEGAL_CITATION_RE.finditer(ctext)),
        *(match.span() for match in _PINPOINT_RE.finditer(ctext)),
    ]
    quantified = _span_action_links(
        doc,
        _QUANTITY_RE,
        excluded_ranges=excluded_numeric_ranges,
    )
    causal = _causal_support_links(doc)
    distributional = _distributional_links(doc)
    relation_results = {
        "actionable_target_dependency": _result("actionable_target_dependency", actionable),
        "corrective_target_dependency": _filtered_actionable(
            actionable,
            "corrective_target_dependency",
            action_lemmas=_CORRECTIVE_ACTIONS,
        ),
        "privacy_restriction_action_link": _filtered_actionable(
            actionable,
            "privacy_restriction_action_link",
            text_terms=_PRIVACY_TERMS,
        ),
        "identity_authenticity_action_link": _filtered_actionable(
            actionable,
            "identity_authenticity_action_link",
            text_terms=_IDENTITY_TERMS,
        ),
        "supported_actionable_target_graph": _supported_actionable_graph(actionable),
        "legal_authority_action_link": _result("legal_authority_action_link", legal),
        "pinpoint_provision_action_link": _result(
            "pinpoint_provision_action_link", pinpoint
        ),
        "quantified_action_link": _result("quantified_action_link", quantified),
        "causal_support_action_link": _result("causal_support_action_link", causal),
        "distributional_group_impact_link": _result(
            "distributional_group_impact_link", distributional
        ),
        "uncertainty_bound_relation": _specialized_relation(
            doc,
            "uncertainty_bound_relation",
            (_UNCERTAINTY_RE, _QUANTITY_RE),
            minimum_patterns=2,
        ),
        "burden_breakdown_relation": _specialized_relation(
            doc,
            "burden_breakdown_relation",
            (_BURDEN_ENTITY_RE, _QUANTITY_RE),
            minimum_patterns=2,
        ),
        "cost_comparison_relation": _specialized_relation(
            doc,
            "cost_comparison_relation",
            (_COST_RE, _COMPARISON_RE, _QUANTITY_RE),
            minimum_patterns=2,
        ),
        "time_value_relation": _specialized_relation(
            doc,
            "time_value_relation",
            (_TIME_VALUE_RE, _QUANTITY_RE),
            minimum_patterns=2,
        ),
    }
    return {
        "schema": SCHEMA,
        "program_id": PROGRAM_ID,
        "discovery_mode": DISCOVERY_MODE,
        "parser_model": PARSER_MODEL,
        "parser_pipes": list(_nlp().pipe_names),
        "input_characters": len(ctext),
        "sentence_count": sum(1 for _ in doc.sents),
        "relations": {
            relation: relation_results[relation].as_dict()
            for relation in sorted(RELATION_DEPTHS)
        },
    }
