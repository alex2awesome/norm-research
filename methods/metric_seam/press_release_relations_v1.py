"""Additive, outcome-blind relations over hierarchy press-release ``ctext``.

The module deliberately emits replayable local witnesses rather than scores or
criterion verdicts.  It is an evolution of the retrospective press-release h0
programs and :mod:`methods.metric_seam.hybrids.ops_capability`: attribution,
sentence structure, dates, quantities, named entities, and links are combined
into relation-specific graphs.  No relation resolves URLs, retrieves documents,
loads a corpus, calls a generative/remote model, or consults a reference/outcome
value.  The dependency and NER operations use a disclosed local CPU spaCy parser.

Depth is the depth of the *matched local relation*, not the size of this file:

``D2``
    a parser-backed structural measurement;
``D3``
    a positive relation computed by graph traversal, within-document retrieval,
    dependency binding, or arithmetic recomputation.

Absence of a witness is never an absence certificate for the source document.
Hierarchy items can be truncated at the declared 4,000-character projection.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import functools
import math
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

from methods.metric_seam.hybrids.ops_capability import number_consistency


SCHEMA = "metric-seam.press-release-local-relations.v1"
PROGRAM_ID = "press_release_local_relations_v1"

RELATION_SPECS: dict[str, dict[str, Any]] = {
    "attribution_claim_binding": {
        "matched_depth": 3,
        "depth_meaning": "dependency-bound speaker-to-claim or speaker-to-quote edge",
        "implemented_relation": (
            "A named or grammatical speaker is bound to the exact attributed clause "
            "or direct-quotation span in the presented ctext."
        ),
        "does_not_establish": (
            "speaker independence, credibility, diversity, factual accuracy, quote tone, "
            "or completeness of attribution"
        ),
    },
    "quote_integration_structure": {
        "matched_depth": 2,
        "depth_meaning": "sentence/quote-span structural computation",
        "implemented_relation": (
            "Direct-quotation spans are located and their sentence position, length, "
            "speaker binding, and block-versus-integrated structure are measured."
        ),
        "does_not_establish": "insight, authenticity, news value, or aesthetic quote quality",
    },
    "entity_evidence_graph": {
        "matched_depth": 3,
        "depth_meaning": "named-entity to local evidence path in a sentence graph",
        "implemented_relation": (
            "Named entities are connected to quantities, dates, or URLs through explicit "
            "same-sentence or adjacent-sentence paths."
        ),
        "does_not_establish": (
            "truth, causal support, source independence, external verification, or whether "
            "the evidence is sufficient"
        ),
    },
    "claim_evidence_alignment": {
        "matched_depth": 3,
        "depth_meaning": "within-document retrieval plus shared-entity relation",
        "implemented_relation": (
            "A syntactically claim-like sentence is aligned to a local evidence-bearing "
            "sentence by self-containment, shared entities, or TF-IDF similarity."
        ),
        "does_not_establish": "entailment, factual correctness, or external corroboration",
    },
    "date_quantity_internal_consistency": {
        "matched_depth": 3,
        "depth_meaning": "calendar parsing and arithmetic recomputation when checkable",
        "implemented_relation": (
            "Date and quantity mentions are bound to their local noun/clause context; "
            "explicit count-percentage and before-after percentage relations are recomputed."
        ),
        "does_not_establish": (
            "external accuracy, event actuality, chronological narrative quality, or a "
            "negative finding when no re-derivable arithmetic is present"
        ),
    },
    "url_role_clause_binding": {
        "matched_depth": 3,
        "depth_meaning": "URL-to-action/source/contact clause graph edge",
        "implemented_relation": (
            "A URL-like span is parsed and bound to an explicit source, action, contact, "
            "asset, social, or organizational clause in the presented ctext."
        ),
        "does_not_establish": (
            "link reachability, destination contents, official status, rights clearance, "
            "freshness, or external evidentiary quality"
        ),
    },
    "opening_information_graph_alignment": {
        "matched_depth": 3,
        "depth_meaning": "sentence graph centrality and opening-to-body alignment",
        "implemented_relation": (
            "Opening sentences are compared with central body sentences and central named "
            "entities to measure front-loading in the presented text."
        ),
        "does_not_establish": (
            "that the inferred first sentence is a true headline/lede, accuracy, novelty, "
            "timeliness, or overall newsworthiness"
        ),
    },
    "sentence_dependency_readability": {
        "matched_depth": 2,
        "depth_meaning": "dependency-tree and sentence-distribution computation",
        "implemented_relation": (
            "Sentence length, dependency depth, passive syntax, long-token density, and "
            "explicit acronym-definition bindings are measured."
        ),
        "does_not_establish": (
            "reader comprehension, AP-style correctness, jargon meaning, cultural or "
            "linguistic accessibility, or prose quality"
        ),
    },
    "cta_resource_binding": {
        "matched_depth": 3,
        "depth_meaning": "action-clause to resource/contact graph edge",
        "implemented_relation": (
            "An imperative or explicit action clause is bound to a URL, email, or phone "
            "resource in the same or adjacent sentence."
        ),
        "does_not_establish": (
            "audience benefit, persuasion, conversion, destination validity, or that one "
            "action is strategically optimal"
        ),
    },
    "boilerplate_contact_structure": {
        "matched_depth": 2,
        "depth_meaning": "footer/section and contact-resource structural binding",
        "implemented_relation": (
            "Explicit About, media-contact, investor-contact, and resource-footer cues are "
            "located and bound to nearby organizations or contact resources."
        ),
        "does_not_establish": "completeness, currency, factual accuracy, or usefulness",
    },
    "section_scannability_structure": {
        "matched_depth": 2,
        "depth_meaning": "line/section/list structure over exact projected text",
        "implemented_relation": (
            "Preserved headings, list markers, paragraph breaks, and short-block structure "
            "are measured from the exact ctext representation."
        ),
        "does_not_establish": (
            "visual layout, typography, contrast, mobile rendering, or source formatting "
            "that was discarded before ctext projection"
        ),
    },
    "event_logistics_binding": {
        "matched_depth": 3,
        "depth_meaning": "event clause to actor/date/time/place dependency/NER binding",
        "implemented_relation": (
            "An event-like clause is bound to explicit actors and date/time/place entities "
            "through paths in the same sentence's dependency graph."
        ),
        "does_not_establish": "event actuality, legal compliance, attendance value, or completeness",
    },
    "attribution_scoped_claim_language": {
        "matched_depth": 3,
        "depth_meaning": "lexical claim cue scoped through an attribution graph",
        "implemented_relation": (
            "A closed, disclosed set of superlative/alarm/promotional cues is assigned to "
            "document voice, a bound speaker, or unresolved quotation scope."
        ),
        "does_not_establish": (
            "sensationalism, deception, evidentiary proportionality, or whether the wording "
            "is justified"
        ),
    },
    "significance_comparison_binding": {
        "matched_depth": 3,
        "depth_meaning": "comparative/causal clause to entity or quantity binding",
        "implemented_relation": (
            "Comparative or explicit significance/causal clauses are bound to named entities "
            "or quantities in their sentence."
        ),
        "does_not_establish": "importance, causal validity, audience value, or benchmark fairness",
    },
    "uncertainty_claim_scope_binding": {
        "matched_depth": 3,
        "depth_meaning": "modal/limitation cue to governed claim dependency",
        "implemented_relation": (
            "Explicit uncertainty, preliminary-status, confidence, or limitation cues are "
            "bound to the clause they syntactically modify."
        ),
        "does_not_establish": (
            "whether uncertainty is calibrated, complete, prominent enough, or scientifically valid"
        ),
    },
    "commitment_action_binding": {
        "matched_depth": 3,
        "depth_meaning": "modal/commitment predicate to concrete action dependency path",
        "implemented_relation": (
            "An explicit promise, plan, intention, or future modal is bound to a concrete "
            "action predicate and its local object, timeline, or resource."
        ),
        "does_not_establish": (
            "that the action occurred, is adequate, prevents recurrence, repairs harm, or "
            "constitutes transparent progress over time"
        ),
    },
    "opening_locality_binding": {
        "matched_depth": 3,
        "depth_meaning": "opening-clause predicate to place-entity dependency path",
        "implemented_relation": (
            "A GPE, LOC, or FAC entity in the opening stack is bound through the sentence "
            "dependency graph and checked for recurrence later in the presented ctext."
        ),
        "does_not_establish": (
            "target-market identity, local relevance, impact, outlet fit, or whether the place "
            "is a strategically meaningful angle"
        ),
    },
}


_REPORTING_LEMMAS = {
    "acknowledge",
    "add",
    "announce",
    "argue",
    "assert",
    "comment",
    "confirm",
    "conclude",
    "declare",
    "explain",
    "indicate",
    "note",
    "predict",
    "report",
    "say",
    "state",
    "tell",
    "warn",
    "write",
}
_CREDENTIAL_WORDS = {
    "analyst",
    "ceo",
    "chair",
    "chief",
    "cfo",
    "cofounder",
    "coo",
    "cto",
    "director",
    "doctor",
    "executive",
    "founder",
    "officer",
    "president",
    "professor",
    "researcher",
    "scientist",
    "spokesperson",
    "svp",
    "vice",
    "vp",
}
_ENTITY_LABELS = {"EVENT", "FAC", "GPE", "LOC", "NORP", "ORG", "PERSON", "PRODUCT"}
_EVIDENCE_LABELS = {"CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME"}
_EVENT_WORDS = {
    "announce",
    "ceremony",
    "conference",
    "event",
    "festival",
    "launch",
    "meeting",
    "opening",
    "premiere",
    "summit",
    "tour",
    "webinar",
}
_ACTION_LEMMAS = {
    "apply",
    "attend",
    "book",
    "buy",
    "call",
    "contact",
    "download",
    "email",
    "join",
    "learn",
    "register",
    "request",
    "reserve",
    "subscribe",
    "visit",
}
_CLAIM_LANGUAGE = {
    "amazing",
    "best",
    "breakthrough",
    "cure",
    "groundbreaking",
    "guaranteed",
    "historic",
    "incredible",
    "leading",
    "miracle",
    "revolutionary",
    "unmatched",
    "unprecedented",
    "world-class",
}
_UNCERTAINTY_LEMMAS = {
    "appear",
    "estimate",
    "may",
    "might",
    "potential",
    "preliminary",
    "suggest",
    "uncertain",
}
_COMMITMENT_LEMMAS = {"commit", "intend", "plan", "pledge", "promise"}
_NONACTION_LEMMAS = {"be", "do", "have", "say", "seem"}
_REMEDIAL_ACTIONS = {
    "address",
    "correct",
    "fix",
    "improve",
    "investigate",
    "prevent",
    "protect",
    "restore",
    "review",
    "safeguard",
    "update",
}
_REPORTING_ACTIONS = {"disclose", "publish", "report", "update"}
_REMEDIAL_CONTEXT_RE = re.compile(
    r"\b(?:breach|corrective|failure|fix|harm|incident|investigat|recurrence|"
    r"remed|review|safeguard|security|vulnerabil)\w*\b",
    re.I,
)
_REPORTING_CONTEXT_RE = re.compile(
    r"\b(?:corrective|finding|outcome|performance|progress|public|reform|review|"
    r"safeguard|security|status|transparen)\w*\b",
    re.I,
)
_LIMITATION_RE = re.compile(
    r"\b(?:caveat|confidence interval|early[- ]stage|limited by|limitation|"
    r"not yet|preliminary|subject to|uncertain|unknown)\b",
    re.I,
)
_SIGNIFICANCE_RE = re.compile(
    r"\b(?:because|compared with|compared to|in contrast|means that|more than|"
    r"less than|outperform(?:s|ed|ing)?|resulting in|therefore|which means|"
    r"why (?:this|it) matters)\b",
    re.I,
)
_URL_RE = re.compile(
    r"(?<!@)(?:https?://[^\s<>\]\[(){}\"']+|"
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:com|edu|gov|io|net|org)(?:/[^\s<>\]\[(){}\"']*)?)",
    re.I,
)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\d)"
)
_DATE_SURFACE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?"
    r"|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\.?\s*,?\s*\d{4}"
    r"|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.I,
)


@dataclass(frozen=True)
class SentenceRecord:
    index: int
    start: int
    end: int
    text: str


_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy

        spacy.require_cpu()
        _NLP = spacy.load("en_core_web_sm")
        _NLP.max_length = 2_000_000
    return _NLP


def _round(value: float | int | None, digits: int = 4) -> float | int | None:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _excerpt(text: str, start: int, end: int, *, margin: int = 45) -> str:
    lo = max(0, start - margin)
    hi = min(len(text), end + margin)
    return text[lo:hi]


def _sentences(doc) -> list[SentenceRecord]:
    return [
        SentenceRecord(i, sent.start_char, sent.end_char, sent.text)
        for i, sent in enumerate(doc.sents)
        if sent.text.strip()
    ]


def _sentence_index(sentences: Sequence[SentenceRecord], offset: int) -> int | None:
    for sent in sentences:
        if sent.start <= offset < sent.end:
            return sent.index
    return None


def _sentence_token_span(doc, sentence: SentenceRecord):
    return doc.char_span(sentence.start, sentence.end, alignment_mode="expand")


def _subtree_text(token) -> str:
    tokens = list(token.subtree)
    if not tokens:
        return token.text
    return token.doc[tokens[0].i : tokens[-1].i + 1].text


def _normalize_entity(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value.casefold())).strip()


def _is_credentialed(speaker: str, sentence: str) -> bool:
    words = set(re.findall(r"[a-z]+", f"{speaker} {sentence}".casefold()))
    return bool(words & _CREDENTIAL_WORDS)


def _quote_spans(text: str) -> list[tuple[int, int, str]]:
    patterns = [
        re.compile(r"“([^”]{4,1200})”", re.S),
        re.compile(r'(?<!\w)"([^"\n]{4,1200})"', re.S),
    ]
    spans: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            start, end = match.start(1), match.end(1)
            if any(a < end and start < b for a, b in occupied):
                continue
            spans.append((start, end, match.group(1)))
            occupied.append((start, end))
    return sorted(spans)


def _speaker_for_span(doc, start: int, end: int) -> dict[str, Any] | None:
    window_start = max(0, start - 220)
    window_end = min(len(doc.text), end + 220)
    window = doc.char_span(window_start, window_end, alignment_mode="expand")
    if window is None:
        return None
    candidates: list[tuple[int, dict[str, Any]]] = []
    for token in window:
        if token.lemma_.casefold() not in _REPORTING_LEMMAS or token.pos_ not in {
            "AUX",
            "VERB",
        }:
            continue
        subject = next(
            (
                child
                for child in token.children
                if child.dep_ in {"csubj", "nsubj", "nsubjpass"}
            ),
            None,
        )
        if subject is None:
            continue
        speaker = _subtree_text(subject)
        distance = min(abs(token.idx - start), abs(token.idx - end))
        candidates.append(
            (
                distance,
                {
                    "speaker": speaker,
                    "verb": token.text,
                    "verb_lemma": token.lemma_.casefold(),
                    "binding": "dependency_reporting_verb",
                    "credentialed_in_sentence": _is_credentialed(speaker, token.sent.text),
                    "sentence": token.sent.text,
                },
            )
        )
    if candidates:
        return min(candidates, key=lambda row: row[0])[1]

    nearby_entities = [
        ent
        for ent in doc.ents
        if ent.label_ in {"ORG", "PERSON"}
        and ent.end_char >= window_start
        and ent.start_char <= window_end
    ]
    if not nearby_entities:
        return None
    nearest = min(
        nearby_entities,
        key=lambda ent: min(abs(ent.start_char - end), abs(ent.end_char - start)),
    )
    return {
        "speaker": nearest.text,
        "verb": None,
        "verb_lemma": None,
        "binding": "entity_proximity_only",
        "credentialed_in_sentence": _is_credentialed(nearest.text, nearest.sent.text),
        "sentence": nearest.sent.text,
    }


def _claim_like(sentence_span) -> bool:
    if sentence_span is None or len(sentence_span) < 4:
        return False
    root = sentence_span.root
    has_subject = any(
        token.dep_ in {"csubj", "expl", "nsubj", "nsubjpass"}
        for token in sentence_span
    )
    return root.pos_ in {"ADJ", "AUX", "NOUN", "VERB"} and has_subject


def _resources(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, pattern in (("url", _URL_RE), ("email", _EMAIL_RE), ("phone", _PHONE_RE)):
        for match in pattern.finditer(text):
            value = match.group(0).rstrip(".,;:")
            rows.append(
                {
                    "kind": kind,
                    "value": value,
                    "start": match.start(),
                    "end": match.start() + len(value),
                }
            )
    return sorted(rows, key=lambda row: (row["start"], row["kind"]))


def _pack_result(
    relation_id: str,
    *,
    status: str,
    summary: Mapping[str, Any],
    witnesses: Sequence[Mapping[str, Any]],
    realized_depth: int | None = None,
) -> dict[str, Any]:
    spec = RELATION_SPECS[relation_id]
    return {
        "relation_id": relation_id,
        "status": status,
        "program_relation_depth_ceiling": spec["matched_depth"],
        "matched_relation_depth": realized_depth,
        "realized_depth": realized_depth,
        "depth_meaning": spec["depth_meaning"],
        "implemented_relation": spec["implemented_relation"],
        "does_not_establish": spec["does_not_establish"],
        "summary": dict(summary),
        "witness_count": len(witnesses),
        "witnesses": list(witnesses)[:12],
        "absence_certificate": False,
    }


def _attribution_relations(
    text: str,
    doc,
    sentences: Sequence[SentenceRecord],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    quote_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    graph = nx.DiGraph()

    for index, (start, end, quote) in enumerate(_quote_spans(text)):
        sentence_index = _sentence_index(sentences, start)
        speaker = _speaker_for_span(doc, start, end)
        quote_node = f"quote:{index}"
        graph.add_node(quote_node, kind="quote")
        binding = None if speaker is None else speaker["binding"]
        if speaker is not None:
            speaker_key = _normalize_entity(speaker["speaker"]) or speaker["speaker"]
            speaker_node = f"speaker:{speaker_key}"
            graph.add_node(speaker_node, kind="speaker", text=speaker["speaker"])
            graph.add_edge(speaker_node, quote_node, binding=binding)
        quote_rows.append(
            {
                "quote_index": index,
                "start": start,
                "end": end,
                "sentence_index": sentence_index,
                "word_count": len(re.findall(r"\b\w+\b", quote)),
                "multi_sentence_or_long_block": quote.count(".") > 1 or len(quote) > 320,
                "speaker": None if speaker is None else speaker["speaker"],
                "speaker_binding": binding,
                "credentialed_in_sentence": (
                    None if speaker is None else speaker["credentialed_in_sentence"]
                ),
                "exact_quote": quote,
            }
        )
        if speaker is not None and binding == "dependency_reporting_verb":
            attribution_rows.append(
                {
                    "kind": "direct_quote",
                    "speaker": speaker["speaker"],
                    "reporting_verb": speaker["verb"],
                    "claim_start": start,
                    "claim_end": end,
                    "exact_claim": quote,
                    "sentence_index": sentence_index,
                    "credentialed_in_sentence": speaker["credentialed_in_sentence"],
                }
            )

    for token in doc:
        if token.lemma_.casefold() not in _REPORTING_LEMMAS or token.pos_ not in {
            "AUX",
            "VERB",
        }:
            continue
        subject = next(
            (
                child
                for child in token.children
                if child.dep_ in {"csubj", "nsubj", "nsubjpass"}
            ),
            None,
        )
        complement = next(
            (child for child in token.children if child.dep_ in {"ccomp", "xcomp"}),
            None,
        )
        if subject is None or complement is None:
            continue
        claim_tokens = list(complement.subtree)
        if not claim_tokens:
            continue
        claim_start = claim_tokens[0].idx
        claim_end = claim_tokens[-1].idx + len(claim_tokens[-1].text)
        speaker = _subtree_text(subject)
        if any(
            row["speaker"] == speaker
            and int(row["claim_start"]) <= claim_start
            and claim_end <= int(row["claim_end"])
            for row in attribution_rows
        ):
            continue
        claim = text[claim_start:claim_end]
        sentence_index = _sentence_index(sentences, claim_start)
        attribution_rows.append(
            {
                "kind": "governed_clause",
                "speaker": speaker,
                "reporting_verb": token.text,
                "claim_start": claim_start,
                "claim_end": claim_end,
                "exact_claim": claim,
                "sentence_index": sentence_index,
                "credentialed_in_sentence": _is_credentialed(speaker, token.sent.text),
            }
        )
        speaker_node = f"speaker:{_normalize_entity(speaker) or speaker}"
        claim_node = f"claim:{claim_start}:{claim_end}"
        graph.add_node(speaker_node, kind="speaker", text=speaker)
        graph.add_node(claim_node, kind="claim")
        graph.add_edge(speaker_node, claim_node, binding="dependency_reporting_verb")

    direct_quote_bindings = sum(
        row["speaker_binding"] == "dependency_reporting_verb" for row in quote_rows
    )
    quote_result = _pack_result(
        "quote_integration_structure",
        status="measured" if quote_rows else "relation_not_instantiated",
        summary={
            "quotation_spans": len(quote_rows),
            "dependency_bound_quotation_spans": direct_quote_bindings,
            "proximity_only_speaker_candidates": sum(
                row["speaker_binding"] == "entity_proximity_only" for row in quote_rows
            ),
            "long_or_multisentence_quote_spans": sum(
                bool(row["multi_sentence_or_long_block"]) for row in quote_rows
            ),
            "distinct_bound_speakers": len(
                {row["speaker"] for row in quote_rows if row["speaker"]}
            ),
        },
        witnesses=quote_rows,
        realized_depth=2 if quote_rows else None,
    )
    attribution_result = _pack_result(
        "attribution_claim_binding",
        status="witnessed" if attribution_rows else "relation_not_instantiated",
        summary={
            "dependency_bound_claims": len(attribution_rows),
            "direct_quote_claims": sum(row["kind"] == "direct_quote" for row in attribution_rows),
            "governed_clause_claims": sum(
                row["kind"] == "governed_clause" for row in attribution_rows
            ),
            "distinct_speakers": len({row["speaker"] for row in attribution_rows}),
            "credentialed_bindings": sum(
                bool(row["credentialed_in_sentence"]) for row in attribution_rows
            ),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
        },
        witnesses=attribution_rows,
        realized_depth=3 if attribution_rows else None,
    )
    return attribution_result, quote_result, attribution_rows, quote_rows


def _entity_evidence_relation(
    text: str,
    doc,
    sentences: Sequence[SentenceRecord],
    resources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    graph = nx.Graph()
    sent_features: dict[int, dict[str, Any]] = {
        sent.index: {"entities": set(), "evidence": [], "resources": []}
        for sent in sentences
    }
    for sent in sentences:
        graph.add_node(f"sentence:{sent.index}", kind="sentence", index=sent.index)
        if sent.index + 1 < len(sentences):
            graph.add_edge(
                f"sentence:{sent.index}",
                f"sentence:{sent.index + 1}",
                kind="adjacent_sentence",
                weight=1,
            )

    for ent_index, ent in enumerate(doc.ents):
        sent_index = _sentence_index(sentences, ent.start_char)
        if sent_index is None:
            continue
        if ent.label_ in _ENTITY_LABELS:
            entity_key = f"entity:{ent.label_}:{_normalize_entity(ent.text)}"
            graph.add_node(entity_key, kind="entity", label=ent.label_, text=ent.text)
            graph.add_edge(f"sentence:{sent_index}", entity_key, kind="contains", weight=2)
            sent_features[sent_index]["entities"].add((ent.label_, ent.text))
        elif ent.label_ in _EVIDENCE_LABELS:
            evidence_key = f"evidence:{ent_index}:{ent.start_char}"
            graph.add_node(evidence_key, kind="evidence", label=ent.label_, text=ent.text)
            graph.add_edge(f"sentence:{sent_index}", evidence_key, kind="contains", weight=2)
            sent_features[sent_index]["evidence"].append(
                {
                    "kind": ent.label_.casefold(),
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )

    for resource_index, resource in enumerate(resources):
        sent_index = _sentence_index(sentences, int(resource["start"]))
        if sent_index is None:
            continue
        key = f"resource:{resource_index}:{resource['start']}"
        graph.add_node(key, kind="evidence", label=resource["kind"], text=resource["value"])
        graph.add_edge(f"sentence:{sent_index}", key, kind="contains", weight=2)
        sent_features[sent_index]["resources"].append(dict(resource))
        sent_features[sent_index]["evidence"].append(dict(resource))

    witnesses: list[dict[str, Any]] = []
    for sent in sentences:
        features = sent_features[sent.index]
        if not features["entities"] or not features["evidence"]:
            continue
        for label, entity in sorted(features["entities"]):
            witnesses.append(
                {
                    "path_kind": "same_sentence",
                    "sentence_index": sent.index,
                    "entity": entity,
                    "entity_label": label,
                    "evidence": [row["text"] if "text" in row else row["value"] for row in features["evidence"][:5]],
                    "exact_sentence": sent.text,
                }
            )

    for left in range(len(sentences) - 1):
        right = left + 1
        left_features = sent_features[left]
        right_features = sent_features[right]
        for entity_side, evidence_side, direction in (
            (left_features, right_features, "entity_then_evidence"),
            (right_features, left_features, "evidence_then_entity"),
        ):
            if not entity_side["entities"] or not evidence_side["evidence"]:
                continue
            shared = entity_side["entities"] & evidence_side["entities"]
            if not shared:
                continue
            witnesses.append(
                {
                    "path_kind": "adjacent_sentence_shared_entity",
                    "direction": direction,
                    "sentence_indices": [left, right],
                    "shared_entities": [text_value for _label, text_value in sorted(shared)],
                    "evidence": [
                        row.get("text", row.get("value"))
                        for row in evidence_side["evidence"][:5]
                    ],
                }
            )

    evidence_sentence_count = sum(bool(row["evidence"]) for row in sent_features.values())
    entity_sentence_count = sum(bool(row["entities"]) for row in sent_features.values())
    components = list(nx.connected_components(graph)) if graph.number_of_nodes() else []
    result = _pack_result(
        "entity_evidence_graph",
        status="witnessed" if witnesses else "relation_not_instantiated",
        summary={
            "sentences": len(sentences),
            "entity_bearing_sentences": entity_sentence_count,
            "evidence_bearing_sentences": evidence_sentence_count,
            "same_or_adjacent_entity_evidence_paths": len(witnesses),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "graph_components": len(components),
        },
        witnesses=witnesses,
        realized_depth=(
            3
            if any(row["path_kind"] == "adjacent_sentence_shared_entity" for row in witnesses)
            else 2
            if witnesses
            else None
        ),
    )
    return result, sent_features


def _claim_evidence_relation(
    doc,
    sentences: Sequence[SentenceRecord],
    sent_features: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    claim_indices = [
        sent.index
        for sent in sentences
        if _claim_like(_sentence_token_span(doc, sent))
    ]
    evidence_indices = [
        index for index, features in sent_features.items() if features["evidence"]
    ]
    if not claim_indices or not evidence_indices:
        return _pack_result(
            "claim_evidence_alignment",
            status="relation_not_instantiated",
            summary={
                "claim_like_sentences": len(claim_indices),
                "evidence_bearing_sentences": len(evidence_indices),
                "alignment_edges": 0,
            },
            witnesses=[],
        )

    texts = [sent.text for sent in sentences]
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
        similarities = (matrix @ matrix.T).toarray()
    except ValueError:
        similarities = None
    witnesses: list[dict[str, Any]] = []
    graph = nx.Graph()
    for claim_index in claim_indices:
        graph.add_node(f"claim:{claim_index}", kind="claim")
        claim_entities = {
            _normalize_entity(value)
            for _label, value in sent_features[claim_index]["entities"]
        }
        if claim_index in evidence_indices:
            witnesses.append(
                {
                    "claim_sentence_index": claim_index,
                    "evidence_sentence_index": claim_index,
                    "alignment_kind": "same_sentence",
                    "tfidf_similarity": 1.0,
                    "shared_entities": sorted(claim_entities),
                    "exact_claim_sentence": sentences[claim_index].text,
                }
            )
            graph.add_edge(f"claim:{claim_index}", f"evidence:{claim_index}", weight=1.0)
            continue
        candidates: list[tuple[float, int, set[str]]] = []
        for evidence_index in evidence_indices:
            evidence_entities = {
                _normalize_entity(value)
                for _label, value in sent_features[evidence_index]["entities"]
            }
            shared = claim_entities & evidence_entities
            similarity = (
                0.0 if similarities is None else float(similarities[claim_index, evidence_index])
            )
            relation_score = similarity + min(0.6, 0.3 * len(shared))
            if shared or similarity >= 0.18:
                candidates.append((relation_score, evidence_index, shared))
        if not candidates:
            continue
        relation_score, evidence_index, shared = max(candidates)
        witnesses.append(
            {
                "claim_sentence_index": claim_index,
                "evidence_sentence_index": evidence_index,
                "alignment_kind": "shared_entity_or_tfidf_retrieval",
                "alignment_score": _round(relation_score),
                "tfidf_similarity": (
                    None
                    if similarities is None
                    else _round(float(similarities[claim_index, evidence_index]))
                ),
                "shared_entities": sorted(shared),
                "exact_claim_sentence": sentences[claim_index].text,
                "exact_evidence_sentence": sentences[evidence_index].text,
            }
        )
        graph.add_edge(
            f"claim:{claim_index}",
            f"evidence:{evidence_index}",
            weight=relation_score,
        )

    return _pack_result(
        "claim_evidence_alignment",
        status="witnessed" if witnesses else "relation_not_instantiated",
        summary={
            "claim_like_sentences": len(claim_indices),
            "evidence_bearing_sentences": len(evidence_indices),
            "alignment_edges": len(witnesses),
            "same_sentence_edges": sum(
                row["alignment_kind"] == "same_sentence" for row in witnesses
            ),
            "retrieval_edges": sum(
                row["alignment_kind"] == "shared_entity_or_tfidf_retrieval"
                for row in witnesses
            ),
            "unlinked_claim_like_sentences": len(claim_indices) - len(witnesses),
        },
        witnesses=witnesses,
        realized_depth=(
            3
            if any(
                row["alignment_kind"] == "shared_entity_or_tfidf_retrieval"
                for row in witnesses
            )
            else 2
            if witnesses
            else None
        ),
    )


def _date_quantity_relation(
    text: str,
    doc,
    sentences: Sequence[SentenceRecord],
) -> dict[str, Any]:
    import datetime

    from dateutil import parser as date_parser

    date_rows: list[dict[str, Any]] = []
    previous = None
    for match in _DATE_SURFACE_RE.finditer(text):
        raw = match.group(0)
        try:
            # A deterministic leap-year default is required for yearless month/day
            # surfaces.  Invalid surfaces such as April 31 are retained as typed
            # counter-witnesses instead of being silently dropped.
            parsed = date_parser.parse(
                raw,
                fuzzy=False,
                default=datetime.datetime(2000, 1, 1),
            )
        except Exception as error:
            date_rows.append(
                {
                    "surface": raw,
                    "start": match.start(),
                    "end": match.end(),
                    "valid_calendar_date": False,
                    "error_type": type(error).__name__,
                }
            )
            continue
        current = parsed.date()
        delta = None if previous is None else (current - previous).days
        date_rows.append(
            {
                "surface": raw,
                "start": match.start(),
                "end": match.end(),
                "valid_calendar_date": True,
                "normalized_date": current.isoformat(),
                "days_since_previous_text_order_date": delta,
                "sentence_index": _sentence_index(sentences, match.start()),
            }
        )
        previous = current

    quantity_rows: list[dict[str, Any]] = []
    for ent in doc.ents:
        if ent.label_ not in {"CARDINAL", "MONEY", "ORDINAL", "PERCENT", "QUANTITY"}:
            continue
        root = ent.root
        quantity_rows.append(
            {
                "surface": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "head": root.head.text if root.head is not root else root.text,
                "dependency": root.dep_,
                "sentence_index": _sentence_index(sentences, ent.start_char),
                "context": _excerpt(text, ent.start_char, ent.end_char),
            }
        )
    arithmetic_rows = number_consistency(text)
    witnesses: list[dict[str, Any]] = []
    witnesses.extend(
        {
            **row,
            "arithmetic_relation_kind": row["kind"],
            "kind": "arithmetic_recomputation",
        }
        for row in arithmetic_rows
    )
    witnesses.extend(
        {"kind": "invalid_calendar_surface", **row}
        for row in date_rows
        if not row["valid_calendar_date"]
    )
    witnesses.extend(
        {"kind": "date_context_binding", **row}
        for row in date_rows[:4]
        if row["valid_calendar_date"]
    )
    witnesses.extend({"kind": "quantity_head_binding", **row} for row in quantity_rows[:4])
    applicable = bool(date_rows or quantity_rows or arithmetic_rows)
    realized_depth = 3 if arithmetic_rows else 2 if applicable else None
    return _pack_result(
        "date_quantity_internal_consistency",
        status="measured" if applicable else "relation_not_instantiated",
        summary={
            "date_surfaces": len(date_rows),
            "valid_calendar_dates": sum(row["valid_calendar_date"] for row in date_rows),
            "invalid_calendar_date_surfaces": sum(
                not row["valid_calendar_date"] for row in date_rows
            ),
            "text_order_date_reversals": sum(
                isinstance(row.get("days_since_previous_text_order_date"), int)
                and row["days_since_previous_text_order_date"] < 0
                for row in date_rows
            ),
            "quantity_mentions": len(quantity_rows),
            "rederived_arithmetic_relations": len(arithmetic_rows),
            "arithmetic_consistent": sum(
                bool(row.get("consistent")) for row in arithmetic_rows
            ),
            "arithmetic_inconsistent": sum(
                row.get("consistent") is False for row in arithmetic_rows
            ),
        },
        witnesses=witnesses,
        realized_depth=realized_depth,
    )


def _resource_role(sentence: str) -> str | None:
    low = sentence.casefold()
    role_cues = (
        ("source", ("according to", "data", "report", "research", "source", "study")),
        ("contact", ("contact", "email", "media inquiries", "press contact")),
        ("asset", ("b-roll", "download", "image", "media kit", "photo", "video")),
        ("social", ("facebook", "instagram", "linkedin", "social", "twitter", "x.com")),
        ("action", ("apply", "buy", "learn more", "register", "subscribe", "tickets", "visit")),
        ("organization", ("about", "investor relations", "newsroom", "website")),
    )
    for role, cues in role_cues:
        if any(cue in low for cue in cues):
            return role
    return None


_ROLE_CUE_LEMMAS = {
    "source": {"accord", "data", "report", "research", "source", "study"},
    "contact": {"contact", "email", "inquiry", "media", "press"},
    "asset": {"download", "image", "kit", "media", "photo", "video"},
    "social": {"facebook", "instagram", "linkedin", "social", "twitter"},
    "action": _ACTION_LEMMAS | {"ticket"},
    "organization": {"about", "investor", "newsroom", "website"},
}


def _dependency_binding(span, left_token, candidate_tokens) -> dict[str, Any] | None:
    graph = nx.Graph()
    token_ids = {token.i for token in span}
    for token in span:
        graph.add_node(token.i)
        if token.head.i != token.i and token.head.i in token_ids:
            graph.add_edge(token.i, token.head.i)
    candidates = []
    for candidate in candidate_tokens:
        try:
            path = nx.shortest_path(graph, left_token.i, candidate.i)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        candidates.append((len(path) - 1, candidate, path))
    if not candidates:
        return None
    distance, candidate, path = min(candidates, key=lambda row: row[0])
    return {
        "bound_token": candidate.text,
        "bound_lemma": candidate.lemma_,
        "dependency_distance": distance,
        "dependency_path": [span.doc[index].text for index in path],
    }


def _url_relation(
    text: str,
    doc,
    sentences: Sequence[SentenceRecord],
    resources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    witnesses: list[dict[str, Any]] = []
    url_rows: list[dict[str, Any]] = []
    for resource in resources:
        if resource["kind"] != "url":
            continue
        sentence_index = _sentence_index(sentences, int(resource["start"]))
        if sentence_index is None:
            continue
        sentence = sentences[sentence_index]
        role = _resource_role(sentence.text)
        value = str(resource["value"])
        parsed_value = value if "://" in value else f"https://{value}"
        parsed = urlsplit(parsed_value)
        span = _sentence_token_span(doc, sentence)
        dependency_binding = None
        if span is not None and role is not None:
            resource_token = min(
                span,
                key=lambda token: min(
                    abs(token.idx - int(resource["start"])),
                    abs(token.idx + len(token.text) - int(resource["end"])),
                ),
            )
            cue_tokens = [
                token
                for token in span
                if token.lemma_.casefold() in _ROLE_CUE_LEMMAS[role]
                and token is not resource_token
            ]
            dependency_binding = _dependency_binding(
                span,
                resource_token,
                cue_tokens,
            )
        row = {
            "url": value,
            "domain": parsed.netloc.casefold(),
            "path": parsed.path,
            "start": resource["start"],
            "end": resource["end"],
            "sentence_index": sentence_index,
            "bound_role": role,
            "dependency_binding": dependency_binding,
            "exact_clause": sentence.text,
            "network_resolution_attempted": False,
        }
        url_rows.append(row)
        if role is not None and dependency_binding is not None:
            witnesses.append(row)
    return (
        _pack_result(
            "url_role_clause_binding",
            status="witnessed" if witnesses else (
                "untyped_or_unbound_url_present"
                if url_rows
                else "relation_not_instantiated"
            ),
            summary={
                "url_spans": len(url_rows),
                "role_cue_urls": sum(row["bound_role"] is not None for row in url_rows),
                "dependency_bound_urls": len(witnesses),
                "untyped_or_unbound_urls": len(url_rows) - len(witnesses),
                "role_counts": dict(
                    sorted(Counter(row["bound_role"] for row in witnesses).items())
                ),
                "network_resolution_attempted": False,
            },
            witnesses=witnesses,
            realized_depth=3 if witnesses else 2 if url_rows else None,
        ),
        url_rows,
    )


def _opening_relation(
    doc,
    sentences: Sequence[SentenceRecord],
    sent_features: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if len(sentences) < 3:
        return _pack_result(
            "opening_information_graph_alignment",
            status="insufficient_sentence_structure",
            summary={"sentences": len(sentences)},
            witnesses=[],
        )
    texts = [sent.text for sent in sentences]
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
        similarity = (matrix @ matrix.T).toarray()
    except ValueError:
        similarity = None
    graph = nx.Graph()
    graph.add_nodes_from(range(len(sentences)))
    if similarity is not None:
        for left in range(len(sentences)):
            for right in range(left + 1, len(sentences)):
                weight = float(similarity[left, right])
                if weight >= 0.08:
                    graph.add_edge(left, right, weight=weight)
    centrality = (
        nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {}
    )
    top_indices = sorted(centrality, key=centrality.get, reverse=True)[: min(5, len(sentences))]
    opening_count = min(2, max(1, math.ceil(0.15 * len(sentences))))
    opening_indices = set(range(opening_count))
    central_in_opening = [index for index in top_indices if index in opening_indices]

    entity_sentence_counts: Counter[str] = Counter()
    entity_first: dict[str, int] = {}
    for index, features in sent_features.items():
        for _label, value in features["entities"]:
            key = _normalize_entity(value)
            if not key:
                continue
            entity_sentence_counts[key] += 1
            entity_first.setdefault(key, index)
    central_entities = [key for key, _count in entity_sentence_counts.most_common(8)]
    opening_entities = [key for key in central_entities if entity_first[key] < opening_count]
    witnesses = [
        {
            "kind": "central_sentence_in_opening",
            "sentence_index": index,
            "pagerank": _round(centrality[index]),
            "exact_sentence": sentences[index].text,
        }
        for index in central_in_opening
    ]
    witnesses.extend(
        {
            "kind": "central_entity_first_mentioned_in_opening",
            "entity": entity,
            "first_sentence_index": entity_first[entity],
            "sentence_frequency": entity_sentence_counts[entity],
        }
        for entity in opening_entities
    )
    return _pack_result(
        "opening_information_graph_alignment",
        status="measured",
        summary={
            "sentences": len(sentences),
            "opening_sentence_count": opening_count,
            "sentence_graph_edges": graph.number_of_edges(),
            "sentence_graph_centrality_available": bool(centrality),
            "top_central_sentence_indices": top_indices,
            "top_central_sentences_in_opening": len(central_in_opening),
            "central_entities": len(central_entities),
            "central_entities_first_mentioned_in_opening": len(opening_entities),
        },
        witnesses=witnesses,
        realized_depth=3 if centrality else 2 if central_entities else None,
    )


def _dependency_depth(token) -> int:
    depth = 0
    seen = set()
    current = token
    while current.head is not current and current.i not in seen:
        seen.add(current.i)
        current = current.head
        depth += 1
    return depth


def _readability_relation(doc, sentences: Sequence[SentenceRecord]) -> dict[str, Any]:
    sentence_lengths: list[int] = []
    dependency_depths: list[int] = []
    passive_sentences = 0
    acronym_bindings: list[dict[str, Any]] = []
    for sentence in sentences:
        span = _sentence_token_span(doc, sentence)
        if span is None:
            continue
        words = [token for token in span if token.is_alpha]
        sentence_lengths.append(len(words))
        dependency_depths.extend(_dependency_depth(token) for token in span)
        passive_sentences += any(
            token.dep_ in {"auxpass", "nsubjpass"} or token.morph.get("Voice") == ["Pass"]
            for token in span
        )
        for match in re.finditer(r"\b([A-Z][A-Za-z -]{3,80})\s+\(([A-Z]{2,10})\)", sentence.text):
            acronym_bindings.append(
                {
                    "sentence_index": sentence.index,
                    "expanded_form": match.group(1).strip(),
                    "acronym": match.group(2),
                    "exact_sentence": sentence.text,
                }
            )
    words = [token for token in doc if token.is_alpha]
    long_words = [token for token in words if len(token.text) >= 10]
    witnesses: list[dict[str, Any]] = list(acronym_bindings)
    longest = sorted(
        zip(sentence_lengths, sentences, strict=False), key=lambda row: row[0], reverse=True
    )[:3]
    witnesses.extend(
        {
            "kind": "long_sentence_structure",
            "sentence_index": sentence.index,
            "word_count": length,
            "exact_sentence": sentence.text,
        }
        for length, sentence in longest
        if length >= 30
    )
    return _pack_result(
        "sentence_dependency_readability",
        status="measured" if sentences else "insufficient_sentence_structure",
        summary={
            "sentences": len(sentence_lengths),
            "mean_words_per_sentence": _round(
                sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else None
            ),
            "median_words_per_sentence": (
                None
                if not sentence_lengths
                else _round(sorted(sentence_lengths)[len(sentence_lengths) // 2])
            ),
            "maximum_words_per_sentence": max(sentence_lengths) if sentence_lengths else None,
            "mean_dependency_depth": _round(
                sum(dependency_depths) / len(dependency_depths) if dependency_depths else None
            ),
            "maximum_dependency_depth": max(dependency_depths) if dependency_depths else None,
            "passive_sentence_fraction": _round(
                passive_sentences / len(sentence_lengths) if sentence_lengths else None
            ),
            "long_token_fraction": _round(len(long_words) / len(words) if words else None),
            "explicit_acronym_definition_bindings": len(acronym_bindings),
        },
        witnesses=witnesses,
        realized_depth=2 if sentences else None,
    )


def _action_sentences(doc, sentences: Sequence[SentenceRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sentence in sentences:
        span = _sentence_token_span(doc, sentence)
        if span is None:
            continue
        roots = [token for token in span if token.dep_ == "ROOT"]
        action_tokens = [
            token
            for token in span
            if token.lemma_.casefold() in _ACTION_LEMMAS
            and token.pos_ in {"AUX", "VERB"}
        ]
        if not action_tokens:
            continue
        imperative = any(
            token in roots and not any(child.dep_ in {"nsubj", "nsubjpass"} for child in token.children)
            for token in action_tokens
        )
        explicit = bool(
            re.search(r"\b(?:for more|please|to (?:apply|buy|learn|register|subscribe|visit))\b", sentence.text, re.I)
        )
        if not imperative and not explicit:
            continue
        rows.append(
            {
                "sentence_index": sentence.index,
                "action_lemmas": sorted({token.lemma_.casefold() for token in action_tokens}),
                "imperative_root": imperative,
                "explicit_action_phrase": explicit,
                "exact_sentence": sentence.text,
            }
        )
    return rows


def _cta_relation(
    doc,
    sentences: Sequence[SentenceRecord],
    resources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actions = _action_sentences(doc, sentences)
    resource_by_sentence: dict[int, list[Mapping[str, Any]]] = {}
    for resource in resources:
        sentence_index = _sentence_index(sentences, int(resource["start"]))
        if sentence_index is not None:
            resource_by_sentence.setdefault(sentence_index, []).append(resource)
    witnesses: list[dict[str, Any]] = []
    for action in actions:
        action_index = action["sentence_index"]
        candidates = [
            resource
            for index in (action_index - 1, action_index, action_index + 1)
            for resource in resource_by_sentence.get(index, [])
        ]
        if not candidates:
            continue
        witnesses.append(
            {
                **action,
                "bound_resources": [
                    {"kind": resource["kind"], "value": resource["value"]}
                    for resource in candidates[:5]
                ],
                "binding_distance_sentences": min(
                    abs(
                        action_index
                        - int(_sentence_index(sentences, int(resource["start"])) or 0)
                    )
                    for resource in candidates
                ),
            }
        )
    return _pack_result(
        "cta_resource_binding",
        status="witnessed" if witnesses else (
            "unbound_action_present" if actions else "relation_not_instantiated"
        ),
        summary={
            "action_clauses": len(actions),
            "resource_bound_action_clauses": len(witnesses),
            "unbound_action_clauses": len(actions) - len(witnesses),
            "distinct_action_lemmas": sorted(
                {lemma for row in actions for lemma in row["action_lemmas"]}
            ),
        },
        witnesses=witnesses,
        realized_depth=3 if witnesses else 2 if actions else None,
    )


def _boilerplate_relation(
    doc,
    sentences: Sequence[SentenceRecord],
    resources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    about_re = re.compile(
        r"\b(?:ABOUT|About)\s+(?:(?:THE|The)\s+)?[A-Z][A-Za-z0-9&.'-]*"
    )
    other_cue_re = re.compile(
        r"\b(?:investor relations|media contacts?|press contacts?|"
        r"for media inquiries|newsroom)\b",
        re.I,
    )
    cue_rows: list[dict[str, Any]] = []
    for sentence in sentences:
        if not about_re.search(sentence.text) and not other_cue_re.search(sentence.text):
            continue
        entities = [
            ent.text
            for ent in doc.ents
            if ent.label_ == "ORG"
            and ent.start_char >= sentence.start
            and ent.end_char <= sentence.end
        ]
        nearby_resources = [
            {"kind": resource["kind"], "value": resource["value"]}
            for resource in resources
            if sentence.start - 120 <= resource["start"] <= sentence.end + 240
        ]
        cue_rows.append(
            {
                "sentence_index": sentence.index,
                "relative_position": _round(sentence.index / max(1, len(sentences) - 1)),
                "organizations": entities,
                "nearby_resources": nearby_resources[:6],
                "exact_sentence": sentence.text,
            }
        )
    return _pack_result(
        "boilerplate_contact_structure",
        status="witnessed" if cue_rows else "relation_not_instantiated",
        summary={
            "boilerplate_or_contact_cues": len(cue_rows),
            "cues_with_organization": sum(bool(row["organizations"]) for row in cue_rows),
            "cues_with_resource": sum(bool(row["nearby_resources"]) for row in cue_rows),
            "cues_in_final_third": sum(
                isinstance(row["relative_position"], float)
                and row["relative_position"] >= 2 / 3
                for row in cue_rows
            ),
        },
        witnesses=cue_rows,
        realized_depth=2 if cue_rows else None,
    )


def _scannability_relation(text: str) -> dict[str, Any]:
    lines = text.split("\n")
    nonempty = [(index, line) for index, line in enumerate(lines) if line.strip()]
    headings = [
        (index, line.strip())
        for index, line in nonempty
        if (
            line.strip().endswith(":")
            or (
                len(line.strip()) <= 100
                and re.search(r"[A-Z]", line)
                and line.strip() == line.strip().upper()
            )
        )
    ]
    bullets = [
        (index, line.strip())
        for index, line in nonempty
        if re.match(r"\s*(?:[-*•]|\d+[.)])\s+", line)
    ]
    paragraphs = [block for block in re.split(r"\n\s*\n", text) if block.strip()]
    witnesses = [
        {"kind": "heading", "line_index": index, "text": line}
        for index, line in headings[:8]
    ]
    witnesses.extend(
        {"kind": "list_item", "line_index": index, "text": line}
        for index, line in bullets[:8]
    )
    preserved_structure = len(lines) > 1
    return _pack_result(
        "section_scannability_structure",
        status="measured" if preserved_structure else "representation_not_instantiated",
        summary={
            "newline_characters": text.count("\n"),
            "nonempty_lines": len(nonempty),
            "paragraph_blocks": len(paragraphs),
            "heading_like_lines": len(headings),
            "list_item_lines": len(bullets),
            "short_line_fraction": _round(
                sum(len(line.split()) <= 12 for _index, line in nonempty) / len(nonempty)
                if nonempty
                else None
            ),
            "source_layout_recoverable": False,
        },
        witnesses=witnesses,
        realized_depth=2 if preserved_structure else None,
    )


def _event_relation(doc, sentences: Sequence[SentenceRecord]) -> dict[str, Any]:
    witnesses: list[dict[str, Any]] = []
    for sentence in sentences:
        span = _sentence_token_span(doc, sentence)
        if span is None:
            continue
        event_token_objects = [
            token
            for token in span
            if token.lemma_.casefold() in _EVENT_WORDS
        ]
        if not event_token_objects:
            continue
        entities = [
            ent
            for ent in doc.ents
            if ent.start_char >= sentence.start and ent.end_char <= sentence.end
        ]
        entity_groups = {
            "actor": [ent for ent in entities if ent.label_ in {"ORG", "PERSON"}],
            "date_or_time": [ent for ent in entities if ent.label_ in {"DATE", "TIME"}],
            "place": [ent for ent in entities if ent.label_ in {"FAC", "GPE", "LOC"}],
        }
        if not all(entity_groups.values()):
            continue
        dependency_graph = nx.Graph()
        sentence_token_ids = {member.i for member in span}
        for token in span:
            dependency_graph.add_node(token.i, text=token.text)
            if token.head.i != token.i and token.head.i in sentence_token_ids:
                dependency_graph.add_edge(token.i, token.head.i)
        bound_paths: dict[str, dict[str, Any]] = {}
        for group_name, group_entities in entity_groups.items():
            candidates = []
            for event_token in event_token_objects:
                for entity in group_entities:
                    try:
                        path = nx.shortest_path(
                            dependency_graph,
                            event_token.i,
                            entity.root.i,
                        )
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                    candidates.append((len(path) - 1, event_token, entity, path))
            if not candidates:
                break
            distance, event_token, entity, path = min(candidates, key=lambda row: row[0])
            bound_paths[group_name] = {
                "event_token": event_token.text,
                "entity": entity.text,
                "entity_label": entity.label_,
                "dependency_distance": distance,
                "dependency_path": [doc[index].text for index in path],
            }
        if set(bound_paths) != set(entity_groups):
            continue
        witnesses.append(
            {
                "sentence_index": sentence.index,
                "event_tokens": [token.text for token in event_token_objects],
                "bound_dependency_paths": bound_paths,
                "exact_sentence": sentence.text,
            }
        )
    return _pack_result(
        "event_logistics_binding",
        status="witnessed" if witnesses else "relation_not_instantiated",
        summary={
            "complete_actor_date_place_event_bindings": len(witnesses),
            "opening_bindings": sum(row["sentence_index"] <= 1 for row in witnesses),
        },
        witnesses=witnesses,
        realized_depth=3 if witnesses else None,
    )


def _claim_language_relation(
    doc,
    sentences: Sequence[SentenceRecord],
    attributions: Sequence[Mapping[str, Any]],
    quote_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    witnesses: list[dict[str, Any]] = []
    attribution_spans = [
        (int(row["claim_start"]), int(row["claim_end"]), row["speaker"])
        for row in attributions
    ]
    quote_spans = [
        (int(row["start"]), int(row["end"]), row.get("speaker")) for row in quote_rows
    ]
    for token in doc:
        cue = token.lemma_.casefold()
        surface = token.text.casefold()
        if cue not in _CLAIM_LANGUAGE and surface not in _CLAIM_LANGUAGE:
            continue
        scope = "document_voice"
        speaker = None
        for start, end, bound_speaker in attribution_spans:
            if start <= token.idx < end:
                scope = "dependency_bound_speaker"
                speaker = bound_speaker
                break
        if scope == "document_voice":
            for start, end, bound_speaker in quote_spans:
                if start <= token.idx < end:
                    scope = "quoted_bound_speaker" if bound_speaker else "unresolved_quote"
                    speaker = bound_speaker
                    break
        sentence_index = _sentence_index(sentences, token.idx)
        witnesses.append(
            {
                "cue": token.text,
                "start": token.idx,
                "end": token.idx + len(token.text),
                "scope": scope,
                "speaker": speaker,
                "sentence_index": sentence_index,
                "exact_sentence": token.sent.text,
            }
        )
    return _pack_result(
        "attribution_scoped_claim_language",
        status="witnessed" if witnesses else "relation_not_instantiated",
        summary={
            "closed_vocabulary_cues": len(witnesses),
            "scope_counts": dict(sorted(Counter(row["scope"] for row in witnesses).items())),
            "closed_vocabulary": sorted(_CLAIM_LANGUAGE),
        },
        witnesses=witnesses,
        realized_depth=3 if witnesses else None,
    )


def _significance_relation(doc, sentences: Sequence[SentenceRecord]) -> dict[str, Any]:
    witnesses: list[dict[str, Any]] = []
    for sentence in sentences:
        relation_matches = list(_SIGNIFICANCE_RE.finditer(sentence.text))
        if not relation_matches:
            continue
        entity_objects = [
            ent
            for ent in doc.ents
            if ent.start_char >= sentence.start
            and ent.end_char <= sentence.end
            and (ent.label_ in _ENTITY_LABELS or ent.label_ in _EVIDENCE_LABELS)
        ]
        if not entity_objects:
            continue
        span = _sentence_token_span(doc, sentence)
        if span is None:
            continue
        comparative_tokens = []
        comparative_tokens = [
            token.text
            for token in span
            if token.tag_ in {"JJR", "JJS", "RBR", "RBS"}
            or token.dep_ in {"advcl", "prep"}
            and token.lemma_.casefold() in {"compare", "result"}
        ]
        dependency_bindings = []
        for relation_match in relation_matches:
            cue_offset = sentence.start + relation_match.start()
            relation_token = min(span, key=lambda token: abs(token.idx - cue_offset))
            binding = _dependency_binding(
                span,
                relation_token,
                [entity.root for entity in entity_objects],
            )
            if binding is not None:
                dependency_bindings.append(
                    {
                        "relation_cue": relation_match.group(0),
                        "relation_token": relation_token.text,
                        **binding,
                    }
                )
        if not dependency_bindings:
            continue
        witnesses.append(
            {
                "sentence_index": sentence.index,
                "relation_cues": [match.group(0) for match in relation_matches],
                "comparative_dependency_tokens": comparative_tokens,
                "bound_entities_or_quantities": [
                    {"text": entity.text, "label": entity.label_}
                    for entity in entity_objects[:8]
                ],
                "dependency_bindings": dependency_bindings,
                "exact_sentence": sentence.text,
            }
        )
    return _pack_result(
        "significance_comparison_binding",
        status="witnessed" if witnesses else "relation_not_instantiated",
        summary={
            "bound_comparative_or_significance_clauses": len(witnesses),
            "opening_bindings": sum(row["sentence_index"] <= 1 for row in witnesses),
            "bindings_with_quantity": sum(
                any(
                    entity["label"] in _EVIDENCE_LABELS
                    for entity in row["bound_entities_or_quantities"]
                )
                for row in witnesses
            ),
        },
        witnesses=witnesses,
        realized_depth=3 if witnesses else None,
    )


def _uncertainty_relation(doc, sentences: Sequence[SentenceRecord]) -> dict[str, Any]:
    witnesses: list[dict[str, Any]] = []
    for sentence in sentences:
        span = _sentence_token_span(doc, sentence)
        if span is None:
            continue
        cue_tokens = []
        for token in span:
            lemma = token.lemma_.casefold()
            if lemma in {"may", "might"}:
                # Calendar-month "May" is commonly tagged PROPN/NOUN; only the
                # modal auxiliary is an uncertainty operator.
                if token.tag_ == "MD" or token.pos_ == "AUX":
                    cue_tokens.append(token)
                continue
            if lemma in _UNCERTAINTY_LEMMAS or _LIMITATION_RE.fullmatch(token.text):
                cue_tokens.append(token)
        phrase_matches = list(_LIMITATION_RE.finditer(sentence.text))
        if not cue_tokens and not phrase_matches:
            continue
        bindings = []
        for token in cue_tokens:
            target = token.head
            bindings.append(
                {
                    "cue": token.text,
                    "dependency": token.dep_,
                    "governed_or_head_token": target.text,
                    "governed_or_head_lemma": target.lemma_,
                }
            )
        token_cue_surfaces = {token.text.casefold() for token in cue_tokens}
        bindings.extend(
            {
                "cue": match.group(0),
                "dependency": "phrase_scope",
                "governed_or_head_token": span.root.text,
                "governed_or_head_lemma": span.root.lemma_,
            }
            for match in phrase_matches
            if match.group(0).casefold() not in token_cue_surfaces
        )
        witnesses.append(
            {
                "sentence_index": sentence.index,
                "bindings": bindings,
                "exact_sentence": sentence.text,
            }
        )
    return _pack_result(
        "uncertainty_claim_scope_binding",
        status="witnessed" if witnesses else "relation_not_instantiated",
        summary={
            "uncertainty_or_limitation_clauses": len(witnesses),
            "opening_bindings": sum(row["sentence_index"] <= 1 for row in witnesses),
        },
        witnesses=witnesses,
        realized_depth=3 if witnesses else None,
    )


def _commitment_relation(
    doc,
    sentences: Sequence[SentenceRecord],
    resources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    witnesses: list[dict[str, Any]] = []
    for sentence in sentences:
        span = _sentence_token_span(doc, sentence)
        if span is None:
            continue
        candidates: list[tuple[Any, Any, str]] = []
        for token in span:
            lemma = token.lemma_.casefold()
            if lemma in _COMMITMENT_LEMMAS and token.pos_ == "VERB":
                action = next(
                    (
                        child
                        for child in token.children
                        if child.dep_ in {"ccomp", "pcomp", "xcomp"}
                        and child.pos_ == "VERB"
                    ),
                    None,
                )
                if action is None:
                    action = next(
                        (
                            descendant
                            for descendant in token.subtree
                            if descendant is not token
                            and descendant.pos_ == "VERB"
                            and descendant.lemma_.casefold() not in _NONACTION_LEMMAS
                        ),
                        None,
                    )
                if action is not None:
                    candidates.append((token, action, "lexical_commitment"))
            elif token.tag_ == "MD" and lemma in {"shall", "will"}:
                action = token.head
                if (
                    action.pos_ == "VERB"
                    and action.lemma_.casefold() not in _NONACTION_LEMMAS
                ):
                    candidates.append((token, action, "future_modal"))

        seen = set()
        for commitment, action, commitment_kind in candidates:
            key = (commitment.i, action.i)
            if key in seen:
                continue
            seen.add(key)
            binding = _dependency_binding(span, commitment, [action])
            if binding is None:
                continue
            if any(token.dep_ == "neg" for token in action.subtree):
                continue
            action_nouns = [
                token.text
                for token in action.subtree
                if token.pos_ in {"NOUN", "PROPN"}
                and token.dep_ in {"attr", "dobj", "oprd", "pobj"}
            ]
            action_lemma = action.lemma_.casefold()
            if (
                not action_nouns
                and action_lemma not in _REMEDIAL_ACTIONS | _REPORTING_ACTIONS
            ):
                continue
            is_remedial = (
                action_lemma in _REMEDIAL_ACTIONS
                and bool(_REMEDIAL_CONTEXT_RE.search(sentence.text))
            )
            is_public_reporting = (
                action_lemma in _REPORTING_ACTIONS
                and bool(_REPORTING_CONTEXT_RE.search(sentence.text))
            )
            if is_remedial:
                action_class = "remedial_commitment"
            elif is_public_reporting:
                action_class = "public_reporting_commitment"
            else:
                action_class = "other_concrete_future_action"
            dates = [
                ent.text
                for ent in doc.ents
                if ent.label_ in {"DATE", "TIME"}
                and ent.start_char >= sentence.start
                and ent.end_char <= sentence.end
            ]
            local_resources = [
                {"kind": resource["kind"], "value": resource["value"]}
                for resource in resources
                if sentence.start <= int(resource["start"]) < sentence.end
            ]
            witnesses.append(
                {
                    "sentence_index": sentence.index,
                    "commitment_kind": commitment_kind,
                    "commitment_token": commitment.text,
                    "action_token": action.text,
                    "action_lemma": action_lemma,
                    "action_objects": action_nouns[:8],
                    "action_class": action_class,
                    "dependency_binding": binding,
                    "dates_or_times": dates,
                    "resources": local_resources[:5],
                    "exact_sentence": sentence.text,
                }
            )
    return _pack_result(
        "commitment_action_binding",
        status="witnessed" if witnesses else "relation_not_instantiated",
        summary={
            "commitment_action_bindings": len(witnesses),
            "remedial_commitment_bindings": sum(
                row["action_class"] == "remedial_commitment" for row in witnesses
            ),
            "public_reporting_commitment_bindings": sum(
                row["action_class"] == "public_reporting_commitment"
                for row in witnesses
            ),
            "bindings_with_timeline": sum(bool(row["dates_or_times"]) for row in witnesses),
            "bindings_with_resource": sum(bool(row["resources"]) for row in witnesses),
        },
        witnesses=witnesses,
        realized_depth=3 if witnesses else None,
    )


def _locality_relation(doc, sentences: Sequence[SentenceRecord]) -> dict[str, Any]:
    if not sentences:
        return _pack_result(
            "opening_locality_binding",
            status="insufficient_sentence_structure",
            summary={"sentences": 0, "opening_sentence_count": 0},
            witnesses=[],
        )
    opening_count = min(2, max(1, math.ceil(0.15 * len(sentences))))
    later_places = Counter(
        _normalize_entity(ent.text)
        for ent in doc.ents
        if ent.label_ in {"FAC", "GPE", "LOC"}
        and (_sentence_index(sentences, ent.start_char) or 0) >= opening_count
    )
    witnesses: list[dict[str, Any]] = []
    for sentence in sentences[:opening_count]:
        span = _sentence_token_span(doc, sentence)
        if span is None:
            continue
        place_entities = [
            ent
            for ent in doc.ents
            if ent.label_ in {"FAC", "GPE", "LOC"}
            and ent.start_char >= sentence.start
            and ent.end_char <= sentence.end
        ]
        for entity in place_entities:
            binding = _dependency_binding(span, span.root, [entity.root])
            if binding is None:
                continue
            normalized = _normalize_entity(entity.text)
            witnesses.append(
                {
                    "sentence_index": sentence.index,
                    "place": entity.text,
                    "place_label": entity.label_,
                    "opening_predicate": span.root.text,
                    "dependency_binding": binding,
                    "later_sentence_mentions": later_places[normalized],
                    "recurs_after_opening": later_places[normalized] > 0,
                    "exact_sentence": sentence.text,
                }
            )
    return _pack_result(
        "opening_locality_binding",
        status="witnessed" if witnesses else "relation_not_instantiated",
        summary={
            "sentences": len(sentences),
            "opening_sentence_count": opening_count,
            "opening_place_bindings": len(witnesses),
            "opening_places_repeated_later": sum(
                row["recurs_after_opening"] for row in witnesses
            ),
        },
        witnesses=witnesses,
        realized_depth=3 if witnesses else None,
    )


@functools.lru_cache(maxsize=1)
def implementation_dependencies() -> dict[str, str]:
    """Return versions for the local algorithmic dependencies."""

    import dateutil
    import sklearn
    import spacy

    nlp = _get_nlp()
    return {
        "python_spacy": spacy.__version__,
        "spacy_pipeline": nlp.meta.get("name", "unknown"),
        "spacy_pipeline_version": nlp.meta.get("version", "unknown"),
        "networkx": nx.__version__,
        "scikit_learn": sklearn.__version__,
        "python_dateutil": dateutil.__version__,
    }


def analyze_press_release_ctext(text: str) -> dict[str, Any]:
    """Analyze one exact ctext value without corpus or external state."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("ctext must be a nonempty string")
    nlp = _get_nlp()
    doc = nlp(text)
    sentences = _sentences(doc)
    resources = _resources(text)

    attribution, quote, attribution_rows, quote_rows = _attribution_relations(
        text, doc, sentences
    )
    entity_evidence, sent_features = _entity_evidence_relation(
        text, doc, sentences, resources
    )
    url_relation, _url_rows = _url_relation(text, doc, sentences, resources)
    relations = {
        "attribution_claim_binding": attribution,
        "quote_integration_structure": quote,
        "entity_evidence_graph": entity_evidence,
        "claim_evidence_alignment": _claim_evidence_relation(
            doc, sentences, sent_features
        ),
        "date_quantity_internal_consistency": _date_quantity_relation(
            text, doc, sentences
        ),
        "url_role_clause_binding": url_relation,
        "opening_information_graph_alignment": _opening_relation(
            doc, sentences, sent_features
        ),
        "sentence_dependency_readability": _readability_relation(doc, sentences),
        "cta_resource_binding": _cta_relation(doc, sentences, resources),
        "boilerplate_contact_structure": _boilerplate_relation(
            doc, sentences, resources
        ),
        "section_scannability_structure": _scannability_relation(text),
        "event_logistics_binding": _event_relation(doc, sentences),
        "attribution_scoped_claim_language": _claim_language_relation(
            doc, sentences, attribution_rows, quote_rows
        ),
        "significance_comparison_binding": _significance_relation(doc, sentences),
        "uncertainty_claim_scope_binding": _uncertainty_relation(doc, sentences),
        "commitment_action_binding": _commitment_relation(
            doc, sentences, resources
        ),
        "opening_locality_binding": _locality_relation(doc, sentences),
    }
    if set(relations) != set(RELATION_SPECS):
        raise AssertionError("relation implementation/contract registry mismatch")
    return {
        "schema": SCHEMA,
        "program_id": PROGRAM_ID,
        "input_contract": {
            "field": "ctext",
            "exact_input_chars": len(text),
            "corpus_state_loaded": False,
            "external_resources_loaded": False,
            "network_access_used": False,
            "outcomes_or_references_loaded": False,
        },
        "document_structure": {
            "sentences": len(sentences),
            "tokens": len(doc),
            "named_entities": len(doc.ents),
            "resources": len(resources),
        },
        "relations": relations,
    }
