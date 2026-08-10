"""Deterministic numeric/comparative-only projection of the strict Science code.

The archived strict verifier ranks a global top five across numeric, comparative,
theoretical, empirical, and qualitative claims.  Its ``evidence_link`` decision is
defined only for the latter three relation classes.  This additive projection does
not alter that verifier or any archived output.  It reuses the frozen strict parser,
retrieval, edge predicates, and exact matching while changing one declared selection
step: filter to numeric/comparative candidates *before* ranking the top five.

Every selected claim receives exactly one projection decision:
``supported``, ``contradicted``, or ``insufficient``.  Unmatched claims are explicit
``insufficient`` selections.  Thus this artifact is the exact future reconstruction
target for the implementation-disclosed prompt plane; it is not the whole archived
code vector and has no ``evidence_link`` state.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
import json
from typing import Any, Iterator

from . import core as v2
from . import core_relation_strict as strict


SCHEMA = "metric-seam.science-numeric-comparative-code-projection.v1"
STATUS = "cpu_projection_complete_no_model_api_gpu"
RELATIONS = {"numeric", "comparative"}
DECISIONS = {"supported", "contradicted", "insufficient"}
SELECTION_LIMIT = 5

_BASE_EXTRACT_CLAIMS = v2.extract_claims


class NumericComparativeProjectionError(ValueError):
    """Raised when the frozen relation-local projection contract drifts."""


def select_numeric_comparative_claims(
    abstract: str, *, limit: int = SELECTION_LIMIT
) -> list[v2.Claim]:
    """Filter before top-five ranking, then restore document order and indices."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise NumericComparativeProjectionError("selection limit must be positive")
    # A large limit retains every executable candidate from the frozen extractor.
    candidates = [
        claim
        for claim in _BASE_EXTRACT_CLAIMS(abstract, limit=1_000_000)
        if claim.relation in RELATIONS
    ]
    selected = sorted(
        candidates,
        key=lambda claim: (-claim.selection_score, claim.sentence.index),
    )[:limit]
    selected.sort(key=lambda claim: claim.sentence.index)
    return [
        v2.Claim(
            index,
            claim.sentence,
            claim.relation,
            claim.quantities,
            claim.comparison,
            claim.selection_score,
        )
        for index, claim in enumerate(selected)
    ]


@contextmanager
def _strict_relation_bindings() -> Iterator[None]:
    """Reuse the frozen strict quantity/comparison/edge implementations."""

    with strict._strict_bindings():  # type: ignore[attr-defined]
        yield


def _source_span(source: str, sentence: v2.Sentence) -> dict[str, Any]:
    exact = source[sentence.start : sentence.end]
    return {
        "sentence_index": sentence.index,
        "start": sentence.start,
        "end": sentence.end,
        "exact_source_excerpt": exact,
        "normalized_parser_text": sentence.text,
    }


def _empty_result(item_key: str, *, reason: str) -> dict[str, Any]:
    return {
        "item_key": item_key,
        "status": "abstain",
        "reason": reason,
        "selected_claim_count": 0,
        "decision_counts": {},
        "selections": [],
        "graph": {
            "claim_nodes": 0,
            "evidence_nodes": 0,
            "edges": 0,
            "matched_edges": 0,
            "matching": "exact_max_weight_bipartite",
        },
    }


def project_document(item_key: str, abstract: str, body: str) -> dict[str, Any]:
    """Project one article onto the exact numeric/comparative prompt target."""

    abstract = abstract or ""
    body = body or ""
    if not abstract.strip():
        return _empty_result(item_key, reason="missing_abstract")
    if not body.strip():
        return _empty_result(item_key, reason="missing_fullpaper_body")

    with _strict_relation_bindings():
        abstract_sentences = v2.segment_sentences(abstract)
        abstract_hashes = {
            v2._sentence_fingerprint(sentence.text)  # type: ignore[attr-defined]
            for sentence in abstract_sentences
        }
        body_all = v2.segment_sentences(body)
        body_sentences = [
            sentence
            for sentence in body_all
            if v2._sentence_fingerprint(sentence.text)  # type: ignore[attr-defined]
            not in abstract_hashes
        ]
        body_sentences = [
            v2.Sentence(index, sentence.start, sentence.end, sentence.text)
            for index, sentence in enumerate(body_sentences)
        ]
        claims = select_numeric_comparative_claims(abstract)
        if not claims:
            return _empty_result(
                item_key, reason="no_numeric_or_comparative_claim_relation"
            )

        edges: list[v2.Edge] = []
        if body_sentences:
            index = v2.DocumentBM25(body_sentences)
            for claim in claims:
                for evidence_index, score in index.retrieve(
                    claim.sentence.text, k=8
                ):
                    edge = strict.evaluate_edge(
                        claim,
                        body_sentences[evidence_index],
                        score,
                    )
                    if edge is not None:
                        if edge.decision not in DECISIONS:
                            raise NumericComparativeProjectionError(
                                "strict numeric/comparative edge emitted invalid decision"
                            )
                        edges.append(edge)
        matched = v2._max_weight_matching(  # type: ignore[attr-defined]
            edges, len(claims)
        )
        by_claim = {edge.claim_index: edge for edge in matched}
        selections: list[dict[str, Any]] = []
        for claim in claims:
            edge = by_claim.get(claim.index)
            if edge is None:
                decision = "insufficient"
                reason = "no_distinct_matched_body_sentence"
                evidence = None
                checks = None
            else:
                decision = edge.decision
                reason = edge.reason
                sentence = body_sentences[edge.evidence_index]
                evidence = {
                    **_source_span(body, sentence),
                    "quantities": [
                        asdict(quantity)
                        for quantity in strict.extract_quantities(sentence.text)
                    ],
                    "comparison": (
                        asdict(strict.extract_comparison(sentence.text))
                        if strict.extract_comparison(sentence.text) is not None
                        else None
                    ),
                }
                checks = {
                    "bm25": round(edge.bm25, 6),
                    "claim_term_coverage": round(edge.lexical_coverage, 6),
                    "quantity_matches": edge.quantity_matches,
                    "quantity_required": edge.quantity_required,
                    "relation_state": edge.relation_state,
                }
            if decision not in DECISIONS:
                raise NumericComparativeProjectionError(
                    "projection emitted an invalid decision"
                )
            selections.append(
                {
                    "decision": decision,
                    "reason": reason,
                    "claim": {
                        **_source_span(abstract, claim.sentence),
                        "relation": claim.relation,
                        "selection_score": claim.selection_score,
                        "quantities": [asdict(value) for value in claim.quantities],
                        "comparison": (
                            asdict(claim.comparison)
                            if claim.comparison is not None
                            else None
                        ),
                    },
                    "evidence": evidence,
                    "checks": checks,
                }
            )

    decisions = Counter(row["decision"] for row in selections)
    if decisions["supported"] and decisions["contradicted"]:
        status, reason = "mixed", "support_and_contradiction_projection"
    elif decisions["contradicted"]:
        status, reason = "contradicted", "contradiction_projection"
    elif decisions["supported"]:
        status, reason = "supported", "support_projection"
    else:
        status, reason = "insufficient", "no_relation_certificate_projection"
    return {
        "item_key": item_key,
        "status": status,
        "reason": reason,
        "selected_claim_count": len(claims),
        "decision_counts": dict(sorted(decisions.items())),
        "selections": selections,
        "graph": {
            "claim_nodes": len(claims),
            "evidence_nodes": len(body_sentences),
            "edges": len(edges),
            "matched_edges": len(matched),
            "matching": "exact_max_weight_bipartite",
        },
    }


def build_projection(
    split_items: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    parse_ctext,
) -> dict[str, Any]:
    """Execute the projection over frozen label-free split items."""

    rows: list[dict[str, Any]] = []
    by_phase: dict[str, Any] = {}
    total_decisions: Counter[str] = Counter()
    for phase, items in split_items.items():
        phase_rows = []
        for item in items:
            if set(item) != {"item_key", "ctext"}:
                raise NumericComparativeProjectionError(
                    "projection item exposes fields beyond item_key/ctext"
                )
            abstract, body = parse_ctext(item["ctext"])
            projected = project_document(item["item_key"], abstract, body)
            row = {"phase": phase, **projected}
            phase_rows.append(row)
            rows.append(row)
        status_counts = Counter(row["status"] for row in phase_rows)
        decisions = Counter(
            decision
            for row in phase_rows
            for decision, count in row["decision_counts"].items()
            for _ in range(count)
        )
        total_decisions.update(decisions)
        by_phase[phase] = {
            "items": len(phase_rows),
            "status_counts": dict(sorted(status_counts.items())),
            "selected_claims": sum(
                row["selected_claim_count"] for row in phase_rows
            ),
            "decision_counts": dict(sorted(decisions.items())),
        }

    forbidden = {
        selection["decision"]
        for row in rows
        for selection in row["selections"]
    } - DECISIONS
    if forbidden:
        raise NumericComparativeProjectionError(
            f"projection contains forbidden decisions: {sorted(forbidden)}"
        )
    payload = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "task": "peer-review",
        "method_origin": "additive_projection_of_frozen_strict_code",
        "archived_artifacts_modified": False,
        "selection_contract": {
            "candidate_relations": ["comparative", "numeric"],
            "relation_classification_priority": [
                "comparative",
                "theoretical",
                "numeric",
                "empirical",
                "qualitative",
            ],
            "filter_before_top_five_ranking": True,
            "candidate_score": {
                "explicit_claim_cue": 2,
                "directed_comparison": 2,
                "parsed_quantity": 1,
                "result_or_theory_marker": 1,
                "minimum": 2,
            },
            "ranking": "selection_score descending, sentence index ascending",
            "output_order": "source sentence order",
            "limit": SELECTION_LIMIT,
        },
        "decision_contract": {
            "reconstruction_target": [
                "contradicted",
                "insufficient",
                "supported",
            ],
            "evidence_link_in_reconstruction_target": False,
            "unmatched_selected_claim": "insufficient",
        },
        "not_whole_archived_vector": True,
        "by_phase": by_phase,
        "summary": {
            "items": len(rows),
            "selected_claims": sum(row["selected_claim_count"] for row in rows),
            "decision_counts": dict(sorted(total_decisions.items())),
            "evidence_link_decisions": 0,
        },
        "rows": rows,
    }
    # Freeze the exact JSON value domain now (not only when a writer happens to
    # serialize it), so tuple-bearing dataclass fields replay as canonical lists.
    return json.loads(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
