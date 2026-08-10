#!/usr/bin/env python3
"""Exact-address, label-free code comparator for the science-v8 prompt arm.

This additive instrument consumes the exact ``A####``/``B####`` source-span objects in
the verified v8 requests.  It never reconstructs continuous abstract/body strings and
never uses the historical code-segmenter crosswalk.  Claim selection, document-local
BM25 retrieval, corrected relation parsing, and exact maximum-weight matching therefore
run over the same source addresses and exact span texts supplied to the prompt.

The resulting certificates are executable, document-local relation witnesses.  They
are not external scientific truth.  Agreement with a prompt result may later measure
reconstruction agreement; agreement alone does not establish isomorphism.

Historical provenance is intentionally explicit: the deep decomposition was manually
constructed as a mock of a discovered verifiability program.  This comparator is a
selected retrospective seed, not an automatically discovered program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from . import addressed_pipeline as addressed
from . import core as frozen
from . import core_corrected as corrected


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = (
    ROOT / "outputs/metric_seam_pilot/science_articulability_v8_hardened_prepared"
)
DEFAULT_OUT = (
    ROOT / "outputs/metric_seam_pilot/science_verifiability_v8_exact_address_code"
)

RESULT_SCHEMA = "science-verifiability-exact-address-result-v8"
STRUCTURAL_RESULT_SCHEMA = "science-verifiability-exact-address-abstention-v8"
MANIFEST_SCHEMA = "science-verifiability-exact-address-bundle-v8"
SOURCE_MAP_SCHEMA = addressed.SOURCE_MAP_SCHEMA
RESULTS_NAME = "code_results.jsonl"
REPORT_NAME = "REPORT.md"

METHOD_PROVENANCE: dict[str, Any] = {
    "historical_deep_decomposition_origin": (
        "manually_constructed_mock_of_discovered_deep_verifiability"
    ),
    "current_pipeline_selection": "selected_retrospective_seed",
    "automatically_discovered": False,
    "claim_selection_origin": "manually_written_relation_heuristics",
    "relation_parser_origin": "manually_written_and_audit_corrected",
    "retrieval_and_matching_origin": "manually_selected_bm25_and_exact_bipartite_matching",
    "interpretation": (
        "The run measures what this frozen executable witness reconstructs; it does "
        "not measure an autonomous system discovering the decomposition from scratch."
    ),
}


def canonical_bytes(value: Any) -> bytes:
    return addressed.canonical_bytes(value)


def hash_value(value: Any) -> str:
    return addressed.hash_value(value)


def hash_file(path: Path) -> str:
    return addressed.hash_file(path)


def display_path(path: Path) -> str:
    return addressed.display_path(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _implementation_bindings() -> dict[str, dict[str, str]]:
    from . import addressed_pipeline_v8 as v8

    modules = {
        "exact_address_comparator": Path(__file__),
        "corrected_relation_ops": Path(corrected.__file__),
        "frozen_bm25_matching_ops": Path(frozen.__file__),
        "v8_bundle_verifier": Path(v8.__file__),
        "addressed_source_map_contract": Path(addressed.__file__),
    }
    return {
        name: {"path": display_path(path), "sha256": hash_file(path)}
        for name, path in modules.items()
    }


def _comparison_terms(text: str) -> tuple[str, ...]:
    """Use the frozen entity normalizer while keeping parsing local to this arm."""

    return frozen._entity_terms(text)  # type: ignore[attr-defined]


_PASSIVE_COMPARATORS: tuple[tuple[re.Pattern[str], int], ...] = (
    (
        re.compile(
            r"\b(?:is|are|was|were|be|been|being)\s+"
            r"(?:outperformed|surpassed|exceeded|beaten)\s+by\b",
            re.I,
        ),
        -1,
    ),
)
_ACTIVE_COMPARATORS: tuple[tuple[re.Pattern[str], int], ...] = (
    (
        re.compile(
            r"\b(?:outperform(?:s|ed|ing)?|surpass(?:es|ed|ing)?|"
            r"exceed(?:s|ed|ing)?|beat(?:s|en|ing)?)\b",
            re.I,
        ),
        1,
    ),
    (re.compile(r"\bimprov(?:e|es|ed|ing)\s+(?:on|over|upon)\b", re.I), 1),
    (re.compile(r"\b(?:better|higher|faster|more accurate|more efficient)\s+than\b", re.I), 1),
    (re.compile(r"\b(?:worse|lower|slower|inferior)\s+than\b", re.I), -1),
    (re.compile(r"\bunderperform(?:s|ed|ing)?\b", re.I), -1),
)


def extract_comparison(text: str) -> frozen.Comparison | None:
    """Parse directed roles, prioritizing passives before their active participle.

    The frozen parser checks ``outperformed`` as an active cue before reaching its
    passive rule.  Here a passive span is a single cue: text before the auxiliary is
    the disadvantaged left entity and text after ``by`` is the advantaged right entity.
    """

    source = str(text or "")
    for pattern, base_polarity in (*_PASSIVE_COMPARATORS, *_ACTIVE_COMPARATORS):
        match = pattern.search(source)
        if not match:
            continue
        left = source[max(0, match.start() - 120):match.start()]
        right = source[match.end():min(len(source), match.end() + 120)]
        left = re.split(r"[.;:]|\b(?:that|whether)\b", left, flags=re.I)[-1]
        right = re.split(r"[.;:]", right)[0]
        right = re.split(
            r"\b(?:by|on|with|using|for|at|while|under|across)\b",
            right,
            maxsplit=1,
            flags=re.I,
        )[0]
        polarity = base_polarity
        if re.search(
            r"\b(?:not|never|fails?\s+to|doesn['’]t|didn['’]t)\b",
            left[-45:],
            re.I,
        ):
            polarity *= -1
        return frozen.Comparison(
            cue=match.group(0),
            polarity=polarity,
            left_terms=_comparison_terms(left),
            right_terms=tuple(_comparison_terms(right)[:8]),
        )
    return None


def _relation(
    text: str,
    quantities: tuple[frozen.Quantity, ...],
    comparison: frozen.Comparison | None,
) -> str:
    return frozen._relation(text, quantities, comparison)  # type: ignore[attr-defined]


def _fingerprint(text: str) -> str:
    normalized = " ".join(frozen.tokens(text, content_only=False))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_source_map(source_map: dict[str, Any], paper_id: str) -> None:
    """Validate exact address identities, hashes, offsets, and ordering."""

    if not isinstance(source_map, dict):
        raise ValueError("source_map must be an object")
    if source_map.get("schema_version") != SOURCE_MAP_SCHEMA:
        raise ValueError("unsupported source-map schema")
    if source_map.get("paper_id") != paper_id:
        raise ValueError("source-map paper_id mismatch")
    expected_keys = {"schema_version", "paper_id", "abstract", "body"}
    if set(source_map) != expected_keys:
        raise ValueError("source-map keys differ from exact addressed contract")
    for section, prefix in (("abstract", "A"), ("body", "B")):
        spans = source_map.get(section)
        if not isinstance(spans, list):
            raise ValueError(f"source-map {section} must be a list")
        prior_end = -1
        for index, span in enumerate(spans):
            if not isinstance(span, dict):
                raise ValueError(f"source-map {section} span must be an object")
            if set(span) != {
                "sentence_id", "section", "sentence_index", "start", "end",
                "text", "text_sha256",
            }:
                raise ValueError(f"source-map {section} span keys changed")
            expected_id = f"{prefix}{index + 1:04d}"
            if span["sentence_id"] != expected_id:
                raise ValueError(f"noncanonical address sequence: {expected_id}")
            if span["section"] != section or span["sentence_index"] != index:
                raise ValueError(f"source-map {expected_id} section/index mismatch")
            start, end = span["start"], span["end"]
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or start < prior_end
            ):
                raise ValueError(f"source-map {expected_id} offsets invalid")
            text = span["text"]
            if not isinstance(text, str) or not text or text != text.strip():
                raise ValueError(f"source-map {expected_id} exact text invalid")
            if end - start != len(text):
                raise ValueError(f"source-map {expected_id} offset length mismatch")
            if span["text_sha256"] != hash_value(text):
                raise ValueError(f"source-map {expected_id} text hash mismatch")
            prior_end = end


def _source_address(span: dict[str, Any]) -> dict[str, Any]:
    """Return a replayable address without copying source prose into the result."""

    return {
        "sentence_id": span["sentence_id"],
        "section": span["section"],
        "sentence_index": span["sentence_index"],
        "start": span["start"],
        "end": span["end"],
        "text_sha256": span["text_sha256"],
    }


def _quantity_payload(quantity: frozen.Quantity) -> dict[str, Any]:
    return {
        "value": quantity.value,
        "unit": quantity.unit,
        "span_relative_start": quantity.start,
        "span_relative_end": quantity.end,
        "raw_sha256": hash_value(quantity.raw),
    }


def _comparison_payload(comparison: frozen.Comparison | None) -> dict[str, Any] | None:
    if comparison is None:
        return None
    payload = asdict(comparison)
    payload["cue_sha256"] = hash_value(payload.pop("cue"))
    payload["left_terms"] = list(payload["left_terms"])
    payload["right_terms"] = list(payload["right_terms"])
    return payload


def _claim_payload(
    claim: frozen.Claim, abstract_spans: list[dict[str, Any]]
) -> dict[str, Any]:
    span = abstract_spans[claim.sentence.index]
    return {
        "source_address": _source_address(span),
        "relation": claim.relation,
        "selection_score": claim.selection_score,
        "quantities": [_quantity_payload(quantity) for quantity in claim.quantities],
        "comparison": _comparison_payload(claim.comparison),
    }


def select_claims(abstract_spans: list[dict[str, Any]], *, limit: int = 5) -> list[frozen.Claim]:
    """Apply the retrospective claim-selection seed to exact A-address spans."""

    candidates: list[frozen.Claim] = []
    for span in abstract_spans:
        text = span["text"]
        quantities = corrected.extract_quantities(text)
        comparison = extract_comparison(text)
        score = 0.0
        if frozen._CLAIM_RE.search(text):  # type: ignore[attr-defined]
            score += 2.0
        if comparison is not None:
            score += 2.0
        if quantities:
            score += 1.0
        if frozen._RESULT_RE.search(text) or frozen._THEORY_RE.search(text):  # type: ignore[attr-defined]
            score += 1.0
        if score < 2.0:
            continue
        sentence = frozen.Sentence(
            span["sentence_index"], span["start"], span["end"], text
        )
        candidates.append(
            frozen.Claim(
                index=len(candidates),
                sentence=sentence,
                relation=_relation(text, quantities, comparison),
                quantities=quantities,
                comparison=comparison,
                selection_score=score,
            )
        )
    selected = sorted(
        candidates, key=lambda claim: (-claim.selection_score, claim.sentence.index)
    )[:limit]
    selected.sort(key=lambda claim: claim.sentence.index)
    return [
        frozen.Claim(
            index=index,
            sentence=claim.sentence,
            relation=claim.relation,
            quantities=claim.quantities,
            comparison=claim.comparison,
            selection_score=claim.selection_score,
        )
        for index, claim in enumerate(selected)
    ]


def _evaluate_edge(
    claim: frozen.Claim, evidence: frozen.Sentence, bm25: float
) -> frozen.Edge | None:
    """Corrected quantity/entity and comparison-role evaluation on exact spans."""

    claim_tokens = frozen.tokens(claim.sentence.text, content_only=True)
    evidence_tokens = frozen.tokens(evidence.text, content_only=True)
    coverage = len(set(claim_tokens) & set(evidence_tokens)) / max(
        1, len(set(claim_tokens))
    )
    if coverage < 0.08:
        return None
    evidence_quantities = corrected.extract_quantities(evidence.text)
    quantity_matches = sum(
        any(
            corrected.quantity_relation_equal(
                claim.sentence.text,
                claim_quantity,
                evidence.text,
                evidence_quantity,
            )
            for evidence_quantity in evidence_quantities
        )
        for claim_quantity in claim.quantities
    )
    raw_value_matches = sum(
        any(frozen.quantity_equal(claim_quantity, evidence_quantity)
            for evidence_quantity in evidence_quantities)
        for claim_quantity in claim.quantities
    )
    relation_state = frozen._comparison_state(  # type: ignore[attr-defined]
        claim.comparison, extract_comparison(evidence.text)
    )
    decision, witness_kind = "insufficient", "none"
    reason = "retrieved_but_relation_not_certified"
    if claim.relation == "comparative":
        if relation_state in {"reversed_roles", "direction_mismatch"} and coverage >= 0.18:
            decision, witness_kind, reason = (
                "contradicted", "relation_certificate", relation_state
            )
        elif relation_state not in {"aligned", "aligned_reversed"}:
            reason = relation_state
        elif claim.quantities and quantity_matches < len(claim.quantities):
            reason = (
                "quantity_entity_binding_failed"
                if raw_value_matches == len(claim.quantities)
                else "claim_quantity_not_reproduced"
            )
        elif coverage >= 0.16:
            decision, witness_kind, reason = (
                "supported", "relation_certificate", "aligned_comparison"
            )
    elif claim.relation == "numeric":
        if (
            quantity_matches == len(claim.quantities)
            and quantity_matches > 0
            and coverage >= 0.13
        ):
            decision, witness_kind, reason = (
                "supported",
                "relation_certificate",
                "normalized_quantity_entity_and_terms_match",
            )
        else:
            reason = (
                "quantity_entity_binding_failed"
                if raw_value_matches == len(claim.quantities)
                and raw_value_matches > quantity_matches
                else "claim_quantity_not_reproduced"
            )
    elif claim.relation == "theoretical":
        if frozen._THEORY_RE.search(evidence.text) and coverage >= 0.18:  # type: ignore[attr-defined]
            decision, witness_kind, reason = (
                "evidence_link", "evidence_link", "theory_marker_and_terms_match"
            )
        else:
            reason = "missing_theory_witness"
    elif claim.relation == "empirical":
        if (
            frozen._EVIDENCE_RE.search(evidence.text)  # type: ignore[attr-defined]
            and frozen._ASSERTION_RE.search(evidence.text)  # type: ignore[attr-defined]
            and coverage >= 0.20
        ):
            decision, witness_kind, reason = (
                "evidence_link", "evidence_link", "empirical_artifact_and_terms_match"
            )
        else:
            reason = "missing_empirical_assertion_witness"
    elif frozen._EVIDENCE_RE.search(evidence.text) and coverage >= 0.25:  # type: ignore[attr-defined]
        decision, witness_kind, reason = (
            "evidence_link", "evidence_link", "qualitative_evidence_and_terms_match"
        )

    relation_bonus = {
        "aligned": 1.0,
        "aligned_reversed": 1.0,
        "not_required": 0.45,
        "reversed_roles": 0.35,
        "direction_mismatch": 0.35,
        "missing": 0.0,
        "baseline_mismatch": 0.0,
    }.get(relation_state, 0.0)
    weight = (
        coverage
        + 0.12 * math.log1p(bm25)
        + 0.20 * quantity_matches
        + relation_bonus
    )
    return frozen.Edge(
        claim.index,
        evidence.index,
        weight,
        coverage,
        bm25,
        quantity_matches,
        len(claim.quantities),
        relation_state,
        decision,
        witness_kind,
        reason,
    )


def _empty_result(
    paper_id: str,
    source_map: dict[str, Any],
    *,
    reason: str,
    excluded_repeats: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    repeats = excluded_repeats or []
    return {
        "paper_id": paper_id,
        "status": "abstain",
        "reason": reason,
        "claim_count": 0,
        "certificate_count": 0,
        "evidence_link_count": 0,
        "decision_counts": {},
        "certificates": [],
        "evidence_links": [],
        "matches": [],
        "selected_claims": [],
        "coverage": {
            "abstract_addresses": len(source_map["abstract"]),
            "body_addresses": len(source_map["body"]),
            "independent_body_addresses": len(source_map["body"]) - len(repeats),
            "repeated_abstract_addresses_excluded": len(repeats),
            "selected_claim_addresses": 0,
            "matched_claim_addresses": 0,
        },
        "excluded_repeated_abstract_addresses": [_source_address(span) for span in repeats],
        "graph": {
            "claim_nodes": 0,
            "evidence_nodes": len(source_map["body"]) - len(repeats),
            "edges": 0,
            "matched_edges": 0,
            "matching": "exact_max_weight_bipartite",
        },
    }


def verify_addressed_document(paper_id: str, source_map: dict[str, Any]) -> dict[str, Any]:
    """Verify one document using only its exact source-map address objects."""

    validate_source_map(source_map, paper_id)
    abstract_spans = source_map["abstract"]
    body_spans = source_map["body"]
    if not abstract_spans:
        return _empty_result(paper_id, source_map, reason="missing_abstract_addresses")
    if not body_spans:
        return _empty_result(paper_id, source_map, reason="missing_fullpaper_body_addresses")

    abstract_hashes = {_fingerprint(span["text"]) for span in abstract_spans}
    repeats = [span for span in body_spans if _fingerprint(span["text"]) in abstract_hashes]
    independent_spans = [
        span for span in body_spans if _fingerprint(span["text"]) not in abstract_hashes
    ]
    if not independent_spans:
        return _empty_result(
            paper_id,
            source_map,
            reason="abstract_only_no_independent_addressed_evidence",
            excluded_repeats=repeats,
        )

    claims = select_claims(abstract_spans)
    if not claims:
        return _empty_result(
            paper_id,
            source_map,
            reason="no_executable_claim_relation",
            excluded_repeats=repeats,
        )

    # Local graph indices are intentionally separate from immutable B-address indices.
    evidence_nodes = [
        frozen.Sentence(local_index, span["start"], span["end"], span["text"])
        for local_index, span in enumerate(independent_spans)
    ]
    index = frozen.DocumentBM25(evidence_nodes)
    edges: list[frozen.Edge] = []
    for claim in claims:
        for evidence_index, score in index.retrieve(claim.sentence.text, k=8):
            edge = _evaluate_edge(claim, evidence_nodes[evidence_index], score)
            if edge is not None:
                edges.append(edge)
    if not edges:
        result = _empty_result(
            paper_id,
            source_map,
            reason="no_retrievable_evidence",
            excluded_repeats=repeats,
        )
        result["claim_count"] = len(claims)
        result["selected_claims"] = [
            _claim_payload(claim, abstract_spans) for claim in claims
        ]
        result["coverage"]["selected_claim_addresses"] = len(claims)
        result["graph"]["claim_nodes"] = len(claims)
        return result

    matched = frozen._max_weight_matching(edges, len(claims))  # type: ignore[attr-defined]
    matches: list[dict[str, Any]] = []
    for edge in matched:
        claim = claims[edge.claim_index]
        claim_span = abstract_spans[claim.sentence.index]
        evidence_span = independent_spans[edge.evidence_index]
        evidence_quantities = corrected.extract_quantities(evidence_span["text"])
        evidence_comparison = extract_comparison(evidence_span["text"])
        matches.append(
            {
                "reconstruction_key": {
                    "claim_sentence_id": claim_span["sentence_id"],
                    "evidence_sentence_id": evidence_span["sentence_id"],
                    "relation": claim.relation,
                },
                "decision": edge.decision,
                "witness_kind": edge.witness_kind,
                "reason": edge.reason,
                "claim": {
                    "source_address": _source_address(claim_span),
                    "relation": claim.relation,
                    "selection_score": claim.selection_score,
                    "quantities": [_quantity_payload(q) for q in claim.quantities],
                    "comparison": _comparison_payload(claim.comparison),
                },
                "evidence": {
                    "source_address": _source_address(evidence_span),
                    "quantities": [_quantity_payload(q) for q in evidence_quantities],
                    "comparison": _comparison_payload(evidence_comparison),
                },
                "checks": {
                    "bm25": round(edge.bm25, 6),
                    "claim_term_coverage": round(edge.lexical_coverage, 6),
                    "quantity_matches": edge.quantity_matches,
                    "quantity_required": edge.quantity_required,
                    "relation_state": edge.relation_state,
                    "executable_replay": (
                        "hydrate_bound_source_addresses_then_rerun_frozen_code"
                    ),
                },
            }
        )

    decisions = Counter(match["decision"] for match in matches)
    certificates = [
        match for match in matches if match["witness_kind"] == "relation_certificate"
    ]
    evidence_links = [
        match for match in matches if match["witness_kind"] == "evidence_link"
    ]
    if decisions["supported"] and decisions["contradicted"]:
        status, reason = "mixed", "support_and_contradiction_certificates"
    elif decisions["contradicted"]:
        status, reason = "contradicted", "contradiction_certificate"
    elif decisions["supported"]:
        status, reason = "supported", "support_certificate"
    elif decisions["evidence_link"]:
        status, reason = "evidence_link", "surface_evidence_link_only"
    else:
        status, reason = "insufficient", "retrieved_without_relation_certificate"
    return {
        "paper_id": paper_id,
        "status": status,
        "reason": reason,
        "claim_count": len(claims),
        "certificate_count": len(certificates),
        "evidence_link_count": len(evidence_links),
        "decision_counts": dict(sorted(decisions.items())),
        "certificates": certificates,
        "evidence_links": evidence_links,
        "matches": matches,
        "selected_claims": [
            _claim_payload(claim, abstract_spans) for claim in claims
        ],
        "coverage": {
            "abstract_addresses": len(abstract_spans),
            "body_addresses": len(body_spans),
            "independent_body_addresses": len(independent_spans),
            "repeated_abstract_addresses_excluded": len(repeats),
            "selected_claim_addresses": len(claims),
            "matched_claim_addresses": len(matched),
        },
        "excluded_repeated_abstract_addresses": [_source_address(span) for span in repeats],
        "graph": {
            "claim_nodes": len(claims),
            "evidence_nodes": len(evidence_nodes),
            "edges": len(edges),
            "matched_edges": len(matched),
            "matching": "exact_max_weight_bipartite",
        },
    }


def _fixture_source_map(abstract_text: str, body_text: str) -> dict[str, Any]:
    paper = {"paper_id": "metamorphic", "abstract": abstract_text, "body": body_text}
    return addressed.build_source_map(paper)


def metamorphic_self_check() -> dict[str, bool]:
    """Guard exact-address transport plus numeric, entity, role, and direction logic."""

    number_claim = "We report results on 100k examples."
    exact_number = verify_addressed_document(
        "metamorphic",
        _fixture_source_map(number_claim, "Table 2 reports results on 100k examples."),
    )
    changed_number = verify_addressed_document(
        "metamorphic",
        _fixture_source_map(number_claim, "Table 2 reports results on 10 examples."),
    )
    entity_claim = "We show robust performance across 28 LoRA adapters."
    correct_entity = verify_addressed_document(
        "metamorphic",
        _fixture_source_map(
            entity_claim, "Table 2 shows robust performance across 28 LoRA adapters."
        ),
    )
    wrong_entity = verify_addressed_document(
        "metamorphic",
        _fixture_source_map(
            entity_claim, "Table 2 shows robust performance across 28 image tasks."
        ),
    )
    comparison_claim = "We show that our method outperforms BERT."
    direction = verify_addressed_document(
        "metamorphic",
        _fixture_source_map(
            comparison_claim,
            "Table 2 shows that our method is outperformed by BERT.",
        ),
    )
    roles = verify_addressed_document(
        "metamorphic",
        _fixture_source_map(
            comparison_claim, "Table 2 shows that BERT outperforms our method."
        ),
    )
    repeated = verify_addressed_document(
        "metamorphic", _fixture_source_map(comparison_claim, comparison_claim)
    )
    missing = verify_addressed_document(
        "metamorphic", _fixture_source_map(comparison_claim, "")
    )
    exact_map = _fixture_source_map(
        comparison_claim, "Table 2 shows that our method outperforms BERT."
    )
    exact = verify_addressed_document("metamorphic", exact_map)
    exact_cert = exact["certificates"][0] if exact["certificates"] else {}
    checks = {
        "complete_number_suffix_supports": exact_number["status"] == "supported",
        "number_suffix_mutation_invalidates": changed_number["status"] != "supported",
        "quantity_entity_match_supports": correct_entity["status"] == "supported",
        "quantity_entity_mutation_invalidates": wrong_entity["status"] != "supported",
        "comparison_direction_is_contradiction": (
            direction["status"] == "contradicted"
            and direction["certificates"][0]["checks"]["relation_state"]
            == "direction_mismatch"
        ),
        "comparison_role_swap_is_contradiction": (
            roles["status"] == "contradicted"
            and roles["certificates"][0]["checks"]["relation_state"]
            == "reversed_roles"
        ),
        "exact_claim_address_preserved": (
            exact_cert.get("claim", {}).get("source_address")
            == _source_address(exact_map["abstract"][0])
        ),
        "exact_evidence_address_preserved": (
            exact_cert.get("evidence", {}).get("source_address")
            == _source_address(exact_map["body"][0])
        ),
        "repeated_abstract_is_excluded": (
            repeated["status"] == "abstain"
            and repeated["reason"] == "abstract_only_no_independent_addressed_evidence"
        ),
        "missing_body_abstains": (
            missing["status"] == "abstain"
            and missing["reason"] == "missing_fullpaper_body_addresses"
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"exact-address comparator invariant(s) failed: {failed}")
    return checks


def _assert_historical_comparator_input_identity(manifest: dict[str, Any]) -> None:
    """Fail closed unless the historical comparator and v8 share source bytes."""

    historical = manifest.get("historical_code_comparator")
    source = manifest.get("input")
    if not isinstance(historical, dict) or not isinstance(source, dict):
        raise ValueError("v8 manifest lacks historical-comparator input identity gate")
    historical_sha = historical.get("input_source_sha256")
    source_sha = source.get("source_file_sha256")
    if (
        not isinstance(historical_sha, str)
        or len(historical_sha) != 64
        or not isinstance(historical.get("input_source_path"), str)
        or not historical["input_source_path"]
    ):
        raise ValueError("historical-comparator input identity is incomplete")
    if historical_sha != source_sha:
        raise ValueError(
            "historical-comparator payload input SHA differs from v8 source SHA"
        )


def _verified_bundle_snapshot(
    bundle: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from . import addressed_pipeline_v8 as v8

    manifest, requests_by_id, abstentions_by_index = v8.verify_bundle(bundle)
    _assert_historical_comparator_input_identity(manifest)
    requests = sorted(requests_by_id.values(), key=lambda row: row["source_index"])
    abstentions = sorted(abstentions_by_index.values(), key=lambda row: row["source_index"])
    snapshot = {
        "manifest_sha256": hash_file(bundle / "manifest.json"),
        "requests_sha256": hash_file(bundle / v8.REQUESTS_NAME),
        "structural_abstentions_sha256": hash_file(bundle / v8.ABSTENTIONS_NAME),
        "request_index_sha256": hash_value(
            [
                [
                    row["source_index"], row["request_id"], row["request_sha256"],
                    row["source_map_sha256"],
                ]
                for row in requests
            ]
        ),
        "abstention_index_sha256": hash_value(
            [
                [
                    row["source_index"], row["abstention_sha256"],
                    row["source_map_sha256"],
                ]
                for row in abstentions
            ]
        ),
        "implementation_bindings": _implementation_bindings(),
    }
    return manifest, requests, abstentions, snapshot


def _request_result(
    request: dict[str, Any], *, bundle_snapshot: dict[str, Any]
) -> dict[str, Any]:
    # This fresh projection is the comparator's complete request interface.  It does
    # not index paper_input, prompt outputs, crosswalk contents, or any corpus outcome.
    admitted = {
        "source_index": request["source_index"],
        "paper_id": request["paper_id"],
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "source_map": request["source_map"],
        "source_map_sha256": request["source_map_sha256"],
    }
    if hash_value(admitted["source_map"]) != admitted["source_map_sha256"]:
        raise ValueError("request source-map hash mismatch at comparator boundary")
    result = verify_addressed_document(admitted["paper_id"], admitted["source_map"])
    material = {
        "schema_version": RESULT_SCHEMA,
        "source_index": admitted["source_index"],
        "paper_id": admitted["paper_id"],
        "request_id": admitted["request_id"],
        "request_sha256": admitted["request_sha256"],
        "source_map_sha256": admitted["source_map_sha256"],
        "bundle_manifest_sha256": bundle_snapshot["manifest_sha256"],
        "requests_file_sha256": bundle_snapshot["requests_sha256"],
        "comparator_sha256": bundle_snapshot["implementation_bindings"]
        ["exact_address_comparator"]["sha256"],
        "method_provenance_sha256": hash_value(METHOD_PROVENANCE),
        "result": result,
    }
    return {**material, "row_sha256": hash_value(material)}


def _structural_result(
    abstention: dict[str, Any], *, bundle_snapshot: dict[str, Any]
) -> dict[str, Any]:
    material = {
        "schema_version": STRUCTURAL_RESULT_SCHEMA,
        "source_index": abstention["source_index"],
        "paper_id": abstention["paper_id"],
        "request_id": None,
        "request_sha256": None,
        "source_map_sha256": abstention["source_map_sha256"],
        "structural_abstention_sha256": abstention["abstention_sha256"],
        "bundle_manifest_sha256": bundle_snapshot["manifest_sha256"],
        "requests_file_sha256": bundle_snapshot["requests_sha256"],
        "comparator_sha256": bundle_snapshot["implementation_bindings"]
        ["exact_address_comparator"]["sha256"],
        "method_provenance_sha256": hash_value(METHOD_PROVENANCE),
        "result": {
            "paper_id": abstention["paper_id"],
            "status": "abstain",
            "reason": "missing_fullpaper_body_addresses",
            "claim_count": 0,
            "certificate_count": 0,
            "evidence_link_count": 0,
            "decision_counts": {},
            "certificates": [],
            "evidence_links": [],
            "matches": [],
            "selected_claims": [],
            "coverage": {
                "abstract_addresses": None,
                "body_addresses": 0,
                "independent_body_addresses": 0,
                "repeated_abstract_addresses_excluded": 0,
                "selected_claim_addresses": 0,
                "matched_claim_addresses": 0,
            },
            "excluded_repeated_abstract_addresses": [],
            "graph": {
                "claim_nodes": 0,
                "evidence_nodes": 0,
                "edges": 0,
                "matched_edges": 0,
                "matching": "exact_max_weight_bipartite",
            },
        },
    }
    return {**material, "row_sha256": hash_value(material)}


def build_rows(
    requests: list[dict[str, Any]],
    abstentions: list[dict[str, Any]],
    *,
    bundle_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        _request_result(request, bundle_snapshot=bundle_snapshot)
        for request in requests
    ]
    rows.extend(
        _structural_result(abstention, bundle_snapshot=bundle_snapshot)
        for abstention in abstentions
    )
    rows.sort(key=lambda row: row["source_index"])
    indices = [row["source_index"] for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("exact-address code rows do not cover the corpus once")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    selected_relation_counts: Counter[str] = Counter()
    matched_relation_counts: Counter[str] = Counter()
    certificate_relation_counts: Counter[str] = Counter()
    relation_state_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for row in rows:
        result = row["result"]
        status_counts[result["status"]] += 1
        reason_counts[result["reason"]] += 1
        decision_counts.update(result.get("decision_counts", {}))
        selected_relation_counts.update(
            claim["relation"] for claim in result.get("selected_claims", [])
        )
        matched_relation_counts.update(
            match["reconstruction_key"]["relation"]
            for match in result.get("matches", [])
        )
        certificate_relation_counts.update(
            certificate["reconstruction_key"]["relation"]
            for certificate in result.get("certificates", [])
        )
        relation_state_counts.update(
            match["checks"]["relation_state"]
            for match in result.get("matches", [])
        )
        totals["certificates"] += result["certificate_count"]
        totals["evidence_links"] += result["evidence_link_count"]
        totals["selected_claim_addresses"] += result["coverage"][
            "selected_claim_addresses"
        ]
        totals["matched_claim_addresses"] += result["coverage"][
            "matched_claim_addresses"
        ]
        totals["repeated_abstract_addresses_excluded"] += result["coverage"][
            "repeated_abstract_addresses_excluded"
        ]
        totals["graph_edges"] += result["graph"]["edges"]
    return {
        "records": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "selected_relation_counts": dict(sorted(selected_relation_counts.items())),
        "matched_relation_counts": dict(sorted(matched_relation_counts.items())),
        "certificate_relation_counts": dict(sorted(certificate_relation_counts.items())),
        "relation_state_counts": dict(sorted(relation_state_counts.items())),
        **dict(sorted(totals.items())),
    }


def build_manifest(
    *,
    bundle: Path,
    source_manifest: dict[str, Any],
    bundle_snapshot: dict[str, Any],
    rows: list[dict[str, Any]],
    results_path: Path,
    metamorphic_checks: dict[str, bool],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": "completed_cpu_no_api_no_gpu",
        "objective": "unsupervised_code_reconstruction_same_source_address_surface",
        "axes": {
            "articulability": "prompt_based_separate_v8_arm",
            "verifiability": "this_code_based_arm",
            "isomorphism": "not_established_by_agreement_alone",
        },
        "representation_contract": {
            "same_evidence_content_as_prompt": True,
            "same_visible_A_B_ids_and_exact_span_texts_as_prompt": True,
            "same_bound_source_map_as_prompt_request": True,
            "exact_offsets_preserved_from_bound_request": True,
            "offsets_rendered_to_model": False,
            "continuous_abstract_or_body_reconstructed": False,
            "historical_code_segmenter_crosswalk_consumed": False,
            "serialized_prompt_bytes_identical": False,
            "scope": (
                "same_evidence_and_visible_address_surface_with_code_only_bound_offsets"
            ),
        },
        "interpretation_guard": {
            "external_scientific_truth_claimed": False,
            "correlation_or_agreement_implies_correctness": False,
            "agreement_alone_licenses_isomorphism": False,
            "negative_result_establishes_tacitness": False,
        },
        "execution": {
            "api_calls": 0,
            "gpu_used": False,
            "execution_device": "cpu",
            "external_sources": "none",
            "external_supervision": "none",
            "prompt_results_consumed": False,
            "historical_acceptance_outcomes_consumed": False,
        },
        "input_boundary": {
            "official_v8_verifier_runs_before_and_after": True,
            "official_v8_trusted_loader_deserializes_source_rows": True,
            "comparator_request_fields_indexed": [
                "source_index",
                "paper_id",
                "request_id",
                "request_sha256",
                "source_map",
                "source_map_sha256",
            ],
            "comparator_indexes_paper_input": False,
            "comparator_indexes_prompt_output": False,
            "comparator_indexes_acceptance_or_judgment_fields": False,
        },
        "executable_program": {
            "stages": [
                "relation_rich_claim_selection_over_exact_A_addresses",
                "repeated_abstract_exclusion_over_exact_B_addresses",
                "document_local_BM25_retrieval_fit_per_paper",
                "complete_token_quantity_and_unit_normalization",
                "local_quantity_entity_binding",
                "directed_comparison_entity_role_and_polarity_parsing",
                "typed_relation_edge_construction",
                "exact_max_weight_one_to_one_claim_evidence_matching",
                "executable_address_bound_certificate_emission",
            ],
            "retrieval_scope": "one_presented_paper_body_only",
            "matching": "exact_max_weight_bipartite",
            "claim_limit_per_paper": 5,
            "retrieval_limit_per_claim": 8,
            "certificate_scope": "document_local_relation_local_parser_scoped",
        },
        "method_provenance": METHOD_PROVENANCE,
        "method_provenance_sha256": hash_value(METHOD_PROVENANCE),
        "source_bundle": {
            "path": display_path(bundle),
            "schema_version": source_manifest["schema_version"],
            "manifest_sha256": bundle_snapshot["manifest_sha256"],
            "requests_sha256": bundle_snapshot["requests_sha256"],
            "structural_abstentions_sha256": bundle_snapshot[
                "structural_abstentions_sha256"
            ],
            "request_index_sha256": bundle_snapshot["request_index_sha256"],
            "abstention_index_sha256": bundle_snapshot["abstention_index_sha256"],
            "official_full_verification_passed_before_and_after_run": True,
            "historical_comparator_input_identity": {
                "input_source_path": source_manifest["historical_code_comparator"][
                    "input_source_path"
                ],
                "input_source_sha256": source_manifest[
                    "historical_code_comparator"
                ]["input_source_sha256"],
                "v8_source_sha256": source_manifest["input"]["source_file_sha256"],
                "machine_equal": (
                    source_manifest["historical_code_comparator"][
                        "input_source_sha256"
                    ]
                    == source_manifest["input"]["source_file_sha256"]
                ),
            },
        },
        "implementation_bindings": bundle_snapshot["implementation_bindings"],
        "metamorphic_checks": {
            "all_passed": all(metamorphic_checks.values()),
            "checks": metamorphic_checks,
        },
        "files": {
            "results": {
                "path": RESULTS_NAME,
                "sha256": hash_file(results_path),
                "count": len(rows),
            },
            "report": {
                "path": REPORT_NAME,
                "verification": "exact_deterministic_render_from_manifest",
            },
        },
        "summary": summarize(rows),
    }


def render_report(manifest: dict[str, Any]) -> str:
    summary = manifest["summary"]
    statuses = ", ".join(
        f"{name}={count:,}" for name, count in summary["status_counts"].items()
    )
    return f"""# Science v8 exact-address code comparator

Status: **completed on CPU**, with **0 API calls** and **no GPU use**.

This code arm consumed the exact A####/B#### span texts, IDs, and offsets in the fully
verified v8 request bundle. The model-visible address surface contains the same IDs and
texts; offsets are bound request metadata but are not rendered to the model. The code arm
did not reconstruct continuous article text and did not consume the historical segmenter
crosswalk, prompt results, acceptance outcomes, external supervision, or external
scientific sources.

## Coverage

- Corpus rows: **{summary['records']:,}**
- Statuses: {statuses}
- Selected exact-address claims: **{summary['selected_claim_addresses']:,}**
- One-to-one matched claims: **{summary['matched_claim_addresses']:,}**
- Executable relation certificates: **{summary['certificates']:,}**
- Evidence links (not relation certificates): **{summary['evidence_links']:,}**
- Repeated abstract addresses excluded from body evidence: **{summary['repeated_abstract_addresses_excluded']:,}**
- Retrieved graph edges: **{summary['graph_edges']:,}**

## Claim boundary

Certificates are document-local executable witnesses under the frozen parser. They do
not establish external scientific truth. A later prompt/code comparison may report
reconstruction agreement over this same address surface; agreement alone does not
establish full isomorphism.

## Provenance

The original deep science decomposition was manually constructed as a mock of a
discovered verifiability program. This implementation is a selected retrospective seed,
not an automatically discovered decomposition. Every body-present result row binds its
v8 request and source map; every missing-body row binds its structural-abstention ledger
entry. All rows bind the bundle manifest, comparator implementation, and provenance
hashes.
"""


def run(bundle: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")
    checks = metamorphic_self_check()
    source_manifest, requests, abstentions, before = _verified_bundle_snapshot(bundle)
    rows = build_rows(requests, abstentions, bundle_snapshot=before)
    source_manifest_after, _, _, after = _verified_bundle_snapshot(bundle)
    if source_manifest != source_manifest_after or before != after:
        raise RuntimeError("v8 source bundle or bound code changed during comparator run")
    output_dir.mkdir(parents=True, exist_ok=False)
    results_path = output_dir / RESULTS_NAME
    _write_jsonl(results_path, rows)
    manifest = build_manifest(
        bundle=bundle,
        source_manifest=source_manifest,
        bundle_snapshot=before,
        rows=rows,
        results_path=results_path,
        metamorphic_checks=checks,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / REPORT_NAME).write_text(render_report(manifest), encoding="utf-8")
    return manifest


def verify_output(bundle: Path, output_dir: Path) -> dict[str, Any]:
    source_manifest, requests, abstentions, snapshot = _verified_bundle_snapshot(bundle)
    actual_rows = _read_jsonl(output_dir / RESULTS_NAME)
    expected_rows = build_rows(requests, abstentions, bundle_snapshot=snapshot)
    if actual_rows != expected_rows:
        raise ValueError("exact-address result rows differ from deterministic replay")
    for row in actual_rows:
        material = {key: value for key, value in row.items() if key != "row_sha256"}
        if row.get("row_sha256") != hash_value(material):
            raise ValueError("exact-address result row hash mismatch")
    checks = metamorphic_self_check()
    expected_manifest = build_manifest(
        bundle=bundle,
        source_manifest=source_manifest,
        bundle_snapshot=snapshot,
        rows=actual_rows,
        results_path=output_dir / RESULTS_NAME,
        metamorphic_checks=checks,
    )
    actual_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if actual_manifest != expected_manifest:
        raise ValueError("exact-address output manifest differs from deterministic replay")
    if (output_dir / REPORT_NAME).read_text(encoding="utf-8") != render_report(
        expected_manifest
    ):
        raise ValueError("exact-address report differs from deterministic render")
    return expected_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    run_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    verify_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    subparsers.add_parser("metamorphic")
    args = parser.parse_args()
    if args.command == "run":
        manifest = run(args.bundle.resolve(), args.out.resolve())
        print(json.dumps({
            "status": manifest["status"],
            "records": manifest["summary"]["records"],
            "certificates": manifest["summary"]["certificates"],
            "api_calls": 0,
            "gpu_used": False,
        }, sort_keys=True))
    elif args.command == "verify":
        manifest = verify_output(args.bundle.resolve(), args.out.resolve())
        print(json.dumps({
            "status": "verified_deterministic_replay",
            "records": manifest["summary"]["records"],
            "certificates": manifest["summary"]["certificates"],
        }, sort_keys=True))
    else:
        print(json.dumps(metamorphic_self_check(), sort_keys=True))


if __name__ == "__main__":
    main()
