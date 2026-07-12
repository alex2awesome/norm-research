"""Deterministic claim -> full-paper evidence verification.

The verifier is deliberately document-local: it fits retrieval statistics only on the
sentences of the paper being checked.  It never reads a judgement/acceptance label and
does not call an LLM.  Its certificates establish that a declared executable relation
was witnessed; they are not a claim that the paper, or the claim, is globally true.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable


_ABBREVIATIONS = {
    "e.g.", "i.e.", "et al.", "fig.", "figs.", "eq.", "eqs.", "sec.",
    "secs.", "ref.", "refs.", "vs.", "dr.", "prof.", "no.", "approx.",
}
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "in", "into",
    "is", "it", "its", "may", "method", "model", "of", "on", "or", "our", "paper",
    "result", "results", "show", "shows", "shown", "that", "the", "their", "these",
    "this", "to", "using", "was", "we", "were", "which", "with", "would",
}
_GENERIC_ENTITY = _STOP | {
    "approach", "baseline", "baselines", "existing", "previous", "prior", "proposed",
    "state", "art", "work", "works", "performance", "system", "systems", "than", "over",
    "across", "achieve", "achieves", "both", "collection", "common", "competitive", "dataset",
    "datasets", "demonstrate", "demonstrates", "extensive", "experiments", "superior", "through",
    "not",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*|\d+(?:\.\d+)?")
_CLAIM_RE = re.compile(
    r"\b(?:we\s+(?:show|find|demonstrate|prove|establish|achieve|obtain|report|observe|"
    r"introduce|present|propose|develop)|our\s+(?:method|model|approach|results?)|"
    r"results?\s+(?:show|demonstrate|indicate|confirm)|experiments?\s+(?:show|demonstrate|"
    r"confirm)|outperform|improv|better\s+than|worse\s+than|state[- ]of[- ]the[- ]art|"
    r"theorem|bound|convergen|significant)\b",
    re.I,
)
_RESULT_RE = re.compile(
    r"\b(?:experiment|evaluation|result|ablation|accuracy|error|loss|score|benchmark|dataset|"
    r"empirical|simulation|statistically|significant|effective|robust|benefit|improv|outperform|"
    r"perform|fail|success)\w*\b",
    re.I,
)
_THEORY_RE = re.compile(
    r"\b(?:proof|prove[sd]?|theorem|lemma|proposition|corollary|bound|guarantee|convergen|"
    r"optimality|complexity)\w*\b",
    re.I,
)
_EVIDENCE_RE = re.compile(
    r"\b(?:figure|fig\.?|table|tab\.?|equation|eq\.?|experiment|evaluation|result|ablation|"
    r"proof|theorem|lemma|appendix)\b",
    re.I,
)
_ASSERTION_RE = re.compile(
    r"\b(?:result|show|demonstrat|confirm|achiev|outperform|improv|increase|decrease|reduc|"
    r"effective|robust|valid|consistent|perform|fail|success|benefit|accuracy|error|loss|"
    r"correlat|significant)\w*\b",
    re.I,
)

# Comparison polarity describes whether the grammatical left-hand entity is claimed to
# have an advantage (+1) or disadvantage (-1).  Entity-role alignment is checked
# separately, so "ours beats BERT" and "BERT beats ours" cannot silently agree.
_COMPARATORS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\b(outperform(?:s|ed|ing)?|surpass(?:es|ed|ing)?|exceed(?:s|ed|ing)?|beat(?:s|en|ing)?)\b", re.I), 1),
    (re.compile(r"\b(improv(?:e|es|ed|ing))\s+(?:on|over|upon)\b", re.I), 1),
    (re.compile(r"\b(better|higher|faster|more accurate|more efficient)\s+than\b", re.I), 1),
    (re.compile(r"\b(underperform(?:s|ed|ing)?|worse|lower|slower|inferior)\s+than\b", re.I), -1),
    (re.compile(r"\b(is|are|was|were)\s+(outperformed|surpassed|exceeded|beaten)\s+by\b", re.I), -1),
)

_UNIT_ALIASES = {
    "%": "percent", "percent": "percent", "percentage": "percent",
    "percentage point": "percentage_point", "percentage points": "percentage_point",
    "point": "point", "points": "point", "x": "ratio", "times": "ratio", "fold": "ratio",
    "ms": "second", "millisecond": "second", "milliseconds": "second",
    "s": "second", "sec": "second", "secs": "second", "second": "second", "seconds": "second",
    "min": "second", "mins": "second", "minute": "second", "minutes": "second",
    "h": "second", "hr": "second", "hrs": "second", "hour": "second", "hours": "second",
    "kb": "byte", "mb": "byte", "gb": "byte", "byte": "byte", "bytes": "byte",
}
_UNIT_SCALE = {
    "%": 0.01, "percent": 0.01, "percentage": 0.01,
    "ms": 0.001, "millisecond": 0.001, "milliseconds": 0.001,
    "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "kb": 1_000.0, "mb": 1_000_000.0, "gb": 1_000_000_000.0,
}
_QUANTITY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<sign>[+\-−]?)\s*(?P<value>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\+?"
    r"\s*(?P<unit>percentage\s+points?|percent|%|points?|times|fold|x|milliseconds?|ms|"
    r"seconds?|secs?|s|minutes?|mins?|min|hours?|hrs?|hr|h|kilobytes?|megabytes?|gigabytes?|"
    r"bytes?|kb|mb|gb)?(?![A-Za-z])",
    re.I,
)


@dataclass(frozen=True)
class Sentence:
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Quantity:
    raw: str
    value: float
    unit: str
    start: int
    end: int


@dataclass(frozen=True)
class Comparison:
    cue: str
    polarity: int
    left_terms: tuple[str, ...]
    right_terms: tuple[str, ...]


@dataclass(frozen=True)
class Claim:
    index: int
    sentence: Sentence
    relation: str
    quantities: tuple[Quantity, ...]
    comparison: Comparison | None
    selection_score: float


@dataclass(frozen=True)
class Edge:
    claim_index: int
    evidence_index: int
    weight: float
    lexical_coverage: float
    bm25: float
    quantity_matches: int
    quantity_required: int
    relation_state: str
    decision: str
    witness_kind: str
    reason: str


def _normal_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def segment_sentences(text: str) -> list[Sentence]:
    """Segment prose while retaining offsets and protecting decimals/abbreviations."""

    text = text or ""
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    for i, char in enumerate(text):
        if char not in ".!?":
            continue
        if char == "." and i > 0 and i + 1 < n and text[i - 1].isdigit() and text[i + 1].isdigit():
            continue
        prefix = text[max(start, i - 10): i + 1].lower().strip()
        if any(prefix.endswith(abbr) for abbr in _ABBREVIATIONS):
            continue
        j = i + 1
        while j < n and text[j] in "\"')]}":
            j += 1
        if j < n and not text[j].isspace():
            continue
        while j < n and text[j].isspace():
            j += 1
        if j < n and not (text[j].isupper() or text[j].isdigit() or text[j] in "(["):
            continue
        spans.append((start, i + 1))
        start = j
    if start < n:
        spans.append((start, n))

    out: list[Sentence] = []
    for a, b in spans:
        raw = text[a:b]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        cleaned = _normal_text(raw)
        if len(cleaned) < 2:
            continue
        out.append(Sentence(len(out), a + leading, a + trailing, cleaned))
    return out


def tokens(text: str, *, content_only: bool = False) -> list[str]:
    result = [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]
    if content_only:
        result = [t for t in result if t not in _STOP and len(t) > 1]
    return result


def extract_quantities(text: str) -> tuple[Quantity, ...]:
    out: list[Quantity] = []
    for match in _QUANTITY_RE.finditer(text or ""):
        if match.start() > 1 and (text or "")[match.start() - 1] in "-_" and (text or "")[match.start() - 2].isalnum():
            continue
        if re.search(r"\b(?:table|tab\.?|figure|fig\.?|equation|eq\.?)\s*$",
                     (text or "")[max(0, match.start() - 18):match.start()], re.I):
            continue
        raw_value = match.group("value").replace(",", "")
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if match.group("sign") in {"-", "−"}:
            value = -value
        unit_raw = (match.group("unit") or "").lower().strip()
        before = (text or "")[max(0, match.start() - 12):match.start()]
        after = (text or "")[match.end():min(len(text or ""), match.end() + 4)]
        is_small_integer = value.is_integer() and 0 <= abs(value) <= 50
        if not unit_raw and is_small_integer:
            parenthesized_index = (
                bool(re.search(r"[\(\[]\s*$", before))
                and bool(re.match(r"\s*[\)\]]", after))
            )
            bare_list_index = bool(re.match(r"\s*[\)\.]\s+[A-Z]", after))
            if parenthesized_index or bare_list_index:
                continue
        # Avoid treating bare publication years as evidentiary measurements.
        if not unit_raw and value.is_integer() and 1900 <= value <= 2100:
            continue
        unit = _UNIT_ALIASES.get(unit_raw, "unitless")
        value *= _UNIT_SCALE.get(unit_raw, 1.0)
        out.append(Quantity(match.group(0), value, unit, match.start(), match.end()))
    return tuple(out)


def quantity_equal(left: Quantity, right: Quantity) -> bool:
    if left.unit != right.unit:
        return False
    scale = max(abs(left.value), abs(right.value), 1e-12)
    return abs(left.value - right.value) <= max(1e-9, 0.005 * scale)


def _entity_terms(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    has_self = bool(re.search(
        r"\b(?:we|us|ours|our\s+(?:method|model|approach|system)|this\s+(?:method|model|approach|system)|"
        r"the\s+proposed\s+(?:method|model|approach|system))\b",
        lowered,
    ))
    vals = [t for t in tokens(text, content_only=True)
            if t not in _GENERIC_ENTITY and not re.fullmatch(r"\d+(?:\.\d+)?", t)]
    if has_self:
        vals.append("selfmethod")
    return tuple(vals[-8:])


def extract_comparison(text: str) -> Comparison | None:
    for pattern, polarity in _COMPARATORS:
        match = pattern.search(text or "")
        if not match:
            continue
        left = text[max(0, match.start() - 120):match.start()]
        right = text[match.end():min(len(text), match.end() + 120)]
        # Limit to the nearest phrase; punctuation usually marks the relation boundary.
        left = re.split(r"[.;:]|\b(?:that|whether)\b", left, flags=re.I)[-1]
        right = re.split(r"[.;:]", right)[0]
        # Comparative complements precede margins, datasets and evaluation clauses.
        right = re.split(r"\b(?:by|on|with|using|for|at|while|under|across)\b", right, maxsplit=1, flags=re.I)[0]
        local_left = left[-45:]
        if re.search(r"\b(?:not|never|fails?\s+to|doesn['’]t|didn['’]t)\b", local_left, re.I):
            polarity *= -1
        return Comparison(
            cue=match.group(0),
            polarity=polarity,
            left_terms=_entity_terms(left),
            right_terms=tuple(_entity_terms(right)[:8]),
        )
    return None


def _relation(sentence: str, quantities: tuple[Quantity, ...], comp: Comparison | None) -> str:
    if comp is not None:
        return "comparative"
    if _THEORY_RE.search(sentence):
        return "theoretical"
    if quantities:
        return "numeric"
    if _RESULT_RE.search(sentence):
        return "empirical"
    return "qualitative"


def extract_claims(abstract: str, *, limit: int = 5) -> list[Claim]:
    candidates: list[Claim] = []
    for sentence in segment_sentences(abstract):
        quantities = extract_quantities(sentence.text)
        comparison = extract_comparison(sentence.text)
        score = 0.0
        if _CLAIM_RE.search(sentence.text):
            score += 2.0
        if comparison is not None:
            score += 2.0
        if quantities:
            score += 1.0
        if _RESULT_RE.search(sentence.text) or _THEORY_RE.search(sentence.text):
            score += 1.0
        if score < 2.0:
            continue
        candidates.append(Claim(
            index=len(candidates),
            sentence=sentence,
            relation=_relation(sentence.text, quantities, comparison),
            quantities=quantities,
            comparison=comparison,
            selection_score=score,
        ))
    # Prefer relation-rich claims but return them in document order for stable certificates.
    selected = sorted(candidates, key=lambda c: (-c.selection_score, c.sentence.index))[:limit]
    selected.sort(key=lambda c: c.sentence.index)
    return [Claim(i, c.sentence, c.relation, c.quantities, c.comparison, c.selection_score)
            for i, c in enumerate(selected)]


def _set_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class DocumentBM25:
    """BM25 index whose IDF is fit on one presented paper body only."""

    def __init__(self, sentences: list[Sentence], k1: float = 1.5, b: float = 0.75):
        self.sentences = sentences
        self.k1 = k1
        self.b = b
        self.docs = [tokens(s.text, content_only=True) for s in sentences]
        self.freqs = [Counter(doc) for doc in self.docs]
        self.lengths = [len(doc) for doc in self.docs]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        self.idf = {term: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
                    for term, count in df.items()}

    def score(self, query: str, doc_index: int) -> float:
        q = set(tokens(query, content_only=True))
        freq = self.freqs[doc_index]
        dl = self.lengths[doc_index]
        total = 0.0
        for term in q:
            tf = freq.get(term, 0)
            if not tf:
                continue
            denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(1e-9, self.avgdl))
            total += self.idf.get(term, 0.0) * tf * (self.k1 + 1.0) / denom
        return total

    def retrieve(self, query: str, *, k: int = 8) -> list[tuple[int, float]]:
        ranked = [(i, self.score(query, i)) for i in range(len(self.sentences))]
        ranked = [(i, s) for i, s in ranked if s > 0.0]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        return ranked[:k]


def _comparison_state(claim: Comparison | None, evidence: Comparison | None) -> str:
    if claim is None:
        return "not_required"
    if evidence is None:
        return "missing"
    direct_left = _set_overlap(claim.left_terms, evidence.left_terms)
    direct_right = _set_overlap(claim.right_terms, evidence.right_terms)
    reverse_left = _set_overlap(claim.left_terms, evidence.right_terms)
    reverse_right = _set_overlap(claim.right_terms, evidence.left_terms)
    direct = direct_left + direct_right
    reverse = reverse_left + reverse_right
    # A contradiction needs both entity roles, not incidental overlap with context words.
    if reverse > direct + 0.08 and reverse_left >= 0.50 and reverse_right >= 0.50:
        # Reversing both roles also reverses the sign needed for semantic agreement.
        return "aligned_reversed" if claim.polarity == -evidence.polarity else "reversed_roles"
    if direct_left < 0.15 or direct_right < 0.15:
        return "baseline_mismatch"
    if claim.polarity != evidence.polarity:
        return "direction_mismatch"
    return "aligned"


def _evaluate_edge(claim: Claim, evidence: Sentence, bm25: float) -> Edge | None:
    ctokens = tokens(claim.sentence.text, content_only=True)
    etokens = tokens(evidence.text, content_only=True)
    coverage = len(set(ctokens) & set(etokens)) / max(1, len(set(ctokens)))
    if coverage < 0.08:
        return None
    eq = extract_quantities(evidence.text)
    matches = sum(1 for q in claim.quantities if any(quantity_equal(q, e) for e in eq))
    comp_state = _comparison_state(claim.comparison, extract_comparison(evidence.text))
    decision = "insufficient"
    witness_kind = "none"
    reason = "retrieved_but_relation_not_certified"

    if claim.relation == "comparative":
        if comp_state in {"reversed_roles", "direction_mismatch"} and coverage >= 0.18:
            decision, witness_kind, reason = "contradicted", "relation_certificate", comp_state
        elif comp_state not in {"aligned", "aligned_reversed"}:
            reason = comp_state
        elif claim.quantities and matches < len(claim.quantities):
            reason = "claim_quantity_not_reproduced"
        elif coverage >= 0.16:
            decision, witness_kind, reason = "supported", "relation_certificate", "aligned_comparison"
    elif claim.relation == "numeric":
        if matches == len(claim.quantities) and matches > 0 and coverage >= 0.13:
            decision, witness_kind, reason = (
                "supported", "relation_certificate", "normalized_quantity_and_terms_match"
            )
        else:
            reason = "claim_quantity_not_reproduced"
    elif claim.relation == "theoretical":
        if _THEORY_RE.search(evidence.text) and coverage >= 0.18:
            decision, witness_kind, reason = (
                "evidence_link", "evidence_link", "theory_marker_and_terms_match"
            )
        else:
            reason = "missing_theory_witness"
    elif claim.relation == "empirical":
        if _EVIDENCE_RE.search(evidence.text) and _ASSERTION_RE.search(evidence.text) and coverage >= 0.20:
            decision, witness_kind, reason = (
                "evidence_link", "evidence_link", "empirical_artifact_and_terms_match"
            )
        else:
            reason = "missing_empirical_assertion_witness"
    elif _EVIDENCE_RE.search(evidence.text) and coverage >= 0.25:
        decision, witness_kind, reason = (
            "evidence_link", "evidence_link", "qualitative_evidence_and_terms_match"
        )

    relation_bonus = {
        "aligned": 1.0, "aligned_reversed": 1.0, "not_required": 0.45, "reversed_roles": 0.35,
        "direction_mismatch": 0.35, "missing": 0.0, "baseline_mismatch": 0.0,
    }.get(comp_state, 0.0)
    weight = coverage + 0.12 * math.log1p(bm25) + 0.20 * matches + relation_bonus
    return Edge(claim.index, evidence.index, weight, coverage, bm25, matches,
                len(claim.quantities), comp_state, decision, witness_kind, reason)


def _max_weight_matching(edges: list[Edge], nclaims: int) -> list[Edge]:
    """Exact maximum-weight one-to-one matching for the small claim graph."""

    by_claim: dict[int, list[Edge]] = {i: [] for i in range(nclaims)}
    for edge in edges:
        by_claim[edge.claim_index].append(edge)
    for values in by_claim.values():
        values.sort(key=lambda e: (-e.weight, e.evidence_index))

    @lru_cache(maxsize=None)
    def solve(ci: int, used: tuple[int, ...]) -> tuple[float, tuple[Edge, ...]]:
        if ci >= nclaims:
            return 0.0, ()
        used_set = set(used)
        best_score, best_edges = solve(ci + 1, used)
        for edge in by_claim.get(ci, []):
            if edge.evidence_index in used_set:
                continue
            nxt = tuple(sorted((*used, edge.evidence_index)))
            score, chosen = solve(ci + 1, nxt)
            score += edge.weight
            if score > best_score + 1e-12:
                best_score, best_edges = score, (edge, *chosen)
        return best_score, best_edges

    return sorted(solve(0, ())[1], key=lambda e: e.claim_index)


def _sentence_fingerprint(sentence: str) -> str:
    normalized = " ".join(tokens(sentence, content_only=False))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def verify_document(paper_id: str, abstract: str, body: str) -> dict[str, Any]:
    """Verify abstract claims against distinct sentences from the presented paper body."""

    abstract = abstract or ""
    body = body or ""
    if not abstract.strip():
        return {"paper_id": paper_id, "status": "abstain", "reason": "missing_abstract",
                "claim_count": 0, "certificate_count": 0, "evidence_link_count": 0,
                "certificates": [], "evidence_links": [], "matches": []}
    if not body.strip():
        return {"paper_id": paper_id, "status": "abstain", "reason": "missing_fullpaper_body",
                "claim_count": 0, "certificate_count": 0, "evidence_link_count": 0,
                "certificates": [], "evidence_links": [], "matches": []}

    abstract_sentences = segment_sentences(abstract)
    abstract_hashes = {_sentence_fingerprint(s.text) for s in abstract_sentences}
    body_all = segment_sentences(body)
    # Repeated abstract prose is claim material, not independent body evidence.
    body_sentences = [s for s in body_all if _sentence_fingerprint(s.text) not in abstract_hashes]
    body_sentences = [Sentence(i, s.start, s.end, s.text) for i, s in enumerate(body_sentences)]
    if not body_sentences:
        return {"paper_id": paper_id, "status": "abstain", "reason": "abstract_only_no_independent_evidence",
                "claim_count": 0, "certificate_count": 0, "evidence_link_count": 0,
                "certificates": [], "evidence_links": [], "matches": []}

    claims = extract_claims(abstract)
    if not claims:
        return {"paper_id": paper_id, "status": "abstain", "reason": "no_executable_claim_relation",
                "claim_count": 0, "certificate_count": 0, "evidence_link_count": 0,
                "certificates": [], "evidence_links": [], "matches": []}

    index = DocumentBM25(body_sentences)
    edges: list[Edge] = []
    for claim in claims:
        for evidence_index, score in index.retrieve(claim.sentence.text, k=8):
            edge = _evaluate_edge(claim, body_sentences[evidence_index], score)
            if edge is not None:
                edges.append(edge)
    if not edges:
        return {"paper_id": paper_id, "status": "abstain", "reason": "no_retrievable_evidence",
                "claim_count": len(claims), "certificate_count": 0, "evidence_link_count": 0,
                "certificates": [], "evidence_links": [], "matches": [],
                "graph": {"claim_nodes": len(claims), "evidence_nodes": len(body_sentences), "edges": 0}}

    matched = _max_weight_matching(edges, len(claims))
    matches: list[dict[str, Any]] = []
    for edge in matched:
        claim = claims[edge.claim_index]
        evidence = body_sentences[edge.evidence_index]
        matches.append({
            "decision": edge.decision,
            "witness_kind": edge.witness_kind,
            "reason": edge.reason,
            "claim": {
                "sentence_index": claim.sentence.index,
                "text": claim.sentence.text,
                "relation": claim.relation,
                "quantities": [asdict(q) for q in claim.quantities],
                "comparison": asdict(claim.comparison) if claim.comparison else None,
            },
            "evidence": {
                "sentence_index": evidence.index,
                "start": evidence.start,
                "end": evidence.end,
                "text": evidence.text,
                "quantities": [asdict(q) for q in extract_quantities(evidence.text)],
                "comparison": asdict(extract_comparison(evidence.text)) if extract_comparison(evidence.text) else None,
            },
            "checks": {
                "bm25": round(edge.bm25, 6),
                "claim_term_coverage": round(edge.lexical_coverage, 6),
                "quantity_matches": edge.quantity_matches,
                "quantity_required": edge.quantity_required,
                "relation_state": edge.relation_state,
            },
        })

    decisions = Counter(c["decision"] for c in matches)
    certificates = [m for m in matches if m["witness_kind"] == "relation_certificate"]
    evidence_links = [m for m in matches if m["witness_kind"] == "evidence_link"]
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
        "graph": {
            "claim_nodes": len(claims),
            "evidence_nodes": len(body_sentences),
            "edges": len(edges),
            "matched_edges": len(matched),
            "matching": "exact_max_weight_bipartite",
        },
    }


def metamorphic_self_check() -> dict[str, bool]:
    """Executable invariants run before a corpus result can be written."""

    abstract = "We show that our method outperforms BERT by 12.5% on image accuracy."
    evidence = (
        "We evaluate both systems on the held-out image benchmark. "
        "Table 2 shows that our method outperforms BERT by 12.5% on image accuracy."
    )
    original = verify_document("metamorphic", abstract, evidence)
    number_removed = verify_document(
        "metamorphic", abstract, evidence.replace("12.5%", "the reported margin")
    )
    direction_swapped = verify_document(
        "metamorphic",
        abstract,
        evidence.replace("our method outperforms BERT", "BERT outperforms our method"),
    )
    baseline_swapped = verify_document(
        "metamorphic", abstract, evidence.replace("BERT", "RoBERTa")
    )
    checks = {
        "original_relation_certifies_support": original["status"] == "supported",
        "remove_evidence_number_invalidates_support": number_removed["status"] != "supported",
        "swap_entity_direction_certifies_contradiction": direction_swapped["status"] == "contradicted",
        "swap_baseline_invalidates_support": baseline_swapped["status"] != "supported",
        "abstract_only_abstains": verify_document("metamorphic", abstract, abstract)["status"] == "abstain",
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"metamorphic verifier invariant(s) failed: {failed}")
    return checks
