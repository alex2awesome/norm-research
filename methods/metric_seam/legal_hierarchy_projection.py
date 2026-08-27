"""Additive, label-free legal-writing relation projections.

The historical ``hybrids/programs_legal`` fleet scores Title VII fact-pattern
criteria.  The R1/R2/R3 hierarchy panel instead contains legal-writing
criteria.  Replaying the old whole scores against that panel would therefore
be a construct substitution.  This module preserves the old files and ports
only capability-shaped ideas that have an exact relation-local target:

* date/event arithmetic from a23/a26/a36/a39;
* finite numeric checks from a21/a28;
* concrete-fact grounding from a34/a46;
* quotation/actor structure from a8/a10.

It adds document, dependency, entity, citation, definition, and discourse
algorithms needed by the writing criteria.  Outputs are measurements of named
sub-relations, never a whole legal-writing score.  There are no prompt fields,
outcomes, references, retrieval calls, network calls, or corpus state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence


SCHEMA = "metric-seam.legal-hierarchy-relation-projection.v1"
PROGRAM_VERSION = "legal-writing-projection-v1"

DEPTH_MEANINGS = {
    1: "surface text, finite lexicon, or local regular-language measurement",
    2: "dependency/entity parse, graph aggregation, date arithmetic, or finite consistency check",
}


RELATIONS: tuple[dict[str, Any], ...] = (
    {
        "relation_id": "plain_language_surface",
        "effective_code_depth": 1,
        "implemented_relation": "density of frozen legalese/wordiness forms and long words",
        "exclusions": ["audience fit", "legal accuracy", "whether a technical term is necessary"],
        "historical_seed_ids": ["a0", "a23", "a49"],
    },
    {
        "relation_id": "sentence_clarity_parse",
        "effective_code_depth": 2,
        "implemented_relation": "parsed sentence length, dependency span, and subordinate-clause load",
        "exclusions": ["semantic clarity", "rhetorical rhythm quality", "legal accuracy"],
        "historical_seed_ids": [],
    },
    {
        "relation_id": "active_voice_parse",
        "effective_code_depth": 2,
        "implemented_relation": "finite-clause active/passive structure and nominalization incidence",
        "exclusions": ["whether passive voice is contextually justified", "verb precision"],
        "historical_seed_ids": ["a10", "a36"],
    },
    {
        "relation_id": "negation_stack_parse",
        "effective_code_depth": 2,
        "implemented_relation": "multiple dependency-linked negations within one sentence",
        "exclusions": ["semantic scope resolution", "whether a negative is legally necessary"],
        "historical_seed_ids": [],
    },
    {
        "relation_id": "concrete_fact_anchors",
        "effective_code_depth": 2,
        "implemented_relation": "named entities grounded near dates, quantities, quotations, or money",
        "exclusions": ["truth", "materiality", "selection quality", "narrative vividness"],
        "historical_seed_ids": ["a28", "a34", "a46"],
    },
    {
        "relation_id": "temporal_order_graph",
        "effective_code_depth": 2,
        "implemented_relation": "strictly parsed dated-event sequence and explicit signaling of backward transitions",
        "exclusions": ["event truth", "causation", "whether chronology is the best organization"],
        "historical_seed_ids": ["a23", "a26", "a36", "a39"],
    },
    {
        "relation_id": "numeric_consistency_check",
        "effective_code_depth": 2,
        "implemented_relation": "recomputed count/percentage and before/after percentage consistency",
        "exclusions": ["unreported denominators", "truth of source numbers", "non-local arithmetic"],
        "historical_seed_ids": ["a21", "a28"],
    },
    {
        "relation_id": "definition_use_graph",
        "effective_code_depth": 2,
        "implemented_relation": "explicit acronym/defined-term declarations linked to later uses",
        "exclusions": ["substantive definition precision", "missing concepts never declared"],
        "historical_seed_ids": ["a46"],
    },
    {
        "relation_id": "citation_format_structure",
        "effective_code_depth": 2,
        "implemented_relation": "finite legal-citation spans, duplicates, and local explanatory-parenthetical structure",
        "exclusions": ["authority validity", "pinpoint accuracy", "precedential weight", "citation completeness"],
        "historical_seed_ids": ["a23", "a49"],
    },
    {
        "relation_id": "quote_attribution_parse",
        "effective_code_depth": 2,
        "implemented_relation": "quoted spans linked to reporting predicates and syntactic actor candidates",
        "exclusions": ["quote accuracy against source", "speaker authority", "quotation necessity"],
        "historical_seed_ids": ["a8", "a10"],
    },
    {
        "relation_id": "discourse_cohesion_graph",
        "effective_code_depth": 2,
        "implemented_relation": "content-lemma overlap graph connectivity across adjacent and nonadjacent sentences",
        "exclusions": ["logical validity", "case-theory quality", "persuasive force"],
        "historical_seed_ids": ["a10", "a18"],
    },
    {
        "relation_id": "paragraph_cohesion_graph",
        "effective_code_depth": 2,
        "implemented_relation": "within-paragraph linkage to the first sentence and single-topic lexical concentration",
        "exclusions": ["claim-bearing topic-sentence semantics", "logical progression"],
        "historical_seed_ids": [],
    },
    {
        "relation_id": "frontloaded_disposition_structure",
        "effective_code_depth": 2,
        "implemented_relation": "issue, result, or requested-relief cues located in opening/coda document zones",
        "exclusions": ["whether the point is dispositive", "strength of reason", "correct relief"],
        "historical_seed_ids": ["a3", "a13", "a18"],
    },
    {
        "relation_id": "counterposition_structure",
        "effective_code_depth": 2,
        "implemented_relation": "paired party-position predicates and explicit contrast/rebuttal linkage",
        "exclusions": ["fairness", "completeness", "substantive response quality"],
        "historical_seed_ids": ["a3", "a10", "a18"],
    },
    {
        "relation_id": "tone_restraint_surface",
        "effective_code_depth": 1,
        "implemented_relation": "incendiary, ad-hominem, and intensifier surface density",
        "exclusions": ["audience alignment", "contextual appropriateness", "implicit disrespect"],
        "historical_seed_ids": ["a8", "a13"],
    },
    {
        "relation_id": "heading_roadmap_structure",
        "effective_code_depth": 2,
        "implemented_relation": "heading-like lines, nesting markers, and roadmap cues",
        "exclusions": ["argumentative substance", "heading truth", "quality of hierarchy"],
        "historical_seed_ids": [],
    },
    {
        "relation_id": "question_frame_structure",
        "effective_code_depth": 2,
        "implemented_relation": "early question-presented cues, interrogative syntax, and yes/no framing",
        "exclusions": ["decisiveness", "case-posture fit", "material issue completeness"],
        "historical_seed_ids": [],
    },
    {
        "relation_id": "inclusive_language_surface",
        "effective_code_depth": 1,
        "implemented_relation": "frozen generic-gender and demeaning-label surface forms",
        "exclusions": ["implicit bias", "referential correctness", "naturalness of a rewrite"],
        "historical_seed_ids": ["a0", "a31", "a41"],
    },
    {
        "relation_id": "deadline_remedy_consequence_structure",
        "effective_code_depth": 2,
        "implemented_relation": "co-occurrence graph over demanded remedy, deadline, and stated consequence",
        "exclusions": ["legal entitlement", "strategic calibration", "evidentiary sufficiency"],
        "historical_seed_ids": ["a26", "a39"],
    },
)

RELATION_BY_ID = {row["relation_id"]: row for row in RELATIONS}


_LEGALESE = {
    "aforementioned", "aforesaid", "hereafter", "hereby", "herein", "hereinafter",
    "heretofore", "hereunder", "notwithstanding", "pursuant", "thereafter", "thereby",
    "therein", "thereof", "thereunder", "whereas", "whereby", "witnesseth",
}
_WORDY_PHRASES = (
    "at this point in time", "by means of", "for the purpose of", "in accordance with",
    "in order to", "in the event that", "prior to", "subsequent to", "with respect to",
)
_INTENSIFIERS = {
    "clearly", "obviously", "undeniably", "indisputably", "plainly", "certainly",
    "absolutely", "utterly", "egregious", "shameless", "ridiculous", "absurd",
}
_AD_HOMINEM = {
    "liar", "dishonest", "corrupt", "fraudster", "incompetent", "scurrilous",
    "outrageous", "bad faith", "witch hunt", "frivolous stunt",
}
_GENERIC_GENDER = (
    re.compile(r"\b(?:reasonable|average|ordinary)\s+man\b", re.I),
    re.compile(r"\bchairman\b|\bpoliceman\b|\bworkman\b", re.I),
    re.compile(r"\bthe\s+(?:judge|lawyer|juror)\s+[^.!?]{0,30}\b(?:he|his)\b", re.I),
)
_REPORTING_LEMMAS = {
    "say", "state", "claim", "argue", "allege", "contend", "report", "testify",
    "explain", "write", "note", "admit", "deny", "tell", "assert", "respond",
}
_POSITION_LEMMAS = {"allege", "argue", "claim", "contend", "assert", "maintain", "deny"}
_PARTY_WORDS = {"plaintiff", "defendant", "petitioner", "respondent", "appellant", "appellee"}
_CONTRAST = re.compile(
    r"\b(?:however|although|but|yet|nevertheless|nonetheless|in contrast|on the other hand|"
    r"responds?|rebut(?:s|ted)?|counters?)\b", re.I
)
_OPEN_CUES = re.compile(
    r"\b(?:the (?:issue|question) is|we (?:hold|conclude|find)|this (?:case|action) concerns|"
    r"plaintiff (?:seeks|requests)|summary judgment|motion to dismiss|the court (?:holds|finds))\b",
    re.I,
)
_RELIEF_CUES = re.compile(
    r"\b(?:therefore|accordingly|for these reasons|wherefore|prays? for|requests? that|"
    r"dismiss(?:ed|al)?|affirm(?:ed)?|reverse(?:d)?|remand(?:ed)?|grant(?:ed)?|deny|denied)\b",
    re.I,
)
_ROADMAP_CUES = re.compile(
    r"\b(?:first|second|third|finally|this (?:section|part)|we begin|turn next|as follows|"
    r"the reasons are|in sum|to summarize)\b", re.I
)
_QUESTION_CUE = re.compile(r"\b(?:question presented|issue presented|whether)\b", re.I)
_REMEDY = re.compile(
    r"\b(?:cease|desist|remove|retract|preserve|pay|compensat\w*|reinstate|correct|cure|refund)\b",
    re.I,
)
_DEADLINE = re.compile(
    r"\b(?:within\s+(?:\d+|one|two|three|five|seven|ten|fourteen|thirty)\s+days?|"
    r"no later than|by\s+(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2}|deadline)\b",
    re.I,
)
_CONSEQUENCE = re.compile(
    r"\b(?:otherwise|failing which|if you do not|legal action|seek an injunction|file suit|"
    r"pursue all remedies|without further notice)\b",
    re.I,
)

_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+"
    r"\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?"
    r"|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s*,?\s*\d{4}|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.I,
)
_TEMPORAL_SIGNAL = re.compile(
    r"\b(?:earlier|later|before|after|previously|subsequently|meanwhile|by then|"
    r"the following|the next|that same day|years? later)\b", re.I
)
_COUNT_PCT_RE = re.compile(
    r"(?P<num>\d[\d,]*)\s*(?:out of|of|/)\s*(?P<denom>\d[\d,]*)"
    r"[^.\n%]{0,25}?\(?\s*(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent)\)?", re.I
)
_DELTA_PCT_RE = re.compile(
    r"from\s+(?P<a>\d[\d,]*(?:\.\d+)?)\s+to\s+(?P<b>\d[\d,]*(?:\.\d+)?)"
    r"[^.\n%]{0,40}?(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent)", re.I
)
_ACRONYM_DEF_RE = re.compile(
    r"\b(?P<long>[A-Z][A-Za-z&' -]{3,80}?)\s*\((?P<acro>[A-Z][A-Z0-9.-]{1,12})\)"
)
_QUOTED_DEF_RE = re.compile(
    r"(?P<term>[\"“][^\"”\n]{2,60}[\"”])\s+(?:means|shall mean|refers to)\b", re.I
)
_CASE_CITE_RE = re.compile(r"\b\d{1,4}\s+[A-Z][A-Za-z.]{0,15}\s+\d{1,6}\b")
_STATUTE_CITE_RE = re.compile(
    r"\b\d+\s+U\.?\s*S\.?\s*C\.?\s*(?:§{1,2}|sec(?:tion)?\.?)\s*[\w().-]+", re.I
)
_RULE_CITE_RE = re.compile(r"\b(?:Fed\.?\s*R\.?\s*(?:Civ|Crim|Evid)\.?\s*P\.?|Rule)\s*\d+(?:\([a-z0-9]+\))*", re.I)
_PAREN_EXPLANATION_RE = re.compile(r"\((?:holding|explaining|finding|stating|noting|recognizing)\b", re.I)
_QUOTE_RE = re.compile(r"[\"“](?P<body>[^\"”\n]{3,600})[\"”]")
_HEADING_RE = re.compile(r"^(?:[IVXLC]+\.|[A-Z]\.|\d+\.|[A-Z][A-Z0-9 ,:'’&-]{3,80})\s*$")


@dataclass(frozen=True)
class ProjectionContext:
    text: str
    doc: Any


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _span(text: str, start: int, end: int) -> dict[str, Any]:
    return {"start": start, "end": end, "surface": text[start:end]}


def _load_nlp():
    """Load the frozen CPU spaCy parser. Fail closed rather than regex-guessing."""

    try:
        import spacy

        spacy.require_cpu()
        nlp = spacy.load(
            "en_core_web_sm",
            disable=[],
        )
        nlp.max_length = 2_000_000
        return nlp
    except Exception:
        return None


def _plain_language(ctx: ProjectionContext) -> dict[str, Any]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", ctx.text)
    if not words:
        return {"value": None, "applicable": False, "certificates": []}
    low = [word.casefold() for word in words]
    legalese = sum(word in _LEGALESE for word in low)
    phrase_hits = sum(len(re.findall(re.escape(phrase), ctx.text, re.I)) for phrase in _WORDY_PHRASES)
    long_words = sum(len(word) >= 10 for word in words)
    penalty = 7.0 * (legalese + phrase_hits) / len(words) + 0.8 * long_words / len(words)
    certs = [
        {"kind": "surface_witness", **_span(ctx.text, m.start(), m.end())}
        for term in sorted(_LEGALESE)
        for m in list(re.finditer(rf"\b{re.escape(term)}\b", ctx.text, re.I))[:3]
    ][:12]
    return {"value": _clip01(1.0 - penalty), "applicable": True, "certificates": certs}


def _sentence_clarity(ctx: ProjectionContext) -> dict[str, Any]:
    doc = ctx.doc
    if doc is None:
        return {"value": None, "applicable": False, "certificates": []}
    sents = list(doc.sents)
    if not sents:
        return {"value": None, "applicable": False, "certificates": []}
    alpha_counts = [sum(token.is_alpha for token in sent) for sent in sents]
    mean_words = sum(alpha_counts) / len(alpha_counts)
    long_fraction = sum(count > 30 for count in alpha_counts) / len(alpha_counts)
    sub_deps = {"advcl", "ccomp", "xcomp", "relcl", "acl"}
    subordinate = sum(token.dep_ in sub_deps for token in doc) / len(sents)
    dep_spans = [abs(token.i - token.head.i) for token in doc if token.is_alpha and token.head is not token]
    mean_dep_span = sum(dep_spans) / len(dep_spans) if dep_spans else 0.0
    penalty = (
        max(0.0, mean_words - 18.0) / 30.0
        + 0.55 * long_fraction
        + max(0.0, subordinate - 1.2) / 5.0
        + max(0.0, mean_dep_span - 3.0) / 18.0
    )
    worst = sorted(zip(alpha_counts, sents), key=lambda row: row[0], reverse=True)[:3]
    certs = [
        {"kind": "long_sentence", "word_count": count, **_span(ctx.text, sent.start_char, sent.end_char)}
        for count, sent in worst if count > 30
    ]
    return {"value": _clip01(1.0 - penalty), "applicable": True, "certificates": certs}


def _active_voice(ctx: ProjectionContext) -> dict[str, Any]:
    doc = ctx.doc
    if doc is None:
        return {"value": None, "applicable": False, "certificates": []}
    finite = [token for token in doc if token.pos_ in {"VERB", "AUX"} and token.dep_ not in {"aux", "auxpass"}]
    if not finite:
        return {"value": None, "applicable": False, "certificates": []}
    passive = [
        token for token in finite
        if any(child.dep_ in {"auxpass", "nsubjpass"} for child in token.children)
        or token.dep_ == "ROOT" and any(child.dep_ == "nsubjpass" for child in token.children)
    ]
    nouns = [token for token in doc if token.pos_ == "NOUN" and token.text.casefold().endswith(("tion", "ment", "ance", "ence", "ity"))]
    active_fraction = 1.0 - len(passive) / len(finite)
    nominal_penalty = min(0.3, len(nouns) / max(1, len(finite)) * 0.15)
    certs = [
        {"kind": "passive_clause", "lemma": token.lemma_, **_span(ctx.text, token.idx, token.idx + len(token.text))}
        for token in passive[:12]
    ]
    return {"value": _clip01(active_fraction - nominal_penalty), "applicable": True, "certificates": certs}


def _negation_stack(ctx: ProjectionContext) -> dict[str, Any]:
    doc = ctx.doc
    if doc is None:
        return {"value": None, "applicable": False, "certificates": []}
    sents = list(doc.sents)
    if not sents:
        return {"value": None, "applicable": False, "certificates": []}
    stacked = []
    for sent in sents:
        negs = [token for token in sent if token.dep_ == "neg" or token.lower_ in {"no", "neither", "nor", "without"}]
        if len(negs) >= 2:
            stacked.append((sent, negs))
    certs = [
        {
            "kind": "stacked_negation",
            "negation_tokens": [token.text for token in negs],
            **_span(ctx.text, sent.start_char, sent.end_char),
        }
        for sent, negs in stacked[:10]
    ]
    return {"value": _clip01(1.0 - len(stacked) / len(sents)), "applicable": True, "certificates": certs}


def _concrete_facts(ctx: ProjectionContext) -> dict[str, Any]:
    doc = ctx.doc
    if doc is None:
        return {"value": None, "applicable": False, "certificates": []}
    word_count = max(1, sum(token.is_alpha for token in doc))
    anchors = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"}:
            anchors.append(ent)
    date_or_num = [ent for ent in anchors if ent.label_ in {"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"}]
    density = 100.0 * len(anchors) / word_count
    grounded_names = 0
    for ent in anchors:
        if ent.label_ not in {"PERSON", "ORG", "GPE", "LOC"}:
            continue
        if any(abs(other.start_char - ent.end_char) <= 100 or abs(ent.start_char - other.end_char) <= 100 for other in date_or_num):
            grounded_names += 1
    value = 1.0 - math.exp(-(density + 0.7 * grounded_names) / 5.0)
    certs = [
        {"kind": "fact_anchor", "entity_type": ent.label_, **_span(ctx.text, ent.start_char, ent.end_char)}
        for ent in anchors[:20]
    ]
    return {"value": _clip01(value), "applicable": True, "certificates": certs}


def _parse_dates(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Strict additive replacement for the frozen silent-drop date helper.

    Invalid surfaces (for example April 31) are emitted as abstention
    certificates rather than silently removed.  Yearless dates are valid
    anchors but are not compared across document order.
    """

    try:
        from dateutil import parser as date_parser
        import datetime
    except ImportError:
        return [], []
    valid, invalid = [], []
    for match in _DATE_RE.finditer(text):
        surface = match.group(0)
        has_year = bool(
            re.search(r"\b(?:19|20)\d{2}\b", surface)
            or re.search(r"/\d{2,4}\s*$", surface)
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}", surface)
        )
        try:
            parsed = date_parser.parse(
                surface,
                fuzzy=False,
                default=datetime.datetime(2000, 1, 1),
            ).date()
        except (ValueError, OverflowError):
            invalid.append({"kind": "invalid_date_abstention", **_span(text, match.start(), match.end())})
            continue
        valid.append(
            {
                "kind": "date_anchor",
                "start": match.start(),
                "end": match.end(),
                "surface": surface,
                "iso_date": parsed.isoformat(),
                "comparable": has_year,
                "date": parsed,
            }
        )
    return valid, invalid


def _temporal_order(ctx: ProjectionContext) -> dict[str, Any]:
    dates, invalid = _parse_dates(ctx.text)
    comparable = [row for row in dates if row["comparable"]]
    if len(comparable) < 2:
        return {"value": None, "applicable": False, "certificates": invalid + [{k: v for k, v in row.items() if k != "date"} for row in dates[:10]]}
    edges = []
    good = 0.0
    for left, right in zip(comparable, comparable[1:]):
        delta = (right["date"] - left["date"]).days
        lo = left["end"]
        hi = right["start"]
        signal = bool(_TEMPORAL_SIGNAL.search(ctx.text[max(0, lo - 30):min(len(ctx.text), hi + 30)]))
        edge_good = delta >= 0 or signal
        good += float(edge_good)
        edges.append(
            {
                "kind": "date_order_edge",
                "left": left["iso_date"],
                "right": right["iso_date"],
                "days": delta,
                "backward_transition_signaled": bool(delta < 0 and signal),
                "accepted": edge_good,
            }
        )
    certs = invalid + [{k: v for k, v in row.items() if k != "date"} for row in dates[:12]] + edges[:12]
    return {"value": good / len(edges), "applicable": True, "certificates": certs}


def _numeric_consistency(ctx: ProjectionContext) -> dict[str, Any]:
    checks = []
    for match in _COUNT_PCT_RE.finditer(ctx.text):
        num = float(match.group("num").replace(",", ""))
        denom = float(match.group("denom").replace(",", ""))
        stated = float(match.group("pct"))
        if denom <= 0:
            continue
        computed = 100.0 * num / denom
        tolerance = max(1.0, 0.08 * abs(stated))
        checks.append(
            {
                "kind": "count_percentage_check",
                "computed": round(computed, 4),
                "stated": stated,
                "consistent": abs(computed - stated) <= tolerance,
                **_span(ctx.text, match.start(), match.end()),
            }
        )
    for match in _DELTA_PCT_RE.finditer(ctx.text):
        left = float(match.group("a").replace(",", ""))
        right = float(match.group("b").replace(",", ""))
        stated = float(match.group("pct"))
        if left == 0:
            continue
        computed = 100.0 * (right - left) / left
        tolerance = max(1.0, 0.08 * abs(stated))
        checks.append(
            {
                "kind": "delta_percentage_check",
                "computed": round(computed, 4),
                "stated": stated,
                "consistent": abs(abs(computed) - abs(stated)) <= tolerance,
                **_span(ctx.text, match.start(), match.end()),
            }
        )
    if not checks:
        return {"value": None, "applicable": False, "certificates": []}
    return {
        "value": sum(check["consistent"] for check in checks) / len(checks),
        "applicable": True,
        "certificates": checks[:20],
    }


def _definition_graph(ctx: ProjectionContext) -> dict[str, Any]:
    declarations = []
    for match in _ACRONYM_DEF_RE.finditer(ctx.text):
        term = match.group("acro").strip(".")
        declarations.append((term, match.start(), match.end(), match.group("long").strip()))
    for match in _QUOTED_DEF_RE.finditer(ctx.text):
        term = match.group("term").strip('"“”')
        declarations.append((term, match.start(), match.end(), term))
    if not declarations:
        return {"value": None, "applicable": False, "certificates": []}
    certs, used = [], 0
    for term, start, end, expansion in declarations:
        later = ctx.text[end:]
        count = len(re.findall(rf"\b{re.escape(term)}\b", later))
        used += int(count > 0)
        certs.append(
            {
                "kind": "definition_edge",
                "term": term,
                "expansion": expansion,
                "later_use_count": count,
                **_span(ctx.text, start, end),
            }
        )
    return {"value": used / len(declarations), "applicable": True, "certificates": certs[:20]}


def _citations(ctx: ProjectionContext) -> dict[str, Any]:
    matches = []
    for kind, pattern in (
        ("case_citation", _CASE_CITE_RE),
        ("statute_citation", _STATUTE_CITE_RE),
        ("rule_citation", _RULE_CITE_RE),
    ):
        for match in pattern.finditer(ctx.text):
            matches.append((match.start(), match.end(), kind, match.group(0)))
    matches.sort()
    if not matches:
        return {"value": None, "applicable": False, "certificates": []}
    normalized = [re.sub(r"\s+", " ", surface.casefold()) for _, _, _, surface in matches]
    duplicates = len(normalized) - len(set(normalized))
    explanatory = sum(bool(_PAREN_EXPLANATION_RE.search(ctx.text[end:end + 100])) for _, end, _, _ in matches)
    # This deliberately scores structure, not authority correctness. Duplicate
    # citations are mildly penalized; explanatory parentheticals add evidence.
    value = _clip01(0.65 + 0.25 * explanatory / len(matches) - 0.15 * duplicates / len(matches))
    certs = [
        {"kind": kind, **_span(ctx.text, start, end)}
        for start, end, kind, _ in matches[:30]
    ]
    return {"value": value, "applicable": True, "certificates": certs}


def _quote_attribution(ctx: ProjectionContext) -> dict[str, Any]:
    doc = ctx.doc
    quotes = list(_QUOTE_RE.finditer(ctx.text))
    if not quotes:
        return {"value": None, "applicable": False, "certificates": []}
    certs, attributed = [], 0
    for quote in quotes:
        lo, hi = max(0, quote.start() - 180), min(len(ctx.text), quote.end() + 180)
        linked = False
        actor_surfaces: set[str] = set()
        if doc is not None:
            tokens = [token for token in doc if lo <= token.idx < hi]
            reporting = [token for token in tokens if token.lemma_.casefold() in _REPORTING_LEMMAS]
            for verb in reporting:
                # Covers ordinary subjects, passive agents, and coordinated
                # subjects/action beats without accepting an unrelated verb in
                # another sentence as attribution.
                sentence_contains_quote = verb.sent.start_char <= quote.start() < verb.sent.end_char
                if not sentence_contains_quote:
                    continue
                linked = True
                for child in verb.children:
                    if child.dep_ in {"nsubj", "nsubjpass", "agent"}:
                        subtree = list(child.subtree)
                        actor_surfaces.add(ctx.text[subtree[0].idx:subtree[-1].idx + len(subtree[-1].text)])
                        for conjunct in child.conjuncts:
                            actor_surfaces.add(conjunct.text)
        if not linked:
            sentence_zone = ctx.text[lo:hi]
            linked = bool(re.search(r"\b(?:said|stated|argued|testified|wrote|reported|explained|claimed|contended)\b", sentence_zone, re.I))
        attributed += int(linked)
        certs.append(
            {
                "kind": "quote_attribution_edge" if linked else "unattributed_quote",
                "actors": sorted(actor_surfaces),
                **_span(ctx.text, quote.start(), quote.end()),
            }
        )
    return {"value": attributed / len(quotes), "applicable": True, "certificates": certs[:20]}


def _sentence_bow(sent: Any) -> set[str]:
    return {
        token.lemma_.casefold()
        for token in sent
        if token.is_alpha and not token.is_stop and len(token.text) > 1
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _discourse_cohesion(ctx: ProjectionContext) -> dict[str, Any]:
    doc = ctx.doc
    if doc is None:
        return {"value": None, "applicable": False, "certificates": []}
    sents = list(doc.sents)[:240]
    if len(sents) < 2:
        return {"value": None, "applicable": False, "certificates": []}
    bows = [_sentence_bow(sent) for sent in sents]
    edges = []
    adjacency = [set() for _ in sents]
    for i, left in enumerate(bows):
        for j in range(i + 1, min(len(bows), i + 9)):
            weight = _jaccard(left, bows[j])
            if weight > 0.08:
                adjacency[i].add(j)
                adjacency[j].add(i)
                edges.append((i, j, weight))
    if not edges:
        return {"value": 0.0, "applicable": True, "certificates": []}
    seen, largest = set(), 0
    for start in range(len(sents)):
        if start in seen:
            continue
        stack, size = [start], 0
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            size += 1
            stack.extend(adjacency[node] - seen)
        largest = max(largest, size)
    connected = largest / len(sents)
    adjacent_overlap = sum(_jaccard(bows[i], bows[i + 1]) for i in range(len(sents) - 1)) / (len(sents) - 1)
    value = _clip01(0.7 * connected + 0.3 * min(1.0, adjacent_overlap / 0.2))
    certs = [
        {"kind": "sentence_graph_edge", "left_sentence": i, "right_sentence": j, "weight": round(weight, 6)}
        for i, j, weight in sorted(edges, key=lambda row: row[2], reverse=True)[:20]
    ]
    return {"value": value, "applicable": True, "certificates": certs}


def _paragraph_cohesion(ctx: ProjectionContext) -> dict[str, Any]:
    doc = ctx.doc
    if doc is None:
        return {"value": None, "applicable": False, "certificates": []}
    sentence_rows = [(sent.start_char, sent.end_char, sent, _sentence_bow(sent)) for sent in doc.sents]
    paragraphs = []
    for match in re.finditer(r"(?:^|\n\s*\n)(?P<body>\S(?:.*?\S)?)(?=\n\s*\n|$)", ctx.text, re.S):
        rows = [row for row in sentence_rows if match.start("body") <= row[0] < match.end("body")]
        if len(rows) >= 2:
            paragraphs.append(rows)
    if not paragraphs:
        return {"value": None, "applicable": False, "certificates": []}
    scores, certs = [], []
    for rows in paragraphs:
        topic = rows[0][3]
        links = [_jaccard(topic, row[3]) for row in rows[1:]]
        score = sum(link > 0.05 for link in links) / len(links)
        scores.append(score)
        certs.append(
            {
                "kind": "paragraph_topic_link_profile",
                "sentence_count": len(rows),
                "linked_later_sentences": sum(link > 0.05 for link in links),
                **_span(ctx.text, rows[0][0], rows[-1][1]),
            }
        )
    return {"value": sum(scores) / len(scores), "applicable": True, "certificates": certs[:20]}


def _frontloaded(ctx: ProjectionContext) -> dict[str, Any]:
    n = len(ctx.text)
    if n == 0:
        return {"value": None, "applicable": False, "certificates": []}
    opening = ctx.text[:max(200, int(n * 0.18))]
    coda_start = min(n, max(0, int(n * 0.82)))
    coda = ctx.text[coda_start:]
    open_hits = list(_OPEN_CUES.finditer(opening))
    coda_hits = list(_RELIEF_CUES.finditer(coda))
    certs = [
        {"kind": "opening_disposition_cue", **_span(ctx.text, match.start(), match.end())}
        for match in open_hits[:8]
    ] + [
        {"kind": "coda_relief_cue", **_span(ctx.text, coda_start + match.start(), coda_start + match.end())}
        for match in coda_hits[:8]
    ]
    return {"value": _clip01(0.65 * bool(open_hits) + 0.35 * bool(coda_hits)), "applicable": True, "certificates": certs}


def _counterposition(ctx: ProjectionContext) -> dict[str, Any]:
    doc = ctx.doc
    if doc is None:
        return {"value": None, "applicable": False, "certificates": []}
    party_positions: dict[str, list[Any]] = {party: [] for party in _PARTY_WORDS}
    for token in doc:
        if token.lemma_.casefold() not in _POSITION_LEMMAS:
            continue
        for child in token.children:
            party = child.lemma_.casefold()
            if child.dep_ in {"nsubj", "nsubjpass"} and party in party_positions:
                party_positions[party].append(token)
                for conjunct in child.conjuncts:
                    conj_party = conjunct.lemma_.casefold()
                    if conj_party in party_positions:
                        party_positions[conj_party].append(token)
    opposing_pairs = (("plaintiff", "defendant"), ("petitioner", "respondent"), ("appellant", "appellee"))
    paired = [pair for pair in opposing_pairs if party_positions[pair[0]] and party_positions[pair[1]]]
    contrasts = list(_CONTRAST.finditer(ctx.text))
    certs = []
    for party, verbs in party_positions.items():
        certs.extend(
            {"kind": "party_position", "party": party, "predicate": verb.lemma_, **_span(ctx.text, verb.idx, verb.idx + len(verb.text))}
            for verb in verbs[:4]
        )
    certs.extend(
        {"kind": "contrast_link", **_span(ctx.text, match.start(), match.end())}
        for match in contrasts[:8]
    )
    value = _clip01(0.7 * bool(paired) + 0.3 * min(1.0, len(contrasts) / 2.0))
    return {"value": value, "applicable": True, "certificates": certs[:20]}


def _tone(ctx: ProjectionContext) -> dict[str, Any]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", ctx.text)
    if not words:
        return {"value": None, "applicable": False, "certificates": []}
    hits = []
    for term in sorted(_INTENSIFIERS | _AD_HOMINEM):
        hits.extend(re.finditer(rf"\b{re.escape(term)}\b", ctx.text, re.I))
    density = len(hits) * 100.0 / len(words)
    certs = [{"kind": "tone_surface", **_span(ctx.text, match.start(), match.end())} for match in hits[:20]]
    return {"value": _clip01(1.0 - density / 2.0), "applicable": True, "certificates": certs}


def _headings(ctx: ProjectionContext) -> dict[str, Any]:
    offsets, cursor = [], 0
    for line in ctx.text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and len(stripped) <= 100 and _HEADING_RE.fullmatch(stripped):
            start = cursor + line.find(stripped)
            offsets.append((start, start + len(stripped), stripped))
        cursor += len(line)
    roadmaps = list(_ROADMAP_CUES.finditer(ctx.text))
    certs = [
        {"kind": "heading", **_span(ctx.text, start, end)}
        for start, end, _ in offsets[:15]
    ] + [
        {"kind": "roadmap_cue", **_span(ctx.text, match.start(), match.end())}
        for match in roadmaps[:15]
    ]
    value = _clip01(0.6 * min(1.0, len(offsets) / 3.0) + 0.4 * min(1.0, len(roadmaps) / 4.0))
    return {"value": value, "applicable": True, "certificates": certs}


def _question_frame(ctx: ProjectionContext) -> dict[str, Any]:
    doc = ctx.doc
    if doc is None:
        return {"value": None, "applicable": False, "certificates": []}
    cutoff = max(300, int(len(ctx.text) * 0.25))
    opening = ctx.text[:cutoff]
    cues = list(_QUESTION_CUE.finditer(opening))
    interrogatives = [sent for sent in doc.sents if sent.start_char < cutoff and sent.text.rstrip().endswith("?")]
    yes_no = [
        sent for sent in interrogatives
        if next((token for token in sent if not token.is_space), None) is not None
        and next(token for token in sent if not token.is_space).lemma_.casefold()
        in {"be", "do", "have", "can", "could", "may", "must", "should", "will", "would"}
    ]
    certs = [
        {"kind": "question_cue", **_span(ctx.text, match.start(), match.end())}
        for match in cues[:10]
    ] + [
        {"kind": "interrogative", "yes_no_form": sent in yes_no, **_span(ctx.text, sent.start_char, sent.end_char)}
        for sent in interrogatives[:10]
    ]
    value = _clip01(0.45 * bool(cues) + 0.35 * bool(interrogatives) + 0.20 * bool(yes_no))
    return {"value": value, "applicable": True, "certificates": certs}


def _inclusive(ctx: ProjectionContext) -> dict[str, Any]:
    matches = [match for pattern in _GENERIC_GENDER for match in pattern.finditer(ctx.text)]
    word_count = max(1, len(re.findall(r"[A-Za-z]+", ctx.text)))
    value = _clip01(1.0 - 100.0 * len(matches) / word_count)
    certs = [{"kind": "generic_gender_surface", **_span(ctx.text, match.start(), match.end())} for match in matches[:20]]
    return {"value": value, "applicable": True, "certificates": certs}


def _deadline_remedy(ctx: ProjectionContext) -> dict[str, Any]:
    remedies = list(_REMEDY.finditer(ctx.text))
    deadlines = list(_DEADLINE.finditer(ctx.text))
    consequences = list(_CONSEQUENCE.finditer(ctx.text))
    value = (bool(remedies) + bool(deadlines) + bool(consequences)) / 3.0
    certs = [
        {"kind": kind, **_span(ctx.text, match.start(), match.end())}
        for kind, matches in (
            ("remedy", remedies), ("deadline", deadlines), ("consequence", consequences)
        )
        for match in matches[:8]
    ]
    return {"value": value, "applicable": True, "certificates": certs}


_ANALYZERS = {
    "plain_language_surface": _plain_language,
    "sentence_clarity_parse": _sentence_clarity,
    "active_voice_parse": _active_voice,
    "negation_stack_parse": _negation_stack,
    "concrete_fact_anchors": _concrete_facts,
    "temporal_order_graph": _temporal_order,
    "numeric_consistency_check": _numeric_consistency,
    "definition_use_graph": _definition_graph,
    "citation_format_structure": _citations,
    "quote_attribution_parse": _quote_attribution,
    "discourse_cohesion_graph": _discourse_cohesion,
    "paragraph_cohesion_graph": _paragraph_cohesion,
    "frontloaded_disposition_structure": _frontloaded,
    "counterposition_structure": _counterposition,
    "tone_restraint_surface": _tone,
    "heading_roadmap_structure": _headings,
    "question_frame_structure": _question_frame,
    "inclusive_language_surface": _inclusive,
    "deadline_remedy_consequence_structure": _deadline_remedy,
}


def analyze_legal_writing_ctext(
    text: str,
    *,
    relation_ids: Sequence[str] | None = None,
    nlp: Any | None = None,
) -> dict[str, Any]:
    """Measure selected relation-local channels on exactly ``text``."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("ctext must be a nonempty string")
    selected = list(relation_ids) if relation_ids is not None else list(RELATION_BY_ID)
    unknown = sorted(set(selected) - set(RELATION_BY_ID))
    if unknown:
        raise ValueError(f"unknown relation ids: {unknown}")
    parser = _load_nlp() if nlp is None else nlp
    doc = parser(text) if parser is not None else None
    ctx = ProjectionContext(text=text, doc=doc)
    values = {relation_id: _ANALYZERS[relation_id](ctx) for relation_id in selected}
    return {
        "schema": SCHEMA,
        "program_version": PROGRAM_VERSION,
        "input_sha256": __import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
        "input_chars": len(text),
        "parser_available": doc is not None,
        "relation_values": values,
        "whole_construct_score_emitted": False,
    }


def load_cpu_parser():
    """Public loader so a runner can reuse one CPU parser across a split."""

    return _load_nlp()
