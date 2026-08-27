"""Additive pure-code relation graphs for the frozen patent ``ctext``.

This module is intentionally separate from :mod:`patent_claim_structure`.
The earlier parser and every artifact made from it remain frozen.  The
capabilities here are retrospective/manual pipeline additions motivated by a
source-level audit.  They consume only the exact presented ``ctext`` bytes and
emit finite, relation-local witnesses.  They do not emit a patent-quality
score, infer facts outside the presented prefix, or establish legal
compliance, clarity, definiteness, patentability, codability, reconstruction,
or isomorphism.

Depth follows the metric-seam vocabulary.  Clause/list parsers whose decision
depends on a relation across spans are depth 2.  Term, numeric, and symbol
graphs that execute resolution across claim-dependency ancestors are depth 3.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Iterable, Mapping

from methods.metric_seam.patent_claim_structure import Claim, parse_claims


SCHEMA = "metric-seam.patent-claim-graph-additive.v2"
DISCOVERY_MODE = "manual_additive_pipeline_expansion_after_source_audit"
MAXIMUM_DEPTH = 3

RELATIONS = (
    {
        "relation_id": "claim_status_and_local_listing_witnesses",
        "implemented_relation": (
            "recognized status parentheticals attached to parsed claim ordinals, plus local "
            "duplicate-ordinal counter-witnesses within the presented claim prefix"
        ),
        "channel": "code",
        "depth": 2,
    },
    {
        "relation_id": "two_part_or_jepson_structure",
        "implemented_relation": (
            "an independently parsed claim contains an explicit Jepson improvement boundary "
            "or EPC-style 'characterised/characterized in that/by' boundary"
        ),
        "channel": "code",
        "depth": 2,
    },
    {
        "relation_id": "markush_closed_group_structure",
        "implemented_relation": (
            "a claim contains the exact closed-group opener 'selected from the group "
            "consisting of' followed by a finite presented alternative list; explicit "
            "mixture/combination qualifiers are separately witnessed"
        ),
        "channel": "code",
        "depth": 2,
    },
    {
        "relation_id": "bounded_antecedent_term_reference_graph",
        "implemented_relation": (
            "bounded article-led noun phrases are linked from 'the/said/such' references "
            "to earlier 'a/an/one-or-more/plurality' introductions in the same claim or an "
            "explicitly referenced ancestor claim"
        ),
        "channel": "code",
        "depth": 3,
    },
    {
        "relation_id": "numeric_constraint_definition_graph",
        "implemented_relation": (
            "presented numeric comparator/range nodes are linked to an adjacent bounded "
            "parameter phrase and, when explicit in ctext, a measurement/definition node"
        ),
        "channel": "code",
        "depth": 3,
    },
    {
        "relation_id": "formula_variable_definition_alignment",
        "implemented_relation": (
            "single-symbol numeric assignments in a claim are linked to explicit symbol "
            "definition clauses in that claim or an explicitly referenced ancestor, and "
            "incompatible numeric equalities incorporated along one dependency path are "
            "emitted as finite contradiction counter-witnesses"
        ),
        "channel": "code",
        "depth": 3,
    },
)

AGGREGATION_RULE = None
CAPABILITIES_USED = (
    "frozen named-section and claim-dependency parser v13 (read-only import)",
    "claim-status and two-part/Jepson clause parsers",
    "closed-group Markush alternative-list parser",
    "bounded article-led noun-phrase parser and ancestor term graph",
    "numeric constraint/measurement-definition evidence graph",
    "single-symbol assignment/definition evidence graph",
)
ABSTENTION_CONDITIONS = (
    "missing named CLAIMS section or no parseable claims",
    "no exact relation-specific syntax in the presented ctext",
    "open-ended or truncated list/claim span",
    "ambiguous term reference with multiple candidate introductions",
    "definition, status, specification, or claim material outside the 4,000-character prefix",
)


_STATUS_RE = re.compile(
    r"^\s*\(\s*(?P<status>currently\s+amended|previously\s+presented|"
    r"original|new|cancell?ed|withdrawn|allowed)\s*\)",
    re.I,
)
_JEPSON_RE = re.compile(
    r"\b(?:the\s+)?improvement\s+(?:compris(?:e|es|ing)|consist(?:s|ing)|"
    r"being|wherein)\b",
    re.I,
)
_TWO_PART_RE = re.compile(
    r"\bcharacteri[sz]ed\s+(?:in\s+that|by)\b",
    re.I,
)
_MARKUSH_RE = re.compile(
    r"\bselected\s+from\s+the\s+group\s+consisting\s+of\b",
    re.I,
)
_MIXTURE_RE = re.compile(
    r"\b(?:mixtures?|combinations?|blends?)\s+(?:of|thereof)\b|"
    r"\band\s+(?:mixtures?|combinations?|blends?)\s+thereof\b",
    re.I,
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_UNIT = (
    r"%|percent|ppm|ppb|nm|(?:micro|milli|centi|kilo)?meters?|mm(?:\s*[23])?|"
    r"cm(?:\s*[23])?|km|(?:micro|milli|kilo)?grams?|mg|kg|hz|khz|mhz|ghz|"
    r"seconds?|minutes?|hours?|days?|degrees?|°\s*[cf]"
)
_NUMERIC_CONSTRAINT_RE = re.compile(
    rf"(?P<surface>(?:(?P<operator>between|from|at\s+least|at\s+most|"
    rf"greater\s+than|less\s+than|more\s+than|no\s+more\s+than|"
    rf"no\s+less\s+than|equal\s+to|approximately|about)\s+)?"
    rf"(?P<left>{_NUMBER})(?:\s*(?P<connector>to|through|and|[-–—])\s*"
    rf"(?P<right>{_NUMBER}))?(?:\s*(?P<unit>{_UNIT}))?)",
    re.I,
)
_MEASUREMENT_TRIGGER_RE = re.compile(
    r"\b(?:is|are)\s+(?P<kind>defined|measured|determined|calculated|computed)"
    r"\s+(?:as|by|using|according\s+to|with)\b",
    re.I,
)
_SYMBOL_ASSIGNMENT_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<symbol>[A-Za-z])\s*(?:_\s*\d+|\d+)?\s*=\s*"
    rf"(?P<value>{_NUMBER})(?![A-Za-z0-9_])"
)
_SYMBOL_DEFINITION_RE = re.compile(
    r"\b(?P<symbols>[A-Za-z](?:\s*(?:,|and|or)\s*[A-Za-z])*)\s+"
    r"(?:is|are)\s+(?:independently\s+)?(?:an?\s+|the\s+)?"
    r"(?P<definition>integer|number|value|ratio|coefficient|constant|parameter|"
    r"variable|length|width|height|distance|angle|concentration|amount|time|step)\b",
    re.I,
)

_INTRODUCER_SEQUENCES = (
    (("a", "plurality", "of"), "introduction"),
    (("one", "or", "more"), "introduction"),
    (("at", "least", "one"), "introduction"),
    (("plurality", "of"), "introduction"),
    (("a",), "introduction"),
    (("an",), "introduction"),
    (("the",), "reference"),
    (("said",), "reference"),
    (("such",), "reference"),
)

# These boundaries deliberately trade recall for replayable phrase identity.
# A recognized node is exact under this grammar; unrecognized prose is not
# converted into an absence or defect.
_PHRASE_BOUNDARIES = {
    "a", "according", "adapted", "an", "and", "are", "as", "at", "based", "being",
    "between", "by", "characterised", "characterized", "comprise", "comprises",
    "comprising", "configured", "consist", "consisting", "contains", "coupled",
    "causes", "closes", "defined", "determined", "during", "emits", "equal", "for",
    "from", "further", "generates", "has", "having", "in", "include", "includes",
    "including", "implemented", "is", "measured", "more", "of", "on", "opens",
    "operative", "or", "performs", "programmed", "provides", "receives", "said",
    "selected", "so", "stores", "such", "that", "the", "therein", "thereof", "to",
    "transmits", "using", "wherein", "which", "with",
}
def _normal_text(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def _head(value: str) -> str:
    words = _normal_text(value).split()
    if not words:
        return ""
    head = words[-1]
    if len(head) > 4 and head.endswith("ies"):
        return head[:-3] + "y"
    if len(head) > 4 and head.endswith("s") and not head.endswith("ss"):
        return head[:-1]
    return head


def _phrase_key(words: Iterable[str]) -> str:
    values = [_normal_text(word) for word in words]
    if values:
        values[-1] = _head(values[-1])
    return " ".join(values)


def _term_mentions(claim: Claim, *, claim_instance: int | None = None) -> list[dict]:
    tokens = [
        {"surface": match.group(0), "word": match.group(0).casefold(), "span": match.span()}
        for match in _WORD_RE.finditer(claim.text)
    ]
    mentions: list[dict] = []
    i = 0
    while i < len(tokens):
        matched: tuple[tuple[str, ...], str] | None = None
        for sequence, kind in _INTRODUCER_SEQUENCES:
            if tuple(row["word"] for row in tokens[i : i + len(sequence)]) == sequence:
                matched = (sequence, kind)
                break
        if matched is None:
            i += 1
            continue
        sequence, kind = matched
        phrase_start = i + len(sequence)
        phrase_tokens = []
        previous_end = tokens[phrase_start - 1]["span"][1]
        for token in tokens[phrase_start : phrase_start + 6]:
            gap = claim.text[previous_end : token["span"][0]]
            if phrase_tokens and re.search(r"[,;:]", gap):
                break
            if token["word"] in _PHRASE_BOUNDARIES:
                break
            phrase_tokens.append(token)
            previous_end = token["span"][1]
        if not phrase_tokens:
            i += len(sequence)
            continue
        start = tokens[i]["span"][0]
        end = phrase_tokens[-1]["span"][1]
        key = _phrase_key(row["surface"] for row in phrase_tokens)
        if not key:
            i += len(sequence)
            continue
        mentions.append(
            {
                "mention_id": (
                    f"p{claim_instance}:c{claim.number}:m{len(mentions) + 1}"
                    if claim_instance is not None
                    else f"c{claim.number}:m{len(mentions) + 1}"
                ),
                "claim": claim.number,
                "kind": kind,
                "article": " ".join(sequence),
                "surface": claim.text[start:end],
                "phrase": " ".join(row["surface"] for row in phrase_tokens),
                "key": key,
                "head": _head(key),
                "span": [start, end],
            }
        )
        i = phrase_start + len(phrase_tokens)
    return mentions


def _ancestor_numbers(number: int, by_number: Mapping[int, Claim]) -> set[int]:
    result: set[int] = set()
    claim = by_number.get(number)
    frontier = list(claim.explicit_dependencies) if claim is not None else []
    while frontier:
        parent = frontier.pop()
        if parent in result or parent not in by_number:
            continue
        result.add(parent)
        frontier.extend(by_number[parent].explicit_dependencies)
    return result


def _resolve_term_graph(claims: list[Claim]) -> dict:
    # Duplicate ordinals make ancestry ambiguous; retain their mention nodes but
    # do not use either duplicate as an ancestor candidate.
    counts: dict[int, int] = defaultdict(int)
    for claim in claims:
        counts[claim.number] += 1
    by_number = {claim.number: claim for claim in claims if counts[claim.number] == 1}
    mentions_by_claim: dict[int, list[dict]] = defaultdict(list)
    for position, claim in enumerate(claims, start=1):
        mentions_by_claim[claim.number].extend(
            _term_mentions(claim, claim_instance=position)
        )
    introductions = [
        row
        for rows in mentions_by_claim.values()
        for row in rows
        if row["kind"] == "introduction"
    ]
    references = [
        row
        for rows in mentions_by_claim.values()
        for row in rows
        if row["kind"] == "reference"
    ]
    edges = []
    for reference in references:
        allowed_claims = _ancestor_numbers(reference["claim"], by_number) | {
            reference["claim"]
        }
        candidates = [
            row
            for row in introductions
            if row["claim"] in allowed_claims
            and (
                row["claim"] != reference["claim"]
                or row["span"][0] < reference["span"][0]
            )
        ]
        exact = [row for row in candidates if row["key"] == reference["key"]]
        head = [row for row in candidates if row["head"] == reference["head"]]
        if len(exact) == 1:
            status, matched = "resolved_exact", exact
        elif len(exact) > 1:
            status, matched = "ambiguous", exact
        elif len(head) > 1:
            status, matched = "ambiguous", head
        elif head:
            status, matched = "head_only_near_match", head
        else:
            status, matched = "unresolved", []
        edges.append(
            {
                "reference_id": reference["mention_id"],
                "claim": reference["claim"],
                "reference_surface": reference["surface"],
                "reference_key": reference["key"],
                "status": status,
                "candidate_introduction_ids": [row["mention_id"] for row in matched],
            }
        )
    return {
        "nodes": introductions + references,
        "edges": edges,
        "counts": {
            "introductions": len(introductions),
            "references": len(references),
            "resolved": sum(row["status"].startswith("resolved") for row in edges),
            "ambiguous": sum(row["status"] == "ambiguous" for row in edges),
            "head_only_near_match": sum(
                row["status"] == "head_only_near_match" for row in edges
            ),
            "unresolved": sum(row["status"] == "unresolved" for row in edges),
        },
    }


def _status_nodes(claims: list[Claim]) -> tuple[list[dict], list[dict]]:
    nodes = []
    counter_witnesses = []
    by_number: dict[int, list[int]] = defaultdict(list)
    for position, claim in enumerate(claims):
        by_number[claim.number].append(position)
        match = _STATUS_RE.match(claim.text)
        if match:
            nodes.append(
                {
                    "claim": claim.number,
                    "status": _normal_text(match.group("status")).replace("canceled", "cancelled"),
                    "surface": match.group(0),
                    "span": list(match.span()),
                }
            )
    for number, positions in sorted(by_number.items()):
        if len(positions) > 1:
            counter_witnesses.append(
                {"claim": number, "reason": "duplicate_presented_ordinal", "positions": positions}
            )
    return nodes, counter_witnesses


def _two_part_nodes(claims: list[Claim]) -> list[dict]:
    nodes = []
    for claim in claims:
        if not claim.is_independent:
            continue
        candidates = [("jepson_improvement", match) for match in _JEPSON_RE.finditer(claim.text)]
        candidates.extend(
            ("epc_characterising_boundary", match)
            for match in _TWO_PART_RE.finditer(claim.text)
        )
        for kind, match in sorted(candidates, key=lambda row: row[1].start()):
            nodes.append(
                {
                    "claim": claim.number,
                    "boundary_kind": kind,
                    "surface": match.group(0),
                    "span": list(match.span()),
                    "preamble_chars": match.start(),
                    "characterising_chars": len(claim.text) - match.end(),
                }
            )
    return nodes


def _markush_nodes(claims: list[Claim]) -> list[dict]:
    nodes = []
    for claim in claims:
        for match in _MARKUSH_RE.finditer(claim.text):
            tail = re.split(r"[;.]", claim.text[match.end() :], maxsplit=1)[0][:800]
            # A closed-group opener without a visibly enumerable tail is a
            # truncated applicability marker, not a finite Markush witness.
            separators = list(re.finditer(r",|\b(?:and|or)\b", tail, flags=re.I))
            if len(separators) < 1:
                continue
            alternatives = [
                part.strip(" :(),")
                for part in re.split(r",|\b(?:and|or)\b", tail, flags=re.I)
                if part.strip(" :(),")
            ]
            if len(alternatives) < 2:
                continue
            mixture = _MIXTURE_RE.search(tail)
            nodes.append(
                {
                    "claim": claim.number,
                    "opener": match.group(0),
                    "span": [match.start(), match.end() + len(tail)],
                    "presented_alternative_count_lower_bound": len(alternatives),
                    "explicit_mixture_or_combination_qualifier": (
                        mixture.group(0) if mixture else None
                    ),
                }
            )
    return nodes


def _nearest_parameter_key(claim: Claim, start: int, mentions: list[dict]) -> str | None:
    prior = [row for row in mentions if row["span"][1] <= start and start - row["span"][1] <= 140]
    if prior:
        return prior[-1]["key"]
    prefix = claim.text[max(0, start - 90) : start]
    words = [match.group(0) for match in _WORD_RE.finditer(prefix)]
    bounded = []
    for word in reversed(words):
        if word.casefold() in _PHRASE_BOUNDARIES:
            if bounded:
                break
            continue
        bounded.append(word)
        if len(bounded) == 3:
            break
    return _phrase_key(reversed(bounded)) or None


def _measurement_nodes(claims: list[Claim]) -> list[dict]:
    nodes = []
    for claim in claims:
        for match in _MEASUREMENT_TRIGGER_RE.finditer(claim.text):
            prefix = claim.text[max(0, match.start() - 100) : match.start()]
            words = [row.group(0) for row in _WORD_RE.finditer(prefix)]
            phrase = []
            for word in reversed(words):
                if word.casefold() in _PHRASE_BOUNDARIES:
                    if phrase:
                        break
                    continue
                phrase.append(word)
                if len(phrase) == 4:
                    break
            key = _phrase_key(reversed(phrase))
            if not key:
                continue
            nodes.append(
                {
                    "definition_id": f"c{claim.number}:d{len(nodes) + 1}",
                    "claim": claim.number,
                    "parameter_key": key,
                    "parameter_head": _head(key),
                    "kind": match.group("kind").casefold(),
                    "surface": match.group(0),
                    "span": list(match.span()),
                }
            )
    return nodes


def _numeric_graph(claims: list[Claim]) -> dict:
    definitions = _measurement_nodes(claims)
    constraints = []
    links = []
    for claim in claims:
        mentions = _term_mentions(claim)
        for match in _NUMERIC_CONSTRAINT_RE.finditer(claim.text):
            operator = match.group("operator")
            connector = match.group("connector")
            unit = match.group("unit")
            surface = match.group("surface")
            prefix = claim.text[max(0, match.start() - 24) : match.start()].casefold()
            # Bare integers are overwhelmingly claim/figure/identifier numbers.
            if not (operator or connector or unit or "." in surface):
                continue
            if re.search(r"(?:claims?|fig(?:ure)?s?|no\.)\s*$", prefix):
                continue
            parameter_key = _nearest_parameter_key(claim, match.start(), mentions)
            node = {
                "constraint_id": f"c{claim.number}:n{len(constraints) + 1}",
                "claim": claim.number,
                "surface": surface,
                "span": list(match.span("surface")),
                "parameter_key": parameter_key,
                "parameter_head": _head(parameter_key or ""),
                "operator": _normal_text(operator or connector or "unit_bearing_value"),
                "left": match.group("left"),
                "right": match.group("right"),
                "unit": _normal_text(unit) if unit else None,
            }
            constraints.append(node)
            exact = [row for row in definitions if row["parameter_key"] == parameter_key]
            head = [
                row
                for row in definitions
                if node["parameter_head"] and row["parameter_head"] == node["parameter_head"]
            ]
            matched = exact if len(exact) == 1 else head if len(head) == 1 else []
            if matched:
                links.append(
                    {
                        "constraint_id": node["constraint_id"],
                        "definition_id": matched[0]["definition_id"],
                        "match": "exact_parameter" if exact else "unique_parameter_head",
                    }
                )
    return {"constraint_nodes": constraints, "definition_nodes": definitions, "links": links}


def _symbol_definitions(claims: list[Claim]) -> list[dict]:
    definitions = []
    for claim in claims:
        for match in _SYMBOL_DEFINITION_RE.finditer(claim.text):
            symbols = re.findall(r"[A-Za-z]", match.group("symbols"))
            for symbol in symbols:
                definitions.append(
                    {
                        "definition_id": f"c{claim.number}:s{len(definitions) + 1}",
                        "claim": claim.number,
                        "symbol": symbol.casefold(),
                        "definition_kind": match.group("definition").casefold(),
                        "surface": match.group(0),
                        "span": list(match.span()),
                    }
                )
    return definitions


def _formula_graph(claims: list[Claim]) -> dict:
    counts: dict[int, int] = defaultdict(int)
    for claim in claims:
        counts[claim.number] += 1
    by_number = {claim.number: claim for claim in claims if counts[claim.number] == 1}
    definitions = _symbol_definitions(claims)
    assignments = []
    links = []
    for claim in claims:
        allowed = _ancestor_numbers(claim.number, by_number) | {claim.number}
        for match in _SYMBOL_ASSIGNMENT_RE.finditer(claim.text):
            node = {
                "assignment_id": f"c{claim.number}:a{len(assignments) + 1}",
                "claim": claim.number,
                "symbol": match.group("symbol").casefold(),
                "value": match.group("value"),
                "surface": match.group(0),
                "span": list(match.span()),
            }
            assignments.append(node)
            candidates = [
                row
                for row in definitions
                if row["symbol"] == node["symbol"] and row["claim"] in allowed
            ]
            if len(candidates) == 1:
                links.append(
                    {
                        "assignment_id": node["assignment_id"],
                        "definition_id": candidates[0]["definition_id"],
                    }
                )
    conflicts = []
    assignments_by_claim: dict[int, list[dict]] = defaultdict(list)
    for row in assignments:
        assignments_by_claim[row["claim"]].append(row)
    for claim in claims:
        allowed = _ancestor_numbers(claim.number, by_number) | {claim.number}
        by_symbol: dict[str, list[dict]] = defaultdict(list)
        for number in allowed:
            for row in assignments_by_claim[number]:
                by_symbol[row["symbol"]].append(row)
        for symbol, rows in sorted(by_symbol.items()):
            values = sorted({row["value"] for row in rows})
            if len(values) > 1:
                conflicts.append(
                    {
                        "claim": claim.number,
                        "symbol": symbol,
                        "incompatible_values": values,
                        "assignment_ids": [row["assignment_id"] for row in rows],
                    }
                )
    return {
        "assignment_nodes": assignments,
        "definition_nodes": definitions,
        "links": links,
        "conflicts": conflicts,
    }


def analyze_patent_claim_graph(ctext: str) -> dict:
    """Return finite additive witnesses for one exact presented text."""

    if not isinstance(ctext, str):
        raise TypeError("ctext must be a string")
    sections, claims = parse_claims(ctext)
    status_nodes, status_counter_witnesses = _status_nodes(claims)
    two_part_nodes = _two_part_nodes(claims)
    markush_nodes = _markush_nodes(claims)
    term_graph = _resolve_term_graph(claims)
    numeric_graph = _numeric_graph(claims)
    formula_graph = _formula_graph(claims)

    relation_values = {
        "claim_status_and_local_listing_witnesses": {
            "value": None,
            "support": {
                "recognized_status_markers": len(status_nodes),
                "duplicate_ordinal_counter_witnesses": len(status_counter_witnesses),
            },
        },
        "two_part_or_jepson_structure": {
            "value": None,
            "support": {"finite_structure_witnesses": len(two_part_nodes)},
        },
        "markush_closed_group_structure": {
            "value": None,
            "support": {
                "finite_closed_group_witnesses": len(markush_nodes),
                "explicit_mixture_or_combination_qualifiers": sum(
                    row["explicit_mixture_or_combination_qualifier"] is not None
                    for row in markush_nodes
                ),
            },
        },
        "bounded_antecedent_term_reference_graph": {
            "value": (
                None
                if not term_graph["counts"]["references"]
                else term_graph["counts"]["resolved"] / term_graph["counts"]["references"]
            ),
            "support": term_graph["counts"],
        },
        "numeric_constraint_definition_graph": {
            "value": None,
            "support": {
                "numeric_constraint_nodes": len(numeric_graph["constraint_nodes"]),
                "measurement_or_definition_nodes": len(numeric_graph["definition_nodes"]),
                "positive_definition_links": len(numeric_graph["links"]),
            },
        },
        "formula_variable_definition_alignment": {
            "value": None,
            "support": {
                "single_symbol_numeric_assignments": len(formula_graph["assignment_nodes"]),
                "symbol_definition_nodes": len(formula_graph["definition_nodes"]),
                "positive_definition_links": len(formula_graph["links"]),
                "incompatible_equality_counter_witnesses": len(formula_graph["conflicts"]),
            },
        },
    }

    certificates = []
    certificates.extend(
        {"relation": "claim_status_and_local_listing_witnesses", "kind": "positive_witness", **row}
        for row in status_nodes
    )
    certificates.extend(
        {"relation": "claim_status_and_local_listing_witnesses", "kind": "counter_witness", **row}
        for row in status_counter_witnesses
    )
    certificates.extend(
        {"relation": "two_part_or_jepson_structure", "kind": "positive_witness", **row}
        for row in two_part_nodes
    )
    certificates.extend(
        {"relation": "markush_closed_group_structure", "kind": "positive_witness", **row}
        for row in markush_nodes
    )
    certificates.extend(
        {
            "relation": "bounded_antecedent_term_reference_graph",
            "kind": "positive_witness" if row["status"].startswith("resolved") else "bounded_counter_witness",
            **row,
        }
        for row in term_graph["edges"]
    )
    certificates.extend(
        {"relation": "numeric_constraint_definition_graph", "kind": "positive_witness", **row}
        for row in numeric_graph["links"]
    )
    certificates.extend(
        {"relation": "formula_variable_definition_alignment", "kind": "positive_witness", **row}
        for row in formula_graph["links"]
    )
    certificates.extend(
        {
            "relation": "formula_variable_definition_alignment",
            "kind": "counter_witness",
            **row,
        }
        for row in formula_graph["conflicts"]
    )

    abstentions = []
    if "CLAIMS" not in sections:
        abstentions.append({"relation": "all", "reason": "named_claims_section_absent"})
    elif not claims:
        abstentions.append({"relation": "all", "reason": "no_numbered_claims_parsed"})
    relation_objects = {
        "claim_status_and_local_listing_witnesses": status_nodes + status_counter_witnesses,
        "two_part_or_jepson_structure": two_part_nodes,
        "markush_closed_group_structure": markush_nodes,
        "bounded_antecedent_term_reference_graph": term_graph["edges"],
        "numeric_constraint_definition_graph": numeric_graph["links"],
        "formula_variable_definition_alignment": formula_graph["links"]
        + formula_graph["conflicts"],
    }
    for relation, objects in relation_objects.items():
        if not objects:
            abstentions.append({"relation": relation, "reason": "no_finite_relation_witness"})

    return {
        "schema": SCHEMA,
        "discovery_mode": DISCOVERY_MODE,
        "channel": "pure_code",
        "maximum_decision_contributing_depth": MAXIMUM_DEPTH,
        "aggregation_rule": AGGREGATION_RULE,
        "scope": {
            "input": "exact presented ctext only",
            "external_supervision_used": False,
            "outcome_or_reference_values_used": False,
            "prompt_outputs_used": False,
            "prior_art_or_examiner_evidence_used": False,
            "whole_patent_construct_established": False,
            "legal_compliance_or_definiteness_established": False,
            "verified_absence_outside_recognized_local_grammar": False,
            "reconstruction_or_isomorphism_measured": False,
        },
        "presented_character_count": len(ctext),
        "claim_count": len(claims),
        "graphs": {
            "status_listing": {"nodes": status_nodes, "counter_witnesses": status_counter_witnesses},
            "two_part_jepson": {"nodes": two_part_nodes},
            "markush": {"nodes": markush_nodes},
            "term_reference": term_graph,
            "numeric_constraint_definition": numeric_graph,
            "formula_variable_definition": formula_graph,
        },
        "relation_values": relation_values,
        "certificates": certificates,
        "abstentions": abstentions,
    }


def score(ctext: str, extracted: Mapping | None = None, ops=None) -> None:
    """Refuse a whole-criterion scalar; consumers must use audited relations."""

    del ctext, extracted, ops
    return None
