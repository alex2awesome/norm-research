"""Pure-code, relation-local structure analysis for patent abstracts and claims.

This capability is an additive manual pipeline seed.  It operates only on the
presented ``ctext`` bytes and does not read outcomes, examiner actions, prior
art, prompt judgements, or corpus state.  Its positive outputs are finite
syntax/graph witnesses (for example, a parsed dependency from claim 3 to claim
1).  They are not legal conclusions about definiteness, unity, enablement,
eligibility, novelty, or patentability.

The highest decision-contributing operation is depth 2 in the metric-seam
vocabulary: parsed claims are connected across spans into a dependency graph.
Surface markers are retained as depth-0/1 observations inside that graph, but
are never promoted to whole-construct scores.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import re
from typing import Iterable, Mapping


SCHEMA = "metric-seam.patent-claim-structure.v13"
DISCOVERY_MODE = "manual_additive_pipeline_seed"
MAXIMUM_DEPTH = 2

RELATIONS = (
    {
        "relation_id": "application_section_presence",
        "implemented_relation": "named ABSTRACT and CLAIMS sections are present in ctext",
        "channel": "code",
        "depth": 1,
    },
    {
        "relation_id": "claim_number_contiguity",
        "implemented_relation": "parsed claim ordinals form a contiguous increasing sequence",
        "channel": "code",
        "depth": 1,
    },
    {
        "relation_id": "claim_dependency_well_formedness",
        "implemented_relation": (
            "explicit claim references in a bounded opening incorporation clause resolve "
            "to earlier-presented, lower-numbered claims and form an acyclic dependency graph"
        ),
        "channel": "code",
        "depth": 2,
    },
    {
        "relation_id": "claim_set_layering",
        "implemented_relation": (
            "the presented claim set contains independently parsed roots and valid explicit "
            "dependent-claim fallback edges from bounded opening incorporation clauses"
        ),
        "channel": "code",
        "depth": 2,
    },
    {
        "relation_id": "antecedent_reference_surface_coverage",
        "implemented_relation": (
            "a bounded noun-head heuristic reports whether definite references have a matching "
            "indefinite introduction in the same claim or an explicitly referenced ancestor; "
            "diagnostic only pending parser validation"
        ),
        "channel": "code",
        "depth": 2,
    },
    {
        "relation_id": "statutory_category_surface_coverage",
        "implemented_relation": (
            "an independent-claim preamble names a process, machine/system/apparatus, "
            "manufacture/article/medium, or composition category"
        ),
        "channel": "code",
        "depth": 1,
    },
    {
        "relation_id": "functional_limitation_incidence",
        "implemented_relation": (
            "a presented claim contains a bounded functional-language marker such as "
            "'means for' or 'configured to'"
        ),
        "channel": "code",
        "depth": 1,
    },
    {
        "relation_id": "numerical_limitation_incidence",
        "implemented_relation": (
            "a presented claim contains a complete numeric token, optional unit, or range"
        ),
        "channel": "code",
        "depth": 1,
    },
    {
        "relation_id": "abstract_word_count",
        "implemented_relation": "the named abstract section has a replayable word count",
        "channel": "code",
        "depth": 1,
    },
)

AGGREGATION_RULE = None
CAPABILITIES_USED = (
    "named-section parser",
    "claim ordinal parser",
    "bounded opening-incorporation parser and claim dependency graph",
    "bounded noun-head antecedent tracker",
    "numeric token and unit parser",
)
ABSTENTION_CONDITIONS = (
    "missing named CLAIMS section",
    "no parseable numbered claims",
    "relation-specific evidence absent (for example, no explicit dependency edge)",
    "open-ended dependency phrase whose referenced claim set is not explicit",
    "explicit dependency range wider than the bounded 100-claim expansion",
)


_SECTION_RE = re.compile(
    r"(?im)^[ \t]*(ABSTRACT|CLAIMS?|DESCRIPTION|BACKGROUND|SUMMARY|"
    r"DETAILED DESCRIPTION|DRAWINGS?|BRIEF DESCRIPTION OF (?:THE )?DRAWINGS)\s*:?\s*$"
)
_CLAIM_START_RE = re.compile(r"(?m)^[ \t]*(\d{1,4})\s*[.)]\s+")
_CANCELED_RANGE_RE = re.compile(
    r"(?im)^[ \t]*(\d{1,4})\s*(?:-|–|—|to)\s*(\d{1,4})\s*[.)]\s*"
    r"\(?\s*cancell?ed\s*\)?"
)
_CLAIM_REF_TERM = r"\d{1,4}(?:\s*(?:-|–|—|to|through)\s*\d{1,4})?"
_CLAIM_REF_LIST = (
    rf"{_CLAIM_REF_TERM}(?:\s*(?:,|and|or)\s*(?:claims?\s+)?"
    rf"{_CLAIM_REF_TERM})*"
)
_DEPENDENCY_LEAD = (
    r"of|in|from|under|according\s+to|accordingly\s+to|in\s+accordance\s+with|"
    r"recited\s+in|set\s+forth\s+in|"
    r"as\s+(?:recited|set\s+forth|claimed)\s+in"
)
_EXPLICIT_CLAIM_REF_RE = re.compile(
    rf"\b(?:{_DEPENDENCY_LEAD})\s+(?:the\s+)?"
    r"(?:(?:any|one)\s+(?:one\s+)?of\s+(?:the\s+)?)?"
    rf"claims?\s+({_CLAIM_REF_LIST})",
    re.I,
)
_DEPENDENCY_MARKER_RE = re.compile(
    rf"\b(?:{_DEPENDENCY_LEAD})\s+(?:the\s+)?"
    r"(?:(?:any|one)\s+(?:one\s+)?of\s+(?:the\s+)?)?claims?\b",
    re.I,
)
_TRUNCATED_DEPENDENCY_MARKER_RE = re.compile(
    rf"\b(?:{_DEPENDENCY_LEAD})\s+(?:the\s+)?clai\s*$",
    re.I,
)
_OPEN_DEPENDENCY_RE = re.compile(
    r"\b(?:"
    r"(?:any|one)\s+(?:one\s+)?of\s+(?:the\s+)?(?:preceding|previous)\s+claims?"
    r"|(?:any|the)\s+(?:preceding|previous)\s+claims?"
    r")\b",
    re.I,
)
_FUNCTIONAL_RE = re.compile(
    r"\b(?:means\s+for|configured\s+to|adapted\s+to|operative\s+to|"
    r"programmed\s+to|instructions\s+(?:that|to)|module\s+configured\s+to)\b",
    re.I,
)
_NUMBER_ATOM = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_UNIT_PATTERN = (
    r"%|percent|ppm|ppb|nm|(?:micro|milli|centi|kilo)?meters?|mm|cm|km|"
    r"(?:micro|milli|kilo)?grams?|mg|kg|hz|khz|mhz|ghz|seconds?|minutes?|"
    r"hours?|days?|°\s*[cf]|degrees?"
)
_NUMBER_RE = re.compile(
    rf"(?<![\w.]){_NUMBER_ATOM}(?:\s*(?:{_UNIT_PATTERN}))?(?!\w)",
    re.I,
)
_RANGE_RE = re.compile(
    r"(?<![\w.])(?:from\s+|between\s+)?"
    rf"(?P<left>{_NUMBER_ATOM})\s*(?P<connector>[-–—]|to|and)\s*"
    rf"(?P<right>{_NUMBER_ATOM})"
    rf"(?:\s*(?P<unit>{_UNIT_PATTERN}))?(?!\w)",
    re.I,
)
_INTRO_RE = re.compile(
    r"(?=\b(?:a|an)\s+(?P<phrase>(?:[a-z][a-z0-9_-]*\s+){0,3}[a-z][a-z0-9_-]*))",
    re.I,
)
_DEFINITE_RE = re.compile(
    r"(?=\b(?:the|said|such)\s+(?P<phrase>(?:[a-z][a-z0-9_-]*\s+){0,3}[a-z][a-z0-9_-]*))",
    re.I,
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

_HEAD_STOPWORDS = {
    "a", "an", "and", "any", "each", "first", "one", "plurality", "said", "second",
    "such", "the", "third",
}

_PHRASE_BOUNDARIES = {
    "about", "according", "adapted", "and", "are", "arranged", "as", "at", "based",
    "being", "between", "by", "comprise", "comprises", "comprising", "configured",
    "consist", "consisting", "coupled", "for", "from", "further", "has", "having", "in",
    "include", "includes", "including", "into", "is", "not", "of", "on", "operative",
    "or", "out", "programmed", "selected", "so", "substantially", "that", "thereof", "to",
    "wherein", "which", "with",
}

_CATEGORY_PATTERNS = (
    ("process", re.compile(r"\b(?:method|process)\b", re.I)),
    ("machine_or_apparatus", re.compile(r"\b(?:machine|system|apparatus|device)\b", re.I)),
    (
        "manufacture_or_article",
        re.compile(r"\b(?:article|manufacture|non[- ]transitory\s+(?:computer[- ]readable\s+)?medium)\b", re.I),
    ),
    ("composition", re.compile(r"\bcomposition\b", re.I)),
)


@dataclass(frozen=True)
class DependencyIssue:
    surface: str
    reason: str


@dataclass(frozen=True)
class Claim:
    number: int
    text: str
    explicit_dependencies: tuple[int, ...]
    dependency_issues: tuple[DependencyIssue, ...]
    open_dependency: bool
    category: str | None
    category_surface: str | None
    category_span: tuple[int, int] | None
    functional_markers: tuple[str, ...]
    numeric_tokens: tuple[str, ...]
    numeric_ranges: tuple[dict, ...]
    introduced_heads: tuple[str, ...]
    definite_heads: tuple[str, ...]
    canceled: bool = False

    @property
    def is_independent(self) -> bool:
        return (
            not self.canceled
            and not self.explicit_dependencies
            and not self.dependency_issues
            and not self.open_dependency
        )


def split_named_sections(ctext: str) -> dict[str, str]:
    """Return normalized named sections without inventing missing boundaries."""

    if not isinstance(ctext, str):
        raise TypeError("ctext must be a string")
    matches = list(_SECTION_RE.finditer(ctext))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).upper()
        if name == "CLAIM":
            name = "CLAIMS"
        end = matches[index + 1].start() if index + 1 < len(matches) else len(ctext)
        body = ctext[match.end():end].strip()
        # If duplicate headings occur, preserve all presented bytes in order.
        sections[name] = "\n".join(value for value in (sections.get(name), body) if value)
    return sections


def _expand_reference_expression(
    expression: str,
) -> tuple[tuple[int, ...], tuple[DependencyIssue, ...]]:
    values: set[int] = set()
    issues = []
    for start_text, end_text in re.findall(
        r"(\d{1,4})(?:\s*(?:-|–|—|to|through)\s*(\d{1,4}))?",
        expression,
        flags=re.I,
    ):
        start = int(start_text)
        if not end_text:
            values.add(start)
            continue
        end = int(end_text)
        surface_match = re.search(
            rf"\b{re.escape(start_text)}\s*(?:-|–|—|to|through)\s*"
            rf"{re.escape(end_text)}\b",
            expression,
            flags=re.I,
        )
        surface = surface_match.group(0) if surface_match else f"{start_text}-{end_text}"
        if end < start:
            issues.append(
                DependencyIssue(surface=surface, reason="descending_dependency_range")
            )
            continue
        if end - start > 100:
            issues.append(
                DependencyIssue(surface=surface, reason="dependency_range_exceeds_bound")
            )
            continue
        values.update(range(start, end + 1))
    return tuple(sorted(values)), tuple(issues)


def _claim_dependencies(
    text: str,
) -> tuple[tuple[int, ...], bool, tuple[DependencyIssue, ...]]:
    incorporation_clause = re.split(
        r"[;:]|\b(?:comprising|consisting|including|having|wherein)\b",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    dependencies: set[int] = set()
    issues = []
    for match in _EXPLICIT_CLAIM_REF_RE.finditer(incorporation_clause):
        expanded, expression_issues = _expand_reference_expression(match.group(1))
        dependencies.update(expanded)
        issues.extend(expression_issues)
    for marker in _DEPENDENCY_MARKER_RE.finditer(incorporation_clause):
        tail = incorporation_clause[marker.end():]
        if not re.match(r"\s+\d{1,4}\b", tail):
            issues.append(
                DependencyIssue(
                    surface=marker.group(0), reason="missing_dependency_ordinal"
                )
            )
    truncated_marker = _TRUNCATED_DEPENDENCY_MARKER_RE.search(incorporation_clause)
    if truncated_marker:
        issues.append(
            DependencyIssue(
                surface=truncated_marker.group(0),
                reason="truncated_dependency_marker",
            )
        )
    return (
        tuple(sorted(dependencies)),
        bool(_OPEN_DEPENDENCY_RE.search(incorporation_clause)),
        tuple(issues),
    )


def _category(
    text: str, *, independent: bool
) -> tuple[str | None, str | None, tuple[int, int] | None]:
    if not independent:
        return None, None, None
    preamble = re.split(
        r"\b(?:comprising|consisting|including|having|configured)\b",
        text[:240],
        maxsplit=1,
        flags=re.I,
    )[0]
    claimed_object_phrase = re.split(
        r"[,;:]|\b(?:for|of|in|by|with|using|at|on|via|wherein|that|which)\b",
        preamble,
        maxsplit=1,
        flags=re.I,
    )[0]
    matches = []
    for priority, (name, pattern) in enumerate(_CATEGORY_PATTERNS):
        for match in pattern.finditer(claimed_object_phrase):
            matches.append(
                (match.start(), -priority, match.end(), name, match.group(0))
            )
    # Compound claimed-object phrases can contain several category words (for
    # example, "semiconductor device manufacturing method").  The rightmost
    # category-bearing noun before a use-context preposition is the head.  Text
    # after "for"/"of"/"by" is not allowed to override that head.
    if not matches:
        return None, None, None
    start, _priority, end, name, surface = max(
        matches, key=lambda row: (row[0], row[1])
    )
    return name, surface, (start, end)


def _functional_markers(text: str) -> tuple[str, ...]:
    return tuple(sorted({" ".join(match.group(0).casefold().split()) for match in _FUNCTIONAL_RE.finditer(text)}))


def _numeric_relations(text: str) -> tuple[tuple[str, ...], tuple[dict, ...]]:
    range_matches = []
    for match in _RANGE_RE.finditer(text):
        prefix = text[max(0, match.start() - 36):match.start()].casefold()
        suffix = text[match.end():min(len(text), match.end() + 16)].casefold()
        surface = match.group(0).casefold().lstrip()
        led_by_between = surface.startswith("between ")
        led_by_range_word = led_by_between or surface.startswith("from ")
        unit = (match.group("unit") or "").casefold()
        connector = match.group("connector").casefold()
        reference_context = bool(
            re.search(
                r"(?:claims?|fig(?:ure)?s?\.?|seq\s+id\s+no\.?|items?)\s*$",
                prefix,
            )
        )
        explicit_range_context = bool(
            led_by_range_word
            or re.search(r"(?:between|from|range(?:d|s)?(?:\s+of)?|approximately|about)\s*$", prefix)
            or re.match(rf"\s*(?:{_UNIT_PATTERN})\b", suffix, flags=re.I)
        )
        if reference_context:
            continue
        if connector == "and" and not (led_by_between or unit):
            continue
        if connector in {"-", "–", "—"} and not (unit or explicit_range_context):
            continue
        range_matches.append(match)
    ranges = tuple(
        {
            "surface": match.group(0),
            "left": match.group("left"),
            "right": match.group("right"),
            "connector": match.group("connector").casefold(),
            "unit": (match.group("unit") or "").casefold() or None,
        }
        for match in range_matches
    )
    tokens = []
    for match in _NUMBER_RE.finditer(text):
        surface = match.group(0).strip()
        prefix = text[max(0, match.start() - 28):match.start()].casefold()
        suffix = text[match.end():min(len(text), match.end() + 12)].casefold()
        in_range = any(
            range_match.start() <= match.start() and match.end() <= range_match.end()
            for range_match in range_matches
        )
        has_unit = bool(re.search(r"[%°a-z]", surface, flags=re.I))
        bounded_dimensionless = bool(
            re.search(
                r"(?:at\s+least|at\s+most|less\s+than|greater\s+than|more\s+than|"
                r"no\s+more\s+than|no\s+less\s+than|value\s+of|ratio\s+of)\s*$",
                prefix,
            )
            or re.match(r"\s*(?:times?|fold)\b", suffix)
            or "." in surface
        )
        claim_or_figure_reference = bool(
            re.search(r"(?:claims?|fig(?:ure)?s?\.?|no\.)\s*$", prefix)
        )
        identifier_context = bool(re.search(r"(?:version|codec|h\.)\s*$", prefix))
        if (
            (in_range or has_unit or bounded_dimensionless)
            and not claim_or_figure_reference
            and not identifier_context
        ):
            tokens.append(surface)
    tokens = tuple(tokens)
    return tokens, ranges


def _head(phrase: str) -> str | None:
    words = [word.casefold() for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", phrase)]
    candidates = []
    for word in words:
        if word in _PHRASE_BOUNDARIES:
            break
        if word not in _HEAD_STOPWORDS and len(word) > 1:
            candidates.append(word)
    return candidates[-1] if candidates else None


def _noun_heads(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
    heads = {_head(match.group("phrase")) for match in pattern.finditer(text)}
    return tuple(sorted(head for head in heads if head))


def parse_claims(ctext: str) -> tuple[dict[str, str], list[Claim]]:
    """Parse presented numbered claims; return no claims when the section is absent."""

    sections = split_named_sections(ctext)
    body = sections.get("CLAIMS", "")
    starts = list(_CLAIM_START_RE.finditer(body))
    canceled_ranges = list(_CANCELED_RANGE_RE.finditer(body))
    markers = sorted(
        [(match.start(), "claim", match) for match in starts]
        + [(match.start(), "canceled_range", match) for match in canceled_ranges],
        key=lambda row: row[0],
    )
    claims: list[Claim] = []
    for index, (_position, kind, match) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(body)
        if kind == "canceled_range":
            start_number, end_number = int(match.group(1)), int(match.group(2))
            if end_number < start_number or end_number - start_number > 100:
                continue
            for number in range(start_number, end_number + 1):
                claims.append(
                    Claim(
                        number=number,
                        text="(canceled)",
                        explicit_dependencies=(),
                        dependency_issues=(),
                        open_dependency=False,
                        category=None,
                        category_surface=None,
                        category_span=None,
                        functional_markers=(),
                        numeric_tokens=(),
                        numeric_ranges=(),
                        introduced_heads=(),
                        definite_heads=(),
                        canceled=True,
                    )
                )
            continue

        number = int(match.group(1))
        text = body[match.end():end].strip()
        canceled = bool(re.match(r"\(?\s*cancell?ed\s*\)?", text, flags=re.I))
        dependencies, open_dependency, dependency_issues = _claim_dependencies(text)
        independent = (
            not canceled
            and not dependencies
            and not dependency_issues
            and not open_dependency
        )
        category, category_surface, category_span = _category(
            text, independent=independent
        )
        numeric_tokens, numeric_ranges = _numeric_relations(text)
        claims.append(
            Claim(
                number=number,
                text=text,
                explicit_dependencies=dependencies,
                dependency_issues=dependency_issues,
                open_dependency=open_dependency,
                category=category,
                category_surface=category_surface,
                category_span=category_span,
                functional_markers=_functional_markers(text),
                numeric_tokens=numeric_tokens,
                numeric_ranges=numeric_ranges,
                introduced_heads=_noun_heads(_INTRO_RE, text),
                definite_heads=_noun_heads(_DEFINITE_RE, text),
                canceled=canceled,
            )
        )
    return sections, claims


def _has_cycle(nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> bool:
    graph: dict[int, list[int]] = defaultdict(list)
    indegree = Counter({node: 0 for node in nodes})
    for child, parent in edges:
        graph[parent].append(child)
        indegree[child] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in graph[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited != len(indegree)


def _ancestor_introductions(
    claim_number: int,
    by_number: Mapping[int, Claim],
    *,
    seen: frozenset[int] = frozenset(),
) -> set[str]:
    if claim_number in seen or claim_number not in by_number:
        return set()
    claim = by_number[claim_number]
    result = set(claim.introduced_heads)
    next_seen = seen | {claim_number}
    for parent in claim.explicit_dependencies:
        result.update(_ancestor_introductions(parent, by_number, seen=next_seen))
    return result


def _claim_payload(claim: Claim, *, introduced_universe: set[str]) -> dict:
    missing = sorted(set(claim.definite_heads) - introduced_universe)
    return {
        "number": claim.number,
        "canceled": claim.canceled,
        "text_length": len(claim.text),
        "explicit_dependencies": list(claim.explicit_dependencies),
        "dependency_parse_issues": [
            {"surface": issue.surface, "reason": issue.reason}
            for issue in claim.dependency_issues
        ],
        "open_dependency": claim.open_dependency,
        "independent": claim.is_independent,
        "statutory_category_surface": claim.category,
        "statutory_category_witness": (
            {
                "surface": claim.category_surface,
                "span": list(claim.category_span),
            }
            if claim.category_surface is not None and claim.category_span is not None
            else None
        ),
        "functional_markers": list(claim.functional_markers),
        "numeric_tokens": list(claim.numeric_tokens),
        "numeric_ranges": list(claim.numeric_ranges),
        "introduced_noun_heads": list(claim.introduced_heads),
        "definite_noun_heads": list(claim.definite_heads),
        "possible_missing_antecedent_heads": missing,
    }


def analyze_patent_ctext(ctext: str) -> dict:
    """Emit a non-aggregated, replayable relation record for one presented text."""

    sections, claims = parse_claims(ctext)
    numbers = [claim.number for claim in claims]
    by_number = {claim.number: claim for claim in claims}
    presented_positions: dict[int, list[int]] = defaultdict(list)
    for index, number in enumerate(numbers):
        presented_positions[number].append(index)
    duplicate_numbers = sorted(number for number, count in Counter(numbers).items() if count > 1)
    expected = list(range(min(numbers), max(numbers) + 1)) if numbers else []
    missing_numbers = sorted(set(expected) - set(numbers))

    edges: list[tuple[int, int]] = []
    valid_edges: list[tuple[int, int]] = []
    invalid_edges: list[dict] = []
    for claim in claims:
        for parent in claim.explicit_dependencies:
            edge = (claim.number, parent)
            edges.append(edge)
            reasons = []
            if parent not in by_number:
                reasons.append("referenced_claim_not_present")
            elif len(presented_positions[parent]) > 1:
                reasons.append("referenced_claim_number_is_duplicated")
            elif by_number[parent].canceled:
                reasons.append("referenced_claim_is_canceled_in_presented_text")
            child_positions = presented_positions[claim.number]
            parent_positions = presented_positions.get(parent, [])
            if (
                len(child_positions) != 1
                or len(parent_positions) != 1
                or parent_positions[0] >= child_positions[0]
            ):
                reasons.append("reference_is_not_to_an_earlier_claim")
            if parent >= claim.number:
                reasons.append("referenced_claim_number_is_not_lower")
            if reasons:
                invalid_edges.append({"child": claim.number, "parent": parent, "reasons": reasons})
            else:
                valid_edges.append(edge)
    cycle = _has_cycle(numbers, [edge for edge in edges if edge[1] in by_number]) if claims else False
    dependency_issues = [
        {
            "claim": claim.number,
            "surface": issue.surface,
            "reason": issue.reason,
        }
        for claim in claims
        for issue in claim.dependency_issues
    ]
    unresolved_dependency_reference = any(
        row["reason"]
        in {
            "dependency_range_exceeds_bound",
            "missing_dependency_ordinal",
            "truncated_dependency_marker",
        }
        for row in dependency_issues
    )
    descending_dependency_ranges = sum(
        row["reason"] == "descending_dependency_range"
        for row in dependency_issues
    )

    claim_rows = []
    definite_total = 0
    missing_antecedent_total = 0
    for claim in claims:
        universe = set(claim.introduced_heads)
        for parent in claim.explicit_dependencies:
            universe.update(_ancestor_introductions(parent, by_number))
        row = _claim_payload(claim, introduced_universe=universe)
        definite_total += len(row["definite_noun_heads"])
        missing_antecedent_total += len(row["possible_missing_antecedent_heads"])
        claim_rows.append(row)

    active_claims = [claim for claim in claims if not claim.canceled]
    independent = [claim for claim in active_claims if claim.is_independent]
    dependent = [claim for claim in active_claims if not claim.is_independent]
    validly_linked_dependents = sorted({child for child, _parent in valid_edges})
    open_dependency_claims = sorted(
        claim.number for claim in active_claims if claim.open_dependency
    )
    layering_value = (
        None
        if (
            not active_claims
            or unresolved_dependency_reference
            or (not valid_edges and open_dependency_claims)
        )
        else float(bool(independent and validly_linked_dependents))
    )
    explicit_edge_count = len(edges)
    dependency_denominator = explicit_edge_count + descending_dependency_ranges
    abstract = sections.get("ABSTRACT", "")
    abstract_words = _WORD_RE.findall(abstract)
    parsed_categories = [claim.category for claim in independent if claim.category]

    relation_values = {
        "application_section_presence": {
            "value": sum(name in sections for name in ("ABSTRACT", "CLAIMS")) / 2.0,
            "support": {"abstract_present": "ABSTRACT" in sections, "claims_present": "CLAIMS" in sections},
        },
        "claim_number_contiguity": {
            "value": (
                None
                if not claims
                else float(
                    not duplicate_numbers
                    and not missing_numbers
                    and numbers == expected
                )
            ),
            "support": {
                "presented_numbers": numbers,
                "duplicate_numbers": duplicate_numbers,
                "missing_numbers": missing_numbers,
                "presented_in_increasing_order": numbers == sorted(numbers),
            },
        },
        "claim_dependency_well_formedness": {
            "value": (
                None
                if dependency_denominator == 0 or unresolved_dependency_reference
                else len(valid_edges) / dependency_denominator
            ),
            "support": {
                "explicit_edges": explicit_edge_count,
                "valid_edges": len(valid_edges),
                "invalid_edges": len(invalid_edges),
                "dependency_parse_issues": dependency_issues,
                "cycle_detected": cycle,
                "open_dependency_claims": [claim.number for claim in claims if claim.open_dependency],
            },
        },
        "claim_set_layering": {
            "value": layering_value,
            "support": {
                "independent_claims": len(independent),
                "parsed_dependent_claims": len(dependent),
                "validly_linked_dependent_claims": validly_linked_dependents,
                "open_dependency_claims": open_dependency_claims,
                "invalid_dependency_edges": len(invalid_edges),
            },
        },
        "antecedent_reference_surface_coverage": {
            "value": None if definite_total == 0 else 1.0 - missing_antecedent_total / definite_total,
            "support": {
                "definite_noun_heads": definite_total,
                "possible_missing_antecedent_heads": missing_antecedent_total,
            },
        },
        "statutory_category_surface_coverage": {
            "value": None if not independent else len(parsed_categories) / len(independent),
            "support": {"independent_claims": len(independent), "categories": dict(Counter(parsed_categories))},
        },
        "functional_limitation_incidence": {
            "value": None if not active_claims else sum(bool(claim.functional_markers) for claim in active_claims) / len(active_claims),
            "support": {"claims_with_marker": [claim.number for claim in active_claims if claim.functional_markers]},
        },
        "numerical_limitation_incidence": {
            "value": None if not active_claims else sum(bool(claim.numeric_tokens) for claim in active_claims) / len(active_claims),
            "support": {"claims_with_number": [claim.number for claim in active_claims if claim.numeric_tokens]},
        },
        "abstract_word_count": {
            "value": len(abstract_words) if abstract else None,
            "support": {"within_150_words": bool(abstract) and len(abstract_words) <= 150},
        },
    }

    certificates = []
    for name in ("ABSTRACT", "CLAIMS"):
        if name in sections:
            certificates.append({"relation": "application_section_presence", "kind": "positive_witness", "section": name})
    certificates.extend(
        {
            "relation": "claim_dependency_well_formedness",
            "kind": "positive_witness",
            "child_claim": child,
            "parent_claim": parent,
        }
        for child, parent in valid_edges
    )
    certificates.extend(
        {
            "relation": "claim_dependency_well_formedness",
            "kind": "counter_witness",
            **row,
        }
        for row in invalid_edges
    )
    certificates.extend(
        {
            "relation": "claim_dependency_well_formedness",
            "kind": "counter_witness",
            **row,
        }
        for row in dependency_issues
        if row["reason"] == "descending_dependency_range"
    )
    for claim in active_claims:
        for marker in claim.functional_markers:
            certificates.append(
                {"relation": "functional_limitation_incidence", "kind": "positive_witness", "claim": claim.number, "surface": marker}
            )
        for surface in claim.numeric_tokens:
            certificates.append(
                {"relation": "numerical_limitation_incidence", "kind": "positive_witness", "claim": claim.number, "surface": surface}
            )
        if claim.category:
            certificates.append(
                {
                    "relation": "statutory_category_surface_coverage",
                    "kind": "positive_witness",
                    "claim": claim.number,
                    "category": claim.category,
                    "surface": claim.category_surface,
                    "span": list(claim.category_span or ()),
                }
            )

    abstentions = []
    if "CLAIMS" not in sections:
        abstentions.append({"relation": "all_claim_relations", "reason": "named_claims_section_absent"})
    elif not claims:
        abstentions.append({"relation": "all_claim_relations", "reason": "no_numbered_claims_parsed"})
    if not edges:
        abstentions.append({"relation": "claim_dependency_well_formedness", "reason": "no_explicit_claim_reference"})
    if any(
        row["reason"] == "dependency_range_exceeds_bound"
        for row in dependency_issues
    ):
        abstentions.append(
            {
                "relation": "claim_dependency_well_formedness",
                "reason": "dependency_range_exceeds_bounded_expansion",
            }
        )
    if any(
        row["reason"] == "missing_dependency_ordinal"
        for row in dependency_issues
    ):
        abstentions.append(
            {
                "relation": "claim_dependency_well_formedness",
                "reason": "dependency_marker_missing_ordinal",
            }
        )
    if any(
        row["reason"] == "truncated_dependency_marker"
        for row in dependency_issues
    ):
        abstentions.append(
            {
                "relation": "claim_dependency_well_formedness",
                "reason": "dependency_marker_truncated_at_presented_boundary",
            }
        )
    if layering_value is None and open_dependency_claims:
        abstentions.append(
            {
                "relation": "claim_set_layering",
                "reason": "open_dependency_not_explicitly_enumerable",
            }
        )
    if definite_total == 0:
        abstentions.append({"relation": "antecedent_reference_surface_coverage", "reason": "no_bounded_definite_noun_head"})
    if not abstract:
        abstentions.append({"relation": "abstract_word_count", "reason": "named_abstract_section_absent_or_empty"})

    return {
        "schema": SCHEMA,
        "discovery_mode": DISCOVERY_MODE,
        "channel": "pure_code",
        "maximum_decision_contributing_depth": MAXIMUM_DEPTH,
        "aggregation_rule": AGGREGATION_RULE,
        "scope": {
            "input": "presented ctext only",
            "external_supervision_used": False,
            "prior_art_or_examiner_evidence_used": False,
            "whole_patent_construct_established": False,
            "legal_validity_or_patentability_established": False,
            "antecedent_surface_is_diagnostic_only": True,
            "verified_absence_established": False,
        },
        "section_names": sorted(sections),
        "claims": claim_rows,
        "graph": {
            "nodes": numbers,
            "edges": [{"child": child, "parent": parent} for child, parent in edges],
            "invalid_edges": invalid_edges,
            "dependency_parse_issues": dependency_issues,
            "cycle_detected": cycle,
        },
        "relation_values": relation_values,
        "certificates": certificates,
        "abstentions": abstentions,
    }


def score(ctext: str, extracted: Mapping | None = None, ops=None) -> None:
    """Deliberately refuse a whole-criterion scalar.

    Consumers must select an independently audited relation from
    :func:`analyze_patent_ctext`; aggregating unrelated structural observations
    would manufacture a patent-quality score.
    """

    del ctext, extracted, ops
    return None
