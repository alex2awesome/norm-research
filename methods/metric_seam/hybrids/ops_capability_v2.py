"""CapabilityOps v2: conservative, certificate-oriented fixes over frozen e2l-v1.

The v1 module remains untouched because historical E2L results depend on its exact
behavior.  This additive wrapper fixes audited counterexamples and prefers ``None`` /
``checkable=False`` over confident answers when the available evidence is ambiguous.
"""

from __future__ import annotations

import datetime
import math
import re

import ops_capability as v1

VERSION = "reconstruction-v2.1"

CAPABILITIES = dict(v1.CAPABILITIES)
CAPABILITIES.update(
    {
        "attributions": {
            **CAPABILITIES["attributions"],
            "v2": (
                "ambiguous multi-organization ownership abstains; reporting verbs "
                "inherit shared subjects across conjunctions; adjacent named-action "
                "beats can anchor immediately following quoted speech"
            ),
        },
        "self_attributed": {
            **CAPABILITIES["self_attributed"],
            "v2": "repeated/out-of-parse-cap spans abstain",
        },
        "stat_consistency": {
            **CAPABILITIES["stat_consistency"],
            "v2": "p-value bounds do not imply decisions they cannot entail",
        },
        "number_consistency": {
            **CAPABILITIES["number_consistency"],
            "v2": "percentage direction is preserved",
        },
        "date_chain": {
            **CAPABILITIES["date_chain"],
            "v2": (
                "missing-year dates use frozen 2000 epoch; matched but invalid calendar "
                "dates are retained as explicit non-checkable rows"
            ),
        },
        "deadline_satisfied": {
            **CAPABILITIES["deadline_satisfied"],
            "v2": "requires 0 <= elapsed <= deadline",
        },
        "is_refrain": {
            **CAPABILITIES["is_refrain"],
            "v2": (
                "one- and two-word refrains are eligible; craft still requires at least "
                "one intervening sentence with progression or a varied callback"
            ),
        },
        "discourse_position": {
            **CAPABILITIES["discourse_position"],
            "v2": "invalid/ambiguous offsets abstain",
        },
    }
)

_CAP = v1._SPACY_CHAR_CAP
_EPOCH = datetime.datetime(2000, 1, 1)


def _issuer_orgs(doc):
    """Return a conservative home-org set, or None when document ownership is ambiguous.

    An organization is accepted only when it is the unique most-frequent ORG, or the
    earliest ORG and no other ORG ties it in frequency.  This deliberately abstains on
    joint releases and organization lists instead of declaring up to three unrelated
    organizations to be the document's own voice.
    """

    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        norm = v1._norm_org(ent.text)
        if len(norm) < 2 or (len(norm) <= 4 and not ent.text.isupper()):
            continue
        counts[norm] = counts.get(norm, 0) + 1
        first.setdefault(norm, ent.start_char)
    if not counts:
        return frozenset()
    ranked = sorted(counts, key=lambda x: (-counts[x], first[x]))
    if len(ranked) > 1 and counts[ranked[0]] == counts[ranked[1]]:
        return None
    return frozenset({ranked[0]})


def _shared_subject(tok):
    """Direct subject, or the subject licensed by a coordinated verb ancestor."""

    subject_deps = ("nsubj", "nsubjpass", "csubj")
    subj = next((c for c in tok.children if c.dep_ in subject_deps), None)
    if subj is not None:
        return subj
    cursor = tok
    # In ``she starved ..., let ..., and told ...``, spaCy attaches ``told`` to
    # ``let`` and ``let`` to ``starved``.  Coordinated predicates share the root's
    # subject; stopping as soon as the chain ceases to be ``conj`` avoids borrowing
    # a subject across an arbitrary embedding boundary.
    for _ in range(8):
        if cursor.dep_ != "conj" or cursor.head is cursor:
            break
        cursor = cursor.head
        subj = next((c for c in cursor.children if c.dep_ in subject_deps), None)
        if subj is not None:
            return subj
    return None


def _reporting_conj_additions(doc):
    """Attributions v1 misses solely because a reporting verb shares its subject."""

    out = []
    for tok in doc:
        if tok.pos_ != "VERB" or tok.lemma_.lower() not in v1._REPORTING_VERBS:
            continue
        direct_subj = next(
            (c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass", "csubj")),
            None,
        )
        if direct_subj is not None:
            continue  # v1 already handles this shape.
        subj = _shared_subject(tok)
        comp = next((c for c in tok.children if c.dep_ in ("ccomp", "xcomp")), None)
        if subj is None or comp is None:
            continue
        s0, s1 = v1._subtree_bounds(subj)
        c0, c1 = v1._subtree_bounds(comp)
        speaker_span = doc[s0 : s1 + 1].text
        sent = tok.sent
        out.append(
            {
                "verb": tok.lemma_.lower(),
                "verb_text": tok.text,
                "speaker_span": speaker_span,
                "span": doc[c0 : c1 + 1].text,
                "span_start": doc[c0].idx,
                "span_end": doc[c1].idx + len(doc[c1].text),
                "sentence": sent.text,
                "speaker_is_first_person_org": None,
                "quote_like": any(q in sent.text for q in ('"', "“", "‘")),
                "attribution_mode": "reporting_verb_shared_subject",
                "checkable": True,
            }
        )
    return out


_QUOTE_CLOSE = {'"': '"', "'": "'", "“": "”", "‘": "’"}


def _action_beat_additions(doc):
    """Associate a named action beat with an immediately following quoted sentence.

    This is deliberately narrower than a general fiction coreference resolver: the action
    sentence must have an explicit proper-noun subject, contain no quotation itself, and be
    immediately followed by a sentence whose first non-space character opens and whose last
    quote closes direct speech.  The row advertises its bounded syntactic mode so callers do
    not confuse this convention-level association with semantic speaker identification.
    """

    out = []
    for previous in doc.sents:
        previous_has_direct_quote = (
            any(q in previous.text for q in ('"', "“", "‘"))
            or re.search(r"(?:^|\s)'(?=\S)", previous.text) is not None
        )
        if previous_has_direct_quote:
            continue
        open_at = previous.end_char
        while open_at < len(doc.text) and doc.text[open_at].isspace():
            open_at += 1
        if open_at >= len(doc.text) or doc.text[open_at] not in _QUOTE_CLOSE:
            continue
        opener = doc.text[open_at]
        closer = _QUOTE_CLOSE[opener]
        # spaCy may split a multi-sentence quotation and may tokenize straight/curly
        # apostrophes inside contractions as the same character used to close speech.
        # Select the first quote followed by a boundary, not the apostrophe in ``don't``.
        close_at = None
        cursor = open_at + 1
        while cursor < len(doc.text):
            candidate = doc.text.find(closer, cursor)
            if candidate < 0:
                break
            following = doc.text[candidate + 1 : candidate + 2]
            if not following or following.isspace() or following in ",.;:!?)]}":
                close_at = candidate
                break
            cursor = candidate + 1
        if close_at is None or close_at <= open_at + 1:
            continue
        root = previous.root
        subj = next(
            (c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")), None
        )
        if subj is None:
            continue
        s0, s1 = v1._subtree_bounds(subj)
        subject_tokens = doc[s0 : s1 + 1]
        if not any(t.pos_ == "PROPN" for t in subject_tokens):
            continue
        speaker_span = subject_tokens.text
        raw_start = open_at + 1
        raw_end = close_at
        while raw_start < raw_end and doc.text[raw_start].isspace():
            raw_start += 1
        while raw_end > raw_start and doc.text[raw_end - 1].isspace():
            raw_end -= 1
        if raw_end <= raw_start:
            continue
        out.append(
            {
                "verb": "action_beat",
                "verb_text": root.text,
                "speaker_span": speaker_span,
                "span": doc.text[raw_start:raw_end],
                "span_start": raw_start,
                "span_end": raw_end,
                "sentence": f"{previous.text} {doc.text[open_at:close_at + 1]}",
                "speaker_is_first_person_org": None,
                "quote_like": True,
                "attribution_mode": "adjacent_named_action_beat",
                "attribution_status": "bounded_syntactic_association",
                "checkable": True,
            }
        )
    return out


def attributions(text):
    rows = v1.attributions(text)
    doc = v1._doc_for(text)
    if doc is None:
        return []
    existing = {
        (r.get("verb_text"), r.get("speaker_span"), r.get("span_start"), r.get("span_end"))
        for r in rows
    }
    for row in _reporting_conj_additions(doc) + _action_beat_additions(doc):
        key = (
            row.get("verb_text"),
            row.get("speaker_span"),
            row.get("span_start"),
            row.get("span_end"),
        )
        if key not in existing:
            rows.append(row)
            existing.add(key)
    issuers = _issuer_orgs(doc)
    out = []
    for row in rows:
        r = dict(row)
        if r.get("attribution_mode") == "adjacent_named_action_beat":
            # The action-beat extension establishes a local actor-to-quote association,
            # not press-release document ownership.  NER can mislabel fiction names or
            # titles as ORG, so the orthogonal own-organization question must abstain.
            r["speaker_is_first_person_org"] = None
            out.append(r)
            continue
        if issuers is None or not issuers:
            # First-person grammar remains direct evidence; absent or ambiguous document-
            # ownership evidence must not be converted into a confident third-party verdict.
            span = r.get("speaker_span", "")
            low_words = re.findall(r"[a-z']+", span.lower())[:2]
            r["speaker_is_first_person_org"] = (
                True if any(w in v1._FIRST_PERSON_WORDS for w in low_words) else None
            )
        else:
            r["speaker_is_first_person_org"] = v1._speaker_is_self(
                r.get("speaker_span", ""), issuers
            )
        out.append(r)
    return out


def self_attributed(text, span):
    if not isinstance(text, str) or not text:
        return None
    if isinstance(span, (tuple, list)) and len(span) == 2:
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            return None
        if start < 0 or end <= start or end > min(len(text), _CAP):
            return None
    elif isinstance(span, str):
        starts = [m.start() for m in re.finditer(re.escape(span), text)] if span else []
        if len(starts) != 1:
            return None
        start, end = starts[0], starts[0] + len(span)
        if end > _CAP:
            return None
    else:
        return None
    rows = attributions(text)
    for row in rows:
        if row["span_start"] < end and start < row["span_end"]:
            return row["speaker_is_first_person_org"]
    return True


def _bound_decision(cmp, value):
    """Decision implied by a p-value report at alpha=.05, else None."""

    if cmp in ("<", "<="):
        return True if value <= 0.05 else None
    if cmp in (">", ">="):
        return False if value >= 0.05 else None
    if cmp == "=":
        return value <= 0.05
    return None


def stat_consistency(text):
    rows = v1.stat_consistency(text)
    for row in rows:
        if not row.get("checkable"):
            continue
        reported = _bound_decision(row.get("reported_cmp"), row.get("reported_p"))
        recomputed = row.get("recomputed_p")
        row["decision_inconsistent"] = (
            None if reported is None or recomputed is None
            else reported != (recomputed <= 0.05)
        )
    return rows


_DELTA_RE = re.compile(
    r"(?:(?P<verb>increas\w*|decreas\w*|rose|risen|grew|fell|fallen|dropped|changed)\b[^.\n]{0,30}?)?"
    r"from\s+(?P<a>\d[\d,]*(?:\.\d+)?)\s+to\s+(?P<b>\d[\d,]*(?:\.\d+)?)"
    r"[^.\n%]{0,40}?(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent)"
    r"(?:\s*(?P<label>increase|decrease|rise|drop|fall|growth|reduction))?",
    re.I,
)


def number_consistency(text):
    if not text:
        return []
    # Retain v1's count/denominator checks; replace its sign-discarding delta rows.
    out = [r for r in v1.number_consistency(text) if r.get("kind") != "delta_pct"]
    for m in _DELTA_RE.finditer(str(text)[: v1._STAT_TEXT_CAP]):
        a = float(m.group("a").replace(",", ""))
        b = float(m.group("b").replace(",", ""))
        if a == 0:
            continue
        stated = float(m.group("pct"))
        computed = 100.0 * (b - a) / a
        cue = (m.group("label") or m.group("verb") or "").lower()
        cue_sign = 0
        if re.search(r"decreas|drop|fall|reduction", cue):
            cue_sign = -1
        elif re.search(r"increas|rise|rose|risen|grew|growth", cue):
            cue_sign = 1
        direction_ok = cue_sign == 0 or (computed > 0) == (cue_sign > 0)
        tol = max(1.0, 0.08 * abs(stated))
        magnitude_ok = abs(abs(computed) - abs(stated)) <= tol
        out.append(
            {
                "kind": "delta_pct",
                "a": a,
                "b": b,
                "stated_pct": stated,
                "computed_pct": round(computed, 2),
                "direction_cue": cue or None,
                "direction_consistent": direction_ok,
                "consistent": magnitude_ok and direction_ok,
                "context": m.group(0),
            }
        )
    return out


def date_chain(text):
    if not text:
        return []
    try:
        from dateutil import parser as dtparser
    except ImportError:
        return []
    out = []
    previous = None
    for m in v1._DATE_SCAN_RE.finditer(str(text)[: v1._STAT_TEXT_CAP]):
        raw = m.group(0)
        try:
            parsed = dtparser.parse(raw, fuzzy=True, default=_EPOCH).date()
        except Exception as exc:
            # A recognized date surface is evidence even when its calendar value is
            # impossible (the corpus contains a real "April 31" clerical error).  Dropping
            # it would make downstream anchor-selection treat the mention as absent.  The
            # explicit row lets callers abstain or apply a separately declared repair rule.
            out.append(
                {
                    "text": raw,
                    "start": m.start(),
                    "date": None,
                    "days_since_prev": None,
                    "parse_status": "INVALID",
                    "checkable": False,
                    "error": f"{type(exc).__name__}: invalid calendar date",
                }
            )
            continue
        out.append(
            {
                "text": raw,
                "start": m.start(),
                "date": parsed.isoformat(),
                "days_since_prev": (parsed - previous).days if previous else None,
                "parse_status": "VALID",
                "checkable": True,
            }
        )
        previous = parsed
    return out


def deadline_satisfied(event_date, filing_date, days):
    try:
        from dateutil import parser as dtparser

        def parse(value):
            if isinstance(value, datetime.datetime):
                return value.date()
            if isinstance(value, datetime.date):
                return value
            return dtparser.parse(str(value), fuzzy=True, default=_EPOCH).date()

        delta = (parse(filing_date) - parse(event_date)).days
        limit = int(days)
        if limit < 0:
            return None
        return 0 <= delta <= limit
    except Exception:
        return None


def is_refrain(text):
    """Recurring-sentence analysis without v1's three-word eligibility floor.

    The progression logic is retained, including the interleaved-boilerplate guard.  A
    callback is classified as craft only when *every* repeat has an intervening sentence;
    adjacency alone is never promoted into craft, even for a varied final line.
    """

    sents = v1._split_sentences(text)
    n = len(sents)
    if n < 2:
        return []
    all_bows = [v1._all_words(s) for _, _, s in sents]
    bows = [v1._content_words(s) for _, _, s in sents]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        if not all_bows[i]:
            continue
        for j in range(i + 1, n):
            if not all_bows[j]:
                continue
            inter = all_bows[i] & all_bows[j]
            if not inter:
                continue
            if len(inter) / len(all_bows[i] | all_bows[j]) >= 0.55:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    repeated_idxs = {i for occ in groups.values() if len(occ) >= 2 for i in occ}
    out = []
    for occ in groups.values():
        if len(occ) < 2:
            continue
        occ = sorted(occ)
        seen_before = set(bows[occ[0]])
        novelty = []
        for left, right in zip(occ, occ[1:]):
            between_words = set()
            for gap_i in range(left + 1, right):
                if gap_i not in repeated_idxs:
                    between_words |= bows[gap_i]
            if between_words:
                novelty.append(len(between_words - seen_before) / len(between_words))
            else:
                novelty.append(0.0)
            seen_before |= between_words | bows[right]
        varied_final = (
            all_bows[occ[0]] != all_bows[occ[-1]]
            and bool(all_bows[occ[0]] & all_bows[occ[-1]])
        )
        has_intervening = all(right - left >= 2 for left, right in zip(occ, occ[1:]))
        avg_novelty = sum(novelty) / len(novelty) if novelty else 0.0
        is_craft = has_intervening and (avg_novelty >= 0.4 or varied_final)
        if not has_intervening:
            reason = "no intervening sentence: adjacent repetition is not craft"
        elif is_craft and avg_novelty >= 0.4:
            reason = "escalating callback: new material between repeats"
        elif is_craft:
            reason = "escalating callback: varied final occurrence (twist line)"
        elif avg_novelty < 0.1:
            reason = "content-free or repeated gaps: no progression between repeats"
        else:
            reason = "low novelty between repeats"
        out.append(
            {
                "sentence": sents[occ[0]][2],
                "occurrences": occ,
                "is_refrain": is_craft,
                "reason": reason,
                "novelty_between": [round(x, 3) for x in novelty],
                "varied_final": varied_final,
                "has_intervening_sentence": has_intervening,
                "minimum_refrain_words": 1,
            }
        )
    return out


def discourse_position(text, span):
    if not isinstance(text, str) or not text:
        return None
    if isinstance(span, (tuple, list)) and len(span) == 2:
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            return None
        if start < 0 or end <= start or end > len(text):
            return None
    elif isinstance(span, str):
        starts = [m.start() for m in re.finditer(re.escape(span), text)] if span else []
        if len(starts) != 1:
            return None
        start = starts[0]
    else:
        return None
    sentences = v1._split_sentences(text)
    idx = next((i for i, (a, b, _s) in enumerate(sentences) if a <= start < b), None)
    if idx is None:
        return None
    opening = max(1, math.ceil(len(sentences) * 0.15))
    coda = len(sentences) - max(1, math.ceil(len(sentences) * 0.10))
    return "opening" if idx < opening else "coda" if idx >= coda else "middle"


class CapabilityOps:
    attributions = staticmethod(attributions)
    self_attributed = staticmethod(self_attributed)
    parse_math = staticmethod(v1.parse_math)
    licensing_does_work = staticmethod(v1.licensing_does_work)
    restates_definition = staticmethod(v1.restates_definition)
    stat_consistency = staticmethod(stat_consistency)
    number_consistency = staticmethod(number_consistency)
    date_chain = staticmethod(date_chain)
    deadline_satisfied = staticmethod(deadline_satisfied)
    sentence_graph = staticmethod(v1.sentence_graph)
    is_refrain = staticmethod(is_refrain)
    discourse_position = staticmethod(discourse_position)
    fact_density = staticmethod(v1.fact_density)
    entities_with_evidence = staticmethod(v1.entities_with_evidence)
