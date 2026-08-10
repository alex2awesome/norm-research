"""a333: Rule of three / triples (hybrid v1 -- agentic round 2).

ROUND-1 (h0) DIAGNOSIS from train residuals:
  - h0 trusts the LLM's `triad_items` field almost blindly whenever it parses
    to exactly 3 comma/semicolon-separated pieces (a=0.65, +0.25 twist bonus
    -> up to 0.90), with code-side corroboration only as a light "nudge"
    (weight 0.25). But the extractor frequently returns 3 pieces that are
    just a plain narrative list INSIDE ONE SENTENCE ("search the shed, break
    every piece of wood, find no marijuana"; "Smith, Jones, Baker"; "got up,
    straightened out his shirt, and stumbled away") -- not three genuinely
    separate, parallel beats (dialogue turns / repeated list items / an
    escalating refrain). These fake-list triads scored 0.90-0.94 in h0 but
    judge=0.00 on every train example checked.
  - Conversely, h0's code-side fallback corroboration (when the LLM found
    NOTHING) used "some 4+ letter content word appears exactly 3 times
    anywhere in the text" as its main structural signal. On texts this short
    that fires by pure coincidence very often (a topic word like "vote" or
    "phone" happening to occur 3x has nothing to do with rule-of-three
    structure) and produced a large cluster of judge=0.00 items parked at a
    near-identical score (~0.24), actively hurting rank because genuine mild
    positives (judge 0.10-0.25, extractor-missed triads like a 5-item
    escalating list) sat BELOW that false-positive floor at h0's flat 0.15.

FIX (this file):
  - New code-side structural signal, computed independently of the LLM
    claim, from sentence/line-level UNITS of the raw text:
      * near-duplicate units (parallel dialogue turns / repeated list lines)
        via a Jaccard clique over content words -- generalizes h0's
        single-word repeat count to actual TEMPLATED repetition across
        separate beats, not incidental word reuse within one beat;
      * repeated content-word BIGRAMS (adjacent content-word pairs) across
        the whole text -- catches near-verbatim refrains ("Give her another
        chance!", "Do you have a banana?") even when individual units don't
        cluster;
      * explicit ordinal sequencing (first...second...third), reused from
        h0;
      * bare "three/triple/trio/triad" mention, kept at a deliberately tiny
        weight per the known-failure note that this is a weak proxy.
    A title-echo guard collapses the corpus's common "Title Title-repeated-
    as-first-sentence..." artifact before any of this runs, so that dataset
    formatting duplication is not mistaken for triadic repetition.
  - NEW GATE: an independent check of whether the LLM's own triad_items best
    match THREE DIFFERENT sentence units (real corroboration -> near-full
    trust) or all collapse into the SAME single unit (probable fake in-
    sentence list -> heavily discounted UNLESS the independent structural
    signal above is separately strong, e.g. when the genuine triad in the
    text is a refrain elsewhere and the extractor grabbed a decoy list).
  - Fallback (LLM found nothing) is now driven mainly by the same
    independent structural signal (rescues genuine extractor-missed
    triads/near-triads) instead of a near-flat floor.
"""
import re
from collections import Counter

LLM_FIELDS = {
    "triad_items": "List the 3 parallel beats/items (comma or semicolon separated) if this text uses a rule-of-three structure, else NONE.",
    "third_twist": "In up to 8 words, say how the third/final beat escalates or subverts the first two beats, else NONE.",
}

_NONE_TOKENS = {"", "none", "n/a", "na", "no", "nothing", "not applicable"}

_NEGATIVE_TWIST_MARKERS = (
    "same as", "no differ", "identical", "no escalat", "no surpris",
    "nothing differ", "no twist", "no change", "not differ",
)

_STOPWORDS = {
    "that", "this", "with", "from", "have", "those", "these", "would",
    "could", "should", "there", "their", "when", "what", "your", "were",
    "been", "into", "just", "like", "some", "them", "then", "also",
    "because", "after", "before", "during", "which", "about", "where",
    "being", "doing", "having", "only", "very", "such", "most", "more",
    "than", "over", "under", "again", "further", "once", "here", "other",
    "same", "each", "both", "against", "between", "through", "above",
    "below", "down", "near", "upon", "within", "without", "will", "shall",
    "went", "walk", "walks", "walked", "walking", "said", "says",
}

_ORDINAL_WORDS = ("first", "second", "third")
_NUMERAL_RE = re.compile(r"\b(three|trio|triple|triad)\b")
_CONTENT_WORD_RE = re.compile(r"\b[a-zA-Z]{4,}\b")
_UNIT_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|(?<=[.!?][\"'’”])\s+|\n+"
)


def _clean_field(value) -> str:
    try:
        return (value or "").strip()
    except Exception:
        return ""


def _is_none_token(s: str) -> bool:
    return s.strip().lower() in _NONE_TOKENS


def _content_words(unit: str):
    return [w for w in _CONTENT_WORD_RE.findall(unit.lower()) if w not in _STOPWORDS]


def _split_units(t: str):
    raw = [u.strip() for u in _UNIT_SPLIT_RE.split(t) if u.strip()]
    # Title-echo guard: this corpus commonly repeats the (headline-like)
    # opening chunk verbatim as the start of the first real sentence, e.g.
    # "Wife checking phone Wife was checking her husband's phone...". That
    # duplication is a formatting artifact, not triadic repetition -- if the
    # first unit's content words are a prefix of the second unit's, drop the
    # short duplicate.
    if len(raw) >= 2:
        u0 = _content_words(raw[0])
        u1 = _content_words(raw[1])
        if len(u0) >= 3 and u0 == u1[:len(u0)]:
            raw = raw[1:]
    return raw[:40]


def _ordinal_sequence_score(t: str) -> float:
    positions = {}
    for w in _ORDINAL_WORDS:
        m = re.search(r"\b" + w + r"\b", t)
        positions[w] = m.start() if m else None
    if all(positions[w] is not None for w in _ORDINAL_WORDS):
        if positions["first"] < positions["second"] < positions["third"]:
            return 0.22
        return 0.12
    if positions["third"] is not None and (
        positions["first"] is not None or positions["second"] is not None
    ):
        return 0.08
    return 0.0


def _repeated_bigram_score(units) -> float:
    counts = Counter()
    for u in units:
        words = _content_words(u)
        for a, b in zip(words, words[1:]):
            counts[(a, b)] += 1
    if not counts:
        return 0.0
    m = max(counts.values())
    if m >= 3:
        return 0.18
    if m == 2:
        return 0.10
    return 0.0


def _max_mutual_clique(units) -> int:
    """Largest MUTUALLY-linked group of near-duplicate units (every pair in
    the group independently passes Jaccard>=0.3 with >=2 shared content
    words) -- corroborates genuinely repeated/parallel beats (dialogue
    turns, escalating list lines) independent of whatever the LLM's
    triad_items claims.

    Deliberately NOT a transitive connected-component: chaining A~B and B~C
    into "a size-3 cluster" when A and C never directly match is exactly how
    generic story-wide overlap (e.g. two characters' names both co-occurring
    with a third, unrelated sentence) produces a spurious triad signal --
    two named characters recurring across many sentences of ANY story is not
    evidence of rule-of-three structure. Requiring every pair in the group
    to independently match keeps the signal to genuine templated repeats.
    """
    n = len(units)
    if n < 2:
        return 0
    wordsets = [set(_content_words(u)) for u in units]

    def linked(i, j):
        wi, wj = wordsets[i], wordsets[j]
        if len(wi) < 2 or len(wj) < 2:
            return False
        inter = wi & wj
        if len(inter) < 2:
            return False
        uni = wi | wj
        return bool(uni) and len(inter) / len(uni) >= 0.3

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if linked(i, j)]
    if not pairs:
        return 0
    best = 2
    pair_set = set(pairs)
    for i, j in pairs:
        for k in range(n):
            if k in (i, j):
                continue
            a, b = (min(i, k), max(i, k))
            c, d = (min(j, k), max(j, k))
            if (a, b) in pair_set and (c, d) in pair_set:
                best = 3
                break
        if best == 3:
            break
    return best


def _structural_score(t: str, units) -> float:
    ordinal = _ordinal_sequence_score(t)
    numeral = 0.03 if _NUMERAL_RE.search(t) else 0.0
    bigram = _repeated_bigram_score(units)
    clique = _max_mutual_clique(units)
    if clique >= 3:
        clique_score = 0.45
    elif clique == 2:
        clique_score = 0.06
    else:
        clique_score = 0.0
    return min(0.85, ordinal + numeral + bigram + clique_score)


_MATCH_WORD_RE = re.compile(r"\b[a-zA-Z']{3,}\b")
_MATCH_EXTRA_STOP = {
    "got", "his", "her", "its", "she", "him", "the", "and", "was", "were",
    "are", "that", "this", "out", "too", "you", "not", "for",
}


def _match_words(s: str):
    return [w for w in _MATCH_WORD_RE.findall(s.lower())
            if w not in _STOPWORDS and w not in _MATCH_EXTRA_STOP]


def _can_separate(cand_sets) -> bool:
    """Bipartite matching: can every item be assigned to a DIFFERENT
    candidate unit? (Simple augmenting-path search; item counts are tiny
    (<=5), so a plain DFS is more than fast enough.)"""
    n = len(cand_sets)
    match_to_item = {}

    def try_assign(i, visited):
        for u in cand_sets[i]:
            if u in visited:
                continue
            visited.add(u)
            if u not in match_to_item or try_assign(match_to_item[u], visited):
                match_to_item[u] = i
                return True
        return False

    count = 0
    for i in range(n):
        if try_assign(i, set()):
            count += 1
    return count == n


def _items_same_unit(items, units):
    """Returns (same_unit: bool, unit_idx: int|None).

    same_unit=True means the LLM's extracted items are NOT separable across
    distinct sentence units -- i.e. likely a plain in-sentence comma list,
    not genuinely separate beats. unit_idx is the shared unit index when
    same_unit is True and resolvable (used for a terminal-position check by
    the caller: a same-unit list that IS the text's own final beat can still
    be a genuine single-sentence rhetorical triad, e.g. "king-size,
    extra-firm, and always laying on my box-spring").

    Uses a lenient (>=3 letter) word filter for placement matching -- items
    are short LLM paraphrases (e.g. "got up") that can be all-short-words
    and would otherwise be unplaceable under the stricter >=4 letter content
    filter used for structural clique/bigram detection. A common failure
    mode this guards against: an early "cast list" sentence that names all
    N items together (e.g. "lil' droplet, lil' feather, and lil' brick ask
    their mothers...") ties with each item's OWN dedicated paragraph later
    in the text; naive argmax always resolves to the first tie (the shared
    intro line), wrongly flagging a genuinely-separate triad as one fake
    list. Bipartite separability over the full tie sets fixes this: if each
    item has an available candidate unit distinct from the others', they are
    treated as separable even though a shared "hub" unit also ties.
    """
    if len(units) < 2:
        return True, (0 if units else None)
    unit_wordsets = [set(_match_words(u)) for u in units]
    cand_sets = []
    for item in items:
        iw = set(_match_words(item))
        if not iw:
            cand_sets.append(set())
            continue
        scores = [len(iw & uw) for uw in unit_wordsets]
        mx = max(scores) if scores else 0
        if mx == 0:
            cand_sets.append(set())
            continue
        cand_sets.append({idx for idx, sc in enumerate(scores) if sc == mx})
    resolved_sets = [s for s in cand_sets if s]
    if len(resolved_sets) < 2:
        return True, None
    if _can_separate(resolved_sets):
        return False, None
    # Not separable -- find the single unit common to (most of) the
    # resolved items, for the caller's terminal-position check.
    common = set.intersection(*resolved_sets) if resolved_sets else set()
    shared_idx = min(common) if common else min(min(s) for s in resolved_sets)
    return True, shared_idx


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            normed = ops.normalize(raw)
        except Exception:
            normed = raw
        t = (normed or raw).lower()
        if not t.strip():
            return 0.5

        units = _split_units(t)
        struct = _structural_score(t, units)

        extracted = extracted or {}
        triad_field = _clean_field(extracted.get("triad_items", ""))
        twist_field = _clean_field(extracted.get("third_twist", ""))
        triad_present = not _is_none_token(triad_field)

        if triad_present:
            items = [x.strip() for x in re.split(r"[;,]", triad_field) if x.strip()]
            n = len(items)
            if n == 3:
                a_raw = 0.60
            elif n in (2, 4):
                a_raw = 0.40
            else:
                a_raw = 0.30

            same_unit, shared_idx = _items_same_unit(items, units)
            if same_unit:
                is_terminal = (
                    shared_idx is not None and len(units) >= 1
                    and shared_idx >= len(units) - 2
                )
                if is_terminal:
                    # Not separable across units, but the list IS the text's
                    # own final beat -- plausibly a genuine single-sentence
                    # rhetorical triad (classic "two short parallel items,
                    # then a longer/escalated third" punchline), not a
                    # mid-narrative procedural list. Moderate trust.
                    gate = 0.45 + 0.40 * min(1.0, struct / 0.50)
                else:
                    # Mid-text, not separable: probable fake in-sentence
                    # list. Trust it only to the extent the text shows
                    # genuine parallel repetition elsewhere.
                    gate = 0.14 + 0.46 * min(1.0, struct / 0.50)
            else:
                # Items genuinely span separate beats/turns -- strong
                # corroboration on its own; struct adds a small extra nudge.
                gate = 0.82 + 0.18 * min(1.0, struct / 0.40)
            gate = max(0.0, min(1.0, gate))
            a = a_raw * gate

            if twist_field and not _is_none_token(twist_field):
                twist_l = twist_field.lower()
                if any(marker in twist_l for marker in _NEGATIVE_TWIST_MARKERS):
                    twist_bonus = 0.0
                else:
                    twist_bonus = 0.08 + 0.14 * gate
            else:
                twist_bonus = 0.0
            a = min(1.0, a + twist_bonus)
        else:
            # LLM found nothing -- rescue on independent structural evidence
            # (e.g. a repeated-list/refrain the extractor missed) instead of
            # a near-flat floor.
            a = 0.05 + 0.55 * struct

        n_sent = None
        try:
            stats = ops.sent_stats(raw)
            if isinstance(stats, dict):
                for k in ("n_sent", "num_sentences", "n_sentences", "sentence_count"):
                    if k in stats:
                        n_sent = stats[k]
                        break
            elif stats:
                n_sent = stats[0]
        except Exception:
            n_sent = None
        if isinstance(n_sent, (int, float)) and n_sent <= 1:
            a *= 0.6

        return max(0.0, min(1.0, a))
    except Exception:
        return 0.5
