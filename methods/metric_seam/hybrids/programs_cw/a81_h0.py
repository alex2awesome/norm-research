"""a81 hybrid metric channel: Multithread and ensemble management.

Criterion: interlocking threads/ensembles with distinct goals; deliberate
alternation of strands; separations doing new work; clean resolution.

Design:
- The predicate lives in code: strand-head diversity (distinct paragraph
  anchors), anchor alternation, internal scene breaks, thread cue words,
  distinct speaker attributions; a monologue penalty for pure single-'I'
  pieces.
- Two LLM fields provide thick-input grounding regex cannot see (unnamed
  dual protagonists, timeline braids): a thread COUNT and an INTERLOCK
  verdict. Code maps them through fixed monotone tables and blends with
  the code signal; empty extraction falls back to code-only, centered
  near the corpus-typical middle (high-NA criterion).
"""

import re

LLM_FIELDS = {
    "thread_count": "Count the distinct narrative threads (separate plotlines, POVs, or storylines the story cuts between); answer with only that number.",
    "interlock": "Do multiple plotlines or characters with distinct goals interlock and resolve together by the end? Answer YES, PARTLY, or NO.",
}

# Capitalized tokens that are never treated as character names.
_CAP_STOP = frozenset("""
The A An And But Or Nor So Yet If Then Than That This These Those There Here
He She It They We You I Me My His Her Its Our Your Their Him Them Us
What When Where Which Who Whom Whose Why How Not No Yes Oh Ah Well Now Just
After Before While During Since Until Once Again Still Also Even Ever Never
Suddenly Finally Maybe Perhaps Instead Meanwhile Elsewhere Everyone Everything
Someone Something Anyone Anything Nobody Nothing All Any Some Both Each Every
One Two Three Four Five Six Seven Eight Nine Ten First Second Third Last Next
Monday Tuesday Wednesday Thursday Friday Saturday Sunday
January February March April May June July August September October November
December Today Tomorrow Yesterday Day Night Morning Evening Time Year Years
Mr Mrs Ms Miss Dr St Ok Okay Hey Hello Look Come Go Stop Wait Please Thanks
Thank Sorry Damn Fuck Shit Jesus Christ Holy New Old Good Bad Big Little
Edit Final Note Update Chapter Part Prologue Epilogue End Fin
Reddit Facebook Google Internet Earth America American English
""".split())

_SAID_VERBS = (
    "said|asked|replied|answered|shouted|whispered|muttered|yelled|exclaimed|"
    "bellowed|interrupted|continued|added|began|started|cried|groaned|sighed|"
    "laughed|snapped|grumbled|hissed|huffed|agreed|repeated|spoke|called|"
    "responded|demanded|offered|admitted|remarked|stated|declared|wondered"
)

_CUE_RE = re.compile(
    r"\bmeanwhile\b|\belsewhere\b|\bacross town\b|\bat the same time\b|"
    r"\bback at (?:the|his|her|their)\b|\bin another (?:part|place|room|world|"
    r"city|land|corner)\b|\bmiles away\b|\bon the other side of (?:town|the "
    r"city|the world)\b|\bunbeknownst\b|\bfor (?:his|her|their) part\b",
    re.IGNORECASE,
)

_BREAK_RE = re.compile(r"^\s*\\?[-_~*#=+.─-╿]{3,}\s*$")

_NOTE_RE = re.compile(
    r"^\s*(?:final\s+)*edit\s*\d*\s*[:\-]|(?:\br/\w+)|thanks? (?:you )?(?:so much )?"
    r"for reading|first time post|long time lurker|constructive criticism|"
    r"feedback (?:is )?(?:welcome|appreciated)|check out (?:my|r/)|if you "
    r"enjoyed|writingprompts|dont tear me apart|don.t tear me apart|"
    r"any support helps|\[wp\]",
    re.IGNORECASE,
)


def _clean(text):
    """Drop author-note/meta lines; return cleaned text."""
    lines = text.split("\n")
    kept = [ln for ln in lines if not _NOTE_RE.search(ln)]
    return "\n".join(kept)


def _paragraphs(text):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) < 4:
        # hard line-wrapped corpus: fall back to single-newline units
        alt = [p.strip() for p in text.split("\n") if p.strip()]
        if len(alt) >= 2 * len(paras):
            paras = alt
    return paras


def _actor_candidates(text):
    """Capitalized tokens seen >=2 times, at least once mid-sentence."""
    counts = {}
    mid = set()
    for m in re.finditer(r"[A-Z][a-z]{2,}", text):
        w = m.group(0)
        if w in _CAP_STOP:
            continue
        counts[w] = counts.get(w, 0) + 1
        j = m.start() - 1
        while j >= 0 and text[j] in " \t":
            j -= 1
        if j >= 0 and text[j] not in '.!?"\n“”*(':
            mid.add(w)
    return {w for w, c in counts.items() if c >= 2 and w in mid}


def _speakers(text, actors):
    """Distinct named + role speakers in dialogue attributions."""
    named = set()
    roles = set()
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\s+(?:%s)\b" % _SAID_VERBS, text):
        if m.group(1) in actors:
            named.add(m.group(1))
    for m in re.finditer(r"\b(?:%s)\s+([A-Z][a-z]{2,})\b" % _SAID_VERBS, text):
        if m.group(1) in actors:
            named.add(m.group(1))
    for m in re.finditer(
        r"\b(?:the|his|her|their|an?)\s+([a-z]{3,})\s+(?:%s)\b" % _SAID_VERBS,
        text,
    ):
        roles.add(m.group(1))
    for m in re.finditer(
        r"\b(?:%s)\s+the\s+([a-z]{3,})\b" % _SAID_VERBS, text
    ):
        roles.add(m.group(1))
    return named, roles


def _anchor(para, actors):
    """Strand-head anchor of a paragraph: first actor name near the start,
    else 'I' for first-person openings, else None."""
    head = para[:140]
    best = None
    best_pos = len(head) + 1
    for a in actors:
        m = re.search(r"\b%s\b" % re.escape(a), head)
        if m and m.start() < best_pos:
            best, best_pos = a, m.start()
    if best is not None:
        return best
    if re.match(r'^["“*]?(?:I|My|Me)\b', para):
        return "I"
    return None


def _code_signal(text, ops):
    try:
        t = ops.normalize(text)
    except Exception:
        t = text
    if not isinstance(t, str) or not t.strip():
        return 0.42
    t = _clean(t)
    paras = _paragraphs(t)
    n_p = max(1, len(paras))

    actors = _actor_candidates(t)
    named_sp, role_sp = _speakers(t, actors)

    # internal scene breaks (ignore final 15% where sign-offs live)
    cut = int(n_p * 0.85)
    breaks = sum(1 for i, p in enumerate(paras[:cut]) if _BREAK_RE.match(p))

    # paragraph strand anchors + alternation
    anchors = [_anchor(p, actors) for p in paras]
    seq = [a for a in anchors if a is not None]
    heads = set(seq)
    n_heads = len(heads)
    alt = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    i_frac = (seq.count("I") / len(seq)) if seq else 0.0

    cues = len(_CUE_RE.findall(t))
    n_speak = len(named_sp) + min(len(role_sp), 3)

    pos = 0.0
    pos += 0.34 * min(max(n_heads - 1, 0), 3) / 3.0
    pos += 0.26 * min(alt, 6) / 6.0
    pos += 0.16 * min(breaks, 2) / 2.0
    pos += 0.12 * min(cues, 3) / 3.0
    pos += 0.12 * min(max(n_speak - 1, 0), 3) / 3.0

    s = 0.32 + 0.44 * pos

    # pure single-strand monologue: one head (or none) and I-dominated
    non_i = heads - {"I"}
    if len(non_i) == 0 and i_frac > 0.5:
        s -= 0.08
    if len(non_i) == 0 and not breaks and cues == 0 and n_speak <= 1:
        s -= 0.04

    return min(max(s, 0.08), 0.92)


_THREAD_MAP = {1: 0.30, 2: 0.58, 3: 0.70, 4: 0.78}


def _parse_threads(raw):
    if not raw or not isinstance(raw, str):
        return None
    m = re.search(r"\d+", raw)
    if not m:
        low = raw.lower()
        if re.search(r"\bone\b|\bsingle\b", low):
            return 1
        if re.search(r"\btwo\b", low):
            return 2
        if re.search(r"\bthree\b", low):
            return 3
        return None
    n = int(m.group(0))
    return min(max(n, 1), 9)


def _parse_interlock(raw):
    if not raw or not isinstance(raw, str):
        return None
    low = raw.lower()
    if "partly" in low or "partial" in low or "somewhat" in low:
        return 0.0
    if re.search(r"\bno\b|\bnone\b", low):
        return -0.07
    if re.search(r"\byes\b", low):
        return 0.07
    return None


def score(text: str, extracted: dict, ops) -> float:
    try:
        code = _code_signal(text if isinstance(text, str) else "", ops)

        ext = extracted if isinstance(extracted, dict) else {}
        n_thr = _parse_threads(ext.get("thread_count", ""))
        ilk = _parse_interlock(ext.get("interlock", ""))

        if n_thr is not None:
            llm = _THREAD_MAP.get(n_thr, 0.80)
            if ilk is not None and n_thr >= 2:
                llm += ilk
            s = 0.62 * llm + 0.38 * code
        elif ilk is not None:
            s = code + 0.5 * ilk
        else:
            s = code
        return float(min(max(s, 0.05), 0.95))
    except Exception:
        return 0.5
