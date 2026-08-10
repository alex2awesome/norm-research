"""Hybrid metric channel for a144: Sentence-level rhythm, sound, and memorability.

Insight from train residuals: the structure-stat baseline (rho ~= .04) fails because
the judge is not scoring raw variance/TTR -- it separates (a) mechanically sloppy
prose (missing apostrophes, lowercase sentence starts, run-ons, unterminated lines)
at the bottom, (b) clean-but-plain prose in the middle, and (c) clean prose with
DELIBERATE cadence control (staccato one-line beats, em-dash/semicolon music,
controlled short/long alternation) at the top.  Code computes a mechanics-error
axis plus a cadence-control axis, with format-aware scaffolding removal (email
headers, chat speaker tags, markdown dividers, link lists) so epistolary/chat
stories are not punished for mimetic surface.  One LLM field grounds the tacit
"is the rhythm deliberate?" judgment regex cannot see; a second extracts concrete
mechanical errors as a cross-check on the code error detector.

b4 extension (blind, construct-driven -- no eval signal used to pick these):
the criterion text explicitly asks for rhythm/diction tuned "to action/emotion"
and for "resonance"/"memorability", two things the h0 code and LLM fields never
touch -- h0's cadence_quality only asks whether variation is SKILLFUL in the
abstract, not whether it is DEPLOYED to track the scene's content, and nothing
in h0 checks for a standout quotable line. Two new fields close that gap:
  - rhythm_matches_content: does the sentence rhythm actually shift with the
    scene's action/emotional intensity (the "diction ... to action/emotion"
    half of the criterion), as opposed to merely varying for its own sake.
  - memorable_line: an extractive quote of the single most striking line,
    used (with a light grounding check against the source text) as a direct
    proxy for the criterion's "memorability" clause, which nothing else in
    the module measures.
"""

import re
import math
import statistics as st

LLM_FIELDS = {
    "cadence_quality": (
        "Rate 1 to 9 how skillfully the sentences vary length, rhythm, and "
        "punctuation for deliberate musical or dramatic effect (1=clumsy, "
        "9=masterful). Answer with the number only."
    ),
    "prose_errors": (
        "List up to 5 unintentional spelling, punctuation, or grammar mistakes "
        "exactly as written in the story, comma-separated; answer NONE if the "
        "prose is clean."
    ),
    "rhythm_matches_content": (
        "Does the sentence rhythm (short/choppy vs. long/flowing) shift to "
        "track the scene's action or emotional intensity, rather than staying "
        "uniform throughout? Answer YES, NO, or MIXED."
    ),
    "memorable_line": (
        "Quote the single most vivid or rhythmically striking sentence or "
        "phrase (<=12 words) from the story, exactly as written. Answer NONE "
        "if nothing stands out as memorable."
    ),
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"https?://\S+|www\.\S+|\[([^\]]*)\]\((?:[^)]*)\)|/?r/[A-Za-z0-9_]+", re.I
)
_ENTITY_RE = re.compile(r"&(?:amp|gt|lt|nbsp|#x200b|#x200B|#8203);", re.I)
_RULE_LINE = re.compile(r"^\s*[-=_*~#\s.]{3,}\s*$")
_HEADER_LINE = re.compile(
    r"^\s*(?:to|from|cc|bcc|subject|date|re|edit|update|ps|p\.s\.?)\s*[:\-]", re.I
)
_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*+•])\s")
_CHAT_TAG = re.compile(r"^\s*[A-Z][A-Za-z'. ]{1,28}:\s+\S")

_APOS_MISS = re.compile(
    r"\b(?:dont|doesnt|didnt|isnt|arent|wasnt|werent|cant|couldnt|wouldnt|"
    r"shouldnt|wont|aint|ive|youre|theyre|theres|whats|hasnt|havent|im|hes|"
    r"wouldve|couldve|shouldve)\b"
)
_LC_I = re.compile(r"(?<![\w'])i(?=[\s,.!?;:])")
_NOSPACE = re.compile(r"[a-z]{2,}[.!?,][A-Za-z]{2,}")
_SPACE_BEFORE = re.compile(r"[A-Za-z]\s+[,.!?;:](?:\s|$)")
_TERMINAL = ('.', '!', '?', '"', "'", '*', ')', ':', '-', ';', ',')
_DASH_MUSIC = re.compile(
    r"(?<=\w)—|—(?=\w)|(?<=\w) — (?=\w)|(?<=\w)--(?=\w)|"
    r"(?<=\w) -- (?=\w)|(?<=[a-z]) - (?=[a-z])"
)
_ELLIPSIS_END = re.compile(r"(?:\.\.\.|…)['\"]?\s*$")


def _prep(text, ops):
    """Normalize and strip non-prose chrome (URLs, entities, markdown)."""
    t = text or ""
    try:
        t = ops.normalize(t)
    except Exception:
        pass
    t = _ENTITY_RE.sub(" ", t)
    t = _URL_RE.sub(" ", t)
    t = t.replace("\r", "")
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("‘", "'").replace("’", "'")
    t = t.replace("[...]", " ")
    t = re.sub(r"\\\*", "*", t)
    lines = []
    for ln in t.split("\n"):
        if _RULE_LINE.match(ln) and ln.strip():
            lines.append("")            # divider -> paragraph break, not prose
        else:
            lines.append(ln)
    t = "\n".join(lines)
    # markdown emphasis / quote markers (after divider removal)
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"^\s*>+", "", t, flags=re.M)
    return t


def _split_lines(t):
    """Return (prose_lines, chat_frac): scaffold-stripped lines plus how
    chat/epistolary-formatted the document is."""
    raw = [ln.strip() for ln in t.split("\n") if ln.strip()]
    if not raw:
        return [], 0.0
    prose, n_format = [], 0
    for ln in raw:
        if _HEADER_LINE.match(ln):
            n_format += 1
            continue                      # email/PS headers: scaffold, drop
        if _LIST_ITEM.match(ln):
            n_format += 1
            continue                      # link/list items: scaffold, drop
        m = _CHAT_TAG.match(ln)
        if m:
            n_format += 1
            prose.append(ln[m.end() - 1:].strip())  # keep message body
            continue
        prose.append(ln)
    return prose, n_format / len(raw)


def _sentences(t):
    parts = re.split(r"(?<=[.!?…])\s+|\n+", t)
    return [p.strip() for p in parts if p.strip()]


def _strict_sentences(t):
    """Sentence starts for capitalization checks: do NOT split at ellipses
    ('wait... nah' voice style is deliberate, not an error)."""
    parts = re.split(r"(?<=[.!?])(?<!\.\.)\s+|\n+", t)
    return [p.strip() for p in parts if p.strip()]


def _clip01(x):
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# code features
# ---------------------------------------------------------------------------

def _code_features(t):
    prose_lines, chat_frac = _split_lines(t)
    body = "\n".join(prose_lines)
    if not body.strip():
        body = t
    is_format_piece = chat_frac > 0.4     # chat log / heavy epistolary

    words = re.findall(r"[A-Za-z']+", body)
    n_words = max(1, len(words))
    sents = _sentences(body)
    lens = [len(re.findall(r"[A-Za-z']+", s)) for s in sents]
    lens = [l for l in lens if l > 0] or [1]
    n_sent = len(lens)
    mean_len = sum(lens) / n_sent

    # ---- mechanics error axis (negatives) ----
    lc_i = len(_LC_I.findall(body))
    apos = len(_APOS_MISS.findall(body))
    nospace = len(_NOSPACE.findall(body))
    spb = len(_SPACE_BEFORE.findall(body))
    lc_starts = 0
    for s in _strict_sentences(body):
        m = re.match(r'^["\'(]*\s*([A-Za-z])', s)
        if m and m.group(1).islower():
            lc_starts += 1
    err = 2.0 * lc_i + 2.0 * apos + 2.0 * nospace + 1.0 * spb + 1.5 * lc_starts
    err_rate = err / n_words * 100.0      # weighted errors per 100 words
    f_err = _clip01(err_rate / 4.0)

    # unterminated substantial lines (missing end punctuation, doggerel)
    subst = [ln for ln in prose_lines if len(ln.split()) >= 4]
    if subst:
        unterm = sum(1 for ln in subst if not ln.endswith(_TERMINAL))
        f_unterm = _clip01(unterm / len(subst) / 0.5)
    else:
        f_unterm = 0.0

    if is_format_piece:
        # mimetic surface (chat-speak, headers): mechanics not trustworthy
        f_err *= 0.25
        f_unterm *= 0.2

    # run-on tendency: very long average sentences
    f_runon = _clip01((mean_len - 22.0) / 14.0)

    # overwrought 'as'-chaining (amateur simultaneity tell)
    as_rate = len(re.findall(r"\bas\b", body.lower())) / n_words * 100.0
    f_as = _clip01((as_rate - 2.5) / 2.5)

    # trailing-off melodrama: paragraphs that END on an ellipsis
    all_paras = [p.strip() for p in re.split(r"\n\s*\n|\n", body) if p.strip()]
    if len(all_paras) >= 4:
        ell_ends = sum(1 for p in all_paras if _ELLIPSIS_END.search(p))
        f_ellip = _clip01(ell_ends / len(all_paras) / 0.3)
    else:
        f_ellip = 0.0

    # lowercase stretch-filler ("uhhhh", "soooo") -- transcribed hedging
    stretch = len(re.findall(r"\b[a-z]*([a-z])\1{3,}[a-z]*\b", body))
    f_fill = _clip01(stretch / 3.0)
    if is_format_piece:
        f_fill *= 0.25                    # chat elongation is mimetic

    # ---- cadence-control axis (positives) ----
    cv = (st.pstdev(lens) / mean_len) if n_sent > 1 else 0.0
    frac_short = sum(1 for l in lens if l <= 5) / n_sent
    frac_long = sum(1 for l in lens if l >= 22) / n_sent
    f_burst = _clip01(cv / 0.9) * _clip01(4.0 * min(frac_short, frac_long + 0.05))

    # punctuation music: em-dash / semicolon woven into running prose
    dashes = len(_DASH_MUSIC.findall(body))
    semis = body.count(";")
    f_music = _clip01((dashes + semis) / max(6.0, n_sent * 0.35))

    # one-line beat paragraphs ("Nothing." / "And again.") -- clean, short,
    # decisively ended (ellipsis endings are melodrama, not staccato)
    paras = all_paras
    if len(paras) >= 6:
        beats = 0
        for p in paras:
            w = p.split()
            if (2 <= len(w) <= 7 and p.endswith(('.', '!', '?'))
                    and not _ELLIPSIS_END.search(p)
                    and not _LIST_ITEM.match(p)
                    and sum(1 for x in w if re.search(r"[A-Za-z]{2}", x)) >= 2):
                beats += 1
        f_beat = _clip01(beats / len(paras) / 0.22)
    else:
        f_beat = 0.0

    return {
        "f_err": f_err, "f_unterm": f_unterm, "f_runon": f_runon,
        "f_as": f_as, "f_ellip": f_ellip, "f_fill": f_fill,
        "f_burst": f_burst, "f_music": f_music, "f_beat": f_beat,
        "chat": 1.0 if is_format_piece else 0.0,
    }


def _code_score(t):
    f = _code_features(t)
    neg = (0.32 * f["f_err"] + 0.17 * f["f_unterm"] + 0.17 * f["f_runon"]
           + 0.10 * f["f_as"] + 0.06 * f["f_ellip"] + 0.05 * f["f_fill"])
    pos = 0.11 * f["f_burst"] + 0.09 * f["f_music"] + 0.10 * f["f_beat"]
    return _clip01(0.50 - neg + pos), f


# ---------------------------------------------------------------------------
# LLM field parsing
# ---------------------------------------------------------------------------

def _parse_rating(ans):
    """Parse a 1-9 rating; return None if unusable."""
    if not ans:
        return None
    m = re.search(r"\d+(?:\.\d+)?", str(ans))
    if not m:
        return None
    v = float(m.group(0))
    if v < 1.0 or v > 10.0:
        return None
    return _clip01((min(v, 9.0) - 1.0) / 8.0)


def _parse_errors(ans):
    """Return error-load in [0,1], or None if field unusable/absent."""
    if ans is None:
        return None
    a = str(ans).strip()
    if not a:
        return 0.0
    if re.match(r"^\W*none\b", a, re.I) or a.lower() in ("n/a", "no errors", "clean"):
        return 0.0
    items = [x for x in re.split(r"[,;\n]+", a) if x.strip()]
    return _clip01(len(items) / 5.0)


def _parse_rhythm_match(ans):
    """Parse the rhythm-matches-content field into [0,1]; None if absent."""
    if ans is None:
        return None
    a = str(ans).strip().upper()
    if not a:
        return None
    if a.startswith("YES"):
        return 1.0
    if a.startswith("MIXED") or a.startswith("PARTIAL") or a.startswith("SOME"):
        return 0.5
    if a.startswith("NO"):
        return 0.0
    return None


def _norm_snippet(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _parse_memorable(ans, doc_text):
    """Return a grounding score in [0,1] for a quoted memorable line, or
    None if the field is absent from `extracted`. A quote that is actually
    findable (near-verbatim) in the source text gets full credit; a quote
    that was supplied but doesn't match anything in the text (paraphrase or
    hallucination) gets partial credit; NONE gets zero."""
    if ans is None:
        return None
    a = str(ans).strip()
    if not a or re.match(r"^\W*none\b", a, re.I):
        return 0.0
    quote = a.strip('"“”\' ')
    words = quote.split()
    if len(words) < 2:
        return 0.3
    probe = " ".join(words[:6])
    probe_n, doc_n = _norm_snippet(probe), _norm_snippet(doc_text)
    if probe_n and probe_n in doc_n:
        return 1.0
    return 0.35


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def score(text: str, extracted: dict, ops) -> float:
    try:
        t = _prep(text, ops)
        if len(t.strip()) < 40:
            return 0.15
        base, _f = _code_score(t)

        extracted = extracted or {}
        rating = _parse_rating(extracted.get("cadence_quality", ""))
        errs = _parse_errors(extracted.get("prose_errors"))

        val = base
        if rating is not None:
            # LLM grounds the tacit cadence judgment; code keeps the anchor.
            val = 0.45 * base + 0.55 * (0.15 + 0.7 * rating)
        if errs is not None and errs > 0:
            val -= 0.12 * errs

        # --- b4: does rhythm actually track action/emotion? ---
        rhythm = _parse_rhythm_match(extracted.get("rhythm_matches_content"))
        if rhythm is not None:
            val = 0.85 * val + 0.15 * rhythm

        # --- b4: grounded memorable-line bonus ---
        mem = _parse_memorable(extracted.get("memorable_line"), t)
        if mem is not None and mem > 0:
            val += 0.06 * mem

        return _clip01(val)
    except Exception:
        return 0.5
