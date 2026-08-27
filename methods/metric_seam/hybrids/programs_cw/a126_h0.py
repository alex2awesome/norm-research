"""Hybrid metric channel for a126: Mode and scene/summary balance and transitions.

Insight from train residuals: the judge rewards DELIBERATE mode control, not
keyword co-presence (baseline's failure).  High scorers commit to a mode --
pure dialogue scene, document forms (letter / email / chat log / news item /
chronicle), or controlled lyrical summary.  Low scorers ramble in retrospective
"let me tell you how it all started" summary with lazy connectives ("anyway",
"eventually", "it turns out"), ellipsis/exclamation-heavy telling, and
wall-of-text paragraphs.  Code detects commitment + amateur-telling markers;
two LLM fields ground the tacit part (deliberateness, transition smoothness).
"""

import re
import statistics

LLM_FIELDS = {
    "mode_control": (
        "Does the story control scene-vs-summary mode deliberately (committed "
        "scene, document/letter/chat format, or stylized chronicle) or drift "
        "between telling and showing? Answer one word: deliberate, mixed, or drifting."
    ),
    "transition_quality": (
        "Are the story's shifts in time or place handled smoothly and "
        "purposefully, or abruptly and confusingly? Answer one word: smooth, "
        "adequate, or jarring."
    ),
}

# ---------------------------------------------------------------------------

_CHROME_RE = re.compile(
    r"^\s*\**\s*(final\s+)*(edit|edit\s*\d*|updated?)\s*\**\s*[:\-]", re.I)

_HEADER_LINE_RE = re.compile(
    r"^\s*[*\[]{0,2}[A-Z][A-Za-z0-9 .'’\-]{0,30}\s*:\s+\S")
_STAGE_LINE_RE = re.compile(r"^\s*\*[^*]{4,90}\*\s*$")
_DOC_MARK_RE = re.compile(
    r"(\(AP\)|\(Reuters\)|^\s*\**\s*(To|From|Subject|CC)\s*\**\s*:"
    r"|^\s*Entry\s+\d|^\s*Accessing\b|has joined|has disconnected)",
    re.I | re.M)

_QUOTE_RE = re.compile(r"[\"“”]")

_LAZY_RE = re.compile(
    r"\b(anyways?|eventually|suddenly|one day|before long|"
    r"it all (started|began)|little did|(it )?turn(s|ed) out|"
    r"needless to say|to make a long story short)\b", re.I)

_OPENER_RE = re.compile(
    r"(ever since i|it all (started|began)|this is the story|"
    r"i (do not|don'?t) remember|i remember (how|when|saying)|"
    r"boy have i|let me tell you|i had been|have i had)", re.I)

_ELLIPSIS_RE = re.compile(r"\.{2,}|…")


def _paragraphs(t):
    paras = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
    if len(paras) <= 2 and t.count("\n") > 6:
        paras = [p for p in t.split("\n") if p.strip()]
    return paras


def _strip_chrome(t):
    # drop trailing author-note lines (Edit:, subreddit plugs) near the end
    lines = t.split("\n")
    cut = len(lines)
    for i in range(len(lines) - 1, max(0, int(len(lines) * 0.7)) - 1, -1):
        s = lines[i].strip()
        if not s:
            continue
        if _CHROME_RE.match(s) or re.match(r"^r/\w+$", s) or re.match(
                r"^[-~=_*]{3,}\s*$", s):
            cut = i
        else:
            break
    return "\n".join(lines[:cut])


def _code_score(text, ops):
    try:
        t = ops.normalize(text)
        if not isinstance(t, str) or not t.strip():
            t = text
    except Exception:
        t = text
    if not isinstance(t, str) or not t.strip():
        return 0.5
    t = _strip_chrome(t)

    words = re.findall(r"\w+", t)
    nw = max(1, len(words))
    per100 = 100.0 / nw

    paras = _paragraphs(t)
    npara = max(1, len(paras))
    lines = [ln for ln in t.split("\n") if ln.strip()]
    nlines = max(1, len(lines))

    # --- mode commitment -------------------------------------------------
    dia = sum(1 for p in paras if _QUOTE_RE.search(p)) / float(npara)
    header_frac = sum(1 for ln in lines
                      if _HEADER_LINE_RE.match(ln) or _STAGE_LINE_RE.match(ln)
                      ) / float(nlines)
    is_doc = header_frac >= 0.18 or bool(_DOC_MARK_RE.search(t))

    if is_doc:
        base = 0.72                       # committed document / chronicle form
    elif dia >= 0.60:
        base = 0.66                       # committed scene with dialogue
    elif dia <= 0.10:
        base = 0.50                       # pure summary: lyrical or amateur
    else:
        base = 0.56                       # scene/summary mix

    # --- craft penalties (amateur-telling markers) -----------------------
    excl = len(re.findall(r"!", t)) * per100
    ell = len(_ELLIPSIS_RE.findall(t)) * per100
    lazy = len(_LAZY_RE.findall(t)) * per100

    pen = 0.0
    pen += min(0.14, 0.045 * excl)
    pen += min(0.12, 0.050 * ell)
    pen += min(0.12, 0.070 * lazy)
    if _OPENER_RE.search(t[:260]):
        pen += 0.07                       # retrospective "how it began" frame

    wpp = nw / float(npara)
    if wpp > 130:
        pen += 0.08                       # wall-of-text paragraphs
    try:
        _, mws, _ = ops.sent_stats(t)
        if mws and mws > 32:
            pen += 0.05                   # rambling run-on sentences
    except Exception:
        pass

    return max(0.0, min(1.0, base - pen))


_MODE_MAP = [("deliberate", 0.90), ("mixed", 0.50), ("drift", 0.12)]
_TRANS_MAP = [("smooth", 0.85), ("adequate", 0.50), ("average", 0.50),
              ("jarring", 0.15), ("abrupt", 0.15), ("confus", 0.15)]


def _map_answer(ans, table):
    if not isinstance(ans, str) or not ans.strip():
        return None
    a = ans.strip().lower()
    if a in ("none", "n/a", "unknown"):
        return None
    for key, val in table:
        if key in a:
            return val
    return None


def score(text, extracted, ops):
    try:
        code = _code_score(text, ops)
        vals = []
        try:
            m = _map_answer((extracted or {}).get("mode_control", ""), _MODE_MAP)
            if m is not None:
                vals.append(m)
            tq = _map_answer((extracted or {}).get("transition_quality", ""),
                             _TRANS_MAP)
            if tq is not None:
                vals.append(tq)
        except Exception:
            vals = []
        if vals:
            out = 0.5 * code + 0.5 * statistics.mean(vals)
        else:
            out = code
        return float(max(0.0, min(1.0, out)))
    except Exception:
        return 0.5
