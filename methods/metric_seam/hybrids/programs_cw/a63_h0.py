"""Hybrid metric channel for a63: Plot progression and turning points.

Construct: inciting incident early; escalating complications; crisis/climax;
apt denouement. Keyword-position proxies (baseline v1_structure, rho=.317)
fail on dialogue-enacted arcs (judged 0.8-0.9, baseline 0.0) and give
spurious credit to static forms (toasts, listicles, diary vignettes, one-beat
dialogue gags). Design: two LLM extraction fields ground the tacit arc facts
(which canonical stages are clearly present; whether there is a decisive
late turning point); the predicate -- stage counting, turn validation, form
penalties, narration/length shaping -- stays in code.
"""

import re
import math

LLM_FIELDS = {
    "plot_stages": (
        "Which of INCITING INCIDENT, RISING COMPLICATIONS, CLIMAX, RESOLUTION does this "
        "story clearly contain? List only the ones present; NONE if none."
    ),
    "end_turn": (
        "In at most 12 words, state the story's decisive turning point or twist; "
        "NONE if the situation never really changes."
    ),
}

# ---------------------------------------------------------------- helpers

_STAGE_PATTERNS = [
    ("incit", "setup", "opening incident"),
    ("rising", "complicat", "escalat"),
    ("climax", "crisis", "peak"),
    ("resolution", "denouement", "resolv"),
]

_NEG_TURN = re.compile(
    r"\b(none|nothing|no (?:real )?(?:change|turn|twist)|unchanged|"
    r"stays? the same|not applicable|n/?a)\b"
)

# Static, non-narrative forms (class-level genre markers, not topic words).
_STATIC_FORM = re.compile(
    r"(\bdear diary\b|\btips?\s*#\s*\d|\bstep\s+\d+\s*:|\bladies and gentlemen\b|"
    r"^\s*\[?poem\]?\s*$)",
    re.IGNORECASE | re.MULTILINE,
)

# Trailing author chrome to strip before computing stats.
_CHROME_LINE = re.compile(
    r"^\s*(edit\s*:|thanks? (?:you )?for (?:reading|the gold)|"
    r"if you (?:liked?|enjoy)|check out|part \d+\]?\(|\(end of recording\)|"
    r"r/\w+\s*$|&amp;?#?x?\w+;?\s*$)",
    re.IGNORECASE,
)

_QUOTE_SPANS = re.compile(r"\"[^\"\n]{0,600}\"|“[^”\n]{0,600}”")

_PROGRESSION = re.compile(
    r"\b(then|suddenly|until|after|before long|finally|at last|meanwhile|"
    r"moments? later|that (?:night|morning|day)|years? later|when|once|"
    r"but now|no longer|realiz|began|started|stopped|turned)\b"
)


def _clamp(x):
    return max(0.0, min(1.0, x))


def _strip_chrome(t):
    lines = t.split("\n")
    kept = [ln for ln in lines if not _CHROME_LINE.match(ln)]
    return "\n".join(kept)


def _stage_fraction(ans):
    """Fraction of the 4 canonical arc stages the extractor found."""
    a = (ans or "").lower()
    if not a.strip():
        return 0.0
    hits = 0
    for pats in _STAGE_PATTERNS:
        if any(p in a for p in pats):
            hits += 1
    if hits == 0:
        return 0.0
    return hits / 4.0


def _turn_score(ans):
    """1.0 for a substantive decisive turn, 0 for NONE/negation, 0.6 thin."""
    a = (ans or "").strip().lower()
    if not a:
        return 0.0
    if _NEG_TURN.search(a) and len(a.split()) <= 6:
        return 0.0
    if _NEG_TURN.search(a):
        return 0.2
    return 1.0 if len(a.split()) >= 3 else 0.6


def _narration_words(t):
    """Words outside quoted dialogue spans."""
    stripped = _QUOTE_SPANS.sub(" ", t)
    return len(re.findall(r"[A-Za-z']+", stripped))


# ---------------------------------------------------------------- score

def score(text: str, extracted: dict, ops) -> float:
    try:
        t = text or ""
        try:
            t = ops.normalize(t)
        except Exception:
            pass
        if not t.strip():
            return 0.0

        body = _strip_chrome(t)
        words = re.findall(r"[A-Za-z']+", body)
        wc = len(words)
        if wc < 15:
            return 0.05

        # ---- code-side structural features -------------------------------
        paras = [p for p in re.split(r"\n\s*\n|\n\s*\*{3,}\s*\n", body) if p.strip()]
        para_score = min(1.0, len(paras) / 6.0)

        prog_hits = len(_PROGRESSION.findall(body.lower()))
        prog_density = prog_hits / max(1.0, wc / 100.0)  # per 100 words
        prog_score = min(1.0, prog_density / 3.0)

        narr_code = 0.5 * para_score + 0.5 * prog_score
        len_score = min(1.0, wc / 250.0)

        try:
            n_sent, mean_wps, _ = ops.sent_stats(body)
        except Exception:
            n_sent, mean_wps = max(1, wc // 15), 15.0
        # Degenerate fragmenting (mojibake / hard-wrapped chaff) or one-block wall.
        shape_ok = 1.0 if (2.0 <= float(mean_wps) <= 45.0 and int(n_sent) >= 3) else 0.6

        # ---- LLM-grounded arc facts (predicate in code) -------------------
        has_llm = isinstance(extracted, dict) and (
            "plot_stages" in extracted or "end_turn" in extracted
        )
        if has_llm:
            stage_frac = _stage_fraction(extracted.get("plot_stages", ""))
            turn = _turn_score(extracted.get("end_turn", ""))
            core = (
                0.50 * stage_frac
                + 0.25 * turn
                + 0.15 * narr_code
                + 0.10 * len_score
            )
        else:
            core = 0.5 * narr_code + 0.3 * len_score + 0.2 * prog_score

        core *= shape_ok

        # ---- class-level form penalties -----------------------------------
        if _STATIC_FORM.search(body):
            core *= 0.45  # toast / listicle / diary: static genre, arc rare
        if _narration_words(body) < 25:
            core *= 0.55  # pure-dialogue gag scene: no narrated progression
        if wc < 130:
            core *= 0.60  # one-beat vignettes seldom carry a full arc
        elif wc < 250:
            core *= 0.85

        return _clamp(core)
    except Exception:
        return 0.5
