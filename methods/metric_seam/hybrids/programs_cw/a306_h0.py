"""a306_h0: hybrid channel for CW criterion a306 "Tension and momentum management".

Train-residual study (30 stratified examples, baseline v2_holistic rho=0.057):
the judge responds to ARC SHAPE, not surface tension keywords. High-judge
stories (0.7-0.9) are in-scene, escalate beat by beat, and end on a
twist/cliffhanger/peak (d01869, d00779, d00682, d01219). Low-judge stories
are digressive rambles (d04046), reportage parodies (d00009), poems (d01355),
single-blob travelogues (d02932), or fizzle-out vignettes, frequently carrying
apologetic author notes ("sorry for bad formatting", "my first post").

Design: two small-enum LLM fields carry the tacit construct (tension arc,
ending type) since Gemma answers a closed one-word choice far more reliably
than an open judgment; code keeps the predicate: enum parsing/mapping,
paragraph-rhythm "punch" signal, and robust class-level dampers (poem layout,
very short texts, single-blob monotone, apologetic meta-notes). Meta-note
penalty is deliberately small because one 0.7 example also apologizes.
retrieve_similar is unused: topical neighbors carry no craft signal here and
the judge scores craft, not topic.
"""

import re
import math
import statistics

LLM_FIELDS = {
    "tension_arc": (
        "Does the story's dramatic tension escalate through rising complications to a peak "
        "(BUILDS), stay static or reportorial (FLAT), or get smothered by digression or summary "
        "(DEFLATES)? Answer exactly one word: BUILDS, FLAT, or DEFLATES."
    ),
    "ending_hook": (
        "Ignoring any author notes, apologies, or links at the very end, how does the story "
        "finish: TWIST, CLIFFHANGER, ESCALATION, RESOLUTION, or FIZZLE? "
        "Answer that one word plus at most five evidence words."
    ),
}

# ---------------------------------------------------------------- enum maps
# Ordered substring match: first hit wins (CLIFF before FLAT etc.).
_ARC_TABLE = (
    ("BUILD", 0.95),
    ("ESCALAT", 0.95),   # tolerated synonym
    ("DEFLAT", 0.05),
    ("MEANDER", 0.20),
    ("STATIC", 0.30),
    ("FLAT", 0.30),
)
_END_TABLE = (
    ("CLIFF", 0.95),
    ("TWIST", 0.95),
    ("REVEAL", 0.90),    # tolerated synonym
    ("ESCALAT", 0.80),
    ("RESOLUT", 0.45),
    ("RESOLVE", 0.45),
    ("FIZZLE", 0.12),
    ("QUIET", 0.30),
    ("FLAT", 0.20),
)

_META_RES = [
    r"sorry for (?:the )?(?:bad|poor|crappy|awful) ?(?:formatting|grammar|spelling)",
    r"don'?t judge(?: my)? typos",
    r"my first (?:post|story|submission|prompt|time (?:posting|writing))",
    r"(?:any |all )?feedback (?:is )?(?:very )?welcome",
    r"let me know what you think",
    r"(?:wrote|writing|written) (?:this )?on (?:my )?(?:phone|mobile)",
    r"\bon mobile\b",
    r"still crap\b",
    r"^\s*edit ?\d* ?:",
    r"had fun with th(?:at|is) prompt",
    r"subscribe to /?r/",
    r"/u/\w+",
    r"part \d(?:-\d)? (?:is )?(?:now )?(?:up|in (?:the )?comments)",
    r"i gotta contribute",
]
_META_PAT = re.compile("|".join("(?:%s)" % p for p in _META_RES),
                       re.IGNORECASE | re.MULTILINE)
_HEAD_ECHO_PAT = re.compile(r"^\s*\\?\[(?:wp|poem|mc|sp|eu|cw)\b", re.IGNORECASE)

_WORD_RE = re.compile(r"[A-Za-z']+")
_TERMINAL_RE = re.compile(r"""[.!?:;,"'”’…)\]*~-]\s*$""")


def _enum_score(answer, table, default=0.5):
    a = (answer or "").strip().upper()
    if not a or a == "NONE":
        return default
    for key, val in table:
        if key in a:
            return val
    return default


def _words(s):
    return _WORD_RE.findall(s or "")


def _paragraphs(t):
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in t.split("\n") if p.strip()]
    return paras


def _clean(text, ops):
    t = text or ""
    try:
        t2 = ops.normalize(t)
        if t2:
            t = t2
    except Exception:
        pass
    t = t.replace("&gt;", " ").replace("&lt;", " ").replace("&amp;", "&")
    t = re.sub(r"&#x?[0-9a-fA-F]+;", " ", t)
    t = re.sub(r"\*\*?", "", t)
    return t


def score(text, extracted, ops):
    try:
        t = _clean(text, ops)
        w = _words(t)
        n_words = len(w)
        if n_words < 10:
            return 0.0

        # ---------------- LLM part: arc shape + ending type -------------
        arc = _enum_score((extracted or {}).get("tension_arc", ""), _ARC_TABLE)
        end = _enum_score((extracted or {}).get("ending_hook", ""), _END_TABLE)
        llm_part = 0.55 * arc + 0.45 * end

        # ---------------- code part: paragraph punch rhythm -------------
        paras = _paragraphs(t)
        plens = [len(_words(p)) for p in paras] or [0]
        mean_p = statistics.fmean(plens) if plens else 0.0
        cv = (statistics.pstdev(plens) / (mean_p + 1.0)) if len(plens) > 1 else 0.0
        n_punch = sum(1 for p, L in zip(paras, plens)
                      if 1 <= L <= 7 and _TERMINAL_RE.search(p))
        punch_ratio = min(1.0, n_punch / 5.0)
        rhythm = 0.55 * min(1.0, cv / 1.1) + 0.45 * punch_ratio
        # pure talking-heads (every paragraph quoted) rarely grounds tension
        n_dialog = sum(1 for p in paras if '"' in p or "“" in p)
        if paras and (n_dialog / len(paras)) > 0.90:
            rhythm *= 0.75
        code_part = max(0.0, min(1.0, rhythm))

        s = 0.65 * llm_part + 0.35 * code_part

        # ---------------- robust class-level dampers ---------------------
        # apologetic / meta author notes (small: one 0.7 train ex has one)
        n_meta = len(_META_PAT.findall(t))
        if _HEAD_ECHO_PAT.search(t[:400]):
            n_meta += 1
        if n_meta:
            s -= min(0.10, 0.05 * n_meta)

        # poem layout: many short unpunctuated lines (verse, not wrapped prose)
        lines = [ln.strip() for ln in t.split("\n") if len(_words(ln)) >= 3]
        if len(lines) >= 8:
            line_lens = [len(_words(ln)) for ln in lines]
            no_punct = sum(1 for ln in lines if not _TERMINAL_RE.search(ln))
            if (no_punct / len(lines)) > 0.5 and statistics.fmean(line_lens) <= 11:
                s = min(s, 0.35)

        # very short text: tension cannot build a page-turn impulse
        if n_words < 180:
            s = 0.7 * s + 0.3 * 0.28

        # single-blob monotone pacing
        if len(paras) <= 2 and n_words > 350:
            s -= 0.07

        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
