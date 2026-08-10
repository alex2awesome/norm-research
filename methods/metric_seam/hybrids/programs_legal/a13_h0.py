"""a13 hybrid: Constructive discharge (Title VII).

Criterion: intolerable working conditions the employer deliberately created,
such that a reasonable person would feel compelled to resign (Suders) — a
higher bar than ordinary hostile environment. Requires severe/sustained
mistreatment, a humiliating demotion, or an ultimatum immediately preceding
the resignation, as distinct from a voluntary departure for an unrelated
reason.

Baseline (v0_keyword) counts a fixed resignation-lexicon ('resigned', 'quit',
'intolerable', 'ultimatum', ...) and normalizes hit-count. That lexicon still
catches the clean explicit cases, so we KEEP it as a code-side backstop. But
it has two known failure modes visible in the train set:

  1. False positives: plaintiffs who "resigned"/"quit" for a reason plainly
     UNRELATED to the alleged mistreatment (e.g. left to take a better-paying
     job elsewhere) still trip the lexicon and score high, even though the
     judge scores these near 0.
  2. False negatives: genuine severe/sustained mistreatment (physical
     harassment, retaliatory demotion + pay cut, drastic hours cut, a real
     "resign or violate your beliefs" ultimatum) that doesn't happen to use
     the baseline's exact phrasing scores 0 under the keyword lexicon even
     though the judge scores these mid-to-high.

Per the corpus notes: legal-TERM presence (including the parties' own
"constructive discharge" label) is a weak proxy for the construct being
FACTUALLY present — a case can literally use the phrase while the surrounding
text is conclusory/vague (thin at the motion-to-dismiss stage), and a case can
show the real fact pattern without ever using the term. So the LLM fields ask
for the underlying FACT (a specific triggering event, or an unrelated reason),
not just term detection — and code keeps: the resignation lexicon backstop,
a severity-breadth counter (distinct types of alleged mistreatment), and a
structural proximity check (a severity indicator appearing textually close to
a resignation mention), which operationalizes the doctrine's "immediately
preceding" requirement without needing an LLM to judge it.
"""
import re

LLM_FIELDS = {
    "resign_trigger": (
        "In <=15 words, name the SPECIFIC factual event or condition just "
        "before any resignation (not a legal label like 'hostile environment' "
        "or 'constructive discharge'); answer NONE if plaintiff never resigned "
        "or no specific triggering fact is given."
    ),
    "unrelated_reason": (
        "In <=12 words, if plaintiff resigned/quit, state any reason given "
        "that is UNRELATED to the alleged mistreatment (e.g. better-paying "
        "job, retirement); answer NONE if the resignation is attributed to "
        "the alleged mistreatment or plaintiff never resigned."
    ),
}

# --- code-side backstop: baseline resignation lexicon (proven signal) ---
_RESIGN_KWS = [
    "resigned", "resignation", "quit", "intolerable", "unbearable",
    "no choice but to", "forced to leave", "compelled to resign",
    "ultimatum", "had no other option",
]

# markers used only for the structural-proximity check below
_RESIGN_MARKERS = [r"resign\w*", r"\bquit\b", r"\bdeparted\b", r"left (her|his) job", r"left work"]

# --- code-side predicate: breadth of distinct severity-type indicators ---
_SEVERITY_PATTERNS = [
    r"demot\w*", r"suspend\w*", r"threat\w*", r"retaliat\w*", r"harass\w*",
    r"hostile\s+work\s+environment", r"humiliat\w*", r"sexual\s+advance\w*",
    r"reduc\w*\s+(her|his)?\s*(salary|pay)", r"cut\w*[\s\w']{0,15}hours",
    r"grabb\w*", r"groped", r"\bnude\b",
]

_UNRELATED_HINTS = [
    "better", "new job", "other job", "unrelated", "retire", "personal reason",
    "voluntar", "pursue other", "accept a position", "accepted a position",
    "higher pay", "better-paying", "better paying",
]

_COERCION_HINTS = [
    "ultimatum", "forced", "no choice", "intoler", "compell", "coerc",
    "demot", "harass", "hostile", "threat", "reduc", "cut", "suspend", "retaliat",
]

_NONE_ANSWERS = {"", "none", "n/a", "na", "not stated", "not mentioned", "unknown", "not applicable"}


def _clean(v):
    v = (v or "").strip().lower()
    return "" if v in _NONE_ANSWERS else v


def score(text: str, extracted: dict, ops) -> float:
    try:
        norm = ops.normalize(text) if text else ""
        t = norm.lower()

        # code layer 1: resignation lexicon (baseline's proven signal)
        lex_hits = sum(1 for k in _RESIGN_KWS if k in t)
        lex_score = min(1.0, lex_hits / 4.0)

        # code layer 2: breadth of distinct severity-type indicators
        sev_matches = [(m.start(), pat) for pat in _SEVERITY_PATTERNS for m in re.finditer(pat, t)]
        distinct_sev = len({pat for _, pat in sev_matches})
        sev_score = min(1.0, distinct_sev / 5.0)

        # code layer 3: structural proximity — a severity marker sitting close
        # to a resignation mention operationalizes "immediately preceding"
        resign_positions = [m.start() for pat in _RESIGN_MARKERS for m in re.finditer(pat, t)]
        proximity = 0.0
        if resign_positions and sev_matches:
            for r in resign_positions:
                if any(0 <= r - s <= 400 for s, _ in sev_matches):
                    proximity = 1.0
                    break

        code_score = 0.45 * lex_score + 0.35 * sev_score + 0.20 * proximity
        code_score = max(0.0, min(1.0, code_score))

        # LLM layer: thick-input grounding for the construct itself
        ext = extracted or {}
        trigger = _clean(ext.get("resign_trigger"))
        unrelated = _clean(ext.get("unrelated_reason"))

        has_trigger = bool(trigger)
        coercive_confirmed = has_trigger and any(h in trigger for h in _COERCION_HINTS)
        unrelated_confirmed = bool(unrelated) and any(h in unrelated for h in _UNRELATED_HINTS)

        result = code_score
        if unrelated_confirmed:
            # plaintiff's own departure reason is unrelated to the alleged
            # mistreatment — cap low regardless of keyword/severity hits
            result = min(result, 0.15)
        elif coercive_confirmed:
            # LLM independently names a specific coercive trigger fact —
            # raise the floor even if code-side signal was weak
            result = max(result, 0.55)
        elif not has_trigger:
            # no resignation, or no specific fact behind it (conclusory/vague
            # legal-label-only language) — cap moderate-low
            result = min(result, 0.2)

        return max(0.0, min(1.0, result))
    except Exception:
        return 0.5
