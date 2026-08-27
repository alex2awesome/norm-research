"""Hybrid metric channel for CW criterion a0: "Spectacle Subordinate to Plot".

Judge construct: pity/fear should arise from the causal structure of the plot
(would move even in reading); visual/sensational display must not replace
poetic construction.

Train-residual analysis (30 stratified examples):
  - All judge=1.0 texts are compact, dialogue-driven or epistolary pieces whose
    payoff is a causally set-up reveal.
  - Low scorers (0.0-0.2) rely on displayed spectacle (gore, action set-pieces,
    gross-out, crude shock), carry meta-chrome (Edit notes, feedback requests,
    serial links), or show broken craft (whitespace walls, lowercase-i pronoun).
  - Baseline causal-connective keywords are pure noise (train rho ~ -0.01).

Design: keep the predicate in code (dialogue share, sensational-display
density, chrome/craft penalties); use two SHORT categorical LLM fields for the
tacit constructs a regex cannot see (does impact rest on spectacle? does the
ending land as a causal consequence?). Categorical YES/SOMEWHAT/NO answers are
robustly answerable by a mid-size model in <=20 words; parsing is fail-open to
a neutral 0.5.
"""

import re

LLM_FIELDS = {
    "spectacle_reliance": (
        "Does the story's impact rely mainly on shocking imagery, gore, action "
        "set-pieces, or gross-out display rather than its plot? Answer YES, "
        "SOMEWHAT, or NO."
    ),
    "causal_payoff": (
        "Does the ending land as a consequence or reveal that was causally set "
        "up earlier in the story? Answer YES, PARTIAL, or NO."
    ),
}

# --- sensational-display lexicon (spectacle substituting for plot) ---------
# Deliberately EXCLUDES abstract death/kill/murder/blade vocabulary: reflective
# or reported violence (letters, dialogue about death) is not staged spectacle.
_SPECTACLE_RE = re.compile(
    r"\b("
    # displayed violence / action set-piece
    r"explos\w*|explod\w*|detonat\w*|grenade\w*|rifle\w*|shotgun\w*|gunfire|"
    r"missile\w*|warhead\w*|sever(?:ed|ing)|gore|gory|blood\w*|scream\w*|"
    r"shriek\w*|flame\w*|blaz(?:e|ing)\w*|burn(?:ing|ed|t)|scorch\w*|"
    r"charred|corpse\w*|siren\w*|roar(?:ed|ing|s)?|slaughter\w*|stabbed|"
    # gross-out display
    r"vomit\w*|threw up|puk(?:e|ed|ing)|piss\w*|feces|rot(?:ting|ted|s)|"
    r"stench|maggot\w*|roach\w*|cockroach\w*|gag(?:ged|ging|s)|"
    # crude shock display
    r"masturbat\w*|phallus|semen|fart\w*"
    r")\b",
    re.IGNORECASE,
)

# --- meta-chrome / author-note markers (only ever seen on judge <= 0.4) ----
_CHROME_PATTERNS = [
    re.compile(r"(?im)^\s*\**\s*(?:final\s+)*edit\s*:"),
    re.compile(r"(?i)\bword count\b"),
    re.compile(r"(?i)let me know what you think"),
    re.compile(r"(?i)thanks for reading"),
    re.compile(r"(?i)constructive criticism"),
    re.compile(r"(?i)feedback is (?:always )?(?:welcome|appreciated)"),
    re.compile(r"(?i)subscribe to"),
    re.compile(r"(?i)\bpart \d+\b[^\n]*http"),
    re.compile(r"(?i)my first post"),
]

# dialogue spans: curly/straight double quotes, curly single quotes
_QUOTE_PAIRS = [("“", "”"), ('"', '"'), ("‘", "’")]
# chat-fiction line: "Speaker Name: utterance"
_CHAT_LINE_RE = re.compile(r"^[A-Z][A-Za-z .'\-]{1,24}:\s")
# lowercase standalone pronoun "i" (craft error)
_LOWER_I_RE = re.compile(r"(?<![A-Za-z])i(?![A-Za-z])")


def _parse_categorical(ans, positive, middle, negative):
    """Map a short free-text answer onto {positive:1.0, middle:0.5-ish, negative:0.0}.

    Fail-open: unparseable or empty -> None (caller substitutes neutral).
    """
    try:
        a = (ans or "").strip().lower()
        if not a:
            return None
        head = a[:60]
        # middle tokens first (they often embed 'yes'/'no' variants)
        for tok in middle:
            if tok in head:
                return 0.5
        neg_hit = re.search(r"\b(" + "|".join(negative) + r")\b", head)
        pos_hit = re.search(r"\b(" + "|".join(positive) + r")\b", head)
        if neg_hit and pos_hit:
            return 0.5
        if pos_hit:
            return 1.0
        if neg_hit:
            return 0.0
        return None
    except Exception:
        return None


def _dialogue_fraction(text):
    """Fraction of characters that sit inside dialogue (quotes or chat lines)."""
    total = max(1, len(text))
    quoted = 0
    for op, cl in _QUOTE_PAIRS:
        i = 0
        while True:
            s = text.find(op, i)
            if s < 0:
                break
            e = text.find(cl, s + 1)
            if e < 0:
                break
            span = e - s
            if 1 < span < 600:  # ignore degenerate/unclosed spans
                quoted += span
            i = e + 1
    chat_chars = 0
    n_lines = 0
    chat_lines = 0
    for line in text.split("\n"):
        if not line.strip():
            continue
        n_lines += 1
        if _CHAT_LINE_RE.match(line.strip()):
            chat_lines += 1
            chat_chars += len(line)
    frac = min(1.0, (quoted + chat_chars) / total)
    chat_frac = chat_lines / max(1, n_lines)
    return frac, chat_frac


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.5
        try:
            t = ops.normalize(text)
            if not t or not isinstance(t, str):
                t = text
        except Exception:
            t = text

        words = max(1, len(re.findall(r"\w+", t)))

        # ---- code channel -------------------------------------------------
        dlg_frac, chat_frac = _dialogue_fraction(t)
        # saturate: >=55% of chars in dialogue counts as fully dialogue-driven
        dialogue_component = min(1.0, dlg_frac / 0.55)

        # sensational-display density (per 1000 words), saturating penalty
        spec_hits = len(_SPECTACLE_RE.findall(t))
        per1000 = spec_hits * 1000.0 / words
        spectacle_pen = 0.30 * min(1.0, per1000 / 12.0)

        # meta-chrome penalty
        chrome_hits = sum(1 for p in _CHROME_PATTERNS if p.search(t))
        chrome_pen = min(0.18, 0.09 * chrome_hits)

        # formatting mess: walls of blank lines
        mess_pen = 0.0
        runs = re.findall(r"\n{3,}", t)
        if runs:
            longest = max(len(r) for r in runs)
            if longest >= 5:
                mess_pen += 0.10 * min(1.0, (longest - 4) / 8.0)
        # raw html-escape blockquote litter
        if t.count("&gt;") >= 3:
            mess_pen += 0.04

        # craft error: lowercase standalone "i" (skip chat-register pieces)
        i_pen = 0.0
        if chat_frac <= 0.25:
            c = len(_LOWER_I_RE.findall(t))
            if c > 2:
                i_pen = 0.08 * min(1.0, (c - 2) / 6.0)

        # ---- LLM channel (thick-input grounding; neutral 0.5 fallback) ----
        ex = extracted if isinstance(extracted, dict) else {}
        spec_ans = _parse_categorical(
            ex.get("spectacle_reliance", ""),
            positive=["no", "not", "none"],
            middle=["somewhat", "partial", "partially", "mixed", "mostly"],
            negative=["yes"],
        )
        # positive == "NO, does not rely on spectacle" -> good (1.0)
        spec_val = 0.5 if spec_ans is None else spec_ans

        causal_ans = _parse_categorical(
            ex.get("causal_payoff", ""),
            positive=["yes"],
            middle=["partial", "partially", "somewhat", "mixed", "mostly"],
            negative=["no", "not", "none"],
        )
        causal_val = 0.5 if causal_ans is None else causal_ans

        # ---- blend --------------------------------------------------------
        out = (
            0.16
            + 0.26 * spec_val
            + 0.24 * causal_val
            + 0.24 * dialogue_component
            - spectacle_pen
            - chrome_pen
            - mess_pen
            - i_pen
        )
        return float(max(0.0, min(1.0, out)))
    except Exception:
        return 0.5
