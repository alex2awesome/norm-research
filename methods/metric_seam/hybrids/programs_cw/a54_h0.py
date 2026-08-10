"""Hybrid metric channel for a54: Clarity and restraint; avoid affectation.

Construct read from train residuals: the judge rewards plain, proportioned,
dialogue-forward prose that is mechanically clean; it punishes (a) showy
elevation / purple diction / ornamental simile / narratorial exclamation and
(b) loss of control (typos, run-ons, lowercase-i, spacing errors).
Keyword topic is irrelevant; register is the signal. The register itself is
tacit, so one categorical LLM field names it; the predicate stays in code.
"""

import re
import math

LLM_FIELDS = {
    "register": ("Classify this story's narration prose style with ONE word: "
                 "plain, conversational, ornate, purple, or erudite."),
    "mechanics": ("Count the story's spelling/grammar/typo errors; "
                  "answer with ONE word: clean, few, or many."),
}

# --- class-level ornate/purple lexicon (poetic-Latinate register markers) ---
_PURPLE = {
    "nonchalantly", "chagrin", "tempestuous", "melancholia", "melancholy",
    "pallor", "pallid", "alabaster", "ethereal", "mesmerizing", "mesmerized",
    "alluring", "enchanting", "resplendent", "luminescent", "luminous",
    "iridescent", "incandescent", "euphoria", "euphoric", "visage", "tresses",
    "cacophony", "symphony", "myriad", "plethora", "wondrous", "magnificence",
    "brilliance", "radiating", "radiant", "gossamer", "celestial", "ephemeral",
    "ineffable", "azure", "obsidian", "amethyst", "amythest", "sanguine",
    "verdant", "halcyon", "effervescent", "mellifluous", "penumbra",
    "eldritch", "grandiose", "opulent", "sumptuous", "exquisite",
    "transcendent", "seraphic", "immaculate", "crystalline", "shimmering",
    "glistening", "cascading", "undulating", "beguiling", "bewitching",
    "intoxicating", "rapturous", "ecstasy", "reverie", "languid", "lithe",
    "dainty", "porcelain", "demigods", "meteoric", "engulfed", "surging",
    "excruciating", "insidious", "precariously", "foretold", "heresy",
    "vagabond", "bastion", "bereft", "brethren", "visceral", "primordial",
}

_LY_STOP = {
    "only", "family", "reply", "supply", "belly", "jelly", "silly", "ugly",
    "holy", "early", "likely", "lonely", "lovely", "friendly", "deadly",
    "weekly", "daily", "july", "fly", "apply", "rely", "ally", "bully",
    "folly", "assembly", "monopoly", "italy", "really", "probably",
    "actually", "finally", "usually", "nearly", "barely", "hardly",
    "exactly", "definitely", "obviously", "seriously", "honestly",
}

_QUOTE_RE = re.compile(r'"[^"]{1,600}"|“[^”]{1,600}”')
_CHAT_RE = re.compile(r"^\s*\*?[A-Z][\w@'. -]{1,28}:\s", re.M)
_WORD_RE = re.compile(r"[A-Za-z']+")


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _good(rate, lo, hi):
    """1.0 when rate<=lo (restrained/clean), 0.0 when rate>=hi."""
    if hi <= lo:
        return 0.5
    return _clamp01(1.0 - (rate - lo) / (hi - lo))


def _strip_chrome(text):
    # markdown / scrape artifacts; keep words intact
    text = re.sub(r"&(amp|gt|lt|nbsp|#x200b);", " ", text, flags=re.I)
    text = text.replace("*", " ").replace("&", " and ")
    text = re.sub(r"\[[^\]]{0,40}\]\([^)]{0,300}\)", " ", text)  # md links
    text = re.sub(r"https?://\S+", " ", text)
    # trailing author-note chrome
    text = re.sub(r"(?im)^\s*(edit\s*:|update\s*:|/?r/\w+|p\.?s\.?\b).*$", " ", text)
    return text


def _split_dialogue(text):
    """Return (narration_text, nonchat_text, dialogue_char_fraction)."""
    total = max(1, len(text))
    dia = 0
    for m in _QUOTE_RE.finditer(text):
        dia += len(m.group(0))
    # placeholder keeps attribution fragments ("... he said") from looking
    # like lowercase sentence starts after quote removal
    narration = _QUOTE_RE.sub(" Q ", text)
    # chat-log / screenplay style lines count as dialogue too
    kept = []
    for line in narration.splitlines():
        if _CHAT_RE.match(line):
            dia += len(line)
        else:
            kept.append(line)
    narration = "\n".join(kept)
    nonchat = "\n".join(l for l in text.splitlines() if not _CHAT_RE.match(l))
    return narration, nonchat, _clamp01(dia / total)


def _verse_frac(narration):
    """Fraction of narration lines that look like verse (short, unpunctuated)."""
    lines = [l.strip() for l in narration.splitlines()]
    lines = [l for l in lines if len(l) > 2 and l[0] not in "\"'“‘*[>-"]
    if len(lines) < 6:
        return 0.0
    versey = sum(1 for l in lines
                 if len(l) <= 60 and 2 <= len(l.split()) <= 9
                 and not l.rstrip().endswith((".", "!", "?", ":", ",", ";")))
    return versey / len(lines)


def _code_score(text, ops):
    try:
        text = ops.normalize(text)
    except Exception:
        pass
    raw = text
    text = _strip_chrome(text)
    narration, nonchat, dia_frac = _split_dialogue(text)

    nwords = [w.lower() for w in _WORD_RE.findall(narration)]
    n_narr = max(30, len(nwords))

    # 1) ornate diction rate (per 1000 narration words)
    purple_hits = sum(1 for w in nwords if w in _PURPLE)
    purple = _good(1000.0 * purple_hits / n_narr, 0.5, 6.0)

    # 2) narratorial exclamation rate
    excl = _good(1000.0 * narration.count("!") / n_narr, 1.0, 14.0)

    # 3) ornamental simile rate in narration
    narr_low = narration.lower()
    sim_hits = len(re.findall(
        r"\blike (?:a|an|the|some)\b|\bmuch like\b"
        r"|\blike \w+ing\b|\bsimilar to\b|\bas \w+ as\b", narr_low))
    sim_hits += len(re.findall(r"\b[Ll]ike [A-Z][a-z]", narration))
    simile = _good(1000.0 * sim_hits / n_narr, 1.2, 7.0)

    # 4) -ly adverb density in narration
    adv_hits = sum(1 for w in nwords
                   if len(w) > 4 and w.endswith("ly") and w not in _LY_STOP)
    adverb = _good(1000.0 * adv_hits / n_narr, 6.0, 30.0)

    # 5) mechanics in NARRATION only (typos inside dialogue/chat are voice):
    #    lowercase-i pronoun, space-before-punct, glued sentences,
    #    lowercase sentence starts
    lc_i = len(re.findall(r"(?:^|[\s\"“(])i(?=[\s'’,.])", narration))
    sp_punct = len(re.findall(r"\s[,.](?:\s|$)", narration))
    glued = len(re.findall(r"[a-z]{2}[.,][A-Za-z]{2}", narration))
    lc_start = len(re.findall(r"[.!?]\s+[a-z]", narration))
    # confusable-grammar errors: authorial, so count in dialogue too (not chat)
    confus = len(re.findall(
        r"\b(?:alot|could of|would of|should of|must of|to late|to many"
        r"|to much|you was|we was|they was|he don'?t|she don'?t|it don'?t"
        r"|(?:i|you|we|they|she|he)(?: never| always)? (?:been|seen|done))\b",
        nonchat.lower()))
    err_rate = 1000.0 * (2.0 * lc_i + sp_punct + glued + lc_start +
                         2.0 * confus) / n_narr
    clean = _good(err_rate, 1.0, 20.0)

    # 6) sustained ALL-CAPS shouting (in prose or quoted dialogue, not chat)
    caps_words = sum(len(m.group(0).split())
                     for m in re.finditer(r"\b[A-Z]{3,}(?:[\s,!?']+[A-Z]{2,}\b)+",
                                          nonchat))
    n_nonchat = max(30, len(_WORD_RE.findall(nonchat)))
    caps = _good(1000.0 * caps_words / n_nonchat, 4.0, 45.0)

    # 7) verse/doggerel layout (short unpunctuated lines) reads as display
    verse = _good(_verse_frac(narration), 0.25, 0.60)

    # 8) plain-showing: dialogue/chat fraction (helps, absence not fatal)
    dialogue = _clamp01(dia_frac / 0.55)

    # 9) proportion: sentence length + long-word load (baseline's weak signal)
    try:
        _, mean_wps, frac_long = ops.sent_stats(raw)
        struct = 0.5 * _good(mean_wps, 20.0, 45.0) + 0.5 * _good(frac_long, 0.18, 0.35)
    except Exception:
        struct = 0.5

    core = (0.21 * purple + 0.12 * excl + 0.12 * simile + 0.08 * adverb +
            0.14 * clean + 0.05 * caps + 0.12 * verse + 0.11 * dialogue +
            0.05 * struct)
    # severe clarity failure gates the whole channel (multiplicative)
    return _clamp01(core * (0.55 + 0.45 * clean))


_REG_MAP = [
    ("purple", 0.05), ("erudite", 0.15), ("ornate", 0.30),
    ("conversational", 0.90), ("plain", 1.00),
]
_MECH_MAP = [("none", 1.00), ("many", 0.10), ("few", 0.55)]


def _map_field(ans, table):
    if not isinstance(ans, str) or not ans.strip():
        return None
    low = ans.lower()
    for key, val in table:
        if re.search(r"\b" + key + r"\b", low):
            return val
    return None


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or len(text.strip()) < 40:
            return 0.5
        code = _code_score(text, ops)
        reg = _map_field((extracted or {}).get("register", ""), _REG_MAP)
        mech = _map_field((extracted or {}).get("mechanics", ""), _MECH_MAP)
        parts = [(code, 0.45)]
        if reg is not None:
            parts.append((reg, 0.40))
        if mech is not None:
            parts.append((mech, 0.15))
        tot = sum(w for _, w in parts)
        return _clamp01(sum(v * w for v, w in parts) / tot)
    except Exception:
        return 0.5
