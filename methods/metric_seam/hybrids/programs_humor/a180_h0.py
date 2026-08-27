"""
Hybrid channel for aspect a180: "Australian humor conventions"
(larrikin anti-authoritarianism, dry/irreverent tone, vernacular slang,
cartooning prominence).

Design rationale (blind, from the pack only):
The v0_keyword baseline (train rho 0.14) scores by raw substring counts of a
short slang list, which both (a) fires on accidental substrings inside
unrelated words (e.g. "oi" inside "poisonous"), and (b) rewards mere topic
vocabulary rather than the actual comedic CONVENTION. The pack's own notes
say structure/economy is the real signal and that topic words/profanity are
weak proxies. Almost all training judge scores are 0.0; the handful of
non-zero ones (0.7, 0.5, 0.35) are not the ones with the most slang hits —
they are jokes with (i) a mocked/undercut authority figure (nun outside a
pub), and/or (ii) a short, dry, understated punchline delivered with
comedic economy, sometimes with almost no explicit Aussie vocabulary at
all. So this module treats explicit vernacular as a MODEST, word-boundary
-safe signal, and leans on two things code cannot reliably judge on its
own: (1) whether an authority figure/institution is being mocked
(larrikin anti-authoritarianism) and (2) whether the punchline reads as
dry/deadpan rather than slapstick (dry/irreverent tone) -- both routed to
LLM_FIELDS. Cartooning prominence (the archetype fully "landing": slang +
authority-mockery + dry delivery together) is approximated with a small
interaction bonus rather than a separate detector, since there is no
literal cartoon/image signal available in text.
"""

import re

LLM_FIELDS = {
    "authority_target": (
        "Name the authority figure/institution mocked or undercut in this "
        "text (e.g. priest, nun, cop, boss, politician, teacher), else NONE."
    ),
    "dry_tone": (
        "Say YES if the punchline lands in a dry, deadpan, understated way "
        "(not slapstick/silly/exclamatory), else NO."
    ),
}

# Distinctive Australian vernacular. Kept to terms unlikely to appear as
# innocuous substrings of unrelated words; matched on word boundaries so a
# single spurious hit (like "oi" inside "poisonous" in the baseline) cannot
# occur here.
_AUSSIE_WORDS = [
    "mate", "bloke", "sheila", "ute", "arvo", "brekkie", "servo", "bogan",
    "straya", "aussie", "crikey", "reckon", "drongo", "galah", "cobber",
    "digger", "dunny", "snag", "barbie", "yakka", "whinge", "ripper",
    "stubby", "schooner", "knackered", "larrikin", "wowser", "bloody",
    "esky", "budgie", "thongs",
]
_AUSSIE_PHRASES = [
    "fair dinkum", "no worries", "she'll be right", "shes right",
    "g'day", "gday", "flat out like a lizard drinking", "up shit creek",
    "tall poppy",
]

_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _AUSSIE_WORDS) + r")\b",
    re.IGNORECASE,
)
_PHRASE_RES = [re.compile(re.escape(p), re.IGNORECASE) for p in _AUSSIE_PHRASES]

# Understatement / litotes markers -- a textual analog of dry Australian
# deadpan delivery that code can check directly.
_UNDERSTATEMENT_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bnot\s+bad\b", r"\bcould\s+be\s+worse\b", r"\bfair\s+enough\b",
        r"\bnothing\s+much\b", r"\bno\s+big\s+deal\b", r"\bshe'?ll\s+be\s+right\b",
        r"\bs?pose\s+so\b",
    ]
]

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _safe_normalize(text, ops):
    try:
        norm = ops.normalize(text)
        if isinstance(norm, str) and norm.strip():
            return norm
    except Exception:
        pass
    return text or ""


def _own_sentence_word_counts(text):
    # Fallback/independent sentence economy signal: don't rely solely on
    # ops.sent_stats' aggregate mean, since a short deadpan punchline can be
    # buried before trailing meta-commentary ("...I had to share it").
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    counts = []
    for p in parts:
        n = len(_WORD_TOKEN_RE.findall(p))
        if n > 0:
            counts.append(n)
    return counts


def _get_sent_stats(norm_text, ops):
    # Contract documents ops.sent_stats as returning a 3-tuple
    # (n_sent, mean_words_per_sent, frac_long_words); tolerate a dict shape
    # too in case the runtime binds it differently.
    n_sent, mean_wps, frac_long = 0, 0.0, 0.0
    try:
        stats = ops.sent_stats(norm_text)
        if isinstance(stats, dict):
            n_sent = stats.get("n_sent", stats.get("num_sentences", 0)) or 0
            mean_wps = stats.get(
                "mean_words_per_sent", stats.get("mean_words_per_sentence", 0.0)
            ) or 0.0
            frac_long = stats.get(
                "frac_long_words", stats.get("frac_long", 0.0)
            ) or 0.0
        elif isinstance(stats, (tuple, list)) and len(stats) >= 3:
            n_sent, mean_wps, frac_long = stats[0], stats[1], stats[2]
    except Exception:
        pass
    return n_sent, float(mean_wps or 0.0), float(frac_long or 0.0)


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        norm = _safe_normalize(text, ops)
        low = norm.lower()

        val = 0.08  # base: most jokes in this corpus are not Australian-coded

        # --- code signal 1: word-boundary-safe vernacular lexicon ---
        n_word_hits = len(set(_WORD_RE.findall(low)))
        n_phrase_hits = sum(1 for p in _PHRASE_RES if p.search(low))
        n_vern_hits = n_word_hits + n_phrase_hits
        val += 0.05 * min(n_vern_hits, 4)  # cap at +0.20

        # --- code signal 2: dry/laconic punchline economy ---
        _n_sent, mean_wps, frac_long = _get_sent_stats(norm, ops)
        own_counts = _own_sentence_word_counts(norm)
        has_economy = False
        if len(own_counts) >= 3:
            trailing = own_counts[1:]  # skip title/opening sentence
            shortest = min(trailing)
            baseline_mean = mean_wps if mean_wps > 0 else (
                sum(own_counts) / len(own_counts)
            )
            if shortest <= 8 and shortest <= 0.6 * max(baseline_mean, 1.0):
                has_economy = True
        if has_economy:
            val += 0.15
        if 0.0 < frac_long < 0.15:
            val += 0.05  # plain vernacular register, not ornate wording

        # --- code signal 3: understatement / litotes phrasing ---
        n_understatement = sum(1 for r in _UNDERSTATEMENT_RES if r.search(low))
        val += 0.08 * min(n_understatement, 2)  # cap at +0.16

        # --- LLM field 1: larrikin anti-authoritarianism ---
        authority = (extracted.get("authority_target") or "").strip()
        has_authority = bool(authority) and authority.upper() not in ("NONE", "N/A", "-")
        if has_authority:
            val += 0.25

        # --- LLM field 2: dry/deadpan tone ---
        dry = (extracted.get("dry_tone") or "").strip().upper()
        is_dry = dry.startswith("Y")
        if is_dry:
            val += 0.20

        # --- interaction: larrikin archetype fully present ---
        if has_authority and is_dry and n_vern_hits >= 1:
            val += 0.10

        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
