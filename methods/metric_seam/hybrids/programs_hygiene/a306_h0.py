"""
Hybrid channel for aspect a306: SSTH/GTVH-based analysis.

Criterion: evaluate via script opposition and the GTVH resources (Script
Opposition, Logical Mechanism, Situation, Narrative Strategy, Language),
integrating incongruity-resolution accounts at the theory level.

Design rationale (from the pack's training data + failure notes):
  - The frozen code baseline used topic/keyword lists ("thought", "seemed",
    "but", "actually", "reveals") as a proxy for "script switch" and scored
    ~0 correlation with the judge (train_rho = -0.083). The pack explicitly
    warns that topic words and profanity are WEAK proxies for judged craft;
    the real signal is structural (setup/punchline placement, economy,
    escalation) and, above all, whether the ending actually *resolves* an
    incongruity (a genuine script opposition / logical mechanism), which
    requires semantic judgment code cannot perform.
  - So the two LLM_FIELDS do the THICK-INPUT work: naming the humor
    mechanism (pun/wordplay vs. misdirection vs. absurdist-without-resolution
    vs. none) and judging whether the ending actually resolves/reinterprets
    the setup. Both are short categorical/keyword answers; the PREDICATE
    (how those categories map to a score, and how they combine with
    structural features) stays entirely in code, deterministic.
  - Code contributes a lightweight, criterion-general structural signal:
    "punchline economy" -- is the final sentence short/tight relative to the
    setup (a hallmark of setup->punch narrative strategy), a small dialogue
    bonus (many high-scoring items are Q&A/dialogue-driven jokes), and a
    mild damping factor for very long, unresolved rambles (only applied when
    the resolution signal itself is weak/absent, so a long-but-well-executed
    joke, e.g. an extended pun payoff, is not penalized for length alone).
  - Deliberately NOT used: topic/profanity keyword lists (explicitly flagged
    as a weak proxy that misled the baseline) and corpus similarity via
    ops.retrieve_similar as a quality predictor (similarity-to-corpus is not
    evidence of judged craft, it would just reward being a common/derivative
    joke template).
"""

import re
import math

LLM_FIELDS = {
    "mechanism": (
        "In <=5 words, name the joke's core mechanism: pun/wordplay, "
        "misdirection twist, ironic reversal, absurdist non-sequitur, or none."
    ),
    "resolution": (
        "In <=8 words: does the ending clearly resolve or reinterpret the "
        "setup (a script switch), or does it ramble/trail off with no punchline?"
    ),
}

# HYGIENE PATCH: bare "pun" `in s` fired on "punchline"/"punishment"/
# "punching"/"punctuation" in the LLM's free-text mechanism answer -- unrelated
# to the pun/wordplay mechanism this is meant to detect. Split "pun" out to a
# word-bounded check (`_PUN_RE`); the other, longer/distinctive phrases keep
# plain substring matching since they carry negligible collision risk.
_MECH_PUN_PHRASES = ("wordplay", "word play", "double meaning", "homophone", "play on words")
_PUN_RE = re.compile(r"\bpuns?\b")
_MECH_TWIST = ("misdirection", "twist", "irony", "ironic", "reversal", "reverse", "subversion")
_MECH_ABSURD = ("absurd", "non-sequitur", "nonsequitur", "non sequitur", "surreal", "random")
_MECH_NONE = ("none", "no mechanism", "n/a", "unclear", "nothing")

_RES_POS_WORDS = ("resolve", "reinterpret", "recontextual", "clear", "punchline", "reveal", "click")
_RES_NEG_WORDS = ("ramble", "rambl", "trail", "meander", "unresolved", "confus", "flat", "nothing")


def _map_mechanism(raw):
    s = raw.strip().lower() if isinstance(raw, str) else ""
    if not s:
        return 0.30  # answered NONE / no extraction -> weak-but-not-zero prior
    if _PUN_RE.search(s) or any(k in s for k in _MECH_PUN_PHRASES):
        return 1.0
    if any(k in s for k in _MECH_TWIST):
        return 0.85
    if any(k in s for k in _MECH_ABSURD):
        return 0.40
    if any(k in s for k in _MECH_NONE):
        return 0.15
    return 0.5  # some other mechanism named -> neutral credit


def _map_resolution(raw):
    s = raw.strip().lower() if isinstance(raw, str) else ""
    if not s:
        return 0.35
    has_no = re.search(r"\bno\b", s) is not None
    has_yes = re.search(r"\byes\b", s) is not None
    pos_hit = has_yes or any(k in s for k in _RES_POS_WORDS)
    neg_hit = any(k in s for k in _RES_NEG_WORDS)
    if pos_hit and not neg_hit:
        return 0.9
    if neg_hit and not pos_hit:
        return 0.15
    if has_no and not pos_hit:
        return 0.2
    return 0.5


def _split_sentences(text):
    parts = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
    return parts


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5

        try:
            norm = ops.normalize(text)
            if not isinstance(norm, str) or not norm.strip():
                norm = text
        except Exception:
            norm = text

        # --- LLM-grounded semantic component (the actual GTVH constructs) ---
        mech_raw = extracted.get("mechanism", "") if isinstance(extracted, dict) else ""
        res_raw = extracted.get("resolution", "") if isinstance(extracted, dict) else ""
        mech_score = _map_mechanism(mech_raw)
        res_score = _map_resolution(res_raw)
        llm_component = 0.55 * mech_score + 0.45 * res_score

        # --- structural component: setup -> short punch economy ---
        sents = _split_sentences(norm)
        n_sent = len(sents)
        if n_sent >= 2:
            word_counts = [len(s.split()) for s in sents]
            final_len = word_counts[-1] if word_counts[-1] > 0 else 1
            setup_lens = word_counts[:-1]
            mean_setup = sum(setup_lens) / max(1, len(setup_lens))
            if mean_setup <= 0:
                mean_setup = final_len
            ratio = final_len / mean_setup
            # short/tight final line relative to setup -> high; long trailing
            # final line relative to setup -> low. Neutral around ratio=1.
            brevity_score = 1.5 - 0.5 * ratio
            brevity_score = max(0.0, min(1.0, brevity_score))
        else:
            # single-clause text: no setup/punch split observable; neutral
            brevity_score = 0.5

        # small dialogue/Q&A bonus (Narrative Strategy: many strong items are
        # dialogue-driven jokes with quoted turns)
        quote_chars = sum(1 for c in norm if c in "\"“”")
        dialogue_bonus = 0.05 if (quote_chars >= 2 and n_sent >= 2) else 0.0

        structural_component = max(0.0, min(1.0, brevity_score))

        base = 0.65 * llm_component + 0.35 * structural_component + dialogue_bonus

        # mild damping for very long, weakly-resolved rambles only (avoid
        # punishing long-but-well-executed jokes whose resolution/mechanism
        # signal is already strong)
        try:
            n_words = len(norm.split())
        except Exception:
            n_words = 0
        if n_words > 220 and res_score < 0.4 and mech_score < 0.4:
            excess = min(1.0, (n_words - 220) / 300.0)
            base *= (1.0 - 0.25 * excess)

        out = max(0.0, min(1.0, base))
        return out
    except Exception:
        return 0.5
