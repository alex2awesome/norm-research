"""Hybrid metric channel for a189: Form-content integration and aesthetic finish.

Design (from train residuals):
- Judge lows are dominated by mechanical sloppiness (lowercase 'i', missing
  apostrophes, doubled words, spacing errors, run-ons, no paragraph breaks)
  and amateur meta-chrome (edit notes, apologies, feedback-begging).
- Judge highs are clean, and the top ones commit to a distinctive imitated
  FORM (news article, chat transcript, numbered tips, refrain structure).
- Keyword/topic is irrelevant; predicate stays in code. LLM fields supply
  what regex cannot see: agreement-level grammar errors, and the name of the
  imitated form. Diegetic typos in transcript-form stories are forgiven.
"""

import re
import math
import statistics
from collections import Counter

LLM_FIELDS = {
    "typo_sample": (
        "Quote verbatim up to three spelling or grammar errors the author makes in narration "
        "(ignore intentional in-character or dialect errors); answer NONE if the prose is clean."
    ),
    "form_device": (
        "In at most 5 words, name any special text format the story imitates "
        "(chat log, news report, numbered tips, letter, diary, transcript); "
        "answer NONE if standard narrative prose."
    ),
}

_FORM_WORDS = (
    "chat", "transcript", "log", "news", "report", "article", "broadcast",
    "list", "tips", "listicle", "letter", "diary", "journal", "epistolary",
    "review", "interview", "script", "screenplay", "manual", "guide",
    "recipe", "advertisement", "faq", "poem", "verse", "monologue",
    "forum", "text message", "memo", "obituary", "instructions", "documentary",
)

_TRANSCRIPT_WORDS = ("chat", "transcript", "log", "forum", "text message", "message board")

_META_PHRASES = (
    "sorry for", "first time post", "first post", "long time lurker",
    "on my phone", "tear me apart", "constructive criticism", "any feedback",
    "feedback is welcome", "let me know what you think", "word count",
    "still crap", "please let me know", "hopefully you liked",
    "hope you enjoyed", "thanks for reading", "thank you for reading",
    "gotta contribute", "not my best", "plot holes, i know", "disclaimer:",
    "[wp]", "i know it's bad", "criticism welcome", "first attempt",
)

_APOSTROPHE_ERRS = (
    "dont", "didnt", "doesnt", "isnt", "wasnt", "couldnt", "wouldnt",
    "shouldnt", "im", "ive", "youre", "theyre", "hadnt", "havent",
    "arent", "werent", "mustve", "couldve", "wouldve", "whats", "thats",
    "theres", "wont", "cant",
)

_DOUBLED_OK = {"had", "that", "very", "so", "no", "ha", "boom", "tick", "knock"}


def _mech_rate(t, raw):
    """Weighted mechanical-error count per 1000 words on normalized text."""
    words = re.findall(r"[A-Za-z']+", t)
    n_words = max(1, len(words))
    w = 0.0
    # standalone lowercase 'i' as pronoun
    w += 2.0 * len(re.findall(r"(?<![A-Za-z])i(?![A-Za-z])", t))
    # doubled words ("but but", "know know")
    dbl = [m for m in re.findall(r"\b([a-z]{2,})\s+\1\b", t) if m not in _DOUBLED_OK]
    w += 2.0 * len(dbl)
    # space before punctuation ("second , every")
    w += 1.5 * len(re.findall(r"[A-Za-z][ \t]+[,;!?]", t))
    w += 1.5 * len(re.findall(r"[A-Za-z][ \t]+\.(?!\.)", t))
    # missing space after punctuation ("doing.It", "now,i")
    w += 1.5 * len(re.findall(r"[a-z][,;][A-Za-z]", t))
    w += 1.5 * len(re.findall(r"[a-z]\.[A-Z]", t))
    # missing-apostrophe contractions (lowercase only; capitalized "Im" too risky except sentence-y ones)
    apo = re.compile(r"\b(?:%s)\b" % "|".join(_APOSTROPHE_ERRS))
    w += 1.5 * len(apo.findall(t))
    # emoticons in the prose
    w += 2.5 * len(re.findall(r"(?:[;:]-?\)|:\(|:D\b|:P\b)", t))
    # shouting caps (capped)
    caps = re.findall(r"\b[A-Z]{4,}\b", t)
    caps = [c for c in caps if c not in ("ISIS", "NASA", "SWAT", "SEAL", "EXIT", "PARIS")]
    w += 0.4 * min(6, len(caps))
    # mid-clause double spaces (two-spaces-after-period is a typing convention, not a flaw)
    w += 0.35 * min(10, len(re.findall(r"[A-Za-z,;]  ", raw or "")))
    return w / n_words * 1000.0


def _meta_penalty(t_low):
    hits = sum(1 for p in _META_PHRASES if p in t_low)
    # edit-note lines ("Edit:", "Edit 1:")
    hits += min(2, len(re.findall(r"\bedit ?\d*\s*:", t_low)))
    return min(0.28, 0.14 * hits)


def _refrain(t):
    """A >=6-word shingle repeated verbatim = deliberate structural refrain."""
    toks = re.findall(r"[a-z']+", t.lower())
    if len(toks) < 60:
        return False
    grams = Counter(tuple(toks[i:i + 6]) for i in range(len(toks) - 5))
    rep = [g for g, c in grams.items() if c >= 2]
    return len(rep) >= 3  # a real refrain repeats a full clause, giving several shingles


def _code_form(t):
    """Code backstop for unmistakable imitated forms."""
    if re.search(r"\(AP\)|\(Reuters\)", t):
        return True
    if re.search(r"has joined|has disconnected", t) and re.search(r"chat|lobby|server", t, re.I):
        return True
    if len(re.findall(r"(?im)^\s*(?:tip\s*#?\s*\d|step\s*\d|rule\s*#?\s*\d)", t)) >= 3:
        return True
    return False


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.0
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw
        # strip URLs/markdown-link guts so they don't trip spacing regexes
        t = re.sub(r"https?://\S+|www\.\S+|&\w+;|&#x?\w+;", " ", t)
        t_low = t.lower()

        # ---- LLM fields ----
        form_ans = (extracted.get("form_device") or "").strip().lower()
        if form_ans in ("none", "none.", "no", "n/a"):
            form_ans = ""
        has_form = bool(form_ans) and any(wd in form_ans for wd in _FORM_WORDS)
        has_form = has_form or _code_form(t)
        is_transcript = bool(form_ans) and any(wd in form_ans for wd in _TRANSCRIPT_WORDS)
        if not is_transcript and re.search(r"has joined|has disconnected", t) \
                and re.search(r"chat|lobby|server", t, re.I):
            is_transcript = True

        typo_ans = (extracted.get("typo_sample") or "").strip()
        if typo_ans.lower().strip(" .") in ("none", "no errors", "n/a", "no"):
            typo_ans = ""
        if typo_ans:
            n_items = 1 + min(2, typo_ans.count(";"))
            quoted = len(re.findall(r'"[^"]{2,}"', typo_ans))
            n_items = min(3, max(n_items, quoted))
        else:
            n_items = 0

        # ---- code predicates ----
        mech = min(1.0, _mech_rate(t, raw) / 12.0)
        meta = _meta_penalty(t_low)
        typo_pen = 0.07 * n_items
        if is_transcript:  # diegetic typos in an imitated chat/log form
            mech *= 0.3
            typo_pen *= 0.3

        try:
            _, mws, _ = ops.sent_stats(t)
        except Exception:
            mws = 18.0
        runon = min(0.12, max(0.0, (float(mws) - 25.0)) * 0.012)
        if is_transcript:  # chat/log lines lack terminal punctuation; not run-ons
            runon = 0.0

        n_wtok = len(re.findall(r"[A-Za-z']+", t))
        no_paras = 0.08 if (n_wtok > 120 and "\n\n" not in raw) else 0.0

        refrain_bonus = 0.06 if _refrain(t) else 0.0
        dash_semi = t.count("—") + t.count("–") + t.count(";") \
            + len(re.findall(r"[a-z] - [a-z]", t_low))
        polish = 0.05 if (dash_semi >= 3 and mech < 0.25) else 0.0

        s = (0.55
             + (0.15 if has_form else 0.0)
             + refrain_bonus
             + polish
             - 0.32 * mech
             - meta
             - typo_pen
             - runon
             - no_paras)
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
