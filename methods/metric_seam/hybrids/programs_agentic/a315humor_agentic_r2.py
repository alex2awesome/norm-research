# Hybrid metric channel for aspect a315 (humor): "Punchline-last placement"
# Criterion: place the funniest word/twist at the final beat; end on the peak
# without follow-on that deflates the laugh.
#
# DIAGNOSIS 1 (from h0's code_score, measured directly against the full
# train set): h0's code-side "structural fallback" used two small
# step-bonuses (ends_clean, short-final-sentence ratio) to spread scores
# within the huge "LLM said punch_ends_text=YES, no deflator" bucket (86% of
# train items). That code_score has ~0 rank-correlation with the judge
# inside this bucket (rho=0.05) -- it is not real signal, it's noise, and
# spreading scores by noise actively HURTS Spearman versus leaving the
# bucket flat (tying the whole bucket at a constant alone raises overall
# train rho 0.364 -> 0.505). The two step-bonuses are dropped for that
# reason; the code fallback now only distinguishes "coda undercuts the
# punch" vs "clean ending", nothing finer.
#
# DIAGNOSIS 2 (from agentic_run.py residuals on this diagnosis-1 candidate):
# the single biggest remaining error class was BOILERPLATE tails -- source
# credit ("Credit from /u/X"), reddit-style dedications ("^(In memory of
# Bob Einstein)"), subreddit/inspiration credits, stray emoji/emoticons, and
# markdown/HTML artifacts (&amp;#x200B;) that trail the real joke text. The
# extractor correctly notices *something* follows the punchline and answers
# NO / names a "deflator", but these are platform metadata, not a coda that
# undercuts the joke -- and the judge scores these documents at or near 1.0.
# Fix: detect this boilerplate class (from the extractor's own stated reason
# AND from raw-text tail patterns) and treat it as a clean ending, not a
# deflating one. This is additive -- genuine apology/moral/disclaimer/
# over-explaining codas (confirmed by the SAME extractor fields) still drag
# the score down hard; only the credit/dedication/emoji subclass is spared.
#
# A meta-question-to-the-audience ending ("do you guys think this is a good
# ending?") and an "over-explaining" opener on the TRUE final sentence
# ("Of course I won't be able to find it there.") are kept as small,
# precisely-anchored code-side additions to the same "coda undercuts the
# punch" construct (matched only at the start of the actual last sentence,
# not a wide window -- an earlier wide-window version of the over-explain
# check false-positived on punchline text itself, e.g. a character's line
# "That's because it is!" that IS the twist, not a deflating coda).
import re

LLM_FIELDS = {
    "punch_ends_text": (
        "Answer YES if the single funniest word/twist of this joke is the very "
        "last thing in the text, with nothing after it that explains, apologizes "
        "for, or undercuts it; otherwise answer NO with a short reason (<=15 words)."
    ),
    "trailing_deflator": (
        "In <=12 words, describe any text AFTER the funniest twist that "
        "apologizes for, explains, moralizes about, or otherwise undercuts the "
        "joke (disclaimer, 'jk', moral, meta-comment); answer 'none' if nothing "
        "follows it."
    ),
}

_DEFLATOR_RE = re.compile(
    r"("
    r"not\s+(a\s+)?racist|no\s+offense|j\s*/?\s*k\b|just\s+kidding|only\s+joking|"
    r"kidding\s+aside|disclaimer|trigger\s+warning|\btw\s*:|\bcw\s*:|"
    r"sorry\s+(if|for)|don'?t\s+(kill|hurt|hate|murder)\s+me|"
    r"please\s+don'?t\s+(kill|murder)|"
    r"the\s+moral\s+(of\s+(this|the)\s+story\s+)?is|\bedit\s*:|\bupdate\s*:|"
    r"get\s+it\?|ba+\s*dum\s*(tss|tiss|ching)?|see\s+what\s+i\s+did\s+there|"
    r"i\s+know\s+(it'?s|this\s+is)?\s*(bad|terrible|dumb|stupid)|not\s+a\s+joke|"
    r"subscribe|click\s+here|read\s+more|related\s+articles?|follow\s+us"
    r")",
    re.IGNORECASE,
)

_META_COMMENT_RE = re.compile(
    r"^(i\s+think|i\s+guess|anyway|so\s+yeah|just\s+saying|ps\b|note\s*:|also,)",
    re.IGNORECASE,
)

# meta-question directed at the audience/reader in the final sentence: a
# direct frame-break that solicits feedback instead of landing the twist.
_META_AUDIENCE_RE = re.compile(
    r"(do\s+(?:you|y'?all|u)\s+(?:guys\s+|all\s+)?think|don'?t\s+you\s+think|"
    r"what\s+do\s+you\s+think|good\s+(?:ending|joke)\s*\??|thoughts\s*\??|"
    r"isn'?t\s+(?:it|that)|am\s*i\s*right|right\s*\?$)",
    re.IGNORECASE,
)

# over-explaining opener on the TRUE final sentence only (anchored match,
# not a window search) -- spelling out the joke's logic in prose instead of
# landing on the twist. Anchoring matters: a window search over the last
# ~200 chars false-positived on punchline dialogue itself (a character's
# "That's because it is!" that IS the twist, two sentences before the real
# last line).
_OVEREXPLAIN_RE = re.compile(
    r"^(of\s+course|that'?s\s+(?:because|why)|the\s+reason\s+(?:is|being)|"
    r"which\s+means|in\s+other\s+words)\b",
    re.IGNORECASE,
)

# Boilerplate/attribution tail: source credit, reddit-style dedications
# ("^(In memory of ...)"), subreddit/inspiration credits, URLs, /u/ or /r/
# references, and scraped markdown/HTML artifacts. This is platform
# metadata riding after the joke, not a coda that undercuts it.
_BOILERPLATE_TAIL_RE = re.compile(
    r"(credit(?:s|ed)?\s+(?:to|from|goes\s+to)|\^\(|in\s+memory\s+of|"
    r"inspired\s+by|https?://|(?:^|[\s(])/?[ur]/\w+|source\s*:|"
    r"&amp;#x200b;|&#x200b;)",
    re.IGNORECASE,
)
_EMOJI_TAIL_RE = re.compile(
    r"[\U0001F300-\U0001FAFF☀-➿←-⇿⬀-⯿]+\s*$"
    r"|[:;]-?[)(dD]+\s*$"
)
# Same boilerplate class, but read off the extractor's OWN stated reason for
# its NO / deflator answer (more reliable than re-deriving it from raw text,
# since the extractor already located the trailing span).
_BOILERPLATE_REASON_RE = re.compile(
    r"(credit|dedicat|memory\s+of|smiley|emoji|emoticon|subreddit|reddit|"
    r"attribution|hashtag|watermark|\burl\b|hyperlink|\blink\b|source\s*:)",
    re.IGNORECASE,
)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_NONE_LIKE = ("", "none", "none.", "n/a", "no", "no.", "nothing", "nothing.")


def _has_boilerplate_tail(text):
    tail = text[-260:]
    if _BOILERPLATE_TAIL_RE.search(tail):
        return True
    if _EMOJI_TAIL_RE.search(text.rstrip()):
        return True
    return False


def _split_sentences(t):
    return [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]


def _code_structure_score(text):
    """Structural fallback/prior: does the ending land on the twist, or does
    a disclaimer / moralizing coda / audience-facing meta-question undercut
    it? (No graded sub-bonuses -- those measured as pure noise on train.)"""
    if not text or not text.strip():
        return 0.5

    sents = _split_sentences(text)
    if not sents:
        return 0.5

    tail_window = text[-220:]  # near-end region; disclaimers/boilerplate live here
    deflator_in_tail = bool(_DEFLATOR_RE.search(tail_window))

    last_sent = sents[-1]
    meta_comment = bool(_META_COMMENT_RE.search(last_sent))
    meta_question = (bool(_META_AUDIENCE_RE.search(last_sent))
                      and last_sent.rstrip().endswith("?"))
    overexplain = bool(_OVEREXPLAIN_RE.match(last_sent.strip()))

    if deflator_in_tail or meta_comment or meta_question or overexplain:
        score = 0.35
        if len(sents) >= 2:
            words_last = len(last_sent.split())
            prior = sents[:-1]
            words_mean = sum(len(s.split()) for s in prior) / max(1, len(prior))
            if words_mean > 0 and words_last > 1.6 * words_mean:
                score -= 0.15  # long rambling coda compounds the deflation
    else:
        score = 0.75

    return max(0.0, min(1.0, score))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.5

        code_score = _code_structure_score(t)

        pe_raw = str(extracted.get("punch_ends_text", "") or "").strip()
        td_raw = str(extracted.get("trailing_deflator", "") or "").strip()
        pe = pe_raw.lower()
        td = td_raw.lower()

        deflator_present = td not in _NONE_LIKE
        # Is the trailing content the extractor is reacting to actually
        # credit/dedication/emoji/markdown boilerplate rather than a coda
        # that undercuts the joke? Check both the extractor's own stated
        # reason and the raw text tail.
        boilerplate = (bool(_BOILERPLATE_REASON_RE.search(pe_raw))
                       or bool(_BOILERPLATE_REASON_RE.search(td_raw))
                       or _has_boilerplate_tail(t))

        if deflator_present and not boilerplate:
            # Confirmed genuine deflator (apology/moral/disclaimer/explains-
            # the-pun): this is the direct violation of the criterion,
            # regardless of what punch_ends_text separately claimed.
            llm_base = 0.15
        elif pe.startswith("yes"):
            llm_base = 0.85
        elif pe.startswith("no"):
            # A NO caused by boilerplate (credit/dedication/emoji) reflects
            # the same clean-landing state as a YES -- the extractor just
            # noticed the metadata riding after the real ending, not a
            # deflating coda -- so treat it at parity with a plain YES.
            llm_base = 0.85 if boilerplate else 0.3
        else:
            llm_base = None

        if llm_base is not None:
            llm_base = max(0.0, min(1.0, llm_base))
            final = 0.7 * llm_base + 0.3 * code_score
        else:
            # No usable LLM signal: fold the deflator field into the code prior.
            final = code_score - 0.25 if (deflator_present and not boilerplate) else code_score

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
