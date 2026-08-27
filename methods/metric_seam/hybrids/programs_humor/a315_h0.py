# Hybrid metric channel for aspect a315: "Punchline-last placement"
# Criterion: place the funniest word/twist at the final beat; end on the peak
# without follow-on that deflates the laugh.
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

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_NONE_LIKE = ("", "none", "none.", "n/a", "no", "no.", "nothing", "nothing.")


def _split_sentences(t):
    return [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]


def _code_structure_score(text):
    """Structural fallback/prior: does the text end cleanly on its final beat,
    or is there a trailing disclaimer / rambling meta-comment coda?"""
    if not text or not text.strip():
        return 0.5

    sents = _split_sentences(text)
    if not sents:
        return 0.5

    tail_window = text[-220:]  # near-end region; disclaimers/boilerplate live here
    deflator_in_tail = bool(_DEFLATOR_RE.search(tail_window))

    last_sent = sents[-1]
    meta_comment = bool(_META_COMMENT_RE.search(last_sent))

    stripped_end = text.rstrip()
    ends_clean = bool(re.search(r"[\"'”)!?.]\s*$", stripped_end))

    score = 0.5
    if deflator_in_tail or meta_comment:
        score -= 0.3
        if len(sents) >= 2:
            words_last = len(last_sent.split())
            prior = sents[:-1]
            words_mean = sum(len(s.split()) for s in prior) / max(1, len(prior))
            if words_mean > 0 and words_last > 1.6 * words_mean:
                score -= 0.15  # long rambling coda compounds the deflation
    else:
        if ends_clean:
            score += 0.15
        if len(sents) >= 2:
            words_last = len(last_sent.split())
            prior = sents[:-1]
            words_mean = sum(len(s.split()) for s in prior) / max(1, len(prior))
            if words_mean > 0 and words_last <= max(3, 0.6 * words_mean):
                score += 0.1  # short, punchy final beat

    return max(0.0, min(1.0, score))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.5

        code_score = _code_structure_score(t)

        pe = str(extracted.get("punch_ends_text", "") or "").strip().lower()
        td = str(extracted.get("trailing_deflator", "") or "").strip().lower()

        deflator_present = td not in _NONE_LIKE

        llm_base = None
        if pe.startswith("yes"):
            llm_base = 0.8
        elif pe.startswith("no"):
            llm_base = 0.3

        if llm_base is not None:
            llm_base += -0.35 if deflator_present else 0.05
            llm_base = max(0.0, min(1.0, llm_base))
            final = 0.7 * llm_base + 0.3 * code_score
        else:
            # No usable LLM signal: fold the deflator field into the code prior.
            final = code_score - 0.25 if deflator_present else code_score

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
