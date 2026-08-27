"""a36_h0 -- Hybrid channel for "Frame Narratives and Guide/Chorus Voices".

Construct: stories presented as TOLD/TRANSMITTED tales (storyteller addressing a
listener, letter/diary/chat-log frames, nested tales, masked narrators, chorus
voices commenting on the action) vs. direct immediate-scene narration.

Design: two LLM extraction fields give thick-input grounding (frame device name;
a verbatim audience-address quote that code VERIFIES against the text). The
predicate stays in code: a told-tale phrase bank, narration-level (dialogue- and
italics-stripped) second-person address density, narration imperatives, a
chat/transcript chorus detector, and a retrospective told-tale opener detector.
Author-note chrome (Edit:/[WP]/subreddit plugs) is stripped first so it cannot
leak "you"/"tell" signals in either direction.
"""

import re

LLM_FIELDS = {
    "frame_device": (
        "If the story is framed as a tale told, written, or transmitted (a storyteller "
        "addressing listeners, letter, diary, chat log, transcript, or a story nested "
        "inside another), name the device in a few words; otherwise answer NONE."
    ),
    "teller_quote": (
        "Quote verbatim (max 10 words) where a storytelling narrator directly addresses "
        "readers or listeners (e.g. 'let me tell you the story'); answer NONE if no such "
        "address exists."
    ),
}

# ----------------------------------------------------------------------------
# text cleanup
# ----------------------------------------------------------------------------

_SMART = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "′": "'",
    "–": "-", "—": "-", "―": "-", " ": " ",
}

_CHROME_PREFIXES = (
    "edit:", "edit :", "edit-", "edit 2", "final edit", "final final edit",
    "update:", "updated:", "a/n", "an:", "author's note", "authors note",
    "obligatory", "[wp]", "[part", "part 2", "part two", "p.s.",
)

_CHROME_SUBSTR = (
    "/r/", "reddit.com", "let me know what you think", "feedback is welcome",
    "feedback welcome", "any feedback", "thanks for reading", "thank you for reading",
    "my first post", "first post here", "if no one else will post",
    "gotta contribute", "constructive criticism", ";)", "blew up",
    "thank you so much for all", "upvote",
)


def _basic_normalize(text):
    for k, v in _SMART.items():
        text = text.replace(k, v)
    text = text.replace("&gt;", ">").replace("&lt;", "<")
    text = text.replace("&amp;", "&").replace("&nbsp;", " ").replace("&#x200b;", " ")
    text = text.replace("[...]", " ")
    return text


def _strip_chrome(text):
    """Drop author-note / scraped-chrome lines so they can't leak signals."""
    kept = []
    for line in text.split("\n"):
        probe = line.strip().lstrip("*#>\\-~ \t[").lower()
        low = line.lower()
        if any(probe.startswith(p) for p in _CHROME_PREFIXES):
            continue
        if any(s in low for s in _CHROME_SUBSTR):
            continue
        if set(line.strip()) and set(line.strip()) <= set("*~-_= "):
            continue  # pure separator line
        kept.append(line)
    return "\n".join(kept)


def _narration_only(text):
    """Remove quoted dialogue and short *emphasis/thought* spans; keep narration."""
    t = re.sub(r"\*[^*\n]{1,140}\*", " ", text)
    out = []
    for para in t.split("\n"):
        if '"' in para:
            parts = para.split('"')
            out.append(" ".join(parts[0::2]))  # even segments = outside quotes
        else:
            out.append(para)
    return "\n".join(out)


# ----------------------------------------------------------------------------
# code predicates
# ----------------------------------------------------------------------------

_STRONG_PHRASES = [
    r"\blet me tell you\b",
    r"\bi(?:'ll| will) tell you\b",
    r"\btell you (?:all )?about\b",
    r"\bi was there when\b",
    r"\bthis is (?:the|a) story\b",
    r"\bthe story of how\b",
    r"\bit all (?:started|began)\b",
    r"\ball (?:started|began) when\b",
    r"\bif you know what i(?:'m| am)? ?(?:saying|mean)\b",
    r"\bwho do you think\b",
    r"\bthis is how i\b",
    r"\bgather (?:round|around)\b",
    r"\bdear reader\b",
    r"\bonce upon a time\b",
    r"\btold (?:me|us|him|her|them) (?:the|a|this|that) (?:tale|story)\b",
    r"\btell (?:me|us) (?:the|a|this) (?:tale|story)\b",
    r"\blong story short\b",
    r"\bas i was saying\b",
    r"\bwhere was i\b",
    r"\bi'll spare you\b",
    r"\bhave i (?:had|got) a\b",
    r"\bthe tale of how\b",
    r"\bso it (?:is|was) (?:said|told)\b",
    r"\byou(?:'re| are) probably wondering\b",
]

_WEAK_PHRASES = [
    r"\byou see,",
    r"\bmind you,",
    r"\bbelieve me\b",
    r"\btrust me\b",
    r"\bi often wonder\b",
    r"\bmy friend,",
    r"\bso, friend\b",
    r"\bi digress\b",
    r"\byou may wonder\b",
]

_IMPERATIVES = {
    "come", "listen", "imagine", "picture", "gather", "behold", "consider",
    "stop", "watch", "go", "take", "throw", "remember", "note", "revel",
}

_CHAT_LINE = re.compile(r"^\*?[A-Z][\w'.\-]*(?: [A-Z][\w'.\-]*){0,2}: \S")

_OPENERS = [
    r"\bit all (?:started|began)\b", r"\bi was there\b", r"\bboy have i\b",
    r"\bonce upon\b", r"\blet me tell\b", r"\bthis is (?:the|a) story\b",
    r"\bhave i got a\b", r"\bso there i was\b", r"\byou won'?t believe\b",
    r"\bi remember when\b", r"\bcome, ", r"\bgather (?:round|around)\b",
]


def _phrase_sig(full_text):
    n = 0.0
    for p in _STRONG_PHRASES:
        if re.search(p, full_text):
            n += 1.0
    for p in _WEAK_PHRASES:
        if re.search(p, full_text):
            n += 0.5
    return min(1.0, 0.55 * n)


def _addr_sig(narr):
    words = re.findall(r"[a-z']+", narr)
    if len(words) < 80:
        return 0.0
    you = sum(1 for w in words if w in ("you", "your", "yours", "yourself"))
    if you < 3:
        return 0.0
    dens = 100.0 * you / len(words)
    return max(0.0, min(1.0, (dens - 0.5) / 1.5))


def _imp_sig(narr):
    hits = 0
    for m in re.finditer(r"(?:^|[.!?]\s+)([A-Z][a-z]+)[ ,]", narr, re.M):
        if m.group(1).lower() in _IMPERATIVES:
            hits += 1
    return min(1.0, hits / 3.0)


def _chat_sig(text):
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 8:
        return 0.0
    hits = sum(1 for l in lines if _CHAT_LINE.match(l.strip()))
    if hits >= 6 and hits >= 0.25 * len(lines):
        return 1.0
    return 0.0


def _opener_sig(text):
    head = text.strip()[:300].lower()
    return 1.0 if any(re.search(p, head) for p in _OPENERS) else 0.0


# ----------------------------------------------------------------------------
# LLM field handling (predicate in code; fields only supply grounding)
# ----------------------------------------------------------------------------

_NEG_ANSWERS = {
    "none", "no", "n/a", "na", "nothing", "absent", "not present", "no frame",
    "no device", "direct", "direct narration", "immediate", "standard",
    "regular", "normal", "unknown",
}

_DEVICE_STRONG = (
    "letter", "diary", "journal", "epistolary", "chat", "transcript", "log",
    "forum", "storyteller", "story-teller", "bard", "campfire", "tavern",
    "story within", "story-within", "nested", "frame", "framed", "tale",
    "told", "recount", "oral", "listener", "audience", "advice column",
    "guide", "video", "album", "metafiction", "author", "manuscript",
    "testimony", "monologue", "address", "second person", "second-person",
)

_DEVICE_WEAK_ONLY = ("first-person", "first person", "third-person",
                     "third person", "past tense", "flashback")


def _clean_answer(extracted, key):
    try:
        v = extracted.get(key, "") if isinstance(extracted, dict) else ""
    except Exception:
        return ""
    if not isinstance(v, str):
        return ""
    v = v.strip().strip('"\'` .')
    if not v:
        return ""
    low = v.lower().strip()
    if low in _NEG_ANSWERS or low.startswith(("none", "no ", "n/a", "not ")):
        return ""
    return v


def _device_sig(extracted):
    ans = _clean_answer(extracted, "frame_device")
    if not ans:
        return 0.0
    low = ans.lower()
    if any(w in low for w in _DEVICE_STRONG):
        return 1.0
    if any(w in low for w in _DEVICE_WEAK_ONLY):
        return 0.25
    return 0.5


def _squash_ws(s):
    return " ".join(re.findall(r"[a-z0-9']+", s.lower()))


def _quote_sig(extracted, full_text):
    ans = _clean_answer(extracted, "teller_quote")
    if not ans:
        return 0.0
    q = _squash_ws(ans)
    if not q:
        return 0.0
    body = _squash_ws(full_text)
    if q in body:
        return 1.0  # verified verbatim quote -> gold
    words = [w for w in q.split() if len(w) > 3]
    if words and sum(1 for w in words if w in body) >= 0.6 * len(words):
        return 0.6  # near-verbatim / lightly paraphrased
    return 0.3  # asserted but unverifiable


# ----------------------------------------------------------------------------
# score
# ----------------------------------------------------------------------------

def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        try:
            t = ops.normalize(text)
            if not isinstance(t, str) or not t.strip():
                t = text
        except Exception:
            t = text
        t = _basic_normalize(t)
        t = _strip_chrome(t)
        if not t.strip():
            return 0.5
        low = t.lower()
        narr = _narration_only(t)
        narr_low = narr.lower()

        code_part = min(1.0,
                        0.50 * _phrase_sig(low) +
                        0.45 * _addr_sig(narr_low) +
                        0.75 * _chat_sig(t) +
                        0.40 * _opener_sig(t) +
                        0.30 * _imp_sig(narr))

        if not isinstance(extracted, dict):
            extracted = {}
        llm_part = min(1.0,
                       0.62 * _device_sig(extracted) +
                       0.62 * _quote_sig(extracted, low))

        combined = min(1.0, 0.60 * llm_part + 0.62 * code_part)
        return max(0.0, min(1.0, 0.08 + 0.90 * combined))
    except Exception:
        return 0.5
