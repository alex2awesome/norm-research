"""Hybrid metric channel for a288: Title effectiveness and alignment (short fiction).

Insight from train residuals: this is WritingPrompts-style fiction where the vast
majority of stories carry NO title at all -- the judge gives 0.0 to 25/30 train
examples. The judge only rewards documents that actually PRESENT a title
(explicit header, bolded name, or the author naming the piece, e.g. "Baby Team
Six" -> 1.0), and grades that title's punchiness and alignment with the story.
The v0 baseline scored the shape of the first prose line as if it were a title,
which is why it landed at rho ~ 0.009.

Design: one LLM field extracts the story's own title verbatim (or NONE).
All predicates stay in code:
  - no title            -> 0.05 (matches the mass of judge zeros)
  - prompt echo [WP]    -> 0.05
  - email/subject-like  -> 0.10 (in-story chrome, not a story title)
  - generic header      -> 0.18 ("Dear Diary", "Part 1", ...)
  - speaker-label guard -> 0.20 (extracted "title" is followed by ':' in text,
                                  i.e. a dialogue tag / header, not a title)
  - real title          -> 0.30 base
                           +0.20 verified verbatim in document (anti-hallucination)
                           +0.10 punchy length (1-8 words)
                           +0.40 * fraction of title content words echoed in body
                                  (premise/theme alignment)
"""

import re

LLM_FIELDS = {
    "story_title": (
        "Quote exactly the story's own title if the document displays or names one "
        "(a standalone header line, a bolded name, or the author naming the piece, "
        "e.g. 'Baby Team Six'); answer NONE if absent. Exclude [WP] prompt lines, "
        "dialogue speaker labels, email subject lines, and chapter headers."
    ),
}

_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with",
    "from", "into", "onto", "over", "under", "that", "this", "these", "those",
    "was", "were", "are", "is", "be", "been", "has", "had", "have", "not",
    "but", "all", "his", "her", "its", "our", "your", "their", "my", "who",
    "what", "when", "where", "why", "how", "you", "she", "he", "it", "they",
}

_GENERIC = {
    "dear diary", "the end", "part 1", "part one", "part i", "prologue",
    "epilogue", "chapter 1", "chapter one", "untitled", "poem", "a poem",
    "edit", "intro", "introduction", "my story", "the story", "short story",
    "a short story", "story", "wp", "writing prompt", "prompt",
}

_NONE_PREFIXES = (
    "none", "n/a", "na ", "no title", "no explicit", "no clear", "not titled",
    "there is no", "the story has no", "the document has no", "nothing",
    "not present", "no story title", "unknown", "untitled",
)


def _clean_title(raw):
    """Strip markdown, quotes, and meta-prefixes from the extractor's answer."""
    t = (raw or "").strip()
    # drop markdown emphasis, quotes, backticks
    t = re.sub(r'[*_`"“”‘’]+', "", t)
    # drop lead-in phrases the LLM may add
    t = re.sub(
        r"^(the\s+)?(story|piece|document|it)?('?s)?\s*(is\s+)?"
        r"(titled?|called|named)\s*[:\-]?\s*",
        "", t, flags=re.IGNORECASE)
    t = re.sub(r"^title\s*[:\-]\s*", "", t, flags=re.IGNORECASE)
    return t.strip(" .:;,-—–'\"")


def _norm_text(text, ops):
    """Lowercased document with markdown chrome removed, whitespace collapsed."""
    try:
        t = ops.normalize(text)
    except Exception:
        t = text or ""
    t = t.lower()
    t = re.sub(r'[*_`>#"“”‘’\\]+', " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = (extracted or {}).get("story_title", "") or ""
        title = _clean_title(raw)
        low = title.lower()

        # --- no title: the dominant judge-zero class -------------------------
        if not low or low.startswith(_NONE_PREFIXES):
            return 0.05
        # verbose meta-answers ("this story does not have an explicit title...")
        if "title" in low and re.search(r"\b(no|not|none|absent|lacks?|without)\b", low):
            return 0.05
        # prompt echo is not a title
        if low.startswith("[wp") or "[wp]" in low:
            return 0.05
        # in-story email chrome is not a story title
        if low.startswith(("re:", "re ", "subject:", "subject ", "fwd:", "cc:", "to:")):
            return 0.10
        # generic / structural headers barely register with the judge
        if low in _GENERIC or re.match(r"^(chapter|part)\b[\s\divx]*$", low):
            return 0.18

        words = low.split()
        nw = len(words)
        if nw > 12:
            # a full sentence: almost certainly a prompt / first line, not a title
            return 0.15

        norm = _norm_text(text, ops)

        # speaker-label guard: "Eldrich The White Knight:" is a dialogue tag,
        # not a title, when its in-text occurrence is followed by a colon.
        if re.search(re.escape(low) + r"\s*:", norm):
            return 0.20

        appears = low in norm

        # alignment: title content words echoed in the body (first occurrence
        # of the title itself removed so it cannot vouch for itself)
        body = norm.replace(low, " ", 1) if appears else norm
        content = [w for w in re.findall(r"[a-z][a-z']+", low)
                   if w not in _STOP and len(w) >= 3]
        if content:
            hits = sum(
                1 for w in content
                if re.search(r"\b" + re.escape(w), body) is not None)
            overlap = hits / float(len(content))
        else:
            overlap = 0.5  # all-stopword title: neutral

        val = 0.30
        if appears:
            val += 0.20
        if 1 <= nw <= 8:
            val += 0.10
        elif nw <= 12:
            val += 0.03
        val += 0.40 * overlap

        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
