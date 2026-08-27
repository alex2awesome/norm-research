# Hybrid module for humor unit u15: "Medium/Format Fit"
# Construct division: code detects concrete surface markers of an adopted
# format (mock-document headers like "Dear Diary"/dateline stamps, mock-ad
# copy like "Attention shoppers"/ALL-CAPS slogans, canonical Q&A joke shape,
# faux-news datelines) and measures whether that format's surface cues
# recur consistently across the piece rather than appearing once and being
# dropped. The two LLM fields carry what code cannot classify: which format
# the piece is actually imitating (open-vocabulary), and whether the piece
# commits to that format's conventions all the way through or abandons them.

import re

LLM_FIELDS = {
    "imitated_format": (
        "In <=8 words, name the format/medium this piece parodies or adopts "
        "(e.g. 'diary entry', 'radio ad', 'classic Q&A joke', 'news report'), "
        "or say NONE if it is just a plain joke/story."
    ),
    "format_committed": (
        "In <=10 words: does the piece maintain that format's conventions "
        "consistently throughout, or drop them partway? Answer 'consistent', "
        "'inconsistent', or 'n/a'."
    ),
}

_DIARY_RE = re.compile(r"\b(dear diary|dear journal)\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\s*[:\-]", re.IGNORECASE)
_AD_RE = re.compile(
    r"\b(attention (?:shoppers|customers)|now (?:only|available)|call now|"
    r"limited time|order (?:now|today)|our (?:clients|customers))\b",
    re.IGNORECASE,
)
_NEWS_RE = re.compile(
    r"\b(breaking news|reports? that|sources say|in a statement|according to)\b",
    re.IGNORECASE,
)
_QA_JOKE_RE = re.compile(
    r"^\s*(why (?:did|do|does|is)|what (?:did|do)|how (?:many|do))", re.IGNORECASE
)
_LETTER_RE = re.compile(r"\bdear\s+\w+\s*[,:]", re.IGNORECASE)

_FORMAT_PATTERNS = {
    "diary": _DIARY_RE,
    "ad": _AD_RE,
    "news": _NEWS_RE,
    "letter": _LETTER_RE,
}


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        # --- code: which format markers appear, and how many times (recurrence
        # is the operational proxy for "committed to the format", as opposed
        # to a single stray phrase) ---
        best_family, best_hits = None, 0
        for name, pat in _FORMAT_PATTERNS.items():
            hits = len(pat.findall(t))
            if hits > best_hits:
                best_family, best_hits = name, hits

        qa_hit = 1.0 if _QA_JOKE_RE.search(t.strip()) else 0.0

        if best_hits >= 2:
            format_commitment = 1.0
        elif best_hits == 1:
            format_commitment = 0.6
        elif qa_hit:
            format_commitment = 0.75  # canonical joke format, self-consistent by construction
        else:
            format_commitment = 0.5  # no detectable adopted format; not penalized, just neutral

        # --- code: sentence-length regularity as a coherence-with-format proxy
        # (a committed format tends to repeat a structural unit: entries,
        # slogans, exchanges) ---
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        if len(sents) >= 3:
            lens = [len(re.findall(r"\w+", s)) for s in sents]
            mean_len = sum(lens) / len(lens)
            var = sum((l - mean_len) ** 2 for l in lens) / len(lens) if mean_len else 0.0
            cv = (var ** 0.5) / mean_len if mean_len else 1.0
            coherence = max(0.3, 1.0 - min(cv, 1.5) / 1.5 * 0.5)
        else:
            coherence = 0.7

        # --- LLM-field grounding ---
        extracted = extracted or {}
        imitated_format = str(extracted.get("imitated_format", "") or "").strip().lower()
        format_committed = str(extracted.get("format_committed", "") or "").strip().lower()

        has_format = imitated_format and imitated_format not in ("none", "n/a", "")
        if has_format:
            if format_committed.startswith("consistent"):
                llm_component = 1.0
            elif format_committed.startswith("inconsistent"):
                llm_component = 0.2
            else:
                llm_component = 0.6
        else:
            llm_component = 0.55  # plain joke, no imitated format to judge fit against

        combined = 0.35 * format_commitment + 0.15 * coherence + 0.50 * llm_component
        return max(0.0, min(1.0, combined))
    except Exception:
        return 0.5
