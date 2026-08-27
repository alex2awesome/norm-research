"""Shared text normalization for ALL contest_corpus sources.

CRITICAL standing rule: identical normalization must be applied to text from
ALL sites so that source-specific typography (curly quotes etc.) cannot leak
the label. (A 0.996 fake AUC was once caused by typographic-quote leakage.)

Pipeline:
  1. unicodedata.normalize("NFKC", s)   (also folds nbsp, ellipsis, ligatures)
  2. curly quotes/apostrophes -> straight ASCII
  3. line-ending normalization, per-line trailing-space strip
  4. collapse runs of 3+ blank lines to 2 (preserves poem stanza breaks)
  5. strip wrapper quotes if the whole text is enclosed in a single pair

raw_text (the unnormalized extraction) is kept separately by the scrapers.
"""
import re
import unicodedata

_QUOTE_MAP = {
    "‘": "'",  # left single
    "’": "'",  # right single / apostrophe
    "‚": "'",  # single low-9
    "‛": "'",  # single high-reversed-9
    "′": "'",  # prime
    "“": '"',  # left double
    "”": '"',  # right double
    "„": '"',  # double low-9
    "‟": '"',  # double high-reversed-9
    "″": '"',  # double prime
    "«": '"',  # left guillemet
    "»": '"',  # right guillemet
}
_QUOTE_RE = re.compile("|".join(_QUOTE_MAP))


def normalize_text(s: str | None, strip_wrapper_quotes: bool = True) -> str | None:
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", s)
    s = _QUOTE_RE.sub(lambda m: _QUOTE_MAP[m.group(0)], s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # strip trailing whitespace per line (preserve leading indent for poems)
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    # collapse 3+ consecutive newlines to 2 (one blank line)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if strip_wrapper_quotes and len(s) > 1:
        for q in ('"', "'"):
            if s.startswith(q) and s.endswith(q) and s.count(q) == 2:
                s = s[1:-1].strip()
    return s
