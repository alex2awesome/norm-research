"""p903_v1_structure -- Corpus distinctiveness, structural/positional heuristic.

Criterion: the release is distinctive relative to the collection, not a
near-duplicate template recycled across many similar announcements.

Corpus access is impossible, so we read the document's ANATOMY.  Templated
releases (and scraped chrome-heavy pages) share a recognizable skeleton:
big navigation blocks of short link-like lines at top/bottom, lines that
repeat verbatim inside the page, and a standard tail (About X / contacts /
forward-looking legal) that begins early and swallows much of the document.
Distinctive one-of-a-kind content instead carries a long, varied run of
prose paragraphs before any standard tail appears.

Components (each in [0,1], higher = more distinctive):
  dup_inv    - 1 - fraction of substantive lines duplicated within the page
  nav_inv    - 1 - fraction of lines that look like nav/link chrome
  tail_pos   - how LATE the standard boilerplate tail begins (relative offset)
  prose_mass - fraction of characters living in real prose paragraphs
  para_var   - coefficient of variation of prose-paragraph lengths
score = 0.25*dup_inv + 0.20*nav_inv + 0.20*tail_pos + 0.25*prose_mass + 0.10*para_var
"""

import re
import statistics
from collections import Counter

_MOJIBAKE = [
    ("\xe2€œ", '"'), ("\xe2€\x9d", '"'),
    ("\xe2€™", "'"), ("\xe2€˜", "'"),
    ("\xe2€“", "-"), ("\xe2€”", "-"),
    ("\xe2€\xa6", "..."), ("\xe2€", '"'),
    ("\xc2\xa0", " "), ("\xc2", ""), ("\xa0", " "),
    ("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "),
    ("&gt;", ">"), ("&lt;", "<"), ("&rsquo;", "'"), ("&lsquo;", "'"),
    ("&rdquo;", '"'), ("&ldquo;", '"'), ("&ndash;", "-"), ("&mdash;", "-"),
]

# markers that open the standard recycled tail of a press release
_TAIL_PATTERNS = [
    re.compile(r"^\s*about\s+[A-Z][\w&.\-]*", re.MULTILINE),
    re.compile(r"forward[-\s]looking\s+statements?", re.IGNORECASE),
    re.compile(r"safe\s+harbor", re.IGNORECASE),
    re.compile(r"media\s+contacts?", re.IGNORECASE),
    re.compile(r"press\s+contacts?", re.IGNORECASE),
    re.compile(r"investor\s+(?:relations|contacts?)", re.IGNORECASE),
    re.compile(r"for\s+(?:more|further)\s+information", re.IGNORECASE),
    re.compile(r"^\s*contacts?\s*:?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"###"),
    re.compile(r"^\s*SOURCE\s+[A-Z]", re.MULTILINE),
    re.compile(r"all\s+rights\s+reserved", re.IGNORECASE),
]

_SENT_PUNCT = re.compile(r"[.!?]")


def _normalize(text):
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    return text.replace("[...]", " ")


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def score(text: str) -> float:
    try:
        text = _normalize(str(text))
        if len(text.strip()) < 200:
            return 0.5
        lines = [ln.strip() for ln in text.split("\n")]
        lines = [ln for ln in lines if ln]
        if not lines:
            return 0.5

        # --- duplicate-line fraction (verbatim internal recycling) --------
        subst = [ln.lower() for ln in lines if len(ln.split()) >= 3]
        if subst:
            counts = Counter(subst)
            dup_frac = sum(c for c in counts.values() if c > 1) / len(subst)
        else:
            dup_frac = 1.0
        dup_inv = 1.0 - _clamp(dup_frac / 0.40)

        # --- nav/link-chrome line fraction ---------------------------------
        def is_navlike(ln):
            w = ln.split()
            if not (1 <= len(w) <= 4):
                return False
            if _SENT_PUNCT.search(ln):
                return False
            caps = sum(1 for t in w if t[:1].isupper() or t[:1].isdigit())
            return caps >= max(1, len(w) - 1)
        nav_frac = sum(1 for ln in lines if is_navlike(ln)) / len(lines)
        nav_inv = 1.0 - _clamp(nav_frac / 0.60)

        # --- onset of the standard boilerplate tail ------------------------
        n = len(text)
        onsets = []
        for rx in _TAIL_PATTERNS:
            m = rx.search(text)
            if m:
                onsets.append(m.start() / n)
        if onsets:
            tail_pos = _clamp((min(onsets) - 0.30) / 0.60)
        else:
            tail_pos = 0.70  # no template tail at all: mildly distinctive

        # --- prose-paragraph mass -------------------------------------------
        prose = [ln for ln in lines
                 if len(ln.split()) >= 25 and _SENT_PUNCT.search(ln)]
        prose_chars = sum(len(ln) for ln in prose)
        total_chars = sum(len(ln) for ln in lines)
        mass = prose_chars / total_chars if total_chars else 0.0
        prose_mass = _clamp(mass / 0.60)

        # --- prose paragraph length variability -----------------------------
        wc = [len(ln.split()) for ln in prose]
        if len(wc) >= 3 and statistics.mean(wc) > 0:
            cv = statistics.pstdev(wc) / statistics.mean(wc)
            para_var = _clamp(cv / 0.80)
        else:
            para_var = 0.35

        s = (0.25 * dup_inv + 0.20 * nav_inv + 0.20 * tail_pos
             + 0.25 * prose_mass + 0.10 * para_var)
        return _clamp(float(s))
    except Exception:
        return 0.5


if __name__ == "__main__":
    demo = ("Home\nProducts\nNews\n" * 5
            + "The town council of Millbrook voted Tuesday on an unusual plan "
              "that residents had debated for months, involving the old mill.\n"
            + "About Millbrook\nMedia Contact\n")
    print(score(demo))
