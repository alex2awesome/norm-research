"""p903_v2_holistic -- Corpus distinctiveness, composite of weak signals.

Criterion: distinctive, one-of-a-kind content scores high; text that reads
like many other releases in the collection (recycled template/boilerplate)
scores low.

Without corpus access, six within-document proxies are blended.  Templated
near-duplicates betray themselves through stock phrasing, flat vocabulary,
internal n-gram and line recycling, and an early-starting standard tail;
one-of-a-kind releases carry rich vocabulary, attributed quotations and a
diverse cast of named entities.

Components (each mapped to [0,1], higher = more distinctive):
  boil_inv  (0.22) - inverse density of stock press-release boilerplate
  richness  (0.22) - mean segmental type-token ratio, 100-word windows
  gram_inv  (0.18) - 1 - within-document repeated 5-gram rate
  dup_inv   (0.14) - 1 - duplicated-line fraction
  tail_pos  (0.12) - how late the standard boilerplate tail begins
  specific  (0.12) - attributed quotations + capitalized-bigram diversity
"""

import math
import re
import statistics
from collections import Counter

_MOJIBAKE = [
    ("â€œ", '"'), ("â€", '"'),
    ("â€™", "'"), ("â€˜", "'"),
    ("â€“", "-"), ("â€”", "-"),
    ("â€¦", "..."), ("â€", '"'),
    ("Â ", " "), ("Â", ""), (" ", " "),
    ("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "),
    ("&gt;", ">"), ("&lt;", "<"), ("&rsquo;", "'"), ("&lsquo;", "'"),
    ("&rdquo;", '"'), ("&ldquo;", '"'), ("&ndash;", "-"), ("&mdash;", "-"),
]

_BOILER = [re.compile(p, re.IGNORECASE) for p in (
    r"pr\s*newswire", r"business\s*wire", r"globe\s*newswire", r"accesswire",
    r"forward[-\s]looking\s+statements?", r"safe\s+harbor",
    r"undue\s+reliance", r"risks\s+and\s+uncertainties",
    r"all\s+rights\s+reserved", r"(?:is\s+)?pleased\s+to\s+announce",
    r"today\s+announced", r"announced\s+today",
    r"leading\s+(?:provider|supplier|manufacturer|developer)",
    r"(?:global|world)\s+leader", r"industry[-\s]leading",
    r"state[-\s]of[-\s]the[-\s]art", r"cutting[-\s]edge", r"award[-\s]winning",
    r"headquartered\s+in", r"for\s+(?:more|further)\s+information",
    r"media\s+contacts?", r"investor\s+relations",
    r"skip\s+to\s+(?:main\s+)?content", r"privacy\s+policy", r"cookies?",
    r"subscribe", r"follow\s+us", r"learn\s+more", r"read\s+more",
    r"contact\s+us", r"visit\s+www\.",
)]

_TAIL = [re.compile(p, f) for p, f in (
    (r"^\s*about\s+[A-Z][\w&.\-]*", re.MULTILINE),
    (r"forward[-\s]looking\s+statements?", re.IGNORECASE),
    (r"safe\s+harbor", re.IGNORECASE),
    (r"media\s+contacts?", re.IGNORECASE),
    (r"investor\s+(?:relations|contacts?)", re.IGNORECASE),
    (r"for\s+(?:more|further)\s+information", re.IGNORECASE),
    (r"###", 0),
    (r"^\s*SOURCE\s+[A-Z]", re.MULTILINE),
)]

_QUOTE = re.compile(
    r'"[^"\n]{25,400}"[\s,]*(?:said|says|stated|noted|added|commented|'
    r'according\s+to|explained|remarked)|(?:said|says|stated|noted|added|'
    r'commented|explained)[\s,:]*"[^"\n]{25,400}"', re.IGNORECASE)

_CAP_BIGRAM = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b")


def _normalize(text):
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    return text.replace("[...]", " ")


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def score(text: str) -> float:
    try:
        text = _normalize(str(text))
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text.lower())
        n_words = len(words)
        if n_words < 30:
            return 0.5

        # 1. inverse boilerplate density (per 1k words) ---------------------
        hits = sum(min(len(rx.findall(text)), 8) for rx in _BOILER)
        density = 1000.0 * hits / max(n_words, 50)
        boil_inv = math.exp(-density / 12.0)

        # 2. lexical richness: mean segmental TTR over 100-word windows -----
        win = 100
        if n_words >= win:
            ttrs = [len(set(words[i:i + win])) / float(win)
                    for i in range(0, n_words - win + 1, win)]
            msttr = statistics.mean(ttrs)
        else:
            msttr = len(set(words)) / float(n_words) * 0.85  # rough deflate
        richness = _clamp((msttr - 0.55) / 0.30)

        # 3. within-document 5-gram repetition ------------------------------
        if n_words >= 40:
            grams = Counter(tuple(words[i:i + 5]) for i in range(n_words - 4))
            repeated = sum(c for c in grams.values() if c > 1)
            rep_rate = repeated / float(n_words - 4)
        else:
            rep_rate = 0.0
        gram_inv = 1.0 - _clamp(rep_rate / 0.30)

        # 4. duplicated-line fraction ----------------------------------------
        lines = [ln.strip().lower() for ln in text.split("\n")
                 if len(ln.split()) >= 3]
        if lines:
            lc = Counter(lines)
            dup = sum(c for c in lc.values() if c > 1) / len(lines)
        else:
            dup = 1.0
        dup_inv = 1.0 - _clamp(dup / 0.40)

        # 5. late onset of standard tail --------------------------------------
        onsets = [rx.search(text).start() / len(text)
                  for rx in _TAIL if rx.search(text)]
        tail_pos = _clamp((min(onsets) - 0.30) / 0.60) if onsets else 0.70

        # 6. specificity: attributed quotes + named-entity diversity ----------
        quotes = len(_QUOTE.findall(text))
        q = min(quotes, 3) / 3.0
        cb = {m.group(0).lower() for m in _CAP_BIGRAM.finditer(text)}
        ent = _clamp((len(cb) / max(n_words, 1)) / 0.015)
        specific = 0.55 * q + 0.45 * ent

        s = (0.22 * boil_inv + 0.22 * richness + 0.18 * gram_inv
             + 0.14 * dup_inv + 0.12 * tail_pos + 0.12 * specific)
        return _clamp(float(s))
    except Exception:
        return 0.5


if __name__ == "__main__":
    print(score('The mayor unveiled a mural. "We never expected the flood to '
                'give us a canvas," said painter Ana Ruiz, laughing at the '
                'crowd gathered along Willow Creek at dawn on Saturday.'))
