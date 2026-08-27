"""p903_v0_keyword -- Corpus distinctiveness, surface/lexical heuristic.

Criterion: the release's content is distinctive relative to other releases
in the collection (not near-duplicate template / boilerplate recycled
across many similar announcements).

We cannot see the rest of the corpus, so we proxy "reads like many others"
with the density of stock press-release phrasing: wire-service slugs,
formulaic announcement cliches, legal safe-harbor blocks, and site/footer
chrome that recur nearly verbatim across templated releases.  A document
saturated with these phrases is, with high probability, a near-duplicate of
many siblings; a document nearly free of them is one-of-a-kind content.

score = exp(-weighted_boilerplate_hits_per_1000_words / 18)
"""

import math
import re

# --- mojibake / entity normalization -------------------------------------
# Sequences written as unicode escapes (UTF-8 bytes mis-decoded as cp1252).
# Longer sequences MUST precede the bare "a-hat euro" fallback.
_MOJIBAKE = [
    ("â€œ", '"'),   # curly left double quote
    ("â€", '"'),   # curly right double quote
    ("â€™", "'"),   # curly right apostrophe
    ("â€˜", "'"),   # curly left apostrophe
    ("â€“", "-"),   # en dash
    ("â€”", "-"),   # em dash
    ("â€¦", "..."), # ellipsis
    ("â€", '"'),         # bare remnant (often right double quote)
    ("Â ", " "),         # mangled non-breaking space
    ("Â", ""),                # stray A-circumflex
    (" ", " "),               # real non-breaking space
    ("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "),
    ("&gt;", ">"), ("&lt;", "<"), ("&rsquo;", "'"), ("&lsquo;", "'"),
    ("&rdquo;", '"'), ("&ldquo;", '"'), ("&ndash;", "-"), ("&mdash;", "-"),
]


def _normalize(text):
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    text = text.replace("[...]", " ")
    return text


# --- boilerplate phrase bank (pattern, weight) ----------------------------
# weight 3: hard template markers (wire slugs, legal blocks)
# weight 2: stock PR announcement cliches
# weight 1: site/footer chrome that recycles across scraped pages
_PHRASES = [
    (r"pr\s*newswire", 3), (r"business\s*wire", 3), (r"globe\s*newswire", 3),
    (r"marketwired", 3), (r"accesswire", 3), (r"news\s*wire", 2),
    (r"forward[-\s]looking\s+statements?", 3), (r"safe\s+harbor", 3),
    (r"undue\s+reliance", 3), (r"no\s+obligation\s+to\s+(?:publicly\s+)?update", 3),
    (r"risks\s+and\s+uncertainties", 3), (r"all\s+rights\s+reserved", 3),
    (r"securities\s+and\s+exchange\s+commission", 2), (r"form\s+10-[kq]", 2),
    (r"(?:is\s+)?pleased\s+to\s+announce", 2), (r"proud\s+to\s+announce", 2),
    (r"today\s+announced", 2), (r"announced\s+today", 2),
    (r"leading\s+(?:provider|supplier|manufacturer|developer)", 2),
    (r"(?:global|world)\s+leader", 2), (r"industry[-\s]leading", 2),
    (r"best[-\s]in[-\s]class", 2), (r"state[-\s]of[-\s]the[-\s]art", 2),
    (r"cutting[-\s]edge", 2), (r"award[-\s]winning", 2),
    (r"is\s+a\s+leading", 2), (r"one\s+of\s+the\s+(?:largest|leading)", 2),
    (r"headquartered\s+in", 2), (r"strategic\s+partnership", 2),
    (r"about\s+[A-Z][\w&.\-]*\s*[:\n]", 2),
    (r"skip\s+to\s+(?:main\s+)?content", 2), (r"privacy\s+policy", 1),
    (r"terms\s+(?:of\s+use|and\s+conditions)", 1), (r"cookies?", 1),
    (r"sign\s+up", 1), (r"subscribe", 1), (r"follow\s+us", 1),
    (r"read\s+more", 1), (r"learn\s+more", 1), (r"click\s+here", 1),
    (r"contact\s+us", 1), (r"media\s+contacts?", 2),
    (r"investor\s+relations", 2), (r"press\s+releases?", 1),
    (r"for\s+(?:more|further)\s+information", 2), (r"to\s+learn\s+more", 1),
    (r"related\s+(?:articles|news|links)", 1), (r"share\s+this", 1),
    (r"back\s+to\s+top", 1), (r"log\s*in", 1), (r"log\s*out", 1),
    (r"visit\s+(?:us\s+at\s+)?www\.", 1), (r"©|copyright", 1),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), w) for p, w in _PHRASES]

_PER_PHRASE_CAP = 8      # one footer word repeated 40x still counts as 8
_DECAY = 18.0            # density (per 1k words) giving score ~0.37


def score(text: str) -> float:
    try:
        text = _normalize(str(text))
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
        n_words = len(words)
        if n_words < 30:
            return 0.5
        total = 0.0
        for rx, w in _COMPILED:
            hits = min(len(rx.findall(text)), _PER_PHRASE_CAP)
            total += w * hits
        density = 1000.0 * total / max(n_words, 50)
        s = math.exp(-density / _DECAY)
        return max(0.0, min(1.0, float(s)))
    except Exception:
        return 0.5


if __name__ == "__main__":
    print(score("Acme Corp today announced a strategic partnership. "
                "Forward-looking statements... For more information visit www.acme.com"))
