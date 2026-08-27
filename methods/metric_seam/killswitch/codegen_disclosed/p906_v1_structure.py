"""p906 v1 -- Persuasive cadence, structural/positional heuristic.

Criterion: "The prose rhythm builds momentum across paragraphs:
sentence-length variation, paragraph pacing, and transitions sustain the
reader's attention from the headline through the closing boilerplate."

Approach (document structure & position):
  * segment the page into paragraph-like blocks and classify each as
    PROSE vs CHROME (nav menus, link lists, cookie banners),
  * measure the "arc": prose present early (headline/lede zone), a
    sustained contiguous prose run in the middle, and boilerplate
    markers ("About X", contact block, ###/SOURCE) confined to the tail,
  * measure paragraph pacing: variation + alternation of prose-paragraph
    lengths across document positions (momentum = successive paragraphs
    that change gear rather than repeat the same size),
  * measure paragraph-initial transitions (connectives placed at the
    head of a paragraph, i.e., positionally doing the bridging work).

Contract: score(text) -> float in [0, 1]; deterministic; never raises.
"""

import re
import math
import statistics
from collections import Counter

# ---------------------------------------------------------------- cleanup

_REPL = [
    ("â€œ", '"'), ("â€\x9d", '"'), ("â€™", "'"), ("â€˜", "'"),
    ("â€”", " -- "), ("â€“", " - "), ("â€¦", "..."), ("â€¢", " "),
    ("Â\xa0", " "), ("Â ", " "), ("Â", ""), ("\xa0", " "),
    ("&amp;", "&"), ("&nbsp;", " "), ("&quot;", '"'), ("&#39;", "'"),
    ("&apos;", "'"), ("&lt;", "<"), ("&gt;", ">"), ("&rsquo;", "'"),
    ("&lsquo;", "'"), ("&ldquo;", '"'), ("&rdquo;", '"'),
    ("&mdash;", " -- "), ("&ndash;", " - "), ("&hellip;", "..."),
]


def _clean(text):
    for bad, good in _REPL:
        text = text.replace(bad, good)
    text = text.replace("[...]", "\n\n")
    return text


# ----------------------------------------------------------- block carving

def _blocks(text):
    """Merge adjacent non-empty lines into paragraph blocks."""
    blocks, cur = [], []
    for rawline in text.split("\n"):
        line = rawline.strip()
        if not line:
            if cur:
                blocks.append(" ".join(cur))
                cur = []
            continue
        # very short fragments end a block on their own
        cur.append(line)
        if len(line) < 30 and not re.search(r"[.!?,;:]$", line):
            blocks.append(" ".join(cur))
            cur = []
    if cur:
        blocks.append(" ".join(cur))
    return blocks


def _is_prose(block):
    words = block.split()
    if len(words) < 15:
        return False
    if not re.search(r"[.!?]", block):
        return False
    lower = sum(1 for w in words if w[:1].islower())
    if lower / len(words) < 0.35:          # Title Case link soup
        return False
    if len(re.findall(r"\|", block)) > 3:  # menu separators
        return False
    return True


_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z\"'(])")

_PARA_TRANSITION = re.compile(
    r"^(however|moreover|furthermore|additionally|in addition|meanwhile|"
    r"as a result|therefore|thus|consequently|in fact|indeed|notably|"
    r"importantly|building on|at the same time|for example|for instance|"
    r"beyond|looking ahead|going forward|together|finally|ultimately|"
    r"first|second|third|next|now|today|since then|to that end|"
    r"the (?:new|latest|combined|expanded)\b|this\b|that\b|these\b|"
    r"with\b|under\b|as\b)",
    re.IGNORECASE,
)

_BOILERPLATE = re.compile(
    r"(^|\s)(about [A-Z][\w&.-]+|media contact|press contact|"
    r"investor relations|for (?:more|further) information|"
    r"forward-looking statements|source:|###|contact:)",
    re.IGNORECASE,
)


def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        text = _clean(text)
        blocks = _blocks(text)
        if not blocks:
            return 0.0
        flags = [_is_prose(b) for b in blocks]
        prose_idx = [i for i, f in enumerate(flags) if f]
        n_blocks = len(blocks)
        n_prose = len(prose_idx)
        if n_prose < 2:
            return 0.1  # no paragraph sequence -> no cross-paragraph cadence

        prose = [blocks[i] for i in prose_idx]
        total_words = sum(len(b.split()) for b in blocks)
        prose_words = sum(len(b.split()) for b in prose)

        # --- signal 1: sustained contiguous prose run --------------------
        best_run, run = 0, 0
        for f in flags:
            run = run + 1 if f else 0
            best_run = max(best_run, run)
        s_run = min(1.0, best_run / 5.0)  # 5+ consecutive prose paras = full

        # --- signal 2: arc from headline zone to boilerplate tail --------
        first_pos = prose_idx[0] / n_blocks           # prose starts early?
        early = max(0.0, 1.0 - first_pos / 0.5)       # 0 if starts past midpoint
        tail_zone = text[int(len(text) * 0.70):]
        head_zone = text[: int(len(text) * 0.50)]
        bp_tail = 1.0 if _BOILERPLATE.search(tail_zone) else 0.0
        bp_head = 1.0 if _BOILERPLATE.search(head_zone) else 0.0
        s_arc = 0.6 * early + 0.4 * bp_tail
        if bp_head and not bp_tail:
            s_arc *= 0.5  # boilerplate sitting up top = broken arc

        # --- signal 3: paragraph pacing (variation + alternation) --------
        lens = [len(b.split()) for b in prose]
        if len(lens) >= 3 and statistics.mean(lens) > 0:
            mean_len = statistics.mean(lens)
            cv = statistics.pstdev(lens) / mean_len
            diffs = [abs(a - b) for a, b in zip(lens, lens[1:])]
            alt = (statistics.mean(diffs) / mean_len) if diffs else 0.0
            s_pace = 0.5 * _band(cv, 0.05, 0.20, 0.80, 1.40) \
                   + 0.5 * _band(alt, 0.05, 0.20, 0.90, 1.60)
        else:
            s_pace = 0.3

        # --- signal 4: sentence-length gearing across paragraphs ---------
        para_means = []
        for b in prose:
            sl = [len(s.split()) for s in _SENT_SPLIT.split(b) if s.split()]
            if sl:
                para_means.append(statistics.mean(sl))
        if len(para_means) >= 3 and statistics.mean(para_means) > 0:
            gcv = statistics.pstdev(para_means) / statistics.mean(para_means)
            s_gear = _band(gcv, 0.02, 0.10, 0.55, 0.95)
        else:
            s_gear = 0.3

        # --- signal 5: paragraph-initial transitions ----------------------
        heads = sum(1 for b in prose if _PARA_TRANSITION.match(b))
        s_trans = _band(heads / n_prose, 0.05, 0.20, 0.85, 1.01)

        # --- signal 6: page is mostly prose (chrome starves cadence) -----
        s_prose = min(1.0, (prose_words / max(1, total_words)) / 0.55)

        raw = (0.20 * s_run + 0.15 * s_arc + 0.22 * s_pace
               + 0.18 * s_gear + 0.13 * s_trans + 0.12 * s_prose)
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5


def _band(x, lo0, lo, hi, hi0):
    """Trapezoid: 0 below lo0, ramp to 1 at lo, flat to hi, ramp to 0 at hi0."""
    if x <= lo0 or x >= hi0:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return (x - lo0) / (lo - lo0)
    return (hi0 - x) / (hi0 - hi)
