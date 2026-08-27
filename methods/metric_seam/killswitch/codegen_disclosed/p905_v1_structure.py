"""p905_v1_structure -- Authentic authorship, structural/positional heuristic.

Criterion: the release reads as written by someone with genuine familiarity
with the company and subject matter, not assembled from generic template
marketing language.

Approach (structure/position only): a genuinely authored release is
dominated by contiguous prose paragraphs, opens with a dateline/ticker,
carries named-quote attributions in its middle, keeps contact/"About"
boilerplate at the tail (not the head), and shows natural sentence-length
variation. Scraped nav chrome (many short link-label lines), template
placeholders ({...}), and boilerplate-dominated pages score low. Each
sub-signal is scored in [0,1] from WHERE things appear and combined as a
weighted sum.
"""

import re
import statistics

# --- mojibake / entity cleanup (scraped-corpus hazard) ---------------------
_MOJI = [
    ("â€œ", '"'), ("â€\x9d", '"'), ("â€\x99", "'"), ("â€\x98", "'"),
    ("â€“", "-"), ("â€”", "-"), ("â€¦", "..."), ("â€¢", "*"),
    ("Â ", " "), ("Â", ""),
    ("&amp;", "&"), ("&nbsp;", " "), ("&quot;", '"'),
    ("&#39;", "'"), ("&lt;", "<"), ("&gt;", ">"),
]
_MOJI_LEFTOVER = re.compile("â€.")

_MONTH = (r"(?:January|February|March|April|May|June|July|August|September|"
          r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)")
_DATELINE = re.compile(
    r"(?:\((?:NYSE|NASDAQ|AMEX|TSX[V]?|LSE|OTC\w*|Euronext)\s*:\s*[A-Z.\-]{1,8}\))"
    r"|(?:\b[A-Z][A-Za-z.\- ]{1,30},\s*(?:[A-Z][a-z]+\.?,?\s+)?" + _MONTH +
    r"\.?\s+\d{1,2},?\s+(?:19|20)\d{2})"
    r"|(?:" + _MONTH + r"\.?\s+\d{1,2},?\s+(?:19|20)\d{2}\s*[-/]+)")
_QUOTE_ATTR = re.compile(
    r'["\']\s*,?\s*said\b'
    r"|\bsaid\s+[A-Z][a-z]+"
    r"|\b[A-Z][a-z]+\s+[A-Z][a-z]+\s*,\s*(?:the\s+)?(?:CEO|CFO|CTO|COO|President|"
    r"Chief|Vice\s+President|VP|Director|Founder|Head\s+of|General\s+Manager)\b")
_BOILER = re.compile(
    r"\b(?:About\s+[A-Z][\w&.\- ]{1,40}\n|For\s+(?:more|further|additional)\s+information|"
    r"Media\s+Contact|Press\s+Contact|Investor\s+(?:Relations|Contact)|"
    r"Contact\s*:|SOURCE\s+[A-Z]|forward-looking\s+statements?)\b",
    re.IGNORECASE)
_PLACEHOLDER = re.compile(r"\{\{?[\w:./ -]{1,40}\}?\}|\[if\b|\[endif\]|<!--")
_EMAIL_PHONE = re.compile(
    r"[\w.\-]+@[\w.\-]+\.\w{2,}|\(?\+?\d{1,3}[)\s.\-]?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]?\d{2,4}")


def _normalize(text):
    for bad, good in _MOJI:
        text = text.replace(bad, good)
    text = _MOJI_LEFTOVER.sub("'", text)
    text = text.replace("[...]", "\n")
    return text


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or len(text.strip()) < 200:
            return 0.5
        t = _normalize(text)
        n = len(t)

        # ---- line classification: prose paragraphs vs nav/label chrome ----
        lines = t.split("\n")
        offsets = []          # (start_offset, line)
        pos = 0
        for ln in lines:
            offsets.append((pos, ln))
            pos += len(ln) + 1

        prose_chars = 0
        total_chars = 0
        prose_flags = []
        for off, ln in offsets:
            s = ln.strip()
            if not s:
                prose_flags.append(False)
                continue
            total_chars += len(s)
            is_prose = (len(s.split()) >= 12) and bool(re.search(r"[.!?]", s))
            prose_flags.append(is_prose)
            if is_prose:
                prose_chars += len(s)
        if total_chars < 200:
            return 0.5
        prose_frac = prose_chars / float(total_chars)

        # longest contiguous prose run (a real body, not scattered snippets)
        best_run = 0
        run = 0
        for flag, (off, ln) in zip(prose_flags, offsets):
            if flag:
                run += len(ln)
                best_run = max(best_run, run)
            else:
                if len(ln.strip()) > 0:
                    run = 0
        run_frac = min(1.0, best_run / float(max(total_chars, 1)))

        # ---- positional signals -------------------------------------------
        # dateline / ticker in the opening 30%
        dl = _DATELINE.search(t, 0, max(1, int(n * 0.30)))
        dateline_sub = 1.0 if dl else 0.0

        # named quote attribution positioned inside the middle of the doc
        quote_sub = 0.0
        for m in _QUOTE_ATTR.finditer(t):
            rel = m.start() / float(n)
            if 0.10 <= rel <= 0.85:
                quote_sub = 1.0
                break
            quote_sub = max(quote_sub, 0.4)   # present but oddly placed

        # boilerplate / contact placement: tail is normal, head is chrome
        boiler_head = 0
        boiler_tail = 0
        for m in list(_BOILER.finditer(t)) + list(_EMAIL_PHONE.finditer(t)):
            rel = m.start() / float(n)
            if rel < 0.30:
                boiler_head += 1
            elif rel > 0.70:
                boiler_tail += 1
        tail_sub = 1.0 if (boiler_tail > 0 and boiler_head == 0) else \
                   (0.5 if boiler_tail == boiler_head == 0 else 0.0)

        # ---- coherence: dominant capitalized token spans the document -----
        caps = [(m.start(), m.group(0)) for m in
                re.finditer(r"\b[A-Z][A-Za-z]{2,}\b", t)]
        stop = {"The", "This", "That", "These", "Those", "And", "But", "For",
                "With", "From", "About", "Our", "Your", "Their", "New", "More",
                "All", "Are", "Was", "Has", "Have", "Will", "Can", "May",
                "Not", "One", "Two", "How", "What", "When", "Where", "Why",
                "Home", "Contact", "News", "Login", "Search", "Read",
                "Skip", "Menu", "Privacy", "Terms", "Cookie", "Cookies"}
        counts = {}
        for off, w in caps:
            if w in stop:
                continue
            counts.setdefault(w, []).append(off)
        coher_sub = 0.0
        if counts:
            top = max(counts.items(), key=lambda kv: (len(kv[1]), kv[0]))
            positions = top[1]
            if len(positions) >= 3:
                thirds = {int(3 * p / (n + 1)) for p in positions}
                coher_sub = (min(len(positions), 8) / 8.0) * (len(thirds) / 3.0)

        # ---- sentence-length variation inside prose ------------------------
        prose_text = " ".join(ln.strip() for flag, (o, ln)
                              in zip(prose_flags, offsets) if flag)
        sents = [s for s in re.split(r"(?<=[.!?])\s+", prose_text)
                 if len(s.split()) >= 3]
        var_sub = 0.0
        if len(sents) >= 4:
            lens = [len(s.split()) for s in sents]
            sd = statistics.pstdev(lens)
            var_sub = min(1.0, sd / 9.0)   # authored prose: sd ~ 8-12 words

        # ---- penalties ------------------------------------------------------
        n_ph = len(_PLACEHOLDER.findall(t))
        placeholder_pen = min(0.15, 0.05 * n_ph)
        # nav chrome: share of nonblank lines that are short link labels
        nonblank = [ln.strip() for ln in lines if ln.strip()]
        shorties = sum(1 for s in nonblank
                       if len(s.split()) <= 4 and not re.search(r"[.!?,]$", s))
        chrome_pen = 0.12 * (shorties / float(max(len(nonblank), 1)))

        val = (0.05
               + 0.34 * prose_frac
               + 0.12 * run_frac
               + 0.12 * dateline_sub
               + 0.13 * quote_sub
               + 0.08 * tail_sub
               + 0.08 * coher_sub
               + 0.08 * var_sub
               - placeholder_pen
               - chrome_pen)
        return max(0.0, min(1.0, float(val)))
    except Exception:
        return 0.5
