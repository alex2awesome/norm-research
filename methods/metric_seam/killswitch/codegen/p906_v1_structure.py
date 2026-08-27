"""p906 v1 -- Persuasive cadence: structural/positional approach.

Criterion: prose rhythm builds momentum across paragraphs; sentence-length
variation, paragraph pacing, and transitions sustain the reader's attention
from headline through closing boilerplate.

This variant looks at WHERE things sit and how their sizes move:
  * filters navigation chrome lines and trailing contact/boilerplate blocks,
  * detects a headline position and a structured close,
  * measures sentence-length variation (coefficient of variation band),
  * measures long/short alternation between adjacent sentences,
  * measures paragraph count and paragraph-size pacing across the body.

score(text) -> float in [0.0, 1.0]; deterministic; 0.5 on unexpected error.
"""

import re
import math
import statistics

# Mojibake repairs (UTF-8 bytes mis-decoded as cp1252). Order matters:
# three-char sequences first, then the two-char right-double-quote remnant.
# Strings are built with chr() because several code points are invisible.
_EU = chr(0xE2) + chr(0x20AC)  # mis-decoded 0xE2 0x80 prefix (a-hat, euro)
_MOJIBAKE = (
    (_EU + chr(0x0153), '"'),    # +oe ligature  -> left curly double quote
    (_EU + chr(0x009D), '"'),    # +invisible 9D -> right curly double quote
    (_EU + chr(0x2122), "'"),    # +trademark    -> apostrophe / right single
    (_EU + chr(0x02DC), "'"),    # +small tilde  -> left single quote
    (_EU + chr(0x201C), " - "),  # +left dquote  -> en dash
    (_EU + chr(0x201D), " - "),  # +right dquote -> em dash
    (_EU + chr(0x00A6), " . "),  # +broken bar   -> ellipsis
    (_EU, '"'),                  # bare remnant: right dquote, 9D stripped
    (chr(0xC2) + chr(0xA0), " "),  # A-circumflex + non-breaking space
    (chr(0xC2), ""),               # stray A-circumflex
    (chr(0xA0), " "),              # remaining non-breaking space
    (chr(0x9D), ""),               # stray invisible control char
)

_ELISION_RE = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]")
_MULTIDOT_RE = re.compile(r"\.{2,}")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

_CHROME_LINE_RE = re.compile(
    r"^\s*(?:home|about(?:\s+us)?|contact(?:\s+us)?|news(?:room)?|products?"
    r"|services|solutions|pricing|blog|careers|events|resources|support"
    r"|sign\s?in|log\s?in|register|subscribe|search|menu|skip to content"
    r"|privacy policy|terms(?:\s+of\s+(?:use|service))?|cookie[s]?(?:\s+\w+)?"
    r"|follow us|share(?:\s+this)?|©.*|copyright.*)\s*$",
    re.IGNORECASE,
)
_BOILER_RE = re.compile(
    r"(?:media contact|press contact|investor contact|contact:|about\s+[A-Z]"
    r"|for (?:more|further) information|visit\s+(?:www\.|http)"
    r"|[\w.+-]+@[\w-]+\.\w{2,}|\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}"
    r"|^###\s*$|^\s*SOURCE\b|forward-looking statements)",
    re.IGNORECASE | re.MULTILINE,
)
_ABBR_RE = re.compile(
    r"\b(Inc|Corp|Co|Ltd|LLC|Mr|Mrs|Ms|Dr|St|No|vs|Jr|Sr|U\.S|U\.K|approx)\."
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(A-Z0-9])")


def _normalize(t):
    for bad, good in _MOJIBAKE:
        t = t.replace(bad, good)
    t = _ELISION_RE.sub("\n\n", t)
    t = _MULTIDOT_RE.sub(".", t)
    return t


def _is_chrome_line(line):
    s = line.strip()
    if not s:
        return True
    if _CHROME_LINE_RE.match(s):
        return True
    if s.count("|") >= 2:
        return True
    words = _WORD_RE.findall(s)
    if not re.search(r"[.!?]\s*$", s) and len(words) <= 6:
        return True
    if s.isupper() and len(words) <= 8:
        return True
    if re.match(r"^(?:https?://|www\.)\S+$", s):
        return True
    return False


def _sentences(paragraph):
    p = re.sub(r"\s+", " ", paragraph).strip()
    p = _ABBR_RE.sub(lambda m: m.group(1) + "<DOT>", p)
    parts = _SENT_SPLIT_RE.split(p)
    out = []
    for s in parts:
        s = s.replace("<DOT>", ".").strip()
        if re.search(r"[A-Za-z]", s):
            out.append(s)
    return out


def _cv_band(cv):
    if cv <= 0.0:
        return 0.0
    if cv < 0.35:
        return cv / 0.35
    if cv <= 0.90:
        return 1.0
    if cv >= 1.80:
        return 0.25
    return 1.0 - 0.75 * (cv - 0.90) / 0.90


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        t = _normalize(text)

        raw_paras = [p for p in re.split(r"\n\s*\n+", t) if p.strip()]
        paras = []
        for p in raw_paras:
            lines = [ln for ln in p.split("\n") if ln.strip()]
            kept = [ln for ln in lines if not _is_chrome_line(ln)]
            if not kept:
                continue
            if len(kept) < max(1, len(lines) // 2):
                continue  # mostly-chrome block
            paras.append(" ".join(kept))
        if not paras:
            return 0.02

        # Headline: first surviving block, one shortish line, no terminal '.'
        headline = False
        if paras:
            first = paras[0].strip()
            wn = len(_WORD_RE.findall(first))
            if 3 <= wn <= 20 and not re.search(r"[.!?]\s*$", first):
                headline = True
                paras = paras[1:]

        # Trailing boilerplate: peel matching blocks off the END only.
        boiler_close = False
        while paras and len(paras) > 1 and _BOILER_RE.search(paras[-1]):
            boiler_close = True
            paras.pop()

        body = paras
        if not body:
            return 0.05

        sent_lens = []
        para_word_counts = []
        para_sent_counts = []
        for p in body:
            sents = _sentences(p)
            para_sent_counts.append(len(sents))
            para_word_counts.append(len(_WORD_RE.findall(p)))
            for s in sents:
                n = len(_WORD_RE.findall(s))
                if n > 0:
                    sent_lens.append(n)

        body_words = sum(para_word_counts)
        if body_words == 0 or not sent_lens:
            return 0.02

        # 1) sentence-length variation
        if len(sent_lens) >= 3:
            mean_len = statistics.mean(sent_lens)
            cv = (statistics.pstdev(sent_lens) / mean_len) if mean_len else 0.0
            s_var = _cv_band(cv)
        else:
            s_var = 0.1

        # 2) long/short alternation between adjacent sentences
        if len(sent_lens) >= 4:
            flips = 0
            pairs = 0
            for a, b in zip(sent_lens, sent_lens[1:]):
                pairs += 1
                if abs(a - b) / float(max(a, b)) >= 0.30:
                    flips += 1
            s_alt = min(1.0, (flips / float(pairs)) / 0.55)
        else:
            s_alt = 0.1

        # 3) paragraph count and pacing
        n_paras = len(body)
        base = {0: 0.0, 1: 0.25, 2: 0.55, 3: 0.80}.get(n_paras, 1.0)
        if n_paras >= 2:
            mp = statistics.mean(para_word_counts)
            cv_p = (statistics.pstdev(para_word_counts) / mp) if mp else 0.0
            pace = min(1.0, 0.5 + cv_p)
        else:
            pace = 0.5
        s_para = base * pace

        # 4) shape: headline on top, hook up front, structured close
        s_shape = 0.0
        if headline:
            s_shape += 0.4
        if n_paras >= 2 and para_word_counts[0] < statistics.median(
                para_word_counts):
            s_shape += 0.3  # short lead paragraph = hook
        if boiler_close or (n_paras >= 3 and para_word_counts[-1]
                            == min(para_word_counts)):
            s_shape += 0.3  # release closes into boilerplate / tapers off
        s_shape = min(1.0, s_shape)

        gate = min(1.0, body_words / 60.0)
        raw = (0.35 * s_var + 0.25 * s_alt + 0.22 * s_para + 0.18 * s_shape)
        return max(0.0, min(1.0, gate * raw))
    except Exception:
        return 0.5
