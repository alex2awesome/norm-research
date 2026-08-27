"""p906 v2 -- Persuasive cadence: composite of several weak signals.

Criterion: prose rhythm builds momentum across paragraphs; sentence-length
variation, paragraph pacing, and transitions sustain the reader's attention
from headline through closing boilerplate.

Weighted blend of nine weak signals, none decisive alone:
  s_var    sentence-length coefficient-of-variation band
  s_alt    long/short alternation between adjacent sentences
  s_trans  transition/connective marker density
  s_para   paragraph count and paragraph-size pacing
  s_punct  rhythm punctuation density (commas/semicolons/colons/dashes)
  s_hook   short opening sentence relative to document mean
  s_flow   paragraphs that OPEN with a connective or a short sentence
  s_close  structured close (contact/about/### boilerplate near the end)
  s_len    mean sentence length inside the readable 10-28 word band
Chrome-heavy pages are damped by the junk fraction; tiny texts are gated.

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
_RHYTHM_PUNCT_RE = re.compile(r"[,;:]|\s-\s|--")

_TRANSITION_RE = re.compile(
    r"\b(?:however|moreover|furthermore|additionally|consequently|therefore"
    r"|meanwhile|ultimately|finally|notably|importantly|crucially|indeed"
    r"|similarly|likewise|nonetheless|nevertheless|in addition|as a result"
    r"|at the same time|in turn|what's more|for example|for instance"
    r"|in fact|on top of that|building on|looking ahead|going forward"
    r"|to that end|this means|most importantly|since then|first|second"
    r"|third|next|then)\b",
    re.IGNORECASE,
)
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
    if _CHROME_LINE_RE.match(s) or s.count("|") >= 2:
        return True
    words = _WORD_RE.findall(s)
    if not re.search(r"[.!?]\s*$", s) and len(words) <= 6:
        return True
    if s.isupper() and len(words) <= 8:
        return True
    return bool(re.match(r"^(?:https?://|www\.)\S+$", s))


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

        total_words = len(_WORD_RE.findall(t))
        if total_words == 0:
            return 0.0

        raw_paras = [p for p in re.split(r"\n\s*\n+", t) if p.strip()]
        body, chrome_words = [], 0
        for p in raw_paras:
            lines = [ln for ln in p.split("\n") if ln.strip()]
            kept = [ln for ln in lines if not _is_chrome_line(ln)]
            dropped = [ln for ln in lines if _is_chrome_line(ln)]
            chrome_words += sum(len(_WORD_RE.findall(ln)) for ln in dropped)
            if kept and len(kept) >= max(1, len(lines) // 2):
                body.append(" ".join(kept))
        if not body:
            return 0.02

        # structured close: boilerplate cues in the last quarter of blocks
        tail = body[max(0, len(body) - max(1, len(body) // 4)):]
        s_close = 1.0 if any(_BOILER_RE.search(p) for p in tail) else 0.0
        while body and len(body) > 1 and _BOILER_RE.search(body[-1]):
            body.pop()  # keep boilerplate out of the cadence measurements

        para_sents = [_sentences(p) for p in body]
        para_sents = [s for s in para_sents if s]
        if not para_sents:
            return 0.02
        sent_lens = [len(_WORD_RE.findall(s))
                     for sents in para_sents for s in sents]
        sent_lens = [n for n in sent_lens if n > 0]
        body_words = sum(sent_lens)
        if not sent_lens or body_words == 0:
            return 0.02
        mean_len = statistics.mean(sent_lens)

        # (1) variation
        if len(sent_lens) >= 3 and mean_len > 0:
            s_var = _cv_band(statistics.pstdev(sent_lens) / mean_len)
        else:
            s_var = 0.1

        # (2) alternation
        if len(sent_lens) >= 4:
            flips = sum(1 for a, b in zip(sent_lens, sent_lens[1:])
                        if abs(a - b) / float(max(a, b)) >= 0.30)
            s_alt = min(1.0, (flips / float(len(sent_lens) - 1)) / 0.55)
        else:
            s_alt = 0.1

        # (3) transitions
        body_text = "\n".join(body)
        trans_per100 = 100.0 * len(_TRANSITION_RE.findall(body_text)) \
            / float(body_words)
        s_trans = 1.0 - math.exp(-trans_per100 / 1.8)
        if trans_per100 > 9.0:
            s_trans *= max(0.4, 1.0 - (trans_per100 - 9.0) / 12.0)

        # (4) paragraph count and pacing
        n_paras = len(para_sents)
        base = {1: 0.25, 2: 0.55, 3: 0.80}.get(n_paras, 1.0 if n_paras else 0.0)
        pw = [sum(len(_WORD_RE.findall(s)) for s in sents)
              for sents in para_sents]
        if n_paras >= 2 and statistics.mean(pw) > 0:
            cv_p = statistics.pstdev(pw) / statistics.mean(pw)
            s_para = base * min(1.0, 0.5 + cv_p)
        else:
            s_para = base * 0.5

        # (5) rhythm punctuation
        punct_per100 = 100.0 * len(_RHYTHM_PUNCT_RE.findall(body_text)) \
            / float(body_words)
        s_punct = 1.0 - math.exp(-punct_per100 / 4.0)

        # (6) hook: first sentence noticeably tighter than the average
        first_len = sent_lens[0]
        s_hook = 1.0 if (first_len <= 20 and first_len <= mean_len) else \
            (0.5 if first_len <= 25 else 0.0)

        # (7) flow: paragraph openings carry a connective or a short sentence
        good_openers = 0
        for sents in para_sents:
            head = sents[0]
            if _TRANSITION_RE.match(head.strip()) or \
                    len(_WORD_RE.findall(head)) <= max(8, 0.8 * mean_len):
                good_openers += 1
        s_flow = good_openers / float(n_paras) if n_paras else 0.0

        # (8) readable mean sentence length band
        if 10.0 <= mean_len <= 28.0:
            s_len = 1.0
        elif mean_len < 10.0:
            s_len = max(0.0, (mean_len - 3.0) / 7.0)
        else:
            s_len = max(0.0, 1.0 - (mean_len - 28.0) / 15.0)

        raw = (0.18 * s_var + 0.14 * s_alt + 0.14 * s_trans + 0.12 * s_para
               + 0.10 * s_punct + 0.08 * s_hook + 0.08 * s_flow
               + 0.08 * s_close + 0.08 * s_len)
        junk_frac = chrome_words / float(total_words)
        gate = min(1.0, body_words / 60.0)
        return max(0.0, min(1.0, raw * gate * (1.0 - 0.8 * junk_frac)))
    except Exception:
        return 0.5
