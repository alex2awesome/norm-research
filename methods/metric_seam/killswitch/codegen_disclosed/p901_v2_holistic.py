"""p901_v2_holistic -- Quantitative support (criterion p901), composite arm.

Blends five weak signals into one score: (1) length-normalized density of
typed numeric evidence, (2) fraction of prose sentences carrying a figure,
(3) variety of figure TYPES used (currency / percent / magnitude /
measurement / date / plain count), (4) quantitative-vs-vague-language
balance (numbers vs "significant/leading/robust" talk), and (5)
specificity -- the share of figures that are typed rather than bare
integers. Nav chrome, URLs, phones, emails, and the trailing contact
block are stripped first.
"""

import re
import math

# --- mojibake / entity normalization (longest first; residual last) ---------
# Explicit escapes: â€... are the visible mojibake of UTF-8 curly
# punctuation mis-decoded as cp1252 (e.g. "â€œ" renders "â€œ").
_MOJIBAKE = [
    ("â€œ", '"'),   # curly left double quote
    ("â€", '"'),   # curly right double quote
    ("â€™", "'"),   # curly apostrophe
    ("â€˜", "'"),   # curly left single quote
    ("â€“", "-"),   # en dash
    ("â€”", "-"),   # em dash
    ("â€¦", "..."), # ellipsis
    ("â‚¬", "€"),  # euro sign
    ("â€", '"'),         # residual quote fragment (must be last)
    ("Â ", " "),         # mojibake non-breaking space
    ("Â ", " "),
    ("Â", ""),
    (" ", " "),
    ("&amp;", "&"),
    ("&nbsp;", " "),
    ("&quot;", '"'),
    ("&#39;", "'"),
    ("&rsquo;", "'"),
    ("&lsquo;", "'"),
    ("&rdquo;", '"'),
    ("&ldquo;", '"'),
    ("&gt;", " "),
    ("&lt;", " "),
]

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?:\+?1[\s\-.])?\(?\d{3}\)?[\s\-.]\d{3}[\s\-.]\d{4}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

_TAIL_MARKER_RE = re.compile(
    r"^(?:media\s+)?contacts?\b|^about\s+\S+|^for\s+(?:more|further)\s+"
    r"information\b|^source[:\s]|^###|^investor\s+relations\b|"
    r"^press\s+(?:contact|office)\b|forward-looking\s+statements|"
    r"safe\s+harbor", re.IGNORECASE)

# --- typed figure detectors --------------------------------------------------
_TYPED = {
    "currency": re.compile(
        r"[$£€¥]\s?\d[\d,]*(?:\.\d+)?"
        r"(?:\s?(?:million|billion|trillion|thousand|[MBK])\b)?"
        r"|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|euros|pounds|USD|EUR|GBP)\b",
        re.IGNORECASE),
    "percent": re.compile(
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent(?:age\s+points?)?)",
        re.IGNORECASE),
    "magnitude": re.compile(
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion|thousand|mn|bn)\b",
        re.IGNORECASE),
    "measure": re.compile(
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:kg|km|mg|ml|mm|cm|kWh?|MWh?|GWh?|GW|MW|"
        r"mph|sq\.?\s?ft|square\s+(?:feet|meters|metres|miles)|acres?|"
        r"miles?|meters?|metres?|feet|tons?|tonnes?|barrels?|gallons?|"
        r"liters?|litres?|hectares?|ounces?|degrees?|hours?|days?|weeks?|"
        r"months?|years?|employees|patients|customers|members|countries|"
        r"stores|locations|units|jobs|people)\b",
        re.IGNORECASE),
    "date": re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|"
        r"Nov|Dec)\.?\s+\d{1,2}(?:,?\s+\d{4})?\b"
        r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:19|20)\d{2}\b",
        re.IGNORECASE),
}
_BARE_NUM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

# vague qualitative-claim vocabulary (the "rather than" side of the criterion)
_VAGUE_RE = re.compile(
    r"\b(?:significant(?:ly)?|substantial(?:ly)?|considerabl[ye]|"
    r"dramatic(?:ally)?|strong(?:ly)?|robust|rapid(?:ly)?|many|numerous|"
    r"several|vast|huge|major|leading|world-class|best-in-class|"
    r"state-of-the-art|cutting-edge|innovative|unparalleled|exceptional|"
    r"tremendous|remarkable|extensive|widely|premier|outstanding|"
    r"unprecedented|broad(?:ly)?|high-quality)\b", re.IGNORECASE)

_WORD_RE = re.compile(r"\b\w+\b")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize(text):
    for bad, good in _MOJIBAKE:
        if bad in text:
            text = text.replace(bad, good)
    text = text.replace("[...]", "\n")
    text = _URL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    return text


def _body_text(text):
    """Drop nav chrome lines and the trailing contact/boilerplate block."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return ""
    tail_start = int(len(lines) * 0.70)
    cut = len(lines)
    for i in range(tail_start, len(lines)):
        if _TAIL_MARKER_RE.search(lines[i]):
            cut = i
            break
    lines = lines[:cut]
    keep = [ln for ln in lines
            if len(ln) >= 60
            or (len(ln) >= 25 and re.search(r"[.!?](?:\s|$)", ln))]
    return " ".join(keep)


def _sat(x, k):
    return 1.0 - math.exp(-x / k) if k > 0 else 0.0


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        t = _normalize(text)
        body = _body_text(t)
        if len(_WORD_RE.findall(body)) < 20:
            body = t  # nothing prose-like survived; fall back to full text
        words = len(_WORD_RE.findall(body))
        if words < 10:
            return 0.0

        # non-overlapping typed matches, then leftover bare numbers
        claimed = []

        def _free(s, e):
            for cs, ce in claimed:
                if s < ce and e > cs:
                    return False
            return True

        typed_counts = {}
        for name, pat in _TYPED.items():
            c = 0
            for m in pat.finditer(body):
                if _free(m.start(), m.end()):
                    claimed.append((m.start(), m.end()))
                    c += 1
            typed_counts[name] = c
        bare = 0
        for m in _BARE_NUM_RE.finditer(body):
            if _free(m.start(), m.end()):
                claimed.append((m.start(), m.end()))
                bare += 1

        typed_total = sum(typed_counts.values())
        total = typed_total + bare

        # (1) density of figures per 100 words (typed count 1.5x)
        density = 100.0 * (1.5 * typed_total + bare) / float(words)
        s_density = _sat(density, 2.5)

        # (2) sentence coverage
        sents = [s for s in _SENT_SPLIT_RE.split(body)
                 if len(_WORD_RE.findall(s)) >= 6]
        if sents:
            cov = sum(1 for s in sents if re.search(r"\d", s)) / len(sents)
        else:
            cov = 0.0
        s_cov = _sat(cov, 0.25)

        # (3) variety of figure types (counting "bare count" as a type)
        types_present = sum(1 for c in typed_counts.values() if c > 0)
        types_present += 1 if bare > 0 else 0
        s_var = min(1.0, types_present / 4.0)

        # (4) quantitative vs vague-language balance
        vague = len(_VAGUE_RE.findall(body))
        if total + vague > 0:
            s_bal = total / float(total + vague)
        else:
            s_bal = 0.3
        # (5) specificity: typed share of all figures
        s_spec = (typed_total / float(total)) if total > 0 else 0.0

        raw = (0.30 * s_density + 0.25 * s_cov + 0.15 * s_var
               + 0.15 * s_bal + 0.15 * s_spec)
        raw *= min(1.0, words / 50.0)  # damp very short docs
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
