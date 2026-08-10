# -*- coding: utf-8 -*-
"""p901_v1_structure -- structural/positional metric for 'Quantitative support'.

Approach: the document is split into paragraph blocks; navigation-chrome
blocks (short menu-like lines without sentence punctuation) are dropped,
and a trailing contact/boilerplate zone (from the first contact-style
marker in the last 40% of blocks) is discarded so its phone numbers and
addresses do not count.  On the surviving body blocks the metric measures
WHERE numeric evidence sits: position-weighted paragraph coverage
(inverted pyramid -- earlier paragraphs count more), a lede bonus when
the opening paragraph is quantified, and body figure density.

Source is pure ASCII: all non-ASCII characters are built with chr() so
the module is immune to encoding mangling.
"""

import re
import math

_AC = chr(0x00E2)            # a-circumflex
_EU = chr(0x20AC)            # euro sign
_MOJI_PREFIX = _AC + _EU

_MOJIBAKE = [
    (_MOJI_PREFIX + chr(0x0153), '"'),   # left curly double quote
    (_MOJI_PREFIX + chr(0x009D), '"'),   # right curly double quote (invisible U+009D)
    (_MOJI_PREFIX + chr(0x2122), "'"),   # apostrophe
    (_MOJI_PREFIX + chr(0x02DC), "'"),   # left single quote
    (_MOJI_PREFIX + chr(0x201C), "-"),   # en dash
    (_MOJI_PREFIX + chr(0x201D), "-"),   # em dash
    (_MOJI_PREFIX + chr(0x00A6), "..."), # ellipsis
    (_AC + chr(0x201A) + chr(0x00AC), _EU),  # euro mojibake -> euro sign
    (_MOJI_PREFIX, '"'),                 # leftover bare form
    (chr(0x00C2), ""),                   # A-circumflex before nbsp
]

_STRIP_RES = [
    re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    re.compile(r"\S+@\S+"),
    re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b"),
    re.compile(r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}"),
    re.compile("(?:" + chr(0x00A9) + r"|\(c\)|copyright)\s*\d{4}",
               re.IGNORECASE),
    re.compile(r"\[\.\.\.\]"),
]

_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
           r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
           r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")

_UNITS = (r"(?:units?|employees?|customers?|users?|members?|subscribers?|"
          r"patients?|stores?|locations?|offices?|countries|states|cities|"
          r"acres?|square\s+(?:feet|meters?)|miles?|kilometers?|km|meters?|"
          r"tons?|tonnes?|pounds?|kg|liters?|gallons?|barrels?|megawatts?|"
          r"gigawatts?|MW|GW|kWh|MWh|jobs?|shares?|people|homes?|"
          r"households?|vehicles?|devices?|downloads?|transactions?|"
          r"hours?|days?|weeks?|months?|years?|quarters?)")

_CURRENCY_CLASS = "[$" + chr(0x20AC) + chr(0x00A3) + chr(0x00A5) + "]"

# "Strong" numeric evidence: figures that plausibly back a claim.
_STRONG_RES = [
    re.compile("(?:" + _CURRENCY_CLASS + r"|\bUS\$|\bUSD\s|\bEUR\s|\bGBP\s)"
               r"\s?\d[\d,]*(?:\.\d+)?", re.IGNORECASE),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|"
               r"percentage\s+points?\b|basis\s+points\b|bps\b)",
               re.IGNORECASE),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?\s(?:million|billion|trillion|"
               r"thousand)\b", re.IGNORECASE),
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?" + _UNITS + r"\b", re.IGNORECASE),
    re.compile(r"\b" + _MONTHS + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?"
               r"(?:,?\s+\d{4})?\b", re.IGNORECASE),
    re.compile(r"\b(?:Q[1-4]|FY|H[12])\s?'?\d{2,4}\b"),
]

_NAV_VOCAB = ("home", "about", "about us", "contact", "contact us", "news",
              "newsroom", "blog", "products", "services", "menu", "search",
              "login", "log in", "sign in", "sign up", "subscribe", "share",
              "print", "email", "tweet", "facebook", "twitter", "linkedin",
              "instagram", "youtube", "privacy policy", "terms of use",
              "terms of service", "cookie policy", "sitemap", "careers",
              "investors", "media", "events", "resources", "support",
              "skip to content", "read more", "back to top",
              "all rights reserved")

_CONTACT_MARKERS = ("contact:", "contacts:", "media contact", "press contact",
                    "investor contact", "investor relations",
                    "media relations", "for more information",
                    "for further information", "about ", "###", "source:",
                    "source ", "tel:", "tel.", "telephone:", "phone:",
                    "e-mail", "email:", "media inquiries", "press office")

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENT_PUNCT_RE = re.compile(r"[.!?;:]")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
_BLANKLINE_RE = re.compile(r"\n\s*\n")


def _is_chrome(block):
    words = _WORD_RE.findall(block)
    stripped = block.strip().lower().rstrip(" |>-")
    if not words:
        return True
    if stripped in _NAV_VOCAB:
        return True
    # Short line with no sentence punctuation: menu item / heading chrome.
    if len(words) <= 5 and not _SENT_PUNCT_RE.search(block):
        return True
    return False


def _is_contact_marker(block):
    low = block.strip().lower()
    for marker in _CONTACT_MARKERS:
        if low.startswith(marker) or (len(low) < 80 and marker in low):
            return True
    return False


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        cleaned = text
        for bad, good in _MOJIBAKE:
            cleaned = cleaned.replace(bad, good)
        for rx in _STRIP_RES:
            cleaned = rx.sub(" ", cleaned)

        # Paragraph blocks: prefer blank-line separation, else single lines.
        if _BLANKLINE_RE.search(cleaned):
            raw_blocks = _PARA_SPLIT_RE.split(cleaned)
        else:
            raw_blocks = cleaned.split("\n")
        blocks = [b.strip() for b in raw_blocks if b.strip()]
        if not blocks:
            return 0.0

        # Trailing contact/boilerplate zone: first marker in last 40%.
        n = len(blocks)
        cut = n
        for i, b in enumerate(blocks):
            if i >= max(1, int(0.6 * n)) and _is_contact_marker(b):
                cut = i
                break
        blocks = blocks[:cut]

        body = [b for b in blocks if not _is_chrome(b)]
        if not body:
            return 0.0

        # Position-weighted paragraph coverage (inverted pyramid).
        weight_sum = 0.0
        hit_sum = 0.0
        quantified = []
        for i, b in enumerate(body):
            w = 1.0 / (1.0 + 0.4 * i)
            has_fig = False
            for rx in _STRONG_RES:
                if rx.search(b):
                    has_fig = True
                    break
            quantified.append(has_fig)
            weight_sum += w
            if has_fig:
                hit_sum += w
        coverage = hit_sum / max(weight_sum, 1e-9)

        # Lede bonus: figures up front is the press-release signature.
        if quantified[0]:
            lede = 1.0
        elif len(quantified) > 1 and quantified[1]:
            lede = 0.5
        else:
            lede = 0.0

        # Body figure density (spans consumed so nothing double-counts).
        body_text = "\n".join(body)
        n_words = max(len(_WORD_RE.findall(body_text)), 40)
        n_strong = 0
        working = body_text
        for rx in _STRONG_RES:
            found = rx.findall(working)
            if found:
                n_strong += len(found)
                working = rx.sub(lambda m: " " * len(m.group(0)), working)
        density = 100.0 * n_strong / float(n_words)
        density_sig = 1.0 - math.exp(-density / 3.0)

        value = 0.55 * coverage + 0.25 * lede + 0.20 * density_sig
        return max(0.0, min(1.0, value))
    except Exception:
        return 0.5
