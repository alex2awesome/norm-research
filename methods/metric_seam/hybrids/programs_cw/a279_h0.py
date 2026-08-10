"""a279 hybrid channel: Ending effectiveness, resonance, and closure.

Design (from train residuals):
  - Judge lows are dominated by non-endings: serial continuation markers
    ("To be continued", "Part 2 in comments", "Edit: part two below"),
    stories that stop mid-arc, and chatty author-note tails.
  - Judge highs almost always RESOLVE the arc and land on a short, punchy
    final narrative line (often dialogue / exclamation / one-line paragraph).
  - Keyword closure vocab (baseline) is a weak proxy; the predicate here is
    structural: does the story end, and does the last line land.

LLM fields ground the two things regex cannot see:
  - arc_closure: 3-way constrained label (resolution is tacit, unregexable)
  - final_line: the true final narrative sentence, skipping reddit chrome
The predicate (caps, weights, punchiness, penalties) stays in code.
"""

import re

LLM_FIELDS = {
    "arc_closure": (
        "Does the story's final scene decisively resolve or land its central "
        "arc? Answer exactly one word: RESOLVED, CLIFFHANGER, or ABRUPT."
    ),
    "final_line": (
        "Quote the story's final narrative sentence, ignoring any author "
        "notes, edit notes, links, or subreddit signatures."
    ),
}

# --- serial / continuation markers (searched in tail; 'part X of Y' anywhere)
_SERIAL_PATS = [
    r"to be continued",
    r"will be continued",
    r"\bpart\s*(?:\d+|one|two|three|four|ii\b|iii\b|iv\b)[^.\n]{0,40}"
    r"(?:below|here|soon|next|coming|comments|tomorrow|\bup\b)",
    r"(?:edit|update)\s*:?[^\n]{0,60}\bpart\b",
    r"\bin (?:the )?comments\b",
    r"\bnext (?:part|chapter|installment|episode)\b",
    r"\bstay tuned\b",
    r"\bcontinued (?:in|on|at)\b",
    r"tell you what happens (?:tomorrow|later|next)",
    r"\bmore parts?\b",
    r"\brest of the story\b",
]
_SERIAL_ANYWHERE = re.compile(r"\bpart\s*\d+(?:\s*-\s*\d+)?\s*(?:of|/)\s*\d+\b")

# --- author meta-chatter in the tail (about the writing, not the story)
_META_PATS = [
    r"\blol\b",
    r"\b(?:this|that|the) prompt\b",
    r"\bhad fun (?:with|writing)\b",
    r"\bthanks for reading\b",
    r"\bfeedback\b",
    r"\bcriti(?:que|cism)\b",
    r"\bfirst (?:post|story|prompt|attempt)\b",
    r"\bhope you enjoy",
    r"\bwriting until after\b",
    r"\bsorry (?:if|for|about)\b",
    r"\bthat was rough\b",
    r"\brushed\b",
]

# --- chrome lines to strip from the end before finding the final line
_CHROME_LINE_PATS = [
    r"^[\s\*\-_=~>\.#]{2,}$",                       # separators *** ---
    r"^\s*&(?:amp;)?#x200b;?\s*$",                  # zero-width junk lines
    r"^\s*\(?\s*(?:edit|update|a/?n|author'?s? notes?)\b",
    r"https?://",
    r"\breddit\.com\b",
    r"^\s*\*{0,2}\s*/?r/\w+",                       # /r/subreddit signatures
    r"^\s*\*{0,2}\s*part\s*\d",                     # *Part 2-3 in comments.*
    r"\bsubscribe\b",
    r"^\s*\[[^\]]*\]\([^)]*\)\s*\.?\s*$",           # bare markdown link line
    r"^\s*\(\s*end of (?:recording|log|entry|transmission|part)[^)]*\)\s*$",
    r"\bthanks for reading\b|\bfeedback\b|\bcriti(?:que|cism)\b",
    r"\b(?:this|that) prompt\b|\blol\b",
    r"^\s*\[\s*(?:wp|sp|eu|cw|tt|pi|ff)\s*\]",      # prompt tags
]
_CHROME_LINE = [re.compile(p, re.I) for p in _CHROME_LINE_PATS]

_WORD = re.compile(r"[A-Za-z']+")


def _words(s):
    return _WORD.findall(s or "")


def _strip_chrome_tail(norm):
    """Drop trailing chrome lines; return remaining text (may be '')."""
    lines = norm.rstrip().split("\n")
    while lines:
        ln = lines[-1].strip()
        if not ln:
            lines.pop()
            continue
        ln_clean = ln.replace("​", "").replace("&amp;#x200B;", "").replace("&#x200B;", "")
        if not ln_clean.strip():
            lines.pop()
            continue
        if any(p.search(ln) for p in _CHROME_LINE):
            lines.pop()
            continue
        break
    return "\n".join(lines).rstrip()


def _last_sentence(paragraph):
    parts = re.split(r"(?<=[.!?…])\s+", paragraph.strip())
    parts = [p for p in parts if _words(p)]
    return parts[-1] if parts else paragraph.strip()


def _closure_value(ans):
    """Map the extractor's 3-way answer to [0,1]; 0.5 when absent/unparsed."""
    if not ans or not isinstance(ans, str):
        return 0.5
    low = ans.lower()
    keys = [
        ("resolved", 1.0), ("resolves", 1.0), ("resolve", 1.0),
        ("cliffhanger", 0.0), ("unresolved", 0.0), ("abrupt", 0.1),
    ]
    best_pos, best_val = None, 0.5
    for k, v in keys:
        i = low.find(k)
        if i >= 0 and (best_pos is None or i < best_pos):
            best_pos, best_val = i, v
    return best_val


def _punch(sentence, paragraph):
    """Punchiness of the landing line: short, standalone, spoken/exclaimed."""
    wc = len(_words(sentence))
    if wc == 0:
        return 0.0
    if wc <= 9:
        p = 1.0
    elif wc >= 28:
        p = 0.0
    else:
        p = (28.0 - wc) / 19.0
    bonus = 0.0
    if re.search(r"[!?]\s*[\"'”’\*]*\s*$", sentence):
        bonus += 0.10                       # exclamatory / question landing
    if re.match(r"^\s*[\"'“‘\*]", sentence) or re.search(
            r"[\"'”’]\s*$", sentence):
        bonus += 0.05                       # spoken last line
    if len(_words(paragraph)) <= 14:
        bonus += 0.10                       # standalone one-line paragraph
    return min(1.0, p + bonus)


def score(text, extracted, ops):
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        try:
            norm = ops.normalize(text)
            if not isinstance(norm, str) or not norm.strip():
                norm = text
        except Exception:
            norm = text
        low_tail = norm[-800:].lower()

        # 1) serial / continuation markers -> hard cap (story does not end)
        serial_hits = sum(1 for p in _SERIAL_PATS if re.search(p, low_tail))
        if _SERIAL_ANYWHERE.search(norm.lower()):
            serial_hits += 1

        # 2) author meta-chatter near the end -> graded penalty
        meta_hits = sum(1 for p in _META_PATS if re.search(p, norm[-500:].lower()))
        meta_pen = min(0.38, 0.14 * meta_hits)

        # 3) find the landing line (prefer the LLM extraction, code fallback)
        stripped = _strip_chrome_tail(norm)
        para = ""
        for chunk in reversed(stripped.split("\n")):
            if _words(chunk):
                para = chunk.strip()
                break
        code_sent = _last_sentence(para) if para else ""
        fl = ""
        if isinstance(extracted, dict):
            fl = (extracted.get("final_line") or "").strip()
        if fl and fl.lower() not in ("none", "n/a") and 1 <= len(_words(fl)) <= 25:
            sent = fl
        else:
            sent = code_sent
        punch = _punch(sent, para if para else sent)

        # 4) arc closure from the constrained LLM label
        ans = extracted.get("arc_closure") if isinstance(extracted, dict) else ""
        closure = _closure_value(ans)

        # 5) structure: story-length floor + clean terminal punctuation
        n_words = len(_words(norm))
        floor = min(1.0, n_words / 150.0)
        body = stripped if stripped else norm.rstrip()
        terminal = 1.0 if body.endswith(
            (".", "!", "?", '"', "”", "'", "’", "*")) else 0.3
        structure = 0.6 * terminal + 0.4 * floor

        # 6) combine; a punchy line only counts if the arc actually lands
        punch_eff = punch * (0.35 + 0.65 * closure)
        raw = 0.12 + 0.40 * closure + 0.30 * punch_eff + 0.15 * structure
        raw -= meta_pen
        if serial_hits:
            raw = min(raw, 0.15 - 0.02 * min(serial_hits, 3) + 0.04 * punch)
        return float(max(0.0, min(1.0, raw)))
    except Exception:
        return 0.5
