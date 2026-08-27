"""a104 hybrid: Scannable, media-friendly formatting.

Judge behavior on TRAIN falls into bands:
  ~0.0  nav-chrome / ToS / non-release pages
  ~0.1  index / investor / blog pages with a little content
  ~0.15 REAL releases that are dense walls of text
  ~0.35 releases (and structured reports) with some layout: subheads,
        labeled fields, highlights
  ~0.45 clear subheads + consistently short paragraphs
  ~0.55 tables / stat blocks / (flattened) highlight lists

Design: score = gate * (0.12 + 0.88 * structure)
  gate      = is there a release/document here at all? Release markers
              (newswire tags, dateline, FOR IMMEDIATE RELEASE, said-quotes,
              contact/SOURCE blocks) OR generic document-ness (prose mass),
              where the doc-ness path is scaled by structure so that an
              unformatted wall of legal prose stays low while a structured
              non-release report (analyst study, agency notice) passes.
              Nav-chrome dominance and blog/index markers damp the gate.
  structure = MEASURED layout, not keywords (presence != quality; robust to
              counterfactual keyword injection): bullet-glyph/numbered lines,
              distinct subheads adjacent to prose (nav runs and repeated
              chrome lines excluded), short labeled fields ("TIME AND
              DATE:"), numeric table/stat lines, the short-paragraph mass of
              the prose, and flattened highlight lists (runs of capitalized
              past-tense verbs mid-sentence: the scrape signature of a
              bulleted highlights block whose bullets lost their periods).

Scrape robustness: hard-wrapped (PDF-extracted) lines are first unwrapped
(line ends in [a-z,] and next line starts lowercase/digit) so paragraph and
prose-mass features see real paragraphs. Works with extracted == {}; the two
LLM fields sharpen exactly the two stages (doc kind -> gate, layout
elements -> structure) when present.
"""
import re

LLM_FIELDS = {
    "doc_kind": ("Is this one press release/announcement (not a nav page, "
                 "article index, blog, news article, or FAQ)? Answer "
                 "RELEASE, ARTICLE, or NAV"),
    "layout_elems": ("Which does the main body visibly use: subheadings, "
                     "bullet list, numbered list, data table, highlights "
                     "block? List them, or NONE"),
}

_WORD_RE = re.compile(r"[A-Za-z0-9$%']+")
_ALPHA_RE = re.compile(r"[A-Za-z']+")

# ---- release markers (gate only; damping factor, never the quality signal)
_NEWSWIRE_RE = re.compile(
    r"PRNewswire|PR Newswire|BUSINESS WIRE|Business Wire|GLOBE NEWSWIRE|"
    r"Globe Newswire|Marketwired|ACCESSWIRE|ACCESS Newswire")
_IMMEDIATE_RE = re.compile(r"FOR IMMEDIATE RELEASE|\bPRESS RELEASE\b")
_DATELINE_RE = re.compile(  # 'INDIANAPOLIS - Dec. 19, 2018' / 'DUBAI, UAE, September 11, 2017'
    r"\b[A-Z]{3,}[A-Za-z., ]{0,30}[-,(]\s*"
    r"(?:[A-Z][a-z]{2,8}\.?,? \d{1,2},? \d{4}|\d{1,2} [A-Z][a-z]{2,8},? \d{4})"
    r"|\b[A-Z]{3,}[A-Z .]{0,20},\s+[A-Z][a-z]+\.?\s*[-–—]{1,2}\s")  # 'FORT WORTH, Texas - '
_ANNOUNCE_RE = re.compile(
    r"\btoday (?:announce[ds]?|report(?:ed|s)?|launch(?:ed|es)?|introduced|"
    r"released|issued|unveiled|named|filed)\b|"
    r"\bannounc(?:es|ed|ing) (?:today|its|the|plans)\b|"
    r"\b(?:announced|reported|launched|unveiled) today\b", re.I)
_TICKER_RE = re.compile(r"\((?:NYSE|NASDAQ|Nasdaq|OTC|TSX|LSE|AMEX)[ A-Za-z]*:\s*[A-Z]{1,6}\b")
_SAID_RE = re.compile(r"[,\"] (?:said|says) [A-Z]|\" (?:said|says)|(?:said|says) [A-Z][a-z]+ [A-Z]")
_CONTACT_RE = re.compile(
    r"\b(?:Media Contact|Press Contact|Media Inquiries|Press Office|"
    r"Media Hotline|Press inquiries|SOURCE [A-Z][A-Za-z]|###)")
_ABOUT_RE = re.compile(r"\bAbout [A-Z][A-Za-z&]")

# blog / index / FAQ anti-markers
_ANTI_RE = re.compile(
    r"frequently asked questions|Read [Mm]ore|Learn [Mm]ore|Posted In:|"
    r"View all news|View All\b|Older Posts|Related News|Top Stories|"
    r"Continue reading|Click here", re.I)

_BULLET_RE = re.compile(r"^\s*(?:[•‣▪·–>*+-]|\d{1,2}[.)])\s+\S")
_LABEL_RE = re.compile(r"^([A-Z][A-Za-z0-9 &/'().-]{1,45}):(?!//)\s*(\S.*)?$")
_CAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ,&/'()–.-]{5,70}$")
_NUMTOK_RE = re.compile(r"\d[\d,.]*%?")
_DECIMAL_RE = re.compile(r"\d+\.\d|\d+%|\$\s?\d")
# flattened highlight lists: capitalized past-tense verb where a sentence
# boundary was scraped away (preceding token ends in lowercase/number/%)
_FLAT_VERBS = (
    "Reported|Announced|Launched|Returned|Accrued|Delivered|Achieved|"
    "Completed|Executed|Adopted|Built|Opened|Signed|Raised|Reduced|"
    "Increased|Expanded|Generated|Recorded|Declared|Repurchased|Acquired|"
    "Introduced|Extended|Awarded|Named|Secured|Invested|Improved")
_FLAT_MID_RE = re.compile(r"[a-z0-9%)\"']\s+(?:" + _FLAT_VERBS + r")\s+[a-z$\d]")
_FLAT_SENT_RE = re.compile(r"^(?:" + _FLAT_VERBS + r")\b")
_COLON_INTRO_RE = re.compile(
    r"(?:following|highlights|as follows|includ(?:e|es|ing)|these steps):", re.I)
_WRAP_END_RE = re.compile(r"[a-z,]$")
_WRAP_START_RE = re.compile(r"^[a-z0-9]")


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _unwrap(text):
    """Re-join hard-wrapped (PDF-extracted) lines: a line ending in a
    lowercase letter or comma whose successor starts lowercase/digit is a
    wrap artifact, not a layout break."""
    raw = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out = []
    for ln in raw:
        if (out and _WRAP_END_RE.search(out[-1]) and _WRAP_START_RE.match(ln)
                and len(_WORD_RE.findall(out[-1])) >= 3):
            out[-1] = out[-1] + " " + ln
        else:
            out.append(ln)
    return out


def _classify_lines(lines):
    """Word counts + nav-run mask. Nav chrome = runs of >=5 consecutive
    short, number-free lines, plus any short line whose exact text repeats
    3+ times ('Learn More', 'Read More' link chrome). Numeric lines are
    protected: data tables also come as runs of short lines."""
    wcs = [len(_WORD_RE.findall(ln)) for ln in lines]
    counts = {}
    for ln, wc in zip(lines, wcs):
        if wc <= 6:
            counts[ln] = counts.get(ln, 0) + 1
    navish = [wcs[i] <= 6 and not _DECIMAL_RE.search(lines[i])
              for i in range(len(lines))]
    in_nav = [navish[i] and counts.get(lines[i], 0) >= 3
              for i in range(len(lines))]
    i = 0
    while i < len(lines):
        if navish[i]:
            j = i
            while j < len(lines) and navish[j]:
                j += 1
            if j - i >= 5:
                for k in range(i, j):
                    in_nav[k] = True
            i = j
        else:
            i += 1
    return wcs, in_nav


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        text = ops.normalize(text)
        lines = _unwrap(text)
        if not lines:
            return 0.0
        wcs, in_nav = _classify_lines(lines)
        n_lines = len(lines)
        total_words = sum(wcs) or 1

        # ---------- structure features (measured layout only) ----------
        bullets = labels = stat_lines = 0
        subhead_texts = set()
        prose_blocks = []           # word counts of body prose lines
        for i, ln in enumerate(lines):
            wc = wcs[i]
            if in_nav[i]:
                continue
            is_stat = (wc <= 10 and
                       (len(_NUMTOK_RE.findall(ln)) >= 2 or
                        (_DECIMAL_RE.search(ln) and wc <= 6)))
            if is_stat:
                stat_lines += 1
                continue
            if _BULLET_RE.match(ln) and wc >= 3:
                bullets += 1
                continue
            m = _LABEL_RE.match(ln)
            if (m and len(_WORD_RE.findall(m.group(1))) <= 8
                    and len(_WORD_RE.findall(m.group(2) or "")) <= 30):
                labels += 1
                continue
            if wc >= 20 and "." in ln:
                prose_blocks.append(wc)
                continue
            if (2 <= wc <= 12 and not re.search(r"[.!?]$", ln)
                    and not re.search(r"[.!?] ", ln)):
                near_prose = any(
                    0 <= j < n_lines and wcs[j] >= 25
                    for j in (i - 2, i - 1, i + 1, i + 2))
                if near_prose:
                    subhead_texts.add(ln)
        subheads = min(len(subhead_texts), 6) + sum(
            1 for t in list(subhead_texts)[:6] if _CAPS_RE.match(t))

        prose_words = sum(prose_blocks)
        # short-paragraph mass: fraction of prose in digestible blocks
        if len(prose_blocks) >= 2:
            short_para = (sum(w for w in prose_blocks if w <= 100)
                          / prose_words)
            mean_block = prose_words / len(prose_blocks)
            short_para *= 1.0 - 0.6 * _clamp((mean_block - 150) / 250.0)
        else:
            short_para = 0.0

        # flattened highlight lists (scrape-collapsed bullets)
        body = " ".join(ln for i, ln in enumerate(lines)
                        if not in_nav[i] and wcs[i] >= 20)
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body)
                 if len(_ALPHA_RE.findall(s)) >= 4]
        flat = (len(_FLAT_MID_RE.findall(body)) +
                sum(1 for s in sents if _FLAT_SENT_RE.match(s)))
        flat_score = _clamp((flat - 2) / 5.0)
        colon_intro = 1.0 if (_COLON_INTRO_RE.search(body)
                              and (flat >= 2 or bullets or stat_lines >= 3)
                              ) else 0.0

        structure = _clamp(
            0.28 * _clamp(bullets / 5.0) +
            0.18 * _clamp(subheads / 6.0) +
            0.22 * _clamp(labels / 4.0) +
            0.26 * _clamp((stat_lines - 2) / 8.0) +
            0.28 * short_para +
            0.26 * flat_score +
            0.08 * colon_intro)

        # ---------- gate: release-likeness / document-ness ----------
        pts = (2.0 * bool(_NEWSWIRE_RE.search(text)) +
               1.5 * bool(_IMMEDIATE_RE.search(text)) +
               1.5 * bool(_DATELINE_RE.search(text)) +
               1.0 * bool(_ANNOUNCE_RE.search(text)) +
               0.75 * bool(_TICKER_RE.search(text)) +
               0.75 * _clamp(len(_SAID_RE.findall(text)) / 2.0) +
               0.5 * bool(_CONTACT_RE.search(text)) +
               0.5 * bool(_ABOUT_RE.search(text)))
        release = _clamp(pts / 3.0)

        doc_like = min(_clamp((prose_words - 80) / 250.0),
                       _clamp((prose_words / total_words) / 0.45))
        gate = max(release, 0.25 * doc_like + 0.55 * doc_like * structure)

        # nav-chrome damping, relaxed for marker-confirmed releases (real
        # releases drag newswire/Cision footers along in the scrape)
        nav_ratio = sum(1 for v in in_nav if v) / n_lines
        gate *= 1.0 - 0.6 * _clamp((nav_ratio - 0.45) / 0.4) * (1.0 - 0.75 * release)
        if release < 0.6:
            anti = len(_ANTI_RE.findall(text))
            gate *= 1.0 - 0.5 * _clamp(anti / 6.0)

        # ---------- LLM thick-input grounding (optional) ----------
        # NOTE (hygiene patch): doc_kind is a single-category enum answer
        # (RELEASE/ARTICLE/NAV); bare `"X" in dk` false-fired on unrelated
        # words containing the token as a substring (e.g. "NAV" inside
        # "navigation"/"coronavirus", "RELEASE" inside "released"). \b-anchor
        # so the category token must appear as its own word.
        dk = str((extracted or {}).get("doc_kind", "")).strip().upper()
        if dk:
            if re.search(r"\bRELEASE\b", dk):
                gate = max(gate, 0.85)
            elif re.search(r"\bNAV\b", dk):
                gate = min(gate, 0.25)
            elif re.search(r"\bARTICLE\b", dk):
                gate = min(gate, 0.5)
        le = str((extracted or {}).get("layout_elems", "")).strip()
        if le:
            if le.upper() == "NONE":
                structure = min(structure, 0.6 * structure)
            else:
                # NOTE (hygiene patch): "table"/"bullet" bare substring
                # false-fired on unrelated words ("tablets"/"injectables" for
                # "table"; "bulletin"/"bulletshares" for "bullet"). \b-anchor
                # with a small inflection whitelist for the layout-element
                # keywords that legitimately pluralize/verb-form.
                le_l = le.lower()
                n_el = sum(1 for kw in (r"subhead\w*", r"bullets?", r"numbered",
                                        r"tables?", r"highlight\w*")
                           if re.search(r"\b" + kw + r"\b", le_l))
                structure = _clamp(0.7 * structure + 0.3 * (n_el / 4.0))

        if gate <= 0.04:
            return 0.02
        return _clamp(gate * (0.12 + 0.88 * structure))
    except Exception:
        return 0.5
