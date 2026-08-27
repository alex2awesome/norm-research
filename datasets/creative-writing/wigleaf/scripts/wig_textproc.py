#!/usr/bin/env python3
"""Shared text extraction + presentation normalization for the Wigleaf cw-B cell.

CRITICAL DE-CONFOUNDING ROLE: Top50 (live story URLs) and longlist (recovered via
web search) come from DIFFERENT fetch paths and DIFFERENT magazines. To prevent a
presentation/source leak, this module applies the *identical* pipeline to every
text regardless of class or fetch_source:

  extract_main_text(html)  -> largest paragraph-rich block, wayback toolbar removed
  strip_bio_tail(text)     -> remove trailing author-bio paragraph (~37% of pages)
  strip_cms_boilerplate    -> drop nav/CMS lines (issue headers, "Read more", etc.)
  normalize_text(text)     -> NFKC, curly->straight quotes, dashes, ellipsis,
                              collapse whitespace, strip wrapper quotes
  normalize_magazine(name) -> canonical casing (SmokeLong/Smokelong, elimae/Elimae)

The same functions are imported by both the Top50 fetcher and the longlist fetcher.
"""
import re
import unicodedata

from bs4 import BeautifulSoup

# ---------------------------------------------------------------- normalization
CURLY = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "′": "'", "«": '"', "»": '"', "‚": "'", "„": '"',
    "—": "-", "–": "-", "‒": "-", "―": "-",  # dashes -> hyphen
    "…": "...",                                            # ellipsis
    " ": " ", " ": " ", " ": " ", " ": " ", # thin/nbsp spaces
    "�": "'",                                              # replacement char
}


def normalize_text(s):
    """NFKC, curly->straight quotes, dashes->hyphen, ellipsis, collapse ws,
    strip wrapper quotes. Applied IDENTICALLY to both classes."""
    if not s:
        return ""
    # remove soft hyphens + zero-width chars (Wayback re-encoding artifacts that
    # mangle words like 'pub\xadlished'); NFKC does NOT strip these
    s = re.sub(r"[­​‌‍﻿]", "", s)
    s = unicodedata.normalize("NFKC", s)
    for k, v in CURLY.items():
        s = s.replace(k, v)
    s = re.sub(r"<[^>]+>", " ", s)           # leftover markup
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    # strip wrapper quotes around the whole piece
    if len(s) > 2 and s[0] == '"' and s[-1] == '"' and s.count('"') == 2:
        s = s[1:-1].strip()
    return s


# canonical magazine names: lowercase-key -> display name. Audit flagged
# SmokeLong/Smokelong and elimae/Elimae casing inconsistencies that leak the
# parse pipeline. Map every spelling variant to one canonical form.
_MAG_CANON = {
    "smokelong quarterly": "SmokeLong Quarterly",
    "smokelong": "SmokeLong Quarterly",
    "elimae": "elimae",
    "pindeldyboz": "Pindeldyboz",
    "wigleaf": "Wigleaf",
    "necessary fiction": "Necessary Fiction",
    "jellyfish review": "Jellyfish Review",
    "flash fiction online": "Flash Fiction Online",
    "fiction southeast": "Fiction Southeast",
    "fictionaut": "Fictionaut",
    "atticus review": "Atticus Review",
    "matter press": "Matter Press",
    "the journal of compressed creative arts": "Matter Press",
    "pithead chapel": "Pithead Chapel",
    "wigleaf top 50": "Wigleaf",
    "cease, cows": "Cease, Cows",
    "gone lawn": "Gone Lawn",
    "corium magazine": "Corium Magazine",
    "corium": "Corium Magazine",
    "okay donkey": "Okay Donkey",
    "lost balloon": "Lost Balloon",
    "the rumpus": "The Rumpus",
    "hobart": "Hobart",
    "pank": "PANK",
    "diagram": "DIAGRAM",
    "tin house": "Tin House",
    "guernica": "Guernica",
    "joyland": "Joyland",
    "passages north": "Passages North",
    "split lip magazine": "Split Lip Magazine",
    "split lip": "Split Lip Magazine",
    "wigleaf.": "Wigleaf",
}


def normalize_magazine(name):
    if not name:
        return ""
    n = unicodedata.normalize("NFKC", str(name))
    for k, v in CURLY.items():
        n = n.replace(k, v)
    n = re.sub(r"\s+", " ", n).strip().rstrip(".,")
    key = n.lower().strip()
    if key in _MAG_CANON:
        return _MAG_CANON[key]
    # title-case ALL-CAPS magazine names so "ELIMAE" doesn't differ from "elimae"
    if n.isupper() and len(n) > 3:
        n = n.title()
    return n


# ------------------------------------------------------------- bio-tail removal
# Trailing author-bio paragraph. Audit found bio tails in ~37% of pages. They
# read "<Author> is the author of...", "She lives in...", "He teaches at...",
# "<Name>'s work has appeared in...". We strip the LAST paragraph(s) if they
# match these third-person-bio cues. Conservative: only trailing block, and
# only when it is clearly a bio (not story prose).
_BIO_CUES = re.compile(
    r"\b("
    r"is the author of|is the editor of|author of the|"
    r"(her|his|their) (work|stories|fiction|writing|poems?|essays?) (has|have) (appeared|been published)|"
    r"(she|he|they) (lives?|teaches?|holds? (a|an)|earned (a|an)|received (a|an)|works? as|is (a|an) (writer|editor|professor|candidate|graduate|student|teacher))|"
    r"(she|he|they) (is|are) (currently )?(a |an )?(mfa|phd|ph\.d|writer|editor|professor)|"
    r"(she|he|they) can be (found|reached)|"
    r"(lives|resides) (in|with) (his|her|their|a)|"
    r"more of (her|his|their) work|"
    r"recently appeared (from|in)|"
    r"is a writer (and|living|based)|"
    r"is a graduate of|"
    r"(her|his|their) (debut|first) (novel|collection|book)|"
    r"forthcoming (from|in)|"
    r"(she|he|they) (also )?(writes?|edits?)|"
    r"(mfa|m\.f\.a) (from|at|candidate)|"
    r"(@\w+ on (twitter|instagram)|on twitter (at|@)|tweets? (at|@))|"
    r"is pursuing (a|an|her|his)|"
    r"holds an mfa|"
    r"find (her|him|them) (at|on)|"
    r"(her|his|their) work can be (found|seen)|"
    r"about the author|"
    r"(collection|novel|book|chapbook|story) [a-z' ]{0,30}(won|will be published|is forthcoming|was published|is out|appeared) (the |from |by |in )|"
    r"won the \d{4}|(won|received|nominated for) (a |an |the |two |three )?(pushcart|tiptree|shirley jackson|o\.?\s?henry|best (small|of)|national book|iowa short fiction|award)|"
    r"(her|his|their) (fiction|work|stories) (has|have) been translated|"
    r"(work|stories|fiction|writing|poems?|essays?) (have |has )?appear(s|ed)? in (journals?|magazines?|including)|"
    r"appear(s|ed)? in journals|"
    r"(her|his|their) (writing|work|stories|fiction) (appears?|has appeared)|"
    r"(is|was) born in \d{4}|(is|was) the recipient of|"
    r"holds (a|an) (ma|mfa|phd|m\.a|b\.a|ba) (in|from)|"
    r"teaches (writing|english|creative|fiction|at)|"
    r"is (a|an) [a-z ]{0,30}(writer|poet|editor|novelist|essayist|attorney|artist|translator|journalist|professor|teacher|musician|filmmaker|photographer|student|candidate)\b|"
    r"(currently )?(lives?|resides?|residing|living|based) (in|near|with|and) [a-z]|"
    r"(is|are) (currently )?(based|residing|living) in|"
    r"co-?founded|co-?editor of|founding editor|"
    r"has (published|written) (five|four|three|two|several|numerous|a)"
    r")\b",
    re.I,
)

# Trailing CMS / promo / credit lines (NOT bios but still presentation junk):
# copyright stamps, "More fiction at X", image credits, cookie banners, nav,
# "Return to Top", "Permalink", contest promos. Strip these from the END.
_TRAIL_JUNK = re.compile(
    r"^("
    r"copyright\b|©|\(c\)\s*\d|©\s*\d|\d{4}\s*©|"
    r"more (fiction|stories|work) (at|from|by)|read more (on|about)|"
    r"(photo|illustration|image|art(work)?|cover)\s*(by|credit|:|©)|photo[: ]|"
    r"return to top|back to top|top of page|"
    r"permalink|tags?:|posted (on|by|in)|filed under|"
    r"this data is anonymized|cookie|privacy policy|withdraw permission|"
    r"feel like submitting|check out our submission|submission guidelines|"
    r"\(next:|\(previous:|next:|previous:|next story|previous story|"
    r"page optimized|powered by|wordpress|wp minify|"
    r"share (this|on)|tweet this|"
    r"\d{1,2}/\d{1,2}/\d{2,4}[, ]|"  # date stamps like 5/6/2008
    r"strictly necessary cookie"
    r")",
    re.I,
)


# a copyright / "contents of X are Copyright" / "Thoughts? Tell us." clause glued
# to the end of the last story line — cut at the last sentence boundary before it.
_TRAIL_CLAUSE = re.compile(
    r"(thoughts\?\s*tell us|copyright\s*©|©\s*\d{4}|copyright\s*\d{4}|"
    r"all rights reserved|except where noted|contents of \w+\.com|"
    r"return to top|more fiction at|originally (appeared|published))", re.I)


def _cut_trailing_clause(text):
    tail = text[-300:]
    m = _TRAIL_CLAUSE.search(tail)
    if not m:
        return text
    abs_pos = len(text) - 300 + m.start()
    before = text[:abs_pos]
    b = max(before.rfind(". "), before.rfind("? "), before.rfind("! "),
            before.rfind(".\n"), before.rfind("?\n"), before.rfind("!\n"),
            before.rfind('."'), before.rfind('?"'), before.rfind('!"'))
    if b > 200:
        return text[:b + 1].strip()
    # no sentence boundary: just drop from the clause start if enough remains
    if abs_pos > 200:
        return text[:abs_pos].strip()
    return text


# a run of >=3 entries separated by «», "", |, • = an issue TOC / contributor
# list (SmokeLong-era), not story prose. Separators may be raw (pre-normalize).
_TOC_RUN = re.compile(r'(?:[^\n«»"|•]{1,45}\s*(?:«»|""|\|\||•|\s\|\s)\s*){3,}')


def _strip_contributor_toc(text):
    """Cut a trailing issue-TOC / contributor list (a long run of name «» name «»
    name ...). Find the EARLIEST start of such a run in the tail and cut there."""
    if not text:
        return text
    m = _TOC_RUN.search(text[-900:])
    if m:
        abs_pos = len(text) - 900 + m.start()
        before = text[:abs_pos]
        b = max(before.rfind(". "), before.rfind("? "), before.rfind("! "),
                before.rfind('.\n'), before.rfind('?\n'), before.rfind('!\n'),
                before.rfind('."'), before.rfind('?"'), before.rfind('!"'),
                before.rfind('.»'), before.rfind('?»'))
        if b > 200:
            return text[:b + 1].strip()
        if abs_pos > 200:
            return text[:abs_pos].strip()
    return text


_MONTH_NAV = re.compile(
    r"(\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|archives?|submit|about|social media|excerpt)\b[\s\n]*){4,}", re.I)


def _strip_archive_nav(text):
    """Cut a trailing archive/month-list nav widget (Hobart-style:
    'Archives January December November ... Submit About Social Media')."""
    if not text:
        return text
    m = _MONTH_NAV.search(text[-600:])
    if m:
        abs_pos = len(text) - 600 + m.start()
        before = text[:abs_pos]
        b = max(before.rfind(". "), before.rfind("? "), before.rfind("! "),
                before.rfind('."'), before.rfind('?"'), before.rfind('!"'))
        if b > 200:
            return text[:b + 1].strip()
        if abs_pos > 200:
            return text[:abs_pos].strip()
    return text


def strip_trailing_junk(text):
    """Drop trailing CMS/promo/credit lines. Operates on the tail, line by line."""
    if not text:
        return text
    text = _cut_trailing_clause(text)
    text = _strip_contributor_toc(text)
    text = _strip_archive_nav(text)
    lines = text.split("\n")
    end = len(lines)
    for i in range(len(lines) - 1, max(-1, len(lines) - 10), -1):
        s = lines[i].strip()
        if not s:
            end = i
            continue
        words = len(s.split())
        # short junky lines, or any line matching the junk patterns
        if _TRAIL_JUNK.match(s) and words < 25:
            end = i
        elif words <= 4 and (s.isupper() or set(s) <= set("•*#+-—– ")):
            end = i  # decorative separators / all-caps nav
        else:
            break
    return "\n".join(lines[:end]).strip()


def strip_bio_tail(text):
    """Remove a trailing author-bio block. Operates paragraph-by-paragraph from
    the end: drop trailing paragraphs that look like a third-person author bio,
    stop at the first non-bio paragraph (the story body)."""
    if not text:
        return text
    paras = [p for p in re.split(r"\n\s*\n", text)]
    if len(paras) <= 1:
        # single block: try last-sentence-window bio strip
        sents = re.split(r"(?<=[.!?])\s+", text)
        cut = len(sents)
        for i in range(len(sents) - 1, max(-1, len(sents) - 6), -1):
            if _BIO_CUES.search(sents[i]):
                cut = i
            else:
                # stop scanning once we hit a clearly-story sentence below a bio
                if cut < len(sents):
                    break
        if cut < len(sents):
            return " ".join(sents[:cut]).strip()
        return text
    # strip trailing bio paragraphs
    n = len(paras)
    cut = n
    for i in range(n - 1, max(-1, n - 4), -1):
        p = paras[i].strip()
        if not p:
            continue
        words = len(p.split())
        # bio paragraphs are usually short-to-medium and third-person about author
        if _BIO_CUES.search(p) and words < 160:
            cut = i
        else:
            break
    body = "\n\n".join(paras[:cut]).strip()
    # don't let bio strip eat the whole story
    if len(body) < 200 and len(text) > 400:
        return text
    return body


# CMS / nav boilerplate lines often left at the TOP of the extracted block:
# magazine masthead, issue label, "Read More", social buttons. Strip leading
# lines that are short and look like nav/headers, until we hit story prose.
_CMS_LINE = re.compile(
    r"^("
    r"home|about|archives?|guidelines?|submissions?|contact|masthead|issues?|blog|news|links|store|donate|subscribe|menu|search|"
    r"read more|continue reading|share (this|on)|tweet|like|follow|next|previous|back to top|"
    r"issue\s*\d+|issue\s*[:#]|vol(?:ume)?\.?\s*\d+|winter|spring|summer|fall|autumn|"
    r"posted (on|by)|filed under|tags?:|categor(y|ies):|comments?|leave a (reply|comment)|"
    r"copyright|all rights reserved|powered by|"
    r"art(work)? by |illustration by |image by |photo(graph)? by |cover art|"
    r"(\w+ ){0,3}editor .{0,50}(on (today|this)|introduces|on our|picks|bonus flash|flash:)|"
    r"(read|listen to) .{0,30}(interview|intro|conversation)|"
    r"on today'?s? (bonus |flash|story)|this week'?s? (flash|story)|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|"
    r"by\s+[A-Z]"
    r")",
    re.I,
)


def _is_prose_line(s):
    """A story/prose line: long, or ends with sentence punctuation, or contains
    enough words to be a sentence. Nav/menu/masthead lines are short fragments."""
    words = len(s.split())
    if words >= 9:
        return True
    if s.rstrip().endswith((".", "!", "?", '"', "'", ":", ",", ";")) and words >= 4:
        return True
    return False


# focused leading-boilerplate forms safe to force-drop even when long-ish:
# editorial intros and art/photo credits (NOT the generic 'by [Name]').
_LEAD_BOILER = re.compile(
    r"^("
    r"art(work)? by |illustration by |image by |photo(graph)? by |cover art|"
    r"(\w+ ){0,3}editor .{0,50}(on (today|this)|introduces|on our|picks|bonus flash|flash:)|"
    r"(read|listen to) .{0,30}(interview|intro|conversation)|"
    r"on today'?s? (bonus |flash|story)|this week'?s? (flash|story)"
    r")", re.I)


def strip_cms_boilerplate(text):
    """Drop leading nav/CMS/masthead lines until story prose starts. Magazine nav
    menus are runs of short fragment lines ('web features', 'print issues',
    'sf/ld books', 'handbooks', ...) — drop ANY leading short non-prose line, not
    only keyword matches, up to a bounded scan (so we never eat the story)."""
    if not text:
        return text
    lines = text.split("\n")
    start = 0
    # scan up to the first 20 lines; stop at the first clear prose line
    for i, ln in enumerate(lines[:20]):
        s = ln.strip()
        if not s:
            start = i + 1
            continue
        # pipe/bullet-separated nav bar ("main | info | archive | print | prize"):
        # 2+ separators of |,»,› = a menu, not prose
        if len(re.findall(r"\s[|»›·]\s", s)) >= 2 and not _is_prose_line(s):
            start = i + 1
            continue
        # editorial intro / image-credit lines can be LONG (look like prose) but
        # are still boilerplate — force-drop only at the very top (first 2 kept
        # lines), matched by the explicit keyword set, never deeper.
        # force-drop a leading editorial-intro / image-credit line (these look
        # like prose but are boilerplate). Only at the very top, only these
        # specific keyword forms, and only when the line is itself short-ish OR
        # the credit is the whole line (avoid eating a story that merely contains
        # 'editor' or 'by' mid-sentence).
        if i <= 3 and _LEAD_BOILER.match(s) and (len(s) < 120 or s.rstrip().endswith(":")):
            start = i + 1
            continue
        if _is_prose_line(s):
            break
        words = len(s.split())
        # drop short fragment lines (nav/menu/masthead/byline/issue labels),
        # explicit CMS-keyword lines, all-caps headers, and menu-y endings
        if (words <= 6 or _CMS_LINE.match(s) or s.isupper()
                or s.endswith(("|", ">", "»"))):
            start = i + 1
        else:
            break
    # a leading nav line may also be glued to the first prose with the title in
    # between ("...print | prize Title by Author If God were..."); cut a leading
    # nav prefix off the first kept line.
    out = "\n".join(lines[start:]).strip()
    m = re.match(r"^([a-z][a-z ]*?(?:[|»›·]\s*[a-z][a-z ]*?){2,})\s+(?=[A-Z])", out)
    if m and len(m.group(1)) < 60:
        out = out[m.end():].strip()
    return out


# --------------------------------------------------------------- html -> text
def extract_main_text(html_str):
    """Largest paragraph-rich text block. Wayback toolbar removed. Same heuristic
    as the validated pilot fetcher (trafilatura not installed)."""
    soup = BeautifulSoup(html_str, "html.parser")
    for sel in ["script", "style", "nav", "header", "footer", "form", "aside",
                "noscript", "iframe", "svg", "button", "figure"]:
        for t in soup.find_all(sel):
            t.decompose()
    for tid in ["wm-ipp-base", "wm-ipp", "donato"]:
        t = soup.find(id=tid)
        if t:
            t.decompose()
    total_len = len(soup.get_text(" ", strip=True)) or 1
    for t in soup.find_all(attrs={"class": re.compile(
            r"comment|sidebar|share|related|menu|nav-|footer|header|widget|breadcrumb",
            re.I)}):
        if t.name in ("body", "html", "main", "article") or t.parent is None:
            continue
        if len(t.get_text(" ", strip=True)) > 0.6 * total_len:
            continue
        t.decompose()
    candidates = soup.find_all(["article", "main", "div", "td", "section", "body"])
    best, best_len = None, 0
    for c in candidates:
        paras = c.find_all(["p", "br"])
        txt = c.get_text(" ", strip=True)
        link_txt = sum(len(a.get_text()) for a in c.find_all("a"))
        score = len(txt) - 2 * link_txt + 50 * min(len(paras), 30)
        if score > best_len:
            best, best_len = c, score
    if best is None:
        return ""
    parts = []
    for block in best.find_all(["p", "h1", "h2", "blockquote", "li"]) or [best]:
        t = block.get_text(" ", strip=True)
        if t:
            parts.append(t)
    text = "\n\n".join(parts) if parts else best.get_text("\n", strip=True)
    return text.strip()


def strip_inline_bio(text):
    """Some pages append the bio to the LAST story sentence with no break, e.g.
    '...merely gaslighting him? Maureen Kingston lives and works in eastern
    Nebraska. She is...'. Find a sentence boundary followed by a clear bio start
    in the last few sentences and cut there."""
    if not text:
        return text
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) < 2:
        return text
    # inspect the last up-to-5 sentences; cut at the EARLIEST trailing bio
    # sentence (trailing bios are contiguous: 'X is a writer. He lives in Y.')
    lo = max(1, len(sents) - 5)
    cut = None
    for i in range(len(sents) - 1, lo - 1, -1):
        if _BIO_CUES.search(sents[i]):
            cut = i
        elif cut is not None:
            break
    if cut is not None:
        cand = " ".join(sents[:cut]).strip()
        if len(cand) > 200:
            return cand
    # fallback: a bio cue mid-sentence (no clean boundary). Find the cue in the
    # tail and cut at the last sentence-end BEFORE it.
    tail = text[-400:]
    m = _BIO_CUES.search(tail)
    if m:
        abs_pos = len(text) - 400 + m.start()
        # find last [.!?] before abs_pos
        before = text[:abs_pos]
        b = max(before.rfind(". "), before.rfind("? "), before.rfind("! "))
        if b > 200:
            return text[:b + 1].strip()
    return text


# Pages where extraction grabbed a cookie banner, privacy notice, paywall, or
# sidebar schedule instead of the story. Detect and reject (identically for both
# classes) so these never enter the dataset as "text".
_JUNK_PAGE = re.compile(
    r"(this data is anonymized|strictly necessary cookie|cookie settings|"
    r"recognising you when you return|privacy policy page|withdraw permission|"
    r"please enable javascript|enable cookies|"
    r"you are being redirected|subscribe to (read|continue)|"
    r"this content is for (members|subscribers)|"
    r"\bTBD\b.*\bTBD\b.*\bTBD\b)", re.I)


def looks_like_junk_page(text):
    if not text or len(text) < 120:
        return True
    if _JUNK_PAGE.search(text):
        return True
    # mostly dates/short tokens (sidebar schedule)
    words = text.split()
    if words:
        datey = sum(1 for w in words if re.match(r"\d{1,2}/\d{1,2}", w) or w == "TBD")
        if datey / len(words) > 0.2:
            return True
    return False


def full_pipeline(html_str):
    """html -> (raw_text, clean_text). raw_text = extracted+CMS-stripped (pre-bio,
    pre-normalize); clean_text = bio-stripped + trailing-junk-stripped + fully
    normalized. Both classes go through this so presentation never differs by
    source."""
    raw = extract_main_text(html_str)
    raw = strip_cms_boilerplate(raw)
    clean = strip_trailing_junk(raw)
    clean = strip_bio_tail(clean)
    clean = strip_inline_bio(clean)
    clean = strip_trailing_junk(clean)   # second pass: junk exposed after bio cut
    clean = normalize_text(clean)
    return raw, clean


if __name__ == "__main__":
    import sys
    h = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    r, c = full_pipeline(h)
    print("RAW:\n", r[:500], "\n\nCLEAN:\n", c[:500])
