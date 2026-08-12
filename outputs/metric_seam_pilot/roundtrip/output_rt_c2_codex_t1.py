# AUTO: blind rule compilation chunk c2
import re


def _clean_visible(text):
    text = "" if text is None else str(text)
    text = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _units(text):
    text = "" if text is None else str(text)
    parts = re.split(r"\n\s*\n|\n(?=(?:[-*•]|\d+[.)])\s)|(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def _substantive_comment(comment):
    low = comment.lower()
    technical = (
        "architect", "design", "api", "behavior", "correct", "bug", "race", "thread",
        "performance", "memory", "cache", "security", "test", "failure", "error", "edge case",
        "maintain", "coupl", "dependency", "interface", "contract", "algorithm", "complexity",
        "transaction", "concurr", "serialize", "validation", "logic", "state", "implementation",
        "backward compat", "refactor", "instead", "alternative", "because", "so that", "why",
    )
    trivial = (
        "typo", "whitespace", "formatting", "nit:", "rename this", "fixed", "done", "ditto",
        "add license", "delete this line", "remove this", "lint", "formatter", "style only",
    )
    if any(x in low for x in trivial) and not any(x in low for x in technical):
        return False
    words = re.findall(r"[A-Za-z0-9_]+", comment)
    has_technical = any(x in low for x in technical)
    has_reasoning = any(x in low for x in ("because", "since", "otherwise", "which means", "for example"))
    return has_technical and (len(words) >= 7 or has_reasoning or "?" in comment)


def _review_stats(text):
    comments = _units(text)
    if not comments:
        return [], 0, 0
    substantive = sum(_substantive_comment(c) for c in comments)
    long_comments = sum(len(re.findall(r"\w+", c)) >= 25 for c in comments)
    return comments, substantive, long_comments


def _dateline_parts(text):
    text = "" if text is None else str(text)
    month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    city = r"[A-Z][A-Z .'-]{2,}"
    place = city + r",\s*(?:[A-Z]{2}|[A-Z][A-Za-z .'-]+)"
    date = month + r"\s+\d{1,2}(?:,\s*\d{4})?"
    has_place_date = bool(re.search(place + r"\s*,?\s*" + date, text))
    has_full_date = bool(re.search(place + r"\s*,?\s*" + month + r"\s+\d{1,2},\s*\d{4}", text))
    wire = bool(re.search(r"(?i)/(?:PR\s*Newswire|Business\s*Wire|GlobeNewswire)/|News provided by|\b(?:PR\s*Newswire|Business\s*Wire|GlobeNewswire)\b", text))
    return has_place_date, has_full_date, wire


def score__press_releases__a86(text):
    has_date, _, wire = _dateline_parts(text)
    dateline = has_date and wire
    if not dateline:
        return 0.0
    low = ("" if text is None else str(text)).lower()
    boilerplate = any(x in low for x in ("contact cision", "online member center", "888-776-0942", "a cision company"))
    return 10.0 if boilerplate else 3.3


def score__press_releases__a42(text):
    visible = _clean_visible(text)
    if not visible:
        return 0.0
    nav_terms = (
        "home", "about us", "contact us", "menu", "search", "sign in", "log in", "subscribe",
        "share", "facebook", "twitter", "linkedin", "privacy policy", "terms of use", "terms and conditions",
        "cookie policy", "accessibility", "site map", "sitemap", "all rights reserved", "copyright",
        "careers", "investor relations", "press room", "skip to content",
    )
    lines = [x.strip() for x in visible.splitlines() if x.strip()]
    if len(lines) <= 1:
        lines = re.split(r"\s*[|•·]\s*", visible)
    total = sum(len(re.findall(r"\w+", x)) for x in lines)
    boiler = 0
    for line in lines:
        n = len(re.findall(r"\w+", line))
        low = line.lower()
        hits = sum(term in low for term in nav_terms)
        if hits >= 2 or (hits >= 1 and n <= 12):
            boiler += n
    ratio = boiler / max(1, total)
    if ratio >= 0.85:
        return 10.0
    if ratio >= 0.60:
        return 8.8
    if ratio >= 0.15:
        return 2.1
    return 0.0


def score__press_releases__a64(text):
    visible = _clean_visible(text).lower()
    if not visible:
        return 0.0
    health = (
        "health", "healthcare", "medical", "medicine", "patient", "clinical", "hospital", "disease",
        "condition", "treatment", "therapy", "pharmaceutical", "drug", "vaccine", "diagnos", "physician",
        "doctor", "nurse", "mental health", "wellness", "fitness", "sleep", "nutrition", "animal health",
        "animal welfare", "veterinary", "pet care", "public health",
    )
    navigation = ("404", "page not found", "log in", "sign in", "skip to content", "privacy policy", "site map")
    health_hits = sum(len(re.findall(r"\b" + re.escape(k) + (r"\w*" if k.endswith("s") else r"\b"), visible)) for k in health)
    nav_hits = sum(visible.count(k) for k in navigation)
    words = len(re.findall(r"\w+", visible))
    return 10.0 if health_hits >= 2 and health_hits * 20 >= max(20, words) and nav_hits <= health_hits else 0.0


def score__math__a120(text):
    s = "" if text is None else str(text)
    units = _units(s)
    if not units:
        return 0.0
    questions = s.count("?")
    q_units = sum("?" in u for u in units)
    hint_hits = len(re.findall(r"(?i)\b(?:hint|consider|think about|what if|can you|could you|try|notice|ask yourself|suppose)\b", s))
    direct = len(re.findall(r"(?i)\b(?:the answer is|therefore|thus|hence|we get|we conclude|solution|equals)\b", s))
    q_ratio = q_units / len(units)
    if q_ratio >= 0.75 and hint_hits >= 2 and direct == 0:
        return 10.0
    if q_ratio >= 0.35 and questions >= 2 and hint_hits >= 1:
        return 7.0 if direct == 0 else 5.0
    if questions and hint_hits:
        return 3.0
    return 0.0


def score__code_review__a216(text):
    comments, substantive, _ = _review_stats(text)
    if not comments:
        return 0.0
    r = substantive / len(comments)
    if r >= 0.85:
        return 10.0
    if r >= 0.60:
        return 7.0
    if r >= 0.30:
        return 3.0
    if substantive:
        return 1.0
    return 0.0


def score__code_review__a0(text):
    comments, substantive, long_comments = _review_stats(text)
    if not comments:
        return 0.0
    low = str(text).lower()
    polite = sum(low.count(x) for x in ("please", "could we", "would you", "thanks", "suggest", "what do you think"))
    solutions = sum(low.count(x) for x in ("for example", "instead", "consider", "suggest", "could use", "how about"))
    blunt = sum(low.count(x) for x in ("fixed it", "rename this", "remove.", "delete.", "wrong."))
    depth = substantive / len(comments)
    score = 2.0 + 5.0 * depth + min(1.5, 0.5 * long_comments) + min(1.0, 0.25 * (polite + solutions)) - min(2.0, blunt)
    return round(max(0.0, min(10.0, score)), 1)


def score__math__a174(text):
    s = "" if text is None else str(text)
    low = s.lower()
    verify = bool(re.search(r"\b(?:verify|check|is)\b.{0,45}\b(?:proof|argument|reasoning)\b|\b(?:proof|argument)\b.{0,35}\b(?:correct|valid)\b", low))
    if not verify:
        return 0.0
    conclusion = bool(re.search(r"\b(?:qed|therefore|thus|hence|which proves|we conclude)\b|□", low))
    proof_markers = len(re.findall(r"\\(?:begin|end|forall|exists|implies|Rightarrow|leq|geq)|\$[^$]+\$|\b(?:assume|suppose|lemma|case \d|induction)\b", s, re.I))
    equations = len(re.findall(r"(?:^|\n)\s*(?:\$+|\\\[)?.{0,100}(?:=|≤|≥|\\le|\\ge|\\equiv|\\Rightarrow).{0,100}", s))
    if conclusion and proof_markers >= 3 and equations >= 2:
        return 10.0
    if conclusion and len(re.findall(r"\w+", s)) >= 80:
        return 7.5
    return 5.0


def score__math__a60(text):
    s = "" if text is None else str(text)
    frac = len(re.findall(r"\\(?:d?frac|tfrac)\s*\{[^{}]+\}\s*\{[^{}]+\}", s))
    slash_math = len(re.findall(r"(?:\b\d+(?:\.\d+)?|[A-Za-z)}\]])\s*/\s*(?:\d+(?:\.\d+)?\b|[A-Za-z({\[])", s))
    count = frac + slash_math
    if count == 0:
        return 0.0
    complex_bonus = min(2.0, 0.5 * len(re.findall(r"\\(?:sum|int|prod|sqrt|lim)|[=+*^]", s)))
    display_bonus = 1.0 if re.search(r"\$\$|\\begin\{(?:align|equation)\}|\\\[", s) else 0.0
    return round(min(10.0, 3.0 + 1.5 * min(count, 4) + complex_bonus + display_bonus), 1)


def score__code_review__a117(text):
    comments = _units(text)
    if not comments:
        return 0.0
    request = re.compile(r"(?i)(?:^|\b)(?:please\s+)?(?:remove|use|fix|add|change|replace|move|rename|avoid|update|make|extract|ensure|consider|should|need to|could you|can you|I suggest|let(?:'s| us))\b")
    direct = sum(bool(request.search(c)) for c in comments)
    ratio = direct / len(comments)
    if ratio >= 0.80:
        return 10.0
    if ratio >= 0.55:
        return 6.7
    if ratio >= 0.20:
        return 3.3
    return 0.0


def score__math__a108(text):
    s = "" if text is None else str(text)
    n = len(s)
    elided = "[...]" in s
    if n < 2000 and not elided:
        return 0.0
    if n < 2000:
        return round(min(4.9, 0.5 + 4.4 * n / 2000.0), 1)
    if n < 2500:
        return round(5.0 + 1.9 * (n - 2000) / 500.0, 1)
    if n < 3000:
        return round(7.0 + 1.9 * (n - 2500) / 500.0, 1)
    return round(min(10.0, 9.0 + (n - 3000) / 1000.0), 1)


def score__code_review__a90(text):
    comments, substantive, long_comments = _review_stats(text)
    if not comments:
        return 0.0
    low = str(text).lower()
    replies = len(re.findall(r"(?im)^(?:reply|response|reviewer|author)\s*[:#]", str(text)))
    trivial = sum(low.count(x) for x in ("fixed", "done", "ditto", "nit:", "remove", "typo", "lint"))
    depth = substantive / len(comments)
    score = 1.0 + 5.5 * depth + min(2.0, 0.5 * long_comments) + min(1.0, 0.25 * replies) - min(2.0, 0.2 * trivial)
    return round(max(0.0, min(10.0, score)), 1)


def score__press_releases__a111(text):
    s = "" if text is None else str(text)
    has_date, full_date, wire = _dateline_parts(s)
    if has_date and full_date and wire:
        return 10.0
    if has_date and (full_date or wire):
        return 9.0
    if has_date:
        return 8.0
    low = s.lower()
    corporate = sum(x in low for x in ("about us", "contact us", "privacy policy", "terms of use", "careers", "investor relations", "copyright", "©")) >= 2
    return 2.0 if corporate else 0.0


def score__press_releases__a112(text):
    s = "" if text is None else str(text)
    low = s.lower()
    visible = _clean_visible(s)
    if not visible:
        return 0.0
    error_nav = any(x in low for x in ("404 error", "page not found")) or (len(re.findall(r"\w+", visible)) < 35 and "|" in s)
    ascii_letters = len(re.findall(r"[A-Za-z]", visible))
    other_letters = len(re.findall(r"[^\W\d_A-Za-z]", visible, re.UNICODE))
    if error_nav or (other_letters > ascii_letters):
        return 0.0
    has_date, _, wire = _dateline_parts(s)
    attribution = wire or "news provided by" in low
    announcement = bool(re.search(r"(?i)\b(?:announce[sd]?|launch(?:es|ed)?|acqui(?:re[sd]?|sition)|appoint(?:s|ed|ment)?|reports?\s+(?:financial|quarterly|annual)|new initiative|partnership)\b", s))
    markers = sum((has_date, attribution, announcement))
    if markers == 3:
        return 10.0
    if markers == 2:
        return 8.0
    if announcement:
        return 2.0
    return 1.0


def score__CAL__CAL6(text):
    count = len(re.findall(r"(?i)\bthe\b", "" if text is None else str(text)))
    if count > 10:
        return 10.0
    if count >= 3:
        return 5.0
    return 0.0


def score__patents__a90(text):
    low = ("" if text is None else str(text)).lower()
    if not low:
        return 0.0
    semiconductor = ("semiconductor", "transistor", "integrated circuit", "circuitry", "electrode", "diode", "led", "power converter", "voltage", "current source", "wafer", "gate oxide", "photodetector")
    hardware = ("processor", "memory", "cache", "camera", "sensor", "display", "controller", "bus", "logic circuit", "storage device", "audio device", "video device")
    software = ("software", "user interface", "protocol", "application", "server", "database", "network message", "printing")
    unrelated = ("chemical", "pharmaceutical", "biological", "polymer", "mechanical", "business method", "composition of matter", "agricultural")
    e = sum(low.count(x) for x in semiconductor)
    h = sum(low.count(x) for x in hardware)
    sw = sum(low.count(x) for x in software)
    un = sum(low.count(x) for x in unrelated)
    claims_components = bool(re.search(r"\b(?:claim|comprising|includes?)\b.{0,120}\b(?:circuit|processor|memory|transistor|electrode|semiconductor|sensor)\b", low, re.S))
    if e >= 2 and (claims_components or e >= h + sw + un):
        return 10.0
    if h >= 2 and h + e > sw + un:
        return 8.0
    if h + e >= 1 or sw >= 1:
        return 3.0
    return 0.0


def score__code_review__a18(text):
    comments, substantive, _ = _review_stats(text)
    if not comments:
        return 0.0
    ratio = substantive / len(comments)
    if ratio >= 0.90:
        return 10.0
    if ratio >= 0.70:
        return 8.0
    if ratio >= 0.50:
        return round(6.0 + 1.5 * (ratio - 0.50) / 0.20, 1)
    if substantive <= 2 and ratio > 0:
        return 2.0
    return 0.0


def score__code_review__a63(text):
    comments, substantive, long_comments = _review_stats(text)
    if not comments:
        return 0.0
    low = str(text).lower()
    dialogue = len(re.findall(r"(?im)^(?:reply|response|reviewer|author)\s*[:#]", str(text)))
    alternatives = sum(low.count(x) for x in ("instead", "alternative", "tradeoff", "another option", "for example"))
    ratio = substantive / len(comments)
    if ratio >= 0.75 and long_comments and (dialogue or alternatives):
        return min(10.0, 9.0 + min(1.0, 0.25 * (dialogue + alternatives)))
    if ratio >= 0.50:
        return round(min(8.0, 6.0 + 2.0 * ratio), 1)
    if ratio >= 0.15:
        return round(3.0 + 2.0 * ratio, 1)
    return 1.0 if comments else 0.0


def score__press_releases__a175(text):
    low = ("" if text is None else str(text)).lower()
    copyright_notice = "©" in low or "copyright" in low or "all rights reserved" in low
    legal = ("privacy policy", "terms of use", "terms and conditions", "accessibility", "site map", "sitemap", "cookie policy", "legal notice")
    count = sum(term in low for term in legal)
    if copyright_notice and count >= 2:
        return 10.0
    if copyright_notice and count == 1:
        return 7.0
    if count >= 2:
        return 6.5
    if copyright_notice or count == 1:
        return 4.0
    return 0.0


def score__math__a48(text):
    s = "" if text is None else str(text)
    rendered = bool(re.search(r"\\begin\{(?:CD|tikzcd|tikzpicture|xymatrix|array)\}.*?\\end\{(?:CD|tikzcd|tikzpicture|xymatrix|array)\}", s, re.S))
    exact_sequence = bool(re.search(r"(?:\\longrightarrow|\\rightarrow|--?>|→).*(?:\\longrightarrow|\\rightarrow|--?>|→)", s, re.S))
    mentions = bool(re.search(r"(?i)\b(?:commutative diagram|mathematical diagram|exact sequence|diagram commutes|commuting diagram)\b", s))
    if rendered and (mentions or exact_sequence):
        return 10.0
    if mentions:
        return 5.0
    return 0.0


def score__math__a216(text):
    s = "" if text is None else str(text)
    low = s.lower()
    latex = len(re.findall(r"\$\$.*?\$\$|\$[^$\n]+\$|\\begin\{(?:align\*?|equation\*?|gather\*?|proof)\}|\\\[.*?\\\]", s, re.S))
    proof = len(re.findall(r"\b(?:proof|assume|suppose|therefore|thus|hence|we have|it follows|qed|derive|solution)\b", low))
    excluded = bool(re.search(r"\b(?:give me a hint|hint only|verify my proof|is my proof correct|request for examples?)\b", low))
    math_ops = len(re.findall(r"[=<>≤≥]|\\(?:implies|Rightarrow|sum|int|forall|exists|leq|geq)", s))
    return 10.0 if latex >= 2 and proof >= 3 and math_ops >= 2 and not excluded else 0.0


def score__code_review__a297(text):
    comments = _units(text)
    if not comments:
        return 0.0
    substantive = sum(_substantive_comment(c) for c in comments)
    return round(10.0 * substantive / len(comments), 1)


JOB_IDS = [
    "press_releases__a86",
    "press_releases__a42",
    "press_releases__a64",
    "math__a120",
    "code_review__a216",
    "code_review__a0",
    "math__a174",
    "math__a60",
    "code_review__a117",
    "math__a108",
    "code_review__a90",
    "press_releases__a111",
    "press_releases__a112",
    "CAL__CAL6",
    "patents__a90",
    "code_review__a18",
    "code_review__a63",
    "press_releases__a175",
    "math__a48",
    "math__a216",
    "code_review__a297",
]
