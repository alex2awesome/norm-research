# AUTO: blind rule compilation chunk c2
import re


def _text(value):
    return value if isinstance(value, str) else "" if value is None else str(value)


def _visible(value):
    s = _text(value)
    s = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = re.sub(r"(?i)\b(?:https?://|www\.)\S+", " ", s)
    s = re.sub(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);", " ", s)
    return re.sub(r"[ \t]+", " ", s).strip()


_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_PLACE = r"[A-Z][A-Z .'-]{1,35}"
_REGION = r"(?:[A-Z]{2}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"


def _dateline_parts(text):
    s = _visible(text)
    full = re.search(
        rf"\b{_PLACE},\s*{_REGION},?\s+{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
        s,
    )
    no_day = re.search(rf"\b{_PLACE},\s*{_REGION},?\s+{_MONTH}\s+\d{{4}}\b", s)
    wire = re.search(r"(?i)(?:/\s*(?:PR\s*)?Newswire\s*/|news\s+provided\s+by|\b(?:PR\s*Newswire|Business\s+Wire|GlobeNewswire)\b)", s)
    return bool(full), bool(no_day), bool(wire)


def _lines(text):
    s = _visible(text)
    return [x.strip() for x in re.split(r"[\r\n]+|\s*[|•]\s*", s) if x.strip()]


def _comments(text):
    s = _text(text).strip()
    if not s:
        return []
    chunks = re.split(r"\n\s*\n+|\n(?=(?:[-*]\s+|(?:comment|reviewer|author|reply)\s*[:#]))", s, flags=re.I)
    chunks = [re.sub(r"^(?:[-*]\s+|(?:comment|reviewer|author|reply)\s*[:#]\s*)", "", c.strip(), flags=re.I)
              for c in chunks if c.strip()]
    if len(chunks) == 1:
        lines = [x.strip() for x in s.splitlines() if x.strip()]
        if len(lines) > 1:
            chunks = lines
    return chunks


_TECH = re.compile(r"(?i)\b(?:architect(?:ure|ural)?|design|API|interface|contract|behavior|logic|bug|incorrect|correctness|race|thread|concurren|async|deadlock|security|validation|error|exception|performance|latency|memory|cache|database|schema|transaction|algorithm|complexity|test|coverage|mock|dependency|coupl|maintain|compatib|edge case|overflow|null|state|invariant|refactor|implementation)\b")
_REASON = re.compile(r"(?i)\b(?:because|since|so that|otherwise|which (?:means|causes)|in order to|for example|e\.g\.|instead|alternative|trade-?off|consider|what if|could|would)\b")
_TRIVIAL = re.compile(r"(?i)\b(?:nit|typo|whitespace|format(?:ting)?|indent(?:ation)?|spelling|rename|license|lint|prettier|semicolon|comma|style)\b")
_ACK = re.compile(r"(?i)^\s*(?:fixed(?: it)?|done|ditto|thanks|thank you|ack|agreed|resolved|lgtm|ok(?:ay)?|remove(?: this)?|delete(?: this| it)?)[.!\s]*$")


def _substantive(comment):
    c = comment.strip()
    if not c or _ACK.fullmatch(c):
        return False
    words = re.findall(r"\b\w+\b", c)
    tech = bool(_TECH.search(c))
    reason = bool(_REASON.search(c))
    question = "?" in c and (tech or len(words) >= 10)
    code = bool(re.search(r"`[^`]+`|```|\bdef\s+\w+|\bclass\s+\w+|\w+\([^)]*\)", c))
    trivial_only = bool(_TRIVIAL.search(c)) and not tech and not reason and len(words) < 18
    return not trivial_only and (question or (tech and (reason or len(words) >= 12)) or (code and len(words) >= 10) or len(words) >= 32)


def _direct_request(comment):
    c = comment.strip()
    if not c or _ACK.fullmatch(c):
        return bool(re.match(r"(?i)^\s*(?:remove|delete|fix)\b", c))
    return bool(re.search(
        r"(?i)(?:^|[.!?]\s+)(?:please\s+)?(?:add|remove|delete|use|change|fix|replace|move|extract|rename|avoid|make|update|ensure|consider|prefer|document|test|handle|return|split|merge|keep|drop)\b|"
        r"\b(?:should|need to|needs to|must|please|I suggest|I'd recommend|can you|could you|would you)\b",
        c,
    ))


def _interp(x, points):
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def score__press_releases__a86(text):
    full, _, wire = _dateline_parts(text)
    if not full:
        return 0.0
    tail = _visible(text)[-3000:]
    boiler = re.search(r"(?i)(?:contact\s+cision|online\s+member\s+center|888[- ]776[- ]0942|a\s+cision\s+company)", tail)
    return 10.0 if wire and boiler else 3.3


def score__press_releases__a42(text):
    lines = _lines(text)
    if not lines:
        return 0.0
    nav = re.compile(r"(?i)\b(?:home|about us|contact us|menu|navigation|search|sign in|log in|subscribe|share|facebook|twitter|linkedin|instagram|privacy(?: policy)?|terms(?: of (?:use|service))?|cookie|accessibility|site ?map|copyright|all rights reserved|careers|investors|press room)\b")
    counts = {}
    for line in lines:
        key = line.lower()
        counts[key] = counts.get(key, 0) + 1
    boiler = total = 0.0
    for line in lines:
        words = re.findall(r"\b\w+\b", line)
        weight = max(1, len(words))
        total += weight
        signals = len(nav.findall(line))
        is_boiler = signals >= 1 and (len(words) <= 14 or signals >= 2)
        is_boiler = is_boiler or (counts[line.lower()] > 1 and len(words) <= 20)
        if is_boiler:
            boiler += weight
    ratio = boiler / total if total else 0.0
    return round(max(0.0, min(10.0, _interp(ratio, [(0.0, 0.0), (0.12, 2.1), (0.55, 7.0), (0.75, 8.8), (0.92, 10.0)]))), 1)


def score__press_releases__a64(text):
    s = _visible(text).lower()
    if not s:
        return 0.0
    health = re.compile(r"\b(?:health(?:care)?|medical|medicine|patient|doctor|physician|hospital|clinic|disease|condition|treatment|therapy|diagnos|symptom|pharmaceutical|drug|vaccine|public health|mental health|wellness|fitness|exercise|sleep|nutrition|animal health|animal welfare|veterinary|veterinarian|pet care|nursing|surgery|cancer|diabetes|infection)\w*\b")
    other = re.compile(r"\b(?:stock|shareholder|revenue|profit|earnings|merger|acquisition|software|cloud|cryptocurrency|election|campaign|consumer electronics|automobile|real estate)\w*\b")
    hs = len(health.findall(s))
    os = len(other.findall(s))
    words = len(re.findall(r"\b\w+\b", s))
    navigation = len(re.findall(r"\b(?:menu|sign in|log in|404|page not found|privacy policy|site map)\b", s))
    substantive = words >= 20 and navigation * 12 < words
    return 10.0 if substantive and hs >= 2 and hs >= os else 0.0


def score__math__a120(text):
    s = _visible(text)
    if not s:
        return 0.0
    sentences = [x for x in re.split(r"(?<=[.!?])\s+|\n+", s) if x.strip()]
    questions = sum("?" in x for x in sentences)
    hints = len(re.findall(r"(?i)\b(?:hint|consider|think about|what (?:if|happens|does)|how (?:would|could|can)|can you|try|notice|suppose|ask yourself)\b", s))
    direct = len(re.findall(r"(?i)\b(?:the answer is|therefore|thus|hence|we obtain|we get|the solution is|equals)\b", s))
    qratio = questions / max(1, len(sentences))
    score = 10.0 * min(1.0, qratio * 1.35 + min(hints, 4) * 0.10) - min(5.0, direct * 1.5)
    if questions == 0 and hints == 0:
        score = 0.0
    elif direct >= 2 and qratio < 0.35:
        score = min(score, 3.0)
    return round(max(0.0, min(10.0, score)), 1)


def score__code_review__a216(text):
    comments = _comments(text)
    if not comments:
        return 0.0
    ratio = sum(_substantive(c) for c in comments) / len(comments)
    return round(max(0.0, min(10.0, _interp(ratio, [(0.0, 0.0), (0.15, 1.0), (0.45, 3.0), (0.72, 7.0), (0.92, 10.0)]))), 1)


def score__code_review__a0(text):
    comments = _comments(text)
    if not comments:
        return 0.0
    whole = " ".join(comments)
    sub = sum(_substantive(c) for c in comments) / len(comments)
    reason = sum(bool(_REASON.search(c)) for c in comments) / len(comments)
    polite = len(re.findall(r"(?i)\b(?:please|thanks|thank you|could we|could you|would you|suggest|consider|perhaps|good point)\b", whole)) / len(comments)
    code = sum(bool(re.search(r"`[^`]+`|```|\b(?:def|class|return|if|for)\b", c)) for c in comments) / len(comments)
    terse = sum(len(re.findall(r"\b\w+\b", c)) <= 5 for c in comments) / len(comments)
    score = 1.5 + 4.0 * sub + 2.0 * reason + 1.4 * min(1.0, polite) + 1.4 * code - 2.0 * terse
    return round(max(0.0, min(10.0, score)), 1)


def score__math__a174(text):
    s = _visible(text)
    request = bool(re.search(r"(?i)\b(?:verify|check|is)\b.{0,45}\b(?:proof|argument|reasoning|solution)\b|\b(?:proof|argument)\b.{0,35}\b(?:correct|valid|sound)\b", s))
    if not request:
        return 0.0
    conclusion = bool(re.search(r"(?i)\b(?:therefore|thus|hence|which proves|q\.?e\.?d|conclude)\b", s))
    stuck = bool(re.search(r"(?i)\b(?:stuck|not sure how|don't know how|cannot finish|incomplete|so far)\b", s))
    steps = len(re.findall(r"(?:\\begin\{(?:align|equation)|\$[^$]+\$|\\(?:implies|therefore|forall|exists)|(?:^|\n)\s*\d+[.)])", _text(text)))
    proofish = len(re.findall(r"(?i)\b(?:assume|suppose|let|then|implies|case|lemma|theorem|therefore|thus|hence)\b", s))
    if stuck or not conclusion:
        return 5.0
    return 10.0 if steps >= 3 and proofish >= 3 else 7.5


def score__math__a60(text):
    s = _text(text)
    frac = len(re.findall(r"\\(?:d?frac|tfrac)\s*\{[^{}]*\}\s*\{[^{}]*\}", s))
    slash = len(re.findall(r"(?<![A-Za-z:])(?:\d+|[A-Za-z)]|\})\s*/\s*(?:\d+|[A-Za-z(]|\{)", s))
    n = frac + slash
    if n == 0:
        return 0.0
    displays = len(re.findall(r"\$\$|\\begin\{(?:align|equation|gather)\}|\\\[", s))
    complexity = len(re.findall(r"[=+\-*^_]|\\(?:sum|int|sqrt|lim)", s))
    score = 3.5 + min(4.0, 1.4 * (n - 1)) + min(1.5, 0.3 * complexity) + min(1.0, 0.5 * displays)
    return round(min(10.0, score), 1)


def score__code_review__a117(text):
    comments = _comments(text)
    if not comments:
        return 0.0
    ratio = sum(_direct_request(c) for c in comments) / len(comments)
    if ratio >= 0.8:
        return 10.0
    if ratio > 0.5:
        return 6.7
    if ratio >= 0.25:
        return 3.3
    return 0.0


def score__math__a108(text):
    s = _visible(text)
    n = len(s)
    elided = "[...]" in s
    if n < 2000:
        return round(min(4.9, 4.9 * n / 2000.0), 1) if elided else 0.0
    if n < 2500:
        return round(5.0 + 1.9 * (n - 2000) / 500.0, 1)
    if n < 3000:
        return round(7.0 + 1.9 * (n - 2500) / 500.0, 1)
    return round(min(10.0, 9.0 + (n - 3000) / 1500.0), 1)


def score__code_review__a90(text):
    comments = _comments(text)
    if not comments:
        return 0.0
    sub = sum(_substantive(c) for c in comments) / len(comments)
    lengths = [len(re.findall(r"\b\w+\b", c)) for c in comments]
    thought = sum(n >= 22 for n in lengths) / len(lengths)
    dialogue = min(1.0, max(0, len(comments) - 1) / 5.0)
    terse = sum(n <= 5 for n in lengths) / len(lengths)
    score = 1.0 + 6.0 * sub + 1.8 * thought + 1.2 * dialogue - 2.0 * terse
    return round(max(0.0, min(10.0, score)), 1)


def score__press_releases__a111(text):
    full, no_day, wire = _dateline_parts(text)
    if full and wire:
        return 10.0
    if full:
        return 9.0
    if no_day and wire:
        return 9.0
    if no_day or re.search(rf"\b{_PLACE},\s*{_REGION}\b", _visible(text)):
        return 8.0
    corporate = re.search(r"(?i)\b(?:about us|investor relations|careers|contact us|privacy policy|terms of use|site map|all rights reserved|corporate)\b", _visible(text))
    return 2.0 if corporate else 0.0


def score__press_releases__a112(text):
    s = _visible(text)
    if not s:
        return 0.0
    words = re.findall(r"\b\w+\b", s)
    error_nav = len(re.findall(r"(?i)\b(?:404|page not found|menu|navigation|log in|sign in|site map)\b", s))
    sentences = re.split(r"(?<=[.!?])\s+", s)
    if len(words) < 35 and (error_nav or len(sentences) < 2):
        return 0.0
    ascii_letters = sum(ord(c) < 128 and c.isalpha() for c in s)
    all_letters = sum(c.isalpha() for c in s)
    if all_letters and ascii_letters / all_letters < 0.55:
        return 0.0
    full, _, wire = _dateline_parts(s)
    provided = bool(re.search(r"(?i)\bnews provided by\b", s))
    announce = len(re.findall(r"(?i)\b(?:announce[sd]?|launch(?:es|ed)?|appoint(?:s|ed)?|acquir(?:es|ed|ing)|merger|initiative|financial results?|reports?\s+(?:results|revenue|earnings)|introduc(?:es|ed)|agreement)\b", s))
    narrative = len(words) >= 100
    if (full or provided or wire) and announce and narrative:
        return min(10.0, 8.0 + float(full) + 0.5 * float(provided or wire) + 0.5 * float(announce >= 2))
    if announce and narrative:
        return 5.0
    return 2.0 if len(words) >= 30 else 1.0


def score__CAL__CAL6(text):
    n = len(re.findall(r"\bthe\b", _text(text), flags=re.I))
    return 10.0 if n > 10 else 5.0 if n >= 3 else 0.0


def score__patents__a90(text):
    s = _visible(text).lower()
    if not s:
        return 0.0
    semiconductor = len(re.findall(r"\b(?:semiconductor|transistor|integrated circuit|diode|led|electrode|gate dielectric|wafer|photolithograph|mosfet|circuitry|power converter|voltage regulator)\w*\b", s))
    hardware = len(re.findall(r"\b(?:processor|cpu|gpu|memory|cache|register|bus|camera|sensor|display|audio|video|antenna|receiver|transmitter|electronic device|circuit|controller)\w*\b", s))
    software = len(re.findall(r"\b(?:software|user interface|application|protocol|data processing|network message|web|business method)\w*\b", s))
    unrelated = len(re.findall(r"\b(?:chemical|polymer|pharmaceutical|biological|therapeutic|mechanical linkage|gear|combustion|agricultural)\w*\b", s))
    claims_physical = bool(re.search(r"(?i)\b(?:claim|comprising|includes?)\b.{0,120}\b(?:circuit|semiconductor|processor|memory|electrode|transistor|sensor|display)\b", s))
    if semiconductor >= 2 or (semiconductor and claims_physical):
        return 10.0
    if hardware >= 2 and claims_physical:
        return 8.0
    if hardware >= 2:
        return 6.0
    if software or hardware == 1:
        return 3.0
    return 0.0 if unrelated or not hardware else 2.0


def score__code_review__a18(text):
    comments = _comments(text)
    if not comments:
        return 0.0
    ratio = sum(_substantive(c) for c in comments) / len(comments)
    if ratio >= 0.9:
        return 10.0
    if ratio >= 0.72:
        return 8.0
    if ratio >= 0.5:
        return round(6.0 + 1.5 * (ratio - 0.5) / 0.22, 1)
    if ratio <= 0.2:
        return 0.0 if ratio == 0 else 1.0 + 5.0 * ratio
    return round(2.0 + 4.0 * (ratio - 0.2) / 0.3, 1)


def score__code_review__a63(text):
    comments = _comments(text)
    if not comments:
        return 0.0
    qualities = []
    for c in comments:
        words = len(re.findall(r"\b\w+\b", c))
        q = 1.0 if _substantive(c) else 0.0
        q += 0.7 * bool(_REASON.search(c)) + 0.5 * bool("?" in c and _TECH.search(c))
        q += 0.5 * bool(re.search(r"(?i)\b(?:alternative|trade-?off|architecture|design)\b", c))
        q += 0.3 * (words >= 25)
        qualities.append(q)
    avg = sum(qualities) / len(qualities)
    dialogue = 0.6 if len(comments) >= 3 and re.search(r"(?i)\b(?:reply|agree|because|follow.?up|what about)\b", _text(text)) else 0.0
    score = min(10.0, 1.0 + 3.4 * avg + dialogue)
    return round(score, 1)


def score__press_releases__a175(text):
    s = _visible(text)
    copyright_notice = bool(re.search(r"(?i)(?:©|\bcopyright\b|\ball rights reserved\b)", s))
    patterns = [r"privacy(?: policy)?", r"terms(?: of (?:use|service))?", r"accessibility", r"site ?map", r"cookie policy", r"legal notice"]
    legal = sum(bool(re.search(rf"(?i)\b{p}\b", s)) for p in patterns)
    if copyright_notice and legal >= 2:
        return 10.0
    if copyright_notice and legal == 1:
        return 6.5
    if legal >= 2:
        return 7.0
    if copyright_notice:
        return 4.0
    if legal == 1:
        return 3.0
    return 0.0


def score__math__a48(text):
    s = _text(text)
    rendered = bool(re.search(r"\\begin\{(?:CD|tikzpicture|tikzcd|xymatrix)\}|\\xymatrix\b", s, flags=re.I))
    exact_formatted = bool(re.search(r"\\begin\{(?:array|align|aligned)\}.*?(?:\\to|\\longrightarrow|\\rightarrow).*?\\end\{(?:array|align|aligned)\}", s, flags=re.I | re.S))
    if rendered or exact_formatted:
        return 10.0
    if re.search(r"(?i)\b(?:commutative diagram|mathematical diagram|exact sequence|diagram commutes|commuting diagram)\b", _visible(s)):
        return 5.0
    return 0.0


def score__math__a216(text):
    raw = _text(text)
    s = _visible(raw)
    latex = len(re.findall(r"\$[^$\n]+\$|\$\$|\\begin\{(?:align|equation|gather|proof)\}|\\(?:frac|sum|int|lim|implies|forall|exists)\b", raw))
    reasoning = len(re.findall(r"(?i)\b(?:proof|assume|suppose|let|then|therefore|thus|hence|derive|solution|it follows|we have)\b", s))
    conclusion = bool(re.search(r"(?i)\b(?:therefore|thus|hence|which proves|q\.?e\.?d|final(?:ly)?|answer)\b", s))
    disqualify = bool(re.search(r"(?i)\b(?:give me a hint|hint only|can you verify|is my proof correct)\b", s))
    words = len(re.findall(r"\b\w+\b", s))
    return 10.0 if latex >= 3 and reasoning >= 3 and conclusion and words >= 40 and not disqualify else 0.0


def score__code_review__a297(text):
    comments = _comments(text)
    if not comments:
        return 0.0
    return round(10.0 * sum(_substantive(c) for c in comments) / len(comments), 1)


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
