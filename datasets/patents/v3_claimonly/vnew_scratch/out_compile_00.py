import re
import math


_DEGREE_TERMS = (
    "about", "approximately", "approximate", "substantially", "generally",
    "essentially", "relatively", "near", "nearly", "roughly", "at least",
    "at most", "no more than", "no less than", "greater than", "less than",
    "higher", "lower", "similar", "rapidly", "slowly", "thin", "thick",
    "small", "large", "minimal", "maximum", "minimum",
)

_RELATION_PATTERNS = (
    r"\b(?:on|onto|in|into|within|inside|outside|over|under|between|through|"
    r"across|along|from|to|toward|towards|away from|relative to)\b",
    r"\b(?:attached|coupled|connected|secured|mounted|fixed|disposed|positioned|"
    r"located|contiguous|responsive|configured)\s+(?:to|with|on|in|within|between)\b",
    r"\b(?:extends|faces|projects|communicates|interfaces|abuts)\s+"
    r"(?:through|from|to|toward|towards|with|on|into|between)\b",
)

_NEGATIVE_PATTERNS = (
    r"\bfree\s+(?:of|from)\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bexcluding\b",
    r"\bexcept(?:\s+for)?\b",
    r"\bother\s+than\b",
    r"\bin\s+the\s+absence\s+of\b",
    r"\bdoes\s+not\b",
    r"\bdo\s+not\b",
    r"\bnot\b",
    r"\bnon[-\w]*\b",
)

_EXTERNAL_HEADS = {
    "method", "apparatus", "system", "device", "composition", "claim",
    "element", "text", "amount", "range", "position", "state", "group",
}


def _clean(text):
    try:
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text.strip())
    except Exception:
        return ""


def _clamp(value):
    try:
        return float(max(0.0, min(10.0, value)))
    except Exception:
        return 0.0


def _word_count(text):
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def _head_words(phrase):
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", phrase.lower())
    if not words:
        return ""
    stop = {
        "first", "second", "third", "fourth", "respective", "individual",
        "corresponding", "encrypted", "minimum", "maximum", "fuel",
        "payment", "skin", "contacting", "flat", "steady", "set", "point",
    }
    while len(words) > 1 and words[-1] in stop:
        words.pop()
    return words[-1] if words else ""


def _introduced_heads(text):
    pattern = (
        r"\b(?:a|an|at\s+least\s+\d+|at\s+most\s+\d+|one\s+or\s+more|"
        r"two\s+or\s+more)\s+"
        r"([^,;:.()]{{1,100}}?)(?=\b(?:of|to|from|when|wherein|"
        r"which|that|and|or|with|having|comprising|configured)\b|[,;:.()]|$)"
    )
    heads = []
    for match in re.finditer(pattern, text, re.I):
        head = _head_words(match.group(1))
        if head:
            heads.append((head, match.start()))
    return heads


def _definite_heads(text):
    pattern = (
        r"\b(?:the|said)\s+"
        r"([^,;:.()]{{1,100}}?)(?=\b(?:of|to|from|when|wherein|"
        r"which|that|and|or|with|having|comprising|configured)\b|[,;:.()]|$)"
    )
    heads = []
    for match in re.finditer(pattern, text, re.I):
        head = _head_words(match.group(1))
        if head:
            heads.append((head, match.start()))
    return heads


def pb03(text: str) -> float:
    text = _clean(text)
    if not text or _word_count(text) < 2:
        return 0.0

    words = _word_count(text)
    score = 8.0

    if text.count("(") != text.count(")") or text.count("[") != text.count("]"):
        score -= 3.0
    if re.search(r"[,:;]\s*(?:and|or|wherein|which|that)\s*[,:;]", text, re.I):
        score -= 1.5
    if re.search(r"\b(?:a|an)\s+(?:and|or|of|to)\b", text, re.I):
        score -= 2.0
    if re.search(r"\b(?:and|or|wherein|which|that|to|of|with)\s*[,.!?;:]?\s*$", text, re.I):
        score -= 2.0
    if re.search(r"\b(\w+)\s+\1\b", text, re.I):
        score -= 1.0
    if re.search(r"[,.!?;:]{2,}", text):
        score -= 1.0
    if text.count(",") > max(8, words // 3):
        score -= 1.0
    if re.search(r"\b(?:respectively|such|same|above|below)\b", text, re.I):
        score -= 0.3
    if re.search(r"\b(?:wherein|comprising|including|configured to|such that)\b", text, re.I):
        score += 0.5
    if re.search(r"\b(?:when|if|responsive to|based on)\b.+\b(?:then|wherein|such that)\b", text, re.I):
        score += 0.3

    return _clamp(score)


def pb04(text: str) -> float:
    text = _clean(text)
    if not text or _word_count(text) < 2:
        return 0.0

    hits = []
    lower = text.lower()
    for term in _DEGREE_TERMS:
        for match in re.finditer(r"\b" + re.escape(term) + r"\b", lower):
            hits.append(match)

    if not hits:
        return 10.0

    quality = []
    for match in hits:
        left = max(0, match.start() - 90)
        right = min(len(text), match.end() + 110)
        context = text[left:right].lower()

        anchored = 0
        if re.search(r"\b\d+(?:\.\d+)?\s*%?\b", context):
            anchored += 1
        if re.search(r"\b(?:than|relative to|compared with|compared to|"
                     r" 기준|baseline|reference|threshold|limit|range|set point|"
                     r"maximum|minimum|target|specified|predetermined)\b", context):
            anchored += 1
        if re.search(r"\b(?:between|from)\b[^.;]{0,70}\b(?:and|to)\b", context):
            anchored += 1
        if re.search(r"\b(?:defined|measured|determined|calculated|based on)\b", context):
            anchored += 1

        if re.search(r"\b(?:substantially|approximately|about|roughly|near|"
                     r"relatively|generally|essentially)\b", context):
            quality.append(0.35 + 0.2 * min(3, anchored))
        else:
            quality.append(0.55 + 0.15 * min(3, anchored))

    score = 10.0 * (sum(quality) / len(quality))
    if re.search(r"\b(?:about|approximately|substantially|near|roughly)\b"
                 r"[^.;]{0,35}\b(?:or more|or less|as needed|as desired)\b", lower):
        score -= 2.0
    return _clamp(score)


def pb05(text: str) -> float:
    text = _clean(text)
    if not text or _word_count(text) < 2:
        return 0.0

    introduced = _introduced_heads(text)
    definite = _definite_heads(text)

    article_errors = len(re.findall(r"\ba\s+[aeiou]\w*", text, re.I))
    if not introduced:
        score = 7.0
        if definite:
            score -= min(3.0, 0.35 * len(definite))
        return _clamp(score - article_errors)

    introduced_heads = [head for head, _ in introduced]
    later_refs = 0
    for head, position in introduced:
        if any(ref_head == head and ref_pos > position for ref_head, ref_pos in definite):
            later_refs += 1

    coverage = later_refs / len(introduced)
    score = 4.5 + 4.5 * coverage

    dangling = 0
    known = set(introduced_heads) | _EXTERNAL_HEADS
    for head, position in definite:
        prior = any(ref_head == head and ref_pos < position for ref_head, ref_pos in introduced)
        if not prior and head not in known:
            dangling += 1

    score -= min(3.5, 0.8 * dangling)
    score -= min(2.0, 1.0 * article_errors)

    if re.search(r"\bsaid\s+(?:first|second|third|respective)\b", text, re.I):
        score -= 0.7
    if re.search(r"\b(?:the|said)\s+\w+\s+of\s+\w+\b", text, re.I):
        score += 0.2

    return _clamp(score)


def pb06(text: str) -> float:
    text = _clean(text)
    if not text or _word_count(text) < 2:
        return 0.0

    clear = 0
    for pattern in _RELATION_PATTERNS:
        clear += len(re.findall(pattern, text, re.I))

    ambiguous = len(re.findall(
        r"\b(?:associated|related|corresponding|adjacent|proximate|near|"
        r"substantially aligned|in communication|in relation)\s+(?:to|with)?\b",
        text, re.I
    ))
    dangling = bool(re.search(
        r"\b(?:on|in|to|from|with|between|through|within|toward|attached to)\s*$",
        text, re.I
    ))
    vague = len(re.findall(r"\b(?:thereof|therein|therebetween|same)\b", text, re.I))

    if clear == 0:
        score = 5.5
    else:
        score = min(10.0, 5.0 + 1.1 * math.log1p(clear) + 0.35 * min(clear, 5))

    score -= min(3.0, 1.0 * ambiguous)
    score -= min(1.5, 0.35 * vague)
    if dangling:
        score -= 3.0
    if clear and re.search(r"\b(?:between|from)\b.+\b(?:and|to)\b", text, re.I):
        score += 0.4

    return _clamp(score)


def pb07(text: str) -> float:
    text = _clean(text)
    if not text or _word_count(text) < 2:
        return 0.0

    lower = text.lower()
    matches = []
    for pattern in _NEGATIVE_PATTERNS:
        matches.extend(re.finditer(pattern, lower))

    if not matches:
        return 10.0

    qualities = []
    for match in matches:
        tail = lower[match.end():match.end() + 100]
        has_object = bool(re.match(
            r"\s+(?:a|an|the|any|each|such|one|more|less|\w+)", tail
        ))
        bounded = bool(re.search(
            r"\b(?:a|an|the|any|each|at least|at most|more than|less than|"
            r"specified|particular|selected|listed|following)\b", tail
        ))
        vague = bool(re.search(
            r"\b(?:substantially|essentially|generally|normally|usually|"
            r"as needed|as desired|not limited to)\b", lower[max(0, match.start()-20):match.end()+100]
        ))

        quality = 0.35
        if has_object:
            quality += 0.3
        if bounded:
            quality += 0.25
        if vague:
            quality -= 0.25
        qualities.append(max(0.05, min(1.0, quality)))

    score = 10.0 * (sum(qualities) / len(qualities))
    if re.search(r"\bnot\s+(?:limited|necessarily|required)\b", lower):
        score -= 1.5
    if re.search(r"\bfree\s+(?:of|from)\s*$|\bwithout\s*$", lower):
        score -= 3.0
    return _clamp(score)


def pb08(text: str) -> float:
    text = _clean(text)
    if not text or _word_count(text) < 2:
        return 0.0

    lower = text.lower()
    numbers = [
        float(value.replace(",", ""))
        for value in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", lower)
    ]
    if not numbers:
        return 10.0

    score = 7.0
    bounded_ranges = 0

    for match in re.finditer(
        r"\b(?:between|from)\s+(\d[\d,]*(?:\.\d+)?)\s*"
        r"(?:and|to|-|–)\s*(\d[\d,]*(?:\.\d+)?)", lower
    ):
        low = float(match.group(1).replace(",", ""))
        high = float(match.group(2).replace(",", ""))
        bounded_ranges += 1
        if low > high:
            score -= 4.0
        else:
            score += 0.8

    for match in re.finditer(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:to|-|–)\s*"
        r"(\d[\d,]*(?:\.\d+)?)\b", lower
    ):
        low = float(match.group(1).replace(",", ""))
        high = float(match.group(2).replace(",", ""))
        bounded_ranges += 1
        if low > high:
            score -= 4.0
        else:
            score += 0.6

    clear_bounds = len(re.findall(
        r"\b(?:at least|at most|no more than|no less than|not less than|"
        r"not greater than|less than|greater than|up to)\s+\d", lower
    ))
    score += min(1.2, 0.35 * clear_bounds)

    approximate = len(re.findall(
        r"\b(?:about|approximately|approximate|roughly|around)\s+\d", lower
    ))
    score -= min(2.0, 0.5 * approximate)

    if re.search(r"\bup\s+to\s+\d[^.;]{0,25}\bor\s+more\b", lower):
        score -= 4.0
    if re.search(r"\b(?:at least|greater than|more than)\s+(\d+(?:\.\d+)?)"
                 r"[^.;]{0,45}\b(?:at most|less than|no more than)\s+(\d+(?:\.\d+)?)", lower):
        score -= 3.0
    if re.search(r"\b(?:between|from)\s+\d[^.;]{0,45}\b(?:and|to)\s*$", lower):
        score -= 3.0
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:or more|or greater|or less)\b", lower):
        score -= 0.3

    if bounded_ranges == 0 and clear_bounds == 0:
        score -= min(2.0, 0.35 * len(numbers))
    return _clamp(score)


REGISTRY = {
    "pb03": pb03,
    "pb04": pb04,
    "pb05": pb05,
    "pb06": pb06,
    "pb07": pb07,
    "pb08": pb08,
}