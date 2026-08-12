# AUTO: blind rule compilation chunk c3
import re


def _text(text):
    return text if isinstance(text, str) else "" if text is None else str(text)


def _words(text):
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", _text(text))


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])(?:[\"'”’)]*)\s+", _text(text)) if s.strip()]


def _has_any(low, terms):
    return any(term in low for term in terms)


def _math_features(text):
    t = _text(text)
    low = t.lower()
    words = _words(t)
    display = bool(re.search(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\begin\s*\{[^}]+\}", t))
    inline = bool(re.search(r"(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)", t))
    equation = bool(re.search(r"(?:^|\s)[A-Za-z0-9_(){}^+*/\\.-]+\s*(?:=|≤|≥|<|>|\\leq|\\geq|\\equiv|\\to)\s*[^\s,.;]+", t))
    notation = display or inline or equation or bool(re.search(r"\\(?:frac|sum|int|sqrt|forall|exists|in|cdot|times|alpha|beta|theta)\b|[∑∫√∞∀∃∈⊂≠≈]", t))
    proof = _has_any(low, ("proof", "suppose", "assume", "therefore", "thus", "hence", "q.e.d", "qed", "contradiction", "induction"))
    calculation = notation or bool(re.search(r"\d\s*[-+*/^=]\s*\d", t))
    resolution = _has_any(low, ("therefore", "thus", "hence", "so the answer", "it follows", "we conclude", "counterexample", "solution", "equals", "is equal to"))
    explanatory = len(words) >= 25 and _has_any(low, ("because", "since", "which", "where", "then", "implies", "follows", "means"))
    return len(words), notation, display, proof, calculation, resolution, explanatory


def _review_depth(text):
    t = _text(text)
    low = t.lower()
    words = _words(t)
    if not words:
        return 0.0
    reason = len(re.findall(r"\b(?:because|since|therefore|however|whereas|trade-?off|reason|why|so that|in order to|prefer(?:able)?|consider)\b", low))
    technical = len(re.findall(r"\b(?:architecture|design|interface|api|algorithm|complexity|performance|concurren\w*|thread|cache|database|schema|type|memory|security|behavior|implementation|dependency|state|error|test|function|class|method|pattern|system)\b", low))
    questions = t.count("?")
    dialogue = len(re.findall(r"(?:^|\n)\s*(?:reviewer|author|reply|response|comment)\b", low))
    directive = len(re.findall(r"\b(?:remove this|fix typo|nit:|formatting|add empty line|rename this|done|lgtm)\b", low))
    length_part = min(3.0, len(words) / 65.0)
    substance = min(5.0, reason * 0.8 + technical * 0.28 + questions * 0.35)
    engagement = min(2.0, dialogue * 0.45 + (1.0 if questions and reason else 0.0))
    score = length_part + substance + engagement - min(3.0, directive * 0.65)
    if len(words) < 8:
        score = min(score, 2.5)
    return max(0.0, min(10.0, score))


def _claims_body(text, require_colon=False):
    t = _text(text)
    suffix = r"\s*:[ \t]*$" if require_colon else r"\s*:?[ \t]*$"
    match = re.search(r"(?im)^\s*CLAIMS?" + suffix, t)
    if not match:
        return None
    tail = t[match.end():]
    end = re.search(r"(?im)^\s*(?:ABSTRACT|DESCRIPTION|BACKGROUND|SUMMARY|DRAWINGS?)\s*: ?", tail)
    return tail[:end.start()] if end else tail


def _numbered_claims(body):
    if body is None:
        return []
    return re.findall(r"(?ms)^\s*(\d+)\s*[.)]\s*(.+?)(?=^\s*\d+\s*[.)]\s|\Z)", body)


def _abstract_body(text):
    t = _text(text)
    match = re.search(r"(?im)^\s*ABSTRACT\s*: ?", t)
    if not match:
        return ""
    tail = t[match.end():]
    end = re.search(r"(?im)^\s*(?:CLAIMS?|DESCRIPTION|BACKGROUND|SUMMARY|DRAWINGS?)\s*: ?", tail)
    return tail[:end.start()] if end else tail


def score__math__a222(text):
    n, notation, display, proof, calculation, resolution, explanatory = _math_features(text)
    low = _text(text).lower()
    if n == 0:
        return 0.0
    evasive = _has_any(low, ("i can't answer", "cannot answer", "what do you think", "up to you", "not enough information"))
    substantive = proof or calculation or _has_any(low, ("counterexample", "algorithm", "construct", "let ", "define "))
    if evasive and not substantive:
        return 0.0
    score = 1.0 + 3.0 * substantive + 2.0 * resolution + 1.5 * explanatory + 1.0 * notation + 1.0 * display
    if n < 12:
        score = min(score, 4.0)
    return float(max(0.0, min(10.0, score)))


def score__code_review__a162(text):
    return float(round(_review_depth(text), 2))


def score__math__a30(text):
    n, notation, display, proof, calculation, resolution, explanatory = _math_features(text)
    if n == 0:
        return 0.0
    core = proof or calculation or notation
    if not core:
        return 1.0 if n < 20 else 2.0
    score = 2.0 + 2.0 * proof + 1.5 * calculation + 1.0 * notation + 1.5 * resolution + 1.5 * explanatory
    if n < 18:
        score = min(score, 3.0)
    elif not resolution:
        score = min(score, 6.0)
    return float(max(0.0, min(10.0, score)))


def score__math__a18(text):
    n, notation, display, proof, calculation, resolution, explanatory = _math_features(text)
    if n == 0:
        return 0.0
    low = _text(text).lower()
    hint = _has_any(low, ("hint:", "as a hint", "try ", "you might try"))
    if hint and n < 45:
        return 2.0
    score = 1.0 + 1.8 * (proof or calculation) + 1.2 * notation + 1.5 * resolution + 1.7 * explanatory
    score += min(2.0, n / 90.0)
    if proof and resolution and explanatory and n >= 60:
        score = max(score, 9.0)
    elif calculation and resolution:
        score = max(score, 6.0)
    return float(max(0.0, min(10.0, score)))


def score__patents__a192(text):
    return 10.0 if re.search(r"(?im)^\s*CLAIMS?\s*:?[\t ]*$", _text(text)) else 0.0


def score__press_releases__a113(text):
    t = _text(text).strip()
    if not t:
        return 0.0
    units = [u.strip() for u in re.split(r"\n+|(?<=[.!?])\s+", t) if u.strip()]
    if not units:
        return 0.0
    boiler = re.compile(r"^(?:home|menu|search|subscribe|sign in|privacy|terms|contact|about|read more|copyright|©)(?:\b|$)", re.I)
    complete = 0.0
    weight = 0.0
    for unit in units:
        wc = len(_words(unit))
        w = max(1.0, min(20.0, wc))
        weight += w
        grammatical = wc >= 4 and bool(re.search(r"[.!?][\"'”’)]?$", unit)) and bool(re.match(r"[\"'“‘(]*[A-Z0-9]", unit))
        if grammatical and not boiler.match(unit):
            complete += w
    ratio = complete / weight if weight else 0.0
    return float(round(10.0 * ratio, 2))


def score__patents__a96(text):
    t = _text(text)
    abstract = bool(re.search(r"(?im)^\s*ABSTRACT\s*:?[\t ]*$", t))
    body = _claims_body(t)
    claims = _numbered_claims(body)
    patentish = abstract and body is not None and bool(re.search(r"\b(?:claim|invention|embodiment|apparatus|method|system|comprising)\b", t, re.I))
    return 10.0 if patentish and claims else 0.0


def score__code_review__a252(text):
    t = _text(text)
    if not t.strip():
        return 0.0
    comments = [c for c in re.split(r"\n\s*\n|(?im)^\s*(?:comment|reviewer)\s*\d*\s*:\s*", t) if c.strip()]
    automated = 0
    for c in comments:
        low = c.lower()
        bot = bool(re.search(r"\b(?:bot|lint|linter|gofmt|prettier|eslint|format go code|automated)\b", low))
        suggestion = "```suggestion" in low or bool(re.search(r"\b(?:format|formatting)\b", low))
        automated += int(bot and suggestion)
    return float(round(10.0 * automated / len(comments), 2)) if comments else 0.0


def score__patents__a36(text):
    low = _text(text).lower()
    if not low.strip():
        return 0.0
    dynamic = len(re.findall(r"\b(?:control(?:ling|led|ler)?|feedback|sense[ds]?|sensor|detect(?:ing|ed|or)?|process(?:ing|or)?|comput(?:e|er|ing|ational)|execute[sd]?|signal|data|transmit|receive|monitor|adjust|responsive to|based on)\b", low))
    static = len(re.findall(r"\b(?:composition|compound|molecule|protein|peptide|polymer|alloy|material|layer|substrate|housing|frame|bracket|fastener|chemical|formulation)\b", low))
    active_loop = _has_any(low, ("feedback loop", "based on a sensed", "based on sensed", "in response to", "dynamically adjust", "closed-loop"))
    if dynamic == 0:
        return 1.0 if static else 0.0
    score = 3.0 + min(5.0, dynamic * 0.7) + (2.0 if active_loop else 0.0) - min(3.0, static * 0.25)
    return float(max(0.0, min(10.0, round(score, 2))))


def score__math__a228(text):
    t = _text(text).strip()
    words = _words(t)
    if not words:
        return 0.0
    low = t.lower()
    n = len(words)
    filler = len(re.findall(r"\b(?:basically|actually|obviously|clearly|as you can see|it is worth noting|in conclusion|to be honest|perhaps|maybe|I think)\b", t, re.I))
    quote_words = sum(len(_words(q)) for q in re.findall(r"(?ms)^\s*>.*?(?=^\s*[^>]|\Z)", t))
    structure_penalty = 2.5 if re.search(r"\.{3,}|\[\s*(?:\.\.\.|omitted)\s*\]", t, re.I) else 0.0
    useful = _has_any(low, ("answer", "therefore", "thus", "hence", "equals", "proof", "solution", "because", "so ")) or bool(re.search(r"[=$]|\\(?:frac|sqrt|sum)", t))
    if not useful:
        return max(0.0, 3.0 - min(3.0, n / 100.0))
    score = 10.0 - max(0.0, (n - 120) / 80.0) - filler * 0.45 - 4.0 * quote_words / max(1, n) - structure_penalty
    if n < 8:
        score = min(score, 7.0)
    return float(max(0.0, min(10.0, round(score, 2))))


def score__patents__a234(text):
    abstract = _abstract_body(text)
    if not abstract:
        return 0.0
    low = abstract.lower()
    problem = _has_any(low, ("problem in the prior art", "problem of the prior art", "conventionally", "drawback", "deficiency", "limitation", "disadvantage", "fails to", "consumes bandwidth", "excessive bandwidth", "however"))
    solution = _has_any(low, ("to solve", "overcome", "address", "therefore", "provides", "proposes", "according to the invention", "is configured to", "thereby", "reduces", "improves"))
    technical = _has_any(low, ("method", "system", "device", "apparatus", "processor", "signal", "network", "circuit", "data", "controller"))
    return 10.0 if problem and solution and technical else 0.0


def score__patents__a42(text):
    body = _claims_body(text, require_colon=True)
    if body is None:
        return 0.0
    claims = _numbered_claims(body)
    if not claims:
        return 0.0
    first = claims[0][1].lower()
    canceled = bool(re.search(r"\b(?:cancel+ed|withdrawn)\b", first))
    dependent = bool(re.search(r"\bclaim\s+\d+\b", first))
    medical = bool(re.search(r"\b(?:treat(?:ing|ment)?|administer(?:ing)?|patient|disease|therapeutic)\b", first))
    concrete = bool(re.search(r"\b(?:comprising|includes?|consisting|steps? of|receiv(?:e|ing)|transmit(?:ting)?|process(?:ing)?|determining|forming|coupled|configured|providing)\b", first))
    if canceled or dependent or medical:
        return 1.5
    return 10.0 if concrete else 1.5


def score__press_releases__a118(text):
    t = _text(text).strip()
    if not t:
        return 0.0
    words = _words(t)
    if not words:
        return 0.0
    ascii_letters = len(re.findall(r"[A-Za-z]", t))
    all_letters = len(re.findall(r"[^\W\d_]", t, re.UNICODE))
    if all_letters and ascii_letters / all_letters < 0.45:
        return 0.0
    lines = [line.strip() for line in t.splitlines() if line.strip()]
    date = bool(re.search(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", t, re.I))
    byline = bool(re.search(r"(?im)^\s*(?:by\s+[A-Z][^\n]{2,60}|author\s*:\s*[^\n]+)$", t))
    headline = bool(lines and len(_words(lines[0])) >= 3 and len(_words(lines[0])) <= 20)
    body_sentences = sum(1 for s in _sentences(t) if len(_words(s)) >= 8)
    news_terms = _has_any(t.lower(), ("announced", "press release", "news", "today", "company", "according to", "said ", "reports"))
    nav = len(re.findall(r"(?im)^\s*(?:home|products?|services?|about us|contact|menu|search|privacy|terms|login|sign in)\s*$", t))
    if date and byline and headline and body_sentences >= 4:
        return 10.0
    if date and byline and body_sentences >= 2:
        return 7.5
    if date and news_terms and body_sentences >= 1:
        return 6.2
    if nav >= 3 and news_terms:
        return 5.0
    if _has_any(t.lower(), ("about us", "our company", "our mission")) and not date:
        return 1.2
    return 0.0 if nav >= max(2, body_sentences) else min(5.0, 1.0 + body_sentences * 0.7)


def score__math__a234(text):
    t = _text(text)
    if re.search(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\begin\s*\{[^}]+\}", t):
        return 10.0
    if re.search(r"(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)", t):
        return 4.4
    return 0.0


def score__code_review__a9(text):
    return float(round(_review_depth(text), 2))


def score__patents__a0(text):
    t = _text(text)
    body = _claims_body(t, require_colon=True)
    abstract = bool(re.search(r"(?im)^\s*ABSTRACT\s*:", t))
    if body is None:
        return 1.5 if abstract else 0.0
    claims = _numbered_claims(body)
    readable = [c for c in claims if len(_words(c[1])) >= 3]
    if len(readable) >= 2:
        return 10.0
    if len(readable) == 1:
        return 5.0
    return 2.0


def score__code_review__a306(text):
    return float(round(_review_depth(text), 2))


def score__patents__a179(text):
    abstract = _abstract_body(text)
    return 10.0 if re.search(r"\[\s*\.\.\.\s*\]", abstract) else 0.0


def score__math__a114(text):
    t = _text(text)
    low = t.lower()
    n = len(_words(t))
    if n == 0:
        return 0.0
    user_ref = _has_any(low, ("your proof", "your reasoning", "your work", "your argument", "your calculation", "your approach", "you wrote", "you assumed", "your notation"))
    validation = _has_any(low, ("correct", "valid", "indeed", "works", "right", "error", "mistake", "flaw", "not valid", "incorrect"))
    constructive = len(re.findall(r"\b(?:because|however|instead|could|should|clarify|notation|alternative|improve|step|specifically)\b", low))
    if not user_ref:
        return 0.0
    score = 2.0 + 2.0 * validation + min(3.0, constructive * 0.6) + min(3.0, n / 60.0)
    return float(max(0.0, min(10.0, round(score, 2))))


def score__math__a180(text):
    n, notation, display, proof, calculation, resolution, explanatory = _math_features(text)
    if n == 0 or not notation:
        return 0.0
    score = 5.0 + 1.0 * display + 1.0 * proof + 1.0 * resolution + 1.0 * explanatory + (1.0 if n >= 50 and calculation else 0.0)
    return float(max(5.0, min(10.0, score)))


def score__press_releases__a104(text):
    t = _text(text)
    low = t.lower()
    words = _words(t)
    if not words:
        return 0.0
    numbers = re.findall(r"(?<!\w)(?:[$€£]\s*)?\d+(?:[.,]\d+)*(?:\s*%|\s*(?:million|billion|bps|basis points?))?", t, re.I)
    finance_terms = re.findall(r"\b(?:expense ratio|return|yield|revenue|earnings|profit|margin|assets? under management|aum|investment|portfolio|benchmark|dividend|ebitda|cash flow|market cap|performance|quarter|fiscal|basis points?)\b", low)
    metric_terms = re.findall(r"\b(?:expense ratio|annualized|year-over-year|yoy|return on|earnings per share|eps|basis points?|percent|percentage|cagr|volatility|sharpe)\b", low)
    density = 100.0 * (len(numbers) + len(finance_terms)) / len(words)
    if len(numbers) >= 8 and len(finance_terms) >= 5 and len(metric_terms) >= 2 and density >= 5.0:
        return 10.0
    if finance_terms or (numbers and _has_any(low, ("financial", "business", "company", "market"))):
        return 5.0
    return 0.0


JOB_IDS = [
    "math__a222",
    "code_review__a162",
    "math__a30",
    "math__a18",
    "patents__a192",
    "press_releases__a113",
    "patents__a96",
    "code_review__a252",
    "patents__a36",
    "math__a228",
    "patents__a234",
    "patents__a42",
    "press_releases__a118",
    "math__a234",
    "code_review__a9",
    "patents__a0",
    "code_review__a306",
    "patents__a179",
    "math__a114",
    "math__a180",
    "press_releases__a104",
]
