# AUTO: blind rule compilation chunk c5
import re


def score__math__a168(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    advanced = [
        "abstract algebra", "category theory", "differential geometry", "algebraic geometry",
        "functional analysis", "complex analysis", "real analysis", "measure theory",
        "topology", "manifold", "homology", "cohomology", "group action", "ring homomorphism",
        "sigma-algebra", "lebesgue", "hilbert space", "banach space", "functor", "morphism",
        "riemannian", "galois", "spectral sequence", "distribution theory"
    ]
    intermediate = [
        "calculus", "derivative", "integral", "linear algebra", "matrix", "eigenvalue",
        "probability", "variance", "number theory", "differential equation", "vector space",
        "limit", "continuity", "gradient", "series", "modulo", "induction"
    ]
    formal = [
        "theorem", "lemma", "proposition", "corollary", "definition", "suppose that",
        "assume that", "we claim", "it follows", "therefore", "hence", "contradiction",
        "if and only if", "q.e.d", "qed", "proof"
    ]
    elementary = ["add", "subtract", "multiply", "divide", "arithmetic", "solve for", "equation"]
    a = sum(t.count(x) for x in advanced)
    m = sum(t.count(x) for x in intermediate)
    f = sum(t.count(x) for x in formal)
    e = sum(t.count(x) for x in elementary)
    symbols = len(re.findall(r"(?:\\forall|\\exists|\\in|\\sum|\\int|[∀∃∈⊂∑∫])", text))
    if a:
        return float(min(10, 8.0 + min(1.2, 0.3 * a) + min(0.8, 0.12 * (f + symbols))))
    if m:
        return float(min(7.8, 4.5 + min(1.6, 0.28 * m) + min(1.7, 0.18 * (f + symbols))))
    if f >= 3:
        return float(min(6.0, 2.5 + 0.45 * f))
    if e or re.search(r"\d\s*[-+*/=^]\s*\d", text):
        return float(min(2.5, 0.8 + 0.3 * e + 0.15 * f))
    return float(min(2.0, 0.3 + 0.2 * f))


def score__code_review__a126(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    deep = [
        "architecture", "trade-off", "tradeoff", "race condition", "deadlock", "memory leak",
        "edge case", "complexity", "thread-safe", "concurrency", "transaction", "rollback",
        "security", "vulnerability", "invariant", "backward compatibility", "failure mode",
        "root cause", "instead", "alternative", "because", "otherwise", "would fail"
    ]
    minor = ["typo", "rename", "formatting", "whitespace", "nit", "lint", "indent", "spelling"]
    technical = [
        "function", "method", "class", "api", "database", "query", "cache", "exception",
        "error", "test", "type", "null", "async", "performance", "algorithm", "interface"
    ]
    d = sum(t.count(x) for x in deep)
    n = sum(t.count(x) for x in minor)
    tech = sum(t.count(x) for x in technical)
    explanations = len(re.findall(r"\b(?:because|since|so that|which means|for example|e\.g\.)\b", t))
    code = len(re.findall(r"`[^`]+`|\b\w+\([^\n)]*\)|\b(?:if|for|while|return|try|except)\b", text))
    terse = len(re.findall(r"(?mi)^\s*(?:lgtm|looks good|fix this|remove|rename|nit)\W*$", text))
    substantive = d * 1.25 + tech * 0.25 + explanations * 0.8 + min(code, 8) * 0.2
    score = 1.0 + min(8.5, substantive) - min(4.0, n * 0.55 + terse)
    return float(max(0.0, min(10.0, score)))


def score__math__a198(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    verify = bool(re.search(
        r"\b(?:is|was)\s+(?:this|my|the)\s+(?:proof|solution|argument)\s+(?:correct|valid|sound)|"
        r"\b(?:check|verify|validate|review)\s+(?:this|my|the)\s+(?:proof|solution|argument)|"
        r"\bwhere did i go wrong\b", t
    ))
    proof_markers = [
        "proof", "suppose", "assume", "therefore", "hence", "thus", "it follows",
        "we have", "we claim", "contradiction", "without loss of generality", "qed", "q.e.d"
    ]
    p = sum(t.count(x) for x in proof_markers)
    math = len(re.findall(r"[=<>≤≥∑∫√^]|\\(?:frac|sum|int|sqrt|begin)|\b(?:theorem|lemma|integer|function|set|matrix)\b", text))
    words = len(re.findall(r"\b\w+\b", text))
    if verify and (p >= 2 or math >= 3):
        return 10.0
    if p >= 5 and math >= 3 and words >= 100:
        return 6.7
    if (p >= 2 and math >= 1) or (math >= 2 and words >= 45):
        return 3.3
    return 0.0


def score__code_review__a153(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    substance = [
        "architecture", "design", "edge case", "race condition", "performance", "security",
        "complexity", "api", "interface", "database", "transaction", "concurrency", "failure",
        "test case", "alternative", "because", "rationale", "backward compatibility"
    ]
    superficial = ["nit", "typo", "rename", "format", "whitespace", "spelling", "lint"]
    s = sum(t.count(x) for x in substance)
    shallow = sum(t.count(x) for x in superficial)
    detail = len(re.findall(r"\b(?:because|since|so that|for example|otherwise|instead|consider)\b", t))
    code = len(re.findall(r"`[^`]+`|```|\b\w+\([^)]*\)", text))
    terse = len(re.findall(r"(?mi)^\s*(?:remove|rename|fix this|why\??|nit:?|lgtm)\s*[.!]?$", text))
    dialogue = len(re.findall(r"(?mi)^\s*(?:reviewer|author|reply|response|comment)\s*[:\-]", text))
    score = 1.2 + min(8.8, 0.75 * s + 0.6 * detail + 0.18 * min(code, 10) + 0.25 * dialogue)
    score -= min(4.0, 0.45 * shallow + 0.8 * terse)
    return float(max(0.0, min(10.0, score)))


def score__CAL__CAL5(text):
    if not isinstance(text, str):
        return 0.0
    count = len(re.findall(r"(?<![A-Za-z])[A-Z]{3,}(?![A-Za-z])", text))
    return 10.0 if count >= 5 else (5.0 if count >= 1 else 0.0)


def score__code_review__a81(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    deep = [
        "architecture", "design", "trade-off", "tradeoff", "edge case", "concurrency",
        "performance", "security", "invariant", "failure mode", "api", "database", "algorithm",
        "why", "how would", "what happens", "because", "rationale", "alternative"
    ]
    shallow = ["formatting", "whitespace", "rename", "nit", "lint", "typo", "indent", "remove this"]
    d = sum(t.count(x) for x in deep)
    n = sum(t.count(x) for x in shallow)
    questions = len(re.findall(r"[^?\n]{3,}\?", text))
    roles = set(re.findall(r"(?mi)^\s*(reviewer|author|reply|response|maintainer)\s*[:\-]", text))
    replies = len(re.findall(r"(?mi)^\s*(?:author|reply|response|maintainer)\s*[:\-].{20,}", text))
    explain = len(re.findall(r"\b(?:because|since|the reason|in order to|so that|this ensures)\b", t))
    if len(roles) >= 2 and replies and (d + questions + explain >= 5):
        return float(min(10.0, 7.7 + 0.25 * min(6, d + questions + explain)))
    score = 1.0 + min(7.0, 0.6 * d + 0.35 * questions + 0.5 * explain + 0.25 * replies)
    score -= min(3.0, 0.45 * n)
    return float(max(0.0, min(7.5, score)))


def score__press_releases__a80(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    press = bool(re.search(r"\b(?:press release|news release|for immediate release|prnewswire|business wire|globe newswire)\b", t))
    dateline = bool(re.search(r"(?m)^\s*[A-Z][A-Z .'-]{2,},\s*(?:[A-Z]{2}|[A-Z][a-z]+)?\s*,?\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b", text))
    announcement = bool(re.search(r"\b(?:announc(?:e|es|ed|ing)|launch(?:es|ed|ing)?|introduc(?:e|es|ed|ing)|unveil(?:s|ed)?|debut(?:s|ed)?)\b", t))
    offering = bool(re.search(r"\b(?:new|innovative|next-generation)\s+(?:product|service|technology|platform|solution|application|brand|initiative|program|device|system)\b", t))
    general = bool(re.search(r"\b(?:appoint(?:s|ed)?|executive|conference|event|quarterly|corporate update|partnership|acquisition|award)\b", t))
    navigation = len(re.findall(r"(?mi)^\s*(?:home|about us|contact|products|services|privacy|terms|sign in|menu)\s*$", text))
    if (press or dateline) and announcement and offering:
        return 10.0
    if (press or dateline) and (announcement or general):
        return 7.5
    if announcement and offering and navigation == 0:
        return 7.5
    return 0.0


def score__press_releases__a105(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text)
    if not words:
        return 0.0
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n\s*\n", text.strip()) if len(re.findall(r"\b\w+\b", s)) >= 4]
    long_sentences = sum(len(re.findall(r"\b\w+\b", s)) >= 8 for s in sentences)
    nav = len(re.findall(r"(?mi)^\s*(?:home|about(?: us)?|contact|menu|search|products|services|privacy|terms|sign in|log in|next|previous)\s*$", text))
    list_lines = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text))
    t = text.lower()
    official = bool(re.search(r"\b(?:press release|announc(?:e|es|ed)|for immediate release|prnewswire|business wire)\b", t))
    coherence = min(1.0, (long_sentences * 12.0) / max(1, len(words)))
    if len(words) >= 120 and long_sentences >= 5 and nav <= 2:
        score = 8.0 + min(2.0, coherence * 1.3 + min(len(words), 800) / 1000.0)
    elif len(words) >= 50 and long_sentences >= 2:
        score = 5.0 + min(2.5, coherence * 2.0 + len(words) / 400.0)
        if official:
            score = max(5.0, min(score, 7.0))
    else:
        score = 2.0 + min(2.5, long_sentences * 0.5)
    score -= min(4.0, nav * 0.6 + max(0, list_lines - long_sentences) * 0.15)
    return float(max(0.0, min(10.0, score)))


def score__press_releases__a76(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    prose_words = 0
    nonprose_words = 0
    nav_terms = re.compile(r"^(?:home|about(?: us)?|contact|menu|search|products|services|privacy|terms|sign in|log in|next|previous|subscribe)$", re.I)
    for line in lines:
        n = len(re.findall(r"\b\w+\b", line))
        looks_prose = n >= 8 and (bool(re.search(r"[.!?](?:[\"')\]]*)$", line)) or n >= 16)
        looks_list = bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line)) or bool(nav_terms.match(line))
        if looks_prose and not looks_list:
            prose_words += n
        else:
            nonprose_words += n
    ratio = prose_words / max(1, prose_words + nonprose_words)
    return float(max(0.0, min(10.0, 10.0 * ratio)))


def score__math__a54(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    answer = t.split("answer", 1)[1] if "answer" in t else t
    words = len(re.findall(r"\b\w+\b", answer))
    math = len(re.findall(r"[=<>≤≥∑∫√^]|\\(?:frac|sum|int|sqrt)|\b(?:theorem|lemma|equation|function|integer|matrix)\b", answer))
    reasoning = sum(answer.count(x) for x in ["because", "therefore", "hence", "thus", "since", "it follows", "suppose", "we have", "so that"])
    structure = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+|\b(?:first|next|finally)\b", answer))
    uncertainty = len(re.findall(r"\b(?:maybe|probably|not sure|i think)\b", answer))
    if words < 8 or (math == 0 and reasoning == 0):
        return 0.0
    if words >= 220 and reasoning >= 6 and math >= 6 and structure >= 2 and uncertainty == 0:
        return 10.0
    if words >= 90 and reasoning >= 3 and math >= 3:
        return 7.5
    if words >= 25 and (reasoning >= 1 or math >= 2):
        return 5.0
    return 0.0


def score__patents__a228(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    moving = [
        "rotat", "pivot", "slide", "reciprocat", "hinge", "gear", "shaft", "motor", "actuator",
        "piston", "spring", "valve", "bearing", "cam ", "lever", "wheel", "pulley", "conveyor",
        "movable", "movement", "mechanism"
    ]
    physical = [
        "housing", "chamber", "frame", "assembly", "fluid", "nozzle", "pipe", "structural",
        "fastener", "bracket", "plate", "container", "inlet", "outlet", "mechanical", "component"
    ]
    device = ["device", "sensor", "electrode", "material", "chemical", "substrate", "apparatus"]
    software = ["algorithm", "data processing", "database", "software", "protocol", "memory", "logic circuit", "network", "business method"]
    mv = sum(t.count(x) for x in moving)
    ph = sum(t.count(x) for x in physical)
    dv = sum(t.count(x) for x in device)
    sw = sum(t.count(x) for x in software)
    relations = len(re.findall(r"\b(?:coupled to|connected to|mounted on|disposed within|engages|configured to move|in fluid communication)\b", t))
    if mv >= 4 and ph >= 4 and relations >= 2:
        return 10.0
    if mv >= 1 and ph >= 3:
        return float(min(9.5, 7.2 + 0.25 * min(5, mv + relations)))
    if ph >= 2 or dv >= 2:
        return float(min(7.5, 4.5 + 0.25 * min(8, ph + dv)))
    if sw and not (mv or ph or dv):
        return 0.0
    return float(min(4.0, 0.5 * (mv + ph + dv)))


def score__press_releases__a25(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    cleaned = re.sub(r"<[^>]*>|https?://\S+|www\.\S+|```.*?```|`[^`]*`", " ", text, flags=re.I | re.S)
    tokens = re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)?", cleaned, flags=re.UNICODE)
    if not tokens:
        return 0.0
    common = {
        "the", "and", "of", "to", "a", "in", "is", "that", "for", "on", "with", "as", "by",
        "at", "from", "this", "be", "are", "was", "an", "or", "it", "we", "will", "has", "have",
        "not", "but", "our", "its", "their", "which", "can", "new", "about", "more", "said", "company"
    }
    ascii_tokens = [w for w in tokens if w.isascii() and re.fullmatch(r"[A-Za-z]+(?:['-][A-Za-z]+)?", w)]
    nonascii = len(tokens) - len(ascii_tokens)
    common_hits = sum(w.lower() in common for w in ascii_tokens)
    # ASCII alone is not proof of English; require common function words when there is enough prose.
    if len(ascii_tokens) >= 8 and common_hits == 0:
        english_est = min(0.35 * len(ascii_tokens), 2.0)
    else:
        english_est = len(ascii_tokens)
    ratio = english_est / max(1.0, english_est + nonascii)
    return float(max(0.0, min(10.0, 10.0 * ratio)))


def score__press_releases__a262(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    beginning = text.lstrip()[:1500]
    months = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep(?:t)?\.?|Oct\.?|Nov\.?|Dec\.?)"
    standard = re.search(
        r"(?m)^\s*[A-Z][A-Z .'-]{2,},\s*(?:[A-Z][a-z]+|[A-Z]{2})?(?:,)?\s*" + months +
        r"\s+\d{1,2},\s+\d{4}(?:\s*/(?:PRNewswire|Business Wire|GlobeNewswire)/)?\s*(?:--|—|-)",
        beginning
    )
    if standard:
        return 10.0
    partial = re.search(months + r"\s+\d{1,2},\s+\d{4}|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|/(?:PRNewswire|Business Wire|GlobeNewswire)/", beginning, re.I)
    return 2.0 if partial else 0.0


def score__press_releases__a75(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = re.findall(r"\b[A-Za-z][A-Za-z.'-]*\b", text)
    if not words:
        return 0.0
    caps = sum(w.isupper() and len(w.strip(".'-")) >= 2 for w in words)
    capitalized = sum(w[0].isupper() for w in words)
    cap_ratio = (caps * 1.6 + capitalized) / len(words)
    list_lines = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+|^\s*[A-Z][A-Za-z ]{1,30}:\s*$", text))
    menu_lines = len(re.findall(r"(?mi)^\s*(?:home|about(?: us)?|contact|products|services|news|investors|careers|privacy|menu|search)\s*$", text))
    structure = min(1.0, (list_lines + 2 * menu_lines) / max(3.0, len(text.splitlines())))
    density = min(1.0, cap_ratio / 0.45)
    score = 1.0 + 6.2 * density + 2.8 * structure
    if cap_ratio < 0.03:
        score = min(score, 1.0)
    elif cap_ratio < 0.08:
        score = min(score, 2.5)
    return float(max(0.0, min(10.0, score)))


def score__code_review__a72(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    precise = [
        "indent", "whitespace", "lint", "format", "syntax", "semicolon", "trailing comma",
        "style guide", "pep 8", "line length", "rename", "remove", "replace", "use `", "should be",
        "must ", "unused import", "type annotation"
    ]
    conversational = [
        "why", "what do you think", "could you explain", "i wonder", "rationale", "architecture",
        "trade-off", "tradeoff", "because", "i think", "perhaps", "discussion", "alternative"
    ]
    p = sum(t.count(x) for x in precise)
    c = sum(t.count(x) for x in conversational)
    questions = text.count("?")
    directives = len(re.findall(r"(?mi)^\s*(?:please\s+)?(?:add|remove|rename|replace|use|fix|indent|format|change)\b", text))
    exact_code = len(re.findall(r"`[^`]+`|```", text))
    formal = p + directives + min(5, exact_code) * 0.4
    chat = c + questions * 0.7
    total = formal + chat
    if total == 0:
        return 3.0
    ratio = formal / total
    return float(max(0.0, min(10.0, 1.0 + 8.5 * ratio)))


def score__press_releases__a100(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    dateline = bool(re.search(r"(?m)^\s*[A-Z][A-Z .'-]{2,},.{0,30}(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{4})", text))
    release = bool(re.search(r"\b(?:press release|news release|for immediate release|prnewswire|business wire|globe newswire)\b", t))
    news = bool(re.search(r"\b(?:announc(?:e|es|ed)|launch(?:es|ed)?|introduc(?:e|es|ed)|unveil(?:s|ed)|acquir(?:e|es|ed)|appoint(?:s|ed))\b", t))
    boiler = bool(re.search(r"\b(?:about\s+[A-Z][\w&.-]+|media contact|investor relations|forward-looking statements|safe harbor)\b", text, re.I))
    bad = len(re.findall(r"(?mi)^\s*(?:home|menu|products|add to cart|cookie settings|privacy|terms|404|page not found|sign in)\s*$", text))
    elements = sum([dateline, release, news, boiler])
    score = [0.0, 3.0, 6.0, 8.5, 10.0][elements]
    if bad:
        score -= min(5.0, bad * 0.8)
    if not news:
        score = min(score, 3.0)
    return float(max(0.0, min(10.0, score)))


def score__math__a6(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    words = re.findall(r"\b\w+\b", text)
    n = len(words)
    if n < 8:
        return 0.0
    steps = len(re.findall(r"(?mi)^\s*(?:step\s+)?\d+[.)]\s+|\b(?:first|second|next|finally)\b", text))
    explanations = len(re.findall(r"\b(?:because|since|therefore|so that|this means|in other words|for example|notice that)\b", t))
    analogies = len(re.findall(r"\b(?:like|analogous to|think of|imagine|intuition|intuitively)\b", t))
    jargon = len(re.findall(r"\b(?:functor|isomorphism|homeomorphism|sigma-algebra|eigenbasis|bijection|manifold|asymptotic|homomorphism)\b", t))
    avg_sentence = n / max(1, len(re.findall(r"[.!?]+", text)))
    score = 2.0 + min(5.5, steps * 0.7 + explanations * 0.65 + analogies * 0.7 + min(n, 180) / 80.0)
    if 8 <= avg_sentence <= 26:
        score += 1.0
    elif avg_sentence > 40:
        score -= 1.2
    score -= min(3.0, max(0, jargon - explanations) * 0.45)
    if n < 25 and explanations == 0:
        score = min(score, 2.0)
    return float(max(0.0, min(10.0, score)))


def score__press_releases__a41(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    nav_re = re.compile(r"^(?:home|about(?: us)?|contact|menu|search|products|services|news|investors|careers|privacy|terms|sign in|log in|register|subscribe|language|english|next|previous|cookie settings)$", re.I)
    structural = 0
    prose = 0
    for line in lines:
        n = len(re.findall(r"\b\w+\b", line))
        links = len(re.findall(r"https?://|www\.|\[[^]]+\]\([^)]*\)|<a\b", line, re.I))
        if nav_re.match(line) or links or re.match(r"^(?:[-*•]|\d+[.)])\s+", line) or n <= 3:
            structural += max(1, n)
        else:
            prose += n
    ratio = structural / max(1, structural + prose)
    return float(max(0.0, min(10.0, 10.0 * ratio)))


def score__press_releases__a66(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    finance = [
        "revenue", "earnings", "net income", "operating income", "gross margin", "ebitda",
        "earnings per share", "eps", "cash flow", "fiscal", "quarter", "dividend", "stock",
        "share price", "market capitalization", "guidance", "balance sheet", "assets", "liabilities",
        "analyst", "investment", "investor", "acquisition", "merger"
    ]
    f = sum(t.count(x) for x in finance)
    figures = len(re.findall(r"(?:[$€£¥]\s?\d[\d,.]*(?:\s?(?:million|billion|m|bn))?|\b\d+(?:\.\d+)?%|\b(?:USD|EUR|GBP)\s?\d[\d,.]*)", text, re.I))
    comparisons = len(re.findall(r"\b(?:increased|decreased|grew|declined|up|down)\s+(?:by\s+)?\d+(?:\.\d+)?%", t))
    words = max(1, len(re.findall(r"\b\w+\b", text)))
    density = (f * 4 + figures * 5 + comparisons * 4) / words
    if figures >= 6 and f >= 5 and density >= 0.16:
        return float(min(10.0, 8.5 + min(1.5, density * 3)))
    if f >= 2 or figures >= 2:
        return float(min(7.0, 3.5 + min(3.5, 0.22 * f + 0.32 * figures)))
    return float(min(3.0, 0.35 * f + 0.45 * figures))


def score__CAL__CAL4(text):
    if not isinstance(text, str):
        return 0.0
    return 10.0 if re.search(r"(?m)^#", text) else 0.0


def score__press_releases__a73(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    nav_re = re.compile(r"^(?:home|about(?: us)?|contact|menu|search|products|services|news|investors|careers|privacy|terms|sign in|log in|register|subscribe|language|next|previous|cookie settings)$", re.I)
    meaningful = 0
    boiler = 0
    for line in lines:
        n = len(re.findall(r"\b\w+\b", line))
        is_boiler = bool(nav_re.match(line)) or bool(re.search(r"\b(?:all rights reserved|cookie policy|terms of use|legal disclaimer)\b", line, re.I))
        is_link = bool(re.search(r"https?://|www\.|\[[^]]+\]\([^)]*\)|<a\b", line, re.I))
        is_fragment = n <= 3 and not re.search(r"[.!?]$", line)
        if is_boiler or is_link or is_fragment:
            boiler += max(1, n)
        else:
            meaningful += n
    if meaningful == 0:
        return 0.0
    ratio = meaningful / max(1, meaningful + boiler)
    # Non-English-only excerpts are treated as non-substantive by this rule.
    letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
    if letters:
        ascii_ratio = sum(ch.isascii() for ch in letters) / len(letters)
        if ascii_ratio < 0.35:
            return 0.0
    return float(max(0.0, min(10.0, 10.0 * ratio)))


JOB_IDS = [
    "math__a168",
    "code_review__a126",
    "math__a198",
    "code_review__a153",
    "CAL__CAL5",
    "code_review__a81",
    "press_releases__a80",
    "press_releases__a105",
    "press_releases__a76",
    "math__a54",
    "patents__a228",
    "press_releases__a25",
    "press_releases__a262",
    "press_releases__a75",
    "code_review__a72",
    "press_releases__a100",
    "math__a6",
    "press_releases__a41",
    "press_releases__a66",
    "CAL__CAL4",
    "press_releases__a73",
]
