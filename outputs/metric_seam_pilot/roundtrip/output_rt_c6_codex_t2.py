# AUTO: blind rule compilation chunk c6
import re
import math


def _text(text):
    return text if isinstance(text, str) else "" if text is None else str(text)


def _words(text):
    return re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)?", _text(text).lower())


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", _text(text)) if s.strip()]


def _hits(text, terms):
    low = _text(text).lower()
    return sum(len(re.findall(r"\b" + re.escape(term) + r"\b", low)) for term in terms)


def _clamp(value):
    return float(max(0.0, min(10.0, value)))


def _headline_dateline_news(text):
    raw = _text(text).strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return False, False, 0
    first = lines[0]
    headline = (len(first.split()) <= 18 and len(first) <= 150 and
                (first.isupper() or first.istitle() or not re.search(r"[.!?]$", first)))
    months = r"January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\."
    date = rf"(?:{months})\s+\d{{1,2}}(?:,\s*\d{{4}})?|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}"
    dateline = bool(re.search(rf"(?im)^(?:[A-Z][A-Z .'-]+(?:,\s*[A-Z]{{2}})?\s*[-—,]\s*)?(?:{date})\s*[-—–,]", raw))
    news_terms = ["announces", "announced", "reports", "reported", "results", "acquisition",
                  "acquires", "appoints", "appointed", "launches", "agreement", "merger",
                  "quarter", "fiscal", "earnings", "revenue", "dividend", "statement"]
    return headline, dateline, _hits(raw, news_terms)


def _technical_profile(text):
    words = _words(text)
    n = max(1, len(words))
    advanced = _hits(text, ["theorem", "lemma", "corollary", "proposition", "proof", "bijection",
                            "isomorphism", "topology", "manifold", "measure", "sigma algebra",
                            "hilbert", "banach", "eigenvalue", "asymptotic", "convergence",
                            "complexity", "invariant", "induction", "contradiction", "derivative",
                            "integral", "algorithm", "recurrence", "optimization"])
    formulas = len(re.findall(r"(?:[=<>\u2264\u2265∈∀∃∑∫√→]|(?:sin|cos|log|lim|det)\s*\(|\$[^$]+\$|\\(?:frac|sum|int|begin))", _text(text)))
    return n, advanced, formulas


def score__press_releases__a117(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    headline, dateline, news = _headline_dateline_news(raw)
    company = _hits(raw, ["company", "corporation", "inc", "ltd", "plc", "group", "investors",
                          "shareholders", "chief executive officer", "ceo", "board of directors"])
    nav = _hits(raw, ["home", "menu", "products", "shopping cart", "sign in", "subscribe", "events"])
    paragraphs = len([p for p in re.split(r"\n\s*\n", raw) if p.strip()])
    score = 1.0 + 2.0 * headline + 2.5 * dateline + min(2.5, news * 0.7) + min(1.5, company * 0.35)
    if paragraphs >= 2:
        score += 0.5
    if nav > news + company and news == 0:
        score = min(score, 3.0)
    if headline and dateline and news and company:
        score = max(score, 9.0)
    elif (headline or dateline) and news:
        score = max(5.0, min(7.0, score))
    return _clamp(score)


def score__math__a102(text):
    raw = _text(text)
    n, advanced, formulas = _technical_profile(raw)
    if n == 1 and not raw.strip():
        return 0.0
    proof_markers = _hits(raw, ["proof", "suppose", "assume", "therefore", "thus", "hence",
                                "it follows", "because", "by induction", "contradiction", "qed",
                                "consequently", "we have", "which proves", "we conclude"])
    step_lines = len(re.findall(r"(?m)^\s*(?:step\s+\d+|\d+[.)])\s+", raw, re.I))
    hints = _hits(raw, ["hint", "try", "left as an exercise", "can someone help", "how do i"])
    answer_only = n < 45 and formulas <= 2 and proof_markers <= 1
    score = min(4.0, n / 90.0) + min(2.0, formulas / 3.0) + min(3.0, proof_markers / 2.5) + min(1.0, step_lines / 2.0)
    if answer_only:
        score = min(score, 2.0)
    if hints and proof_markers < 3:
        score = min(score, 2.0)
    if n >= 180 and formulas >= 4 and proof_markers >= 5:
        score = max(score, 7.0)
    return _clamp(score)


def score__math__a126(text):
    raw = _text(text)
    n, advanced, formulas = _technical_profile(raw)
    if not raw.strip() or (n < 35 and formulas < 2):
        return 0.0
    derivation_lines = len(re.findall(r"(?m)^.*(?:=|\u2264|\u2265|\u2192|\\frac|\\sum|\\int).*$", raw))
    density = (advanced + formulas) / max(1.0, n / 100.0)
    length_score = min(8.0, 8.0 * math.log1p(n / 35.0) / math.log1p(1000.0 / 35.0))
    depth = min(2.0, formulas / 10.0 + advanced / 18.0 + derivation_lines / 30.0)
    score = length_score + depth
    if n < 120:
        score = min(score, 3.5)
    elif n < 450:
        score = min(score, 7.0)
    if n >= 900 and density >= 3 and (formulas >= 12 or advanced >= 12):
        score = max(score, 9.0)
    return _clamp(score)


def score__patents__a30(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    core = _hits(raw, ["software", "database", "data structure", "data processing", "computer system",
                       "algorithm", "memory", "processor", "server", "storage", "indexing", "scheduler",
                       "protocol", "network", "digital communication", "control logic", "machine learning"])
    physical = _hits(raw, ["mechanical", "chemical", "agricultural", "vehicle", "radio frequency",
                           "sensor", "motor", "circuit", "material", "composition", "apparatus"])
    score = 10.0 * core / max(3.0, core + physical + 1.0)
    if core >= 5:
        score = max(score, 7.0)
    if core >= 10 and core >= 2 * physical:
        score = max(score, 9.0)
    if core == 0:
        score = 0.0
    elif physical >= 2 * core:
        score = min(score, 5.0)
    return _clamp(score)


def score__math__a156(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    n, advanced, formulas = _technical_profile(raw)
    deep = _hits(raw, ["axiom", "definition", "theorem", "lemma", "category", "homology", "cohomology",
                       "topological", "functional analysis", "measure theory", "formal logic", "model theory",
                       "non-trivial", "invariant", "generalization", "necessary and sufficient", "abstract"])
    reasoning = _hits(raw, ["proof", "therefore", "hence", "because", "implies", "it follows",
                            "contradiction", "conceptually", "interpretation", "observe that"])
    routine = _hits(raw, ["simplify", "plug in", "calculator", "basic algebra", "differentiate", "homework"])
    score = min(3.0, n / 110.0) + min(3.5, (advanced + 2 * deep) / 5.0) + min(2.5, reasoning / 3.0) + min(1.0, formulas / 8.0)
    if deep < 2 and advanced < 4:
        score = min(score, 5.0)
    if routine > deep + advanced and deep == 0:
        score = min(score, 2.0)
    if deep >= 5 and reasoning >= 5 and n >= 180:
        score = max(score, 8.0)
    return _clamp(score)


def score__CAL__CAL1(text):
    count = len(re.findall(r"[0-9]", _text(text)))
    return 10.0 if count >= 3 else 5.0 if count else 0.0


def score__press_releases__a65(text):
    raw = _text(text)
    headline, dateline, news = _headline_dateline_news(raw)
    corporate = _hits(raw, ["company", "corporation", "inc", "plc", "investor", "shareholder",
                            "acquisition", "financial results", "chief executive", "board"])
    finance = _hits(raw, ["revenue", "earnings", "net income", "sales", "fiscal", "quarter", "dividend"])
    web = _hits(raw, ["portal", "home", "navigation", "products", "learn more", "sign in", "article"])
    if dateline and news and corporate:
        return 10.0
    if (corporate and (news or finance)) or (finance >= 2 and web):
        return 4.0
    return 0.0


def score__code_review__a108(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    speaker_labels = re.findall(r"(?m)^\s*(?:@?[A-Za-z][\w.-]*|author|reviewer)\s*:", raw, re.I)
    reply_marks = len(re.findall(r"(?im)^\s*(?:reply|re:|author:|reviewer:|>\s|@\w+)", raw))
    dialogue = _hits(raw, ["thanks", "agreed", "agree", "good point", "you're right", "you are right",
                           "clarify", "to clarify", "i changed", "updated", "resolved", "done", "instead"])
    technical = _hits(raw, ["function", "class", "method", "api", "database", "thread", "memory", "cache",
                            "exception", "type", "architecture", "performance", "race condition", "test",
                            "dependency", "interface", "implementation", "algorithm"])
    exchanges = max(reply_marks, max(0, len(speaker_labels) - 1))
    if exchanges == 0:
        return 0.0
    score = 2.5 + min(2.5, exchanges * 0.8) + min(2.5, technical * 0.35) + min(2.5, dialogue * 0.45)
    if exchanges >= 4 and technical >= 5 and dialogue >= 3:
        score = max(score, 8.5)
    elif technical >= 4 and exchanges >= 2:
        score = max(score, 7.5)
    elif exchanges <= 1:
        score = min(score, 5.0)
    return _clamp(score)


def score__press_releases__a97(text):
    raw = _text(text)
    words = max(1, len(_words(raw)))
    money = len(re.findall(r"(?i)(?:[$€£]\s?\d[\d,.]*|\b\d+(?:\.\d+)?\s*(?:million|billion|trillion)\s+(?:dollars|euros|pounds)?|\b(?:USD|EUR|GBP)\s?\d)", raw))
    percent = len(re.findall(r"\b\d+(?:\.\d+)?\s*%", raw))
    eps = len(re.findall(r"(?i)\b(?:EPS|earnings per share|AUM|assets under management)\b(?:\s*(?:of|was|were|:)?\s*[$€£]?\d[\d,.]*)?", raw))
    result = len(re.findall(r"(?i)\b(?:revenue|sales|net income|operating income|profit|margin|buyback|cash flow)\b[^.\n]{0,45}\d", raw))
    table_rows = len(re.findall(r"(?m)^\s*[^\n]{0,45}\s{2,}[$€£]?[\d,(]", raw))
    metrics = money + percent + eps + result + table_rows
    if metrics == 0:
        return 0.0
    density = metrics * 100.0 / words
    financial = money + percent + eps + result
    if financial >= 10 or (financial >= 6 and density >= 3.0) or table_rows >= 5:
        return 10.0
    if financial >= 4 or (financial >= 2 and density >= 1.5):
        return 7.0
    if financial >= 2:
        return 6.0
    return 4.0 if metrics >= 2 else 2.0


def score__code_review__a198(text):
    lines = [line.strip() for line in _text(text).splitlines() if line.strip()]
    title = lines[0] if lines else ""
    if re.search(r"(?i)\bimprov(?:e?ments?|ments?|ments|ment|ements?|ments?)\b|\bimprovments\b", title):
        return 10.0
    ticket = bool(re.search(r"(?i)\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?|issue)\s*#?\s*\d+\b", title))
    trailing = bool(re.search(r"(?:#\s*\d+|(?:^|\s)[([]?\d+[)\]]?)\s*$", title))
    return 5.0 if ticket or trailing else 0.0


def score__patents__a6(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    tech = _hits(raw, ["computer", "software", "processor", "processing unit", "digital data", "database",
                       "memory", "electronic", "sensor", "image processing", "controller", "circuit",
                       "network", "server", "algorithm", "display"])
    physical = _hits(raw, ["mechanical", "chemical", "composition", "compound", "biological", "molecule",
                           "shaft", "housing", "fastener", "alloy", "polymer", "fluid"])
    if tech == 0:
        return 0.0
    ratio = tech / max(1.0, tech + physical)
    score = 1.0 + 9.0 * ratio * min(1.0, tech / 5.0)
    if tech >= 7 and ratio >= 0.65:
        score = max(score, 9.0)
    elif physical and tech and physical >= tech:
        score = max(4.0, min(6.0, score))
    return _clamp(score)


def score__press_releases__a0(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    core_terms = ["ethics", "ethical", "compliance", "code of conduct", "anti-corruption", "anticorruption",
                  "anti-bribery", "bribery", "legal standards", "regulatory compliance", "whistleblower",
                  "conflict of interest", "corporate responsibility", "responsible business", "integrity"]
    tangential = ["regulatory", "disclosure", "legal", "law", "policy", "standards", "governance"]
    core = _hits(raw, core_terms)
    related = _hits(raw, tangential)
    words = max(1, len(_words(raw)))
    sentences = max(1, len(_sentences(raw)))
    focused_sentences = sum(any(term in s.lower() for term in core_terms) for s in _sentences(raw))
    if core == 0 and related == 0:
        return 0.0
    if core == 0 or (core <= 2 and focused_sentences <= 1):
        return _clamp(3.0 + min(2.0, (core + related) / 3.0))
    share = focused_sentences / sentences
    score = 4.0 + min(3.0, core / 3.0) + min(2.0, share * 5.0) + min(1.0, (core + related) * 100 / words / 3.0)
    if core >= 8 and share >= 0.35:
        score = max(score, 9.0)
    return _clamp(score)


def score__patents__a134(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    cs = _hits(raw, ["algorithm", "software", "machine learning", "parallel processing", "data processing",
                     "memory operation", "user interface", "software-defined", "database", "computer program",
                     "execution", "operating system", "data structure", "processor", "server"])
    electronics = _hits(raw, ["electronic", "circuit", "semiconductor", "telecommunication", "wireless",
                              "transmitter", "receiver", "signal", "antenna", "computing hardware"])
    physical = _hits(raw, ["mechanical", "manufacturing", "agriculture", "chemical", "composition", "biological",
                           "vehicle", "material", "shaft", "housing", "molecule"])
    generic = _hits(raw, ["processor", "computer"])
    if cs == 0 and electronics == 0:
        return 0.0 if physical == 0 else 2.0
    if cs <= generic and physical >= 3:
        cs = 0
    if cs >= 5 and cs >= electronics and cs >= physical:
        return _clamp(8.0 + min(2.0, (cs - 5) / 4.0))
    if electronics >= cs and electronics >= physical:
        return _clamp(4.0 + min(3.0, (electronics + cs) / 5.0))
    if physical > cs + electronics:
        return _clamp(1.0 + min(2.0, (cs + electronics) / 2.0))
    return _clamp(4.0 + 4.0 * cs / max(1.0, cs + electronics + physical))


def score__patents__a84(text):
    raw = _text(text)
    words = _words(raw)
    if not words:
        return 0.0
    sentences = _sentences(raw)
    avg_sentence = len(words) / max(1, len(sentences))
    long_word_ratio = sum(len(w) >= 10 for w in words) / len(words)
    clauses = len(re.findall(r"[,;:]|\b(?:wherein|whereby|such that|configured to|comprising)\b", raw, re.I)) / max(1, len(sentences))
    jargon = _hits(raw, ["wherein", "comprising", "aforementioned", "plurality", "substrate", "transceiver",
                         "electromechanical", "semiconductor", "configuration", "corresponding", "thereof"])
    score = 10.0
    score -= min(3.5, max(0.0, avg_sentence - 14.0) / 8.0)
    score -= min(2.5, long_word_ratio * 12.0)
    score -= min(2.0, clauses / 2.5)
    score -= min(2.0, jargon / max(1.0, len(words) / 100.0) / 5.0)
    if avg_sentence > 35 or (long_word_ratio > 0.22 and clauses > 3):
        score = min(score, 3.0)
    elif avg_sentence > 24 or long_word_ratio > 0.16:
        score = min(6.0, max(4.0, score))
    return _clamp(score)


def score__press_releases__a87(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    emails = re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", raw, re.I)
    phones = re.findall(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]\d{4}(?!\w)", raw)
    addresses = re.findall(r"(?im)\b\d{1,6}\s+[A-Za-z0-9.' -]{2,45}\s(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Lane|Ln\.?|Drive|Dr\.?|Way)\b", raw)
    named_lines = re.findall(r"(?im)^(?:contact|media contact|investor relations)?\s*:?[ \t]*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[^\n]{0,100}(?:@|\d{3}[ .)-])", raw)
    details = len(emails) + len(phones) + len(addresses)
    if details == 0:
        return 0.0
    if details >= 3 and (len(named_lines) >= 1 or (emails and phones)):
        return 10.0
    return 5.0


def _simple_arithmetic_falsehood(text):
    pattern = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*([+*\-/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)(?![\w.])")
    for match in pattern.finditer(text):
        a, op, b, claimed = match.groups()
        a, b, claimed = float(a), float(b), float(claimed)
        if op == "+":
            actual = a + b
        elif op == "-":
            actual = a - b
        elif op == "*":
            actual = a * b
        elif b == 0:
            return True
        else:
            actual = a / b
        if not math.isclose(actual, claimed, rel_tol=1e-9, abs_tol=1e-9):
            return True
    return False


def score__math__a42(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    explicit_failure = re.search(r"(?i)\b(?:this (?:answer|proof|claim) is (?:incorrect|false|invalid)|counterexample\s*:|contradiction\s*:|does not follow|division by zero|false conclusion|logical fallacy)\b", raw)
    malformed = re.search(r"(?i)\b(?:0\s*=\s*1|1\s*=\s*0|sqrt\s*\(\s*-\d+(?:\.\d+)?\s*\)\s*(?:is|=)\s*(?:a\s+)?real|all\s+(?:numbers|integers|reals)\s+are\s+equal)\b", raw)
    return 10.0 if explicit_failure or malformed or _simple_arithmetic_falsehood(raw) else 0.0


def score__code_review__a36(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    comments = [c.strip() for c in re.split(r"\n\s*\n|(?m)^\s*[-*]\s+", raw) if c.strip()]
    sentences = _sentences(raw)
    trivial = _hits(raw, ["remove", "empty line", "indentation", "typo", "whitespace", "nit", "done", "ok",
                          "rename", "formatting", "semicolon"])
    technical = _hits(raw, ["architecture", "design", "interface", "abstraction", "performance", "complexity",
                            "race condition", "thread safety", "database", "transaction", "api", "algorithm",
                            "dependency", "maintainability", "security", "edge case", "failure mode", "invariant"])
    reasoning = _hits(raw, ["because", "therefore", "otherwise", "this means", "for example", "instead",
                            "consider", "why", "could", "would", "suggest", "tradeoff"])
    long_comments = sum(len(_words(c)) >= 25 for c in comments)
    explanatory = min(len(sentences), reasoning + long_comments)
    total_signals = trivial + technical + explanatory
    if total_signals == 0:
        return min(4.0, len(_words(raw)) / 25.0)
    substantive_ratio = (technical + explanatory) / total_signals
    score = 1.0 + 7.0 * substantive_ratio + min(2.0, (technical + long_comments) / 4.0)
    if technical >= 5 and explanatory >= 4:
        score = max(score, 8.0)
    if trivial >= 2 * (technical + explanatory) or (technical == 0 and explanatory <= 1):
        score = min(score, 3.0)
    return _clamp(score)


def score__patents__a240(text):
    raw = _text(text)
    labeled = re.search(r"(?im)^\s*CLAIMS\s*:\s*$", raw)
    if not labeled:
        return 0.0
    tail = raw[labeled.end():]
    formal = re.search(r"(?m)^\s*1\.\s+\S", tail)
    claim_language = re.search(r"(?i)\b(?:claim\s+1|what is claimed|comprising|wherein)\b", tail)
    return 10.0 if formal and claim_language else 0.0


def score__patents__a60(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    ui = _hits(raw, ["user interface", "graphical user interface", "gui", "display", "touchscreen", "touch screen",
                     "user input", "interactive", "interaction", "menu", "cursor", "screen", "visualization",
                     "human-machine interface", "button", "gesture"])
    supporting = _hits(raw, ["software", "computer", "imaging", "printing", "data handling", "application",
                             "processor", "system", "network"])
    other = _hits(raw, ["chemical", "mechanical", "composition", "radio", "antenna", "material", "engine"])
    if ui == 0:
        return 0.0
    focus = ui / max(1.0, ui + supporting + other)
    score = 3.0 + min(7.0, ui * 1.2 + focus * 3.0)
    if ui >= 6 and focus >= 0.35:
        score = max(score, 9.0)
    elif supporting >= ui or other >= ui:
        score = min(7.0, max(4.0, score))
    return _clamp(score)


def score__math__a132(text):
    raw = _text(text)
    if not raw.strip():
        return 0.0
    geom_terms = ["geometric", "geometry", "visualize", "visualization", "diagram", "shape", "space", "curve",
                  "surface", "angle", "triangle", "circle", "polygon", "coordinate", "dimension", "distance",
                  "length", "area", "volume", "intersection", "symmetry", "rotation", "projection", "manifold"]
    intuition_terms = ["intuition", "interpretation", "picture", "view", "see", "imagine", "represents", "means"]
    algebra_terms = ["algebra", "calculate", "compute", "equation", "substitute", "formal", "logic", "simplify"]
    geom = _hits(raw, geom_terms)
    intuition = _hits(raw, intuition_terms)
    algebra = _hits(raw, algebra_terms)
    sentences = _sentences(raw)
    geom_sentences = sum(any(term in s.lower() for term in geom_terms) for s in sentences)
    if geom == 0:
        return 0.0
    share = geom_sentences / max(1, len(sentences))
    if geom <= 2 and share < 0.25:
        return 3.0 if algebra > 0 else 4.0
    if share >= 0.55 and geom >= 6 and intuition >= 2:
        return 10.0
    if share >= 0.3 or geom >= 5:
        return 7.0 if intuition or geom >= algebra else 6.0
    return 4.0


JOB_IDS = [
    "press_releases__a117",
    "math__a102",
    "math__a126",
    "patents__a30",
    "math__a156",
    "CAL__CAL1",
    "press_releases__a65",
    "code_review__a108",
    "press_releases__a97",
    "code_review__a198",
    "patents__a6",
    "press_releases__a0",
    "patents__a134",
    "patents__a84",
    "press_releases__a87",
    "math__a42",
    "code_review__a36",
    "patents__a240",
    "patents__a60",
    "math__a132",
]
