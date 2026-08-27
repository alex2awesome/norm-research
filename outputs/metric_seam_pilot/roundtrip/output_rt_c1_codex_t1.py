# AUTO: blind rule compilation chunk c1
def score__patents__a24(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    words = re.findall(r"[a-z][a-z0-9-]*", s)
    if not words:
        return 0.0
    interactive = [
        "user interface", "graphical interface", "human-computer", "human computer",
        "touch screen", "touchscreen", "user input", "user interaction", "interactive",
        "display screen", "user-selectable", "user controllable", "user-controllable",
        "keyboard", "mouse", "gesture", "cursor", "menu", "button", "editing",
        "digital content", "application software", "user command", "user device",
    ]
    computing = [
        "computer", "processor", "processing", "software", "digital", "electronic",
        "memory", "data", "server", "network", "controller", "circuit", "signal",
        "database", "algorithm", "communication protocol", "transmit", "receiver",
    ]
    physical = [
        "mechanical", "manufacturing", "material", "chemical", "composition", "polymer",
        "molecule", "biological", "pharmaceutical", "alloy", "fabrication", "machining",
        "housing", "shaft", "gear", "fastener", "fluid", "combustion", "protocol",
    ]
    i = sum(s.count(x) for x in interactive)
    c = sum(s.count(x) for x in computing)
    p = sum(s.count(x) for x in physical)
    direct_user = len(re.findall(r"\buser\w*\b", s)) + len(re.findall(r"\b(operator|viewer|player)\b", s))
    if i >= 4 or (i >= 2 and direct_user >= 2):
        return float(min(10.0, 8.0 + 0.25 * min(8, i - 2) + 0.1 * min(5, direct_user)))
    if i >= 1 and c >= 2:
        return float(min(8.5, 6.5 + 0.35 * i + 0.1 * min(c, 8)))
    if c >= 3 and c >= p:
        return float(min(7.0, 4.0 + 0.3 * min(c, 10)))
    if c >= 1:
        return float(min(4.5, 2.5 + 0.35 * c))
    return float(min(3.5, 0.5 + 0.2 * p)) if p else 0.0


def score__math__a150(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.strip()
    lower = s.lower()
    # When explicit answer markers exist, discard the answer as required.
    parts = re.split(r"(?im)^\s*(?:answer|solution|response)\s*:?​?\s*$", s, maxsplit=1)
    q = parts[0]
    ql = q.lower()
    advanced = [
        "topology", "homotopy", "homology", "cohomology", "manifold", "banach",
        "hilbert space", "measure theory", "lebesgue", "functional analysis", "algebraic geometry",
        "category theory", "galois", "noetherian", "spectral sequence", "lie algebra",
        "representation theory", "distribution", "sobolev", "fundamental group", "sheaf",
        "scheme", "ring isomorphism", "compactness theorem", "ergodic", "stochastic process",
    ]
    theoretical = [
        "prove", "proof", "theorem", "lemma", "if and only if", "necessary and sufficient",
        "show that", "derive", "converge", "continuity", "isomorphism", "bijection",
        "eigenvalue", "group", "ring", "field", "vector space", "sequence", "series",
        "integral", "differentiable", "generalize", "counterexample", "existence", "unique",
    ]
    basic = [
        "calculate", "compute", "evaluate", "simplify", "solve for", "what is the value",
        "find the derivative", "find the integral", "factor", "expand", "arithmetic",
        "percentage", "decimal", "multiply", "add the", "subtract",
    ]
    a = sum(ql.count(x) for x in advanced)
    t = sum(ql.count(x) for x in theoretical)
    b = sum(ql.count(x) for x in basic)
    symbols = len(re.findall(r"[=<>∈∉⊂⊆∀∃∑∫√]|\\(?:frac|sum|int|lim|forall|exists)", q))
    conceptual = sum(ql.count(x) for x in ["why", "under what conditions", "structure", "generalization", "conceptual", "relationship", "equivalent", "characterize"])
    if a >= 2 or (a >= 1 and (t + conceptual) >= 3):
        return float(min(10.0, 8.0 + 0.35 * min(a, 4) + 0.2 * min(t + conceptual, 3)))
    if a == 1:
        return float(min(8.8, 7.2 + 0.2 * min(t + conceptual, 5)))
    if t >= 3 or (t >= 2 and conceptual):
        return float(min(7.5, 5.0 + 0.45 * min(t, 5) + 0.2 * conceptual))
    if b and t == 0 and conceptual == 0:
        return float(min(2.5, 0.8 + 0.12 * min(symbols, 8) + 0.2 * (len(re.findall(r"\b\w+\b", q)) > 30)))
    if t >= 1 or symbols >= 3:
        return float(min(5.5, 3.2 + 0.45 * t + 0.12 * min(symbols, 8)))
    return 1.0 if "?" in q else 0.5


def score__math__a204(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.strip()
    split = re.split(r"(?im)^\s*(?:answer|solution|response)\s*:?[ \t]*$", s, maxsplit=1)
    answer = split[1].strip() if len(split) == 2 else s
    words = re.findall(r"\b[\w'-]+\b", answer)
    n = len(words)
    if n < 3:
        return 0.5 if n else 0.0
    lower = answer.lower()
    math_tokens = len(re.findall(r"[=<>±×÷∑∫√^]|\\(?:frac|sum|int|sqrt|lim|begin)", answer))
    reasoning = sum(lower.count(x) for x in [
        "because", "therefore", "thus", "hence", "since", "it follows", "suppose",
        "let ", "we have", "implies", "by the", "proof", "consequently", "case",
    ])
    completion = sum(lower.count(x) for x in ["therefore", "thus", "hence", "which proves", "as required", "q.e.d", "qed", "final answer", "conclude"])
    vague = sum(lower.count(x) for x in ["maybe", "perhaps", "not sure", "i think", "try ", "hint:", "should be"])
    paragraphs = len([p for p in re.split(r"\n\s*\n", answer) if len(re.findall(r"\w+", p)) >= 12])
    score = 1.0 + min(3.0, n / 45.0) + min(2.4, reasoning * 0.45) + min(1.5, math_tokens * 0.12) + min(1.0, completion * 0.35) + min(0.8, paragraphs * 0.25)
    score -= min(2.0, vague * 0.5)
    if n < 15:
        score = min(score, 3.0)
    elif n < 40:
        score = min(score, 6.0)
    elif reasoning == 0 and math_tokens < 2:
        score = min(score, 5.0)
    return float(max(0.0, min(10.0, round(score, 1))))


def score__code_review__a27(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.strip()
    lower = s.lower()
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    technical = [
        "architecture", "design", "trade-off", "tradeoff", "alternative", "interface",
        "implementation", "dependency", "performance", "complexity", "concurrency", "thread",
        "cache", "database", "api", "error handling", "security", "memory", "test", "refactor",
        "race condition", "backward compatibility", "maintainability", "scalability",
    ]
    depth = sum(lower.count(x) for x in technical)
    reasoning = sum(lower.count(x) for x in ["because", "however", "instead", "consider", "why", "what if", "would it", "could we", "the reason", "this means"])
    speakers = set()
    for line in lines:
        m = re.match(r"(?:comment by\s+)?@?([A-Za-z][\w.-]{1,30})\s*(?::|said:)", line)
        if m:
            speakers.add(m.group(1).lower())
    reply_markers = len(re.findall(r"(?im)^\s*(?:reply|author|reviewer|response|@\w+)\s*:?", s))
    turns = max(len(speakers), min(5, reply_markers))
    superficial = len(re.findall(r"(?im)^\s*(?:done|fixed|lgtm|nit:?|thanks|remove this|looks good|agreed|ok(?:ay)?)\W*$", s))
    if turns >= 2 and depth >= 4 and reasoning >= 3:
        return float(min(10.0, 7.0 + 0.35 * min(depth, 5) + 0.3 * min(reasoning, 4) + 0.2 * min(turns - 2, 2)))
    if depth + reasoning >= 3:
        return float(min(6.8, 3.0 + 0.35 * min(depth, 6) + 0.3 * min(reasoning, 5) + 0.2 * min(turns, 2)))
    if superficial >= max(1, len(lines) // 2) or len(re.findall(r"\w+", s)) < 20:
        return float(max(0.0, min(2.0, 1.5 - 0.2 * superficial + 0.1 * depth)))
    return float(min(3.5, 1.5 + 0.35 * (depth + reasoning)))


def score__press_releases__a233(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.strip()
    lower = s.lower()
    wire = sum(x in lower for x in ["news provided by", "share this article", "pr newswire", "prnewswire", "cision", "/prnewswire/"])
    immediate = "for immediate release" in lower
    dateline = bool(re.search(r"(?im)^\s*[A-Z][A-Z .'-]{2,30}(?:,\s*[A-Z]{2}|,\s*[A-Z][a-z]+)?\s*[-—,]\s*(?:[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}\s+[A-Z][a-z]+\s+\d{4})", s))
    announcement = sum(x in lower for x in ["announces", "announced today", "launches", "unveils", "has appointed", "today announced", "new partnership", "acquisition"])
    newsroom = sum(x in lower for x in ["press release", "newsroom", "media contact", "about ", "company news"])
    hub = sum(x in lower for x in ["view all", "latest news", "search news", "categories", "navigation", "read more"])
    if wire >= 1 and dateline and announcement:
        return float(min(10.0, 9.0 + 0.25 * min(wire, 3) + 0.25 * min(announcement, 2)))
    if (immediate or dateline) and announcement and newsroom:
        return 7.5
    if announcement and newsroom and len(re.findall(r"\w+", s)) >= 80:
        return 6.0
    if hub >= 2 or (announcement and not (dateline or immediate)):
        return 2.5
    return 0.0


def score__math__a0(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.strip()
    split = re.split(r"(?im)^\s*(?:answer|solution|response)\s*:?[ \t]*$", s, maxsplit=1)
    answer = split[1].strip() if len(split) == 2 else s
    lower = answer.lower()
    words = re.findall(r"\b[\w'-]+\b", answer)
    if len(words) < 3:
        return 0.5 if words else 0.0
    equations = len(re.findall(r"[=<>≤≥∈∑∫√]|\\(?:frac|sum|int|sqrt|lim)", answer))
    valid_reasoning = sum(lower.count(x) for x in ["because", "therefore", "thus", "hence", "since", "it follows", "implies", "by theorem", "substituting", "simplifying", "proof"])
    error_signals = sum(lower.count(x) for x in ["not sure", "i don't know", "cannot solve", "impossible to answer", "maybe", "guess", "no answer"])
    gap_signals = sum(lower.count(x) for x in ["left as an exercise", "details omitted", "obviously", "clearly"])
    completion = sum(lower.count(x) for x in ["therefore", "hence", "final", "conclude", "qed", "as required"])
    score = 2.0 + min(2.5, equations * 0.25) + min(2.5, valid_reasoning * 0.5) + min(1.3, len(words) / 90.0) + min(1.0, completion * 0.35)
    score -= min(5.0, error_signals * 1.5 + gap_signals * 0.3)
    if len(words) < 12:
        score = min(score, 4.5)
    elif valid_reasoning == 0 and equations < 2:
        score = min(score, 5.5)
    elif completion and valid_reasoning >= 3 and equations >= 3:
        score = max(score, 7.0)
    return float(max(0.0, min(10.0, round(score, 1))))


def score__patents__a102(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    specific_patterns = [
        r"\b(?:escherichia coli|e\.?\s*coli|staphylococcus aureus|saccharomyces cerevisiae|bacillus subtilis)\b",
        r"\b(?:dna|rna|mrna|trna|adenosine triphosphate|atp|glucose|ethanol|methanol|acetone|benzene|toluene|sodium chloride|calcium carbonate|hydrochloric acid|sulfuric acid)\b",
        r"\b(?:hydrogen|helium|lithium|carbon|nitrogen|oxygen|fluorine|sodium|magnesium|aluminum|silicon|phosphorus|sulfur|chlorine|potassium|calcium|iron|copper|zinc|silver|gold)\b",
        r"\b(?:protein|enzyme|antibody|antigen|receptor|cytokine|hemoglobin|insulin|collagen|keratin|dopamine|serotonin)\s+[A-Z0-9-]*\b",
        r"\b(?:cancer|carcinoma|melanoma|diabetes|alzheimer(?:'s)?|parkinson(?:'s)?|hypertension|arthritis|influenza|covid-19|pneumonia|leukemia)\b",
        r"\b[A-Z][a-z]+\s+(?:coli|aureus|subtilis|cerevisiae)\b",
        r"\b(?:C\d{1,3}H\d{1,3}(?:O\d{1,3})?|NaCl|H2O2|CO2|NH3)\b",
    ]
    if any(re.search(p, text, re.I) for p in specific_patterns):
        return 10.0
    general = [
        "chemical", "chemistry", "biological", "biology", "organism", "microorganism",
        "cell", "tissue", "medical", "disease", "compound", "molecule", "protein",
        "bacteria", "virus", "fungus", "pharmaceutical", "drug", "organic", "inorganic",
        "acid", "base", "solvent", "reaction", "polymer", "patient", "therapeutic",
    ]
    return 5.0 if any(re.search(r"\b" + re.escape(x) + r"s?\b", s) for x in general) else 0.0


def score__CAL__CAL3(text):
    import re
    n = len(re.findall(r"\b\w+(?:['’-]\w+)*\b", text if isinstance(text, str) else ""))
    if n > 150:
        return 10.0
    if n >= 50:
        return 5.0
    return 0.0


def score__press_releases__a33(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.strip()
    words = re.findall(r"\b[\w'-]+\b", s)
    n = len(words)
    if n == 0:
        return 0.0
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", s) if len(re.findall(r"\w+", x)) >= 4]
    paragraphs = [p for p in re.split(r"\n\s*\n", s) if len(re.findall(r"\w+", p)) >= 20]
    long_sentence_ratio = sum(len(re.findall(r"\w+", x)) >= 8 for x in sentences) / max(1, len(sentences))
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    fragments = sum(1 for x in lines if len(re.findall(r"\w+", x)) <= 5 and not re.search(r"[.!?][\"')\]]?$", x))
    boiler = sum(s.lower().count(x) for x in ["copyright", "all rights reserved", "privacy policy", "cookie", "sign in", "navigation", "skip to", "terms of use", "read more", "http://", "https://"])
    data_lines = sum(bool(re.search(r"(?:\$|%|\b\d{2,}(?:\.\d+)?\b).*(?:\$|%|\b\d{2,}(?:\.\d+)?\b)", x)) for x in lines)
    coherence = min(4.0, len(sentences) * 0.35) + min(2.0, len(paragraphs) * 0.6) + 1.5 * long_sentence_ratio
    substance = min(2.0, n / 100.0)
    penalty = min(5.0, boiler * 0.55 + fragments / max(1, len(lines)) * 2.5 + data_lines / max(1, len(lines)) * 2.0)
    score = coherence + substance - penalty
    if n < 40:
        score = min(score, 3.0)
    return float(max(0.0, min(10.0, round(score, 1))))


def score__patents__a12(text):
    import re
    if not isinstance(text, str):
        return 0.0
    m = re.search(r"(?im)^\s*CLAIMS\s*:", text)
    if not m:
        return 0.0
    section = text[m.end():]
    if re.search(r"(?m)^\s*1(?:[.)]|\s)", section):
        return 10.0
    if re.search(r"(?m)^\s*\d+(?:[.)]|\s)", section):
        return 3.3
    return 1.1


def score__math__a66(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.strip()
    split = re.split(r"(?im)^\s*(?:answer|solution|response)\s*:?[ \t]*$", s, maxsplit=1)
    answer = split[1].strip() if len(split) == 2 else s
    lower = answer.lower()
    prompts = [
        "can you finish", "can you see", "what do you think", "what happens if",
        "can you show", "can you prove", "can you compute", "try to", "your turn",
        "what would", "how might", "which theorem", "does this suggest", "next step",
    ]
    p = sum(lower.count(x) for x in prompts)
    questions = answer.count("?")
    withholding = sum(lower.count(x) for x in ["complete the", "fill in", "left for you", "finish the", "remaining step", "without giving", "rather than giving"])
    complete = sum(lower.count(x) for x in ["final answer", "therefore the answer", "thus the answer", "we conclude that", "hence the result", "qed", "which proves"])
    ends_question = bool(re.search(r"\?\s*$", answer))
    if (p >= 1 or withholding >= 1) and questions >= 1 and complete == 0:
        return float(min(10.0, 6.0 + 0.8 * min(p, 3) + 0.7 * min(withholding, 2) + (0.8 if ends_question else 0.0)))
    if questions >= 2 and complete == 0 and ends_question:
        return 6.0
    return 0.0


def score__press_releases__a31(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.strip()
    blocks = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    def sentence_count(p):
        return len([x for x in re.split(r"(?<=[.!?])(?:\s+|$)", p) if len(re.findall(r"\b\w+\b", x)) >= 5])
    coherent = [p for p in blocks if len(re.findall(r"\b\w+\b", p)) >= 25 and sentence_count(p) >= 2]
    boiler_terms = ["copyright", "privacy policy", "terms of use", "contact us", "sign in", "navigation", "skip to content"]
    if coherent:
        narrative_words = sum(len(re.findall(r"\b\w+\b", p)) for p in coherent)
        boiler = sum(s.lower().count(x) for x in boiler_terms)
        return 10.0 if narrative_words >= 35 and boiler < 4 else 8.0
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", s) if len(re.findall(r"\b\w+\b", x)) >= 5]
    readable_words = sum(len(re.findall(r"\b\w+\b", x)) for x in sentences)
    if len(sentences) >= 2:
        return float(min(5.0, 2.0 + readable_words / 25.0))
    if len(sentences) == 1:
        return float(min(3.5, 1.0 + readable_words / 15.0))
    return 0.0


def score__press_releases__a103(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    words = re.findall(r"\b[a-z][a-z-]*\b", s)
    n = len(words)
    finance = [
        "revenue", "net income", "earnings", "eps", "profit", "loss", "cash flow",
        "ebitda", "margin", "dividend", "shareholder", "stock", "shares", "portfolio",
        "investment", "investor", "interest rate", "basis points", "market capitalization",
        "assets", "liabilities", "equity", "bond", "loan", "credit", "banking", "fiscal",
        "quarter", "financial results", "economic growth", "inflation", "gdp", "expense",
    ]
    business = [
        "company", "corporate", "business", "operations", "employee", "workforce", "human resources",
        "chief executive", "ceo", "acquisition", "merger", "partnership", "customer", "sales",
        "strategy", "management", "board of directors", "appoint", "market", "industry",
    ]
    f = sum(s.count(x) for x in finance)
    metrics = len(re.findall(r"(?:\$|€|£)\s*\d|\b\d+(?:\.\d+)?\s*(?:%|million|billion|trillion|basis points)\b", s))
    b = sum(s.count(x) for x in business)
    density = (f + metrics) * 100.0 / max(30, n)
    if f + metrics >= 8 and density >= 2.0:
        return float(min(10.0, 8.0 + 0.12 * min(f + metrics - 8, 15)))
    if f + metrics >= 4 and density >= 1.0:
        return float(min(8.5, 6.5 + 0.15 * (f + metrics)))
    if b >= 4:
        return float(min(7.5, 5.0 + 0.18 * min(b, 12) + 0.1 * min(f, 5)))
    if f + metrics >= 2:
        return float(min(5.5, 3.5 + 0.4 * (f + metrics)))
    return float(min(4.0, 0.4 * b))


def score__math__a144(text):
    import re
    if not isinstance(text, str):
        return 0.0
    marker = text.rfind("[...]")
    if marker < 0:
        return 0.0
    tail = text[marker + 5:].strip()
    words = len(re.findall(r"\b\w+\b", tail))
    paragraphs = len([p for p in re.split(r"\n\s*\n", tail) if re.search(r"\w", p)])
    sentences = len([x for x in re.split(r"(?<=[.!?])\s+", tail) if len(re.findall(r"\w+", x)) >= 3])
    equations = len(re.findall(r"[=∑∫√]|\\(?:frac|sum|int|begin|align|sqrt)", tail))
    if words <= 20 and sentences <= 1 and equations <= 1:
        return 0.7
    if words < 90 and paragraphs <= 1 and equations < 4:
        return 3.4
    if words >= 350 or paragraphs >= 6 or equations >= 15:
        return float(min(10.0, 8.0 + min(2.0, words / 500.0 + paragraphs * 0.12 + equations * 0.04)))
    if words >= 120 or paragraphs >= 3 or equations >= 6:
        return float(min(7.0, 6.0 + min(1.0, words / 350.0 + paragraphs * 0.08 + equations * 0.025)))
    return 3.4


def score__code_review__a324(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.strip()
    chunks = [x.strip() for x in re.split(r"\n\s*\n|(?im)^(?:comment|reviewer|author|reply)\s*(?:by\s+\w+)?\s*:", s) if x.strip()]
    if len(chunks) == 1:
        chunks = [x.strip() for x in s.splitlines() if x.strip()]
    tech = [
        "function", "method", "class", "variable", "api", "database", "query", "test",
        "exception", "error", "performance", "memory", "thread", "security", "architecture",
        "interface", "dependency", "refactor", "algorithm", "complexity", "cache", "validation",
        "type", "return", "parameter", "implementation", "race condition", "null",
    ]
    superficial_pattern = re.compile(r"(?i)^\s*(?:done|fixed|lgtm|looks good|thanks|thank you|agreed|ok(?:ay)?|nice|nit:?|\+1)[.!\s]*$")
    substantive = 0
    superficial = 0
    for c in chunks:
        lower = c.lower()
        wc = len(re.findall(r"\w+", c))
        detail = sum(x in lower for x in tech)
        action = sum(x in lower for x in ["because", "consider", "suggest", "instead", "should", "could", "please", "what if", "why", "recommend"])
        if wc >= 12 and detail >= 1 and action >= 1:
            substantive += 1
        elif superficial_pattern.match(c) or wc <= 5:
            superficial += 1
    total = max(1, len(chunks))
    ratio = substantive / total
    if ratio >= 0.7 and substantive >= 2:
        return float(min(10.0, 7.0 + 3.0 * ratio))
    if ratio >= 0.25 or substantive >= 1:
        return float(min(6.8, 3.0 + 4.0 * ratio + 0.25 * min(substantive, 3)))
    if superficial / total >= 0.5 or len(re.findall(r"\w+", s)) < 20:
        return float(max(0.0, 2.0 - 1.5 * superficial / total))
    return 2.5


def score__press_releases__a115(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    periodic = sum(x in s for x in ["quarterly results", "annual results", "financial results", "fiscal year", "first quarter", "second quarter", "third quarter", "fourth quarter", "year ended", "quarter ended"])
    metrics_terms = ["revenue", "net income", "earnings per share", "eps", "profit", "operating income", "cash flow", "ebitda", "gross margin", "expenses"]
    metrics = sum(s.count(x) for x in metrics_terms)
    figures = len(re.findall(r"(?:\$|€|£)\s*\d|\b\d+(?:\.\d+)?\s*(?:%|million|billion)\b", s))
    investor = sum(x in s for x in ["investor relations", "shareholder", "dividend", "financial performance", "balance sheet", "securities and exchange commission"])
    announcement = sum(x in s for x in ["announces", "launches", "acquisition", "partnership", "conference", "event", "appoints"])
    boiler = sum(x in s for x in ["navigation", "page not found", "404 error", "sign in", "privacy policy"])
    if periodic and metrics >= 2 and figures >= 2:
        return float(min(10.0, 9.0 + 0.15 * min(metrics + figures - 4, 7)))
    if metrics >= 5 and (figures >= 3 or investor >= 2):
        return float(min(9.0, 8.0 + 0.1 * min(metrics + figures, 10)))
    if metrics >= 2 and (figures >= 1 or investor):
        return float(min(8.0, 5.0 + 0.3 * min(metrics + figures + investor, 10)))
    if announcement and not periodic:
        return 2.5
    if boiler or len(re.findall(r"\w+", s)) < 20:
        return 0.0
    return 0.5


def score__patents__a222(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text
    lower = s.lower()
    patent = bool(re.search(r"(?im)^\s*(?:claims?|what is claimed is)\s*:", s)) or "patent application" in lower
    if not patent:
        return 0.0
    m = re.search(r"(?im)^\s*(?:claims?|what is claimed is)\s*:\s*", s)
    claims = s[m.end():] if m else s
    independent = re.search(r"(?ms)^\s*1(?:[.)]|\s)\s*(.*?)(?=^\s*2(?:[.)]|\s)|\Z)", claims)
    if not independent:
        independent = re.search(r"(?ms)^\s*\d+(?:[.)]|\s)\s*(.*?)(?=^\s*\d+(?:[.)]|\s)|\Z)", claims)
    if not independent:
        return 0.0
    claim = independent.group(1).lower()
    excluded = [
        "chemical composition", "pharmaceutical composition", "formulation", "nucleic acid",
        "polypeptide", "protein", "molecule", "compound of formula", "business method",
        "computer-implemented method", "data processing method", "communication protocol",
        "method for manufacturing", "method of testing", "method of measuring",
    ]
    if any(x in claim for x in excluded):
        return 0.0
    device = re.search(r"\b(?:apparatus|device|connector|toy|rest|assembly|machine|system|mechanism|fixture|housing|structure)\b", claim)
    parts = len(re.findall(r"\b(?:housing|frame|member|shaft|arm|connector|fastener|surface|wall|opening|base|support|spring|gear|motor|electrode|circuit|sensor|actuator|plate|body|chamber|valve|tube|wheel|hinge)\b", claim))
    links = len(re.findall(r"\b(?:coupled|connected|attached|mounted|disposed|positioned|engaged|extending|adjacent|between|within|secured|pivotally|electrically connected)\b", claim))
    method_only = bool(re.match(r"\s*(?:a|the)\s+method\b", claim)) and not device
    return 10.0 if device and parts >= 2 and links >= 1 and not method_only else 0.0


def score__press_releases__a81(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.strip()
    lower = s.lower()
    immediate = "for immediate release" in lower
    dateline = bool(re.search(r"(?im)^\s*[A-Z][A-Z .'-]{2,30}(?:,\s*(?:[A-Z]{2}|[A-Z][a-z]+))?\s*[-—,]\s*(?:[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}\s+[A-Z][a-z]+\s+\d{4})", s))
    headline = bool(re.search(r"(?m)^\s*[A-Z][^\n.!?]{15,120}\s*$", s))
    announcement = sum(x in lower for x in ["announces", "announced today", "launches", "unveils", "appoints", "has entered into", "new partnership"])
    footer = sum(x in lower for x in ["newsroom", "view all press releases", "investor relations", "media contact", "about ", "forward-looking statements"])
    body = len(re.findall(r"\b\w+\b", s)) >= 100 and len(re.findall(r"[.!?](?:\s|$)", s)) >= 4
    negative = sum(x in lower for x in ["add to cart", "product specifications", "video transcript", "legal memorandum", "login", "sign in"])
    elements = immediate + dateline + headline + bool(announcement) + bool(footer) + body
    if dateline and announcement and body and elements >= 4:
        return float(min(10.0, 8.0 + 0.4 * (elements - 4)))
    if announcement and body:
        return 5.0 if not dateline else 7.0
    if negative or not dateline:
        return float(max(0.0, min(2.0, 0.5 * elements - negative)))
    return 3.0


def score__press_releases__a79(text):
    import re
    if not isinstance(text, str):
        return 0.0
    s = text
    standard = re.search(r"(?im)^\s*[A-Z][A-Z .'-]{2,35},\s*(?:[A-Z]{2}|[A-Z][a-z]+),?\s+(?:[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\s*(?:/PRNewswire/)?\s*[-—]+", s)
    wire = re.search(r"(?im)^\s*[A-Z][A-Z .'-]{2,35},.*?/PRNewswire/\s*[-—]+", s)
    if standard or wire:
        return 10.0
    immediate = re.search(r"(?is)FOR IMMEDIATE RELEASE\s*(?:\r?\n|[-—:]\s*){1,3}\s*(?:[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}\s+[A-Z][a-z]+\s+\d{4})", s)
    if immediate:
        return 7.5
    partial = re.search(r"(?im)^\s*(?:[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\s*$", s)
    city_dash = re.search(r"(?m)^\s*[A-Z][A-Z .'-]{2,35}\s*[-—]\s+", s)
    return 6.7 if partial or city_dash or "FOR IMMEDIATE RELEASE" in s.upper() else 0.0


def score__CAL__CAL2(text):
    return 10.0 if isinstance(text, str) and "?" in text else 0.0


def score__press_releases__a2(text):
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.strip()
    lower = s.lower()
    dateline = bool(re.search(r"(?im)^\s*[A-Z][A-Z .'-]{2,35}(?:,\s*(?:[A-Z]{2}|[A-Z][a-z]+))?\s*[-—,]\s*(?:[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}\s+[A-Z][a-z]+\s+\d{4})", s))
    headline = bool(re.search(r"(?m)^\s*[A-Z][^\n.!?]{15,120}\s*$", s))
    lead = dateline and len(re.findall(r"[.!?](?:\s|$)", s)) >= 2
    quotes = len(re.findall(r"[\"“][^\"”]{25,}[\"”]", s)) >= 1 or bool(re.search(r"\b(?:said|stated|commented|added)\b", lower))
    footer = sum(x in lower for x in ["media contact", "press contact", "about ", "investor relations", "for more information", "forward-looking statements"])
    announcement = sum(x in lower for x in ["announces", "announced today", "launches", "unveils", "appoints", "acquisition", "partnership"])
    dilution = sum(x in lower for x in ["sign in", "log in", "navigation", "cookie settings", "skip to content", "stock quote", "frequently asked questions", "page not found", "error 404"])
    elements = dateline + headline + lead + quotes + bool(footer) + bool(announcement)
    if dateline and lead and announcement and elements >= 4:
        return float(min(10.0, 8.0 + 0.45 * (elements - 4) - 0.25 * min(dilution, 2)))
    if announcement and len(re.findall(r"\w+", s)) >= 70:
        return float(max(4.0, min(7.0, 4.0 + 0.65 * elements - 0.4 * dilution)))
    if dilution >= 1 or len(re.findall(r"\w+", s)) < 50:
        return float(max(0.0, min(3.0, 0.6 * elements - 0.5 * dilution)))
    return float(min(3.0, 0.5 * elements))


JOB_IDS = [
    "patents__a24",
    "math__a150",
    "math__a204",
    "code_review__a27",
    "press_releases__a233",
    "math__a0",
    "patents__a102",
    "CAL__CAL3",
    "press_releases__a33",
    "patents__a12",
    "math__a66",
    "press_releases__a31",
    "press_releases__a103",
    "math__a144",
    "code_review__a324",
    "press_releases__a115",
    "patents__a222",
    "press_releases__a81",
    "press_releases__a79",
    "CAL__CAL2",
    "press_releases__a2",
]
