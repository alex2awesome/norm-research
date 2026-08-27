# AUTO: blind rule compilation chunk c1
import re


def score__patents__a24(text):
    t = (text or "").lower()
    if not t.strip():
        return 0.0
    interactive = ["user interface", "graphical interface", "touchscreen", "touch screen", "user input", "human-computer", "human computer", "interactive", "display screen", "cursor", "keyboard", "mouse", "gesture", "user-select", "user control", "user-controllable", "digital content", "menu", "widget", "application interface"]
    computing = ["computer", "processor", "software", "digital", "electronic", "memory", "data processing", "server", "network", "algorithm", "controller", "circuit"]
    excluded = ["mechanical", "manufacturing", "alloy", "polymer", "chemical", "compound", "pharmaceutical", "protein", "bacteria", "composition", "protocol", "fabrication"]
    ih = sum(t.count(x) for x in interactive)
    ch = sum(t.count(x) for x in computing)
    eh = sum(t.count(x) for x in excluded)
    if ih >= 5 or (ih >= 2 and ih * 2 >= eh):
        return min(10.0, 8.0 + min(2.0, 0.25 * ih))
    if ih >= 1:
        return min(7.9, 5.5 + 0.5 * ih + min(1.0, 0.1 * ch))
    if ch >= 3 and ch >= eh:
        return min(7.0, 4.0 + 0.25 * ch)
    if ch:
        return min(4.8, 3.2 + 0.2 * ch)
    return min(3.9, 0.5 + 0.25 * eh)


def score__math__a150(text):
    t = text or ""
    parts = re.split(r"(?im)^\s*(?:answer|solution|assistant|a)\s*:\s*", t, maxsplit=1)
    q = parts[0].lower()
    if not q.strip():
        return 0.0
    advanced = ["topology", "manifold", "homology", "cohomology", "measure theory", "functional analysis", "banach", "hilbert", "galois", "scheme", "algebraic geometry", "category theory", "lie algebra", "distribution", "stochastic process", "spectral sequence", "fundamental group", "operator algebra", "riemannian", "noetherian"]
    theoretical = ["prove", "proof", "theorem", "lemma", "conjecture", "necessary and sufficient", "generalize", "counterexample", "isomorphism", "convergence", "continuity", "compact", "group", "ring", "field", "eigenvalue", "integral", "derivative", "sequence", "rigorous"]
    routine = ["calculate", "compute", "simplify", "evaluate", "solve for", "what is", "find the value", "homework"]
    ah = sum(q.count(x) for x in advanced)
    th = sum(q.count(x) for x in theoretical)
    rh = sum(q.count(x) for x in routine)
    if ah:
        return min(10.0, 8.0 + 0.45 * min(ah, 3) + 0.15 * min(th, 4))
    if th >= 2 or (th and len(re.findall(r"\b\w+\b", q)) >= 45):
        return min(7.5, 5.0 + 0.45 * min(th, 5))
    if th:
        return 5.0
    if rh or re.search(r"\d\s*[-+*/^=]", q):
        return max(0.5, 2.0 - 0.35 * min(rh, 4))
    return 3.0


def score__math__a204(text):
    t = text or ""
    parts = re.split(r"(?im)^\s*(?:answer|solution|assistant|a)\s*:\s*", t, maxsplit=1)
    a = (parts[1] if len(parts) > 1 else t).strip()
    words = re.findall(r"\b\w+\b", a)
    if not words:
        return 0.0
    math_marks = len(re.findall(r"[=<>+*/^]|\\(?:frac|sum|int|lim|sqrt)|\b(?:theorem|proof|therefore|hence|suppose|let|since|implies|equation|lemma)\b", a, re.I))
    steps = len(re.findall(r"(?im)^\s*(?:\d+[.)]|step\s+\d+|[-*])\s+", a))
    conclusions = len(re.findall(r"\b(?:therefore|thus|hence|consequently|we conclude|this proves|qed|final answer)\b", a, re.I))
    if len(words) < 8:
        return min(1.5, 0.2 + 0.15 * len(words) + 0.1 * math_marks)
    score = 2.0 + min(3.2, len(words) / 45.0) + min(2.0, math_marks / 4.0) + min(1.2, steps * 0.3) + min(1.2, conclusions * 0.6)
    if len(words) < 35:
        score = min(score, 5.8)
    elif len(words) < 90:
        score = min(score, 8.0)
    return round(min(10.0, score), 1)


def score__code_review__a27(text):
    t = text or ""
    if not t.strip():
        return 0.0
    technical = re.findall(r"\b(?:architecture|design|trade-?off|alternative|implementation|interface|api|database|schema|cache|thread|race|performance|complexity|dependency|abstraction|refactor|security|transaction|algorithm|invariant|test|error handling)\b", t, re.I)
    dialogue = len(re.findall(r"(?im)^\s*(?:reviewer|author|developer|maintainer|reply|response|commenter)\s*[:\-]", t))
    questions = t.count("?")
    turns = max(dialogue, len(re.findall(r"(?im)^\s*(?:>|@\w+)", t)))
    superficial = len(re.findall(r"(?im)^\s*(?:done|fixed|lgtm|thanks|remove this|nit:?|wrong copyright|looks good)[.!]?\s*$", t))
    if turns >= 4 and len(technical) >= 6 and questions >= 2:
        return min(10.0, 7.0 + 0.35 * min(turns - 3, 5) + 0.2 * min(len(technical) - 5, 7))
    if len(technical) >= 2 or questions >= 2:
        return min(6.5, 3.0 + 0.3 * min(len(technical), 8) + 0.2 * min(questions, 4))
    return max(0.0, min(2.5, 1.5 + 0.15 * len(technical) - 0.3 * superficial))


def score__press_releases__a233(text):
    t = text or ""
    low = t.lower()
    if not low.strip():
        return 0.0
    wire = ["news provided by", "share this article", "pr newswire", "prnewswire", "cision", "/business wire/", "globenewswire"]
    formal = ["for immediate release", "press release", "news release", "official newsroom"]
    event = ["announces", "announced", "launches", "introduced", "unveils", "appoints", "acquires", "reports", "today announced"]
    wh = sum(x in low for x in wire)
    fh = sum(x in low for x in formal)
    eh = sum(low.count(x) for x in event)
    dateline = bool(re.search(r"(?m)^\s*[A-Z][A-Z .'-]{2,30},\s+(?:[A-Z]{2},?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+20\d{2}", t))
    if wh and eh:
        return min(10.0, 8.5 + 0.5 * wh + 0.25 * min(eh, 2))
    if (fh or dateline) and eh:
        return 7.5
    if eh and len(re.findall(r"\b\w+\b", t)) > 100:
        return 4.0 if fh else 2.5
    if any(x in low for x in ["newsroom", "latest news", "all news"]):
        return 2.5
    return 0.0


def score__math__a0(text):
    t = text or ""
    parts = re.split(r"(?im)^\s*(?:answer|solution|assistant|a)\s*:\s*", t, maxsplit=1)
    a = (parts[1] if len(parts) > 1 else t).strip()
    words = re.findall(r"\b\w+\b", a)
    if not words:
        return 0.0
    uncertainty = len(re.findall(r"\b(?:maybe|perhaps|not sure|i think|guess|approximately)\b", a, re.I))
    explicit_bad = len(re.findall(r"\b(?:cannot solve|don't know|do not know|no idea|unrelated|insufficient information)\b", a, re.I))
    derivation = len(re.findall(r"[=<>+*/^]|\\(?:frac|sum|int|lim|sqrt)|\b(?:because|since|therefore|thus|hence|proof|suppose|let)\b", a, re.I))
    conclusion = bool(re.search(r"\b(?:therefore|thus|hence|answer is|we obtain|we conclude|qed)\b", a, re.I))
    if explicit_bad:
        return 0.5
    if len(words) < 5 and derivation == 0:
        return 1.5
    score = 3.0 + min(3.5, derivation * 0.45) + min(1.5, len(words) / 80.0) + (1.2 if conclusion else 0.0) - 0.7 * uncertainty
    if derivation == 0:
        score = min(score, 4.5)
    if len(words) < 25:
        score = min(score, 7.8)
    return round(max(0.0, min(10.0, score)), 1)


def score__patents__a102(text):
    t = (text or "").lower()
    if not t.strip():
        return 0.0
    specific = [r"\b(?:[a-z]+(?:ane|ene|yne|ol|one|acid|oxide|chloride|sulfate|phosphate))\b", r"\b(?:dna|rna|mrna|enzyme|receptor|protein|antibody|antigen|cytokine|peptide|glucose|ethanol|sodium|potassium|calcium|carbon|oxygen|hydrogen|nitrogen)\b", r"\b(?:e\.\s*coli|staphylococcus|streptococcus|bacillus|salmonella|candida|influenza|cancer|diabetes|alzheimer'?s|parkinson'?s)\b", r"\b[A-Z][a-z]?\d?\b"]
    if any(re.search(p, text or "", re.I if p != specific[-1] else 0) for p in specific):
        return 10.0
    general = ["chemical", "biological", "biology", "chemistry", "organism", "cellular", "cell", "tissue", "medical", "disease", "microbe", "molecule", "organic", "inorganic", "patient", "physiological"]
    return 5.0 if any(x in t for x in general) else 0.0


def score__CAL__CAL3(text):
    n = len(re.findall(r"\b\w+\b", text or ""))
    if n > 150:
        return 10.0
    if n >= 50:
        return 5.0
    return 0.0


def score__press_releases__a33(text):
    t = text or ""
    words = re.findall(r"\b\w+\b", t)
    if not words:
        return 0.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    full = [s for s in sentences if len(re.findall(r"\b\w+\b", s)) >= 7]
    paragraphs = [p for p in re.split(r"\n\s*\n", t) if len(re.findall(r"\b\w+\b", p)) >= 25]
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    short_lines = sum(len(re.findall(r"\b\w+\b", x)) <= 4 for x in lines)
    boiler = len(re.findall(r"\b(?:copyright|all rights reserved|privacy policy|terms of use|sign in|navigation|cookie|subscribe|view all|contact us)\b|https?://|www\.", t, re.I))
    numeric = len(re.findall(r"\b\d+(?:[.,]\d+)*%?\b", t)) / max(1, len(words))
    score = 1.0 + min(4.0, len(full) * 0.55) + min(2.0, len(paragraphs) * 0.8) + min(1.5, len(words) / 130.0)
    score -= min(3.0, boiler * 0.25 + short_lines / max(1, len(lines)) * 2.0 + numeric * 8.0)
    return round(max(0.0, min(10.0, score)), 1)


def score__patents__a12(text):
    t = text or ""
    m = re.search(r"(?im)^\s*CLAIMS\s*:\s*", t)
    if not m:
        return 0.0
    claims = t[m.end():]
    if re.search(r"(?m)^\s*1(?:[.)]|\s)", claims):
        return 10.0
    if re.search(r"(?m)^\s*\d+(?:[.)]|\s)", claims):
        return 3.3
    return 1.1


def score__math__a66(text):
    t = text or ""
    if not t.strip():
        return 0.0
    invitation = len(re.findall(r"\b(?:can you|could you|try to|what happens if|do you see|can you see|how would you|now show|finish|your turn)\b", t, re.I))
    withheld = len(re.findall(r"\b(?:left to you|for you to complete|without giving|next step|remaining step|work out the rest|complete the argument)\b", t, re.I))
    final = len(re.findall(r"\b(?:final answer|therefore the answer|thus the answer|answer is|we conclude that|qed)\b", t, re.I))
    ends_question = bool(re.search(r"\?\s*$", t))
    if (invitation or withheld) and ends_question and final == 0:
        return min(10.0, 6.0 + 1.0 * min(invitation, 2) + 1.0 * min(withheld, 2))
    if invitation and final == 0:
        return 6.0
    return 0.0


def score__press_releases__a31(text):
    t = text or ""
    if not t.strip():
        return 0.0
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    coherent = 0
    fragments = 0
    for p in paras:
        words = re.findall(r"\b\w+\b", p)
        sentences = re.findall(r"(?:^|(?<=[.!?])\s+)[A-Z][^.!?]{15,}[.!?]", p)
        if len(words) >= 30 and len(sentences) >= 2:
            coherent += 1
        elif sentences or (len(words) >= 12 and re.search(r"[.!?]", p)):
            fragments += 1
    if coherent:
        return 10.0
    if fragments:
        return min(5.0, 1.5 + 1.2 * fragments)
    return 0.0


def score__press_releases__a103(text):
    t = (text or "").lower()
    words = re.findall(r"\b\w+\b", t)
    if not words:
        return 0.0
    finance = ["revenue", "net income", "earnings", "eps", "profit", "loss", "stock", "shares", "portfolio", "investment", "investor", "interest rate", "market capitalization", "cash flow", "ebitda", "dividend", "fiscal", "quarter", "assets", "liabilities", "banking", "inflation", "gdp", "economic indicator", "operating margin"]
    business = ["company", "corporate", "business", "operations", "employee", "management", "acquisition", "merger", "partnership", "chief executive", "workforce", "human resources", "customer"]
    fh = sum(t.count(x) for x in finance)
    bh = sum(t.count(x) for x in business)
    density = 100.0 * fh / max(1, len(words))
    if fh >= 8 or density >= 2.0:
        return min(10.0, 8.0 + min(2.0, fh / 8.0))
    if fh >= 3:
        return min(8.5, 5.5 + 0.4 * fh)
    if bh >= 4:
        return min(7.0, 5.0 + 0.15 * bh)
    if fh:
        return 4.0
    return min(3.5, 0.3 * bh)


def score__math__a144(text):
    t = text or ""
    pos = t.rfind("[...]")
    if pos < 0:
        return 0.0
    tail = t[pos + 5:].strip()
    words = len(re.findall(r"\b\w+\b", tail))
    paras = len([p for p in re.split(r"\n\s*\n", tail) if p.strip()])
    equations = len(re.findall(r"[=<>]|\\(?:frac|sum|int|lim|begin)|\$[^$]+\$", tail))
    sentences = len(re.findall(r"[.!?](?:\s|$)", tail))
    if words <= 20 and sentences <= 1:
        return 0.7
    if words <= 80 and paras <= 2 and sentences <= 5:
        return 3.4
    if words >= 350 or paras >= 6 or equations >= 12:
        return min(10.0, 8.0 + min(2.0, words / 500.0 + equations / 20.0))
    if words >= 120 or paras >= 3 or equations >= 5:
        return min(7.0, 6.0 + min(1.0, words / 300.0 + equations / 15.0))
    return 3.4


def score__code_review__a324(text):
    t = text or ""
    if not t.strip():
        return 0.0
    chunks = [c.strip() for c in re.split(r"\n\s*\n|(?im)^\s*(?:comment|reviewer|author|reply)\s*:\s*", t) if c.strip()]
    if not chunks:
        chunks = [t]
    substantive = 0
    superficial = 0
    tech_pattern = r"\b(?:because|consider|suggest|should|could|instead|refactor|test|handle|avoid|architecture|design|performance|security|race|api|interface|algorithm|complexity|error|exception|dependency|database|cache|thread)\b"
    for c in chunks:
        wc = len(re.findall(r"\b\w+\b", c))
        tech = len(re.findall(tech_pattern, c, re.I))
        if wc >= 18 and (tech >= 2 or "?" in c):
            substantive += 1
        elif wc <= 8 or re.fullmatch(r"(?is)\s*(?:done|fixed|lgtm|thanks|agreed|looks good|nice)[.!]?\s*", c):
            superficial += 1
    ratio = substantive / max(1, len(chunks))
    if ratio >= 0.7 and substantive:
        return min(10.0, 7.0 + 3.0 * ratio)
    if substantive:
        return min(6.5, 3.0 + 4.0 * ratio)
    return max(0.0, min(2.0, 1.5 - 0.25 * superficial + 0.01 * len(re.findall(r"\b\w+\b", t))))


def score__press_releases__a115(text):
    t = (text or "").lower()
    if not t.strip():
        return 0.0
    period = len(re.findall(r"\b(?:first|second|third|fourth|1st|2nd|3rd|4th) quarter\b|\bq[1-4]\b|\bannual (?:financial )?results\b|\bfiscal year\b|\byear ended\b", t))
    metrics = ["revenue", "net income", "earnings per share", "eps", "profit", "operating income", "cash flow", "ebitda", "gross margin", "financial results"]
    mh = sum(t.count(x) for x in metrics)
    figures = len(re.findall(r"(?:\$|€|£)\s?\d|\b\d+(?:\.\d+)?\s*(?:million|billion|%)", t))
    corporate_event = len(re.findall(r"\b(?:announces|launches|acquires|appoints|conference|event|partnership)\b", t))
    if period and mh >= 2 and figures >= 2:
        return min(10.0, 9.0 + 0.15 * min(mh + figures, 7))
    if mh >= 5 and figures >= 2:
        return min(9.0, 8.0 + 0.1 * min(mh, 10))
    if corporate_event:
        return 3.0
    if any(x in t for x in ["navigation", "page not found", "error 404", "sign in"]):
        return 0.0
    return 0.0


def score__patents__a222(text):
    t = text or ""
    low = t.lower()
    if not re.search(r"\b(?:claim|claims|what is claimed)\b", low):
        return 0.0
    claims_match = re.search(r"(?is)(?:what is claimed is|claims?\s*:)(.*)", t)
    claims = claims_match.group(1) if claims_match else t
    independents = re.split(r"(?m)^\s*\d+(?:[.)]|\s)", claims)
    independent = next((c for c in independents[1:] if c.strip() and not re.search(r"^\s*(?:the|an?)\s+\w+\s+(?:of|according to)\s+claim\s+\d+", c, re.I)), "")
    if not independent:
        return 0.0
    physical = len(re.findall(r"\b(?:apparatus|device|connector|housing|frame|member|shaft|gear|spring|fastener|surface|wall|body|chamber|electrode|motor|sensor|lever|support|mounted|coupled|connected|attached|adjacent|between|comprising)\b", independent, re.I))
    excluded = len(re.findall(r"\b(?:composition|compound|molecule|protein|pharmaceutical|polymer formulation|algorithm|software|data processing|business method|communication protocol|computer-implemented method)\b", independent, re.I))
    method_only = bool(re.match(r"\s*(?:a|the)\s+method\b", independent, re.I)) and physical < 4
    return 10.0 if physical >= 4 and physical > excluded * 2 and not method_only else 0.0


def score__press_releases__a81(text):
    t = text or ""
    low = t.lower()
    if not low.strip():
        return 0.0
    immediate = "for immediate release" in low
    dateline = bool(re.search(r"(?m)^\s*[A-Z][A-Z .'-]{2,30},\s*(?:[A-Z]{2}\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+20\d{2}", t))
    announcement = bool(re.search(r"\b(?:announces|today announced|launches|unveils|reports|appoints|acquires)\b", low))
    footer = any(x in low for x in ["newsroom", "view all press releases", "investor relations", "media contact", "about the company"])
    headline = any(len(re.findall(r"\b\w+\b", line)) >= 5 and line.strip().isupper() for line in t.splitlines()[:8])
    elements = sum([immediate, dateline, announcement, footer, headline])
    if immediate and dateline and announcement and elements >= 4:
        return min(10.0, 8.0 + 0.5 * (elements - 3))
    if announcement:
        return 5.0 if elements >= 2 else 4.0
    return min(2.0, 0.5 * elements)


def score__press_releases__a79(text):
    t = text or ""
    months = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
    full = re.search(r"(?m)^\s*[A-Z][A-Z .'-]{2,30},\s*(?:[A-Z][a-z]+|[A-Z]{2})?,?\s*" + months + r"\s+\d{1,2},\s+20\d{2}\s*(?:/PRNewswire/\s*)?(?:--|[-–—])", t)
    if full:
        return 10.0
    immediate = re.search(r"(?is)FOR IMMEDIATE RELEASE\s*(?:[:\-]?\s*)" + months + r"\s+\d{1,2},\s+20\d{2}", t)
    if immediate:
        return 7.5
    partial = re.search(r"\b" + months + r"\s+\d{1,2},\s+20\d{2}\b|(?m)^\s*[A-Z][A-Z .'-]{2,30}\s*[-–—]\s+", t)
    return 6.7 if partial else 0.0


def score__CAL__CAL2(text):
    return 10.0 if "?" in (text or "") else 0.0


def score__press_releases__a2(text):
    t = text or ""
    low = t.lower()
    if not low.strip():
        return 0.0
    dateline = bool(re.search(r"(?m)^\s*[A-Z][A-Z .'-]{2,30},\s*(?:[A-Z]{2},?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+20\d{2}", t))
    announcement = bool(re.search(r"\b(?:announces|today announced|launches|unveils|reports|appoints|acquires|introduces)\b", low))
    quote = bool(re.search(r'["“][^"”]{25,}["”]', t)) or " said" in low
    boiler = any(x in low for x in ["about ", "media contact", "press contact", "investor relations", "for more information"])
    headline = any(5 <= len(re.findall(r"\b\w+\b", line)) <= 22 for line in t.splitlines()[:6] if line.strip())
    bad = len(re.findall(r"\b(?:sign in|log in|page not found|error 404|stock quote|frequently asked questions|faq|navigation)\b", low))
    elements = sum([dateline, announcement, quote, boiler, headline])
    if dateline and announcement and elements >= 4:
        return min(10.0, 8.0 + 0.5 * (elements - 3))
    if announcement or (dateline and headline):
        return max(4.0, min(7.0, 3.0 + elements - min(2, bad)))
    return max(0.0, min(3.0, 0.5 * elements - 0.75 * bad))


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
