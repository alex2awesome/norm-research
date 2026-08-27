# AUTO: blind rule compilation chunk c1
import re
import math
import string
import collections

_WORD_RE = re.compile(r"[A-Za-z']+")


def _words(text):
    if not text:
        return []
    return _WORD_RE.findall(text)


def _sentences(text):
    if not text:
        return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def _count_any(t_lower, keywords):
    return sum(t_lower.count(k) for k in keywords)


def _clip(x, lo=0.0, hi=10.0):
    return max(lo, min(hi, x))


def score__patents__a24(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    ui_kw = [
        "user interface", "graphical user interface", "gui", "touchscreen",
        "touch screen", "input device", "user input", "cursor", "click",
        "icon", "menu", "display screen", "user-controllable", "interactive",
        "webpage", "browser", "keyboard", "mouse", "gesture", "user experience",
        "user-selected", "user selects", "digital content", "content manipulation",
        "human-computer", "human computer", "user-interactive"
    ]
    digital_kw = [
        "processor", "circuit", "electronic", "signal processing",
        "computing device", "memory", "software", "algorithm", "network",
        "wireless", "data processing", "computer-implemented", "microprocessor"
    ]
    mech_kw = [
        "mechanical", "gear", "shaft", "valve", "engine", "manufacturing process",
        "material composition", "chemical compound", "biological", "protein",
        "alloy", "communication protocol", "physical material", "molded",
        "injection mold"
    ]
    ui_c = _count_any(t, ui_kw)
    dig_c = _count_any(t, digital_kw)
    mech_c = _count_any(t, mech_kw)
    total = ui_c + dig_c + mech_c
    if total == 0:
        return 4.0
    score = 4.0 + 6.0 * (ui_c / total) - 4.0 * (mech_c / total)
    return round(_clip(score), 2)


def score__math__a150(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    advanced_kw = [
        "topology", "manifold", "homomorphism", "isomorphism", "functor",
        "category theory", "measure theory", "lebesgue", "sigma-algebra",
        "banach space", "hilbert space", "abstract algebra", "group theory",
        "ring theory", "field extension", "galois", "real analysis",
        "complex analysis", "differential geometry", "algebraic geometry",
        "generalization", "generalize", "conjecture", "counterexample",
        "structure of", "abstractly", "functional analysis", "spectral theory",
        "representation theory"
    ]
    basic_kw = [
        "homework", "how do i calculate", "simplify", "solve for x",
        "what is the derivative of", "compute the", "evaluate the integral",
        "basic question", "simple question", "word problem",
        "how do you add", "how do you multiply", "times table"
    ]
    proof_kw = [
        "prove", "proof", "show that", "theorem", "lemma", "generalize",
        "generalization", "structure", "abstract"
    ]
    adv_c = _count_any(t, advanced_kw)
    basic_c = _count_any(t, basic_kw)
    proof_c = _count_any(t, proof_kw)
    n = len(_words(text))
    if adv_c > 0:
        score = 8.0 + min(2.0, 0.5 * adv_c)
    elif basic_c > 0 and proof_c == 0:
        score = max(0.0, 2.0 - 0.3 * (basic_c - 1))
    elif proof_c > 0:
        score = 5.0 + min(2.0, 0.5 * proof_c)
    else:
        if n < 15:
            score = 2.0
        elif n < 60:
            score = 5.0
        else:
            score = 6.5
    return round(_clip(score), 2)


def score__math__a204(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    n = len(_words(text))
    sentences = _sentences(text)
    rigor_kw = [
        "therefore", "thus", "hence", "qed", "proof", "we conclude",
        "it follows that", "by definition", "by induction", "claim",
        "lemma", "theorem"
    ]
    rigor_c = _count_any(t, rigor_kw)
    equation_c = len(re.findall(r"[=]", text)) + len(re.findall(r"\\[a-zA-Z]+", text))
    weak_kw = ["not sure", "i think", "maybe", "todo", "incomplete", "hint:", "try to"]
    weak_c = _count_any(t, weak_kw)
    length_score = min(n / 150.0, 1.0) * 6.0
    rigor_score = min(rigor_c * 0.8, 3.0)
    eq_score = min(equation_c * 0.3, 1.5)
    score = 1.0 + length_score + rigor_score + eq_score - min(weak_c * 1.0, 3.0)
    return round(_clip(score), 2)


def score__code_review__a27(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    tech_kw = [
        "architecture", "trade-off", "tradeoff", "design", "alternative",
        "consider", "because", "however", "approach", "refactor",
        "complexity", "performance", "scalability", "interface",
        "implementation", "edge case", "consistency", "pattern"
    ]
    directive_kw = [
        "done", "fixed", "wrong copyright", "remove this", "lgtm",
        "nit:", "typo", "thanks"
    ]
    tech_c = _count_any(t, tech_kw)
    directive_c = _count_any(t, directive_kw)
    q_c = text.count('?')
    n = len(_words(text))
    turns = max(1, len(re.split(r'\n\s*\n|\n-{2,}|\n>{1}', text)))
    score = (1.0 + min(tech_c * 0.6, 5.0) + min(q_c * 0.4, 2.0)
             + min(turns * 0.15, 1.5) + min(n / 300.0, 1.0)
             - min(directive_c * 0.5, 3.0))
    return round(_clip(score), 2)


def score__press_releases__a233(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    wire_kw = [
        "news provided by", "share this article", "pr newswire", "prnewswire",
        "cision", "businesswire", "business wire", "globe newswire",
        "globenewswire"
    ]
    fir_kw = ["for immediate release"]
    dateline = re.search(
        r'\b[A-Z][A-Za-z\.]+(,\s*[A-Z]{2})?,?\s+'
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}',
        text
    )
    hub_kw = ["newsroom", "press releases", "view all press releases", "media center"]
    nav_kw = ["home", "about us", "contact us", "sign in", "subscribe"]
    wire_c = _count_any(t, wire_kw)
    fir_c = _count_any(t, fir_kw)
    hub_c = _count_any(t, hub_kw)
    if wire_c > 0 and dateline:
        return 10.0
    if fir_c > 0 or dateline:
        return 7.5
    if hub_c > 0:
        return 2.5
    nav_c = _count_any(t, nav_kw)
    if nav_c > 0 and len(_words(text)) < 60:
        return 0.0
    return 2.5 if (wire_c + fir_c + hub_c) > 0 else 0.0


def score__math__a0(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    error_kw = [
        "incorrect", "this is wrong", "mistake", "error in",
        "that's not right", "not correct", "doesn't work", "fails to"
    ]
    resolve_kw = [
        "therefore", "thus", "hence", "the answer is", "we conclude",
        "qed", "final answer", "in conclusion"
    ]
    hedge_kw = ["not sure", "i think", "maybe", "might be", "i believe", "possibly"]
    n = len(_words(text))
    error_c = _count_any(t, error_kw)
    resolve_c = _count_any(t, resolve_kw)
    hedge_c = _count_any(t, hedge_kw)
    equation_c = text.count('=')
    if error_c > 0:
        score = max(0.0, 2.5 - error_c * 0.5)
    else:
        score = (3.0 + min(resolve_c * 1.2, 3.0) + min(equation_c * 0.15, 2.0)
                  + min(n / 200.0, 2.0) - min(hedge_c * 0.8, 2.0))
    return round(_clip(score), 2)


def score__patents__a102(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    specific_kw = [
        "escherichia coli", "e. coli", "staphylococcus", "bacteria", "bacterium",
        "protein", "enzyme", "receptor", "antibody", "polymer", "polypeptide",
        "nucleotide", "dna", "rna", "hydroxyl", "methyl", "amine", "carboxyl",
        "catalyst", "monomer", "copolymer", "peptide", "antigen", "virus",
        "cell membrane", "glucose", "amino acid"
    ]
    general_kw = [
        "chemical", "biological", "organic", "compound", "molecule",
        "composition", "reaction", "formulation", "biocompatible", "pharmaceutical"
    ]
    specific_c = _count_any(t, specific_kw)
    general_c = _count_any(t, general_kw)
    if specific_c > 0:
        return 10.0
    if general_c > 0:
        return 5.0
    return 0.0


def score__CAL__CAL3(text):
    if not text or not text.strip():
        return 0.0
    n = len(text.split())
    if n > 150:
        return 10.0
    elif n >= 50:
        return 5.0
    else:
        return 0.0


def score__press_releases__a33(text):
    if not text or not text.strip():
        return 0.0
    boilerplate_kw = [
        "all rights reserved", "copyright ©", "privacy policy", "terms of use",
        "subscribe", "sign in", "cookie policy", "navigation", "skip to content",
        "home | about"
    ]
    t = text.lower()
    boiler_c = _count_any(t, boilerplate_kw)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    n_lines = len(lines)
    short_lines = sum(1 for l in lines if len(l.split()) <= 4)
    frag_ratio = short_lines / n_lines if n_lines else 1.0
    sentences = _sentences(text)
    ns = len(sentences)
    n = len(_words(text))
    avg_sent_len = n / ns if ns else 0
    score = 10.0
    score -= min(frag_ratio * 8.0, 6.0)
    score -= min(boiler_c * 1.5, 4.0)
    if avg_sent_len < 4 or avg_sent_len > 45:
        score -= 2.0
    if n < 30:
        score -= 3.0
    return round(_clip(score), 2)


def score__patents__a12(text):
    if not text or not text.strip():
        return 0.0
    m = re.search(r'CLAIMS\s*:', text, re.IGNORECASE)
    if not m:
        return 0.0
    after = text[m.end():]
    m1 = re.search(r'(?:^|\n)\s*1\s*[\.\)]', after)
    if m1:
        return 10.0
    mnum = re.search(r'(?:^|\n)\s*\d+\s*[\.\)]', after)
    if mnum:
        return 3.3
    return 1.1


def score__math__a66(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    socratic_kw = [
        "can you finish", "can you see", "can you show", "i'll leave",
        "left as an exercise", "try to finish", "what do you get",
        "can you complete", "see if you can", "can you continue"
    ]
    direct_kw = [
        "the answer is", "therefore the", "final answer", "in conclusion",
        "thus we have", "so the result is"
    ]
    soc_c = _count_any(t, socratic_kw)
    direct_c = _count_any(t, direct_kw)
    if soc_c > 0:
        score = 6.0 + min(4.0, soc_c * 1.5)
        if direct_c > 0:
            score -= min(direct_c * 1.0, 3.0)
        return round(_clip(score), 2)
    return 0.0


def score__press_releases__a31(text):
    if not text or not text.strip():
        return 0.0
    sentences = _sentences(text)
    long_sentences = [s for s in sentences if len(_words(s)) >= 6]
    if not long_sentences:
        return 0.0
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    best_para_sentcount = 0
    for p in paragraphs:
        ps = [s for s in re.split(r'(?<=[.!?])\s+', p) if len(_words(s)) >= 5]
        best_para_sentcount = max(best_para_sentcount, len(ps))
    if best_para_sentcount >= 3:
        return 10.0
    total_words_in_long = sum(len(_words(s)) for s in long_sentences)
    score = min(5.0, 1.0 + total_words_in_long / 30.0)
    return round(_clip(score), 2)


def score__press_releases__a103(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    fin_kw = [
        "revenue", "earnings", "stock", "stocks", "shares", "portfolio",
        "investment", "interest rate", "interest rates", "dividend", "equity",
        "market cap", "ebitda", "profit margin", "net income", "balance sheet",
        "cash flow", "ipo", "hedge fund", "bond", "bonds", "asset", "assets",
        "liability", "liabilities", "gdp", "inflation", "valuation",
        "acquisition price", "merger", "nasdaq", "nyse"
    ]
    biz_kw = [
        "business", "corporate", "company", "operations", "management", "hr",
        "human resources", "employees", "partnership", "announcement"
    ]
    fin_c = _count_any(t, fin_kw)
    biz_c = _count_any(t, biz_kw)
    n = len(_words(text)) or 1
    fin_density = fin_c / (n / 100.0)
    if fin_density > 0.5:
        score = 8.0 + min(2.0, fin_density - 0.5)
    elif biz_c > 0 or fin_c > 0:
        score = 5.0 + min(2.0, fin_density * 2 + biz_c * 0.1)
    else:
        score = 2.0
    return round(_clip(score), 2)


def score__math__a144(text):
    if not text:
        return 0.0
    idx = text.find('[...]')
    if idx == -1:
        return 0.0
    after = text[idx + 5:]
    n = len(_words(after))
    if n <= 6:
        return 0.7
    elif n <= 40:
        return 3.4
    elif n <= 150:
        return 6.5
    else:
        return 9.0


def score__code_review__a324(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    substantive_kw = [
        "suggest", "consider", "because", "why not", "instead", "alternative",
        "architecture", "design", "approach", "refactor", "edge case",
        "performance", "should we", "could we", "what about",
        "i think we should"
    ]
    superficial_kw = [
        "done", "fixed", "lgtm", "+1", "nice", "thanks", "ok", "sounds good",
        "agreed", "nit:"
    ]
    sub_c = _count_any(t, substantive_kw)
    sup_c = _count_any(t, superficial_kw)
    total = sub_c + sup_c
    if total == 0:
        n = len(_words(text))
        return 3.0 if n > 40 else 1.0
    ratio = sub_c / total
    return round(_clip(ratio * 10.0), 2)


def score__press_releases__a115(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    earnings_kw = [
        "quarterly results", "annual results", "fiscal year", "q1 20", "q2 20",
        "q3 20", "q4 20", "earnings release", "earnings report", "net income",
        "eps", "earnings per share", "revenue of", "reported revenue",
        "financial results"
    ]
    fin_perf_kw = ["financial performance", "investor", "expense", "operating margin", "gross margin"]
    generic_pr_kw = [
        "announced today", "launch of", "new product", "acquisition of",
        "partnership with", "press release"
    ]
    nav_kw = ["page not found", "404", "navigation", "sign in", "home page"]
    earnings_c = _count_any(t, earnings_kw)
    finperf_c = _count_any(t, fin_perf_kw)
    generic_c = _count_any(t, generic_pr_kw)
    nav_c = _count_any(t, nav_kw)
    n = len(_words(text))
    if nav_c > 0 and n < 80:
        return 0.0
    if earnings_c > 0:
        return round(min(10.0, 9.0 + min(1.0, earnings_c * 0.2)), 2)
    if finperf_c > 0:
        return round(min(9.0, 8.0 + min(1.0, finperf_c * 0.3)), 2)
    if generic_c > 0:
        return round(min(3.0, 2.0 + min(1.0, generic_c * 0.2)), 2)
    return 0.0


def score__patents__a222(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    software_kw = [
        "computer-implemented", "software", "algorithm", "business method",
        "data processing", "computer program", "communication protocol",
        "protocol for", "instructions stored"
    ]
    chem_kw = [
        "composition comprising", "chemical compound", "polymer", "formulation",
        "pharmaceutical composition", "biological", "protein", "molecule"
    ]
    struct_kw = [
        "apparatus comprising", "device comprising", "a housing", "a shaft",
        "a connector", "coupled to", "attached to", "affixed to", "a frame",
        "a bracket", "pivotally", "rotatably", "a body portion", "structural",
        "interconnection", "mechanically coupled"
    ]
    struct_c = _count_any(t, struct_kw)
    sw_c = _count_any(t, software_kw)
    chem_c = _count_any(t, chem_kw)
    if struct_c > 0 and sw_c == 0 and chem_c == 0:
        return 10.0
    return 0.0


def score__press_releases__a81(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    fir = bool(re.search(r'for immediate release', t))
    dateline = bool(re.search(r'\b[A-Z][A-Za-z\.]+,\s*[A-Z]{2,}\b[,\s\-–—]', text))
    boiler_kw = ["newsroom", "view all press releases", "investor relations"]
    boiler_c = _count_any(t, boiler_kw)
    nav_kw = ["add to cart", "buy now", "product specifications", "sign in", "subscribe now", "transcript"]
    nav_c = _count_any(t, nav_kw)
    n = len(_words(text))
    if fir and dateline and boiler_c > 0:
        return 10.0
    if fir and dateline:
        return 8.5
    if nav_c > 0 and not fir and not dateline:
        return 0.5
    if (fir or dateline) and n > 40:
        return 4.5
    return 1.0


def score__press_releases__a79(text):
    if not text or not text.strip():
        return 0.0
    wire_dateline = re.search(
        r'\b[A-Z][A-Za-z\. ]{1,25},\s*[A-Za-z][A-Za-z\.]*,?\s*'
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}'
        r'\s*/?\s*(PRNewswire|PR Newswire|Business Wire|BusinessWire)?/?\s*--',
        text
    )
    if wire_dateline:
        return 10.0
    if re.search(r'for immediate release', text, re.IGNORECASE) and re.search(r'\d{4}', text):
        return 7.5
    partial_date = re.search(
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}', text
    )
    partial_city = re.search(r'\b[A-Z][A-Za-z]+\s*[\-–—]', text[:200])
    if partial_date or partial_city:
        return 6.7
    return 0.0


def score__CAL__CAL2(text):
    if not text:
        return 0.0
    return 10.0 if '?' in text else 0.0


def score__press_releases__a2(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    dateline = bool(re.search(r'\b[A-Z][A-Za-z\.]+,\s*[A-Z]{2,}\b', text[:300]))
    first_line = text.strip().split("\n")[0]
    quote = bool(re.search(r'["“][^"”]{20,}["”]', text))
    boiler_kw = ["contact:", "media contact", "about ", "for more information", "©", "all rights reserved"]
    boiler_c = _count_any(t, boiler_kw)
    nav_kw = ["sign in", "log in", "password", "stock quote", "frequently asked questions", "faq", "page not found"]
    nav_c = _count_any(t, nav_kw)
    n = len(_words(text))
    if nav_c > 0 and n < 100:
        return round(max(0.0, 3.0 - nav_c), 2)
    signals = sum([dateline, quote, boiler_c > 0])
    if signals >= 2 and n > 80:
        return round(min(10.0, 8.0 + signals * 0.5), 2)
    if n > 50:
        return round(min(7.0, 4.0 + signals * 1.0), 2)
    return round(max(0.0, 2.0 - nav_c * 0.5), 2)


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
