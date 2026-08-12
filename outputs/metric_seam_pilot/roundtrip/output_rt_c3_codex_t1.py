# AUTO: blind rule compilation chunk c3
import re


def score__math__a222(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    low = text.lower()
    math_hits = len(re.findall(
        r"(?:\$[^$]+\$|\\(?:frac|sum|int|sqrt|begin)|[=<>\u2264\u2265\u00b1\u221a]|\b(?:theorem|proof|calculate|equation|function|integer|matrix|probability|derivative|integral|counterexample)\b)",
        text, re.I))
    constructive = len(re.findall(
        r"\b(?:therefore|thus|hence|so|implies|we (?:get|have|find|show|prove|construct)|the answer is|solution|counterexample|substitut|simplif|compute|let)\b",
        low))
    steps = len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s|[=\u21d2]\s*[^\n]+", text))
    evasive = bool(re.search(r"\b(?:cannot answer|can't answer|not enough information|what do you think|it depends)\b", low))
    rhetorical_only = text.count("?") > 0 and math_hits == 0 and constructive == 0
    if (evasive or rhetorical_only) and math_hits == 0:
        return 0.0
    score = 1.0 + min(4.0, 1.15 * math_hits) + min(3.0, 1.1 * constructive) + min(2.0, 0.7 * steps)
    if len(words) < 12:
        score = min(score, 4.0)
    if math_hits and constructive and len(words) >= 25:
        score = max(score, 7.5)
    return round(max(0.0, min(10.0, score)), 1)


def score__code_review__a162(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    low = text.lower()
    reasoning = len(re.findall(r"\b(?:because|since|therefore|however|trade-?off|alternative|consider|why|how|so that|in order to|would|could|might|architecture|design|behavior|impact|risk|performance)\b", low))
    dialogue = len(re.findall(r"\b(?:i think|i agree|good point|what if|do we|could we|thanks|reply|perhaps|makes sense)\b|\?", low))
    directives = len(re.findall(r"(?:^|[.!?\n])\s*(?:please\s+)?(?:remove|add|fix|rename|format|nit|typo|change|delete|use)\b", low))
    sentences = len(re.findall(r"[.!?](?:\s|$)", text))
    depth = min(5.0, reasoning * 0.65) + min(2.0, dialogue * 0.5) + min(2.0, len(words) / 65.0) + min(1.0, sentences / 8.0)
    if directives and reasoning == 0:
        depth = min(depth, 2.5)
    if reasoning >= 4 and dialogue >= 2:
        depth = max(depth, 8.0)
    return round(max(0.0, min(10.0, depth)), 1)


def score__math__a30(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    low = text.lower()
    notation = len(re.findall(r"\$[^$]+\$|\\(?:frac|sum|int|sqrt|begin)|[=<>\u2264\u2265\u2208\u2200\u2203\u21d2]", text))
    rigor = len(re.findall(r"\b(?:proof|suppose|assume|let|definition|lemma|theorem|because|therefore|thus|hence|implies|if and only if|contradiction|case)\b", low))
    completion = len(re.findall(r"\b(?:therefore|thus|hence|which proves|as required|the answer is|we conclude|q\.?e\.?d)\b", low))
    if len(words) < 6 or (notation == 0 and rigor == 0):
        return round(min(1.0, len(words) / 8.0), 1)
    score = 1.0 + min(3.2, notation * 0.65) + min(3.2, rigor * 0.55) + min(1.8, completion * 0.9) + min(0.8, len(words) / 120.0)
    if completion == 0:
        score = min(score, 6.0 if len(words) >= 35 else 3.5)
    if notation >= 2 and rigor >= 4 and completion:
        score = max(score, 8.0)
    return round(max(0.0, min(10.0, score)), 1)


def score__math__a18(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    low = text.lower()
    formal = len(re.findall(r"\$[^$]+\$|\\(?:frac|sum|int|sqrt|begin)|[=<>\u2264\u2265\u2208\u21d2]|\b(?:theorem|lemma|proof)\b", text, re.I))
    depth = len(re.findall(r"\b(?:because|therefore|thus|hence|suppose|assume|consider|implies|contradiction|case|observe|note that|in other words|geometrically|intuitively)\b", low))
    teaching = len(re.findall(r"\b(?:why|means|interpret|intuition|for example|example|notice|key idea|concept)\b", low))
    conclusion = bool(re.search(r"\b(?:therefore|thus|hence|we conclude|which proves|the answer is|q\.?e\.?d)\b", low))
    score = 0.7 + min(3.1, formal * 0.62) + min(3.1, depth * 0.55) + min(1.5, teaching * 0.5) + min(1.0, len(words) / 110.0)
    if not conclusion:
        score = min(score, 7.0 if len(words) >= 35 else 3.5)
    if len(words) < 15:
        score = min(score, 2.0)
    if conclusion and formal >= 3 and depth >= 4 and len(words) >= 50:
        score = max(score, 9.0)
    return round(max(0.0, min(10.0, score)), 1)


def score__patents__a192(text):
    text = text or ""
    return 10.0 if re.search(r"(?im)^\s*CLAIMS?\s*:?[ \t]*$", text) else 0.0


def score__press_releases__a113(text):
    text = text or ""
    if not text.strip():
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    units = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    units = [u.strip() for u in units if u.strip()]
    if not units:
        return 0.0
    complete = 0.0
    boiler = 0.0
    for unit in units:
        words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", unit)
        is_list = bool(re.match(r"^(?:[-*\u2022]|\d+[.)])\s", unit))
        is_boiler = bool(re.search(r"\b(?:cookie|privacy policy|terms of use|all rights reserved|subscribe|sign in|menu|contact us)\b", unit, re.I))
        if len(words) >= 5 and re.search(r"[.!?][\"')\]]?$", unit) and not is_list and not is_boiler:
            complete += 1.0
        elif len(words) >= 9 and not is_list and not is_boiler:
            complete += 0.6
        if is_list or is_boiler or len(words) < 4:
            boiler += 1.0
    ratio = complete / max(1.0, len(units))
    if complete == 0:
        return 0.0
    score = 10.0 * ratio
    if boiler == 0 and ratio >= 0.9:
        score = 10.0
    return round(max(0.0, min(10.0, score)), 1)


def score__patents__a96(text):
    text = text or ""
    abstract = bool(re.search(r"(?im)^\s*ABSTRACT\s*:?[ \t]*$", text))
    match = re.search(r"(?ims)^\s*CLAIMS\s*:?[ \t]*$(.*)", text)
    numbered = bool(match and re.search(r"(?m)^\s*1[.)]\s+\S", match.group(1)))
    patentish = bool(re.search(r"\b(?:patent|invention|embodiment|what is claimed|claim)\b", text, re.I))
    return 10.0 if abstract and numbered and patentish else 0.0


def score__code_review__a252(text):
    text = text or ""
    if not text.strip():
        return 0.0
    blocks = [b for b in re.split(r"\n\s*\n|(?m)(?=^\s*(?:Comment|Review|Reviewer)\s*\d*\s*:)", text) if b.strip()]
    automated = 0.0
    for block in blocks:
        low = block.lower()
        bot = bool(re.search(r"\b(?:bot|lint|linter|automated|formatter|gofmt|prettier|format go code)\b", low))
        suggestion = bool(re.search(r"```\s*suggestion\b|\bformat(?:ting)?\s+(?:suggestion|issue|code)\b", low))
        mechanical = bool(re.search(r"\b(?:whitespace|indentation|trailing space|style check)\b", low))
        if (bot and (suggestion or mechanical)) or "format go code:" in low:
            automated += 1.0
        elif suggestion and mechanical:
            automated += 0.75
    return round(max(0.0, min(10.0, 10.0 * automated / len(blocks))), 1)


def score__patents__a36(text):
    text = text or ""
    if not text.strip():
        return 0.0
    low = text.lower()
    active = len(re.findall(r"\b(?:control(?:ler|ling|led)?|feedback|sense[ds]?|sensor|detect(?:s|ed|ing)?|processor|processing|compute|execut(?:e|es|ing)|signal|circuit|algorithm|adjust(?:s|ed|ing)?|monitor|transmit|receive|data interaction|in response to|based on)\b", low))
    steps = len(re.findall(r"\b(?:determining|receiving|generating|processing|controlling|transmitting|storing|selecting|updating)\b", low))
    static = len(re.findall(r"\b(?:composition|compound|molecule|polymer|alloy|material|layer|substrate|housing|frame|bracket|fastener|chemical|biological|protein|peptide)\b", low))
    feedback = len(re.findall(r"\b(?:feedback loop|in response to|based on (?:a |the )?(?:sensed|detected|measured)|closed-loop|dynamically adjust)\b", low))
    score = min(7.0, active * 0.7 + steps * 0.45) + min(3.0, feedback * 1.5)
    if active == 0 and static:
        return 0.0
    if active <= 2 and static > active:
        score = min(score, 4.5)
    return round(max(0.0, min(10.0, score)), 1)


def score__math__a228(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    low = text.lower()
    math = len(re.findall(r"\$[^$]+\$|\\(?:frac|sum|int|sqrt|begin)|[=<>\u2264\u2265\u21d2]|\b(?:therefore|thus|hence|answer|solution|proof)\b", text, re.I))
    filler = len(re.findall(r"\b(?:basically|actually|just|really|very|perhaps|maybe|in my opinion|it is worth noting|as you may know|hope this helps|great question|of course)\b", low))
    quotes = sum(len(m.group(0).split()) for m in re.finditer(r"(?s)```.*?```|>[^\n]+", text))
    conclusion = bool(re.search(r"\b(?:the answer is|therefore|thus|hence|we get|we conclude|equals?)\b|=", low))
    if not conclusion and math == 0:
        return max(0.0, round(2.0 - min(2.0, len(words) / 100.0), 1))
    direct = 10.0
    if len(words) > 80:
        direct -= min(4.0, (len(words) - 80) / 80.0)
    if len(words) > 300:
        direct -= min(3.0, (len(words) - 300) / 150.0)
    direct -= min(3.0, filler * 0.45)
    direct -= min(4.0, 5.0 * quotes / max(1, len(words)))
    if math < 2:
        direct -= 1.5
    return round(max(0.0, min(10.0, direct)), 1)


def score__patents__a234(text):
    text = text or ""
    match = re.search(r"(?ims)^\s*ABSTRACT\s*:?[ \t]*$(.*?)(?=^\s*(?:CLAIMS?|DESCRIPTION|BACKGROUND|FIELD)\s*:?[ \t]*$|\Z)", text)
    abstract = match.group(1) if match else ""
    low = abstract.lower()
    problem = bool(re.search(r"\b(?:problem|limitation|deficien|drawback|shortcoming|prior art|conventional(?:ly)?|however|fails? to|unable to|consum(?:es|ing) (?:excessive )?bandwidth|inefficien|difficult)\b", low))
    solution = bool(re.search(r"\b(?:solve|overcome|address|improv|provide[sd]?|propos|thereby|reduce[sd]?|avoid[sd]?|enable[sd]?|according to the invention)\b", low))
    technical = bool(re.search(r"\b(?:system|method|apparatus|device|processor|circuit|network|signal|data|controller|module|algorithm)\b", low))
    return 10.0 if problem and solution and technical else 0.0


def score__patents__a42(text):
    text = text or ""
    match = re.search(r"(?ims)^\s*CLAIMS:\s*$([\s\S]*)", text)
    if not match:
        return 0.0
    claims = match.group(1)
    first = re.search(r"(?ims)^\s*1\.\s*(.*?)(?=^\s*2\.|\Z)", claims)
    if not first:
        return 1.0
    claim = first.group(1).strip().lower()
    if re.search(r"\b(?:cancel(?:ed|led)|withdrawn)\b", claim):
        return 1.5
    dependent = bool(re.search(r"\b(?:according to claim|of claim|as recited in claim)\s+\d+", claim))
    medical = bool(re.search(r"\b(?:treating|treatment|administering|patient|subject in need)\b", claim))
    concrete = bool(re.search(r"\b(?:comprising|including|a method|a system|an apparatus|device|processor|circuit|receiving|determining|generating|transmitting|forming|coupled to|configured to)\b", claim))
    if concrete and not dependent and not medical:
        return 10.0
    return 1.5


def score__press_releases__a118(text):
    text = text or ""
    if not text.strip():
        return 0.0
    low = text.lower()
    alpha = re.findall(r"\b[A-Za-z]+\b", text)
    english = len(re.findall(r"\b(?:the|and|of|to|in|for|is|on|with|that)\b", low))
    if len(alpha) >= 20 and english / len(alpha) < 0.025:
        return 0.0
    date = bool(re.search(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", text, re.I))
    byline = bool(re.search(r"(?im)^\s*(?:by\s+[A-Z][\w.'-]+|author\s*:|written by\s+)", text))
    headline = bool(re.search(r"(?m)^\s*(?:#{1,2}\s+.{5,}|[A-Z][^.!?\n]{8,80})\s*$", text))
    body_sentences = len(re.findall(r"[.!?](?:\s|$)", text))
    article = bool(re.search(r"\b(?:announc(?:es|ed|ement)|press release|news|today|company|reported|launch(?:es|ed)?)\b", low))
    nav = len(re.findall(r"\b(?:home|menu|products|services|contact|privacy|cookie|sign in|search|sitemap)\b", low))
    if headline and byline and date and body_sentences >= 5:
        return 10.0
    if byline and date and body_sentences >= 4:
        return 7.5
    if date and article and body_sentences >= 2:
        return 6.2
    if nav >= 4 and (article or body_sentences >= 1):
        return 5.0
    if re.search(r"\b(?:about us|our company|who we are|our mission)\b", low) and not date:
        return 1.2
    return 0.0 if nav >= 3 or body_sentences == 0 else 2.5


def score__math__a234(text):
    text = text or ""
    if re.search(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\begin\s*\{[^}]+\}", text):
        return 10.0
    if re.search(r"(?<!\$)\$(?!\$)[^$\n]+(?<!\$)\$(?!\$)", text):
        return 4.4
    return 0.0


def score__code_review__a9(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    low = text.lower()
    deep = len(re.findall(r"\b(?:because|since|therefore|however|architecture|design|trade-?off|preferable|why do we|what if|how does|behavior|performance|maintainability|compatibility|implementation|alternative|could we|might)\b", low))
    dialogue = len(re.findall(r"\b(?:i think|i agree|good point|thanks|makes sense|interesting|reply)\b|\?", low))
    trivial = len(re.findall(r"\b(?:remove|add empty line|fix typo|typo|formatting|nit|lint|done|lgtm)\b", low))
    ratio = (deep + 0.6 * dialogue) / max(1.0, deep + dialogue + trivial)
    score = 1.0 + 7.5 * ratio + min(1.5, len(words) / 100.0)
    if deep == 0:
        score = min(score, 3.0)
    if deep >= 4 and dialogue >= 2:
        score = max(score, 9.0)
    return round(max(0.0, min(10.0, score)), 1)


def score__patents__a0(text):
    text = text or ""
    claims_match = re.search(r"(?ims)^\s*CLAIMS:\s*$([\s\S]*)", text)
    abstract = bool(re.search(r"(?im)^\s*ABSTRACT:\s*$", text))
    if not claims_match:
        return 1.5 if abstract else 0.0
    claims_text = claims_match.group(1)
    numbered = re.findall(r"(?m)^\s*\d+[.)]\s+\S", claims_text)
    if len(numbered) >= 2:
        return 10.0
    if len(numbered) == 1:
        return 5.0
    readable = len(re.findall(r"\b\w+\b", claims_text))
    return 5.0 if readable >= 8 else 2.0


def score__code_review__a306(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    low = text.lower()
    depth = len(re.findall(r"\b(?:because|since|therefore|however|trade-?off|architecture|design|pattern|system behavior|implementation choice|alternative|performance|scalability|maintainability|why|how|rationale|impact)\b", low))
    exchange = len(re.findall(r"\b(?:i think|i agree|what if|could we|would it|good point|makes sense|reply|thanks)\b|\?", low))
    shallow = len(re.findall(r"\b(?:fix this|done|nit|typo|lgtm|formatting|bot|lint)\b", low))
    score = min(5.5, depth * 0.8) + min(2.0, exchange * 0.5) + min(2.5, len(words) / 70.0)
    if depth == 0:
        score = min(score, 3.0)
    elif depth <= 2 and exchange <= 1:
        score = max(4.0, min(score, 6.0))
    if depth >= 4 and exchange >= 2:
        score = max(score, 7.5)
    score -= min(2.0, shallow * 0.35)
    return round(max(0.0, min(10.0, score)), 1)


def score__patents__a179(text):
    text = text or ""
    match = re.search(r"(?ims)^\s*ABSTRACT\s*:?[ \t]*$(.*?)(?=^\s*(?:CLAIMS?|DESCRIPTION|BACKGROUND|FIELD)\s*:?[ \t]*$|\Z)", text)
    abstract = match.group(1) if match else ""
    return 10.0 if "[...]" in abstract else 0.0


def score__math__a114(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    low = text.lower()
    engagement = len(re.findall(r"\b(?:your (?:work|reasoning|proof|approach|calculation|argument|step|notation)|you (?:wrote|assumed|claimed|correctly)|your result)\b", low))
    verification = len(re.findall(r"\b(?:correct|incorrect|valid|verify|check|mistake|error|flaw|issue|missing|holds|does not follow)\b", low))
    constructive = len(re.findall(r"\b(?:instead|you could|better to|clarify|rewrite|alternative|notation|because|however|note that|add|explain)\b", low))
    standalone = bool(re.search(r"\b(?:the answer is|solution:|we solve|let us solve)\b", low)) and engagement == 0
    if standalone or engagement == 0:
        return 0.0
    score = min(4.0, engagement * 1.2) + min(2.5, verification * 0.7) + min(2.5, constructive * 0.55) + min(1.0, len(words) / 100.0)
    if len(words) < 20:
        score = min(score, 3.0)
    if engagement >= 2 and verification >= 2 and constructive >= 3 and len(words) >= 55:
        score = max(score, 8.5)
    return round(max(0.0, min(10.0, score)), 1)


def score__math__a180(text):
    text = text or ""
    if not text.strip():
        return 0.0
    notation = len(re.findall(r"\$[^$]+\$|\\(?:frac|sum|int|sqrt|begin|forall|exists)|[=<>\u2264\u2265\u2208\u2200\u2203\u2211\u222b\u21d2\u221a]|\b\w+\s*\([^)]*\)\s*=", text))
    if notation == 0:
        return 0.0
    low = text.lower()
    rigor = len(re.findall(r"\b(?:assume|suppose|let|definition|lemma|theorem|proof|because|therefore|thus|hence|implies|if and only if|contradiction|case|we conclude)\b", low))
    words = len(re.findall(r"\b\w+\b", text))
    score = 4.5 + min(2.8, notation * 0.55) + min(2.2, rigor * 0.4) + min(0.5, words / 160.0)
    if notation >= 4 and rigor >= 5 and words >= 45:
        score = 10.0
    return round(max(0.0, min(10.0, score)), 1)


def score__press_releases__a104(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    low = text.lower()
    numbers = re.findall(r"(?<!\w)(?:[$\u00a3\u20ac]\s*)?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|basis points?|bps|million|billion|trillion|x)?", text, re.I)
    metrics = re.findall(r"\b(?:expense ratio|return on equity|return on assets|earnings per share|eps|revenue|net income|ebitda|margin|yield|assets under management|aum|market cap|price-to-earnings|p/e|performance|annualized|quarter-over-quarter|year-over-year|cash flow|dividend|basis points?)\b", low)
    finance = re.findall(r"\b(?:invest(?:ment|or|ing)|financial|fund|portfolio|equity|stock|bond|share|fiscal|quarter|capital|market)\b", low)
    density = 100.0 * (len(numbers) + 1.5 * len(metrics)) / len(words)
    if not numbers and not metrics:
        return 0.0
    if density >= 8.0 and len(numbers) >= 6 and len(metrics) >= 3:
        return 10.0
    if finance or metrics:
        return round(max(1.0, min(9.5, 3.0 + density * 0.65)), 1)
    return round(min(4.0, density * 0.5), 1)


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
