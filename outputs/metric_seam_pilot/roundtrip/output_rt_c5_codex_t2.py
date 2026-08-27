# AUTO: blind rule compilation chunk c5
import re


def score__math__a168(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    advanced = [
        "abstract algebra", "group theory", "ring theory", "field theory",
        "real analysis", "complex analysis", "functional analysis",
        "differential geometry", "algebraic geometry", "category theory",
        "topology", "measure theory", "hilbert space", "banach space",
        "manifold", "homology", "cohomology", "functor", "isomorphism",
        "compactness", "lebesgue", "riemannian", "eigenvalue theorem"
    ]
    intermediate = [
        "calculus", "derivative", "integral", "gradient", "matrix",
        "linear algebra", "vector space", "probability", "variance",
        "number theory", "induction", "differential equation", "eigenvalue",
        "limit", "convergence", "series", "determinant"
    ]
    proof = len(re.findall(r"\b(?:proof|theorem|lemma|proposition|corollary|suppose|assume|therefore|hence|qed)\b", s))
    formal = len(re.findall(r"(?:\bfor all\b|\bthere exists\b|\bif and only if\b|⇒|⇔|∀|∃|∎)", s))
    adv = sum(s.count(x) for x in advanced)
    mid = sum(s.count(x) for x in intermediate)
    equations = len(re.findall(r"(?:[=<>≤≥]|\b(?:sin|cos|log|lim|sum)\b)", s))
    if adv >= 3 and proof + formal >= 3:
        return 10.0
    if adv >= 1 and proof + formal >= 2:
        return 9.0
    if adv >= 1:
        return 7.5
    if mid >= 3 and proof >= 2:
        return 7.0
    if mid >= 1 and (proof or equations >= 2):
        return 5.0
    if mid >= 1 or proof >= 2:
        return 3.5
    if equations or re.search(r"\b(?:add|subtract|multiply|divide|arithmetic|algebra|equation|number)\b", s):
        return 1.0
    return 0.0


def score__code_review__a126(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    deep = len(re.findall(r"\b(?:architecture|design|race condition|deadlock|security|vulnerability|complexity|performance|memory leak|transaction|concurrency|invariant|regression|edge case|failure mode|trade-?off|backward compatibility|root cause)\b", s))
    reasoning = len(re.findall(r"\b(?:because|therefore|otherwise|causes?|results? in|so that|in order to|instead|consider|for example|e\.g\.)\b", s))
    tests = len(re.findall(r"\b(?:test|coverage|assert|mock|fixture)\b", s))
    trivial = len(re.findall(r"\b(?:typo|nit|naming|rename|formatting|whitespace|indent|lint|semicolon|spelling)\b", s))
    terse = len(re.findall(r"(?im)^\s*(?:lgtm|looks good|fix this|remove this|rename this|done|thanks|\+1)[.!]?\s*$", text))
    words = len(re.findall(r"\b\w+\b", text))
    raw = 1.0 + min(5.0, deep * 1.25) + min(3.0, reasoning * 0.55) + min(1.5, tests * 0.4)
    if words >= 120 and deep + reasoning >= 4:
        raw += 1.0
    raw -= min(4.0, trivial * 0.45 + terse * 1.0)
    return round(max(0.0, min(10.0, raw)), 1)


def score__math__a198(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    attempted = bool(re.search(r"\b(?:is|was) (?:my|this|the following) proof (?:correct|valid|right|sound)\b|\b(?:check|verify|validate|review) (?:my|this) proof\b|\bmy (?:attempt|proof)\b.*\b(?:correct|valid|check|verify)\b", s, re.S))
    proof_terms = len(re.findall(r"\b(?:proof|assume|suppose|let|theorem|lemma|therefore|thus|hence|implies|contradiction|qed)\b|∎|⇒", s))
    reasoning = len(re.findall(r"\b(?:because|since|therefore|thus|hence|so|which implies|it follows)\b", s))
    words = len(re.findall(r"\b\w+\b", text))
    question_only = text.strip().endswith("?") and proof_terms < 2 and reasoning == 0
    if attempted:
        return 10.0
    if proof_terms >= 6 and reasoning >= 3 and words >= 100:
        return 6.7
    if proof_terms >= 4 and reasoning >= 2 and words >= 55:
        return 6.7
    if proof_terms >= 2 or (reasoning >= 2 and words >= 30):
        return 3.3
    if question_only or words < 20 or (re.search(r"\b(?:calculate|compute|evaluate|what is)\b", s) and proof_terms == 0):
        return 0.0
    return 0.0


def score__code_review__a153(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    substantive = len(re.findall(r"\b(?:architecture|design|interface|abstraction|edge case|race condition|security|performance|complexity|compatibility|transaction|concurrency|failure|bug|regression|alternative|refactor)\b", s))
    explanation = len(re.findall(r"\b(?:because|since|therefore|otherwise|so that|which (?:means|causes)|for example|consider|suggest|instead)\b", s))
    specific = len(re.findall(r"`[^`\n]+`|\b(?:line|function|method|class|variable|parameter|return value|exception|test case)\b", text, re.I))
    superficial = len(re.findall(r"\b(?:nit|typo|rename|formatting|whitespace|lint|spelling)\b", s))
    terse = len(re.findall(r"(?im)^\s*(?:remove|rename|fix this|change this|why\??|done|lgtm|thanks)[.!]?\s*$", text))
    words = len(re.findall(r"\b\w+\b", text))
    value = 0.8 + min(4.0, substantive * 0.8) + min(2.8, explanation * 0.55) + min(2.0, specific * 0.25)
    if words >= 100 and substantive and explanation:
        value += 1.0
    value -= min(3.5, superficial * 0.35 + terse * 0.8)
    return round(max(0.0, min(10.0, value)), 1)


def score__CAL__CAL5(text):
    if not isinstance(text, str):
        return 0.0
    count = len(re.findall(r"(?<![\w])(?=[A-ZÀ-Þ]{3,}(?![\w]))[A-ZÀ-Þ]+", text))
    if count >= 5:
        return 10.0
    if count >= 1:
        return 5.0
    return 0.0


def score__code_review__a81(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    technical = len(re.findall(r"\b(?:architecture|design|trade-?off|edge case|concurrency|performance|security|complexity|interface|api|invariant|regression|implementation|failure mode|race condition)\b", s))
    probing = len(re.findall(r"(?:\bwhy\b|\bhow (?:does|would|can|should)\b|\bwhat (?:if|happens)\b|\bcould (?:we|you)\b)[^?\n]{0,120}\?", s))
    rationale = len(re.findall(r"\b(?:because|since|therefore|otherwise|the reason|so that|in order to|this (?:ensures|avoids|prevents|allows))\b", s))
    author_reply = len(re.findall(r"(?im)^\s*(?:author|reply|response|developer|op)\s*[:\-]", text))
    superficial = len(re.findall(r"\b(?:formatting|whitespace|typo|nit|naming|rename|lint|indentation|semicolon)\b", s))
    terse = len(re.findall(r"(?im)^\s*(?:remove this|rename this|fix this|done|lgtm|thanks|\+1)[.!]?\s*$", text))
    score = 1.0 + min(3.5, technical * 0.7) + min(2.0, probing * 0.8) + min(2.5, rationale * 0.45)
    if author_reply and rationale:
        score += min(2.0, author_reply * 0.6)
    elif technical + probing >= 2:
        score = min(score, 7.0)
    score -= min(4.0, superficial * 0.4 + terse * 0.8)
    return round(max(0.0, min(10.0, score)), 1)


def score__press_releases__a80(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    release = bool(re.search(r"\b(?:press release|news release|for immediate release|prnewswire|business wire|globe newswire|media contact)\b", s))
    dateline = bool(re.search(r"(?m)^\s*[A-Z][A-Z .'-]{2,},\s*(?:[A-Z][a-z]+,?\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}", text))
    announcement = bool(re.search(r"\b(?:announc(?:e|es|ed|ing)|launch(?:es|ed|ing)?|introduc(?:e|es|ed|ing)|unveil(?:s|ed|ing)?|debut(?:s|ed|ing)?|release(?:s|d|ing)?)\b", s))
    offering = bool(re.search(r"\b(?:new|innovative|next-generation)\s+(?:product|service|technology|platform|solution|application|brand|program|initiative|device|system|offering)\b|\binitiative\b", s))
    corporate_general = bool(re.search(r"\b(?:appoint(?:s|ed|ment)|conference|event|award|quarterly|earnings|acquisition|merger|organizational update|board of directors)\b", s))
    web_junk = len(re.findall(r"\b(?:home|menu|sign in|log in|cookie|privacy policy|terms of use|shopping cart|skip to content)\b", s))
    if (release or dateline) and announcement and offering:
        return 10.0
    if (release or dateline) and (announcement or corporate_general):
        return 7.5
    if release and corporate_general:
        return 7.5
    if web_junk >= 2 or not (release or dateline):
        return 0.0
    return 0.0


def score__press_releases__a105(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = re.findall(r"\b\w+\b", text)
    sentences = [x for x in re.split(r"(?<=[.!?])\s+|\n\s*\n", text.strip()) if len(re.findall(r"\b\w+\b", x)) >= 5]
    long_sentences = sum(len(re.findall(r"\b\w+\b", x)) >= 10 for x in sentences)
    s = text.lower()
    navigation = len(re.findall(r"\b(?:home|menu|navigation|sign in|log in|subscribe|cookie|privacy policy|terms of use|contact us|skip to content|read more|site map)\b", s))
    linkish = len(re.findall(r"(?:https?://|www\.|\[[^\]]+\]\([^)]*\)|(?m)^\s*[•*\-]\s+\S)", text))
    release = len(re.findall(r"\b(?:press release|announc(?:es|ed)|company|corporation|media contact|prnewswire)\b", s))
    coherence = min(1.0, (long_sentences * 15) / max(1, len(words)))
    junk = min(1.0, (navigation * 6 + linkish * 3) / max(1, len(words)))
    if len(words) >= 150 and long_sentences >= 5 and coherence >= 0.35 and junk < 0.12:
        return round(min(10.0, 8.0 + min(2.0, len(words) / 500.0)), 1)
    if len(words) >= 70 and long_sentences >= 3 and junk < 0.25:
        return 6.0 if release else 7.5
    if len(words) >= 35 and long_sentences >= 2:
        return 5.0
    if navigation + linkish >= max(2, long_sentences * 2):
        return 1.0
    return 2.0


def score__press_releases__a76(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = re.findall(r"\b\w+\b", text)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    prose_words = 0
    nonprose_words = 0
    nav_re = re.compile(r"\b(?:home|menu|sign in|log in|subscribe|cookie|privacy|terms|contact|search|read more|next|previous)\b", re.I)
    for block in blocks:
        bw = len(re.findall(r"\b\w+\b", block))
        sentences = len(re.findall(r"[.!?](?:\s|$)", block))
        listlike = bool(re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+|https?://|\|", block))
        if bw >= 20 and sentences >= 1 and not listlike and len(nav_re.findall(block)) <= 1:
            prose_words += bw
        else:
            nonprose_words += bw
    if not words:
        return 0.0
    ratio = prose_words / max(1, prose_words + nonprose_words)
    score = ratio * 10.0
    if len(words) < 25:
        score = min(score, 4.0)
    return round(max(0.0, min(10.0, score)), 1)


def score__math__a54(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    words = len(re.findall(r"\b\w+\b", text))
    reasoning = len(re.findall(r"\b(?:because|since|therefore|thus|hence|which implies|it follows|suppose|assume|consider|proof)\b", s))
    math = len(re.findall(r"(?:[=<>≤≥]|\b(?:theorem|lemma|derivative|integral|matrix|equation|function|limit|probability|integer|vector)\b)", s))
    structure = len(re.findall(r"(?m)^\s*(?:\d+[.)]|step\s+\d+|[-*•])\s+", text))
    uncertainty = len(re.findall(r"\b(?:maybe|probably|i think|not sure|guess)\b", s))
    if words < 15 or (reasoning == 0 and math == 0):
        return 0.0
    if words >= 220 and reasoning >= 7 and math >= 5 and structure >= 2 and uncertainty == 0:
        return 10.0
    if words >= 80 and reasoning >= 3 and math >= 2 and uncertainty == 0:
        return 7.5
    if words >= 35 and (reasoning >= 1 or math >= 2):
        return 5.0
    return 0.0


def score__patents__a228(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    moving = len(re.findall(r"\b(?:rotat(?:e|es|ing|able)|pivot(?:s|ing|able)?|slide|sliding|reciprocat(?:e|ing)|gear|hinge|shaft|bearing|piston|spring|lever|cam|actuator|motor|valve|pump|wheel|conveyor|linkage|movable|movement)\b", s))
    physical = len(re.findall(r"\b(?:housing|frame|chassis|assembly|component|member|surface|wall|chamber|tube|pipe|fluid|nozzle|sensor|electrode|substrate|material|layer|device|apparatus|fastener|bracket)\b", s))
    interaction = len(re.findall(r"\b(?:coupled to|connected to|mounted (?:on|to)|engages?|adjacent to|disposed (?:in|on|between)|configured to|relative to|between a|attached to)\b", s))
    software = len(re.findall(r"\b(?:algorithm|database|data processing|neural network|software|protocol|packet|memory|processor|logic circuit|business method|user interface)\b", s))
    if moving >= 3 and physical >= 5 and interaction >= 3:
        return 10.0
    if moving >= 1 and physical >= 4 and interaction >= 2:
        return 8.1
    if physical >= 4 and interaction >= 1:
        return 7.0
    if physical >= 2:
        return 5.1
    if software and not moving and physical < 2:
        return 0.0
    if physical:
        return 3.0
    return 0.0


def score__press_releases__a25(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    cleaned = re.sub(r"<[^>]*>|https?://\S+|www\.\S+|```.*?```|`[^`]*`", " ", text, flags=re.S | re.I)
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žΑ-ωА-я一-鿿ぁ-ゟ゠-ヿ가-힣]+(?:'[A-Za-z]+)?", cleaned)
    if not tokens:
        return 0.0
    common = set("the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because these give day most us is are was were been being has had does did should may company said press release product service today announced more information contact business provides customers market including through".split())
    english = 0.0
    nonenglish = 0.0
    for tok in tokens:
        low = tok.lower()
        if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", low):
            if low in common or len(low) >= 2:
                english += 1.0
            else:
                english += 0.5
        else:
            nonenglish += 1.0
    ratio = english / max(1.0, english + nonenglish)
    return round(max(0.0, min(10.0, ratio * 10.0)), 1)


def score__press_releases__a262(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    beginning = text.lstrip()[:1200]
    months = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    standard = re.search(r"(?m)^\s*[A-Z][A-Z .'-]{2,}(?:,\s*(?:[A-Z]{2}|[A-Z][a-z]+))?,?\s*[—-]?\s*" + months + r"\s+\d{1,2},\s+\d{4}(?:\s*/(?:PRNewswire|Business Wire|GlobeNewswire)/)?\s*(?:--|—|-)", beginning)
    if standard:
        return 10.0
    partial = re.search(months + r"\s+\d{1,2},\s+\d{4}|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|/(?:PRNewswire|Business Wire|GlobeNewswire)/", text, re.I)
    return 2.0 if partial else 0.0


def score__press_releases__a75(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = re.findall(r"\b[A-Za-z][A-Za-z.'’-]*\b", text)
    if not words:
        return 0.0
    caps = sum(w.isupper() and len(w) >= 2 for w in words)
    titled = sum(w[0].isupper() for w in words)
    density = (caps * 1.7 + titled) / len(words)
    list_lines = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+|(?m)^\s*[^\n]{1,35}\s*$", text))
    links = len(re.findall(r"https?://|www\.|\b(?:home|menu|contact|about|products|services|news|search|login|sign in)\b", text, re.I))
    structure = min(1.0, (list_lines + links) / 10.0)
    combined = min(1.0, density / 0.45) * 7.0 + structure * 3.0
    if density < 0.03 and structure == 0:
        return 0.0
    if density < 0.09 and structure < 0.2:
        return 2.0
    if combined >= 8.5:
        return 10.0
    if combined >= 4.5:
        return 6.0
    return round(max(2.0, min(5.5, combined)), 1)


def score__code_review__a72(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    precise = len(re.findall(r"\b(?:syntax|formatting|lint|whitespace|indentation|style guide|pep ?8|semicolon|trailing comma|line length|type annotation|rename|remove|replace|use [a-z_][a-z0-9_]*(?:\(\))?)\b", s))
    directives = len(re.findall(r"(?im)^\s*(?:please\s+)?(?:add|remove|rename|replace|change|use|move|indent|format|sort|fix|delete)\b", text))
    code = len(re.findall(r"`[^`\n]+`|```", text))
    conversational = len(re.findall(r"\b(?:because|i think|i wonder|perhaps|maybe|what do you think|could we|would it|the reason|trade-?off|architecture|rationale)\b", s))
    questions = text.count("?")
    objective = precise + directives + min(code, 4)
    discussion = conversational + questions
    if objective == 0 and discussion == 0:
        return 2.0
    ratio = objective / max(1, objective + discussion)
    score = 1.0 + 9.0 * ratio
    if objective < 2:
        score = min(score, 4.0)
    if discussion == 0 and objective >= 4:
        score = max(score, 8.0)
    return round(max(0.0, min(10.0, score)), 1)


def score__press_releases__a100(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    dateline = bool(re.search(r"(?m)^\s*[A-Z][A-Z .'-]{2,},\s*(?:[A-Z]{2}|[A-Z][a-z]+)?\s*,?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}", text))
    announcement = bool(re.search(r"\b(?:announc(?:e|es|ed|ing)|launch(?:es|ed)?|introduc(?:e|es|ed)|unveil(?:s|ed)|acquir(?:e|es|ed)|appoint(?:s|ed))\b", s))
    corporate = len(re.findall(r"\b(?:company|corporation|inc\.?|ltd\.?|organization|business|customers?|market|chief executive|ceo)\b", s))
    boiler = len(re.findall(r"\b(?:about [A-Z][\w&.-]+|forward-looking statements|media contact|investor relations|safe harbor|for more information|source:)\b", text, re.I))
    explicit_release = bool(re.search(r"\b(?:press release|news release|for immediate release|prnewswire|business wire|globe newswire)\b", s))
    junk = len(re.findall(r"\b(?:shopping cart|cookie settings|page not found|404 error|sign in|log in|navigation menu|product catalog)\b", s))
    elements = int(dateline) + int(announcement) + int(corporate >= 2) + int(boiler >= 1) + int(explicit_release)
    if elements >= 4 and announcement:
        return 10.0
    if elements >= 3 and announcement:
        return 8.0
    if elements == 2 and announcement:
        return 6.0
    if junk >= 2 or elements == 0:
        return 0.0
    return 3.0


def score__math__a6(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = re.findall(r"\b\w+\b", text)
    n = len(words)
    s = text.lower()
    steps = len(re.findall(r"(?m)^\s*(?:step\s+\d+|\d+[.)]|[-*•])\s+|\b(?:first|next|then|finally)\b", s))
    explanation = len(re.findall(r"\b(?:because|since|therefore|thus|this means|in other words|for example|notice that|so that|the idea|think of|analog)\b", s))
    jargon = len(re.findall(r"\b(?:isomorphism|homomorphism|eigenbasis|sigma-algebra|functorial|homeomorphism|asymptotic|diffeomorphism|ergodic|tensorial|noncommutative)\b", s))
    formulas = len(re.findall(r"[=<>≤≥]|\b(?:integral|derivative|equation|function|matrix|probability)\b", s))
    avg_sentence = n / max(1, len(re.findall(r"[.!?](?:\s|$)", text)))
    if n < 15 or (formulas and explanation == 0 and steps == 0 and n < 50):
        return 1.0
    score = 4.0 + min(2.5, steps * 0.5) + min(2.5, explanation * 0.45)
    if 40 <= n <= 350:
        score += 0.8
    if avg_sentence <= 24:
        score += 0.5
    score -= min(3.0, jargon * 0.5)
    if explanation == 0:
        score = min(score, 6.0)
    return round(max(0.0, min(10.0, score)), 1)


def score__press_releases__a41(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = len(re.findall(r"\b\w+\b", text))
    s = text.lower()
    nav = len(re.findall(r"\b(?:home|menu|navigation|search|sign in|log in|register|subscribe|account|language|select language|contact us|about us|products|services|newsroom|careers|cookie|privacy policy|terms of use|site map|skip to content|read more|share this)\b", s))
    links = len(re.findall(r"https?://|www\.|\[[^\]]+\]\([^)]*\)|(?m)^\s*(?:[-*•]|\|)\s*\S", text))
    fragments = sum(1 for line in text.splitlines() if 0 < len(re.findall(r"\b\w+\b", line)) <= 5)
    prose = sum(1 for x in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if len(re.findall(r"\b\w+\b", x)) >= 12)
    boiler_weight = nav * 5 + links * 3 + fragments * 2
    prose_weight = prose * 15
    ratio = boiler_weight / max(1, boiler_weight + prose_weight)
    if nav == 0 and links == 0 and fragments == 0:
        return 0.0
    score = ratio * 10.0
    if prose >= 3 and score < 4.0:
        score = min(3.0, score)
    if words < 25 and boiler_weight:
        score = max(score, 8.0)
    return round(max(0.0, min(10.0, score)), 1)


def score__press_releases__a66(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = text.lower()
    finance = len(re.findall(r"\b(?:revenue|earnings|profit|loss|ebitda|eps|cash flow|dividend|share price|stock|market capitalization|fiscal|quarter|guidance|margin|assets|liabilities|investment|investor|analyst|acquisition|merger|sales|net income|operating income)\b", s))
    numeric = len(re.findall(r"(?:[$€£]\s?\d[\d,.]*(?:\s*(?:million|billion|m|bn))?|\b\d+(?:\.\d+)?\s*%|\b(?:q[1-4]|fy\s?\d{2,4})\b|\$?\d+\.\d+\s+(?:per share|million|billion))", s))
    analyst = len(re.findall(r"\b(?:analysts?|consensus|forecast|estimate|year-over-year|quarter-over-quarter|basis points|outlook)\b", s))
    words = len(re.findall(r"\b\w+\b", text))
    density = (finance * 3 + numeric * 4 + analyst * 3) / max(1, words)
    if finance >= 8 and numeric >= 5 and density >= 0.15:
        return 10.0
    if finance >= 5 and numeric >= 3:
        return 9.0
    if finance >= 3 or (finance >= 1 and numeric >= 2):
        return 6.0
    if finance >= 1 or numeric >= 1:
        return 3.0
    return 0.0


def score__CAL__CAL4(text):
    if not isinstance(text, str):
        return 0.0
    return 10.0 if re.search(r"(?m)^#", text) else 0.0


def score__press_releases__a73(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = re.findall(r"\b\w+\b", text)
    total = len(words)
    if total == 0:
        return 0.0
    s = text.lower()
    boiler = len(re.findall(r"\b(?:home|menu|navigation|search|sign in|log in|register|subscribe|account|contact us|about us|cookie|privacy policy|terms of use|site map|skip to content|copyright|all rights reserved|legal disclaimer|read more)\b", s))
    linkish = len(re.findall(r"https?://|www\.|\[[^\]]+\]\([^)]*\)|(?m)^\s*(?:[-*•]|\|)\s*\S", text))
    short_lines = sum(1 for line in text.splitlines() if 0 < len(re.findall(r"\b\w+\b", line)) <= 5)
    prose_sentences = sum(1 for x in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if len(re.findall(r"\b\w+\b", x)) >= 10)
    meaningful = prose_sentences * 14
    junk = boiler * 5 + linkish * 3 + short_lines * 2
    if meaningful == 0:
        return 0.0 if junk or total < 20 else 2.0
    ratio = meaningful / max(1, meaningful + junk)
    score = ratio * 10.0
    if prose_sentences >= 5 and junk == 0:
        return 10.0
    if ratio >= 0.7:
        score = max(7.0, score)
    elif ratio <= 0.4:
        score = min(4.0, score)
    return round(max(0.0, min(10.0, score)), 1)


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
