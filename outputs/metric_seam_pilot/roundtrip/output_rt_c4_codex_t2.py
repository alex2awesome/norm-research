# AUTO: blind rule compilation chunk c4
import re


def score__math__a24(text):
    t = (text or "")[:300000]
    if not t.strip():
        return 0.0
    low = t.lower()
    words = re.findall(r"\b\w+\b", low)
    math_tokens = len(re.findall(r"[=<>≤≥∫∑√^]|\\(?:frac|sum|int|lim|begin)|\b(?:theorem|lemma|proof|derive|equation|function|integer|matrix|probability)\b", t, re.I))
    reasoning = len(re.findall(r"\b(?:because|therefore|thus|hence|since|implies|consequently|suppose|assume|case|it follows)\b", low))
    completion = len(re.findall(r"\b(?:q\.?e\.?d|proved|therefore the answer|final answer|we conclude|as required)\b", low))
    gaps = len(re.findall(r"\b(?:not sure|maybe|probably|cannot solve|left as an exercise|without proof|approximately)\b", low))
    if math_tokens == 0:
        return min(2.5, len(words) / 120.0)
    score = 2.5 + min(2.5, math_tokens / 6.0) + min(2.5, reasoning / 4.0) + min(1.5, len(words) / 180.0) + min(1.0, completion * 0.5) - min(3.0, gaps * 0.8)
    return round(max(0.0, min(10.0, score)), 2)


def score__press_releases__a204(text):
    t = (text or "")[:300000]
    words = re.findall(r"\b\w+\b", t.lower())
    if not words:
        return 0.0
    ui_patterns = [
        r"\bsign in\b", r"\blog ?in\b", r"\bsearch\b", r"\bmenu\b", r"\bregister\b",
        r"\bsubscribe\b", r"\bcontact(?: us)?\b", r"\bsite ?map\b", r"\bprivacy\b",
        r"\bterms(?: of use)?\b", r"\bcookie", r"\bhome\b", r"\bskip to", r"\baccount\b",
        r"\bpassword\b", r"\busername\b", r"\bfooter\b", r"\bhelp\b", r"\bdownload\b",
        r"\bclick here\b", r"\bread more\b", r"\blearn more\b", r"https?://|www\."
    ]
    hits = sum(len(re.findall(p, t, re.I)) for p in ui_patterns)
    short_lines = sum(1 for x in t.splitlines() if 0 < len(x.split()) <= 4)
    prose_sentences = len(re.findall(r"\b[A-Z][^.!?\n]{45,}[.!?]", t))
    density = (hits * 12.0 + short_lines * 2.0) / max(30.0, len(words))
    score = 10.0 * (1.0 - pow(2.718281828, -density))
    score -= min(3.0, prose_sentences / 5.0)
    return round(max(0.0, min(10.0, score)), 2)


def score__patents__a216(text):
    t = (text or "")[:300000].lower()
    if not t.strip():
        return 0.0
    fields10 = r"\b(?:consumer good|sporting|recreational|exercise equipment|toy|game apparatus|household|appliance|furniture|fixture|fastener|hinge|bracket|container|kitchen|cleaning device)\w*\b"
    fields5 = r"\b(?:automotive|vehicle|vehicular|automobile|engine|transmission|telecommunication|wireless|cellular|radio network|base station|antenna|fuel cell|fuel generation|energy generation|power generation|battery charging)\w*\b"
    fields0 = r"\b(?:software|computer|processor|semiconductor|integrated circuit|transistor|chemical|polymer|alloy|manufacturing|medical|surgical|pharmaceutical|diagnostic|database|neural network)\w*\b"
    a = len(re.findall(fields10, t))
    b = len(re.findall(fields5, t))
    c = len(re.findall(fields0, t))
    if a > b and a > c:
        return 10.0
    if b > a and b > c:
        return 5.0
    if a == b and a > c and a > 0:
        return 7.5
    return 0.0


def score__math__a72(text):
    t = (text or "")[:300000]
    if not t.strip():
        return 0.0
    parts = re.split(r"(?im)^\s*(?:answer|solution|response)\s*:\s*", t, maxsplit=1)
    a = parts[-1]
    low = a.lower()
    words = re.findall(r"\b\w+\b", a)
    equations = len(re.findall(r"[=<>≤≥∫∑√^]|\\(?:frac|sum|int|lim)|\$[^$]+\$", a))
    logic = len(re.findall(r"\b(?:because|therefore|thus|hence|since|implies|suppose|assume|case|follows)\b", low))
    finish = bool(re.search(r"\b(?:q\.?e\.?d|we conclude|final answer|as required|therefore)\b", low))
    uncertainty = len(re.findall(r"\b(?:not sure|maybe|probably|cannot|i don't know|left as an exercise)\b", low))
    if len(words) < 12 or equations == 0:
        return round(max(0.0, min(3.0, len(words) / 10.0 - uncertainty)), 2)
    score = 3.0 + min(2.0, equations / 5.0) + min(2.0, logic / 4.0) + min(2.0, len(words) / 150.0) + (1.0 if finish else 0.0) - uncertainty
    return round(max(0.0, min(10.0, score)), 2)


def score__press_releases__a110(text):
    t = (text or "")[:300000]
    if not t.strip():
        return 0.0
    low = t.lower()
    dateline = bool(re.search(r"(?m)^(?:[A-Z][A-Z .'-]{2,},?\s+(?:[A-Z]{2},?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2}/\d{1,2}/\d{2,4})|[A-Z][A-Za-z .'-]+,\s+[A-Z]{2}\s*[-—])", t))
    announcement = len(re.findall(r"\b(?:announces?|announced|launches?|unveils?|reports? (?:results|earnings)|acquires?|merger|appoints?|partnership|initiative|today announced)\b", low))
    company = len(re.findall(r"\b(?:inc\.?|corp(?:oration)?\.?|company|ltd\.?|plc|llc|nasdaq|nyse)\b", low))
    press = bool(re.search(r"\b(?:press release|for immediate release|news provided by|media contact)\b", low))
    unrelated = bool(re.search(r"\b(?:government of|stock screener|login portal)\b", low))
    non_ascii = sum(ord(c) > 127 for c in t) / max(1, len(t))
    if unrelated or non_ascii > 0.2:
        return 0.5 if announcement else 0.0
    score = 1.0 + (2.5 if dateline else 0.0) + min(3.0, announcement * 1.5) + min(1.5, company * 0.3) + (2.0 if press else 0.0)
    if announcement == 0:
        score = min(score, 6.0)
    return round(max(0.0, min(10.0, score)), 2)


def score__math__a84(text):
    t = (text or "")[:300000].lower()
    if not t.strip():
        return 0.0
    advanced = len(re.findall(r"\b(?:topolog\w*|differential geometry|manifold\w*|abstract algebra|group theory|ring theory|field theory|complex analysis|measure theory|functional analysis|stochastic process|martingale|hilbert space|banach space|homolog\w*|category theory|lie algebra|galois|fourier analysis)\b", t))
    undergraduate = len(re.findall(r"\b(?:calculus|derivative|integral|limit|linear algebra|matrix|eigenvalue|differential equation|real analysis|sequence|series|vector space|probability|proof by induction|partial derivative)\b", t))
    elementary = len(re.findall(r"\b(?:arithmetic|addition|subtraction|multiplication|division|percent|fraction|quadratic|triangle|high school|puzzle|algorithm|counting)\b", t))
    formal = len(re.findall(r"\b(?:theorem|lemma|corollary|proof|axiom|isomorphism|compact|continuous|converges)\b|[∀∃∫∑]", t))
    if advanced:
        return round(min(10.0, 7.0 + min(2.0, advanced * 0.5) + min(1.0, formal * 0.15)), 2)
    if undergraduate:
        return round(min(6.9, 3.0 + min(2.5, undergraduate * 0.35) + min(1.4, formal * 0.15)), 2)
    if elementary or re.search(r"\d+\s*[+*/=-]\s*\d+", t):
        return round(min(2.9, 1.0 + elementary * 0.3), 2)
    return 0.0


def score__press_releases__a67(text):
    t = (text or "")[:300000]
    low = t.lower()
    words = re.findall(r"\b\w+\b", low)
    if not words:
        return 0.0
    terms = re.findall(r"\b(?:animal(?:s)?|animal welfare|animal rights|welfare|humane|slaughter\w*|abattoir\w*|livestock|cruelty|ethical treatment|supply chain|free-range|cage-free|pets?)\b", low)
    if not terms:
        return 0.0
    lines = [x.strip().lower() for x in t.splitlines() if x.strip()]
    prominent = sum(bool(re.search(r"\b(?:animal|welfare|humane|slaughter|abattoir|livestock|cruelty)\b", x)) for x in lines[:8])
    density = len(terms) * 100.0 / len(words)
    if prominent >= 2 and (density >= 2.0 or len(terms) >= 8):
        return min(10.0, 8.5 + min(1.5, density / 4.0))
    if prominent or len(terms) >= 4:
        return round(min(8.0, 5.5 + min(2.5, density)), 2)
    return round(min(2.0, 0.75 + 0.4 * len(terms)), 2)


def score__math__a12(text):
    t = (text or "")[:300000].lower()
    if not t.strip():
        return 0.0
    high = len(re.findall(r"\b(?:abstract algebra|functional analysis|topolog\w*|measure theory|differential geometry|manifold|hilbert|banach|lebesgue|homology|galois|lie group|operator algebra|category theory)\b", t))
    medium = len(re.findall(r"\b(?:multivariable calculus|linear algebra|real analysis|differential equation|eigenvalue|vector space|partial derivative|gradient|jacobian|convergence|calculus)\b", t))
    rigor = len(re.findall(r"\b(?:theorem|lemma|proof|corollary|suppose|therefore|implies|if and only if|axiom)\b|[∀∃]", t))
    elementary = len(re.findall(r"\b(?:arithmetic|percent|fraction|quadratic|high school|solve for x|addition|multiplication)\b", t))
    if high:
        return round(min(10.0, 8.0 + high * 0.35 + rigor * 0.08), 2)
    if medium:
        return round(min(7.0, 4.0 + medium * 0.35 + rigor * 0.12), 2)
    if rigor >= 5:
        return min(7.0, 3.5 + rigor * 0.3)
    return round(min(3.0, 0.5 + rigor * 0.25 + elementary * 0.2), 2)


def score__patents__a204(text):
    t = (text or "")[:300000]
    match = re.search(r"(?is)(?:^|\n)\s*ABSTRACT\s*[:\n]?\s*(.*?)(?=\n\s*(?:CLAIMS?|BACKGROUND|FIELD OF (?:THE )?INVENTION|SUMMARY|DESCRIPTION|BRIEF DESCRIPTION)\b|\Z)", t)
    return 10.0 if match and "The present invention" in match.group(1) else 0.0


def score__press_releases__a101(text):
    t = (text or "")[:300000]
    low = t.lower()
    if not low.strip():
        return 0.0
    dateline = bool(re.search(r"(?m)^[A-Z][A-Z .'-]{2,},?\s+(?:[A-Z]{2},?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2}/\d{1,2})", t))
    announcement = bool(re.search(r"\b(?:announces?|launches?|unveils?|reports? (?:results|earnings)|acquires?|appoints?|today announced|merger)\b", low))
    quote = bool(re.search(r"[“\"][^”\"]{25,}[”\"]|\b(?:said|stated|commented)\s+[A-Z]", t))
    boiler = sum(bool(re.search(p, low)) for p in [r"news provided by", r"terms of use", r"site ?map", r"copyright|©", r"about (?:the |us|[a-z])", r"media contact"])
    nav = len(re.findall(r"\b(?:sign in|log in|menu|search|shopping cart|services|stock quote)\b", low))
    if not dateline and not announcement and not quote:
        return 0.0
    score = (2.5 if dateline else 0.0) + (3.0 if announcement else 0.0) + (2.0 if quote else 0.0) + min(2.5, boiler * 0.5)
    score -= min(2.0, nav * 0.25)
    return round(max(0.0, min(10.0, score)), 2)


def score__code_review__a144(text):
    t = (text or "")[:300000]
    low = t.lower()
    words = re.findall(r"\b\w+\b", low)
    if not words:
        return 0.0
    design = len(re.findall(r"\b(?:architecture|design|trade-?off|alternative|api|backward compatib|interface|abstraction|coupling|dependency|scalab|consistency|concurrency|transaction|invariant|migration)\w*\b", low))
    exchange = len(re.findall(r"(?m)^(?:reviewer|author|reply|comment|response|[A-Za-z][\w.-]*):", t))
    questions = len(re.findall(r"\?", t))
    long_sentences = len(re.findall(r"\b[^.!?\n]{100,}[.!?]", t))
    trivial = len(re.findall(r"\b(?:nit|typo|whitespace|rename|formatting|done|fixed|lgtm)\b", low))
    if design == 0 or (len(words) < 50 and long_sentences == 0):
        return 0.0
    score = 2.5 + min(3.5, design * 0.45) + min(2.0, exchange * 0.35) + min(1.5, questions * 0.25) + min(1.0, long_sentences * 0.25) - min(2.5, trivial * 0.2)
    return round(max(0.0, min(10.0, score)), 2)


def score__code_review__a279(text):
    t = (text or "")[:300000]
    low = t.lower()
    words = re.findall(r"\b\w+\b", low)
    if not words:
        return 0.0
    deep = len(re.findall(r"\b(?:architecture|race condition|deadlock|complexity|performance|latency|memory|edge case|failure mode|invariant|concurren\w*|transaction|atomic|cache|scalab\w*|security|algorithm|api contract|backward compatib\w*|system behavior|overflow)\b", low))
    reasoning = len(re.findall(r"\b(?:because|therefore|otherwise|trade-?off|consider|would cause|in order to|for example|specifically)\b", low))
    trivial = len(re.findall(r"\b(?:typo|nit|naming|whitespace|blank line|formatting|style|lint|semicolon|spelling)\b", low))
    detail = min(2.0, len(words) / 180.0)
    score = 1.0 + min(5.0, deep * 0.55) + min(2.0, reasoning * 0.3) + detail - min(4.0, trivial * 0.35)
    if deep == 0:
        score = min(4.0, score)
    return round(max(0.0, min(10.0, score)), 2)


def score__math__a36(text):
    t = (text or "")[:300000]
    if not t.strip():
        return 0.0
    parts = re.split(r"(?im)^\s*(?:answer|solution|response)\s*:\s*", t, maxsplit=1)
    a = parts[-1]
    low = a.lower()
    words = re.findall(r"\b\w+\b", a)
    math = len(re.findall(r"[=<>≤≥∫∑√^]|\\(?:frac|sum|int|lim)|\b(?:theorem|proof|equation|function|matrix)\b", a, re.I))
    logic = len(re.findall(r"\b(?:because|therefore|thus|hence|since|implies|first|next|finally|suppose|case)\b", low))
    correction = len(re.findall(r"\b(?:however|the mistake|misconception|not quite|instead|note that|the issue)\b", low))
    fragmented = len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S{1,15}\s*$", a))
    if len(words) < 15:
        return round(min(3.0, len(words) / 6.0), 2)
    score = 2.0 + min(2.5, math / 5.0) + min(2.5, logic / 4.0) + min(1.5, len(words) / 140.0) + min(1.5, correction * 0.4) - min(2.0, fragmented * 0.3)
    return round(max(0.0, min(10.0, score)), 2)


def score__code_review__a54(text):
    t = (text or "")[:300000]
    pattern = r"(?im)(?:^|[\s:;,.!?])(?:fixed|done|verified|removed|updated|resolved|addressed|implemented|corrected|changed|completed)(?:[\s.!?,;:]|$)"
    return 10.0 if re.search(pattern, t) else 0.0


def score__math__a96(text):
    t = (text or "")[:300000]
    if not t.strip():
        return 0.0
    q = re.split(r"(?im)^\s*(?:answer|solution|response)\s*:\s*", t, maxsplit=1)[0]
    low = q.lower()
    words = re.findall(r"\b\w+\b", low)
    if not words:
        return 0.0
    advanced = len(re.findall(r"\b(?:topolog\w*|manifold|measure theory|functional analysis|abstract algebra|galois|homology|category theory|complex analysis|stochastic|hilbert|banach|generaliz\w*|necessary and sufficient)\b", low))
    exploration = len(re.findall(r"\b(?:why|intuition|deeper|generaliz\w*|what if|under what conditions|alternative proof|relationship|interpretation|motivation|extend|conceptual|explain)\b", low))
    routine = len(re.findall(r"\b(?:calculate|compute|evaluate|solve for|find the value|homework|verify|simplify|plug in)\b", low))
    score = 1.5 + min(4.0, advanced * 0.8) + min(3.5, exploration * 0.65) + min(1.0, len(words) / 120.0) - min(3.0, routine * 0.7)
    if exploration == 0 and advanced == 0:
        score = min(score, 3.9)
    elif advanced == 0:
        score = min(score, 6.9)
    return round(max(0.0, min(10.0, score)), 2)


def score__patents__a54(text):
    t = (text or "")[:300000].lower()
    if not t.strip():
        return 0.0
    core = len(re.findall(r"\b(?:processor|cpu|gpu|memory controller|operating system|machine learning|neural network|computer-implemented|data processing|computing architecture|instruction set|cache memory|virtual machine|database|software)\b", t))
    applied = len(re.findall(r"\b(?:controller|navigation|communication network|wireless|data management|sensor data|control system|computer system|user interface)\b", t))
    physical = len(re.findall(r"\b(?:mechanical|agricultural|biological|vehicle|engine|apparatus|housing|shaft|gear|valve|crop|medical)\b", t))
    if core >= 3 and core >= applied:
        return round(min(10.0, 8.5 + core * 0.2), 2)
    if core >= 1 and (core + applied >= 3):
        return round(min(9.0, 6.5 + core * 0.35 + applied * 0.2), 2)
    if applied >= 2:
        return round(min(8.0, 5.0 + applied * 0.3), 2)
    if applied == 1 or core == 1:
        return 5.0 if physical else 6.0
    if physical:
        return 1.0 if physical >= 3 else 2.5
    return 0.0


def score__press_releases__a28(text):
    t = (text or "")[:300000]
    low = t.lower()
    words = re.findall(r"\b\w+\b", low)
    if not words:
        return 0.0
    opening = t[:1200]
    immediate = bool(re.search(r"for immediate release", opening, re.I))
    dateline = bool(re.search(r"(?m)^[A-Z][A-Z .'-]{2,},?\s+(?:[A-Z]{2},?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2}/\d{1,2})", opening))
    announcement = len(re.findall(r"\b(?:announces?|launches?|unveils?|reports?|acquires?|appoints?|initiative|event|partnership|today announced)\b", low))
    narrative = len(re.findall(r"\b[A-Z][^.!?\n]{50,}[.!?]", t))
    nav = len(re.findall(r"\b(?:sign in|menu|search|product menu|site map|stock quote|privacy|terms of use|shopping cart)\b", low))
    score = (3.0 if immediate else 0.0) + (3.0 if dateline else 0.0) + min(2.5, announcement * 1.0) + min(1.5, narrative * 0.15) - min(4.0, nav * 0.25)
    if not immediate and not dateline:
        score = min(7.0, score + (2.0 if announcement and narrative >= 2 else 0.0))
    return round(max(0.0, min(10.0, score)), 2)


def score__math__a78(text):
    t = (text or "")[:300000]
    if not t.strip():
        return 0.0
    parts = re.split(r"(?im)^\s*(?:answer|solution|response)\s*:\s*", t, maxsplit=1)
    answer = parts[-1]
    low = answer.lower()
    conceptual = len(re.findall(r"\b(?:misunderstanding|misconception|the flaw|the mistake|the issue|confus\w*|because|reason|intuition|what this means|not necessarily|however|you cannot|the assumption|instead)\b", low))
    direct = len(re.findall(r"\b(?:calculate|substitute|plug in|equals|result is|answer is|simplify)\b|=", low))
    words = len(re.findall(r"\b\w+\b", answer))
    if conceptual >= 3 and words >= 45 and conceptual >= direct:
        return 10.0
    if conceptual >= 1:
        return 5.0
    return 0.0


def score__press_releases__a146(text):
    t = (text or "")[:300000]
    words = re.findall(r"\b\w+\b", t)
    if not words:
        return 0.0
    patterns = [r"https?://\S+", r"www\.\S+", r"\[[^\]]+\]\([^\)]+\)", r"\bclick here\b", r"\bread more\b", r"\blearn more\b", r"\bsign in\b", r"\bregister\b", r"\bsearch\b", r"\bmenu\b", r"\bsubscribe\b", r"\bcontact(?: us)?\b", r"\bsite ?map\b", r"\bdownload\b", r"\badd to cart\b", r"\bsubmit\b", r"\bbutton\b"]
    hits = sum(len(re.findall(p, t, re.I)) for p in patterns)
    linkish_lines = sum(1 for x in t.splitlines() if 0 < len(x.split()) <= 5 and re.search(r"(?:>|→|\||menu|home|contact|search|login|sign in|more)", x, re.I))
    density = (hits * 100.0 + linkish_lines * 35.0) / len(words)
    score = 10.0 * (1.0 - pow(2.718281828, -density / 8.0))
    return round(max(0.0, min(10.0, score)), 2)


def score__patents__a72(text):
    t = (text or "")[:300000]
    low = t.lower()
    if not low.strip():
        return 0.0
    components = len(re.findall(r"\b(?:comprises?|comprising|coupled to|configured to|module|processor|sensor|housing|layer|substrate|circuit|valve|chamber|shaft|memory|controller|electrode|element)\b", low))
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:nm|mm|cm|m|kg|g|mg|%|°c|v|a|hz|mhz|ghz|seconds?|minutes?|degrees?)\b", low))
    algorithms = len(re.findall(r"\b(?:algorithm|step of|calculat\w*|iterat\w*|threshold|formula|equation|sequence|protocol)\b", low))
    formulations = len(re.findall(r"\b(?:wt\.?\s*%|molar|compound|composition|mixture|formula|ratio of|concentration)\b", low))
    vague = len(re.findall(r"\b(?:may include|in some embodiments|generally|desired result|various means|suitable|optionally)\b", low))
    concrete = components + 2 * numbers + algorithms + formulations
    if concrete >= 12 and (numbers + algorithms + formulations >= 2):
        return 10.0
    if concrete >= 4:
        return 5.0 if concrete < 9 or vague > concrete / 2 else 8.0
    return 0.0


def score__press_releases__a291(text):
    t = (text or "")[:300000]
    low = t.lower()
    if not low.strip():
        return 0.0
    first_person = len(re.findall(r"\b(?:i|i'm|i've|my|mine|me)\b", low))
    investing = len(re.findall(r"\b(?:invest\w*|portfolio|trading|holding\w*|asset allocation|passive income|dividend|stock\w*|bond\w*|etf|shares?|position|yield|retirement account)\b", low))
    analysis = len(re.findall(r"\b(?:analysis|strategy|opinion|believe|think|my goal|my approach|I own|I hold|risk tolerance)\b", t, re.I))
    corporate = len(re.findall(r"\b(?:press release|for immediate release|today announced|news provided by|media contact|the company announced)\b", low))
    byline = bool(re.search(r"(?im)^\s*(?:by|author:)\s+[A-Z][A-Za-z .'-]{2,}$", t))
    if first_person >= 4 and investing >= 4 and analysis >= 1 and (byline or first_person >= 7) and corporate == 0:
        return 10.0
    return 0.0


JOB_IDS = [
    "math__a24",
    "press_releases__a204",
    "patents__a216",
    "math__a72",
    "press_releases__a110",
    "math__a84",
    "press_releases__a67",
    "math__a12",
    "patents__a204",
    "press_releases__a101",
    "code_review__a144",
    "code_review__a279",
    "math__a36",
    "code_review__a54",
    "math__a96",
    "patents__a54",
    "press_releases__a28",
    "math__a78",
    "press_releases__a146",
    "patents__a72",
    "press_releases__a291",
]
