# AUTO: blind rule compilation chunk c6
import re
import math


def score__press_releases__a117(text):
    text = text or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    first = lines[0]
    headline = len(first.split()) >= 3 and len(first) <= 180 and not first.endswith(".")
    date_words = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    date_pat = rf"(?:{date_words}\s+\d{{1,2}}(?:,\s*\d{{4}})?|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}})"
    dateline = bool(re.search(rf"(?im)^(?:[A-Z][A-Z .'-]{{2,40}},?\s*(?:--|—|-)?\s*)?{date_pat}\b", text))
    dateline = dateline or bool(re.search(rf"(?im)^[A-Z][A-Z .'-]{{2,40}},?\s*(?:--|—|-)\s*(?:{date_words}|\d{{4}}-)", text))
    corporate = len(re.findall(r"\b(?:company|corporation|corp\.?|inc\.?|ltd\.?|plc|business|investors?|shareholders?)\b", text, re.I))
    news = len(re.findall(r"\b(?:announc(?:e[sd]?|ement)|report(?:s|ed)?|appoint(?:s|ed|ment)|acquir(?:e[sd]?|es|ing|ition)|launch(?:es|ed)?|results?|revenue|earnings|merger|agreement|award(?:ed)?|quarter|fiscal)\b", text, re.I))
    release_markers = len(re.findall(r"\b(?:press release|news release|for immediate release|media contact|investor relations)\b", text, re.I))
    body_words = len(re.findall(r"\b\w+\b", text))
    nav = len(re.findall(r"\b(?:home|menu|shop|cart|login|sign up|privacy|cookie|site map|products)\b", text, re.I))
    if headline and dateline and news >= 1 and corporate >= 1 and body_words >= 60:
        return float(min(10, 8.5 + min(1.0, news * 0.2) + min(0.5, release_markers * 0.25)))
    if news and (headline or dateline or release_markers or corporate):
        return float(min(7, 4.5 + 0.7 * headline + 0.8 * dateline + min(1.0, news * 0.2)))
    if body_words and corporate + news:
        return float(max(1, min(4, 1.5 + 0.4 * (corporate + news))))
    return float(max(0, min(2, 1.0 if body_words > 30 and nav < 4 else 0.0)))


def score__math__a102(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    lower = text.lower()
    proof_terms = len(re.findall(r"\b(?:proof|prove|suppose|assume|therefore|thus|hence|because|implies|lemma|theorem|contradiction|induction|consequently|qed|wlog)\b", lower))
    step_terms = len(re.findall(r"\b(?:first|second|next|then|finally|case\s+\d+|base case|inductive step)\b", lower))
    math_lines = len(re.findall(r"(?m)^(?=.*(?:=|≤|≥|<|>|\^|\\|∑|∫)).{3,}$", text))
    equations = len(re.findall(r"(?:=|≤|≥|\b(?:equals|converges|diverges)\b)", text, re.I))
    asks_help = bool(re.search(r"\b(?:any hints?|help me|how (?:do|can|would) (?:i|you)|stuck)\b", lower))
    final_only = len(words) < 80 and proof_terms < 2
    if asks_help and proof_terms < 2:
        return 1.0 if equations else 0.0
    depth = proof_terms * 0.65 + step_terms * 0.4 + min(3, equations * 0.2) + min(2, math_lines * 0.25)
    length_support = min(2.5, len(words) / 180.0)
    score = depth + length_support
    if final_only:
        score = min(score, 2.0)
    elif proof_terms < 3:
        score = min(score, 5.0)
    elif len(words) < 140:
        score = min(score, 6.5)
    return float(max(0, min(10, round(score, 1))))


def score__math__a126(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    n = len(words)
    if n < 35:
        return 0.0
    equations = len(re.findall(r"(?:\$[^$]{2,}\$|\\\(|\\\[|(?m)^\s*[^\n]{0,80}(?:=|≤|≥|∑|∫|→)[^\n]{1,80}$)", text))
    technical = len(re.findall(r"\b(?:theorem|lemma|proof|definition|algorithm|complexity|induction|derivative|integral|matrix|function|probability|convergence|implementation|runtime|invariant|equation)\b", text, re.I))
    paragraphs = len([p for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 12])
    length_score = 10.0 * (1.0 - math.exp(-max(0, n - 25) / 850.0))
    density_bonus = min(1.5, equations * 0.12 + technical * 0.04 + paragraphs * 0.03)
    score = length_score + density_bonus
    if n < 180:
        score = min(score, 3.5)
    elif n < 400:
        score = min(score, 6.0)
    elif n < 800:
        score = min(score, 8.0)
    if equations + technical < 2:
        score *= 0.55
    return float(max(0, min(10, round(score, 1))))


def score__patents__a30(text):
    text = text or ""
    if not text.strip():
        return 0.0
    computing = len(re.findall(r"\b(?:software|computer|computing|processor|database|data|memory|storage|index(?:ing)?|algorithm|server|client|network|digital|protocol|packet|schedule|scheduling|controller|program|instruction|virtual|cloud|encryption|file system|operating system)\b", text, re.I))
    architecture = len(re.findall(r"\b(?:architecture|data management|data processing|distributed|parallel processing|memory management|communication protocol|control protocol|software-defined|application programming interface|database management)\b", text, re.I))
    physical = len(re.findall(r"\b(?:mechanical|shaft|gear|vehicle|agricultur|chemical|compound|composition|molecule|antenna|radio frequency|sensor|motor|engine|valve|circuit|voltage|optical|medical|biological)\w*\b", text, re.I))
    if computing == 0:
        return 0.0
    ratio = computing / max(1, computing + physical)
    score = 1.0 + 7.0 * ratio + min(2.0, architecture * 0.45)
    if architecture >= 2 and ratio >= 0.6:
        score = max(score, 9.0)
    elif computing >= 5 and ratio >= 0.5:
        score = max(score, 7.0)
    elif physical > computing * 2:
        score = min(score, 5.0)
    return float(max(0, min(10, round(score, 1))))


def score__math__a156(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    advanced = len(re.findall(r"\b(?:topolog|manifold|measure theor|functional analysis|category theor|homolog|cohomolog|algebraic geometry|spectral|hilbert|banach|sigma-algebra|axiom|formal logic|model theory|galois|lie algebra|stochastic process|distribution theory|compactness|isomorphism|eigenvalue|convexity|variational)\w*\b", text, re.I))
    rigor = len(re.findall(r"\b(?:definition|theorem|lemma|proposition|corollary|proof|suppose|assume|therefore|thus|hence|contradiction|necessary|sufficient|if and only if|qed)\b", text, re.I))
    elementary = len(re.findall(r"\b(?:simplify|calculate|plug in|quadratic formula|basic algebra|arithmetic|differentiate|homework|solve for [a-z]|factor(?:ize)?|multiplication|addition)\b", text, re.I))
    conceptual = len(re.findall(r"\b(?:intuition|interpretation|structure|fundamental|invariant|generaliz|abstract|conceptual|underlying|insight|equivalent)\w*\b", text, re.I))
    score = min(4.5, advanced * 0.75) + min(3.5, rigor * 0.3) + min(2.0, conceptual * 0.35) + min(1.0, len(words) / 500.0)
    if advanced == 0:
        score = min(score, 4.0 if rigor + conceptual >= 5 else 2.0)
    if elementary > advanced * 2:
        score = min(score, 3.0)
    if len(words) < 100:
        score = min(score, 5.0)
    if advanced >= 4 and rigor >= 6 and len(words) >= 250:
        score = max(score, 8.0)
    return float(max(0, min(10, round(score, 1))))


def score__CAL__CAL1(text):
    count = len(re.findall(r"[0-9]", text or ""))
    if count >= 3:
        return 10.0
    if count >= 1:
        return 5.0
    return 0.0


def score__press_releases__a65(text):
    text = text or ""
    if not text.strip():
        return 0.0
    date_words = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    date = rf"(?:{date_words}\s+\d{{1,2}}(?:,\s*\d{{4}})?|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}})"
    dateline = bool(re.search(rf"(?im)^(?:[A-Z][A-Z .'-]{{2,40}},?\s*(?:--|—|-)\s*)?{date}\b", text))
    dateline = dateline or bool(re.search(rf"(?im)^[A-Z][A-Z .'-]{{2,40}},?\s*(?:--|—|-)\s*(?:{date_words}|\d{{4}})", text))
    news = bool(re.search(r"\b(?:announce|announces|announced|acquisition|acquires|financial results|earnings|appoints|appointment|merger|quarterly results|fiscal year|enters? (?:an? )?agreement|launches)\b", text, re.I))
    corporate = bool(re.search(r"\b(?:company|corporation|corp\.?|inc\.?|ltd\.?|plc|investor|shareholder|revenue|net income|chief executive|board of directors)\b", text, re.I))
    formal = bool(re.search(r"\b(?:press release|news release|for immediate release|media contact|investor relations)\b", text, re.I))
    if dateline and news and corporate:
        return 10.0
    business_content = len(re.findall(r"\b(?:business|financial|finance|market|sales|revenue|income|investor|stock|shares|acquisition|executive|company)\b", text, re.I))
    web_portal = bool(re.search(r"\b(?:investor portal|investor relations|navigation|sign in|subscribe|read more|website)\b", text, re.I))
    if corporate and (news or business_content >= 3 or formal or web_portal):
        return 4.0
    return 0.0


def score__code_review__a108(text):
    text = text or ""
    if not text.strip():
        return 0.0
    reply_markers = len(re.findall(r"(?im)(?:^|\n)\s*(?:reply|author|reviewer|maintainer|developer|@[\w-]+)\s*:|\b(?:replied|responded)\b", text))
    exchange = len(re.findall(r"\b(?:agree|agreed|good point|makes sense|you're right|you are right|thanks|resolved|clarif|what about|instead|because|however|could we|why|how about)\w*\b", text, re.I))
    technical = len(re.findall(r"\b(?:architecture|algorithm|implementation|performance|complexity|thread|race condition|memory|database|api|interface|function|class|method|test|exception|cache|dependency|refactor|security|type|schema)\b", text, re.I))
    speakers = set(re.findall(r"(?im)^\s*([A-Za-z][\w.-]{1,30}|@[\w-]+)\s*:", text))
    turns = len(re.findall(r"(?m)^\s*(?:[A-Za-z][\w.-]{1,30}|@[\w-]+)\s*:", text))
    if reply_markers == 0 and len(speakers) < 2:
        return 0.0
    if turns >= 6 and len(speakers) >= 2 and exchange >= 4 and technical >= 4:
        return 10.0
    if (turns >= 3 or reply_markers >= 2) and exchange >= 2 and technical >= 3:
        return 7.5
    if turns >= 2 or reply_markers >= 1:
        return 5.0 if exchange + technical >= 3 else 2.5
    return 0.0


def score__press_releases__a97(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    amounts = len(re.findall(r"(?:[$€£¥]\s?\d[\d,.]*(?:\s?(?:million|billion|trillion|mn|bn))?|\b\d[\d,.]*\s?(?:dollars?|euros?|pounds?|million|billion|trillion|mn|bn)\b|\b\d+(?:\.\d+)?\s?%)", text, re.I))
    finance_terms = len(re.findall(r"\b(?:EPS|earnings per share|revenue|net income|operating income|AUM|assets under management|buyback|repurchase|cash flow|EBITDA|gross margin|sales|profit|loss|dividend|fiscal|quarter)\b", text, re.I))
    numeric_results = len(re.findall(r"\b(?:increased?|decreased?|grew|declined|rose|fell|totaled|reached|reported)\b[^.!?\n]{0,60}\d", text, re.I))
    explicit = amounts + min(finance_terms, amounts + numeric_results) + numeric_results
    density = 1000.0 * explicit / max(80, len(words))
    if amounts >= 8 and finance_terms >= 4 and density >= 12:
        return 10.0
    if explicit >= 5 and finance_terms >= 2:
        return float(min(7, 5.5 + min(1.5, density / 12.0)))
    if explicit >= 1:
        return float(min(4, 1.5 + explicit * 0.6))
    return 0.0


def score__code_review__a198(text):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    title = lines[0] if lines else ""
    if re.search(r"\b(?:improvement|improvements|improvment|improvments)\b", title, re.I):
        return 10.0
    issue = re.search(r"\b(?:fixe?s?|fixed|closes?|closed|resolves?|issue)\s*:?#?\s*\d+\b", title, re.I)
    trailing = re.search(r"(?:#\s*\d+|\bissue\s+\d+|\(\s*#?\d+\s*\))\s*$", title, re.I)
    return 5.0 if issue or trailing else 0.0


def score__patents__a6(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tech = len(re.findall(r"\b(?:computer|computing|software|hardware|processor|microprocessor|memory|database|data processing|digital|electronic|electronics|sensor|controller|image processing|signal processing|circuit|network|server|algorithm|program instructions|semiconductor)\b", text, re.I))
    physical = len(re.findall(r"\b(?:mechanical|chemical|composition|compound|polymer|molecule|biological|protein|shaft|gear|valve|housing|agricultur|fluid|alloy|pharmaceutical|material)\w*\b", text, re.I))
    automation = len(re.findall(r"\b(?:automated|controller|electronically|control signal|data handling|processor-controlled|feedback control)\b", text, re.I))
    if tech == 0:
        return 0.0
    ratio = tech / max(1, tech + physical)
    score = 1.0 + 8.0 * ratio + min(1.0, math.log1p(tech) / 3.0)
    if tech >= 5 and ratio >= 0.65:
        score = max(score, 9.0)
    elif physical >= tech and automation:
        score = max(4.0, min(6.0, score))
    elif physical > tech and not automation:
        score = min(score, 4.0)
    return float(max(0, min(10, round(score, 1))))


def score__press_releases__a0(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    core = len(re.findall(r"\b(?:ethics|ethical|compliance|comply|code of conduct|legal standards?|anti-corruption|anti-bribery|bribery|corruption|whistleblower|regulatory compliance|business conduct|integrity|conflict of interest|fiduciary|governance)\b", text, re.I))
    social = len(re.findall(r"\b(?:corporate social responsibility|responsible business|human rights|sustainability|accountability|fair dealing|workplace conduct)\b", text, re.I))
    legal = len(re.findall(r"\b(?:regulation|regulatory|law|legal|disclosure|policy|policies|standards?|prohibited|requirement)\b", text, re.I))
    if core + social + legal == 0:
        return 0.0
    density = 1000.0 * (core * 2 + social * 1.5 + legal * 0.5) / len(words)
    dedicated = core >= 6 and density >= 18
    if dedicated and re.search(r"\b(?:code of (?:ethics|conduct)|anti-corruption|anti-bribery|compliance program|ethics policy)\b", text, re.I):
        return float(min(10, 8.5 + min(1.5, density / 30.0)))
    if core + social >= 3 and density >= 8:
        return float(min(8, 5.5 + min(2.5, density / 12.0)))
    return float(min(5, 2.5 + min(2.5, (core + social + legal) * 0.4)))


def score__patents__a134(text):
    text = text or ""
    if not text.strip():
        return 0.0
    core = len(re.findall(r"\b(?:algorithm|software|machine learning|neural network|parallel processing|data processing|database|memory operation|user interface|software-defined networking|program instructions|operating system|compiler|data structure|virtual machine|computer science|encryption)\b", text, re.I))
    general = len(re.findall(r"\b(?:computer|processor|memory|electronic|electronics|circuit|telecommunication|network|transmitter|receiver|digital|semiconductor|server|device controller)\b", text, re.I))
    physical = len(re.findall(r"\b(?:mechanical|manufactur|agricultur|chemical|chemistry|biolog|compound|composition|molecule|shaft|gear|valve|vehicle|alloy|polymer|pharmaceutical|fluid)\w*\b", text, re.I))
    incidental = core == 0 and general <= 2 and physical >= 2
    if core == 0 and general == 0:
        return 0.0 if physical == 0 else 1.0
    if incidental:
        return float(min(3, 1 + general))
    if core >= 3 and core >= physical * 0.5:
        return float(min(10, 8 + min(2, core / 5.0)))
    if core >= 1 and core + general >= physical:
        return float(min(8, 6 + min(2, core / 2.0)))
    if general >= 3 and general >= physical * 0.5:
        return float(min(7, 4 + general * 0.3))
    return float(min(3, 1 + (core + general) * 0.3))


def score__patents__a84(text):
    text = text or ""
    words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text)
    if not words:
        return 0.0
    sentences = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", text) if re.search(r"[A-Za-z]", s)]
    avg_sentence = len(words) / max(1, len(sentences))
    long_words = sum(len(w) >= 12 for w in words) / len(words)
    jargon = len(re.findall(r"\b(?:wherein|thereof|therein|aforementioned|configured to|plurality|substrate|semiconductor|electromagnetic|transceiver|actuator|circumferential|longitudinal|microprocessor|photolithograph|heterogeneous)\w*\b", text, re.I)) / len(words)
    nested = sum(max(s.count(","), s.count(";")) >= 4 for s in sentences) / max(1, len(sentences))
    score = 10.0
    score -= max(0, avg_sentence - 14) * 0.11
    score -= long_words * 12.0
    score -= jargon * 45.0
    score -= nested * 2.0
    if avg_sentence >= 35 or jargon >= 0.05:
        score = min(score, 3.0)
    elif avg_sentence >= 25 or jargon >= 0.025:
        score = min(score, 6.0)
    return float(max(0, min(10, round(score, 1))))


def score__press_releases__a87(text):
    text = text or ""
    if not text.strip():
        return 0.0
    emails = re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I)
    phones = re.findall(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]\d{4}(?!\d)", text)
    addresses = re.findall(r"(?im)\b\d{1,6}\s+[A-Za-z0-9.' -]{2,50}\s+(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Lane|Ln\.?|Drive|Dr\.?|Way|Suite|Floor)\b[^\n,]*", text)
    name_lines = re.findall(r"(?im)^(?:contact\s*:\s*)?(?:[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\s+){1,3}[A-Z][a-z]+(?:\s*[,|]\s*[^\n]+)?$", text)
    details = len(emails) + len(phones) + len(addresses)
    pairing = 0
    for line in text.splitlines():
        if (re.search(r"@|\d{3}[ .)-]\d{3}[ .-]\d{4}", line) and
                re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", line)):
            pairing += 1
    boilerplate = bool(re.search(r"(?i)^(?:media|press|investor) contact", text.strip())) or bool(re.search(r"(?im)^contacts?\s*:", text))
    if details >= 3 and (pairing >= 1 or len(name_lines) >= 2):
        return 10.0
    if details >= 1 or name_lines:
        return 5.0
    if boilerplate:
        return 5.0
    return 0.0


def score__math__a42(text):
    text = text or ""
    if not text.strip():
        return 0.0
    lower = text.lower()
    explicit_error = bool(re.search(r"\b(?:this is (?:false|incorrect|wrong)|counterexample|contradiction|logical fallacy|invalid proof|does not follow|disproves?|cannot be true|calculation error|mistakenly)\b", lower))
    bad_arithmetic = False
    for match in re.finditer(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*([+*×-])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)(?![\w.])", text):
        a, op, b, claimed = match.groups()
        a, b, claimed = float(a), float(b), float(claimed)
        actual = a + b if op == "+" else a - b if op == "-" else a * b
        if abs(actual - claimed) > 1e-9 * max(1.0, abs(actual), abs(claimed)):
            bad_arithmetic = True
            break
    bad_division = False
    for match in re.finditer(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(?:/|÷)\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)(?![\w.])", text):
        a, b, claimed = map(float, match.groups())
        if b == 0 or abs(a / b - claimed) > 1e-9 * max(1.0, abs(a / b), abs(claimed)):
            bad_division = True
            break
    false_constants = bool(re.search(r"\b(?:pi\s*=\s*3(?!\.)|sqrt\s*\(?\s*4\s*\)?\s*=\s*-2|1\s*/\s*0\s*=|0\s*=\s*1\b)", lower))
    return 10.0 if explicit_error or bad_arithmetic or bad_division or false_constants else 0.0


def score__code_review__a36(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    comments = [c.strip() for c in re.split(r"\n\s*\n|(?m)^(?:[-*]|\d+[.)])\s+", text) if c.strip()]
    trivial = len(re.findall(r"\b(?:remove|typo|whitespace|indentation|empty line|newline|rename|nit|nits|done|ok|okay|lgtm|fix this|add a comma)\b", text, re.I))
    technical = len(re.findall(r"\b(?:architecture|design|algorithm|complexity|performance|concurrency|race condition|memory leak|api|interface|abstraction|dependency|security|transaction|database|cache|invariant|edge case|error handling|test coverage|maintainability|coupling|refactor)\b", text, re.I))
    reasoning = len(re.findall(r"\b(?:because|therefore|otherwise|which means|so that|in order to|consider|instead|alternative|trade-?off|this (?:would|will|can)|for example|why)\b", text, re.I))
    long_comments = sum(len(re.findall(r"\b\w+\b", c)) >= 25 for c in comments)
    substantive = technical + reasoning + long_comments * 2
    if substantive == 0:
        return float(max(0, min(3, 1.0 if trivial else 2.0)))
    ratio = substantive / max(1, substantive + trivial)
    score = 2.0 + ratio * 6.0 + min(2.0, (technical + long_comments) * 0.25)
    if technical >= 5 and reasoning >= 3 and (long_comments >= 2 or len(words) >= 180):
        score = max(score, 8.0)
    if trivial > substantive * 2:
        score = min(score, 4.0)
    return float(max(0, min(10, round(score, 1))))


def score__patents__a240(text):
    text = text or ""
    claims_heading = re.search(r"(?im)^\s*CLAIMS\s*:\s*$", text)
    if not claims_heading:
        return 0.0
    after = text[claims_heading.end():]
    numbered_claim = re.search(r"(?m)^\s*1\.\s+(?:A|An|The|What is claimed|A method|A system|A device)", after, re.I)
    claim_language = re.search(r"\b(?:comprising|wherein|according to claim|claim 1|the method of claim|the system of claim)\b", after, re.I)
    return 10.0 if numbered_claim and claim_language else 0.0


def score__patents__a60(text):
    text = text or ""
    if not text.strip():
        return 0.0
    ui = len(re.findall(r"\b(?:user interface|graphical user interface|GUI|touchscreen|touch screen|display|displaying|screen|menu|button|icon|window|dialog|user input|interactive|interaction|gesture|cursor|visualization|human-machine interface|user-facing)\b", text, re.I))
    core_ui = len(re.findall(r"\b(?:graphical user interface|user interface|touchscreen|interactive display|displaying (?:to|for) (?:a )?user|receiving user input|user interaction|human-machine interface)\b", text, re.I))
    software = len(re.findall(r"\b(?:software|computer|processor|application|program|data|image processing|printing|server|network|memory|algorithm)\b", text, re.I))
    other = len(re.findall(r"\b(?:chemical|compound|mechanical|shaft|gear|agricultur|molecule|antenna|radio frequency|engine|valve|polymer)\w*\b", text, re.I))
    if ui == 0:
        return 0.0
    relevance = ui / max(1, ui + software + other)
    if core_ui >= 3 and relevance >= 0.25:
        return float(min(10, 8.5 + min(1.5, core_ui * 0.3)))
    if ui >= 2 and software >= 2:
        return float(min(8, 5.5 + min(2.5, ui * 0.4)))
    return float(min(6.5, 2.0 + ui * 0.8 + core_ui * 0.6))


def score__math__a132(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    geometric = len(re.findall(r"\b(?:geometr|visuali[sz]|diagram|figure|shape|circle|triangle|polygon|sphere|surface|curve|angle|distance|area|volume|coordinate|vector|dimension|rotation|symmetry|projection|intersection|parallel|perpendicular|manifold|topolog|spatial)\w*\b", text, re.I))
    intuition = len(re.findall(r"\b(?:intuition|intuitive|visual meaning|geometric meaning|picture|imagine|viewed as|interpret(?:ation)?|corresponds to|represents)\b", text, re.I))
    algebraic = len(re.findall(r"\b(?:algebra|equation|calculate|compute|simplify|formula|symbolic|formal logic|truth table|differentiate|integrate)\w*\b", text, re.I))
    if geometric == 0 and intuition == 0:
        return 0.0
    density = (geometric + intuition * 1.5) / max(1, len(words) / 100.0)
    if geometric >= 8 and intuition >= 3 and density >= 4 and geometric + intuition >= algebraic:
        return 10.0
    if geometric >= 4 and (intuition >= 1 or density >= 4) and geometric + intuition >= algebraic * 0.5:
        return float(min(7, 5.5 + min(1.5, (geometric + intuition) / 8.0)))
    return float(min(4, 2.5 + min(1.5, (geometric + intuition) * 0.25)))


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
