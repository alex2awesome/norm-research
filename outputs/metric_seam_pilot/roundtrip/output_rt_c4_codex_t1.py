# AUTO: blind rule compilation chunk c4
import re


def _clean(text):
    return text if isinstance(text, str) else "" if text is None else str(text)


def _words(text):
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", _clean(text))


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", _clean(text)) if s.strip()]


def _count(text, patterns):
    low = _clean(text).lower()
    return sum(len(re.findall(p, low)) for p in patterns)


def _bounded(value):
    return float(max(0.0, min(10.0, value)))


def _answer_part(text):
    text = _clean(text)
    matches = list(re.finditer(r"(?im)^\s*(?:answer|solution|response)\s*:?[ \t]*$", text))
    if matches:
        return text[matches[-1].end():]
    return text


def _question_part(text):
    text = _clean(text)
    marker = re.search(r"(?im)^\s*(?:answer|solution|response)\s*:?[ \t]*$", text)
    return text[:marker.start()] if marker else text


def _math_profile(text):
    text = _clean(text)
    wc = max(1, len(_words(text)))
    equations = len(re.findall(r"(?:[$=<>]|\\(?:frac|sum|int|lim|begin|forall|exists)|\b\w+\s*[+*/^]\s*\w+)", text))
    rigor = _count(text, [
        r"\bproof\b", r"\btherefore\b", r"\bthus\b", r"\bhence\b",
        r"\bassum(?:e|ing)\b", r"\bimplies\b", r"\bif and only if\b",
        r"\blet\b", r"\bcase\s+\d", r"\bcontradiction\b", r"\bq\.?e\.?d\b"
    ])
    completion = _count(text, [
        r"\btherefore\b", r"\bthus\b", r"\bhence\b", r"\bconsequently\b",
        r"\bthe answer is\b", r"\bwe conclude\b", r"\bq\.?e\.?d\b"
    ])
    gaps = _count(text, [
        r"\bnot sure\b", r"\bmaybe\b", r"\bprobably\b", r"\bi think\b",
        r"\bsketch\b", r"\bhint\b", r"\bwithout proof\b", r"\bleft to (?:the )?reader\b"
    ])
    return wc, equations, rigor, completion, gaps


def _math_answer_quality(text):
    answer = _answer_part(text)
    wc, equations, rigor, completion, gaps = _math_profile(answer)
    if not answer.strip():
        return 0.0
    score = 1.0
    score += min(2.0, wc / 90.0)
    score += min(2.0, equations * 0.35)
    score += min(3.0, rigor * 0.42)
    score += min(2.0, completion * 0.65)
    score -= min(2.5, gaps * 0.7)
    if wc < 20:
        score = min(score, 3.0)
    elif wc < 55:
        score = min(score, 6.5)
    return _bounded(score)


def score__math__a24(text):
    """Proxy correctness by coherent derivation, rigor, explicit conclusions, and gaps."""
    answer = _answer_part(text)
    score = _math_answer_quality(answer)
    contradictions = _count(answer, [
        r"\bthis is (?:false|incorrect|wrong)\b", r"\bdoes not follow\b",
        r"\bcontradict(?:s|ion)\b.*\b(?:my|our)\b", r"\berror\b"
    ])
    return _bounded(score - min(2.5, contradictions * 0.8))


def score__press_releases__a204(text):
    text = _clean(text)
    wc = max(1, len(_words(text)))
    nav = _count(text, [
        r"\bsign in\b", r"\blog ?in\b", r"\bsearch\b", r"\bmenu\b",
        r"\bregister\b", r"\bsubscribe\b", r"\bcontact(?: us)?\b", r"\bsite ?map\b",
        r"\bprivacy\b", r"\bterms(?: of use)?\b", r"\bsettings\b", r"\bdashboard\b",
        r"\bmy account\b", r"\bclick here\b", r"\bread more\b", r"https?://|www\."
    ])
    ui = _count(text, [r"\bsign in\b", r"\blog ?in\b", r"\bsearch\b", r"\bform\b", r"\bsettings\b", r"\bdashboard\b", r"\bpassword\b", r"\busername\b"])
    long_prose = sum(1 for s in _sentences(text) if len(_words(s)) >= 18)
    density = nav * 100.0 / wc
    if nav >= 4 and ui >= 2 and density >= 3.0:
        return _bounded(7.0 + min(3.0, (density - 3.0) * 0.45 + ui * 0.25))
    if nav >= 5 and long_prose == 0:
        return 0.0
    if nav == 0:
        return 1.0 if long_prose else 0.0
    prose_share = long_prose / max(1, len(_sentences(text)))
    return _bounded(2.0 + min(4.5, density * 0.7 + nav * 0.15) - prose_share * 1.5)


def score__patents__a216(text):
    text = _clean(text)
    high = _count(text, [
        r"\bconsumer (?:good|product)s?\b", r"\bsport(?:ing)?\b", r"\brecreation(?:al)?\b",
        r"\bhousehold\b", r"\bappliance\b", r"\bfixture\b", r"\bfurniture\b",
        r"\bkitchen\b", r"\btoy\b", r"\bgarment\b", r"\bshoe\b", r"\bexercise\b"
    ])
    middle = _count(text, [
        r"\bvehicle\b", r"\bautomotive\b", r"\bengine\b", r"\bwireless\b",
        r"\btelecommunicat", r"\bnetwork\b", r"\bfuel\b", r"\benergy\b",
        r"\bbattery\b", r"\bpower generation\b"
    ])
    other = _count(text, [
        r"\bsoftware\b", r"\bcomputer\b", r"\bsemiconductor\b", r"\belectronic\b",
        r"\bchemical\b", r"\bpolymer\b", r"\bmanufactur", r"\bmedical\b",
        r"\bsurgical\b", r"\bdiagnos", r"\bpharmaceutical\b"
    ])
    if high > max(middle, other):
        return 10.0
    if middle > max(high, other):
        return 5.0
    if high and high == middle and high > other:
        return 7.5
    if middle and middle == other and middle > high:
        return 2.5
    return 0.0


def score__math__a72(text):
    answer = _answer_part(text)
    score = _math_answer_quality(answer)
    wc, equations, rigor, completion, gaps = _math_profile(answer)
    if equations == 0 and rigor == 0:
        score = min(score, 3.0)
    if gaps or (wc < 100 and completion == 0):
        score = min(score, 7.0)
    return _bounded(score)


def score__press_releases__a110(text):
    text = _clean(text)
    dateline = bool(re.search(r"(?im)^(?:[A-Z][A-Z .'-]+(?:,\s*[A-Z]{2})?\s*[-—]|[A-Z][A-Za-z .'-]+,\s*(?:[A-Z][a-z]+\s+\d{1,2}|[A-Z]{2})[^\n]{0,30}[-—])", text))
    announcement = _count(text, [
        r"\bannounc(?:e|es|ed|ement)\b", r"\breports? (?:results|revenue|earnings)\b",
        r"\bacquir(?:e|es|ed|quisition)\b", r"\bmerger\b", r"\blaunch(?:es|ed)?\b",
        r"\bappoint(?:s|ed|ment)\b", r"\bintroduc(?:es|ed)\b"
    ])
    company = _count(text, [r"\b(?:inc\.?|corp\.?|corporation|company|ltd\.?|plc|llc)\b", r"\bceo\b", r"\bchief executive\b"])
    press = _count(text, [r"\bfor immediate release\b", r"\bnews provided by\b", r"\bmedia contact\b", r"\babout [A-Z]"])
    nav = _count(text, [r"\bsign in\b", r"\bsearch\b", r"\bsite ?map\b", r"\bhome\b", r"\bservices\b"])
    foreign = len(re.findall(r"[^\x00-\x7F]", text)) > max(20, len(text) * 0.15)
    if foreign or not text.strip():
        return 0.0
    if dateline and announcement and company:
        return _bounded(8.5 + min(1.5, press * 0.5 + announcement * 0.15))
    if announcement and company:
        return _bounded(6.5 + min(1.5, press * 0.4))
    if company or (nav >= 3 and len(_words(text)) > 40):
        return 5.5
    return 0.5 if nav or text.strip() else 0.0


def score__math__a84(text):
    text = _clean(text)
    advanced = _count(text, [
        r"\btopolog", r"\bdifferential geometr", r"\babstract algebra\b", r"\bgroup theory\b",
        r"\bring\b", r"\bfield extension\b", r"\bcomplex analys", r"\bmeasure theor",
        r"\bfunctional analys", r"\bmanifold\b", r"\bhomolog", r"\bstochastic process",
        r"\bmartingale\b", r"\bbanach\b", r"\bhilbert\b", r"\blebesgue\b"
    ])
    moderate = _count(text, [
        r"\bcalculus\b", r"\bderivative\b", r"\bintegral\b", r"\blinear algebra\b",
        r"\bmatrix\b", r"\beigenvalue\b", r"\bdifferential equation\b", r"\bsequence\b",
        r"\bseries\b", r"\bproof\b", r"\bprobability\b"
    ])
    elementary = _count(text, [r"\barithmetic\b", r"\bpercentage\b", r"\btriangle\b", r"\bquadratic\b", r"\bcombinatorial puzzle\b", r"\balgorithm\b", r"\bprogramming\b"])
    if advanced:
        return _bounded(7.0 + min(3.0, advanced * 0.55 + moderate * 0.08))
    if moderate:
        return _bounded(3.2 + min(3.7, moderate * 0.45))
    if elementary or re.search(r"\d+\s*[-+*/]\s*\d+", text):
        return _bounded(1.0 + min(1.8, elementary * 0.3))
    return 0.0


def score__press_releases__a67(text):
    text = _clean(text)
    terms = [r"\banimals?\b", r"\banimal welfare\b", r"\banimal rights?\b", r"\bhumane\b", r"\bslaughter", r"\babattoir", r"\blivestock\b", r"\bcruelty\b", r"\bpets?\b", r"\bsupply chain\b"]
    total = _count(text, terms)
    if total == 0:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    heading_text = "\n".join(lines[:3])
    opening = " ".join(_words(text)[:100])
    prominent = _count(heading_text + "\n" + opening, terms)
    wc = max(1, len(_words(text)))
    density = total * 100.0 / wc
    welfare_specific = _count(text, [r"\banimal welfare\b", r"\banimal rights?\b", r"\bhumane\b", r"\bcruelty\b", r"\bslaughter", r"\babattoir"])
    if total >= 6 and prominent >= 2 and density >= 1.0:
        return _bounded(8.5 + min(1.5, density * 0.3 + welfare_specific * 0.08))
    if welfare_specific >= 2 or prominent >= 2:
        return _bounded(6.0 + min(2.0, welfare_specific * 0.3 + density * 0.15))
    return 2.0 if total >= 2 else 1.0


def score__math__a12(text):
    text = _clean(text)
    advanced = _count(text, [
        r"\babstract algebra\b", r"\bfunctional analys", r"\btopolog", r"\bmeasure theor",
        r"\bdifferential geometr", r"\bmanifold\b", r"\bbanach\b", r"\bhilbert\b",
        r"\blebesgue\b", r"\bhomomorph", r"\bcompactness\b", r"\bsigma-algebra\b"
    ])
    undergraduate = _count(text, [
        r"\bmultivariable calculus\b", r"\blinear algebra\b", r"\breal analys",
        r"\bdifferential equation", r"\beigenvalue\b", r"\bgradient\b", r"\bintegral\b",
        r"\bconvergence\b", r"\bvector space\b"
    ])
    _, equations, rigor, _, _ = _math_profile(text)
    if advanced:
        return _bounded(8.0 + min(2.0, advanced * 0.35 + rigor * 0.08))
    if undergraduate:
        return _bounded(4.0 + min(3.0, undergraduate * 0.35 + rigor * 0.12))
    if rigor >= 4 and equations >= 2:
        return min(7.0, 4.0 + rigor * 0.2)
    if equations or re.search(r"\b(?:algebra|geometry|arithmetic|equation)\b", text, re.I):
        return _bounded(1.0 + min(2.0, equations * 0.2 + rigor * 0.12))
    return 0.0


def score__patents__a204(text):
    text = _clean(text)
    heading = re.search(r"(?im)^\s*ABSTRACT\s*:?[ \t]*$", text)
    if not heading:
        return 0.0
    rest = text[heading.end():]
    next_heading = re.search(r"(?m)^\s*(?:CLAIMS?|DESCRIPTION|BACKGROUND|SUMMARY|FIELD(?: OF THE INVENTION)?|DETAILED DESCRIPTION)\s*:?[ \t]*$", rest, re.I)
    abstract = rest[:next_heading.start()] if next_heading else rest
    return 10.0 if re.search(r"\bThe present invention\b", abstract) else 0.0


def score__press_releases__a101(text):
    text = _clean(text)
    dateline = bool(re.search(r"(?im)^(?:[A-Z][A-Z .'-]+(?:,\s*[A-Z]{2})?[^\n]{0,35}[-—]|[A-Z][A-Za-z .'-]+,\s*[A-Z]{2},?\s+[A-Z][a-z]+\s+\d{1,2})", text))
    corporate = _count(text, [r"\bannounc", r"\bcompany\b", r"\b(?:inc\.?|corp\.?|ltd\.?|plc|llc)\b", r"\blaunch", r"\bacquir", r"\bearnings\b"])
    quotes = len(re.findall(r"[\"“][^\"”]{20,}[\"”]", text)) + _count(text, [r"\bsaid\b", r"\baccording to\b"])
    boiler = _count(text, [r"\bnews provided by\b", r"\bterms of use\b", r"\bsite ?map\b", r"\bcopyright\b", r"\bmedia contact\b", r"\babout (?:the )?company\b"])
    nav = _count(text, [r"\bsign in\b", r"\bproduct menu\b", r"\bsearch\b", r"\bstock quote\b", r"\bservices\b"])
    components = int(dateline) + int(corporate >= 2) + int(quotes > 0) + int(boiler > 0)
    if components == 4:
        return _bounded(9.0 + min(1.0, (corporate + boiler) * 0.08))
    if components >= 2:
        return _bounded(4.5 + components * 0.7 - min(1.5, nav * 0.2))
    if components == 1 and corporate:
        return 3.0
    return 0.0


def score__code_review__a144(text):
    text = _clean(text)
    sentences = _sentences(text)
    design = _count(text, [
        r"\barchitectur", r"\btrade-?offs?\b", r"\balternative\b", r"\bdesign\b",
        r"\bapi (?:stability|compatibility|contract)\b", r"\bbackward compatib",
        r"\bcoupling\b", r"\babstraction\b", r"\binterface\b", r"\bdependency\b",
        r"\bscalab", r"\bmaintainab"
    ])
    reasoning = _count(text, [r"\bbecause\b", r"\bhowever\b", r"\bif we\b", r"\bwould (?:mean|cause|allow)\b", r"\bimplication\b", r"\bon the other hand\b"])
    exchanges = len(re.findall(r"(?im)^(?:author|reviewer|reply|response|comment|[A-Za-z][\w.-]{1,20})\s*:", text))
    substantive = sum(1 for s in sentences if len(_words(s)) >= 15)
    trivial = _count(text, [r"\bnit\b", r"\btypo\b", r"\bformat", r"\bwhitespace\b", r"^\s*(?:done|fixed|updated)\s*[.!]?$" ] )
    if design >= 3 and reasoning >= 2 and substantive >= 3 and exchanges >= 2:
        return _bounded(8.0 + min(2.0, design * 0.15 + reasoning * 0.12))
    if design and (reasoning or substantive >= 2):
        return _bounded(5.0 + min(2.0, design * 0.25 + reasoning * 0.2))
    if trivial or text.strip():
        return 0.0
    return 0.0


def score__code_review__a279(text):
    text = _clean(text)
    deep = _count(text, [
        r"\barchitectur", r"\bperformance\b", r"\bcomplexity\b", r"\brace condition\b",
        r"\bconcurren", r"\bthread safe", r"\bedge case\b", r"\bsystem behavior\b",
        r"\bmemory\b", r"\blatency\b", r"\bscalab", r"\binvariant\b", r"\bdeadlock\b",
        r"\bapi contract\b", r"\bfailure mode\b", r"\btransaction\b"
    ])
    reasoning = _count(text, [r"\bbecause\b", r"\btherefore\b", r"\bhowever\b", r"\bif (?:this|we|the)\b", r"\botherwise\b", r"\btrade-?off\b"])
    routine = _count(text, [r"\btypo\b", r"\bnit\b", r"\bwhitespace\b", r"\bblank line\b", r"\bnaming\b", r"\bformatting\b", r"\bstyle\b"])
    substantive = sum(1 for s in _sentences(text) if len(_words(s)) >= 16)
    if deep >= 3 and (reasoning >= 2 or substantive >= 4):
        return _bounded(7.0 + min(3.0, deep * 0.2 + reasoning * 0.12 + substantive * 0.04))
    if deep or reasoning >= 2:
        return _bounded(4.0 + min(3.0, deep * 0.3 + reasoning * 0.18 + substantive * 0.05) - min(1.0, routine * 0.1))
    if not text.strip():
        return 0.0
    return _bounded(min(4.0, 1.0 + substantive * 0.12 - routine * 0.08))


def score__math__a36(text):
    answer = _answer_part(text)
    score = _math_answer_quality(answer)
    clarity = _count(answer, [r"\bfirst\b", r"\bnext\b", r"\bfinally\b", r"\bin other words\b", r"\bthis means\b", r"\bnote that\b"])
    correction = _count(answer, [r"\bmisconception\b", r"\bnot quite\b", r"\bthe issue is\b", r"\bthe mistake\b", r"\bactually\b"])
    score += min(1.0, clarity * 0.15 + correction * 0.2)
    if len(_words(answer)) < 35:
        score = min(score, 4.5)
    return _bounded(score)


def score__code_review__a54(text):
    text = _clean(text)
    confirmation = r"(?im)(?:^|(?:author|reviewer|reply|response|comment)\s*:\s*|\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+)(?:fixed|done|verified|removed|updated|resolved|addressed|changed|implemented|corrected)(?:\b|\s+it\b)"
    return 10.0 if re.search(confirmation, text) else 0.0


def score__math__a96(text):
    question = _question_part(text)
    advanced = _count(question, [
        r"\btopolog", r"\bmanifold\b", r"\bmeasure theor", r"\babstract algebra\b",
        r"\bfunctional analys", r"\bhomolog", r"\bcategory theor", r"\bstochastic\b",
        r"\bcomplex analys", r"\bdifferential geometr"
    ])
    exploration = _count(question, [
        r"\bintuition\b", r"\bgenerali[sz]", r"\bwhy\b", r"\bwhat if\b",
        r"\bunder what conditions\b", r"\bdeeper\b", r"\binterpret", r"\bextend\b",
        r"\balternative proofs?\b", r"\bconceptual\b", r"\brelationship\b"
    ])
    routine = _count(question, [r"\bcalculate\b", r"\bcompute\b", r"\bsolve for\b", r"\bhomework\b", r"\bverify (?:my|this) proof\b", r"\bfind the (?:value|answer)\b"])
    if advanced and exploration:
        return _bounded(7.0 + min(3.0, advanced * 0.35 + exploration * 0.3))
    if exploration >= 3:
        return _bounded(7.0 + min(2.0, exploration * 0.2))
    if exploration:
        return _bounded(4.0 + min(2.9, exploration * 0.55 + advanced * 0.25))
    if advanced:
        return min(6.0, 4.0 + advanced * 0.3)
    return _bounded(max(0.5 if question.strip() else 0.0, 2.5 - routine * 0.35))


def score__patents__a54(text):
    text = _clean(text)
    core = _count(text, [
        r"\bprocessor\b", r"\bmemory (?:controller|management|access)\b", r"\boperating system\b",
        r"\bmachine learning\b", r"\bneural network\b", r"\bcomputer-implemented\b",
        r"\bdata processing\b", r"\bcomputing architecture\b", r"\binstruction set\b",
        r"\bdatabase\b", r"\bvirtual machine\b"
    ])
    applied = _count(text, [
        r"\bnavigation\b", r"\bcommunication(?:s)? network\b", r"\bdata management\b",
        r"\bcontrol system\b", r"\bsensor data\b", r"\bwireless\b", r"\brouting\b"
    ])
    incidental = _count(text, [r"\bcontroller\b", r"\bmicroprocessor\b", r"\buser interface\b", r"\bdisplay\b", r"\bcomputer\b"])
    physical = _count(text, [r"\bmechanical\b", r"\bagricultur", r"\bbiolog", r"\bvehicle\b", r"\bapparatus\b", r"\bshaft\b", r"\bvalve\b"])
    if core >= 2 or (core and core + applied >= 3):
        return _bounded(9.0 + min(1.0, core * 0.12))
    if core or applied >= 2:
        return _bounded(7.0 + min(1.0, applied * 0.15))
    if incidental >= 2:
        return 5.5 if physical else 6.0
    if incidental:
        return 3.5 if physical else 5.0
    return 1.0 if physical == 0 and text.strip() else 0.0


def score__press_releases__a28(text):
    text = _clean(text)
    start = text[:600]
    immediate = bool(re.search(r"\bFOR IMMEDIATE RELEASE\b", start, re.I))
    dateline = bool(re.search(r"(?im)^(?:[A-Z][A-Z .'-]+(?:,\s*[A-Z]{2})?[^\n]{0,40}[-—]|[A-Z][A-Za-z .'-]+,\s*[A-Z]{2},?\s+[A-Z][a-z]+\s+\d{1,2})", start))
    announcement = _count(text, [r"\bannounc", r"\blaunch", r"\bacquir", r"\bappoint", r"\breports? (?:results|earnings)\b", r"\binitiative\b"])
    narrative = sum(1 for s in _sentences(text) if len(_words(s)) >= 15)
    nav = _count(text, [r"\bsign in\b", r"\bmenu\b", r"\bsearch\b", r"\bsite ?map\b", r"\bstock quote\b", r"\bproducts?\b", r"\bservices\b"])
    if (immediate or dateline) and announcement and narrative >= 3:
        return _bounded(8.0 + min(2.0, int(immediate) * 0.5 + narrative * 0.08 + announcement * 0.12))
    if announcement and narrative >= 2:
        return _bounded(5.0 + int(immediate or dateline) * 1.2 - min(1.0, nav * 0.1))
    if nav >= 3 or narrative == 0:
        return _bounded(max(0.0, 2.5 - nav * 0.25))
    return 4.0


def score__math__a78(text):
    question = _question_part(text)
    answer = _answer_part(text)
    confusion = _count(question, [r"\bwhy\b", r"\bdon't understand\b", r"\bconfus", r"\bwhat is wrong\b", r"\bwhere.*mistake\b", r"\bhow can\b", r"\bintuition\b", r"\bseems (?:wrong|contradictory|impossible)\b"])
    conceptual = _count(answer, [r"\bbecause\b", r"\bthe reason\b", r"\bthe mistake\b", r"\bthe issue\b", r"\bmeans that\b", r"\bin other words\b", r"\bintuition\b", r"\bnot the same as\b", r"\bhowever\b"])
    computation = len(re.findall(r"(?:=|\bcalculate\b|\bcompute\b|\bsolve\b)", answer, re.I))
    overlap_terms = set(w.lower() for w in _words(question) if len(w) > 5) & set(w.lower() for w in _words(answer))
    if confusion and conceptual >= 2 and len(overlap_terms) >= 2:
        return 10.0
    if conceptual and (confusion or len(overlap_terms) >= 2):
        return 5.0
    if computation or answer.strip():
        return 0.0
    return 0.0


def score__press_releases__a146(text):
    text = _clean(text)
    wc = max(1, len(_words(text)))
    elements = _count(text, [
        r"\bclick here\b", r"\bread more\b", r"\blearn more\b", r"\bsign in\b",
        r"\bregister\b", r"\bsearch\b", r"\bmenu\b", r"\bsubscribe\b",
        r"\bcontact(?: us)?\b", r"\bsite ?map\b", r"https?://|www\.",
        r"\bdownload\b", r"\bshare\b", r"\bview all\b", r"\badd to cart\b"
    ])
    markdown_links = len(re.findall(r"\[[^\]]+\]\([^\)]+\)", text))
    html_links = len(re.findall(r"<a\b", text, re.I))
    total = elements + markdown_links + html_links
    if total == 0:
        return 0.0
    density = total * 100.0 / wc
    return _bounded(min(10.0, density * 1.15 + min(3.0, total * 0.16)))


def score__patents__a72(text):
    text = _clean(text)
    components = _count(text, [
        r"\bcompris(?:e|es|ing)\b", r"\bcomponent\b", r"\bmodule\b", r"\bprocessor\b",
        r"\bmemory\b", r"\bsensor\b", r"\blayer\b", r"\bmember\b", r"\bassembly\b",
        r"\bconfigured to\b", r"\bcoupled to\b"
    ])
    exact = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|mm|cm|m|kg|g|mg|nm|°[CF]?|volts?|watts?|hz|mhz|ghz|seconds?|minutes?|hours?)\b", text, re.I))
    formulations = _count(text, [r"\bformula(?:tion)?\b", r"\bweight percent\b", r"\bmolar\b", r"\bchemical composition\b"])
    algorithms = _count(text, [r"\balgorithm\b", r"\bfor each\b", r"\biteration\b", r"\bcalculate\b", r"\bthreshold\b", r"\bif .{0,40} then\b"])
    mechanisms = components + exact * 2 + formulations * 2 + algorithms
    vague = _count(text, [r"\bmay include\b", r"\bany suitable\b", r"\bgenerally\b", r"\bdesired result\b", r"\bvarious means\b"])
    if mechanisms >= 10 and (exact or formulations or algorithms >= 2):
        return 10.0
    if mechanisms >= 3:
        return 5.0
    if mechanisms and vague == 0:
        return 5.0
    return 0.0


def score__press_releases__a291(text):
    text = _clean(text)
    investing = _count(text, [
        r"\binvest(?:ing|ment|or)\b", r"\bportfolio\b", r"\btrading\b", r"\bholdings?\b",
        r"\basset allocation\b", r"\bpassive income\b", r"\bdividend\b", r"\bstocks?\b",
        r"\bbonds?\b", r"\bshares?\b", r"\bposition\b", r"\breturn\b"
    ])
    first_person = _count(text, [r"\bi\b", r"\bmy\b", r"\bme\b", r"\bi'm\b", r"\bi've\b"])
    opinion = _count(text, [r"\bi (?:believe|think|expect|prefer|recommend)\b", r"\bmy strategy\b", r"\bmy portfolio\b", r"\bin my view\b", r"\bmy goal\b"])
    corporate = _count(text, [r"\bfor immediate release\b", r"\bcompany announced\b", r"\bnews provided by\b", r"\bmedia contact\b"])
    byline = bool(re.search(r"(?im)^\s*by\s+[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3}\s*$", text))
    if investing >= 3 and first_person >= 2 and (opinion or byline) and corporate == 0:
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
