# AUTO: blind rule compilation chunk c5

import re
import math
import statistics
import string
import collections


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _clean_words(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or "")


def _sentences(text):
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def _lines(text):
    if not text:
        return []
    return text.splitlines()


def _clamp(x, lo=0.0, hi=10.0):
    try:
        if x != x:  # nan check
            return lo
    except Exception:
        return lo
    return max(lo, min(hi, x))


_MONTHS = (r'(?:January|February|March|April|May|June|July|August|September|'
           r'October|November|December)')


# ---------------------------------------------------------------------------
# math__a168: mathematical maturity, depth, and level of abstraction
# ---------------------------------------------------------------------------

def score__math__a168(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    adv_kw = ["abstract algebra", "group theory", "ring theory", "field theory", "galois",
              "topology", "topological space", "manifold", "differential geometry",
              "real analysis", "complex analysis", "measure theory", "functional analysis",
              "category theory", "homomorphism", "isomorphism", "homeomorphism",
              "diffeomorphism", "sigma-algebra", "banach space", "hilbert space",
              "compactness", "metric space", "holomorphic", "cohomology", "module theory",
              "lie algebra", "lie group", "sheaf", "functor"]
    mid_kw = ["calculus", "derivative", "integral", "linear algebra", "matrix", "matrices",
              "eigenvalue", "eigenvector", "vector space", "differential equation",
              "taylor series", "probability distribution", "random variable",
              "number theory", "modular arithmetic", "combinatorics", "limit of"]
    basic_kw = ["arithmetic", "basic algebra", "solve for x", "elementary school",
                "multiplication table", "long division", "simple fraction", "percentage"]
    adv_count = sum(t.count(k) for k in adv_kw)
    mid_count = sum(t.count(k) for k in mid_kw)
    basic_count = sum(t.count(k) for k in basic_kw)
    proof_kw = ["proof", "theorem", "lemma", "corollary", "qed", "axiom", "by induction",
                "by contradiction", "rigorously", "formally define"]
    proof_count = sum(t.count(k) for k in proof_kw)

    if adv_count > 0:
        base = 9.0 + min(1.0, proof_count * 0.2)
        return _clamp(base)
    if mid_count > 0:
        base = 5.0 + min(1.5, proof_count * 0.15)
        if basic_count > mid_count:
            base -= 0.3
        return _clamp(base)
    if basic_count > 0:
        return 1.0
    has_math_symbols = bool(re.search(r'[=+\-*/^]|\d', t))
    return 0.5 if has_math_symbols else 0.0


# ---------------------------------------------------------------------------
# code_review__a126: technical depth and substantive value of review comments
# ---------------------------------------------------------------------------

def score__code_review__a126(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    words = _clean_words(text)
    wc = max(1, len(words))
    high_kw = ["architecture", "architectural", "trade-off", "tradeoff", "edge case",
               "race condition", "memory leak", "deadlock", "complexity", "security vulnerability",
               "root cause", "explain why", "alternative approach", "refactor", "design pattern",
               "this could fail", "this will break"]
    low_kw = ["nit:", "typo", "formatting", "lgtm", "thanks", "looks good", "rename",
              "remove this", "fix this", "unused import", "whitespace", "+1"]
    explain_kw = ["because", "since", "therefore", "however", "in order to", "this is because"]
    high_count = sum(t.count(k) for k in high_kw)
    low_count = sum(t.count(k) for k in low_kw)
    explain_count = sum(t.count(k) for k in explain_kw)

    norm = max(0.5, wc / 50.0)
    score = 3.0 + (high_count / norm) * 2.2 - (low_count / norm) * 1.5
    score += min(2.0, explain_count * 0.4)
    if wc < 8 and high_count == 0:
        score = min(score, 2.0)
    return _clamp(score)


# ---------------------------------------------------------------------------
# math__a198: proof-verification request / complete proof / partial proof / none
# ---------------------------------------------------------------------------

def score__math__a198(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    verify_kw = ["is my proof correct", "check my proof", "verify my proof",
                 "is this proof correct", "am i correct", "is this correct",
                 "did i do this right", "review my proof", "is my solution correct",
                 "is my reasoning correct", "is this proof valid", "is my proof valid"]
    if any(k in t for k in verify_kw):
        return 10.0

    proof_kw = ["proof", "qed", "theorem", "lemma", "∎"]
    has_proof_markers = any(k in t for k in proof_kw)
    words = _clean_words(text)
    wc = len(words)
    sentences = _sentences(text)
    sc = len(sentences)

    if has_proof_markers and wc > 60 and sc >= 3:
        return 6.7

    explain_kw = ["because", "since", "therefore", "thus", "hence", "this means", "note that"]
    explain_count = sum(t.count(k) for k in explain_kw)
    if (wc > 15 and explain_count >= 1) or (has_proof_markers and wc <= 60):
        return 3.3

    return 0.0


# ---------------------------------------------------------------------------
# code_review__a153: technical depth, clarity, constructiveness of review comments
# ---------------------------------------------------------------------------

def score__code_review__a153(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    words = _clean_words(text)
    wc = max(1, len(words))
    terse_kw = ["remove", "rename", "fix this", "nit", "typo", "unused", "lgtm", "+1",
                "done", "fixed"]
    high_kw = ["architecture", "architectural", "edge case", "trade-off", "tradeoff",
               "alternative implementation", "consider using", "the reason this",
               "this could fail", "suggest we", "instead of this", "detailed explanation",
               "design decision"]
    high_count = sum(t.count(k) for k in high_kw)
    terse_count = sum(t.count(k) for k in terse_kw)
    sentences = _sentences(text)
    avg_sent_len = wc / max(1, len(sentences))

    score = 4.0
    score += min(4.0, high_count * 1.3)
    score -= min(4.0, terse_count * 0.9)
    if avg_sent_len > 8:
        score += min(1.5, (avg_sent_len - 8) * 0.05)
    if wc < 6:
        score = min(score, 2.0)
    return _clamp(score)


# ---------------------------------------------------------------------------
# CAL__CAL5: uppercase-emphasis word count
# ---------------------------------------------------------------------------

def score__CAL__CAL5(text):
    if not text:
        return 0.0
    words = re.findall(r"[A-Za-z']+", text)
    count = sum(1 for w in words if len(w) >= 3 and w.isupper())
    if count >= 5:
        return 10.0
    if count >= 1:
        return 5.0
    return 0.0


# ---------------------------------------------------------------------------
# code_review__a81: collaborative substantive technical dialogue
# ---------------------------------------------------------------------------

def score__code_review__a81(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    words = _clean_words(text)
    wc = max(1, len(words))
    question_count = t.count("?")
    architecture_kw = ["architecture", "design", "trade-off", "tradeoff", "approach",
                       "pattern", "scalab"]
    arch_count = sum(t.count(k) for k in architecture_kw)
    explain_kw = ["because", "the reason", "this is due to", "rationale", "in order to", "since"]
    explain_count = sum(t.count(k) for k in explain_kw)
    superficial_kw = ["nit:", "typo", "formatting", "remove this", "rename", "lgtm",
                      "fix this", "minor"]
    superficial_count = sum(t.count(k) for k in superficial_kw)

    substantive_signal = question_count * 0.6 + arch_count * 1.0 + explain_count * 0.8
    score = 2.0 + substantive_signal - superficial_count * 0.7
    if wc < 10:
        score = min(score, 2.0)
    return _clamp(score)


# ---------------------------------------------------------------------------
# press_releases__a80: press release announcing a specific new offering
# ---------------------------------------------------------------------------

def score__press_releases__a80(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    not_pr_kw = ["add to cart", "home page", "sign in", "log in", "subscribe now",
                 "table of contents", "opinion", "editorial", "cookie policy",
                 "privacy policy", "terms of service", "navigation menu"]
    specific_launch_kw = ["announces the launch of", "unveils", "introduces its new",
                          "new product", "launches new", "announces new", "today announced",
                          "proud to announce", "new service", "new technology", "debuts"]
    general_pr_kw = ["press release", "announced today", "appoints", "appointment as",
                     "board of directors", "quarterly results", "fiscal year", "named as",
                     "promotes", "earnings report", "prnewswire", "businesswire", "globenewswire"]

    specific_count = sum(t.count(k) for k in specific_launch_kw)
    general_count = sum(t.count(k) for k in general_pr_kw)
    notpr_count = sum(t.count(k) for k in not_pr_kw)

    if specific_count > 0 and notpr_count == 0:
        return 10.0
    if general_count > 0:
        return 7.5
    return 0.0


# ---------------------------------------------------------------------------
# press_releases__a105: narrative completeness / informational value
# ---------------------------------------------------------------------------

def score__press_releases__a105(text):
    if not text or not text.strip():
        return 0.0
    lines = [l for l in _lines(text) if l.strip()]
    if not lines:
        return 0.0
    sentences = _sentences(text)
    words = _clean_words(text)
    wc = len(words)
    short_line_frac = sum(1 for l in lines if len(l.strip()) < 40) / len(lines)
    avg_sent_len = wc / max(1, len(sentences))

    prose_score = 0.0
    if avg_sent_len >= 12:
        prose_score += 5
    elif avg_sent_len >= 6:
        prose_score += 2
    prose_score += (1 - short_line_frac) * 5
    return _clamp(prose_score)


# ---------------------------------------------------------------------------
# press_releases__a76: proportion of continuous coherent prose (structure only)
# ---------------------------------------------------------------------------

def score__press_releases__a76(text):
    if not text or not text.strip():
        return 0.0
    lines = [l for l in _lines(text) if l.strip()]
    if not lines:
        return 0.0
    sentences = _sentences(text)
    words = _clean_words(text)
    wc = len(words)
    avg_sent_len = wc / max(1, len(sentences))
    short_line_frac = sum(1 for l in lines if len(l.strip()) < 35) / len(lines)
    long_para_frac = sum(1 for l in lines if len(l.strip()) >= 120) / len(lines)

    prose_signal = long_para_frac * 6 + (1 - short_line_frac) * 4
    if avg_sent_len < 5:
        prose_signal *= 0.5
    return _clamp(prose_signal)


# ---------------------------------------------------------------------------
# math__a54: mathematical quality of the provided Answer
# ---------------------------------------------------------------------------

def score__math__a54(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    words = _clean_words(text)
    wc = len(words)
    sentences = _sentences(text)
    incorrect_kw = ["i don't know", "not sure", "cannot solve", "no idea"]
    if any(k in t for k in incorrect_kw) or wc < 5:
        return 0.0

    explain_kw = ["because", "therefore", "thus", "hence", "step", "first,", "next,",
                  "note that", "this means", "in other words"]
    explain_count = sum(t.count(k) for k in explain_kw)
    rigor_kw = ["proof", "qed", "theorem", "rigorous", "formally"]
    rigor_count = sum(t.count(k) for k in rigor_kw)

    if wc > 150 and explain_count >= 4 and (rigor_count >= 1 or len(sentences) >= 8):
        return 10.0
    if wc > 40 and explain_count >= 2:
        return 7.5
    if wc >= 10:
        return 5.0
    return 0.0


# ---------------------------------------------------------------------------
# patents__a228: mechanical / electromechanical / physical hardware disclosure
# ---------------------------------------------------------------------------

def score__patents__a228(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    mech_assembly_kw = ["gear", "shaft", "piston", "hinge", "lever", "linkage", "bracket",
                        "pivot", "actuator", "spring-loaded", "bearing assembly",
                        "crankshaft", "cam follower", "hydraulic cylinder",
                        "pneumatic cylinder", "moving part", "mechanical assembly",
                        "rotates about", "coupled to a shaft"]
    hw_device_kw = ["housing", "valve", "nozzle", "chassis", "enclosure", "fluid conduit",
                    "structural member", "mounting bracket", "sensor housing",
                    "physical configuration"]
    device_lacking_moving_kw = ["sensor", "electrode", "membrane", "chemical composition",
                                "material layer", "coating", "substrate"]
    software_kw = ["algorithm", "software module", "data processing", "processor configured to",
                  "computer-readable medium", "neural network", "database",
                  "communication protocol", "business method", "circuitry",
                  "memory storing instructions", "logic gate", "transmitting data"]

    mech_count = sum(t.count(k) for k in mech_assembly_kw)
    hw_count = sum(t.count(k) for k in hw_device_kw)
    device_count = sum(t.count(k) for k in device_lacking_moving_kw)
    sw_count = sum(t.count(k) for k in software_kw)

    if mech_count >= 2:
        return 10.0
    if mech_count == 1 or hw_count >= 2:
        return 8.1
    if hw_count >= 1 or device_count >= 1:
        return 5.1
    if sw_count > 0:
        return 0.0
    return 2.5


# ---------------------------------------------------------------------------
# press_releases__a25: proportion English vs non-English text
# ---------------------------------------------------------------------------

def score__press_releases__a25(text):
    if not text or not text.strip():
        return 0.0
    t = re.sub(r'<[^>]+>', ' ', text)
    t = re.sub(r'https?://\S+', ' ', t)
    t = re.sub(r'`[^`]*`', ' ', t)
    letters = re.findall(r'[^\W\d_]', t, re.UNICODE)
    if not letters:
        return 5.0
    ascii_letters = [c for c in letters if ord(c) < 128]
    frac = len(ascii_letters) / len(letters)
    return _clamp(frac * 10.0)


# ---------------------------------------------------------------------------
# press_releases__a262: formal dateline (capitalized location + date)
# ---------------------------------------------------------------------------

def score__press_releases__a262(text):
    if not text or not text.strip():
        return 0.0
    full_pattern = re.compile(
        r'\b[A-Z][A-Za-z.\s]{1,30},\s*[A-Za-z.]{2,20},?\s+' + _MONTHS +
        r'\s+\d{1,2},?\s+\d{4}\s*(?:/PRNewswire/|/--\s|--\s)'
    )
    city_date_pattern = re.compile(
        r'\b[A-Z][A-Z\s]{2,25},\s+' + _MONTHS + r'\s+\d{1,2},?\s+\d{4}'
    )
    if full_pattern.search(text) or city_date_pattern.search(text):
        return 10.0

    partial_pattern = re.compile(_MONTHS + r'\s+\d{1,2},?\s+\d{4}')
    prnewswire_pattern = re.compile(r'/PRNewswire/|/--\s')
    if partial_pattern.search(text) or prnewswire_pattern.search(text):
        return 2.0
    return 0.0


# ---------------------------------------------------------------------------
# press_releases__a75: density of capitalized words/acronyms/proper nouns + lists
# ---------------------------------------------------------------------------

def score__press_releases__a75(text):
    if not text or not text.strip():
        return 0.0
    words = _clean_words(text)
    wc = max(1, len(words))
    cap_count = sum(1 for w in words if w.isupper() or (w[:1].isupper() and len(w) > 1))
    cap_density = cap_count / wc
    lines = [l for l in _lines(text) if l.strip()]
    short_line_frac = sum(1 for l in lines if len(l.strip()) < 40) / max(1, len(lines))

    score = cap_density * 22 + short_line_frac * 3
    return _clamp(score)


# ---------------------------------------------------------------------------
# code_review__a72: technical precision and formality of review comments
# ---------------------------------------------------------------------------

def score__code_review__a72(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    directive_kw = ["indent", "whitespace", "trailing space", "semicolon", "lint",
                    "style guide", "should be", "use camelcase", "use snake_case", "align",
                    "spacing", "tab width", "line length", "format this", "rename this to",
                    "change this to"]
    conversational_kw = ["why", "how about", "what if", "i think", "in my opinion",
                         "could we discuss", "curious", "wondering", "let's talk"]
    directive_count = sum(t.count(k) for k in directive_kw)
    conv_count = sum(t.count(k) for k in conversational_kw)
    question_marks = t.count("?")

    score = 5.0 + directive_count * 1.0 - (conv_count + question_marks) * 0.8
    return _clamp(score)


# ---------------------------------------------------------------------------
# press_releases__a100: self-contained corporate press release
# ---------------------------------------------------------------------------

def score__press_releases__a100(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    dateline = bool(re.search(
        r'\b[A-Z][A-Za-z\s]{1,25},\s+[A-Za-z]{2,20},?\s+' + _MONTHS + r'\s+\d{1,2}', text
    ))
    announcement_kw = ["announced today", "is pleased to announce", "announces",
                       "today announced", "press release"]
    boilerplate_kw = ["about the company", "for more information", "forward-looking statements",
                      "contact:", "media contact", "all rights reserved", "trademarks of"]
    low_kw = ["add to cart", "sign in", "log in", "cookie", "404", "page not found",
             "subscribe", "home page", "navigation menu", "product catalog"]

    ann = any(k in t for k in announcement_kw)
    boiler = any(k in t for k in boilerplate_kw)
    low = any(k in t for k in low_kw)

    score = 1.0
    if dateline:
        score += 4
    if ann:
        score += 3
    if boiler:
        score += 2
    if low:
        score = min(score, 3.0)
    return _clamp(score)


# ---------------------------------------------------------------------------
# math__a6: pedagogical clarity, directness, appropriateness of the answer
# ---------------------------------------------------------------------------

def score__math__a6(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    words = _clean_words(text)
    wc = len(words)
    sentences = _sentences(text)
    step_kw = ["step 1", "first,", "next,", "then,", "finally,", "for example",
              "think of it as", "in other words", "intuitively", "let's break", "imagine",
              "analogy"]
    step_count = sum(t.count(k) for k in step_kw)
    jargon_kw = ["it follows that", "trivially", "clearly,", "by definition", "q.e.d", "∎"]
    jargon_count = sum(t.count(k) for k in jargon_kw)
    avg_sent_len = wc / max(1, len(sentences))

    score = 5.0
    score += min(4.0, step_count * 1.2)
    if wc < 12:
        score -= 3.0
    if avg_sent_len > 30:
        score -= 2.0
    score -= min(2.0, jargon_count * 0.7)
    return _clamp(score)


# ---------------------------------------------------------------------------
# press_releases__a41: density of UI/web navigation boilerplate
# ---------------------------------------------------------------------------

def score__press_releases__a41(text):
    if not text or not text.strip():
        return 0.0
    lines = [l for l in _lines(text) if l.strip()]
    if not lines:
        return 0.0
    nav_kw = ["home", "menu", "sign in", "log in", "subscribe", "cookie", "privacy policy",
             "terms of use", "select language", "account", "cart", "navigation",
             "skip to content", "all rights reserved"]
    nav_line_count = sum(1 for l in lines if any(k in l.lower() for k in nav_kw))
    short_line_frac = sum(1 for l in lines if len(l.strip()) < 30) / len(lines)
    nav_frac = nav_line_count / len(lines)

    score = nav_frac * 6 + short_line_frac * 4
    return _clamp(score)


# ---------------------------------------------------------------------------
# press_releases__a66: density of explicit financial/corporate/investment data
# ---------------------------------------------------------------------------

def score__press_releases__a66(text):
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    words = _clean_words(text)
    wc = max(1, len(words))
    fin_kw = ["revenue", "earnings", "eps", "ebitda", "net income", "gross margin",
             "stock price", "shares outstanding", "quarterly", "fiscal year", "analyst",
             "dividend", "market cap", "investor", "forecast", "guidance"]
    fin_count = sum(t.count(k) for k in fin_kw)
    dollar_count = len(re.findall(r'\$\s?\d', text))
    percent_count = len(re.findall(r'\d+(?:\.\d+)?\s?%', text))
    number_density = len(re.findall(r'\b\d[\d,\.]*\b', text)) / max(1, wc / 20)

    score = fin_count * 1.3 + dollar_count * 0.8 + percent_count * 0.8 + min(2.0, number_density * 0.3)
    return _clamp(score)


# ---------------------------------------------------------------------------
# CAL__CAL4: presence of a line beginning with '#'
# ---------------------------------------------------------------------------

def score__CAL__CAL4(text):
    if not text:
        return 0.0
    for line in text.splitlines():
        if line.startswith('#'):
            return 10.0
    return 0.0


# ---------------------------------------------------------------------------
# press_releases__a73: proportion of coherent substantive content vs boilerplate
# ---------------------------------------------------------------------------

def score__press_releases__a73(text):
    if not text or not text.strip():
        return 0.0
    lines = [l for l in _lines(text) if l.strip()]
    if not lines:
        return 0.0
    nav_kw = ["home", "menu", "sign in", "log in", "subscribe", "cookie", "privacy policy",
             "terms of use", "select language", "account", "cart", "navigation",
             "skip to content", "all rights reserved", "click here", "read more"]
    boiler_line_count = sum(
        1 for l in lines if any(k in l.lower() for k in nav_kw) or len(l.strip()) < 25
    )
    substantive_frac = 1 - (boiler_line_count / len(lines))
    sentences = _sentences(text)
    words = _clean_words(text)
    avg_sent_len = len(words) / max(1, len(sentences))

    score = substantive_frac * 10
    if avg_sent_len < 5:
        score *= 0.6
    return _clamp(score)


# ---------------------------------------------------------------------------
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
