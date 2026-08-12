# AUTO: blind rule compilation chunk c6

import re
import math
import statistics
import string
import collections


# ---------------------------------------------------------------------------
# shared helpers (not scoring functions themselves)
# ---------------------------------------------------------------------------

def _words(text):
    if not text:
        return []
    return re.findall(r"[A-Za-z']+", text)


def _sentences(text):
    if not text or not text.strip():
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def _lower(text):
    return (text or "").lower()


def _count_kw(text_lower, kws):
    return sum(text_lower.count(k) for k in kws)


def _clamp(x, lo=0.0, hi=10.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return lo
    if x != x:  # NaN
        return lo
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# press_releases__a117
# ---------------------------------------------------------------------------

_A117_DATELINE_RE = re.compile(
    r'^[A-Z][A-Za-z\.\s]{1,30},\s*(?:[A-Z]{2}\s*,?\s*)?'
    r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
    r'Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}',
    re.MULTILINE,
)

_A117_WIRE_TAGS = ["prnewswire", "pr newswire", "business wire", "businesswire",
                    "globenewswire", "globe newswire", "marketwired", "/prn/"]

_A117_NEWS_KW = ["announced", "announces", "today announced", "reported today",
                  "quarterly results", "fiscal year", "fiscal quarter", "earnings",
                  "acquisition", "appointed", "named as", "financial results",
                  "results for the quarter"]

_A117_NAV_KW = ["home", "about us", "contact us", "sign in", "log in", "subscribe",
                 "navigation", "copyright ©", "all rights reserved", "click here",
                 "privacy policy", "terms of service", "add to cart", "site map"]


def score__press_releases__a117(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    head = text[:400]
    has_dateline = bool(_A117_DATELINE_RE.search(head)) or bool(_A117_DATELINE_RE.search(text))
    has_wire = any(tag in tl for tag in _A117_WIRE_TAGS)
    news_hits = _count_kw(tl, _A117_NEWS_KW)
    nav_hits = _count_kw(tl, _A117_NAV_KW)

    if (has_dateline or has_wire) and news_hits >= 1:
        score = 9.5
    elif has_wire or news_hits >= 2:
        score = 6.5
    elif news_hits >= 1 or has_dateline:
        score = 5.5
    elif nav_hits >= 2:
        score = 1.0
    else:
        score = 2.5

    score -= min(nav_hits, 3) * 0.4
    return _clamp(score)


# ---------------------------------------------------------------------------
# math__a102
# ---------------------------------------------------------------------------

_A102_PROOF_KW = ["proof:", "q.e.d", "we prove", "we will show", "by induction",
                   "by contradiction", "let us show", "this completes the proof",
                   "hence proven", "lemma", "theorem", "claim:"]
_A102_PARTIAL_KW = ["outline", "sketch", "roughly speaking", "intuitively",
                     "without proof", "it can be shown"]
_A102_WEAK_KW = ["please help", "can someone", "i don't know how", "not sure how",
                  "the answer is", "final answer:"]


def score__math__a102(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    proof_hits = _count_kw(tl, _A102_PROOF_KW)
    partial_hits = _count_kw(tl, _A102_PARTIAL_KW)
    weak_hits = _count_kw(tl, _A102_WEAK_KW)
    step_count = len(re.findall(r'\bstep\s*\d+', tl)) + len(re.findall(r'^\s*\d+[\.\)]', text, re.MULTILINE))
    eq_count = text.count('=')

    if weak_hits >= 1 and proof_hits == 0 and eq_count < 2:
        return 1.0
    if proof_hits >= 2 and (step_count >= 2 or eq_count >= 5):
        return 8.5
    if proof_hits >= 1 and eq_count >= 2:
        return 6.0
    if partial_hits >= 1 or proof_hits >= 1:
        return 4.5
    if eq_count >= 1:
        return 2.0
    return 0.5


# ---------------------------------------------------------------------------
# math__a126
# ---------------------------------------------------------------------------

_A126_EQ_RE = re.compile(r'\\(frac|int|sum|prod|sqrt|partial|nabla|infty|alpha|beta|theta|sigma|lim)')


def score__math__a126(text):
    text = text or ""
    if not text.strip():
        return 0.0
    wc = len(_words(text))
    eq_markers = len(_A126_EQ_RE.findall(text)) + text.count('=') + text.count('$')

    if wc < 30:
        length_score = 0.0
    elif wc < 100:
        length_score = 2.0
    elif wc < 250:
        length_score = 4.5
    elif wc < 500:
        length_score = 6.5
    elif wc < 900:
        length_score = 8.0
    else:
        length_score = 9.5

    bonus = min(1.5, eq_markers * 0.08)
    return _clamp(length_score + bonus)


# ---------------------------------------------------------------------------
# patents__a30
# ---------------------------------------------------------------------------

_A30_CORE_KW = ["software", "data processing", "database", "algorithm", "processor",
                 "computing", "computer system", "memory", "storage device", "indexing",
                 "index of", "scheduling", "protocol", "network", "server", "cache",
                 "query", "instructions executable", "non-transitory computer-readable",
                 "data structure"]
_A30_OTHER_KW = ["mechanical", "engine", "valve", "vehicle", "agricultur", "crop",
                  "chemical composition", "radio frequency", "antenna",
                  "telecommunications", "sensor housing", "circuit board", "motor",
                  "hydraulic", "pump", "wheel", "gear", "fertilizer"]


def score__patents__a30(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    core = _count_kw(tl, _A30_CORE_KW)
    other = _count_kw(tl, _A30_OTHER_KW)
    if core == 0:
        return 0.0
    ratio = core / float(core + other + 1)
    if ratio >= 0.6 and core >= 3:
        return _clamp(9.0 + min(ratio, 1.0))
    if ratio >= 0.45:
        return _clamp(7.0 + ratio * 2)
    if core >= 1:
        return _clamp(2.0 + ratio * 4)
    return 0.0


# ---------------------------------------------------------------------------
# math__a156
# ---------------------------------------------------------------------------

_A156_ADV_KW = ["theorem", "lemma", "proof", "topology", "manifold", "homomorphism",
                 "isomorphism", "abstract algebra", "measure theory",
                 "functional analysis", "category theory", "formal logic", "axiom",
                 "rigorous", "converge", "epsilon-delta", "foundational",
                 "generalize", "non-trivial", "corollary"]
_A156_ROUTINE_KW = ["solve for x", "simplify", "plug in", "homework", "basic algebra",
                     "elementary", "how do i solve", "calculate the", "step by step",
                     "arithmetic", "word problem"]


def score__math__a156(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    adv = _count_kw(tl, _A156_ADV_KW)
    rou = _count_kw(tl, _A156_ROUTINE_KW)

    if adv >= 3 and rou == 0:
        return 9.5
    if adv >= 2:
        return 7.5
    if adv >= 1 and rou <= 1:
        return 5.5
    if rou >= 2:
        return 1.0
    return 3.0


# ---------------------------------------------------------------------------
# CAL__CAL1
# ---------------------------------------------------------------------------

def score__CAL__CAL1(text):
    text = text or ""
    digits = sum(1 for c in text if c.isdigit())
    if digits >= 3:
        return 10.0
    if digits >= 1:
        return 5.0
    return 0.0


# ---------------------------------------------------------------------------
# press_releases__a65
# ---------------------------------------------------------------------------

_A65_CORP_KW = ["acquisition", "financial results", "executive appointment",
                 "appointed", "named chief", "quarterly results", "earnings",
                 "fiscal year", "board of directors", "announced today",
                 "today announced"]
_A65_BIZ_KW = ["investor relations", "investor portal", "stock price", "market cap",
                "revenue", "shares outstanding", "financial news", "business news"]
_A65_NAV_KW = ["home", "about us", "contact us", "sign in", "navigation", "menu",
                "product page", "privacy policy", "terms of service"]


def score__press_releases__a65(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    head = text[:400]
    has_dateline = bool(_A117_DATELINE_RE.search(head)) or bool(_A117_DATELINE_RE.search(text))
    has_wire = any(tag in tl for tag in _A117_WIRE_TAGS)
    corp_hits = _count_kw(tl, _A65_CORP_KW)
    biz_hits = _count_kw(tl, _A65_BIZ_KW)
    nav_hits = _count_kw(tl, _A65_NAV_KW)

    if (has_dateline or has_wire) and corp_hits >= 1:
        return 10.0
    if biz_hits >= 1 or corp_hits >= 1:
        return 4.0
    if nav_hits >= 1:
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# code_review__a108
# ---------------------------------------------------------------------------

_A108_REPLY_KW = ["re:", "reply", "replied", "responded", "response:"]
_A108_AGREE_KW = ["agreed", "lgtm", "makes sense", "good point", "sounds good",
                   "sgtm", "sounds right", "fair point"]


def score__code_review__a108(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    quote_lines = len(re.findall(r'^\s*>', text, re.MULTILINE))
    reply_hits = _count_kw(tl, _A108_REPLY_KW)
    agree_hits = _count_kw(tl, _A108_AGREE_KW)
    questions = text.count('?')

    total = quote_lines + reply_hits + agree_hits + min(questions, 5)

    if total >= 8:
        return 10.0
    if total >= 5:
        return 7.5
    if total >= 2:
        return 5.0
    if total >= 1:
        return 2.5
    return 0.0


# ---------------------------------------------------------------------------
# press_releases__a97
# ---------------------------------------------------------------------------

_A97_DOLLAR_RE = re.compile(r'\$\s?\d[\d,\.]*\s?(million|billion|thousand|m|bn)?', re.IGNORECASE)
_A97_PCT_RE = re.compile(r'\d+(\.\d+)?\s?%')
_A97_TERM_KW = ["eps", "earnings per share", "revenue", "aum", "assets under management",
                 "buyback", "share repurchase", "net income", "operating income",
                 "gross margin"]


def score__press_releases__a97(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    dollar_hits = len(_A97_DOLLAR_RE.findall(text))
    pct_hits = len(_A97_PCT_RE.findall(text))
    term_hits = _count_kw(tl, _A97_TERM_KW)

    wc = max(1, len(_words(text)))
    density = (dollar_hits + pct_hits + term_hits) / (wc / 200.0)

    if density == 0:
        return 0.0
    if density >= 4:
        return 10.0
    if density >= 1.2:
        return 6.5
    return 3.0


# ---------------------------------------------------------------------------
# code_review__a198
# ---------------------------------------------------------------------------

_A198_IMPROV_RE = re.compile(r'\bimprov\w*', re.IGNORECASE)
_A198_TICKET_RE = re.compile(
    r'\b(fix(es|ed)?|close[sd]?|resolve[sd]?|issue)\s*#\s*\d+', re.IGNORECASE
)
_A198_TRAILING_RE = re.compile(r'#\d+\s*$')


def score__code_review__a198(text):
    text = text or ""
    if not text.strip():
        return 0.0
    title = text.strip().split('\n')[0]

    if _A198_IMPROV_RE.search(title):
        return 10.0
    if _A198_TICKET_RE.search(title) or _A198_TRAILING_RE.search(title.strip()):
        return 5.0
    return 0.0


# ---------------------------------------------------------------------------
# patents__a6
# ---------------------------------------------------------------------------

_A6_CORE_KW = ["computer", "processor", "software", "digital data", "electronic control",
                "image processing", "sensor", "integrated circuit", "microprocessor",
                "memory", "data system", "computing device"]
_A6_MID_KW = ["automated control", "electronic component", "actuator", "control unit",
               "imaging device", "circuit board"]


def score__patents__a6(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    core = _count_kw(tl, _A6_CORE_KW)
    mid = _count_kw(tl, _A6_MID_KW)

    if core >= 2:
        return 9.5
    if core >= 1:
        return 7.0
    if mid >= 2:
        return 5.0
    if mid >= 1:
        return 4.5
    return 0.5


# ---------------------------------------------------------------------------
# press_releases__a0
# ---------------------------------------------------------------------------

_A0_KW = ["code of ethics", "code of conduct", "compliance", "anti-corruption",
           "anti-bribery", "whistleblower", "regulatory disclosure", "legal standards",
           "corporate governance", "ethical", "corporate social responsibility",
           "sustainability report", "regulatory compliance", "sec filing", "gdpr",
           "fcpa", "ethics policy"]


def score__press_releases__a0(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    hits = _count_kw(tl, _A0_KW)

    if hits == 0:
        return 0.0
    if hits <= 2:
        return 4.0
    if hits <= 4:
        return 7.0
    return 9.5


# ---------------------------------------------------------------------------
# patents__a134
# ---------------------------------------------------------------------------

_A134_CORE_KW = ["algorithm", "machine learning", "neural network", "parallel processing",
                   "memory operation", "user interface", "software-defined network",
                   "data processing", "computer program", "instructions stored",
                   "processor configured to", "software module"]
_A134_GENERIC_KW = ["electronic device", "telecommunications", "wireless", "antenna",
                      "transceiver", "circuit", "hardware component", "integrated circuit"]
_A134_MECH_KW = ["mechanical", "engine", "valve", "chemical compound", "agricultural",
                   "biological", "enzyme", "harvest", "crop", "motor", "pump", "gear",
                   "vehicle chassis"]


def score__patents__a134(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    core = _count_kw(tl, _A134_CORE_KW)
    generic = _count_kw(tl, _A134_GENERIC_KW)
    mech = _count_kw(tl, _A134_MECH_KW)

    if core >= 1:
        return _clamp(8.0 + min(core, 2))
    if generic >= 1:
        return _clamp(4.0 + min(generic, 3))
    if mech >= 1:
        return _clamp(1.0 + min(mech, 2))
    return 0.0


# ---------------------------------------------------------------------------
# patents__a84
# ---------------------------------------------------------------------------

_A84_JARGON_KW = ["wherein", "said ", "plurality", "comprising", "aforementioned",
                    "heretofore", "notwithstanding", "whereby", "thereof",
                    "hereinafter", "hereto"]


def score__patents__a84(text):
    text = text or ""
    if not text.strip():
        return 5.0
    tl = _lower(text)
    words = _words(text)
    wc = len(words)
    sents = _sentences(text)
    num_sents = max(1, len(sents))
    avg_sent_len = wc / float(num_sents)
    avg_word_len = (sum(len(w) for w in words) / float(wc)) if wc else 0.0
    jargon_hits = _count_kw(tl, _A84_JARGON_KW)

    penalty = 0.0
    if avg_sent_len > 25:
        penalty += 3.0
    elif avg_sent_len > 18:
        penalty += 1.5

    if avg_word_len > 5.5:
        penalty += 2.0
    elif avg_word_len > 4.8:
        penalty += 1.0

    penalty += min(4.0, jargon_hits * 0.5)

    return _clamp(10.0 - penalty)


# ---------------------------------------------------------------------------
# press_releases__a87
# ---------------------------------------------------------------------------

_A87_EMAIL_RE = re.compile(r'[\w\.\-]+@[\w\.\-]+\.\w+')
_A87_PHONE_RE = re.compile(r'(\+?\d{1,2}[\s\-.])?\(?\d{3}\)?[\s\-.]\d{3}[\s\-.]\d{4}')
_A87_ADDR_RE = re.compile(r'\b\d{1,5}\s+[A-Za-z0-9\.\s]{2,30}\b(street|st\.|avenue|ave\.|blvd|boulevard|suite|drive|dr\.)', re.IGNORECASE)
_A87_CONTACT_WORDS = ["contact:", "media contact", "investor relations contact",
                        "investor contact", "press contact", "for more information"]


def score__press_releases__a87(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    emails = len(_A87_EMAIL_RE.findall(text))
    phones = len(_A87_PHONE_RE.findall(text))
    addrs = len(_A87_ADDR_RE.findall(text))
    has_contact_word = any(k in tl for k in _A87_CONTACT_WORDS)

    total = emails + phones + addrs

    if total >= 2:
        return 10.0
    if total == 1 or has_contact_word:
        return 5.0
    return 0.0


# ---------------------------------------------------------------------------
# math__a42
# ---------------------------------------------------------------------------

_A42_ERROR_KW = ["incorrect", "wrong", "mistake", "error", "invalid", "false",
                  "disprove", "counterexample", "contradiction", "flaw",
                  "doesn't hold", "fails to", "not true", "fallacy"]
_A42_CORRECT_KW = ["correct", "valid", "true", "q.e.d", "hence proven",
                    "thus proven", "completes the proof", "holds", "verified"]


def score__math__a42(text):
    text = text or ""
    if not text.strip():
        return 5.0
    tl = _lower(text)
    err = _count_kw(tl, _A42_ERROR_KW)
    cor = _count_kw(tl, _A42_CORRECT_KW)

    if err == 0 and cor == 0:
        return 5.0
    if err > cor:
        return _clamp(6.0 + min(err, 4))
    if cor > err:
        return _clamp(4.0 - min(cor, 4))
    return 5.0


# ---------------------------------------------------------------------------
# code_review__a36
# ---------------------------------------------------------------------------

_A36_TRIVIAL_KW = ["remove", "add empty line", "wrong indentation", "typo",
                     "fix whitespace", "nit:", "done", "lgtm", "+1"]
_A36_DEEP_KW = ["why are these", "docstring", "architecture", "design",
                 "consider using", "instead of", "because", "in order to",
                 "this could lead to", "edge case", "performance", "refactor",
                 "explanation:"]


def score__code_review__a36(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    trivial = _count_kw(tl, _A36_TRIVIAL_KW)
    deep = _count_kw(tl, _A36_DEEP_KW)

    if deep == 0 and trivial == 0:
        return 3.0

    score = 10.0 * deep / float(deep + trivial + 1)
    return _clamp(score)


# ---------------------------------------------------------------------------
# patents__a240
# ---------------------------------------------------------------------------

_A240_CLAIMS_RE = re.compile(r'\bCLAIMS?\s*:', re.IGNORECASE)
_A240_WHAT_IS_CLAIMED_RE = re.compile(r'what is claimed is', re.IGNORECASE)
_A240_CLAIM1_RE = re.compile(r'\bclaim\s+1\b', re.IGNORECASE)
_A240_WHEREIN_RE = re.compile(r'\bwherein\b', re.IGNORECASE)


def score__patents__a240(text):
    text = text or ""
    if not text.strip():
        return 0.0
    if _A240_CLAIMS_RE.search(text) or _A240_WHAT_IS_CLAIMED_RE.search(text):
        return 10.0
    if _A240_CLAIM1_RE.search(text) and _A240_WHEREIN_RE.search(text):
        return 10.0
    return 0.0


# ---------------------------------------------------------------------------
# patents__a60
# ---------------------------------------------------------------------------

_A60_UI_KW = ["user interface", "graphical user interface", " gui ", "touchscreen",
               "touch screen", "display screen", "user-facing", "interactive display",
               "icon", "widget", "input device", "cursor", "on-screen", "user input device"]
_A60_SUPPORT_KW = ["software architecture", "imaging", "printing", "data handling",
                     "rendering", "display device", "output device"]


def score__patents__a60(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = " " + _lower(text) + " "
    ui_hits = _count_kw(tl, _A60_UI_KW)
    support_hits = _count_kw(tl, _A60_SUPPORT_KW)

    if ui_hits >= 2:
        return 10.0
    if ui_hits == 1 and support_hits >= 1:
        return 7.0
    if ui_hits == 1:
        return 6.0
    if support_hits >= 1:
        return 2.0
    return 0.0


# ---------------------------------------------------------------------------
# math__a132
# ---------------------------------------------------------------------------

_A132_CORE_KW = ["geometric", "geometry", "visualize", "visualization", "diagram",
                   "picture", "intuitively", "intuition", "shape", "angle", "triangle",
                   "circle", "sphere", "curve", "surface", "manifold",
                   "coordinate plane", "plot", "graph of"]


def score__math__a132(text):
    text = text or ""
    if not text.strip():
        return 0.0
    tl = _lower(text)
    hits = _count_kw(tl, _A132_CORE_KW)

    if hits >= 5:
        return 10.0
    if hits >= 3:
        return 6.5
    if hits >= 1:
        return 3.5
    return 0.0


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
