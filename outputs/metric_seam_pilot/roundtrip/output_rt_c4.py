# AUTO: blind rule compilation chunk c4

import re
import math
import statistics
import string
import collections


# ---------------------------------------------------------------------------
# Shared helpers (used by multiple score__ functions below)
# ---------------------------------------------------------------------------

def _safe_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    return text


def _words(text):
    return re.findall(r"[A-Za-z']+", text)


def _sentences(text):
    parts = re.split(r"[.!?]+(?:\s|$)", text)
    return [p.strip() for p in parts if p.strip()]


def _clip(x, lo=0.0, hi=10.0):
    try:
        x = float(x)
    except Exception:
        return lo
    if math.isnan(x):
        return lo
    return max(lo, min(hi, x))


def _count_kw(text_lower, keywords):
    total = 0
    for k in keywords:
        k = k.lower()
        if " " in k or "-" in k:
            total += text_lower.count(k)
        else:
            total += len(re.findall(r"\b" + re.escape(k) + r"\b", text_lower))
    return total


def _math_symbol_count(text):
    return len(re.findall(r"[=<>≤≥∑∫∂√±∞×÷]|\\[a-zA-Z]+|\$[^$]{1,200}\$|\b\d+\.\d+\b", text))


# ---------------------------------------------------------------------------
# math__a24 -- correctness/rigor/completeness of a math answer
# ---------------------------------------------------------------------------

def score__math__a24(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()
    words = _words(text)
    wc = len(words)

    rigor_kw = ["proof", "qed", "therefore", "thus", "hence", "by definition",
                "theorem", "lemma", "corollary", "we conclude", "which completes",
                "q.e.d", "as required", "as desired"]
    error_kw = ["incorrect", "mistake", "is wrong", "error", "not sure",
                "unclear", "doesn't work", "does not work", "counterexample fails",
                "i don't know", "no idea"]

    rigor_hits = _count_kw(tl, rigor_kw)
    error_hits = _count_kw(tl, error_kw)
    sym_count = _math_symbol_count(text)

    score = 5.0
    score += min(3.0, rigor_hits * 0.8)
    score += min(2.0, sym_count / 6.0)
    if wc > 150:
        score += 1.0
    elif wc < 15:
        score -= 3.0
    score -= min(4.5, error_hits * 1.6)

    return _clip(round(score, 3))


# ---------------------------------------------------------------------------
# press_releases__a204 -- UI/navigation boilerplate density
# ---------------------------------------------------------------------------

def score__press_releases__a204(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()
    words = _words(text)
    wc = max(1, len(words))

    boilerplate_kw = ["menu", "navigation", "home", "about us", "contact us",
                       "privacy policy", "terms of use", "terms & conditions",
                       "copyright", "all rights reserved", "sign in", "log in",
                       "login", "register", "subscribe", "footer", "site map",
                       "sitemap", "click here", "read more", "cookie", "follow us",
                       "search"]
    interactive_kw = ["login", "log in", "sign in", "password", "username",
                       "search", "submit", "form", "dashboard", "settings",
                       "register", "subscribe now", "enter your email"]

    bp_hits = _count_kw(tl, boilerplate_kw)
    inter_hits = _count_kw(tl, interactive_kw)

    density = bp_hits / wc * 100.0
    score = min(10.0, density * 3.0)
    if inter_hits >= 3:
        score = max(score, 8.0)
    if bp_hits == 0:
        score = 0.0

    return _clip(round(score, 3))


# ---------------------------------------------------------------------------
# patents__a216 -- primary technological field
# ---------------------------------------------------------------------------

def score__patents__a216(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    kw_10 = ["toy", "game", "sporting", "recreational", "exercise equipment",
              "fitness equipment", "household appliance", "kitchen appliance",
              "furniture", "fixture", "hinge", "utensil", "cookware", "garment",
              "footwear", "shoe", "apparel", "cleaning device", "vacuum cleaner"]
    kw_5 = ["vehicle", "automobile", "automotive", "engine", "wireless",
             "telecommunication", "cellular", "antenna", "base station",
             "fuel", "energy generation", "power plant", "turbine", "battery",
             "solar", "combustion", "network node"]
    kw_0 = ["software", "computer", "semiconductor", "integrated circuit",
             "chemical composition", "polymer", "alloy", "manufacturing process",
             "medical device", "pharmaceutical", "diagnostic", "surgical",
             "implant", "algorithm", "processor"]

    h10 = _count_kw(tl, kw_10)
    h5 = _count_kw(tl, kw_5)
    h0 = _count_kw(tl, kw_0)

    if h10 == 0 and h5 == 0 and h0 == 0:
        return 0.0
    if h10 >= h5 and h10 >= h0:
        return 10.0
    if h5 >= h10 and h5 >= h0:
        return 5.0
    return 0.0


# ---------------------------------------------------------------------------
# math__a72 -- mathematical quality/completeness of answer
# ---------------------------------------------------------------------------

def score__math__a72(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()
    words = _words(text)
    wc = len(words)

    rigor_kw = ["proof", "therefore", "thus", "hence", "theorem", "lemma",
                "by definition", "we conclude", "qed"]
    conceptual_only_kw = ["in general", "conceptually", "intuitively", "roughly"]
    error_kw = ["incorrect", "mistake", "is wrong", "error", "not sure"]

    rigor_hits = _count_kw(tl, rigor_kw)
    error_hits = _count_kw(tl, error_kw)
    sym_count = _math_symbol_count(text)

    if wc < 6:
        return _clip(1.0)

    score = 3.5
    score += min(3.5, rigor_hits * 1.1)
    score += min(2.5, sym_count / 5.0)
    if wc > 80:
        score += 0.7
    score -= min(4.0, error_hits * 1.7)
    if sym_count == 0 and _count_kw(tl, conceptual_only_kw) > 0 and rigor_hits == 0:
        score = min(score, 3.0)

    return _clip(round(score, 3))


# ---------------------------------------------------------------------------
# press_releases__a110 -- corporate/business news announcement clarity
# ---------------------------------------------------------------------------

_DATELINE_RE = re.compile(
    r"\b[A-Z][A-Za-z\.]*(?:\s[A-Z][A-Za-z\.]*)?,\s*(?:[A-Z]{2}|[A-Za-z]+)\s*[-–—]\s"
)

def score__press_releases__a110(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    announcement_kw = ["announced", "announces", "today announced", "reported",
                        "earnings", "launch", "launches", "appointed", "names",
                        "acquisition", "merger", "quarterly results",
                        "for immediate release", "press release"]
    boilerplate_kw = ["home", "about us", "contact", "navigation", "sign in",
                       "products", "services", "careers"]
    portal_kw = ["stock quote", "ticker symbol", "nasdaq:", "nyse:", ".gov",
                 "government", "municipal"]

    has_dateline = bool(_DATELINE_RE.search(text))
    ann_hits = _count_kw(tl, announcement_kw)
    bp_hits = _count_kw(tl, boilerplate_kw)
    portal_hits = _count_kw(tl, portal_kw)

    ascii_letters = sum(1 for c in text if c in string.ascii_letters)
    non_english = ascii_letters < max(1, len(text) * 0.3)

    if has_dateline and ann_hits >= 1:
        return 9.5
    if ann_hits >= 2:
        return 8.0
    if non_english or (portal_hits >= 1 and ann_hits == 0):
        return 0.2
    if bp_hits >= 2 and ann_hits == 0:
        return 6.0
    if ann_hits == 0 and bp_hits == 0 and portal_hits == 0:
        return 0.3
    return 3.0


# ---------------------------------------------------------------------------
# math__a84 -- mathematical sophistication / concept complexity
# ---------------------------------------------------------------------------

def score__math__a84(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    advanced_kw = ["topology", "topological", "manifold", "differential geometry",
                    "abstract algebra", "group theory", "ring theory",
                    "field theory", "galois", "functional analysis",
                    "measure theory", "complex analysis", "holomorphic",
                    "banach", "hilbert space", "homomorphism", "homeomorphism",
                    "sigma-algebra", "lebesgue", "stochastic process",
                    "martingale", "representation theory", "algebraic geometry",
                    "category theory", "lie algebra", "lie group"]
    undergrad_kw = ["derivative", "integral", "matrix", "eigenvalue",
                     "vector space", "linear algebra", "calculus",
                     "differential equation", "limit", "continuity",
                     "real analysis", "induction", "taylor series"]
    elementary_kw = ["addition", "subtraction", "multiplication", "division",
                       "fraction", "percentage", "arithmetic", "algorithm",
                       "programming", "combinatorics", "puzzle", "elementary",
                       "high school"]

    adv = _count_kw(tl, advanced_kw)
    ug = _count_kw(tl, undergrad_kw)
    elem = _count_kw(tl, elementary_kw)

    if adv > 0:
        return _clip(round(7.0 + min(3.0, adv * 0.6), 3))
    if ug > 0:
        return _clip(round(3.0 + min(3.9, ug * 0.7), 3))
    if elem > 0:
        return _clip(round(min(2.9, elem * 0.5), 3))
    return _clip(1.5)


# ---------------------------------------------------------------------------
# press_releases__a67 -- animal-welfare terminology density/prominence
# ---------------------------------------------------------------------------

def score__press_releases__a67(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    terms = ["animal", "welfare", "humane", "slaughter", "abattoir",
             "supply chain", "cruelty", "livestock", "factory farm",
             "cage-free", "free-range", "animal rights", "inhumane",
             "animal cruelty"]

    total_hits = _count_kw(tl, terms)
    opening = tl[:250]
    opening_hits = _count_kw(opening, terms)

    if total_hits == 0:
        return 0.0
    if opening_hits >= 1 or total_hits >= 6:
        return 10.0
    if total_hits >= 3:
        return _clip(6.0 + min(2.0, (total_hits - 3) * 0.5))
    return _clip(1.0 + min(1.0, total_hits * 0.5))


# ---------------------------------------------------------------------------
# math__a12 -- mathematical maturity/rigor/advanced level
# ---------------------------------------------------------------------------

def score__math__a12(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    advanced_kw = ["abstract algebra", "functional analysis", "topology",
                    "measure theory", "differential geometry", "manifold",
                    "banach", "hilbert space", "sigma-algebra", "homomorphism",
                    "galois", "ring theory", "group theory"]
    undergrad_kw = ["multivariable calculus", "linear algebra", "real analysis",
                     "differential equation", "eigenvalue", "vector space",
                     "matrix", "derivative", "integral"]
    elementary_kw = ["arithmetic", "high school", "elementary", "basic algebra",
                       "addition", "subtraction", "multiplication", "division"]

    proof_kw = ["proof", "theorem", "lemma", "corollary", "rigorous",
                "formal proof", "qed"]

    adv = _count_kw(tl, advanced_kw)
    ug = _count_kw(tl, undergrad_kw)
    elem = _count_kw(tl, elementary_kw)
    proof_hits = _count_kw(tl, proof_kw)

    if adv > 0:
        return _clip(round(8.0 + min(2.0, adv * 0.4 + proof_hits * 0.2), 3))
    if ug > 0:
        return _clip(round(4.0 + min(3.0, ug * 0.5 + proof_hits * 0.3), 3))
    if elem > 0:
        return _clip(round(min(3.0, elem * 0.5), 3))
    return _clip(1.5)


# ---------------------------------------------------------------------------
# patents__a204 -- ABSTRACT contains "The present invention" (or similar)
# ---------------------------------------------------------------------------

def score__patents__a204(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0

    m = re.search(r"ABSTRACT\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z \-]{3,}\n|\Z)",
                  text, re.DOTALL)
    section = m.group(1) if m else text

    if re.search(r"the present invention", section, re.IGNORECASE):
        return 10.0
    return 0.0


# ---------------------------------------------------------------------------
# press_releases__a101 -- standard press release format match
# ---------------------------------------------------------------------------

def score__press_releases__a101(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    has_dateline = bool(_DATELINE_RE.search(text))
    has_quote = bool(re.search(r'"[^"]{15,}"', text)) or bool(re.search(r"\bsaid\b", tl))
    footer_kw = ["news provided by", "terms of use", "site map", "sitemap",
                 "copyright", "all rights reserved", "privacy policy"]
    ann_kw = ["announced", "announces", "today announced", "launch", "launches",
              "appointed", "acquisition", "merger"]
    nav_kw = ["home", "products", "services", "login", "sign in",
              "stock quote", "nasdaq:", "nyse:"]

    has_footer = _count_kw(tl, footer_kw) >= 1
    has_ann = _count_kw(tl, ann_kw) >= 1
    nav_hits = _count_kw(tl, nav_kw)

    features = sum([has_dateline, has_quote, has_footer, has_ann])

    if features == 0 and nav_hits >= 2:
        return 0.0
    if features == 0:
        return 0.5

    score = 10.0 * (features / 4.0)
    return _clip(round(score, 3))


# ---------------------------------------------------------------------------
# code_review__a144 -- depth of technical design discussion
# ---------------------------------------------------------------------------

def score__code_review__a144(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()
    words = _words(text)

    design_kw = ["architecture", "design", "trade-off", "tradeoff",
                 "alternative approach", "api stability", "scalability",
                 "refactor", "abstraction", "interface design",
                 "backward compat", "performance implications", "complexity",
                 "maintainability", "extensibility"]
    brief_kw = ["done", "fixed", "lgtm", "nit:", "typo", "looks good", "+1", "ok"]

    design_hits = _count_kw(tl, design_kw)
    brief_hits = _count_kw(tl, brief_kw)
    sentences = _sentences(text)
    long_sentences = sum(1 for s in sentences if len(_words(s)) > 12)

    if design_hits == 0:
        return 0.0
    score = min(10.0, design_hits * 2.2 + long_sentences * 0.4)
    if brief_hits > design_hits * 2 and design_hits <= 1:
        score = min(score, 4.0)

    return _clip(round(score, 3))


# ---------------------------------------------------------------------------
# code_review__a279 -- technical depth/substantive value of review
# ---------------------------------------------------------------------------

def score__code_review__a279(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    technical_kw = ["architecture", "architectural", "edge case", "race condition",
                     "performance", "complexity", "big o", "memory leak",
                     "thread safety", "concurrency", "deep dive", "benchmark",
                     "trade-off", "tradeoff", "scalability", "algorithm"]
    trivial_kw = ["typo", "whitespace", "naming convention", "blank line",
                  "style violation", "nit", "rename variable", "indentation"]

    tech_hits = _count_kw(tl, technical_kw)
    triv_hits = _count_kw(tl, trivial_kw)

    score = 4.5 + tech_hits * 1.3 - triv_hits * 0.8
    if tech_hits >= 3:
        score = max(score, 7.0)
    if tech_hits == 0 and triv_hits > 0:
        score = min(score, 3.0)
    if tech_hits == 0 and triv_hits == 0:
        score = 4.0

    return _clip(round(score, 3))


# ---------------------------------------------------------------------------
# math__a36 -- mathematical quality and pedagogical clarity
# ---------------------------------------------------------------------------

def score__math__a36(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()
    words = _words(text)
    wc = len(words)

    rigor_kw = ["proof", "therefore", "thus", "hence", "theorem", "lemma",
                "by definition", "qed"]
    clarity_kw = ["because", "note that", "in other words", "let's clarify",
                  "to clarify", "misconception", "actually", "the reason",
                  "the key idea", "this is because"]
    error_kw = ["incorrect", "mistake", "is wrong", "error", "unclear"]

    rigor_hits = _count_kw(tl, rigor_kw)
    clarity_hits = _count_kw(tl, clarity_kw)
    error_hits = _count_kw(tl, error_kw)
    sym_count = _math_symbol_count(text)

    if wc < 8:
        return _clip(1.5)

    score = 3.0
    score += min(3.0, rigor_hits * 1.0)
    score += min(2.5, clarity_hits * 0.9)
    score += min(1.5, sym_count / 8.0)
    score -= min(4.0, error_hits * 1.6)
    if wc < 20 and rigor_hits == 0 and clarity_hits == 0:
        score = min(score, 3.5)

    return _clip(round(score, 3))


# ---------------------------------------------------------------------------
# code_review__a54 -- explicit acknowledgment of resolved changes
# ---------------------------------------------------------------------------

def score__code_review__a54(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    ack_kw = ["fixed", "done", "verified", "removed", "updated", "resolved",
              "addressed", "implemented", "corrected"]
    for k in ack_kw:
        if re.search(r"\b" + re.escape(k) + r"\b", tl):
            return 10.0
    return 0.0


# ---------------------------------------------------------------------------
# math__a96 -- depth/conceptual richness of the question
# ---------------------------------------------------------------------------

def score__math__a96(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()
    words = _words(text)
    wc = len(words)

    exploratory_kw = ["generalize", "generalization", "intuition", "why does",
                       "what if", "is there a", "conjecture", "open problem",
                       "explore", "deeper understanding", "alternative proof",
                       "different approach", "interpret", "meaning of",
                       "motivation behind", "is it true that"]
    routine_kw = ["solve for", "compute", "calculate", "find the value",
                  "evaluate", "homework", "simplify", "what is the derivative of",
                  "prove that"]

    exp_hits = _count_kw(tl, exploratory_kw)
    rout_hits = _count_kw(tl, routine_kw)
    qmarks = text.count("?")

    if exp_hits >= 2:
        return _clip(round(7.0 + min(3.0, exp_hits * 0.6), 3))
    if exp_hits == 1:
        return _clip(round(5.5 + min(1.4, qmarks * 0.2), 3))
    if rout_hits >= 1 or wc < 25:
        return _clip(round(min(3.9, 1.0 + rout_hits * 0.3), 3))
    return _clip(4.5)


# ---------------------------------------------------------------------------
# patents__a54 -- computer science / software / computing architecture strength
# ---------------------------------------------------------------------------

def score__patents__a54(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    core_kw = ["processor", "memory", "operating system", "machine learning",
                "neural network", "algorithm", "data processing",
                "computing architecture", "central processing unit", "software",
                "computer program", "instructions stored"]
    applied_kw = ["navigation system", "communication system", "control system",
                   "data management", "wireless communication", "gps",
                   "autonomous"]
    physical_proc_kw = ["controller", "embedded controller", "user interface",
                          "graphical user interface", "business method"]
    incidental_kw = ["agricultural", "biological", "vehicle", "mechanical"]

    core = _count_kw(tl, core_kw)
    applied = _count_kw(tl, applied_kw)
    phys = _count_kw(tl, physical_proc_kw)
    incid = _count_kw(tl, incidental_kw)

    if core > 0:
        return _clip(round(9.0 + min(1.0, core * 0.15), 3))
    if applied > 0:
        return _clip(round(7.0 + min(1.0, applied * 0.25), 3))
    if phys > 0:
        return _clip(round(5.0 + min(1.0, phys * 0.3), 3))
    if incid > 0:
        return _clip(round(2.0 + min(2.0, incid * 0.4), 3))
    return _clip(0.5)


# ---------------------------------------------------------------------------
# press_releases__a28 -- standalone corporate press release strength
# ---------------------------------------------------------------------------

def score__press_releases__a28(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    has_dateline = bool(_DATELINE_RE.search(text))
    has_fir = "for immediate release" in tl
    nav_kw = ["menu", "navigation", "sign in", "login", "site map", "sitemap",
              "stock quote", "ticker", "footer", "search"]
    nav_hits = _count_kw(tl, nav_kw)
    words = _words(text)
    sentences = _sentences(text)
    avg_sent_len = (len(words) / len(sentences)) if sentences else 0

    if (has_dateline or has_fir) and avg_sent_len > 8:
        return _clip(round(8.5 + (1.5 if has_dateline and has_fir else 0.0), 3))
    if nav_hits >= 3 and not has_dateline and not has_fir:
        return _clip(round(max(0.0, 2.5 - nav_hits * 0.1), 3))
    if has_dateline or has_fir or (avg_sent_len > 10 and nav_hits < 2):
        return _clip(round(4.0 + min(3.0, (2 if has_dateline else 0) + (2 if has_fir else 0)), 3))
    return _clip(2.0)


# ---------------------------------------------------------------------------
# math__a78 -- addresses core conceptual misunderstanding vs. pure computation
# ---------------------------------------------------------------------------

def score__math__a78(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    conceptual_kw = ["misconception", "confusion", "the issue is", "the reason is",
                       "note that", "actually", "the mistake here", "conceptually",
                       "intuitively", "in other words", "to clarify",
                       "the key insight", "this is because", "misunderstand"]

    conceptual_hits = _count_kw(tl, conceptual_kw)

    if conceptual_hits >= 2:
        return 10.0
    if conceptual_hits == 1:
        return 5.0
    return 0.0


# ---------------------------------------------------------------------------
# press_releases__a146 -- density of hyperlinks/interactive UI elements
# ---------------------------------------------------------------------------

def score__press_releases__a146(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()
    words = _words(text)
    wc = max(1, len(words))

    link_kw = ["click here", "read more", "learn more", "sign in", "register",
               "search", "menu", "subscribe", "contact", "site map", "sitemap"]
    url_hits = len(re.findall(r"https?://\S+|www\.\S+", text))

    kw_hits = _count_kw(tl, link_kw)
    density = (kw_hits + url_hits) / wc * 100.0

    score = min(10.0, density * 2.5)
    return _clip(round(score, 3))


# ---------------------------------------------------------------------------
# patents__a72 -- technical specificity of disclosed subject matter
# ---------------------------------------------------------------------------

def score__patents__a72(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    numeric_units = len(re.findall(
        r"\d+(\.\d+)?\s?(mm|cm|kg|g|%|°c|v|hz|nm|mol|psi|degrees|inches|lbs)", tl))
    chem_formula = len(re.findall(r"\b[A-Z][a-z]?\d+[A-Za-z0-9]*\b", text))
    reference_numerals = len(re.findall(r"\(\d{1,3}\)", text))
    structural_kw = ["comprising", "wherein said", "step of", "coupled to",
                       "affixed to", "disposed within"]
    structural_hits = _count_kw(tl, structural_kw)

    vague_kw = ["generally", "broadly", "may include", "various embodiments",
                "any suitable", "in general", "some implementations"]
    vague_hits = _count_kw(tl, vague_kw)

    specific_count = numeric_units + reference_numerals + structural_hits + min(3, chem_formula)

    if specific_count >= 3:
        return 10.0
    if specific_count >= 1:
        return 5.0
    if vague_hits >= 1 or len(_words(text)) < 15:
        return 0.0
    return 2.5


# ---------------------------------------------------------------------------
# press_releases__a291 -- personal investing/opinion piece vs. corporate content
# ---------------------------------------------------------------------------

def score__press_releases__a291(text):
    text = _safe_text(text)
    if not text.strip():
        return 0.0
    tl = text.lower()

    personal_kw = ["i bought", "i sold", "i invested", "my portfolio",
                    "my holdings", "in my opinion", "my strategy",
                    "passive income", "my dividend", "my retirement",
                    "i've been investing", "i plan to", "my investment"]
    corporate_kw = ["announced today", "press release", "the company today",
                      "for immediate release", "quarterly results", "nasdaq:",
                      "nyse:", "inc.", "corp."]

    personal_hits = _count_kw(tl, personal_kw)
    corporate_hits = _count_kw(tl, corporate_kw)

    if personal_hits >= 1 and personal_hits > corporate_hits:
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
