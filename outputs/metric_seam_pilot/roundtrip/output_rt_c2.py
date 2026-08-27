# AUTO: blind rule compilation chunk c2

import re


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SUBSTANTIVE_KEYWORDS = [
    'why', 'because', 'consider', 'what if', 'what about', 'could we', 'should we', 'design',
    'architecture', 'api', 'behavior', 'behaviour', 'test', 'testing', 'edge case', 'performance',
    'race condition', 'thread', 'concurrency', 'security', 'memory leak', 'complexity',
    'maintainab', 'alternative', 'instead of', 'trade-off', 'tradeoff', 'propose', 'suggestion',
    'i think', 'i wonder', 'does this', 'is there a reason', "won't this", "wouldn't this",
    'what happens if', 'how does', 'question', 'concerned', 'issue is', 'problem is',
    'reasoning', 'rationale', 'explain', 'clarify',
]

TRIVIAL_KEYWORDS = [
    'nit:', 'nit ', 'typo', 'rename', 'formatting', 'format', 'style', 'whitespace', 'indent',
    'lgtm', 'looks good', 'thanks', 'thank you', 'done', 'fixed', 'ok', 'okay', '+1', 'ditto',
    'remove this', 'delete this', 'add license', 'unused import', 'lint',
]

IMPERATIVE_VERBS = [
    'remove', 'use', 'fix', 'add', 'rename', 'delete', 'change', 'update', 'move',
    'extract', 'avoid', 'replace', 'revert', 'wrap', 'initialize', 'check', 'ensure', 'make',
    'set', 'call', 'return', 'drop', 'split', 'inline', 'rewrite', 'simplify', 'refactor',
]


def _split_comments(text):
    if not text or not text.strip():
        return []
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(paras) >= 2:
        return paras
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines if lines else [text.strip()]


def _is_substantive(comment):
    cl = comment.lower()
    words = comment.split()
    if any(k in cl for k in TRIVIAL_KEYWORDS) and len(words) <= 6:
        return False
    if any(k in cl for k in SUBSTANTIVE_KEYWORDS):
        return True
    if '?' in comment and len(words) >= 6:
        return True
    if len(words) >= 20:
        return True
    return False


def _is_trivial_comment(comment):
    cl = comment.lower().strip()
    trivial_exact = ['fixed', 'done', 'remove', 'ditto', 'ok', 'okay', '+1', 'nit', 'nit:']
    if cl in trivial_exact:
        return True
    if len(comment.split()) <= 3:
        return True
    return any(k in cl for k in TRIVIAL_KEYWORDS)


def _is_blunt(comment):
    words = comment.split()
    if len(words) <= 3:
        return True
    cl = comment.lower().strip()
    blunt_starts = ['fixed', 'remove', 'rename', 'delete', 'fix this', 'change this']
    return any(cl.startswith(b) for b in blunt_starts) and len(words) <= 5


def _is_direct_request(comment):
    c = comment.strip()
    if not c:
        return False
    m = re.match(r"^[\"'\-\*\s]*([A-Za-z']+)", c)
    if not m:
        return False
    fw = m.group(1).lower()
    if fw in IMPERATIVE_VERBS:
        return True
    if re.match(
        r"^(please\s+)?(remove|use|fix|add|rename|delete|change|update|move|extract|avoid|"
        r"don'?t|replace|revert|wrap|initialize|check|ensure|make|set|call|return)\b",
        c.lower(),
    ):
        return True
    return False


_MONTH = (
    r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
    r'Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
)


# ---------------------------------------------------------------------------
# press_releases__a86
# ---------------------------------------------------------------------------

def score__press_releases__a86(text):
    if not text or not text.strip():
        return 0.0
    tl = text.lower()
    dateline_re = re.compile(
        r"\b[A-Z][A-Za-z\.\-' ]{2,35},\s*(?:[A-Z]{2}|[A-Za-z\.]{3,20}),?\s+"
        + _MONTH + r"\.?\s+\d{1,2},?\s+\d{4}"
    )
    has_dateline = bool(dateline_re.search(text))
    boilerplate_markers = [
        'cision', 'prnewswire', 'pr newswire', '888-776-0942',
        'online member center', 'contact cision', 'a cision company',
    ]
    has_boilerplate = any(m in tl for m in boilerplate_markers)
    if has_dateline and has_boilerplate:
        return 10.0
    if has_dateline:
        return 3.3
    return 0.0


# ---------------------------------------------------------------------------
# press_releases__a42
# ---------------------------------------------------------------------------

def score__press_releases__a42(text):
    if not text or not text.strip():
        return 0.0
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    nav_keywords = [
        'home', 'about us', 'about', 'contact us', 'contact', 'privacy policy',
        'terms of use', 'terms of service', 'sitemap', 'site map', '© ', 'copyright',
        'all rights reserved', 'menu', 'search', 'subscribe', 'follow us', 'sign in',
        'log in', 'login', 'cookie', 'skip to content', 'share this', 'facebook',
        'twitter', 'linkedin', 'instagram', 'youtube', 'careers', 'investor relations',
        'faq', 'help center',
    ]

    def is_boiler(line):
        ll = line.lower()
        if any(kw in ll for kw in nav_keywords):
            return True
        if re.match(r"^([\w &/'-]{1,24}\s*[|>\u00b7\u2022]\s*){2,}[\w &/'-]{0,24}$", line):
            return True
        if len(line) <= 20 and re.match(r"^[A-Z][a-zA-Z&' ]*$", line) and len(line.split()) <= 3:
            return True
        return False

    total_chars = sum(len(l) for l in lines)
    if total_chars == 0:
        return 0.0
    boiler_chars = sum(len(l) for l in lines if is_boiler(l))
    ratio = boiler_chars / total_chars
    return round(min(10.0, max(0.0, ratio * 10)), 1)


# ---------------------------------------------------------------------------
# press_releases__a64
# ---------------------------------------------------------------------------

def score__press_releases__a64(text):
    if not text or not text.strip():
        return 0.0
    tl = text.lower()
    health_kw = [
        'health', 'healthcare', 'health care', 'medical', 'medicine', 'patient', 'physician',
        'doctor', 'hospital', 'clinic', 'clinical', 'treatment', 'therapy', 'disease',
        'diagnosis', 'pharmaceutical', 'drug', 'vaccine', 'wellness', 'fitness', 'sleep',
        'mental health', 'veterinary', 'animal health', 'symptom', 'surgery', 'nurse',
        'fda approval', 'pharma', 'oncology', 'cardiology', 'diabetes', 'cancer',
    ]
    other_kw = [
        'revenue', 'earnings', 'stock price', 'shares', 'technology platform', 'software',
        'election', 'senate', 'congress', 'vehicle', 'automotive', 'consumer electronics',
        'retail sales', 'real estate', 'financial results', 'quarterly report',
    ]
    nav_kw = ['sign in', 'log in', '404', 'page not found', 'navigation menu', 'site map']
    if any(kw in tl for kw in nav_kw) and not any(kw in tl for kw in health_kw):
        return 0.0
    health_count = sum(tl.count(kw) for kw in health_kw)
    other_count = sum(tl.count(kw) for kw in other_kw)
    if health_count >= 2 and health_count >= other_count:
        return 10.0
    return 0.0


# ---------------------------------------------------------------------------
# math__a120
# ---------------------------------------------------------------------------

def score__math__a120(text):
    if not text or not text.strip():
        return 0.0
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return 0.0
    question_sents = [s for s in sentences if s.endswith('?')]
    ratio = len(question_sents) / len(sentences)
    tl = text.lower()
    hint_kw = [
        'what do you think', 'try to', 'can you', 'hint:', 'what if', 'consider what',
        'think about', 'what would happen', 'how might',
    ]
    hint_bonus = 0.5 if any(k in tl for k in hint_kw) else 0.0
    score = ratio * 10 + hint_bonus
    return round(min(10.0, max(0.0, score)), 1)


# ---------------------------------------------------------------------------
# code_review__a216
# ---------------------------------------------------------------------------

def score__code_review__a216(text):
    comments = _split_comments(text)
    if not comments:
        return 0.0
    subst = sum(1 for c in comments if _is_substantive(c))
    return round(10.0 * subst / len(comments), 1)


# ---------------------------------------------------------------------------
# code_review__a0
# ---------------------------------------------------------------------------

def score__code_review__a0(text):
    comments = _split_comments(text)
    if not comments:
        return 0.0
    total_words = sum(len(c.split()) for c in comments)
    avg_words = total_words / len(comments)
    subst = sum(1 for c in comments if _is_substantive(c))
    subst_ratio = subst / len(comments)
    blunt = sum(1 for c in comments if _is_blunt(c))
    blunt_ratio = blunt / len(comments)
    code_blocks = text.count('```')
    tl = text.lower()
    polite_kw = ['please', 'thanks', 'thank you', 'could you', 'would you', 'i think', 'maybe',
                 'suggest', 'consider']
    polite_count = sum(tl.count(k) for k in polite_kw)
    score = (
        3.0
        + subst_ratio * 5.0
        + min(1.5, avg_words / 20.0)
        + min(1.0, code_blocks * 0.3)
        + min(0.5, polite_count * 0.1)
        - blunt_ratio * 3.0
    )
    return round(min(10.0, max(0.0, score)), 1)


# ---------------------------------------------------------------------------
# math__a174
# ---------------------------------------------------------------------------

def score__math__a174(text):
    if not text or not text.strip():
        return 0.0
    tl = text.lower()
    verify_kw = [
        'verify my proof', 'check my proof', 'is this proof correct', 'is my proof correct',
        'check this proof', 'verify this proof', 'is this correct', 'critique my proof',
        'verify the following proof', 'please check my proof', 'is my proof valid',
        'proof attempt', 'my attempt at a proof', 'is this a valid proof',
    ]
    if not any(k in tl for k in verify_kw):
        return 0.0
    incomplete_kw = [
        "i'm stuck", 'i am stuck', 'not sure how to proceed', 'how do i continue',
        "don't know how to proceed", 'do not know how to proceed', 'got stuck', 'incomplete',
        "i'm not sure", 'i am not sure', 'how to finish',
    ]
    if any(k in tl for k in incomplete_kw):
        return 5.0
    conclude_kw = [
        'therefore', 'hence', 'qed', 'thus', 'this completes the proof',
        'which proves', 'as required', '\u220e', 'q.e.d',
    ]
    has_conclusion = any(k in tl for k in conclude_kw)
    latex_markers = len(re.findall(r'\$\$|\\begin\{align|\\begin\{equation|\\\[|\$[^$\n]+\$', text))
    if has_conclusion and latex_markers >= 3:
        return 10.0
    if has_conclusion:
        return 7.5
    return 5.0


# ---------------------------------------------------------------------------
# math__a60
# ---------------------------------------------------------------------------

def score__math__a60(text):
    if not text or not text.strip():
        return 0.0
    frac_count = len(re.findall(r'\\frac\s*\{', text))
    slash_count = 0
    for line in text.splitlines():
        if 'http://' in line or 'https://' in line or 'www.' in line:
            continue
        slash_count += len(re.findall(r'(?<![/\w])[\w\)\}]+\s*/\s*[\w\(\{]+(?![/\w])', line))
    total = frac_count * 2 + slash_count
    if total <= 0:
        return 0.0
    return round(min(10.0, 2.0 + total * 1.2), 1)


# ---------------------------------------------------------------------------
# code_review__a117
# ---------------------------------------------------------------------------

def score__code_review__a117(text):
    comments = _split_comments(text)
    if not comments:
        return 0.0
    direct = sum(1 for c in comments if _is_direct_request(c))
    return round(10.0 * direct / len(comments), 1)


# ---------------------------------------------------------------------------
# math__a108
# ---------------------------------------------------------------------------

def score__math__a108(text):
    if not text:
        return 0.0
    n = len(text)
    has_elision = '[...]' in text or '[\u2026]' in text
    if not has_elision:
        if n < 2000:
            return 0.0
        return round(min(10.0, 5.0 + (n - 2000) / 500.0), 1)
    if n < 2000:
        return round(max(0.0, min(4.9, n / 2000.0 * 5.0)), 1)
    if n < 2500:
        return round(5.0 + (n - 2000) / 500.0 * 1.9, 1)
    if n < 3000:
        return round(7.0 + (n - 2500) / 500.0 * 1.9, 1)
    return round(min(10.0, 9.0 + min(1.0, (n - 3000) / 2000.0)), 1)


# ---------------------------------------------------------------------------
# code_review__a90
# ---------------------------------------------------------------------------

def score__code_review__a90(text):
    comments = _split_comments(text)
    if not comments:
        return 0.0
    subst = sum(1 for c in comments if _is_substantive(c))
    subst_ratio = subst / len(comments)
    avg_words = sum(len(c.split()) for c in comments) / len(comments)
    trivial = sum(1 for c in comments if _is_trivial_comment(c))
    trivial_ratio = trivial / len(comments)
    score = subst_ratio * 7.0 + min(2.0, avg_words / 15.0) - trivial_ratio * 3.0 + 1.0
    return round(min(10.0, max(0.0, score)), 1)


# ---------------------------------------------------------------------------
# press_releases__a111
# ---------------------------------------------------------------------------

def score__press_releases__a111(text):
    if not text or not text.strip():
        return 0.0
    tl = text.lower()
    full_dateline_re = re.compile(
        r"\b[A-Z][A-Za-z\.\-' ]{2,35},\s*[A-Z]{2},?\s+" + _MONTH +
        r"\.?\s+\d{1,2},?\s+\d{4}\s*(?:/\s*\w[\w\s]*\w\s*/|--)"
    )
    partial_dateline_re = re.compile(
        r"\b[A-Z][A-Za-z\.\-' ]{2,35},\s*[A-Z]{2},?\s+" + _MONTH +
        r"\.?\s+\d{1,2},?\s+\d{4}"
    )
    wire_markers = [
        'prnewswire', 'pr newswire', 'businesswire', 'business wire', 'globe newswire',
        'globenewswire', 'cision', 'ap news', 'reuters',
    ]
    if full_dateline_re.search(text):
        if any(m in tl for m in wire_markers):
            return 10.0
        return 9.0
    if partial_dateline_re.search(text):
        return 8.0
    corp_markers = [
        'privacy policy', 'terms of use', 'terms of service', 'sitemap', 'site map',
        'menu', 'navigation', 'copyright', '© ', 'all rights reserved', 'footer',
    ]
    if any(m in tl for m in corp_markers):
        return 2.0
    return 0.0


# ---------------------------------------------------------------------------
# press_releases__a112
# ---------------------------------------------------------------------------

def score__press_releases__a112(text):
    if not text or not text.strip():
        return 0.0
    tl = text.lower()
    non_ascii_ratio = sum(1 for ch in text if ord(ch) > 127) / max(1, len(text))
    if non_ascii_ratio > 0.3:
        return 0.0
    nav_markers = ['404', 'page not found', 'home | about', 'skip to content', 'site map']
    if any(m in tl for m in nav_markers) and len(text.split()) < 60:
        return 0.0
    dateline_re = re.compile(
        r"\b[A-Z][A-Za-z\.\-' ]{2,35},\s*(?:[A-Z]{2}|[A-Za-z]{3,20}),?\s+" + _MONTH +
        r"\.?\s+\d{1,2},?\s+\d{4}"
    )
    has_dateline = bool(dateline_re.search(text))
    has_provider = 'news provided by' in tl or 'source:' in tl
    narrative_kw = [
        'today announced', 'announced today', 'is pleased to announce',
        'acquisition', 'acquires', 'appointed', 'appoints', 'financial results',
        'quarterly results', 'launch of', 'launches', 'unveils', 'partnership with',
    ]
    has_narrative = any(k in tl for k in narrative_kw)
    if has_dateline and (has_provider or has_narrative):
        return 9.0
    if has_dateline or has_provider or has_narrative:
        return 6.0
    return 1.5


# ---------------------------------------------------------------------------
# CAL__CAL6
# ---------------------------------------------------------------------------

def score__CAL__CAL6(text):
    if not text:
        return 0.0
    count = len(re.findall(r'\bthe\b', text, flags=re.IGNORECASE))
    if count > 10:
        return 10.0
    if count >= 3:
        return 5.0
    return 0.0


# ---------------------------------------------------------------------------
# patents__a90
# ---------------------------------------------------------------------------

def score__patents__a90(text):
    if not text or not text.strip():
        return 0.0
    tl = text.lower()
    tier10_kw = [
        'integrated circuit', 'semiconductor', 'transistor', 'power conversion',
        'diode', 'capacitor', 'resistor', 'circuitry', 'circuit board', 'led',
        'display panel', 'voltage regulator', 'silicon wafer', 'doping',
        'fabrication of a semiconductor',
    ]
    tier8_kw = [
        'processor', 'memory cache', 'cache memory', 'cpu', 'microprocessor',
        'camera module', 'image sensor', 'computing hardware', 'motherboard', 'ram',
        'rom', 'field-programmable gate array', 'fpga', 'central processing unit',
    ]
    tier4_kw = [
        'software', 'user interface', 'communication protocol', 'graphical user interface',
        'application program', 'network protocol', 'algorithm', 'computer program',
        'printing device',
    ]
    tier0_kw = [
        'chemical compound', 'pharmaceutical composition', 'mechanical linkage',
        'biological sample', 'polymer', 'gear assembly', 'business method',
        'financial transaction', 'catalyst', 'organic compound',
    ]

    def count_kw(kws):
        return sum(tl.count(k) for k in kws)

    c10 = count_kw(tier10_kw)
    c8 = count_kw(tier8_kw)
    c4 = count_kw(tier4_kw)
    c0 = count_kw(tier0_kw)
    if c10 > 0 and c10 >= c8 and c10 >= c4:
        return round(min(10.0, 8.0 + c10 * 0.3), 1)
    if c8 > 0 and c8 >= c4:
        return round(min(8.0, 6.0 + c8 * 0.3), 1)
    if c4 > 0:
        return round(min(4.0, 2.0 + c4 * 0.2), 1)
    return 0.0


# ---------------------------------------------------------------------------
# code_review__a18
# ---------------------------------------------------------------------------

def score__code_review__a18(text):
    comments = _split_comments(text)
    if not comments:
        return 0.0
    subst = sum(1 for c in comments if _is_substantive(c))
    ratio = subst / len(comments)
    if ratio >= 0.9:
        return 10.0
    if ratio >= 0.7:
        return 8.0
    if ratio >= 0.4:
        return round(6.0 + (ratio - 0.4) / 0.3 * 1.5, 1)
    if ratio > 0.05:
        return round(1.0 + ratio * 5.0, 1)
    return 0.0


# ---------------------------------------------------------------------------
# code_review__a63
# ---------------------------------------------------------------------------

def score__code_review__a63(text):
    comments = _split_comments(text)
    if not comments:
        return 0.0
    subst = sum(1 for c in comments if _is_substantive(c))
    ratio = subst / len(comments)
    avg_words = sum(len(c.split()) for c in comments) / len(comments)
    dialogue = text.count('>') + text.count('Reply') + text.count('reply to')
    if ratio >= 0.7 and avg_words >= 25:
        return round(min(10.0, 9.0 + min(1.0, dialogue * 0.05)), 1)
    if ratio >= 0.4:
        return round(6.0 + min(2.0, ratio * 2.0), 1)
    if ratio >= 0.15 or avg_words >= 10:
        return round(3.0 + min(2.0, avg_words / 20.0), 1)
    return round(min(2.0, ratio * 10), 1)


# ---------------------------------------------------------------------------
# press_releases__a175
# ---------------------------------------------------------------------------

def score__press_releases__a175(text):
    if not text or not text.strip():
        return 0.0
    tl = text.lower()
    has_copyright = bool(re.search(r'©|\(c\)\s*\d{0,4}|copyright\s*(?:©|\(c\))?\s*\d{0,4}', tl))
    legal_kw = [
        'privacy policy', 'terms of use', 'terms of service', 'accessibility',
        'site map', 'sitemap', 'cookie policy', 'legal notice', 'terms & conditions',
        'terms and conditions',
    ]
    legal_count = sum(1 for k in legal_kw if k in tl)
    if has_copyright and legal_count >= 2:
        return 10.0
    if has_copyright and legal_count == 1:
        return 7.0
    if legal_count >= 2:
        return 6.5
    if has_copyright:
        return 5.0
    if legal_count == 1:
        return 3.0
    return 0.0


# ---------------------------------------------------------------------------
# math__a48
# ---------------------------------------------------------------------------

def score__math__a48(text):
    if not text or not text.strip():
        return 0.0
    rendered_re = re.compile(
        r'\\begin\{CD\}.*?\\end\{CD\}|\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}|'
        r'\\begin\{tikzcd\}.*?\\end\{tikzcd\}',
        re.DOTALL,
    )
    if rendered_re.search(text):
        return 10.0
    tl = text.lower()
    mention_kw = [
        'commutative diagram', 'exact sequence', 'diagram below', 'diagram above',
        'the following diagram', 'diagram chase', 'short exact sequence',
    ]
    if any(k in tl for k in mention_kw):
        return 5.0
    return 0.0


# ---------------------------------------------------------------------------
# math__a216
# ---------------------------------------------------------------------------

def score__math__a216(text):
    if not text or not text.strip():
        return 0.0
    tl = text.lower()
    exclude_kw = [
        'give an example', 'give me an example', 'can you give an example',
        'just a hint', 'give me a hint', 'without giving the full',
    ]
    if any(k in tl for k in exclude_kw):
        return 0.0
    latex_markers = (
        len(re.findall(r'\$\$', text))
        + len(re.findall(r'\\begin\{align', text))
        + len(re.findall(r'\\\[', text))
        + len(re.findall(r'\$[^$\n]+\$', text))
    )
    proof_kw = [
        'proof', 'derive', 'derivation', 'therefore', 'qed', 'hence', 'thus',
        'we conclude', 'q.e.d',
    ]
    has_proof_lang = any(k in tl for k in proof_kw)
    if latex_markers >= 3 and has_proof_lang:
        return 10.0
    return 0.0


# ---------------------------------------------------------------------------
# code_review__a297
# ---------------------------------------------------------------------------

def score__code_review__a297(text):
    comments = _split_comments(text)
    if not comments:
        return 0.0
    subst = sum(1 for c in comments if _is_substantive(c))
    return round(10.0 * (subst / len(comments)), 1)


JOB_IDS = [
    "press_releases__a86",
    "press_releases__a42",
    "press_releases__a64",
    "math__a120",
    "code_review__a216",
    "code_review__a0",
    "math__a174",
    "math__a60",
    "code_review__a117",
    "math__a108",
    "code_review__a90",
    "press_releases__a111",
    "press_releases__a112",
    "CAL__CAL6",
    "patents__a90",
    "code_review__a18",
    "code_review__a63",
    "press_releases__a175",
    "math__a48",
    "math__a216",
    "code_review__a297",
]
