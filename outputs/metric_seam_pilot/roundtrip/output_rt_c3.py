# AUTO: blind rule compilation chunk c3

import re

# ---------------------------------------------------------------------------
# shared helpers (not scoring functions themselves)
# ---------------------------------------------------------------------------

def _clamp(x, lo=0.0, hi=10.0):
    try:
        x = float(x)
    except Exception:
        return lo
    if x != x:  # NaN guard
        return lo
    return max(lo, min(hi, x))


def _words(text):
    if not text:
        return []
    return re.findall(r"[A-Za-z']+", text)


def _sentences(text):
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _math_symbol_count(t):
    if not t:
        return 0
    return len(re.findall(r'[=<>≤≥∑∫√±∞^_]|\\[a-zA-Z]+', t))


_SUBSTANTIVE_KW = [
    'because', 'i think', 'why do we', 'why does', 'why not', 'consider',
    'alternative', 'trade-off', 'tradeoff', 'architecture', 'design',
    'preferable', 'interesting', 'concern', 'however', 'in my opinion',
    'what if', 'could we', 'reasoning', 'explanation', 'i believe',
    'the reason', 'this is because'
]
_DIRECTIVE_KW = [
    'remove', 'fix typo', 'typo', 'nit:', 'nit ', 'style', 'formatting',
    'add empty line', 'done', 'lgtm', 'please fix', 'unused', 'rename',
    'extra space', 'trailing whitespace'
]


def _review_signals(text):
    """Split a review-comment blob into blocks and count how many read as
    substantive/dialogic vs. terse/directive. Returns (sub, direc, total)."""
    if not text or not text.strip():
        return 0.0, 0.0, 0
    blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    if not blocks:
        blocks = [text.strip()]
    sub = 0.0
    direc = 0.0
    for b in blocks:
        lb = b.lower()
        wc = len(_words(b))
        sub_hit = any(k in lb for k in _SUBSTANTIVE_KW)
        direc_hit = any(k in lb for k in _DIRECTIVE_KW)
        is_dialogue_question = ('?' in b) and wc > 8
        if (sub_hit or is_dialogue_question) and wc >= 8:
            sub += 1
        elif direc_hit or wc <= 6:
            direc += 1
        else:
            if wc > 15:
                sub += 0.5
                direc += 0.5
            else:
                direc += 1
    total = len(blocks)
    return sub, direc, total


# ---------------------------------------------------------------------------
# scoring functions, one per job_id
# ---------------------------------------------------------------------------

def score__math__a222(text):
    if not text or not text.strip():
        return 0.0
    t = text.strip()
    words = _words(t)
    n = len(words)
    if n < 5:
        return 0.0
    lower = t.lower()
    resolve_kw = ['proof', 'therefore', 'hence', 'thus', 'qed', 'counterexample',
                  'solution', 'answer is', 'we conclude', 'implies', 'let ',
                  'define', 'compute', 'calculate', 'construct']
    resolve_score = sum(lower.count(k) for k in resolve_kw)
    math_symbols = _math_symbol_count(t)
    numbers = len(re.findall(r'\d', t))
    evasive_kw = ['why do you ask', 'what do you mean', 'have you considered',
                  'it\'s up to you', 'not sure what you', 'philosophically',
                  'depends on what you mean', 'that\'s subjective']
    evasive_score = sum(lower.count(k) for k in evasive_kw)
    sentences = _sentences(t)
    q_ratio = t.count('?') / max(1, len(sentences))

    signal = resolve_score * 1.5 + min(math_symbols, 10) * 0.3 + min(numbers, 20) * 0.05
    penalty = evasive_score * 3.0 + (q_ratio * 5 if q_ratio > 0.5 else 0.0)
    raw = signal - penalty
    if signal <= 0 or (evasive_score > 0 and signal < 2):
        return 0.0
    return _clamp(2.0 + raw)


def score__code_review__a162(text):
    sub, direc, total = _review_signals(text)
    if total == 0:
        return 0.0
    ratio = sub / total
    if ratio >= 0.7:
        return _clamp(8.0 + (ratio - 0.7) / 0.3 * 2.0)
    if ratio <= 0.2:
        return _clamp(ratio / 0.2 * 2.5)
    return _clamp(4.0 + (ratio - 0.2) / 0.5 * 3.0)


def score__math__a30(text):
    if not text or not text.strip():
        return 0.0
    t = text.strip()
    words = _words(t)
    n = len(words)
    if n < 3:
        return 0.0
    lower = t.lower()
    proof_hits = sum(lower.count(k) for k in
                      ['proof', 'qed', 'therefore', 'hence', 'thus', 'lemma',
                       'theorem', 'by induction', 'by contradiction'])
    calc_hits = len(re.findall(r'\d+\s*[+\-*/^=]\s*\d+', t)) + t.count('=')
    def_hits = sum(lower.count(k) for k in ['define', 'defined as', 'definition'])
    symbol_density = _math_symbol_count(t) / max(1, n)
    hint_hits = sum(lower.count(k) for k in
                     ['hint:', 'you might try', 'consider trying', 'as a starting point'])

    substantive = proof_hits * 2 + min(calc_hits, 10) + def_hits * 1.5 + symbol_density * 20
    if n < 8 and substantive == 0:
        return 0.0
    if substantive >= 4:
        return _clamp(8.0 + min(substantive - 4, 4) * 0.5)
    if substantive >= 1 or hint_hits > 0 or n >= 15:
        return _clamp(3.0 + substantive * 0.5 - hint_hits * 0.3)
    return 0.5


def score__math__a18(text):
    if not text or not text.strip():
        return 0.0
    t = text.strip()
    words = _words(t)
    n = len(words)
    if n < 3:
        return 0.5
    lower = t.lower()
    proof_hits = sum(lower.count(k) for k in
                      ['proof', 'qed', 'therefore', 'hence', 'thus', 'lemma',
                       'theorem', 'by induction', 'by contradiction'])
    concept_hits = sum(lower.count(k) for k in
                        ['intuitively', 'the idea is', 'conceptually', 'this means',
                         'in other words', 'geometrically', 'the reason'])
    symbol_density = _math_symbol_count(t) / max(1, n)
    hint_hits = sum(lower.count(k) for k in ['hint:', 'you might try', 'as a starting point'])

    depth = proof_hits * 1.5 + concept_hits * 1.5 + symbol_density * 15 + min(n / 50.0, 3)
    if depth >= 6:
        return _clamp(9.0 + min(depth - 6, 2) * 0.5)
    if depth >= 2.5:
        return _clamp(6.0 + (depth - 2.5) * 0.5)
    if hint_hits > 0 or n < 15:
        return _clamp(1.0 + depth * 0.3)
    return 2.0


def score__patents__a192(text):
    if not text or not text.strip():
        return 0.0
    if re.search(r'(?im)^\s*claims?\s*:?\s*$', text) or re.search(r'(?i)\bclaims?\s*:', text):
        return 10.0
    return 0.0


def score__press_releases__a113(text):
    if not text or not text.strip():
        return 0.0
    lines = [l.strip() for l in re.split(r'[\n\r]+', text) if l.strip()]
    if not lines:
        lines = [text.strip()]
    total = 0
    good = 0
    for line in lines:
        chunks = _sentences(line) if len(line) > 200 else [line]
        if not chunks:
            chunks = [line]
        for c in chunks:
            total += 1
            words_c = _words(c)
            is_bullet = bool(re.match(r'^\s*([-*\u2022]|\d+[.)])\s+', c))
            ends_punct = c.rstrip().endswith(('.', '!', '?'))
            long_enough = len(words_c) >= 5
            if ends_punct and long_enough and not is_bullet:
                good += 1
    if total == 0:
        return 0.0
    ratio = good / total
    if ratio >= 0.9:
        return 10.0
    if ratio >= 0.6:
        return _clamp(7.0 + (ratio - 0.6) / 0.4 * 1.6)
    if ratio >= 0.3:
        return _clamp(4.0 + (ratio - 0.3) / 0.3 * 1.0)
    if ratio > 0:
        return _clamp(1.0 + ratio / 0.3 * 2.0)
    return 0.0


def score__patents__a96(text):
    if not text or not text.strip():
        return 0.0
    has_abstract = bool(re.search(r'(?i)\babstract\b', text))
    has_claims_heading = bool(re.search(r'(?i)\bclaims?\s*:?', text))
    numbered_claims = len(re.findall(r'(?m)^\s*\d{1,3}\s*\.\s+', text))
    if has_abstract and has_claims_heading and numbered_claims >= 1:
        return 10.0
    return 0.0


def score__code_review__a252(text):
    if not text or not text.strip():
        return 0.0
    blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    if not blocks:
        blocks = [text.strip()]
    total = len(blocks)
    bot_count = 0
    for b in blocks:
        if ('```suggestion' in b
                or re.search(r'(?i)format\s+\w+\s+code\s*:', b)
                or re.search(r'(?i)\b(gofmt|clang-format|eslint|prettier|black|golangci-lint)\b', b)):
            bot_count += 1
    ratio = bot_count / total
    return _clamp(ratio * 10.0)


_DYNAMIC_KW = ['control', 'feedback', 'sensor', 'sensed', 'processor', 'processing',
               'algorithm', 'signal', 'controller', 'actuator', 'execute', 'computing',
               'monitor', 'adjust', 'real-time', 'circuit', 'software', 'instructions']
_STATIC_KW = ['composition', 'compound', 'molecule', 'formulation', 'alloy', 'polymer',
              'chemical', 'material', 'mixture', 'crystalline', 'protein', 'antibody',
              'gene sequence']
_MECH_KW = ['gear', 'lever', 'spring', 'hinge', 'bracket', 'shaft', 'valve', 'mechanical']


def score__patents__a36(text):
    if not text or not text.strip():
        return 5.0
    lower = text.lower()
    dyn = sum(lower.count(k) for k in _DYNAMIC_KW)
    stat = sum(lower.count(k) for k in _STATIC_KW)
    mech = sum(lower.count(k) for k in _MECH_KW)
    if dyn == 0 and stat == 0 and mech == 0:
        return 5.0
    if dyn > stat and dyn > 0:
        return _clamp(7.0 + min(dyn, 10) * 0.3)
    if stat > dyn:
        return 1.0
    if mech > 0:
        return 5.0
    return 5.0


def score__math__a228(text):
    if not text or not text.strip():
        return 0.0
    t = text.strip()
    words = _words(t)
    n = len(words)
    lower = t.lower()
    filler_kw = ['by the way', 'as an aside', 'interestingly', 'note that',
                 'it is worth noting', 'alternatively', 'another way',
                 'just to clarify', 'in general']
    filler_hits = sum(lower.count(k) for k in filler_kw)
    elision = lower.count('[...]') + lower.count('...')
    quote_lines = len(re.findall(r'(?m)^\s*>', t))
    has_final_answer = bool(re.search(r'(?i)(answer is|therefore|=\s*\d)', t))

    if n <= 40 and filler_hits == 0 and quote_lines == 0:
        return _clamp(10.0 - max(0, n - 15) * 0.05)
    if n <= 150 and elision < 3:
        return _clamp(6.0 - filler_hits * 0.3 - max(0, (n - 40)) * 0.01)
    penalty = elision * 1.0 + quote_lines * 0.5 + max(0, n - 150) * 0.01
    score = 4.0 - penalty
    if not has_final_answer:
        score -= 1.0
    return _clamp(score)


def score__patents__a234(text):
    if not text or not text.strip():
        return 0.0
    lower = text.lower()
    problem_kw = ['prior art', 'conventionally', 'conventional', 'problem',
                  'limitation', 'deficienc', 'drawback', 'disadvantage',
                  'consumes', 'consume bandwidth', 'shortcoming', 'issue with']
    solution_kw = ['solve', 'to overcome', 'to address', 'provides a',
                   'in order to', 'the present invention', 'according to the invention',
                   'this disclosure', 'improve']
    has_problem = any(k in lower for k in problem_kw)
    has_solution = any(k in lower for k in solution_kw)
    if has_problem and has_solution:
        return 10.0
    return 0.0


def score__patents__a42(text):
    if not text or not text.strip():
        return 0.0
    m = re.search(r'(?i)claims\s*:', text)
    if not m:
        return 0.0
    after = text[m.end():]
    canceled_nearby = bool(re.search(r'(?i)\(canceled\)|\(cancelled\)', after[:500]))
    numbered = re.findall(r'(?m)^\s*(\d{1,3})\s*\.\s+(.*)', after)
    if not numbered:
        return 0.0
    first_num, first_content = numbered[0]
    first_canceled = bool(re.search(r'(?i)canceled|cancelled', first_content)) and len(_words(first_content)) < 6
    concrete_kw = ['comprising', 'wherein', 'configured to', 'step of',
                   'a method', 'a system', 'an apparatus']
    has_concrete = any(k in after.lower()[:2000] for k in concrete_kw)
    if canceled_nearby or first_canceled:
        return 1.5
    if has_concrete:
        return 10.0
    return 1.5


def score__press_releases__a118(text):
    if not text or not text.strip():
        return 0.0
    lower = text.lower()
    non_english_ratio = len(re.findall(r'[^\x00-\x7F]', text)) / max(1, len(text))
    if non_english_ratio > 0.3:
        return 0.0
    date_pattern = re.search(
        r'(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b'
        r'|\b\d{1,2}/\d{1,2}/\d{2,4}\b', text)
    byline = re.search(r'(?i)\bby\s+[A-Z][a-zA-Z\'.-]+\s+[A-Z][a-zA-Z\'.-]+\b', text)
    n = len(_words(text))
    nav_kw = ['home', 'sitemap', 'privacy policy', 'terms of use', 'subscribe',
              'menu', 'navigation', 'click here', 'all rights reserved', 'copyright']
    nav_hits = sum(lower.count(k) for k in nav_kw)

    if date_pattern and byline and n > 150:
        return 10.0
    if byline and date_pattern and n > 50:
        return 7.5
    if date_pattern and 30 < n <= 150:
        return 6.2
    if date_pattern and nav_hits > 3:
        return 5.0
    if 'about us' in lower and not date_pattern:
        return 1.2
    if nav_hits > 0 and not date_pattern:
        return 0.0
    if n < 20:
        return 0.0
    return 3.0


def score__math__a234(text):
    if not text or not text.strip():
        return 0.0
    if (re.search(r'\$\$.*?\$\$', text, re.S)
            or re.search(r'\\\[.*?\\\]', text, re.S)
            or re.search(r'\\begin\{[a-zA-Z*]+\}', text)):
        return 10.0
    if re.search(r'\$[^$\n]+\$', text):
        return 4.4
    return 0.0


def score__code_review__a9(text):
    sub, direc, total = _review_signals(text)
    if total == 0:
        return 0.0
    ratio = sub / total
    if ratio >= 0.75:
        return _clamp(9.0 + (ratio - 0.75) / 0.25 * 1.0)
    if ratio <= 0.15:
        return _clamp(ratio / 0.15 * 3.0)
    return _clamp(3.1 + (ratio - 0.15) / 0.6 * 5.9)


def score__patents__a0(text):
    if not text or not text.strip():
        return 0.0
    m = re.search(r'(?i)claims\s*:?', text)
    has_abstract = bool(re.search(r'(?i)\babstract\s*:?', text))
    if not m:
        return 1.5 if has_abstract else 0.0
    after = text[m.end():]
    numbered = re.findall(r'(?m)^\s*\d{1,3}\s*\.\s+\S', after)
    if len(numbered) >= 2:
        return 10.0
    if len(numbered) == 1:
        return 5.0
    return 2.0


def score__code_review__a306(text):
    sub, direc, total = _review_signals(text)
    if total == 0:
        return 0.0
    ratio = sub / total
    if ratio >= 0.65:
        return _clamp(7.0 + (ratio - 0.65) / 0.35 * 3.0)
    if ratio <= 0.25:
        return _clamp(ratio / 0.25 * 3.0)
    return _clamp(4.0 + (ratio - 0.25) / 0.4 * 2.0)


def score__patents__a179(text):
    if not text:
        return 0.0
    if '[...]' in text or '[\u2026]' in text:
        return 10.0
    return 0.0


def score__math__a114(text):
    if not text or not text.strip():
        return 0.0
    t = text.strip()
    lower = t.lower()
    words = _words(t)
    n = len(words)
    you_hits = len(re.findall(r"\byour\b|\byou\b|\byou're\b", lower))
    verify_kw = ['correct', 'verify', 'confirms', 'confirm', 'valid', 'checks out', 'is right', 'looks good']
    flaw_kw = ['mistake', 'error', 'flaw', 'incorrect', 'issue with', 'problem with', 'however', 'but you']
    feedback_kw = ['alternatively', 'a cleaner way', 'better notation', 'you could also',
                   'consider using', 'nice approach', 'good job', 'well done']
    hint_kw = ['hint:', 'try considering', 'think about']

    verify_hits = sum(lower.count(k) for k in verify_kw)
    flaw_hits = sum(lower.count(k) for k in flaw_kw)
    feedback_hits = sum(lower.count(k) for k in feedback_kw)
    hint_hits = sum(lower.count(k) for k in hint_kw)

    engagement = you_hits + verify_hits * 2 + flaw_hits * 2 + feedback_hits * 2
    if engagement == 0:
        return 0.0
    if engagement >= 6 and (verify_hits > 0 or feedback_hits > 0):
        return _clamp(8.0 + min(engagement - 6, 4) * 0.5)
    if engagement >= 2:
        return _clamp(6.0 + min(engagement - 2, 4) * 0.25)
    if hint_hits > 0 or (you_hits > 0 and n < 40):
        return 3.0
    return 0.0


def score__math__a180(text):
    if not text or not text.strip():
        return 0.0
    t = text.strip()
    n = max(1, len(_words(t)))
    symbol_count = _math_symbol_count(t)
    latex_count = len(re.findall(r'\$[^$]+\$|\\\[.*?\\\]|\\begin\{[a-zA-Z*]+\}', t, re.S))
    density = symbol_count / n
    if symbol_count == 0 and latex_count == 0:
        return 0.0
    if latex_count >= 1 and density > 0.15:
        return _clamp(9.0 + min(density - 0.15, 0.05) * 20)
    if latex_count >= 1 or density > 0.03:
        return _clamp(5.0 + min(density, 0.15) / 0.15 * 4)
    return _clamp(1.0 + density * 30)


_FIN_KW = ['expense ratio', 'basis points', 'p/e ratio', 'yield', 'nav ', 'dividend',
           'return on', 'assets under management', 'benchmark', 'portfolio',
           'volatility', 'sharpe ratio', 'quarterly earnings', 'revenue of', 'net income']


def score__press_releases__a104(text):
    if not text or not text.strip():
        return 0.0
    lower = text.lower()
    currency = len(re.findall(r'\$\s?\d[\d,]*(\.\d+)?|\d+(\.\d+)?\s?%', text))
    fin_hits = sum(lower.count(k) for k in _FIN_KW)
    if currency >= 5 or fin_hits >= 3:
        return 10.0
    if currency > 0 or fin_hits > 0:
        return 5.0
    return 0.0


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
