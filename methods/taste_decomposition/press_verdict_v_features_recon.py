"""Faithful reconstruction of codex_pr_vrescue.py's extract_one/lexical_counts
feature extraction, recovered via bytecode disassembly of
__pycache__/codex_pr_vrescue.cpython-312.pyc (source .py not present on disk;
this repo's git history has no record of it either -- it was an untracked
scratch/codex-agent script referenced in notes/2026-06-25__press-release-audit.md
lines 403/429/456-457 as "the codex 'win' ... 88 vs 20 features ... ~74 clean
auditable features"). Reconstructed feature-for-feature from co_consts/bytecode;
see the disassembly transcript for the mapping. Not a guess: every regex string,
constant, and STORE_FAST target below was read directly out of the compiled
bytecode, not inferred from prose.
"""
import re
import numpy as np

MONTH_RE = r'Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?'
STOP_PROPER = frozenset({'About', 'Business', 'For', 'Share', 'Press', 'These', 'News', 'Media', 'The', 'Wire',
                          'LLC', 'Corporation', 'Release', 'Those', 'All', 'Copyright', 'This', 'Investor',
                          'That', 'More', 'Today', 'Inc', 'Contact', 'Ltd', 'Corp', 'Company', 'Limited'})
SEED = 20260626


def clean_text(s):
    if not isinstance(s, str):
        return ''
    return s.replace('\x00', ' ')


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def count_re(pattern, text, flags=re.I):
    return len(re.findall(pattern, text, flags))


def first_window(text, n=700):
    return text[:n]


def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]


def count_syllables(word):
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return 0
    groups = re.findall(r'[aeiouy]+', word)
    n = len(groups)
    if word.endswith('e') and n > 1:
        n -= 1
    return max(n, 1)


def lexical_counts(text_lower):
    specs = {
        'kw_announces': r'\b(announce[sd]?|announcing|announcement)\b',
        'kw_launch_unveil': r'\b(launch(?:es|ed|ing)?|unveil(?:s|ed|ing)?|introduc(?:e|es|ed|ing)|debut(?:s|ed|ing)?)\b',
        'kw_acquire_deal': r'\b(acquir(?:e|es|ed|ing|es)|acquisition|merger|merge[sd]?|definitive agreement|deal|transaction)\b',
        'kw_funding_invest': r'\b(raises?|raised|funding|financing|series [abcde]|invest(?:s|ed|ing|ment)?|capital raise)\b',
        'kw_earnings_guidance': r'\b(earnings|revenue|net income|eps|guidance|outlook|fiscal|quarterly results|financial results)\b',
        'kw_regulatory_legal': r'\b(sec charges?|settlement|lawsuit|litigation|enforcement|fda|approval|clearance|patent|bankruptcy|chapter 11|indictment)\b',
        'kw_government_policy': r'\b(administration|department|agency|rule|regulation|grant|funding award|national security|public health|federal)\b',
        'kw_research_survey': r'\b(study|survey|research|report|according to|findings|poll)\b',
        'kw_superlative': r'\b(first|largest|record|biggest|highest|lowest|best|leading|world-class|groundbreaking|breakthrough|milestone)\b',
        'kw_customer_product': r'\b(customer|consumer|users?|subscribers?|platform|product|service|solution|app|software|technology)\b',
        'kw_partnership': r'\b(partner(?:s|ed|ship|ing)?|collaborat(?:e|es|ed|ion|ing)|alliance|joins? forces)\b',
        'kw_award_recognition': r'\b(award(?:s|ed)?|recogniz(?:e|es|ed|ing)|honou?red|ranked|named to)\b',
        'kw_expansion_growth': r'\b(expand(?:s|ed|ing|sion)?|growth|growing|scale|global|international|new market)\b',
        'kw_crisis_risk': r'\b(crisis|risk|threat|pandemic|emergency|recall|fraud|identity theft|climate change)\b',
        'kw_media_event': r'\b(webcast|conference call|presentation|live coverage|opening ceremony|premiere|big game)\b',
    }
    return {name: len(re.findall(pattern, text_lower, re.I)) for name, pattern in specs.items()}


def extract_one(text):
    text = clean_text(text)
    lower = text.lower()
    fw = first_window(text)
    words = re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)?|\d+(?:[.,:/-]\d+)*%?", text)
    alpha_words = [w for w in words if re.search('[A-Za-z]', w)]
    word_count = len(words)
    alpha_count = len(alpha_words)
    char_count = len(text)
    sentences = split_sentences(text)
    sent_count = len(sentences)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', text.strip()) if p.strip()]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    first_line = lines[0].strip() if lines else text[:160].strip()
    first_line_words = re.findall(r'[A-Za-z]+', first_line)
    numbers = re.findall(r'(?<![A-Za-z])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?', text)
    distinct_numbers = {n.replace(',', '') for n in numbers}
    sentences_with_number = sum(1 for s in sentences if re.search(r'\d', s))

    dollar_sym = r'\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion|m|bn|b))?'
    dollar_word = r'\b(?:usd|us\$|dollars?)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion)\s?(?:usd|dollars?)\b'
    big_money = r'(?:\$\s?\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion|m|bn|b)\b|\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion)\s?(?:dollars?|usd)\b)'
    percent = r'\b\d+(?:\.\d+)?\s?(?:%|percent|percentage points?)\b'
    year = r'\b(?:19|20)\d{2}\b'
    date = rf'\b(?:{MONTH_RE})\.?\s+\d{{1,2}}(?:,\s*\d{{4}})?\b|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b'
    time_re = r'\b\d{1,2}:\d{2}\s?(?:a\.?m\.?|p\.?m\.?|am|pm|ET|PT|CT|UTC)?\b'
    phone = r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b'
    ranges = r'\b\d+(?:\.\d+)?\s?(?:-|–|—|to)\s?\d+(?:\.\d+)?\b'
    round_numbers = r'\b[1-9]\d{2,}(?:,?0{2,})\b'
    units = (r'\b\d[\d,.]*\s?(?:tons?|tonnes?|gw|mw|kw|kwh|mwh|employees?|workers?|jobs?|units?|shares?|screens?|'
             r'stores?|locations?|customers?|users?|subscribers?|members?|homes?|businesses?|square feet|sq\.?\s?ft\.?|'
             r'miles?|acres?|barrels?|boe|patients?|students?|teams?)\b')
    up_down_pct = r'\b(?:up|down|increase[sd]?|decrease[sd]?|rose|fell|grew|declined|higher|lower|improved|reduced|expanded|contracted)\b.{0,50}?\d+(?:\.\d+)?\s?(?:%|percent|percentage points?)'
    guidance = r'\b(?:guidance|outlook|expects?|forecast|target(?:s|ed)?|project(?:s|ed|ion)|anticipates?|full[- ]year|fiscal year)\b'
    super_num = r'\b(?:largest|first|record|biggest|highest|lowest|best|No\.?\s?1|#1)\b.{0,80}?\d|\d.{0,80}?\b(?:largest|first|record|biggest|highest|lowest|best)\b'

    emails = re.findall(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', text, re.I)
    urls = re.findall(r'https?://\S+|www\.\S+|\b[A-Za-z0-9.-]+\.(?:com|org|net|gov|edu|io|co)\b', text, re.I)
    quote_chars = text.count('"') + text.count('“') + text.count('”')
    quote_pairs = quote_chars // 2
    said_terms = count_re(r'\b(said|says|stated|commented|explained|added|according to)\b', text)
    exec_titles = count_re(r'\b(ceo|chief executive|cfo|chief financial|coo|chief operating|president|chairman|chair|director|vice president|svp|evp|founder)\b', text)
    exec_quote_presence = int(quote_pairs > 0 and said_terms > 0 and exec_titles > 0)
    contact_terms = count_re(r'\b(media contact|press contact|investor contact|investor relations|for media inquiries|contact:|public relations|pr hotline|newsroom)\b', text)
    about_terms = count_re(r'(?:^|\n)\s*about\s+[A-Z][^\n]{2,120}', text, flags=re.I | re.M)
    boilerplate_terms = count_re(r'\b(forward-looking statements?|safe harbor|trademarks?|all rights reserved|copyright|privacy policy|terms of use)\b', text)
    dateline = int(bool(re.search(r"^[A-Z][A-Z .,'-]{2,80}(?:--|—|-)\s", fw, re.M)) or bool(re.search(r'\b(?:BUSINESS WIRE|PRNewswire|GlobeNewswire)\b', fw)))
    wire_terms = count_re(r'\b(PRNewswire|PR Newswire|Business Wire|GlobeNewswire|news provided by)\b', text)
    ticker_terms = count_re(r'\((?:NYSE|NASDAQ|Nasdaq|OTC|TSX|LSE|ASX):\s?[A-Z.]{1,6}\)|\b(?:NYSE|NASDAQ|Nasdaq|OTC|TSX|LSE|ASX):\s?[A-Z.]{1,6}\b', text)
    byline_terms = count_re(r"(?:^|\n|\b)(?:By|Author)\s+[A-Z][A-Za-z.'-]+", fw)
    scrape_failure = int(bool(re.search(r'news_release_found|provided raw page content|does not contain a press release|result` is an empty string', text, re.I)))
    read_more_terms = count_re(r'\b(read more|sign up for|share this|posted|updated on|theme|segment|tags)\b', text)

    proper_phrases = re.findall(r"\b(?:[A-Z][A-Za-z&.'-]+)(?:\s+(?:[A-Z][A-Za-z&.'-]+)){0,3}\b", text)
    proper_filtered = set()
    for p in proper_phrases:
        if p.split()[0] in STOP_PROPER:
            continue
        if re.fullmatch('[A-Z]{1,3}', p.strip()):
            continue
        proper_filtered.add(p.strip())
    cap_tokens = re.findall(r"\b[A-Z][A-Za-z&.'-]{2,}\b", text)
    all_caps_tokens = re.findall(r'\b[A-Z]{2,}\b', text)

    punctuation = {
        'n_commas': text.count(','),
        'n_semicolons': text.count(';'),
        'n_colons': text.count(':'),
        'n_parentheses': text.count('(') + text.count(')'),
        'n_bullets': len(re.findall(r'(?:^|\n)\s*(?:[•*]|\d+\.|-)\s+', text)),
    }
    syllables = sum(count_syllables(w) for w in alpha_words)
    avg_sentence_words = safe_div(word_count, sent_count)
    avg_word_len = safe_div(sum(len(w) for w in alpha_words), alpha_count)
    if alpha_count and sent_count:
        fk_grade = 0.39 * avg_sentence_words + 11.8 * safe_div(syllables, alpha_count) - 15.59
    else:
        fk_grade = 0.0
    if alpha_count and sent_count:
        flesch = 206.835 - 1.015 * avg_sentence_words - 84.6 * safe_div(syllables, alpha_count)
    else:
        flesch = 0.0
    titlecase_ratio_first = safe_div(sum(1 for w in first_line_words if w[:1].isupper()), len(first_line_words))

    out = {
        'char_count': char_count,
        'word_count': word_count,
        'alpha_word_count': alpha_count,
        'sentence_count': sent_count,
        'paragraph_count': len(paragraphs),
        'line_count': len(lines),
        'avg_sentence_words': avg_sentence_words,
        'avg_word_len': avg_word_len,
        'flesch_kincaid_grade': fk_grade,
        'flesch_reading_ease': flesch,
        'uppercase_char_ratio': safe_div(sum(1 for c in text if c.isupper()), max(1, sum(1 for c in text if c.isalpha()))),
        'punctuation_density': safe_div(sum(punctuation.values()), word_count),
        'has_short_headline': int(3 <= len(first_line_words) <= 22 and len(first_line) < 180),
        'first_line_word_count': len(first_line_words),
        'first_line_titlecase_ratio': titlecase_ratio_first,
        'n_numbers': len(numbers),
        'n_distinct_numbers': len(distinct_numbers),
        'numeric_density': safe_div(len(numbers), word_count),
        'number_sentence_ratio': safe_div(sentences_with_number, sent_count),
        'n_dollar_sym': count_re(dollar_sym, text),
        'n_dollar_word': count_re(dollar_word, text),
        'n_money': count_re(dollar_sym, text) + count_re(dollar_word, text),
        'has_money': int(bool(re.search(dollar_sym, text, re.I)) or bool(re.search(dollar_word, text, re.I))),
        'n_bigmoney': count_re(big_money, text),
        'n_percent': count_re(percent, text),
        'has_percent': int(bool(re.search(percent, text, re.I))),
        'n_years': count_re(year, text, flags=0),
        'n_dates': count_re(date, text),
        'n_times': count_re(time_re, text),
        'n_phone': count_re(phone, text),
        'n_ranges': count_re(ranges, text),
        'n_round_numbers': count_re(round_numbers, text),
        'n_units': count_re(units, text),
        'n_up_down_percent': count_re(up_down_pct, text),
        'n_guidance_terms': count_re(guidance, text),
        'n_superlative_num': count_re(super_num, text),
        'n_emails': len(emails),
        'n_urls': len(urls),
        'url_density': safe_div(len(urls), word_count),
        'email_density': safe_div(len(emails), word_count),
        'n_quote_chars': quote_chars,
        'n_quote_pairs': quote_pairs,
        'quote_density': safe_div(quote_pairs, word_count),
        'n_said_terms': said_terms,
        'n_exec_titles': exec_titles,
        'exec_quote_presence': exec_quote_presence,
        'n_contact_terms': contact_terms,
        'n_about_sections': about_terms,
        'n_boilerplate_terms': boilerplate_terms,
        'has_dateline': dateline,
        'n_wire_terms': wire_terms,
    }
    out.update({
        'n_ticker_terms': ticker_terms,
        'n_byline_terms': byline_terms,
        'scrape_failure_artifact': scrape_failure,
        'n_news_article_artifacts': read_more_terms,
        'n_distinct_proper_phrases': len(proper_filtered),
        'n_capitalized_tokens': len(cap_tokens),
        'capitalized_token_ratio': safe_div(len(cap_tokens), max(1, alpha_count)),
        'n_all_caps_tokens': len(all_caps_tokens),
        'all_caps_token_ratio': safe_div(len(all_caps_tokens), max(1, alpha_count)),
    })
    out.update(punctuation)
    out.update(lexical_counts(lower))
    out['money_and_guidance'] = out['has_money'] * int(out['n_guidance_terms'] > 0)
    out['percent_and_updown'] = out['has_percent'] * int(out['n_up_down_percent'] > 0)
    out['quote_and_exec'] = exec_quote_presence
    out['superlative_and_number'] = int(out['n_superlative_num'] > 0)
    out['launch_and_product'] = int(out['kw_launch_unveil'] > 0 and out['kw_customer_product'] > 0)
    out['deal_and_money'] = int(out['kw_acquire_deal'] > 0 and out['has_money'] > 0)
    out['research_and_numbers'] = int(out['kw_research_survey'] > 0 and out['n_numbers'] >= 3)
    out['contact_or_about'] = int(out['n_contact_terms'] > 0 or out['n_about_sections'] > 0)
    return out


def extract_features(texts):
    rows = [extract_one(clean_text(t)) for t in texts]
    X = np.array([[r[k] for k in rows[0].keys()] for r in rows], dtype=float)
    names = list(rows[0].keys())
    return X, names
