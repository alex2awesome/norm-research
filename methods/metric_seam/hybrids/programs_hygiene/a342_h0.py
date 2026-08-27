"""
Hybrid metric channel for a342: "Mechanical correctness and style adherence".

Design notes (from studying the 30 train residuals):
  - The pure-code baseline (v1_structure, train rho=0.246) saturates near 1.0 for
    almost every document (its features -- end punctuation, matched quotes/parens,
    sentence-initial caps -- are true of most scraped text), so it has almost no
    discriminative range where the judge does.
  - The judge's low scores (0.05-0.25) cluster on documents with heavy, *involuntary*
    surface breakdown: bare lowercase "i" as the first-person pronoun, missing
    apostrophes in contractions (dont/couldnt/wasnt), and real misspellings -- often
    accompanied by the author's own apology ("sorry for bad formatting and grammar").
    These are robust, class-level, regex-visible signals, so they anchor the code term.
  - But the judge is NOT a rule-checker: one perfect-score (1.0) train example is a
    multi-speaker chat-log transcript saturated with "errors" (u, srsly, wtf, ALLCAPS,
    lowercase starts) that are a deliberately executed character-voice convention, not
    sloppiness. Conversely, a couple of mid-low examples (0.2-0.25) have clean surface
    mechanics but real word-choice slips a regex cannot see (e.g. a malapropism like
    "tilde wave" for "tidal wave", "pall park" for "ball park") -- the judge scores craft
    of usage, not just rule conformity.
  - Regex can't tell "deliberate stylized voice" from "author can't spell", and can't
    catch semantic word-choice errors. That's exactly the thick-input gap an LLM read of
    the FULL text can close cheaply with one short categorical field. The code keeps the
    predicate (how the verdict is combined with surface stats into a score); the LLM only
    supplies the one grounding fact code structurally cannot compute.
"""

import re
import statistics
from collections import Counter

LLM_FIELDS = {
    "usage_verdict": (
        "Read the whole story once. Answer with exactly one word: "
        "CLEAN if grammar/spelling/word-choice are correct throughout; "
        "STYLE if any nonstandard spelling/grammar/punctuation is clearly a deliberate, "
        "skillfully-executed stylistic voice (e.g. chat/text messages, dialect, broken "
        "speech-as-characterization); ERRORS if there are real unintentional grammar, "
        "spelling, or word-choice mistakes (e.g. a wrong/garbled word, a typo). "
        "If genuinely unsure, answer CLEAN."
    ),
}

_MISSING_APOS = {
    "dont", "cant", "wont", "isnt", "wasnt", "werent", "arent", "didnt", "doesnt",
    "couldnt", "wouldnt", "shouldnt", "hasnt", "havent", "neednt", "mustnt",
    "youre", "theyre", "im", "ive", "youve", "theyve", "weve", "youll", "theyll",
    "whats", "thats", "wheres", "hows", "lets", "shouldve", "wouldve", "couldve",
    "aint",
}
_CORRECT_APOS = {
    "don't", "can't", "won't", "isn't", "wasn't", "weren't", "aren't", "didn't",
    "doesn't", "couldn't", "wouldn't", "shouldn't", "hasn't", "haven't", "needn't",
    "mustn't", "you're", "they're", "i'm", "i've", "you've", "they've", "we've",
    "you'll", "they'll", "what's", "that's", "where's", "how's", "let's",
    "should've", "would've", "could've",
}

_MISSPELLINGS = re.compile(
    r"\b(?:teh|recieve|recieved|seperate|seperated|definately|definetly|alot|"
    r"wierd|untill|occured|occurence|thier|wich|becuase|arguement|greatful|"
    r"neccessary|embarass|occassion|priviledge|publically|tommorow|begining|"
    r"concious|forsee|goverment|harrass|independant|intelligance|knowlege|"
    r"liesure|maintainance|mispell|noticable|persistant|posession|prefered|"
    r"rediculous|relevent|succesful|suprise|tounge|truely|writting|grammer)\b",
    re.IGNORECASE,
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b[A-Za-z']+\b")
_APOS_CHARS = str.maketrans({"’": "'", "‘": "'"})


def _tokenize_word_forms(t):
    # normalize curly apostrophes to straight so set lookups work regardless of
    # whether ops.normalize() already ran
    t2 = t.translate(_APOS_CHARS)
    return re.findall(r"\b[A-Za-z']+\b", t2), t2


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw
        t = t.translate(_APOS_CHARS)

        words, _ = _tokenize_word_forms(t)
        n_words = max(1, len(words))

        sents = [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]
        n_sent = max(1, len(sents))

        # --- 1. missing-apostrophe contraction rate ---
        wl = [w.lower() for w in words]
        wc = Counter(wl)
        missing = sum(wc[w] for w in _MISSING_APOS)
        correct = sum(wc[w] for w in _CORRECT_APOS)
        denom = missing + correct
        f_apos = 1.0 - (missing / denom) if denom > 0 else 1.0

        # --- 2. bare lowercase first-person "i" ---
        lower_i = len(re.findall(r"(?<![A-Za-z'])i(?![A-Za-z'])", t))
        upper_i = len(re.findall(r"(?<![A-Za-z'])I(?![A-Za-z'])", t))
        denom_i = lower_i + upper_i
        f_i = 1.0 - (lower_i / denom_i) if denom_i > 0 else 1.0

        # --- 3. sentence-initial capitalization ---
        cap_ok = 0
        for s in sents:
            core = s.lstrip("\"'“‘*_-—– \t0123456789.,:;")
            if not core:
                continue
            ch = core[0]
            if ch.isalpha():
                cap_ok += 1 if ch.isupper() else 0
            else:
                cap_ok += 1  # non-letter start (emoji/number/symbol) -- don't penalize
        f_cap = cap_ok / n_sent

        # --- 4. misspelling density (per 200 words, capped) ---
        n_misspell = len(_MISSPELLINGS.findall(t))
        f_spell = 1.0 - min(1.0, n_misspell / max(1.0, n_words / 200.0))

        # --- 5. run-on / comma-splice sentences ---
        runon = 0
        for s in sents:
            wcount = len(_WORD.findall(s))
            commas = s.count(",")
            if wcount > 50 and commas >= 4:
                runon += 1
        f_runon = 1.0 - min(1.0, runon / max(1, n_sent / 6))

        # --- 6. end punctuation sanity ---
        end_ok = sum(1 for s in sents if s.rstrip()[-1:] in ".!?\"'”’")
        f_end = end_ok / n_sent

        # cross-check with ops.sent_stats for extreme run-on tendency (mild)
        try:
            _, mean_wps, _frac_long = ops.sent_stats(t)
            if mean_wps and mean_wps > 45:
                f_runon = min(f_runon, 0.5)
        except Exception:
            pass

        code_score = (
            0.25 * f_apos
            + 0.20 * f_i
            + 0.15 * f_cap
            + 0.15 * f_spell
            + 0.15 * f_runon
            + 0.10 * f_end
        )
        code_score = max(0.0, min(1.0, code_score))

        # --- LLM grounding: deliberate voice vs. real errors, code can't see this ---
        verdict = (extracted.get("usage_verdict") or "").strip().upper()
        if "STYLE" in verdict:
            code_score = max(code_score, 0.75)
        elif re.search(r"\bERRORS?\b", verdict):
            code_score = min(code_score, 0.4)
        # CLEAN / "" / unparseable -> trust the code term as-is

        return max(0.0, min(1.0, code_score))
    except Exception:
        return 0.5
