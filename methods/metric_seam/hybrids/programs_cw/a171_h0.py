"""a171 hybrid: Diction clarity and precise, evocative word choice.

Core insight from train residuals: the judge is scoring CONTROL of the prose,
not topic or keyword imagery. Low-scored stories are dense with unintentional
mechanical errors in the NARRATION (subject-verb agreement, missing
apostrophes, lowercase 'i', malapropisms, run-on participial chains); mid
stories are clean but plain; high stories are clean AND precise. Errors inside
quoted dialogue / chat-log lines are diegetic (character voice) and must NOT
count: a chat-log story full of 'wtf n00b' scored 0.9. So: strip dialogue and
chat chrome, measure error density on what remains (code predicate), and use
two verified LLM extractions for what regex cannot see (non-pattern
misspellings and agreement slips; one vivid-phrase probe to separate
clean-plain from clean-precise). All LLM answers are verified verbatim against
the text before they contribute.
"""
import re
import math

LLM_FIELDS = {
    "flaw_quotes": ("Quote verbatim up to 3 misspelled or grammatically wrong "
                    "words/short phrases from the story's narration (not inside "
                    "dialogue), comma-separated; answer NONE if the prose is clean."),
    "vivid_quote": ("Quote verbatim the story's single most precise, evocative "
                    "image or phrase (max 10 words); answer NONE if the prose is "
                    "generic, plain, or clumsy."),
}

# ---- text hygiene -----------------------------------------------------------

_ENTITY_RE = re.compile(r"&(amp|gt|lt|nbsp|#x200b|#8203);?", re.I)
_AUTHOR_NOTE_RE = re.compile(
    r"^\s*(\[?(final\s+)*edit\b|edit:|edits?:|ps\b|p\.s\.|thank you|thanks\b|"
    r"update:|obligatory|first (wp|post)|hope(fully)? you|feel free|"
    r"if you (liked|enjoyed)|check out|come on over|/?r/|part \d|"
    r"\*{3,}|~{3,}|-{4,}|_{4,}|={4,}|\[poem\]|\[wp\]|\[pi\])", re.I)
_CHAT_LINE_RE = re.compile(r"^\s*\**[*_]*[A-Za-z][\w .@'\-]{0,30}[*_]*\s*:\s+\S")
_STAGE_RE = re.compile(r"^\s*\*[^*]{1,80}\*\s*$")

def _clean(text, ops):
    try:
        t = ops.normalize(text)
        if not isinstance(t, str) or not t:
            t = text
    except Exception:
        t = text
    t = _ENTITY_RE.sub(" ", t)
    # unify curly quotes
    t = (t.replace("“", '"').replace("”", '"')
          .replace("‘", "'").replace("’", "'"))
    t = t.replace("**", "").replace("—", " - ")
    return t

def _split_lines(t):
    story, n_chat, n_lines = [], 0, 0
    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            continue
        n_lines += 1
        if _AUTHOR_NOTE_RE.match(s):
            continue
        if _CHAT_LINE_RE.match(s) or _STAGE_RE.match(s):
            n_chat += 1
            continue
        story.append(ln)
    chat_frac = (n_chat / n_lines) if n_lines else 0.0
    return "\n".join(story), chat_frac

_QUOTE_RE = re.compile(r'"[^"\n]{1,600}"')
_SQUOTE_RE = re.compile(r"(?<![A-Za-z])'[^'\n]{2,400}'(?![A-Za-z])")
_MARK = "\x01"

def _narration(t):
    """Story text with quoted speech replaced by a marker (so sentence
    punctuation inside dialogue can't fake narration errors)."""
    t = _QUOTE_RE.sub(_MARK, t)
    t = _SQUOTE_RE.sub(_MARK, t)
    kept = []
    for ln in t.split("\n"):
        s = ln.strip()
        # still opens with a quote char => unclosed multi-paragraph speech
        if s[:1] in ('"', "'") and len(s) > 1:
            kept.append(_MARK)
            continue
        # stylized/distorted text (letter-spaced speech, ascii art)
        toks = re.findall(r"[A-Za-z]+", s)
        if len(toks) >= 4 and sum(1 for w in toks if len(w) == 1) > 0.4 * len(toks):
            kept.append(_MARK)
            continue
        kept.append(ln)
    t = "\n".join(kept)
    # dangling unmatched quote to end of line (interrupted dialogue)
    t = re.sub(r'"[^"\n]*$', _MARK, t, flags=re.M)
    return t

# ---- code error predicates (narration only) ---------------------------------

_MISSING_APOS = re.compile(
    r"(?<![A-Za-z'])(dont|didnt|doesnt|isnt|wasnt|werent|wouldnt|couldnt|"
    r"shouldnt|cant|wont|havent|hasnt|hadnt|youre|theyre|weve|youve|theyve|"
    r"thats|whats|theres|im|ive|lets|arent|aint)(?![A-Za-z'])")
_LOWER_I = re.compile(r"(?:^|[\s\"(])i(?=[\s'.,!?;:])")
_WRONG_OF = re.compile(r"\b(should|could|would|must|might) of\b", re.I)
_ALOT = re.compile(r"\b(alot|aswell|abit|noone|everytime|incase|infact)\b"
                   r"|\bin regards to\b", re.I)
_A_VOWEL = re.compile(r"\b[Aa] (?!one\b|uni|use|user|euro|ubiq|eu\b)"
                      r"[aeiouAEIOU][a-z]{3,}")
_TO_LATE = re.compile(r"\b(was|is|be|it's|its) to late\b", re.I)
_NO_SPACE = re.compile(r"[a-z][.!?][A-Z][a-z]")
_DOUBLE_WORD = re.compile(r"\b([A-Za-z]{2,})\s+\1\b", re.I)
_DW_OK = {"had", "that", "very", "really", "so", "no", "long", "many"}
_CLICHES = [r"\bin the blink of an eye\b", r"\bat the end of the day\b",
            r"\btime stood still\b", r"\bdrunk as a skunk\b",
            r"\bchills? (ran|went) down\b", r"\bdark and stormy\b",
            r"\bcold sweat\b", r"\bheart of gold\b", r"\bsent shivers\b"]

_LC_START = re.compile(r"(?:^[ \t]*|(?<!\.)(?<=[.!?])[ \t]+)([a-z][a-z']+)",
                       re.M)

def _lc_sentence_starts(narr):
    n = 0
    for m in _LC_START.finditer(narr):
        if m.group(1) == "i" or _MARK in m.group(0):
            continue
        n += 1
    return n

def _code_errors(narr):
    e = 0.0
    e += len(_MISSING_APOS.findall(narr))
    e += len(_LOWER_I.findall(narr))
    e += 1.5 * len(_WRONG_OF.findall(narr))
    e += 1.5 * len(_ALOT.findall(narr))
    e += 1.5 * len(_TO_LATE.findall(narr))
    e += 1.0 * len(_A_VOWEL.findall(narr))
    e += 0.8 * len(_NO_SPACE.findall(narr))
    e += 0.8 * sum(1 for m in _DOUBLE_WORD.finditer(narr)
                   if m.group(1).lower() not in _DW_OK)
    e += 0.7 * _lc_sentence_starts(narr)
    return e

# ---- LLM field verification --------------------------------------------------

def _norm_snip(s):
    s = (s or "").lower()
    s = (s.replace("“", '"').replace("”", '"')
          .replace("‘", "'").replace("’", "'"))
    return re.sub(r"[^a-z0-9']+", " ", s).strip()

def _verified_flaws(field, hay_norm):
    if not field or field.strip().upper() in ("", "NONE", "NONE."):
        return 0
    n = 0
    for part in re.split(r"[,;\n]| and ", field):
        p = _norm_snip(part.strip(" '\"`."))
        if 2 <= len(p) <= 60 and p in hay_norm:
            n += 1
    return min(n, 3)

def _verified_vivid(field, hay_norm):
    if not field:
        return False
    f = field.strip().strip("'\"` .")
    if f.upper() in ("", "NONE"):
        return False
    p = _norm_snip(f)
    words = p.split()
    return bool(3 <= len(words) <= 14 and len(p) >= 12 and p in hay_norm)

# ---- main --------------------------------------------------------------------

def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or len(text.strip()) < 40:
            return 0.5
        t = _clean(text, ops)
        story, chat_frac = _split_lines(t)
        chat_mode = chat_frac > 0.5
        if not story.strip():
            story = "" if chat_mode else t
        narr = _narration(story)
        nwords = len(re.findall(r"[A-Za-z']+", narr))
        total_words = max(len(re.findall(r"[A-Za-z']+", story)), 1)
        hay_norm = _norm_snip(t)

        # 1) mechanical error density (per 1000 narration words)
        errs = _code_errors(narr) if not chat_mode else 0.0
        try:
            v = _verified_flaws((extracted or {}).get("flaw_quotes", ""), hay_norm)
        except Exception:
            v = 0
        if chat_mode:
            v = min(v, 1)  # chat 'errors' are usually diegetic
        errs += 2.0 * v
        errs = max(0.0, errs - 1.2)  # allowance: a lone typo is not flab
        denom = max(nwords, int(0.35 * total_words), 120)
        density = 1000.0 * errs / denom
        err_pen = 0.50 * (1.0 - math.exp(-density / 6.0))

        # 2) run-on / breathless-chain penalty (narration)
        style_pen = 0.0
        if not chat_mode and nwords >= 80:
            sents = [s for s in re.split(r"[.!?]+", narr) if len(s.split()) > 2]
            if sents:
                mws = sum(len(s.split()) for s in sents) / len(sents)
                if mws > 26:
                    style_pen += min(0.10, 0.01 * (mws - 26))
            as_rate = len(re.findall(r"\bas\b", narr.lower())) * 100.0 / nwords
            ing_rate = len(re.findall(r",\s*\w+ing\b|\bcausing\b|\bbefore \w+ing\b",
                                      narr.lower())) * 100.0 / nwords
            if as_rate > 2.2:
                style_pen += min(0.08, 0.03 * (as_rate - 2.2))
            if ing_rate > 1.2:
                style_pen += min(0.08, 0.04 * (ing_rate - 1.2))
            bang_rate = narr.count("!") * 100.0 / nwords
            if bang_rate > 1.0:
                style_pen += min(0.06, 0.03 * (bang_rate - 1.0))
        cl = sum(1 for p in _CLICHES if re.search(p, t.lower()))
        style_pen += min(0.06, 0.03 * cl)
        # verse/doggerel layout: many short unpunctuated lines
        if not chat_mode:
            lines = [ln.strip() for ln in story.split("\n") if ln.strip()]
            if len(lines) >= 10:
                loose = sum(1 for ln in lines
                            if 2 <= len(ln.split()) <= 12
                            and ln[-1] not in ".!?\"'):;,-")
                if loose / len(lines) > 0.45:
                    style_pen += 0.10
        style_pen = min(style_pen, 0.22)

        # 3) precision bonus: verified vivid phrase (separates clean-plain
        #    from clean-precise); only meaningful when prose is clean.
        try:
            vivid = _verified_vivid((extracted or {}).get("vivid_quote", ""), hay_norm)
        except Exception:
            vivid = False
        clean = math.exp(-density / 6.0)
        vivid_bonus = 0.10 * clean if vivid else 0.0

        base = 0.60 if not chat_mode else 0.62
        s = base - err_pen - style_pen + vivid_bonus
        return max(0.02, min(0.98, s))
    except Exception:
        return 0.5
