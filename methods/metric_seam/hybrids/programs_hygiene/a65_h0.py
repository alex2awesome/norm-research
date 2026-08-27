"""a65 — Context and significance (hybrid channel).

Construct: does the document supply BACKGROUND and RATIONALE so a reader can
interpret importance, trends, and implications?

Design (presence != quality):
  * Work only over PROSE sentences (function-word density gate) so keyword hits
    inside nav chrome / spec tables / tag clouds do not count.
  * Score six cue FAMILIES whose predicates require co-occurring content
    (number + change-verb + temporal comparator in ONE sentence, etc.), not
    bare keywords. Family diversity is rewarded: real context sections mix
    history + trend + quantified comparison + rationale.
  * Damp by absolute prose mass (link farms have little prose) and by an
    announcement/attribution gate (bio pages / FAQs rarely have datelines,
    "announced/reported/said/according to").
  * Optional LLM fields are verbatim quotes; code verifies grounding
    (3-gram containment) before using them, keeping the predicate in code.
"""
import re

LLM_FIELDS = {
    "context_quote": "Quote verbatim (<=15 words) a passage giving background, history, or prior-trend context for the main news; NONE if absent.",
    "significance_quote": "Quote verbatim (<=15 words) a passage stating why the news matters, its implications, or what it changes; NONE if absent.",
}

_FUNC_WORDS = frozenset(
    "the a an of to in for with on at by from as that this these those it its "
    "is are was were be been has have had will would can could and or but "
    "their our we they he she who which when while".split())

# --- cue families (compiled once) -------------------------------------------
_NUM = re.compile(r"\d")
# NOTE (hygiene patch): "fall\w*" was an unrestricted wildcard and false-fired
# on "fallout" (consequences, unrelated to a decline trend) and "fallows"/
# "fallow" (a surname / fallow land, also unrelated). Restricted to the
# inflections the decline-trend concept actually needs (fall/falls/fallen/
# falling); "fell"/"rose"/etc. stay exact since they're already finite forms.
_CHANGE = re.compile(
    r"\b(?:increas\w*|decreas\w*|declin\w*|grew|grow\w*|rose|rising|fell|"
    r"fall(?:s|en|ing)?|dropp?\w*|surpass\w*|exceed\w*|doubl\w*|tripl\w*|shrank|"
    r"slipp?\w*|jump\w*|climb\w*|gain\w*|remained|down|up|higher|lower|"
    r"larger|smaller|\d+x)\b", re.I)
_TEMPORAL_CMP = re.compile(
    r"compared\s+(?:to|with)|versus|\bvs\.?\s|year[-\s]over[-\s]year|\byoy\b|"
    r"same\s+(?:period|quarter|month)|a\s+year\s+(?:ago|earlier)|"
    r"last\s+(?:year|quarter|month|spring|summer|fall|winter|may)|"
    r"prior[-\s]year|previous\s+(?:year|quarter)|over\s+(?:the\s+)?last|"
    r"from\s+[\w\s]{0,25}(?:19|20)\d{2}|since\s+(?:19|20)\d{2}|"
    r"consecutive\s+(?:quarter|year|month)|than\s+(?:in\s+)?q?\d|"
    r"higher\s+than|lower\s+than|above\s+(?:19|20)\d{2}\s+levels|"
    r"far\s+cry\s+from|earlier\s+this\s+year", re.I)
_MAGNITUDE = re.compile(r"%|\bper\s?cent\w*\b|\bmillion\b|\bbillion\b|\btrillion\b", re.I)
_B_HIST = re.compile(
    r"since\s+(?:19|20)\d{2}|years?\s+since|"
    r"since\s+the\s+\w+\s+(?:agreement|crisis|pandemic|recession|launch)|"
    r"\b\d+\s+years?\s+(?:ago|old)\b|[-\s]year[-\s]old\b|"
    r"(?:founded|established|launched|debuted|created|began)\s+in\s+(?:19|20)\d{2}|"
    r"first\s+(?:implemented|introduced|launched|opened|began)|"
    r"\boriginally\b|\bin\s+(?:19|20)\d{2},|\b\d+(?:th|st|nd|rd)\s+annual\b", re.I)
_C_MILESTONE = re.compile(
    r"first\s+time|world'?s\s+first|first[-\s]ever|milestone|"
    r"record\s+(?:high|low|revenue|quarter|year|\$)|"
    r"(?:highest|lowest|largest|biggest|best|most\s+\w+)\s+(?:\w+\s+)?(?:ever|on\s+record)|"
    r"(?:lowest|highest)\s+(?:\w+\s+)?since|most\s+significant|"
    r"of\s+particular\s+significance|unprecedented|mark(?:s|ed)?\s+a\s+turning\s+point|"
    r"(?:most|largest|biggest)\s+[\w\s]{0,30}to\s+date", re.I)
_D_TREND = re.compile(
    r"\btrends?\b|fastest[-\s]growing|continue[sd]?\s+to|continuing\s+to|"
    r"accelerat\w+|momentum|increasingly|shift\s+(?:from|to|toward|away)|"
    r"rebound|\bboom\b|slowdown|downturn|growing\s+(?:demand|interest|number)|"
    r"demand\s+for|a\s+thing\s+of\s+the\s+past|industry[-\s]wide|persists?\b|"
    r"likely\s+to|expected\s+to|forecasts?\b|outlook|predict\w*|"
    r"projections?\b|\bsurge\b|stagnant", re.I)
# Legal safe-harbor / forward-looking boilerplate: never count cues in these.
_BOILER = re.compile(
    r"forward[-\s]looking|differ\s+materially|safe\s+harbor|"
    r"no\s+obligation\s+to\s+(?:publicly\s+)?update|risks?,?\s+uncertainties", re.I)
_E_CAUSAL = re.compile(
    r"driven\s+by|due\s+to|as\s+a\s+result|\bbecause\b|le[ad]d?s?\s+to|"
    r"in\s+response\s+to|\breflecting\b|thanks\s+to|result(?:ed|s)?\s+in|"
    r"\bamid\b|attribut\w+\s+to|owing\s+to|stemm\w+\s+from", re.I)
_F_PURPOSE = re.compile(
    r"with\s+the\s+goal\s+of|\bgoal\s+of\b|aims?\s+to|aimed\s+at|designed\s+to|"
    r"intended\s+to|in\s+order\s+to|in\s+an\s+effort\s+to|"
    r"as\s+part\s+of\s+(?:our|its|an?|the)[\w\s]{0,20}effort|seeks?\s+to|"
    r"initiative\s+to|help\s+(?:reduce|prevent|deter|address|combat|improve|"
    r"protect|increase|accelerate)|(?:more|increasing)\s+accessib|"
    r"can\s+be\s+(?:burdensome|difficult|challenging|costly)|barriers?\s+to", re.I)
_ANNOUNCE = re.compile(
    r"/prnewswire|businesswire|globe\s?newswire|for\s+immediate\s+release|"
    r"\bannounc\w+|today\s+(?:reported|released|issued|launched|filed)|"
    r"(?:reported|released|issued|launched|filed|unveiled)\s+today|"
    r"\bsaid\b|\bsays\b|according\s+to", re.I)

_WORD = re.compile(r"[A-Za-z']+")


def _prose_sentences(t):
    """Keep lines that look like running prose, then sentence-split."""
    kept = []
    for ln in t.split("\n"):
        ws = _WORD.findall(ln)
        if len(ws) < 5:
            continue
        fw = sum(1 for w in ws if w.lower() in _FUNC_WORDS)
        if fw < 2 or fw / len(ws) < 0.15:
            continue
        kept.append(ln.strip())
    blob = " ".join(kept)
    sents = re.split(r"(?<=[.!?])\s+", blob)
    out = []
    for s in sents:
        ws = _WORD.findall(s)
        if len(ws) < 8:
            continue
        fw = sum(1 for w in ws if w.lower() in _FUNC_WORDS)
        if fw < 2 or fw / len(ws) < 0.15:
            continue
        out.append(s)
    return out


def _grounded(quote, low_text):
    """Quote is grounded if some 3 consecutive words appear in the text."""
    toks = _WORD.findall(quote.lower())
    if len(toks) < 4:
        return False
    n = 3
    for i in range(len(toks) - n + 1):
        if " ".join(toks[i:i + n]) in low_text:
            return True
    return False


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text)
        sents = _prose_sentences(t)
        prose_words = sum(len(_WORD.findall(s)) for s in sents)

        a1 = a2 = b = c = d = e = f = 0
        for s in sents:
            if _BOILER.search(s):
                continue
            has_num = bool(_NUM.search(s))
            has_chg = bool(_CHANGE.search(s))
            has_mag = bool(_MAGNITUDE.search(s))
            has_tmp = bool(_TEMPORAL_CMP.search(s))
            if has_num and has_tmp and (has_chg or has_mag):
                a1 += 1          # quantified comparison anchored in time
            elif has_num and has_chg and has_mag:
                a2 += 1          # quantified change, no explicit anchor
            b += len(_B_HIST.findall(s))
            c += len(_C_MILESTONE.findall(s))
            d += len(_D_TREND.findall(s))
            e += len(_E_CAUSAL.findall(s))
            f += len(_F_PURPOSE.findall(s))

        raw = (1.0 * min(a1, 4) + 0.4 * min(a2, 2) + 0.6 * min(b, 3) +
               0.55 * min(c, 3) + 0.5 * min(d, 3) + 0.5 * min(e, 1) +
               0.5 * min(f, 3))
        fams = sum(1 for v in (a1 + a2, b, c, d, e, f) if v > 0)

        # --- LLM thick-input grounding (predicate stays in code) ---
        low = re.sub(r"\s+", " ", t.lower())
        ctx = extracted.get("context_quote") if extracted else None
        sig = extracted.get("significance_quote") if extracted else None
        both_none = (ctx == "" and sig == "")
        if ctx and _grounded(ctx, low):
            raw += 0.7
            fams += 1
        if sig and _grounded(sig, low):
            raw += 0.6
            fams += 1

        div = (0.0, 0.6, 0.85, 1.0, 1.1)[min(fams, 4)]
        core = min(1.0, raw / 3.2) * div
        damp = min(1.0, prose_words / 120.0)          # link farms -> low
        gate = 1.0 if _ANNOUNCE.search(t) else 0.6    # no announce/quote cue
        # blog-index / aggregator chrome ("Read more", subscribe widgets, ...)
        junk = len(re.findall(
            r"read\s+more|subscribe|sign\s+in|tag\s+cloud|\barchives?\b|"
            r"older\s+posts|related\s+(?:articles|posts)|shopping\s+cart",
            t, re.I))
        jdamp = 0.45 if junk >= 3 else (0.8 if junk >= 2 else 1.0)
        s_val = core * damp * gate * jdamp
        if both_none:
            s_val *= 0.55                             # extractor found nothing
        return max(0.0, min(1.0, s_val))
    except Exception:
        return 0.3
