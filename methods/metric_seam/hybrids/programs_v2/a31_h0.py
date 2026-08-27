"""a31 - Claim restraint and non-sensational language (hybrid channel).

Design: the judge scores the RESTRAINT OF THE WORDING itself, not release-ness
(dry non-releases score high; hyped releases score ~0). Code channel measures
per-1000-word densities of a graded sensational lexicon:
  STRONG  = unambiguous hype / outrage / implication-leap markers (soft-start:
            one stray hit per ~800 words is forgiven so a single colorful word
            or award name cannot tank an otherwise measured document),
  MILD    = corporate puffery (tolerated at low density in real releases),
  CTA     = promotional call-to-action register (interacts with puffery),
plus exclamation density, first-person-plural promo register (we/our), and a
markup-junk penalty (template/href debris obscuring claims). Evidence-register
bonus (money/percent/quantified figures, calibrated hedging) is GATED on low
strong-hype density so specifics cannot buy back a sensational document
(presence != quality; injection-safe: injecting hype lowers the score, and
injecting bare numbers into hyped text does not raise it).
LLM channel (optional) supplies grounded quotes of the single most sensational
phrase and one unsupported superlative; quotes are verified against the text
before they carry full weight. Works identically with extracted={}.
"""
import re

LLM_FIELDS = {
    "sensational_quote": "Quote the most sensational, exaggerated, or emotionally loaded phrase in the document; answer NONE if the wording stays measured.",
    "unsupported_superlative": "Quote one superlative or dramatic claim not backed by nearby numbers or evidence; answer NONE if none.",
}

_STRONG = [
    r"revolutionar\w+", r"breakthroughs?\b", r"miracles?\b", r"miraculous\w*",
    r"stunning\w*", r"shocking\w*", r"astonishing", r"amazing\w*",
    r"incredibl\w+", r"unbelievabl\w+", r"insane\w*", r"phenomenal\w*",
    r"sensational\w*", r"spectacular\w*", r"unforgettable", r"whopping",
    r"staggering", r"skyrocket\w*", r"top-?notch", r"\bkiller\b",
    r"jaw-?dropping", r"mind-?blowing", r"unmatched", r"unrival+ed",
    r"unparalleled", r"must-?have", r"plagues?\b", r"\bchaos\b",
    r"catastroph\w+", r"disastrous|disasters?\b", r"disgrace\w*",
    r"sabotage\w*", r"scandal\w*", r"outrage\w*", r"shameful",
    r"nightmare\w*", r"devastat\w+", r"horrif\w+", r"extortion",
    r"\bcrap\b", r"botch\w+", r"rip-?offs?\b", r"kludge",
    r"game-?chang\w+", r"once-?in-?a-?lifetime",
    r"sound\w* the alarm", r"nickel[- ](?:and|&)[- ]dim\w+",
    r"squeez\w+ the pockets?", r"thumb\w+ \w+ nose", r"state terrorism",
    r"criminal scheme", r"witness tampering", r"record-?breaking",
    r"hot streak", r"(?:best|most \w+) \w+(?:[ -]\w+){0,2} in the world",
]
_MILD = [
    r"\bleading\b", r"(?:market|industry|world|global|category) leaders?\b",
    r"\bleader in\b", r"\bpremier\b", r"premium\w*", r"innovat\w+",
    r"exclusive\w*", r"seamless\w*", r"best-?of-?breed", r"renowned",
    r"\bproud\w*", r"passionate\w*", r"excellence", r"award-?winning",
    r"unprecedented", r"historic (?:crisis|first|moment|achievement|victory)",
    r"massive\w*", r"\bhuge\b", r"\bbold\b", r"exciting", r"memorable",
    r"hassle-?free", r"unfair\w*", r"crisis|crises", r"\bmeek\w*",
    r"oppress\w+", r"fraudulent", r"unique\w*", r"reinvent\w+",
    r"transformativ\w+", r"\bstark\b", r"\bworst\b", r"falter\w+",
    r"destroy\w+", r"\belite\b", r"iconic", r"legendary", r"\bultimate\b",
    r"perfect\w*", r"delight\w*", r"greatest", r"\bessential\b",
    r"world-?class", r"best-?in-?class", r"cutting-?edge",
    r"state-?of-?the-?art", r"first-?ever", r"nosedive\w*",
    r"one of the best", r"\w+ing the world\b", r"unlock\w+ value",
    r"recognized as", r"with confidence", r"got you covered",
]
_CTA = [
    r"book now", r"buy now", r"shop now", r"order now", r"sign up",
    r"subscri\w+", r"learn more", r"read more", r"register", r"request a demo",
    r"act now", r"limited time", r"don'?t miss", r"click here", r"join now",
    r"get started", r"\bdeals?\b", r"\bfree\b", r"\bbuy\b", r"\bstore\b",
    r"\bcart\b", r"% off",
]
_STRONG_RE = re.compile("|".join(_STRONG), re.I)
_MILD_RE = re.compile("|".join(_MILD), re.I)
_CTA_RES = [re.compile(p, re.I) for p in _CTA]  # per-phrase, capped (nav chrome repeats)
_WE_RE = re.compile(r"\b(?:we|we're|we've|we'll|our|ours)\b", re.I)
_JUNK_RE = re.compile(r"href=|class=|xdm:|\{\{|&gt;|&lt;|\\r\\n|</a>")
_MONEY_RE = re.compile(r"[$£€]\s?\d")
_PCT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent)")
_QTY_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion)\b", re.I)
_TICKER_RE = re.compile(r"\((?:NYSE|NASDAQ|Nasdaq)\s*:")
_HEDGE_RE = re.compile(
    r"\b(?:may|might|could|suggests?|indicates?|appears?|approximately|"
    r"estimated?|preliminary|subject to|risks?|uncertaint\w+|assumptions?)\b", re.I)


def _sat(x):
    return x if x < 1.0 else 1.0


def _grounded(quote, low_text):
    """True if a 3-word shingle of the quote appears verbatim in the text."""
    toks = re.sub(r"[^a-z0-9 ]", " ", quote.lower()).split()
    if len(toks) < 3:
        return bool(toks) and " ".join(toks) in low_text
    return any(" ".join(toks[i:i + 3]) in low_text for i in range(len(toks) - 2))


def score(text, extracted, ops):
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.5
        low = re.sub(r"[^a-z0-9 ]", " ", t.lower())
        low = re.sub(r" +", " ", low)
        wn = max(1, len(re.findall(r"[A-Za-z']+", t)))
        per_k = 1000.0 / wn

        strong_d = len(_STRONG_RE.findall(t)) * per_k
        mild_d = len(_MILD_RE.findall(t)) * per_k
        # cap each CTA phrase at 2: repeated identical link text is nav chrome,
        # not an escalation of promotional register
        cta_d = sum(min(2, len(rx.findall(t))) for rx in _CTA_RES) * per_k
        excl_d = t.count("!") * per_k
        we_d = len(_WE_RE.findall(t)) * per_k
        junk_d = len(_JUNK_RE.findall(t)) * per_k
        spec_d = (len(_MONEY_RE.findall(t)) + len(_PCT_RE.findall(t))
                  + len(_QTY_RE.findall(t)) + len(_TICKER_RE.findall(t))) * per_k
        hedge_d = len(_HEDGE_RE.findall(t)) * per_k

        pen = (0.62 * _sat(max(0.0, strong_d - 1.2) / 3.0)  # soft-start
               + 0.20 * _sat(mild_d / 10.0)
               + 0.06 * _sat(cta_d / 14.0)
               + 0.03 * _sat(excl_d / 4.0)
               + 0.10 * _sat(we_d / 40.0)
               + 0.30 * _sat(max(0.0, junk_d - 15.0) / 30.0))
        # promotional-register interaction: puffery + CTA chrome together
        pen += 0.18 * _sat(mild_d / 9.0) * _sat(cta_d / 14.0)

        bonus = 0.0
        # gate: specifics/hedging can't buy back a hyped or puffery-dense document
        if strong_d < 2.0 and excl_d < 2.0 and mild_d < 8.0:
            bonus = 0.08 * _sat(spec_d / 8.0) + 0.05 * _sat(hedge_d / 5.0)

        s = 0.93 - pen + bonus

        # ---- optional LLM channel (thick grounding; no-op when extracted={}) ----
        if extracted:
            q1 = (extracted.get("sensational_quote") or "").strip()
            q2 = (extracted.get("unsupported_superlative") or "").strip()
            if q1.upper() == "NONE":
                q1 = ""
            if q2.upper() == "NONE":
                q2 = ""
            if q1:
                s -= 0.16 if _grounded(q1, low) else 0.07
            if q2:
                s -= 0.10 if _grounded(q2, low) else 0.04
            if not q1 and not q2:
                s += 0.06
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
