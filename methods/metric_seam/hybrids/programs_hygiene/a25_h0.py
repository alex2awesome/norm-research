"""a25 — Truthfulness, substantiation, and non-deception (no hype/manipulation).

Predicate (kept in code): a document is credible on this criterion when it is
SUBSTANTIATED (attributed quotes, precise figures, wire/source/disclosure
markers, real contact blocks) and SOBER (no superlative hype, no emotionally
loaded/derisive advocacy, no ad-style calls-to-action, no raw template junk).
Presence of substantiation tokens alone is NOT quality: loaded advocacy and
hype multiplicatively crush the score even when numbers/quotes are present,
and CTA/junk penalties are gated so data-dense pages with site chrome are
not punished for the chrome.
"""
import math
import re

LLM_FIELDS = {
    "content_type": "In <=6 words classify the page's MAIN content: press release, news article, research/product information, navigation/link listing, or advertisement/marketing.",
    "manipulative_language": "List up to 3 hyped, exaggerated, or emotionally loaded persuasive phrases from the text; answer NONE if it is sober and factual.",
}

# --- lexicons -----------------------------------------------------------
_HYPE = re.compile(
    r"\b(?:amazing|incredible|unforgettable|unbelievable|revolutionary|"
    r"unprecedented|game[- ]?chang\w*|miracle|magical|world[- ]class|"
    r"best[- ]in[- ]class|cutting[- ]edge|state[- ]of[- ]the[- ]art|ultimate|"
    r"stunning|breathtaking|extraordinary|phenomenal|spectacular|sensational|"
    r"unparalleled|unmatched|unrivaled|hassle[- ]free|once[- ]in[- ]a[- ]lifetime|"
    r"top[- ]notch|insane(?:ly)?|killer|hottest|hot streak|must[- ]have|"
    r"exclusive|premier|innovative|luxurious|jaw[- ]dropping|epic|"
    r"guarantee[ds]?\b|100% (?:safe|free|effective))\b", re.I)

_LOADED = re.compile(
    r"\b(?:whopping|disgrace\w*|sabotag\w*|conceal\w*|scandal\w*|hoax\w*|"
    r"crooked|botch\w*|kludge\w*|crap\w*|pathetic|fud|astroturf\w*|"
    r"propaganda|witch[- ]?hunt\w*|fake news|cover[- ]up|corrupt\w*|"
    r"outrageous|deceit\w*|dishonest\w*|rips?[- ]off\w*|ripped[- ]off|"
    r"impersonat\w*|mutiny|cartel|prop up|faltering|so[- ]called|"
    r"open[- ]borders|radical (?:left|right)|disastrous|catastroph\w*|"
    r"shameful|sham\b|debunk\w*|smear\w*)\b", re.I)

_CTA = re.compile(
    r"\b(?:book now|buy now|order now|shop now|sign[- ]up|free trial|"
    r"start now|join now|act now|call today|limited time|% off|learn more|"
    r"get started|subscribe|register now|click here|don'?t miss|claim your|"
    r"special offer|money[- ]back|deals?\b)\b", re.I)

_JUNK = re.compile(r"href=|\{\{|\\r\\n|&gt;|&lt;|xdm:|class=\"|id=\"|</a>")

# --- substantiation feature regexes -------------------------------------
_ATTR_VERB = re.compile(
    r"\b(?:said|announced|stated|reported|told|noted|added|cautioned|"
    r"acknowledged|according to)\b", re.I)
_NAMED_ATTR = re.compile(
    r"\bsaid\s+[A-Z][a-z]+|\b[A-Z][a-z]+\s+(?:said|announced)\b")
_QUOTE = re.compile(r'"[^"\n]{25,400}"')
_ATTR_QUOTE = re.compile(
    r'"[^"\n]{25,400}[,"]?\s*(?:said|says|added|noted|charged|stated)\b|'
    r'\b(?:said|says|added|noted|stated)[^"\n]{0,60}"[^"\n]{25,400}"')
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?")
_PCT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent|per cent)", re.I)
_BIGNUM = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
_DECNUM = re.compile(r"\b\d+\.\d+\b")
_SCALE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:million|billion|trillion)\b", re.I)
_TICKER = re.compile(r"\b(?:NYSE|NASDAQ|Nasdaq)\s*:\s*[A-Z]+")
_SRC_STRONG = re.compile(
    r"prnewswire|globe newswire|business ?wire|for immediate release|"
    r"news provided by|forward[- ]looking statements|"
    r"private securities litigation reform act|media contact|press contact|"
    r"media inquiries|disclosure:|not receiving compensation|"
    r"expresses my own opinions|no business relationship", re.I)
# mild self-promotional spin tolerated by the judge only with a small dock
_MILD_SPIN = re.compile(
    r"record[- ](?:low|high|breaking|setting)|most robust", re.I)
_SRC_SOFT = re.compile(r"\bSOURCE\s+[A-Z][A-Za-z]|press releases?\b|news releases?\b", re.I)
_EMAIL = re.compile(r"[\w.\-]+@[\w\-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|\(\d{3}\)\s*\d{3}[-.\s]\d{4}|\b1-\d{3}-\d{3}-\d{4}\b")
_PROSE_SENT = re.compile(r"[A-Z][^.!?\n]{40,}[.!?]")


def _sat(x, k):
    """Saturating squash: 0 at 0, ->1 as x grows; k = softness."""
    return 1.0 - math.exp(-x / float(k))


def score(text, extracted, ops):
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        n_words = max(1, len(re.findall(r"[A-Za-z][A-Za-z'\-]*", t)))

        # ---- substantiation S (decoupled: structured evidence, not keywords)
        n_aq = len(_ATTR_QUOTE.findall(t))
        n_named = len(_NAMED_ATTR.findall(t))
        n_attr = len(_ATTR_VERB.findall(t))
        nums = (len(_MONEY.findall(t)) + len(_PCT.findall(t)) +
                len(_BIGNUM.findall(t)) + len(_DECNUM.findall(t)) +
                len(_SCALE.findall(t)) + len(_TICKER.findall(t)))
        n_src = min(4, len(_SRC_STRONG.findall(t))) + 0.5 * min(3, len(_SRC_SOFT.findall(t)))
        n_contact = min(4, len(_EMAIL.findall(t)) + len(_PHONE.findall(t)))
        s_raw = (1.6 * min(3, n_aq) + 0.8 * min(3, n_named) +
                 0.4 * min(6, n_attr) + 0.35 * min(10, nums) +
                 0.9 * n_src + 0.5 * n_contact)
        S = _sat(s_raw, 4.0)

        # ---- hype H (density; a single puff word is forgiven)
        n_hype = len(_HYPE.findall(t))
        H = min(1.0, max(0.0, n_hype - 1) / 4.0)

        # ---- loaded/derisive advocacy L
        n_loaded = len(_LOADED.findall(t))
        L = min(1.0, n_loaded / 3.0)

        # ---- promo CTA P (gated by substantiation later)
        n_cta = len(_CTA.findall(t))
        P = min(1.0, max(0.0, n_cta - 2) / 6.0)

        # ---- raw template/HTML junk J (with slack for stray anchors)
        n_junk = len(_JUNK.findall(t))
        J = min(1.0, max(0.0, n_junk - 4) / 10.0)

        # ---- thick-input adjustments from LLM extractions (optional)
        ex = extracted or {}
        ml = (ex.get("manipulative_language") or "").strip()
        if ml and ml.lower() not in ("none", "n/a", "no", "-"):
            items = [s for s in re.split(r"[;,\n]", ml) if s.strip()]
            L = max(L, min(1.0, 0.35 + 0.2 * len(items)))
        ct = (ex.get("content_type") or "").strip().lower()
        ct_mult = 1.0
        ct_floor = 0.0
        if ct:
            # NOTE (hygiene patch): content_type is short free text, and most
            # of these keywords are deliberate stems (navigat->navigation,
            # aggregat->aggregate, advertis->advertise*, promo->promot*,
            # news->news*, research->researcher*, report->report*) which stay
            # bare/\w*-stemmed since their observed hosts are genuine kin.
            # "link" and "product" were TRUE bugs: bare "link" matched
            # "linkedin"/"blinking"/"nvlink-c" and bare "product" matched
            # "production"/"productivity"/"counterproductive" -- unrelated
            # words. Both are now \b-anchored with only the inflections the
            # concept actually needs.
            if (re.search(r"\bnavigat\w*\b", ct) or "listing" in ct
                    or "menu" in ct or "index" in ct or "directory" in ct
                    or re.search(r"\baggregat\w*\b", ct)
                    or re.search(r"\blinks?\b", ct)):
                ct_mult = 0.35
            elif (re.search(r"\badvertis\w*\b", ct) or "marketing" in ct
                    or re.search(r"\bpromo\w*\b", ct) or "sales" in ct):
                ct_mult = 0.45
            elif ("press release" in ct or re.search(r"\bnews\w*\b", ct)
                    or re.search(r"\bresearch\w*\b", ct)
                    or re.search(r"\bproducts?\b", ct)
                    or "information" in ct or re.search(r"\breport\w*\b", ct)):
                ct_floor = 0.45  # substantive content the code's regexes may miss

        # ---- combine: substantiation crushed multiplicatively by deception
        tone = 1.0 - 0.85 * max(H, L)
        mild = 1.0 - 0.1 * min(3, len(_MILD_SPIN.findall(t)))
        p_eff = P * (1.0 - S)          # chrome CTAs forgiven on data-dense docs
        j_eff = J * (1.0 - 0.6 * S)    # stray anchors forgiven on real releases
        base = (max(S, ct_floor) * tone * mild *
                (1.0 - 0.45 * p_eff) * (1.0 - 0.5 * j_eff))

        # ---- sober-prose floor: coherent factual prose with zero spin gets
        # partial credit even without classic PR substantiation markers.
        # Prose CHARACTER coverage separates paragraph pages from headline/
        # nav aggregators (short title lines contribute no covered chars).
        prose_chars = sum(len(m) for m in _PROSE_SENT.findall(t))
        coverage = prose_chars / max(1, len(t))
        floor = 0.0
        if n_words >= 150:
            floor = (min(0.4, 0.55 * coverage) * (1.0 - max(H, L)) *
                     (1.0 - P) * (1.0 - J))
        val = max(base, floor) * ct_mult
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.3
