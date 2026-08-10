"""a72 hybrid agentic: Theme, idea, and lasting resonance.

Judge intuition (from train residuals): low scores are gag/parody/crude-humor
pieces (punchline structure, ALL-CAPS shouting, profanity-dense, Reddit
edit-churn); high scores are reflective, thematically controlled prose with
resonant endings. Topic keywords (v0 baseline) are a weak proxy. The tacit
mode/theme call goes to two short LLM fields; the predicate and all surface
robustness checks stay in code.

Round-1 diagnosis (agentic pass on top of h0): the structural formula (mode
gate + theme gate + diction bonuses + sentence-shape band) already tracks the
judge well (h0 TRAIN rho 0.687). The worst residuals are dominated by cases
where the LLM "mode"/"theme" fields say the right thing but the judge still
disagrees (mode=reflective pieces judged low, mode=plain pieces with a named
theme judged low) -- i.e. residual variance the code CANNOT see without
inventing new LLM fields (disallowed by the budget), not a formula bug.

One genuine, *measured* lexicon fix survived a density-vs-judge correlation
ablation (see below): `loneli\\w*` (meant to catch the loneliness/isolation
theme) never matches the single most common surface form "lonely" -- it's
spelled with a "y" where the regex root has an "i", so the regex only ever
fires on "loneliness"/"lonelier", never the base word. Fixing this raised the
raw density-vs-judge rho for that bucket (0.226 -> 0.238 on TRAIN) and is
kept. A parallel silent-e root fix (`forgive\\w*`->`forgiv\\w*` so "forgiving"
counts; `sacrifice`->`sacrific\\w*` so "sacrificial" counts) is roughly
rho-neutral and kept as a legitimate precision fix (no measured harm).

IMPORTANT NEGATIVE RESULT (measured, not assumed): broader recall expansion
of the same "_ABSTRACT" resonance lexicon -- adding `\\w*` to bare words
(beauty/wonder/regret/legacy/truth/peace/purpose/meaning/death/love/human/
mortal/eternal) and/or adding new on-topic resonance words tied to the
criterion's own name (remember/linger/endure/resonance/haunt/poignant/
unforgettable/profound/timeless) -- was tried and *measurably degraded* the
density-vs-judge correlation (0.226 -> 0.115 to 0.184 depending on variant;
mode=plain flips sign to negative). This is the known_failure_pattern
firing exactly as warned: more keyword recall just means more surface
topic-word hits inside mediocre genre pieces, diluting rather than
sharpening the signal for THIS criterion. Both expansions were reverted;
only the two narrow, measured-positive/neutral fixes above are kept.

Round-2/3/4 (see inline comments at point of use): mode/theme weight
recalibration to measured TRAIN group means (+0.013 rho), a new "narrative
development room" length term (+0.011 rho), and removing two h0 predicates
(dialogue-skit-shape penalty, flat sentence-length "flowing prose" band)
that were measurably net-negative for this criterion once checked against
full TRAIN rho on both odd/even halves independently. h0's TRAIN rho 0.687
-> 0.720 (candidate).

Round-5/6 (exploratory, both REJECTED, reported as negative results): (a)
a TF-IDF corpus-similarity "genre typicality" feature via
`ops.retrieve_similar` correlated with judge in comedy/reflective but
*negatively* in plain mode (the bucket most in need of help) -- dropped,
no consistent story; (b) further micro-regrids of crude/shout/churn/theme
weights each bought <0.003 TRAIN rho and were NOT robust across odd/even
TRAIN halves (one grid improved the full set but *degraded* one half) --
all rejected. Two consecutive rounds under the +0.005 plateau bar: this
program is left at the round-4 state, a considered stopping point, not a
forced one.

Residual, NOT fixable in code without a new LLM field (irreducibly
field-dominated, per worst-residual inspection): mode=reflective pieces
that are competent, sentimental, but derivative "premise+twist" WritingPrompts
pieces (e.g. a Christmas-truce retelling, a "kids literally playing gods"
frame, a Garden-of-Eden retelling) get the full reflective-mode credit yet
are judged low -- the criterion's "controlling idea... over mere clockwork"
distinction is a judgment about whether a theme is *earned* vs recycled,
which surface diction/length/mode signals cannot resolve; and symmetrically,
mode=plain pieces that name a real theme in passing (over a genre
action/horror plot) get theme credit they don't fully deserve. Both require
literary judgment a regex/length feature set cannot see.
"""
import re

LLM_FIELDS = {
    "mode": ("Classify the piece's dominant mode in ONE word: comedy "
             "(gags/parody/absurd humor), plain (straight genre storytelling), "
             "or reflective (emotionally or philosophically resonant)."),
    "theme": ("In at most 10 words, name the universal human theme the story "
              "seriously explores (e.g. mortality, love, loss, sacrifice); "
              "answer NONE if it is mainly a joke or parody."),
}

_CRUDE = re.compile(
    r"\b(fuck\w*|shit\w*|cock|cocks|dicks?|wank\w*|masturbat\w*|phallus|"
    r"boner|bitch\w*|douche\w*|assholes?|ass|tits?|crap\w*|piss\w*)\b",
    re.IGNORECASE)

_SHOUT = re.compile(r"\b[A-Z]{4,}\b")

_META_EDIT = re.compile(r"(?mi)^\s*\**\s*(final\s+)*edit\s*\d*\s*[:\-]")
_META_APOLOGY = re.compile(
    r"(?i)(sorry for (the )?(bad|poor|awful)?\s*(formatting|grammar|spelling)|"
    r"wrote this on my phone|word count\s*:)")

_SIMILE = re.compile(r"(?i)\b(like an?|as if|as though)\b")

# --- resonance/theme diction: h0's lexicon, with two narrow measured fixes ---
# (1) loneli\w* never matches "lonely" itself (loneli- vs lonely spelling) --
#     only "loneliness"/"lonelier" fired. Measured: fixing this raises the
#     density-vs-judge rho of this bucket (0.226 -> 0.238 on TRAIN).
# (2) forgive\w*/sacrifice miss the "-ing"/"-ial" inflections because the
#     root keeps a trailing silent "e" (forgiv\w*, sacrific\w* fix this).
#     Measured: rho-neutral (0.226 -> 0.220 alone, 0.232 combined with (1)).
# Everything else in this bucket is h0's ORIGINAL lexicon, unchanged. Broader
# recall expansion (bare-word \w* suffixes, new resonance words) was tried
# and measurably HURT (see module docstring) -- reverted.
_ABSTRACT = re.compile(
    r"(?i)\b(mortality|mortal|eternity|eternal|souls?|fate|destiny|grief|"
    r"sorrow|hope|memory|memories|silence|loneli\w*|lonely|alone|death|dying|love|"
    r"loss|meaning|purpose|humanity|human|beauty|wonder|forever|regret|"
    r"forgiv\w*|sacrific\w*|legacy|truth|peace)\b")

_TAIL_JUNK = re.compile(
    r"(?mi)^\s*(\**\s*(final\s+)*edit\b.*|.*\b(/r/|/u/|https?://|subreddit|"
    r"thanks? for reading|word count).*)$")

_MODE_COMEDY = re.compile(r"(?i)(comed|humor|humour|satir|parod|joke|absurd|farc|gag)")
_MODE_REFLECT = re.compile(r"(?i)(reflect|philosoph|poignan|melanchol|thought|"
                           r"literar|emotion|resonan|contemplat|introspect)")


def _sat(x):
    """Saturating ramp: 0 -> 0, 1+ -> 1."""
    if x <= 0.0:
        return 0.0
    return x if x < 1.0 else 1.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or len(text.strip()) < 40:
            return 0.5
        try:
            t = ops.normalize(text)
            if not isinstance(t, str) or not t:
                t = text
        except Exception:
            t = text

        words = re.findall(r"[A-Za-z']+", t)
        n_words = max(len(words), 1)
        kw = n_words / 1000.0

        s = 0.5

        # ---- code predicates: surface robustness signals ----
        # crude/profane punchline density (strong low-band marker)
        crude = len(_CRUDE.findall(t))
        s -= 0.12 * _sat((crude / kw) / 8.0)

        # ALL-CAPS shouting
        shout = len(_SHOUT.findall(t))
        s -= 0.08 * _sat((shout / kw) / 10.0)

        # reddit meta-churn: repeated edit notes, formatting apologies
        churn = 0.0
        if len(_META_EDIT.findall(t)) >= 2:
            churn += 1.0
        elif len(_META_EDIT.findall(t)) == 1:
            churn += 0.4
        if _META_APOLOGY.search(t):
            churn += 1.0
        s -= 0.05 * _sat(churn / 2.0)

        # Round-4 REMOVED (measured harm, see docstring): h0 had a
        # "dialogue-skit shape" penalty here (paragraphs mostly opening with
        # a quote -> -0.06). For THIS criterion that predicate is noise, not
        # signal: several of the highest-judged TRAIN pieces (e.g. an entire
        # devil/human dialogue scene, an author-confronts-his-creation
        # dialogue scene) are almost pure dialogue, and several of the
        # lowest-judged pieces are narration. Zeroing this term raised TRAIN
        # rho on both odd/even halves independently (not a single-fold
        # artifact) -- kept removed.

        # reflective / figurative diction (weak positive, saturating)
        simile = len(_SIMILE.findall(t))
        abstract = len(_ABSTRACT.findall(t))
        s += 0.06 * _sat((simile / kw) / 5.0)
        s += 0.06 * _sat((abstract / kw) / 14.0)

        # resonant ending: reflective diction near the close, plugs stripped
        body = _TAIL_JUNK.sub("", t).rstrip()
        tail = body[-500:]
        tail_hits = len(_ABSTRACT.findall(tail)) + len(_SIMILE.findall(tail))
        s += 0.05 * _sat(tail_hits / 4.0)

        # Round-4 REMOVED (measured harm, see docstring): h0 had a flat
        # "flowing prose band" bonus/penalty here keyed on mean-words-per-
        # sentence (13-26 words -> +0.04, <8 words -> -0.03), applied the
        # SAME direction regardless of mode. Measured: raw mwps correlates
        # NEGATIVELY with judge inside "comedy" (rho=-0.29, short punchy
        # sentences read as wittier) but POSITIVELY inside "reflective"
        # (rho=+0.31) -- a genuine interaction the flat band can't
        # represent, and forcing one direction on both nets negative on
        # TRAIN (checked both odd/even halves). Rather than invent a new
        # mode-conditional formula shape (higher overfitting risk for a
        # small measured gain), this term is simply dropped.
        #
        # The `flw` (fraction of >=9-letter words) lexical-richness term
        # that lived in the same h0 block is INTENTIONALLY left out too --
        # not because it hurt, but because it never fired at all: h0 gated
        # it at flw > 0.15, and on this corpus flw never exceeds ~0.13
        # (mean 0.048, max 0.129, checked across all 150 TRAIN docs), so it
        # was dead code in h0. Lowering the threshold to where it WOULD
        # fire doesn't help either -- raw flw is itself weakly NEGATIVELY
        # correlated with judge on TRAIN (rho=-0.07) -- so "fixing" this
        # inert term would make things worse, not better; left out.

        # Round-3: narrative "room to develop/test a theme" (measured signal
        # h0 didn't use -- h0 only ever looks at mean-words-per-SENTENCE, not
        # overall document length). Many worst residuals under h0/round-2
        # were short vignettes/quick-gag pieces that hit a controlling idea
        # in passing without room to embody or test it, and long-form pieces
        # that develop one scene/idea fully. Raw word count correlates with
        # judge score inside EVERY mode bucket on TRAIN (comedy rho=0.35,
        # plain rho=0.15, reflective rho=0.40) -- it is measured structure,
        # not a keyword predicate, and gates nothing; it only adds a modest,
        # saturating credit once a piece has passed a minimum-development
        # length, capped well below "reward raw length" territory.
        s += 0.06 * _sat((n_words - 200.0) / 600.0)

        # ---- LLM fields: tacit mode/theme grounding ----
        # Round-2 recalibration (measured, see docstring): h0's mode/theme
        # weights (-0.28 comedy / +0.22 reflective / -0.08 no-theme) were
        # NOT proportioned to the actual TRAIN group-mean gaps. Measured
        # judge means by mode: comedy=0.245, plain=0.315, reflective=0.437
        # (comedy is only ~0.07 below plain, not the ~2x-larger gap h0's
        # -0.28 implies); measured means by theme presence: no-theme=0.220,
        # has-theme=0.365 (a ~0.145 gap, more than h0's -0.08 implies). Only
        # these three existing scalar weights were retuned to the measured
        # gaps (comedy softened, reflective/no-theme sharpened); the
        # predicates themselves and everything else in the formula are
        # unchanged. Verified on full TRAIN (+0.013 rho) AND on both
        # odd/even TRAIN halves independently (both halves improve), so
        # this is not a single-fold artifact.
        ex = extracted if isinstance(extracted, dict) else {}
        mode = str(ex.get("mode", "") or "").strip().lower()
        if mode:
            if _MODE_COMEDY.search(mode):
                s -= 0.15
            elif _MODE_REFLECT.search(mode):
                s += 0.28
            # 'plain'/anything else: neutral

        theme = str(ex.get("theme", "") or "").strip()
        if "theme" in ex:
            tl = theme.lower()
            if not tl or tl in ("none", "none.", "n/a", "no"):
                s -= 0.14
            else:
                s += 0.05
                if _ABSTRACT.search(tl):
                    s += 0.05

        if s < 0.0:
            return 0.0
        if s > 1.0:
            return 1.0
        return s
    except Exception:
        return 0.5
