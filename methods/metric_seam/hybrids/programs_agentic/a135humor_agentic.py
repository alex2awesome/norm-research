# Hybrid scoring module for humor aspect a135:
# "Platform/broadcast standards and constraints" -- sponsor/network/platform
# taste, profanity, libel, and censorship rules that govern what can be shown
# or said and where it can run.
#
# Agentic revision of a135_h0.py. Round-1 diagnosis on TRAIN found three
# concrete boundary-matching bugs in h0's lexicon regexes (all confirmed by
# real mismatched train items, not hypothetical):
#   (1) `_SEVERE_SLUR_RE` used `spic\w*` inside a `\b...\b` wrapper. Because
#       `\w*` is greedy and the closing `\b` only needs to land at the next
#       non-word boundary, `spic\w*` matches the ENTIRE word "spice" (spic +
#       the trailing "e" consumed by `\w*`, boundary satisfied after it).
#       This zeroed out d02055 (a clean "spice rack" pun, judge=1.00) as if
#       it contained a severe slur.
#   (2) Same bug class in `_STRONG_PROFANITY_RE`'s `cock\w*`, which matches
#       "cockpit" and "cocktail" wholesale. d04493 ("the pilot in the
#       cock-pit hears the noise", judge=1.00) got flagged as profanity.
#   (3) `_VIOLENCE_VERB_RE`'s `punch\w*` matches "punchline" -- an extremely
#       common word in a joke corpus. Crossed with `_VULNERABLE_TARGET_RE`
#       ("nun"), this fired a violence hazard on d03691 (clean, judge=1.00)
#       purely because the text contains the word "punchline" near "nun".
# Fix: a shared boundary-aware matcher (`_pattern`/`_lex_any`, ported from
# the a351 fix on this same fleet) that restricts single-word lexicon terms
# to a small explicit inflectional-suffix whitelist (('s|s|es|ed|ing)) instead
# of an open `\w*`, so "spic"/"cock"/"punch" no longer swallow the rest of an
# unrelated host word. Multi-word phrases are matched verbatim as before.
#
# Second finding: aggregate TRAIN stats (binned by the two LLM fields) show
# h0's flat severity constants are miscalibrated:
#   harm_severity=''      (n=124) mean_judge=0.82  -> severity should be ~0
#   harm_severity='MILD'  (n=13)  mean_judge=0.39  -> severity should be ~0.5,
#                                                     NOT 0.5 blended with a
#                                                     flat +0.4 target bump
#   harm_severity='SEVERE'(n=13)  mean_judge=0.16  -> severity should be ~0.8,
#                                                     not the hard 1.0 h0 used
#                                                     (SEVERE items still
#                                                     average judge=0.16, not
#                                                     0 -- h0's hard zero
#                                                     overshoots systematically)
#   target_group present   (n=26) mean_judge=0.53  -> a bare named target
#                                                     (no severity flag) is a
#                                                     WEAK, noisy signal on its
#                                                     own (means range 0.06 to
#                                                     1.00 depending on what
#                                                     else co-occurs) -- h0's
#                                                     flat -0.4 penalty is too
#                                                     strong when nothing else
#                                                     is flagged, and stacks
#                                                     incorrectly with severity.
# These constants below (0.50 / 0.78 base, +0.25 target boost) are refit from
# that binned table, not from any single held-out item.
#
# Third finding: the LLM fields never fire on explicit-anatomy/gross-out
# vocabulary (penis, dildo, semen, vomiting-for-a-bet, etc.) because that's
# not "harm to a protected group/self" -- it's a plain broadcast-decency
# violation the extractor isn't asked about. All train items containing this
# vocabulary score low (mean_judge=0.40, n=10) yet h0's code channel has no
# lexicon for it at all, so they fell through as a false "clean" (1.0). Added
# a small explicit-content code lexicon (gated on concrete anatomical/bodily
# terms, not topic words) to catch this residual code-reachable class.
#
# Output: 1.0 = clean / broadcast-safe, 0.0 = would clearly violate
# platform/broadcast standards.

import re
import math

LLM_FIELDS = {
    "harm_severity": (
        "Rate how much serious harm (violence, sexual assault/exploitation, "
        "hate, self-harm, child harm) this joke plays for laughs: "
        "NONE, MILD, or SEVERE."
    ),
    "target_group": (
        "Name a real protected group (race, religion, ethnicity, gender, "
        "orientation, disability) this joke demeans or mocks; else say NONE."
    ),
}

# ---------------------------------------------------------------------------
# Boundary-aware lexicon matching (fixes bug classes 1-3 above).
# ---------------------------------------------------------------------------

_PATTERN_CACHE = {}


def _pattern(term):
    """Boundary-aware, light-inflection-aware regex for one lexicon term.

    Phrases (space/hyphen inside) match verbatim between word boundaries.
    Single words allow a small whitelist of regular inflectional suffixes
    ('s / s / es / ed / ing) instead of an open `\\w*`, so "spic" does not
    swallow "spice" and "punch" does not swallow "punchline" -- a plain
    `\\bterm\\w*\\b` matches both because the greedy `\\w*` is free to consume
    the rest of the host word up to its own boundary. This was the single
    biggest source of mis-scored train items in h0.
    """
    cached = _PATTERN_CACHE.get(term)
    if cached is not None:
        return cached
    esc = re.escape(term)
    if " " in term or "-" in term:
        pat = r"\b" + esc + r"\b"
    else:
        pat = r"\b" + esc + r"(?:'s|s|es|ed|ing)?\b"
    compiled = re.compile(pat, re.I)
    _PATTERN_CACHE[term] = compiled
    return compiled


def _lex_any(hay, needles):
    try:
        return any(_pattern(n).search(hay) for n in needles)
    except Exception:
        return False


def _lex_count(hay, needles):
    try:
        return sum(len(_pattern(n).findall(hay)) for n in needles)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Lexicons (single words go through the whitelist-suffix matcher above;
# phrases are matched verbatim). Values kept close to h0's, only fixing the
# specific collision-prone roots and adding the new explicit/gross category.
# ---------------------------------------------------------------------------

_SEVERE_SLURS = [
    "nigger", "nigga", "chink", "spic", "kike", "faggot", "retard", "retarded",
    "tranny", "trannies", "wetback", "coon",
]

# Strong profanity is NOT one uniform severity tier. Per-word TRAIN means
# (binned on items where that single word is the only strong-profanity hit):
# "fuck" n=11 mean_judge=0.245 (severe), "shit"/"dick" n=6/2 mean~0.38 (more
# moderate -- often used as a literal/mild exclamation, "oh shit", "lake of
# shit", rather than a targeted attack word). TIER_A gets the h0 peak; TIER_B
# is dialed back to match its weaker observed effect.
_STRONG_PROFANITY_TIER_A = ["fuck", "cunt", "motherfucker"]
_STRONG_PROFANITY_TIER_B = ["shit", "pussy", "asshole", "dumbass", "jackass"]
# "cock" and "dick" are handled separately below (`_cock_hits`/`_dick_hits`):
# both collide with common benign words/names ("cockpit", the name "Dick")
# that a plain boundary+suffix match cannot distinguish on its own. "cock"
# folds into TIER_B's weaker peak; "dick" gets its own peak matching its
# observed (thin, n=2) TRAIN mean.

_COCK_BENIGN_COMPOUNDS = (
    "cockpit", "cocktail", "cockatoo", "cockroach", "peacock",
    "shuttlecock", "gamecock", "cockerel",
)

# MILD interjections ("hell", "crap", "bloody", "damn") barely move the judge
# at all when they're the only code signal present (TRAIN: mild-hit-only
# items mean_judge=0.84, ~= the no-hit baseline of 0.84). "bastard" is the
# outlier (n=2, mean_judge=0.20 -- it's used as a real insult, not an
# interjection, in both train occurrences) but the sample is too thin to
# split out its own tier without overfitting two items; left in this list,
# the shared peak is now low enough not to swamp cases where it's the only
# hit while still contributing to `secondary_sum` when paired with a
# stronger signal.
_MILD_PROFANITY = ["damn", "hell", "crap", "bastard", "bloody", "piss"]
_MILD_PEAK = 0.12

_HAZARD_HATE = [
    "nazi", "hitler", "holocaust", "genocide", "gas chamber", "final solution",
    "mein fuhrer", "mein führer", "fuhrer", "führer", "kkk", "lynch",
    "concentration camp",
]

_HAZARD_SELFHARM = [
    "suicide", "suicidal", "kill myself", "self-harm", "self harm",
    "overdose", "hanged himself", "hanged herself", "hanged themselves",
    "hanging himself", "hanging herself", "hanging themselves",
]

_HAZARD_SEXUAL = [
    "rape", "raping", "raped", "molest", "molested", "non-consensual",
    "nonconsensual", "anus", "semen", "sperm",
]

_META_RACISM_RE = re.compile(r"\bn[- ]word\b", re.I)

_VIOLENCE_VERBS = [
    "beat", "kick", "punch", "stomp", "stab", "strangl", "chok", "slap",
    "assault",
]

_VULNERABLE_TARGETS = [
    "nun", "priest", "child", "kid", "kids", "boy", "girl", "baby", "babies",
    "infant", "toddler", "elderly", "old man", "old woman", "disabled",
    "orphan",
]

# NEW: plain broadcast-decency vocabulary the two LLM fields never fire on
# (it isn't "harm to a group/self", just explicit content). Concrete
# anatomical/sexual-act nouns only -- no topic words like "sex" or "naked"
# alone, which are too generic and would gate on innuendo rather than the
# actual predicate.
_EXPLICIT_ANATOMY = [
    "penis", "dildo", "vagina", "vulva", "testicle", "testicles", "orgasm",
    "ejaculate", "ejaculating", "semen", "sperm", "cum", "boner", "clitoris",
    "erection", "erectile", "masturbat", "vibrator",
]
_EXPLICIT_PHRASES = ["jacked off", "jerked off", "jacking off", "jerking off"]
_GROSS_BODILY = ["vomit", "vomiting", "vomited", "puke", "puked", "puking"]

_STEREOTYPE_PHRASES = ["those people", "you people", "typical"]

_EMPTY_ANSWER_PREFIXES = (
    "none", "n/a", "na", "no", "unsure", "unknown", "not applicable",
    "not clear", "unclear",
)


def _is_empty_answer(s):
    s2 = (s or "").strip().strip(".").lower()
    if not s2:
        return True
    return any(s2.startswith(p) for p in _EMPTY_ANSWER_PREFIXES)


def _cock_hits(low):
    """Count real profane uses of "cock", excluding benign compounds the
    boundary+suffix matcher alone can't rule out ("cock-pit" hyphenated --
    the hyphen itself is a regex word boundary, so `\\bcock\\b` still matches
    inside it even though the suffix whitelist correctly rejects the
    unhyphenated "cockpit"). Checked via a small local context window rather
    than a fixed regex lookaround so it also catches "cock tail", "pea cock",
    etc. with a space instead of a hyphen."""
    hits = 0
    for m in _pattern("cock").finditer(low):
        window = low[max(0, m.start() - 6):m.end() + 9].replace("-", "").replace(" ", "")
        if any(bc in window for bc in _COCK_BENIGN_COMPOUNDS):
            continue
        hits += 1
    return hits


def _dick_hits(orig, low):
    """Count real profane uses of "dick", excluding the common personal name
    (Dick/Richard). Written English overwhelmingly capitalizes the name and
    lowercases the profanity in this casual reddit-style corpus, so require
    the ORIGINAL (non-lowercased) text to have the match in lowercase --
    "Dick graciously replies" / "Dick's out for her ombre" (the name) are
    skipped, but "what a dick" would still count."""
    hits = 0
    for m in _pattern("dick").finditer(low):
        if orig[m.start():m.end()] == low[m.start():m.end()]:
            hits += 1
    return hits


def _code_severity(orig, low):
    """Lexical hazard channel -> magnitude in [0, 1]. Boundary-aware
    throughout (see `_pattern`), so short collision-prone roots (spic, cock,
    punch, chok) only match real inflections of the intended word, not host
    words that happen to contain them as a substring."""
    components = []

    if _lex_any(low, _SEVERE_SLURS):
        components.append(1.0)

    if _lex_any(low, _HAZARD_HATE):
        components.append(0.75)
    if _lex_any(low, _HAZARD_SELFHARM):
        components.append(0.75)
    if _lex_any(low, _HAZARD_SEXUAL):
        components.append(0.65)
    if _META_RACISM_RE.search(low):
        components.append(0.6)

    # Violence: graduated by how many distinct violent-verb hits co-occur
    # with a vulnerable target, rather than a flat constant. h0's flat 0.7
    # over-penalized a single hypothetical/verbal threat (e.g. one "choke")
    # as hard as sustained depicted violence (four verbs: kicked/beating/
    # kicking/punching) -- the two are not equally certain signals of a real
    # broadcast-standards violation.
    n_violence = _lex_count(low, _VIOLENCE_VERBS)
    if n_violence >= 1 and _lex_any(low, _VULNERABLE_TARGETS):
        violence_sev = 0.35 + 0.40 * (1.0 - math.exp(-0.55 * (n_violence - 1)))
        components.append(min(0.75, violence_sev))

    n_expl = _lex_count(low, _EXPLICIT_ANATOMY) + _lex_any(low, _EXPLICIT_PHRASES)
    if n_expl:
        components.append(min(0.60, 0.40 + 0.10 * (n_expl - 1)))
    if _lex_any(low, _GROSS_BODILY):
        components.append(0.40)

    tier_a_hits = _lex_count(low, _STRONG_PROFANITY_TIER_A)
    tier_b_hits = _lex_count(low, _STRONG_PROFANITY_TIER_B) + _cock_hits(low)
    dick_hits = _dick_hits(orig, low)
    mild_hits = _lex_count(low, _MILD_PROFANITY)

    tier_a_sat = 1.0 - math.exp(-1.3 * tier_a_hits)
    tier_b_sat = 1.0 - math.exp(-1.3 * tier_b_hits)
    dick_sat = 1.0 - math.exp(-1.3 * dick_hits)
    mild_sat = 1.0 - math.exp(-0.8 * mild_hits)
    components.append(0.75 * tier_a_sat)
    components.append(0.40 * tier_b_sat)
    components.append(0.45 * dick_sat)
    components.append(_MILD_PEAK * mild_sat)

    if _lex_any(low, _STEREOTYPE_PHRASES):
        components.append(0.4)

    if not components:
        return 0.0
    base = max(components)
    secondary_sum = sum(c for c in components if c < base)
    return min(1.0, base + 0.15 * min(1.0, secondary_sum))


# Recalibrated against TRAIN aggregate stats binned by (harm_severity,
# target_group present), not against any single held-out item:
#   ('', False)      n=107 mean_judge=0.84
#   ('', True)       n=17  mean_judge=0.73  -- BUT this bucket is bimodal, not
#                    a shifted-down clean population: 7/17 items are an exact
#                    judge=1.00 (the LLM named a target -- nationality of a
#                    spy character, a nun's religion, a disability metaphor
#                    name -- for what is really just a neutral descriptor, not
#                    a demeaning joke) while the rest range .25-.95. Since
#                    harm_severity='' means the LLM itself saw no harm, a
#                    bare target name alone is too weak/bimodal a signal to
#                    move score on -- boosting here made TRAIN rho *worse*
#                    (0.787 with no boost vs 0.782 with the flat +0.25 boost
#                    round-1 used). Confirmed by a boost-value sweep on TRAIN:
#                    gating the boost to only fire alongside a MILD/SEVERE
#                    harm flag raised rho from 0.782 to 0.793; the bare-name
#                    case is left at the harm_severity-only base rate.
#   ('MILD', False)  n=8   mean_judge=0.49
#   ('MILD', True)   n=5   mean_judge=0.21
#   ('SEVERE', False)n=9   mean_judge=0.21
#   ('SEVERE', True) n=4   mean_judge=0.06
# The bucket means above nominally point to base~0.50/0.78 with a +0.20
# boost, but once the code channel was retiered (profanity split into
# fuck/cunt vs shit/dick tiers, explicit-anatomy lexicon added) the code and
# LLM channels compete more often via `max()`, which shifts the LLM
# constants that best fit TRAIN as a whole. A small grid search around the
# bucket-derived values (never touching individual items, just these 3
# scalars) found 0.40/0.70/+0.30 nudges TRAIN rho up further (0.797->0.800);
# kept one step back from that grid optimum to avoid chasing 4-decimal noise
# in a 150-item grid search.
_LLM_BASE_SEVERITY = {"": 0.0, "MILD": 0.42, "SEVERE": 0.72}
_TARGET_BOOST = 0.28


def _llm_severity(extracted):
    if not isinstance(extracted, dict):
        return 0.0
    harm_txt = (extracted.get("harm_severity") or "").strip().upper()
    target_txt = (extracted.get("target_group") or "").strip()

    base = _LLM_BASE_SEVERITY.get(harm_txt, 0.0)
    if harm_txt not in _LLM_BASE_SEVERITY:
        # Unrecognized free-text answer (extractor didn't follow the
        # NONE/MILD/SEVERE contract) -- treat as a mild-strength signal
        # rather than ignoring it outright or over-trusting it.
        base = 0.35

    # A named target only moves the score when it co-occurs with a MILD/
    # SEVERE harm flag -- see note above on why a bare target name alone
    # (harm_txt == '') is not applied.
    if harm_txt and not _is_empty_answer(target_txt):
        base = min(1.0, base + _TARGET_BOOST)

    return base


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        t = ops.normalize(raw)
        if not t or not t.strip():
            return 0.5
        low = t.lower()

        code_sev = _code_severity(t, low)
        llm_sev = _llm_severity(extracted)

        severity = max(code_sev, llm_sev)
        s = 1.0 - severity
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        return s
    except Exception:
        return 0.5
