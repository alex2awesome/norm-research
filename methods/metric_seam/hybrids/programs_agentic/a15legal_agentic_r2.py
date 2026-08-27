# Hybrid module for legal_title_vii aspect a15: "hwe based on protected trait
# and unwelcome" (Oncale v. Sundowner). Hostile-environment harassment must be
# BOTH (a) linked to a protected trait (slur, trait-based comment/joke, sexual
# advance/touching) AND (b) shown to be unwelcome (plaintiff objected,
# reported it, told the harasser to stop, rebuffed it, avoided them). Bare
# doctrinal labels ("hostile work environment", "harassment") recited in a
# procedural aside (exhaustion, timeliness, retaliation-only, accommodation
# cases) do NOT satisfy the criterion.
#
# h0 diagnosis (FIELD-DOMINATED, rho=0.502 baseline -> h0 improves via two
# LLM fields carrying ~65% of the score weight, code channel a flat lexical
# density count). That makes h0's signal mostly a re-statement of the LLM's
# own yes/no read of the construct, with code doing little independent work.
#
# Recode strategy: move the CONDUCT/UNWELCOME predicate itself into code as
# far as regex can reach, using two upgrades over the plain keyword baseline
# that specifically target its two documented failure modes:
#   1. FALSE POSITIVES (bare "harassment"/"complained about" in a procedural
#      aside): replaced generic keyword hits with categorized, DESCRIPTIVE
#      conduct patterns (concrete verbs: groped/mocked/mimicked/stared at...)
#      -- a bare doctrinal label never matches any category, so it can no
#      longer buy partial credit on its own.
#   2. Trait-mockery verbs (mock/imitate/ridicule/belittle) are only counted
#      as protected-trait conduct when a protected-trait word (race, sex,
#      religion, age, national origin, disability, ...) appears within a
#      short character window of the verb -- this keeps ordinary personality-
#      conflict bullying (Oncale's "equal opportunity" abuse) from counting.
#   3. FALSE NEGATIVES (vivid concrete harassment that skips the exact
#      baseline keywords): widened conduct categories (sexual-advance verbs,
#      lewd-material mentions, trait-linked mockery/ridicule) plus a quoted-
#      speech boost -- a quoted remark that itself contains conduct-category
#      language is strong, code-legible evidence of a real, specific incident
#      (as opposed to a paraphrased legal label).
#   4. UNWELCOME is scored by its own reaction-verb category (rebuffed,
#      reported, objected, told to stop, complained about) and is only
#      counted at full strength when a reaction span sits within a short
#      distance of a conduct span -- linking "she complained" to the SAME
#      incident it is a reaction to, rather than to an unrelated grievance.
#   5. Severity scales with how many DISTINCT conduct categories are present
#      (a single crude remark vs. slurs+advances+isolation) which tracks how
#      the judge separates 0.5 (one described incident) from 0.85-1.0
#      (multiple, repeated, or escalating incidents).
# The two existing LLM fields (trait_conduct / unwelcome_reaction) are kept
# as a smaller-weight hedge/backstop for constructs the regex categories miss
# (paraphrase variety code can't enumerate), not as the primary driver.

import re

LLM_FIELDS = {
    "trait_conduct": (
        "In <=15 words, quote or paraphrase the SPECIFIC protected-trait-linked "
        "conduct alleged AGAINST THE PLAINTIFF (a slur, a race/sex/religion/age/"
        "disability-linked remark or joke, an unwanted sexual advance or "
        "touching). Answer NONE if the facts only use general labels like "
        "'harassment' or 'hostile work environment' WITHOUT describing any "
        "concrete incident directed at the plaintiff."
    ),
    "unwelcome_reaction": (
        "In <=15 words, state how the plaintiff showed that SAME trait-linked "
        "conduct was unwelcome (objected, reported it, told the harasser to "
        "stop, rebuffed/refused it, avoided them). Answer NONE if no such "
        "reaction to trait-linked conduct is described."
    ),
}

_NONE_LIKE = ("", "none", "n/a", "na", "not applicable", "not stated",
              "not mentioned", "no", "nothing", "unclear", "unknown")

_LABEL_ONLY = (
    "harassment", "hostile work environment", "discrimination",
    "unwelcome conduct", "unwelcome", "sexual harassment", "harassed",
)


def _clamp(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _present(raw):
    s = (raw or "").strip().strip(".").lower()
    if s in _NONE_LIKE:
        return False
    if s in _LABEL_ONLY:
        return False
    return len(s) >= 3


# ---- code layer: categorized, descriptive conduct/reaction detectors ----

_TRAIT_WORD_RE = re.compile(
    r"\b(?:rac(?:e|ial|ist)|sex(?:ual|ist)?|gender|religio(?:n|us)|"
    r"age(?:d|ist)?|national origin|ethnic(?:ity)?|disab(?:led|ility)|"
    r"color|nationality|accent|pregnan(?:t|cy)|asian|black|african[- ]american|"
    r"hispanic|latin[oa]|muslim|jewish|christian|gay|lesbian|homosexual)\b",
    re.I)

_CONDUCT_CATS = {
    "slur": re.compile(
        r"\b(?:racial slur|the n-word|n-word|ethnic slur|slurs?)\b", re.I),
    "sexual_advance": re.compile(
        r"\b(?:sexual advances?|propositioned|groped|fondled|"
        r"grabbed (?:her|his|him|their)\s+\w*(?:leg|legs|thigh|thighs|"
        r"buttocks?|arm|arms|breasts?|chest|waist|shirt)|"
        r"touched (?:her|him)(?:\s+\w+){0,2}\s+inappropriately|"
        r"ask(?:ed|ing) (?:her|him) (?:for|to have)\s+(?:oral|vaginal|sexual)|"
        r"invited (?:her|him) to (?:his|her) (?:house|home|apartment)|"
        r"stared at (?:her|his)\s+(?:breasts?|body|legs|buttocks|crotch)|"
        r"put (?:his|her) hands? (?:on|in|down)|"
        r"kissed (?:her|him)|unwanted (?:sexual )?touching|"
        r"discussed (?:his|her|the) sexual)\b", re.I),
    "lewd": re.compile(
        r"\b(?:lewd|obscene|sexually explicit|explicit (?:poster|clipping|"
        r"comments?|jokes?)|pornographic)\b", re.I),
    "trait_mockery": re.compile(
        r"\b(?:mock(?:ed|ing|s|ery)?|imitat(?:e|ed|ing|es|ion)|"
        r"mimic(?:k?ed|king|s)?|made fun of|jok(?:e|es|ed|ing)s? about|"
        r"ridicul(?:e|ed|ing)|derogatory (?:comments?|remarks?)|"
        r"disparaging (?:comments?|remarks?)|belittl(?:e|ed|ing|ement))\b",
        re.I),
}

_REACTION_RE = re.compile(
    r"\b(?:rebuffed|refused (?:his|her|their) advances|"
    r"reject(?:ed|ing) (?:his|her|their) advances|objected to|"
    r"told (?:him|her|them) to stop|"
    r"reported (?:the|his|her|this)?\s*(?:conduct|behavior|behaviour|"
    r"harassment|incident)s?|complained (?:about|to \w+ about)|"
    r"informed (?:her|his) supervisors?(?:\s+about)?|"
    r"filed a (?:complaint|charge)s? (?:about|regarding|alleging) "
    r"harassment)\b", re.I)

_QUOTE_RE = re.compile(r'"([^"]{3,220})"')

# Excludes matches that are actually about a THIRD PARTY's incident, not
# conduct against the plaintiff (e.g. a supervisor-plaintiff who merely
# relays/investigates a report of a slur made against a different associate
# -- the "hostile environment" here belongs to someone else's claim, not the
# plaintiff's own a15 predicate).
_THIRD_PARTY_RE = re.compile(
    r"\b(?:one of (?:his|her|their) (?:associates|employees)|"
    r"received a report (?:of|from)|a report (?:of|from) (?:one of|another))\b",
    re.I)

# Adverse-action / accommodation vocabulary the LLM field sometimes leans on
# even though the instruction asks for harassing CONDUCT (a remark/joke/
# slur/advance) -- demotions, terminations, and accommodation denials are
# disparate-treatment / failure-to-accommodate claims, not hostile-
# environment harassment, unless harassment-type language is also present.
_ADVERSE_ACTION_RE = re.compile(
    r"\b(?:demot\w*|terminat\w*|fired|discharg\w*|salary reduction|pay cut|"
    r"reduction in (?:force|pay)|denied? (?:a |an )?promotion|"
    r"failure to accommodate|denial of (?:the |a |an )?(?:accommodations?|"
    r"requests?)|unwanted \"?accommodations?\"?|reassign\w*|laid off|layoff|"
    r"disciplinary action|performance improvement plan|\bpip\b|"
    r"denied (?:a )?transfer|not(?:\s+be)? renewed|non-?renewal|"
    r"faking|not (?:really |actually )?(?:disabled|injured|sick)|"
    r"questioned (?:the legitimacy|whether)|disputed (?:his|her|the) "
    r"(?:medical|disability))\b", re.I)

_ANY_CONDUCT_RE = re.compile(
    "|".join(p.pattern for p in _CONDUCT_CATS.values()), re.I)


def _field_is_adverse_action_only(s):
    """True if the LLM field text reads as an adverse-action/accommodation
    claim rather than harassing conduct."""
    if not s:
        return False
    return bool(_ADVERSE_ACTION_RE.search(s)) and not _ANY_CONDUCT_RE.search(s)


def _conduct_spans(t):
    """Category -> list of match spans, with trait_mockery gated on a nearby
    protected-trait word (keeps ordinary personality-conflict bullying out)
    and all categories gated against a third-party-incident context."""
    hit_cats = {}
    for cat, pat in _CONDUCT_CATS.items():
        spans = []
        for m in pat.finditer(t):
            lo0, hi0 = max(0, m.start() - 220), min(len(t), m.end() + 220)
            if _THIRD_PARTY_RE.search(t[lo0:hi0]):
                continue
            if cat == "trait_mockery":
                lo, hi = max(0, m.start() - 70), min(len(t), m.end() + 70)
                if not _TRAIT_WORD_RE.search(t[lo:hi]):
                    continue
            spans.append(m.span())
        if spans:
            hit_cats[cat] = spans
    return hit_cats


def _quote_boost(t, hit_cats):
    """A quoted remark whose CONTENTS match a conduct category is strong,
    code-legible evidence of a concrete (not merely labeled) incident."""
    boost_cats = set(hit_cats)
    for qm in _QUOTE_RE.finditer(t):
        q = qm.group(1)
        for cat, pat in _CONDUCT_CATS.items():
            if cat in boost_cats:
                continue
            if pat.search(q) or (cat == "trait_mockery" and _TRAIT_WORD_RE.search(q)):
                boost_cats.add(cat)
    return boost_cats


def _reaction_linked(t, conduct_spans_flat):
    reaction_spans = [m.span() for m in _REACTION_RE.finditer(t)]
    if not reaction_spans:
        return False, False
    linked = any(
        abs(rs[0] - cs[0]) <= 400
        for rs in reaction_spans for cs in conduct_spans_flat
    )
    return True, linked


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        hit_cats = _conduct_spans(t)
        cats_present = _quote_boost(t, hit_cats)
        conduct_spans_flat = [sp for spans in hit_cats.values() for sp in spans]

        reaction_present, reaction_linked = _reaction_linked(t, conduct_spans_flat)

        conduct_strength = _clamp(len(cats_present) / 2.5)
        reaction_strength = 1.0 if reaction_linked else (0.5 if reaction_present else 0.0)
        code_core = _clamp(0.6 * conduct_strength + 0.4 * reaction_strength)

        ext = extracted or {}
        conduct_field = ext.get("trait_conduct", "")
        has_conduct = (_present(conduct_field)
                       and not _field_is_adverse_action_only(conduct_field))
        has_unwelcome = _present(ext.get("unwelcome_reaction", ""))
        llm_signal = 0.5 * has_conduct + 0.5 * has_unwelcome

        if not cats_present and not has_conduct:
            # Neither channel found concrete trait-linked conduct -- the
            # baseline's main false-positive mode (bare label / procedural
            # aside). Keep near zero regardless of any leftover LLM
            # unwelcome-only signal.
            return _clamp(0.10 * llm_signal)

        return _clamp(0.80 * code_core + 0.20 * llm_signal)
    except Exception:
        return 0.5
