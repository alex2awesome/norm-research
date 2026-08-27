"""
Hybrid metric channel for aspect a351: "Representation ethics and harm minimization".

Criterion: avoid minimizing trauma for laughs; attend to ableist, racist, and gendered
harms; avoid false equivalences and exploitative portrayals; pursue remediation and
sensitivity.

Design notes (agentic round 2 -- see file history for round-1 diagnosis):
- ROUND-1 DIAGNOSIS found two classes of bug in the h0 program, both fixed here:
  (1) naive substring lexicon matching produced false hazard hits inside unrelated
      words ("spic" inside "spice rack", "punch" inside "punchline") and missed
      regular inflections of real hits ("kids"/"jumps"). Fixed with a boundary-aware,
      light-inflection-aware regex matcher (`_lex_any`) used everywhere a keyword list
      is checked, in both code_hazard and the LLM-stance reader.
  (2) the LLM-hazard reader used undifferentiated flat constants (0.85 mock / 0.3
      critique / 0.55 ambiguous) regardless of *what* was demeaned, so a mild banter
      stereotype (blonde/nationality jokes) got the same penalty as a named-protected-
      identity or atrocity/trauma target, and a target-empty-but-stance-flagged case
      was silently zeroed out. Fixed with a 3-tier target-severity classifier
      (OFFTOPIC/DEFAULT/SEVERE, based on keywords in the *harm_target* extraction)
      crossed with a stance modifier, and a fallback path so a clearly-trivializing
      stance is not ignored just because the target field came back empty.
  (3) stance text is discursive LLM prose, not a keyword list, so naive substring
      matching on stance words is negation-blind: "situational irony rather than
      *endorsing* harm" hit the trivialize keyword "endors" although the sentence
      explicitly disclaims endorsement. Added a small negation-window check
      (`_negated`) so a hit preceded by "not/without/rather than/..." is discarded.
- Code still carries the PREDICATE: explicit slurs, atrocity/genocide references,
  explicit sexual-violence vocabulary, and lexical co-occurrence patterns for
  well-known exploitative tropes (undressing near a minor with a dismissive
  punchline, violence against a vulnerable/protected target, child-abuse-for-laughs,
  suicidal-vulnerable-person + sexual solicitation, and a NEW pattern: sexual content
  directed at someone described as incapacitated/unable to consent).
- Two LLM_FIELDS (UNCHANGED from h0 -- same names, same budget) supply THICK-INPUT
  grounding for constructs code cannot reliably reach from lexicon alone: (a) *which*
  protected group / trauma type the joke's humor depends on demeaning, and (b) the
  text's *stance* toward that target.
- Final score = 1 - max(code_hazard, llm_hazard) + small remediation bonus, clipped to
  [0, 1]. Using max (not sum/average) means either channel alone is sufficient to flag
  harm -- recall matters more than precision here (false negatives on exploitative
  content are worse than a few over-cautious flags).
"""

import re
import math
import statistics
from collections import Counter

LLM_FIELDS = {
    "harm_target": "Name the protected/vulnerable group, trauma type, or atrocity the joke's punchline demeans or trivializes (e.g. child sexual abuse, racial stereotype, disability mockery, domestic violence, genocide), or say NONE.",
    "harm_stance": "In <=20 words: does the text mock/trivialize/endorse harm toward that target, or instead critique/is self-aware/neutral about it, or NONE.",
}

_SLURS = [
    "retard", "retarded", "faggot", "fag", "tranny", "nigger", "nigga",
    "kike", "spic", "chink", "gook", "cripple", "midget", "coon",
]

_GENOCIDE_MARKERS = [
    "mein fuhrer", "mein führer", "final solution", "gas chamber",
    "concentration camp", "holocaust", "zyklon", "auschwitz",
    "ss officer", "ethnic cleansing", "genocide",
]

_SEXUAL_VIOLENCE_WORDS = [
    "rape", "raping", "raped", "molest", "molested",
    "non-consensual", "nonconsensual",
]

_UNDRESS_MARKERS = ["unbuckl", "unbutton", "unzip"]
_MINOR_MARKERS = ["boy", "girl", "kid", "child", "children", "little"]
_DISMISS_MARKERS = [
    "not your day", "not my day", "sorry kid",
    "wait at the bottom", "just not your day",
]

_SUICIDE_CONTEXT = ["jump", "bridge", "cliff", "ledge", "roof"]
_SOLICIT_MARKERS = ["sex", "fuck me", "have sex", "touch"]

_VIOLENCE_VERBS = [
    "kick", "beat", "beating", "punch", "beaten", "assault",
]
_VULNERABLE_TARGETS = [
    "nun", "homeless", "elderly", "blind", "deaf", "disabled",
    "wheelchair", "baby", "babies", "newborn", "infant", "toddler",
]

_CHILD_ABUSE_MARKERS = ["cigarette"]
_CHILD_ABUSE_VICTIMS = ["baby", "babies", "newborn", "infant"]

_STEREOTYPE_PHRASES = ["those people", "you people", "typical"]

_REMEDIATION_MARKERS = ["content warning", "trigger warning", "cw:"]

# NEW (round 2): sexual content directed at someone described as incapacitated /
# unable to consent -- a distinct exploitative trope not covered by the explicit
# sexual-violence word list (no "rape"/"molest" vocabulary is used, but consent is
# structurally impossible given the described state).
_INCAPACITATION_MARKERS = [
    "coma", "unconscious", "passed out", "sedated", "unresponsive",
    "knocked out", "blacked out",
]
_SEXUAL_CONTENT_MARKERS = [
    "private area", "private parts", "sponge bath", "stimulated",
    "aroused", "turned on", "oral sex", "orgasm", "climax", "fondle", "grope",
]

_TRIVIALIZE_STANCE = [
    "trivial", "normaliz", "endors", "mock", "demean", "dehuman",
    "laugh at", "false equivalen", "glorif", "reinforc",
]
# Round 4 split: "genuine" mitigation words are evaluative -- the extractor is
# reporting that the text itself disapproves, apologizes for, or reflects on the
# harm -- and are trusted regardless of target. "framing" words (self-aware,
# subvert, satire) merely describe TONE, not evaluation, and round-3 showed they
# over-forgive dark humor about real trauma/death just because the extractor
# called it "satirical" (a joke trivializing a friend's combat death, self-
# reported as "neutral/satirical", was almost fully forgiven -- judge disagreed
# sharply). Political/institutional satire (mocking a government, a dictator, a
# state security service) is exactly what "framing" words are meant to credit;
# jokes about war/death/disaster/casualties are not -- see _TRAUMA_TARGET_TERMS
# gate below.
_GENUINE_MITIGATE_STANCE = [
    "critiqu", "critical", "condemn", "apolog",
    "sympath", "reflect", "sensitiv", "remediat",
]
_FRAMING_MITIGATE_STANCE = ["self-aware", "self aware", "subvert", "satir"]
_STRONG_MITIGATE_STANCE = _GENUINE_MITIGATE_STANCE + _FRAMING_MITIGATE_STANCE

_TRAUMA_TARGET_TERMS = [
    "war", "veteran", "casualt", "combat", "grave", "cemetery",
    "trauma", "disaster",
]

# harm_target severity tiers (round 2). The criterion explicitly scopes to
# "ableist, racist, gendered harms" and "trauma" -- generic crude/gross-out humor
# that happens to get flagged by an over-eager extractor is OUT of scope (near-zero
# weight); a named protected-identity/atrocity/trauma target is IN scope at the
# high end; everything else flagged (unnamed "stereotype", banter categories like
# blonde/nationality jokes, mild content) sits in between.
_OFFTOPIC_TARGET_TERMS = [
    "bestiality", "zoophil", "toilet humor", "gross-out", "flatulence",
    "body fluid", "scatolog", "profanity",
]
_SEVERE_TARGET_TERMS = [
    "disab", "ableis", "wheelchair", "blind", "deaf", "autis", "down syndrome",
    "lgbtq", "gay", "lesbian", "transgender", "trans", "bisexual", "queer",
    "black people", "african american", "racis", "antisemit", "jewish",
    "muslim", "islam", "genocide", "holocaust", "ethnic cleansing", "atrocity",
    "torture", "sexual assault", "rape", "molest", "child abuse", "pedophil",
    "domestic violence", "gendered violence", "misogyn", "hate crime",
    "war", "veteran", "casualt", "grave", "cemetery", "trauma", "disaster",
]

_OFFTOPIC_PEAK = 0.08
_DEFAULT_PEAK = 0.38
_SEVERE_PEAK = 0.60
_STRONG_MITIGATE_FACTOR = 0.15   # fraction of tier peak once self-aware/critique fires
_TRAUMA_FRAMING_FACTOR = 1.0     # "framing" words don't discount a trauma-tier target
_AMBIGUOUS_FACTOR = 0.6          # target named, stance empty
_WEAK_FACTOR = 0.8               # target named, stance present but non-committal
_BARE_NEUTRAL_FACTOR = 0.35      # stance field is JUST "neutral", no elaboration
_BARE_NEUTRAL_RE = re.compile(r"^neutral\.?$")

_NEGATORS = (
    "not ", "n't", "never ", "without ", "no ", "rather than ",
    "instead of ", "isn't", "aren't", "doesn't", "don't", "didn't",
)

_PATTERN_CACHE = {}
_STEM_PATTERN_CACHE = {}


def _safe_lower(s):
    try:
        return s.lower()
    except Exception:
        return ""


def _pattern(term):
    """Boundary-aware, light-inflection-aware regex for one CONCRETE lexicon term
    (slurs, violence verbs, minor/vulnerable-target markers, etc).

    Phrases (containing a space or hyphen) match verbatim between word
    boundaries. Single words allow a small whitelist of regular inflectional
    suffixes so "jump" still matches "jumps"/"jumped"/"jumping"/"jump's", but
    "spic" does not match inside "spice" and "punch" does not match inside
    "punchline" -- a plain substring check conflates both and was the single
    biggest source of mis-scored train items in round 1. These are short,
    common words where the collision risk with unrelated words is real, so a
    narrow whitelist (rather than an open-ended stem match) is the safer
    default.
    """
    cached = _PATTERN_CACHE.get(term)
    if cached is not None:
        return cached
    esc = re.escape(term)
    if " " in term or "-" in term:
        pat = r"\b" + esc + r"\b"
    else:
        pat = r"\b" + esc + r"(?:'s|s|es|ed|ing)?\b"
    compiled = re.compile(pat)
    _PATTERN_CACHE[term] = compiled
    return compiled


def _stem_pattern(term):
    """Open-ended prefix match (`term` + any word chars) for LONG, DISTINCTIVE
    discursive stems used to read the LLM's free-text stance sentence (e.g.
    "dehuman", "sensitiv", "satir", "critiqu"). Unlike the short concrete words
    above, these multi-syllable stems have no realistic unrelated-word collision
    (nothing common starts with "dehuman" besides the dehumanize family), so an
    open suffix is the correct way to catch derivational forms a fixed
    inflection whitelist would miss (e.g. "satire"/"satirical", "critique"/
    "critical", "sensitivity", "remediation" -- none of which are reachable by
    a small s/es/ed/ing suffix set).
    """
    cached = _STEM_PATTERN_CACHE.get(term)
    if cached is not None:
        return cached
    esc = re.escape(term)
    if " " in term or "-" in term:
        pat = r"\b" + esc + r"\b"
    else:
        pat = r"\b" + esc + r"\w*"
    compiled = re.compile(pat)
    _STEM_PATTERN_CACHE[term] = compiled
    return compiled


def _lex_any(hay, needles):
    try:
        return any(_pattern(n).search(hay) for n in needles)
    except Exception:
        return False


def _negated(hay, term, stem=False):
    """True if the nearest hit of `term` is preceded (within a short window) by a
    negator, so "rather than endorsing harm" is not read as an endorsement."""
    pat = _stem_pattern(term) if stem else _pattern(term)
    m = pat.search(hay)
    if not m:
        return False
    window = hay[max(0, m.start() - 32):m.start()]
    return any(neg in window for neg in _NEGATORS)


def _lex_any_unnegated(hay, needles, stem=True):
    """Stance-keyword check: stem-mode prefix match + negation guard. `stem`
    defaults to True because this helper is only ever called on the discursive
    stance keyword lists, never on the concrete code_hazard lexicons."""
    try:
        pat_fn = _stem_pattern if stem else _pattern
        return any(pat_fn(n).search(hay) and not _negated(hay, n, stem=stem) for n in needles)
    except Exception:
        return False


def _code_hazard(text):
    """Lexical / co-occurrence hazard detectors -> magnitude in [0, 1]."""
    try:
        tl = _safe_lower(text)
        if not tl.strip():
            return 0.0
        hazards = []

        if _lex_any(tl, _SLURS):
            hazards.append(1.0)

        if _lex_any(tl, _GENOCIDE_MARKERS):
            hazards.append(0.95)

        if _lex_any(tl, _SEXUAL_VIOLENCE_WORDS):
            hazards.append(0.9)

        if _lex_any(tl, _UNDRESS_MARKERS) and _lex_any(tl, _MINOR_MARKERS):
            hazards.append(0.9)
        if _lex_any(tl, _UNDRESS_MARKERS) and _lex_any(tl, _DISMISS_MARKERS):
            hazards.append(0.85)

        if _lex_any(tl, _SUICIDE_CONTEXT) and _lex_any(tl, _SOLICIT_MARKERS):
            hazards.append(0.75)

        if _lex_any(tl, _VIOLENCE_VERBS) and _lex_any(tl, _VULNERABLE_TARGETS):
            hazards.append(0.8)

        if _lex_any(tl, _CHILD_ABUSE_MARKERS) and _lex_any(tl, _CHILD_ABUSE_VICTIMS):
            hazards.append(0.9)

        if _lex_any(tl, _INCAPACITATION_MARKERS) and _lex_any(tl, _SEXUAL_CONTENT_MARKERS):
            hazards.append(0.85)

        if _lex_any(tl, _STEREOTYPE_PHRASES):
            hazards.append(0.4)

        if not hazards:
            return 0.0
        base = max(hazards)
        extra = 0.0
        if len(hazards) >= 2:
            extra = min(0.15, 0.05 * (len(hazards) - 1))
        return max(0.0, min(1.0, base + extra))
    except Exception:
        return 0.0


def _target_tier_peak(target_l):
    if _lex_any(target_l, _OFFTOPIC_TARGET_TERMS):
        return _OFFTOPIC_PEAK
    if _lex_any(target_l, _SEVERE_TARGET_TERMS):
        return _SEVERE_PEAK
    return _DEFAULT_PEAK


def _llm_hazard(extracted):
    """LLM-grounded hazard + remediation bonus, from harm_target / harm_stance.

    Tiered by target severity (named protected-identity/atrocity/trauma target vs.
    generic banter stereotype vs. off-topic/out-of-scope content), then modulated
    by stance: an unnegated self-aware/critique/subvert stance mitigates strongly
    regardless of whether trivializing words also appear in the same sentence
    (satire framing recontextualizes surface "mocks"); an unnegated trivializing
    stance takes the full tier peak; an empty or non-committal stance takes a
    fraction of the tier peak rather than a flat constant.
    """
    try:
        if not isinstance(extracted, dict):
            return 0.0, 0.0
        target = (extracted.get("harm_target") or "").strip()
        stance = (extracted.get("harm_stance") or "").strip()
        target_present = bool(target) and target.upper() not in ("NONE", "N/A")
        stance_l = stance.lower()

        if not target_present:
            # Fallback: even without a named target, a stance sentence that both
            # (a) names an in-scope protected/trauma category itself and (b)
            # unambiguously trivializes it still indicates real signal -- don't
            # zero it out just because the target field came back empty. This is
            # deliberately narrow: round-2 tried firing on ANY "mocks X" stance
            # with no target and it flagged harmless institutional/political
            # satire ("mocks Republicans", "mocks the Arkansas Fire Department",
            # "mocks the target's intelligence") that the criterion does not cover
            # -- gating on an explicit severe-tier keyword in the stance text
            # itself removes those false positives while keeping the narrower,
            # genuinely in-scope case.
            if stance_l and _lex_any(stance_l, _SEVERE_TARGET_TERMS) and \
                    _lex_any_unnegated(stance_l, _TRIVIALIZE_STANCE) and \
                    not _lex_any_unnegated(stance_l, _STRONG_MITIGATE_STANCE):
                return _SEVERE_PEAK, 0.0
            return 0.0, 0.0

        target_l = target.lower()
        tier_peak = _target_tier_peak(target_l)
        is_trauma_target = _lex_any(target_l, _TRAUMA_TARGET_TERMS)

        if not stance_l:
            return tier_peak * _AMBIGUOUS_FACTOR, 0.0

        # Genuine evaluative mitigation (the extractor reports the text itself
        # disapproves/apologizes/reflects) is trusted regardless of target.
        if _lex_any_unnegated(stance_l, _GENUINE_MITIGATE_STANCE):
            return tier_peak * _STRONG_MITIGATE_FACTOR, 0.05

        # "Framing" mitigation (self-aware/subverts/satire) describes TONE, not
        # evaluation. It correctly forgives political/institutional satire, but
        # must not forgive dark humor about real trauma/death just because the
        # extractor labelled the tone "satirical" -- for a trauma-tier target it
        # therefore does not discount below the full tier peak.
        if _lex_any_unnegated(stance_l, _FRAMING_MITIGATE_STANCE):
            if is_trauma_target:
                return tier_peak * _TRAUMA_FRAMING_FACTOR, 0.0
            return tier_peak * _STRONG_MITIGATE_FACTOR, 0.05

        if _lex_any_unnegated(stance_l, _TRIVIALIZE_STANCE):
            return tier_peak, 0.0

        if _BARE_NEUTRAL_RE.match(stance_l):
            # A terse, unelaborated "Neutral." carries little evidence either way
            # -- weight it down further than a stance that explains its neutrality.
            return tier_peak * _BARE_NEUTRAL_FACTOR, 0.0

        # stance present but neither clearly mitigating nor clearly trivializing
        # (e.g. an elaborated-but-noncommittal self-report) -- keep most of the
        # tier weight rather than discounting heavily, since self-reported
        # "neutral" framing does not reliably mean the underlying content is
        # harmless. For a trauma-tier target this applies at full strength (same
        # rationale as the framing-mitigation gate above): an elaborated "neutral,
        # situational irony" self-report about a joke built on graves/war/disaster
        # is not good evidence the trauma wasn't minimized.
        if is_trauma_target:
            return tier_peak * _TRAUMA_FRAMING_FACTOR, 0.0
        return tier_peak * _WEAK_FACTOR, 0.0
    except Exception:
        return 0.0, 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5

        try:
            t = ops.normalize(text)
        except Exception:
            t = text
        if not isinstance(t, str) or not t.strip():
            t = text

        code_pen = _code_hazard(t)
        llm_pen, bonus = _llm_hazard(extracted)

        penalty = max(code_pen, llm_pen)

        tl = _safe_lower(t)
        if _lex_any(tl, _REMEDIATION_MARKERS):
            bonus += 0.05

        raw = 1.0 - penalty + bonus
        if raw < 0.0:
            raw = 0.0
        if raw > 1.0:
            raw = 1.0
        return raw
    except Exception:
        return 0.5
