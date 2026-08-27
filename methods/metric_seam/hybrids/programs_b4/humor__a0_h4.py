# Hybrid module (4-field) for humor aspect a0: "Ethical target selection:
# punch up, avoid harm and cruelty"
#
# Design: code carries the deterministic, unambiguous hazards (explicit slurs,
# explicit child+harm combinations, explicit "suicide" mentions, explicit
# punching-up institution/self markers). The two ORIGINAL LLM fields carry
# thick semantic judgment code cannot reach on its own: (1) who/what actually
# bears the brunt of the joke, and (2) whether the humor depends on a
# marginalized group, a real tragedy, or someone's suffering. The LLM only
# returns short factual extractions; all scoring logic (the predicate) lives
# here in code.
#
# GAP the original 2 fields plausibly miss (construct grounds only -- no
# judge/eval signal consulted): the criterion explicitly names two more
# axes that neither `target` nor `harm_class` probes, and that code cannot
# reliably probe on its own:
#   (a) "calibrate tone/aggression to teasing not bullying" -- the existing
#       code only proxies aggression with an ALL-CAPS shouting regex, which
#       is orthogonal to actual cruelty (a venomous line can be delivered in
#       lowercase; a shouted line can be pure slapstick, e.g. "OH SHIT!!").
#       A direct LLM read of delivery tone (playful vs cruel/degrading) is a
#       thick-input construct.
#   (b) "avoid stereotyping and appearance-based or humiliating shots" -- the
#       existing fields identify WHO is targeted and WHETHER a vulnerable
#       class is invoked, but not the MECHANISM: whether the punchline's
#       humor specifically trades on a negative stereotype about a group's
#       traits (lazy, dumb, promiscuous, etc.) rather than an individual's
#       specific action. That mechanism judgment is thick-input too.
#
# New fields `tone` and `stereotype` probe those two gaps directly from the
# text. Both are short categorical/extractive reads. Scoring stays additive
# and gracefully degrades: if the fields are absent from `extracted` (not
# yet extracted), the new blocks are skipped entirely and score() reduces
# exactly to the original 2-field behavior.

import re

LLM_FIELDS = {
    "target": (
        "In <=8 words, name who or what bears the brunt of this joke's "
        "mockery/violence (e.g. 'a politician', 'a crying child', "
        "'a racial group', 'the narrator himself'), or say 'none'."
    ),
    "harm_class": (
        "In <=10 words, say whether the joke's humor depends on a slur, a "
        "marginalized/protected group, a child being harmed, sexual "
        "assault, suicide/self-harm, or a real tragedy -- name which one -- "
        "or say 'none'."
    ),
    "tone": (
        "In <=6 words, categorize the joke's tone toward its target: "
        "playful/affectionate, neutral/factual, or cruel/degrading."
    ),
    "stereotype": (
        "In <=10 words, does the punchline rely on a negative stereotype "
        "about a group's traits (name the trait), or 'no'?"
    ),
}

_SLUR_RE = re.compile(
    r"\b(nigger[s]?|nigga[s]?|kike[s]?|spic[s]?|chink[s]?|gook[s]?|"
    r"fagg?ot[s]?|tranny|trannies|retard(?:ed|s)?|cripple[ds]?|midget[s]?|"
    r"beaner[s]?|wetback[s]?|towelhead[s]?)\b",
    re.IGNORECASE,
)

_CHILD_TERMS_RE = re.compile(
    r"\b(boy|girl|kid|kids|child|children|son|daughter|toddler|infant|baby|babies|newborn)\b",
    re.IGNORECASE,
)

_CHILD_HARM_CUES_RE = re.compile(
    r"\b(unbuckl\w*|unbutton\w*|unzip\w*|molest\w*|rap(?:e|ed|ing)|"
    r"cigarette|burn\w*|beat\w*|slap\w*|choke\w*|abus\w*|kill\w*|murder\w*|"
    r"drown\w*|starv\w*|neglect\w*|molested)\b",
    re.IGNORECASE,
)

_RACE_GROUP_RE = re.compile(
    r"\b(black people|white people|mexicans?|asians?|jews?|muslims?|latinos?|immigrants?)\b",
    re.IGNORECASE,
)
_ORIGIN_MYTH_RE = re.compile(
    r"\b(how (?:we|they) got|that'?s how|explode[ds]?|grill)\b", re.IGNORECASE
)

_SUICIDE_WORD_RE = re.compile(r"\bsuicid\w*\b", re.IGNORECASE)

_SHOUT_RE = re.compile(r"[A-Z]{5,}")

_POWER_KEYWORDS_RE = re.compile(
    r"\b(politician\w*|president|senator|congress\w*|ceo|corporation\w*|"
    r"government|boss|billionaire\w*|celebrity|celebrities|landlord\w*|"
    r"narrator|myself|himself|herself|oneself|his own|her own|the author|"
    r"the speaker|self[- ]?deprecat\w*)\b",
    re.IGNORECASE,
)

_VULN_KEYWORDS_RE = re.compile(
    r"\b(child|kid|boy|girl|baby|infant|disab\w*|autis\w*|intellectual\w*|"
    r"race|racial|racist|ethnic\w*|black|white people|asian|jewish|jew|"
    r"muslim|mexican|latino|immigrant\w*|refugee\w*|gay|lesbian|trans\w*|"
    r"homeless|elderly|women|woman|victim\w*|marginali[sz]\w*|rape|molest\w*|"
    r"suicid\w*|tragedy|slur|bully\w*|harass\w*|exploit\w*)\b",
    re.IGNORECASE,
)

_MUTUAL_PAIR_RE = re.compile(r"\bhusband\b.*\bwife\b|\bwife\b.*\bhusband\b", re.IGNORECASE | re.DOTALL)

# --- new-field parsing vocab --------------------------------------------

_CRUEL_TONE_RE = re.compile(
    r"\b(cruel|degrad\w*|mean[- ]?spirited|bully\w*|harsh|contempt\w*|"
    r"vicious|mocking|derisive|humiliat\w*|belittl\w*|scorn\w*)\b",
    re.IGNORECASE,
)
_GENTLE_TONE_RE = re.compile(
    r"\b(playful|affectionate|light[- ]?hearted|gentle|warm|teasing|"
    r"friendly|silly|absurd|whimsical|goofy|good[- ]?natured)\b",
    re.IGNORECASE,
)

_STEREOTYPE_NONE_WORDS = {"", "no", "none", "n/a", "na", "not applicable", "unclear"}


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw
        tl = t.lower()

        hazard = 0.0
        boost = 0.0

        # --- deterministic code hazards ---
        slur_hits = len(_SLUR_RE.findall(tl))
        if slur_hits:
            hazard += 0.55 + 0.15 * min(slur_hits - 1, 2)

        if _CHILD_TERMS_RE.search(tl) and _CHILD_HARM_CUES_RE.search(tl):
            hazard += 0.55

        if _RACE_GROUP_RE.search(tl) and _ORIGIN_MYTH_RE.search(tl):
            hazard += 0.5

        if _SUICIDE_WORD_RE.search(tl):
            hazard += 0.2

        # tone/aggression calibration: shouted delivery amplifies an
        # already-detected hazard (teasing vs. bullying)
        if hazard > 0.0 and _SHOUT_RE.search(raw):
            hazard *= 1.2

        # --- deterministic code boosts (punching up / fair targets) ---
        if _POWER_KEYWORDS_RE.search(tl):
            boost += 0.15
        if _MUTUAL_PAIR_RE.search(tl):
            boost += 0.1

        # --- LLM-field grounding (thick-input constructs code can't reach) ---
        extracted = extracted or {}
        harm_class = str(extracted.get("harm_class", "") or "").strip().lower()
        target = str(extracted.get("target", "") or "").strip().lower()

        if harm_class and harm_class != "none":
            if _VULN_KEYWORDS_RE.search(harm_class):
                hazard += 0.4
            else:
                # extractor flagged *something* but it doesn't match our
                # known vulnerable-construct vocabulary; treat as a mild flag
                hazard += 0.15

        if target and target != "none":
            if _POWER_KEYWORDS_RE.search(target):
                boost += 0.25
            elif _VULN_KEYWORDS_RE.search(target):
                hazard += 0.25

        # --- NEW: tone field -- direct read of teasing vs. bullying delivery.
        # Only fires when the extractor has actually produced this field
        # (graceful degradation to the original 2-field behavior otherwise).
        if "tone" in extracted:
            tone_val = str(extracted.get("tone", "") or "").strip().lower()
            is_cruel = bool(_CRUEL_TONE_RE.search(tone_val))
            is_gentle = bool(_GENTLE_TONE_RE.search(tone_val))
            if is_cruel and not is_gentle:
                hazard += 0.25
            elif is_gentle and not is_cruel:
                boost += 0.15

        # --- NEW: stereotype field -- names the MECHANISM (group-trait
        # stereotype) the original target/harm_class fields don't isolate.
        # A stereotype hit is a hazard signal independent of, and additive
        # to, harm_class (e.g. an appearance/trait stereotype with no slur
        # and no named "marginalized group" phrasing that harm_class would
        # otherwise miss).
        if "stereotype" in extracted:
            stereo_val = str(extracted.get("stereotype", "") or "").strip().lower()
            if stereo_val not in _STEREOTYPE_NONE_WORDS:
                if _VULN_KEYWORDS_RE.search(stereo_val):
                    hazard += 0.3
                else:
                    hazard += 0.15

        hazard = min(hazard, 1.2)
        boost = min(boost, 0.65)

        s = 0.5 + boost - hazard
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
