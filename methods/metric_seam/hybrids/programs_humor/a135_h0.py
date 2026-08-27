# Hybrid scoring module for humor aspect a135:
# "Platform/broadcast standards and constraints" -- sponsor/network/platform
# taste, profanity, libel, and censorship rules that govern what can be shown
# or said and where it can run.
#
# Design: broadcast/platform standards are violated by (a) explicit profanity
# and slurs, scaled by severity/frequency, and (b) content that depicts or
# endorses serious real-world harm (violence, sexual exploitation, hate,
# self-harm) toward people, especially vulnerable targets or protected
# groups -- even when no swear word is present (e.g. an implied assault, a
# euphemistic genocide reference, a racist premise with no slur). Code
# lexicons catch the former and several code-detectable hazard patterns of
# the latter; two LLM fields catch the residual "reads-as-fine-to-a-keyword-
# scanner but is clearly not broadcast-safe" cases that require semantic
# judgment (implied non-consensual harm to a vulnerable party; whether a
# protected group is being demeaned) -- constructs plain regex cannot reach.
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

_SEVERE_SLUR_RE = re.compile(
    r"\b(nigger\w*|nigga\w*|chink\w*|spic\w*|kike\w*|faggot\w*|retard(?:ed)?|"
    r"tranny|trannies|wetback\w*|coon\w*)\b",
    re.I,
)

_STRONG_PROFANITY_RE = re.compile(
    r"\b(fuck\w*|shit\w*|cunt\w*|cock\w*|dick\w*|pussy|bitch\w*|"
    r"motherfucker\w*|asshole\w*|dumbass|jackass)\b",
    re.I,
)

_MILD_PROFANITY_RE = re.compile(
    r"\b(damn\w*|hell|crap\w*|bastard\w*|bloody|piss\w*)\b",
    re.I,
)

_HAZARD_HATE_RE = re.compile(
    r"\b(nazi\w*|hitler|holocaust|genocide|gas\s+chamber\w*|final\s+solution|"
    r"mein\s+f[uü]hrer|f[uü]hrer|kkk|lynch\w*|concentration\s+camp)\b",
    re.I,
)

_HAZARD_SELFHARM_RE = re.compile(
    r"\b(suicide\w*|suicidal|kill\s+myself|self[- ]harm|overdose|"
    r"hang(?:ed|ing)\s+(?:himself|herself|themselves))\b",
    re.I,
)

_HAZARD_SEXUAL_RE = re.compile(
    r"\b(rape\w*|molest\w*|pedophil\w*|paedophil\w*|non-?consensual\w*|"
    r"anus\w*|semen|sperm)\b",
    re.I,
)

_META_RACISM_RE = re.compile(r"\bn[- ]word\b", re.I)

_VIOLENCE_VERB_RE = re.compile(
    r"\b(beat\w*|kick\w*|punch\w*|stomp\w*|stab\w*|strangl\w*|chok\w*|"
    r"slap\w*|assault\w*)\b",
    re.I,
)

_VULNERABLE_TARGET_RE = re.compile(
    r"\b(nun\w*|priest\w*|child\w*|kids?|boy|girl|bab(?:y|ies)|infant\w*|"
    r"toddler\w*|elderly|old\s+man|old\s+woman|disabled|orphan\w*)\b",
    re.I,
)

_EMPTY_ANSWER_PREFIXES = (
    "none", "n/a", "na", "no", "unsure", "unknown", "not applicable",
    "not clear", "unclear",
)


def _is_empty_answer(s):
    s2 = (s or "").strip().strip(".").lower()
    if not s2:
        return True
    return any(s2.startswith(p) for p in _EMPTY_ANSWER_PREFIXES)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        t = ops.normalize(raw)
        if not t or not t.strip():
            return 0.5
        low = t.lower()

        slur_hit = bool(_SEVERE_SLUR_RE.search(low))
        strong_hits = len(_STRONG_PROFANITY_RE.findall(low))
        mild_hits = len(_MILD_PROFANITY_RE.findall(low))
        hazard_hate = bool(_HAZARD_HATE_RE.search(low))
        hazard_selfharm = bool(_HAZARD_SELFHARM_RE.search(low))
        hazard_sexual = bool(_HAZARD_SEXUAL_RE.search(low))
        meta_racism = bool(_META_RACISM_RE.search(low))
        violence_hit = bool(_VIOLENCE_VERB_RE.search(low)) and bool(
            _VULNERABLE_TARGET_RE.search(low)
        )

        strong_sat = 1.0 - math.exp(-1.3 * strong_hits)
        mild_sat = 1.0 - math.exp(-0.8 * mild_hits)

        components = []
        if slur_hit:
            components.append(1.0)
        if hazard_hate:
            components.append(0.75)
        if hazard_selfharm:
            components.append(0.75)
        if hazard_sexual:
            components.append(0.65)
        if violence_hit:
            components.append(0.7)
        if meta_racism:
            components.append(0.6)
        components.append(0.55 * strong_sat)
        components.append(0.25 * mild_sat)

        code_severity = max(components) if components else 0.0
        secondary_sum = sum(c for c in components if c < code_severity)
        code_severity = min(1.0, code_severity + 0.15 * min(1.0, secondary_sum))

        harm_txt = ""
        target_txt = ""
        if isinstance(extracted, dict):
            harm_txt = (extracted.get("harm_severity") or "").strip().lower()
            target_txt = (extracted.get("target_group") or "").strip()

        llm_severity = 0.0
        if any(k in harm_txt for k in ("severe", "extreme")):
            llm_severity = 1.0
        elif any(k in harm_txt for k in ("mild", "moderate", "some", "low")):
            llm_severity = 0.5

        target_penalty = 0.0
        if not _is_empty_answer(target_txt):
            target_penalty = 0.4

        severity = max(code_severity, llm_severity)
        severity = min(1.0, severity + target_penalty)

        s = 1.0 - severity
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        return s
    except Exception:
        return 0.5
