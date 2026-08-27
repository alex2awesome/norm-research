import re
import math

_BUDGET = 25000

_INVESTIGATION_KEYWORDS = [
    "investigat", "internal investigation", "prompt investigation",
    "thorough investigation", "independent investigation",
    "outside investigator", "third-party investigator", "hired an investigator",
    "hired an outside", "conducted an investigation", "hr investigation",
    "human resources investigation", "human resources conducted",
    "eeoc charge", "eeoc investigated", "equal employment opportunity",
]

_ACTION_KEYWORDS = [
    "prompt corrective action", "corrective action", "remedial action",
    "remedial measure", "took prompt", "promptly investigat",
    "promptly remed", "promptly correct", "promptly address",
    "promptly respond", "remediate", "remedied",
    "disciplinary action", "disciplined the", "disciplined him",
    "disciplined her", "issued a warning", "issued a reprimand",
    "verbal warning", "written warning", "terminated the harasser",
    "suspended the", "reprimanded the", "separated the",
    "transferred the harasser", "training", "sensitivity training",
    "anti-harassment training", "mandatory training",
]

_INVESTIGATED_BY_EMPLOYER = [
    "employer investigated", "company investigated",
    "investigation revealed", "investigation found",
    "investigation determined", "investigation concluded",
    "investigation confirmed", "investigation showed",
    "investigation substantiat", "investigation unconfirm",
    "investigator found", "investigator concluded",
    "investigator determined", "investigator interviewed",
    "investigation was prompt", "investigation was thorough",
    "investigation was reasonable", "investigation was adequate",
]

_HR_ACTED = [
    "hr investigated", "human resources investigated",
    "personnel department", "human resources promptly",
    "hr promptly", "management investigated",
    "supervisor investigated", "department investigated",
    "conducted interviews", "interviewed witnesses",
    "interviewed the complainant", "witness statements",
    "took statements", "gathered evidence",
    "reviewed the evidence", "reviewed surveillance",
    "reviewed video", "reviewed the complaint",
]

_OUTCOME = [
    "found to have violated", "violation of policy",
    "violation of the policy", "substantiated",
    "unsubstantiated", "could not substantiate",
    "insufficient evidence to support", "found no merit",
    "determined no harassment", "determined no discrimination",
    "concluded no", "no policy violation",
    "corrective action was taken", "appropriate action was taken",
]

_NEGATIVE = [
    "no investigation", "did not investigate", "failed to investigate",
    "never investigated", "no corrective action", "no remedial action",
    "failed to take corrective", "failed to take remedial",
    "did not take corrective", "did not take remedial",
    "did not take any action", "took no action",
    "no action was taken", "no remedial steps",
    "no remedial measures", "ignored the complaint",
    "ignored the report", "disregarded the complaint",
    "disregarded the report", "failed to act",
    "failed to respond", "no effort to investigate",
    "did nothing", "no disciplinary action",
    "employer failed to take",
]

_NO_NEED = [
    "no investigation was necessary", "no investigation was needed",
    "no investigation was required", "no need to investigate",
    "investigation was unnecessary", "remedial action was unnecessary",
    "no corrective action was necessary",
]

_DISTRACTORS = [
    "eeoc investigated", "eeoc investigation",
    "court investigated", "court's investigation",
    "jury investigated", "agency investigation",
    "department of labor investigated", "dol investigated",
    "ofccp investigated", "commission investigated",
    "police investigation", "law enforcement investigation",
    "criminal investigation", "fbi investigated",
    "agent investigated",
]


def _count_phrase(text, phrase):
    try:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        return len(pattern.findall(text))
    except Exception:
        try:
            low = text.lower()
            return low.count(phrase.lower())
        except Exception:
            return 0


def _count_any(text, phrases):
    return sum(_count_phrase(text, p) for p in phrases)


def _count_regex(text, pattern):
    try:
        c = re.compile(pattern, re.IGNORECASE)
        return len(c.findall(text))
    except Exception:
        return 0


def _has_regex(text, pattern):
    try:
        return bool(re.search(pattern, text, re.IGNORECASE))
    except Exception:
        return False


def _score_investigation_conducted(text):
    score = 0.0
    score += 2.5 * _count_any(text, _INVESTIGATION_KEYWORDS)
    score += 3.0 * _count_any(text, _INVESTIGATED_BY_EMPLOYER)
    score += 2.5 * _count_any(text, _HR_ACTED)
    score += 2.0 * _count_regex(
        text,
        r"\b(?:investigat|inquir|review|examin|look(?:ed)?\s+into)\w*"
        r"(?:\s+\w+){0,3}\s+(?:the\s+)?(?:matter|alleg|complaint|incid|report|claim|conduct|facts|accusation)"
    )
    score += 2.5 * _count_regex(
        text,
        r"\b(?:interview|took|obtained|gathered|collected)\w*\s+(?:\w+\s+){0,2}(?:witness|statement|evidence|fact|information)"
    )
    score += 2.0 * _count_regex(
        text,
        r"\b(?:made|reach)\w*\s+(?:a\s+)?(?:determination|finding|conclusion)"
    )
    score += 1.5 * _count_regex(
        text,
        r"\b(?:employer|defendant|company|management|hr|department|supervisor|city)\s+(?:promptly\s+|immediately\s+)?(?:investigat|inquir|review|examin)"
    )
    return score


def _score_corrective_action(text):
    score = 0.0
    score += 3.0 * _count_any(text, _ACTION_KEYWORDS)
    score += 2.0 * _count_regex(
        text,
        r"\b(?:took|take|implement|implemented|adopt|adopted)\w*\s+(?:\w+\s+){0,3}(?:corrective|remedial)\s+(?:action|measure|step)"
    )
    score += 1.5 * _count_regex(
        text,
        r"\b(?:promptly|immediately|without\s+delay)\s+(?:\w+\s+){0,2}(?:investigat|remediat|correct|address|respond|suspend|disciplin|terminat|transfer|warn|reprimand)"
    )
    score += 1.5 * _count_regex(
        text,
        r"\b(?:disciplin|suspend|reprimand|terminat|transfer|warn|demote|reassig|separat|train|counsel)\w*\s+(?:\w+\s+){0,2}(?:harasser|accused|alleged|perpetrator|offender|respondent|him|her|the\s+(?:employee|individual|supervisor|coworker|manager))"
    )
    score += 1.5 * _count_regex(
        text,
        r"\b(?:anti[\s-]?harassment|sensitivity|diversity|workplace|sexual\s+harassment)\s+training"
    )
    return score


def _score_outcome(text):
    score = 0.0
    score += 2.0 * _count_any(text, _OUTCOME)
    score += 2.0 * _count_regex(
        text,
        r"\b(?:investigation|inquiry|review)\s+(?:revealed|found|determined|concluded|showed|confirm(?:ed|s)?|demonstrat\w*)\s+that"
    )
    score += 1.5 * _count_regex(
        text,
        r"\b(?:substantiat|unsubstantiat|unfounded|confirm\w*|verif\w*)\w*\s+(?:\w+\s+){0,3}(?:alleg|complaint|claim|harass|discrimin|conduct|accusation)"
    )
    return score


def _score_correction_narrative(text):
    score = 0.0
    score += 2.0 * _count_regex(
        text,
        r"\b(?:once|after|when|upon|following)\s+(?:\w+\s+){0,3}(?:report\w*|complain\w*|notifi\w*|inform\w*|learn\w*|became\s+aware),?\s+(?:\w+\s+){0,4}(?:investigat|inquir|review|examin|interview|correct|remediat|disciplin|suspend|terminat|transfer|warn|reprimand)"
    )
    score += 2.0 * _count_regex(
        text,
        r"\b(?:in\s+response\s+to|in\s+light\s+of)\s+(?:the\s+)?(?:\w+\s+){0,3}(?:complaint|report|alleg|incid|conduct|harass|discrimin),?\s+(?:\w+\s+){0,4}(?:investigat|correct|remediat|disciplin|suspend|terminat|transfer|warn|reprimand|address|respond)"
    )
    score += 1.5 * _count_regex(
        text,
        r"\b(?:promptly|immediately|without\s+delay|timely|quickly|expeditiously|forthwith)\s+(?:\w+\s+){0,4}(?:investigat|correct|remediat|disciplin|suspend|terminat|transfer|warn|reprimand|address|respond)"
    )
    return score


def _penalize_negative(text):
    penalty = 0.0
    penalty += 3.0 * _count_any(text, _NEGATIVE)
    penalty += 2.0 * _count_regex(
        text,
        r"\b(?:no|never|did\s+not|failed\s+to|without|lack(?:ed|ing)?\s+(?:of\s+)?|absence\s+of)\s+(?:\w+\s+){0,4}(?:investigat|corrective\s+action|remedial\s+action|remedial\s+measure|disciplin|action|response)"
    )
    penalty += 2.0 * _count_any(text, _NO_NEED)
    return penalty


def _penalize_distractors(text):
    penalty = 0.0
    penalty += 2.0 * _count_any(text, _DISTRACTORS)
    penalty += 2.0 * _count_regex(
        text,
        r"\b(?:eeoc|court|jury|commission|agency|department\s+of\s+labor|dol|ofccp|nrc|fbi|police|sheriff|law\s+enforcement)\s+(?:\w+\s+){0,2}(?:investigat|investigation|inquiry)"
    )
    return penalty


def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = text[:500000].lower()

        raw = 0.0
        raw += _score_investigation_conducted(t)
        raw += _score_corrective_action(t)
        raw += _score_outcome(t)
        raw += _score_correction_narrative(t)

        neg = _penalize_negative(t)
        dis = _penalize_distractors(t)

        net = max(0.0, raw - neg - dis * 0.5)

        if net <= 0:
            return 0.5
        if net >= 20:
            return 10.0

        s = 0.5 + (net / 20.0) * 9.5
        return max(0.5, min(10.0, s))
    except Exception:
        return 0.5