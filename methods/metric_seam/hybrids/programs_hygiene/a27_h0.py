"""Hybrid metric channel for a27: "Anchor the Fantastical in Concrete Realism".

Construct: surreal/absurd invention reads as high when (1) the NARRATION treats
it deadpan / matter-of-factly, (2) the story is grounded in mundane everyday
life (homes, jobs, meals, errands), and (3) the prose shows restraint
(few narratorial exclamations, little epic-register vocabulary) plus sharp
ordinary specificity (money amounts, clock times, weekdays, everyday nouns).

(1) and (2) are tacit constructs -> two LLM extraction fields with strict
one-word enums; the predicate (mapping + blending + all surface features)
stays in code.
"""

import re
import math

LLM_FIELDS = {
    "tone": ("Does the narration present the story's impossible or absurd events with deadpan "
             "matter-of-fact calm, or with shock and melodrama? Answer one word: deadpan, mixed, "
             "or melodramatic."),
    "setting": ("Is the story grounded in mundane everyday life (homes, jobs, meals, errands, "
                "small routines)? Answer one word: yes, partly, or no."),
}

# ---------------- lexicons (class-level, not example-specific) ----------------

_EVERYDAY = [
    # household
    "kitchen", "couch", "sofa", "table", "chair", "counter", "sink", "fridge",
    "freezer", "refrigerator", "toaster", "oven", "stove", "microwave",
    "garage", "yard", "lawn", "porch", "blanket", "pillow", "bedroom",
    "bathroom", "shower", "towel", "laundry", "dishes", "chores", "mug",
    "spoon", "fork", "plate", "jar", "napkin", "doorbell", "mailbox", "keys",
    # food & drink
    "coffee", "tea", "toast", "bread", "pizza", "sandwich", "casserole",
    "stew", "cereal", "bacon", "cookie", "cookies", "dinner", "lunch",
    "breakfast", "snack", "grocery", "groceries", "leftovers", "ice cream",
    "beer", "wine", "soda",
    # transit / city
    "bus", "cab", "taxi", "sedan", "traffic", "intersection", "sidewalk",
    "curb", "subway", "parking", "driveway", "commute", "fender",
    # work / school / admin
    "office", "boss", "job", "paycheck", "meeting", "desk", "email",
    "coworker", "school", "teacher", "homework", "classroom", "backpack",
    "store", "shop", "mall", "receipt", "rent", "tab", "tip", "bank",
    "account", "accountant", "accounting", "audit", "tax", "taxes",
    "insurance", "paperwork", "wallet", "cash", "salary", "invoice",
    "customer", "customers", "client", "clients", "firm",
    # mundane tech / media
    "phone", "cellphone", "telephone", "laptop", "computer", "television",
    "remote", "internet", "facebook", "twitter", "youtube", "voicemail",
    "notification", "notifications", "forums", "download", "newspaper",
    # neighbors & routine
    "neighbor", "neighbors", "roommate", "landlord", "weekend", "errand",
    "errands", "mower", "lawnmower", "pajamas", "umbrella",
]

_GRANDIOSE = [
    r"prophec\w*", "destiny", "destined", "foretold", "empire", "emperor",
    "kingdom", "realm", "throne", "eternal", "eternity", "ancient", "cosmic",
    "almighty", "divine", "heresy", "wrath", "vengeance", "tyranny",
    r"legion\w*", "banner", r"oath\w*", r"vanquish\w*", "darkness", "abyss",
    "supreme", "sword", "blade", r"warship\w*", "carnage", "heinous",
    "sinister", "unimaginable", "unspeakable",
]

_DRAMATIC_VERBS = [
    "screamed", "screaming", "yelled", "yelling", "roared", "shrieked",
    "screeched", "exclaimed", "boomed", "bellowed", "wailed", "thundered",
    "sobbed",
]
_SAID_VERBS = [
    "said", "says", "asked", "replied", "muttered", "murmured", "whispered",
    "answered", "responded", "added", "noted",
]

_EVERYDAY_RE = re.compile(r"\b(?:" + "|".join(_EVERYDAY) + r")\b")
_GRANDIOSE_RE = re.compile(r"\b(?:" + "|".join(_GRANDIOSE) + r")\b")
_DRAMATIC_RE = re.compile(r"\b(?:" + "|".join(_DRAMATIC_VERBS) + r")\b")
_SAID_RE = re.compile(r"\b(?:" + "|".join(_SAID_VERBS) + r")\b")

_MONEY_RE = re.compile(r"\$\s?\d|\b\d[\d,]*\s?(?:dollars?|bucks?|grand|cents)\b"
                       r"|\b(?:twenty|fifty|hundred|thousand|million)\s?dollar")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm|a\.m\.|p\.m\.)\b")
_WEEKDAY_RE = re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b")
_UNIT_RE = re.compile(r"\b\d[\d,]*\s?(?:years?|minutes?|hours?|weeks?|months?|days?|miles?|feet|inches)\b")
_ELLIPSIS_RE = re.compile(r"\.\.\.+|…")
_QUOTE_SPAN_RE = re.compile(r"[\"“][^\"“”\n]{0,800}[\"”]")


def _clip01(x):
    return max(0.0, min(1.0, x))


def _enum_tone(ans):
    """deadpan -> 1.0, mixed -> 0.5, melodramatic/shock -> 0.0, unknown -> 0.5"""
    a = (ans or "").strip().lower()
    if not a:
        return 0.5
    # "both"/"matter" are anchored with \b: bare substring falsely fired on
    # unrelated words "bothered" and "smattering" (scanner-caught bug).
    if "mixed" in a or re.search(r"\bboth\b", a):
        return 0.5
    if "melodram" in a or "shock" in a:
        return 0.0
    if "deadpan" in a or re.search(r"\bmatter\b", a) or "calm" in a:
        return 1.0
    if "dramatic" in a:  # after 'deadpan' check; catches bare "dramatic"
        return 0.0
    return 0.5


def _enum_setting(ans):
    """yes -> 1.0, partly -> 0.6, no -> 0.0, unknown -> 0.5"""
    a = (ans or "").strip().lower()
    if not a:
        return 0.5
    if "partly" in a or "partial" in a or "somewhat" in a or "mostly" in a:
        return 0.6
    if re.search(r"\byes\b", a):
        return 1.0
    if re.search(r"\bno\b", a):
        return 0.0
    return 0.5


def _code_bundle(text, ops):
    """Surface craft signals in [0,1]; length-robust (per-100-word densities)."""
    try:
        t = ops.normalize(text)
    except Exception:
        t = text
    tl = t.lower()
    n_words = max(1, len(re.findall(r"\w+", tl)))
    per100 = 100.0 / n_words
    sents = [s for s in re.split(r"[.!?\n]+", t) if s.strip()]
    n_sents = max(1, len(sents))

    # 1. everyday concrete-noun density (positive)
    ev = len(_EVERYDAY_RE.findall(tl)) * per100
    c_everyday = min(1.0, ev / 2.0)

    # 2. ordinary specificity: money, clock times, weekdays, numeric units (positive)
    sp = (len(_MONEY_RE.findall(tl)) + len(_TIME_RE.findall(tl))
          + len(_WEEKDAY_RE.findall(tl)) + len(_UNIT_RE.findall(tl))) * per100
    c_spec = min(1.0, sp / 0.45)

    # 3. narratorial exclamation restraint (exclamations OUTSIDE quotes; negative)
    narration = _QUOTE_SPAN_RE.sub(" ", t)
    n_excl = narration.count("!")
    c_excl = 1.0 - min(1.0, (n_excl / n_sents) * 5.0)

    # 4. epic-register restraint (negative)
    gr = len(_GRANDIOSE_RE.findall(tl)) * per100
    c_grand = 1.0 - min(1.0, gr / 1.1)

    # 5. deadpan dialogue attribution: said-family vs shouty verbs
    n_said = len(_SAID_RE.findall(tl))
    n_dram = len(_DRAMATIC_RE.findall(tl))
    c_said = 0.5 if (n_said + n_dram) == 0 else n_said / float(n_said + n_dram)

    # 6. ellipsis restraint (negative)
    c_ell = 1.0 - min(1.0, (len(_ELLIPSIS_RE.findall(t)) / n_sents) * 2.5)

    # 7. sentence-length control (very long run-ons -> penalty)
    try:
        _, mean_wps, _ = ops.sent_stats(t)
    except Exception:
        mean_wps = n_words / float(n_sents)
    c_sent = 1.0 if mean_wps <= 28.0 else max(0.0, 1.0 - (mean_wps - 28.0) / 25.0)

    bundle = (0.28 * c_everyday + 0.20 * c_spec + 0.20 * c_excl
              + 0.14 * c_grand + 0.08 * c_said + 0.05 * c_ell + 0.05 * c_sent)
    # renormalize to [0,1] (weights sum to 1.0)
    return _clip01(bundle)


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        ex = extracted if isinstance(extracted, dict) else {}
        r_tone = _enum_tone(ex.get("tone", ""))
        r_set = _enum_setting(ex.get("setting", ""))
        cb = _code_bundle(text, ops)
        out = 0.42 * r_tone + 0.23 * r_set + 0.35 * cb
        return float(_clip01(out))
    except Exception:
        return 0.5
