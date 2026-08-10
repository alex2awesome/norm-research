"""a18_h0: HYBRID metric channel for 'Suspense and Sustained Pressure' (short fiction).

Design (class-level signals, not example keywords):
  1. Code-side dread/menace lexicon density (fear, violence, entrapment, pursuit)
     -> core tension signal, tanh-squashed.
  2. Escalation: dread density in the FINAL third vs the FIRST third. Suspense
     that builds toward the ending is rewarded; static gore-dumps are not.
  3. Time pressure (clocks, countdowns, deadlines) and personal stakes
     (my/his/her daughter/sister/life...) -> small additive bonuses.
  4. Comedy register (laugh/grin/joke/tease/wink density) and author-note
     chrome (Edit:, [WP], 'thanks for the gold') -> penalties. The judged-0
     cluster is dominated by parody/banter pieces.
  5. Two LLM fields carry the tacit part regex cannot see:
       tone  -> comedic/parodic pieces with no laugh-words (deadpan parody)
       peril -> whether any character is concretely endangered, and how gravely.
     Field answers only ADJUST the code score; the predicate stays in code.
"""

import math
import re

LLM_FIELDS = {
    "tone": ("Answer one word, COMEDIC or SERIOUS: is this story's dominant "
             "tone comedic, parodic, or absurdist?"),
    "peril": ("In at most 12 words, name the gravest concrete danger any "
              "character faces during the story; answer NONE if nobody is "
              "endangered."),
}

# --- lexicons (compiled once; class-level, deliberately generic) -------------

_DREAD = re.compile(
    # NOTE (hygiene patch): "kill", "stab", "demon" were bare `\w*`-stemmed and
    # picked up unrelated common words ("Killian", "stable"/"stabilise",
    # "demonstrate"). Replaced with explicit inflection whitelists that keep
    # the intended violence/menace senses without the accidental collisions.
    r"\b(fear\w*|afraid|terror|terrified|terrifying|dread\w*|panic\w*|"
    r"horror|horrified|horrifying|scream\w*|shriek\w*|blood|bloody|bleed\w*|"
    r"kill(?:s|ed|ing|er|ers)?|murder\w*|dead|death\w*|die|died|dies|dying|corpse\w*|"
    r"knife|gun|blade|weapon\w*|monster\w*|demon(?:s|ic|ically)?|"
    r"threat\w*|danger\w*|warn(?:ed|ing|s)?|trap(?:ped|s)?|"
    r"escape\w*|flee|fled|hide|hid|hiding|"
    r"silence|silent|whisper\w*|trembl\w*|shiver\w*|shudder\w*|pale|"
    r"pound(?:ed|ing)|pulse|racing|froze|frozen|paraly[sz]ed|"
    r"desperate\w*|sinister|ominous|menac\w*|hostile|"
    r"hunt(?:ed|ing)?|chas(?:e|ed|ing)|victim\w*|snarl\w*|growl\w*|grim|"
    r"stab(?:s|bed|bing)?|strangl\w*|wound\w*|scared|scary|fright\w*|tortur\w*|"
    r"gore|enemy|enemies|doom\w*|sacrifice\w*|survive|survival)\b",
    re.I)

_PRESSURE = re.compile(
    r"(\b(?:clock|countdown|deadline|ticking|ticked)\b"
    r"|\btoo late\b|\blast chance\b|\brunning out\b|\bout of time\b"
    r"|\b(?:one|two|three|four|five|ten|\d+)\s+(?:minutes?|seconds?|hours?)\s+"
    r"(?:left|to|until|before|remaining)\b)",
    re.I)

_STAKES = re.compile(
    r"\b(?:my|his|her|our|your)\s+"
    r"(?:daughter|son|child|children|wife|husband|sister|brother|family|"
    r"people|village|life|lives|soul)\b",
    re.I)

_PROTECT = re.compile(r"\b(protect\w*|save|saving|rescue\w*)\b", re.I)

_COMEDY = re.compile(
    r"\b(laugh\w*|giggl\w*|chuckl\w*|grin(?:ned|ning|s)?|smirk\w*|"
    r"jok(?:e|es|ed|ing)|teas(?:e|ed|ing)|wink(?:ed|ing)?|hilarious|"
    r"funny|comed\w*|prank\w*|silly|haha\w*|lol)\b",
    re.I)

_META = re.compile(
    r"(?mi)(^\s*(?:final\s+)*edit\s*:|\[\s*wp\s*\]|"
    r"thank(?:s| you)\s+for\s+the\s+(?:gold|silver|platinum))")

_WORD = re.compile(r"[A-Za-z']+")


def _density(rx, text, n_words):
    return len(rx.findall(text)) * 100.0 / max(1, n_words)


def _clip(x):
    return max(0.0, min(1.0, x))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = text or ""
        try:
            t = ops.normalize(t) or t
        except Exception:
            pass

        words = _WORD.findall(t)
        n = len(words)
        if n < 30:
            return 0.1  # fragments can't sustain pressure

        # --- core dread signal -------------------------------------------
        dread_d = _density(_DREAD, t, n)          # hits per 100 words
        base = math.tanh(dread_d / 4.0)           # ~0.4-0.9 for tense fiction

        # --- escalation: final third vs first third ----------------------
        third = max(1, len(t) // 3)
        head_d = _density(_DREAD, t[:third], max(1, n // 3))
        tail_d = _density(_DREAD, t[-third:], max(1, n // 3))
        esc = 0.10 * _clip((tail_d - head_d) / 3.0) if tail_d > head_d else 0.0

        # --- bonuses ------------------------------------------------------
        pressure = 0.08 * min(1.0, len(_PRESSURE.findall(t)) / 2.0)
        stakes = 0.08 * min(1.0, len(_STAKES.findall(t)) / 2.0)
        protect = 0.04 * min(1.0, len(_PROTECT.findall(t)) / 2.0)

        # --- penalties ----------------------------------------------------
        comedy_d = _density(_COMEDY, t, n)
        comedy_pen = 0.35 * math.tanh(comedy_d / 1.2)
        meta_pen = 0.08 * min(2, len(_META.findall(t)))

        long_ramble = 0.0
        try:
            _, mws, _ = ops.sent_stats(t)
            if mws and mws > 26.0:                 # rambling run-on register
                long_ramble = 0.05
        except Exception:
            pass

        s = (0.12 + 0.58 * base + esc + pressure + stakes + protect
             - comedy_pen - meta_pen - long_ramble)

        # --- LLM-field adjustments (evidence, not predicate) --------------
        ex = extracted if isinstance(extracted, dict) else {}

        tone = str(ex.get("tone", "") or "").strip().lower()
        if tone:
            if re.search(r"comed|parod|absurd|humor|humour|satir|farc", tone):
                s -= 0.28
            elif re.search(r"serious|dark|tense|grim|dramatic", tone):
                s += 0.04

        peril = str(ex.get("peril", "") or "").strip().lower()
        if peril in ("", "none", "none."):
            s -= 0.14
        elif re.search(r"\bnone\b|no danger|not.{0,12}danger", peril):
            s -= 0.10
        elif re.search(
                r"death|dead|die|dies|dying|kill|murder|execut|drown|devour|"
                r"eaten|torture|destro|war|monster|doom|sacrific|maul|"
                r"life|lives|fatal|lethal", peril):
            s += 0.09
        else:
            s += 0.03

        return _clip(s)
    except Exception:
        return 0.5
