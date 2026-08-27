"""a72 hybrid h0: Theme, idea, and lasting resonance.

Judge intuition (from train residuals): low scores are gag/parody/crude-humor
pieces (punchline structure, ALL-CAPS shouting, profanity-dense, Reddit
edit-churn); high scores are reflective, thematically controlled prose with
resonant endings. Topic keywords (v0 baseline) are a weak proxy. The tacit
mode/theme call goes to two short LLM fields; the predicate and all surface
robustness checks stay in code.
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
    r"boner|bitch\w*|douche\w*|assholes?|ass|tits?|crap|piss\w*)\b",
    re.IGNORECASE)

_SHOUT = re.compile(r"\b[A-Z]{4,}\b")

_META_EDIT = re.compile(r"(?mi)^\s*\**\s*(final\s+)*edit\s*\d*\s*[:\-]")
_META_APOLOGY = re.compile(
    r"(?i)(sorry for (the )?(bad|poor|awful)?\s*(formatting|grammar|spelling)|"
    r"wrote this on my phone|word count\s*:)")

_SIMILE = re.compile(r"(?i)\b(like an?|as if|as though)\b")

_ABSTRACT = re.compile(
    r"(?i)\b(mortality|mortal|eternity|eternal|souls?|fate|destiny|grief|"
    r"sorrow|hope|memory|memories|silence|loneli\w*|alone|death|dying|love|"
    r"loss|meaning|purpose|humanity|human|beauty|wonder|forever|regret|"
    r"forgive\w*|sacrifice|legacy|truth|peace)\b")

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

        # dialogue-skit shape: most paragraphs open with a quote
        paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=\")", t) if p.strip()]
        if len(paras) >= 4:
            dfrac = sum(1 for p in paras
                        if p[0] in "\"“'") / float(len(paras))
            if dfrac > 0.55:
                s -= 0.06 * _sat((dfrac - 0.55) / 0.35)

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

        # sentence shape: flowing prose band + lexical richness
        try:
            n_sent, mwps, flw = ops.sent_stats(t)
            if n_sent and n_sent > 3:
                if 13.0 <= mwps <= 26.0:
                    s += 0.04
                elif mwps < 8.0:
                    s -= 0.03
                s += 0.04 * _sat((float(flw) - 0.15) / 0.15)
        except Exception:
            pass

        # ---- LLM fields: tacit mode/theme grounding ----
        ex = extracted if isinstance(extracted, dict) else {}
        mode = str(ex.get("mode", "") or "").strip().lower()
        if mode:
            if _MODE_COMEDY.search(mode):
                s -= 0.28
            elif _MODE_REFLECT.search(mode):
                s += 0.22
            # 'plain'/anything else: neutral

        theme = str(ex.get("theme", "") or "").strip()
        if "theme" in ex:
            tl = theme.lower()
            if not tl or tl in ("none", "none.", "n/a", "no"):
                s -= 0.08
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
