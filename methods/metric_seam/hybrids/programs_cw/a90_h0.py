"""Hybrid metric channel for a90: Sensory immersion and spatial dynamism.

Signal model (from train residuals):
- Judge-low texts are dialogue/chat/script/memo-dominant, or expository
  narration with near-zero concrete sensory description.
- Judge-high texts are narration-heavy prose dense with sound/smell/touch/
  light/motion imagery, spread across several sense modalities.
The predicate stays in code: dialogue stripping + format detection + a
multi-modal sensory lexicon over NARRATION only, times a narration-fraction
gate, plus spatial-preposition density (the part of the baseline that works).
One LLM field grounds what the lexicon can't see (novel sensory phrasing),
as a constrained "which senses" checklist a mid-size LLM answers reliably.
"""

import re
import math

LLM_FIELDS = {
    "senses_evoked": (
        "Which senses does the story's descriptive narration (not dialogue) "
        "concretely evoke: sight, sound, smell, taste, touch, temperature, "
        "motion? List only matching sense names, or NONE."
    ),
}

# --- sensory lexicons: one word-boundary regex per modality ---
# ambiguous short stems get exact suffix classes (mist!=mistake, numb!=number,
# sour!=source, grit!=gritted[teeth], shadow!=shadowsofclouds-url)

_RE_SOUND = re.compile(r"\b(?:"
    r"echo\w*|creak\w*|crunch\w*|rustl\w*|hiss\w*|clatter\w*|squelch\w*|"
    r"jingl\w*|thud\w*|whisper\w*|wheez\w*|murmur\w*|rumbl\w*|buzz\w*|"
    r"screech\w*|crackl\w*|rattl\w*|chime\w*|dripp?\w*|staccato|shriek\w*|"
    r"hush\w*|silen(?:ce|t)\w*|cacophon\w*|clank\w*|squeak\w*|clang\w*|"
    r"sizzl\w*|gurgl\w*|patter(?:ed|ing)?|howl\w*|hooves|hoofbeats?|"
    r"footsteps?|deafening|shrill"
    r")\b")
_RE_SMELL = re.compile(r"\b(?:"
    r"smell\w*|smelt|stench|scent\w*|reek\w*|odou?rs?|aroma\w*|"
    r"sulphur\w*|sulfur\w*|musty|fetid|waft\w*|perfum\w*|acrid|"
    r"smoke|smoky|smoking|incense|tast(?:e|es|ed|ing)|sour(?:ed|ness)?|"
    r"bitter\w*|honeyed"
    r")\b")
_RE_TOUCH = re.compile(r"\b(?:"
    r"cold\w*|warm\w*|chill\w*|damp\w*|wet|rough\w*|smooth\w*|sting\w*|"
    r"icy|frost\w*|sweat\w*|clammy|tingl\w*|shiver\w*|goose[- ]?pimpl\w*|"
    r"goosebumps?|numb(?:ed|ing|ness)?|slick|sticky|grit(?:s|ty)?|coarse|"
    r"silk\w*|velvet\w*|prickl\w*|burn(?:ed|ing|t)?|scald\w*|heat|breez\w*|"
    r"drench\w*|sodden|trembl\w*|papery|leathery|parchment|foam\w*|froth\w*|"
    r"searing|soggy|slimy|slime"
    r")\b")
_RE_LIGHT = re.compile(r"\b(?:"
    r"glow\w*|gleam\w*|glint\w*|shimmer\w*|flicker\w*|shadows?|shadowy|"
    r"dim(?:ly|med|mer)?|gloom\w*|blinding|dazzl\w*|moonli\w*|dusk|"
    r"twilight|lanterns?|flames?|embers?|lumin\w*|glisten\w*|haz(?:e|y)|"
    r"mist(?:s|y|ier)?|murky|pale(?:r|st|ness)?|scarlet|crimson|golden|"
    r"silvery|violet|turquoise|caramel|amber|copper|ivory|emerald|"
    r"polish(?:ed|ing)?|gilded"
    r")\b")
_RE_MOTION = re.compile(r"\b(?:"
    r"crep[t]|creep\w*|crawl\w*|glid(?:e|es|ed|ing)|swirl\w*|tumbl\w*|"
    r"scuttl\w*|drift\w*|plung\w*|stumbl\w*|lurch\w*|soar\w*|dart(?:ed|ing)|"
    r"slither\w*|spiral\w*|descend\w*|float\w*|sway\w*|flail\w*|stagger\w*|"
    r"sprawl\w*|loom\w*|hover\w*|scrambl\w*|surg(?:e|es|ed|ing)|cascad\w*|"
    r"rippl\w*|flutter\w*|edd(?:y|ies|ied)|billow\w*|swept|sweep\w*|"
    r"burst\w*|swam|flood\w*|stream(?:ed|ing)|splay\w*|splash\w*|bubbl\w*"
    r")\b")
_RE_SCENE = re.compile(r"\b(?:"
    r"marble|mahogany|granite|brass|cobblestone\w*|gravel|asphalt|"
    r"rafters?|floorboards?|carpet\w*|curtains?|duvet|linen|cushion\w*|"
    r"tiles?|columns?|pillars?|archway\w*|doorways?|corridors?|hallways?|"
    r"stairwells?|staircases?|windowsill\w*|pavement|sidewalks?|ceilings?|"
    r"chandelier\w*|waterfalls?|pews?|alcoves?|ledges?|boulders?|"
    r"caverns?|dust\w*|cobwebs?|panell?ed|parquet|veranda\w*|awnings?"
    r")\b")
_MODALITIES = (_RE_SOUND, _RE_SMELL, _RE_TOUCH, _RE_LIGHT, _RE_MOTION,
               _RE_SCENE)

_SPATIAL_PREP = frozenset((
    "above", "below", "beneath", "under", "over", "behind", "beside",
    "between", "beyond", "across", "through", "inside", "outside",
    "toward", "towards", "into", "onto", "past", "around", "along",
    "underneath", "amid", "amidst", "atop", "against",
))

_SENSE_NAMES = ("sight", "sound", "smell", "taste", "touch", "temperature",
                "motion", "movement", "hearing", "visual")

# speaker-label line: "**Name**:", "Name:", "*Name*:", memo/chat headers
_SPEAKER_RE = re.compile(
    r"^\s*[*_]{0,2}[A-Z][A-Za-z0-9'’ .\-]{0,30}[*_]{0,2}\s*:")
_MEMO_RE = re.compile(
    r"^\s*[*_]{0,2}(to|cc|from|subject|re|entry|server|players?|accessing)"
    r"[*_]{0,2}\s*[:\d]", re.IGNORECASE)
_STAGE_RE = re.compile(r"^\s*\*[^*\n]{1,80}\*\s*$")  # *Peter has joined...*
_QUOTE_RE = re.compile(r'"[^"\n]{1,800}"')
_WORD_RE = re.compile(r"[a-z']+")


def _strip_markdown(t):
    t = re.sub(r"&gt;|&amp;|&lt;", " ", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)  # markdown links
    t = re.sub(r"\bhttps?://\S+", " ", t)
    t = re.sub(r"/r/\w+", " ", t)
    t = t.replace("**", "").replace("__", "")
    return t


def _narration(t):
    """Remove dialogue/chat/memo material; return narration text and stats."""
    lines = t.split("\n")
    kept, n_label, n_lines = [], 0, 0
    for ln in lines:
        if not ln.strip():
            continue
        n_lines += 1
        if _MEMO_RE.match(ln) or _STAGE_RE.match(ln) or _SPEAKER_RE.match(ln):
            n_label += 1
            continue
        kept.append(ln)
    body = "\n".join(kept)
    total_words = len(_WORD_RE.findall(body.lower()))
    # strip quoted dialogue (ops.normalize converts smart quotes first)
    narr = _QUOTE_RE.sub(" ", body)
    label_frac = n_label / n_lines if n_lines else 0.0
    return narr, total_words, label_frac


def _sat(hits, words, scale):
    """Saturating density score in [0,1)."""
    if words <= 0:
        return 0.0
    return 1.0 - math.exp(-(hits / words) / scale)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = text or ""
        try:
            t = ops.normalize(t)
        except Exception:
            pass
        t = _strip_markdown(t)
        for a, b in (("“", '"'), ("”", '"'),
                     ("‘", "'"), ("’", "'")):
            t = t.replace(a, b)

        narr, body_words, label_frac = _narration(t)
        low = narr.lower()
        n = len(_WORD_RE.findall(low))
        if n < 30:
            code = 0.05
        else:
            narr_frac = min(1.0, n / body_words) if body_words else 0.0

            mod_hits = [len(rx.findall(low)) for rx in _MODALITIES]
            total_hits = sum(mod_hits)
            # breadth (any evidence) + depth (repeated evidence) of modalities
            cov = (sum(1 for h in mod_hits if h >= 1)
                   + sum(1 for h in mod_hits if h >= 3)) / (2.0 * len(mod_hits))

            words = _WORD_RE.findall(low)
            p_hits = sum(1 for w in words if w in _SPATIAL_PREP)

            s_sens = _sat(total_hits, n, 0.022)
            s_prep = _sat(p_hits, n, 0.022)

            quality = 0.48 * s_sens + 0.22 * cov + 0.30 * s_prep
            code = quality * (0.30 + 0.70 * narr_frac)
            # chat/script/memo formats: immersion judged near-zero
            if label_frac > 0.25:
                code *= 0.35
            elif label_frac > 0.10:
                code *= 0.7

        # LLM grounding: distinct senses the extractor saw in narration
        ans = extracted.get("senses_evoked") if isinstance(extracted, dict) \
            else None
        if ans is None:
            final = code
        else:
            a = str(ans).lower()
            if not a.strip() or a.strip() == "none":
                llm = 0.0
            else:
                found = set()
                for s in _SENSE_NAMES:
                    if s in a:
                        found.add("motion" if s == "movement" else
                                  "sound" if s == "hearing" else
                                  "sight" if s == "visual" else s)
                llm = min(1.0, len(found) / 5.0)
            final = 0.72 * code + 0.28 * llm
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
