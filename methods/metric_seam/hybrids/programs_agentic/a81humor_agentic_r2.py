"""
Hybrid metric channel (agentic recode, round 2) for aspect a81:
"Topical angle and anchoring" -- choose salient CURRENT topics, recast them
with a sharp angle, and tie to the moment with agile callbacks/updates.

Diagnosis of h0 (train rho 0.5619):
  h0 gates almost all of its credit on the LLM field `topic_anchor` firing
  ("is there SOME real-world referent"), and treats "has_topic" as a proxy
  for "is CURRENT." In practice the field fires just as readily on
  historical/Cold-War/timeless referents (Hitler, Soviet Union, KGB, Desert
  Storm, September 11, a biscuit brand) as on genuinely current ones, even
  though its own instruction says "not a historical-fiction setting" --
  the field cannot reliably enforce that distinction. That was the single
  largest source of h0's worst residuals: historical-reference jokes the
  judge scored 0.0 were scored 0.45-0.73 by h0 purely because *some* named
  entity + a twist were present.

  RECODE: "is this actually CURRENT / tied to the moment" is a fact about
  the surface text (recent years, ongoing-event language, immediacy
  adverbs, contemporary named figures) and about the extractor's OWN
  wording (does its short answer read as a Cold-War/WWII/ancient marker,
  or as a brand/franchise/fictional label?) -- both fully CODE-checkable,
  deterministic, and independent of the field's own "is it historical"
  judgment call, which is exactly what fails in h0.

  So this version keeps h0's topic/specificity/twist credit essentially
  as-is (that part already worked), but adds a CODE-DOMINANT hard filter:
  `_is_historical_or_legacy` scans the raw text AND the topic/twist field
  strings for an enumerable historical/legacy/franchise marker list, and
  if any hit, the score HARD-COLLAPSES to the same code-owned currency
  floor used for topic-less documents (`base`, computed purely from
  code-detected recency evidence in the raw text) -- regardless of how
  confidently the field asserted a real-world anchor. This directly
  overrides the field's currency judgment with a code-owned one, moving
  the criterion's central predicate -- "is the referent current, not just
  real" -- from field to code, the FIELD-DOMINATED -> CODE-DOMINANT
  restructuring requested. (Earlier iterations tried gating topic credit
  multiplicatively by *positive* recency evidence instead of penalizing
  *negative*/historical evidence; that punished legitimately-current but
  undated topics like "shootings in America" or "Israel-Palestine
  conflict" that don't need a date to be current, and scored worse on
  train -- the negative/enumerable-blacklist framing generalizes better.)

  The two LLM fields are kept (same names/instructions as h0, reused
  as-is so train iteration sees real field values) for the residual
  THICK-INPUT judgments code truly cannot make: (a) whether a specific
  real-world referent is present at all, and (b) whether the punchline
  genuinely recasts/subverts it (vs. just borrowing its name as a label).
  Their surface WORDING is now also read by code (for legacy/franchise
  labels), which is a genuinely new code-dominant use of the field output.
"""

import re
import math
import statistics

LLM_FIELDS = {
    "topic_anchor": (
        "In <=8 words, name the specific real public figure, institution, "
        "brand, or CURRENT/ongoing event this joke is built on (not a "
        "historical-fiction setting); say NONE if it is a generic/timeless "
        "joke with no real-world anchor."
    ),
    "angle_twist": (
        "In <=15 words, describe the ironic/subversive twist the joke "
        "applies to that real-world topic; say NONE if the topic is only "
        "used as a label/character name with no genuine recast."
    ),
}

_NONE_WORDS = {
    "none", "n/a", "na", "no", "no topic", "not applicable", "-", "", "null",
    "n/a.", "none.",
}

_GENERIC_OPENERS = re.compile(
    r"^\s*[\"']?\s*(a|an|the)\s+(man|woman|guy|blonde|priest|rabbi|nun|"
    r"farmer|doctor|lawyer|husband|wife|dog|cat|duck|goose|parrot|"
    r"irishman|englishman|scotsman|robber|genie)\b",
    re.IGNORECASE,
)

# --- CODE-ONLY currency evidence (the crux of the recode) ---------------
# These are checked against the raw TEXT, never against the LLM fields, so
# they are a fully code-owned signal of "tied to the moment" that does not
# depend on the extractor's judgment call about historical-vs-current.
_CURRENT_EVENT_MARKERS = [
    "covid", "coronavirus", "pandemic", "lockdown", "vaccine", "quarantine",
    "election", "president", "prime minister", "parliament", "senate",
    "congress", "climate", "global warming", "inflation", "recession",
    "shutdown", "twitter", "tiktok", "facebook", "instagram", "brexit",
    "ceasefire", "sanctions", "refugee", "immigra", "protest", "strike",
    "trending", "viral", "breaking news", "war in", "wildfire", "hurricane",
    "supreme court", "impeach", "midterm", "lawsuit", "stock market",
]
# Contemporary (roughly 2015-2026-era) named public figures/institutions --
# a name-only reference ("Trump", "Merkel", "Putin") carries no numeral for
# the year-regex below to catch but is just as clear a currency cue as a
# date; this is world knowledge, not label-derived.
_CURRENT_FIGURES = [
    "trump", "biden", "obama", "putin", "merkel", "boris johnson",
    "kim jong", "xi jinping", "modi", "macron", "trudeau", "zelensky",
    "elon musk", "bezos", "zuckerberg", "greta thunberg", "isis", "taliban",
    "fsb",
]
_RECENT_YEAR_RE = re.compile(r"\b20(1[5-9]|2\d)\b")  # 2015-2029
_IMMEDIACY_RE = re.compile(
    r"\b(today|this week|this month|right now|these days|nowadays|"
    r"just announced|just released|latest|breaking)\b", re.IGNORECASE)

# CODE-owned NEGATIVE evidence: explicit Cold-War/WWII/ancient-era markers.
# A "real-world referent" that is transparently historical is the dominant
# false-positive class the field's own "not historical-fiction" judgment
# fails to filter (Hitler, Soviet Union, KGB, Desert Storm, 9/11 nostalgia
# all still get tagged as valid anchors by the field). Rather than
# requiring a POSITIVE current-events keyword for every legitimately
# ongoing-but-undated topic (immigration, gun violence, a sports league,
# a geopolitical conflict -- none of which need a date to be "current"),
# code instead penalizes the much smaller, enumerable set of markers that
# are unambiguously NOT current.
_HISTORICAL_MARKERS = [
    "soviet", "ussr", "kgb", "hitler", "nazi", "third reich", "world war",
    "wwii", "ww2", "cold war", "desert storm", "gulf war", "vietnam war",
    "stalin", "khrushchev", "gorbachev", "berlin wall", "communist bloc",
    "ancient egypt", "pharaoh", "mummy", "medieval", "middle ages",
    "september 11", "9/11", "twin towers", "world trade center",
]
# CODE-owned check on the field's OWN short answer text (not just the
# document): if the extractor's topic/twist string itself reads as a
# generic/perennial brand, franchise, or fictional-character label
# ("Parle (biscuit brand)", "brand name", "Batman", "Doctor Who"), that is
# evergreen wordplay, not a topic tied to a moment -- discount it the same
# way, using only the fields' surface wording, no semantics beyond regex.
_LEGACY_LABEL_WORDS = [
    "brand", "biscuit", "cereal", "soda company", "product line", "mascot",
    "cartoon character", "sitcom", "franchise", "fictional",
]
_FICTIONAL_FRANCHISES = [
    "batman", "superman", "spider-man", "spiderman", "doctor who",
    "star wars", "star trek", "harry potter", "james bond", "sherlock",
    "marvel", "dc comics", "lord of the rings", "pokemon", "pokémon",
]

_SHARP_CUES = [
    "iron", "twist", "subvert", "pun", "revers", "mock", "satir", "flip",
    "undercut", "pointed", "sarcas", "absurd", "juxtapos", "contrast",
    "double meaning", "double-meaning", "misdirect",
]


def _is_none(s):
    if not isinstance(s, str):
        return True
    return s.strip().strip(".").lower() in _NONE_WORDS


def _recency_code_evidence(t, low, ops):
    """CODE-ONLY: is this text actually anchored to a recent/current
    moment, independent of what the LLM's topic_anchor field claims?"""
    hits = sum(1 for m in _CURRENT_EVENT_MARKERS if m in low)
    fig_hits = sum(1 for m in _CURRENT_FIGURES if m in low)
    recent_years = len(_RECENT_YEAR_RE.findall(t))
    try:
        dates = ops.extract_dates(t) or []
    except Exception:
        dates = []
    immediacy = len(_IMMEDIACY_RE.findall(low))
    raw = (
        0.30 * min(hits, 2)
        + 0.35 * min(fig_hits, 2)
        + 0.35 * min(recent_years, 1)
        + 0.10 * min(len(dates), 2)
        + 0.15 * min(immediacy, 2)
    )
    return max(0.0, min(1.0, raw))


def _is_historical_or_legacy(low, topic_low, twist_low):
    """CODE-ONLY hard filter: does EITHER the raw text OR the extractor's
    own field strings contain an explicit Cold-War/WWII/ancient-era
    marker, or a brand/franchise/fictional-character label? Checking the
    field strings too matters because the joke text itself often never
    spells out "Hitler" or "September 11" even though the inferred label
    names it -- the marker is only visible in the field's own wording."""
    combined = " ".join([low, topic_low, twist_low])
    if any(m in combined for m in _HISTORICAL_MARKERS):
        return True
    if any(w in topic_low or w in twist_low for w in _LEGACY_LABEL_WORDS):
        return True
    if any(f in combined for f in _FICTIONAL_FRANCHISES):
        return True
    return False


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
        low = t.lower()

        ext = extracted if isinstance(extracted, dict) else {}
        topic = ext.get("topic_anchor", "")
        twist = ext.get("angle_twist", "")
        has_topic = not _is_none(topic)
        has_twist = has_topic and not _is_none(twist)

        rec_ev = _recency_code_evidence(t, low, ops)

        # --- CODE-OWNED CURRENCY FLOOR -----------------------------------
        # Shared by BOTH branches below. This is what makes the criterion
        # code-dominant: whether a document is "tied to the moment" at all
        # is decided purely from surface recency evidence, independent of
        # whatever the topic_anchor field claims.
        base = 0.06 + 0.14 * rec_ev

        if not has_topic:
            return max(0.0, min(1.0, base))

        # A referent is claimed -- start from full h0-style credit (a real,
        # specific, capitalized, twisted anchor is genuinely worth a lot),
        # then apply the CODE-DOMINANT part: a multiplicative discount from
        # the historical/legacy-label penalty below. That penalty is what
        # actually separates "current" from "merely real" -- the field's
        # own "not historical-fiction" instruction cannot enforce this
        # reliably (it still tags Hitler, Soviet Union, KGB, Desert Storm,
        # 9/11 nostalgia, and legacy brand puns as valid anchors).
        val = 0.28
        if _GENERIC_OPENERS.match(t.strip()):
            val -= 0.08

        topic_words = [w for w in topic.split() if w]
        cap_words = sum(1 for w in topic_words if w[:1].isupper())
        if cap_words >= 1:
            val += 0.06
        if len(topic_words) >= 2:
            val += 0.04

        val += 0.18 * rec_ev

        twist_low = twist.lower()
        if has_twist:
            sharp = any(c in twist_low for c in _SHARP_CUES)
            val += 0.30 if sharp else 0.12

        # --- CODE-DOMINANT DISCOUNT ---------------------------------------
        # Hard collapse (not a partial multiplicative fade) back to the
        # shared currency floor whenever code detects the claimed anchor is
        # historical, or a brand/franchise/fictional label -- this is the
        # actual current-vs-merely-real distinction the field cannot make,
        # and every train item hitting this filter was judged ~0.0.
        topic_low = topic.lower()
        if _is_historical_or_legacy(low, topic_low, twist_low):
            val = base
        else:
            # Never fall below the shared code-owned currency floor.
            val = max(val, base)

        # Economy/agility: tight, punchy delivery reads as an "agile"
        # callback rather than a meandering retelling. Structural, so left
        # ungated by currency.
        try:
            stats = ops.sent_stats(t)
            n_sent = mean_wps = frac_long = None
            if isinstance(stats, dict):
                n_sent = stats.get("n_sent", stats.get("n_sentences"))
                mean_wps = stats.get("mean_words_per_sent", stats.get("mean_wps"))
                frac_long = stats.get("frac_long_words")
            elif stats:
                n_sent, mean_wps, frac_long = stats
            if n_sent and mean_wps and mean_wps <= 16 and n_sent <= 8:
                val += 0.05
        except Exception:
            pass

        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
