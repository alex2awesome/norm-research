"""
Hybrid metric channel for aspect a81: "Topical angle and anchoring"
(choose salient current topics, recast them with a sharp angle, and tie to
the moment with agile callbacks and updates).

Design rationale:
  The frozen code baseline (v2_holistic) fired on generic narrative
  connectives ("today", "yesterday", "just") and inline year strings, which
  are WEAK proxies -- they fire equally on evergreen bar/animal/priest jokes
  that merely narrate a sequence of days. The real signal is whether the
  joke is actually built on a real-world referent (a named public figure,
  institution, brand, or ongoing/current event) AND whether that referent is
  genuinely *recast* with a twist, rather than just named as a stock label
  or character name.

  "Is there a real, current-world anchor" and "is the twist genuinely a
  recasting of it" are THICK-INPUT constructs -- judging whether "Soviet
  adviser" or "Putin" or "COVID19" counts as a live anchor, and whether the
  punchline actually comments on/subverts that anchor (vs. just borrowing
  its name), requires world knowledge and reading comprehension code can't
  do reliably. Those two judgments are delegated to the two LLM fields
  below. The PREDICATE -- how presence/specificity/recency/sharpness combine
  into a score -- stays entirely in code, deterministic and stdlib-only.
"""

import re
import math
import statistics
from collections import Counter

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

_RECENCY_MARKERS = [
    "covid", "coronavirus", "pandemic", "lockdown", "vaccine",
    "election", "president", "prime minister", "parliament", "senate",
    "congress", "climate", "global warming", "inflation", "recession",
    "shutdown", "twitter", "tiktok", "facebook", "instagram", "brexit",
    "ceasefire", "sanctions", "refugee", "immigra", "protest", "strike",
    "trending", "viral", "breaking news", "war in",
]

_SHARP_CUES = [
    "iron", "twist", "subvert", "pun", "revers", "mock", "satir", "flip",
    "undercut", "pointed", "sarcas", "absurd", "juxtapos", "contrast",
    "double meaning", "double-meaning", "misdirect",
]

_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")


def _is_none(s):
    if not isinstance(s, str):
        return True
    return s.strip().strip(".").lower() in _NONE_WORDS


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

        # --- computation-op evidence for "tie to the moment" ---
        hits = sum(1 for m in _RECENCY_MARKERS if m in low)
        try:
            dates = ops.extract_dates(t) or []
        except Exception:
            dates = []
        years = _YEAR_RE.findall(t)
        recency_evidence = min(1.0, 0.25 * hits + 0.2 * len(dates) + 0.2 * len(years))

        if not has_topic:
            # No genuine real-world anchor -> criterion not met, regardless
            # of surface date-ish words (those are what fooled the baseline).
            base = 0.06
            return max(0.0, min(1.0, base + 0.10 * recency_evidence))

        val = 0.28  # credit for having *some* real-world topical anchor

        # Generic stock-character openers ("A blonde walks into...") often
        # smuggle a topic label in as a character descriptor, not a genuine
        # current-events anchor -- discount slightly even if a topic fired.
        if _GENERIC_OPENERS.match(t.strip()):
            val -= 0.08

        # Specificity of the named anchor: multi-token / capitalized names
        # read as concrete referents rather than vague categories.
        topic_words = [w for w in topic.split() if w]
        cap_words = sum(1 for w in topic_words if w[:1].isupper())
        if cap_words >= 1:
            val += 0.06
        if len(topic_words) >= 2:
            val += 0.04

        # Explicit recency/dating evidence in the text itself.
        val += 0.18 * recency_evidence

        # The sharp-angle RECAST is the dominant criterion component: a
        # topic that's merely named (no twist) earns little beyond the
        # anchor credit above; a genuinely ironic/subversive twist earns
        # the largest single bonus.
        if has_twist:
            twist_low = twist.lower()
            sharp = any(c in twist_low for c in _SHARP_CUES)
            val += 0.30 if sharp else 0.12

        # Economy / agility: tight, punchy delivery reads as an "agile"
        # callback rather than a meandering retelling.
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
