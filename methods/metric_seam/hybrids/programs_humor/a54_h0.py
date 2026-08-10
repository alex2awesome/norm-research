"""
Hybrid channel for a54: "Originality and genuine craft over formula or
clout-chasing" (humor task).

Design rationale (derived only from the improver pack for a54):
The frozen baseline (v1_structure, train rho=0.147) scores originality via a
POS/NEG keyword-density bag ("original", "unique", "fresh", ... vs "cliche",
"hack", ...). The pack's own train examples show this backfires: a text that
literally announces "An original one I wrote while bored in class" (d03394)
gets baseline_code_score=0.77 (near max) purely for saying the word
"original", while the judge rates it only 0.4 (mid-pack) -- the self-report
does not correlate with judged craft. Conversely the single highest-judged
example (d03656, judge=0.85, a quiet monk/"celebrate not celibate" pun) has
no such vocabulary and the baseline underscores it (0.37).

So this hybrid drops keyword self-report entirely and instead operationalizes
"formula/clout-chasing" as detectable SURFACE STRUCTURE:
  - well-known joke templates/memes (walks-into-a-bar, "in Soviet Russia",
    knock-knock, nationality trios, riddle formulas) -- these are the
    textual analogue of "algorithm-bait" / manufactured, mass-produced setups.
  - canned "performance" markers that substitute for craft (ROFL, HEY-OOO,
    LOL, exclamation runs, shouted ALL-CAPS, emoji bursts) -- these signal
    forwarded/chain-mail delivery rather than an authored punchline.
  - pre-emptive apology/disclaimer patterns ("please don't kill me", "not
    racist", "no offense") which the corpus shows accompany shock-value
    material used AS a manufactured-conflict crutch (worst-judged examples).
  - title-echo (title glued directly onto the joke body), a scrape artifact
    correlated with recycled/aggregator content rather than freshly told
    jokes.
These are penalties, not the POS-word rewards the baseline used.

For genuine craft we use structural economy (sentence-length regularity from
ops.sent_stats) and a spoken/quoted final line (dialogue punch-delivery,
common in the higher-judged examples) as mild positive priors, and -- more
importantly -- an EVIDENCE signal: ops.retrieve_similar over the corpus.
Reddit-style joke corpora are full of reposts/paraphrases of the same
handful of classic jokes; a text with a near-duplicate elsewhere in the
corpus is by definition "formula" (retold, not freshly crafted), while a
text with no close neighbor is corroborated as distinctive. This lets the
metric detect "formula" directly from corpus repetition rather than from
surface vocabulary, which the pack's failure notes call the weak proxy.

Two constructs are genuinely hard for regex and are handed to an LLM
extractor (kept SHORT, used only as booleans in code):
  - joke_template: does this match a widely-recognized joke template/meme
    (thick pattern-recognition over world knowledge of joke genres)?
  - craft_twist: is there an actual fresh/surprising turn in the punchline,
    as opposed to a telegraphed, expected resolution (a judgment about
    surprise that keyword/regex cannot make)?
The predicate on both fields (empty/NONE vs non-empty) stays in code.
"""

import re

# ---------------------------------------------------------------------------
# LLM extraction fields (thick-input grounding only; predicate stays in code)
# ---------------------------------------------------------------------------
LLM_FIELDS = {
    "joke_template": "Name the well-known joke template/meme/format this text follows in <=6 words, or NONE if it doesn't recognizably follow one.",
    "craft_twist": "In <=10 words, name the fresh/surprising turn that makes the ending land, or NONE if the punchline is predictable/telegraphed.",
}

# ---------------------------------------------------------------------------
# Structural "formula" detectors (compiled once at import time)
# ---------------------------------------------------------------------------
_FORMULA_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"walks?\s+into\s+a\s+bar",
    r"walked\s+into\s+a\s+bar",
    r"in\s+soviet\s+russia",
    r"knock,?\s*knock",
    r"yo\s+mama",
    r"why\s+did\s+the\s+\w+\s+cross\s+the\s+road",
    r"roses\s+are\s+red",
    r"(englishman|irishman|scotsman)\b.{0,60}\b(englishman|irishman|scotsman)",
    r"how\s+many\s+\w+.{0,25}\bdoes\s+it\s+take\s+to\s+(change|screw)",
    r"a\s+priest,?\s+a\s+rabbi",
]]

_CANNED_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"hey-?oo+",
    r"\brofl\b",
    r"\blol+\b",
    r"!{3,}",
    r"\b[A-Z]{5,}\b",
]]

_DISCLAIMER_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"please\s+don.?t\s+(kill|murder|hate)\s+me",
    r"not\s+a\s+racist",
    r"no\s+offense",
    r"i.?m\s+going\s+to\s+hell",
]]

_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

_QUOTE_TAIL_RE = re.compile(r'["“”\'][^"“”\']{0,80}["“”\']\s*\.?\s*$')
_SPOKEN_TAIL_RE = re.compile(r"\b(says?|replies?|asks?)\b.{0,60}$", re.IGNORECASE)


def _unpack_sent_stats(stats):
    """Accept either a dict or a (n_sent, mean_wps, frac_long) tuple/list."""
    if stats is None:
        return None, None, None
    if isinstance(stats, dict):
        n_sent = stats.get("n_sent")
        mean_wps = stats.get("mean_words_per_sent", stats.get("mean_wps"))
        frac_long = stats.get("frac_long_words")
        return n_sent, mean_wps, frac_long
    try:
        n_sent, mean_wps, frac_long = stats[0], stats[1], stats[2]
        return n_sent, mean_wps, frac_long
    except Exception:
        return None, None, None


def _max_dup_similarity(sims):
    """sims: list of 2-tuples whose element order may be (sim, id) or (id, sim).
    Returns the highest similarity to another (non-self) corpus item, or None."""
    if not sims:
        return None
    pairs = []
    for item in sims:
        try:
            a, b = item[0], item[1]
        except Exception:
            continue
        if isinstance(a, (int, float)) and not isinstance(a, bool):
            pairs.append(float(a))
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            pairs.append(float(b))
    if not pairs:
        return None
    pairs.sort(reverse=True)
    # drop a single near-exact self-match (the queried doc appearing in its own corpus)
    if pairs and pairs[0] >= 0.999:
        pairs = pairs[1:]
    if not pairs:
        return None
    return pairs[0]


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.5

        try:
            t = ops.normalize(text) if ops is not None and hasattr(ops, "normalize") else text
        except Exception:
            t = text
        if not t or not isinstance(t, str):
            t = text
        tl = t.lower()

        penalty = 0.0
        bonus = 0.0

        # --- known joke-template / formula openers ---
        formula_hits = sum(1 for p in _FORMULA_RE if p.search(tl))
        penalty += min(0.24, 0.08 * formula_hits)

        # --- canned "performance" / forwarded-chain markers ---
        canned_hits = sum(1 for p in _CANNED_RE if p.search(t))
        emoji_hits = len(_EMOJI_RE.findall(t))
        penalty += min(0.18, 0.06 * (canned_hits + (1 if emoji_hits >= 2 else 0)))

        # --- pre-emptive shock/conflict disclaimers ---
        disclaimer_hits = sum(1 for p in _DISCLAIMER_RE if p.search(tl))
        penalty += min(0.20, 0.10 * disclaimer_hits)

        # --- title-echo scrape artifact (aggregator title glued to body) ---
        words = tl.split()
        if len(words) > 12:
            head_phrase = " ".join(words[:6])
            if head_phrase and tl.count(head_phrase) >= 2:
                penalty += 0.05

        # --- economy / controlled structure (mild positive prior) ---
        try:
            stats = ops.sent_stats(text) if ops is not None and hasattr(ops, "sent_stats") else None
            n_sent, mean_wps, _frac_long = _unpack_sent_stats(stats)
            if isinstance(mean_wps, (int, float)) and 8.0 <= mean_wps <= 22.0:
                bonus += 0.05
            if isinstance(n_sent, (int, float)) and 2 <= n_sent <= 12:
                bonus += 0.02
        except Exception:
            pass

        # --- dialogue punch-delivery: closes on a short spoken/quoted line ---
        tail = t.strip()[-140:]
        if _QUOTE_TAIL_RE.search(tail) or _SPOKEN_TAIL_RE.search(tail.lower()):
            bonus += 0.04

        # --- evidence op: corpus-duplication as ground truth for "formula" ---
        try:
            sims = ops.retrieve_similar(text, k=5) if ops is not None and hasattr(ops, "retrieve_similar") else None
            dup_sim = _max_dup_similarity(sims)
            if dup_sim is not None:
                if dup_sim >= 0.6:
                    penalty += 0.20
                elif dup_sim >= 0.35:
                    penalty += 0.08
                elif dup_sim <= 0.12:
                    bonus += 0.10
        except Exception:
            pass

        # --- LLM fields: thick judgments regex cannot make ---
        ex = extracted if isinstance(extracted, dict) else {}
        joke_template = str(ex.get("joke_template") or "").strip().lower()
        craft_twist = str(ex.get("craft_twist") or "").strip().lower()
        if joke_template and joke_template not in ("none", "n/a", "na", "none.", "-"):
            penalty += 0.15
        if craft_twist and craft_twist not in ("none", "n/a", "na", "none.", "-"):
            bonus += 0.15

        s = 0.5 + bonus - penalty
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
