"""a63: Benign-violation calibration (hybrid v0).

Criterion: norm violations must be strong enough to be funny yet framed as
safe/benign -- penalize both extremes: too threatening (graphic/cruel/real)
and too safe (no real violation at all, i.e. inert wordplay with no bite).

Design:
  - Code layer handles what regex CAN see: presence of taboo/transgressive
    topic words (is there a violation at all?), surface tone-of-threat cues
    (shouting via caps ratio, distress verbs, defensive self-aware
    disclaimers like "not a racist" / "please don't kill me" which are a
    strong tell that the author knows the joke oversteps), and a light
    economy/rambling penalty from sentence stats.
  - LLM_FIELDS supply the THICK judgment code cannot make: whether the
    violation (once detected) is framed as absurd/fictional/undercut by the
    punchline (benign) vs. graphic/plausible/dwelt-upon (threatening), and
    whether the ending actually defuses the tension. These catch cases like
    an understated implied assault where no lexicon word ever fires.
  - ops.retrieve_similar is used only as a safety valve: if a short text is
    an outlier relative to the corpus (near-zero similarity to its nearest
    neighbors), it may be scraped chrome/stub rather than a joke, so we stay
    neutral (0.5) instead of confidently scoring noise.
"""
import re

TABOO_TERMS = [
    "sex", "fuck", "fucking", "screw", "nude", "naked", "breast", "boobs",
    "penis", "vagina", "virgin", "orgasm", "porn", "condom",
    "rape", "raped", "molest", "molested", "murder", "kill", "killed",
    "killing", "dead", "death", "blood", "bleeding", "gun", "shoot", "shot",
    "stab", "stabbed", "torture", "tortured", "suicide", "abuse", "abused",
    "beat", "beaten", "punch", "punched", "slap", "slapped",
    "racist", "racism", "slur", "nigger", "retard", "retarded",
    "nazi", "hitler", "holocaust", "genocide",
    "drunk", "alcoholic", "drugs", "cocaine", "heroin",
    "prison", "jail", "steal", "stole", "robbery",
    "cancer", "disabled", "crippled", "blind", "deaf", "crazy", "insane",
    "pedophile", "incest",
]
TABOO_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in TABOO_TERMS) + r")\b")

DISTRESS_TERMS = [
    "sobbed", "sobbing", "wept", "weeping", "screamed", "screaming",
    "begged", "begging", "pleaded", "pleading", "strangled", "choked",
    "choking", "mutilated", "dismembered", "shrieked", "wailing", "wailed",
]
DISTRESS_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in DISTRESS_TERMS) + r")\b")

DISCLAIMER_RE = re.compile(
    r"(not\s+(a\s+)?racist|please\s+don.?t\s+(kill|hurt|murder|hate)\s*me|"
    r"no\s+offense|just\s+kidding|\bjk\b|too\s+soon|"
    r"not\s+trying\s+to\s+offend|sorry\s+if\s+this\s+offends|\b/s\b)",
    re.I,
)

LLM_FIELDS = {
    "violation_calibration": (
        "Classify this joke's norm violation in one phrase: TOO_SAFE (no real "
        "transgression), WELL_CALIBRATED (clear violation framed as safe/absurd/"
        "undercut), or TOO_THREATENING (graphic, cruel, or genuinely distressing)."
    ),
    "tension_resolved": (
        "Does the ending defuse or undercut the transgression (yes), or leave it "
        "raw and unresolved (no)? Answer yes, no, or unclear."
    ),
}


def _sent_stats(t, ops):
    try:
        s = ops.sent_stats(t)
    except Exception:
        return (0, 0.0, 0.0)
    try:
        if isinstance(s, dict):
            n = s.get("n_sent", s.get("num_sentences", 0)) or 0
            mw = s.get("mean_words_per_sent", s.get("mean_words", 0.0)) or 0.0
            fl = s.get("frac_long_words", 0.0) or 0.0
        else:
            n, mw, fl = s[0], s[1], s[2]
        return (n, mw, fl)
    except Exception:
        return (0, 0.0, 0.0)


def _neighbor_similarities(t, ops):
    sims = []
    try:
        neigh = ops.retrieve_similar(t, k=5)
        for item in neigh or []:
            try:
                a, b = item[0], item[1]
            except Exception:
                continue
            if isinstance(a, (int, float)) and not isinstance(a, bool):
                sims.append(float(a))
            elif isinstance(b, (int, float)) and not isinstance(b, bool):
                sims.append(float(b))
    except Exception:
        return []
    return sims


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        extracted = extracted if isinstance(extracted, dict) else {}

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw
        tl = t.lower()

        # --- corpus-typicality safety valve (evidence op) ---
        sims = _neighbor_similarities(t, ops)
        if sims:
            mean_sim = sum(sims) / len(sims)
            if mean_sim < 0.03 and len(t) < 120:
                return 0.5  # likely off-corpus scrap, not a joke -> stay neutral

        # --- code layer: is a violation even present? ---
        taboo_hits = len(TABOO_RE.findall(tl))
        violation_presence = min(1.0, taboo_hits / 3.0)

        # --- code layer: threat / non-benign tone cues ---
        disclaimer_hit = 1.0 if DISCLAIMER_RE.search(tl) else 0.0

        letters = [c for c in t if c.isalpha()]
        upper_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
        caps_penalty = min(1.0, max(0.0, (upper_ratio - 0.08) / 0.25))

        distress_hits = len(DISTRESS_RE.findall(tl))
        distress_penalty = min(1.0, distress_hits / 2.0)

        threat_penalty = min(1.0, 0.45 * disclaimer_hit + 0.30 * caps_penalty + 0.35 * distress_penalty)

        # --- code layer: light economy/rambling adjustment ---
        _n_sent, _mean_wps, frac_long = _sent_stats(t, ops)
        economy_adj = -0.03 if (frac_long and frac_long > 0.35) else 0.0

        code_score = 0.3 + 0.55 * violation_presence - 0.55 * threat_penalty + economy_adj
        if taboo_hits == 0:
            code_score = min(code_score, 0.30)  # nothing violated -> "too safe" ceiling
        code_score = max(0.0, min(1.0, code_score))

        # --- LLM layer: thick judgment on framing ---
        cal = str(extracted.get("violation_calibration") or "").strip().lower()
        res = str(extracted.get("tension_resolved") or "").strip().lower()

        llm_score = None
        if cal:
            if "too_safe" in cal or "too safe" in cal or "no real" in cal or "no violation" in cal:
                llm_score = 0.18
            elif ("too_threatening" in cal or "too threatening" in cal or "threatening" in cal
                  or "graphic" in cal or "cruel" in cal or "distressing" in cal or "disturbing" in cal):
                llm_score = 0.08
            elif ("well_calibrated" in cal or "well calibrated" in cal or "calibrated" in cal
                  or "benign" in cal or "safe" in cal):
                llm_score = 0.85

        adj = 0.0
        if res:
            if res.startswith("yes") or "defuse" in res or "undercut" in res or "resolved" in res:
                adj += 0.08
            elif res.startswith("no") or "raw" in res or "unresolved" in res:
                adj -= 0.08

        if llm_score is not None:
            final = 0.65 * llm_score + 0.35 * code_score + adj
        else:
            final = code_score + 0.5 * adj

        # Defensive self-aware disclaimers ("please don't kill me", "not a
        # racist") are strong direct evidence the joke oversteps benign.
        if disclaimer_hit:
            final -= 0.35

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
