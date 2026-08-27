"""
Hybrid metric for a66: "Crux and key ideas foregrounded" (Math StackExchange answers).

Design rationale (see reply for the 4-line version):
The v0 keyword baseline (regex for "key idea", "crux", etc.) scored rho=-0.041 on
train -- keyword presence is not a proxy for this judgment. Reading the 30 train
examples, the judge instead rewards answers where a SINGLE insight is stated
plainly and briefly (score ~1.0: one-paragraph answers that state a lemma/fact
and stop), and punishes (a) long walls of chained display-equations with no
explanatory prose even when correct (score ~0.1-0.2: the crux is technically
present but drowned in mechanical algebra), and (b) non-answers/deflections
(counter-questions, "see example 3" pointers) that never state a crux at all
(score ~0.0-0.05). Whether an insight is "stated plainly, separate from
bookkeeping" vs. "buried in routine derivation" is a genuinely tacit judgment
(salience/foregrounding), not something a regex on keywords can see -- hence an
LLM_FIELDS extraction pair for it, cross-checked against cheap code-side
structural signals (answer length, math-span density vs. sentence count,
proof-skeleton connective/qed hits, deflection-pattern match) so a single LLM
mis-read can't dominate the score.
"""

import re

LLM_FIELDS = {
    "insight": (
        "In at most 8 words, name the ONE key mathematical insight or trick "
        "this answer relies on, or reply NONE if there isn't a distinct one."
    ),
    "foregrounded": (
        "Does the answer state its key insight plainly and early, clearly "
        "separated from routine algebra/bookkeeping? Reply YES, PARTIAL, or NO."
    ),
}

_ANSWER_SPLIT_RE = re.compile(r"\bAnswer\s*:\s*", re.IGNORECASE)

# Structural deflection patterns: counter-questions or bare pointers elsewhere,
# rather than any stated mathematical content. This is a *structural* signature
# (interrogative/pointer shape), not a keyword-presence proxy for quality, so it
# does not repeat the baseline's failure mode.
_DEFLECT_RE = re.compile(
    r"^\s*(do you|how do you|why (not|do)|what about|have you tried|"
    r"is (this|that)|are you|can you|see (the |this |example|link|above|below)|"
    r"this (link|prover|paper|post|reference) )",
    re.IGNORECASE,
)


def _answer_portion(text):
    parts = _ANSWER_SPLIT_RE.split(text)
    tail = parts[-1].strip() if len(parts) >= 2 else text.strip()
    return tail if tail else text.strip()


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = ops.normalize(text)
        ans = _answer_portion(t)
        if not ans:
            ans = t

        words = re.findall(r"[A-Za-z]+", ans)
        n_words = len(words)

        # --- code-side structural features (all deterministic, math-aware ops) ---
        try:
            spans = ops.extract_math_spans(ans)
        except Exception:
            spans = []
        math_chars = sum(len(c) for _, c in spans) if spans else 0
        math_density = math_chars / max(len(ans), 1)

        try:
            n_disp, n_inl, n_num, avg_tok = ops.equation_stats(ans)
        except Exception:
            n_disp, n_inl, n_num, avg_tok = 0, 0, 0, 0.0

        try:
            n_sent, mean_wps, frac_long = ops.sent_stats(ans)
        except Exception:
            n_sent, mean_wps, frac_long = 1, 0.0, 0.0

        try:
            skeleton = ops.proof_skeleton(ans)
            connective_hits = len(skeleton.get("connective", []))
            qed_hits = len(skeleton.get("qed", []))
        except Exception:
            connective_hits, qed_hits = 0, 0

        stripped = ans.strip()
        is_deflection = bool(_DEFLECT_RE.match(stripped))
        if n_words <= 10 and math_chars == 0 and connective_hits == 0:
            is_deflection = True

        # --- LLM-informed base (tacit foregrounding judgment) ---
        insight = (extracted or {}).get("insight", "")
        fg = (extracted or {}).get("foregrounded", "")
        insight = insight.strip() if insight else ""
        fg = fg.strip().upper() if fg else ""
        has_insight = bool(insight) and insight.upper() not in ("NONE", "N/A", "NO", "NONE.", "NA")

        if not extracted:
            base = None  # signal to use pure code fallback below
        elif not has_insight:
            base = 0.15
        elif fg.startswith("Y"):
            base = 0.85
        elif fg.startswith("P"):
            base = 0.5
        elif fg.startswith("N"):
            base = 0.2
        else:
            base = 0.45  # extractor named an insight but gave an unparsed foregrounded verdict

        if base is None:
            # Pure code fallback (no LLM signal available): reward short,
            # connective-marked answers; penalize long equation walls.
            length_score = 1.0 / (1.0 + n_words / 60.0)
            connective_bonus = min(connective_hits / 3.0, 1.0) * 0.15
            wall_penalty = 0.25 if (math_density > 0.55 and n_sent <= 3 and n_words > 30) else 0.0
            base = _clip(0.25 + 0.6 * length_score + connective_bonus - wall_penalty)

        # --- structural modulation: guards against an LLM over-crediting a
        # long derivation just because it technically contains a true fact ---
        if n_words > 220:
            base -= 0.15
        elif n_words > 120:
            base -= 0.08

        if math_density > 0.6 and n_sent <= 2 and n_words > 25:
            base -= 0.2  # wall of chained equations, no prose scaffolding

        if qed_hits > 0 and n_words > 150:
            base -= 0.05  # long formal derivation: crux likely diluted even if marked QED

        if is_deflection:
            base = min(base, 0.15)

        return _clip(base)
    except Exception:
        return 0.5
