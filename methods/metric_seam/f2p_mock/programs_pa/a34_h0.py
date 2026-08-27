"""a34 -- Novelty and public-disclosure bars (hybrid, prior-art evidence-op enabled).

Construct: is the claimed invention new over a single enabling prior-art disclosure, and
is the application clean of public-disclosure/on-sale exposure? The prior_art evidence op
is the natural anchor for the FIRST half: a claim with >=1 disclosing reference (discloses=
True) is the textbook anticipation signature (single-reference novelty rejection), so
frac_claims_any_disclose / max_frac_disclose drive the score down, while a well-searched
claim set (decent retrieval_top_scores) with ZERO disclosing refs is affirmative evidence of
novelty and drives the score up. Text-side signals (novelty-framing language, explicit
prior-art discussion, disclosure/grace-period statements) modulate lightly. The LLM layer
asks the document to name its own closest prior art and its own distinguishing feature,
each validated for containment in the normalized text before it earns a (small) weight.
"""
import re

LLM_FIELDS = {
    "closest_art": "in at most 15 words, what single prior-art reference or product does "
                   "the document itself identify as closest; answer NONE if none is named",
    "distinguishing": "in at most 15 words, what one feature does the document say "
                      "distinguishes the claims from that prior art; answer NONE if absent",
}

_NOVEL_KW = re.compile(r"\b(novel|new(ly)?|unlike (the )?prior art|not (been )?disclosed|"
                       r"no prior art (teaches|discloses)|distinguish(es|ed)? over)\b", re.I)
_PRIORART_DISC = re.compile(r"\b(prior art|background art|conventional(ly)?|known in the art|"
                            r"previously (known|disclosed))\b", re.I)
_DISCLOSURE_STMT = re.compile(r"\b(grace period|public(ly)? disclos\w*|on.?sale bar|"
                              r"printed publication|prior disclosure by (the )?inventor)\b", re.I)


def score(text, extracted, ops, dpid=None):
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        head = t[:4000]
        s = 0.5

        # --- CODE layer: weak textual novelty-framing proxies ------------------------
        n_novel = len(_NOVEL_KW.findall(head))
        s += min(0.08, 0.03 * n_novel)                  # explicit novelty framing
        if _PRIORART_DISC.search(head):
            s += 0.04                                    # engages with prior art at all
        if _DISCLOSURE_STMT.search(t):
            s += 0.03                                    # addresses disclosure/on-sale bars
            if re.search(r"\bgrace period\b", t, re.I):
                s += 0.02                                # invokes the 1-year grace period

        # --- TOOL layer: prior-art disclosure evidence op (largest weight) ----------
        pa = None
        try:
            pa = ops.prior_art(dpid) if dpid else None
        except Exception:
            pa = None
        if pa:
            frac_any = pa.get("frac_claims_any_disclose", 0.0) or 0.0
            frac_max = pa.get("max_frac_disclose", 0.0) or 0.0
            s -= 0.35 * frac_any                         # anticipated claims -> novelty threat
            s -= 0.15 * frac_max                          # worst-case single-claim exposure
            top = pa.get("retrieval_top_scores") or []
            searched_well = bool(top) and (sum(top[:3]) / min(3, len(top))) >= 0.25
            if frac_any == 0.0 and searched_well:
                s += 0.15                                # well-searched, nothing discloses
            elif frac_any == 0.0 and not top:
                s += 0.03                                # sparse retrieval, weak signal either way

        # --- LLM layer: self-identified closest art + distinguishing feature --------
        ca = (extracted or {}).get("closest_art", "").strip()
        di = (extracted or {}).get("distinguishing", "").strip()
        norm_t = re.sub(r"\s+", "", t)
        if ca and ca.upper() != "NONE":
            token = re.sub(r"\s+", "", ca)[:60]
            if token and token in norm_t:
                s += 0.04                                 # grounded acknowledgment of closest art
        if di and di.upper() != "NONE":
            token = re.sub(r"\s+", "", di)[:60]
            if token and token in norm_t:
                s += 0.05                                 # grounded distinguishing feature

        return max(0.0, min(1.0, s))
    except Exception:
        return 0.3
