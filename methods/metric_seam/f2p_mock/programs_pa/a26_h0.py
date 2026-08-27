"""a26 -- Non-obviousness / inventive step, substantive requirement (hybrid, prior-art
evidence-op enabled).

Construct: would the claim have been obvious to a POSITA over the prior art? Obviousness
(35 U.S.C. 103) is a COMBINATION question, so the prior_art op is the natural anchor: per
claim, n_disclose counts how many of the K retrieved refs independently teach it. 0 refs is
affirmative inventive distance; exactly 1 is mainly a novelty/anticipation signature (a26's
sibling criterion) so it is down-weighted; >=2 is the textbook motivation-to-combine fact
pattern, so it dominates the negative side. mean_frac_disclose adds a coarse "how much of the
claim set is individually known" prior; weak retrieval_top_scores argue the field is sparse
(harder to build a combination case). Text signals (unexpected-results framing, independent-
vs-dependent claim breadth) modulate lightly. The LLM layer extracts the document's own
asserted advantage and any combination-synergy statement, grounded by containment in the
text and only weighted when the tool layer does not already show high combination risk.
"""
import re

LLM_FIELDS = {
    "asserted_advantage": "in at most 15 words, what specific technical advantage or "
                          "unexpected result does the document assert over prior art; "
                          "answer NONE if absent",
    "synergy_claim": "in at most 15 words, what synergistic or unexpected effect does the "
                     "document assert from combining the claimed elements; answer NONE if absent",
}

_ADV_KW = re.compile(r"\b(unexpected(ly)?|surprisingly|superior to|significant(ly)? "
                     r"(improv\w*|better|higher|reduc\w*)|synergistic|synergy|"
                     r"criticality|unpredictab\w*)\b", re.I)
_CLAIMS_SEC = re.compile(r"\bCLAIMS?:?\s*(.*?)(?=\n[A-Z][A-Z /]{4,}:|\Z)", re.S)
_CLAIM_HEAD = re.compile(r"^\s*(\d+)\s*\.\s+", re.M)
_DEP_REF = re.compile(r"\bclaim\s+\d+\b", re.I)
_CANCELED = re.compile(r"\bcanceled\b", re.I)


def score(text, extracted, ops, dpid=None):
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        head = t[:4000]
        s = 0.5

        # --- CODE layer: weak secondary-consideration framing + claim breadth -------
        n_adv = len(_ADV_KW.findall(head))
        s += min(0.06, 0.03 * n_adv)                     # weak proxy: unexpected-results framing

        m = _CLAIMS_SEC.search(t)
        if m:
            body = m.group(1)
            starts = [x.start() for x in _CLAIM_HEAD.finditer(body)] + [len(body)]
            segs = [body[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
            segs = [seg for seg in segs if not _CANCELED.search(seg[:40])]
            n_dep = sum(1 for seg in segs if _DEP_REF.search(seg))
            if segs and n_dep == 0:
                s -= 0.04                                 # only broad claims, no narrowing fallback

        # --- TOOL layer: prior-art disclosure evidence op (largest weight) ----------
        pa = None
        try:
            pa = ops.prior_art(dpid) if dpid else None
        except Exception:
            pa = None
        if pa:
            s -= 0.20 * (pa.get("mean_frac_disclose", 0.0) or 0.0)  # elements known -> more obvious

            claims = pa.get("claims") or []
            if claims:
                def n_disc(c):
                    return c.get("n_disclose", sum(1 for r in (c.get("refs") or []) if r.get("discloses")))
                # 0 refs=inventive distance, 1 ref=novelty issue (down-weight), 2+=combination risk
                risk = [0.0 if not n else 0.30 if n == 1 else 1.0 for n in map(n_disc, claims)]
                s -= 0.25 * (sum(risk) / len(risk))
                s += 0.10 * (sum(1 for r in risk if r == 0.0) / len(claims))
            else:
                s -= 0.15 * (pa.get("frac_claims_any_disclose", 0.0) or 0.0)  # degrade gracefully

            top = pa.get("retrieval_top_scores") or []
            if top:
                avg_top = sum(top[:5]) / min(5, len(top))
                s += 0.06 if avg_top < 0.35 else -0.04 if avg_top > 0.75 else 0.0

        # --- LLM layer: self-asserted advantage / synergy, grounded ------------------
        adv = (extracted or {}).get("asserted_advantage", "").strip()
        syn = (extracted or {}).get("synergy_claim", "").strip()
        norm_t = re.sub(r"\s+", "", t)
        combo_risk_high = bool(pa and pa.get("claims") and
                                (sum(1 for c in pa["claims"]
                                     if (c.get("n_disclose") or 0) >= 2) / len(pa["claims"])) > 0.5)
        if adv and adv.upper() != "NONE":
            token = re.sub(r"\s+", "", adv)[:80]
            if token and token in norm_t and not combo_risk_high:
                s += 0.05                                  # grounded advantage, not contradicted
        if syn and syn.upper() != "NONE":
            token = re.sub(r"\s+", "", syn)[:80]
            if token and token in norm_t and not combo_risk_high:
                s += 0.04                                  # grounded combination-synergy argument

        return max(0.0, min(1.0, s))
    except Exception:
        return 0.3
