"""a35 -- Core patentability requirements: novelty, inventive step, industrial applicability
(hybrid, prior-art evidence-op enabled; TRIPS-consistent triad).

Construct: this criterion asks all three patentability pillars at once, so the score BLENDS
three sub-scores rather than owning any one deeply (contrast the dedicated a34/a26 siblings).
Novelty sub-score (tool-driven, largest weight): the prior_art evidence op's per-application
frac_claims_any_disclose / max_frac_disclose are the anticipation-exposure signature -- >=1
disclosing reference on a claim is a single-reference novelty threat, so higher values drive
the score down; a well-searched claim set (decent retrieval_top_scores) with ZERO disclosures
is affirmative novelty evidence and drives it up. Inventive-step sub-score (tool-driven,
second-largest weight): mean_frac_disclose is a coarse "how much of the claim set is
individually known" prior across refs, and the spread of n_disclose across claims flags
uneven, piecemeal coverage -- elements individually scattered across multiple references is
the textbook combination-obviousness fact pattern -- so both push the score down. Industrial-
applicability sub-score (code+LLM, smallest weight, since the tool layers are the experiment
here): concrete practical-use / apparatus-method framing and technical-vocabulary density are
weak text proxies; the LLM field extracts the document's own stated practical application,
validated by containment in the normalized text before it earns any weight.
"""
import re

LLM_FIELDS = {
    "practical_use": "in at most 15 words, what concrete practical or industrial application "
                     "does the document state for the invention; answer NONE if absent",
}

_NOVEL_KW = re.compile(r"\b(novel|new(ly)?|unlike (the )?prior art|not (been )?disclosed)\b",
                      re.I)
_INDUSTRIAL_KW = re.compile(
    r"\b(apparatus|method for|device for|system for|process for|manufactur\w*|industrial(ly)?|"
    r"commercial(ly)?|utility|may be used (to|in|for)|used (in|for) (the )?(manufacture|"
    r"production|treatment)|automotive|pharmaceutical|semiconductor|agricultural|aerospace|"
    r"telecommunications)\b", re.I)


def score(text, extracted, ops, dpid=None):
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        head = t[:4000]
        s = 0.5

        # --- TOOL layer: prior-art disclosure evidence op (largest combined weight) --
        pa = None
        try:
            pa = ops.prior_art(dpid) if dpid else None
        except Exception:
            pa = None

        if pa:
            frac_any = pa.get("frac_claims_any_disclose", 0.0) or 0.0
            frac_max = pa.get("max_frac_disclose", 0.0) or 0.0
            mean_fd = pa.get("mean_frac_disclose", 0.0) or 0.0

            # novelty: anticipation exposure vs. clean, well-searched claim set
            s -= 0.22 * frac_any                          # any disclosure across claims
            s -= 0.08 * frac_max                          # worst single-claim exposure
            top = pa.get("retrieval_top_scores") or []
            searched_well = bool(top) and (sum(top[:3]) / min(3, len(top))) >= 0.25
            if frac_any == 0.0 and searched_well:
                s += 0.12                                  # searched hard, nothing anticipates

            # inventive step: breadth/spread of disclosure = combination-obviousness exposure
            claims = pa.get("claims") or []
            ndis = [c.get("n_disclose", 0) or 0 for c in claims]
            spread = (max(ndis) - min(ndis)) if len(ndis) >= 2 else 0
            s -= 0.16 * mean_fd                           # elements broadly known piecemeal
            if spread >= 3 and mean_fd > 0:
                s -= 0.05                                  # uneven coverage across the claim set
            elif mean_fd == 0.0 and frac_any == 0.0:
                s += 0.05                                  # nothing found anywhere in the K refs
        else:
            if _NOVEL_KW.search(head):
                s += 0.03                                  # text-only fallback, weak proxy

        # --- CODE layer: industrial-applicability language --------------------------
        n_ind = len(_INDUSTRIAL_KW.findall(t))
        s += min(0.08, 0.02 * n_ind)
        _, _, frac_long = ops.sent_stats(head)
        if frac_long > 0.12:
            s += 0.02                                      # technical/industrial vocabulary density

        # --- LLM layer: stated practical application, grounded ----------------------
        pu = (extracted or {}).get("practical_use", "").strip()
        if pu and pu.upper() != "NONE":
            token = re.sub(r"\s+", "", pu)[:60]
            norm_t = re.sub(r"\s+", "", t)
            if token and token in norm_t:
                s += 0.05                                  # grounded, self-stated industrial use

        return max(0.0, min(1.0, s))
    except Exception:
        return 0.3
