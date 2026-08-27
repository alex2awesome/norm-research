"""a60 — Prior-art differentiation and argumentation (hybrid, evidence-op enabled).

Construct: does the application draft and ARGUE its distinction from the prior art well —
identify missing claim elements, avoid harmful admissions/disclaimers, mount reasoned
(KSR-resistant) traversals, and articulate distinguishing features in the spec? This is about
the DOCUMENT'S drafting/argumentation quality, never about whether the invention IS novel.
The prior_art evidence op (BM25/dense retrieval, K=8 refs/claim, Gemma disclosure verdicts)
supplies the actual prior-art landscape, so the code layer's differentiation TALK can be
checked against real differentiation SUBSTANCE: claims whose elements no retrieved reference
discloses. Never touches label/grant/judgement fields.
"""
import re

LLM_FIELDS = {
    "distinguishing_feature": "quote the sentence stating the invention's key feature "
                               "distinguishing it from prior art; NONE if absent",
    "admission_line": "quote any phrase admitting a feature is well-known/conventional/prior "
                       "art; NONE if none",
}

_PA_DISCUSS = re.compile(r"\bprior art\b|U\.S\.\s*Pat|\bUS20\d{8}\b|\b\d{7,8}\b", re.I)
_DIFF_LANG = re.compile(r"\b(in contrast|unlike|distinguish(?:es|ed|able)?|does not disclose|"
                        r"fails? to (?:teach|disclose|suggest)|is silent (?:on|as to)|"
                        r"no (?:disclosure|teaching|suggestion) of|absent from)\b", re.I)
_KSR = re.compile(r"\b(KSR|motivation to combine|teaches? away|hindsight|unexpected result|"
                  r"no reasonable expectation of success|obviousness)\b", re.I)
_ADMIT = re.compile(r"\b(well[- ]known|conventional(?:ly)?|admitted prior art|state of the "
                   r"art|commonly (?:used|known)|as is (?:well )?understood)\b", re.I)


def score(text, extracted, ops, dpid=None):
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        n_pa = len(_PA_DISCUSS.findall(t))
        n_diff = len(_DIFF_LANG.findall(t))
        n_ksr = len(_KSR.findall(t))
        n_admit = len(_ADMIT.findall(t))
        discusses_pa = n_pa >= 1
        argues_diff = n_diff >= 1

        # --- CODE layer: prior-art talk, differentiation, KSR-resistance, admissions ---
        s = 0.5
        if discusses_pa:
            s += 0.06                          # engages with prior art at all
        s += min(0.12, 0.04 * n_diff)          # explicit distinguishing language
        s += min(0.1, 0.05 * n_ksr)            # reasoned/KSR-resistant traversal markers
        s -= min(0.15, 0.05 * n_admit)         # harmful "well known"/"conventional" admissions

        # --- TOOL layer: talk vs. substance, via prior-art evidence op -----------------
        pa = None
        try:
            pa = ops.prior_art(dpid) if dpid else None
        except Exception:
            pa = None
        if pa and (pa.get("n_claims") or 0) > 0:
            frac_any = pa.get("frac_claims_any_disclose", 0.0) or 0.0
            mean_frac = pa.get("mean_frac_disclose", 0.0) or 0.0
            if argues_diff and frac_any < 0.5:
                s += 0.15                      # substantive: most claims genuinely undisclosed
            elif argues_diff and mean_frac > 0.7:
                s -= 0.08                      # talk unsupported: refs actually disclose most
            if not discusses_pa and frac_any < 0.3:
                s -= 0.06                      # real gaps exist but doc never argues them
            zero_ref = sum(1 for c in pa.get("claims", []) if (c.get("n_disclose") or 0) == 0)
            if zero_ref and argues_diff:
                s += min(0.08, 0.02 * zero_ref)  # differentiation talk matches clean claims

        # --- LLM layer: grounded distinguishing-feature quote + admission check --------
        df = (extracted or {}).get("distinguishing_feature", "").strip()
        ad = (extracted or {}).get("admission_line", "").strip()
        norm_t = re.sub(r"\s+", "", t.lower())
        if df and df.upper() != "NONE":
            token = re.sub(r"\s+", "", df.lower())[:80]
            if token and token in norm_t:
                s += 0.06                      # concrete, grounded distinguishing claim
        if ad and ad.upper() != "NONE":
            token = re.sub(r"\s+", "", ad.lower())[:80]
            if token and token in norm_t:
                s -= 0.05                      # LLM-caught harmful admission, grounded

        return max(0.0, min(1.0, s))
    except Exception:
        return 0.3
