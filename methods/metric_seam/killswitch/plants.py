"""Planted-criterion registry for the E-S1 end-to-end kill-switch (proposal §E-S1).

Seven plants of KNOWN placement, laundered as ordinary aspect ids p901-p907. Ground truth
lives ONLY in truth.json + DESIGN.md; every downstream agent (codegen, improver) sees the
same pack format as real bank aspects and is never told these are plants.

Truth types and the pipeline verdict each plant must produce for the kill-switch to PASS:
  p901 code           — pure surface program reaches ceiling; S6 verdict CODE, LLM share ~0
  p902 code+comp_op   — codable, date-normalization computation op carries the signal;
                        S5 op ablation positive IF program uses ops; stronger-executor-
                        without-tool CAN do it (computation op, DPI: no new info)
  p903 code+evid_op   — truth = corpus-retrieval quantity; code-only PLATEAUS below ceiling,
                        hybrid WITH retrieve_similar reaches it; without-tool executor CANNOT
                        (evidence op: Z beyond x). Sharpest op-type recovery test.
  p904 mixed          — deterministic predicate over a semantic field (distinct quoted named
                        people, Claude-labeled truth); pure code plateaus, code+LLM-field
                        hybrid reaches ceiling; S6 verdict MIXED
  p905 a_layer        — holistic tacit criterion (Claude-labeled); pipeline must NOT certify
                        CODE placement (false-positive alarm test; directional truth only)
  p906 null           — channel is pure noise behind a plausible description; reliability ~0,
                        ceiling degenerate, nothing certifiable. Any gate pass = alarm.
  p907 code_known_noise — trivially codable truth (length deciles) + noise calibrated to
                        rel1~0.6: validates the S1 attenuation-ceiling FORMULA empirically
                        (perfect program exists; measured rho vs noisy channel ~= ceiling).
"""
import re

# Construct-validity controls (G2, 2026-07-14).
#
# `positive` controls are ordinary satisfied/violated contrasts.  A verifier
# should score text_pos above text_neg.  `negative_proxy_trap` controls hold the
# construct satisfied in BOTH texts while turning a tempting surface proxy on
# only in text_neg.  A verifier "fires" when it scores text_pos above text_neg;
# validity-v2 requires zero such fires.  Keeping these controls in the planted
# registry makes them reusable by the scalar contract harness and by typed
# three-state verifiers.
VALIDITY_CONTROLS = {
    "math.a12.rigor.contextual_equation_use": [
        {
            "control_id": "a12_pminus_definition_similarity",
            "polarity": "negative_proxy_trap",
            "text_pos": "Let B denote the transformed matrix used below.",
            "text_neg": "Let $B = S X S^{-1}$. We therefore use $B = S X S^{-1}$ below.",
            "why": "A repeated definition is rigorous even when its sides are not a symbolic identity.",
            "expected_role": "definition",
        },
        {
            "control_id": "a12_pminus_equation_to_solve",
            "polarity": "negative_proxy_trap",
            "text_pos": "Solve the following linear equation for r.",
            "text_neg": "Solve $2r + 4 = r - 7$ for r.",
            "why": "An equation-to-solve is supposed to have unequal sides before solving.",
            "expected_role": "equation_to_solve",
        },
        {
            "control_id": "a12_pminus_constraint",
            "polarity": "negative_proxy_trap",
            "text_pos": "Assume the point lies on the stated circle.",
            "text_neg": "Assume $a^2 + b^2 = 1$ throughout the proof.",
            "why": "A constraint is not an asserted algebraic identity.",
            "expected_role": "hypothesis",
        },
        {
            "control_id": "a12_pminus_conservation_definition",
            "polarity": "negative_proxy_trap",
            "text_pos": "Define the total population as the sum of its groups.",
            "text_neg": "Define $N_b + N_s = N$.",
            "why": "A definitional relation can introduce a symbol without being a tautology.",
            "expected_role": "definition",
        },
        {
            "control_id": "a12_pplus_false_identity",
            "polarity": "positive",
            "text_pos": "Expanding gives $(x + 1)^2 = x^2 + 2x + 1$.",
            "text_neg": "Expanding gives $(x + 1)^2 = x^2 + 1$.",
            "why": "The second asserted algebraic step drops the cross term.",
            "expected_role": "asserted_identity_step",
        },
        {
            "control_id": "a12_pplus_arithmetic_error",
            "polarity": "positive",
            "text_pos": "Thus $4 + 3 = 7$.",
            "text_neg": "Thus $4 + 3 = 8$.",
            "why": "The second asserted arithmetic step is false.",
            "expected_role": "asserted_identity_step",
        },
    ],
}


def validity_controls(unit_id):
    """Return defensive copies of the canonical G2 controls for a unit."""
    return [dict(row) for row in VALIDITY_CONTROLS.get(unit_id, ())]

# rel1 targets for the synthetic 2-pass channels
REL_TARGET = {"p901": 0.85, "p902": 0.85, "p903": 0.85, "p904": 0.85,
              "p905": 0.80, "p906": None, "p907": 0.60}

TRUTH_TYPE = {"p901": "code", "p902": "code+comp_op", "p903": "code+evidence_op",
              "p904": "mixed_llm_field", "p905": "a_layer", "p906": "null",
              "p907": "code_known_noise"}

PLANTS = [
    {"aspect_id": "p901", "name": "Quantitative support",
     "description": ("The release backs its claims with concrete figures — dollar amounts, "
                     "percentages, counts, dates-in-numbers, measurements — rather than "
                     "purely qualitative language. Density matters: a release whose text is "
                     "rich in specific numbers scores high; one that makes only qualitative "
                     "claims scores low.")},
    {"aspect_id": "p902", "name": "Temporal anchoring",
     "description": ("The release anchors its claims to specific calendar dates — event "
                     "dates, availability dates, deadlines, fiscal periods. The more "
                     "distinct explicit dates the document commits to, the stronger the "
                     "anchoring; a release with no concrete dates scores low.")},
    {"aspect_id": "p903", "name": "Corpus distinctiveness",
     "description": ("The release's content is distinctive relative to other releases in "
                     "this collection: it is not near-duplicate template or boilerplate "
                     "text recycled across many similar announcements. Releases that read "
                     "like many others in the corpus score low; one-of-a-kind content "
                     "scores high.")},
    {"aspect_id": "p904", "name": "Voice diversity",
     "description": ("The release features direct quotations from multiple distinct named "
                     "people — for example an executive plus a customer, partner, or "
                     "official — with each quote clearly attributed. A single spokesperson "
                     "(or no quotes at all) scores low; three or more distinct quoted "
                     "voices scores high.")},
    {"aspect_id": "p905", "name": "Authentic authorship",
     "description": ("The release reads as if written by someone with genuine familiarity "
                     "with the company and its subject matter — concrete, specific, and "
                     "internally coherent — rather than assembled from generic template "
                     "marketing language that could describe any company.")},
    {"aspect_id": "p906", "name": "Persuasive cadence",
     "description": ("The prose rhythm builds momentum across paragraphs: sentence-length "
                     "variation, paragraph pacing, and transitions sustain the reader's "
                     "attention from the headline through the closing boilerplate.")},
    {"aspect_id": "p907", "name": "Comprehensive detail",
     "description": ("The release provides substantial informative detail: background, "
                     "specifics, supporting facts, and context. Fuller, more complete "
                     "releases score higher; terse stub announcements score low.")},
]

_NUM = re.compile(r"(?<![\w.])(?:\$\s?)?\d[\d,]*(?:\.\d+)?%?")

_MONTH = ("jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec")
_MNUM = {m: i % 12 + 1 for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split())}
_D_MDY = re.compile(
    rf"\b({_MONTH})[a-z]*\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,\s*(\d{{4}}))?",
    re.IGNORECASE)
_D_DMY = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})[a-z]*\.?(?:\s+(\d{{4}}))?",
    re.IGNORECASE)
_D_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_D_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def truth_p901_raw(ctext):
    """Numeric-token density per 1000 words (decile-ranked into 0-10 by the builder).
    Near-independent of document length (rho=-0.20 on v1 items), unlike raw count."""
    words = len(re.findall(r"[A-Za-z']+", ctext))
    return 1000 * len(_NUM.findall(ctext)) / max(words, 1)


def truth_p902(ctext):
    """Distinct day-precision calendar dates (many surface formats), min(n,5)*2."""
    seen = set()
    for m in _D_MDY.finditer(ctext):
        mon = _MNUM[m.group(1).lower()[:3]]
        seen.add((m.group(3) or "?", mon, int(m.group(2))))
    for m in _D_DMY.finditer(ctext):
        mon = _MNUM[m.group(2).lower()[:3]]
        seen.add((m.group(3) or "?", mon, int(m.group(1))))
    for m in _D_SLASH.finditer(ctext):
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mo <= 12 and 1 <= d <= 31:
            y = ("20" + y if len(y) == 2 else y)
            seen.add((y, mo, d))
    for m in _D_ISO.finditer(ctext):
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            seen.add((y, mo, d))
    # merge year-unknown duplicates of known-year dates
    known = {(mo, d) for (y, mo, d) in seen if y != "?"}
    seen = {(y, mo, d) for (y, mo, d) in seen if not (y == "?" and (mo, d) in known)}
    return min(len(seen), 5) * 2


def truth_p907_raw(ctext):
    """Word count (decile-ranked into 0-10 by the builder)."""
    return len(re.findall(r"[A-Za-z']+", ctext))


def map_p904(n_speakers):
    return {0: 0, 1: 4, 2: 7}.get(n_speakers, 10)
