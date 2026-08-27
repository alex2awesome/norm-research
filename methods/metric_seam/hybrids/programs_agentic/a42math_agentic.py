"""a42 - Use of visuals and concrete examples (Math StackExchange).

Best TRAIN rho 0.6827 (h0 reference 0.4185), reached over 4 agentic rounds
against a42_h0 (see agentic_run.py output history):

ROUND 1 -- 0.4185 -> 0.5955. Two changes, both driven by residual diagnosis
on h0 itself:
  (a) TIER TABLE now keys off BOTH LLM fields jointly, not richness-only-
      when-grounds==YES. h0 collapses every "grounds_with_example: NO" doc
      straight to the bottom tier regardless of what example_richness said.
      Checked directly against train judge scores: the two fields are NOT
      redundant. Even when the extractor answered NO to the strict
      grounding question, an example_richness of ONE/MULTIPLE still
      predicts a much higher judge mean than richness NONE/empty (0.083
      over n=15 vs 0.026 over n=110) -- the richness question apparently
      catches illustrative material (a named instance, a small case) that
      the stricter "does this really ground a general argument" question
      rejects. That statistic is now its own tier instead of being thrown
      away.
  (b) DROPPED h0's "numeric-density" code-side proxy (fraction of parsed
      math spans containing any digit). On this corpus almost every math
      answer has *some* digit somewhere in its formulas (indices, small
      constants, exponents), so that term saturated near its cap for most
      of the 110 "no grounding, no richness" documents regardless of the
      judge's actual verdict -- exactly why h0's worst residuals were
      judge=0.0 docs ranked at the 70-90th percentile (routine algebra, not
      exemplification, pushed them up).

ROUND 2 -- 0.5955 -> 0.6096. Added a numeral-presence GATE inside the
  single ambiguous cell (grounds=NO, richness=ONE): the richness field's
  "ONE" is unreliable there since the stricter grounds question already
  rejected the doc. Checked on train: when the claimed example has NO
  concrete numeral anywhere in the answer (an abstract named-object mention
  like "such as the vector space of ..." or "consider X=R"), judge is 0.0
  in every train case (3/3); when >=1 numeral is present, judge is nonzero
  in 7/8 cases. Numeral-free docs in this cell are demoted to the bottom
  tier instead of credited as a genuine (if softer) example.

ROUND 3 -- 0.6096 -> 0.6110. Recalibrated the (grounds=NO, richness=MULTI)
  base down (0.16 -> 0.10): train mean there (n=4) sits BELOW the
  numeral-gated (grounds=NO, richness=ONE) mean, i.e. an extractor claim of
  "MULTIPLE" examples is, if anything, LESS reliable than "ONE" once
  grounds_with_example has already said NO -- plausibly because multi-part
  or multi-case PROOFS (not multi-example groundings) trip the richness
  question into overcounting. Small-n, flagged as thin evidence below.

ROUND 4 -- 0.6110 -> 0.6827 (biggest single gain). REMOVED h0's leftover
  code-side "richness proxy" (marker-phrase / case-split / length nudge)
  entirely instead of merely shrinking it. Direct ablation on train: ANY
  additive nudge inside the bottom tier -- even capped at 0.02-0.05, even
  using a length/inline-equation substance proxy that looks weakly
  correlated in isolation (Spearman ~0.25-0.27 within that subgroup alone)
  -- nets WORSE full-corpus rho than a flat tier value (0.61 vs 0.68). The
  110-item "no grounding, no richness" bulk is overwhelmingly tied at
  judge=0.0; any within-tier nudge just reshuffles that tied mass in ways
  uncorrelated with the few true nonzero docs, which costs more rank
  agreement elsewhere than it buys. Confirmed by direct swap-back: removing
  the nudge and leaving the tier flat is what took rho from 0.611 to 0.683.

LLM_FIELDS is UNCHANGED from h0 (same two fields, same instructions;
field-budget frozen). No new code ops beyond a plain numeral regex (used
only as an on/off gate, per the "keyword regex may gate but not be the
signal" convention) and the existing ops.delimiter_health malformed-LaTeX
penalty (kept from h0, contributes negligibly on train but is principled --
the criterion explicitly asks that visual/example content be
"well-integrated").
"""
import re

LLM_FIELDS = {
    "grounds_with_example": (
        "Does the answer use a concrete numeric example, specific named object, "
        "or small worked toy case to illustrate a general idea -- NOT merely "
        "computing the specific numbers already given in the question? "
        "Answer YES or NO."
    ),
    "example_richness": (
        "How many separate concrete illustrative examples or toy cases does "
        "the answer give to ground its reasoning: NONE, ONE, or MULTIPLE?"
    ),
}


def _fields(extracted):
    """(has_grounding_YES, richness_level in {0,1,2}) read from BOTH fields."""
    g = (extracted.get("grounds_with_example") or "").strip().upper()
    r = (extracted.get("example_richness") or "").strip().upper()
    has = g.startswith("Y")
    if "MULTI" in r or "MANY" in r or "SEVERAL" in r:
        rich = 2
    elif "ONE" in r:
        rich = 1
    else:
        rich = 0
    return has, rich


# base score per (grounds_with_example==YES, richness tier) cell. The four
# cells that actually occur in-corpus are calibrated to the monotone ladder
# observed in train judge means (NO+none < NO+some < YES+one < YES+multi);
# the two cells with no in-corpus support ((True,0) i.e. YES but richness
# came back empty) get a conservative fallback between their neighbors so
# the table stays well-defined without being fit to zero evidence.
_TIER_BASE = {
    (False, 0): 0.03,
    (False, 1): 0.14,
    (False, 2): 0.10,
    (True, 0): 0.20,
    (True, 1): 0.32,
    (True, 2): 0.55,
}


def _answer_text(t):
    return t.split("Answer:", 1)[1] if "Answer:" in t else t


_STANDALONE_NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def _has_numeral(ans):
    """Any standalone number token in the answer (not a subscript/identifier
    letter run). Used only as an on/off GATE inside one specific ambiguous
    cell below -- never as the primary richness signal."""
    return bool(_STANDALONE_NUM_RE.search(ans))


def _malformed_penalty(text, ops):
    """Small penalty for badly broken LaTeX -- an example that isn't
    'well-integrated' (contract's phrase) shouldn't get full credit."""
    try:
        dh = ops.delimiter_health(text) or {}
    except Exception:
        return 0.0
    issues = 0
    for v in dh.values():
        try:
            if int(v) > 0:
                issues += 1
        except Exception:
            continue
    if issues >= 4:
        return 0.08
    if issues >= 2:
        return 0.04
    return 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if len(t) < 30:
            return 0.0

        extracted = extracted or {}
        has, rich = _fields(extracted)
        base = _TIER_BASE.get((has, rich), 0.20)

        if not has and rich == 1 and not _has_numeral(_answer_text(t)):
            # ROUND 2 gate: within the (grounds=NO, richness=ONE) cell, the
            # richness field's "ONE" is unreliable on its own -- the
            # stricter grounds question already rejected the doc. See
            # module docstring for the train evidence. Numeral-free docs in
            # this cell are demoted to the bottom tier instead of credited
            # as a genuine (if softer) example.
            base = _TIER_BASE[(False, 0)]

        penalty = _malformed_penalty(t, ops)

        s = base - penalty
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
