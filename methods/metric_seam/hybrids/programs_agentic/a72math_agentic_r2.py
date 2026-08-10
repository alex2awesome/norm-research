"""a72 'Appropriate and economical proof technique' -- AGENTIC rescue (round 2).

h0 TRAIN rho was 0.4806 (baseline code rho 0.09). Residual inspection against
the live judge scores (agentic_run.py worst-residuals + a full train
groupby(technique_label, completeness_label) -> judge_score table) found ONE
dominant, systematic miscalibration in h0's scoring formula, not a missing
signal:

  h0 discounts "partial" completeness to 0.30x the technique anchor whenever
  the technique label is good (appropriate/elegant). But the judge does NOT
  do this. Empirical TRAIN means (n=139, all real extracted labels -- the
  completeness field only ever returns "complete"/"partial"/"" verbatim, it
  never says "none"/"no proof" in practice, so h0's "none" completeness
  branch is dead code on this data):

      technique     completeness   n    judge mean   judge median
      appropriate   complete       32   0.861        1.00
      appropriate   partial        21   0.879        1.00   <- HIGHER than complete
      elegant       complete       32   0.948        1.00
      elegant       partial        14   0.907         1.00 (mult mostly 1.0)
      overkill      complete        9   0.444        0.35
      overkill      partial         6   0.308        0.28   <- real gap here
      tedious       complete        4   0.387        0.35
      tedious       partial         5   0.300        0.30   <- real gap here
      vacuous       partial         3   0.483        0.35
      mismatched    partial         3   0.167        0.20
      appropriate   "" (missing)    5   0.530        0.30   (bimodal: 1,1,.3,.2,.15)

  i.e. this criterion cares about the METHOD being right, not about whether
  the write-up is a fully worked-out completion vs. a correct hint/sketch --
  a well-aimed hint is judged as economical/appropriate as a full write-up.
  The completeness signal only matters as a real discount when the TECHNIQUE
  is already bad (overkill/tedious): finishing a clumsy computation scores a
  bit higher than an unfinished clumsy computation, but both are low. This
  matches the criterion definition ("appropriate ... proof TECHNIQUE") more
  than h0's design, which conflated "technique quality" with "completeness"
  as if they were substitutable.

  Fix: per-technique anchors recalibrated near the measured group means
  (elegant .95, appropriate .85, overkill .40, tedious .33, vacuous .25,
  mismatched .13), with a completeness MULTIPLIER that is near-1.0 (barely
  discounts) when technique is good, and a real ~0.72-0.85x discount when
  technique is bad/absent -- the opposite emphasis from h0's fixed 0.30x-for-
  partial rule, which was backwards for the largest (good-technique) mass of
  the distribution.

  A code-only safety net (near-empty math-free non-answer override, and a
  penalty for either heavy casework or dense unexplained symbol-pushing) is
  kept from h0 largely as-is since it targets failure modes the two LLM
  fields under-detect (e.g. genuinely math-free non-answers where both
  fields come back empty).

Round 2 (this file, cumulative over rounds): after the anchor/multiplier
rewrite above (TRAIN rho 0.4806 -> 0.6182), two more rounds of residual
inspection found:

  (r2) Even WITHIN the "elegant"/"appropriate" LLM technique bucket, TRAIN
  judge score has a real negative correlation with display-equation count
  and explicit qed/"as desired"-style closing-phrase count (n=104, rho
  -0.23 / -0.27) -- consistent with the criterion text itself ("avoid ...
  tedious computation"). Added a small bounded per-span/per-marker nudge on
  top of the technique/completeness anchor for good-technique items only
  (0.6182 -> 0.6771).

  (r3) Replaced the anchor x completeness-multiplier formula with a direct
  (technique, completeness) lookup fit to the measured TRAIN group means
  for cells with enough support (n>=3), still using ONLY the two existing
  LLM fields as the lookup key. The "vacuous"/"mismatched" x partial cells
  are pulled toward the group MEDIAN rather than mean, since their means
  are each skewed by one confirmed LLM-mislabeled item (d00122: a
  genuinely elegant, appropriate answer -- concise, correct, ends "more or
  less by definition" -- that the extractor tagged "vacuous"; checking all
  7 other vacuous/mismatched-labeled TRAIN items shows NO general
  content-vs-label mismatch pattern, i.e. this is isolated per-item
  extractor noise, not a fixable systematic gap, so it is left as residual
  rather than papered over with a corpus-wide override) (0.6771 -> 0.6974).
  Final TRAIN rho: 0.6974 (h0 reference 0.4806; code baseline_train_rho 0.09).

LLM_FIELDS are UNCHANGED from h0 across all rounds (technique_label,
completeness_label) -- same two fields, same field budget, reused for both
TRAIN signal (via f_orig) and post-hoc extraction. No new field names
declared.
"""

import re

LLM_FIELDS = {
    "technique_label": (
        "Read the ANSWER only (ignore the question). In one word, "
        "describe its proof technique: elegant, appropriate, overkill, "
        "tedious, mismatched, or vacuous."
    ),
    "completeness_label": (
        "Read the ANSWER only. In one word: complete, partial, or none -- "
        "does it carry out a full proof, a partial/hint-only argument, "
        "or no proof at all?"
    ),
}

_TECH_BUCKETS = [
    ("vacuous", ("vacuous", "no proof", "not a proof", "spam", "nonsense",
                 "irrelevant answer", "non-answer", "nonanswer",
                 "empty answer")),
    ("mismatched", ("mismatch", "off-topic", "off topic", "unrelated",
                    "wrong problem", "doesn't address", "does not address",
                    "irrelevant")),
    ("tedious", ("tedious", "heavy computation", "brute force", "laborious",
                 "excessive computation", "symbol-pushing",
                 "symbol pushing")),
    ("overkill", ("overkill", "unnecessar", "convoluted", "clumsy",
                  "gratuitous", "overcomplicat", "too complex",
                  "heavy machinery")),
    ("appropriate", ("appropriate", "natural", "suited", "fitting",
                      "reasonable", "proper", "well-matched",
                      "well matched")),
    ("elegant", ("elegant", "simple", "clean", "direct", "concise",
                 "economical", "straightforward", "trivial", "minimal",
                 "short proof")),
]

_COMPLETE_BUCKETS = [
    ("none", ("none", "no proof", "not a proof", "empty", "nothing")),
    ("partial", ("partial", "hint", "outline", "sketch", "incomplete")),
    ("complete", ("complete", "full", "entire", "finished", "yes")),
]

# per-technique anchors, recalibrated to measured TRAIN group means (see
# docstring table). Ordered worst -> best for readability only.
_TECH_ANCHOR = {
    "mismatched": 0.16,
    "vacuous": 0.24,
    "tedious": 0.36,
    "overkill": 0.40,
    "appropriate": 0.86,
    "elegant": 0.95,
}
_GOOD_TECH = {"appropriate", "elegant"}

# direct (technique, completeness) anchors fit to measured TRAIN group means
# where support was decent (n>=3); "vacuous"/"mismatched" partial cells are
# pulled toward the group MEDIAN rather than mean -- their means are each
# skewed by one confirmed LLM-mislabeled item (a genuinely elegant answer
# the extractor called "vacuous"), so trusting the mean would overfit to a
# single known-noisy point. Falls back to _TECH_ANCHOR x a completeness
# multiplier for any (tech, comp) combination without enough TRAIN support.
_TECH_COMP_ANCHOR = {
    ("elegant", "complete"): 0.95,
    ("elegant", "partial"): 0.91,
    ("appropriate", "complete"): 0.86,
    ("appropriate", "partial"): 0.87,
    ("overkill", "complete"): 0.42,
    ("overkill", "partial"): 0.31,
    ("tedious", "complete"): 0.38,
    ("tedious", "partial"): 0.30,
    ("vacuous", "partial"): 0.22,
    ("mismatched", "partial"): 0.18,
}

# completeness multipliers used only as a FALLBACK when the (tech, comp)
# pair isn't in the direct table above: near-1.0 for good technique (the
# judge does not meaningfully discount a correct-approach hint/sketch), a
# real discount for bad/absent technique.
_COMP_MULT_GOOD = {"complete": 1.00, "partial": 0.97, "none": 0.35, None: 0.65}
_COMP_MULT_BAD = {"complete": 1.00, "partial": 0.75, "none": 0.60, None: 0.85}


def _classify(s, buckets, default=None):
    s = (s or "").strip().lower()
    if not s:
        return default
    for label, keywords in buckets:
        for kw in keywords:
            if kw in s:
                return label
    return default


def _split_answer(text):
    # Corpus format is consistently "Question: ...\n\nAnswer: ...".
    # Score the ANSWER's technique, not the (sometimes casework-heavy,
    # sometimes off-topic) question the asker wrote.
    m = re.search(r"\bAnswer\s*:\s*", text, flags=re.IGNORECASE)
    if m:
        return text[m.end():]
    return text


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = text or ""
        if len(t.strip()) < 10:
            return 0.0
        try:
            t = ops.normalize(t)
        except Exception:
            pass
        answer = _split_answer(t)
        if len(answer.strip()) < 5:
            answer = t

        ext = extracted or {}
        tech = _classify(ext.get("technique_label", ""), _TECH_BUCKETS)
        comp = _classify(ext.get("completeness_label", ""), _COMPLETE_BUCKETS)

        if tech is None:
            # No usable technique signal -> fall back to completeness alone
            # (weak evidence on its own, kept close to neutral).
            if comp == "complete":
                llm_score = 0.55
            elif comp == "partial":
                llm_score = 0.40
            elif comp == "none":
                llm_score = 0.15
            else:
                llm_score = 0.5
        elif (tech, comp) in _TECH_COMP_ANCHOR:
            llm_score = _TECH_COMP_ANCHOR[(tech, comp)]
        else:
            anchor = _TECH_ANCHOR[tech]
            mult_table = _COMP_MULT_GOOD if tech in _GOOD_TECH else _COMP_MULT_BAD
            mult = mult_table.get(comp, mult_table[None])
            llm_score = anchor * mult

        # --- code safety-net nudges (small, bounded; never the primary
        # signal) ---
        nudge = 0.0
        try:
            n_words = len(answer.split())
        except Exception:
            n_words = 0
        try:
            n_disp, n_inl, _n_num, avg_tok = ops.equation_stats(answer)
        except Exception:
            n_disp = n_inl = 0
            avg_tok = 0.0
        n_spans = (n_disp or 0) + (n_inl or 0)

        if n_words < 15 and n_spans == 0:
            # Near-empty, math-free "answer" -> almost certainly a
            # non-answer regardless of what the LLM field returned.
            nudge -= 0.35
        try:
            skeleton = ops.proof_skeleton(answer)
            n_cases = len(skeleton.get("case", []))
            n_qed = len(skeleton.get("qed", []))
        except Exception:
            n_cases = 0
            n_qed = 0
        if n_cases > 4:
            nudge -= 0.05
        if n_spans > 10 and (avg_tok or 0) > 20:
            nudge -= 0.05

        # Verbosity/machinery safety net: even WITHIN the LLM's
        # appropriate/elegant technique bucket, TRAIN data shows a real,
        # non-trivial negative correlation between judge score and
        # display-equation count / explicit "qed"-style closing-phrase
        # count (n=104, rho -0.23 and -0.27 resp.) -- a chain of many
        # displayed derivation steps or belabored "as desired"/"which was
        # to be shown" closers reads as more tedious/machinery-heavy even
        # when the LLM calls the overall technique appropriate. Bounded
        # and code-only; does not touch the technique/completeness anchor
        # itself, only nudges around it.
        if tech in _GOOD_TECH:
            nudge -= 0.018 * min(6, n_disp)
            nudge -= 0.02 * min(4, n_qed)

        return max(0.0, min(1.0, llm_score + nudge))
    except Exception:
        return 0.5
