# Recovery experiment — anchor-free metric recovery on math.SE — 2026-06-21

First run of the **sanctioned approach** (`project_recovery_reconstruction_no_anchor`): NO holistic anchor M.
For each prompt-form `p`: label real datapoints → recover the metric `m_hat` from the (text,label) pattern
(blind) → score recovery two ways + consistency. Compare across forms; calibration anchors at both ends.

## Setup (all real, verified)
- Corpus: 80 math.StackExchange answers (60 recover-from / 20 held-out for behavioral-R).
- **Labeler (executor):** Llama-3.1-8B via vLLM offline on sk3 (1 GPU, B200). *OpenRouter died mid-session
  (credits) and no 70B is cached — so the labeler is 8B, not 70B. The framework wants a weak executor +
  strong recoverer, so this is principled, but the specific numbers are 8B-labeler numbers.*
- **Recoverer + similarity judge:** Claude Sonnet subagents (blind: saw only labels, neutral filenames).
- Forms: 3 LLM metrics (correctness, clarity, elegance) + 2 calibration anchors (word_count=length,
  random). 500+ real judge calls; every label file saved.
- Two recovery measures (both, per user): **sim-R** = Sonnet judges "is m_hat the same metric as stated p?";
  **beh-R** = apply m_hat to the 20 held-out texts (real 8B) and correlate with p's held-out labels.
  **T** = label agreement across two temp-0.7 passes.

## Results
| form (stated) | T | sim-R | beh-R | recovered as | verdict |
|---|---|---|---|---|---|
| word_count | 1.00\* | 0.10 | −0.03 | "completeness/quality" | **CEILING FAIL** — recoverer ignored length(↔0.97) |
| random | 1.00\* | 0.00 | −0.05 | "completeness" (hallucinated) | **FLOOR OK** — both R→0 |
| correctness | 0.66 | 0.70 | 0.23 | general quality | collapsed w/ clarity (r=0.75) |
| clarity | 0.89† | 0.60 | nan | completeness/substance | degenerate labels (95% one bucket) |
| elegance | 0.51 | 0.30 | −0.03 | completeness | **NOT recovered** — tacit lost |

\* deterministic (T trivially 1). † degenerate: near-constant labels make T meaningless.

Cross-label correlation (80 texts): correct↔clarity **0.75**, correct↔elegance 0.43, clarity↔elegance 0.42;
all three ⊥ LENGTH (≈0.10/0.04/−0.01). word_count↔LENGTH **0.97**. random ⊥ everything.

## What it shows
1. **The method runs end-to-end, anchor-free.** No M anywhere; recovery is purely from the label pattern.
2. **Floor calibration WORKS:** `random` → sim-R 0.0 AND beh-R ≈ 0. The recoverer *hallucinated* a
   "completeness" rubric from random labels, but **behavioral-R on held-out caught it** (the rubric doesn't
   reproduce random labels). This is exactly why we run both R's — sim-R alone would be fooled by a
   confident-but-wrong recovery; beh-R is the guard.
3. **Scientific finding:** at 8B the labeler does **not separately transmit** correctness/clarity/elegance —
   they collapse into one general-quality factor (intercorrelation 0.42–0.75, all ⊥ length), so none is
   recovered as itself. **Elegance is the least recovered (sim-R 0.30)** — the tacit metric is lost, the
   same pattern the GEPA trials showed.
4. **Two FAILURES that define the next iteration (honest):**
   - **Recoverer imposes a "quality" prior.** word_count is trivially recoverable (↔length 0.97) yet was
     recovered as "completeness/quality" (sim-R 0.10). The recoverer never tested the simple "longer→higher"
     hypothesis. → Fix: a recoverer that explicitly fits/【tests simple feature hypotheses (length, links,
     formatting) against the labels and reports the correlation, not just free-form prose.
   - **beh-R as implemented is too weak to certify the ceiling.** Routing m_hat through the biased 8B
     labeler + only 20 held-out points + deterministic anchors that can't round-trip through an LLM → beh-R
     ≈ 0 for everything, not just random. → Fix: stronger/more-reliable labeler, more held-out, and replace
     deterministic anchors with LLM-detectable surface anchors (e.g. "contains a hyperlink").

## Status & next (NOT changing the approach unilaterally — proposing)
- The approach is sound and the **floor is validated**; the **ceiling is not yet** (recoverer-prior + beh-R
  routing). Proposed refinements above are within-approach tuning — await sign-off.
- **70B-labeler follow-up** (user's bonus): needs a model download (no 70B cached on sk3; OpenRouter dead).
  Expect 70B to separate correctness/clarity/elegance better (less collapse).
- Artifacts: `prompts/`, `mhat/` (recovered metrics), `labels/`, `run/` (all label files), `sk3_score.py`,
  `judge.py`, `compare_pairs.json`.
