# E7 bounded-articulability pilot — results & verification (2026-06-12)

**One line:** the pilot validated that the E7 machinery runs end-to-end across three domains
(law, creative-writing, code) and produced a precise *catalog of instrument failures to fix*,
but **no interpretable measurement**: every headline number is within sampling noise at this
design, and the verified honest framing is *"instrument calibration with directional,
non-significant point estimates,"* not "measured articulability brackets + a thickness ordering."

Findings below were adversarially verified (6-agent workflow, every figure re-derived from raw
registry files); verdicts on the five candidate claims were 1× supported, 4× overclaimed, and
the writeup has been corrected accordingly. Spend: ~$2.23 OpenRouter total (under the $45 cap).
**Going forward this pipeline runs on sk3 vLLM offline batch only** ([[feedback_metric_implementer_sk3_only]]).

## What was built
- `law` task preset: 360 de-leaked Title VII fact sections; thin→mid→thick ladder
  `citation_grounding` / `factual_specificity` / `element_mapping` (prompt + code seeds).
- CW ladder extended to thin→mid→thick: `adverb_restraint` / `show_dont_tell` / `distinctive_voice`.
- CLI: `--metrics` filter, `--oracle-items` cost cap, per-task `runs/` mkdir; `e7_brackets.py`
  (zero-spend bracket + 2×2 interaction analysis with credibility flags).
- Fixed mid-run: reconstruction NaN (pool-exhaustion), per-task runs dir, and the bracket
  script's triad-panel selection (was sorting by `version_id` not date → pulled a stale 4-tier
  06-11 panel for code, inflating its words_share to 0.75; corrected to date-sort → 0.17).

## The bracket table (pilot, n≈24, apples-to-apples 06-12 seed panels)

`F_lo` = best construct-tracking achieved (max of triad anchor-agreement Spearman, grid oracle
Spearman); `F_hi` = √(pass-pass retest ρ). **Every row carries an artifact flag — do not read
any bracket as a measurement.**

| task | metric | tier | F_lo | F_hi | ρ_tier | words_share | flag |
|---|---|---|---|---|---|---|---|
| law | element_mapping | gemma-4b | .50 | .96 | .93 | .10 | F_hi from collapsed judge; F_lo max-over-small-n |
| law | element_mapping | llama-8b | .66 | .82 | .68 | .10 | F_lo max-over-small-n |
| cw | distinctive_voice | gemma-4b | .38 | .98 | .97 | .00 | (cw words_share floored) |
| cw | distinctive_voice | llama-8b | .30 | .77 | .59 | .00 | F_lo max-over-small-n |
| code | edge_case | gemma-4b | .36 | 1.00 | 1.00 | .17 | F_hi from collapsed judge |
| code | edge_case | llama-8b | .58 | **.48** | .23 | .17 | INVERTED — both ends suspect |

## Verified findings (corrected)

1. **words_share thin/thick ordering — OVERCLAIMED (directional only).** Point estimates
   reproduce exactly and are ordered as design intent predicts under the project convention
   (*low* words_share = thick / reader-decides): distinctive_voice **0.00** (thickest) <
   element_mapping **0.10** < edge_case **0.17** (thinnest). But this is **not statistically
   resolvable**: parametric Monte-Carlo at the true design gives words_share sampling SD
   0.14–0.36 ≫ the 0.07–0.10 gaps; the strict ordering survives only ~24% of resamples (chance
   1/6 = 17%); pairwise P(cw<law)=0.53, P(law<code)=0.54. **cw's 0.00 is a *floored* value**
   (σ²_item clipped at 0), not a measurement — so the ordering really rests on two interior
   estimates (law 0.10, code 0.17). Additional contamination: the 06-12 panels are 2-judge,
   and one judge (gemma) is near-constant (below), so the variance decomposition itself is
   unreliable beyond sampling noise.

2. **Chinchilla-null interaction I does NOT track thickness at pilot scale — SUPPORTED.**
   I_oracle is positive on all three (law +0.14, cw +0.22, code +0.30 — the "words help the
   capable reader more" direction) but its rank does not match the thickness order, **and it is
   uninterpretable**, not merely mis-ranked: code's entire I=+0.30 is manufactured by the
   degenerate gemma weak judge (cells −0.61/−0.91) against two byte-identical llama cells
   (+0.577/+0.577, words_effect_strong = exactly 0); the oracle Spearmans are over only n≈8
   items; and the optimizer returned worse-than-seed in **7/12 scaling cells** (the `frontier`
   axis silently substitutes seed there). Two budget points × 1 round × 1 seed cannot separate
   a budget effect from optimizer noise. *(Note: the C2 verifier inverted the thin/thick
   direction internally — corrected here against `triad.py`; its conclusion is unaffected.)*

3. **Judge-tier degeneracy — two distinct failure modes (C3 OVERCLAIMED on specifics).**
   - **llama-3.2-1b = applicability collapse**: returns valid JSON but no scorable items
     (`too_few_applicable`, oracle n=0, fidelity NaN). Caveat: it was actually run **only on
     law** in the saved data, and saved scorecards **cannot distinguish deliberate abstention
     from parse failure** (both route to the same skip) — so "marks everything inapplicable on
     every task / deliberately" is inferred, not verified.
   - **gemma-3-4b = score collapse**, and worse than first thought: near-constant on **both**
     law (3 values 0.75–0.9, std **0.038**) and code (2 values {0.6,0.7}, std **0.046**), only
     marginally varied on CW (5 values, std 0.082). Its ρ=0.93–1.00 are collapse artifacts, and
     its grid-oracle Spearman on code is **−0.61**. **Net: the pilot had effectively one
     informative judge tier (llama-8b) + the anchor** — so it cannot trace a judge-capability
     frontier at all.

4. **2-pass reliability cannot bound F_hi — OVERCLAIMED (the fix is necessary but insufficient).**
   The code/edge inversion (F_lo .58 > F_hi .48) is real but **both ends are artifacts**: F_hi
   = √0.232 where ρ=0.232 is statistically indistinguishable from 0 (bootstrap 95% CI
   [−0.19, 0.61]); F_lo = 0.577 = 1/√3, produced by a *single rank swap among only 4 applicable
   oracle items* (winner's curse). So ≥3 passes fixes F_hi but not the tiny-n max-selection
   inflating F_lo. **Correction to the plan: the "D-study target 3×3 → Eρ²≈0.84" figure was
   wrong** — the pilot's own G-study gives Eρ² = 0.22 (code) / 0.23 (law) / **0.00** (cw) at
   3 judges×3 passes, maxing ~0.33 at 5×3. The 0.84 came from a different (06-11, optimized,
   4-tier) panel and is deleted.

5. **code↔judge convergence "domain-patterned" — OVERCLAIMED.** Six seed-vs-seed Spearmans
   reproduce (law: citation .19, factual_specificity .62, element .39; cw: adverb −.03, show
   −.06, distinctive .33) but the law>CW contrast is **not clean** — the ranking interleaves
   (distinctive_voice .33 > citation .19) and a 3-vs-3 domain test cannot reject law=CW
   (Mann-Whitney p=0.10). Crucially every code seed is **one hand-written regex (v000 only)**,
   so near-zero adverb/show convergence cannot separate "construct not code-articulable" from
   "this regex is a poor proxy." Suggestive verifiability-gap hypothesis, **not** a causal
   reading or a demonstrated domain dichotomy.

## What the pilot CANNOT support (reader guardrails)
- No frontier **shape** / scaling curve (2 caps × 1 round; 7/12 cells frontier<seed).
- No **power law** / functional form (2 tiers, one degenerate).
- No per-task **thin/thick verdict** (ordering not resolvable; cw value floored; panel contaminated).
- No **causal verifiability gap** (crude single-regex code seeds, no improved-seed comparison).
- No reliability near 0.84 (real 3×3 Eρ² is 0.00–0.23).

## Top protocol fix (single root cause of 4/5 problems)
Scale the panel on **both** axes before reading any signal: **n≥60 items × 3 passes × 3
*informative* judges**, dropping gemma-3-4b (collapses on law+code) and llama-3.2-1b
(abstains) — replace with genuinely distinct, score-spreading tiers (e.g. llama-8b /
llama-70b / qwen-72b, all hostable on sk3). This simultaneously (a) shrinks the words_share
sampling SD below the gaps it must resolve, (b) lifts F_hi out of the indistinguishable-from-0
band, (c) removes the degenerate-judge artifact that fabricates code's interaction, and (d)
enlarges oracle subsets so F_lo stops being a max-over-n≈4. **Secondary, independent fixes:**
improved (multi-version) code seeds for the convergence claim, and >1 GEPA round / multi-seed
so `frontier` stops silently equalling `seed`. All future runs on sk3 offline batch.
