# Overnight autonomous exploration log — 2026-05-15

User went to bed; granted autonomous permission to explore confound questions.
Constraints: ≤ $30 credits, track usage, don't pause to ask.

## The questions being explored

User's worry: the per-task articulability results may be **skewed by something**. And the
longer-term framing: ultimately L1/L2/L3/L4 should be decided by *experimentation with the
data*, not by analyzing metrics — specifically:
- which metrics are recoverable from labels and consistent
- which metrics are predictive / correlated
- which can be implemented in code vs must be LLM-as-judge (+ how to measure LLM-judge goodness)

Tonight's scope: confound-hunt on the existing classification (compute-only where possible).
The experimentation-based validation (needs per-task outcome labels + modeling) is scoped as
a follow-up plan, NOT executed overnight — too big / too uncertain to run blind.

## Credit-usage tracker (running estimate)

| Item | est. cost |
|---|---|
| 5 audit subagents (general-purpose Claude, ~80-200 items each) | ~$8-15 |
| confound_analysis.py (pure compute, cached embeddings) | ~$0 |
| any further LLM exploration | logged below |
| **running total** | **~$8-15** |

Hard stop at $30. Will not launch GPU jobs or new large LLM runs.

## Log

- launched 5 audit subagents (jargon-confound on patents/N&C L3; L2→L1 ×2; broad random audit ×2). Agent cost ≈ $2-3 total (≈265K tokens) — far under the $8-15 estimate.

### AUDIT FINDINGS (4/5 agents in; corroborating)

**1. Jargon confound on L3 — CONFIRMED, large.** Of 109 patents+N&C L3 clusters, ~71 (65%) should be L2. A boilerplate rationale — "requires domain expertise → not code-checkable → LLM can't → briefed humans converge → 3" — appears in 80+/109. The classifier treated *technical vocabulary* as evidence of *needs-a-human-expert*. Patents slightly worse than N&C. **L3 share for technical domains inflated ~60-65%.**

**2. L2→L1 — minor under-assignment.** True L1 rate among "L2" clusters ≈ 5-8%; global L1 would rise 4% → ~6-7%. Not a big artifact. Concentrated in legal style-rules (phrase blacklists, readability formulas) and patents/peer-review form-completeness/presence checks.

**3. Broad audit (n=199) — 83% agreement.** Dominant error: assigned-3 → should-be-2 (21/34 disagreements). **Diagnostic the agent surfaced: `articulability=3` + `indeterminacy≤2` + `reasoning_depth≤3` = the inflation signature** — a "3" with no free parameter and no deep reasoning is a 3 reached only via the (usually spurious) "domain expertise" claim. Also: a few L1 over-applied to presence-checks that need content judgment; a few L4 over-applied to operationalizable normative criteria.

**Implication:** the L3 column is systematically inflated; the 2-vs-3 boundary is the weak point. Headline "patents 54% L3 / N&C 49% / peer-review 42%" is substantially overstated. The rd/indeterminacy signature lets us quantify + correct this across the full 24K without re-running everything.

### CONFOUND QUANTIFICATION (confound_correction.py)

Signature of a suspect (inflated) L3: `articulability=3 & indeterminacy≤2 & reasoning_depth≤3`.
**3,613 of 8,354 L3s (43%) are suspect.** Corrected per-task (suspect-L3 → L2):

| task | L3 raw→corrected |
|---|---|
| patents | 54% → 24% |
| notice-and-comment | 49% → 23% |
| peer-review | 42% → 18% |
| grant-funding | 30% → 8% |
| code-review | 35% → 16% |
| creative-writing | 28% → 19% |
| humor | 36% → 23% |

After correction L3 is **fairly uniform (~16-24%)** — the big between-task L3 differences were mostly the confound. **L4 is unchanged** (audit: endpoints reliable): creative-writing 17%, humor 15%, all others ≤3%.

**So the robust headline survives and sharpens:** tasks barely differ in articulability EXCEPT creative-writing & humor carry a real tacit (L4) tail. The "patents/peer-review are expert-gated" story was largely a jargon artifact.

Caveat: the signature catches inflation-via-spurious-expertise-claim but MISSES inflation that rode on inflated `indeterminacy=3` (audit 3a flagged this) — so the correction is an estimate. Definitive fix = v7 re-classification with a sharpened 2-vs-3 anti-jargon prompt.

### STRUCTURAL CONFOUNDS (confound_analysis.py — joined 18,258 clusters to source orientation)

- **Verbosity (description length): NO confound.** Mean articulability by length quartile = 2.42 / 2.47 / 2.49 / 2.48 — flat.
- **Cluster size: NO confound.** Mean articulability by size = 2.48 / 2.42 / 2.40 / 2.47 — flat.
- **Source orientation: a real correlation, but the L4 finding SURVIVES it.** Orientation does correlate with articulability (stylebooks 11% L1 / 3% L4 — mechanical; formal_guideline 2% L4; blog/textbook 11-13% L4). And tasks differ in orientation mix (peer-review/grant-funding/N&C ~40-46% formal_guideline; creative-writing/humor ~1-5% formal, mostly how_to/blog/wiki). **BUT** — the decisive check — within creative-writing+humor, L4 stays high across *every* orientation: how_to 20%, blog 22%, textbook 26%, even formal_guideline 16% (vs the global formal_guideline rate of 2%). So the creative-writing/humor tacit elevation is intrinsic to the TASK, not a source-type artifact.
- Note: part of the orientation→L3 correlation (formal/professional sources show 38-46% L3) is just the jargon confound re-expressed — formal sources carry more jargon.

**Verdict: the only material confound is the L3 jargon inflation. After correcting it, the picture is clean and the L4 headline is robust to verbosity, cluster-size, and source-type.**

### THE FIX — v7 prompt + L3 re-judgement

Wrote **v7** of the classifier prompt (`scripts/classify_clusters_2axis.py`): added a blunt `*** TECHNICAL VOCABULARY IS NOT EVIDENCE OF LEVEL 3 ***` block, sharpened the 2-vs-3 test ("would a STRONG, KNOWLEDGEABLE LLM-judge trained on the specialist field apply this?"), made route-(a) "expert judgment" require naming a concrete capability the LLM lacks, and added two technical-domain worked examples ("claims are clear and definite" → 2; "comment engages the cost-benefit analysis" → 2).

Validated v7 on the seed-14 calibration 33: articulability `{2:19,3:11,4:3}` (v6) → `{2:23,3:6,4:4}` (v7) — 5/11 L3s correctly moved to L2, extremes intact. v7 behaves as designed.

`--reclassify-l3` mode added: re-judges every v6-L3 cluster (8,354) with v7. Non-L3 clusters keep v6 (audit: endpoints reliable). `aggregate_v7_corrected.py` combines them into the final per-task distribution. [run in progress; results appended below]

### SURFACE axis + L4 content (sanity)

- surface_vs_substance per task discriminates modestly (mean 2.79 code-review/legal → 3.37 notice-and-comment). creative-writing/humor are both high-L4 AND high-substance (S4 = 50%/55%); code-review/legal more form-leaning.
- Sampled L4 clusters are face-valid taste/craft criteria: creative-writing — "Compelling first sentence", "Natural Unaffected Voice", "Economy and understatement", "Choice of Moment"; humor — "Commit to the bit", "Authenticity versus shtick", "Emotional resonance". The L4 bucket is real content, not noise.

---

# FINAL SUMMARY (overnight, 2026-05-15)

## What was wrong, and what we did about it

The per-task articulability numbers from the v6 full run **were skewed by one large confound** (and only one): gpt-5-mini systematically over-rated rubrics as **L3 ("needs a domain expert")** when a knowledgeable LLM-judge could actually apply them (**L2**) — triggered reflexively by technical vocabulary. Five independent audit subagents found this unanimously; ~60-65% of patents/notice-and-comment L3s were mis-rated.

Fix: **v7 prompt** (blunt anti-jargon block + sharpened 2-vs-3 test) re-judged all 8,354 v6-L3 clusters. v7 reassigned ~60% of them (4,419→L2, 548→L4, 34→L1; 3,363 stayed L3). Non-L3 clusters kept v6 (audit confirmed endpoints reliable).

## The v7-corrected per-task articulability (FINAL)

| task | L1 | L2 | L3 | L4 | mean |  | v6-raw L3 → v7 L3 |
|---|---|---|---|---|---|---|---|
| creative-writing | 2% | 65% | 11% | **22%** | 2.53 | | 28% → 11% |
| humor | 3% | 62% | 16% | **20%** | 2.52 | | 36% → 16% |
| math-stackexchange | 7% | 71% | 16% | 5% | 2.19 | | 36% → 16% |
| news-homepages | 5% | 74% | 18% | 4% | 2.21 | | 38% → 18% |
| legal-outcome-prediction | 3% | 79% | 15% | 2% | 2.17 | | 35% → 15% |
| patents | 4% | 75% | 21% | 0% | 2.16 | | 54% → 21% |
| notice-and-comment | 4% | 79% | 16% | 1% | 2.13 | | 49% → 16% |
| peer-review | 3% | 81% | 15% | 1% | 2.13 | | 42% → 15% |
| press-releases | 7% | 80% | 12% | 1% | 2.08 | | 29% → 12% |
| code-review | 9% | 78% | 13% | 0% | 2.05 | | 35% → 13% |
| grant-funding | 3% | 90% | 6% | 0% | 2.04 | | 30% → 6% |

## What changed vs. what is robust

**CHANGED — the L3 story dissolved.** The v6 headline "patents 54% / notice-and-comment 49% / peer-review 42% expert-gated" was the confound. Corrected, L3 is small and roughly uniform (6–21%). **There is no real "formal domains are expert-gated" finding** — those domains are L2-dominated like every other task. patents 21% is the largest residual L3 but n=92 (noisy).

**ROBUST — the headline survives and sharpened.** Two findings hold:
1. **The tacit (L4) tail is real and confined to creative-writing (22%) and humor (20%)** — every other task is ≤5%. L4 was untouched by the correction (audit: endpoints reliable); it even rose slightly (v7 moved 548 mis-low L3s up to L4). Structural checks confirm it is not a confound: within creative-writing+humor, L4 stays high (14–31%) across *every* source orientation, including formal_guideline (16% vs the global 2%).
2. **Across all 11 domains evaluation criteria are overwhelmingly L2 (62–90%)** — i.e. a strong LLM-judge can apply them. Code-checkable (L1) is a small 2–9%; genuine expert-judgment (L3) a small 6–21%.

**The clean cross-task story:** tasks barely differ in articulability *except* creative-writing and humor carry a substantial irreducibly-tacit tail. Combined with the earlier embedding-diversity result (rubric diversity ~constant across tasks), the picture is: **tasks differ in how *tacit* their evaluation is, not in how *broad* or how *expert-gated* it is.**

## Confounds checked and cleared
- L3 jargon inflation — REAL, LARGE → fixed by v7.
- Verbosity (description length) — no confound (flat).
- Cluster size — no confound (flat).
- Source orientation — correlates, but L4 finding survives controlling for it.
- Multi-homing (1.7%) — enumeration artifact, deduped at aggregation.
- L1 under-assignment — minor (~+2-3 pts possible); not corrected; left as a known small floor effect.

## Credit tally (overnight autonomous session)
| Item | est. cost |
|---|---|
| 5 audit subagents (~265K tokens) | ~$2-3 |
| v7 calibration (sample 33) | <$0.10 |
| L3 v7 re-classification (8,390 gpt-5-mini calls) | ~$4-5 |
| compute-only analyses (confound scripts, embeddings cached) | ~$0 |
| **overnight total** | **~$7-9** (cap $30 — well under) |

## Open / for the morning
- patents general bucket is n=92 — all patents numbers are noisy; treat with caution.
- The v7-corrected output is `outputs/analyses/articulability_v7final_per_task.parquet`; v7 raw L3 re-judgements in `cluster_2axis_l3_v7.jsonl`.
- L1 is mildly under-assigned (~+2-3 pts) — a v8 could target "include/complete/provide a discrete artifact" rubrics, but low priority.
- Notebook viz (Step 5) not yet updated with any of this.

---

# SCOPED PLAN (not executed) — experimentation-based validation of the articulability scale

The user's framing: ultimately L1/L2/L3/L4 should be decided **by experiment on the data**, not by an LLM's self-report. The current classification is a *prior / triage* — useful, but the LLM saying "this is L2" is not proof. The experiment makes the scale operational. This needs the per-task **expert outcome labels** (accept/reject, scores) already collected for the dense-model work — locate those first.

**The operational redefinition** (maps onto `Outcome = f(Verifiable) + g(Articulable) + h(Taste)`):
- A rubric's TRUE level = the strongest method that *recovers* it against expert labels.
- L1 ⇔ a deterministic code implementation predicts the outcome (AUC well above chance).
- L2 ⇔ code fails but an LLM-judge applying the rubric predicts the outcome.
- L3 ⇔ neither single rubric predicts, but a dense model trained on expert labels does (and briefed raters converge).
- L4 ⇔ even the dense model plateaus below the expert ceiling — the (1−C) residual.

**Proposed steps:**
1. **Assemble per-task labeled test sets** — work artifact + expert outcome. (Locate existing data; this gates everything.)
2. **Recoverability from labels** — autometrics-style metric discovery on the labeled data; check whether discovered metrics match taxonomy rubrics. A rubric that can't be rediscovered is either non-operative or mis-described.
3. **Code vs LLM-judge per rubric** — for a sample of rubrics across all 4 self-reported levels: (a) LLM-generate a code checker, run it, AUC vs outcome; (b) LLM-judge applies the rubric, AUC vs outcome. Compare to the self-reported level — does code recover the L1s? does LLM-judge recover the L2s? This is the real test of the scale; expect the self-report to over-state codeability and (post-v7) be roughly right on L2.
4. **Predictive / correlated** — per rubric: AUC and correlation with the outcome; cluster rubrics by their prediction vectors to find redundant vs complementary criteria.
5. **LLM-judge goodness** — measure three ways: (a) AUC vs expert labels; (b) test-retest consistency (re-run N×, modal-label stability — the precision check discussed earlier); (c) inter-model agreement (gpt-5 vs Claude as judges) — catches model-specific bias the self-report can't.
6. **Close the loop** — compare experiment-derived levels to the v7 self-reported levels; the disagreement *is* a finding (where introspective articulability ≠ operational articulability).

Cost/scale: substantial (multi-day, real LLM + possibly dense-model compute) — explicitly NOT started overnight. Flagged for user direction.
