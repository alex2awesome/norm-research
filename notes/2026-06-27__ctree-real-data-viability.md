# ctree real-data viability — the collapse is a rubric/corpus mismatch, not the model (2026-06-27)

Overnight work toward the goal: *an accurate ctree that maximizes test-set power of the
articulated metric set (train→test), with metric prompts GEPA-optimized for reconstruction
accuracy.* The blocker turned out to be data alignment, not model strength.

## 1. The collapse is MODEL-INDEPENDENT

`diag_supervisor_probe.py` scored the collapsed peer-review rubrics with three supervisors of very
different strength through the z.ai proxy:

| rubric | gemma-4-31b | glm-5.2 | claude-sonnet-4 |
|---|---|---|---|
| Citation practices | all-1 (std 0) | all-1 (std 0) | all-1 (std 0) |
| Phase-1 allocation | all-NA | all-NA | all-NA |
| Anonymization | discriminates | discriminates | discriminates |

Identical degeneracy across all three → **the collapse is NOT model weakness.** GEPA with a
stronger supervisor (the user's first instinct) will not fix *these* rubrics — the rubric concept
genuinely doesn't apply to the corpus texts. (GLM-5.4 is not served by z.ai — "Unknown Model
1211"; glm-5.2 is served and quota works again.)

## 2. peer-review rubric viability (`diag_rubric_viability.py`, 30 rubrics × 100 texts)

- **5/30 viable** — the paper-*content* rubrics: *Description of mathematical setting/algorithm*
  (appl .98), *Specification of contribution and novelty* (appl 1.0), *Statement of limitations*,
  *Proofs for theoretical claims*, *Dataset availability*.
- **22/30 all-NA** — review-*process* rubrics (COI, confidentiality, fairness, deadlines,
  plagiarism, reviewer conduct) that genuinely don't apply to paper texts.
- **3/30 collapsed** (all-1 / single value).

So peer-review's online-rubrics are editorial-process checklists; only the content subscore
applies to the modeling corpus. The corpus IS ML papers (paper_id/venue/domain/year).

## 3. Bug fixes landed (real blockers — peer-review was never runnable)

- `DATASET_CONFIGS["peer-review"]`: `id "id"` → `"paper_id"` (CSV has no `id` column); `split` →
  `peer_review_modeling_dataset.csv.gz` (`load_items` doesn't append extensions).
- `metric_implementer/measures.py:_spearman`: added a constant-input guard (std<1e-9 → NaN) —
  this is what crashed distillation on the all-zeros collapse (`spearmanr` on constant input).
- Wired `creative-writing` into `DATASET_CONFIGS` (LitBench: id="Unnamed: 0", text, judgement;
  87k rows) — better rubric/corpus alignment (prose-quality rubrics on prose), and per memory its
  dense-model ceiling is "still climbing" (real signal, unlike press-release's ~0.55 confound).

## 4. Plan (in progress)

1. Measure baseline articulated-metric power (train→test LR AUC of non-degenerate rubrics) per
   task — `diag_metric_power.py`. Pick the stronger basis.
2. GEPA-optimize the **viable** (non-collapsed) rubric prompts via `metric_implementer.improve`
   (its `fidelity_scalar` already folds in reconstruction accuracy + discrimination). Collapse is
   irrelevant here — viable rubrics already discriminate.
3. Run the ctree end-to-end (proposer fix + composite enabled), report test AUC of {base} vs
   {base + discovered} — the power the infilling adds.
4. Document the power trajectory.

## 5. Open / honest

- Only 5 viable peer-review rubrics (thin). creative-writing likely has more + higher ceiling —
  the stronger demonstration target if peer-review power is low.
- GEPA cannot rescue the all-NA process rubrics (concept doesn't apply) — those are dropped, not
  fixed. The user's "GEPA + strong supervisor" fixes *vague-but-applicable* rubrics, not
  *inapplicable* ones.

## 6. Capstone results (power trajectory, train→test AUC) — `ctree_power.py`

- **peer-review: 1/40 viable** under the strict filter (appl>0.3 AND std>0.1) → aborted ("too few
  viable rubrics"). Its online-rubrics are overwhelmingly inapplicable review-process checklists;
  peer-review cannot anchor a "maximize metric power" demonstration. Confirms §2.
- **creative-writing: passed the viability filter** (judge cache grew past the 40-item probe into
  full materialization) → prose rubrics align with prose texts. Capstone running: baseline AUC →
  infilling → final AUC. (Result pending at write time.)
- GEPA objective check: `metric_implementer/measures.py:fidelity_scalar` already weights
  **reconstruction accuracy** (`w_recon × reconstruction.behavioral`) + reliability + counterfactual
  + discrimination (predictive perf structurally excluded — the evaluate-never-gate). So "GEPA +
  reconstruction accuracy" is met by the existing `improve()`; `gepa_viable.py` runs it scoped to
  viable rubrics (skipping inapplicable ones that GEPA can't fix).
