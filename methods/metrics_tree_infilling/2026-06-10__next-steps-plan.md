# metrics_tree_infilling — review & next-steps plan (2026-06-10)

The ideal capability, restated:

1. **Ingest** a metric set + a labeled text corpus and apply the metrics.
2. **Order the metrics**: which are general, which are specific — a *natural hierarchy*.
3. **Detect** where new metrics are needed, anywhere in the tree.
4. **Code** the new metrics.

## A. Where the method stands against the ideal

| Capability | Status | Evidence / gap |
|---|---|---|
| (1) ingest + apply | **Built, never run on real data** | `io_metrics.py` loads online-rubrics + code metrics; NA→impute+indicator; `run.py` has 4 task configs. But no end-to-end run on any real task has ever happened — every result so far is synthetic. |
| (2) metric hierarchy | **Half-built — the biggest conceptual gap** | We measure generality (coverage) only for *discovered* features (`loop.py:240`). The input metric set never gets a generality score, and no hierarchy artifact (who generalizes whom) is ever produced. The tree is a partition of *items*, not an ordering of *metrics*. |
| (3) detect gaps | **Built and validated (synthetically)** | Gap = terminal node w/ poor held-out fit (`gaps.py`); pooled root-level gaps via `depth_dial.py`; latent regions discoverable with `discovered_feature_role="both"` (`test_latent_split.py`). §9 limitation (missing interaction of absent features) documented. |
| (4) code the metric | **Stub** | `feature_gen.py` emits a *judge rubric* and calls it "distilled." No `score(text)` program is ever generated; `distill_method="embedding"` is config-only, unimplemented. The repo already has the pattern to copy (`runs/validity_full/v2/*/codegen_claude/` per-aspect programs). |

## B. What the tests prove — and where they fall short

**What passes (10/10):**

- `test_mfluctuation.py` — engine detects planted instability, right variable, right cutpoint.
- `validate_against_partykit.py` (Python self-check half) — detection 0.81 / FP 0.0 / cutpoint err 0.008 over planted scenarios.
- `test_loop_smoke.py` — full loop closes a planted gap with a *scripted* proposer.
- `test_scenario/` — two tacit norms recovered from prose-with-synonyms, decoys rejected, coverage ranks song(0.33) > glow(0.23) ≈ true region sizes.
- `test_latent_split.py` — a discovered feature becomes a *new splitting variable* (latent region, no context column).

**Where they fall short (ranked by how much they currently overstate confidence):**

1. **Every test is oracle/scripted at the LLM seams.** The proposer is either hard-coded
   (`_fake_proposer`) or a deterministic separator that *skips known attributes by fiat*
   (`oracle.py`, `test_latent_split._oracle_proposer`). The judges are substring detectors.
   So the tests prove the *algorithm*, zero of them prove that a real model can (a)
   articulate "melodious" from varied prose, (b) follow "not already covered by known
   criteria," or (c) score a rubric consistently enough to pass the reliability gate.
   The `--live` path exists but has never been executed.
2. **R parity has never actually run against R** (R absent on the laptop). The "as close to
   R as possible" claim rests on the Python self-check only.
3. **Pooling (`depth_dial.py`) is exercised by no test.** No scenario has the *same* hidden
   norm spanning ≥3 gap leaves — exactly the case pooling exists for. The clustering
   threshold (0.55 cosine) and the hash-embedding fallback are untested guesses.
4. **No pure-noise gap test.** A region whose labels are irreducible coin flips should
   produce proposals that *all* die at `no_closure` and a loop that terminates honestly.
   Decoys test "wrong feature rejected," not "no feature exists."
5. **All synthetic worlds are easy-mode for the GLM:** 3–6 metrics, full applicability,
   n=1.4–2.4K. Real runs are 40+ rubric metrics, heavy N/A, and `fit_node_glm` is
   *near-unpenalized* (`ridge_C=1e6`) with `min_node_size=20` → 40-feature GLMs on 20–40
   rows will be unidentifiable. Nothing currently guards against this.
6. **Known cosmetic debt:** `minimal_depth` is still computed/reported though we know it's
   misleading for in-X features; coverage uses a hard 0.5 std-coef threshold that conflates
   *narrow* with *weak*.

## C. Plan

### Phase 0 — first contact with real data (do this before any new machinery)

*Everything below is hypothesis until the method touches a real task.*

- [ ] Run `run.py --task peer-review --metrics rubric --max-metrics 25` with precomputed
      levels where possible (v2 cells DB) or vLLM materialization on sk3. Budget one
      evening; treat crashes/degeneracies as the deliverable.
- [ ] Expected failure to fix on contact: per-node GLM with many metrics. Add
      **(a)** elastic-net or moderate ridge at deep nodes (scale C with n/node), and
      **(b)** a root-level lasso pre-screen that caps X to the top-K metrics by
      population-level signal (keep the rest available to z). Config: `max_x_metrics`,
      `node_ridge_c_schedule`.
- [ ] Expected failure #2: rubric applicability is sparse → imputed columns + indicators
      dominate. Add a min-applicability filter (e.g. drop metrics applicable on <30% of
      items) with a logged count of what was dropped.
- [ ] Decide the live key wiring (open since build week): local vLLM for judge +
      what backend for the proposer. **Not** the USC key by default.

### Phase 1 — metric hierarchy as a first-class output (capability 2)

The tree already contains the information; we just never read it out for the *input* metrics.

- [ ] **Continuous coverage** for every metric (input + discovered): replace the 0.5
      threshold with population-weighted standardized effect,
      `cov(m) = Σ_leaves (n_leaf/n) · |β_std(m, leaf)|`, plus the thresholded variant as a
      secondary readout. Fixes narrow-vs-weak conflation.
- [ ] **Leaf-activity profile** per metric: the vector over leaves of `|β_std|` (or
      active/inactive). This is the object the hierarchy is built from.
- [ ] **Hierarchy construction**: metric A *generalizes* B if A's active-leaf set ⊇ B's
      (with tolerance); siblings = same active set. Emit (i) a containment DAG and (ii) an
      agglomerative dendrogram over profile similarity as `hierarchy.json` +
      a rendered text tree in `tree_summary.json`. New module: `hierarchy.py`.
- [ ] **Drop `minimal_depth`** from records/outputs (keep internally only if the importance
      discount still wants it; otherwise delete `measured_importance`'s dependence on it).
- [ ] Test: extend `test_scenario` asserts — size/feeding/pelt profile-cluster together and
      are confined to grove leaves; song/glow are leaf-specific; habitat indicators are
      splits not X-effects. The scenario's answer key already encodes the true hierarchy.

### Phase 2 — close the loop on "code them" (capability 4)

- [ ] Implement the promised distillation tiers in `feature_gen.py`:
      **Tier 1** frozen judge rubric (current behavior, now explicitly a fallback) →
      **Tier 2** generated `score(text)` Python program (reuse the codegen prompt pattern
      from the per-aspect programs work; AST-safety check; same `score()/applies()`
      convention as `load_code_metrics`) →
      acceptance = round-trip fidelity: program vs judge agreement on a held-out sample
      ≥ threshold (this is the `project_rubric_fidelity_validation` recipe applied here).
- [ ] Reliability for code metrics is 1.0 by construction — record *fidelity-to-judge*
      instead as the discount.
- [ ] Persist kept metrics as runnable artifacts:
      `outputs/.../kept_metrics/{name}.py|.json` so the output of one run is a valid
      metric-set *input* to the next (the method becomes self-composing).
- [ ] Test: scenario hidden norms distill to programs that match the oracle judge ≥95% on
      held-out items.

### Phase 3 — harden the tests where they overstate

- [ ] **Live-LLM scenario run** (one-off, recorded): `test_discovery --live` with a real
      proposer + judge on the 2.4K creature corpus. Success = both norms articulated from
      prose; record the articulations verbatim in the README. This is the single most
      informative missing experiment.
- [ ] **Pooling scenario**: a world where one hidden norm (e.g. "musty smell → rejected")
      holds across *all three* habitats while the Code varies — should surface as ≥3 gap
      leaves whose proposals cluster, get pooled, and re-enter near the root. First real
      test of `depth_dial.py`.
- [ ] **Noise-gap scenario**: one region with coin-flip labels; assert nothing is kept for
      it (`no_closure` deaths) and the loop terminates with that gap honestly unfilled.
- [ ] **Adversarial proposer test**: an oracle that does NOT skip known attributes —
      verifies the redundancy guard (not the prompt) is what blocks re-derivation.
      Currently redundancy is doing no work in any test.
- [ ] **R parity, for real**: run `validate_against_partykit.py --n 100` on a machine with
      R + partykit (sk3 or a Docker one-off); commit the agreement report next to this plan.
- [ ] **Seed-stability report**: refit over B bootstrap/seed draws; report split-variable
      and kept-feature stability. Cheap (engine is fast) and tells us whether single-run
      tree readouts are trustworthy on real data.

### Phase 1b — semantic hierarchy (alongside the scope hierarchy)

Two operationalizations of "hierarchy," built from the same fitted objects:

- **Scope** (Phase 1): leaf-activity profiles → containment DAG ("where does this metric
  carry effect"). The articulability-bound workhorse.
- **Semantic**: pairwise entailment between metrics, tested two ways — LLM entailment over
  the rubric texts ("does satisfying B entail satisfying A?") AND empirical containment in
  the materialized levels (P(A=1 | B=1) ≈ 1). An edge requires both; disagreement between
  the two tests is logged as a finding, not an error.
- [ ] Emit both DAGs + their **cross-tab**: semantically-nested-but-scope-disjoint pairs =
      same concept enforced differently across subpopulations (the imposed-vs-emergent
      norms signal).

### Phase 4 — research extensions (after a real-data result exists)

- [ ] *k*-candidate proposals per gap (sample k, keep best closure) — one-line loop change,
      likely the cheapest quality win for live runs.
- [ ] Pairwise interaction screen at flagged gap nodes to soften the §9 XOR limitation
      (lookahead over products of existing X columns before invoking the LLM).
- [ ] Cross-task: do discovered features transfer (the imposed-vs-emergent norms angle)?

## C2. Addendum (2026-06-10): solutions to the theoretical holes

> **Moved/abstracted (2026-06-10, later same day):** the metric-validity machinery in this
> section (judge-noise bounds, counterfactual + reconstruction validity, the
> evaluate-never-gate principle, judge-prompt optimization, and the new articulability
> scaling laws) is **not tree-specific** and now lives as its own first-priority method at
> `methods/metric_implementer/` (README + `2026-06-10__design.md`). This method becomes a
> *consumer*: it calls `compute_scorecard` on its discovered features. The tree-specific
> items below (triage ladder, nonlinear redundancy, confirm split) stay here.

Discussed after review; these upgrade the method from "flags gaps" to "diagnoses gaps with
estimable bounds," all without label ground truth.

### Gap triage ladder (fixes hole 1: gap ≠ missing feature)

Per flagged node, fit four models on the node's held-out items and replace the fixed
deviance/AUC thresholds with their differences:

| Rung | Model | Bounds |
|---|---|---|
| 1 | GLM(X) (current node model) | metrics, as linearly combined |
| 2 | flex(X) (small GBM on metric levels) | metrics, any combination rule |
| 3 | dense(text) (per-task Llama-8B dense RM, or embedding probe) | learnable at all (C layer) |
| 4 | Bayes ceiling via 1-NN twin disagreement (Cover–Hart: Bayes acc ≤ 1 − NN_err/2) | any model ever |

Diagnosis routing: **flex−GLM large** → misspecification → LLM articulates the *combination
rule* (different prompt; output tagged `recombination`); **dense−flex large** → genuinely
missing textual feature → current proposer loop; **ceiling ≈ dense ≈ flex** → irreducible
noise/taste → mark honestly unfillable, spend nothing. Headline per-region readout:
**articulability gap = dense − GLM**, with (ceiling − dense) quarantined as maybe-learnable.

- [ ] Implement ladder in `gaps.py` (new `triage.py`); rung 3 evaluates existing dense
      checkpoints per node; rung 4 = 1-NN over text embeddings within the node.

### Judge-noise bounds (fixes hole 1d) — the judge is ours, so its noise is measurable

1. [ ] **Per-node reliability**: re-score k passes on flagged nodes' items; attenuation-correct
       the node fit by measured ρ (Spearman disattenuation). Gap test becomes "worse than
       expected given ρ." Where ρ is the culprit → ensemble k judge passes, don't infill.
2. [ ] **Validity via synthetic counterfactuals**: for each (discovered or input) rubric,
       generate minimal text pairs with the attribute planted/absent *by construction*
       (generalize the `test_scenario` world generator into a measurement instrument);
       judge accuracy on these = a validity bound needing zero real labels.
       *Known confound*: judge and generator can share a misreading — both implement a
       simpler aligned attribute S instead of target T, and the test certifies S as T.
3. [ ] **Reconstruction validity** (= the "metric rediscovery" protocol): judge labels real
       items → a reconstructor sees only (text, label) pairs and articulates the separating
       rule → compare to the original rubric (semantic match + behavioral agreement of the
       re-implemented guess). Catches judge-measures-S because the reconstructor names what
       the judge *actually* does; no generator in the loop, so no aligned-confound risk.
       Blind spot: S,T naturally correlated in real data are observationally identical.
4. [ ] **Compose 2+3 into the validity loop**: reconstruct first (names the suspect simpler
       readings) → generate counterfactuals that *decorrelate T from each named suspect*
       (vary T holding S fixed and vice versa) → use different model families for judge /
       generator / reconstructor. Certificate is relative, not absolute (underdetermination):
       "no articulable simpler reading survives decorrelation across model families."
5. Guarantee statement: reliability-corrected A ≤ achievable; dense ≤ achievable ≤
   twin-ceiling (all AUC-style numbers on the same held-out items; corrected A = observed A
   disattenuated for judge noise; dense lower-bounds the optimum because it is achieved;
   twins upper-bound it because identical-looking texts with different labels are
   unresolvable). Articulability gap ∈ [dense − corrA, ceiling − corrA]. Predictive claims
   need only reliability (the rubric IS the articulation); semantic claims need the validity
   loop (items 2–4).

### Reconstruction is a classifier, not a gate (avoids the circularity)

Validation-by-reconstruction and measurement-of-articulability share one capability
bottleneck (LLM rule induction from labeled examples). If reconstruction *gates* which
judges survive, complex-but-valid instruments are filtered out for the same reason
complex-but-articulable norms get called tacit → correlated instrument error inflates the
tacit estimate (a reviewer will call this circular, fairly).

- [ ] Reconstruction validity is a **metric-type classifier**, never a keep/drop gate.
      Reliable + predictive + counterfactually-valid + reconstruction-FAILING metrics are a
      reported category ("instrument-level language-tacit: the judge knows more than it can
      tell"), not discards.
- [ ] State the ceiling as operationalization-relative ("articulable by this reconstructor");
      mitigate with cross-family reconstructors.

### Judge-prompt optimization against the fidelity measures (GEPA-style)

All three measures (ρ, counterfactual accuracy, reconstruction match) are automatic → a
prompt-optimization objective with no human labels. Loop: candidate rubric prompt π →
score (ρ via k-pass test-retest; CF-acc on decorrelated planted pairs; recon-match) →
**reflective mutation** using the textual failure signals (the reconstructor literally names
the misreading; missed counterfactual pairs become injected few-shots) → accept on a
never-touched validity batch (fresh counterfactuals + fresh reconstructor, different model
family).

- [ ] **Hard rule: predictive performance (gap closure / label fit) is NEVER in the
      objective** — optimizing it turns the instrument into a classifier and voids the
      semantic claim. Construct fidelity only; prediction stays a held-out evaluation.
- [ ] Anti-Goodhart on the measures themselves: regenerate counterfactuals per round (a fixed
      set distills the generator's interpretation into the judge); different families for
      generator/reconstructor vs judge.
- [ ] Tooling: custom ~150-line loop over existing `LLMClient` + offline-batch vLLM scoring
      (GEPA-the-algorithm without DSPy-the-framework); revisit `dspy.GEPA` only if it grows.
- [ ] Slots into Phase 2 as Tier 1.5: optimize the rubric prompt, then distill to code.

### Nonlinear redundancy (fixes hole 2)

- [ ] Replace `LinearRegression` in `redundancy_check` with conditional-value testing: flex(X)
      vs flex(X + new) on the node; no improvement ⇒ recombination. Re-label (don't discard):
      recombinations feed the semantic decomposition (Phase 1b) as combination rules.

### Confirm split (fixes hole 3: test reuse)

- [ ] Three-way `discover/select/confirm` (≈60/25/15) in `discover_test_split`; guards consult
      *select* per round, *confirm* touched once for headline numbers + final hierarchy.
      (Thresholdout is the rigorous upgrade if rounds ever grow; unnecessary at 6×8.)

## D. Sequencing & effort

| Phase | Effort | Blocked on |
|---|---|---|
| 0 real-data contact | 1 day + 1 sk3 evening | key/backend decision |
| 1 hierarchy | 1 day | nothing — pure Python over existing objects |
| 2 codegen distill | 1–2 days | proposer backend (same decision as 0) |
| 3 test hardening | 1–2 days; R run is one-off | live run shares 0's backend; R needs sk3/Docker |
| 4 extensions | open-ended | a Phase-0 result worth extending |

Recommended order: **0 → 1 → 3(live+pooling+noise) → 2 → 3(rest) → 4.** Phase 1 before the
live experiments because the hierarchy readout is what makes a real-data run *interpretable*;
Phase 2 after, because codegen quality questions are moot until real gaps are being found.
