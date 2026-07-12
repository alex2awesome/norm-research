# `methods/metric_implementer/experiments/` — R2 recovery & certificate drivers

> **Read this first.** It says which driver is canonical, what each one does, what they share, and the
> intended refactor. If you're adding/changing the R2 recovery pipeline, this is the map.

## The one-line answer: which file do I run?

**`run_r2_recovery.py` is the canonical driver.** In its default MCQ mode, the primary per-metric readout
is target-option probability/accuracy and the panel-level readout is **identity `I(J;Jhat)`**. Held-out
`I(M_ω;M′)` is retained as a secondary behavioral-equivalence replay. Run it for the R2 result.

```bash
# sk3 (1 GPU, vLLM executor): free-mode recovery (GLM articulates M̂; X re-executes → I(M_ω; M′))
HOME=/lfs python -m methods.metric_implementer.experiments.run_r2_recovery --mode free \
    --task peer-review --bucket specific --n-metrics 12 \
    --reconstructor-backend zai_anthropic --reconstructor-model glm-4.7 \
    --target-model <vllm-model-or-snapshot-path>
# MCQ mode (default): exact contrastive teaching set + controls + counterbalanced choices:
#   add --mode mcq --distractor contrastive --n-pool 30 --n-options 4 --R 20
# laptop (GLM-API executor + reconstructor, no GPU): use --target-model glm-4.7 (auto → sampled YES/NO)
```

## The two R2 drivers (and why there are two)

| driver | measures | status | run it? |
|---|---|---|---|
| **`run_r2_recovery.py`** | Primary MCQ identity reconstruction (`p(target)` per metric; `I(J;Jhat)` across targets) plus secondary held-out behavioral replay. `--mode free` remains a readable-rule diagnostic. | **canonical** | ✅ yes — this is the R2 result |
| `run_r2_certificate.py` | within-class subset certificate `OPT_Ω = max_S R(C(S))` ("can a subset of Ω re-express the full verdict") + GEPA-prose decomposition (T_prose). | **legacy / diagnostic only** — WRONG metric for the R2 headline (user-corrected 2026-06-25) | ❌ only as a diagnostic |

Both were born from the same plan; `run_r2_certificate` was built first (within-class), then
`run_r2_recovery` after the pivot to recovery. **`run_r2_certificate` should eventually be demoted to a
diagnostic submodule** (its `OPT_Ω` is still a useful Ω-coverage diagnostic) and its GEPA+Ω machinery
moved into the shared layer (see refactor below).

## Shared engine modules (the recovery/certificate stack)

- **`recon_channel.py`** — the recovery ENGINE. `run_metric(mode="free"|"mcq", induce="free"|"gepa")`
  → `iv_transmission` (= I(M_ω; M′), the headline) + `iv_recon` + identification accuracy (MCQ).
  `induce_free`/`induce_gepa`/`induce_mcq`; `_hi_lo_examples`; `_pyes` (logprob P(YES)); `_sampled_binary`.
  **`run_r2_recovery.py` is a thin per-R2-metric wrapper around this.**

  **Clarification (2026-07-12):** reconstruction is an anchor-free annotation-fidelity measurement, not an
  upper-bound method. MCQ uses a target-containing codebook. The default contrastive path selects related,
  non-clone distractors and an exact max-min teaching set on design data only; the reconstructor sees only
  target annotations and option descriptions. Options are counterbalanced, no-demo/shuffled-label controls
  are mandatory by default, and complete choice probabilities are persisted. Canonical-body re-execution on
  untouched items is a separate secondary readout; see theory §11.1a.
- **`cr3_reconstruction_values.py`** — the anchor-free value bridge for prompt mining. It freezes one
  bootstrap-only metric codebook per target, values every scored prompt with the same MCQ/control protocol,
  and emits immutable `[0,1]` marks for `cr_audit.py`. It never drops constant/collinear audit draws and
  never accepts external labels. `cr3_mining_worker.py --stage value` runs it with one resident reconstructor.
- **`omega_certificate.py`** + **`small_omega_brute_force.py`** + **`large_omega.py`** — the within-class
  subset engine (the legacy certificate). `OmegaCertificate.run()` → exact (K≤15) or large-Ω fallback.
- **`run_real_test.py`** — the GEPA+Ω **builder** shared by the legacy certificate: `phase_a_gepa`
  (GEPA optimizes a prompt → registry lineage), `_candidate_pool` (Ω = GEPA-lineage + decompose-p̂ +
  R1/L0-or-scoped-children + free-gen), `phase_b_certificate`, `_score_criteria_signals`, `_load_texts`.
- **`mine_clusters.py`** — Ω **coverage source**: `r2_groups`/`r2_children`/`r2_criteria` (R2 families),
  `r1_criteria`, `l0_reps`, `mine(levels=R1,L0,R2)`. Reads the curated hierarchy.
- **`orthogonalize.py`** — Ω **atomicity filter** (Shannon CMI → atomic units) + `family_coherence`
  (mean pairwise corr + participation ratio).
- **`mine_gepa_omega.py`** / **`harvest_gepa_omega.py`** — re-harvest Ω from GEPA registry lineages.
- **`real_gamma.py`** — `_YESNO`/`_signal`/`_median_split`/`_decompose` (forced-YES/NO + decomposition).

## Known duplicate / split-brain functionality (refactor targets)

1. **Ω construction lives only in `run_real_test._candidate_pool`** (used by the legacy
   `run_r2_certificate`). The canonical `run_r2_recovery` does NOT build the rich Ω (GEPA+R1/L0+free-gen)
   — it scores `merged_description` directly. **Refactor:** lift `_candidate_pool` (Ω builder) into a
   shared module so BOTH drivers use the same Ω, and have recovery consume it (M_ω = C(Ω), or `induce_mcq`
   picking from the Ω pool). See [[project_r2_cert_criterion_pool_mismatch]].
2. **GEPA (`phase_a_gepa`) is invoked only by the legacy driver.** Recovery currently skips it. The agreed
   design folds GEPA-mined criteria into Ω; that wiring is pending.
3. **Two scoring paths**: `_pyes` (logprob P(YES), vLLM-only — fidelity/M_ω) vs `_sampled_binary` (sampled
   YES/NO, API-compatible). The GEPA/fidelity path is vLLM-locked; an all-API-executor run must use sampled
   scoring. Reconcile via a backend-abstract scorer.

## Everything else in this dir

The remaining ~40 files are **analyses / aggregation / clustering utilities** invoked by higher-level
scripts or notebooks (e.g. `aggregate_*`, `compiler_sweep`, `pairwise_*`, `corr_cluster`,
`per_task_T`, `tvd_gap`, `opus_*`, `spectrum_sample`). They are NOT entry points for the R2 recovery
result — read each file's header docstring before using. Core non-driver utilities:
`glm_cluster.get_embeddings` (tfidf/bge/openai embeddings — used for semantic distractors),
`measures.fidelity_scalar` (the GEPA composite objective).

## Open work (updated 2026-07-12)

**Done:**
- **MCQ recovery folded in** — `run_r2_recovery.py --mode mcq` (the standalone `run_r2_recovery_mcq.py`
  is removed). Reports target-option probability/accuracy, control lift, panel identity `I(J;Jhat)`, and
  secondary behavioral replay.
- **Contrastive MCQ design** — design-only distractor screening, exact max-min teaching examples,
  counterbalanced positions, no-demo/shuffled-label controls, optional normalized choice logits, and complete
  split/option/choice/replay persistence are wired and covered by `tests/test_recon_channel_mcq.py`.
- **Backend-aware M_ω scoring** — `run_r2_recovery` auto-detects vLLM (logprob `_pyes`) vs API/GLM
  (sampled `_sampled_binary`) executor → either mode runs laptop-local on GLM-API. (Resolves refactor #3.)
- **Reconstructor-prior fixes in `recon_channel`** — `induce_free_dd` (feature↔label table, Codex #4) and
  `induce_reasoning` (multi-turn critique) + `_hi_lo_examples_wide`. Caveat: both are PARTIAL/ineffective
  on skewed M_ω — the bottleneck is M_ω quality (skew), not the prompt.

**Diagnostic paths retained:**
- Free reconstruction still supports balanced examples, `free_dd`, reasoning, and GEPA. These do not replace
  the primary MCQ identification measurement.
- `m_omega_gepa.py` optimizes discrimination on the design split. It is not a reconstruction optimum or a
  certificate, and the resulting prompt is frozen before held-out scoring.

**Still open:**
- Fold the rich Ω (GEPA+R1/L0+free-gen from `run_real_test._candidate_pool`) into recovery — refactor #1.
- Repeat the complete design over independent item splits when a population distribution over teaching sets,
  rather than the current conditional active-teaching estimand, is required.
