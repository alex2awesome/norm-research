# R2-level per-metric certificate — running notes
Plan: `~/.claude/plans/vivid-giggling-sutton.md`. This file logs progress + decisions as we build.

## Goal
Per R2 cluster (a family of related criteria = one metric): GEPA-optimize a prompt seeded by its
`merged_description`, build Ω = family children + GEPA-mined + LLM free-gen, then measure
**reconstruction** (within-class cert, M = C(Ω), scoped to the family) and **coherence** (separate
diagnostic). Start at R2 (more distinct, fewer → lower bar).

## Decisions
- **M = C(Ω)** (full-Ω verdict), NOT an external/parent anchor. Prior finding: external M depressed
  I(M; M_Ω) when Llama-8 entropy was low → M_Ω is the settled target. ⇒ **no M-injection, no
  omega_certificate.py change.** Parent `merged_description` only seeds GEPA.
- **Ω = children + GEPA-mined + free-gen**, scoped to the family (no whole-task corpus).
- **Coherence** = separate metric (mean pairwise corr + participation ratio on the children signal
  matrix); distinct from reconstruction.
- **v1 scope**: full GEPA-per-R2 (user choice), but ~12 metrics on peer-review `specific` (GLM quota).

## GLM-4.7 audit (2026-06-25)
- **glm-4.7 is a valid `zai_anthropic` model**, live + has quota on `~/.z-ai-api-key-alexander-spangher.txt`
  (1-call probe: HTTP 200). 5.2 quota out until ~06-30.
- **Bug fixed:** `backends._read_key` read `~/.z-ai-api-key.txt` (MISSING) → run_real_test GLM path was
  broken. Now toggles: `ZAI_KEY_FILE`/`GLMCLUSTER_KEY` env → alexander-spangher → spangher → default.
- **v1 uses `--reconstructor-model glm-4.7`** (unblocks now, spares 5.2; reviser/reconstructor/free-gen
  are generation roles, executor X = Llama-8B so no same-family arbiter issue). A/B 4.7-vs-5.2 later.

## Progress
- [x] **T1 family_coherence** (`orthogonalize.py`): `family_coherence(X, tau_corr=0.10, tau_pr=2.0)`
  → {mean_pairwise_corr, participation_ratio, n_criteria, n_usable, coherent}. Selfcheck [g] PASS.
- [x] **T2 R2 mining** (`mine_clusters.py`): `r2_groups` / `r2_criteria` / `r2_children` + `mine(bucket=)`
  accepts R2. peer-review/specific = 130 groups (PRISMA 81, Open-data 79, Ethics 72 leaves).
- [x] **T3 free-gen + scoped Ω** (`run_real_test.py`): `_free_generate` (failure-informed, temp 0.7);
  `_candidate_pool` source (d) + `scoped_criteria`; `phase_b_certificate` passes both + returns
  `crits`/`signals` for coherence.
- [x] **T4 driver** (`run_r2_certificate.py`): per-R2-metric GEPA(seed=merged_description) → scoped-Ω
  cert → coherence; writes `{task}_{bucket}_results.jsonl`.
- [x] **T5 verify (dry-run)**: unit (coherence selfcheck, r2 round-trip) PASS; **E2E dry-run on
      code-review/specific PASS** — 2/2 metrics certified through OmegaCertificate (LARGE-Ω fallback)
      + missing-impact + coherence. Mock signals → numbers are noise, but orchestration is correct.

## Open issues / decisions before live v1
1. **peer-review has NO trial pool** (blocker for peer-review live). `cfg.text_column='text'`, but the
   only laptop pool is `pool_competitive_code.jsonl.gz` (cols: code/…, no `text`) → `KeyError('text')`.
   `_TRIAL_POOLS` covers only code-review/creative-writing/law. Options: (a) build
   `pool_peer_review.jsonl.gz` on sk3 from the real corpus; (b) run v1 on code-review (has pool +
   224 specific R2 groups) or creative-writing (general bucket only). **Needs user call.**
2. **K>15 → large-Ω fallback** (not exact). Orthogonalization keeps 29–36 units from ~60 candidates, so
   every metric hits the fallback (greedy/double-greedy, U₂ withheld for non-monotone R). To get EXACT
   certificates on some metrics, raise `--cmi-thresh` (keep fewer) or lower `--pool-max`. Tuning knob
   for live, not a bug.
3. **Live v1 needs sk3** (vLLM Llama-8B GPU + GLM quota). Can't run from laptop. Use
   `--reconstructor-model glm-4.7` (5.2 out till ~06-30).

## v1 RESULTS — code-review/specific × Llama-8B × GLM-4.7 (2026-06-25, 12/12 done)

**Headline: the pipeline is mechanically correct (GEPA→scoped-Ω→cert→coherence all ran 12/12
end-to-end) but the SIGNAL IS DEGENERATE. Reconstruction I(M,M_ω)≈0.005–0.01 (untrustworthy),
coherent=False on all 12. Root cause = criteria/pool DOMAIN MISMATCH, not a code bug.**

**The mismatch (smoking gun = the rubric files):** the R2 children are pulled from diverse
code-review rubrics spanning MANY languages — g0 criteria are C/C++ preprocessor ("avoid macro
definitions", "wrap macro bodies in do-while(0)"), g155 are CSS linting ("valid CSS code",
"selectors respect specificity", "stylelint-disable comments"). But the executor pool
`pool_competitive_code.jsonl.gz` is competitive-programming solutions (Python/C++/Java). So most
criteria don't *apply* to the items → Llama-8B emits **near-constant soft verdicts**: every
criterion base-rate ≈0.50 with std only 0.06–0.15 across all 60 items (it's nearly indifferent).
That degeneracy then (a) collapses orthogonalization 46→3–7 atomic units (near-constant columns
look redundant under CMI) and (b) zeros reconstruction (the GEPA prose p̂ is near-constant,
T_prose≈0.001 → ⚠ UNTRUSTWORTHY).

Per-metric (all EXACT brute-force, K=3–7, coherent=False):
| g | K | n_children | mean_corr | PR | mode |
|---|---|---|---|---|---|
| 0 | 3 | 56 | 0.481 | 3.69 | EXACT |
| 31 | 5 | 49 | 0.382 | 5.25 | EXACT |
| 22 | 3 | 33 | 0.507 | 3.25 | EXACT |
| 94 | 7 | 29 | 0.417 | 4.43 | EXACT |
| 30 | 3 | 36 | 0.515 | 3.21 | EXACT |
| 26 | 4 | 16 | 0.450 | 3.71 | EXACT |
| 64 | 4 | 32 | 0.395 | 4.88 | EXACT |
|166 | 3 | 20 | 0.488 | 3.40 | EXACT |
| 49 | 6 | 20 | 0.351 | 4.76 | EXACT |
| 50 | 7 | 26 | 0.390 | 4.90 | EXACT |
| 21 | 5 | 19 | 0.389 | 4.45 | EXACT |
|155 | 5 | 29 | 0.465 | 3.84 | EXACT |

**|Ω| status:** candidate POOL = 16–56 (the `n_children` column; ≈30 target met/ exceeded at the
pool level). Post-orthogonalization K = 3–7 — this is a **symptom of the degenerate signal**, not a
pool-size shortfall: near-constant columns are mutually redundant under CMI, so the filter keeps few.
On real (discriminating) signal K should rise; peer-review is the test that disambiguates.

**Coherent=False for all 12** because PR (3.2–5.3) exceeds τ_pr=2.0 even though mean_corr (0.35–0.52)
clears τ_corr=0.10 → these R2 families look multi-dimensional on the (degenerate) signal, not single
constructs. Again likely a degeneracy artifact — to re-check on peer-review's real signal.

**Takeaway / next:** code-review is a *poor* R2-cert test bed (criteria don't map onto the competitive-
code pool). **peer-review is the deconfounded test** — homogeneous review text, criteria like
EQUATOR/reporting-guidelines adherence map naturally onto reviews. If peer-review shows real signal
(larger K, nonzero I(M,M_ω), some coherent=True) → confirms code-review was the mismatch, not the
pipeline or Llama-8B. **peer-review v1 LAUNCHED 2026-06-25 ~06:35** (GPU0, same config), watcher
armed. See [[feedback_apples_to_apples_dense_vs_baseline]] — never compare criteria vs items across
mismatched domains.

## v1 RESULTS — peer-review/specific × Llama-8B (2026-06-25, 12/12 done) — HYPOTHESIS FALSIFIED

**peer-review is ALSO degenerate** (K=2–8, coherent=False all 12, I(M,M_ω) mostly 0.001–0.09, a few
reach 0.17/0.22). So the "criteria-pool mismatch" explanation for code-review is **not the root cause**
— the cleaner domain didn't fix it. Metrics: PRISMA, Open-data, Ethics, Statistical-methods, STROBE,
ARRIVE — criteria that DO map onto review text.

**Corrected root cause = EXECUTOR WEAKNESS (Llama-8B near-constant P(YES)).** Diagnostic on the saved
signals (`peer-review_specific_g{N}.npz.missing_impact.json` → `per_criterion_spread`): the **soft
P(YES) std is ~0.08 mean / ~0.15–0.21 max** across items — and the codebase's OWN discrimination bar is
**std≥0.12** (`build_score_matrix.py:89`). So most peer-review criteria fall below it → Llama-8B hovers
its verdict in a narrow band (≈0.42–0.58) → near-constant → orthogonalization collapses 49–60→2–8 →
reconstruction noise-limited. 7–9/~50 criteria per metric are fully collapsed (std<0.02).
**NOTE on `base_rate`: it is `mean(_median_split(py))` — tautologically ≈0.50 for EVERY criterion by
construction (median split). It is NOT a signal-strength measure; the std of the soft `py` is.**

**Implication:** the fix is a STRONGER executor (sharper opinions → higher std → larger K → real
reconstruction) — exactly the "many different language models" directive. → 70B run below.

## 70B executor A/B — peer-review × Llama-3.3-70B-Instruct-FP8 (2026-06-25, RUNNING)

Same 12 metrics / config as the 8B run — only the executor changes → clean A/B for the
executor-strength hypothesis. **GPU0** (VLLM_GPU_MEM_UTIL=0.55 → 103GB: 70GB weights + 33GB KV,
alongside ahmedah's 65GB, ~13GB free — TIGHT, watch OOM).

**70B tokenizer gotcha (cost ~20min):** the FP8 snapshot has **only weights, NO tokenizer files**
(`.../snapshots/fde04ee…/` has no tokenizer.json/tokenizer.model). miniconda3 transformers can't fetch
them → `ValueError: Couldn't instantiate the backend tokenizer` on EVERY metric (the driver reloads the
model per metric, so all 12 failed instantly at init). **FIX: pass the local SNAPSHOT PATH as
`--target-model`** (NOT the hub id `nvidia/Llama-3.3-70B-Instruct-FP8`) — vLLM then resolves the
tokenizer itself (mirrors cw_pump's `scripts/sk3_v2_judge_runner.py`, which does `LLM(model=MODEL_DIR)`
with the snapshot path and works). Verified: `init engine took 32.8s`, scoring underway. The gemma4 env
can ALSO load the tokenizer (hub-id) but may lack codebase deps — snapshot-path + miniconda3 is cleaner.

> **CORRECTION 2026-06-25:** the "tokenizer gotcha" above was a RED HERRING. The `/hub/` mirror path I
> used is **INCOMPLETE (12/15 shards)** → `'!!!'` garbage → score_binary all-NaN. The REAL fix is the
> **canonical non-`/hub/` path (15 shards + tokenizer)**:
> `/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee…` (NO
> `/hub/`). Verified coherent. See the PIVOT section below.

## ⭐ PIVOT 2026-06-25: within-class cert was WRONG METRIC → RECOVERY C(R(Ω))

**The whole within-class certificate above (OmegaCertificate: "can a subset of Ω re-express the full
verdict", I(M,M_ω)/R(C(S))/T_prose) was measuring the WRONG thing.** User correction (2026-06-25): the
goal is **RECOVERY / articulability = C(R(Ω)) = I(M_ω; M′)** — given executor X's behavioral labels
M_ω(x), can a strong LLM (GLM) articulate a rule M̂ that, when X re-applies it, reproduces those labels?
High = articulable; low = tacit. This is `recon_channel.run_metric(mode="free")` → `iv_transmission`.
**Do NOT report the within-class numbers as the result** ([[feedback_report_recovery_metric_only]]).

**What changed (all done + validated):**
1. **`run_r2_certificate.py` is now LEGACY** (within-class). New canonical driver:
   **`experiments/run_r2_recovery.py`** — per R2 metric: M_ω = X's verdict on `merged_description`
   (holistic construct; `--target conjunction` for the old AND-of-children target) → GLM induces M̂
   (`induce_free`, R reconstructions) → X re-executes M̂ on held-out → records `recovery` = I(M_ω;M′)
   + M_ω P(YES) std (discrimination flag, floor 0.12) + induced rules (eyeball) + consistency.
2. **`seed=` blocker fixed** (`backends.py`): `LLMBackend.generate_batch` now takes `seed=None`
   (dropped — Anthropic API has no seed; diversity via temperature). `recon_channel.induce_*` can now
   target the GLM API reconstructor. Validated live: GLM induced articulate rules.
3. **70B executor unblocked** — `score_binary` was all-NaN because I used the INCOMPLETE `/hub/` mirror
   (12/15 shards → `'!!!'` garbage). FIX: canonical non-`/hub/` path (15 shards + tokenizer).
   Verified coherent gen (`'YES'`) + clean score_binary (`[1.0, ~0, 1.0]`).

**Llama-8B finding (now correctly framed via recovery):** PRISMA on reviews → M_ω mean=0.092, std=0.084
(<0.12 floor) → near-constant label → recovery=NaN (no signal to recover; NOT "tacit"). 8B is too weak
a discriminator for these R2 metrics. → 70B is the executor that should discriminate.

**RUNNING 2026-06-25 ~12:15 (nohup, survives disconnect), sk3 GPU0:** peer-review/specific ×
Llama-3.3-70B (canonical path) × GLM-4.7 inducer, 12 metrics, R=3 → `outputs/r2_recovery/peer-review-70b/`,
log `logs/r2_recovery/peer-review-70b.log`. PID 549619. Watcher armed. **RESUME HERE**: check that log +
results jsonl; report `recovery`=I(M_ω;M′) per metric (+ M_ω std). If 70B discriminates (std≥0.12) → real
C(R(Ω)); then sweep code-review + more R. If 70B ALSO degenerate → criteria/items mismatch is fundamental
(manuscript criteria on review text).
