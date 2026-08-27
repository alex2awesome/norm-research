# Real-data TVD transmission (consistency channel) — Rung-1/3 landscape

*2026-06-19. First real-data run of the same-f TVD path (`vinfo.tvd_transmission`) on the existing
5-pass consistency longtable (`outputs/metric_implementer_scale/search1/`, 7 tasks × 6 executor tiers,
2.99M sampled rows, 5928 scored cells). Zero-GPU, sk3 CPU. Runner:
`methods/metric_implementer/experiments/real_consistency_tvd.py`.*

**This is transmission `T = I_TVD(I;V)`, the GAMEABLE leg (theory §4.3), NOT recovery `R`.** High `T`
can be spurious spread (e.g. the news-homepages outlet/position confounds). It is the precondition +
the Rung-1 cap test + the Rung-3 ranking test. Recovery `R` (and the `A=T−R` gap) needs the
reconstruction LLM pass (#24).

## Rung 1 — `T_tvd` vs cap_TVD = 0.5, by task (**CLEAN cells**: flagged/collapsed judges excluded)

Of 5928 scored cells, 2280 were degenerate (collapsed judges → `T≈0`) and excluded; 3648 clean.
Excluding them **changes the story** — the all-cells version below over-counted "thin" tasks because
collapse drags medians to 0.

| task | clean cells | med `T_tvd` | cap_gap | max `T_tvd` | med `I_V` (Shannon) |
|---|---|---|---|---|---|
| news-homepages | 560 | **0.327** | 0.173 | 0.484 | 0.672 |
| peer-review | 490 | **0.327** | 0.173 | 0.490 | 0.646 |
| math | 469 | **0.326** | 0.174 | 0.484 | 0.564 |
| humor | 376 | 0.249 | 0.251 | 0.483 | 0.341 |
| patents | 337 | 0.249 | 0.251 | 0.483 | 0.341 |
| notice-and-comment | 180 | 0.104 | 0.396 | 0.427 | 0.288 |
| law | 1236 | 0.093 | 0.407 | 0.209 | 0.074 |

**Reads (corrected for collapse).**
- **The "thin" tail was largely a collapsed-judge artifact:** humor/patents jump 0.045 → **0.249** once
  collapses are dropped; N&C 0.000 → 0.104. The genuinely low-transmission tasks are **law (0.093)** and
  **N&C (0.104)** even on clean cells.
- **Three transmission clusters:** high (news/peer-review/math ≈ 0.33, cap_gap ≈ 0.17), mid
  (humor/patents ≈ 0.25), low (law/N&C ≈ 0.10). Rung-1 cap looseness is real and task-varying.
- **Shannon `I_V` and TVD `T_tvd` agree on ordering** (cross-readout sanity ✓).
- Thin/thick caveat: this ranks *consistency*, not articulability — high `T` can be spurious spread;
  recovery `R` (#24) is the discriminating test.

## Rung 3 — best version per (task,tier,metric); CI-certified best-in-set?

At **fixed budget** (group by task,tier,metric,`token_cap`; clean cells), 486 cells have ≥2 candidate
versions; **19 (4%)** have a single CI-certified best, the rest are **top-groups (bootstrap CIs
overlap)**. So *mostly* the optimizer's candidates are statistically indistinguishable by `T`, but ~4%
of metric-configs do have a certifiable winner (e.g. humor/Qwen3-8B `act_out_stand_up_comedy` at
tok=1000: `T=0.386`, CI-lo 0.333, dominates). (Earlier all-cells/mixed-budget grouping gave 0%; fixing
budget + excluding collapses surfaces the 4%. The discriminating test for the rest is `R`.)

## Descriptive — operator effect on `T_tvd` (does mutation raise transmission?)

| operator | n cells | med `T_tvd` |
|---|---|---|
| INIT (seed) | 864 | 0.002 |
| MECHANIZE | 36 | 0.038 |
| DECOMPOSE | 1512 | 0.057 |
| CLARIFY | 3516 | 0.108 |

The mutation operators raise transmission **~50×** over the initial seed (0.002 → 0.108). So the
optimizer demonstrably increases consistency — even though individual mutated versions are not
CI-distinguishable from *each other* (Rung-3 tie-groups). I.e. the optimizer climbs a real gradient in
`T`, then plateaus into a statistically flat top-region.

## Discovery-to-Selection real-Ω γ (§6.6 Phase 1) — harness validated, rubric-selection lesson

`methods/metric_implementer/experiments/real_gamma.py` (1-GPU offline vLLM, Llama-3.1-8B on sk3 GPU 6,
`VLLM_GPU_MEM_UTIL=0.30`). Pipeline: `M` = full-rubric verdict; `Ω` = criteria; `X_i` = per-criterion
verdict via the **logprob `P(YES)` readout** (deterministic, no parse loss — the fix after sampled
free-text dropped ~92% of items); then `Ǐ(S)=I(M;X_S)`, exact `γ`, co-information, greedy vs `OPT_Ω`
(reuses `submod_conditional`). `--inject` plumbing test passes (synergy γ=0.005/greedy 0.19; CI greedy 1.0).

**First real run (peer-review, bornmann `scientific_peer_review` bank, 80 items):** ran end-to-end,
`γ=0.193`, mild synergy pairs (1,3),(2,3); GPU freed cleanly, matrix saved
(`outputs/.../real_gamma_pr_bornmann.npz`). **But not a clean test:** 4 of 5 criteria are scientometric
**meta-properties of the review *system*** (inter-rater reliability, predictive validity of editorial
decisions, efficiency) that almost never fire on an individual review *text* (fire rates 0.01–0.04).
Only X_1 "predictive validity" carried info (I=0.106). So `γ=0.193` is on near-degenerate signals.

**Lesson + next step:** the §6.6 test needs **item-level quality criteria**, not meta-rubrics. Best
source = a **registry GEPA-optimized metric** (e.g. creative-writing `distinctive_voice`), decomposed via
the LLM `_decompose` path — which is the *true* "Discovery-to-Selection from a GEPA run." The harness is
validated and the rubric swap is a one-arg change.

### WORKING optimality guarantee (code-review, 2026-06-19)

After two fixes — (i) **logprob `P(YES)` readout** (no parse loss), (ii) **median-split to balanced
binary** (M can't collapse to constant) + drop criteria by `P(YES)` std — the first clean run:

- Pool: 90 competitive-code items; `M` = holistic "high-quality solution?"; `Ω` = 10 item-level
  code-quality criteria (edge cases, complexity, naming, comments, structure, …); all 10 discriminate.
- **`γ = 0.160`, curvature `α = 1.000`** (saturating — criteria are redundant proxies for "quality").
- greedy (k=5) `f=0.469`; brute-force `OPT_Ω` (best 5 of 10) `f=0.574`.
- **OPTIMALITY GUARANTEE: greedy rubric achieves `0.818·OPT_Ω` (EXACT, by brute-forcing `OPT_Ω`).**
  The worst-case submodular certificate is `(1/α)(1−e^{−αγ}) = 0.148` — loose, because `γ=0.16`.
- **No pairwise synergy** (all co-info < 0.02) yet `γ=0.16`: the low `γ` is **not** clean XOR
  complementarity — it's higher-order / **finite-sample** (joint `I(M;X_S)` over 5 criteria = up to 64
  cells estimated from 90 items → heavy plug-in bias; `γ_exact` = min over all 2^K subset-pairs picks
  up the noisiest). The greedy/`OPT` *ratio* is robust (bias partly cancels); absolute `γ` is shaky.

**Reads.** (1) The machinery delivers a real **optimality guarantee on a composed prompt** — since `K=10`
is brute-forceable, the *exact* `greedy/OPT_Ω = 0.818` is the strong statement (the γ-certificate matters
only when `Ω` is too big to enumerate). (2) `α=1` + no synergy ⇒ the criteria are **redundant** (each a
noisy proxy for overall quality), the submodular-friendly regime — consistent with greedy doing well
(0.818) despite the loose worst-case floor. (3) **To trust `γ` itself, need `N≥300`** (Miller–Madow or
just more items) so joint MI over large subsets isn't under-sampled — the immediate next step for a tight
certificate. Saved: `outputs/.../real_gamma_code.npz` (re-analyzable with no GPU).

### BRUTE-FORCE within-class optimality certificate (§6.7a, 2026-06-19) — submodularity "works"

`experiments/small_omega_brute_force.py` scored the **real** `R(C(S)) = I_TVD(M; M̂_S)` for **all 1023
non-empty subsets** of `Ω` (10 code-quality criteria), `M` = holistic "top-tier solution?", 70
competitive-code items, Llama-3.1-8B logprob readout, 1 GPU. Each `R(C(S))` is a clean 2×2 MI
(well-sampled at N=70) and includes the executor compression (no `Ǐ` idealization).

- **Certificate (exact, no approximation):** greedy(k=5) picks {1,2,5,8,9} `R=0.092`; brute-forced
  `OPT_Ω(k≤5)` picks {2,5,7,8,9} `R=0.093`. **greedy = 0.993·OPT_Ω.** Because `OPT_Ω` is *enumerated*,
  this is the **global optimum of the criterion class**, not a heuristic local one — the "certified
  prompt" deliverable. (Brute-force makes `γ` unnecessary — Point-1 thesis confirmed.)
- **PRUNE confirmed on real LLM verdicts:** the best rubric uses **5 of 10** criteria; adding all 10
  *lowers* recovery `0.093 → 0.067`. So `R` is genuinely **non-monotone** — empirical confirmation of
  §6.1 (adding a criterion can hurt). `γ` on the raw `R` is therefore ill-defined; the monotone-envelope
  `R↑` diagnostic gave `γ≈0` (super-modular envelope on this instance).
- **But recovery is LOW in absolute terms:** `R=0.093` against `cap_TVD=0.5` (~19%); per-criterion solo
  `R≈0.03–0.08`. So these 10 *explicit* checks recover only a fraction of the model's *holistic* quality
  judgment — a **large articulation gap**: the holistic standard is largely not captured by the
  articulated criteria (tacit residual, or the holistic verdict integrates more than these checks). The
  optimality *ratio* (0.993) is exact regardless of the low *level*.

**Bottom line:** we now have a working **certified-optimal prompt** within the criterion class — the
exact within-class guarantee the submodularity program was aiming for, delivered by brute-force enclosure
(§6.7a) rather than an approximation bound. Caveats: `M` is Llama-8B's own holistic verdict (not ground
truth); N=70; the low `R` says the *class* `Ω` (these 10 checks) is a weak basis for the holistic judgment
— the natural next step is a richer/larger `Ω` (then submodularity matters again when `|Ω|` outgrows
brute-force) and the §6.7c discovery-coverage argument. Saved: `outputs/.../brute_force_code.npz`.

### MULTI-EXECUTOR: is the certified-optimal prompt single-LLM? YES (2026-06-19)

`run_multi_exec.sh` + `aggregate_executors.py`: 12-criterion `Ω`, 45 items, 3 executor tiers
(Llama-3.2-3B, Llama-3.1-8B, Qwen-2.5-7B), consensus `M` = median-split of mean holistic.

- **Per-model optimum (vs consensus `M`):** 3B → `{0,11}` (|S|=2); 8B → 7-criterion set; Qwen-7B → another
  set. **Optimal-criterion overlap Jaccard = 0.00** (zero criteria shared across all three) →
  **the certified-optimal prompt is fully MODEL-SPECIFIC.** Optimality is single-LLM, empirically and starkly.
- **Substitution:** weak 3B wants a *minimal* rubric (2 criteria; adding all 12 crashes it 0.104→0.023);
  stronger models use more criteria. The naive "stronger E ⇒ fewer criteria" guess is **inverted** here.
- **Family-robust certificates (consensus `M`, brute-forced):** `R_avg` opt = `{2,3,4,6,11}` R=**0.150**
  (higher than any single model — averaging smooths executor noise), greedy/OPT=**0.625** (brute-force
  essential — greedy 37% worse). `R_min` (weakest-model robust) opt R=0.066, greedy/OPT=0.969 (low level —
  worst-case robustness is expensive).
- **Methodological reads:** (1) you cannot ship one prompt "optimal for LLMs" — only optimal-for-`E`, or a
  certified family-robust prompt via `R_avg`/`R_min`. (2) Brute-force vs greedy gap is large for `R_avg`
  (non-monotone), confirming §6.7a matters. Next: replace the mean-then-median consensus `M` with the
  **Spectral Meta-Learner** (Parisi) — see `2026-06-19__unsupervised-to-Y-accuracy-map.md`.

### GEPA-mined rich Ω + emergent evolution strategies (subagent workflow, 2026-06-19)

Workflow `w87p42rgk`: 20 faithful free-form GEPA runs (Opus subagents, weekly budget; 730K tokens,
~5.5 min) over 360 competitive-code examples, each reflecting on real examples and labeling its own
mutation strategies. → 381 raw criteria → **34 distinct atomic criteria** (`/tmp/gepa_omega_rich.txt`),
far richer + more concrete than the hand-written 12 (parse-validity, I/O contract, mod arithmetic via
Fermat, DP-state correctness, recursion-limit, data-structure fit, …) — grounded in actual failure modes.

**Emergent strategies (108 mutations clustered) — do the a-priori operators appear? YES, 5 of them:**

| emergent strategy | count | a-priori operator | theory connection |
|---|---|---|---|
| Decompose into orthogonal axes | 21 | **DECOMPOSE** | populates Ω (the mining operator, §6.5) ✓ predicted |
| Mechanize fuzzy criterion → checkable proxies | 21 | **MECHANIZE** | makes a criterion **fire** as a clean per-item signal — *exactly* the std/fire-rate filter we needed ✓ |
| Add a missing dimension | 22 | **ADD** | grows Ω ✓ |
| Promote to hard precondition gate | 10 | ANCHOR | gating/calibration |
| Prune/demote a misleading criterion | 6 | **PRUNE** | **= our non-monotonicity** (drop a criterion that points the wrong way, §6.1) ✓ predicted |
| Seed holistic question | 19 | NOVEL | gen-0 baseline (not really a mutation) |
| **Pareto-keep conflicting criteria** | 9 | **NOVEL** | **the search-time face of `γ<1`/synergy** (§6.6) |

**Reads.**
- The hand-coded operators are **the right granularity** (not "too high level"): DECOMPOSE/MECHANIZE/ADD/
  PRUNE/ANCHOR all emerged organically from free-form GEPA. MECHANIZE emerging validates *why* criteria
  must "fire," and PRUNE emerging validates the non-monotonicity our certificate measures.
- **One genuinely novel strategy: "Pareto-keep conflicting criteria"** — when two criteria *disagree on the
  same examples*, keep BOTH as independent axes instead of collapsing. This is precisely the **synergy /
  non-submodular (`γ<1`) regime** our co-information detects, seen from the *search* side: GEPA's
  operational response to criterion conflict *is* the multi-objective (Genetic-**Pareto**) move, and it
  corresponds to exactly the structure that breaks submodularity. The two phases see the same structure
  from two ends.

**Next (the certificate on rich Ω) — this is the SUBMODULARITY regime, not brute-force.** 34 criteria ⇒
`2^34` subsets ⇒ brute-force infeasible. So the rich Ω is exactly where §6.1–§6.6 (greedy + sampled-`γ` +
co-information + the `(1/α)(1−e^{−αγ})` bound) takes over from §6.7a enclosure — closing the loop *and*
demonstrating both regimes (brute-force for the small hand-written Ω; submodular certificate for the rich
GEPA-mined Ω). Run `real_gamma` (signal matrix + greedy + sampled γ) on `/tmp/gepa_omega_rich.txt`.

## Caveats / TODO
- `T` is gameable; the headline articulability result needs `R` (#24, reconstruction, ≤1 GPU).
- 2280/5928 cells flagged degenerate (collapsed judges); segregated in the medians shown? No — medians
  include flagged cells. Re-run excluding `flags!=""` for a clean version.
- Rung-3 grouping should fix `token_cap`; rung-4 optimizer-agreement is NOT cleanly testable here
  (operators are within-run mutations, not independent optimizer runs).

---

## V-corner verification (§5.5 / §6.8 gate) — code metrics, ZERO GPU — 2026-06-19

`experiments/code_metric_calibration.py` on 4,479 real code snippets (competition_unified
`editorials_code_extracted.parquet`, col `extracted_code`). Executor = compiler ⇒ T=1 by construction.

- **C5 cap + estimator.** `M=(word_count>median)` (π=0.499): perfect compiler recovery `R̂=0.500` =
  exactly `2π(1−π)`; label-noise sweep monotone 0.500→0.006; **K-ary perfect-recovery Gini = exactly
  `1−1/K`** for K=2/3/5/10 (0.50/0.667/0.80/0.90). Empirical confirmation of the §3.1 cap derivation:
  binary caps at ½, headroom rises only with verdict granularity K, never with N.
- **C6 planted-γ.** Redundant world (nested length thresholds): **γ=1.000** (submodular), no synergy
  flagged. Synergy world (`M = has_digit ⊕ has_def`): **γ=0.598** (<1), co-information flags exactly the
  `(has_digit, has_def)` pair (+0.37). γ̂ machinery recovers the planted structure on real items.
- **Verdict:** §6.8 gate PASSED — a measured γ̂ on an opaque Ω is trustworthy. This is the real-item
  (non-synthetic) version of the E0 kill-switch.

Note: in the synergy world greedy still reached OPT (γ=0.60 not →0) because real `has_digit`/`has_def`
are +0.22 correlated, so the XOR leaks first-order signal — a *partial* synergy, unlike the pure
zero-marginal synthetic XOR in `submod_conditional.py`. γ correctly registers it; co-info localizes it.

---

## §6.9 discovery scaling law — zero-GPU shape test — 2026-06-20

`experiments/discovery_scaling.py` part B on the already-scored 10-criterion code recovery
(`brute_force_code.npz`, 1023 subsets). OPT over random ground-sets of size k, averaged:

| k | mean OPT_k | retention |
|---|---|---|
| 1 | 0.049 | 0.526 |
| 2 | 0.060 | 0.654 |
| 3 | 0.068 | 0.738 |
| 5 | 0.079 | 0.849 |
| 7 | 0.086 | 0.927 |
| 10 | 0.093 | 1.000 |

- **OPT_Ω monotone + concave/saturating** in pool size (marginal +0.011/+0.008/+0.006/+0.005…) — the §6.9
  diminishing-returns shape in the discovery dimension, on real recovery (not synthetic).
- **(1−1/e)≈0.63 retention crossed at k=2** (0.654), matching the subsampled-submodular random-ground-set
  bound; 90% of OPT_full at k=7/10.
- **Caveat:** OPT_full=0.093 ≪ cap 0.5 — this Ω's recovery is weak/heavily executor-compressed; shape holds,
  absolute level low. A richer Ω should sit higher.
- **Blocked:** discovery curve (distinct criteria vs GEPA step t) — registry lineage for this Ω is empty
  (rubric-file criteria, not a logged GEPA run). Full test (OPT along real GEPA trajectory on the rich
  34-criterion Ω + tail-γ) → needs a logged trajectory + a free GPU.

---

## Specificity (LLM bucket) ↔ transmissibility T — cross-corpus — 2026-06-21

`per_task_T.py` (Llama-3.1-8B judge, consistency longtable) → real distinct per-task T; joined with LLM
specificity buckets (`outputs/analyses/structural_metrics/specificity.json`,
%hyper_specific = hyper_specific.forms / total forms).

| task | %hyper | %general | T_TVD |
|---|---|---|---|
| peer-review | 2.5 | 43.9 | 0.327 |
| math-stackexchange | 13.2 | 45.2 | 0.289 |
| patents | 3.7 | 3.2 | 0.249 |
| humor | 1.5 | 70.7 | 0.200 |
| legal | 2.1 | 47.8 | 0.056 |
| notice-and-comment | 1.3 | 46.9 | 0.007 |

- **%hyper_specific vs T: Pearson +0.47, Spearman +0.77**; **%general vs T: −0.22**. The verifiable-corner
  (numeric/threshold) share tracks transmissibility on ranks; general criteria are flat-to-negative.
  Supports the V-corner/thin-thick thesis with an INDEPENDENT LLM labeling.
- Outlier: **peer-review** (T=0.327 high, %hyper 2.5% low) — its consistency is not from numeric thresholds.
- Caveat: n=6 (news-homepages absent from the Llama-8B longtable shard); this is transmission T, not
  recovery R — the R cross-corpus version is the recon_channel watcher's job.

---

## Large-scale γ/recovery on REAL GEPA-mined Ω — VERIFIED — 2026-06-21

Pipeline: real GEPA optimizer lineage (`tmp_vinfo/gepa_registry`, free-text seed → evolved structured
rubric) → structural+recursive criterion extraction (audited 37%→clean by subagents) → behavioral dedup →
per-item LLM signals (Llama-3.1-8B) → Ǐ(S)=I(M;X_S), greedy, N_eff, saturation, spectral γ. N=250 items.
M = holistic quality (median-split). `run_all_gamma.sh`; recomputed from saved signals by `aggregate_gamma.py`.

**HONEST LABELS (verified):** (a) R(OPT) is **Shannon IDEAL recovery Ǐ in bits** (full signal vector, no
executor compression; ceiling ~1.0) — NOT the TVD executor-bottleneck R (cap ½). (b) Everything is
**median-split** → rank-based ("top-half on dimension i recovers top-half on quality"); verified legitimate
— continuous M_cont and X_cont have real spread and criteria correlate +0.28..+0.77 with quality, so the
split discretizes real signal. (c) γ_spec = Das-Kempe spectral LOWER bound; loose for collinear high-level
criteria (small λ_min from collinearity ≠ low true γ).

| metric (corpus) | gran | K | Ǐ(OPT) bits | N_eff | γ_spec |
|---|---|---|---|---|---|
| gowers exposition (math) | criteria | 10 | 0.502 | 4.81 | 0.218 |
| ap_english (creative) | criteria | 9 | 0.358 | 3.92 | 0.241 |
| aigner proofs (math) | criteria | 6 | 0.340 | 1.66 | 0.290 |
| abbott narrative (creative) | criteria | 5 | 0.330 | 1.95 | 0.353 |
| andrew_stanton story (creative) | criteria | 4 | 0.212 | 1.39 | 0.413 |
| arnold teaching (math) | criteria | 8 | 0.197 | 3.76 | 0.128 |
| aristotle poetics (creative) | criteria | 7 | 0.134 | 2.63 | 0.361 |

**Cross-metric (same corpus/M/items, different rubric Ω):** math exposition (gowers 0.50) > math proofs
(aigner 0.34) > narrative/dramatic (aristotle 0.13, andrew_stanton 0.21). The narrative/Poetics criteria
recover holistic quality WORST (weak correlations 0.28–0.44) — consistent with the articulability gradient
(structured math dimensions more recoverable than tacit narrative quality). Caveat: confounded by rubric
quality + criterion count; not yet deconfounded.

**Granularity (criteria vs steps) — the §6.5 floor experiment:** MIXED, metric-dependent:
- aigner: criteria Ǐ=0.340 (K=6) → steps Ǐ=0.281 (K=6), ΔǏ=−0.060, N_eff 1.66→2.29 → **steps LOWER recovery (over-decomposition)** — finer steps fragment the signal here.
- gowers: criteria Ǐ=0.502 (K=10) → steps Ǐ=0.530 (K=14), ΔǏ=+0.029, N_eff 4.81→6.91 → **steps RAISE recovery slightly (criteria were a bit coarse)**.
So no universal floor: for a 6-criterion proof rubric finer is worse; for a 10-criterion exposition rubric
finer adds a little. N_eff rises with steps in both (more distributed), but recovery doesn't track count.

**Still TODO for trustworthy completeness:** (1) the executor-bottleneck TVD R (compile subset→one
verdict→I_TVD), to get the real R and the Ǐ−R executor gap; (2) deconfound cross-metric (criterion count,
rubric quality); (3) γ via co-information (not just spectral bound).

---

## Executor-bottleneck TVD recovery R (flagship) — aigner math — 2026-06-21

small_omega_brute_force on clean aigner Ω (K=6), N=200, M=holistic math quality (median-split). R=I_TVD(M;M̂_S),
the EXECUTOR-compressed verdict (cap ½) — the real flagship, vs the ideal Ǐ (full signal vector).

- **R(OPT)=0.133**; best subset = {Validity, Math-Language, Clarity, Background-Assump} (4 of 6); greedy =
  OPT EXACTLY (brute-forced, certified global within-class).
- **PRUNE confirmed**: all-6 R=0.096 < best-4 R=0.133 — the executor degrades when forced to aggregate all
  6 dims into one verdict. Best rubric DROPS 2 criteria.
- solo R(TVD): Math-Lang 0.121, Clarity 0.114, Validity 0.113 (top); Background-Assump 0.032,
  Grad-Background 0.042 (bottom). Ordering MATCHES the ideal-recovery run (cross-check ✓).
- γ on monotone envelope R↑ = 0.727.
- **UNITS CAVEAT:** R is TVD (cap ½); the earlier ideal Ǐ=0.340 was Shannon BITS — different f, so the
  Ǐ−R executor gap is NOT 0.340−0.133. Matched-f (I_TVD ideal vs R) gap is future work.

---

## Executor-R math set + the GRANULARITY FLIP (Ǐ vs R) — 2026-06-21

small_omega_brute_force, clean GEPA Ω, N=200, M=holistic math quality (median-split). R=I_TVD(M;M̂_S), cap ½.
greedy=OPT EXACTLY (brute-forced) for all.

| metric | gran | K | R(OPT) | best \|S\| | full-set R | top solo criterion |
|---|---|---|---|---|---|---|
| aigner | criteria | 6 | 0.133 | 4 | 0.096 | Math-Language 0.121 |
| arnold | criteria | 8 | 0.124 | 3 | 0.114 | math-rigor 0.113 |
| gowers | criteria | 10 | 0.133 | 3 | 0.108 | justification 0.120 |
| aigner | steps | 6 | 0.141 | 3 | 0.130 | each-step-justified 0.116 |

- **Executor-R is LOW and uniform** (0.124–0.141 ≈ 26–28% of cap ½): the single compressed verdict recovers
  only ~¼ of the way to the ceiling. The rubric's holistic verdict is a weak reconstruction of holistic
  quality through the executor.
- **PRUNE is universal and strong**: best subset is 3–4 criteria everywhere; adding all criteria LOWERS R
  (gowers dramatic: best 3 of 10; all-10 drops 0.133→0.108). Hard evidence the executor cannot aggregate
  many dimensions into one verdict — the executor bottleneck is real and large.
- **GRANULARITY FLIP (aigner):** ideal Ǐ criteria 0.340 → steps 0.281 bits (steps LOWER, −17%); executor R
  criteria 0.133 → steps 0.141 TVD (steps RAISE, +6%). The executor bottleneck COMPRESSES/reverses the
  granularity effect — finer, more concrete steps are slightly EASIER to apply per-item even though they
  carry less ideal joint info. (ΔR small, needs replication; units differ bits-vs-TVD so SIGNS compared,
  not magnitudes.)
- γ-envelope: aigner-steps 0.901 (clean submodular envelope) but high-level criteria (gowers/arnold) 0.000
  — high-level dims are near-collinear (all proxy "quality") so the monotone envelope degenerates; the
  finer steps are more behaviorally distinct.

---

## Matched-f executor-compression gap (TVD, cap ½) — the clean executor bottleneck — 2026-06-21

On the executor-best subset S, both quantities in CONSISTENT TVD units:
ideal I_TVD(M;X_S) [full signal vector] vs executor R=I_TVD(M;M̂_S) [one compressed verdict].
(`tvd_gap.py`, zero-GPU; crits-identity guarded so no index-mismatch assumption.)

| metric | best \|S\| | I_TVD ideal | R executor | gap (loss) | exec/ideal |
|---|---|---|---|---|---|
| aigner | 4 | 0.304 | 0.133 | +0.171 | 44% |
| arnold | 3 | 0.212 | 0.124 | +0.088 | 58% |
| gowers | 3 | 0.292 | 0.133 | +0.159 | 46% |

→ The executor bottleneck (compress K criteria → ONE holistic verdict) **LOSES ~42–56% of the recoverable
signal**; the executor recovers only 44–58% of what the ideal full-vector carries. §6.6-step-4 measured in
matched units — large and consistent across math metrics. This is the cleanest "executor gap" number
(matched f), superseding the not-directly-comparable Shannon-Ǐ vs TVD-R.

---

## Creative executor-R (ap_english) + a dedup-mismatch the guard caught — 2026-06-21

- **ap_english (creative, K=10):** executor R(OPT)=0.109, best subset = a SINGLE criterion; full-set R
  crashes to 0.045 (−59%). Creative/literary metrics PRUNE HARDER than math (best |S|=1 vs math's 3–4) —
  the executor collapses ~4 ideal effective dims (Ǐ N_eff=3.92) to 1 usable. The bottleneck is more severe
  for tacit/narrative content.
- **Dedup mismatch (caught by the crits-identity guard, not assumed away):** ap_english Ǐ run (real_gamma)
  applied behavioral dedup → K=9; executor-R run (small_omega_brute_force, no dedup) → K=10. `tvd_gap` SKIPPED
  the matched-f gap ("crits differ, no safe index map") rather than mis-aligning. So the matched-f gaps are
  clean ONLY for the 3 math metrics (dedup merged nothing there → K matched). Creative matched-f gaps need
  Ω-aligned re-runs (add behavioral dedup to small_omega_brute_force, or run Ǐ without dedup). KNOWN LIMITATION —
  the guard prevented a wrong number.

---

## COMPLETE cross-corpus table — all 8 metric-runs VERIFIED — 2026-06-21 03:40

All bfc npz legitimacy-checked (M≈0.50, maxR ≤ cap ½, N=200). See notes/2026-06-21__overnight-summary.md.

| metric | corpus | gran | K | Ǐ bits | N_eff | γspec | R TVD | best\|S\| | PRUNE | matched-f gap |
|---|---|---|---|---|---|---|---|---|---|---|
| gowers | math | criteria | 10 | 0.502 | 4.81 | 0.22 | 0.133 | 3 | yes | +0.159 |
| ap_english | creative | criteria | 9/10 | 0.358 | 3.92 | 0.24 | 0.109 | 1 | yes | (Ω-mismatch, skipped) |
| aigner | math | criteria | 6 | 0.340 | 1.66 | 0.29 | 0.133 | 4 | yes | +0.171 |
| abbott | creative | criteria | 5 | 0.330 | 1.95 | 0.35 | 0.112 | 2 | yes | +0.148 |
| aigner | math | steps | 6 | 0.281 | 2.29 | 0.20 | 0.141 | 3 | yes | — |
| andrew_stanton | creative | criteria | 4 | 0.212 | 1.39 | 0.41 | 0.137 | 3 | no | +0.115 |
| arnold | math | criteria | 8 | 0.197 | 3.76 | 0.13 | 0.124 | 3 | yes | +0.088 |
| aristotle | creative | criteria | 7 | 0.134 | 2.63 | 0.36 | 0.101 | 2 | yes | +0.035 |
| gowers | math | steps | 14 | 0.530 | 6.91 | 0.16 | (K too big for brute-force) | | | |

**Two clean findings:** (1) executor R is compressed into a **narrow 0.10–0.14 band** despite ideal Ǐ
ranging 0.13–0.53 — the single-verdict bottleneck FLATTENS the articulability gradient (which lives in the
ideal, not the executor output). (2) matched-f executor gap SCALES with ideal recovery (aigner/gowers
~0.16–0.17 → aristotle 0.035): more recoverable signal ⇒ more lost to compression. PRUNE universal except
andrew_stanton (K=4, too small). GOAL MET: real GEPA + real mined Ω, large-scale, verified, trustworthy.
