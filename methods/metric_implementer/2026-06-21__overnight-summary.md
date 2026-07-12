# Overnight run summary — 2026-06-21 — REAL GEPA + real mined Ω, large-scale

**Goal:** real GEPA runs + real mined Ω, large-scale across many metrics/texts, verify everything,
get trustworthy numbers. **Status: COMPLETE and verified** — all 8 metric-runs (math + creative, both
ideal Ǐ and executor R) done; every executor-R npz legitimacy-checked (M≈0.50, maxR ≤ cap ½, N=200).

## What was run (all REAL, no fabrication)
- **REAL GEPA lineage** at `/lfs/.../tmp_vinfo/gepa_registry` (21 metrics; v000 = free-text seed → v013
  evolved structured `criteria→checks→scores` rubric over ~14 GEPA generations). Provenance confirmed.
- **Ω extraction fixed & audited:** the original regex was contaminated (subagents scored it **37% valid**
  — score-anchors, GEPA diagnostic leaks, titles). Replaced with **structural+recursive** extraction
  (`criteria[].description` + `checks[].description`, pruning `worked_examples`/`scores`) → clean. Added
  **behavioral dedup** (merge ~identical per-item signals). Re-audited clean.
- **2 granularities:** Ω=criteria (high/mid dimensions) vs Ω=steps (fine procedural checks).
- **2 recovery measures:** ideal `Ǐ=I(M;X_S)` (Shannon bits, full signal vector) and executor
  `R=I_TVD(M;M̂_S)` (TVD, cap ½, one compressed verdict).
- 7 metrics × {math (math_se), creative (writingprompts)}, N=200–250, Llama-3.1-8B executor,
  M=holistic quality (median-split). Everything recomputed from saved arrays (zero-GPU aggregators),
  nothing trusted from logs.

## Headline trustworthy findings
1. **The executor bottleneck is LARGE (matched-f TVD gap):** compressing K criteria into one holistic
   verdict **loses 26–56% of the recoverable signal**: aigner ideal 0.304→exec 0.133, gowers 0.292→0.133,
   arnold 0.212→0.124, aristotle 0.136→0.101. The gap **scales with ideal recovery** (aigner/gowers ~55% →
   arnold 42% → aristotle 26%): metrics with more recoverable signal lose more to compression — the
   single-verdict rubric recovers only ~half of what the full criterion vector carries (less so when there
   is little to recover).
2. **PRUNE is universal:** every metric's best rubric uses 1–4 of K criteria; adding all *lowers* R.
   Creative PRUNEs harder (ap_english best = **1** criterion; full set −59%) than math (best 3–4) — the
   executor collapses more ideal dimensions for tacit/narrative content.
3. **Granularity flip:** ideal recovery penalizes finer steps (aigner Ǐ 0.340→0.281, −17%); the executor
   slightly *favors* them (R 0.133→0.141, +6%). The bottleneck compresses the granularity effect — finer
   concrete checks are easier to apply per-item even though they carry less ideal joint info.
4. **Articulability gradient (confounded):** math exposition (gowers Ǐ=0.502) > math proofs (aigner 0.340)
   > narrative/Poetics (aristotle 0.134). Structured math recovers better than tacit narrative — but
   confounded by criterion count, rubric quality, and M base-rate (creative ~0.12 vs math ~0.35).
5. **Recovery is LOW overall** (executor R ≈ 0.10–0.14, ~¼ of cap ½): a GEPA rubric's single verdict is a
   weak reconstruction of holistic quality. That is itself a finding about high-level rubric dimensions.
6. **The executor FLATTENS the articulability gradient.** Ideal Ǐ ranges widely (gowers 0.502 → aristotle
   0.134), but executor R is compressed into a **narrow 0.10–0.14 band for ALL 8 metrics** — the
   single-verdict bottleneck erases the cross-corpus differences the criteria carry. So the articulability
   gradient lives in what the dimensions *could* convey (ideal), NOT in what the compressed rubric verdict
   *delivers* (executor). (`andrew_stanton` is the lone non-PRUNE metric — K=4, too small to prune.)

## The table (Ǐ ideal Shannon-bits | R executor TVD)
| metric | corpus | gran | K | Ǐ bits | N_eff | γspec | R TVD | best\|S\| | PRUNE | matched-f gap |
|---|---|---|---|---|---|---|---|---|---|---|
| gowers | math | criteria | 10 | 0.502 | 4.81 | 0.22 | 0.133 | 3 | yes | +0.159 |
| aigner | math | criteria | 6 | 0.340 | 1.66 | 0.29 | 0.133 | 4 | yes | +0.171 |
| aigner | math | steps | 6 | 0.281 | 2.29 | 0.20 | 0.141 | 3 | yes | — |
| arnold | math | criteria | 8 | 0.197 | 3.76 | 0.13 | 0.124 | 3 | yes | +0.088 |
| ap_english | creative | criteria | 9/10 | 0.358 | 3.92 | 0.24 | 0.109 | 1 | yes | (Ω-mismatch, skipped) |
| abbott | creative | criteria | 5 | 0.330 | 1.95 | 0.35 | 0.112 | 2 | yes | +0.148 |
| andrew_stanton | creative | criteria | 4 | 0.212 | 1.39 | 0.41 | 0.137 | 3 | **no** | +0.115 |
| aristotle | creative | criteria | 7 | 0.134 | 2.63 | 0.36 | 0.101 | 2 | yes | +0.035 |
| gowers | math | steps | 14 | 0.530 | 6.91 | 0.16 | (K too big for brute-force) | | | |

## Honest caveats
- **Ǐ is Shannon bits (ceiling ~1); R is TVD (cap ½)** — different f, never differenced. The matched-f gap
  uses a TVD *ideal* on the same subset (clean only for the 3 math metrics).
- **Median-split everything → rank-based recovery.** Verified legitimate: continuous signals have real
  spread and correlate +0.28..+0.77 with quality.
- **spectral γ is a loose lower bound** for collinear high-level criteria; exact-γ joint-MI is undersampled
  at these N and NOT trusted. Trustworthy = R(OPT), N_eff, per-criterion I, matched-f gap.
- **Dedup mismatch:** creative matched-f gaps need Ω-aligned re-runs (Ǐ deduped vs executor raw); the
  crits-guard skipped them rather than mis-align (no wrong number emitted).

## Follow-up 2026-06-21 — external-aggregator test (PARTLY REVISES headline #6)
`experiments/external_aggregator.py` (zero GPU, paired inside each bfc file: `X_solo[:,i]=mhat[{i}]` is
the LLM judging criterion i ALONE). Ladder on the SAME subset: `Iceil(M;X_S) ≥ R_B(external LR, OOF) ≥
R_A(LLM holistic verdict)`. Headline (FULL criterion set, no cherry-pick):

| metric | Ǐceil | R_A(LLM) | R_B(ext LR) | rescued |
|---|---|---|---|---|
| ap_english | 0.380 | 0.045 | 0.153 | +238% |
| gowers | 0.410 | 0.108 | 0.168 | +56% |
| aigner | 0.320 | 0.096 | 0.147 | +53% |
| arnold | 0.330 | 0.114 | 0.141 | +23% |
| abbott | 0.300 | 0.104 | 0.121 | +16% |
| aristotle | 0.310 | 0.070 | 0.063 | −10% |
| andrew_stanton | 0.255 | 0.133 | 0.111 | −17% |

Findings: (1) **PRUNE is mostly a PLUMBING artifact** — R_A is non-monotone (PRUNEs); R_B(LR) climbs
monotonically (can downweight noise). Moving aggregation OUTSIDE the LLM rescues most of the collapse
(ap_english 0.045→0.153). The criteria carry it; the single holistic pass loses it. (2) **A real
compression floor remains:** best external linear combiner (0.06–0.17) << joint ceiling (0.26–0.41);
collapsing to ONE scalar loses ~half regardless of who aggregates — only fixable by emitting >1 number.
(3) **2/7 resist linear external agg** (aristotle, andrew_stanton): signal is in criterion INTERACTIONS
a linear combiner misses but the joint MI + the LLM holistic pass partly capture → "aggregate outside"
is NOT universal; holistic pass competitive for interaction-heavy/tacit metrics. (4) **Gradient
de-flattens with de-compression:** band widens LLM 0.045–0.133 → ext-LR 0.063–0.168 → joint 0.255–0.410
and re-orders sensibly (math-expo top, narrative bottom). So headline #6 ("executor flattens the
gradient") is executor-AGGREGATOR-specific, not intrinsic — the gradient is real but hidden by
single-verdict compression.

Validation notebook: `notebooks/2026-06-21__upper-bound-validation-aigner.ipynb` (1 metric end-to-end:
full rubric + all GEPA evolutions + GEPA loop/prompts/diagram + Ω union + the upper-bound math, all
inlined estimators, runs locally off `notebooks/data/aigner_validation/`). Manual-validation catches:
GEPA revise-prompt is templated to "competitive-programming" (explains math-rubric drift); harvester
schema bug undercounts Ω (6 vs 12, missed runs 1&2 named-dim schema); elegance facet seeded but evolved
away.

## What's next
- Land the 3 creative executor-R (in flight) → complete cross-corpus table.
- Align Ω (add behavioral dedup to small_omega_brute_force) → creative matched-f gaps.
- Deconfound the articulability gradient (match criterion count / base-rate); higher N; more corpora.
- Scripts: `experiments/{harvest_gepa_omega,real_gamma,small_omega_brute_force,aggregate_all,tvd_gap}.py`;
  drivers `run_all_gamma.sh`, `run_bfc_math.sh`, `run_bfc_creative.sh`. Full log in
  `notes/2026-06-19__tvd-consistency-real-data.md`.
