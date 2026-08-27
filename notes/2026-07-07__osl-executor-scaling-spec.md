# OSL executor-scaling spec (frozen pre-registration) — 2026-07-07

**Question.** How does the recovery ceiling of a metric scale with executor capability — and what is
its asymptote M_{∞,ideal}? The asymptote operationalizes the dense-model ceiling C in the tacitness
decomposition: a criterion whose disattenuated recovery asymptote sits below 1 is inarticulable *for
an ideal reader*, not merely for a weak judge. Follows Ruan/Maddison/Hashimoto (Observational Scaling
Laws, 2024) in structure — within-family compute→capability calibration + shared capability→downstream
law — with two deliberate departures justified by n≈15 executors (a 15-subject study):

1. **Declared instrument, not estimated latent (PRIMARY).** The capability scalar is a frozen,
   ground-truthed anchor battery (below), not PCA of external benchmarks. PCA at n≈15 estimates an
   unstable axis and assumes variance=relevance; external benchmarks needn't span the judging-relevant
   axis. The prior is injected at instrument-design time, where three pre-registered vetoes can
   falsify it (G1–G3 below), two of which never touch Y.
2. **Data-driven latent as SECONDARY (sensitivity arm).** PCA (unsupervised) and PLS-1 (nested inside
   LOEO) over the in-domain base-measurement vector — reported as convergent-validity comparison,
   never as the headline axis.

## Design (fixed-support arm; native-Ω arm explicitly deferred to phase 2)

- **Task:** humor (short texts, clean 8B artifact base). Probes = manifest texts[60:360] (n=300,
  same convention as bank/checkpoints). Battery texts = manifest texts[360:600] (disjoint).
- **Metrics:** 40 of the 285 humor R2 metrics, frozen by sha256 stable-pick on metric name
  (immune to bank growth; recorded below). Plus **5 planted mechanical control metrics**
  (code-truth: length>K, contains-?, contains-digit, quoted-speech, length<K) whose Ω includes the
  planted criterion + its paraphrases → their fitted asymptote must be ≈1 (pipeline+fitter
  positive control; the banks contain ~0 mechanical metrics, so the control must be planted).
- **Fixed support:** per metric, K=60 criterion TEXTS from the 8B-mined `silver_r2/humor`
  checkpoints, sha256 stable-picked, frozen inline in the freeze file. Every executor RE-EXECUTES
  the same texts (retarget discipline: nothing byte-copied — the confound measured 2026-07-02).
  8B is re-run like every other executor (its cached sigs used different pipeline details; no reuse).
- **Executors (~15, all sk3-local vLLM offline, forced-logprob P(YES)):**
  cached: Llama 1B/3B/8B/70B · Gemma-3 4B/12B/27B · Gemma-2 27B · Qwen3.5-122B;
  downloads: Qwen2.5 3B/7B/14B/72B · Mistral-7B · Phi-4. Gemma-4-31B optional (dedicated env).
  No API executors (sk3-only rule; logprob invariance).
- **Recovery (y):** greedy forward selection on EVEN probes (ridge-CV corr, 1-SE stop, ≤8 criteria),
  readout on ODD probes: r_odd = corr(pred, m̄_ω,odd); bits = −½log2(1−r²). Identical procedure for
  every executor (internal comparability is the requirement, not equivalence with the cert pipeline).
- **Disattenuation (the X–Y circularity fix):** target reliability from the form-orbit split
  (forms {1,3} vs {2,4} of m̄_ω, Spearman-Brown to 4 forms, odd probes); y_dis = r_odd/√rel, clipped.
  Raw reported alongside. Separates "executor is noisy" from "criterion is unrecoverable".
- **Degeneracy gate (per executor×metric cell):** std(m̄_ω) ≥ 0.03 AND nan-rate ≤ 0.10, else cell
  excluded and recorded (the applicable-vs-fails distinction; Qwen-3B exact-0.5 lesson). Per-executor
  nan-rate itself is a base measurement (new-family tokenizers can break the YES/NO readout).

## The declared instrument (anchor battery)

~240 items, each (criterion, text, truth) with truth computed BY CODE — no human labels, no LLM
labels (instrument calibration, same discipline as blinded anchor rows in judging batches). Families:
word-count thresholds (difficulty graded by margin), token/word presence, negation, two-clause
composites, paraphrase re-wordings (truth-bearing). Balanced ~50/50. Scalar z_E = logit(AUC(P_YES,
truth)) — threshold-free. v2 fallback (pre-registered): if G3 fails, add a GLM-planted semantic tier
and re-freeze as battery-v2 with the change documented.

## Gates (pre-registered; run before any extrapolation talk)

- **G1 instrument reliability:** battery split-half (items even/odd) AUC correlation across
  executors ≥ 0.9; no-ceiling: top executor AUC ≤ 0.98 (else add harder items → battery-v2).
- **G2 within-family monotonicity:** z_E strictly increasing in log-params inside Llama (4 pts) and
  Gemma-3 (3 pts) — the within-family compute-linearity of OSL, repurposed as instrument falsification.
- **G3 LOEO:** leave-one-executor-out prediction of executor-level mean y_dis; R²_LOEO > 0 vs
  predict-the-mean baseline. Fails → instrument or law wrong; stop, no extrapolation.
- **G4 planted-mech control:** fitted asymptote L_mech ≥ 0.9 and executor-truth AUC rising in z.
- **G5 family pooling license:** permutation test of between-family variance in the residuals of the
  pooled fit; p < .05 → NO pooling, fall back to within-family curves + meta-analytic combination.

## Fit + reporting

Executor-level y = mean y_dis over included bank metrics (per-metric spaghetti + per-kind fits
reported alongside). Saturating y = L/(1+e^{−k(z−z0)}) vs non-saturating linear, AICc + LOEO; if
non-saturating is not rejected, **L is a lower bound, not a point** (non-Lipschitz extrapolation
honesty). Profile-likelihood CI on L. All recovery numbers reported as transmission (house rule).

## Frozen artifacts (shas recorded at freeze time, before first executor run)

- battery: `outputs/osl/battery_humor_v1.json` (240 items) —
  sha256 `ac4c4ed22f9f550d4dc6a44c190b88b3e66c69f627117546d991eab3095e1a07` (frozen 2026-07-07)
- metric freeze: `outputs/osl/freeze_humor40_v1.json` (40 bank + 5 planted, K=60) —
  sha256 `8c62d3cbf8d8c6d37e4ce1346b00d13e242f89b147299533d0edf73ac96eafd2` (frozen 2026-07-07)
- code: `osl_battery.py`, `osl_sweep.py`, `osl_fit.py` + `tests/test_osl.py` (13 CPU tests green;
  full metric_implementer suite 126 green) — frozen at first executor launch 2026-07-07

Deviations after freeze require a documented battery-v2 / freeze-v2, never in-place edits.

## v2 extension (documented deviation, user-approved 2026-07-07 pm)

Pilot v1 read: mechanical law strong (ρ=.96, LOEO .88, L≈1); taste aggregate weak (LOFO+disatt:
ρ=.71, LOEO .43, L=.72 with CI incl. 1) with residual = executor calibration "dialect"; per-metric
plane strong (corr(slope, frontier)=.70; intrinsic cluster = performance/embodied criteria). v2:

1. **Pairwise pilot arm** (ALONGSIDE P(YES), which stays primary): forced-choice A/B over 200
   fixed probe pairs (seed-0, both presentation orders; order-split = reliability), per metric per
   executor. Rationale: cancels absolute-calibration dialect by construction (dual-polarity
   acquiescence + MathlibPR pairwise-A precedents).
2. **freeze-v2** = all-usable humor bank metrics (~285) + same 5 planted controls; m̄-only P(YES)
   runs on it (6× per-metric floor coverage; enables floor × TASTE/CRAFT lexicon-tag cross).
   `outputs/osl/freeze_humor285_v2.json` — sha recorded at build.
3. **Frontier depth**: qwen35-122b canonical-form-only diagnostic (n_forms=1; its battery was fine,
   orbit reformulations break it — pairwise avoids reformulations entirely, second diagnostic);
   gemma-3 12B/27B weight repair + full sweep; gemma-4-31B attempt via its dedicated env.
4. **Analysis adds**: leave-FAMILY-out consensus (LOFO — leave-executor-out let big-family members
   be scored against siblings; singletons were systematically depressed), reliability
   disattenuation, floors normalized to the planted determinate ceiling (underdetermination index),
   per-metric (slope × frontier) plane as a primary deliverable.

## Curve-shape semantics (doc-of-record, 2026-07-07 pm; user-approved interpretation table)

The per-metric articulability curve is y = consensus-agreement recovery (LOFO family-balanced)
vs x = battery z, read against two independently measured horizontal references: the **planted
determinate ceiling** (≈0.813 on humor-v2 — what a fully codable criterion achieves under this
instrument) and the metric's **underdetermination floor** (1 − inter-frontier agreement: how much
frontier executors disagree with each other about what the criterion means).

| shape | verdict | meaning |
|---|---|---|
| bend **at the ceiling** | REACHES | the concept **is articulable** — its description carries enough; mid-scale executors already execute it fully. Positive certificate. |
| bend **below ceiling, above floor** | BOUNDED | capability keeps growing, articulation doesn't — the words don't carry enough. The **tacit-residual signature**; the only route to a negative certificate. |
| bend **at the floor** | (criterion property) | not an executor limit: frontier models disagree about what the criterion *means*. The language underdetermines the concept (worst for social-consequence criteria, floors 0.04–0.5). |
| **no bend** (RISING) | lower bound only | a rising curve can never prove inarticulability — some future model may articulate it. Certifies "at least this articulable, still climbing." |

Logic is asymmetric: REACHES = positive proof; BOUNDED = candidate negative proof (x-axis-dependence
caveat below); RISING = permanently inconclusive-upward. Humor-v2 census: 55 REACHES / 54 RISING /
24 BOUNDED / 19 NOISY. Standing caveat (Beta-IRT cross-check, arXiv:2606.07616 replication): plain
2PL without a ceiling parameter beat ceiling models in LOEO ⇒ some BOUNDED may be battery-axis
compression, not true plateaus — adjudicated by frontier probe points (Llama-405B, GLM-4.7→5.2).

## Smoothness audit (2026-07-07 pm — "are the curves smooth enough to point to bending?")

`curve_smoothness.py` → `outputs/osl/smoothness_humor.json` + per-metric curves for plotting in
`outputs/osl/curves_humor.json` (149 metrics × 10 executors, y ± split-half SE, z-ordered).

- **The RISE is unambiguous:** curve SNR (isotonic fitted range / split-half probe noise) median
  **20.1** (q25 9.6); 95% of metrics have SNR > 3. Per-point y values are highly reliable.
- **The BEND is family-limited:** residual scatter around the monotone fit is **~5.4× the probe
  noise** (only 1% of metrics are noise-limited) ⇒ deviations from a smooth curve are systematic
  **executor/family dialects**, not sampling error. Effective shape-SNR ≈ 20/5.4 ≈ **3.7**: bends
  of order the full dynamic range are resolvable (~4σ); subtle bends are not, and a pooled-panel
  bend can be an artifact of which families happen to sit where on z (the G5 concern, quantified).
- Consequences: (i) error bars for curve-shape claims must be executor-level residuals, not probe
  SEs; (ii) within-family curves are the clean read but have only 4 points (Llama, Qwen2.5) —
  more rungs/families per task is the binding constraint on bend confidence; (iii) the Beta-IRT
  arm helps because θ_i absorbs executor dialects explicitly.

## v3 multi-task panel extension (launched 2026-07-07 pm, user-directed)

Expand the panel to every task with external silver labels: freezes built for creative_writing
(198 bank), press_releases (149), math (87), news_homepages (150), peer_review (88),
notice_and_comment (88), patents (7) — each + 5 planted controls, probes = manifest texts[60:360]
so the v2 `judgement` labels join exactly (same discipline as humor). 767 new bank metrics
(+152 humor = 919 with panel curves). Fleet: 11 executors (Llama 1B–70B, Qwen2.5 3B–72B,
Gemma2-27B, Mistral-7B, Phi-4) on GPUs 2/7 (small, yield to the 405B probe) + GPU 4 (large);
resumable skip-if-exists; freeze shas in `outputs/osl_multi/FLEET_STATUS`. Deliverables: per-task
per-metric curves + knee census, MI→silver item-AUC replication per task (humor result:
ρ(OPT_Ω, item-silver |AUC−.5|) = +0.47, n=152, perm p<.0005), capability dose-response per task.
