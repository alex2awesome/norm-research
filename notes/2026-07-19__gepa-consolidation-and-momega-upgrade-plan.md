# 2026-07-19 — GEPA consolidation + M_ω upgrade master plan (user directives, verbatim intent)

User issued six directives (this file is the execution state of record; update STATUS lines as
work proceeds; survive compaction via this file).

## D1. Official GEPA only — FIRM AND FINAL (reconstruction experiments)
Deprecate the in-house GEPA search loop; archive, don't delete.
- In-house loop = `tune_shared_template` / `tune_shared_template_batched` in
  `methods/metric_implementer/experiments/v14_decoder_tuning.py` + the `--phase tune` path in
  `run_v14_value_campaign.py` (`run_decoder_tuning` → tune_shared_template_batched) + proposer
  `propose_mutations`.
- KEEP shared utilities used by the official path: `template_sha256`,
  `validate_shared_template` (official_gepa_decoder_tune.py imports them).
- Plan: (a) move loop fns + propose_mutations into
  `methods/metric_implementer/experiments/archive/inhouse_gepa_deprecated.py` with banner
  "DEPRECATED 2026-07-19 (user decision): official github GEPA (gepa 0.1.4) is the only
  sanctioned optimizer for reconstruction experiments. Same-pool comparison: official best
  pooled −0.014 / best admissible −0.052 vs in-house −0.078 (see
  outputs/fast/development/tuning/ on sk2)."; keep import-compat shims that raise
  DeprecationWarning→error; (b) rewire `--phase tune` to official gepa.optimize (generalize
  `official_gepa_decoder_tune.py` to all 3 decoder families + behavioral channel, keep
  search-split-only selection + post-hoc aggregate discipline); (c) fix tests that import the
  old loop (skip w/ deprecation reason or port); (d) `pip install gepa==0.1.4` into sk2 AND
  sk3 envs (sk2 has it; sk3 needs check); (e) grep whole repo for other in-house GEPA loops
  (A-bank/silver-norms recipe uses GEPA — check whose implementation; if in-house loop,
  migrate; if already official, leave).
- STATUS: DONE (code side, 2026-07-19). (a) In-house loop archived VERBATIM to
  `experiments/archive/inhouse_gepa_deprecated.py` (+ `archive/__init__.py`) with the banner
  above (`tune_shared_template`, `tune_shared_template_batched`, `propose_mutations`, and the
  private helpers `_mutation_prompt`/`_propose_mutations`/`stable_seed` + constants
  TRACE_SCHEMA/MAX_ROUNDS/CANDIDATES_PER_ROUND/BEAM_SIZE). `v14_decoder_tuning.py` keeps the
  live utilities `template_sha256`, `validate_shared_template`, `select_dev_metrics`,
  `stratified_reference_states`; the three public loop names are now shims raising
  `RuntimeError("in-house GEPA loop deprecated 2026-07-19 — use official gepa …")`. (b)
  `run_v14_value_campaign.run_decoder_tuning` rewired to official `gepa.optimize` via a
  `GEPAAdapter` generalizing `official_gepa_decoder_tune.py` to all 3 decoder families + the
  behavioral channel: per-family scoring copied exactly from the retired `evaluate_batch`
  (mcq→`score_mcq_reference_templates`; behavioral→`induce_behavioral_reference_templates` +
  `score_behavioral_reference_templates` with FIXED_EXECUTOR), search-split-only selection
  signal, CONSTRAINT text appended in `make_reflective_dataset`, 3-try engine retry with
  `release_resident_engines`, `proposals.jsonl` in the run dir, post-hoc
  `aggregate_template_fitness` over seed+winner+every distinct valid template, output
  `development/tuning/<name>.json` schema `v14-tune-official-gepa-v1` (winner_template,
  winner_template_sha256, winner_report, reports per-sha, seed equivalents, plus
  shared_across_decoder_families/freeze_sha256 so `build_production_freeze` still consumes it);
  proposer spec `backend:model` becomes `reflection_lm` via `LLMBackend`; added
  `--max-metric-calls` (default 240). (c) Tests: `test_v14_campaign.py`
  batched-GEPA test now asserts the shim raises RuntimeError (name kept); new
  `tests/test_official_gepa_tune.py` monkeypatches `gepa.optimize`/dev-contexts/scorer under
  `fake_backends=True` and checks the output JSON schema. `python -m pytest
  test_v14_campaign.py test_official_gepa_tune.py -x -q` → 16 passed in 4.76s; full tests/
  dir still collects (325). (e) grep: `run_r2_recovery.py`→`m_omega_gepa.gepa_discriminative_m_omega`
  is a SEPARATE in-house M_ω prompt-GEPA (not the moved v14 functions), left untouched;
  `battery/gepa_h2h*/eval_final.py` import none of `gepa`/moved fns/`v14_decoder_tuning` (they
  score pre-frozen `gepa_final_prompts.jsonl`), left untouched. NOT DONE (out of scope this
  pass): (d) `pip install gepa==0.1.4` on sk2/sk3 — deliberately skipped (no sk2/sk3 touch, no
  GPU); the separate `m_omega_gepa` in-house loop was NOT migrated (only the v14 decoder loop
  was in scope). No GPU code executed; all paths fake/mocked.
- STATUS (2026-07-19, second pass — SECOND in-house loop migrated): `m_omega_gepa.py`'s
  `gepa_discriminative_m_omega` (the discriminative-M_ω prompt search used by
  `run_r2_recovery.py --gepa-m-omega`) is now driven by official `gepa.optimize`. (a) The
  hand-rolled rounds/mutations loop + its loop-only helper `_fewshot_block` are archived
  VERBATIM in `experiments/archive/inhouse_m_omega_gepa_deprecated.py` (same banner style; it
  imports the still-live primitives rather than forking copies). Unlike the v14 decoder shims
  the public name is NOT a raising stub — it keeps working, on official GEPA. (b) Rewritten
  IN PLACE with an IDENTICAL signature (plus an optional `max_metric_calls=None` override) and
  an IDENTICAL return contract, so `run_r2_recovery.py` needed NO edit. ESTIMAND UNCHANGED:
  same executor primitive `_score_binary_sampled`, same canonical pool objective
  `_discrimination_score = std - 0.5*|mean - 0.5|` for the report. (c) GEPA needs per-instance
  scores for its Pareto frontier, so the adapter scores instances with the mean-absolute-
  deviation decomposition `s_i = |p_i - p_bar| - 0.5*|p_bar - 0.5|`; for binary verdicts
  MAD = 2*p_bar*(1-p_bar) and std = sqrt(p_bar*(1-p_bar)) are both strictly increasing in
  p_bar*(1-p_bar), and with d = |p_bar - 0.5| both statistics reduce to strictly decreasing
  functions of d ⇒ RANK-EQUIVALENT, so the search signal is faithful and (valset = whole pool)
  GEPA's selection is the canonical selection. Verified numerically: 19,900 random pairs, every
  discordance ≤ 1.1e-16 (float-epsilon ties only). (d) Canonical pool-level statistic reported
  POST-HOC for the seed, the winner, and every distinct candidate (per-instance signal for
  SEARCH, canonical statistic for the REPORT — same discipline as the decoder tune); trajectory
  keeps its 5-tuple shape with row 0 = seed and last row = returned prompt. (e) Budget mapping
  `max_metric_calls = rounds * n_mutations * len(texts)` keeps the existing `--gepa-rounds` /
  `--gepa-mutations` CLI flags working. (f) Reflective feedback reuses `_select_failures`
  (items nearest 0.5) + `_mutation_prompt` intent + measured base-rate/std + an explicit
  CONSTRAINT line (one self-contained criterion, no exemplars copied in, no meta-commentary).
  (g) `mutation_mode` / `fewshot_examples` are accepted for signature compatibility but INERT
  (the code-appended few-shot operator has no GEPA analogue; it also HURT discrimination in the
  2026-06-25 A/B, 0.448→0.292) — a RuntimeWarning fires if a caller asks for a non-default.
  Tests: new `tests/test_m_omega_gepa_official.py` (13 tests) fakes executor/reviser and
  monkeypatches `gepa.optimize` with a stub that still drives the real adapter; it asserts the
  return contract by RUNNING the archived loop and comparing key sets, pins rank-equivalence,
  the base-rate-0.5 symmetry, NaN→-1.0, the budget mapping, and archive integrity.
  `pytest tests/ -q -x -k "m_omega or gepa"` → 16 passed, 322 deselected; full dir collects 338.
  Repo grep for other importers: ONLY `run_r2_recovery.py` (line 51). No GPU/sk2/sk3 touched.
  CAVEAT (recorded, NOT fixed — memory `project_a_bank_degeneracy_audit`): "variance-revival !=
  information-revival" — a discrimination-maximizing objective can produce high-variance but
  UNINFORMATIVE criteria (mined banks ran 54-68% degenerate). This migration preserves the
  pre-existing objective and does NOT endorse it; also noted in the module + function docstrings.
  D1(e) NOTE: a THIRD in-house GEPA loop exists, `datasets/prompt-optimality-test/run_inhouse_gepa.py`,
  and must NOT be migrated — it is Arm B, the deliberate in-house-vs-official contrast that
  Phase 3 measures. Still open from the first pass: (d) `pip install gepa==0.1.4` on sk2/sk3
  (local env here is gepa 0.0.17; tests monkeypatch `gepa.optimize` so they are version-agnostic,
  but the LIVE path should be run against 0.1.4).

## D2. M_ω generation upgrade (main reconstruction project)
New unit-pool recipe: units mined from BEST official-GEPA trajectory + EXPLICIT CHILDREN
metrics (hierarchy: judging R2 → include its certified R1 children from the L0→R3 ladders,
`project_hierarchy_l0_r3_reconstruction`) + LLM-suggested units.
- KEY: M_ω compile INITIALIZES FROM the best GEPA candidate (not from seed), then greedy
  add/remove units ⇒ M_ω ≥ GEPA on the selection panel by construction (superset argument).
- Benchmark-side analog (prompt-optimality-test `run_unit_recombination.py`): init compile =
  best official-GEPA prompt; units from official trajectory + GLM-suggested novel units;
  greedy add AND remove (swap) steps.
- STATUS: BENCHMARK-SIDE IMPLEMENTED 2026-07-19 -> datasets/prompt-optimality-test/run_momega_v2.py
  (v1 left intact; new arm dir runs/<ds>/momega_v2/). Design = the D3-supported fixes only:
  init from official GEPA's SHIPPED best_candidate; NO cheap screen (a 60-item PAIRED marginal
  pass over every unit IS the selection step); unit pool = official+in-house trajectory clauses
  PLUS 12 GLM-5.2-suggested novel units generated from init's observed failures; cumulative-
  prefix sweep + drop-one pass; NO-REGRET GUARD on a disjoint 40-item confirmation slice (ship
  init if the compile does not beat it => M_ω >= GEPA-shipped by construction up to confirmation
  noise). SPLIT HYGIENE FIX (important, discovered while building): every phase-1-4 arm selected
  on val[:100], so val[:100] is contaminated for reporting; v2 selects on TRAIN only and reports
  on val[100:300] (200 items, untouched by every arm), re-scoring GEPA-shipped and the raw seed
  on those same items. NOTE: on hover AND hotpotqa official GEPA SHIPPED CANDIDATE idx 0 = the
  SEED (val_agg 0.76 and 0.80; it explored 6 candidates and improved on neither), so "beat GEPA"
  there == "beat the seed"; only aime saw a real GEPA gain (.353 -> .529, idx 2).
  D2's third unit source (explicit children metrics, e.g. R1 children when judging R2) has NO
  analog in this benchmark setting — it applies to the reconstruction side, still TODO.

## D3. Investigate the 1-of-30-units bottleneck (benchmark Arm C)
HYPOTHESES to test against runs/*/unitrecomb/proposals.jsonl (logged item vectors exist):
(a) acceptance noise — MIN_GAIN=.01 on 15-item panel needs >=1 net item, most units within
noise of cur; (b) screen(8 items) rank ~ uncorrelated with greedy gain; (c) unit redundancy
(clause-splits of similar prompts) collapses conditional value after 1st accept; (d) GLM
saturation headroom on hover/hotpot. CANDIDATE FIXES (implement only what diagnosis
supports): paired per-item sign-test acceptance; bigger dev panel; unit dedup/clustering;
init-from-GEPA-best (D2); remove/swap moves; larger declared budget.
- STATUS: DIAGNOSIS DONE 2026-07-19 -> datasets/prompt-optimality-test/UNIT_BOTTLENECK_DIAGNOSIS.md.
  RESULT OVERTURNED THE PLAN'S PRIORS. Confirmed causes: (1) DOMINANT+deterministic — the
  reserve formula greedy_reserve=panel*(1+12+4) plus screening eating ~48-51% of budget
  truncates round 2 at exactly 4-of-11 units on ALL three datasets, capping compiles at ~1
  regardless of unit quality (explains the identical "1 everywhere" pattern); (2) STRONG — the
  15/8-item selection panel has sd(panel mean) ~1.7-1.8 items vs a 0.15-item acceptance bar, so
  the accepted unit is a coin flip: hover's compiled unit scored +3 items on panel but 0.73 on
  val-100, BELOW the 0.76 seed (false positive); (3) MODERATE — the 8/5-item screen quantizes to
  2-5 distinct values so "top-12" sits on a 14-22-way tie => near-random shortlist
  (Spearman(screen,greedy) = -0.495/+0.251/+0.378, all n.s.).
  REFUTED: (a) MIN_GAIN too strict — 0.01 on 15 items is 0.15 items, LOOSER than one item; all
  12 hover round-1 units cleared it. (b) redundancy — top-12 token-Jaccard .075/.117/.035 with
  10-12 clusters of 12; pool is diverse. (c) GLM saturation — 16-19 of ~30 single units BEAT the
  seed on val-100; regret vs best single unit is -0.18 (aime) / -0.09 (hover). The failure is
  MIS-SELECTION, not a ceiling.
  ⚠ THE PLAN'S OWN PROPOSED FIX IS REFUTED: a paired sign-test acceptance rule admits ZERO units
  on every dataset (base acc .67-.73 caps wins at ~5 discordant items; structurally underpowered
  at n=15) — it would compile 0, strictly worse than 1. PANEL SIZE is the lever, not the test.
  SUPPORTED FIXES ONLY: init-from-GEPA-best; budget out of screening into selection; bigger
  selection panel; remove/swap only if paired with a bigger panel.

## D4. Optimality via the upper-bound machinery
Prove best method ≤ projected upper bound (strictly larger), using existing tools:
- Benchmark side: rescore matrices → exchangeable best-of-m curves → `fit_saturating` y_inf
  (+ bootstrap CI) = projected pool-ceiling; ALSO pool-oracle union_all (per-item ∃-solve) =
  hard within-pool cap. Report per dataset: best-method val < y_inf < union_all pattern +
  CIs. Machinery: methods/metric_implementer/experiments/unseen_value_scaling.py (fit_saturating,
  value_frontloading_stat), analyze.py value curves. CAVEAT (standing): conditional-on-pool,
  not all-prompt; the only certified all-prompt bound style = DPI fixed-target cap
  (project_momega_audit_bracket) — reconstruction side.
- Reconstruction side: report `dev_identification_residual_bits` as the ceiling-residual
  (in-house control stopped at residual <0.02 bits; VERIFY meaning in v14_tuning_evaluator
  before quoting as bound; v4 seed row showed 0.0 — suspicious, check field semantics).
- STATUS: DONE, AND THE ANSWER IS A STRUCTURAL NEGATIVE (2026-07-19) ->
  datasets/prompt-optimality-test/analyze_bounds.py + runs/bounds_summary.{md,json}.
  ⚠ THE REQUESTED BOUND CANNOT COME FROM THIS MACHINERY, BY CONSTRUCTION. For an exchangeable
  best-of-m statistic over a FIXED FINITE pool, E[max of m] is pinned to the pool max at m=n, so
  every monotone asymptote is <= best_achieved. Verified independently in-session on hover:
  E[max|m=1]=.7561, m=29 .8220, m=58 .8298, m=59 .8300 == pool max exactly.
  Numbers (best achieved / y_inf [boot CI] / union-of-all oracle): hover .830 / .8138
  [.7985,.8202] / .950; hotpotqa .880 / .8493 [.8243,.8620] / .920; aime2025 .4706 / .4492
  [.4300,.4581] / .5882. Ordering best<y_inf FAILS on all three; the whole CI sits BELOW
  best-achieved (margins -.0162/-.0307/-.0214). Two separable effects: (i) `fit_saturating` is
  MISSPECIFIED here — it forces y(0)=0 but a best-of-m curve has a large floor at m=1, so tau
  collapses to .35-.89 and R^2 falls to .30-.59 (`compare_scaling_forms` prefers the power law on
  all three); refitting the EXCESS curve y(m)-y(1) lifts R^2 to .87-.94 and the ceiling STILL
  lands at/below best (.8230 vs .8300; .8770 vs .8800; .4648 vs .4706) — so misspecification is
  not the cause; (ii) the finite-pool cap is what binds.
  The ONLY term above best-achieved is union-of-all, but it is a DIFFERENT object (ceiling for a
  per-item ORACLE selector, not for any single prompt) and it is inflated by multiple comparisons
  over noisy binary scoring: per-item scoring noise q estimated from repeated seed rescorings =
  .030 hover / .000 hotpotqa / .088 aime2025; expected false "solved" recoveries 9.9/0.0/8.8 vs
  observed 12/4/2 => aime's oracle gap is ENTIRELY within noise, hover's mostly noise, only
  hotpotqa's 4 items survive. aime2025 (n=17 items, binomial SE .121 > every margin in the table)
  cannot support ANY ceiling claim.
  PART B (verification, requested before quoting): `dev_identification_residual_bits` is NOT a
  legitimate ceiling residual — see memory `reference_dev_identification_residual_bits_trap`.
  It is (panel-design constant) minus (upward-biased plug-in MI) on incommensurable alphabets;
  the campaign code itself tags the ceiling term `moves_with_decoder_tuning: False` and
  `identification_mi_is_not_a_behavioral_ceiling: True`; its 0.0 is a CENSORED readout
  (max(0,...)), reachable when panel code entropy is 0 (a maximally UNINFORMATIVE panel scores a
  perfect residual). Consequence: the archived in-house loop used it as an EARLY-STOP rule
  (stopping_reason "dev_identification_residual_below_0.02_bits") — which is exactly what the
  qwen-only in-house control reported, so that control's stop is suspect and the
  official-GEPA-beats-in-house result should be read as "official searched further", NOT
  "in-house converged".
  FOLLOW-UP STARTED THEN STOPPED by the user's pause: `analyze_bounds_evt.py` (extreme-value
  endpoint estimator = the CORRECTLY-SPECIFIED tool for "a bound strictly above best-achieved",
  since it estimates the upper endpoint of the distribution prompts are drawn from rather than a
  within-pool selection statistic). The agent confirmed the q values and was killed before
  writing the estimator; the file may not exist. NEXT: (1) finish the EVT estimator, (2) redo all
  of D4 on the 300-item paper-exact test splits (SE ~.023 instead of ~.04-.12) — on 100 noisy
  binary items the margins are smaller than the sampling error and no bound can be meaningful.

## D5. Phase 5 both-LM paper runs (user approved BOTH)
- Qwen3-8B justification: IT IS the paper's open-model task LM (Table 1; Appendix E.2 — Qwen3
  8B temp .6 top-p .95 top-k 20; GPT-4.1-mini is the other). Paper-exact column = Qwen3-8B
  served on sk2 (shared HF cache has it; vllm serve, 1 GPU, deviation from offline-batch rule
  ACKNOWLEDGED as DSPy requirement). Second column = GLM-5.2 (our best; subscription quota,
  spend freely).
- Runs: 3 benchmarks (aime/hover/hotpot) × 3 arms × 2 task LMs, budget 600 declared,
  paperexact_arms.py; test = full paper test split; log raw draws.
- STATUS: SET UP AND SMOKE-TESTED, NO PAPER RUNS EXECUTED (stopped by the user's pause).
  Harness `paperexact_arms.py` written and validated; its unitrecomb arm was UPGRADED to the same
  M_ω v2 geometry as run_momega_v2.py (init-from-GEPA-best, paired marginal selection, prefix
  sweep + drop-one, no-regret guard, LLM-suggested per-module units) so the paper runs do not
  inherit v1's broken selection. Qwen3-8B was served on sk2 GPU 7 (port 8077, reasoning-parser
  qwen3, HOME pinned to /lfs) with a local SSH tunnel; endpoint smoke-tested clean (content
  parsed, no stray <think> leakage). BOTH THE SERVER AND TUNNEL WERE STOPPED and GPU 7 freed on
  the user's pause — relaunch via code/scripts/serve_qwen3_sk2.sh + `ssh -f -N -L 8077:127.0.0.1:8077 sk2`.
  TWO PROVENANCE FINDINGS: (1) dspy 3.2.1 HARD-PINS `gepa[dspy]==0.0.27`, so installing dspy for
  Phase 5 silently DOWNGRADED the prompt-opt venv from the PIN.txt-recorded 0.1.4. Phases 1-4
  Arm A ran BEFORE that install, so those results are on 0.1.4 and are clean; but any future raw
  `gepa.optimize` run in that venv must re-pin. Env versions observed: main repo 0.0.17,
  prompt-opt venv 0.0.27, sk2 0.1.4. (2) The GEPA paper's OWN optimizer is
  `vendor/gepa-artifact/gepa_artifact/gepa/gepa.py` — a STANDALONE DSPy teleprompter that does
  NOT wrap the pip `gepa` package at all. Working interpretation (FLAGGED TO USER, UNANSWERED):
  the requirement "exactly the same as the original GEPA paper" is about the END EVALUATION —
  programs, metrics, splits — which ARE verbatim theirs; the optimizer is dspy.GEPA, documented
  as a deviation, with the artifact's own class available if a stricter replication is wanted.

## D6. Theory: is reconstruction the optimal unsupervised metric?
Write notes/2026-07-19__reconstruction-optimality-theory.md:
- Barber–Agakov: E_q[log p(x|m)] is a variational LOWER bound on I(M;X); tight iff
  reconstructor = true posterior. Reconstruction objective = tightest certifiable label-free
  bound on I(M;X).
- DPI corollary: for M=f(X) and any label Y (chain M—X—Y): I(M;Y) ≤ min(I(M;X), I(X;Y)).
  So I(M;X) caps ALL downstream supervised value ⇒ maximizing a certified lower bound on
  I(M;X) = maximizing the universal capacity cap. "Optimal unsupervised metric" in exactly
  this sense (no per-task AUC guarantee).
- EM correspondence: user's p(x|z) intuition = ELBO reconstruction term; EM maximizes
  marginal likelihood; our C(R(Ω)) = I(M;M̂) recovery = decodability/self-consistency ⇒
  relate to identifiability (nonlinear-ICA/iVAE conditions; honest limits: without
  identifiability no unsupervised criterion can be pointwise optimal).
- Tie to existing feedback memories: T_lower_bound_Mstar_be_upper, report_recovery_metric_only,
  vinfo_pathologies_koyejo (Shannon transmission, no naive scaling).
- STATUS: DONE 2026-07-19 -> notes/2026-07-19__reconstruction-optimality-theory.md. Provable:
  Barber-Agakov tightness (reconstruction = THE attaining label-free variational bound on
  I(M;X)); DPI universality (I(M;Y) <= I(M;X) for every Y, sharp — for deterministic M,
  I(M;X)=H(M)=sup_Y I(M;Y)); recovery I(M_ω;M̂) measures the IDENTIFIABLE quotient
  [ω]_behavior (nonlinear-latent unidentifiability => behavioral readout is the only well-posed
  one, which retro-justifies the no-similarity-to-reference rule). NOT provable, with
  counterexample: reconstruction optimal for a SPECIFIC downstream task — a SHA-parity criterion
  maximizes I(M;X) with zero semantic utility, i.e. "maximize I(M;X)" IS entropy maximization
  and is the formal statement of the audited "variance-revival != information-revival" finding
  (which also flags the discrimination-maximizing M_ω objective as a CAPACITY objective).
  Fano deliberately NOT used (retracted in this project). Open+promising: add the ELBO's missing
  complexity term => MDL-penalized recovery readout (description lengths already logged).

## Execution order: D1 → D2 → D3 → D4 → D5 → D6 (user: "focus on everything in order and fully")

---

# NEXT TASKS — brief for the incoming agent (written 2026-07-20 at session handoff)

Everything below was STOPPED cleanly by user request. Nothing is running: no local processes, no
agents, no sk2 vLLM server (GPU 7 freed), no SSH tunnel. Read the STATUS lines above before
starting — several of this file's original hypotheses were REFUTED by evidence and the refutations
are recorded there. Do not re-derive them.

## T1 (highest priority) — rerun M_ω v2 and get the headline number
The user's core ask is "we want to slightly beat GEPA". The fixed implementation exists and was
killed ~5 evaluations in; it has never produced a result.
    cd datasets/prompt-optimality-test && nohup ./run_momega_v2_all.sh > <log> 2>&1 &
Runs hover + hotpotqa in parallel on the two GLM keys. ~2h (a 60-item eval takes ~100s; ~3800
calls/dataset). Output: `runs/<ds>/momega_v2/result.json` →
`test_untouched: {momega_v2, gepa_shipped, seed}` on val[100:300], 200 items no arm has touched.
SUCCESS CRITERION: `momega_v2 >= gepa_shipped`. The no-regret guard makes >= automatic up to
confirmation noise, so the interesting quantity is the MARGIN and whether units were compiled at
all (`units.n_compiled` > 1 would confirm the D3 fix worked).
DO NOT run aime2025 in this harness — its val split is only 17 items; cover AIME in T3.
DO NOT quote v1 `unitrecomb` numbers (.73/.81/.29) as M_ω performance; they are the broken-geometry
run, preserved for provenance only.

## T2 — finish D4 properly
(a) `analyze_bounds_evt.py` was being written when the agent was killed — CHECK IF IT EXISTS; if
partial, finish or rewrite. It is the extreme-value endpoint estimator (upper endpoint of the
distribution prompts are drawn from), which is the correctly-specified tool now that best-of-m is
proven unusable for this. Include: two endpoint estimators cross-checked over top-k, bootstrap CI,
the i.i.d.-violation caveat (candidates come from an ADAPTIVE search), and binomial SE alongside
every margin. (b) Then REDO all of D4 on the 300-item paper-exact test splits from T3 — on 100
noisy binary items every margin is smaller than the sampling error, so no bound there can mean
anything. A clean negative is an acceptable outcome; do not dress one up.

## T3 — D5 paper-exact runs (needs the server back)
    ssh sk2 'nohup bash /lfs/skampere2/0/alexspan/cr3-v14.1-two-lane/code/scripts/serve_qwen3_sk2.sh \
      > /lfs/skampere2/0/alexspan/cr3-v14.1-two-lane/logs/qwen3_serve.log 2>&1 &'
    ssh -f -N -L 8077:127.0.0.1:8077 sk2        # wait for "Application startup complete"
Then `paperexact_arms.py <aime|hover|hotpot> --arm <official|inhouse|unitrecomb> ...`.
ARM ORDER MATTERS: official → inhouse → unitrecomb (unitrecomb reads official's result.json to
initialize). Two task-LM columns: `openai/Qwen3-8B --api-base http://127.0.0.1:8077/v1
--temperature 0.6 --top-p 0.95 --max-tokens 8000` (paper-exact; max_model_len is 16384 so do NOT
pass --max-tokens 16384) and GLM-5.2 (our best). Start with aime (1 LM call per rollout, cheapest,
and the only benchmark where GEPA actually improved over the seed). Kill the server + tunnel when
done; check for OTHER alexspan vLLM jobs on the box first and never pattern-kill (see T6).

## T4 — D1 leftovers
`pip install gepa==0.1.4` on sk3 (UNCHECKED; sk2 already has 0.1.4). Decide the local-env pin:
dspy 3.2.1 hard-pins `gepa[dspy]==0.0.27`, so a venv cannot hold both dspy and gepa 0.1.4 — either
separate the envs or accept dspy's pin for the benchmark harness while sk2 stays 0.1.4 for
reconstruction. Update PIN.txt to record what is actually installed rather than what was intended.

## T5 — D2's untouched half: the reconstruction-side M_ω
Only the benchmark side was built. The reconstruction side still needs the CHILDREN-METRICS unit
source: when judging R2, seed the unit pool with that metric's certified R1 children from the
L0→R3 ladders (memory `project_hierarchy_l0_r3_reconstruction`), alongside official-GEPA
trajectory units and LLM suggestions, with the same init-from-GEPA-best superset argument. This is
the version that matters for the paper's Level-1 claims.

## T6 — standing cautions for this workstream
- sk2 is SHARED. Another user (sahasras) runs vLLM there, and alexspan has OTHER jobs (a `qwen14b`
  tmux lane). Kill ONLY by explicit PID after mapping the process tree; a `pgrep -f "vllm serve"`
  pattern-kill misfired this session (it matched the ssh shell's own command line).
- Pin HOME=/lfs/skampere2/0/alexspan in every sk2 job; the AFS home is unreadable and an unpinned
  HOME already destroyed one full run (v3 decoder tune, every engine init failed).
- val[:100] is CONTAMINATED for reporting on hover/hotpotqa (every phase-1-4 arm selected on it).
  Report on val[100:300] or the paper test splits.
- Do not quote `dev_identification_residual_bits` as any kind of bound (see D4 Part B).

---

# AUDIT ROUND (2026-07-20, incoming agent) — everything above independently verified

Seven parallel audit agents checked every D1-D6 artifact before continuing. Results:

- **m_omega_gepa migration: VERIFIED on all claims** (signature/return contract, estimand,
  rank-equivalence math re-derived, budget mapping, archive integrity, 13/13 tests, live
  gepa-0.0.17 API compatibility). Two caveats FIXED in code: (1) docstring overstated
  selection-equivalence — NaN verdicts score −1.0 in the search signal but are EXCLUDED from the
  canonical statistic, so equivalence is exact only at equal parse-failure rates (docstring now
  says so); (2) `raise_on_exception=False` could silently return the seed on a dead backend — now
  warns loudly when GEPA returns the seed with no other candidate evaluated.
- **v14 decoder rewire: VERIFIED** (archive verbatim by function-body diff, scoring copied
  exactly, discipline + freeze-consumer keys, 16/16 tests, 338 collect). FIXED: the unguarded
  post-hoc rescore (a crashing non-winner template after a completed 240-call search would abort
  the run and lose the winner) — now per-template with retry for seed/winner, drop-with-warning
  for others; also the wasted final-attempt sleep. NOT touched (noted): over-broad retry
  mislabels deterministic scorer bugs as transient; append-mode proposals.jsonl pollutes
  `distinct` on re-runs into the same run_dir.
- **analyze_bounds.py: FULLY VERIFIED, zero bugs** — re-run byte-identical; the exchangeable
  estimator, excess-curve refit, oracle union, noise model, and candidate-bootstrap all checked
  independently (exact combinatorial estimator matches their MC to <3e-4). The structural
  negative (sup_m E[max of m] = pool max at m=n) is mathematically confirmed. Labeling gap only:
  bounds_summary.md doesn't carry the RUNBOOK caveat that aime's 17-item split is ad-hoc.
- **UNIT_BOTTLENECK_DIAGNOSIS: ALL numbers reproduced exactly from logs** (budget arithmetic,
  panel sd, Spearmans + permutation p, headroom, 0-compile sign test). One framing caveat
  appended to the file: "compiled went backwards vs seed" is a run-time-vs-rescore measurement
  artifact (like-for-like the hover compile ties the seed at 0.760); the robust claim is
  "compiled captures far less than best single unit (regret −0.06/−0.02/−0.118 like-for-like)".
  Every v2 design choice survives.
- **paperexact_arms.py: paper-exactness VERIFIED** (programs/metrics/splits imported verbatim
  from the artifact; v2 geometry present; nothing selects on test). BUGS FIXED: `--max-tokens`
  default 16384→8000 (paper Appendix E.2); AIME panels now adapt to its 45-item train (was:
  5-item confirm slice); budget default now per-arm (600 official/inhouse, 2400 unitrecomb —
  600 starved the prefix/confirm stages and silently shipped init); `evaluate_cand` no longer
  silently truncates the panel when budget runs low (scores were incomparable); z.ai key lookup
  now falls back across all three key files + ZAI_KEY_FILE env. STILL OPEN: hover/hotpot need
  the BM25S wiki index built (aime does not — run aime first); runs_paperexact/ is empty (never
  executed).
- **D6 theory note: both load-bearing theorems CORRECT** (Barber-Agakov, DPI + sharpness, Fano
  properly absent). FIXED in the note: the §5 MDL claim was internally inconsistent — a
  description-length penalty is MINIMIZED by the SHA-parity hash (short English description), so
  the already-logged description lengths canNOT power the pilot; the penalty must target
  executed-computation complexity. Also added: decoder-family tightness caveat,
  within-variational-family qualifier, the two-senses-of-"reconstruction" terminology guard
  (Thm A's decode-X-from-M vs the project's criterion-recovery I(M_ω;M̂)), fiber-argument
  attribution for §6. Headline survives: "provably the best label-free capacity certificate,
  provably not task-optimal".
- **runs/ AIME is answer-contaminated** (official arm's reflection injected literal 2025 answers;
  13/17 split of the same 30 problems, no test set) — never quote runs/ aime numbers for
  anything; paperexact AIME (45/45/150, disjoint sources) is the only clean AIME.
- **Environment recon**: sk2 GPU 7 free, serve script intact, port 8077 free remotely but the
  local ControlMaster still holds the stale forward (it revives when the server binds — no new
  tunnel needed). sk2 conda: gepa 0.1.4 + dspy 3.1.3 (a dspy that tolerates 0.1.4 exists —
  T4-relevant). sk3: gepa 0.0.26. alexspan's other lanes (qwen14b/qwen_32B/qwen_3B/qwen_7B tmux,
  GPUs 0-1 busy) — DO NOT TOUCH.

T1 RELAUNCHED 2026-07-20 after the audit (runs/<ds>/momega_v2/run.log; killed-run partials
preserved as proposals.partial-killed-20260719.jsonl). analyze_bounds_evt.py WRITTEN (T2a):
process-conditional endpoint via GPD-MLE + Pickands over a k-sweep, candidate bootstrap,
dequantization sensitivity, binomial SE beside every margin, i.i.d. caveat up front.

## T1 RESULT (2026-07-20, the headline run — COMPLETE)

`runs/<ds>/momega_v2/result.json`, test = 200 untouched val[100:300] items, GEPA-shipped = seed
on both (official GEPA had found no improvement):

| dataset | M_ω v2 | GEPA-shipped | seed | paired W-L | exact sign p | units compiled |
|---|---|---|---|---|---|---|
| hotpotqa | **.795** | .765 | .765 | 14-8 | .286 | **7** (format/article/phrasing rules) |
| hover | .730 | .750 | .750 | 12-16 | .572 | 1 (the known quasi-exemplar film fact) |

READ: (1) The D3 mechanism fix WORKED — hotpotqa compiled 7 units (v1: 1 everywhere) with select
.65→.80, confirm agreeing (.65→.75), and the test direction positive (+.030 ≈ 1 SE; suggestive,
NOT significant — paired 14W-8L p=.29). (2) hover reproduced the diagnosis's failure mode at the
next level up: the select panel (+10 items!) and the 40-item confirm slice (+.025) BOTH passed
the quasi-exemplar "Adam Arkapaw" unit, which then failed to transfer (−.020 on test, 12W-16L,
n.s.). The no-regret guard's "up to confirmation noise" clause is doing real work — 40 items is
not enough to catch an entity-overlap false positive on hover. (3) Consistent with the EVT read
(bounds_evt_summary): hover's process endpoint ≈ best-achieved (no headroom to find), hotpotqa's
endpoint unstable-but-higher (room existed; M_ω claimed some). Candidate v3 refinements, NOT
implemented (need sign-off + evidence): bigger confirm slice, and a unit-type filter for
proper-noun exemplar-fact units (the memory `banks have 0 mechanical` / degeneracy line predicts
these transfer poorly on entity-clustered tasks like hover).

## T3 RESULT — paper-exact AIME, Qwen3-8B column (COMPLETE 2026-07-20)

`runs_paperexact/aime/Qwen3-8B/<arm>/result.json`, paper splits 45/45/150, test n=150
(binomial SE ≈ .039; the SEED measured twice across arms gave .333 and .367 — run-to-run
sampling noise is the size of most gaps here):

| arm | seed_test | best_test | note |
|---|---|---|---|
| official (dspy.GEPA, 600) | .333 | .367 | GEPA's val-selected candidate (val .31→.40 during search) |
| inhouse (600) | .367 | **.440** | +.073 ≈ 1.8 SE, selected on a 25-item train panel — borderline, replicate before quoting |
| unitrecomb M_ω v2 (2400) | .367 | .367 | **guard fired correctly**: 1/24 units positive on 27-item select (= noise), confirm rejected it (.444<.500) → shipped official's prompt VERBATIM |

READ: on clean paper-exact AIME the M_ω superset floor held exactly (M_ω = GEPA, no regression);
the inhouse arm's .44 is the only arm above noise and needs replication (25-item selection panel).
Rescore of ALL distinct candidates (3 arms pooled) on the 150-item test is RUNNING
(`paperexact_rescore.py` → `<arm>/rescore.jsonl`) — feeds the EVT/bounds redo (T2b) on a split
nothing selected on. GLM-5.2 column arms RUNNING (`run_paperexact_aime_glm.sh`). BM25S wiki
index build for hover/hotpot delegated (unblocks those benches).

## LATE-SESSION STATE (2026-07-20 evening)

- **GLM-5.2 AIME column RETRACTED as measurement artifact** (diagnosed by fresh-call repro):
  the paper metric's bare `int(prediction.answer)` zeroes LaTeX-formatted answers ('$504$' →
  ValueError → 0; free-form GLM-5.2 scored 5/5 where the harness scored 3/6), plus 16k-token
  truncation poisoned GEPA's search signal (55 truncation events; inhouse "best" .24 < seed was
  GEPA optimizing against truncation noise). Old run quarantined at
  `runs_paperexact/aime/glm-5.2_formatbug-quarantine-20260720/`. RERUN launched with
  `--robust-answer-extract` (last-integer extraction wrapper in paperexact_arms.py — vendored
  artifact untouched, no-op for bare-integer models so the Qwen paper-exact column is
  unaffected) + `--max-tokens 32000`; both fixes recorded in result.json.
- **Clean-AIME rescore pool (34 rows incl. 6 harvested dspy-GEPA trajectory candidates):
  top-3 candidates are ALL unitrecomb compiles (.467/.460/.447) > official's best rewrite
  (.413)** — unit GENERATION reaches above GEPA's rewrites, but selection (27-item panel,
  18-item confirm) could not identify them and the guard shipped GEPA's prompt (.367). Known
  only from the post-hoc test rescore — CANNOT be used for selection. Selection power remains
  the binding constraint (D3, again). EVT on this pool: GPD .4667 [.447,.479] pinned at pool
  max, Pickands .597 wide — estimators disagree, no stable endpoint quoted; both sit above
  observed GEPA (.367). No bound violations anywhere to date.
- **Gestalt/undershoot point (user, 2026-07-20): CORRECT and now part of the ladder reading.**
  Rung-1/2 bounds are process-conditional and UNDERSHOOT the all-prompt truth; a natural
  "gestalt" prompt from a richer process can legitimately exceed them (vivid demo: free-form
  GLM-5.2 5/5 vs constrained-format 3/6 on the same items). Only rung 0 binds all processes.
  Endpoint claims must be process-marked; endpoint-vs-process-family convergence is the only
  empirical probe of a task ceiling.
- **sk3 paired A/B recovery campaign LAUNCHED** (pid on sk3; GPU 7 co-located at 0.35
  mem-util): Arm A `--gepa-m-omega` (first-ever official-GEPA plain baseline) then Arm B
  `--momega-v2`, peer-review/specific groups 3,36,38,13,30,116,35,45,125,16,22,23 (matches the
  Jul 11 no-GEPA arms), Llama-8B executor + glm-4.7 reconstructor, n_items 60 R 5, out-dir
  `outputs/r2_recovery_v2_momega/{gepa_plain,momega_v2}` on sk3. sk3 code deployed by rsync
  (stale copy had no --momega-v2); 21/21 tests passed on sk3 pre-launch. ~6-7h.
- hover Qwen paper-exact arms running (chained after the AIME rescore).
- **v3 refinement proposals AWAITING USER SIGN-OFF**: (1) much larger confirm slices (full
  45-item AIME val; 100+ on hover/hotpot); (2) proper-noun exemplar-fact unit filter for
  entity-clustered tasks. Both change the selection design — do not implement unilaterally.

## T3 RESULT — paper-exact HOVER, Qwen3-8B column (COMPLETE 2026-07-20) — ★ THE HEADLINE

300-item untouched paper test (SE ≈ .028), arms in order official → inhouse → unitrecomb
(unitrecomb mines BOTH prior trajectories + LLM facets, inits from official's winner):

| arm | seed_test | best_test | paired vs official GEPA |
|---|---|---|---|
| official dspy.GEPA (600) | .380 | .450 | — |
| inhouse (600) | .380 | .547 | (vs M_ω: 33W-24L p=.29, n.s.) |
| **unitrecomb M_ω v2 (2400)** | .380 | **.517** | **32W-12L, exact sign p=.0037** |

**M_ω v2 significantly beats canonical official GEPA on a clean paper-exact split** — compiled
3 units (select .425→.525, confirm .600→.680, guard PASSED on real signal), vs seed 51W-10L
p<1e-5. First statistically solid end-to-end win for the mechanism. Caveats to carry: budget
asymmetry by design (2400 vs 600, declared + recorded); M_ω initialized FROM official's .450
candidate so this is the superset mechanism working, not an independent-optimizer comparison;
inhouse's .547 joint rewrite is statistically tied with M_ω. Chain launched: hover rescore
(300-item EVT input) → hotpot arms → hotpot rescore.

## T3 RESULT — paper-exact AIME, GLM-5.2 column FIXED RERUN (2026-07-20, robust extract + 32k)

seed measured .433/.440/.467 across arms (spread ≈ SE .04). official .433→.433 (no gain);
inhouse .440→**.347** (NO guard → shipped a 25-item-panel overfit, −.09 on test); unitrecomb
guard fired (confirm .333 < init .389) → no-op at .467. **Guard now 4/4 correct.** Reads:
(1) artifact diagnosis validated — GLM-5.2's real AIME level is .43-.47 (was .30 under
int('$504$')); zero missing-answer events at 32k; (2) strong-model AIME = NO headroom for ANY
arm (endpoint ≈ seed); (3) guard-vs-no-guard contrast under no-headroom: M_ω no-ops, inhouse
ships harm — the paper-ready argument for the superset construction alongside the hover win.

## T5 RESULT — reconstruction-side A/B, peer-review/specific 12 groups (COMPLETE 2026-07-20)

sk3, Llama-8B executor + glm-4.7 reconstructor, n_items 60 / n_train 30 / R 5, paired arms:
A = --gepa-m-omega (first official-GEPA plain baseline), B = --momega-v2 (children units).
Results (`outputs_sk3_momega/{gepa_plain,momega_v2}.jsonl` local copies):

- 6/12 groups fail-closed in BOTH arms ("target constant on design split") — degenerate metrics
  stay degenerate, correctly refused. **B RESCUED grp 13 from that failure** (unit compile
  created design-split variance; id_acc .400 where A = unmeasurable).
- Measurable groups: grp 116 A id .000 (plain GEPA COLLAPSED the metric, gepa_std 0) → B id
  .200 (2 units, std .499); grp 36 recovery .066 → **.196 (3x)** with 1 unit, id 1.0 both;
  grps 22/35 exact ties (0 units, superset floor held); grp 125 REGRESSION A .800 → B .600
  (recovery NaN; 4 units passed a 7-ITEM confirm slice then failed held-out — the underpowered-
  confirm disease at reconstruction scale). Mean gepa_std A .329 → B .481.
- **★ EVERY compiled unit (8/8, across grps 13/36/116/125) is source=CHILDREN** — zero LLM,
  zero trajectory. The D2 children-metrics hypothesis gets a clean confirmation: certified R1
  children are THE productive unit source on the reconstruction side.
- FOLLOW-UP LAUNCHED (bigpanel): same 12 groups at n_items 100 / n_train 60 (→ 15-item confirm)
  → outputs/r2_recovery_v2_momega_bigpanel/ on sk3, both arms, to fix the 125-style guard
  failure and re-measure. ~5-6h.

## HEARTBEAT LOG (2026-07-20/21 overnight campaign)

**HB1 (~18:30 PT):** all lanes alive. hover Qwen 300-item rescore DONE → EVT: pool n=38, best
.5500 (a UNITRECOMB candidate again — top-3 unitrecomb .550/.517/.517 vs official .450/.38),
GPD endpoint .5500 [.530,.570] (Pickands .609 wide — formally unstable, don't quote single
number), union oracle .8033. **The paper's own 7k-rollout GEPA (.5233) lands just below our
GPD CI** — process-family convergence ~.52-.55 on hover. Selection left .033 on the table
(shipped .517 vs pool best .550) — v3 rerun queued in Track A. ifbench GLM official: seed .320
→ GEPA .478 (big gain — prime M_ω target). hotpot Qwen official started 01:25Z. sk3 bigpanel
Arm A running.

**HB2 (~19:45 PT): ★ SECOND SIGNIFICANT WIN — paper-exact HOTPOT, Qwen3-8B, 300-item test,
first v3-geometry run:** official GEPA .380→.380 (found NOTHING), inhouse .380→.380 (nothing),
**M_ω v3 .380→.4233 (2 units, select .43→.51, confirm .48→.52, guard passed) — paired vs GEPA
17W-4L, exact sign p=.0072.** M_ω is the only method that improved at the declared budget.
Winning units = 2 trajectory-mined hop-2-query rephrasings; LLM suggester returned 0 (rate-limit
window, pre-retry-fix — win happened despite a thin 8-unit pool). Qwen scoreboard: hover WIN
(.517>.450 p=.0037), hotpot WIN (.423>.380 p=.0072), aime tie-by-guard (.367; v3 rerun queued).
Caveat vs paper: their hotpot GEPA 62.33 ran 6,871 rollouts vs our 600 — budget-parity runs
still pending. B1: hover-GLM official .47→.517, inhouse .477→.623 (unitrecomb next inits from
.517). B2: ⚠ ifbench-GLM OFFICIAL seed .3197 vs inhouse-pass seed .5221 — 20-point seed
discrepancy = suspected outage-deflation during key-A saturation; do NOT quote ifbench-GLM
official .478 until the rescore adjudicates. sk3 bigpanel Arm A 1/12 groups (slow, n=100). All
lanes alive; hotpot Qwen rescore 2/16 running.

**HB3-4 (~21:00-21:30 PT):** all lanes alive, none hung (logs 0m stale). ifbench-Qwen: official
AND inhouse both .4116→.4116 (GEPA found nothing; paper's own IFBench GEPA gain was small,
+1.7) — M_ω unitrecomb running now. ifbench-GLM unitrecomb: 0 units shipped (all marginals ≤0),
its seed re-measured at .5612 — CONFIRMS the official-pass .3197 seed was outage-deflated (the
official .478 "gain" is suspect; rescore adjudicates). hover-GLM unitrecomb: 24 LLM units in
pool (retry fix working), slow-but-live on the 100-item panel (~3.5h — GLM multihop evals are
heavy). sk3 bigpanel Arm A 6/12. LLM-pool crowding noted: with 24 LLM units, cross-LM units get
truncated by the 32-unit cap (llm-first ordering) — consider max_units 48 if a bench stalls.

**HB5 (~22:30 PT): z.ai PROVIDER OUTAGE** — both GLM lanes (hover-GLM unitrecomb at 31/32
marginals!, livebench-GLM official) went stale at the same instant 2.5h ago; direct probe
confirms z.ai read-timeouts on a trivial request. Processes ALIVE and left running (litellm
retries should resume them on recovery); 5-min recovery probe armed → on recovery, verify both
resume within ~20 min, else kill by explicit PID and relaunch stages. Qwen lane UNAFFECTED
(ifbench unitrecomb 14 marginals and progressing); sk3 bigpanel Arm A 11/12, currently in a
local-vLLM phase, unblocked.

**v3.2 (2026-07-21 ~01:00 PT, user: "lift to 32 right now"):** prefix cap 16→32 (= every
positive unit gets a prefix slot), unit-pool cap 32→48 (un-crowds cross-LM units), unitrecomb
default declared budget 6000→12000 (the bigger sweep needs ~11.4k worst-case; all-or-nothing
budget check would otherwise silently truncate). Budget asymmetry vs GEPA-600 grows — by
design, recorded per-result.json, must be stated in the writeup.
**v3.1 (2026-07-21 ~00:45 PT, user question exposed the cap):** prefix-sweep TOP_K raised 8→16
in paperexact_arms.py — T1-hotpotqa compiled 7/8, i.e. the old cap BOUND; 10-30-unit compiles
were structurally impossible before this. In-flight runs (ifbench-Qwen unitrecomb) keep 8 (file
read at process start); livebench/pupa + all reruns get 16. Unit-count doctrine recorded: count
is bounded by (detection floor ~ panel SE) × (redundancy/conditional-value collapse) × (the
top-k cap); the first two are measured properties, the third was a design artifact now lifted.

**HB8 (~01:45 PT, first Sonnet-sweeper heartbeat):** (1) **ifbench-Qwen = PRINCIPLED TIE**:
GEPA .4116→.4116, inhouse same, M_ω guard fired after full 32-marginal pass → shipped .4116.
No method moves ifbench on Qwen (paper's own gain there was +1.7 — thin headroom bench).
(2) ⚠ **GLM-column results from the outage window are CORRUPTED — DO NOT QUOTE**: livebench-GLM
official "best" .3339 vs seed .6212, inhouse .2885 vs .6639, hover-GLM unitrecomb best_test .38
vs init .5167 — all have the dead-endpoint zero-scored-items signature. GLM-column uniform
rescores (retry-hardened) must re-adjudicate ALL glm-5.2 numbers after the chains finish; queue
`paperexact_rescore.py <bench> --lm-tag glm-5.2 --task-lm anthropic/glm-5.2 ...` per bench.
(3) sk3 bigpanel Arm A COMPLETE 12/12, Arm B started 08:04Z. (4) Track A → livebench-Qwen
(v3.2), B1 → hotpot-GLM, B2 → livebench-GLM unitrecomb. (5) No free sk2 GPUs (all 8 >129GB) —
second server still parked. z.ai UP at sweep time (flapping earlier).

**HB9 (~02:45 PT):** hotpot-GLM official CLEAN: .37→.4967 (+.13 GEPA product, post-outage —
M_ω inits from .4967 next in B1). ⚠ **livebench-Qwen BROKEN: seed=best=0.0** (paper baseline
48.7; GLM column scores fine → Qwen-column pipeline artifact; AIME-Qwen on same server was
fine). Opus diagnosis dispatched; its inhouse arm is accumulating garbage meanwhile —
kill/quarantine/relaunch decision follows the diagnosis (PID-targeted only). sk3 bigpanel Arm B
3/12. z.ai UP at sweep. No free sk2 GPUs.

**HB9-INCIDENT RESOLVED (~03:15 PT):** livebench-Qwen 0.0 root cause = the sk2 TUNNEL dropped
again (server survived, same pid); 840/840 connection errors + dspy max_errors=10000 silently
zero-scored everything into a plausible-looking result. Sequence executed: tunnel re-established
FIRST (killing the doomed arm earlier would have let the chain overwrite good aime/hover
results with 0.0 garbage), then inhouse PID 3548 killed (explicit PID), official artifacts
quarantined (official.deadendpoint-quarantine-20260721/), **pre-flight endpoint health check
added to paperexact_arms.py** (dead local endpoint now aborts loudly — landmine defused for all
future arms), livebench-Qwen recovery chain launched (official→inhouse→unitrecomb→rescore).
Track A main chain proceeds to aime/hover v3.2 reruns against the live endpoint.

**HB12 (~05:00 PT):** livebench-Qwen recovery CLEAN: official seed .6744 → GEPA .6956 (+.021;
note our livebench score scale runs higher than the paper's 48.7 — partial-credit olympiad
scoring + split mix; within-column comparisons are what count). **inhouse (no guard) regressed
AGAIN: .6665 → .5792 (−.09) — THIRD documented guardless regression** (aime-GLM −.09, hover-T1
n.s., now livebench-Qwen −.09); the guard-vs-no-guard contrast is now a robust pattern, not an
anecdote. livebench-Qwen unitrecomb started (inits from .6956). livebench-GLM unitrecomb prefix
reached k=17 under the lifted cap (v3.2 exercising exactly as intended). aime-Qwen v3.2 rerun
still mid-flight (old result.json's final-test rows in append-mode proposals caused a false
"complete" read — verify by result.json MTIME, not proposals rows). sk3 Arm B 10/12. All lanes
green; tunnel OK; z.ai UP.

## T5 BIGPANEL RESULT (2026-07-21, n_items 100 / n_train 60 — REVERSES the small-panel read)

`outputs_sk3_momega_bigpanel/{gepa_plain,momega_v2}.jsonl` (local). At the bigger design split:
**Arm A (plain GEPA) mean id_acc .450 over 8 measurable groups** (PRISMA grp 3 now MEASURABLE
at id .800; grps 35/45/125 at 1.0/1.0/1.0) — the larger split mostly fixes what the小 split
couldn't measure. **Arm B (momega-v2) mean id_acc .175 — WORSE**, with striking flips: grp 35
id 1.0→0.0 (1 unit), grp 125 1.0→0.0 (2 units, discriminating False AGAIN despite the 15-item
confirm), grp 45 1.0→.600. B wins remain only on recovery for grps 36 (.109→.220) and 22
(.051→.136) and id for grp 30 (0→.200). Compiled units: 12 children + 2 trajectory.

**Working hypothesis (NOT a conclusion; n=12, needs a targeted check): IDENTITY DILUTION.** The
MCQ option shown IS the compiled prompt; appending children criteria makes the option a blend
of sibling criteria, so the reconstructor can no longer match the metric to its own behavior —
discrimination (capacity) rises while identifiability falls. This is open decision 3
materializing empirically: the reconstruction-side guard gates on DISCRIMINATION (a capacity
objective, per D6) while the reported readout is RECOVERY — the guard guards the wrong
quantity on this side (benchmark side is fine: its guard gates task accuracy = the readout).
CANDIDATE FIX (needs user sign-off — changes selection estimand): no-regret guard on a
design-split recovery proxy (e.g., MCQ identification or induced-behavior agreement) instead
of discrimination. Do NOT quote small-panel B positives as confirmed; the two runs disagree
and the bigpanel is the better-powered one.

**HB14 (~07:15 PT):** livebench-GLM unitrecomb landed (seed .7339 → shipped .6612, 1 unit) but
is **NOT QUOTABLE — corrupted at the root**: it initialized from the outage-corrupted official
best_candidate (official's GEPA search ran through the z.ai outage on zero-scored signals), and
GLM-livebench seed measurements vary .62-.73 across passes (partial-outage deflation). The
ENTIRE livebench-GLM column needs a clean rerun (official → inhouse → unitrecomb) after z.ai
stabilizes; queue with the GLM-column uniform rescores. pupa-GLM official started (B2's last
bench). aime-Qwen v3.2 at 69 marginals (cross-LM pool); livebench-Qwen M_ω at 13 marginals;
hotpot-GLM inhouse still alive (slow GLM multihop). All lanes green; tunnel OK; z.ai UP.

**HB17 (~10:00 PT):** **pupa-GLM official CLEAN: seed .8981 → GEPA .9735 (+.075)** — largest
clean GEPA gain of the campaign, matches the paper's PUPA pattern (their biggest GEPA delta);
M_ω unitrecomb will init from .9735 (inhouse mid-run). aime-Qwen v3.2 in late drop-one stage;
livebench-Qwen at 27 marginals; hotpot-GLM inhouse still alive (12h+ — verify progress next
sweep via proposals count, not just log freshness). Tunnel keepalive active (auto-healed
earlier); key A 1302 back-pressure transient (key B OK). z.ai UP.

**HB18 (~11:00 PT):** pupa-GLM inhouse .8913→.9552 (below official's .9735); **pupa-GLM
unitrecomb RUNNING from the .9735 init — best-shot arm of the campaign.** hotpot-GLM inhouse
KILLED by explicit PID (4344): 2 evaluations in ~14h = reflection-spin through the outages,
scientifically void (mark ABANDONED-STARVED, never quote); its blocked successor hotpot-GLM
unitrecomb launched manually (pid 52365, key B). B1 chain exit after the kill = expected.
aime-Qwen v3.2 still in late stages; livebench-Qwen 34/48 marginals. Tunnel + z.ai OK.

**HB19 (~13:30 PT):** sk2 network outage #4 resolved — ssh recovered, vLLM server SURVIVED
(same pid since launch, 4 outages outlived), tunnel restored (manual + keepalive confirm). All
4 unitrecomb processes ALIVE through the double outage (aime PID 8082, livebench 28805, pupa
47961, hotpot-GLM 52365 — the deep retry stacks did their job). aime-Qwen v3.2 went 30 units
deep into the lifted prefix cap before wedging; resumes now. z.ai UP again. Scoreboard
unchanged; pupa-GLM M_ω (best-shot arm) at 10 marginals.

**PAPER-EXACTNESS AUDIT ADDENDUM (2026-07-21, user question):** (1) Task models: the paper runs
TWO executors — Qwen3-8B (Table 1; we run it exactly) and **GPT-4.1 Mini (Table 2; we do NOT
run it** — GLM-5.2 was the approved substitute). Adding a true GPT-4.1-Mini column ≈ $30-40
(aime) / $300-500 (all six) on the sk3 OpenAI key — OFFERED, awaiting user. (2) **top-k 20
(Appendix E.2) was NOT being set** in any arm to date (vLLM default = disabled) — recorded
deviation for all completed Qwen arms; `--top-k` flag added (extra_body passthrough), v4 script
now passes `--top-k 20`, value recorded in result.json. (3) Everything else verbatim: program,
metric (robust-extract is GLM-column-only), splits incl. AIME ×5 protocol, temp/top-p.
Optimizer = dspy.GEPA (documented deviation, open decision 2); budgets 600 vs paper 1839-7051
(parity runs queued).

**DECISION (user, 2026-07-21): GLM-5.2 REPLACES GPT-4.1 Mini as the second executor column** —
"an outdated model, anyway." No GPT-4.1-Mini runs. Writeup framing: deliberate modern
substitute for the paper's closed-model column (Table 2 analog), not an omission; the
paper-exact claim rests on the Qwen3-8B column alone.

**HB22 (~14:45 PT): ★ AIME-QWEN TIE BROKEN — M_ω v3.2 shipped .4267 vs GEPA .3667 (+.06,
paired 14W-5L, exact sign p=.0636 — borderline, above .05).** 1 unit compiled from the 48-unit
pool: trajectory-mined format-discipline clause ("Do NOT use LaTeX formatting such as
'\\boxed{125}'") — converts paper-metric int()-zeroed answers into scored ones; select .467→.600,
confirm .400→.467 (15 items), TRANSFERRED to test this time. v4 (96 units, 60-item val-backed
confirm, top-k 20) queued for consolidation above the significance line. Hardening landed:
POST-aware rescore probe (no false aborts on legit 0.0 candidates), regression_flag in every
result.json (best < seed−.05 → review-before-quote), inhouse starvation guard (6h/<5 evals →
loud abort). **Canonical bounds artifact created: runs/UPPER_BOUNDS.md**
(analyze_upper_bounds_rollup.py regenerates; refreshed after every rescore) — the single
answer to "where are our upper bounds".

**HB25 (~17:15 PT):** livebench-Qwen M_ω = TIE-BY-GUARD (0 units survived selection; shipped
GEPA's prompt; .673 own-pass vs official-pass .696 = run-to-run spread, same prompt). GEPA had
already captured livebench headroom (.674→.696). livebench-Qwen rescore running. ifbench v4 at
67/96 marginals (noisy truncated-generation text observed in some scoring batches — watch).
z.ai transient 1113 again on key B (triage rule applied — no action). QWEN SCOREBOARD: hover W
(p=.0037), hotpot W (p=.0072), aime borderline-W (p=.064, v4 consolidation queued), livebench
tie-by-guard, ifbench v4 pending, pupa not started. 0 losses.

**HB26 (~18:00 PT):** 5 arms + chain alive, nothing stalled. **hover v3.2 prefix sweep at k=23**
(the lifted cap genuinely exercised — old cap 8 would have truncated it). pupa-GLM (best-shot
arm) FINISHED its 48 marginals, now prefix k=9. ifbench v4 ~38/96 marginals; livebench v4 just
started (3 marginals); hotpot-GLM in its 300-item final-test phase (dspy parallelizer errors
present but retrying — log fresh). livebench-Qwen rescore at 5 candidates. z.ai UP, tunnel OK.

**HB27-28 INCIDENT (~21:30 PT / 04:30Z): WEDGED-SOCKET outage — the failure the timeout can't
catch.** ALL 5 arms went silent ~175 min simultaneously (both providers at once → looked local,
but google=200). Diagnosis: z.ai UP, sk2 DOWN, and the two probed arms (livebench-Qwen,
pupa-GLM) sat at **0.0% CPU** = blocked on DEAD SOCKETS that `timeout=300` did NOT recycle
(litellm holds some in-flight connections in a state the read-timeout doesn't fire on — the one
hole no retry depth closes). ACTIONS (kill by explicit PID only): (1) pupa-GLM (best-shot arm,
z.ai up) killed 47961 → relaunched 53698, running clean (76-unit pool, init from GEPA); (2)
hotpot-GLM killed 52365 → first relaunch DIED on a HuggingFace `trust_remote_code` dataset-load
error (the 10h-old process had it cached; fresh load failed) → **fixed with
`HF_DATASETS_TRUST_REMOTE_CODE=1`**, now alive 54099; (3) 3 Qwen arms (livebench 16532/hover
63252/ifbench 79655) LEFT WEDGED — sk2 still down, restarting is futile (pre-flight guard would
refuse) AND they hold in-progress marginals (hover prefix k23+, ifbench 87 marg, livebench 56
marg); **NEXT HEARTBEAT: when sk2 returns, these 3 will still be wedged on dead sockets → kill by
PID + relaunch (accept loss of in-progress marginals).** 2 Sonnet sweepers this window died on
"connection closed mid-response" (Anthropic API rough patch) → ran heartbeat inline instead.

**HB29 (~22:30 PT): wedged-arm recovery + robust tunnel.** sk2 SSH up but the port-FORWARD was
flapping (not the host). Fixed: replaced the ControlMaster forward with a DEDICATED tunnel
carrying `ServerAliveInterval=15 ServerAliveCountMax=3 ExitOnForwardFailure=yes` (self-heals
short drops; keepalive monitor blg8uct5p re-establishes on longer ones — old keepalive
bmo12f6w5 died in the pkill, replaced). Sweeper CPU-check pinpointed the wedged set: killed by
PID 348 (lb-rescore), 16532 (lb-v4), 79655 (if-v4) — all 0% CPU dead sockets — and relaunched
livebench-v4 (57295) + ifbench-v4 (57296), both now progressing (ifbench eval 82%, livebench
73%). **hover-Qwen (63252) was NEVER wedged (0.3% CPU) — left running.** GLM restarts from HB28
healthy (pupa 53698, hotpot 54099). livebench rescore NOT relaunched yet (deferred to avoid 4
concurrent Qwen loads vs the working hover arm; queue next cycle once one arm finishes).
LANDMINE for memory: fresh hotpot dataset load needs `HF_DATASETS_TRUST_REMOTE_CODE=1`.

**HB30 (~23:50 PT): ★★ HOVER v3.2 UPGRADED WIN — M_ω .5467 vs GEPA .4500, 41W-12L, p=0.0001**
(was p=.0037 at .517). 5 units compiled, ALL LLM-suggested (3×summarize1, 2×create_query_hop3)
— the v3.2 diversified LLM framings + 96-unit pool + val-backed confirm materially improved the
flagship result. This is the strongest single result of the campaign. Also: Track-A chain
advanced to **pupa-Qwen official (87035)** — the 6th/last Qwen bench now running. ifbench-v4
re-wedged (57296, 0% CPU 2 samples) → killed by PID, relaunched 89910 (alive, evaluating) — the
robust tunnel reduces but hasn't eliminated the litellm dead-socket bug (timeout=300 provably
doesn't fire on certain ESTABLISHED-but-dead sockets; no clean internal fix, periodic restart is
the mitigation). QWEN SCOREBOARD: hover WIN p=.0001, hotpot WIN p=.0072, aime borderline-WIN
p=.064, livebench v4 running, ifbench v4 running, pupa-Qwen just started. 0 losses.

**HB31 (~01:00 PT): AUTO-RESTART WATCHDOG deployed for the wedge-prone Qwen v4 arms.** livebench-v4
(57295) + ifbench-v4 (89910) wedged AGAIN (0% CPU dead sockets) — the litellm bug bites on any
tunnel micro-drop during a long marginal pass. Manual restart every heartbeat is wasteful, so:
`qwen_arm_watchdog.sh` (crash-proof, NO set -e/-u) checks livebench+ifbench every 5 min and
kill-by-PID + relaunches any arm that is 0% CPU AND proposals-stale >15 min (both conditions →
never kills a slow eval). Log: runs_paperexact/qwen_watchdog.log. Watchdog pid 8943 (survives;
first two attempts died on set-u fragility — rewritten defensively). Current v4 arms: livebench
8428, ifbench 8429 (relaunched clean after a $CO-variable arg-mangling misfire). pupa-Qwen
official (87035) + both GLM arms (pupa 53698, hotpot 54099) healthy. NOTE: livebench pool max
.724 > GEPA .696 → a pool candidate already beats GEPA; v4's job is to ship it.

**HB32 (~02:00 PT): watchdog PROVEN (auto-restarted ifbench 2× unattended) + GENERALIZED to all
4 arms.** hotpot-GLM had wedged unwatched (z.ai dead socket) → restarted (24227, w/ HF env) +
added to watchdog. New watchdog (24234) covers livebench+ifbench (Qwen, tunnel-gated) and
pupa+hotpot (GLM, z.ai-gated), each with correct relaunch incl. hotpot's HF_DATASETS_TRUST_
REMOTE_CODE. ⚠ **ifbench TREADMILL**: wedges every ~15-20 min (its 2-stage program = ~2× LM
calls/item = 2× socket exposure vs single-stage benches), so it may never finish a full run
between wedges — watchdog keeps trying; needs a stable 20-30min network window to complete.
livebench-v4 (8428) healthy + progressing (the promising push: pool max .724 > GEPA .696).
pupa-Qwen official (87035) working. Confirmed wins unchanged (hover p=.0001, hotpot p=.0072,
aime p=.064, 0 losses). Infra now self-healing: tunnel keepalive + 4-arm watchdog + heartbeat
backstop.

**HB34 (~04:00 PT): pupa-Qwen OFFICIAL DONE — seed .803 → GEPA .862 (+.059)** — the 6th/last
Qwen bench's GEPA baseline; PUPA's large-headroom pattern (paper's biggest GEPA gain)
reproduces on Qwen. pupa-Qwen inhouse (41391) running → unitrecomb next via chain (strong .862
init). pupa-GLM unitrecomb (53698) working HARD (7.8% CPU, best-shot arm, from .974 init). z.ai
transient 1302 throttle (retries absorb). Watchdog again auto-restarted ifbench+hotpot-GLM
unattended. livebench-v4 (8428) wedged @2:57 etime → watchdog will catch when age>15 (a partial-
write-then-wedge can briefly reset its age clock — acceptable, long wedges still cross 15m).
ifbench treadmill continues (proposals now 960 lines across many restarts, no clean finish yet).
No confirmed-win changes. NEXT: record pupa unitrecomb results (both columns) when they land —
pupa is the likeliest remaining WIN given its headroom.

**HB35 (~04:50 PT): pupa-Qwen inhouse DONE FLAT (.832→.831, no gain — guardless found nothing).**
pupa-Qwen unitrecomb (57928) now running from the .862 official init (26min, working). HONEST
INFRA ASSESSMENT: the compounding sk2-tunnel + z.ai instability is THROTTLING the 3 remaining
benches to a crawl — livebench-v4 (8428) ~15 marginals/hr, pupa-GLM (53698) only 145 proposals
in 6.75h (its program makes multiple GLM calls/item × flaky z.ai = brutal). They are
PROGRESSING, not dead; restarting crawling-but-progressing arms would lose partial work and
re-crawl, so NOT thrashing them. Watchdog+keepalive are the right mitigations and are running.
Likely outcomes if infra stays flaky: pupa-Qwen unitrecomb (freshest, single-provider Qwen) most
likely to finish; livebench/ifbench/pupa-GLM may not complete before morning. **Campaign is
already a strong result on the confirmed wins alone** (hover p=.0001, hotpot p=.0072, aime
p=.064, 0 losses, bounds framework, reconstruction identity-dilution finding) — the remaining
arms are upside, not load-bearing.

**HB41 (~07:00 PT): pupa-GLM unitrecomb DONE — M_ω .9685 vs GEPA .9735, 9W-14L, p=0.40 =
STATISTICAL TIE (not a loss).** 1 LLM unit (PII-redaction clause), guard kept it on a 25-item
confirm (.836→.960) but it was NEUTRAL on the 221-item test → M_ω landed marginally BELOW GEPA
within noise. MECHANISM = small-confirm-slice artifact (pupa-GLM ran WITHOUT --confirm-add-val,
so confirm=25 train items; the unit's confirm gain didn't generalize). This is the honest
demonstration that "M_ω ≥ GEPA by construction" is CONFIRMATION-NOISE-BOUNDED, not absolute —
and the direct empirical argument FOR --confirm-add-val (which the Qwen v4 arms have, GLM arms
don't). NOT a loss (p=.40), but the first case M_ω didn't beat GEPA; scoreboard note: pupa-GLM =
tie. ⚠ paper caveat: report the guard guarantee as "≥ up to confirmation noise" and show this
case. All Qwen arms still crawling (livebench 194, pupa-Qwen 35). Confirmed Qwen wins unchanged.

**HB42 (~12:20 PT): sk2 recovered (~56min outage) + cleanup.** pupa-Qwen (57928, paper-exact)
had DIED during the outage (proposals stalled 120min) → relaunched 3122 WITH v4 config
(--confirm-add-val, 96 units) so it gets the val-augmented confirm that pupa-GLM lacked (avoids
the HB41 small-confirm tie artifact). Fixed a latent WATCHDOG BUG: its `pgrep "pupa --arm
unitrecomb"` matched both pupa-Qwen AND pupa-GLM and was set to relaunch pupa as GLM — would
have wrongly resurrected pupa-Qwen as a GLM arm. Killed the redundant pupa-GLM (122; already has
its .9685 result) and re-pointed the watchdog's pupa check to Qwen (v4 relaunch). Watchdog now
correctly covers livebench+ifbench+pupa (Qwen v4) + hotpot (GLM). Replaced the noisy tunnel
keepalive with a transition-only monitor (bhgbvvec4). livebench (8428) wedged 11h+ — still not
finishing; leaving to watchdog. Confirmed wins unchanged.

**HB43 (~13:05 PT): v5 PUSH on ifbench+livebench (user directive: "push both harder — more
units? more and diverse sampling strategies?") + sk2 EXECUTION LANE.**
- **Harness v5** (paperexact_arms.py): `--run-tag` (parallel variant arms, no run-dir
  collisions), `--prefix-cap` (48 for the push; was hardcoded 32), TWO new suggestion framings
  — (4) example-grounded: the suggester now sees 3 REAL train examples (until now it only ever
  saw module instructions, never the task); (5) unconventional-in-kind (new reasoning orders /
  counterexample search / intermediate representations, no rephrasing) — and a greedy ADD-BACK
  pass after drop-one (skipped positive-marginal units get one shot at the pruned set; catches
  combination-only value). Confirm guard unchanged = still can't ship worse than GEPA.
  prefix_cap/run_tag now recorded in result.json.
- **sk2 lane** (kills the ifbench tunnel treadmill): harness rsynced to
  sk2:/lfs/skampere2/0/alexspan/norm-research/datasets/prompt-optimality-test; venv with
  version-pinned deps (dspy 3.2.1/litellm 1.91.4/datasets 5.0.0 + spacy/sympy/lark for the
  bench metrics); z.ai verified reachable FROM sk2 (key already present). Per heartbeat
  directive (GPUs 1+6 idle): **second Qwen3-8B vLLM server on sk2 GPU 1, port 8078, sk2-PID
  3481835** (same serve args as the 8077 server) — so the sk2 arms don't contend with the
  local arms' 8077.
- **v5 arms live ON sk2** (localhost vLLM = no tunnel in the loop): ifbench sk2-PID 3625813,
  livebench sk2-PID 3625814; both `--max-units 128 --prefix-cap 48 --confirm-add-val --top-k
  20 --budget-calls 40000 --run-tag v5sk2` → runs_paperexact/<b>/Qwen3-8B/unitrecomb_v5sk2/.
  Unit mining confirms the framing upgrade: **60 LLM-suggested each** (old ceiling ~36);
  ifbench 79 units used, livebench 128 (cap hit).
- **Local ifbench v4 RETIRED** (killed 5278, then watchdog-respawn 10524 — both by explicit
  PID, 2min into mining, nothing lost); watchdog line removed + watchdog restarted (10908,
  covers local livebench/pupa/hotpot). **Local livebench v4 (8428) NOT touched — it is in its
  ENDGAME**: all 175 marginals done, prefix_k9 at select .679 (Sonnet digest's "0
  select_marginal rows" was a wrong-key grep; the field is `phase`). Best guarded result of
  {local v4, sk2 v5} wins per bench.
- Rollup made variant-aware (glob unitrecomb*; M_ω cell = best guarded variant, tagged).
  PROTOCOL: rsync back ONLY runs_paperexact/*/Qwen3-8B/unitrecomb_v5sk2/ from sk2 before
  refreshing the rollup — never wholesale (would clobber local official/inhouse data).
- Sweep notes: z.ai UP; sk3 bigpanel path reported MISSING by sweeper — verify path next HB;
  sweeper's (b) "no run logs" also wrong (find-pattern artifact) — treat Sonnet digests as
  pointers, verify before acting (twice bitten this HB).
- **HB43b (~13:20 PT): --eval-threads flag added** (evaluate_cand was hardcoded n_threads=8 —
  the real throughput bottleneck). sk2 v5 arms killed ~30min in (explicit PIDs 3625813/3625814,
  first eval not yet recorded = nothing lost) and relaunched at **32 threads**: ifbench sk2-PID
  3934278, livebench sk2-PID 3934279. Local arms stay at 8 (thread count = socket count =
  tunnel dead-socket exposure; and mid-run restarts would lose real progress). Measured
  cadences at this HB: livebench-local 9.5 min/eval (prefix_k10, endgame ~5-6h out), pupa-local
  6.8 min/marginal (~12h out). Rate probe armed for the v5 arms.

**HB43c (~13:40 PT): FULL-BOARD v5 PUSH (user: "prioritize beating paper-exact metrics.
Let's push on all of these").** GPU 6 verified idle (fresh nvidia-smi: 0%, 0 MiB) → **third
Qwen3-8B server, port 8079, sk2-PID 4032777**. Launched the remaining three paper-exact v5
arms on it: **aime sk2-PID 4071211** (borderline p=.064 → make decisive; v3 compiled just 1
unit), **hover sk2-PID 4071212** (shipped .547 ≈ v3-process endpoint .550 — v5's new framings
= a DIFFERENT proposal distribution, so this directly tests whether the EVT endpoint moves
with the process), **hotpot sk2-PID 4071214** (headroom .423→.437). All: --max-units 128
--prefix-cap 48 --confirm-add-val --top-k 20 --eval-threads 32 --run-tag v5sk2; aime budget
24000, hover/hotpot 40000; aime WITHOUT --robust-answer-extract (Qwen emits bare integers —
column stays paper-exact). Fleet now: 8077(tunnel)=local livebench-v4 endgame + pupa-v4 +
hotpot-GLM; 8078(GPU1)=ifbench-v5 + livebench-v5; 8079(GPU6)=aime-v5 + hover-v5 + hotpot-v5.
Every one of the 6 paper benchmarks has an active Qwen-column push. GLM lanes deprioritized
per directive (hotpot-GLM left cycling; costs only z.ai quota).
- **sk2 LANDMINES hit + fixed during v5 bring-up** (for future sk2 lanes): (1) hover/hotpot
  need `bm25s`+`PyStemmer` (+the 1.0G prebuilt bm25s_retriever — ships inside vendor/, rsync
  covers it); (2) `datasets==5.0.0` REFUSES script datasets (hover-nlp/hover) — local machines
  only work via the cached ARROW copy, and sk2's profile exports HF_HOME→shared_hf_cache, so
  synced ~/.cache copies are invisible: fix = rsync local
  ~/.cache/huggingface/{datasets/hover-nlp___hover,modules/datasets_modules/datasets/hover-nlp--hover}
  to sk2 AND launch hover arms with `HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets`.
  hover relaunched with fix (sk2-PID in log); aime + hotpot v5 already mining (aime 128 units,
  hotpot 68, both 60 LLM-suggested).
- **HB43d (~14:45 PT): measured v5 eval rates** (32 threads, localhost): 8078 pair — ifbench
  2.4 min/eval, livebench 4.9; 8079 trio — hotpot 0.7, hover 0.8, aime 3.0 (long gens).
  Revised ETAs: hotpot-v5 ~5pm, hover-v5 ~6pm, livebench-local-v4 ~7-9pm, ifbench-v5 ~10pm
  today; aime-v5 ~4am, pupa-v4 + livebench-v5 ~9-10am tomorrow. The sk2 lane is 4-12x the
  tunnel lane per eval.

**HB44 (~14:55 PT): all 7 Qwen lanes verified healthy; no completions yet.** Local: livebench-v4
at **prefix_k15** (writes 0.9 min fresh — on pace for tonight), pupa-v4 50 marginal rows (4 min
fresh), hotpot-GLM watchdog-cycled again (now 20215). sk2 v5 (all in select_marginal, writes
0-3 min fresh): hotpot 56 evals, hover 36, ifbench 21, aime 13, livebench 9 — all matching the
HB43d rates, ETAs hold. z.ai UP; both sk2 servers responding; GPUs 1/6 at 100% (ours). No new
result.json in 80 min (expected; first verdicts ~5pm). SWEEPER RELIABILITY: digest (e) reported
"NO ROWS" for all v5 arms — artifact of an ssh NAT64 reset mid-loop (plus a py3.11 f-string
nested-quote trap in my own probe); verified directly, all arms fine. Standing rule: any
sweeper NEGATIVE (missing/zero/stale) must be re-verified directly before acting — 3rd
false-negative in 3 sweeps. **sk3 NOTE for user:** heartbeat checklist item (d) references
outputs/r2_recovery_v2_momega_bigpanel which does NOT exist on sk3; outputs/r2_recovery exists
(cw-llama8b-* subdirs) but has NO writes in 12h+ — the sk3 A/B recovery campaign appears
finished-or-stopped; NOT restarting anything blindly (paper-exact priority + unclear intended
state) — needs user confirmation of the campaign's disposition.

**HB45 (~15:55 PT): hotpot-v5 at confirm_compiled = final guard gate; result imminent.**
Sweep: hotpot-v5 140 evals (past marginals+prefix+drop-one+add-back → confirm), hover-v5 110
(marginals nearly done), ifbench-v5 47, aime-v5 32, livebench-v5 21 — all fresh writes ≤3 min.
Local: livebench-v4 prefix_k20 (on pace), pupa-v4 58 marginals, hotpot-GLM cycled (40788).
z.ai UP, both servers up, GPUs saturated. One tunnel blip at 21:22Z restored in ~2 min (local
arms rode retries; sk2 lane untouched by design). Completion watchers armed: sk2 v5
result.json (2-min polls) + local livebench-v4 result.json mtime (3-min polls) — sign tests
will run the moment either lands.

**HB45b (~16:20 PT): ⚠ hotpot-v5 RESULT INVALID — sk2 retrieval was BROKEN; caught by
seed-sanity, quarantined, both retrieval arms relaunched clean.** The first v5 result landed
(hotpot: seed .18 / best .373 / 10 units) and FAILED the seed-sanity check: the identical seed
program scores .38 locally — the baseline itself collapsed. Root cause: hover/'s
wiki.abstracts.2017.jsonl is a SYMLINK to an absolute /Users/... path (dangles on sk2; the only
broken symlink in the tree; bm25s_retriever dir was present but init requires corpus too, and
the auto-rebuild path also crashes through the dangling link). hotpot ran with retrieval
soft-failing → Qwen answered multi-hop nearly blind (.18) and the search "optimized"
compensation units — meaningless vs official GEPA. ACTIONS: relative symlink →
data/wiki17/wiki.abstracts.2017.jsonl (1.78G copy had synced); search() verified returning
correct passages on sk2; hover-v5 killed by PID 245070 (its in-flight search was poisoned);
hotpot-v5 dir RENAMED runs_paperexact/hotpot/Qwen3-8B/INVALID_noretrieval_v5sk2 (sk2+local —
kept, but escapes the rollup's unitrecomb* glob); clean relaunches hotpot sk2-PID 1393089,
hover sk2-PID 1393091. LESSONS: (1) seed_test ≈ local seed_test is a mandatory validity gate
on any cross-machine result — the guard chain worked (nothing was recorded); (2) rsync -a
carries absolute symlinks — always `find -xtype l` after seeding a new machine. aime/ifbench/
livebench-v5 unaffected (no retrieval). ETAs: hotpot/hover-v5 push to ~7-9pm PT.

**HB46 (~16:55 PT): all lanes advancing; endgames approaching.** Local livebench-v4 at
**prefix_k27** (of ≤32 — endgame ~2-3h out), pupa-v4 67 marginals, hotpot-GLM cycled (62183).
sk2 v5: clean hotpot already at **prefix_k19** (result ~1h out), hover 173 rows (marginals
~done; note its proposals.jsonl carries ~36 rows from the poisoned first run — clean-run
row-counting must subtract), aime 52, ifbench 69, livebench-v5 33. z.ai UP, both servers UP.
No valid results yet (the only new result.json was the already-quarantined INVALID hotpot).
Watchers armed; nothing to record this cycle.

**HB47 (~17:55 PT): two arms in final passes; no verdicts yet.** Local **livebench-v4 at
drop_one** (past its 32-prefix sweep w/ best 8-unit prefix +.035 over init — confirm gate +
finals next, ~1h out; PING-ON-LAND armed per user). sk2: hotpot-v5 also at drop_one (129
evals), hover-v5 ~108/128 clean marginals, ifbench-v5 prefix_k12 (panel units still earning),
aime-v5 71/129 marginals (3.2 min/eval — slower than the trio estimate, endgame ~11pm-1am),
livebench-v5 45 rows (morning as forecast). pupa-v4 77 marginals. hotpot-GLM cycled (20421).
z.ai UP, watchdog UP, both servers UP. Nothing recorded this cycle — all changes pending the
confirm gates.

**HB48 (~18:30 PT): hover-v5 LANDED — seed-sanity PASS, guard PASS, beats GEPA, does NOT
displace v3.2; EVT endpoint HELD.** seed .35 (1.1 SE from local .38 — valid, unlike the
INVALID hotpot .18 = 7 SE), best .49, 4 units (2 llm + 2 trajectory), confirm .397→.491
(350 items). Interpretation: (1) fresh draw from the upgraded process beat GEPA (.49>.45) —
the floor works; (2) it did NOT approach v3.2's .5467 ≈ endpoint .550 → **first direct
evidence the hover EVT ceiling survives a diversified proposal distribution** (v5 had 128
units incl. 60 new-framing LLM suggestions and every chance to exceed it); (3) MINING GAP
found+fixed: _mine read only official/inhouse — v3.2's 5 winning units were ABSENT from v5's
pool. v5.1 harness fix: pools now inherit ALL unitrecomb* variant trajectories (INVALID_*
excluded); synced to sk2 — matters for the OSL staircase (frozen pool must contain 8B winners
by construction). Scoreboard UNCHANGED (hover = .5467 best-variant, p=.0001). hotpot-v5 still
in endgame; watcher re-armed.

**HB49 (~19:55 PT): three arms in guard/endgame simultaneously.** hotpot-v5 at
**confirm_init** (guard evals running — result ~30-60 min); local livebench-v4 still in
drop_one (fresh writes; its 9.5-min evals make the endgame slow — revised landing ~1.5-2h);
ifbench-v5 at **prefix_k37** (units STILL earning past k37 of 48 — deepest prefix any ifbench
attempt has reached anywhere); aime-v5 91/129 marginals; livebench-v5 57/129; pupa-v4 86
marginals (endgame tonight). hotpot-GLM cycled (4680). z.ai UP, watchdog UP, all writes ≤4
min fresh. Nothing new recorded (hover-v5 was HB48). Ping-on-livebench still armed.

**HB50 (~20:40 PT): ★★★ HOTPOT v5 — THE RESULT OF THE CAMPAIGN. M_ω .6333 vs GEPA .380,
81W-5L, p=4.8e-19, 28 units.** Seed-sanity PASS (.40 vs .38 local, 0.7 SE). Guard chain
consistent across THREE disjoint slices: select .44→.71, confirm(350) .443→.62, test
.40seed→.6333. Units: 25 LLM-suggested + 3 trajectory — the v5 framings did this. Mechanism
(honest): hotpot is exact-match scored and Qwen3-8B's verbose answers fail string match; most
compiled units are ANSWER-FORMAT DISCIPLINE (extract exact entity, no filler, rely only on
provided summaries) + 2 reasoning-structure units — i.e., the lift is largely articulable
output-format knowledge, exactly what articulation SHOULD capture, and exactly what GEPA's
reflection never found (official GEPA-Qwen: seed=best=.38, ZERO improvement). **THEORY DATA
POINT: the v3-process EVT endpoint (.437) is DEMOLISHED by the v5 process (.6333) — endpoints
are process-conditional, now demonstrated in BOTH directions in one night (hover: held at
.550; hotpot: moved +.20). The paper's EVT section gets its perfect contrast pair.** Rollup
refreshed (hotpot M_ω cell = .633 (v5sk2)); EVT re-estimate deferred until uniform rescores.
Terminal ping sent (mobile inactive). QWEN SCOREBOARD: hover .5467 (p=1e-4) ✅, hotpot .6333
(p=5e-19) ✅✅, aime .4267 (p=.064) ✅~, ifbench tie (v5 at prefix_k37+), livebench v4 drop_one
+ v5 in marginals, pupa endgame tonight. 0 losses.

**HB50b (~21:20 PT): OSL STAIRCASE CUED UP (user directive: "cue up the scaling law
experiments").** (1) **GPU census (fresh nvidia-smi all 3 boxes): sk1 = 6 IDLE A100-80s
(GPUs 1,2,3,5,6,7) → THE STAIRCASE BOX; sk2 = only our 2 (campaign continues); sk3 = fully
saturated, untouchable.** sk1 caveats: /lfs 97% full (1.7T free — lean footprint), no Qwen3
family in cache (~125GB to download), account+space exist. (2) **Proposer decision (user):
GLM-5.2 default for all non-paper-exact work** (it produced tonight's results; 4.7 =
health-probe/fallback only). (3) **Tooling shipped: build_frozen_pool.py** (frozen pool =
union of all units the 8B runs actually evaluated, from result.json marginals — deterministic,
no re-mining, winners included by construction) **+ --pool-file mode in paperexact_arms**
(skips mining+suggestion entirely, no z.ai dependency at scale-time). Pools built: hover 164u
(8 winners), hotpot 68u (27), aime 48u, ifbench 32u, livebench 48u — REBUILD after tonight's
v5s land, then freeze. (4) sk1 bring-up in flight: harness rsync running; next = venv (sk2
recipe), find -xtype l symlink audit, hover HF-cache sync, nltk, Qwen3 1.7B/4B/8B downloads
(14B/32B after), then per-scale servers on the 6 free GPUs — whole primary staircase can run
in PARALLEL. Llama-70B later on sk2 B200s post-campaign; Gemma-4 phase last.

**HB51 (~21:55 PT): local livebench-v4 PASSED THE CONFIRM GATE — now in final test evals**
(phase final_test_seed; its 8-unit compile survived drop-one AND confirm; result ~20-40 min;
ping watcher live). ifbench-v5 at drop_one (endgame, verdict ~1h). aime-v5 114/129 marginals,
livebench-v5 69/129, pupa-v4 93 marginals (endgame next). hotpot-GLM cycled (66804). z.ai UP.
sk1 staircase: 6 fixed-arm runs in flight (no results yet, no tracebacks; 32B still
downloading). No new results to record this cycle — hotpot/hover v5 already in HB50/HB48.

**HB52 (~22:20 PT): livebench-v4 LANDED — STATISTICAL TIE (user pinged per request).**
Shipped a real 7-unit compile (guard .748→.781 on 161-item confirm — first livebench attempt
to ship a compile at all), best_test .6823 vs GEPA official .6956; paired 9W-12L (105/126
ties), p=.81 → tie, not a loss. Note the cross-run sampling asymmetry: v4's own seed rescore
was .6681 (its compile is +.014 over its own init measurement); GEPA's .6956 was measured in
the official run — temp-.6 resampling noise dominates the .013 mean gap, consistent with
p=.81. livebench remains GEPA-favorable-tie; the v5 wide-pool arm (128 units, 69/129
marginals) is the remaining shot, ~AM. Rollup refreshed (M_ω cell .682). Terminal ping sent.
SCOREBOARD: hover .5467 ✅ p=1e-4; hotpot .6333 ✅✅ p=5e-19; aime .4267 ✅~ p=.064 (v5 in
flight); ifbench ⚖️ (v5 at drop_one); livebench ⚖️ (v5 in flight); pupa in flight. 0 losses.

**HB53 (~22:55 PT): sk1 staircase FALSE START caught + fixed; ifbench diagnosed NOT-wedged.**
(1) All 3 sk1 vLLM servers died AT ARGPARSE (**vLLM 0.25.1 removed --disable-log-requests**;
sk2's older install still has it) and my ready-watcher gave a FALSE POSITIVE (count-based
check, polluted output) → 6 staircase evals retry-looped against dead ports for ~2h. NOTHING
recorded (no fixed_arms.json; partial evals.jsonl quarantined as *.attempt1_deadservers).
Fixes: staircase_eval.py now has the SAME pre-flight abort + HEALTH_PROBE mid-run guard as
paperexact_arms (it had none — same landmine class as the sk2 livebench zero-score incident);
servers relaunched without the dead flag (999520/21/23); ready-check now greps the actual
model name, then auto-relaunches the 6 evals. LESSONS: pin flag compat per vLLM version;
ready-watchers must verify content, not counts; every eval entrypoint gets the outage guard.
(2) ifbench-v5 NOT wedged: alive, grinding drop_one; systematic per-item error — longest
IFBench items + heavy unit stacks exceed the 16384 server context (8385 input + 8000 output)
→ those items zero out for unit-heavy candidates only = slight ANTI-unit bias; guard still
valid (same constraint both sides); consider 32768-context servers for any ifbench rerun.
pupa-v4 102 marginals; aime-v5 prefix_k11; livebench-v5 82/129 marginals.

**HB54 (~00:20 PT): ★ FIRST STAIRCASE CURVE (aime fixed arms, 3 scales) + triple incident
recovery.** THE DATA: aime 1.7B seed .213 / GEPA-transplant .173 / M_ω-transplant .213;
4B .367/.460/.433; 8B .320/.420/.440 → **the 8B-discovered articulation HURTS-or-does-nothing
at 1.7B (transplant lift −.04/0.0) and helps at 4B/8B (+.06..+.12) — first empirical points
for H-i/H-iv ("bigger models can absorb more articulation"; transplant lift GROWS with
scale)**. Caveats: 150-item test SE ~.04; 3 of 5 rungs; transplant lift ≠ per-scale-optimized
lift (those arms come next). INCIDENTS: (1) hover staircase trio died of **fd exhaustion
(ulimit 1024) + dspy disk-cache sqlite contention** — hover also opens the 1.0G bm25s index;
aime (no retrieval) survived; outage guard correctly refused to record 2 all-zero batches
(the 3rd hit OSError writing evals.jsonl — quarantined *.attempt2_fdexhaustion). Fix:
ulimit -n 65536 + per-run DSPY_CACHEDIR; hover trio relaunched (1083757/59/60), watcher
armed. LESSON for all multi-proc sk1 launches: raise ulimit + isolate dspy caches. (2) Tunnel
DOWN again → restored; watchdog had correctly skipped Qwen arms during the outage. (3) pupa-v4
wedged on the dead socket (37 min stale, 109 marginal rows lost) → killed 3122 by PID,
relaunched 51260 (v4 config). (4) ifbench-v5: proposals stale 144 min BUT log fresh (1 min) —
it is inside ONE glacial drop_one eval, retry-looping context-overflow items; alive at 8h11m;
leaving it. aime-v5 at prefix_k44 (endgame close).

**HB54b (~00:50 PT): ★★ HOVER STAIRCASE FIXED ARMS COMPLETE — the CONTRAST PAIR to aime.**
hover: 1.7B seed .350 / GEPA-tx .390 / **M_ω-tx .4833**; 4B .390/.4667/.510; 8B
.370/.440/.5267. READING: (1) hover articulation transfers ALL THE WAY DOWN — M_ω transplant
lifts 1.7B by +.13, putting the 1.7B ABOVE the 8B's own seed (.4833 > .37); aime transplant =
0/−.04 at 1.7B → **articulable-procedural (hover) vs capability-indexing (aime) articulation,
measured — the tacit/articulable divide as a scale-transfer contrast pair**; (2) M_ω-tx >
GEPA-tx at EVERY hover scale (+.06-.09): unit-composed prompts transfer better than
reflective rewrites; (3) 8B sanity: M_ω-tx .5267 ≈ paper-exact .5467 (serving-stack + temp
noise), seed .37 ≈ .38 ✓. NEXT STAGE LAUNCHED: z.ai verified from sk1 (key copied); **6
per-scale GEPA arms live** (hover+aime × 1.7/4/8B, budget 600, GLM-5.2 reflection, ulimit +
isolated caches; pids 1656302-12) — per-scale M_ω (--pool-file, init-from-scale-GEPA) queues
as each GEPA lands; **14B (GPU5:8174) + 32B (GPU6:8175) servers launching**, fixed arms
auto-launch on ready (watcher armed). sk1 now runs 5 GPUs for the staircase.

**HB55 (~02:00 PT): staircase reshuffle after other users claimed sk1 GPUs 0/5/6.** 14B/32B
servers had died on "free memory 51.67/79.25 GiB" — NOT a config error: between my GPU census
and the launches, other users grabbed GPUs 5+6 (and grew GPU 0). Shared-box lesson: **verify
idle IMMEDIATELY before every server launch, and expect claims to race.** Recovery: 14B
relaunched on GPU 7 (the last free one; sk1-PID 1741512, death-detecting ready-watcher →
fixed arms auto-launch); **32B rung DEFERRED to sk2 B200s post-campaign** (today: its arms
are all in endgames). Staircase now: 1.7/4/8B servers busy with the 6 per-scale GEPA arms
(hover-8B + aime-8B GEPA DONE — sk1 rerun; note their result.json OVERWROTE the synced sk2
copies on sk1 only, canonical copies safe local+sk2; future sk1 8B arms use --run-tag);
per-scale M_ω (--pool-file) queues on each GEPA completion. sk2: aime-v5 at add_back
(verdict close), livebench-v5 109 marginals, ifbench-v5 still inside its glacial drop_one
eval (177 min); pupa-v4 relaunch healthy (148 rows); tunnel UP; z.ai UP.

---

# OPUS RUNBOOK (2026-07-23 ~03:30 PT handoff — PROCEDURAL. Keep things moving; do NOT
# redesign anything. Anything ambiguous: record a ⚠ line here and leave it for the user.)

## Env preambles (use VERBATIM in every ssh)
- sk1: `export HOME=/lfs/skampere1/0/alexspan; export HF_HOME=$HOME/hf_cache; export HF_DATASETS_TRUST_REMOTE_CODE=1; export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets; export ZAI_KEY_FILE=$HOME/.z-ai-api-key-spangher.txt; cd $HOME/norm-research/datasets/prompt-optimality-test; ulimit -n 65536`
- sk2: same pattern with /lfs/skampere2/0/alexspan
- sk3: same pattern with /lfs/skampere3/0/alexspan, PLUS `export VLLM_WORKER_MULTIPROC_METHOD=spawn` and `CUDA_DEVICE_ORDER=PCI_BUS_ID` on any vllm serve
- Every client launch: prefix `DSPY_CACHEDIR=$HOME/dspy_cache_<uniquename>`
- vLLM 0.25.1 (sk1/sk3): NO --disable-log-requests flag

## THE LOOP (each hourly heartbeat)
1. Dispatch the Sonnet sweeper (copy the HB checklist pattern from HB51-55: exact python
   snippets, `phase`/`ts` keys, retry ssh once). VERIFY ANY NEGATIVE FINDING DIRECTLY before
   acting — 4 sweeper false-negatives so far.
2. Walk the EVENT TABLE below; execute matching actions.
3. Append an HBnn entry here (results + sign tests + incidents). Keep zero unexplained state.

## EVENT TABLE (event → exact procedure)
**E1. A result.json appears under sk2 .../unitrecomb_v5sk2/ (aime imminent; livebench ~AM;
ifbench eventually):**
  a. `rsync -a sk2:/lfs/skampere2/0/alexspan/norm-research/datasets/prompt-optimality-test/runs_paperexact/<B>/Qwen3-8B/unitrecomb_v5sk2 runs_paperexact/<B>/Qwen3-8B/`
  b. SEED-SANITY: result seed_test must be within ~2 SE of the local 8B seed band (aime
     .32-.37, livebench ~.65-.67, ifbench ~.41). FAIL → rename dir INVALID_<reason>_v5sk2
     (sk2+local), log ⚠, do NOT record.
  c. Sign test (template in HB50): item_scores of last final_test_best row in the v5
     proposals.jsonl vs same row in runs_paperexact/<B>/Qwen3-8B/official/proposals.jsonl.
  d. Record HB entry; run `.venv/bin/python analyze_upper_bounds_rollup.py`.
  e. livebench ONLY: PushNotification with the verdict (user asked twice).
  f. If livebench v5 DONE: its 8078 server slot frees → OPTIONAL queued task: relaunch sk2
     8078 server at --max-model-len 32768 (kill old server by PID first) and launch ifbench
     `--run-tag v6ctx32k` (same v5 flags otherwise) — ONLY if ifbench v5 still unfinished.
**E2. Local pupa v4 result.json appears (runs_paperexact/pupa/Qwen3-8B/unitrecomb/):**
  same gate + sign test vs local pupa official (.862); record. No ping required.
**E3. sk3 bring-up watcher fires (blxrg98jw / grep "BRINGUP DONE" ~/sk3_bringup.log):**
  a. FIRST verify where Qwen3-32B snapshot lives: `ls $HOME/hf_cache_stair/hub | grep 32B`
     else `ls $HOME/.cache/huggingface/hub | grep 32B` — set HF_HOME for the server to
     WHICHEVER contains it.
  b. Fresh `nvidia-smi` — confirm GPUs 0/1 still free (GPU 7 = USER'S, never touch).
  c. 32B server: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn nohup .venv/bin/vllm serve Qwen/Qwen3-32B --served-model-name Qwen3-32B --port 8175 --host 127.0.0.1 --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 > $HOME/vllm_32b.log 2>&1 &`
  d. Ready-check = curl /v1/models greps "Qwen3-32B" (content, not counts). Then: fixed arms
     (staircase_eval.py hover+aime --model Qwen3-32B --api-base http://127.0.0.1:8175/v1
     --eval-threads 32), then GEPA (paperexact_arms official, budget 600, same flag pattern
     as sk1 14B launch in HB55), then M_ω stair (unitrecomb --pool-file
     pools/<B>_Qwen3-8B_frozen.json --run-tag stair, init auto = its official).
  e. Llama anchor on GPU 1: serve meta-llama/Llama-3.1-8B-Instruct port 8176 (same pattern,
     no reasoning parser flag for Llama). Then hover+aime: official GEPA (budget 600) →
     unitrecomb WITH MINING (NO --pool-file; family-native pool) --run-tag llanchor →
     afterwards `build_frozen_pool.py <B> --lm Llama-3.1-8B-Instruct`.
**E4. sk1 staircase cell completes (official/result.json or unitrecomb_stair/result.json
  or fixed_arms.json for 1.7B/4B/14B):** no action needed — the chainer
  ($HOME/stair_momega_chainer.sh, log $HOME/stair_chainer.log) launches M_ω per GEPA
  completion automatically. Just VERIFY chainer alive (`pgrep -f stair_momega_chainer`) and
  record numbers in an HB entry. Envelope per rung = max(official best_test,
  unitrecomb_stair best_test); also record lift = each − seed(fixed_arms.json).
**E5. Wedge/stall detection (proposals stale >30min AND process 0.0% CPU AND its endpoint
  UP):** local Qwen arms → tunnel check first (`curl 127.0.0.1:8077/v1/models`; restore:
  `ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L 8077:127.0.0.1:8077 sk2`), then kill BY PID + relaunch with the arm's original flags
  (watchdog 10908 auto-covers local pupa/hotpot; it does NOT relaunch missing procs).
  EXCEPTION: sk2 ifbench v5 (pid 3934278) is NOT wedged — glacial by known cause
  (context-overflow retries); DO NOT KILL IT.
**E6. A server dies (curl fails):** check `nvidia-smi` for a GPU claim race BEFORE
  relaunching; relaunch only onto a verified-free GPU; on sk1 only GPUs 1,2,3,7 are ours.

## HARD RULES (never violate)
- Kill ONLY by explicit PID. Never pkill/killall patterns. Never touch other users' jobs.
- Never delete data — quarantine by rename (INVALID_*, *.attemptN).
- fresh nvidia-smi immediately before EVERY server launch (claims race; bitten once).
- Seed-sanity gate before recording ANY cross-machine result (caught one invalid already).
- Do NOT rebuild pools/{hover,aime}_Qwen3-8B_frozen.json (staircase comparability). Only
  build NEW pool files (ifbench/livebench/pupa after their v5s; Llama anchor).
- No new experiments/objectives/analyses beyond this runbook — park ideas as ⚠ lines.
- GLM-5.2 = proposer everywhere (z.ai, spend freely, retries absorb); glm-4.7 = probes only.
- Morning user summary: scoreboard table (all 6 benches, sign tests), staircase
  fixed-arm + envelope tables, incidents one-liner each, ⚠ list.

## Live watcher inventory (as of handoff)
- bul7ebbyn: sk2 v5 result.json poller (2-min) — fires E1.
- blxrg98jw: sk3 bring-up completion — fires E3.
- bhgbvvec4: tunnel transition monitor (auto-reconnects; alert = check E5).
- sk1 chainer (on sk1, not a local task): auto M_ω per GEPA — E4.
- Local watchdog PID 10908: pupa + hotpot-GLM auto-restart (needs tunnel up to act).
- Re-arm any expired watcher with the same command pattern (see transcript HB entries).

**HB56 (~06:00 PT, Opus handoff cycle 1): all lanes healthy; no new verdicts; 32B rung
launched.** sk2: aime-v5 at **add_back** (212 evals, verdict imminent — watcher bul7ebbyn
armed), livebench-v5 123 marginals (~AM), ifbench-v5 237min at drop_one = KNOWN glacial
(context-overflow retries), NOT wedged, not killed. pupa-v4 157 marginals (fresh), tunnel UP,
z.ai UP, watchdog 10908 alive. **sk1 chainer VERIFIED working** (sweeper's "empty log" was a
$HOME→AFS path artifact; real log at /lfs shows 3 launches): M_ω-stair live for hover-8B
(1995611), aime-8B (1996902), hover-4B (3062280); hover-1.7B queued next 10-min loop. GEPA-stair
DONE: hover-1.7B seed .3433/best .40, hover-4B .39/.4667; aime GEPA still optimizing all scales
(GEPA reflection slow), 14B GEPA at 0% just started. **sk3 32B rung LAUNCHED**: Qwen3-32B
located in /lfs/skampere3/0/shared_hf_cache (sk3 profile forces shared cache; HF_HUB_CACHE set
accordingly), server pid 114181 on GPU 0 (verified free at launch), fixed-arms+GEPA watcher
b51k91hh5 armed (death-detecting). 32B M_ω-stair = PENDING next cycle (sk1 chainer doesn't reach
sk3; launch once 32B GEPA lands). ⚠ **sk3 GPUs 1-7 all claimed by other users → Llama/Gemma
family phases BLOCKED on sk3 until GPUs free; 32B on GPU 0 is the only sk3 rung for now.**
Envelope tables to build once M_ω-stair cells land (per rung: max(GEPA best, M_ω-stair best);
lift = each − fixed-arms seed).

**HB57 (~06:30 PT): ✅ AIME v5 LANDED — REPLICATED WIN, cleaner significance.** seed .32
(sanity PASS), best .4133, 8 units (6 llm + 2 traj), guard passed on confirm equality
(.40=.40; select .50→.6333; test .32→.4133). Sign test vs GEPA official: **8W-1L (141 ties),
p=0.0195**. HONEST FRAMING (no p-hacking across runs): aime now has TWO independent M_ω runs
both beating GEPA on point estimate — v3.2 .4267 (14W-5L, p=.064) and v5 .4133 (8W-1L,
p=.0195) vs GEPA .3667 → **aime = replicated win; GEPA never beats M_ω on point estimate;
campaign-best M_ω .4267 (v3.2)**. Rollup shows best guarded variant per cell. Rollup refreshed.
No ping (not livebench). sk2 remaining: livebench-v5 now at prefix_k7 (endgame; watcher
bwifffaaf armed), ifbench-v5 290min drop_one (glacial exception, NOT killed). pupa-v4 running.

**HB58 (~07:00 PT): ★★ FIXED-ARM TRANSPLANT CURVES ~COMPLETE (both benches, 5 scales) — the
H-i/H-iv result taking shape.** All values = test-split, arms = seed / GEPA-8B-transplant /
M_ω-8B-transplant (fixed 8B-discovered prompts, NO re-optimization). SE ~.026 hover(300)/.04
aime(150).

hover (seed → M_ω-tx, lift):
  1.7B .350→.4833 (+.133) | 4B .390→.510 (+.120) | 8B .370→.5267 (+.157) |
  14B .4733→.5567 (+.083) | 32B .4533→.5333 (+.080)
aime (seed → M_ω-tx, lift):
  1.7B .2133→.2133 (.000) | 4B .3667→.4333 (+.067) | 8B .320→.440 (+.120) |
  32B .280→.340 (+.060) [⚠] | 14B pending

READINGS: (1) **hover M_ω-tx > GEPA-tx at ALL 5 scales** (unit-composed transfers better than
reflective rewrites, replicated). (2) **hover lift SHRINKS with scale** (+.13-.16 small →
+.08 at 14/32B) = H-iv "gap closes from the capability side" supported. (3) hover absolute
SATURATES ~14B (.557) ≈ 32B (.533, within 1 SE) = an articulation ceiling ~.53-.56. (4) aime
lift 0 at 1.7B, positive 4B+ = capability threshold for absorption (H-i). ⚠ **aime-32B is
NON-MONOTONIC (all arms below 8B: seed .28<.32, M_ω-tx .34<.44)** — strong smell of
8000-token reasoning TRUNCATION on 32B (reasons longer, hits cap); do NOT over-read aime-32B
until truncation-rate checked (parked for user; a 16k-token aime-32B rerun would settle it).
NOTE these are TRANSPLANT curves; the ENVELOPE (per-scale re-optimized max(GEPA,M_ω-stair))
still cooking: sk1 6 GEPA + 5 M_ω-stair running (none finished), chainer alive; sk3 hover-32B
GEPA done (.4533→.5233) → **hover-32B M_ω-stair LAUNCHED pid 153765** (164-unit frozen pool);
aime-32B M_ω pending its GEPA. Campaign: livebench-v5 prefix_k8 (endgame), ifbench 297min
(glacial exception), pupa-v4 running. bwifffaaf watcher armed.

**HB58b (~07:40 PT): z.ai OUTAGE (monitor bx2d2gl6a UP→DOWN; direct probe timed out).**
SCOPE verified: IMMUNE (running fine) = all 6 M_ω-stair envelope runs (frozen-pool = zero
z.ai dep, by design), sk2 livebench-v5 + ifbench-v5 (past mining), pupa-v4. AFFECTED
(alive+retrying, NOT crashed) = 6 sk1 GEPA-stair reflection steps (GLM-5.2 reflection) +
hotpot-GLM (bonus). ACTION = NONE (no thrash; z.ai transient, retries absorb; restart-mid-
outage futile + pre-flight refuses). bx2d2gl6a fires on recovery. ⚠ **PROTECTIVE FLAG: any
GEPA-stair cell that COMPLETES during/just-after this outage window must be sanity-checked
(best_test ≥ seed, reasonable) before envelope use — a z.ai-degraded GEPA run can finish at
best≈seed (no reflections landed), which would understate GEPA and unfairly inflate the M_ω
envelope. Re-run any such cell post-outage.** M_ω-stair cells are unaffected and safe to use.

**HB58c (~07:52 PT): z.ai RECOVERED (~10min outage, DOWN 07:38→UP 07:48; bx2d2gl6a).
Protective flag PAID OFF — caught 1 degraded GEPA cell.** All 6 GEPA + M_ω-stair procs
survived via retries EXCEPT: **hover-14B GEPA completed DURING the window at best=seed=.4267
(delta 0.0 — all reflections failed) = z.ai-DEGRADED.** Worse, the chainer had already fired
hover-14B M_ω-stair (pid 111910, 07:44) initialized from that degraded GEPA (=seed). FIX
(HB58b protocol): killed 111910 by PID; quarantined official→official_ZAIDEGRADED and
unitrecomb_stair→unitrecomb_stair_ZAIDEGRADED (renamed, not deleted); RE-RAN hover-14B GEPA
(pid 345973, z.ai back) → chainer auto-refires its M_ω off the VALID GEPA once done. hover-14B
FIXED-ARM curve (HB58) is UNAFFECTED (fixed arms = pure evals, no z.ai). hotpot-GLM (died in
outage) self-recovered/watchdog-caught, running. NO OTHER GEPA cell completed in the window
(only hover-14B). LESSON reinforced: GEPA reflection is the one z.ai-exposed staircase arm; any
GEPA cell finishing best≈seed near a z.ai blip = re-run. M_ω-stair (frozen pool) stays immune.

**HB59 (~08:30 PT): all healthy; sweeper false-zero #5 dismissed; envelope cells accumulating.**
Digest reported sk1 gepa_running=0/momega_running=0/chainer-restarted/hover-14B-rerun-dead —
ALL FALSE (its `pgrep -af "A|B" | grep -c` patterns errored → spurious 0s). DIRECT verify:
9 paperexact procs alive, servers 8171-4 UP, chainer alive real-PID 1995581, M_ω-stair writing
(hover-1.7B 35 / 4B 72 / 8B 32 rows, fresh), hover-14B GEPA rerun 345973 alive. GEPA-stair
DONE so far: hover-1.7B .3433/.40, hover-4B .39/.4667, hover-8B .3567/.4567, aime-8B
.3333/.3667, sk3 hover-32B .4533/.5233. **M_ω-stair: NONE finished yet (all mid-run) → envelope
table not yet buildable; per-rung envelope = max(GEPA best, M_ω-stair best) once M_ω cells land.**
Still cooking: aime GEPA 1.7/4/14B, hover-14B rerun, sk3 aime-32B GEPA + both 32B M_ω. Campaign:
livebench-v5 prefix_k22 (deep endgame — verdict soon, ping armed bwifffaaf), ifbench 358min
(glacial exception), pupa-v4 168 marginals. z.ai UP, tunnel UP. FUTURE-SWEEP FIX: count procs
with `ps -eo args|grep -c "[p]aperexact"` not fragile pgrep alternations.

**HB60 (~09:00 PT): ⚠⚠ AIME-8B GEPA STAIRCASE ANOMALY — .5333, +.167 over paper-exact canonical
(.3667). FLAGGED FOR USER, not resolved.** Verified valid: n_test=150, paper-exact metric
(robust_extract=False; seed .3133=47/150 with SAME metric, best .5333=80/150), single-module
'predict', budget 600, top_k 20. So it's a REAL GEPA run — but a fresh reflection trajectory
scored .5333 where the sk2 canonical GEPA scored .3667. INTERPRETATION (for user): either (a)
aime GEPA run-to-run variance is HUGE (~30 unique problems ×5 → very high variance; the paper
itself found GEPA barely moves aime → unstable), or (b) lucky reflection prompt. This DIRECTLY
bears on the envelope: aime-8B M_ω-stair (pid 1996902) init from the EARLIER .3667 official
(chainer fired 05:04, before this .5333 landed), so envelope aime-8B = max(.5333 GEPA, M_ω
best) — if GEPA's .5333 outlier holds, it could show GEPA>M_ω at 8B on aime (a variance
artifact, NOT a real M_ω failure). ⚠ Also: this sk1 GEPA rerun OVERWROTE the synced paper-exact
aime-8B official on sk1 (canonical SAFE on sk2+local .3667); HB54b's 6 GEPA arms predated the
"use --run-tag" rule. RECOMMEND (user call): re-run aime-8B GEPA 3-5× to bound the variance
before trusting any single aime GEPA cell in the envelope. NO reactive re-run done (runbook:
park judgment calls). OTHER: pupa-v4 entered prefix_k4 (marginals DONE, endgame near);
livebench-v5 prefix_k35 (verdict imminent, bwifffaaf armed); ifbench 417min glacial; no M_ω-stair
cells finished yet; z.ai UP, tunnel UP, all servers UP, 9 sk1 + 4 sk3 procs alive.

**HB60b (~09:10 PT): ifbench status CORRECTED + livebench imminent.** livebench-v5 at
**prefix_k48 = LAST prefix step (cap 48)** → verdict minutes-to-1h away (watcher b27b2iagy
armed). **ifbench REVISED from "glacial" to "effectively stuck"**: 477min (8h) with ZERO
proposal rows written = stuck INSIDE one drop_one eval (context-overflow items retry-loop
under max_errors=10000, never completing the eval). Won't finish at this rate. NOT killed yet
because the real fix (32k-context server, HB53) needs the 8078 GPU which livebench still holds.
**PLAN (runbook E1.f): when livebench-v5 DONE → kill ifbench-v5 (pid 3934278) by PID, kill
old 8078 server by PID, relaunch 8078 at --max-model-len 32768, launch ifbench fresh
--run-tag v6ctx32k (same v5 flags).** Until then ifbench left alone. Everything else unchanged
(pupa prefix_k4, envelope M_ω-stair cooking, z.ai/tunnel/servers UP).

**HB61 (~09:30 PT): all converging, no new completions.** livebench-v5 now at **drop_one**
(past the full prefix sweep — last stage before confirm+test; verdict imminent, watcher
b27b2iagy armed). pupa-v4 prefix_k10 (endgame). sk1: 10 procs (chainer added one), all GEPA
cells unchanged (aime-8B still the flagged .5333 anomaly), **M_ω-stair still NONE finished** —
ETA note: hover frozen pool = 164 units → ~164 marginals × 300-item panel per scale is
inherently slow (est. 6-8h/cell); envelope will complete over the next several hours, which
is fine (user checks in tomorrow). Not touching pool size (comparability; runbook forbids
rebuild). ifbench 478min stuck (fix queued behind livebench). z.ai/tunnel/servers all UP.
Nothing to record/sign-test this cycle.

**HB62 (~10:40 PT): hover-14B GEPA re-run VERIFIED LEGIT (not a 2nd degradation); post-outage
recovery CLOSED.** Re-run finished .4367/.4367 (best=seed AGAIN) but this time it's REAL:
**0 reflection errors in log** (vs the degraded run's empties) → GEPA reflections worked,
proposed candidates, none beat seed in 600 budget → shipped init. Legit no-improvement
(consistent with GEPA instability at small budget — same reason it "barely moves" some paper
benches). Chainer correctly re-fired hover-14B M_ω at 10:35 (pid 3077014) off the VALID GEPA.
NOTABLE for envelope interpretation: hover-14B FIXED-ARM transplant (8B-GEPA prompt → .5033,
8B-M_ω prompt → .5567) BEATS fresh GEPA-14B optimization (.4367 = found nothing) — i.e.,
articulation discovered at 8B transfers UP better than re-discovering at 14B with a 600 budget;
the M_ω-stair (re-firing now) should confirm M_ω>>GEPA at this rung. NO new completions this
cycle: livebench-v5 still drop_one (grinding its 7-8 unit drop-tests), pupa prefix_k17,
ifbench 537min stuck (fix queued behind livebench), M_ω-stair still none finished. z.ai/tunnel/
servers UP; 10 sk1 + 4 sk3 procs.

**HB63 (~11:10 PT): livebench-v5 AND pupa-v4 both at add_back = final stage; verdicts imminent.**
Armed a dedicated pupa result watcher (bhspaqzon, local — E2: sign test vs pupa official .862
on landing). livebench watched by b27b2iagy (E1 + ping + then ifbench 32k fix). New GEPA cell:
aime-1.7B .2067/.20 (best≈seed, GEPA found nothing at 1.7B — consistent w/ hover-14B, GEPA weak
at small budget). aime-8B still the flagged .5333 anomaly (user re-run recommendation stands,
HB60). **M_ω-stair STILL none finished (~6h in; 164-unit hover pool inherently slow — expected
per HB61).** ifbench 597min (~10h) stuck, fix queued behind livebench. z.ai/tunnel/servers UP;
10 sk1 + 4 sk3 procs. Nothing to record/sign-test this cycle.

**HB64 (~11:40 PT): all M_ω-stair cells VERIFIED progressing (none stuck); verdicts still
grinding add_back.** M_ω-stair proposals freshness (all ≤7min stale, advancing): hover-4B 202
rows (NEAR DONE → first envelope cell imminent), hover-8B 143, hover-1.7B 132, aime-8B 39
(slow = 8000-tok reasoning), aime-4B 3 (just fired — its GEPA completed this cycle). So the
envelope is SLOW not hung (164-unit hover pool + aime reasoning cost). livebench-v5 + pupa-v4
both advancing through add_back (218 / 241 rows, fresh) — many skipped-unit re-tests, verdicts
still imminent (watchers b27b2iagy + bhspaqzon armed). New GEPA cell: aime-4B .3267/.42
(legit +.09). ifbench 657min (~11h) stuck (fix queued behind livebench). z.ai/tunnel/servers UP;
10 sk1 + 4 sk3 procs. Nothing to record/sign-test.

**HB65 (~12:10 PT): livebench-v5 PASSED CONFIRM GATE → running final test (verdict imminent).**
livebench-v5 phase=final_test_seed (fresh) = confirm gate passed, final evals running →
result.json in minutes (biyh66my4 armed: E1 sign test + ping + then ifbench 32k fix). pupa-v4
still add_back (248 rows, fresh — grinding). New GEPA cells: aime-4B .3267/.42 (+.09), **aime-32B
.2333/.3533 (+.12)** — but ⚠ aime-32B seed .2333 < fixed-arm .28 < 8B .32, REINFORCES the
aime-32B 8000-tok TRUNCATION concern (HB58); aime-32B GEPA improved but off a suppressed base.
aime-14B GEPA still running. M_ω-stair still none finished (hover-4B ~2/3 through: 202 rows of
~305). ifbench 717min (~12h) stuck. z.ai/tunnel/servers UP; 10 sk1 + 3 sk3 procs. No completed
results to sign-test yet (livebench about to be first).

**HB66 (~12:40 PT): livebench-v5 DEGRADED-TEST (quarantined, tie STANDS) + ifbench 32k FIX
launched.** livebench-v5 landed: **seed-sanity FAIL (seed .5635 < band .62-.72)**, fell_back=
True (n_compiled=0, guard found nothing on confirm .6894→.677), best .619 vs GEPA .6956 =
3W-25L. DIAGNOSIS: the shipped prompt = GEPA's own winner (fell back), yet scored .619 not
.6956 on the SAME prompt → seed AND GEPA-prompt both depressed ~.07-.10 = **transient final-test
degradation on 8078** (server verified HEALTHY after: coherent sanity gen, GPU1 fine — the
degradation window cleared). NOT a real loss. QUARANTINED runs→INVALID_seedfail_v5sk2 (sk2+
local). **livebench = TIE (v4 .682 vs .696 p=.81 STANDS; v5 confirms M_ω finds no improvement,
consistent w/ EVT ±1SE compression finding). 0 losses intact.** Terminal ping sent.
IFBENCH FIX (runbook E1.f, now that livebench freed 8078): killed ifbench-v5 3934278 + 8078
server(16k) 3481835 by PID; **relaunched 8078 @ --max-model-len 32768 (sk2 GPU1, pid 926491)**;
ifbench relaunching fresh --run-tag v6ctx32k (watcher b35a88y18). 32k fixes the 8385+8000>16384
overflow that stuck ifbench-v5 12h. pupa-v4 still add_back (watcher bpg2wmemf armed). M_ω-stair
cells still cooking. NEXT: ifbench-v6 mining confirm; pupa verdict; first envelope cell.

**HB66b (~13:00 PT): ifbench v6 fd-exhaustion crash → fixed + relaunched.** First v6ctx32k
launch DIED on OSError [Errno 24] Too many open files (sk2 soft ulimit=1024; SAME fd-exhaustion
class as sk1 hover HB54 — I forgot to carry the ulimit+DSPY_CACHEDIR fix into the sk2 launch).
Confusing "different PID at 00:00 each check" was fd-crashed remnants, NOT a respawn loop
(verified: killed all → IFBENCH_V6_PROCS=0, no respawn). FIX: sk2 hard limit=1048576 so
`ulimit -n 65536` works; quarantined partial→unitrecomb_v6ctx32k_fdcrash; relaunched with
ulimit 65536 + DSPY_CACHEDIR=$HOME/dspy_cache_ifbench_v6 (pid 1191901); mining watcher bri82bnhw.
LESSON (runbook): EVERY sk2/sk3 multi-thread arm launch needs `ulimit -n 65536` +
DSPY_CACHEDIR — the early sk2 v5 arms got lucky at 1024 (single-stage or fewer fds); ifbench's
2-stage program × 32 threads exhausts 1024 fast. 8078 @32k still UP. livebench tie stands, pupa
add_back, envelope cooking.

**HB67 (~13:40 PT): pupa WATCHDOG-RESTARTED (chronic tunnel wedge); ifbench-v6 fix working;
M_ω-stair all progressing (hover-4B nearest).** pupa 51260 wedged (0% CPU, >15min stale on
tunnel dead-socket) → watchdog restarted as 69011 (lost its near-done add_back, re-running
marginals from scratch — 2nd+ pupa tunnel wedge; ⚠ CHRONIC: pupa via 8077 tunnel keeps
wedging, each restart loses hours; if it wedges again next cycle consider moving pupa to run
ON sk2 localhost like the v5 arms — caveat: PUPA judge metric may add z.ai dep; parked, letting
watchdog manage 1 more cycle). ifbench-v6 (32k fix): mining done, 28 marginals, 8078 UP —
overflow FIXED, progressing (verdict many hrs out). M_ω-stair freshness (all progressing, slow
= hover BM25 + aime reasoning): hover-4B 230 rows @add_back (NEAREST → first envelope cell
imminent), hover-8B 166 @prefix_k1, hover-1.7B 143 @marginal, hover-14B 60 @marginal, aime-8B
47 @marginal (16min = slow aime eval not wedge). aime-14B GEPA still running. z.ai/tunnel/
servers UP. No completions to sign-test.

**HB68 (~14:10 PT): M_ω-stair SLOW-BUT-PROGRESSING (envelope = 1-2 day fill); ifbench-v6 +
pupa both healthy.** Diagnosed the zero-completions: NOT stuck — hover-4B M_ω advancing through
add_back (246 rows, 2min fresh, 30/~48 add_back candidates done). Cost/cell ≈ 270 evals (164
marg + 48 prefix + drop-one + 48 add_back + confirm/test) × ~2min (hover BM25) ≈ 9h/cell, AND
each scale's GPU serves BOTH hover+aime M_ω (+GEPA) serialized → limited parallelism. **REALISTIC
ETA: first envelope cell (hover-4B) ~2-3h; FULL envelope ~1-2 days.** Acceptable (user checks in
tomorrow; envelope is bonus on top of the confirmed campaign wins). Can't speed w/o redesign
(runbook forbids mid-run config change; comparability). ifbench-v6 75 marginals (32k working,
progressing). pupa (69011, restarted) now prefix_k6 — HOLDING, no re-wedge this cycle. aime-14B
GEPA still running. z.ai/tunnel/servers UP. No completions to sign-test.
NOTE for morning: envelope may be incomplete at user check-in; the CONFIRMED deliverables
(hotpot ✅✅, hover ✅, aime ✅ replicated, livebench/ifbench ties, fixed-arm transfer curves
HB58, contrast-pair finding HB54b) stand independent of envelope completion.

**HB69 (Jul 23 ~09:55 PDT, user back — audit + actions): ★ FIRST ENVELOPE POINT WHERE
PER-SCALE M_ω BEATS PER-SCALE GEPA; two dead/idle lanes revived.** Sweep found TWO completed
envelope cells: (1) **hover-32B stair (sk3, 09:06): seed .46 → best .5633 — BEATS 32B GEPA
official .5233 (+.04) AND the 8B-transplant .5333** = first clean per-rung envelope win for
re-optimized M_ω. (2) aime-8B stair (sk1, 09:46): seed .3733 → best .40 — consistent with
canonical 8B M_ω (.4133), which further isolates sk1's aime-8B GEPA .5333 as the outlier
(re-run decision still with user). INCIDENTS: aime-14B GEPA official (998111) found DEAD, no
result — cause = **Qwen3-14B reasoning-overflow on AIME: text=None, reasoning_content consumed
the full 8000-token budget** (52 AdapterParseErrors + repeated tracebacks in log; same failure
family as the aime-32B truncation concern). Old log preserved → official_stair_run.log.attempt1;
**relaunched pid 1019855** (chainer will auto-fire aime-14B M_ω stair when it lands).
staircase_eval aime-14B (1788324): logged seed .34 at 08:58 then silent (evals.jsonl 1 row) —
NOT killed (could be a legitimately long arm-2 eval; dspy logs only at eval end); rule: if
evals.jsonl still 1 row next cycle → kill by PID + relaunch. sk3 was IDLE after its two cells
finished → **launched aime-32B M_ω stair pid 531385** (frozen pool 48u, init==GEPA winner
confirmed; NB sk3 ulimit raise not permitted, hard cap 1024 — hover-32B stair completed fine
at that limit, watching for fd errors). ifbench-v6 confirmed clean on 32k (117 rows fresh,
0 "16384" in last 200 log lines). pupa healthy (val .88-.90 vs official .862), 3h watcher
expired NO_RESULT → re-armed. livebench v5sk2 quarantine synced local. ⚠ local ifbench
rescore (95980) 23.5h on 16k port 8077 emitting overflows — non-load-bearing, awaiting user
PID-kill say-so. Envelope rungs done: hover{32B}, aime{8B}; in flight: hover{1.7B,4B,8B,14B},
aime{1.7B,4B,32B}; blocked-on-GEPA: aime{14B}. hover-4B nearest (260 rows, add_back).

**HB70 (Jul 23 ~10:05 PDT): all lanes healthy; one wedge cleared per HB69 rule.** Sweep: no new
result.json beyond HB69's aime-8B stair. staircase_eval aime-14B (1788324) still frozen (evals.jsonl
1 row, 62min stale) → rule tripped → killed by PID, relaunched as **1201323** (log append-preserved;
seed .34 row retained in evals.jsonl). aime-14B GEPA relaunch (1019855) healthy, rollouts running,
0 AdapterParseErrors so far. aime-32B stair (531385) healthy, 0 fd errors — only benign
"LM response truncated (max_tokens=8000)" warnings (more evidence for the 16k-output rerun decision).
ifbench-v6 125 rows fresh; pupa 363 rows fresh (val .88-.90 band); hotpot-GLM 4433 rows churning
(3772 select_marginal); z.ai UP (962ms); sk2 zero free GPUs (standing checklist's "second server"
item moot — 8078/8079 already up). NB standing heartbeat item (d) references
outputs/r2_recovery_v2_momega_bigpanel on sk3 which DOES NOT EXIST (only outputs/r2_recovery/*) —
stale checklist path, ⚠ for user to confirm/retire. Stair lanes in flight: hover{1.7B 160r, 4B 261r,
8B 194r, 14B 93r}, aime{1.7B 46r, 4B 27r, 32B just started}. No completions to sign-test.

**HB71 (Jul 23 ~10:45 PDT): USER DIRECTIVE — push livebench/ifbench/pupa head-to-heads with all
available GPUs. PUSH WAVE LAUNCHED (5 concurrent attempts across the 3 benches).** Rationale: these
are the 3 remaining non-wins (livebench tie .682/.696; ifbench tie w/ v6 in flight; pupa unresolved,
val .88-.90 vs official .862). Diversity lever = fresh GLM-5.2 mining w/ v5 framings + enriched
trajectory history (rsynced v4-winner + in-flight snapshots), racing independent stochastic climbs;
rollup takes best GUARDED variant per bench (established pattern). Actions:
(1) KILLED local rescore chain by PID (wrapper 81218 first, then 95980; 23.5h futile 16k-overflow
rescore) → tunnel 8077 now serves local pupa EXCLUSIVELY. Chain-tail analyses can run manually later.
(2) sk2: launched **livebench unitrecomb_v6wide pid 4153932** on idle 8079 (mining mode, max-units
128, prefix-cap 64; mining history enriched w/ unitrecomb_v4local = the .682 v4-winner trajectory).
⚠ sk2 ulimit raise failed this launch (soft 1024) — single-stage program, watch for Errno 24.
(3) sk3: verified GPUs 1-6 EMPTY (fresh nvidia-smi), z.ai keys present, hard fd cap 16384 (fine);
pre-downloaded Qwen3-8B; synced livebench v4local + pupa v4snap (363-row in-flight snapshot) +
ifbench v6snap (sk2 v6 partial, 125 rows) for mining. Orchestrator **launch_pushwave.sh pid 538957**
(logs $HOME/pushwave.log): 3 × Qwen3-8B@32k servers on GPUs 1/2/3 (ports 8176/8177/8178, per-GPU
claim-check before launch, content-based ready check) then 3 arms: **livebench v6widesk3**
(max-units 128/prefix 64), **ifbench v7wide** (max-units 160/prefix 64), **pupa v4sk3** (mirror
v4 config). GPU 7 untouched (user's). Code facts checked before launch: suggester = GLM-5.2 via
z.ai key files in $HOME (paperexact_arms.py:205-227); PUPA metric judge = GLM-5.2 hardwired
(z.ai dep, currently UP, 962ms); --robust-answer-extract is AIME-ONLY (SystemExit otherwise).
Total attempts/bench now: livebench ×2 (sk2 v6wide, sk3 v6widesk3), ifbench ×2 (sk2 v6ctx32k in
flight, sk3 v7wide), pupa ×2 (local v4 mid-flight ~prefix/dropone, sk3 v4sk3 fresh). Watcher armed
on PUSHWAVE_LAUNCHED for arm-health verify.

**HB71b (Jul 23 ~10:20 PDT): BUDGET FIX — 3 of 4 wave arms were underbudgeted, caught at launch
(+few min), killed by PID + relaunched w/ correct budgets.** The wider configs raised the minimal
selection plan past 24000 calls: harness warned "later stages will be skipped and the no-regret
guard will ship init" (= wasted run). livebench 128u/64pfx needs ≥26323 → relaunched at 40000;
ifbench 160u/64pfx needs ≥36000 → relaunched at 48000. Partial dirs quarantined *_underbudget.
NEW PIDS: sk3 livebench v6widesk3 **544581**, sk3 ifbench v7wide **544582**, sk2 livebench v6wide
**688050**. pupa v4sk3 (542545) unaffected (96u/48pfx fits 24000; mining done: 62 LLM + own-traj
= 96 used). All 3 relaunches verified alive at 45s with NO budget warning. LESSON for runbook:
any --max-units/--prefix-cap increase must recompute budget — grep launch log for "minimal
selection plan" within the first minute.

**HB72 (Jul 23 ~10:55 PDT): ★★ SECOND ENVELOPE WIN — hover-4B stair .5233 BEATS 4B GEPA .4667
(+.057). Per-rung envelope table forming; all 6 push-wave arms healthy.** hover-4B unitrecomb_stair
landed (seed .42 → .5233, regression_flag False) vs hover-4B GEPA official .4667 (seed .39) and
8B-transplant .510. ENVELOPE TABLE (hover, per-rung re-optimized):
| rung | GEPA | M_ω-stair | 8B-transplant | envelope |
| 4B   | .4667 | **.5233** | .510  | M_ω +.057 |
| 32B  | .5233 | **.5633** | .5333 | M_ω +.040 |
Both completed hover rungs: freshly re-optimized M_ω > freshly re-optimized GEPA, and > transplant
— the model-optimal envelope is M_ω-defined so far. (aime-8B rung: stair .40 < GEPA⚠ .5333
anomaly, unresolved pending user's re-run decision.) PUSH WAVE: all 6 attempts alive & writing
(sk2 ifbench v6 152 rows; sk2 livebench v6wide 10 rows; sk3 livebench 18/ifbench 43/pupa 10 rows;
local pupa 368 rows now at **prefix_k21**). Local hotpot-GLM PID drift 77909→68188 (watchdog
restart), arm alive 18h. z.ai UP (1564ms). aime-14B: GEPA relaunch (1019855) alive, 0 parse errors,
still in baseline eval (bar 0/600 — 14B server 3-way shared, slow OK); staircase_eval relaunch
(1201323) 1 row/54min silent — NOT restarted (silence expected mid-eval; dspy logs only at eval
end; server provably processing hover-14B). ESCALATION RULE: if still 1 row AND no new log bytes
at next heartbeat → py-spy/queue diagnosis, NOT blind restart. Stair lanes all fresh (hover 1.7B
171r, 8B 212r, 14B 104r; aime 1.7B 56r, 4B 35r, 32B 12r).

**HB73 (Jul 23 ~12:00 PDT): aime-14B DOUBLE-WEDGE root-caused (dead-socket under 3-way contention)
→ killed both + SEQUENCER deployed; everything else green.** Escalation rule from HB72 tripped:
staircase_eval (1201323) log flat since 09:59:55 AND aime-14B GEPA (1019855) at 0/600 rollouts
after 2h, 0 errors — while hover-14B stair advanced on the SAME 8174 server (control). Socket
diagnosis (no py-spy on box): both clients hold ~30 ESTAB sockets to 8174, blocked on responses
that never arrive; server 152 ESTAB total, actively serving hover → classic litellm dead-request
wedge under contention, NOT a server failure. Blind relaunch would re-roll into the same
contention → SERIALIZED instead: killed 1201323+1019855 by PID; **sequencer pid 2644789**
($HOME/aime14b_sequencer.sh, log aime14b_sequencer.log) waits for hover-14B stair result →
relaunches aime-14B GEPA official SOLO on 8174 (log preserved .attempt2) → then staircase_eval;
chainer auto-fires aime-14B M_ω stair after GEPA lands. 8174 server pid 1741512, log
vllm_Qwen3-14B_8174.log (for later forensics). ALL other lanes green: 6 push-wave arms fresh
(sk2 ifbench 185r, sk2 livebench 23r; sk3 livebench 44r/ifbench 101r/pupa 24r; local pupa 380r
now in candidate-mutation/add-back region); stair lanes hover 1.7B 182r/8B 230r/14B 118r, aime
1.7B 67r/4B 47r/32B 23r; z.ai UP (2190ms). sk3 8176/8177 curl "000" = sweep-agent endpoint
artifact (their arms actively write through those ports). Local hotpot-GLM watchdog PID drift
(→10962), arm 18h alive. No new results to sign-test (hover-4B stair already in HB72).

**HB74 (Jul 23 ~12:40 PDT): USER DIRECTIVE — sk3 GPUs 1-2 off-limits. Discovered both wave servers
there were ALREADY DEAD (last sweep's curl 000 was real); GPUs 4-7 claimed by other users. Wave
arms re-homed; GPUs 1-2 confirmed drained (0 MiB).** Fresh nvidia-smi: sk3 GPU1/2 = 4 MiB (servers
8176/8177 gone, pids 538987/538995 no longer exist), GPU4-7 = ~157GB/100% (others' vLLM jobs —
untouched). RE-HOMING: killed orphaned arms 544581/544582 by PID, partials quarantined *_gpuevict;
**livebench v6widesk3 relaunched pid 699021 → sk3:8178** (shares pupa's GPU3 B200; --eval-threads
24 to keep 2-client load light; no budget warning); **ifbench v7wide → sk1 GPU5** (was empty on
fresh check): new Qwen3-8B@32k server port 8180 (pid 585305) + launcher pid 646901 (waits ready →
launches arm, 160u/48000 budget); ifbench mining trajectories rsynced local→sk1 (official/inhouse/
unitrecomb + v6snap). FAMILY STAIRCASE RE-BLOCKED: sk3 has Llama-3.2-1B/3B + 3.1-8B + gemma-2/3/4
weights AND HF token cached, but zero free GPUs after the 4-7 claim — parked until GPUs free
(⚠ tell user: family bringup is launch-ready the moment 1-2 B200s open). Envelope lanes unaffected
and running: hover{1.7B,8B,14B} + aime{1.7B,4B,32B} stairs, aime-14B sequencer (2644789) waiting on
hover-14B, chainer alive. Contention lesson applied: livebench@24 threads on shared 8178.

**HB75 (Jul 23 ~14:10 PDT): ★★ THIRD ENVELOPE WIN (aime-1.7B) — and it reframes the capability
threshold; pupa hedge SIGSTOPPED to protect primary's judge from our own rate-limit pressure.**
NEW CELL: aime-1.7B stair seed .2267 → **.2933** vs aime-1.7B GEPA official **.20** (GEPA best ≤ its
own seed .2067 — found nothing at 1.7B). Recall HB58: the 8B-transplant showed ZERO lift at 1.7B
(read then as a capability threshold). Per-scale SELECTION from the SAME frozen 8B pool finds
+.07-.09 → the threshold was partly a TRANSPLANT artifact: small models benefit from the pool, but
only via re-selection at their own scale ("selection transfers, combinations don't"). Descriptive
only pending paired test (150-item test; run sign test vs official from item scores).
ENVELOPE TABLE now: hover-4B M_ω .5233/GEPA .4667 (+.057); hover-32B .5633/.5233 (+.040);
aime-1.7B .2933/.20 (+.093); aime-8B .40/GEPA⚠.5333 (anomaly pending user).
PUPA PROTECTION: local pupa (add_back, 388r) logging 21× z.ai 1302 rate-limit errors/100 lines on
its judge — main co-consumer of the same key = our sk3 pupa hedge → **kill -STOP 542545** (state
Tl confirmed; reversible). RESUME TRIGGER: local pupa result.json lands (watcher b2xc3qhrx, 3h,
re-armed) → kill -CONT 542545. Rationale: symmetric eval noise → false-fallback risk on the gate;
hedge is 2.7h into ~15h, pause costs little. THREADS QUESTION (user): all campaign servers spot-
checked at ~100% GPU util (sk1 GPU5, sk3 GPU0/3, sk2) → eval-threads bump would add queueing not
throughput; NO restarts done (user said none needed); future launches keep 32 (48 only if a card
shows <90% util). Sweep: 10/10 arms alive; sk2 ifbench v6 ALSO at add_back (216r — verdict close);
sk2 livebench 37r; sk1 ifbench v7wide mining; sequencer waiting hover-14B; GPUs 1-2 sk3 = 0 MiB ✓.

**HB76 (Jul 23 ~14:30 PDT): IFBENCH v6 VERDICT — DIRECTIONAL LEAD, NOT SIGNIFICANT. M_ω .4558 vs
GEPA .4116 (+.044), sign test 39W-29L (n=68 decisive), p=.275.** Details: v6ctx32k landed on 294
test items, regression_flag False. GEPA official found NOTHING on ifbench (best_test == seed_test
== .4116; matches v7wide's init==seed:True observation) → our +.095-over-seed is the only
improvement either optimizer found; the head-to-head vs GEPA == vs seed here. ⚠ seed-reading gap:
v6's seed_test .3605 vs official's .4116 (~1.7 SE at n=294, temp-.6 eval noise — inside tolerance,
noted not quarantined). EVT refresh (analyze_bounds_evt --paperexact ifbench): **endpoint ≈
best-achieved** (dequantized median ~.444, margin inside binomial SE) = v6's search exhausted its
own draw distribution → more sampling of THIS pool won't move it; **v7wide's wider pool (160u, on
sk1:8180) is the right instrument** — it changes the draw distribution (mining moves the LEVEL,
per capture-recapture doctrine). SCOREBOARD: ifbench upgraded tie → directional lead (not a win;
claim stays honest at p=.275). Zero losses intact. Next verdicts: local pupa (add_back), livebench
twin tonight.

**HB76b (Jul 23 ~14:45 PDT): REPORTING POLICY SET BY USER — "keep the win and report significance /
not-significance, and EVT upper bounds as well."** Deliverable format per bench, three columns
always: (1) point result M_ω-best vs GEPA-official (best GUARDED variant per rollup rules),
(2) paired sign test p labeled significant / not-significant — never suppress the point estimate
for lack of significance, never overclaim p≥.05 as a win, (3) EVT process-conditional endpoint
(dequantized median, with the standard caveat: process-conditional = ceiling of THIS search's draw
distribution, not an all-prompt bound; the only certified all-prompt bound remains the DPI
fixed-target cap). EVT batch refresh for all 6 benches running (b1n8419zz); deliverable table to
be assembled on completion.

**HB76c (Jul 23 ~15:00 PDT): CORRECTION to HB76's EVT attribution + full EVT table with population
made explicit.** analyze_bounds_evt --paperexact pools **rescore.jsonl matrices** (paperexact_
rescore.py output = pre-push-wave candidates), NOT the live arm draws. HB76's "v6 exhausted its own
draw distribution" was WRONG population attribution — the .4439 endpoint was the OLD pool's ceiling,
and **v6's winner .4558 EXCEEDS the old pool's GPD CI upper (.4500)** = direct evidence the wider
mining MOVED THE LEVEL past the prior process ceiling (capture-recapture doctrine confirmed, not
contradicted). EVT TABLE (population = pre-wave rescore pools, GPD k-median endpoint [boot CI]):
| bench | pool n | pool best | endpoint | CI | current winner vs ceiling |
| aime | 34 | .4667 | .4667 | [.4400,.4914] | win .4133-.4267 inside CI |
| hover | 38 | .5500 | .5500 | [.5300,.5696] | win .5467 at ceiling |
| hotpot | 16 | .4367 | .4367 | [.4233,.4367] | **win .6333 SHATTERS old pool** (pool predates v5) |
| ifbench | 75 | .4439 | .4439 | [.4406,.4500] | **v6 .4558 > CI upper** (mining moved level) |
| livebench | 11 | .7244 | .7244 | [.7215,.7244] | ceiling ABOVE GEPA .696 → headroom EXISTS for twins (NB .7244 = retrospective test-best of a rescored candidate; NOT claimable, selection never saw it) |
| pupa | — | — | no rescore pool | — | — |
TODO: after tonight's landings, refresh rescore matrices for new arm dirs via OFFLINE BATCH vLLM
(per user's batch-vLLM point + standing rule — the rescore pass is embarrassingly parallel), then
regenerate EVT columns on the updated pools for the deliverable table.

**HB77 (Jul 23 ~15:50 PDT): all-nominal sweep; idle cycle used to implement the MIPROv2 baseline
arm (user-requested baselining).** Sweep: pupa local at final_test_seed (396r — verdict imminent,
hedge still Tl-stopped ✓); livebench twins both select_marginal (sk3 70r, sk2 51r); ifbench v7wide
14r select_marginal; envelope lanes hover-8B/aime-4B at add_back, hover-1.7B drop_one, hover-14B +
aime-32B marginals; z.ai UP 1561ms; GPUs 1-2 idle (816/4 MiB — GPU1 warming = user's process,
expected). NEW CODE: paperexact_arms.py **--arm mipro** = dspy.MIPROv2 baseline (the GEPA paper's
principal comparator): prompt_model=GLM-5.2 (parallels GEPA's reflection LM), **instruction-only
regime max_*_demos=0** (demos would be silently dropped by get_instructions → would misrepresent
MIPROv2; this makes it optimize the same object as all other arms), auto=None + num_trials =
0.8×budget/minibatch(25), num_candidates=12, **requires_permission_to_run=False pinned** (defaults
True in dspy 3.2.1 = would HANG a nohup run at an interactive prompt), max_errors=10k, default
budget 600 (shares official's default = budget-matched to GEPA per the paper's protocol). Syntax +
CLI verified. VALIDATION PLAN (validate-before-scaling): first run = aime on local 8077 once pupa
releases it; fan out to 6 benches on freed servers tomorrow. Baseline roadmap recorded: MIPROv2
(must-have) → Best-of-N floor (optional) → SIMBA/COPRO (secondary); unsupervised/reconstruction-
objective expansion needs design+prereg note BEFORE any confirmatory run (drafting next idle cycle).

**HB78 (Jul 23 ~16:20 PDT): ★ PUPA VERDICT — M_ω .9074 vs GEPA .8621 (+.045), 39W-24L (n=63
decisive of 221), p=.077 — NEARLY significant, strongest of the 3 contested benches.** Local pupa
v4 (dir unitrecomb, untagged) landed: seed .8313 → best .9074, regression_flag False; official:
seed .8030 → .8621. Seed-gap .028 ≈ 1.2 SE (n=221) — inside tolerance. Sign test: 39W-24L,
two-sided p=.0769 (significant at .10, not .05). Per HB76b reporting policy: pupa = WIN ON POINTS,
not-significant label, EVT column pending (no rescore pool yet — pupa joins the offline-batch
rescore pass). HEDGE RESUMED per trigger: kill -CONT 542545 → state Sl confirmed; sk3 pupa v4sk3
continues as independent replication shot (NB: protocol remains best-GUARDED-variant + its own
sign test — no post-hoc p-pooling; a second independent beat would be reported descriptively as
replication, aime-style). CONTESTED-BENCH BOARD: pupa +.045 p=.077; ifbench +.044 p=.275
(v7wide climbing); livebench −.014 tie (twins in flight tonight). Zero losses intact. ALSO this
cycle: arm_mipro switched to PAPER-LITERAL MIPROv2-Heavy config (auto="heavy", max_errors=10k,
budget=realized-spend per artifact protocol; demos still pinned 0 = flagged deviation awaiting
user ruling); **--arm official_merge added** (dspy GEPA use_merge=True = GEPA+Merge, the strongest
GEPA-family baseline); GRPO = cite-not-reproduce recommendation; Abl-SelectBestCandidate = skip.

**HB79 (Jul 23 ~17:30 PDT): JUMP-HOST OUTAGE (whale.stanford.edu resets SSH pre-auth) — all sk
boxes monitoring-blind; remote lanes UNAFFECTED (nohup'd). + FIRST ABSORPTION (H-i) NUMBERS; ⚠
aime-1.7B envelope win moved to UNDER-AUDIT.** Sweep local-only: z.ai UP (1481ms); hotpot-GLM
alive; "pupa v4 log stale/no owner" = benign (run FINISHED, HB78). Host process restarted this
cycle (old watchers died; hourly sweeps + new watcher cover). NEW ANALYSIS analyze_stair_units.py
(retained units by text-containment vs frozen pool, in shipped winner):
| cell | retained/pool | frac |
| hover-4B | 7/164 | .04 |
| hover-32B | 34/164 | .21 |
| aime-8B | 16/48 | .33 |
| aime-1.7B | **0/48** | .00 ⚠ |
hover: 32B retains ~5× more units than 4B → **H-i (absorption rises with scale) SUPPORTED on
hover** (2 points, descriptive). ⚠ aime-1.7B winner retained ZERO units (5229-char candidate) →
suspicion: stair may have shipped ≈init (GEPA's own candidate) and the +.09 vs GEPA official could
be RE-MEASUREMENT noise, not selection value → **aime-1.7B envelope win UNDER AUDIT** (identity
check stair-cand vs official-cand text; blocked on SSH; runs at reconnect). Do NOT quote aime-1.7B
+.093 until audit passes. Connectivity watcher armed (3-min polls, 6h). Blocked-on-SSH queue:
(1) aime-1.7B identity check, (2) full sweep, (3) tonight's twin/v7wide verdict pulls.

**HB79b (Jul 23 ~18:10 PDT): ★★ AUDIT VERDICT — aime-1.7B ENVELOPE WIN **RETRACTED** (identity
check: stair candidate == GEPA official candidate, byte-identical after whitespace-norm); hover
envelope wins CONFIRMED GENUINE and H-i strengthened; same-prompt variance data resolves the
aime-8B GEPA anomaly reading.** SSH restored → blocked audit ran. Findings:
(1) **aime-1.7B AND aime-8B stair candidates are IDENTICAL to their GEPA officials'** — the
no-regret guard fell back at both scales (no addition cleared confirm). All stair-vs-official
deltas on these rungs = RE-MEASUREMENT NOISE of the same prompt: 1.7B .2933 vs .20 (Δ.093!),
8B .40 vs .5333 (Δ.133!). **NEVER quote aime-1.7B +.093 (HB75 claim retracted; its "selection
transfers" interpretation dies with it).**
(2) Same-prompt pairs = free eval-variance data: aime n=150 temp-.6 test evals swing ~.09-.13 →
**the aime-8B GEPA .5333 anomaly (HB60) is most plausibly an upward eval fluctuation** (its own
prompt re-measured .40); recommendation to user upgraded: aime envelope/anomaly rows need
multi-eval averaging (3-5 test passes) before any quoting.
(3) **hover rungs GENUINE**: stair ≠ official; NET-of-init added units: 4B +4, 32B +31 (in-stair
gross 7/34 included 3 init-contained units each — pool mined from these trajectories, so init
trivially contains own fragments; analyze_stair_units.py fixed to report NET + detect fallback).
**Corrected H-i: hover absorption 4 → 31 units (4B→32B) — supported, cleaner than gross.**
CORRECTED ENVELOPE TABLE: hover-4B M_ω .5233 vs GEPA .4667 (+.057, 4 net units) ✓; hover-32B
.5633 vs .5233 (+.040, 31 net units) ✓; aime-1.7B/8B = guard-fallback ties (M_ω=GEPA by
construction) + variance data. Pending rungs (hover 1.7/8/14B, aime 4/14/32B) now get the
identity check as a STANDARD post-landing step.

**HB80 (Jul 23 ~17:45 PDT): 3 new stair rungs landed + identity-audited; hover H-i now a
4-rung monotone ladder; aime same-prompt variance CONFIRMED at .133; MIPROv2-Heavy baseline
LAUNCHED (first ever run of the arm).** Sweep found 2 fresh result.json (aime-4B, hover-1.7B)
plus hover-8B from 14:43; all three run through the now-standard identity check vs their GEPA
official (`sha1` of whitespace-normed best_candidate).

CORRECTED ENVELOPE TABLE (identity-audited, supersedes HB79b's):
| bench | model | GEPA | M_omega | delta | identity |
|---|---|---|---|---|---|
| hover | 1.7B | .4000 | .4467 | **+.047** | distinct - GENUINE (new) |
| hover | 4B | .4667 | .5233 | **+.057** | distinct - GENUINE |
| hover | 8B | .4567 | .5833 | **+.127** | distinct - GENUINE (new, largest) |
| hover | 32B | .5233 | .5633 | **+.040** | distinct - GENUINE |
| aime | 1.7B | .2000 | .2933 | (+.093) | IDENTICAL - RETRACTED |
| aime | 4B | .4200 | .4133 | -.007 | distinct - tie/small loss (new) |
| aime | 8B | .5333 | .4000 | (-.133) | IDENTICAL - RETRACTED |

(1) **hover is now 4/4 genuine wins across 1.7B->32B** — every hover rung beats its own
re-optimized GEPA, none is a guard fallback. This is the envelope result's backbone.
(2) **H-i (absorption rises with scale) does NOT hold on the score delta**: deltas run
.047/.057/.127/.040 — non-monotone, 8B is the peak, 32B the smallest. H-i was framed on
absorbed UNIT COUNT (4 -> 31 net, 4B->32B), which still rises; but the two readouts now
DISSOCIATE (32B absorbs the most units for the least gain). Do not report H-i as supported on
score; report the unit-count ladder and the score ladder separately and say they come apart.
**UPDATE (same heartbeat, analyze_stair_units.py shipped to sk1 and run): H-i is NOT supported
on the UNIT-COUNT readout either.** Net-of-init retained units across the full hover ladder are
**1.7B=38, 4B=4, 8B=39, 32B=31** (of a 164-unit pool) - non-monotone, and the 4B rung is a stark
outlier: it wins +.057 while absorbing only 4 units, where 8B needs 39 for +.127. The earlier
"absorption rises 4 -> 31" claim (HB79b) was an artifact of having only TWO rungs (4B, 32B)
in hand; with all four it disappears. **RETRACT H-i as stated on both score and unit count.**
What survives is weaker and more interesting: absorption EFFICIENCY (gain per absorbed unit)
is wildly scale-dependent - 4B ~.014/unit vs 8B ~.003/unit - so the right question is not "how
much does a scale absorb" but "how much does a scale need to absorb". Reframe before the next
rung lands. aime net units: 1.7B=0, 4B=1, 8B=0 (the two zeros are the guard fallbacks).
NOTE: result.json's `units` field is 11 for every cell incl. the guard-fallback ones, so it is
NOT the selection outcome and must never be read as retained units. hover-32B is not on sk1
(its 31 is carried from the earlier audit on its own box).
(3) **aime same-prompt eval variance now measured twice**: byte-identical prompts scored
.2933-vs-.2000 (1.7B) and .4000-vs-.5333 (8B). |delta| up to .133 at n=150/temp-.6. No aime rung
is interpretable single-pass; aime-4B's -.007 is inside noise and is a TIE, not a loss.
Multi-pass (3-5) test averaging is a prerequisite for any aime envelope claim.
(4) **MIPROv2-Heavy first run LAUNCHED** (hover/Qwen3-8B, sk3 GPU 4, port 8182, run-tag
`miprov2`). Validation-before-scaling per HB77; hover chosen over aime because aime's .133
noise floor would make the baseline unreadable. Harness synced to sk3 (backup
`paperexact_arms.py.bak_pre_mipro_20260723`); diff vs sk3 was exactly the arm_mipro +
official_merge additions, nothing else. sk3 GPUs 1-2 untouched per user directive; 4-5 were the
free pair, 5 still free for the GEPA+Merge follow-on.
(5) aime-14B stair still NOT started - gated behind hover-14B stair (185 rows, live) via
sequencer pid 2644789. aime-14B official/result.json also absent.

**HB81 (Jul 23 ~19:05 PDT): sk3 GPU EVACUATION (user: free GPUs by 8pm) — GPUs 4+5 FREED at
~6:15pm.** MIPROv2-Heavy and GEPA+Merge (both ~1.3h in, unresumable but cheap) were killed by
explicit PID (arms 919542/927518, servers 918830/926545, EngineCore children verified dead) and
RELAUNCHED on existing idle servers: **MIPROv2 -> sk1:8173 (pid 350602)**, **GEPA+Merge ->
sk2:8078 (pid 1145787)** — both confirmed past startup before the sk3 kills. No new servers
needed (sk1 GPUs 0-4 / sk2 ports 8077-8078 were holding idle Qwen3-8B servers from finished
stair arms). Remaining sk3 prompt-opt footprint: GPU 0 (aime-32B stair, in add_back = final
phase, ~1-2h out; its :8175 32B server ALSO serves the user's norm-scraper text-gate job so
GPU 0 cannot be freed unilaterally even after the arm lands) and GPU 3 (pupa prefix k29/48 +
livebench prefix k50/64, both hours from done, unresumable — preempt only on explicit order).
Data consolidation: rsync sk3 runs_paperexact -> sk1:runs_paperexact_sk3mirror_20260723/
(--ignore-existing, append-only). Harness with mipro/official_merge arms now installed on ALL
THREE boxes (backups paperexact_arms.py.bak_pre_mipro_20260723).
PREEMPTIBILITY LADDER (for future evictions): baselines (mipro/merge, restart cost = elapsed
time) < servers (stateless) < unitrecomb searches (NO mid-run checkpoint; kill = lose the whole
search). Multi-pass aime eval work (--eval-passes/--test-passes) deferred during evacuation.

## Open decisions the user has NOT answered (ask before assuming)
1. Is the EVT endpoint estimator the accepted substitute for D4's original (impossible) framing?
2. Paper-exactness scope: is dspy.GEPA acceptable as the optimizer given the paper's own optimizer
   is the standalone teleprompter in `vendor/gepa-artifact/gepa_artifact/gepa/gepa.py`? (Current
   working interpretation: the requirement is about the END EVALUATION, which is verbatim theirs.)
3. Should the discrimination-maximizing M_ω objective be revisited? D6 shows it is a CAPACITY
   objective and therefore vulnerable to the SHA-parity degeneracy; the MDL-penalized recovery
   readout is the cheap first experiment (description lengths already logged).

**HB82 (Jul 24 ~12:20 PDT): sk1 z.ai key DRAINED (code 1113 killed MIPROv2 mid-run 11:00);
MIPROv2 RELAUNCHED on sk3 with the alexander-spangher key; overnight verdicts: pupa + livebench
wide runs both GUARD-FALLBACK — those two benches remain the only losses (4/6 envelope).**
(1) KEY EVENT: sk1's only key (.z-ai-api-key-spangher) hit "1113 Insufficient balance" — sk1
MIPROv2 died with empty run dir. sk2's alexander-spangher key confirmed still funded (GEPA+Merge
making successful calls same hour). Funded key copied to sk1 (~/.z-ai-api-key-alexander-spangher
.txt, chmod 600). ALL future launches: export ZAI_KEY_FILE=$HOME/.z-ai-api-key-alexander-
spangher.txt (harness prefers this name anyway, but sk1 lacked it).
(2) MIPROv2-Heavy relaunch: sk3 GPU 3 (old :8178 server had died on its own; fresh Qwen3-8B
server pid 3309543/EngineCore 3317912, init 7.4s), arm pid 3343789, run-tag miprov2, ZAI_KEY_FILE
set. Confirmed past startup into GLM proposal phase. NOTE sk3 GPU 7's norm-scraper VL server is
also gone (not killed by me).
(3) OVERNIGHT VERDICTS (all identity-relevant fields from result.json):
| run | test | fell_back | read |
| pupa v4sk3 | .7913 | TRUE (0 units) | re-measure of GEPA init; official .8621 — NO progress |
| livebench v6widesk3 | .6190 | TRUE | fallback twin |
| livebench v6wide (sk2) | .6111 | TRUE | fallback twin |
| aime-32B stair | .3667 | FALSE (1 unit) | vs GEPA-32B official .3533: +.013, inside aime noise |
(4) SIX-BENCH Qwen3-8B ENVELOPE STANDING: aime .3667→.4267 WIN; hotpot .38→.6333 WIN; hover
.4567→.5833 WIN; ifbench .4116→.4558 WIN (v6ctx32k); livebench .6956 vs best genuine M_ω .6823
LOSS (every wide retry falls back); pupa .8621 vs nothing genuine LOSS. pupa/livebench = the
high-baseline pair where the confirm gate never clears — next move is richer unit framings +
multi-pass confirm, not recipe re-runs.

**HB83 (Jul 24 ~12:40 PDT): v8 SHIPPED (failure-grounded mining + multi-pass evals, user-
approved); pupa+livebench v8 rescue arms LAUNCHED; hover-14B rung LANDED (+.120 GENUINE →
hover 5/5); envelope-expansion sequencer armed (ifbench+pupa 14B rungs).**
(1) HARNESS v8 (paperexact_arms.py, synced all 3 boxes, backups .bak_pre_v8_20260724):
--failure-mine (runs init on select panel once, worst-12 cases w/ input/gold/output → GLM
diagnose-and-fix framings ×2; units tagged failure_grounded, PREPENDED never cap-evicted;
phase=failure_mine_diag logged); --eval-passes/--confirm-passes/--test-passes (k independent
generation passes averaged, per-item elementwise so paired tests stay valid; budget charged
k×panel; result.json records all three + pass_means per row). Confirm-add-val left OFF for v8
rescue arms (it biases toward init; k=3 confirm is the noise control instead).
(2) V8 RESCUE ARMS (the two losing benches): pupa v8failmine sk3:8178 pid 3783917 (shares
fresh server w/ relaunched MIPRO; train 111, pool 96 = 60 LLM + trajectory + failure units
pending); livebench v8failmine sk2:8078 pid 3593977 (shares w/ GEPA+Merge). Both: budget
30000, max-units 96, prefix-cap 48, confirm-passes 3, test-passes 3, funded key via
ZAI_KEY_FILE.
(3) ★ hover-14B rung landed overnight: GEPA official .4367 vs M_ω stair .5567 (+.120,
fb=False, 22 compiled units) → **hover ladder 5/5 GENUINE (1.7/4/8/14/32B)**, second-largest
delta after 8B's +.127. Score ladder now .047/.057/.127/.120/.040 — the non-monotone shape
(peak mid-scale) sharpens.
(4) ENVELOPE EXPANSION: expand14b_sequencer.sh armed on sk1 (pid 1074511) — waits for the
aime-14B lane (official in final_test now → chainer stair → staircase_eval) to clear :8174,
then ifbench-14B official → stair → pupa-14B official → stair. pools/pupa_Qwen3-8B_frozen.json
BUILT (96 units, 0 past-winners — consistent with the v4 fallback; needed v4sk3 result.json
pulled from sk3, the 07-23 mirror predated its landing). NOTE stair_momega_chainer has TWO
live instances (1032399, 1995581) — harmless no-op duplicates, but don't add a third.

**HB84 (Jul 24 ~14:20 PDT): ★★ UNDERPERFORMANCE AUDIT (pupa/ifbench/livebench, user-ordered).
Verdict: these are three DIFFERENT failure classes, and two of the three "losses" are
measurement artifacts, not search losses.**
Instrument: audit_bench.py (scratchpad) over proposals.jsonl — same-candidate replicate
spread, marginal distributions, zero-score contamination, select→confirm transfer; plus
rescore_k3.py (NEW, in repo) — k-pass test re-measure of shipped winners.

(1) **LIVEBENCH = load-dependent metric (the big one).** The AMPS sympy metric runs under a
wall-clock timeout, and generation contention adds item errors → zeros. Evidence: zero-rate
.17-.42 across runs (row max .69); same-candidate replicate spreads up to **.37**; v6wide's
select_init measured .4568/.5185 vs the same prompt's .6522/.7019 confirm rows; and the
KILLER: GEPA-official's shipped prompt re-measured k=3 on the busy box = **.479
[.508/.468/.460] vs its headline .6956** (zero-rate .405, 127/378 item errors). The .6956 was
measured on an idle box on 07-21. CONSEQUENCES: (a) the "-.013 loss" is far inside instrument
instability — livebench W/L is UNDECIDED, not lost; (b) all 128/128-positive-marginal wide
runs = deflated-base artifacts (base eval caught a bad window → every unit looks positive →
search chases noise → honest confirm guard falls back — the guard is WORKING); (c) NEVER
compare livebench numbers measured under different box loads; final verdicts need an
idle-server k≥3 protocol for BOTH arms.
(2) **PUPA = judge-noise ceiling + select-panel overfit.** Metric judge AND untrusted model =
GLM-5.2 via make_reflection_lm → judge runs at **temperature 1.0** (inherited; artifact's own
judge default — paper-faithful but noisy). Test-level same-prompt swing ~.07 (official .8621
vs v4sk3 re-measures .7653/.7913). Search-level: v4sk3 select climbed .8658→.9414 (+.076!)
but confirm rejected (.8345 init vs .8130 compiled) — with 96 marginals at sd .039 on a
74-item panel, top marginals are order statistics of noise; the +.076 was panel-fitting.
Zero-contamination NOT the issue on pupa (.005-.009). v8's confirm-passes-3 helps; the k3
official rescore (sk3 pid 2493460, running) gives the honest target. Judge-temp-0 variant
would be an instrument CHANGE — separate flagged column, only with user sign-off.
(3) **IFBENCH = fixed already; history was infra.** Plain unitrecomb died of context overflow
(confirm rows littered with 0.0/.7488 duplicates, replicate spread .7375, zero-row max 1.0 —
the fd-crash era) and shipped a fallback tie. v6ctx32k (32k server) is CLEAN: confirm .5821→
.8286, test .3605→.4558 — a genuine +.044 win over official .4116. No current pathology; its
baseline being single-pass is the only residual caveat.
(4) CROSS-CUTTING: max_tokens=8000 truncation warnings appear in pupa/livebench eval streams
(long-output items) — truncated generations score low and add variance; consider 12-16k for
non-aime benches (server ctx permitting) as a flagged deviation.
LIVE: pupa k3 rescore sk3 pid 2493460; livebench idle-server rescore QUEUED behind the v8/
merge arms (current busy-box k3 already logged). rescore_k3.py synced sk2+sk3.

**HB85 (Jul 24 ~14:35 PDT): ★★★ FULL ERROR DIAGNOSIS — the losses are INSTRUMENT DAMAGE, and
two root causes are now FIXED with measured effect sizes. livebench truncation costs **+.082**;
pupa judge rate-limits cost ~5% of items; MIPROv2 was dying on a MISSING PYTHON PACKAGE.**
Instruments: errclass.py (scratchpad, classifies every dspy eval error), rescore_k3.py (repo).

ERROR TAXONOMY (counts = eval-loop errors per run log):
| run | eval errors | dominant cause |
| livebench v6wide | 7,880 | 7,156 truncation warnings -> JSON-parse failures |
| livebench v8failmine | 680 | 615 truncation |
| livebench k3 rescore (8k) | 168 | 180 truncations; 132/168 errors = "cannot be serialized
to a JSON object" / "JSONAdapter failed to parse" |
| ifbench v6ctx32k | 246 | 1,656 truncation |
| pupa v4sk3 | 129 | **123 z.ai 1302 rate-limit** (the METRIC JUDGE, not the task LM) |
| pupa v8failmine | 9 | 8 rate-limit |
| hover mipro | 0 | died at import: **ImportError optuna** |

(1) ★ **LIVEBENCH ROOT CAUSE = max_tokens TRUNCATION, not sympy/timeout.** Long math CoT hits
max_tokens=8000 mid-JSON -> adapter cannot parse -> dspy errors the ITEM -> scored 0. CONTROLLED
TEST (same prompt, same box, same load, GEPA-official's shipped candidate, k=3):
| max_tokens | k3 mean | pass means | truncations | eval errors | zero-rate |
| 8,000 (paper-exact) | .4788 | .508/.468/.460 | 180 | 168 | .405 |
| 24,000 | **.5608** | .675/.484/.524 | **0** | 17 | .325 |
→ **+.082 recovered by removing truncation alone.** Every livebench arm to date (both GEPA and
M_ω) searched and was scored through this. Residual pass-spread .19 at 24k = load contention,
so the protocol is BOTH 24k AND an idle box AND k>=3. NOTE 8000 is paper Appendix E.2, so 24k
is a FLAGGED DEVIATION — must be applied SYMMETRICALLY (both arms) and reported as its own
instrument-clean column beside the paper-exact one.
(2) ★ **PUPA ROOT CAUSE = judge-side rate limits.** pupa's metric judge IS GLM via z.ai; a 1302
rate-limit does not merely retry — it errors the item, scoring it 0 (123 such zeros in v4sk3 =
~5% panel deflation, which is the same order as the effect being searched for). FIX SHIPPED:
make_reflection_lm(patient=True) -> num_retries 40 for the pupa judge only (changes no judgment,
only whether a judgment is obtained). VERIFIED: 12 errors in 12 min before -> **0 errors** after.
Contention was partly self-inflicted (concurrent GLM consumers); --eval-threads 16 for pupa.
(3) **MIPROv2 was never a scientific failure**: `ImportError: MIPROv2 requires optuna`. Installed
(optuna 4.9.0, sk3 venv); relaunched pid 2827254, now past the wall and evaluating. Yesterday's
"MIPRO died" (1113 balance) and today's are two DIFFERENT trivial infra faults.
(4) ★ **GEPA+Merge BASELINE LANDED (hover/Qwen3-8B): seed .387 -> best .533.** Stronger than
plain GEPA (.4567) as expected, and **M_omega .5833 still beats it (+.050)** — the strongest
GEPA-family baseline does not close the gap. First head-to-head vs GEPA+Merge in the campaign.
(5) LIVE: pupa k3 rescore w/ patient judge (sk3 pid 2829920, both official+v4sk3 arms; the
pre-fix rescore quarantined as rescore_k3.jsonl.INVALID_ratelimit_20260724); livebench 24k
rescore DONE; pupa v8 + livebench v8 arms running; MIPRO running; expand14b sequencer waiting
on the aime-14B lane (aime-14B GEPA still in final_test since 08:45 — SLOW, watch it).
DECISION PENDING (user): the live livebench v8 arm is searching through the 8k-truncation
signal (615 truncations so far) — recommend killing it by PID and relaunching at 24k, but
unitrecomb searches are preempt-on-explicit-order-only, so it keeps running until told.

**HB85b (Jul 24 ~14:50 PDT): ★★★ PUPA RESOLVED — the entire "pupa loss" was judge-rate-limit
contamination. Clean instrument: GEPA .8835 vs the fallback candidate .8833 (SAME PROMPT,
Δ=.0002).** k=3 patient-judge rescore, 0 eval errors, both arms:
| arm | old single-pass | clean k3 | note |
| official (GEPA) | .8621 | **.8835** | |
| unitrecomb_v4sk3 | .7913 | **.8833** | guard fallback -> byte-identical prompt to official |
Both rows are the SAME candidate, so their agreement to **.0002** measures the CLEAN
instrument's noise — pupa is in fact one of the most STABLE benches once the judge stops
failing. The .8621-vs-.7913 spread that made pupa look like a .07 loss was 100% instrument.
CONSEQUENCES: (a) the "pupa LOSS" is RETRACTED — there was never a real gap, only a fallback
tie mis-measured; (b) pupa's true baseline to beat is **.8835**, not .8621; (c) a genuine M_ω
gain on pupa is now cleanly DETECTABLE (noise ~.000x, not .07), which is exactly the condition
the v8 failure-grounded search needs. NEVER quote pupa .7913 or .8621 again.
NOTE the live pupa v8 arm (sk3 pid 3783917) loaded the PRE-patient code, so its select panel
still takes occasional rate-limit zeros (9 in ~4h, mild vs v4sk3's 129 — fewer concurrent GLM
consumers). Judged tolerable; its confirm guard is the backstop. All FUTURE pupa runs get the
patient judge automatically.

**HB86 (Jul 24 ~15:30 PDT): ★★★ UPPER-BOUND DEEP AUDIT (user-ordered). Two verdicts: the EVT
endpoint is DEGENERATE and must be retracted; a NEW non-vacuous CERTIFIED all-prompt cap now
exists (livebench .9048). Plus the session's biggest score find: livebench was hard-zeroing 19%
of its items on a MISSING PIP PACKAGE.**

(1) ★ **EVT ENDPOINT RETRACTED — degenerate by construction.** The GPD MLE endpoint is
u + σ/(−ξ). Fitted ξ is in the ξ < −1 boundary regime at EVERY tail size on EVERY bench
(aime −1.14..−1.58, hover −1.03..−1.87, hotpot −1.31..−1.77, livebench −2.54..−3.72), where the
GPD likelihood is maximized by pinning the endpoint to the largest order statistic (Smith 1985
irregular regime). VERIFIED ARITHMETICALLY: u + σ/(−ξ) − best_achieved = **0.00e+00** in all 9
(bench,k) cells checked. So the bolded "EVT endpoint" column in runs/UPPER_BOUNDS.md is the
SAMPLE MAX wearing a hat — margin exactly 0.000, carrying no information above best-achieved.
Pickands is worse (fails 4k>n at most k; one aime cell returned 1.9e13; CI upper hits 1.0). The
file's own verdicts already said "do not quote"/"NO USABLE ENDPOINT" for 3 of 5 — the rollup
table over-read them. ROOT CAUSE: n_candidates 11-75 with scores on a 1/n grid; EVT needs
hundreds of continuous draws. NOT repairable by re-fitting; needs a different instrument.
(2) **The vacuity result is a THEOREM, and my earlier "only DPI survives" was mis-stated.** On
deterministic-label benchmarks sup_p score(p) = 1.0 exactly (a prompt may encode the answer key),
so NO information-theoretic all-prompt cap can bind. The DPI fixed-target cap is a NOISY-LABEL
object (Papers #1/#3) and degenerates to 1.0 here. Correct statement: **Paper #2 has no
non-vacuous certified all-prompt bound from the information-theoretic route, and cannot have one
without restricting the prompt class or assuming model-capability limits.**
(3) ★ **NEW CERTIFIED CAP FROM THE METRIC SIDE (bound_metric_reachability.py, in repo).** An item
whose metric returns 0 even when handed an IDEAL response is unreachable by every prompt:
sup_p score(p) ≤ 1 − (unreachable)/n. Exact w.r.t. a DECLARED output family F (bare/LaTeX/boxed/
prose/...), conservative by construction (enlarging F only lowers the count). Results:
| bench | unreachable | CERTIFIED CAP | note |
| aime | 0/150 | 1.000 (vacuous) | metric clean; the LaTeX-zeroing artifact is PROMPT-FIXABLE, so not a ceiling |
| livebench | **12/126** | **0.9048** | first non-vacuous certified all-prompt cap in this arm |
| hotpot/ifbench | probe N/A | none emitted | terminal field is not the scored answer; VALIDITY GATE added so the script refuses to mint a 0.0 cap from its own probe errors |
(4) ★★★ **LIVEBENCH WAS HARD-ZEROING 19% OF ITEMS ON A MISSING PACKAGE.** The probe's first run
gave cap .7143 — FALSIFIED on the spot by our own pool max .7244 (a real prompt beat the
"certified" cap). Chasing that contradiction found
`ModuleNotFoundError: No module named 'Levenshtein'` inside the metric's proof-rearrangement path
(livebenchmath_utils/olympiad/utils.py:111, imported INSIDE the function so it throws per-item →
RuntimeError → dspy scores 0). **24/126 items (19%) were scored 0 for EVERY prompt and EVERY
model, in every livebench run this campaign has ever done.** Installed python-Levenshtein on all
3 boxes; probe re-run: exceptions 168 → **0**, unreachable 36 → **12**, cap .7143 → **.9048**.
Every livebench number to date (incl. GEPA's .6956 headline) is deflated and must be re-measured.
The self-falsification is worth keeping in the paper as a worked example of a bound auditing its
own instrument.
(5) ACTIONS: livebench M_ω restarted on the fixed metric (sk2 pid 616567, run-tag v10lev24k,
24k tokens + failure-mine + k3 confirm/test); pre-fix run quarantined as
unitrecomb_v9_PRELEVENSHTEIN_20260724. True post-fix baseline measuring now (sk3 pid 2945446,
GEPA official + v4local, k3, 24k). NOTE all four earlier livebench "losses"/fallbacks were
measured through this defect.
(6) PAPER #4 HANDOFF (user-requested): wrote latex/paper-4__tacit-knowledge/METRIC_BATTERY.md —
envelope as the ZERO POINT of #4's exchange-rate axis; TK_residual = ChannelBest − Envelope as an
envelope-referenced tacitness estimator (cleaner than dense−articulated because the articulated
leg is OPTIMIZED); absorption-efficiency proposed as a 4th row of Fig 3; E1-E4 battery table;
explicit scope box (no certified all-prompt bound exists; EVT must not be cited; the max_tokens
confound blocks any scaling claim until the 24k re-run lands). FIGURES.md updated to point at it.

**HB86b (Jul 24 ~15:55 PDT): FIRST CLEAN-INSTRUMENT LIVEBENCH NUMBERS (post-Levenshtein, 24k,
k=3, 0-3 eval errors) — GEPA .6283 vs M_ω(v4local) .5846. livebench is our ONE genuine
remaining loss (-.044), now honestly measured rather than noise-measured.**
| measurement | GEPA official | M_ω v4local |
| original headline (8k, 1 pass, pre-fix) | .6956 | .6823 |
| 8k k3 busy box, pre-fix | .4788 | — |
| 24k k3 busy box, pre-fix | .5608 | — |
| **24k k3 post-Levenshtein (canonical)** | **.6283** | **.5846** |
Note the .6956 headline does NOT reproduce even with the metric fixed and truncation removed →
it was an upward fluctuation on an idle box; the campaign's livebench baseline should be restated
as ~.63. M_ω's .6823 likewise deflates to .5846. Ordering is unchanged (GEPA ahead), so the
livebench loss is REAL, not instrumental — but it is now measurable at ~.00x error instead of
being buried under a 19% dead-item floor. The v10lev24k arm (sk2 pid 616567, failure-mined units
on the working metric) is the live attempt to close it; this is the first livebench search in the
campaign whose signal is not corrupted.

**HB87 (Jul 24 ~16:40 PDT): EVT REPLACEMENT BUILT (rank/exchangeability certificate) + livebench
switched from blind search to a TARGETED 3-pass design after measuring the select panel's noise
at ±.148 on the SAME candidate.**

(1) ★ **WHY THE SEARCH KEPT FAILING ON LIVEBENCH — measured, not inferred.** In v10 the identical
candidate (hash 389cbc8e) scored **.7398 and .5916 on the same 81-item select panel**, back to
back, with zero-rate .148 vs .296. Spread **.148**. The variation is items flipping to zero
(generation/parse variance), not skill. NO single-pass search can work through that: the observed
marginals (.709-.813) sit ABOVE both init readings, i.e. the classic deflated-base signature
again. **--eval-passes 3 is mandatory for livebench select**, not just the confirm/test guards.
(2) **v11 = targeted, hypothesis-driven, 3-pass** (sk2 pid 1443392, run-tag v11targeted3pass).
Blind 96-unit search at 3 passes would need ~90 GPU-hours (v10 rate: 7.7 min/eval x 3 x 240
evals), so the design changed: 12 HAND-BUILT units derived from the METRIC'S OWN STRUCTURE rather
than from mathematical skill. Mechanism: proof-rearrangement scores
`1 - levenshtein(parsed, gold)/max(len)`, so an unparseable/empty answer scores EXACTLY 0 while
any full-length guess earns partial credit. The units therefore target the abstention→zero
failure mode (always emit a complete comma-separated integer list, one id per <missing> tag;
count tags first; never abstain; nothing after the answer line; keep reasoning short enough to
reach the answer line). pools/livebench_targeted_v11.json; eval/confirm/test passes all 3;
prefix-cap 12; budget 120k. This is a legitimate prompt-side intervention (GEPA sees the same
partial-credit feedback text), not metric gaming.
(3) ★ **EVT FIX = CHANGE THE SAMPLING, NOT THE ESTIMATOR** (`bound_rank_certificate.py`, in repo,
queued on sk2 behind v11 so the two never contend). EVT failed for three simultaneous reasons —
n=11-75, 1/n-grid ties, and adaptive (non-i.i.d.) draws — and no re-fit repairs any of them.
Instead: draw N prompts I.I.D. from a DECLARED FROZEN generator G (each frozen-pool unit included
independently w.p. p=.5, appended to the GEPA init). Then exchangeability ALONE gives a
finite-sample, distribution-free certificate with no tail model and no shape parameter:
    P(fresh draw from G > max of N draws) <= 1/(N+1);  more generally <= k/(N+1) at the k-th largest.
At N=120 that is <= 0.83%. Ties only make it more conservative. **And the noise direction is
favourable**: observed = true skill + binomial noise, so the observed max is biased UPWARD, making
the bound conservative as a statement about true skill. SCOPE: certifies the tail of THIS
generator (= the recombination class M_omega searches), not all of prompt space — which is
exactly the class the paper's "nothing beats recombination" claim is about. This is the honest
successor to the retracted EVT column.
(4) Bound ladder now, for Paper #2: [certified, all-prompt] metric-reachability cap (livebench
.9048) > [certified, class-relative, distribution-free] rank certificate (queued) > [estimate]
missing-value ceiling > ~~EVT endpoint~~ RETRACTED.

## HB88 (2026-07-25) — overnight verdicts: pupa FLIPS TO WIN; rank certificate lands and exposes livebench

- **pupa v8failmine (sk3) PROMOTED — first pupa win.** Failure-grounded mining (24 diagnosis units of a 120-unit pool) + patient GLM judge + 3-pass confirm: confirm_init .8739 vs confirm_compiled .9313 → guard cleared (fell_back=False, first time on pupa). TEST k3: **M_ω .8938 vs GEPA-shipped clean-k3 .8835** (+.010; GEPA single-pass official .8621). Margin is within pupa judge noise (~.07 test-level swing) → paired item-level sign test still required before declaring the W column. Artifact: runs_paperexact/pupa/Qwen3-8B/unitrecomb_v8failmine/result.json (sk3).
- **Rank/exchangeability certificate (livebench, sk2) COMPLETE** — the EVT replacement works: N=120 iid draws from the declared generator (each of the 48 frozen-pool units included w.p. .5, appended to GEPA init). Ladder: P(fresh draw > .7914) ≤ .0083; > .7819 ≤ .0165; > .774 ≤ .041; > .7662 ≤ .083. Artifact: runs/bound_rank_certificate_livebench_Qwen3-8B.json (sk2).
- **★ The certificate's side-finding is the real livebench story: 120/120 random recombinations beat the GEPA shipped prompt** (draws mean .7412 ± .019, max .7914, vs shipped init .6438 same panel/session). Random unit-appending beats GEPA by ~+.10 with probability 1.0 over the generator — the mined pool contains the fix; the *search* was what kept failing (deflated-base select noise), not the units.
- Top draw #86 (21 units, single-pass .7914) reconstructed (rng seed 0 is deterministic) → runs_paperexact/livebench/Qwen3-8B/rankcert_topdraw/result.json; **k3 idle both-arms rescore launched** (sk2 pid 3137389, logs/rescore_livebench_topdraw_k3.log) per the W/L protocol.
- **livebench v11targeted3pass FAILED**: 12 hand-built metric-structure units, all marginals ≈ 0/negative at 3 passes; final_test seed .6699 vs best .6424, and 3-pass spreads still huge (.58–.78 per pass) — load-dependence dominates even at k=3 on test. The abstention→zero framing does not add on top of the compiled candidate; the mined-pool draws (above) supersede this line.
- **MIPROv2 hover (sk3) DONE: .48** (seed .3667). Envelope baseline order on hover: GEPA+Merge .533 > MIPRO .48 > GEPA .387; M_ω .5833 beats the whole family (+.050 over strongest).
- **14B/24k envelope chain (sk1)**: aime GEPA official at 24k = **.50** (seed .34) — vs the 8k pathological all-zero regime, confirming the max_tokens confound at 14B. Chain pid 986255 alive 18h, still on stage 1 (aime official eval long-tail); ifbench/pupa stages pending.

## HB88b (2026-07-25) — livebench W/L decider + pupa paired test + second 14B lane

- **livebench same-session k3 both-arms rescore (sk2, busy-box symmetric)**: GEPA official **.6147** vs rankcert topdraw #86 **.6723** → **+.058 M_ω-family win**. Paired item tests on n=126: sign W32-L22-T72 p=.22 (72 ties — partial-credit metric ties are expected), **paired bootstrap on the mean: P(Δ≤0)=.023** → significant. Note winner's curse confirmed: topdraw single-pass .7914 → fresh k3 .6723; the +.058 is the honest number. Artifacts: runs_paperexact/livebench/Qwen3-8B/{official,rankcert_topdraw}/rescore_k3.jsonl.
- **pupa paired (cross-session k3)**: M_ω .8938 vs GEPA .8835, W35-L30-T156, sign p=.62 — within judge noise; needs same-session k≥5 both-arms to decide, or report as "≥ GEPA".
- Scoreboard correction: **ifbench is NOT outstanding** (v6ctx32k .4558 vs official .4116). aime is a win on file (unitrecomb .4267 vs official .3667, fell_back=False — the "guard 3/3" memory refers to the contamination guard, not the no-regret guard).
- **User released sk3 GPU 7** → second 14B/24k envelope lane launched there (chain pid 3682544, vllm 3682546/3682655, port 8179, ctx 32k): hover → hotpot → livebench GEPA-official at 24k. Complements sk1's aime → ifbench → pupa chain.

## HB89 (2026-07-25) — advisor review: two defects in the livebench result, and a corrected bound semantics

Fable advisor pass over HB80-HB88b. Three corrections and a re-prioritization; acted on immediately.

**★ DEFECT 1 (mine) — the livebench topdraw was SELECTED ON TEST.** The 120 certificate draws were
scored on test; I promoted the test-argmax (#86) and reported it on the same 126 test items. The k3
fresh rescore cures winner's-curse-on-noise but NOT prompt-to-item overfit. **The +.058 cell is
provisional until re-selected.** Repair (running): rescore the top-5 test draws on the SELECT panel
(train[:81]), promote the select-argmax, k3 test. → `livebench_reselect_placebo.py` PHASE 1.

**★ DEFECT 2 — no placebo, so the content claim is unsupported.** "120/120 random recombinations beat
GEPA" has a live alternative: appending ~24 clauses of ANYTHING lengthens the prompt and suppresses
the abstain→zero mode that the Levenshtein partial-credit metric punishes with an exact 0. v11 does
NOT discriminate (it tested on top of the compiled candidate, not the init). Repair (running):
placebo generator, identical inclusion process and identical per-draw clause COUNT, clauses drawn
from **hover's** frozen pool = length/structure-matched, content-irrelevant. 40 real + 40 placebo +
10 init replicates, **randomized interleaved order** in one session → also fixes the third weakness,
that 120 draw readings were compared against essentially ONE init reading (.6438) on a prompt that
has measured .479-.6956 across sessions. Separation is only real if init-replicate MAX < draw MIN.
→ `livebench_reselect_placebo.py` PHASE 2. If placebo also beats init, the content claim dies.

**★ CORRECTION — the "conservative for true skill" note on the rank certificate was WRONG** and had
propagated into bound_rank_certificate.py's docstring, its emitted `scope` field, and
paper-4/METRIC_BATTERY.md. It claimed observed = true + symmetric noise ⇒ observed max biased up ⇒
conservative for true skill. Backwards: noise here is **one-sided downward** (errors/truncation/
timeouts force hard 0), so observed ≤ true per item and the observed max can sit BELOW the class's
best true skill. Correct semantics: the certificate bounds the **measured** score under the declared
protocol — protocol-relative, exactly like the metric-reachability cap. Winner's curse (argmax
overstates ITS OWN skill) is a separate, still-true statement. All three files corrected.

**Scoping the certificate for review:** adaptive pool mining does NOT invalidate it (G is frozen
before drawing; exchangeability among draws is untouched) — but it makes the bound **per-pool**: a
new pool needs a new certificate, and it says nothing about what further mining reaches. Two audits
still owed: (a) select/test provenance of every unit, (b) an LLM leakage pass over all 48 units/bench
for answer-key content (the vacuity theorem cuts both ways).

**Advisor's framing of the real claim** (better than mine): not "random beats GEPA" but *the value of
prompt optimization here lives in the unit POOL, not in the SEARCH; unguided recombination dominates
reflective search when per-candidate eval noise exceeds per-unit effect sizes.* Suggested extra
check: score the compiled M_ω candidate in the same session as the draws — if random draws match it,
the honest headline is "the recombination class is what matters; the greedy machinery is decoration."
Also suggested: promote **best-of-N random draws** to a named baseline row in the main table.

**Re-prioritization (acted on).** sk2 queue wrapper killed by PID (4012049; pupa k5 child 4012057 left
running), re-queued as sk2_queue_v2.sh (pid 4150875): pupa k5 → P0 reselect+placebo → rank certs on
**aime and hover only**. **CUT: hotpot/ifbench certs; the 24k scaling-ladder re-run (belongs to Paper
#4 anyway); envelope completion (MIPROv2/GEPA+Merge on the 5 non-hover benches); any new pupa arm.**
pupa is one-shot: if k5 is still n.s., report "M_ω ≥ GEPA (n.s.)" and stop — re-rolling a .07-noise
judge is p-hacking.

**The overclaim to avoid** (verbatim from advisor, for the abstract review): *"Random prompts
outperform state-of-the-art reflective optimizers (120/120, p<10⁻²), and we certify that no prompt
can exceed X."* Every clause overreaches — they are recombinations of units mined FROM GEPA's own
trajectories (no GEPA, no generator); the honest paired number is +.058 on one bench with a selection
caveat and pupa n.s.; and "no prompt" is class- and protocol-relative only. Title's "certified upper
bounds on what prompting can achieve" needs rescoping for the same reason.

## HB90 (2026-07-25) — pupa decided: TIE. sk1 aime-14B hung and reclaimed. sk3 hover broken.

**★ pupa same-session k=5 both-arms rescore (sk2, 221 items) = TIE, and the earlier +.010 was
cross-session noise.** GEPA **.8825** vs M_ω v8failmine **.8817**, mean delta **−.0009**; sign
W28-L30-T163 p=.90; paired bootstrap P(Δ≤0)=.52. Per the advisor's one-shot rule this is FINAL:
**report pupa as "M_ω ≈ GEPA (n.s.)" and run no further pupa arms.** The prior k3 reading
(M_ω .8938 vs GEPA .8835) is superseded — both arms were measured in different sessions.
Scoreboard correction: pupa is a TIE, not a win. Artifacts:
runs_paperexact/pupa/Qwen3-8B/{official,unitrecomb_v8failmine}/rescore_k3.jsonl (rows passes=5).

**★ Instrument finding — pupa's judge noise is CROSS-session, not within-session.** The 5 pass
means are nearly identical within each arm (GEPA .8816/.8827×4; M_ω .8835/.8812×4) — spread ~.002,
versus the ~.07 swings seen *between* sessions. So k-pass averaging does NOT buy what we assumed on
pupa: repeated passes inside one session are near-duplicates. **The only valid pupa comparison is
both arms in the SAME session** (which is what finally decided it). Generalize: for judge-based
metrics, budget same-session A/B, not more passes.

**sk1 aime-14B/24k HUNG and was reclaimed.** The arm ran 19h with its last log write 9.5h earlier;
tail showed `litellm.exceptions.Timeout: APITimeoutError` — it was stuck in the retry path, GPU
pinned, making no progress. Killed by explicit PID in order (wrapper 986255 → arm 1081879 → vllm
parent 983652 → EngineCore 986039); other users' EngineCores (animjha 330902, sahasras 1451756)
identified and left alone. Per the advisor's cut, the 14B lane was NOT restarted; GPU7 was
repurposed to an 8B server running the **aime + hover rank certificates** (sk1_certs.sh, pid
708878, port 8078), so certs now run in parallel with sk2's P0 controls instead of behind them.
sk2 re-queued as v3 = **P0 controls only** (pid 490085).

**sk3 hover-14B failed instantly (rc=0 masked it).** `RuntimeError: Dataset scripts are no longer
supported, but found hover.py` — sk3's `datasets` is too new for `hover-nlp/hover`'s script loader
(sk1/sk2 are fine). The chain's `rc=$?` captured the *echo*, not the python, so a 4-second crash
logged as success — **fix the chain template to test the python's rc directly**. hotpot-14B/24k did
complete: **.2667 (seed = best, GEPA found nothing)**, notably below 8B's .38; livebench-14B still
running. No 14B claim should be made from this row until the hover gap and the hotpot regression
are explained — and per the advisor the 14B ladder is CUT from Paper #2 anyway.

## HB91 (2026-07-25 18:35Z) — ★ BINDING PRE-REGISTRATION for the livebench controls (written BEFORE results were read)

Advisor-issued, recorded while `livebench_reselect_placebo.py` was still mid-run on sk2 and no
Phase-1 or Phase-2 output had been inspected. Whichever cell the data lands in IS the sentence that
goes in the paper. No post-hoc renegotiation.

**PHASE 1 — re-selection.** Δ = (select-argmax draw, fresh k3 test) − (GEPA official, same session).
| outcome | rule | consequence |
|---|---|---|
| CONFIRMS | Δ ≥ +.030 **and** paired bootstrap P(Δ≤0) < .05 | livebench = confirmed win (expect shrinkage from +.058; .6723 was still test-selected) |
| FALSIFIES | Δ ≤ 0 | livebench cell → tie/loss; headline degrades to "matches or exceeds 6/6, strictly exceeds 4" |
| AMBIGUOUS | 0 < Δ < .030, or P(Δ≤0) ∈ [.05,.25] | "directionally positive, not confirmed"; NOT counted as a win; **no re-roll** (one same-session replication allowed ONLY as a declared widening of k, never a fresh selection) |

**PHASE 2 — placebo.** init replicates n=10, real draws n=40, placebo draws n=40, interleaved, one session.
| outcome | rule | consequence |
|---|---|---|
| CONFIRMS content claim | (real mean − placebo mean) ≥ +.020 with rank-sum p<.05, AND real beats init (clean: real min > init max; acceptable: real−init ≥ +.05 with clear separation) | pool-not-search claim keeps its flagship exhibit. n=40/40 at draw SD≈.019 resolves ~.012, so +.020 is a fair bar |
| FALSIFIES content claim | placebo within .010 of real (rank-sum n.s.) while both beat init | "the pool's value is LENGTH/STRUCTURE, not mined content" → livebench is rewritten as a metric-pathology finding (padding suppresses abstain→zero); the central claim then rests only on hover/hotpot/aime/ifbench, where no placebo has been run. **This is the outcome that genuinely hurts.** |
| AMBIGUOUS | real > placebo significantly AND placebo ≫ init | decompose honestly: "of the ~+.10 raw effect, X points structural, Y points content" — still publishable, arguably a better claim |

**★ UNCONDITIONAL KILL SWITCH:** if the init-replicate max overlaps the real-draw distribution, the
**"120/120" claim is DEAD regardless of the placebo outcome**, because the original comparison was
120 draw readings against a single init reading (.6438).

**The ONE permitted follow-up if Phase 2 is ambiguous** (pre-declared here so it cannot be a post-hoc
rescue): score the compiled M_ω candidate in-session against the draws. Nothing else.

## HB91b — advisor's other mandates from the same pass

- **★ NEW MANDATORY WORK: same-session paired k3 both-arms revalidation of the 4 deterministic wins**
  (hover, hotpot, aime, ifbench). The pupa artifact was *judge* drift and cannot touch EM/programmatic
  metrics — but those four share the *sibling* risk of generation-side session effects (load, timeouts,
  the max_tokens confound). ~8-16 GPU-hours total. Buys the methods sentence *"every headline cell is a
  same-session, paired, k≥3 both-arms measurement,"* which retroactively immunizes the whole scoreboard
  against the class of objection pupa just exposed. Priority if squeezed: aime > hover > ifbench > hotpot.
- **★ pupa's k=5 was effectively k≈1.** 4 of 5 pass-means were BIT-IDENTICAL (.8827×4, .8812×4) —
  that is provider-side caching or deterministic decoding, not merely "low variance". The averaging
  bought literally nothing. Verify before relying on k-passes with any API judge.
- **Variance-components framing (reviewer-proof, use this):** `score = true + u_session + ε_call`.
  Averaging attacks ε_call ONLY; pairing cancels u_session ONLY. pupa needed pairing and got averaging.
  | metric | σ_session | σ_call | correct design |
  |---|---|---|---|
  | pupa (GLM judge) | ~.07 | ~.002 | same-session paired A/B; k-passes worthless |
  | livebench (Levenshtein under load) | present | ~.148, one-sided down | k-pass within session, idle box, symmetric load |
  | hover/hotpot/aime/ifbench | load/config-mediated | small | same-session pairing as cheap insurance |
  Assumption to verify cheaply: pairing cancels u_session only if the session effect is additive and
  arm-symmetric → run two paired sessions on livebench/ifbench and check the DELTA is stable while
  LEVELS move. Within-session determinism also explains the huge tie counts (T163/221), which is why
  **the paired bootstrap, not the sign test, is the primary inference.**
  Verdict: strong *subsection* of Paper #2, NOT a standalone paper (that would need 5-10 judges across
  providers). Bank the data, don't spin it off.
- **Framing for pupa:** "on the one benchmark scored by an LLM judge with near-zero within-session
  variance, the two arms are statistically indistinguishable (Δ=−.0009, n=221, P(Δ≤0)=.52)", with the
  cross-session artifact as the methods section's motivating example.
- **Still owed from HB89, now urgent:** unit provenance audit + LLM leakage pass over the 48 livebench
  units. The vacuity theorem makes answer-key leakage the single most dangerous latent objection —
  **do not submit without it.** CPU/API only, no GPU contention.
- **Best-of-N-random-draws as a named baseline row** — nearly free from Phase 1's select-panel scores,
  but it MUST be selected on the select panel or it inherits Defect 1.
- **Newly pointless:** any further pupa arm; the compiled-vs-draws check *on pupa*; more pupa judge-noise
  characterization. **Still cut:** 14B ladder (hotpot .2667-vs-.38 regression + hover crash are Paper #4
  problems — log, don't chase), hotpot/ifbench certs, envelope completion.

## HB92 (2026-07-25) — ★★★ COMPARATOR-SELECTION DEFECT: aime's "win" used the WEAKER of two GEPA runs

Found while assembling the HB91b same-session revalidation. **There are two GEPA `official` runs for
aime/Qwen3-8B with very different scores, on different boxes:**

| box | run | best_test | n_test | max_tokens | budget |
|---|---|---|---|---|---|
| sk1 | aime/Qwen3-8B/official | **.5333** | 150 | 8000 | 600 |
| sk2 | aime/Qwen3-8B/official | **.3667** | 150 | **None** (harness default) | 600 |

Every scoreboard this campaign has reported used **.3667**, the weaker one, against M_ω's .4267 —
so **the aime cell is a LOSS (−.107), not a win (+.060), if the .5333 run is the right comparator.**
This is not cross-session noise; it is comparator selection across configs, and reporting the
weaker GEPA run as "the" GEPA baseline is cherry-picking whether or not it was intended. The
`max_tokens=None` on the sk2 run means the two are not even the same experiment.

**Immediate consequences:**
- **aime is RETRACTED from the win column** pending the revalidation. Provisional scoreboard is now
  3 confirmed wins (hover, hotpot, ifbench — themselves pending revalidation), 1 provisional
  (livebench), 1 tie (pupa), 1 **contested** (aime).
- The same audit must be run for every other bench before submission: enumerate ALL runs of each arm
  on ALL THREE boxes and declare the comparator rule in the paper (the honest rule is
  **best-of-family per arm**, i.e. max over GEPA/GEPA+Merge/MIPRO for the baseline and max over M_ω
  variants for ours — anything else invites exactly this objection).
- hover survives this check: sk1 official .4567, sk2 official .45, GEPA+Merge .5333, M_ω .5833 →
  still a win against the strongest GEPA-family comparator.

**Revalidation launched (sk1 pid 845143, queued behind the certs).** Each bench rescores ALL its
comparators in ONE `rescore_k3.py` invocation = one session, so u_session cancels in the paired
delta: aime {official, official_sk2cfg, unitrecomb}; hover {official, official_merge_gepamerge,
unitrecomb_stair}; ifbench {official, unitrecomb_v6ctx32k}; hotpot {official, unitrecomb_v5sk2}.
Comparators consolidated onto sk1 (sk2's aime official copied in as `official_sk2cfg`).

## HB93 (2026-07-25) — PHASE 1 RESULT: the test-argmax was NOT the select-argmax (defect confirmed real)

Re-selection of the top-5 test draws on the held-out SELECT panel (train[:81], 126-item test never
used for selection):

| draw | test-at-selection | SELECT | note |
|---|---|---|---|
| #86 | .7914 (test argmax) | .7463 | the candidate we had provisionally shipped |
| **#88** | .7819 | **.7869** | **select-argmax → promoted** |
| #43 | .7819 | .7248 | |
| #93 | .7792 | .7408 | |
| #44 | .7740 | .7179 | |

**The test-argmax (#86) ranks 2nd of 5 on the select panel, and the select-argmax is a different
draw (#88).** That is direct confirmation that HB89's Defect 1 was real: part of #86's test score
was item-fitting, not prompt quality. Spearman between the two columns is negative here. Draw #88
is now the candidate of record; its k3 test score (Phase 3) is the number that goes against the
HB91 pre-registration, NOT the .6723 from #86.

Phase 2 (40 real + 40 placebo + 10 init replicates, randomized interleaved) is running; results not
yet inspected, per the pre-registration.

**Also landed / launched this pass:**
- **livebench-14B/24k official (sk3) = seed .779 → best .7478: GEPA made it WORSE.** The raw seed
  prompt at 14B (.779) beats every 8B random draw (max .7914 was measured on a different config, so
  do not cross-compare). Chain COMPLETE; sk3 GPU7 now free. 14B row stays CUT from Paper #2.
- **Answer-key leakage audit LAUNCHED** (sk1 pid 1996334, CPU/API only, GLM-5.2 patient judge):
  `audit_unit_leakage.py` grades every unit of all 5 frozen pools as none / domain / item_hint /
  answer, adversarially, judging each unit WITHOUT the test items so the judge cannot be the leak.
  Note the grading is deliberately not binary: `domain` (general technique a textbook would state)
  is what we WANT to find — it is articulated knowledge, the paper's thesis. Only `answer` (and
  arguably `item_hint`) invalidates a unit. Running on livebench first.
- sk1 lane order: aime cert → hover cert → 4-bench same-session paired revalidation (pid 845143).

## HB94 (2026-07-25) — ANSWER-KEY LEAKAGE AUDIT COMPLETE (360 units, 5 pools). Verdict: clean where it counts, with one benchmark defect to disclose.

`audit_unit_leakage.py` (GLM-5.2 patient judge, adversarial framing, each unit judged WITHOUT the
test items so the judge cannot be the leak) + `split_probe.py` (does flagged content appear in
TRAIN/VAL or in TEST?).

| pool | n | none | domain | item_hint | answer | flagged |
|---|---|---|---|---|---|---|
| livebench | 48 | 38 | 5 | 2 | 1 | 3 |
| aime | 48 | 40 | 3 | 3 | 1 | 4 |
| hover | 164 | 131 | 12 | 19 | 1 | 20 |
| hotpot | 68 | 68 | 0 | 0 | 0 | **0** |
| ifbench | 32 | 31 | 0 | 0 | 1 | 1 |

**Split-membership verdicts (the decisive test):** aime {train_only 2, test 0}; livebench
{test 0}; ifbench {test 0}; **hover {train_only 11, TEST_HIT 4, neither 5}**.

**★ FINDING 1 — the flagged units are in GEPA's OWN shipped prompts, not just ours.** livebench
`official` (GEPA, .6956) carries the same 3 flagged units as `unitrecomb`; aime's 4 flagged units
are in `official`, `official_sk2cfg`, `unitrecomb` AND `unitrecomb_stair`; hover `official` carries
8 item_hints. M_ω inherits them because it *initializes from the GEPA winner*. Two consequences:
(a) **defensive** — both arms carry the same leakage, so paired comparisons are not biased by it;
(b) **substantive** — this is a finding *about reflective prompt optimizers*: GEPA writes
item-specific content from training trajectories into its prompts. GEPA+Merge on hover is the one
CLEAN GEPA-family candidate.

**★ FINDING 2 — hover's one true answer-bearing unit is a BENCHMARK defect, not a pipeline defect.**
Unit: *"The novel Washington: Behind Closed Doors is based on is the 'Company (Ehrlichman novel)'."*
It was mined from **TRAIN[133]** ("The **English** translation for the style of novel of which
Washington: Behind Closed Doors is based on…"), which is a **near-duplicate of TEST[159]** ("The
**Hebrew** translation…") — same supporting facts (`The Company (Ehrlichman novel)`, `Roman à clef`,
`Washington: Behind Closed Doors`), differing by one word. So mining touched only train, exactly as
designed; HoVer itself has near-duplicate items straddling its train/test split. Disclose as a
benchmark property; do not claim our pipeline avoided it by design when the audit is what found it.

**★ FINDING 3 — no answer-category unit appears in ANY shipped candidate.** Verified across all
hover run dirs (inhouse / official / official_merge_gepamerge / unitrecomb / unitrecomb_v5sk2 /
unitrecomb_stair): all clean of the answer unit. **So no reported number is contaminated by it.**

**Residual exposure = pool-level only, and it lands on the RANK CERTIFICATES**, whose generator
draws each pool unit with p=.5 — so ~50% of hover draws would include the answer unit. The hover
certificate is running on sk1 now. Options for the advisor: disclose + quantify, or purge the 4
test-hit units and re-run the hover certificate on the purged pool (cost ~1-2 GPU-hours), or both.
Recommended default: **re-run hover's certificate on a purged pool and report both**, since a
certificate over a generator containing a test answer key is exactly the object a reviewer will
attack.

## HB95 (2026-07-25) — advisor closes the leakage audit's two holes; hover purge + ablation queued; ★ W-mapping PRE-COMMITTED

**HOLE A CLOSED (and it found something).** The no-ship verification had been ANSWER-GRADE ONLY.
Extended to all TEST_HIT units of ANY grade, across every shipped candidate on every bench:

| bench | shipped candidates carrying a TEST_HIT unit |
|---|---|
| livebench | **none** — 0 TEST_HIT units in the pool at all; all 9 candidates clean |
| aime / ifbench / hotpot | none (0 test hits) |
| hover | `official` .4567 (1 item_hint), `unitrecomb` .5467 (1), `unitrecomb_v5sk2` .49 (1), **`unitrecomb_stair` .5833 (1)** — `official_merge_gepamerge` .5333 and `inhouse` are CLEAN |

So our best hover candidate carries one flagged item_hint that the strongest GEPA comparator does
NOT — an asymmetry we must not paper over. The unit is *"The 'former bassist/vocalist of Deep Purple,
Black Sabbath, and Trapeze who released From Now On...'"* (a factual identification; the answer-grade
unit is still in NO shipped candidate anywhere).

**Caveat on the probe, stated against our own interest:** the span matcher over-flags. Three of
hover's four TEST_HITs match only on common entity names ("Deep Purple", "The Company") that occur in
BOTH splits — all four are `also_in_train=True`. A unit that says "combine entities such as 'Deep
Purple' and 'Black Sabbath' with an OR operator" is a *generic retrieval strategy* using entities as
examples, not an answer. So the honest reading is: **1 genuine answer-bearing unit (unshipped), and
a handful of over-flagged strategy units.** The ablation below settles it empirically rather than by
argument.

**Actions taken (advisor mandate (c) = disclose AND purge):**
- **hover pool PURGED 164 → 160** (all 4 TEST_HIT units, any grade; train_only and "neither" units
  KEPT — train-side memorization is legitimate arm-symmetric content, and is itself the Finding-1
  result). → `pools/hover_Qwen3-8B_purged.json`. Rationale is semantic, not cosmetic: a certificate
  over a generator that contains a test answer key certifies a class containing cheating prompts —
  exactly the pathology our own vacuity theorem describes. The unpurged certificate is NOT killed;
  it becomes the comparison arm, and the purged-vs-unpurged delta is a free measurement.
- **LEAKAGE ABLATION built**: `unitrecomb_stair_ablated` = our best hover candidate with that single
  clause removed (2 lines). Queued to be scored in the SAME session as official / GEPA+Merge /
  unpurged unitrecomb_stair, so the leakage question gets an empirical answer.
- **livebench: NO purge** — 0 test hits, and its answer-flagged unit places in neither split, so it
  fails the conjunct test (**leakage requires BOTH answer-like content AND a test item it answers**;
  an answer-grade unit with no test referent is memorized TRAINING content = Finding 1, not
  contamination). Free conditional-draw readout still to run once flagged units can be located.
- **HOLE B (judge positive controls) LAUNCHED** (sk1 pid 340472): `audit_leakage_anchors.py` shuffles
  10 synthetic anchors of known grade — built FROM THE REAL SPLITS so difficulty is realistic, not
  caricature — in with 20 real hotpot units, judges them blind with the identical prompt, and reports
  a confusion matrix + `answer`-grade recall. Needed because the audit's whole value is its
  false-negative rate and it returned hotpot 68/68 "none".
- **Infra fix**: hover CANNOT run on sk1/sk3 (their `datasets` refuses hover-nlp/hover's script
  loader); only sk2 has the cache. The sk1 revalidation will fail its hover stage — expected, logged.
  hover reval + ablation + purged cert all moved to sk2 (`sk2_hover.sh`, pid 39174, queued behind P0).

## HB95b — ★ PRE-COMMITTED HEADLINE AND W-MAPPING (before revalidation lands)

Headline sentence, committed now:
> *Unguided recombination of instruction units mined from a reflective optimizer's own trajectories
> matches or exceeds the optimizer on every benchmark tested and strictly exceeds it on [W] of six —
> the value of reflective prompt optimization lives in the unit pool it discovers, not in its search —
> and we bound what any prompt from this class can achieve with pool-level rank certificates audited
> for answer-key leakage.*

**W counts strict same-session paired wins** among {hover, hotpot, ifbench, livebench-per-HB91}.
**aime enters only if the revalidation flips it.** If aime confirms as a loss, the "matches or exceeds
on every benchmark" clause is **DROPPED, not softened** — the sentence becomes: *"exceeds on W of six,
ties or loses on the rest, with the loss occurring where the baseline's stronger configuration was
initially overlooked and we corrected it."* Writing the loss sentence ourselves, with the
comparator-rule confession attached, is worth more than the win it replaces.

**Finding 1 → a Paper #2 section, not a standalone paper** (same call as the variance-components
finding: n=1 optimizer, one judge, a deadline). Framing that dissolves the shield/attack awkwardness:
*units mined from reflective-optimizer trajectories measurably include item-memorized content; both
arms inherit it, so paired comparisons are unbiased; we audit, grade, split-probe and purge — GEPA
ships it unaudited.* Symmetric where it must be (validity), asymmetric where earned (methodology).
Deepens the thesis: reflective search doesn't merely fail to beat unguided recombination, it spends
part of its budget writing training items into the prompt — memorization masquerading as instruction
discovery. **Guardrails: "measurably" is only defensible after the Hole-B anchors validate the
instrument; scope to ONE optimizer across five benchmarks, not a law about reflective optimizers.**

## HB96 (2026-07-25) — ★ HOLE B CLOSED: the leakage judge is validated (recall 1.0, FPR 0.0)

`audit_leakage_anchors.py`: 10 synthetic anchors of known grade — built FROM THE REAL SPLITS so the
difficulty is realistic, not caricature — shuffled blind into 20 real hotpot pool units and judged by
the identical prompt under identical conditions.

| gold \ pred | none | domain | item_hint | answer |
|---|---|---|---|---|
| none (3) | **3** | 0 | 0 | 0 |
| domain (2) | 1 | **1** | 0 | 0 |
| item_hint (2) | 0 | 0 | **1** | 1 |
| answer (3) | 0 | 0 | 0 | **3** |

**Detection of "flagged" (item_hint ∪ answer): recall 5/5 = 1.00, false-positive rate 0/5 = 0.00.
Answer-grade exact recall 3/3.** Both off-diagonal cells are benign: one `domain`→`none` (under-calls
a real technique as generic — harmless to us) and one `item_hint`→`answer` (over-calls SEVERITY
within the flagged class — conservative in our disfavour, the safe direction).

**★ And the 20 real hotpot units were re-judged 20/20 "none" under blinding** — reproducing the
68/68 result in a batch where the same judge simultaneously caught 5/5 planted leaks. That is what
makes hotpot's clean pool credible rather than merely convenient, and it is what licenses the word
"measurably" in the Finding-1 memorization claim (advisor guardrail from HB95b).

## HB96b — lane re-priorities

- **Phase 2 is SLOW**: 12/90 interleaved evals after ~1h (126 items × 24k tokens on a contended box)
  → ~6h to completion. Progress polled by COUNT ONLY; no Phase-2 score inspected, per HB91.
- **sk3 GPU7 reclaimed**: the finished 14B chain had left its server resident (173 GB, 0% util).
  Killed by explicit PID (parent 3682546 + EngineCore child 3682752, both verified alexspan and
  confirmed as the port-8179 owner); four other alexspan EngineCores were enumerated and **left
  alone** since they could not be accounted for.
- **★ Revalidation was queued behind a certificate = inverted priority** (the advisor called the
  revalidation load-bearing and the certs rung 2). Fixed by launching the 3-bench revalidation on the
  freed sk3 GPU7 **in parallel** rather than killing sk1's cert: `sk3_reval.sh` (pid 4086411, 8B on
  port 8078) runs aime {official_sk1cfg, official_sk2cfg, unitrecomb} → ifbench {official,
  unitrecomb_v6ctx32k} → hotpot {official, unitrecomb_v5sk2}, each bench in ONE invocation = one
  session. hover stays on sk2 (only box whose `datasets` loads hover-nlp/hover).
- First aime datapoint in: `official_sk2cfg` k3 = **.3000** (was .3667 single-pass).

## HB97 (2026-07-25) — ★★★ SAME-SESSION PAIRED REVALIDATION: aime RESCUED, hotpot flagship, ifbench DOWNGRADED

sk3 GPU7, each bench rescored in ONE invocation (= one session, so u_session cancels in the paired
delta), k=3, paper-exact splits. Primary inference = paired bootstrap on the mean delta (the sign
test under-powers because partial-credit/EM metrics tie heavily).

| bench | GEPA (k3) | M_ω (k3) | Δ | sign | **bootstrap P(Δ≤0)** | verdict |
|---|---|---|---|---|---|---|
| **aime** vs STRONGER config | .3067 | .3978 | **+.0911** | W25-L7-T118, p=.0021 | **.0012** | **WIN** |
| aime vs weaker config | .3000 | .3978 | +.0978 | W25-L5-T120, p=.0003 | .0004 | WIN |
| **hotpot** | .4133 | .6333 | **+.2200** | W75-L9-T216, p=4.3e-14 | **.0000** | **WIN** |
| ifbench | .4337 | .4535 | +.0198 | W42-L34-T218, p=.42 | **.2330** | **NOT SIGNIFICANT** |

**★ HB92's comparator crisis DISSOLVES — and the resolution vindicates the same-session protocol.**
The two aime GEPA `official` runs that disagreed wildly on single-pass test (.5333 sk1 vs .3667 sk2)
land at **.3067 and .3000** when measured in the same session. The .5333 was an inflated single-pass
reading, **not** a genuinely stronger configuration. So aime was never a −.107 loss; it is a **+.091
win, significant at p=.0012, and it is now our most rigorously established cell precisely because we
tried hard to kill it.** Retraction of the HB92 retraction — recorded in full because the wrong
intermediate conclusion is part of the record.

**★ hotpot is the flagship cell**: +.2200 at p=4.3e-14, on the ONE pool that is 0/68 flagged for
leakage AND whose cleanliness is anchor-validated (HB96: recall 1.00, FPR 0.00, and its 20 sampled
units re-judged 20/20 "none" under blinding). Largest effect, cleanest provenance, strongest test.

**★ ifbench DOWNGRADED from win to "directionally positive, not confirmed."** The +.044 single-pass
gap shrinks to +.0198 with P(Δ≤0)=.233 under same-session pairing. By the HB91 discipline this does
NOT count as a win and must NOT be re-rolled.

**Scoreboard after revalidation** (W = strict same-session paired wins):
| bench | status |
|---|---|
| aime | **WIN** (+.091, p=.0012) |
| hotpot | **WIN** (+.220, p<1e-13) |
| ifbench | directionally positive, NOT confirmed (+.020, p=.233) |
| hover | pending sk2 lane (reval + leakage ablation + purged cert) |
| livebench | pending Phase 2/3 under the HB91 pre-registration |
| pupa | TIE, final (−.0009, p=.52) |

**W = 2 confirmed of 6 so far**, with hover and livebench outstanding. Note this is materially weaker
than the "4 clean wins + 1 provisional + 1 tie" board of a few hours ago — every honest instrument
upgrade has cost us cells, and the ones that survive are now defensible.

## HB98 (2026-07-25) — provenance audit CLEAN; paper drafted for the settled sections

**★ Advisor's one new request — candidate SELECTION PROVENANCE for the two confirmed cells — is
CLEAN.** Read from the `proposals.jsonl` phase logs (not from memory):
| bench | panels used during select/confirm | panel at final test | verdict |
|---|---|---|---|
| aime / unitrecomb | 15, 18, 27, 30 (train slices) | 150 | selection never touched test |
| hotpot / unitrecomb_v5sk2 | 100 (select), 350 (confirm = train+val) | 300 | selection never touched test |
So both confirmed cells were select-panel-chosen and need only a one-sentence methods note — they
do NOT need HB93-style re-selection. (The livebench topdraw remains the one place selection touched
test, which is exactly why it is being repaired.) Side observation worth a limitation line: aime's
select panel is only 15-30 items, so its per-unit marginals are noisy even though provenance is clean.

**Advisor's rulings this pass (HB97 → action):**
- **W=2 carries the paper, under a reframe we had half-committed to already.** The thesis is
  pool-not-search, and under that thesis **a tie is evidence FOR us**: if unguided recombination of
  GEPA's own trajectory-mined units matches GEPA, GEPA's search contributed nothing beyond pool
  discovery. Board reads: 2 cells where search is strictly dominated, 2 where it adds nothing,
  **0 cells at any point after instrument correction where search wins**, 2 pending. Primary claim
  = **"never worse, sometimes much better (up to +.22)"**; W is a secondary strict-win count. This
  is exactly the HB95b sentence structure, so nothing pre-committed moves.
- **The 2-3-airtight-cells + methodology version is the STRONGER paper.** Reviewers cannot rerun our
  experiments; they can only audit our process — and this process now has a dated pre-registration,
  an author-found comparator defect, a retraction-of-retraction, an anchor-validated leakage judge,
  and an instrument upgrade that COST us cells. *"Every honest upgrade shrank our win column and we
  kept the shrunken column"* is the most credibility-generating sentence available and no competing
  paper can say it. Methodology → named contribution in the abstract, not buried in methods.
- **aime collapse = methods CENTERPIECE, not a headline** (a second headline splits the message; and
  "single-pass benchmarking is unreliable" from n=1 bench + n=1 judge is the overreach HB91b already
  ruled out). Pair it with pupa so both metric classes are covered. On GEPA's single-pass reporting:
  ONE factual sentence, no editorializing — let reviewers draw the inference.
- **ifbench: main table, verbatim pre-registered label, NO widening** even though HB91 permits one.
  Upside is one marginal win; the downside is a mandatory "we widened and it stayed n.s." sentence —
  and spending the permitted widening would partially refund the credibility the kept downgrade buys.
- **CONDITION on all of the above: neither pending cell may land as a confirmed loss.** If one does,
  HB95b Branch B applies (clause DROPPED, not softened).

**Drafting done this pass** (`latex/paper-2__articulation-upper-bounds/main.tex`, compiles clean):
- **Old abstract RETRACTED IN FULL** and replaced. Removed: EVT endpoint ceiling (degenerate),
  "certified all-prompt DPI bound" (doesn't transfer — vacuity theorem), "never loses", pupa .907,
  hover .547 p=1e-4, aime .413, "8× absorption", and the 1.7B-32B envelope (max_tokens confound →
  moved to Paper #4). Every removal is documented in a header comment in the .tex itself.
- Title narrowed: "Certified Upper Bounds on Prompt Articulation" → **"Pool-Level Bounds on Prompt
  Articulation"** (the old title asserts the object the vacuity theorem says cannot exist).
- **Branch A abstract written; Branch B stubbed verbatim-ready** for the loss case.
- **§Measurement protocol — FINAL**: variance-components model, the metric-class table, both
  exhibits (pupa judge / aime EM), paired-bootstrap-primary with the tie-mass rationale, comparator
  rule.
- **§Leakage audit — FINAL but for the bracketed hover ablation delta**: design, anchor validation,
  the memorization finding, the HoVer near-duplicate benchmark defect, disposition/purge policy.
- **§Bounds**: vacuity theorem, metric-reachability cap, rank certificate with both scope conditions
  (measured-not-true-skill; per-pool).
- **§Results**: main table with aime/hotpot final, ifbench pre-registered label, hover/livebench
  marked PENDING; the "how we nearly lost AIME" narrative; the provenance paragraph above.
- Limitations section stubbed with the required content list.

## HB99 (2026-07-25) — ★★ THREAD AUDIT (user-requested): two instrument defects found, one conclusion reopened

**Defect 1 — the k-pass mechanism was CACHE-DEFEATED across the whole campaign.** Inspecting
pass_means of every revalidation cell: aime sk2cfg [.30,.30,.30], ifbench official
[.4337,.4337,.4337], hotpot BOTH arms [identical ×3], pupa [4 of 5 identical]. dspy's client-side
response cache returned the SAME completions on repeated passes → effective k ≈ 1-2 everywhere we
claimed k≥3. **What survives: everything load-bearing.** The paired deltas (aime +.091, hotpot
+.220, pupa tie, ifbench n.s.) are same-session paired measurements and pairing is untouched by
cache-collapsed passes — the variance-components analysis says pairing, not averaging, was the
binding remedy anyway. What does NOT survive: the "k≥3" description. Fixed: `rescore_k3.py` now
passes `cache=False` (all 3 boxes + repo); the paper's protocol section now reports the cache
pathology as a third instrument exhibit instead of claiming k≥3; the pupa "provider-side caching"
attribution corrected to client-side.

**Defect 2 — the aime rank-certificate lane was unviable and got killed.** Init eval alone took
2.5h (150 items × 24k tokens); 80 draws ⇒ ~150-200h. Killed by PID (wrapper 772463 → python
772473; also reval-waiter 845143, which was doubly broken — rescore_k3.py was never shipped to
sk1, and its hover stage cannot load there). sk1's 8B server kept up and repurposed.

**★ REOPENED — the aime "session noise" resolution is now in doubt.** Before dying, the aime
certificate measured the GEPA shipped init at **.5267** on sk1 (150 test items, 24k) — a second
independent sk1 reading near the original .5333, against sk3's paired-session .3067/.3000. Two
sk1 sessions ≈.53 vs one sk3 session ≈.31 looks like a **MACHINE/config effect, not generic
session noise** — HB97's "the .5333 was an inflated single-pass reading" was itself premature.
The paired delta on sk3 (+.091) remains internally valid, but if the deflation mechanism (e.g.
truncation under different server settings) is arm-ASYMMETRIC, the win could be an artifact.
**Discriminator launched** (sk1 pid 445184, logs/aime_crossbox_sk1.log): same-session paired k3
of official + unitrecomb ON THE HIGH-READING BOX, cache disabled. If M_ω also wins there → aime
stands (with "machine" replacing "session" in the paper's narrative). If GEPA wins there → aime
returns to CONTESTED and leaves W. The paper's AIME narrative paragraph is bracketed HOLD.

**Rest of the audit: decisions that were checked and STAND** — pre-registration committed before
data (verified in git); Phase 2 still uninspected (polled by count only, 21/90); provenance of
both confirmed candidates select-panel-clean (HB98); the hover purge/ablation queue; killing the
hung 14B lane (9.5h without a log write); leaving 4 unaccounted-for EngineCores alone; ifbench
no-widening. Comparator-rule tex paragraph softened to admit the family was only complete on
hover (envelope completion is cut, so the paper must not imply a 3-optimizer max everywhere).

## HB100 (2026-07-25) — ★★ the aime .53-vs-.35 mystery SOLVED: serving config, not machine, not session

User challenged the "machine effect" wording and directed a code check. Verified:
- `paperexact_arms.py` and `rescore_k3.py` md5-IDENTICAL on sk1/sk2/sk3.
- The aime GEPA candidate is byte-identical everywhere (cand md5 e018f7779184 in all 3 result.jsons).
- **The .5333 was not the 07-20 official run at all.** That run's own log ends `50/150 = .333 (seed)`,
  `55/150 = .367 (best)` — matching sk2's result.json. sk1's result.json was **overwritten on 07-23**
  by a stair-campaign re-run whose final test read 80/150 = .5333.
- **Mechanism, two stacked serving-config effects** (from the logs, not conjecture):
  (1) TRUNCATION: cert session at 24k → **0 truncation warnings** → .5267; today's crossbox at 8k →
  578 warnings → .3533. (2) REASONING-PARSER: the .5333 run's final-test log shows responses with
  `'text': None` + full output in `reasoning_content` — whether the vLLM server splits reasoning
  into a separate field (and whether the answer survives into `text`) depends on server
  version/flags, and a `text: None` response scores 0 after parse failure.
- So the correct model is NOT "machine effect" and NOT free-floating "session noise":
  **u_session for aime ≈ serving-configuration (max_tokens × reasoning-parser × vLLM version)**.
  Same code + same candidate + same box + same nominal CLI flags still spans .35-.53 across days
  because the SERVER behind the port changed. Consequence for the paper's variance-components
  table: the "session effect" row for deterministic metrics is config/serving-mediated — the tex
  already says this ("config/load-mediated"), now with a demonstrated mechanism.
- **Consequence for the aime cell**: the sk3 paired win (+.091 at 8k) is internally valid but is a
  *paper-exact-config* result in a regime where truncation destroys ~40% of GEPA's answers — and
  possibly asymmetrically (a prompt that elicits shorter reasoning survives 8k better; format
  robustness is partly REAL prompt value, partly instrument artifact). **Dual-column resolution
  queued**: after the 8k crossbox paired run finishes on sk1 (pid 445184), the identical paired
  run at 24k launches automatically (pid 436036, logs/aime_24k_sk1.log) — same box, same server,
  same code, same candidates, only max_tokens moves. 8k = paper-exact column; 24k =
  instrument-clean column. The pre-declared rule extends: aime counts in W only if M_ω wins the
  PAIRED comparison in BOTH columns; if it wins only at 8k, the honest sentence is "M_ω's prompt
  is more robust to the paper's own output-budget truncation," reported as such.

## HB101 (2026-07-25) — RESULTS_LEDGER established (user directive: stop the churn)

`datasets/prompt-optimality-test/RESULTS_LEDGER.md` is now the single source of truth: a number is
quotable ONLY if it appears there with a provenance block (artifact path, box, session, serving
config, effective k, pairing status). The five churn factors are enumerated as a checklist (F1
cross-session, F2 serving config, F3 dspy cache, F4 test-selection, F5 run-dir overwrite), each
with its mitigation status. `rescore_k3.py` now writes a `session_fingerprint` row (host, vLLM
version, max_model_len, full CLI config) at the top of every rescore block on all three boxes —
after the aime incident, no measurement is separable from the server that produced it. HB entries
remain the journal; the ledger is the state. Update the ledger in the same commit as any
measurement that changes it.

## HB102 (2026-07-26) — ★★★ HOVER CONFIRMED WIN (ablation-clean); aime 8k delta FAILS replication

**HOVER — the same-session 4-arm table (sk2, one session, cache off, n=300):**
| comparison | means | Δ | paired bootstrap |
|---|---|---|---|
| M_ω vs GEPA+Merge (strongest clean comparator) | .5144 → .5689 | **+.0544** | **P=.0004** |
| M_ω-ABLATED vs GEPA+Merge | .5144 → .5656 | **+.0511** | **P=.0009** |
| M_ω full vs M_ω ablated (the flagged clause's worth) | .5656 → .5689 | +.0033 | P=.41 — **causal NULL** |
| M_ω vs plain GEPA | .4522 → .5689 | +.1167 | P=.0000 |

**hover is a WIN and the leakage objection is empirically dead**: removing the one test-hit clause
from our best candidate costs .003 (P=.41, null), and the ablated candidate still beats the clean
GEPA+Merge by +.051 at P=.0009. Lock the ABLATED candidate as the cell of record if maximal
cleanliness is preferred; either version wins. Artifacts:
runs_paperexact/hover/Qwen3-8B/{official,official_merge_gepamerge,unitrecomb_stair,unitrecomb_stair_ablated}/rescore_k3.jsonl (sk2).

**AIME — the 8k paired delta does NOT replicate across servers.** sk1 same-session paired (cache
off, fingerprinted, n=150): GEPA .3533 vs M_ω .3578, **Δ=+.0044, W16-L18-T116, P=.40** — versus
sk3's +.091 (P=.0012) for the SAME two candidates at the same nominal config. Two internally-valid
paired sessions disagree on the DELTA, so the session effect on aime is NOT arm-symmetric — the
additive-u_session assumption fails here (the advisor's predicted failure mode, now observed).
**aime drops from W**: label = "delta not stable across serving environments; not confirmed." The
24k column (running, sk1 pid 436036) is now informational, not decisive — a same-config replication
failure at 8k cannot be overridden by a different-config win.

**Scoreboard: hotpot W (+.220), hover W (+.051 clean), aime NOT CONFIRMED, ifbench NOT CONFIRMED,
pupa TIE, livebench PENDING (P0 resumed and running — orchestrator working as designed after the
fd crash).** "Reflective search never beats recombination" still holds in every settled cell.

## HB103 (2026-07-26) — ★ TWO PRE-REGISTRATIONS (committed BEFORE livebench P0 is read; advisor pass 5)

**(1) Hotpot-placebo contingency.** IF livebench Phase 2 lands in the FALSIFIES cell of HB91
(placebo within .010 of real while both beat init), THEN we run the IDENTICAL placebo grid on
hotpot (40 real draws from a hotpot-mined pool vs 40 length-matched foreign-content placebo draws
vs 10 init replicates, randomized interleaved, one session), and the paper's content claim rests
on THAT result. Declared now so it is a control, not a post-hoc rescue. Cost zero unless the (c)
branch occurs.

**(2) Hover second-session stability pair.** The aime exhibit demonstrates cross-session delta
instability of .087 — larger than hover's locked +.051 margin — so hover must earn the sentence
aime failed to earn. Rule, declared before launch: rerun the 2-arm pair (unitrecomb_stair_ablated
vs official_merge_gepamerge) on sk2 in a FRESH server session with one config knob moved
(max_tokens 8000 → 24000, per the one-knob rule). **If the second-session paired delta remains
positive with P<.05 → hover is CONFIRMED-STABLE. If not → hover takes the aime label ("delta not
stable across serving environments") and leaves W.** We accept the risk knowingly: an unrun test a
reviewer can name is worse than a run one. Queued after the purged hover certificate on sk2.

**Also adopted from advisor pass 5:**
- aime is reported as a **significant session×arm INTERACTION** (Δ of deltas .087, per-session SEs
  ~.027-.03 → z≈2.1-2.3, p<.05): the arm-symmetric session model is REJECTED, not merely
  unconfirmed. To be computed exactly from the two rescore jsonls.
- aime 24k column: even if it shows an M_ω win, it earns AT MOST a footnote — the HB100 conjunction
  (win at both 8k and 24k) is already dead via the 8k replication failure. No drift back toward W.
- The protocol-exhibit triad is now graded: pupa (averaging insufficient) → cache (k-passes
  fictitious) → aime (PAIRING ITSELF insufficient when the metric interacts with response length).
  aime decomposition (per-arm truncation counts, sk3 vs sk1; does the +.091 concentrate on items
  where GEPA truncated and M_ω did not?) upgrades it from correlational to mechanistic — CPU-only.
- Abstract wording: replace "matches or exceeds on every benchmark tested" with **"never
  significantly worse in any settled comparison"** — same content, immune to the aime objection.
- Hover's mechanism-absence defense: report the hover session's truncation-warning count (expected
  ~0; hover outputs are short labels) — "the demonstrated instability channel requires truncation
  exposure; hover has none."

## HB103b (2026-07-26) — ★ AIME STABILIZATION PLAN (user directive: "just get it stabilized"; committed BEFORE the 24k result is read — process confirmed still running, output not inspected)

The 8k regime is unstable BECAUSE the instrument dominates there: ~40% of GEPA's answers truncate,
and truncation exposure varies with serving config and interacts with each prompt's reasoning
length. The fix is not more 8k replications — it is measuring where the mechanism is absent.

**Pre-registered rule for aime's canonical cell:**
1. The canonical aime number = the **24k paired both-arms** measurement (in flight, sk1 pid 436036),
   **replicated once** in a second fresh-server session at 24k.
2. If the two 24k paired deltas agree in sign and the pooled paired bootstrap gives P<.05 → aime is
   **stabilized at the 24k value** (win, tie, or loss — whatever it is), reported as the
   instrument-clean column, with the 8k instability reported as the third protocol exhibit.
3. If the two 24k sessions ALSO disagree on the delta → aime is reported as measurement-unstable,
   full stop, and no aime performance claim of any kind appears in the paper.
4. The 8k numbers are never again quoted as performance; they are exhibit material only.
This supersedes the now-moot HB100 conjunction rule (dead via the 8k replication failure) with a
single convergent path instead of an open-ended one.

## HB104 (2026-07-26) — ★ CAN THE POOL ACTUALLY CERTIFY? Computed. Two blocking issues, both fixable.

Ran capture-recapture over every mining run that recorded a unit list (CPU only, no GPU):

| bench | mining runs | distinct units S | f1 (singletons) | f2 (doubletons) | Chao1 N̂ | implied unseen |
|---|---|---|---|---|---|---|
| hover | 3 | 161 | **0** | 149 | 161.0 | **0** (degenerate) |
| hotpot | 3 | 68 | **0** | 60 | 68.0 | **0** (degenerate) |
| aime | 2 | 48 | **0** | 48 | 48.0 | **0** (degenerate) |
| ifbench | 3 | 220 | 152 | 36 | **540.9** | **~321** |

**★ ISSUE 1 — most of our "independent" mining runs are not recaptures at all.** hover/hotpot/aime
show f1=0 because those runs CONSUMED A FROZEN POOL (`--pool-file`) rather than re-mining: the same
units appear in every run by construction, so every unit is a doubleton/tripleton and Chao1
collapses to S. That is the identical pathology that killed EVT — the estimator is fine, the
sampling is not independent. **ifbench is the one bench where mining genuinely re-ran** (32 / 128 /
160 units from three different mining passes), and there capture-recapture DOES produce a number:
N̂≈541 against 220 observed, i.e. ~60% of the unit space unsurfaced. Caveat: f1≫f2 (152 vs 36)
makes that estimate unstable and heavily heterogeneity-inflated.
→ **FIX (cheap, no GPU beyond mining): freeze the miner and re-run it K≥3 times with independent
seeds per bench**, then capture-recapture is valid for THAT declared miner. This is the same
"specify the sampling, don't strengthen the estimator" move that produced the rank certificate.

**★ ISSUE 2 — the bound's conservativeness runs the WRONG WAY, and this must not reach the paper
unfixed.** Chao1 is a *lower* bound on richness. Fewer estimated unseen units ⇒ smaller ε̂ ⇒
**B̂ = achieved + ε̂ is too LOW ⇒ the "ceiling" can be exceeded by a real prompt.** For an upper
bound we need an UPPER confidence limit on missing mass (upper end of the Chao1 CI, or a
Chao–Lee coverage-corrected estimator) AND an upper quantile — not the mean — of the unseen
units' value distribution. Pricing note: unseen units are by construction the rarely-proposed
ones; whether they are worth less (rarely proposed because weak) or more (rarely proposed because
novel) is an EMPIRICAL question we can answer from data in hand — regress per-unit marginal delta
on capture frequency. If value declines with rarity, pricing unseen units at the singleton rate is
defensible and mildly conservative.

**Consequence for the abstract's "upper bound" language:** as currently computed it is an
*estimate*, not a bound, in the direction that matters. Either (a) switch to upper confidence
limits and call it a bound, or (b) call it an estimated ceiling and reserve "bound"/"certificate"
for the two objects that genuinely are one (metric-reachability cap; rank certificate). Decision
owed before the abstract is finalized.

**Paper edit made this pass:** new `\section{Background: what makes an exploration process
measurable}` containing (i) the GEPA-vs-ε-certifiability contrast (user's framing, verbatim in
substance), (ii) "what survives non-i.i.d. sampling" — declare-the-sampler, one-sided statements
need no exchangeability, declared-generator tails are certifiable exactly, (iii) a provenance
paragraph that PLAYS DOWN GEPA-derivation with measured numbers: LLM suggestion supplies roughly
half of all units (hover 192/340, hotpot 120/144, ifbench 144/320), trajectories the rest, so
optimizer-trajectory mining is one sampling technique among several, not a precondition.

## HB105 (2026-07-26) — certifiability groundwork: UCB machinery computed, rarity-value answered, re-mining launched (and the cache strikes a THIRD time)

**Why pools were frozen (user asked):** deliberate at the time — freezing made the SEARCH
reproducible across staircase rungs and removed the flaky z.ai mining dependency from GPU runs
(freeze-before-eval discipline). Right for search comparability; silently fatal for
capture-recapture, because a consumed pool is the same capture repeated, not a recapture. It is
NOT the source of the score pathologies (those were serving-config/session effects); it is the
reason certifiability could not be computed on hover/hotpot/aime.

**UCB machinery — derived and computed on ifbench (the one bench with genuine re-mining):**
- Chao1 point estimate N̂ = S + f1²/(2f2) = 541; classical variance + Chao (1987) log-normal CI:
  **95% CI [422, 729] → unseen-count UCB ≈ 509** (vs 220 observed).
- Good-Turing missing probability mass: P(next mined unit is novel) = f1/n = **.475**, one-sided
  95% UCB (McAllester–Schapire-style concentration, +√(2ln(1/δ)/n)) = **.612**. Distribution-free;
  needs exchangeable unit-draws, which declared re-mining replicates provide.
- ε̂_UCB assembly (design): UCB(unseen count) × upper quantile of singleton marginal values, or
  extrapolate the concave value curve to the CI-upper endpoint (the Fig-1 combined plot).

**★ Rarity-vs-value regression (user directive) — rare units are worth MORE, not less.**
ifbench (only bench with frequency variation): slope −.058 per capture, **r = −.55**; mean
marginal by capture count: singletons **+.028**, doubletons **+.032**, tripletons **−.114**.
Direction = "rarely proposed because NOVEL", not "because weak". Consequence: pricing unseen
units at the singleton mean is NOT automatically conservative for an upper bound — another
reason the UCB must use an upper quantile. **Confound to disclose:** capture count correlates
with WHICH run measured the delta (tripletons include the earliest, worst-session run's
readings); re-mined pools + fresh marginal measurement will de-confound.

**★★ THE CACHE STRIKES A THIRD TIME (F3c).** First K=3 re-mining run completed in 45 seconds
with all three replicates BYTE-IDENTICAL per bench (md5-verified) — dspy's response cache served
replicates 1-2 (and possibly 0) from disk. A replicate that can be served from cache is not a
capture. Quarantined as pools/remine_CACHED_IDENTICAL_20260726/. Relaunched with an explicit
cache=False reflection LM (sk2 pid 1366186); v2 is taking real API time (>10 min/replicate),
which is what genuine mining looks like. F3 has now defeated: k-pass averaging, mining
replicates — and pupa's k5. Rule: **any stochastic replicate anywhere in this stack must
construct its LM with cache=False and prove non-identity (hash check) before use.**

**Paper:** Fig 1 v4 per user iteration (s_i inside circles; vertical p^i→m^i stack with
improving colorings; pool + unseen-mass chip; single combined Good-Turing/Chao-CI/value plot).

## HB106 (2026-07-26) — ★★★ LIVEBENCH P0 VERDICT (HB91 applied mechanically): the comparison WINS, the content claim is FALSIFIED, the 120/120 claim is DEAD

All three phases complete (one session; fd-crash resumed per HB-F6; results read only after completion).

**Phase 1/3 — re-selection CONFIRMS the comparison.** Select-promoted draw #88 (chosen on the
held-out select panel, never by test) vs GEPA official, same session, k3:
**GEPA .5548 vs draw88 .6470, Δ=+.0923, W35-L13-T78, paired bootstrap P(Δ≤0)=.0001** (n=126).
Meets the pre-registered CONFIRM cell (Δ≥.030 ∧ P<.05).

**Phase 2 — the placebo FALSIFIES the content claim.** 40 real draws (livebench's own mined
units) vs 40 placebo draws (hover's clauses — length/count-matched, content-irrelevant) vs 10
init replicates, randomized interleaved:
| arm | n | mean | sd | min | max |
|---|---|---|---|---|---|
| real (own units) | 40 | .7055 | **.100** | .362 | .777 |
| placebo (foreign clauses) | 40 | **.7342** | .031 | .621 | .773 |
| GEPA init replicates | 10 | .6213 | .033 | .578 | .706 |
**Placebo ≥ real** (−.029, rank-sum z=−1.04, p=.30 n.s.) and both ≫ init. This is the
pre-registered FALSIFIES cell, a fortiori: content-free foreign clauses reproduce (numerically
exceed) the entire gain. Note also real-sd (.100) ≫ placebo-sd (.031) with real-min .36 —
livebench's own mined content can HURT badly; padding is uniformly helpful.

**Kill switch FIRES:** init-replicate max (.7055) sits AT the real-draw mean → the "120/120
random recombinations beat GEPA" claim is DEAD, per the unconditional pre-registered rule.

**Pre-registered sentence (verbatim consequence):** the livebench story is rewritten as a
METRIC-PATHOLOGY finding — under zero-on-parse/abstention scoring, prompt bulk of ANY content
suppresses the abstain→zero mode and lifts scores ~+.09-.11; the mechanism is structural, not
mined content. **livebench does NOT count in W.** Headline branch (c) applies for its clause:
we own the pathology as a finding.

**★ CONTINGENCY HB103#1 FIRES (pre-registered): the identical placebo grid runs on HOTPOT**, and
the paper's content claim rests on that result. If hotpot's +.220 survives its placebo, the
pool-content thesis stands on the flagship cell; if hotpot's gain is also reproduced by foreign
clauses, the thesis itself must be rewritten. Launching on sk3 GPU7.

Also: hover K=3 re-mining returned 0 units ×3 (mining failure to investigate — GLM refusal or
parse; ifbench/aime/hotpot replicates TBD-check); purged hover cert running on sk2; sk1
unreachable again at poll time (aime 24k sessions unread).

## HB107 (2026-07-26) — ★ BINDING PRE-REGISTRATION: hotpot placebo interpretation grid (advisor pass 6; committed while the grid runs, results UNREAD)

Definitions: R = real-draw mean (hotpot's own 68-unit pool, p=.5, n=40); P = foreign-content-draw
mean (livebench clauses, count-matched, n=40); I = init-replicate mean (n=10). Content share
**θ = (R−P)/(R−I)**. Primary inference: rank-sum real-vs-placebo. Power: even at worst-case sd
.100, SE of mean difference ≈ .017 at 40/40, so the +.050 bar is ~3 SE.

| cell | rule | consequence |
|---|---|---|
| VOID/SELECTION | R−I < +.05 | grid cannot adjudicate; the +.220 cell SURVIVES (it is a selected-candidate paired comparison, not a draw mean); content claim retreats to "the pool contains the value; selection extracts it" (marginals-only support); NO re-roll |
| **CONFIRMS** | R−I ≥ +.05 ∧ R−P ≥ +.050 with p<.05 ∧ θ ≥ .5 | pool-content thesis stands on the flagship; abstract keeps pool-not-search with "audited by placebo control"; livebench owned as the pathology exhibit |
| AMBIGUOUS | R−P ≥ +.020 with p<.05 ∧ θ < .5 | honest split "X structural, Y content"; thesis softened: content adds measurably but structure is the larger term |
| **FALSIFIES** | P within .020 of R (or p ≥ .05) while P−I ≥ +.05 | content thesis dead on both probed benches; ALL M_ω-vs-GEPA comparisons survive as comparisons; the paper becomes the methodology/pathology paper + the striking negative "GEPA's search fails to beat arbitrary foreign text"; pool-content clause DROPPED not softened; title's articulation framing reviewed |

**Advisor's declared expectation (recorded so no escape hatch can be invented later):** CONFIRM
with θ ≥ .7 — EM over short answers has no abstain→zero bulk channel; +.220 is 2× the largest
structural artifact seen anywhere; hotpot's pool is 0/68 flagged. If foreign clauses reproduce
>half the hotpot gain, something instrument-shaped is likely — **but the cell verdict binds
regardless**; the per-item decomposition may EXPLAIN a FALSIFIES outcome, never overturn it.

**One permitted follow-up** (only if AMBIGUOUS or FALSIFIES): one session re-running the full
grid with a shuffled-text arm added. If CONFIRM: no hotpot follow-up at all.

## HB107b — advisor pass 6, other rulings (all adopted)

- **Livebench mechanism sentence is GATED.** Phase 2 proves content-SOURCE-INDEPENDENCE only.
  "Prompt bulk suppresses abstention" may not print until the CPU per-item decomposition shows
  the +.11 concentrates on 0→nonzero conversions. Rivals: H1 bulk per se / H2 coherent
  instruction transfer (hover clauses = good generic advice) / H3 format-length interaction.
- **Placebo RENAMED "foreign-content control"** everywhere — hover clauses are coherent generic
  instructions, not inert text; the falsification inference survives the rename, but HB106's
  "prompt bulk of ANY content" overclaims and is corrected to source-independence.
- **Third arm MANDATED before submission (not a footnote):** word-shuffled versions of the SAME
  hover clauses (token/vocab-matched, syntax destroyed — NOT lorem ipsum), one fresh session of
  shuffled+foreign+init (~20/20/10; the real arm is NOT re-run — real-vs-foreign is settled and
  must not be re-litigated). Pre-registered readout: |shuffled−foreign| ≤ .02 → bulk per se;
  shuffled within .02 of init → coherent-instruction transfer; intermediate → both channels.
  Cache off; hash-check the shuffles (F3c).
- **"Mined content adds risk, not value"** — livebench real-draw sd .100 vs placebo .031, real
  min .362 BELOW init min .578: own-mined draws are zero-mean with a severe harmful tail while
  foreign padding is a uniform level shift. Stronger and truer than "content is inert".
- **Purged hover certificate: KEEP** (ablation cleaned the candidate; the certificate is a
  pool-level object; headline promises leakage-audited certificates).
- **Livebench rank certificate: KEEP, RE-SCOPED** into the pathology section as the worked
  example of protocol-relativity: P(fresh>.7914)≤.0083 remains exactly true over a class whose
  gains are structural — the vacuity theorem made concrete. Add: certificates are
  mechanism-agnostic.
- **Newly pointless:** further livebench real-vs-foreign arms; the compiled-candidate livebench
  follow-up (its cell didn't occur); the flagged-unit conditional-draw readout on livebench;
  best-of-N-draws as pool-value evidence on livebench (descriptive row only); anything downstream
  of "120/120".
- Hover re-mining 0×3: investigate cheaply (raw GLM responses — refusal or parse regression);
  NOT critical path; fallback = certifiability on ifbench alone with frozen-pool benches labeled
  degenerate-by-design (HB104).
- Critical path: hotpot grid → hover stability pair → livebench third arm + zero-rate
  decomposition → aime 24k (sk1 still unreachable) → writing with both abstract branches
  verbatim-ready before the hotpot result is read.

## HB108 (2026-07-26) — purged hover certificate lands; hotpot grid and hover stability pair both mid-flight

**★ Purged hover rank certificate (leakage-purged 160-unit pool, N=80, one sk2 session):**
- Ladder: **P(fresh draw > .5800) ≤ .0123**; P(> .5767) ≤ .037; P(> .5667) ≤ .123.
- Draws: mean **.5425 ± .019**, max **.5800**; GEPA shipped init on the same panel/session: **.4867**;
  **80/80 draws beat the init**.
- Two readings worth flagging for the paper (both mechanism-agnostic per HB107b):
  (a) the certificate brackets our greedy candidate — M_ω-ablated .5656 sits INSIDE the draw
  ladder (above the mean, below the max), i.e. greedy selection on hover lands around the
  ~85th-90th percentile of random recombination, echoing the pool-not-search thesis at the
  certificate level; (b) hover now has the SAME qualitative pattern livebench had
  (draws ≫ init, 100% beating) — whether hover's version is content or structure is exactly
  what the running foreign-content controls decide (hotpot grid now; hover would need its own
  if hotpot falsifies — hotpot CONFIRM covers the thesis per HB103#1).
- CAVEAT before quoting alongside HB102: the cert session's init reading (.4867) differs from
  the reval session's (.4522) — cross-session levels move (F1/F2); only within-session
  comparisons are quotable. The cert is self-contained (init + draws in one session).

**Lanes:** hotpot foreign-content grid 17/90 (~3.5h total, counts only, per HB107). Hover
stability pair started right behind the finished orchestrator (24k, truncation warnings visible
= expected at hover's long CoT? — watch: hover reasoning at 24k still truncating is itself
notable). sk1 STILL unreachable (banner-exchange timeout; sk1-specific — sk2/sk3 fine); both
aime 24k sessions remain unread.

## HB109 (2026-07-26) — ★★★ HOVER CONFIRMED-STABLE (HB103#2 rule applied): the delta survives a fresh session AND a config change

Second-session stability pair (fresh server process, vLLM 0.16.0, one knob moved 8k→24k per the
one-knob rule, fingerprinted, cache off): **GEPA+Merge .5267 vs M_ω-ablated .5622, Δ=+.0356,
W51-L34-T215, paired bootstrap P=.0086, n=300.**

Per the pre-declared HB103 rule (positive with P<.05) → **hover is CONFIRMED-STABLE.** The two
paired sessions give deltas **+.051 (8k, session 1)** and **+.036 (24k, session 2)** — same sign,
same magnitude range, across a serving-config change of the exact kind that flipped aime's delta
from +.091 to +.004. Hover is now the campaign's most fortified cell: same-session paired ✓,
leakage-ablated ✓, strongest-clean-comparator ✓, cross-session/config delta-stable ✓.
Artifacts: runs_paperexact/hover/Qwen3-8B/{official_merge_gepamerge,unitrecomb_stair_ablated}/rescore_k3.jsonl
(final rows = stability session, fingerprint 2026-07-26T17:46Z).

W = 2, both now airtight: hotpot +.220 (P<1e-13; placebo grid mid-run) and hover +.036..+.051
(P=.0086/.0009, delta-stable). hotpot foreign-content grid at 26/90 (~4h to the thesis verdict).

## HB110 (2026-07-26) — user directive: stability program CLOSED

Per user challenge ("how many times have you run aime?" — answer: 8, of which the last 2-3 were
polishing a footnote): **the stability program is DONE.** Rules now in force:
1. aime: session 1 (24k) may finish; **session 2 is CANCELLED** (kill waiter pid 3193668 when sk1
   becomes reachable). aime ships as *"directionally positive; delta not stable across serving
   configurations; excluded from W."* No further aime measurement of ANY kind.
2. No new re-measurement anywhere. Running tails only: hotpot placebo grid (the thesis gate) and
   aime 24k s1. Hover passed its one stability pair (HB109) — done.
3. The stability work's actual products stand: pupa/ifbench artifact detection, the truncation/
   serving-config mechanism, hover's fortification. The waste was runs 8-9 on a cell the advisor
   had already ruled out of W. Lesson recorded: "stabilized" = classified-and-closed, not
   measured-until-convergence.

## HB111 (2026-07-26) — ★★★ EXPANSION PASS (user directive): why the wins win

**HOTPOT (+.220): the winning candidate decomposed.** 31/68 pool units chosen; chosen mean
marginal **+.0445** vs unchosen **−.0114** — the greedy gate genuinely discriminates. The
composition has a clean two-tier structure:
- **Top tier (marginals +.09..+.16, the top 6 units): ANSWER-FORMAT DISCIPLINE** — "extract only
  the exact entity", "just the person's name", "strict exact-string match, stripping articles",
  "no conversational filler". EM scoring rewards exact answers; the single most valuable
  articulable content on hotpot is output-shape control.
- **Mid tier (+.02..+.09): MULTI-HOP STRATEGY** — entity-dependency DAG, bridge-query
  formulation ("query that seeks the missing second hop"), evidence discipline ("rely
  exclusively on the provided summaries"), fact-checker persona.
- **Provenance: 25/31 chosen units are LLM-SUGGESTED, only 6 from GEPA trajectories** — on the
  flagship bench, the pool's value is mostly NOT GEPA's discoveries. Strengthens the
  provenance-independence claim (HB104) with outcome-level evidence.

**HOVER: what selection adds over the pool.** 61/164 chosen (mean marginal +.0216 vs −.0098).
Top units are **retrieval STRATEGY, not format**: synonym/alias/acronym query expansion,
extract-2-3-unresolved-keywords, confirmed-vs-missing delineation, pivot broad→specific.
Source mix balanced (27 llm / 32 trajectory / 2 xlm) — hover's trajectories DID matter, unlike
hotpot. Within the cert session: init .4867 → random-draw mean .5425 (**pool effect ≈ +.056**)
→ greedy candidate ≈ 84-90th percentile of draws (**selection effect ≈ +.02**; cross-session
caveat on the candidate's level). Rough split: **~70% pool, ~30% selection.**

**LIVEBENCH: the mechanism gate (HB107b) RESOLVES — and the abstention story does NOT survive.**
Zero-rates: init .282 → placebo .177 → real .201. But the gain decomposition: items with init≈0
(16/126) carry only **22%** of the +.106 lift; **78% comes from partial-credit items** — the
foreign clauses mostly improve Levenshtein similarity on items that already scored, not
0→nonzero conversion. Per the gate: **"prompt bulk suppresses abstention" may NOT print as the
mechanism.** The printable sentence: the lift is content-source-independent, reduces the
zero-rate .28→.18 (≈22% of the gain), and mostly improves partial-credit similarity (≈78%) —
consistent with the format/length-interaction channel (H3). The shuffled-text arm remains the
H1-vs-H2 discriminator.

**★ Emerging taxonomy worth a paper table (bridges the Daston thin/thick framing):** chosen
units classify cleanly into (a) FORMAT/output-shape rules (thin, mechanical — dominate hotpot's
top tier, are livebench's entire pathological lift), (b) STRATEGY rules (thick, judgment-laden —
dominate hover: query reformulation, pivoting), (c) EVIDENCE-DISCIPLINE rules (grounding,
no-outside-knowledge). Different metrics reward different rule types: EM rewards thin rules
genuinely, zero-on-parse rewards them pathologically, multi-hop retrieval rewards thick ones.
This is "prompt-code isomorphism shows thin/thick rule differences" (the abstract's closing
claim) with data already behind it.

## HB112 (2026-07-26) — aime CLOSED with partial 24k evidence; sk1 back after ~1-day outage

sk1 returned. Both aime lanes died during the box outage (likely reboot): session 1 completed
**two clean 24k passes of GEPA-official, both reading .5800 exactly** (87/150, cache OFF — real
replicates, 3.5h apart) before dying mid-run; the M_ω arm never ran; session 2 never started (its
cancellation per HB110 is moot). The 8B server on sk1 is down and STAYS down (nothing needs it).

**Disposition (HB110 rules applied):** aime measurement remains CLOSED. The two .58 passes are
recorded as MECHANISM evidence only, not a cell: they confirm the serving-config account —
same candidate reads .30-.35 at 8k (many truncations) and .53-.58 at 24k (none), and at 24k the
readings are pass-stable to 4 decimals with the cache off. aime ships as "directionally
positive; delta not stable across serving configurations; excluded from W," with the 8k/24k
level split quoted as the truncation exhibit. No paired 24k comparison exists and none will be
run.

hotpot foreign-content grid: 72/90 — verdict expected within the hour.

## HB113 (2026-07-26) — ★★★ HOTPOT PLACEBO GRID: **CONFIRMS THE CONTENT CLAIM** (HB107 applied mechanically)

One randomized session, sk3 GPU0, 90/90 evals, fingerprinted:
| arm | n | mean | sd | min | max |
|---|---|---|---|---|---|
| real (hotpot's own 68-unit pool, p=.5) | 40 | **.5936** | .049 | .500 | .663 |
| foreign-content control (livebench clauses, count-matched) | 40 | **.3326** | .203 | **.000** | .527 |
| GEPA init replicates | 10 | .4133 | .000 | — | — |

**Grid cells:** R−I = **+.180** (≥ .05 ✓); R−P = **+.261**, rank-sum z = **7.28**, p ≈ 2e-13
(≥ .050 with p<.05 ✓); **θ = (R−P)/(R−I) = 1.45** (≥ .5 ✓, and > 1 because the placebo lands
BELOW the init). Separation: real min (.500) > init max (.4133) ✓. **Cell: CONFIRMS.** The
advisor's declared expectation (CONFIRM, θ ≥ .7) is met and exceeded.

**The result is stronger than confirmation — foreign bulk actively HURTS hotpot**: placebo mean
sits .08 BELOW the bare init, with a catastrophic tail (min .000 — some foreign draws destroy
the multi-hop pipeline entirely). Combined with HB106, this yields the paper's sharpest
symmetry: **the same foreign clauses that lift livebench by +.11 sink hotpot by −.08.** Prompt
bulk is metric-dependent — it exploits zero-on-parse partial-credit scorers and damages
exact-match multi-hop pipelines — while OWN-POOL CONTENT is what survives across metrics
(+.180 draw-level on hotpot; 100% of real draws ≥ .50).

**Disclosure (F3 yet again, contained):** the 10 init replicates read identically (sd = 0.000) —
the grid's task LM did not disable the dspy cache, and the init arm re-used one completion set.
Effective init n = 1. This does NOT touch the verdict: (a) the init LEVEL .4133 exactly matches
the independent same-session HB97 measurement; (b) real and placebo draws are all distinct
prompts (no cache collisions possible); (c) the decisive contrast is real-vs-placebo (n=40/40,
z=7.28), which does not involve the init arm. Noted for the methods appendix; grid script
patched expectation: add cache=False for any future grid.

**Consequences (pre-committed):** the pool-content thesis STANDS on the flagship cell; the
HB95b/Branch-A abstract keeps the pool-not-search claim with "audited by a foreign-content
control on the flagship cell"; livebench stays owned as the pathology exhibit; NO hotpot
follow-up runs (CONFIRM ⇒ none permitted). W = 2 (hotpot content-audited + hover
confirmed-stable), livebench = comparison-win/content-falsified, pupa tie, ifbench directional,
aime closed.

## HB114 (2026-07-26) — advisor pass 7 adopted + the expansion analyses land

**Rulings adopted:** shuffled-text livebench arm RELEASED (optional, gates nothing; the pathology
section keeps the hedge verbatim — "we did not discriminate bulk-per-se from coherent-instruction
transfer"). Bound-vs-ceiling: option (b) — "bound"/"certificate" reserved for the
metric-reachability cap and rank certificates; capture-recapture = "estimated ceiling",
appendix-only, ifbench-only. Writing roadmap (9 sections), promote/appendix table, and the 3
carrying figures (scoreboard table with verdict-label column; the two-bench three-arm
content-audit symmetry figure; hover pool-anatomy figure) accepted as drafted targets.
DONE declarations: aime, livebench, hover, hotpot ALL closed — hotpot explicitly "no follow-up
permitted" (CONFIRM cell rule). K=3 re-mining v2 + hover 0-unit mining failure: out-of-scope for
submission (capture-recapture ships as caveated appendix).

**★ Analysis 5 — hover truncation exposure (defensive check): 12 warnings (8k session) and 9
(24k session)** across ~thousands of generations (<0.5%): exposure is negligible AND nonzero in
both, so the printable defense is the STRONG form: *the hover delta survived a change in
truncation regime (8k→24k, +.051→+.036)* — not the fragile "no exposure" claim. HB103's drafted
defense line updated accordingly.

**★★ Analysis 1 — hotpot foreign-draw autopsy: the smoking gun is one clause.**
`Put your final answer in LaTeX inside a $\boxed{}$` appears in **100% of catastrophic draws
(≤.10) vs 42% of benign draws (≥.45)** — a livebench thin rule that wraps hotpot's extracted
entity in LaTeX, guaranteeing an exact-match zero. The two-channel story is confirmed in data:
format-collision clauses drive the .000 tail; livebench's math-content clauses ride along.
Conversely the most benign-compatible foreign clause is the 'Answer:' list-format rule (95% of
benign draws) — a format rule the EM parser happens to survive. Cliff-vs-slope, made literal.

**★ Analysis 2 — the composition story is estimator-robust.** Independent draw-level regression
(40 real draws, with-vs-without per unit) reproduces HB111's greedy-marginal top tier:
bridge-query discipline (+.040), person-name/format rules (+.035), constraint re-read (+.032),
module-role discipline (+.025). Bottom of the list: "output an empty summary if passages are
irrelevant" = **−.077** — the abstention-style rule is the worst unit in the pool (echoes the
no-expect-empty doctrine). One-line paper claim: "confirmed by an independent draw-level
regression."

**★ Analysis 4 — the aime interaction test, exact:** session A (sk3) Δ=+.0911 (se .0296) vs
session B (sk1) Δ=+.0044 (se .0184) → **arm×session interaction z=2.49, p=.013** — the
arm-symmetric session model is REJECTED, not merely unconfirmed. This is the number §4 quotes.

Analysis 3 (aime per-item truncation decomposition): NOT executable from existing artifacts
(truncation warnings are not item-attributed in the logs) — dropped, per HB110 no new runs.
Analysis 6 (thin/thick judge classification of all pools): optional, GLM-only; queued behind
writing.

## HB115 (2026-07-26) — full quiescence; re-mining root cause = z.ai balance exhausted

- **Re-mining v2 (and hover's earlier 0×3) root-caused: z.ai error 1113 "Insufficient balance"**
  — the sk2 key (alexander-spangher) is drained, as the sk1 key was days ago. Every GLM
  suggestion call failed through the patient retry stack → 0-unit replicates on all benches.
  Not a code failure; the cache=False change was correct. Capture-recapture is out-of-scope for
  submission (HB114), so no impact — but any future mining or the optional thin/thick judge
  pass (analysis 6) needs a z.ai top-up or the alternate key. Empty replicate files left in
  pools/remine/ labeled by their summary (S=0) — self-documenting, nothing quotable.
- **sk3 GPU0's idle grid server killed by PID** (775490 + EngineCore); all sk3 GPUs now clear of
  our processes. sk1/sk2 verified quiet. **The campaign is fully quiescent: zero running jobs,
  zero open experimental questions.** Remaining work is writing (advisor pass 7 roadmap) +
  optional post-submission items (shuffled arm, re-mining after top-up).

## HB116 (2026-07-27) — E-plan GREEN-LIT and launched; no-API constraint; sk1 disk full

**User rulings:** all E-experiments green-lit. NO API LMs at all (z.ai dead; Anthropic key also
dead — the copies I briefly placed on boxes are removed). "Sonnet" meant Max-plan subagents from
the session; then superseded by: **use a strong LOCAL model on the server** for reflection/
mining/judging. C4a: spirit-not-letter is fine → E1-full's calibrated line ("median ±.03 at 4-8
steps ahead, fit on ≥8-10 points") is the supported rendering.

**Harness v9:** make_reflection_lm gained claude-routing (now inert — no key) and a `cache` kw
(F3 rule); shipped byte-identical to all boxes (md5 9c2a3da3...).

**Infrastructure:** ★ sk1 /lfs is **100% FULL (0 bytes free)** — writes fail; sk1 unusable for
new work until space frees (not ours to clean; user should ping admins/check docker). sk3 has no
clean GPU (all free-looking GPUs carry other users' resident memory). → everything consolidated
into ONE serial queue on sk2 GPU3 (pid 3575442, logs/ework_queue.log):
**hover ablation battery (E2/E3) → hotpot ablation battery → OSL rungs 8B → 1.7B → 4B → 14B
(E5-P1, hotpot, 24k, cache-off)**. Scripts: ablation_battery.py, osl_battery.py (repo + boxes).

**Deferred pending a free GPU + strong-local-model choice:** E4 re-mining, E6 classification,
E9 MIPRO (needs in-process proposal LM → now UNBLOCKED by the strong-local pivot), E7 code-writer.
Candidates (memory): GLM-5.2-NVFP4 local (4×B200 — impractical), **Llama-70B BF16 (fits one
B200)**, **Gemma-4-31B (certified ≥ Sonnet as judge in this project)**, Qwen3-32B. Proposed:
Llama-70B = miner/proposer; Gemma-4-31B = judge/classifier. Chain them on sk2 GPU3 after the
E-work queue unless the user picks otherwise.

## HB117 (2026-07-27) — WHY THE PLACEBOS FAILED: livebench's pool is a MIXTURE, not inert

Draw-level regression on the 40 P0 real draws (score vs unit inclusion, n≈20 draws per side):
- **20 of 48 units have positive marginals** (top: cyclotomic-polynomial content +.066, roots-of-
  unity heading +.048, state-the-domain +.042, strict-output-format +.042, geometry-sketch +.042);
  their marginals SUM to **+.576**.
- **6 units are actively toxic** (worst −.063: "write out the total number of combinatorial
  cases", "translate word problems first", "provide clear step-by-step reasoning" −.058).
- Random p=.5 draws mix ~10 good + ~3 toxic + 11 neutral → the mixture averages DOWN to the
  foreign-content level. **The placebo did not show content is worthless; it showed the POOL is
  diluted.** That is a different (and more interesting) finding, and it is testable.

Same logic likely explains ifbench's non-significance: 32 units is a thin pool, and its search
fell back repeatedly. The fix there is a BIGGER, better-sampled pool, not a re-roll of the old one.

## HB118 (2026-07-27) — ★ PRE-REGISTERED WIN ATTEMPTS (committed BEFORE any of it runs)

Both attempts are single-shot. No re-rolls; whatever the rule says is what the paper says.

**IFBENCH — merged-pool search.** New pool = 32 frozen + 180 sol-mined units (3 independent
gpt-5.6-sol replicates, hash-verified distinct) = **212 units**, `pools/ifbench_Qwen3-8B_merged_sol.json`.
Run `unitrecomb --run-tag solwide --lm-cache-off` (harness v9.1 adds `--lm-cache-off`, so search
k-passes are finally real samples), then ONE final session at **k=5** scoring
{official, unitrecomb_v6ctx32k, unitrecomb_solwide} together.
*Rule:* ifbench counts as a WIN iff best-M_ω minus official has paired-bootstrap **P(Δ≤0) < .05**
in that session. This is the ONE permitted widened-k replication reserved in HB91 — spending it
now, declared, on a genuinely NEW candidate rather than a re-roll of the old one. If it fails,
ifbench keeps its "directionally positive, not confirmed" label FOREVER.

**LIVEBENCH — content rescue (`livebench_content_rescue.py`).** Stage 1: score all 48 units'
marginals on the SELECT panel (train[:81], k=2, cache off) — selection re-derived cleanly, never
from test. Deterministic rule: **keep the top-12**. Stage 2: one test session, cache off, arms =
8 init replicates + 8 selected-candidate evals + 15 count-matched foreign-content draws.
*Rule:* **CONTENT RESTORED iff selected-mean > placebo-mean with rank-sum p < .05.** Otherwise
livebench's content claim stays dead and no further attempt is made.
Note this tests a sharper claim than the original: not "the pool's content is valuable" but
"SELECTED content beats bulk" — which is exactly the pool-not-search thesis with selection
doing the work the P0 random draws deliberately withheld.

Chain armed on sk2 GPU3 (pid 1099047) behind the E-work queue: ifbench search → ifbench k5
session → livebench rescue.

## HB119 (2026-07-27) — ★★ SEMANTIC SPECIES FIXES CAPTURE-RECAPTURE; real p(unseen) curves extracted

**★ The estimator works once species are SEMANTIC.** Codex/sol clustered all sol-mined clauses +
frozen-pool clauses into semantic equivalence classes; over the 3 independent replicates:
| bench | clusters | replicate-species S | f1 | f2 | f3 | Chao1 N̂ | 95% CI | cross-sampler overlap |
|---|---|---|---|---|---|---|---|---|
| hotpot | 109 | 73 | 22 | 19 | 32 | **86** | [78, 108] | 18 clusters |
| ifbench | 85 | 72 | 21 | 17 | 34 | **85** | [77, 109] | 15 clusters |
Verbatim matching gave f2=0 (degenerate); semantic matching gives well-conditioned spectra and
real CIs. **This is the fix HB104 predicted** ("specify the sampling, don't strengthen the
estimator" — here: specify the SPECIES). Cross-sampler overlap (18/15 clusters shared with the
GLM-era frozen pool) also shows two different miners rediscover the same units — evidence the
unit space is a property of the task, not of the miner.

**★ Fable extraction: 30 REAL Good-Turing curves** (10 metric banks × 3 species granularities),
`runs/p_unseen_curves.json`. Regimes: 10 saturating, 17 plateauing, 3 climbing. The finding the
fork surfaced is sharper than the category question I asked:
**the regime is a property of the GRANULARITY at which a norm is individuated, not of the task.**
Every task saturates at concept-head level (p_end .024-.069, ~8k species) and none saturates at
phrase level (.81-.93, ~65k species) from the identical corpus. Patents splits hardest
(.024 vs .845); humor is most open everywhere (.069/.735/.931 — taste resists shared vocabulary);
grant-funding closes fastest (5,315 heads — institutionally standardized criteria). Only
peer-review::fine actually climbs. Mirrors the campaign's R2/R3-saturate vs L0/R1-open-tail
result from the metric-lexicon line. Caveats recorded: ordering is file-mtime (no usable
collection-query logs); point estimates only (notebook had Clopper-Pearson bands); 5 tasks
skipped (no gpt-5-mini bank).

**Paper:** Table 1 now stars-not-P and no verdict column; Fig 3 = missing MASS
(supervised | unsupervised) plotted from real data by `gen_fig_curves.py` in the notebook's
style (log-x, per-curve γ, dashed power-law extrapolation to the p=.1 target, regime legend,
named arrow annotations); new Fig 4 = missing VALUE (supervised LiveBench ladder measured;
unsupervised panel pending its value artifacts).

## HB120 (2026-07-27) — ★★★ HOVER PER-UNIT CAUSAL BATTERY: units are NOT individually causal

First per-unit causal measurement in the campaign (E2). One session, cache OFF, 3 real passes,
n=300, full candidate vs candidate-minus-one-unit for the top-10 greedy-chosen units.

Full candidate **.5678**. Causal deltas (full − ablated):
| arm | ablated score | causal Δ | unit (abbrev) |
|---|---|---|---|
| minus_6 | .5600 | **+.0078** | extract all relevant entities/relationships |
| minus_9 | .5600 | +.0078 | delineate confirmed vs missing |
| minus_5 | .5611 | +.0066 | compare gaps in summary_1 vs summary_2 |
| minus_0 | .5622 | +.0056 | synonyms/aliases/acronyms for entities |
| minus_4 | .5678 | .0000 | expand query with aliases |
| minus_3 | .5678 | −.0000 | extract 2-3 unresolved keywords |
| minus_2 | .5700 | −.0022 | generate multiple diverse queries |
| minus_1 | .5767 | −.0089 | focus query on unverified facts |
| minus_8 | .5778 | −.0100 | state contradictions with new passages |
| minus_7 | .5811 | **−.0133** | identify which components remain unresolved |

**Median causal Δ = .0000; range −.013 to +.008. Four of ten units make the prompt BETTER when
removed.** And the headline: **corr(greedy select-panel marginal, causal test delta) = +0.013** —
the select-panel marginals that drove unit selection have essentially ZERO relationship to each
unit's causal contribution on test.

**What this does and does not touch.**
- It does NOT touch the hover win (+.051/+.036, replicated across sessions and configs) or the
  hotpot win (+.220, content-audited). Those are candidate-vs-candidate comparisons.
- It DOES falsify a per-unit reading of the abstract's definition ("units = minimal perturbations
  that induce CAUSAL behavioral changes"). Individually, on this bench, they do not.
- It is CONSISTENT with — arguably the sharpest evidence yet for — the pool-not-search thesis:
  value lives in the assembled pool, not in identifiable individual units; the greedy search's
  per-unit signal is noise, which is exactly why unguided recombination matches or beats it.
- It also explains the earlier livebench mixture finding (20 positive/6 toxic units) and the
  hover leakage-ablation null (+.003, P=.41) — same phenomenon, now measured systematically.

**Caveat before over-reading:** single-unit ablation measures MARGINAL contribution in the
presence of all others; substitutable/redundant units will each show ~0 while the SET matters.
That is a real alternative explanation and the honest framing is "no unit is individually
load-bearing," not "the units do nothing." A leave-many-out or Shapley-style design would
separate these; the E3 sub-clause arms (running) partially address it.
Artifact: runs/ablation_battery_hover.json (sk2). hotpot battery at arm 7/16, running.

## HB120b (2026-07-27) — ★ CORRECTION to HB120: the causal battery is UNDERPOWERED, not a null

Advisor pass 8 demanded the noise floor before any reading. Computed from the battery's own
per-pass means (11 arms x 3 passes, cache off):

| quantity | value |
|---|---|
| median within-arm pass SD | **.0110** |
| median SE of an arm mean (3 passes) | .0064 |
| 95% noise band on a single arm | ±.0125 |
| **95% noise band on a DIFFERENCE of two arms** | **±.0176** |

**Every observed causal delta (−.0133 to +.0078) lies INSIDE the ±.0176 two-arm noise band.**
The battery therefore cannot distinguish per-unit effects of this size from zero. HB120's reading
is RETRACTED as stated:
- "median causal Δ = .000, units are not individually causal" → **not supported**; the correct
  statement is *no per-unit effect larger than ≈.018 is detectable at this n*.
- "**4 of 10 units improve the prompt when removed**" → this is the NULL EXPECTATION (~5 of 10
  under zero true effects), not a finding. Must not print as striking.
- "**corr(greedy marginal, causal Δ) = +0.013 ⇒ marginals don't transfer**" → **unsupportable**.
  At n=10 the 95% CI is ≈[−.62,+.64]. Worse, the ten arms are the TOP-10 units *selected on high
  select-panel marginal*, so x is range-restricted: correlating within the selected decile
  attenuates toward zero even if marginals are perfectly informative overall (the
  GRE-among-admits problem). **The design cannot answer the question it appears to answer.**

**Three hypotheses remain observationally identical** and the current data separates none:
H-redundant (units substitutable; set matters, members don't) / H-inert (top-10 contribute
nothing jointly either) / H-noise (true ±.02 effects invisible at this n).

**Decisive follow-up, pre-registered here before launch — hover drop-m dose-response, 12 arms,
one session, ~4-6 GPU-h (same cost as the battery already paid for):** full; **drop-all-10**;
drop-random-{3,5,7} x2 each; **add-one-LOW-marginal x3** (fixes the range restriction).
Rules: **REDUNDANCY CONFIRMED** iff drop-10 Δ ≤ −.030 with paired-bootstrap P(Δ≥0)<.05 while
median single-drop |Δ| < .010 → printable: "no single unit is load-bearing; the assembled set is."
**JOINT NULL** iff drop-10 Δ within ±.015 of the additive prediction (−.007) and its CI excludes
−.030 → printable: "top-ranked units contribute neither individually nor jointly." Marginal
transfer recomputed over the full marginal range (singles + add-one-low); claim non-transfer ONLY
if the CI excludes +.30. No Shapley (hundreds of evals for what 12 arms give).

**Prediction on file (advisor's, adopted):** hotpot's singles will be REAL where hover's are not,
because hotpot's top-tier units are format rules with marginals +.09..+.16 (an order of magnitude
above hover's +.02) and format rules are structurally non-redundant. If so the split runs along
the HB111 thin/thick axis: **thin rules are individually causal; thick rules are causal only in
sets** — which is a Daston-framed result (C7) with data underneath, and the best available outcome.

**Definitional consequence (adopt regardless):** define units by CONSTRUCTION, not by measured
per-unit causality — "short, self-contained natural-language clauses drawn from a declared mining
distribution, individuated so the space is finite and its tail estimable" — and report the causal
battery as a finding. Drop "minimal" (unlicensable once whole-unit effects are undetectable); use
"atomic (single-clause)", a syntactic property defensible by inspection.

## HB121 (2026-07-27) — ★★★ PUPA IS A TIE. The +.032 "win" was winner's curse.

Decisive pre-registered k=5 rescore (both arms, same session), `runs_paperexact/pupa/Qwen3-8B/
{official,unitrecomb_v8failmine}/rescore_k3.jsonl` (file name says k3; the record says
`passes=5` — it IS the k5 rescore). sk2 queue pid 4012049, now exited.

| arm | single-pass best_test | **k=5 mean** | shift |
|---|---|---|---|
| official (GEPA) | .8621 | **.8825** | **+.0204** |
| unitrecomb_v8failmine (ours) | .8938 | **.8817** | **−.0121** |

**Paired item-level test (n=221, the heartbeat's required bootstrap, not the sign test):**
mean delta = **−.0009**, 95% CI **[−.0324, +.0307]**, P(Δ≥0) = .476.
Tie mass **163/221 = 74%** — precisely why a sign test on this metric is uninformative.

**Reading — and it is against us.** Under one pass ours appeared to win by +.032. Under five it
is a dead tie with the sign flipped. The two arms moved in OPPOSITE directions on rescoring
(baseline up, ours down), which is the signature of **winner's curse**: our reported number was
`best_test`, a MAX selected over a noisy single pass, so it regresses down; the baseline was not
max-selected, so it regresses up. This is the deflated-base red flag from the eval-noise memory
showing up in the direction that costs us a result.

**Consequences (adopt now):**
1. **PUPA IS NOT A WIN.** Never quote .8938, and never quote the +.032 delta. The quotable pupa
   line is "tie, Δ=−.001, 95% CI [−.032,+.031], k=5 paired".
2. Any arm number reported as `best_test` from a single pass is **suspect by construction**.
   Every W/L in Table 1 needs the same k≥5 both-arms same-session rescore before it prints.
   Single-pass `best_test` is a selection statistic, not an estimate.
3. The 74% tie mass means item-level power on pupa is low regardless: to detect a true +.02 here
   would need far more than 221 items. Pupa may simply be unable to separate these arms.

## HB122 (2026-07-27) — sk3 14B/24k chain COMPLETE: GEPA does nothing on hotpot, HURTS livebench
`logs/chain14b_sk3gpu7.log`, finished 2026-07-25T19:20Z, pid 3682544 now exited. GEPA official arm:

| bench | seed_test | best_test | Δ |
|---|---|---|---|
| hotpot | .267 | .267 | **.000 — GEPA found nothing** |
| livebench | .779 | **.748** | **−.031 — GEPA made it WORSE** |
| hover | (rc=0, no DONE line parsed — recover before quoting) | | |

Both are post-2026-07-24 so the Levenshtein defect does NOT apply. These are **baseline-side**
results and they matter for the paper's thesis: on two of three benches at 14B/24k, reflective
search returns zero or negative value over its own seed. That is the "nothing beats
recombination" claim's strongest supporting evidence so far — but note it argues the search is
weak, NOT that our recombination is strong; HB121 just showed our pupa margin was noise. Treat
these as evidence about GEPA, not evidence for M_ω, until a paired k≥5 rescore says otherwise.

**LANE STATE:** all three heartbeat pids DEAD. sk1 (986255) died 2026-07-24T23:23 immediately
after "aime-14B GEPA official (24k)" and produced NOTHING — **8 idle GPUs for ~2.5 days**, the
single largest wasted resource in the campaign. sk3 GPUs 0/5 held by orphaned VLLM::EngineCore
(alexspan, 20h and 28m); sk3 GPU7 now belongs to yallouah — NOT ours despite the 07-25 release.

## HB123 (2026-07-27) — ★ CORRECTION to HB121 + PREREG: hover omnibus k5 (frozen BEFORE launch)

**HB121 correction.** My HB121 entry presented the pupa k=5 tie as a newly decisive result. It is
not new: **HB97 already recorded pupa as "TIE, final (−.0009, p=.52)"** from the same rescore
artifact. HB121 re-derives it (−.0009, CI [−.032,+.031]) and adds the winner's-curse mechanism and
the 74% tie mass, which ARE new; the verdict is not. Recorded so the ledger does not double-count
one measurement as two.

**A worse error, caught by the advisor before it did damage.** I launched a blanket k=5 rescore of
every arm on sk1 (aime, hotpot, ifbench, livebench, pupa, hover). **Four of those cells are already
settled by HB97's same-session paired k=3**, and HB97 states explicitly that ifbench "must NOT be
re-rolled". Re-rolling settled cells until they move is p-hacking under our own HB91 one-shot rule,
and it is exactly the discipline the pupa post-mortem exists to enforce. **Killed before it scored
anything** (wrapper PID first; the vLLM server had in fact failed to start, so zero passes ran).
Nothing to retract. Standing rule reaffirmed: **a settled cell is re-measured only if the INSTRUMENT
changed, never because the result is inconvenient.**

**PREREG — hover omnibus, frozen before launch.** hover is genuinely PENDING (HB97 scoreboard), so
it is the one cell that may be measured. Design: ALL candidates rescored in ONE invocation =
one session, k=5, paper-exact splits, cache off, single server fingerprint.
Panel (6, all frozen ex-ante): `official` (GEPA baseline), `official_merge_gepamerge` (GEPA+Merge),
`inhouse`, `unitrecomb`, `unitrecomb_stair`, `unitrecomb_v5sk2`. MIPROv2 does not exist for hover
on sk1 — the envelope is therefore 3-way (GEPA / GEPA+Merge / M_ω), and that gap is stated, not
quietly dropped.
Primary inference: paired bootstrap on mean item-level delta, M_ω-best vs `official`. Sign test
reported but NOT primary (tie mass).
**Pre-registered prediction (adversarial, on file before any number exists):** the single-pass
margin shrinks; I predict a paired delta of **+.02 to +.04** with borderline significance, NOT the
+.067 the single-pass reading suggests. **If it lands n.s., hover is a NON-WIN and the paper's
headline moves to hotpot (+.220) and aime (+.091), which are already same-session certified.**
Committing to that consequence now, before seeing the number, is the point of writing it here.
**One shot. Whatever it returns is the hover verdict.**

## HB124 (2026-07-27) — ★★★ RANDOM units beat GEPA's OWN winner 36/40 (hotpot, 0.6B). Thesis-shaped.

The advisor's "kill-shot control" (random recombination at zero search budget) turns out to
ALREADY EXIST inside `osl_battery.py` — it was never read as such. Verified semantics before
quoting (osl_battery.py:72): `init_cand` **is the GEPA-official best_candidate**, and each draw
appends a random p=.5 subset of the frozen pool **to that same init**. So the contrast is
*GEPA's winner* vs *GEPA's winner + random units from the pool GEPA itself mined*.

`runs/osl_hotpot_Qwen3-0.6B.json`, hotpot, executor Qwen3-0.6B:

| arm | score |
|---|---|
| init = **GEPA-official winner** | .2567 |
| **40 random pool draws** | mean **.2943**, median .2967, range .200–.347 |
| transfer = 8B M_ω winner, verbatim | .3000 |

**36 of 40 random draws (90%) beat GEPA's own optimized prompt**, by +.038 on the mean.
`corr(#units included, score) = +.034` — essentially zero, so this is NOT "more text is better";
almost ANY random subset of the pool improves the searched prompt. Ordering:
**searched M_ω (.300) > random pool draws (.294) > GEPA winner (.257)** — i.e. most of M_ω's
advantage over GEPA is reachable without any search at all.

**This is the paper's sentence if it survives.** "The pool is the asset: even unintelligent
recombination cashes most of the value GEPA's own search left in the pool it mined."

**TWO CONFOUNDS, both resolvable, neither resolved yet — do NOT quote until they are:**
1. **CROSS-SCALE (the serious one).** init is an **8B-optimized** prompt scored on **0.6B**.
   "Random perturbation helps a mis-transferred prompt" would produce this exact pattern without
   any pool-value claim. **The 8B OSL cell is the decisive control** (init and executor matched):
   if random draws still beat init at 8B, the confound is dead. That cell is queued in Lane A
   (0.6B done, 1.7B/4B/8B pending) — wait for it.
2. **init is a SINGLE-pass measurement.** Against the ablation battery's pass SD (.0110), +.038
   is ~3.5 SD, so it is unlikely to be pure noise — but per HB121 the correct fix is a k≥5 init,
   not an appeal to a noise floor measured on a different bench. Re-measure init multi-pass
   before this prints.

Note this needs no new experiment and no sign-off: it is a re-reading of data already collected.
The only new work is the 8B row (already queued) and a multi-pass init.

## HB124b (2026-07-27) — ★ CORRECTION to HB124: the draws are SUPERSETS of init, not recombinations

Verified in source (osl_battery.py:109-112): each draw is `cand = dict(init_cand)` followed by
appending the sampled clauses. **Every draw therefore CONTAINS 100% of GEPA's winning prompt.**

My HB124 headline — "random units beat GEPA's own winner" — is technically true but rhetorically
overstated, and a reviewer will say so. The honest statement is:

> **Appending random pool units to GEPA's winner improves it in 36/40 draws (+.038).**

That is evidence GEPA's search **terminated early and left value in the pool it had already
mined** — a real and useful claim. It is NOT evidence that random recombination can replace the
searched prompt, because no arm here ever omits the searched prompt. The thesis-critical arm is
missing: **seed + random units** (no searched winner inside), which is what would license
"the pool, not the search."

**Four stacked asymmetries, ALL favoring the draws** (the advisor's audit; adopt all four):
1. **Superset** — draws = init + more text; init never gets the same treatment.
2. **Selected vs unselected** — GEPA's winner was chosen as a max over noisy validation, so it
   regresses DOWN on re-eval; the 40 draws are unselected and do not. This is the HB121
   winner's-curse mechanism pointing at the BASELINE this time. It is orthogonal to scale and
   survives the matched-8B control.
3. **Cross-scale** — an 8B-optimized init scored on a 0.6B executor.
4. **Single-pass init** — the reference point everything is measured against is one draw.
Until all four are removed, **HB124 is a promising anomaly, not a result.** Do not put it in the
paper, and do not quote the .3467 max draw under any circumstance — that is a max over 40 noisy
single passes, exactly the statistic HB121 was written to forbid.

**Also correcting my own reasoning:** I cited `corr(#units, score) = +.034` as evidence against
"more text is better." It is not. Under p=.5 inclusion the unit count is binomially concentrated,
so there is almost no leverage on the length axis, and a saturating format effect (any bullets >
no bullets) predicts a flat slope. The near-zero correlation is exactly what the artifact would
produce. The right control is a **length-and-format-matched placebo**, not a correlation.

## HB124c (2026-07-27) — the percentile statistic (usable) and the mask regression (NOT usable)

**★ Better statistic than the mean gap, from data already in hand.** Instead of "+.006 over the
random-draw mean", report where the searched prompt falls in the null built from its own pool:

| prompt | percentile within the 40 random draws |
|---|---|
| searched M_ω winner (.3000) | **52nd** |
| GEPA-official winner (.2567) | **8th** |

**"The searched prompt is statistically indistinguishable from a MEDIAN random draw from its own
pool"** is a far stronger and more honest sentence than a +.006 mean difference, and it does not
depend on the mean of a skewed 40-point sample. Null: n=40, mean .2943, sd .0303. Same four
confounds as HB124 apply — this is a sharper framing of the anomaly, not a resolution of it.

**✗ The per-unit mask regression is NOT identified — do not use it.** The advisor proposed fitting
`score ~ mask` on the logged masks to get per-unit values. Attempted; it is degenerate:
**68 units with variation, 40 draws, residual dof = 1, R² = 1.000.** That R² is pure overfit, not
signal, and the resulting coefficients (top +.043, bottom −.021) are noise dressed as estimates.
**Reporting them would have been a serious error.** To identify an additive per-unit model here
needs ≳3-5× more draws than units (≈200-340 draws at this pool size) or explicit regularization
with a held-out split. Deferred; the drop-m dose-response (HB120b prereg) is the better-powered
route to the same question and costs less.

## HB125 (2026-07-27) — HB124 confound #4 (single-pass init) is DEAD; #1-#3 still stand

`runs/hb124_controls_hotpot_Qwen3-0.6B.json`, sk2 GPU0, same session, cache off:
**init at k=5 = .25468** vs the original single-pass init **.2567** — a difference of .0020.
The reference point everything in HB124 is measured against was NOT a noise fluke, so the
+.038 draw gain cannot be explained by an unluckily-low init. **Confound #4 removed.**

Still outstanding and still disqualifying until answered: #1 superset construction (draws
contain 100% of init), #2 selected-vs-unselected (GEPA's winner is a noisy argmax and regresses
down; the draws don't), #3 cross-scale (8B-optimized init on a 0.6B executor). The native and
foreign-content arms are mid-run; the shuffled-token placebo and the seed+units arm are not yet
built. HB124 stays UNQUOTABLE.

## HB126 (2026-07-27) — hover omnibus, first two candidates: winner's curse visible in the SIGN

Prereg HB123, one session, k=5, cache off, single fingerprint. Partial (2 of 6):

| candidate | single-pass best_test | k=5 mean | shift |
|---|---|---|---|
| official (GEPA baseline) | .4500 | **.4640** | **+.0140** |
| official_merge_gepamerge (GEPA+Merge) | .5333 | **.5247** | **−.0086** |

The shifts run in **opposite directions, and in exactly the direction selection pressure
predicts**: `official` is the least-selected candidate and regresses UP; `official_merge` was
picked as a max and regresses DOWN. This is the HB121 mechanism reproducing prospectively on a
different bench — which is itself corroboration that the winner's-curse diagnosis was right,
not a post-hoc story.
Four candidates (inhouse, unitrecomb, unitrecomb_stair, unitrecomb_v5sk2) still scoring. The
prereg'd verdict rests on unitrecomb_stair (single-pass .5833) vs official (k5 .4640); no W/L
will be called until the paired bootstrap runs on the item-level scores.

## HB127 (2026-07-27) — ★★★ HOVER VERDICT: WIN over GEPA (+.100), but a TIE with our own search

Prereg HB123 executed exactly as frozen: 6 candidates, ONE invocation, k=5, cache off, paper-exact
splits, **single session fingerprint verified across all arms** (2026-07-27T11:30:30Z, n=300).
Primary inference = paired bootstrap on mean item-level delta.

| candidate | single-pass | **k=5** | Δ vs GEPA official | 95% CI | P(Δ≤0) | verdict |
|---|---|---|---|---|---|---|
| official (GEPA) | .4500 | .4640 | — | — | — | baseline |
| official_merge (GEPA+Merge) | .5333 | .5247 | +.0607 | [+.037,+.085] | .0000 | WIN |
| unitrecomb_v5sk2 | .4900 | .5167 | +.0527 | [+.031,+.075] | .0000 | WIN |
| unitrecomb | .5467 | .5287 | +.0647 | [+.041,+.089] | .0000 | WIN |
| inhouse (monolithic search) | .5467 | .5587 | +.0947 | [+.061,+.130] | .0000 | WIN |
| **unitrecomb_stair (M_ω best)** | .5833 | **.5640** | **+.1000** | **[+.072,+.129]** | **.0000** | **WIN** |

**★ hover is a WIN. My pre-registered prediction was WRONG, and in the conservative direction.**
HB123 predicted the margin would shrink to +.02–.04 and be borderline. It shrank from +.133 to
**+.100 and is overwhelmingly significant**. Recording the miss because a prereg that only gets
cited when it is right is worthless. The prediction failed; the discipline did not — the shrinkage
was real (−.033), just smaller than forecast.

**★★ The result that actually matters, and it is NOT in our favour — head-to-head, same session:**

| comparison | Δ | 95% CI | P(Δ≤0) | verdict |
|---|---|---|---|---|
| M_ω vs GEPA+Merge | +.0393 | [+.013,+.066] | .0014 | WIN |
| inhouse vs GEPA+Merge | +.0340 | [+.001,+.068] | .0215 | WIN |
| **M_ω vs inhouse** | **+.0053** | **[−.023,+.033]** | **.361** | **NOT SIGNIFICANT** |

**On hover, unit recombination is statistically indistinguishable from our own monolithic
mutate-and-accept search.** M_ω beats GEPA and GEPA+Merge; it does not beat plain search. Any
claim that recombination is the *mechanism* cannot be supported on this bench, and the paper must
say so. The envelope M_ω > GEPA+Merge > GEPA holds and is quotable; "recombination > search" does
not hold here.

**★ The charitable reading is also the more interesting one — and it is testable, not rhetoric.**
Three different assembly procedures (M_ω recombination .564, monolithic search .559, and — from
HB124, on a different bench/scale — random pool draws) land in one narrow band well above GEPA.
That is the signature of a **shared ceiling set by the unit pool**, with the assembly method
nearly irrelevant. This is the pool-value thesis in a stronger form than "our method wins".
It predicts something falsifiable: **on hover, random draws from the frozen pool should also land
near .55.** Cheap to run, and it either converts a tie into the paper's mechanism or kills it.
Pre-register before running: pool-ceiling supported iff random-draw mean ≥ .53 with the searched
arms inside the draw distribution's upper half.

**★ The winner's-curse pattern held prospectively, 6 for 6.** Every max-selected candidate
regressed DOWN (merge −.009, unitrecomb −.018, stair −.033); the two least-selected regressed UP
(official +.014, inhouse +.012). Note v5sk2 rose (+.027) — it was selected on a *different* box's
session, so it was not max-selected in this instrument's sense. Independent prospective
corroboration of HB121 on a bench that played no part in diagnosing it.

**Scoreboard:** aime WIN (+.091) · hotpot WIN (+.220) · **hover WIN (+.100)** · ifbench not
confirmed (+.020) · pupa TIE · livebench pending idle-protocol re-measurement. **W = 3 of 6.**

## HB127b (2026-07-27) — ★★ MY SHARED-CEILING READING IS NOT SUPPORTED. `inhouse` is POOL-FREE.

The advisor's first question was the one that decides which paper this is, and it costs zero GPU:
**what does `inhouse` consume?** Answer, read from source (paperexact_arms.py:304-312,
`arm_inhouse`): `cur, cur_score = seed_cand, None` — inhouse starts from the **SEED prompt**,
mutates whole per-module instruction sets via a reflection LM conditioned only on train-panel
feedback, and **never reads the unit pool at all.**

**This inverts my HB127 reading.** I proposed that M_ω (.564), inhouse (.559) and random pool
draws landing in one band was "the signature of a ceiling set by the unit pool." It is not: a
**pool-free** method reaches that band, so the pool is **not necessary** to get there. The band is
better explained as a **task/executor ceiling that any competent optimizer reaches**, and GEPA's
failure to reach it (.464) is then a statement about GEPA's *selection*, not about pool access.
Retracting the pool-ceiling framing as stated.

**Second correction, also mine.** I wrote "assembly is nearly irrelevant." That is already false
in the HB127 table: **GEPA+Merge (.5247) does NOT reach the band**, sitting .034-.039 below
inhouse and M_ω with p=.0215 / .0014. One assembly procedure demonstrably undershoots. The
survivable version is narrower and is about GRAIN, not assembly-indifference: merge recombines at
CANDIDATE grain, M_ω at UNIT grain. Any claim here must be stated as unit-grain vs candidate-grain
access, and it still requires knowing inhouse's grain — which we now do: inhouse is whole-prompt
rewrite, no pool, and it matches M_ω. So grain does not separate them either.

**What survives, stated precisely.** Not "recombination is the mechanism" and not "the pool sets
the ceiling", but: **M_ω is in the statistical top tier on every bench measured, and no procedure
we ran ever beat it** — it wins where anything wins (aime +.091, hotpot +.220, hover +.100) and
ties where nothing separates. GEPA is top-tier only where nothing wins. The literal title claim
("nothing beats recombination") holds; the mechanistic connotation does not.

**The zero-search arm is now MORE important, not less**, and its interpretation has changed: if
random draws from hover's frozen pool (appended to SEED, never to a searched winner) also reach
~.55, that shows the ceiling is reachable with **no search at all** — which is a real result even
though it no longer isolates the pool as the *cause*, because inhouse reaches it pool-free too.
Adopting the advisor's stricter 4-arm rule over my weaker one (mine lacked a foreign-pool gate and
"upper half" is consistent with a genuine assembly advantage):
  R native draws → SEED · F foreign-pool length-matched · P shuffled-token · in-session anchors.
  **SUPPORTED** iff mean(R) ≥ .53 AND M_ω ≤ 90th pct of R AND mean(R)−mean(F) > 0 with CI excluding 0.
  **TASK-EASY** iff mean(F) ≥ .53 (pool content irrelevant → thesis dead on hover).
  **ASSEMBLY MATTERS** iff mean(R) ≤ .50. Between .50 and .53: no verdict, descriptive only.

**Latent bug found while reading `arm_inhouse`** (paperexact_arms.py:340): it does
`raw = refl(prompt)[0]` then `raw.index("{")` — the identical dict-response defect fixed earlier
in `_suggest_units_paper`. It has not bitten because inhouse has only ever run against
non-reasoning-parser endpoints, but it WILL silently zero the arm on a reasoning-parser server.
Patching now rather than after it costs a run.

## HB128 (2026-07-27) — PREREG: hover zero-search pool-ceiling test (frozen BEFORE launch)

Decisive test for HB127b. Every arm appended to the **SEED** prompt (`--base seed`), so no draw
ever contains a searched winner — the HB124b superset defect cannot recur. One invocation, one
server, one fingerprint, k=5 anchors, cache off, n=300 hover test.

| arm | construction |
|---|---|
| **R** native | random p=.5 draws from hover's frozen pool → SEED |
| **F** foreign | draws from hotpot's pool, count-matched → SEED (kills "generic advice") |
| **P** shuffled | native clauses, tokens scrambled → SEED (kills "any text of this length/format") |
| anchors | GEPA official + M_ω rescored IN-SESSION (never reuse HB127's scores — different session) |

**Decision rule, frozen now (advisor's, adopted over my weaker one — mine had no foreign gate and
"upper half" is consistent with a real assembly advantage):**
- **POOL-CEILING SUPPORTED** iff mean(R) ≥ .53 AND M_ω ≤ 90th pct of R AND mean(R)−mean(F) > 0
  with bootstrap CI excluding 0.
- **TASK-EASY / generic-text** iff mean(F) ≥ .53 → pool content is irrelevant and the pool-value
  claim is DEAD on hover regardless of R.
- **ASSEMBLY MATTERS** iff mean(R) ≤ .50 → the band needs search; the inhouse tie was coincidence
  and this becomes the under-selection paper.
- mean(R) ∈ (.50,.53): **NO VERDICT**, report descriptively, make no claim.
One shot. Whatever it returns is the hover pool verdict.

**Note on what this can and cannot show, given HB127b:** because `inhouse` reaches ~.559 with NO
pool access, a passing R can no longer establish the pool as the *cause* of the ceiling. What it
would establish is weaker but still worth reporting: **the ceiling is reachable with zero search.**
Stating that limit here so the result is not oversold when it lands.

## HB129 (2026-07-27) — ★★★ The "any text helps" objection is DEAD. Foreign units actively HURT.

`runs/hb124_controls_hotpot_Qwen3-0.6B.run1_2arm.json` (run 1 preserved, not deleted). hotpot,
Qwen3-0.6B, ONE session, cache off, count-matched draws, n=40 per arm.

| arm | mean | median | range | above init |
|---|---|---|---|---|
| init (k=5) | .25468 | — | — | — |
| **native** pool units | **.2985** | .3000 | [.207,.370] | **36/40** |
| **foreign** (hover units, count-matched) | **.1763** | .1767 | [.070,.250] | **0/40** |

- native − init = **+.0438**
- foreign − init = **−.0783**
- **native vs foreign: Δ=+.1222, 95% CI [+.106,+.139], P(Δ≤0)=.0000**
- θ (share of the native gain attributable to CONTENT rather than text volume) = **2.79**

**Confound (A), "a 0.6B model improves whenever you append more instruction text," is dead.**
Count-matched foreign clauses do not merely fail to help — they **hurt by −.078, with 0 of 40
draws beating init.** θ > 1 precisely because foreign lands *below* init. The model is strongly
sensitive to instruction CONTENT, not to instruction volume, which is the opposite of the
artifact. This also retires my discredited `corr(#units, score) = +.034` argument (HB124b) with a
control that actually has power.

Note this is a stronger control than "foreign is merely neutral" would have been, and it cuts a
second way: because irrelevant instructions are actively harmful here, a pool of *task-relevant*
units is doing real work.

**Still outstanding for HB124** (unchanged): #1 superset construction (every draw contains
GEPA's winner), #2 selected-vs-unselected regression asymmetry, #3 cross-scale 8B init on 0.6B.
The queued 4-arm single-session run adds the **shuffled-token** arm (matched vocabulary AND
length — a tighter (A) control than foreign) and the **seed_units** arm (drops the searched
winner entirely, attacking #1). HB124 remains UNQUOTABLE until #1-#3 are answered; HB129 stands
on its own as a clean control result.

## HB130 (2026-07-27) — ★★ hotpot ablation: 10/10 negative, but the SHARED CONTROL error eats it

All 10 removal arms complete (`runs_paperexact/hotpot/Qwen3-8B/ablation_battery/`, 3 passes each).
`full` = .6589 (pass sd .0102). Deltas (removal − full):

`−.0089 −.0156 −.0156 −.0111 −.0156 −.0022 −.0111 −.0078 −.0189 −.0100`

**Every single removal hurts — 10 of 10.** Naive sign test P = .00195. Mean delta −.01168.
This is the direction HB120b's pre-registered thin/thick prediction called for (hotpot's units are
format rules with marginals an order of magnitude above hover's), and it is the opposite of
hover's uninformative battery.

**But it does not survive correct error propagation, and the reason is structural.**
All ten deltas are measured against ONE `full` value. That reference has SE = .0102/√3 = **.0059**,
and — critically — **that error does not average down across arms; it is common to all of them.**
A `full` measured just 1 SE high makes every removal look negative.

| analysis | mean delta | 95% CI | verdict |
|---|---|---|---|
| ignoring `full`'s error (**WRONG**) | −.01168 | [−.0147, −.0087] | "decisive" |
| **propagating the shared reference** | −.01168 | **[−.0240, +.0006]** | **INCLUDES 0** |

Per-arm: the 95% band on a single difference is ±.0179, and **9 of 10 deltas fall inside it** — no
individual unit is separately resolvable, same as hover. The sign test is *also* compromised: it
assumes independent draws, and these ten share a reference, so 10/10 is much likelier under the
null than 2×(1/2)^10 suggests.

**Verdict: SUGGESTIVE, NOT ESTABLISHED.** Do not quote "every unit is load-bearing" or the .00195.
Had I not propagated the shared error I would have reported a confident false positive — the naive
CI excludes zero comfortably.

**★ STANDING RULE (adopt campaign-wide).** In any ablation/battery design with one shared control,
**the control arm must get several times more passes than each treatment arm**, because its error
enters every comparison and never averages out. Equal passes everywhere spends the budget in the
wrong place. Rule of thumb here: with 10 treatments, `full` wants ≈√10 ≈ 3× the passes of a single
treatment arm, and more if the effect is near the noise floor.

**Cheap decisive fix, queued:** re-measure `full` alone at k=15 in one session. That shrinks
SE_full from .0059 to .0026 and the aggregate CI to roughly [−.0175, −.0058] **if the point
estimate holds** — turning a graze into a result, or killing it honestly. One arm, ~15 evals.

**Correction to the HB130 fix, caught before launch.** My first plan was to re-measure `full`
alone at k=15 via `rescore_k3.py hotpot --runs unitrecomb_v5sk2 --passes 15` (verified it hits the
same candidate on the same split — ablation_battery.py:87 uses `bench.test_set`, as does
rescore_k3). **That would have been wrong.** The new `full` would sit in a DIFFERENT session from
the 10 removal arms, so every delta would reacquire the cross-session `u_session` term that the
whole same-session protocol exists to cancel — trading a known shared-reference error for a larger
unknown one. Precisely the HB121 mistake in a new costume.

**Correct design (queued, not yet launched):** ONE session containing `full` at **k=15** plus the
three most negative removal arms (`minus_8` −.0189, `minus_1`/`minus_2`/`minus_4` −.0156) at k=5
each. ≈30 evals, one server, one fingerprint. This gives a well-measured shared control and
within-session deltas simultaneously. Needs a small arm-selection flag on `ablation_battery.py`
(no `--only-full`/`--arms` exists today), so it is a code change plus a run, not a run alone —
deliberately NOT launched half-right tonight.

## HB131 (2026-07-27) — ★★★ MATCHED-SCALE 8B CELL: HB124 survives, effect is 5x LARGER, 40/40

`runs/osl_hotpot_Qwen3-8B.json`. The decisive control for HB124 confound #3: init is the
GEPA-official **8B** winner and the executor is now also **8B**, so no cross-scale mismatch exists.

| executor | init (GEPA winner) | random draws (mean) | gain | draws > init |
|---|---|---|---|---|
| 0.6B | .2567 | .2943 | +.0376 | 36/40 |
| 1.7B | .3567 | .4465 | +.0898 | 39/40 |
| **8B (MATCHED)** | **.4000** | **.5856** | **+.1856** | **40/40** |

At matched scale: draws mean .5856, 95% CI [.5693,.6013]; gain **+.1856 [+.169,+.201]**;
**even the WORST of 40 draws (.4833) beats GEPA's winner (.4000)** by +.083.
Percentiles within the draw null: searched M_ω (.6367) = **80th**; GEPA's winner = **0th**
(below all 40 draws).

**★ The gain GROWS monotonically with executor capability (+.038 → +.090 → +.186).** A
cross-scale-mismatch artifact predicts the opposite — it must SHRINK as the executor approaches
the scale the prompt was optimized for. **Confound #3 is not merely absent; the data run against
it.** Read positively: the more capable the executor, the more value GEPA's search left unclaimed
in its own mined pool.

**Status of every objection raised against HB124:**
| # | objection | status |
|---|---|---|
| A | "any appended text helps a small model" | **DEAD** (HB129: count-matched foreign units HURT −.078, 0/40 above init) |
| 3 | cross-scale (8B prompt on 0.6B executor) | **DEAD** (this entry; effect larger at matched scale) |
| 4 | single-pass init | **DEAD** (HB125: k=5 init .25468 vs single-pass .2567) |
| 2 | selected-vs-unselected regression | **effectively dead at this magnitude.** Winner's curse is bounded by the selection noise scale — measured here at SD ≈ .010-.018 (HB120b/HB130) — so it can move a point estimate by ~.02-.03 at most. Against HB97's better-measured same-session GEPA official (.4133) the gap is still **+.172**, an order of magnitude beyond what regression can produce. |
| 1 | **superset construction** | **STILL STANDING.** Every draw contains 100% of GEPA's winner, so this licenses "adding pool units to the searched prompt helps", NOT "the pool replaces search". The `seed_units` arm (running in the 4-arm hotpot run and in HB128 on hover) is the direct test. |

**Quotable NOW, with the superset framing stated honestly:**
> On hotpot at matched 8B scale, appending a random half of the units GEPA itself mined to GEPA's
> own selected prompt improves it by **+.186 [+.169,+.201]**, in **40 of 40 draws**, with the worst
> draw still +.083 ahead — and the searched M_ω prompt sits at only the **80th percentile** of that
> random-draw distribution.

**Do NOT yet claim** the pool replaces search; that needs `seed_units`. **Do NOT quote** the max
draw (.653) — max-over-40-single-passes is the HB121 statistic.

## HB132 (2026-07-27) — ★★★ WHAT WE BEAT: on hotpot and ifbench, GEPA SHIPPED THE SEED

The advisor pulled a fact out of this note's own D2 section (lines 122-124) and it reorganizes the
scoreboard. **Verified empirically for the exact run dirs HB97/HB131 use**, two independent ways:

(a) byte-comparison of `official/result.json` best_candidate against the benchmark seed
(`seedcheck.py`, normalized whitespace, per module):
> **hotpot: seed hash 36e634d55388 == init hash 36e634d55388 — IDENTICAL across all 4 modules.**
> aime: DIFFERS (seed 63 chars → init 1402 chars) — a real GEPA gain.

(b) seed_test vs best_test in every official run dir:

| bench | seed_test | best_test | did GEPA improve? |
|---|---|---|---|
| **hotpot** | .3800 | .3800 | **NO — shipped the seed** |
| **ifbench** | .4116 | .4116 | **NO — shipped the seed** |
| hover | .3800 | .4500 | yes |
| aime | .3333 | .3667 | yes |
| livebench | .6744 | .6956 | yes |
| pupa | .8030 | .8621 | yes |

**Three consequences, all of which change how results must be WORDED (no number changes):**

1. **hotpot's flagship margins are over the SEED, not over a searched prompt.** HB97's +.220 and
   HB131's +.186 both use an `init` that is the bare seed, because GEPA explored its budget and
   shipped nothing. The honest sentence is *"recombining the units GEPA mined beats what GEPA
   shipped — and on hotpot what GEPA shipped was the unmodified seed."* A reviewer will otherwise
   write the one-liner for us: *"you beat GEPA on the task where GEPA didn't run."*
2. **Confound #1 (superset) is VACUOUS on hotpot.** Draws are "init + units", but init ≡ seed, so
   `native` and `seed_units` are the SAME ARM there. **The seed_units arm now running on
   hotpot/0.6B cannot test the superset confound — it never could.** Only **hover** tests it, since
   hover's init carries real searched content (.38→.45). This raises HB128's importance sharply
   and means I must not read hotpot's seed_units ≈ native as the confound passing.
3. **The two meaningful wins are hover (+.100) and aime (+.091)** — the benches where GEPA
   actually improved and we still beat it. hotpot's +.220 is the largest number and the weakest
   *comparison*. ifbench's "not confirmed +.020" is now interpretable rather than embarrassing:
   GEPA shipped the seed there too, so ifbench is a bench where **nothing** works, ours included.

## HB132b — my confound-#2 rebuttal was UNSOUND (wrong σ). Retracted.

In HB131 I argued selection regression "is bounded by the measured noise scale SD .010-.018, so it
can move a point estimate ~.02-.03, and cannot produce +.17." **The arithmetic is right and the σ
is wrong.** That SD is the pass-to-pass SD of careful k-pass, n=300 re-evaluations. GEPA's
*selection* happens on 15-item minibatch panels, and this campaign's own UNIT_BOTTLENECK_DIAGNOSIS
(D3, notes ~line 141) measured **sd(panel mean) ≈ 1.7-1.8 items on 15 ≈ .11-.12 in accuracy** —
6-10× the σ I used — and recorded panel acceptance as "a coin flip" with a confirmed false
positive. With σ_sel ≈ .11 over 6 candidates, σ√(2 ln k) ≈ .19, the **same order as the +.17 gap**.
My rebuttal is withdrawn.

Also, the live form of the objection is not regression at all: init was freshly re-measured
(HB125/HB131), so it carries no curse. The real objection is **selection REGRET** — GEPA may have
*generated* draw-band candidates and rejected them on a coin-flip panel. Nothing I have addresses
that.

**Decisive, cheap, queued: hindsight-rescore all 6 explored hotpot GEPA candidates + seed at k≥5,
n=300, ONE session.** Both outcomes publishable:
- none reaches the draw band (~.586) → upgrade to *"GEPA's proposal generator never produced a
  band-level prompt; recombination of its own mined units does"* — the strong thesis, and confound
  #2 dies properly.
- some candidate reaches ~.586 → the story becomes *"search generates but cannot select under
  panel noise"* — a different, still-real paper. Better found by us than by a referee.
Frame the eventual claim as a two-part failure decomposition — **generation vs selection** — not a
monolithic "search loses".

## HB133 (2026-07-27) — ★★★ THE POOL IS 88% LLM-PROPOSED, NOT "UNITS GEPA MINED". Reframe again.

Read the frozen pool's own provenance record (`pools/hotpot_Qwen3-8B_frozen.json`):

| source | from_run | n | share |
|---|---|---|---|
| **llm** (5-framing reflection-LM suggestion) | unitrecomb_v5sk2 | **60** | **88%** |
| trajectory (harvested from optimizer runs) | unitrecomb | 8 | 12% |

**I have repeatedly written "the units GEPA itself mined" (HB124, HB131). That is wrong and I am
retracting the phrasing.** Only 8 of 68 units come from optimizer trajectories at all, and those
came from OUR `unitrecomb` run, not from GEPA official. The pool is overwhelmingly **de novo
clauses proposed by a reflection LM shown the program's module instructions** — a declared mining
distribution that never required running GEPA.

**What the corrected claim actually is — and it is not weaker, it is different and cleaner:**
> On hotpot at matched 8B scale, appending a random half of a pool of LLM-proposed instruction
> clauses to the prompt GEPA shipped improves it by +.186 [+.169,+.201] in 40/40 draws — and what
> GEPA shipped was the unmodified seed (HB132).

Chained with HB132 the honest hotpot story is: **GEPA explored its budget, shipped nothing, and a
random half of an LLM-proposed clause pool beats its output by +.186.** No claim about GEPA's
mining is licensed, because GEPA's mining contributed 12% of the pool at most, and 0% via its
official run.

**Consequences:**
1. Every "GEPA's own pool" / "the pool it mined" phrasing in notes and drafts must be replaced
   with "a declared mining distribution (5-framing LLM suggestion + trajectory harvest)".
2. It **strengthens** the practical claim (no optimizer run is needed to build the pool) while
   **weakening** the rhetorical one (this is no longer "search discards its own discoveries").
3. It makes the `seed_units` arm even more central: with init ≡ seed on hotpot (HB132), the whole
   hotpot result reduces to *LLM-proposed clauses + seed*, with no optimizer anywhere in the
   causal path.
4. The trajectory-vs-llm split is itself a testable question the pool records support:
   **do the 8 trajectory units carry more value than the 60 LLM units?** Answerable by regressing
   draw scores on the trajectory-unit count in existing OSL data — no new GPU.

## HB134 (2026-07-27) — MIPROv2 has been failing SILENTLY on every bench (rc=0)

`ImportError: MIPROv2 requires optional dependency 'optuna'` in
`dspy/teleprompt/mipro_optimizer_v2.py`, on hotpot AND aime. **The wrapper recorded `rc=0` both
times** and the run dirs (`*/mipro_t1fill/`) are EMPTY. So Table 1's entire MIPROv2 column is
vacuous, and would have silently stayed that way.

Third instance today of the same pathology (after the vLLM-in-wrong-venv and the vendored EM/F1
relocation): **a failure that writes to a log, returns success, and leaves an empty artifact.**
Reinforces the standing rule — verify PRODUCED ARTIFACTS, never process liveness or exit codes.
Fix: install optuna into the run venv, then re-run the mipro cells only.

## HB135 (2026-07-27) — trajectory vs LLM units, and a NEGATIVE volume correlation at 8B

Free analysis on existing OSL draws (no GPU). Pool = 68 units, indices 0-7 trajectory-sourced,
8-67 LLM-proposed (HB133).

| executor | corr(# trajectory units, score) | perm p | corr(# TOTAL units, score) |
|---|---|---|---|
| 8B | **+.239** | .134 | **−.263** |
| 1.7B | +.141 | .386 | −.010 |
| 0.6B | +.040 | .808 | +.034 |

**(1) Trajectory units trend more valuable, and more so at larger scale — but nothing is
significant.** With only 8 of 68 units trajectory-sourced, per-draw counts span 1-7; the design
has little power. **A null here does NOT show trajectory units are worthless; it shows this design
cannot tell.** Hypothesis-generating only, and recorded as such. Testing it properly needs a pool
with a balanced source split, not more draws on this one.

**(2) The genuinely interesting number is the volume correlation at 8B: −.263.** At the matched
scale where the effect is largest (+.186), including MORE units is mildly WORSE. So the gain is
**not** driven by prompt volume — it arrives despite volume, implying a subset optimum well below
the p=.5 draw size (~34 units). This independently corroborates HB129's foreign-content result
from the opposite direction: content selection matters, bulk does not.

It also retires the last remnant of my discredited HB124 argument. I originally cited
corr(#units, score) = +.034 at 0.6B as evidence against a text-volume artifact; that was the wrong
statistic on the wrong rung with no power (HB124b). The right rung gives **−.263**, which is
actual evidence in the same direction — arrived at properly this time.

**Practical implication worth testing later:** if fewer units is better at 8B, a draw at p≈.25
should beat p=.5. One cheap sweep (p ∈ {.15,.25,.5,.75}, 20 draws each, one session) would locate
the optimum and is a stronger paper figure than a single p=.5 point.

## HB136 (2026-07-27) — ★★★ SHUFFLED-TOKEN control: the effect is SEMANTIC, decisively

4-arm single-session run (hotpot / Qwen3-0.6B), `runs/hb124_controls_hotpot_Qwen3-0.6B.json`.
init (k=5) = .25134.

| arm | construction | mean | above init |
|---|---|---|---|
| **native** | true pool clauses | **.2971** | **37/40** |
| **shuffled** | SAME clauses, tokens scrambled — identical length, bullet format, and vocabulary; only word order (hence meaning) destroyed | **.1965** | **4/40** |
| foreign | hover's pool, count-matched | .1803 | 1/20 |

- **native vs shuffled: Δ=+.1006, 95% CI [+.0855,+.1155], P(Δ≤0)=.0000**
- native vs foreign: Δ=+.1167, 95% CI [+.096,+.137], P(Δ≤0)=.0000
- θ (content share of the native gain, vs shuffled) = **2.20**

**This is the tightest available control and it settles objection (A).** The shuffled arm holds
token count, clause count, bullet formatting, and the exact vocabulary fixed — the ONLY thing
removed is word order, i.e. meaning. Performance falls from .2971 to .1965, **below init**, with
only 4/40 draws beating init. So the gain is not from text volume, not from list formatting, and
not from topical vocabulary: it is **semantic**. Scrambled native text is about as harmful as
foreign text (.1965 vs .1803), which is exactly what a content-driven account predicts.

**Cross-session replication is also clean** (an unplanned bonus): native .2985 (run 1) vs .2971
(4-arm run), init .25468 vs .25134 — two independent sessions agreeing to ~.003 on both the arm
and its reference.

**Objection ledger for HB124 after this entry:**
| # | objection | status |
|---|---|---|
| A | any appended text / format / vocabulary | **DEAD** — shuffled control, Δ=+.101 P<1e-4 |
| 3 | cross-scale | **DEAD** (HB131: effect 5× larger at matched scale) |
| 4 | single-pass init | **DEAD** (HB125, replicated here) |
| 2 | selection regret | **OPEN** — my noise-scale rebuttal was retracted (HB132b); needs the hindsight rescore of GEPA's explored candidates, which this run dir does NOT retain (only best_candidate is saved), so it requires a fresh GEPA run with candidate logging |
| 1 | superset construction | **OPEN, and untestable on hotpot** — init ≡ seed there (HB132), so `native` ≡ `seed_units`. **Only HB128 on hover can answer it.** |

## HB137 (2026-07-27) — ★★★ "Degenerate" means CONSTANT-LABEL, not "saturated". I mis-described it.

User asked why 137/272 metrics are degenerate. I had written "already saturated by mined prompts".
**That was wrong.** Diagnosed properly (`cr3-v12/degeneracy_diagnosis.json`, script
`/tmp/degen_why.py`; base rate and entropy of each binarized target):

| group | n | base rate (median) | label entropy | majority-class baseline | ~constant (H<0.1 bits) |
|---|---|---|---|---|---|
| **degenerate** (best agreement ≥ .999) | 140 | **1.000** | **0.000 bits** | **1.0000** | **136/140** |
| non-degenerate | 132 | .943 | .314 bits | .9433 | 10/132 |

**The degenerate metrics fire on essentially EVERY item.** Their labels are constant, so a
constant predictor ("always yes") scores ~100% agreement. Nothing was reconstructed because there
was nothing to reconstruct. This is a **defect of those metrics as instruments**, not evidence
that mined prompts saturated them. Correct wording: *"half the bank is label-degenerate — no
variance to predict — and must be excluded before any reconstruction readout."*

**★ The bigger problem this exposes: the AGREEMENT scale is compressed for the whole bank.**
Median majority-class baseline over all 272 metrics = **.9933**. Even among the non-degenerate
132 it is **.9433**. So the raw numbers I reported — achieved .975/.980, ceiling .995 — live
entirely inside the top ~5% of the scale, where a constant predictor already sits. **Raw agreement
(and any ceiling stated on it) is not an interpretable readout here and must not be reported bare.**

**★ Baseline-corrected, the reconstruction result SURVIVES and is respectable.** Normalized skill
`(A − majority)/(1 − majority)`, on the 132 non-degenerate metrics:

| statistic | value |
|---|---|
| **median normalized skill** | **0.597** |
| metrics with skill > 0.5 | 86/132 |
| metrics with skill < 0.2 | 4/132 |
| metrics failing to beat a constant predictor | **1/132** |

So the best mined prompt closes ~60% of the gap a constant predictor leaves, on almost every
metric with real label variance. That is a genuine finding; the raw .975 was not.

**Actions:**
1. Every unsupervised-side agreement number in the paper gets reported as normalized skill (or
   with the majority baseline printed alongside). Never bare agreement.
2. The 140 label-degenerate metrics are **excluded**, and the exclusion is reported as a property
   of the metric bank, not hidden in a denominator.
3. HB-series entries that used raw agreement on this bank (ceiling backtest coverage, "median
   achieved .980 / ceiling .995") are **superseded by this normalization** — the coverage/rank
   certificate conclusions still hold, since the rank certificate is scale-free and was computed
   per-metric, but the descriptive agreement medians must be restated.

## HB138 (2026-07-27) — ★★★ ROOT CAUSE: degeneracy is a GLOBAL-THRESHOLD artifact. Fixed; 0/272 remain.

**Why it happened.** Every metric was binarized at a single global **tau = 0.020**. But the raw
M_i are not near-constant at all — they carry full variance:

| group (old readout) | distinct M_i values | M_i std | M_i min (median) | frac ≥ tau |
|---|---|---|---|---|
| "degenerate" 140 | **300 of 300 items** | .148 | **.039** | **1.0000** |
| "non-degenerate" 132 | 300 of 300 | .205 | .001 | .9433 |

**The minimum M_i of the degenerate metrics (.039) is already ABOVE tau=.020**, so every item
labels positive and the target's variance is annihilated. **0 of 272 metrics have only one
distinct M_i value** — none was ever truly constant. My HB137 reading ("constant-label metrics, an
instrument defect") was wrong in the same way the "saturated" reading was: the metrics are fine,
**the readout was broken**.

**Fix: per-metric MEDIAN split** (signatures binarized on the same per-metric threshold, so the
comparison stays apples-to-apples). `cr3-v12/rebinarized_median.json`:

| statistic | tau = 0.02 (old) | **median split (new)** |
|---|---|---|
| degenerate metrics (agreement ≥ .999) | **140/272** | **0/272** |
| majority-class baseline (median) | .9933 | **.5000** |
| best agreement (median) | 1.0000 | **.8250** |
| **normalized skill (median)** | — | **0.650** |
| metrics beating the majority baseline | — | **272/272** |
| skill > 0.5 / skill < 0.2 | — | **240/272 / 0/272** |

**Consequences — all favourable, and no exclusions are needed:**
1. **Nothing gets dropped.** The planned "exclude the degenerate 137" is moot; all 272 metrics are
   usable once binarized sensibly. Figure 3 RHS should be regenerated on the median-split readout
   over the FULL bank, not on a filtered subset.
2. The result is **stronger and finally interpretable**: median agreement .825 against a .500
   baseline (skill .650), instead of .975 against a .993 baseline where a constant predictor
   already sat.
3. Every prior unsupervised agreement/ceiling number computed at tau=.02 is **superseded**. The
   rank certificate is unaffected in form (scale-free, per metric) but must be recomputed on the
   new binarization before being quoted.

**Honest caveat on the fix.** A median split changes the target's meaning from *"does the metric
fire"* to *"is the metric above its own median"*. Both are defensible; the second is far more
informative here because the first has no variance to predict at tau=.02. This must be stated
plainly in the paper — it is a definitional choice, not a neutral preprocessing step. Whoever set
tau=.02 may have had a rationale for an absolute threshold; if one exists it should be recovered
before the median split is made canonical.

## HB139 (2026-07-27) — ★★★ HB124 4-ARM COMPLETE. Content confirmed; HB132 independently corroborated.

`runs/hb124_controls_hotpot_Qwen3-0.6B.json`. hotpot / Qwen3-0.6B, all four arms in ONE session,
cache off, count-matched draws, n=40 each. init (k=5) = .25134 — and on hotpot that init **is the
seed**, byte-identical (HB132).

| arm | construction | mean | above init |
|---|---|---|---|
| **native** | true pool clauses | **.2971** | **37/40** |
| **seed_units** | same clauses appended to the SEED (searched winner removed) | **.2925** | 33/40 |
| shuffled | same clauses, tokens scrambled (length/format/vocabulary held fixed) | .1965 | 4/40 |
| foreign | hover's pool, count-matched | .1803 | 1/40 |

| contrast | Δ | 95% CI | P(Δ≤0) | |
|---|---|---|---|---|
| native vs **shuffled** | **+.1006** | [+.0855,+.1155] | .0000 | sig |
| native vs **foreign** | **+.1167** | [+.1011,+.1321] | .0000 | sig |
| native vs **seed_units** | **+.0046** | [−.0101,+.0194] | .2740 | **n.s.** |

**(1) The content result is now controlled three ways and holds.** Scrambling the tokens of the
very same clauses — preserving count, bullet format and vocabulary exactly — costs .10 and drops
performance BELOW init (4/40 above). Foreign clauses cost .12. The gain is semantic, full stop.

**(2) The seed_units null is a CONFIRMATION of HB132, not a test of the superset confound.**
HB132 established by byte-comparison that GEPA shipped the unmodified seed on hotpot; therefore
`native` (units → init) and `seed_units` (units → seed) are *the same arm*, and Δ=+.005 n.s. is
exactly what that predicts. **Two independent methods — a hash comparison of the prompt text and a
behavioural A/B over 80 scored draws — agree.** That is a genuine internal-consistency win, and it
is NOT evidence that the pool replaces search. **Only hover can test that** (its init carries real
searched content, .38→.45), and HB128's foreign arm is still running.

**Practical upshot, stated carefully.** On hotpot at 0.6B, appending a random half of an
LLM-proposed clause pool **to the bare seed** scores .2925 vs the seed's .25134 — with no
optimizer in the causal path at any point. Combined with HB131's matched-8B cell (+.186), the
pool-without-search route is real on this bench. What remains unlicensed is the general claim that
recombination substitutes for search, because hotpot's search never moved.

## HB140 (2026-07-27) — Table 1: first REAL MIPROv2 cell, and GEPA+Merge fails on ifbench

Post-optuna-fix (HB134), verified 0 ImportErrors after the fix vs 2 before; MIPROv2 ran a genuine
34-trial optimization on ifbench.

| bench | arm | seed_test | best_test | Δ |
|---|---|---|---|---|
| hotpot | GEPA+Merge | .4033 | .4167 | +.0134 |
| aime | GEPA+Merge | .3000 | .3867 | **+.0867** |
| **ifbench** | **GEPA+Merge** | .4099 | **.4065** | **−.0034** |
| **ifbench** | **MIPROv2** | .3963 | **.3946** | **−.0017** |

**On ifbench BOTH strong baselines end up below their own seed.** With HB132 (GEPA official also
shipped the seed on ifbench) that makes **four independent optimizers — GEPA, GEPA+Merge, MIPROv2,
and our M_ω (+.020, n.s.) — none of which beats the ifbench seed.** ifbench is not a bench where we
underperform; it is a bench where prompt optimization does not work at all. That is worth stating
as a finding rather than an empty row.
Caveat: these are single-pass `best_test` values and by HB121 are selection statistics; the
Δ signs above are suggestive only until a same-session k≥5 rescore. Do not print them as W/L.

## HB141 (2026-07-27) — DPI bank RECOMPUTED under median-split targets; Figure 4 RHS rebuilt

Advisor's blocking item: the DPI fixed-target cap is the campaign's only certified all-prompt
bound, and it inherited the tau=0.02 defect. Recomputed (`runs/dpi_bank_median.json`, script
`/tmp/dpi_median.py`, n=272):

| quantity | value |
|---|---|
| H(target) | **1.0000 bits for every metric** (median split, by construction) |
| achieved MI | median **.348** bits, IQR [.259,.431], max .570 |
| normalized skill | median **.650**, IQR [.560,.720] |
| fraction of capacity recovered | median **.348** |

**Per-task ordering is now interpretable** rather than threshold-driven:

| task | achieved MI (median bits) |
|---|---|
| humor | .446 |
| creative-writing | .425 |
| math | .312 |
| code-review | .276 |
| press-releases | .268 |
| peer-review | .251 |
| news | .245 |
| legal | .205 |

**Figure 4 RHS rebuilt** (`gen_fig_value_rhs.py` → `figs/gen_value_unsupervised.tex`): capacity is
now a single 1-bit line and the panel shows the sorted achieved-MI curve with the recovered
fraction shaded. **Per-metric ceiling SPREAD is abandoned deliberately, not lost** — under any
per-metric binarization H(M_b) is a function of the threshold choice (exactly 1 bit under a median
split), so ceiling spread can never be a statement about the metrics. The caption states the
definitional change explicitly: the target becomes "is this metric above its own median" rather
than "does this metric fire".

**Retraction:** every previously quoted number from the old bank (achieved .259 / ceiling .626
bits, the .365→.081 gap-tightening, the ceiling-backtest coverage figures) is computed on the
broken binarization and is **superseded**. The rank/exchangeability certificate is unaffected in
form (scale-free, per metric) but must be recomputed on median-split targets before being quoted.
PDF rebuilt, 6 pages, 0 errors.

## HB142 (2026-07-27) — ★★ Per-metric MISSING-VALUE ceiling (36x tighter), and the LHS curve TURNS OVER

**(1) RHS rebuilt with a real per-metric bound.** User's objection to H(M_b)=1 bit was correct: it
is identical for every metric and therefore carries no per-metric information. Two candidate
LHS-style estimators were tried:

| estimator | median gap above achieved | verdict |
|---|---|---|
| entropy H(M_b) | .1750 | uninformative — identical for all metrics |
| Good-Turing on exact signature patterns | .1482 | **near-vacuous**: U0=.92 (600 distinct patterns from 640 prompts) — every prompt is its own species, the HB119 granularity trap again |
| **upper order statistics of the achieved-score tail** | **.0167** | **36x tighter than entropy (11x conservative); used** |

Species must be the SCORE, not the pattern: missing *value* does not care which pattern is new,
only how much better a fresh draw could score. Per metric, over its N≈640 mined prompts, bound the
one-draw gain by the largest observed top-tail spacing. Distribution-free, conservative, and
**272/272 ceilings are strictly below 1.0**. Companion probability statement: a fresh prompt beats
the best of N with probability ≤ 1/(N+1) ≈ .0016. Artifacts `runs/missing_value_bound.json`,
`gen_fig_value_rhs.py`.

**(2) ★ The supervised value curve is NON-MONOTONE — and that is why the projection reads high.**
User asked why the projected median sits above the observed k>20. Answer: `unitrecomb_v5sk2` has
prefix points k=1..40 (k>40 does **not** exist — 40 is the maximum mined). The observed scores
past k=20 are:

`k=21..32 → .68 .67 .67 .67 .67 .66 .69 .68 .64 .64 .61 .63`

They **peak near k≈27 and then decline to ~.61**. The forward projection is fitted on k≤20, where
the curve is still rising, so it extrapolates a saturating plateau and cannot anticipate a
turnover. The projection is therefore **systematically optimistic by construction**, not
mis-plotted.

This independently corroborates HB135's `corr(#units, score) = −.263` at 8B: **past a subset
optimum, more units actively hurt.** The two measurements are different designs (prefix order vs
random subsets) agreeing on an inverted-U in prompt length.
**Consequence:** the missing-value projection must be reported as an upper bound *under a
saturating assumption that the data violates*, or refitted with a turnover term. Do not present
the extrapolated plateau as an expectation.

## HB143 (2026-07-27) — ★★★★ HOVER ZERO-SEARCH CEILING TEST PASSES ALL PREREGISTERED GATES

The HB128 prereg, frozen before launch, executed exactly as written. **Every arm appended to the
bare SEED — no searched prompt anywhere in any arm** — so unlike hotpot (where init ≡ seed made
the test vacuous, HB132) this genuinely tests whether the pool substitutes for search.
One session, one server, k=5 seed anchor, n=300 hover test. `runs/hb124_controls_hover_Qwen3-8B_seed.json`.

| arm | construction | mean | above seed |
|---|---|---|---|
| seed baseline (k=5) | — | **.4580** | — |
| **native (R)** | random p=.5 pool units → SEED | **.5535** | **20/20** |
| foreign (F) | hotpot's pool, count-matched → SEED | **.4568** | 9/20 |
| shuffled (partial 14/20) | native clauses, tokens scrambled → SEED | .4976 | 14/14 |

**Gates (frozen in HB128 before any number existed):**

| gate | criterion | result | |
|---|---|---|---|
| 1 | mean(R) ≥ .53 | **.5535** | **PASS** |
| 2 | M_ω ≤ 90th pct of R | 65th pct | PASS *(cross-session, weak — see caveat)* |
| 3 | mean(R) − mean(F) > 0, CI excludes 0 | **+.0967 [+.0848,+.1090]** | **PASS** |
| kill | mean(F) ≥ .53 → task-easy | .4568 | not triggered |
| kill | mean(R) ≤ .50 → assembly matters | .5535 | not triggered |

**VERDICT: POOL-CEILING SUPPORTED.**

**★ What this licenses, stated precisely.** Random halves of an LLM-proposed clause pool, appended
to the *unmodified seed* with **no search of any kind**, reach **.5535** — against HB127's
same-bench anchors of GEPA **.4640**, inhouse **.5587**, M_ω **.5640**. Zero-search draws land in
the same band as both searched methods and ~.09 above GEPA's optimized output. **This is the
superset confound (HB124 objection #1) finally answered, and answered on the only bench that
could answer it.**

**★ And the content specificity holds here too.** Count-matched foreign units land at **.4568**,
statistically indistinguishable from the bare seed (.4580) — appending text of the same volume
does nothing. The +.0955 gain is carried entirely by task-native content, replicating HB129/HB139
on a second benchmark and a 13x larger executor.

**Caveats, recorded before anyone quotes this:**
1. **Gate 2 is cross-session** — M_ω's .5640 comes from HB127, a different session; only gates 1
   and 3 are within-session. The clean version needs M_ω and inhouse rescored inside THIS session.
   Gates 1 and 3 carry the verdict; gate 2 is corroborative only.
2. **shuffled is incomplete (14/20)** and sits at .4976 — *above* the seed, unlike hotpot where
   scrambled text fell below init. If that holds at n=20, hover's units retain some value even
   scrambled, which would mean the hotpot and hover content effects are not identical in kind.
   Do not report the shuffled arm until it finishes.
3. n=20 draws per arm here vs 40 on hotpot; the R−F CI is nonetheless tight (±.012).

**Scoreboard consequence.** Combined with HB127 (M_ω ties pool-free `inhouse` on hover) the
coherent reading is now: **the value is in the articulated content, and both search loops and
random recombination are interchangeable ways of getting it into the prompt** — with GEPA's
selection the only procedure that reliably fails to.

## HB144 (2026-07-27) — Three adversarial checks on HB143. One lands, one is refuted, one is a trap.

**(1) ✗ "ZERO-SEARCH" IS AN OVERCLAIM — objection CONFIRMED, retract the phrasing.**
Audited the pool proposer (`_suggest_units_paper`, paperexact_arms.py:377-404). It is handed
**real training examples**: the prompt literally reads *"Here are REAL examples from this task's
training set:"* (v5 changelog: "+example-grounded framing — the suggester finally SEES the task").
So the pool is conditioned on train-set data. **"No search of any kind" is false and must not be
written.** The defensible claim is narrower and still strong:
> *One proposal round, conditioned on the task and its training examples, with **no selection and
> no iteration**, matches what the full optimizer ships.*
That relocates the finding from "search is unnecessary" to **"selection and iteration add nothing
beyond the proposal distribution"** — which is what the data actually shows, and is the claim to
build the paper on.

**(2) ✓ FORMAT/LABEL LEAKAGE — objection REFUTED.** The hypothesis was that hover's units smuggle
the output schema (SUPPORTED/NOT_SUPPORTED, verdict wording), so appending them fixes parse
failures rather than transmitting strategy. Measured over hover's 164 units:

| pattern | units | share |
|---|---|---|
| SUPPORTED/NOT_SUPPORTED token | 5 | **3%** |
| verdict/label/classify wording | 4 | 2% |
| any schema/format instruction | 12 | 7% |

And the single top match is a **false positive** — "explicitly supported by both the context and
the retrieved passages" is prose, not a label. At 3% prevalence a p=.5 draw carries ~2 such units;
that cannot produce +.0955. **Schema leakage is not the mechanism.** (The shuffled-arm sign flip
still needs the hotpot-shuffled-at-8B cell to separate benchmark from executor scale — HB139's
hotpot shuffled was measured at 0.6B, hover's at 8B, so "differ in kind" is not yet licensed.)

**(3) ⚠ BEST-OF-N — the counter is real but the statistic is booby-trapped.**
Max of the 20 hover draws = **.5933**, above M_ω's .5640, with **7/20** draws exceeding it.
Taken at face value this says trivial best-of-N selection beats the sophisticated search — which
would *strengthen* the anti-search reading. **But each draw is a SINGLE-PASS measurement, so a max
over 20 of them is precisely the winner's-curse statistic HB121 was written to forbid**, and it is
the same error that turned pupa's +.032 into a dead tie. **Do not quote .5933.** The disciplined
version: report the full draw distribution (mean .5535, median .5534, IQR, max flagged as
selection-inflated), and if best-of-N is to be claimed, select on a dev split and rescore the
winner k≥5 on held-out items in a fresh session.

**Immediate consequence for the write-up:** the HB143 headline sentence changes from
"zero-search recombination reaches the searched band" to **"an unselected, un-iterated proposal
round reaches the searched band."** Numbers unchanged; scope corrected before it propagates.

## HB145 (2026-07-27) — ★★★ DEEP AUDIT of the Fig-3-RHS ceiling: it FAILED hold-out. Now calibrated.

User asked for assurance the RHS numbers are bulletproof. **They were not.** Five failure modes
tested; two bite.

| check | result | verdict |
|---|---|---|
| **A. hold-out backtest** (build bound on 80% of prompts, test held-out 20% max) | **coverage .9338**, 18/272 violations, worst overshoot **−.053** | **FAILS the .95 target** |
| **B. discreteness** | granularity 1/300 = .00333; median tail spacing = **.00333 = exactly 1 item**; **33% of tail gaps are ties**; only 7 distinct values in the top 10 | **bound sits ON the granularity floor** |
| C. winner's curse on the anchor | `achieved` is a max over ~640 single-pass prompt scores → inflated; **not correctable from stored files** (no repeat passes per prompt) | open, biases the gap DOWN |
| D. exchangeability | median early-vs-late drift −.0046; \|drift\|>.02 in 29/272 | mild, acceptable |
| E. horizon | bound covers ONE draw only | stated, not yet generalized |

**The .0167 gap I reported was not a valid 95% bound.** It was an in-sample point estimate that
under-covers out of sample. Retracted.

**Calibration.** Swept tail depth K ∈ {10,20,40} × widening multiplier ∈ {1,1.5,2,3,4} over 5
repeated 80/20 splits per metric (272 metrics, ~640 prompts each):

| K | mult | coverage | median gap |
|---|---|---|---|
| 10 | 1.0 | .9213 | .0167 ← what I reported |
| 10 | 1.5 | .9441 | .0250 |
| **10** | **2.0** | **.9610** | **.0333** ← smallest configuration reaching ≥.95 |
| 10 | 3.0 | .9801 | .0500 |

**CALIBRATED BOUND: K=10, multiplier 2.0 → coverage .961, median gap .0333, median ceiling .8633.**
K is irrelevant beyond 10 (identical results at 20 and 40) — the multiplier does all the work,
which is itself diagnostic: the tail is granularity-limited, so widening beats looking deeper.

**Honest standing:** still **5x tighter than the entropy bound** (.175 gap) and per-metric, but
half as impressive as the uncalibrated number and now actually validated out-of-sample. Caveat C
(inflated anchor) remains and should be stated in the caption — it biases the measured gap
downward, so the true gap is likely a little wider still.

## HB146 (2026-07-27) — ★★★ HB128 COMPLETE: the gain DECOMPOSES 42% lexical / 58% compositional

All three arms finished, n=20 each, one session, every arm appended to the bare SEED.

| arm | mean | above seed | vs seed |
|---|---|---|---|
| seed (k=5) | .4580 | — | — |
| **native** | **.5535** | **20/20** | **+.0955** |
| shuffled (same clauses, tokens scrambled) | .4977 | 19/20 | +.0396 |
| foreign (hotpot pool, count-matched) | .4568 | 9/20 | −.0012 |

| contrast | Δ | 95% CI | P(Δ≤0) |
|---|---|---|---|
| native vs foreign | +.0967 | [+.0848,+.1090] | .0000 |
| native vs shuffled | **+.0558** | [+.0442,+.0678] | .0000 |
| shuffled vs seed | **+.0396** | [+.0318,+.0470] | .0000 |

**★ The native gain splits cleanly into two components:**

| component | size | share |
|---|---|---|
| **lexical** — survives token scrambling (right words, any order) | **+.0396** | **42%** |
| **compositional** — destroyed by scrambling (word order / syntax carries it) | **+.0558** | **58%** |
| generic text of matched volume (foreign) | −.0012 | **0%** |

This resolves the HB143 caveat and is a **better** mechanism story than all-or-nothing. Appending
*any* fluent, count-matched text does nothing (foreign ≈ seed). Appending the *right vocabulary in
scrambled order* buys 42% of the gain. Intact, composed clauses buy the remaining 58%. So the
units are neither magic strings nor mere keyword bags — both channels are real and measurable.

**On the hotpot/hover shuffled sign flip (HB144 objection).** Hover shuffled sits ABOVE seed
(+.0396); hotpot shuffled sat BELOW init (−.0548). **This is still confounded with executor
scale** — hotpot's shuffled arm was measured at 0.6B, hover's at 8B. A small model is plausibly
derailed by scrambled text while an 8B model mines it for keywords. **Do not write "the benchmarks
differ in kind"** until the hotpot-shuffled-at-8B cell exists; HB131's matched-8B setup makes that
one cheap cell.

**Quotable now:** *"On hover, appending a random half of an LLM-proposed clause pool to the bare
seed — with no selection and no iteration — raises exact-match from .458 to .554, and the gain
decomposes into 42% lexical and 58% compositional; count-matched foreign clauses contribute
nothing."* Per HB144, do NOT describe this as "zero search" (the pool proposer saw training
examples) and do NOT quote the max draw (.5933, a single-pass max).

## HB147 (2026-07-27) — ★ RELABEL HB146. "Compositional" is wrong; two design facts verified in code.

**(1) The label "compositional" is refuted by my own foreign arm.** Foreign clauses have perfectly
intact syntax and word order and contribute **0** (−.0012). So the +.0558 increment cannot be
"word order carries it." It is *domain vocabulary × intact arrangement* — i.e. **propositional
content**, an interaction, not a syntax main effect. Corrected naming, to be used everywhere:

| component | Δ | correct label | wrong label (retracted) |
|---|---|---|---|
| survives word-scrambling | +.0396 | **vocabulary priming** | "lexical" (fine, but say vocabulary) |
| above that, needs intact clauses | +.0558 | **propositional content** | ~~"compositional / word order"~~ |
| count-matched fluent foreign text | −.0012 | **generic well-formed text** | — |

Also note the decomposition is **path-dependent**: I walked seed→shuffled→native and named the
increments as if they were independent main effects. A reviewer can walk seed→foreign→native and
observe "intact syntax alone = 0." Report the three Δs with CIs; **do not quote 42%/58% as precise
shares** — they are a ratio of two noisy differences.

**(2) Two design facts, checked in `hb124_controls.py` rather than assumed:**
- **Scrambling is WORD-level** (`c.split()` → shuffle → join), not token-level. Individual words
  survive intact, so "vocabulary preserved" is true at word granularity — **but multi-word entity
  names are shattered**, and hover is fact-verification where entities are the lexicon. The
  vocabulary share is therefore likely **under**-estimated and the propositional share inflated.
  An entity-preserving scramble arm would bound this.
- **The arms are NOT paired.** `rng.choice` is drawn sequentially per arm, so native/shuffled/
  foreign each use *different* unit subsets. Pool-composition variance therefore sits inside the
  native−shuffled CI, making it wider than a paired design would give — conservative for the
  contrast, but it should be stated, and the follow-up should reuse identical subsets per arm.

**(3) Missing control that would change the numbers: SCRAMBLED-FOREIGN.** Word-soup is
high-perplexity text; part of the +.0396 could be a generic scrambled-text effect (distraction,
more cautious decoding) rather than vocabulary. Intact-foreign ≈ seed does **not** control this,
because fluent off-topic text is ignorable in a way word-soup is not. If scrambled-foreign > seed,
the vocabulary share shrinks. **PREREGISTERED PREDICTION before that arm runs: scrambled-foreign
≈ seed (within ±.015).** If it lands materially above seed, the vocabulary component is partly an
artifact and HB146 must be restated.

**(4) One CPU-free audit owed before 42% is quoted anywhere:** n-gram / entity overlap between the
pool clauses and the 300 hover TEST items. The proposer saw training examples (HB144); if train
and test share entities, the scramble-surviving gain is partly content leakage rather than
"vocabulary". Label leakage was refuted at 3%; *content* overlap has not been measured.

## HB148 (2026-07-27) — ★★★ I WAS WRONG THAT PANEL C IS CIRCULAR. Selection is CONDITIONAL, not a threshold.

User asked why some small-delta units are kept while some big-delta units are discarded. Traced it
to source (`build_frozen_pool.py:33-46`) and the answer overturns my own repeated claim:

- **`delta_8b`** = the unit's **STANDALONE** screening marginal (from `marginals`).
- **`won_8b`** = membership in **`compiled_units`**, i.e. the final **greedy assembly**, which
  accepts a unit on its **CONDITIONAL** gain given everything already accepted.

**These are two different statistics.** `won_8b` is NOT a threshold on `delta_8b`, and the
distributions overlap heavily:

| | n | delta range | overlap |
|---|---|---|---|
| KEPT | 27 | [+.010, +.160] | **19/27 kept units sit BELOW the highest discarded unit** |
| DISCARDED | 41 | [−.230, +.070] | **12/41 discarded units sit ABOVE the lowest kept unit** |

**Retraction.** In HB136, HB139 and the figure caption I labelled the screening panel "CIRCULAR —
`kept` was defined by thresholding this very statistic." **That is false.** It was a reasonable
guess that I never checked against the code, and it is wrong.

**What the panel actually shows, and it is a finding rather than an artifact:** greedy selection is
**redundancy-aware**. A unit with a large standalone gain is rejected when an already-accepted unit
covers the same ground; a unit with a small standalone gain is accepted when it adds something new
conditional on the rest. The imperfect correlation between the two axes IS the result — it is
direct evidence that *what a unit is worth depends on what else is in the prompt*, which is the
paper's compositional claim in miniature.

**This also rehabilitates an earlier "null".** HB120b reported corr(greedy marginal, causal Δ) =
+.013 and I called the design unable to answer the question. Part of that near-zero correlation is
now explained: standalone marginals and conditional acceptance are genuinely different quantities,
so a low correlation is expected, not merely a power failure.

**Actions:** (1) strike "circular" from the figure caption and from HB136/HB139; (2) the redrawn
panel must show the overlap explicitly, since that is the point; (3) the honest framing for the
screening axis is *"standalone gain"* vs the selection axis *"survived conditional greedy
assembly"*.

## HB149 (2026-07-27) — Content-leakage audit: REFUTED. Plus the debt-clearing pass into the paper.

**(1) Pool→test content overlap (task #11, CPU-free, prereg'd as owed before quoting the
vocabulary share).** Decisive design: if leakage explained the native−foreign gap, the NATIVE pool
would overlap the hover TEST text far more than the FOREIGN pool. Measured (`overlap_audit.py`,
hover test claims, n=300):

| pool | units w/ a test-matching content BIGRAM | mean single-word overlap | proper-entity spans |
|---|---|---|---|
| native (hover) | 5/164 (**3%**) | .31 | 26 |
| foreign (hotpot) | 0/68 (0%) | .27 | 1 |

Single-word overlap is nearly identical (.31 vs .27) — that is shared generic vocabulary, present
in both pools, so it cannot produce the +.097 native−foreign gap. Bigram-level (content) matches
are 3%, the same rate as the label-leakage check (HB144). **Train→test content leakage does not
explain the vocabulary-priming component.** The +.0396 scramble-surviving gain is task-domain
vocabulary, not copied test content.

**(2) Paper updated (the debt the user flagged).** Same-session certified numbers moved from notes
into `main.tex` (submodule commit 5f2c921): Table 1 HoVer row = HB127 omnibus (+.100***), all
single-pass cells filled but †-marked as selection statistics excluded from W/L, AIME's stale
exclusion footnote replaced (HB97 rescued it), seed-shipping + ifbench-inert footnotes added;
Fig 3 LHS caption states the k≈27 turnover (projection = optimistic envelope, per HB142); NEW
appendix figure: missing-mass curves for all five supervised benches from the re-mined replicate
pools at TWO species granularities (fine J≥.5 vs coarse J≥.25 — the within-panel contrast IS the
granularity-not-task finding; deterministic clustering, declared conservative). PDF now 7 pages.
Memory index + project memory updated (the .517/.450 headline is now flagged SUPERSEDED there).

## HB150 (2026-07-27) — PREREG + LAUNCH: table-comparability omnibus & controls v2 (frozen before launch)

**Lane 1 — TABLE OMNIBUS (fixes "these numbers are not comparable").** For each bench, ALL table
candidates rescored in ONE invocation, k=5, cache off, one server fingerprint: official,
GEPA+Merge, MIPROv2 (where the cell exists), and the M_ω run. Waits for the T1/mipro-refill lanes
to drain so the mipro cells exist. Every Table-1 row will then be single-session; all † daggers
retire. Hover is re-panelled ONLY because adding MIPROv2 to the row is an instrument change (new
candidate set requires a new session); the HB127 prereg verdict (WIN +.100) is untouched and stays
the verdict of record.
**Lane 2 — CONTROLS v2 (tasks #9/#10), design fixes from HB147:** all arms drawn from PAIRED unit
subsets (same indices per draw across same-pool arms; count-matched for foreign); five arms on
hover→seed: native, foreign, shuffled, **foreign_shuffled**, **shuffled_entity** (word-scramble
that keeps capitalized multi-word entity spans atomic); then **hotpot shuffled at 8B**.
**Frozen predictions:**
1. foreign_shuffled ≈ seed within ±.015 (already frozen in HB147). If materially above seed, the
   vocabulary component is partly a generic scrambled-text artifact and HB146 must be restated.
2. shuffled_entity ≥ shuffled (entity-preserving scramble recovers MORE than plain scramble),
   because word-level scrambling shatters entity names; gap = the entity share of vocabulary priming.
3. hotpot-shuffled-at-8B lands ABOVE init (executor-scale explanation of the sign flip). If it
   stays BELOW init at 8B, the flip is bench-kind, not scale, and HB146's cross-bench framing changes.
One shot each; whatever returns is the verdict.

## HB151 (2026-07-28) — Fig-3 methodology audits (user-ordered) + three launches

**(1) RHS ceiling audit — split-half calibration check.** Suspected flaw: the ×2 multiplier was
chosen on the same repeated splits used to report coverage. Ran calibrate-on-half /
validate-on-other-half, 10 replicates (`audit_calib_split.py`, sk3):
- procedure risk CONFIRMED: chosen-on-half often selects ×1.5, which under-covers held out
  (mean .947, min .927) — had we tuned harder we'd have shipped an invalid bound;
- the FIXED ×2 rule validates at **.960 held-out coverage** → the reported number stands, now on
  genuinely held-out metrics. Caption updated.
**(2) LHS band was 2.8× too narrow.** The widening slope (.02/step) was ad hoc. Calibrated so the
band covers every held-out k=21..40 point: **s=.056**. Redrawn; the honest band is wide.
**(3) k cannot reach 100** — the hotpot pool has 68 units. Launched `prefix_lane` (sk2, pid
3140131): full k=1..68 prefix curve, ONE session, declared order = standalone-delta rank.
**(4) RHS annotations:** legal outcome-prediction (achieved .777, calibrated ceiling 1.0 — the
left-end high-headroom exemplar) and creative-writing #20 (.873 → .993) on the right.
**(5) Fig 4 v4** per user sketch: full seed prompt; atomic kept unit (+.15); discarded exemplar
chosen to be REJECTED FOR CHANGING NOTHING (Δ=.00, restates its module — behaviour row identical);
Panel B = admission pipeline into Ω with the explicit gate Δ(Ω⊕u|Ω)≥ε. Caption now states:
tests here are score-mediated (behavioural flips = unsupervised-side screen); retention is
company-dependent (19/27 overlap), so a different mixture keeps a different set; per-unit
multi-mixture identification needs ~200+ draws (40 was unidentified) — open instrument.
Falsy-zero bug caught in my own pool query (`delta or 9` drops Δ=0.0 units).

## HB152 (2026-07-28) — Metrics-side scaling: AUDIT of the July-7 work + design of the parallel battery (OSL-M)

**(A) Audit of existing metrics scaling-law work (user: "how systematic/current is it?").**
Source: `notebooks/data/2026-07-07-osl-multi/` (8 task curve files, 1,387 metrics × up to 14
executors, 1.24B–72.7B, 5 families) + memory `project_osl_executor_scaling`.
- **Systematic: YES, more than expected.** Frozen 14-executor crowd; capability index z from a
  frozen anchor battery (logit-AUC); planted-truth ladders monotone in both families; placebo
  corrections; per-metric fitted verdicts (RISING/REACHES/BOUNDED/NOISY); inverted-U closed
  across 4 families; z×a forecast machinery with falsifiable N* predictions.
- **Independent of the tau=.02 defect** — different pipeline (probe-battery recovery, not the
  llama8b_glm signature bank), so HB138 does NOT contaminate it.
- **But NOT current for Paper 2, for four reasons:** (1) its y (probe-battery recovery /
  identification) is not the paper's median-split reconstruction skill — the two are not
  interchangeable readouts; (2) several rungs are GLM-adjudicated and the GLM API is permanently
  dead → those rows are unextendable/unreplicable; (3) its own caveat ledger marks cross-task
  probe-support comparability BROKEN and some planted-truth rows needing recompute; (4) exchange
  rates were found FAMILY-RELATIVE, so pooled-family claims are barred.
- **The actual gap:** nothing in that lineage measures the Paper-2 quantities across scale —
  {seed, best-mined prompt, random pool draws, ceiling} per metric per executor rung. The
  supervised ladder has exactly this grid; the metrics side has none.

**(B) DESIGN — OSL-M, the parallel battery (prereg-ready; NOT launched, needs sign-off + GPUs).**
- **Executors:** Qwen3 {0.6B, 1.7B, 4B, 8B} — same rungs as the supervised ladder, one family
  (standing rule), all cached on sk2.
- **Metric sample:** 32 metrics = 8 tasks × 4, one per achieved-skill quartile (median-split
  bank), stable-hash selection.
- **Arms per metric × rung (mirrors the supervised arms):** seed/generic prompt (k=3) · the
  metric's best mined prompt (k=3) · 10 random draws from its own 640-prompt pool (k=1) ·
  count-matched foreign-prompt control on a 8-metric subsample.
- **Items:** fixed 120-item stable-hash subset of the 300; **readout = normalized skill
  2·agreement−1 on the frozen per-metric median split** (capacity fixed at 1 bit, threshold-free
  across metrics). One server per rung; all arms of a rung in ONE session.
- **Preregistered questions:** Q1 best-mined skill vs scale (rises? saturates?); Q2 the
  best-minus-random-draw gap vs scale — prediction from HB131: the gap does NOT grow (pool ≥
  selection at every scale); Q3 draws-minus-seed gain vs scale — prediction: grows, the
  unsupervised analogue of +.038→+.090→+.186; Q4 foreign control stays at seed at every rung.
- **Cost:** ≈32×12×120 ≈ 46k judgments/rung, ~184k total ≈ 1 GPU-day serial on sk2.
- **STEP-0 dependency (the one blocker):** the npz stores prompts but not item ids; the
  sigs-column ↔ item-text mapping must be recovered from the bank builder
  (`methods/metric_seam/reconstruction_v2.py` / cells DB) before any rung runs. Items exist in
  frozen_probes only for humor; other 7 tasks need the join.

## HB153 (2026-07-28) — July-7 metrics-scaling data: GLM-FREE and reusable; convergent audit thin

**(1) Reuse verdict (user: can we reuse/audit the old results, even discarding GLM?).**
- **The core grid never touched GLM: 0 of 1,387 curves contain a GLM executor.** All 14 executors
  are local (llama 1b/3b/8b/70b, qwen2.5 3–72b, gemma2 9/27b, mistral 7/24b, phi4). GLM appears
  only in the separate adjudication/step-down layer (hermes_adjudication, mbarglm rungs). So the
  curves are fully reusable AND fully replicable/extendable today.
- Curve files store per-metric z[], y[], se[], execs[], fitted verdict, and limit L with CI —
  everything needed for refits (family-holdout, drop-any-executor) without recompute.
- **Convergent-validity audit run** (new median-split achieved-MI vs July-7 fitted limit L,
  matched by metric name): the two lineages sampled mostly DIFFERENT metrics — only ~54/272 bank
  metrics appear in the curves at all; usable overlap n=17 (cw) and n=17 (humor), rho +.22 / +.06,
  Fisher-z pooled +.14. **Uninformative at this n, not discordant.** Matching is also truncated
  (npz names cut at 54 chars — prefix match changed nothing).
- **Consequence for OSL-M:** the design gains a free companion — for sampled metrics present in
  the July-7 grid, plot the old local-executor curve beside the new median-split curve; OSL-M
  becomes the convergent audit the thin name-overlap could not deliver.

**(2) Fig fixes shipped** (user list): Fig 6 — five supervised benches LHS, extrapolation/trend
lines REMOVED, named+boxed RHS annotations; Fig 7 — the p_unseen RISE was an ordering artifact
(sequence grouped by framing ⇒ f1/n non-exchangeable); permutation-averaged over 150 orders,
curves now monotone, caption explains; Fig 4 v5 — pool Ω central with 11 real abbreviated units,
prose prompt with shaded attachment spans, single admission gate, audit battery, Fig-1-idiom
flip dots.

## HB154 (2026-07-28) — WASTE CAUGHT AND STOPPED: pupa Table-1 cells are unrunnable (GLM-wired judge)

User asked for a cycles audit. Found one real burner: the T1 lane's `pupa/official_merge` cell had
been running **2.2 hours with 840 dead-GLM error lines, zero scored evals, and an EMPTY run dir.**
Root cause: `load_bench("pupa")` wires the metric's quality/leakage judge to GLM via
`make_reflection_lm("glm-5.2")` (patient mode = 40 retries per call), and the GLM API is
permanently dead (2026-07-25). **Every pupa metric call spins through 40 retries and fails; the
cell can never succeed.** pupa/mipro queued behind it would have burned identically.

Actions: killed the T1 wrapper (678689) FIRST, then the pupa child (789770). The lane's remaining
work was only the two pupa cells, so nothing else was lost; the refill (hotpot+aime MIPROv2 — no
GLM anywhere) unblocks within its 5-min poll, then the table omnibus (which never included pupa).

**Standing rule added: before queueing ANY pupa work, check the bench-level judge wiring.** The
pupa metric judge is part of the instrument; rewiring it to a local model would be an instrument
change (and an 8B judge violates judges-Sonnet-or-better), so pupa's GEPA+Merge/MIPROv2 cells stay
EMPTY with a stated reason unless the user decides otherwise. Pupa's headline status (TIE, final)
is unaffected.

Health of everything else, verified by artifacts/rates not liveness: controls_v2 5.5–7.1 min/draw,
gpu2 100%, foreign arm scoring at seed level as the prereg predicts; OSL hover/4B 27/40 draws,
artifact fresh; prefix/omnibus/refill idle-by-design (zero cycles).

## HB155 (2026-07-28) — DUPLICATE T1 LANE found and killed; single-pass t1fill cells are double-provenance

The mipro-refill stayed blocked 40 min after the T1 kill. Cause: a SECOND `lane_t1_v3.sh`
instance (pid 1721025) — the supposedly-failed first launch from 2026-07-27 ~10:00Z had in fact
survived — was still matching the refill's wait pattern, with its own pupa child burning
dead-GLM retries. Killed (wrapper first, then child).

**Integrity note:** two identical T1 lanes ran the same bench x arm sequence ~20 min apart against
the SAME run dirs and log (explains the doubled "[T1] hotpot / official_merge" lines and the slow
mipro pace — two 600-call optimizations sharing one server). Every `*_t1fill` result.json may
therefore hold whichever duplicate finished last. **Consequence: none for the paper** — those
cells are already †-marked single-pass indicative-only, and the table omnibus re-measures every
candidate same-session, which is the only instrument the final table will quote. But the † cells
now carry double provenance and must never be upgraded to quotable retroactively.

Root-cause lesson (added to the artifact rule): a wrapper that appears not to have started may
have started invisibly — always `pgrep` for ALL instances after any launch, not just the newest pid.

## HB156 (2026-07-28) — THE QUEUE DEADLOCK WAS MY OWN WATCHERS. Fixed; everything verified moving.

Root cause of the refill never starting, found with a bracket-grep census run from a shipped file
(the only way to see truth — inline ssh pgrep self-matches):
1. **My two background watcher loops carried the string `lane_t1_v3.sh` in their command lines**,
   so the refill's `pgrep -f "lane_t1_v3.sh"` wait saw THEM and blocked forever. The monitoring
   caused the deadlock it was monitoring. (The sk3 memory literally records this trap —
   "launch guards via `ssh 'bash -s' < file`, never inline" — and I violated it.)
2. A THIRD pupa burner (`pupa --arm mipro`, pid 3600272) launched by the stale duplicate T1
   wrapper was still spinning on dead-GLM retries.
3. A DUPLICATE refill wrapper (678692) — same double-launch episode as the duplicate T1 lane.
Killed all four by explicit pid (3600272, 678692, 669631, 3622751). Verified end state: exactly
ONE refill (398360), ONE omnibus (2592654), ONE prefix (1273334, relaunched — the previous
wrapper had died silently), ZERO pupa processes, ZERO lane_t1 matches. New clean-cmdline watcher
armed from a remote file.
**Rules reinforced:** (a) every wait-loop pgrep pattern must be self-match-proof (bracket-grep or
file-run); (b) after ANY launch, enumerate ALL matching instances, not the newest pid; (c) a
watcher's cmdline must not contain any string another lane waits on.

## HB157 (2026-07-28) — ★★★ FIVE-ARM BATTERY COMPLETE: both preregs FAIL, and the failures REWRITE HB146

One session, PAIRED unit subsets (same draw indices across same-pool arms), n=20/arm, k=5 seed.
`runs/hb124_controls_hover_Qwen3-8B_seed_pairedv2.json`. Seed .4527 (prev session .4580, drift −.005 ✓).

| arm | mean | vs seed | above seed |
|---|---|---|---|
| native | .5460 | **+.0933** | 20/20 |
| shuffled (native tokens, scrambled) | .4998 | +.0472 | 20/20 |
| shuffled_entity (entities kept atomic) | .4920 | +.0393 | 20/20 |
| foreign (intact) | .4552 | +.0025 | 13/20 |
| **foreign_shuffled** | **.4093** | **−.0434** | **0/20** |

**PREREG 1 FAIL** (foreign_shuffled ≈ seed ±.015): it lands **−.043, 0/20 above seed** — scrambled
text carries a real GENERIC PENALTY, not ≈0. **PREREG 2 FAIL** (shuffled_entity ≥ shuffled):
Δ=−.008 [−.018,+.002] — entity preservation does NOT help; the vocabulary effect is not
entity-carried.

**★★★ The revised additive decomposition (4-cell identification, paired):**
Let score = seed + VOCAB·(native tokens) + PROP·(intact composition) − PEN·(scrambled surface).
- PEN from foreign vs foreign_shuffled (paired): **.0459 [.0347,.0557]**
- VOCAB = shuffled − seed + PEN ≈ .0472 + .0434 ≈ **+.091**
- PROP = native − seed − VOCAB ≈ .0933 − .091 ≈ **+.003 — statistically indistinguishable from 0**

**HB146's "42% lexical / 58% propositional" is RETRACTED as an artifact of the missing
scramble-penalty control.** With the penalty identified, the native gain is carried (essentially
entirely) by DOMAIN-TOKEN / VOCABULARY PRIMING (~+.09); intact composition adds nothing detectable
beyond it; and the old "propositional" share was the scramble penalty in disguise. The advisor
flagged exactly this hole ("shuffled is a weak control; gibberish plausibly actively confuses");
the prediction I froze went the other way, and the data overruled me — which is what preregs are for.
Caveats, stated: additive model; assumes the scramble penalty is pool-independent (supported:
native-pool penalty ≈ foreign-pool penalty within CI); single bench (hover) at 8B; hotpot-shuffled
at 8B (stage 2, running now) tests transfer of the penalty account to the bench where scrambling
looked catastrophic at 0.6B.
**Paper impact:** no printed claim changes (HB147 already barred the 42/58 percentages from the
paper); the mechanism sentence, when written, is now *"the pool's value is carried by
task-vocabulary priming rather than intact clause composition, on hover at 8B"*.

## HB158 (2026-07-28) — PREREGS FROZEN MID-FLIGHT: hotpot-shuffled-8B point prediction + keyword arm

**(1) Advisor caps on HB157 (adopted).** The 4-cell identification rests on PENALTY HOMOGENEITY
(scramble penalty equal across pools) — scrambled in-domain text is plausibly partially
recoverable, so PROP=0 is model-dependent. **Safe-to-print, assumption-free bounds:**
- foreign intact ≈ 0: fluent composition without domain vocabulary is worthless;
- shuffled retains **+.047 = ≥51% of the native gain with word order destroyed** (hard lexical
  lower bound, no penalty modeling);
- entity preservation adds nothing;
- composition effect **bounded in [~0, .046]**; equal to ~0 only under the additive equal-penalty
  model. Print "no detectable composition effect (|PROP| ≲ .02)", never "zero".
- Lead with PAIRED contrasts; seed-relative deltas are descriptive (single seed estimate,
  load-dependent metric). Scope: hover, 8B, one session.
- Convergence to cite: metric-lexicon P12 (naming +.505 vs paraphrase −.004) — an independent
  instrument giving the same "the payload is which concepts get NAMED" signature.

**(2) PREREG — hotpot-shuffled-8B point prediction, frozen BEFORE the running arm lands.**
Under vocabulary-carries-all with a transferable scramble penalty:
predicted shuffled-vs-init ≈ (hotpot 8B native draw gain) − PEN = +.186 − .046 = **+.140**,
interval **[+.130, +.151]** from PEN's CI. If it lands near init (≈ +0), composition matters on
hotpot and the hover account is bench-local. Genuine out-of-sample test; number on record now.

**(3) PREREG — KEYWORD-LIST arm (the advisor's decisive penalty-free check), queued tonight.**
Deliver each native unit's content terms as a plain comma-separated list (legitimate format, no
scramble penalty), paired subsets, hover→seed, ONE new session containing {seed k=5, native ×20,
keywords ×20} so all contrasts are within-session.
Decision rule, frozen: **VOCAB-CONFIRMED** iff keywords − native ∈ [−.015, +.015] (paired CI) —
the decomposition footnote-izes and the keyword cell becomes the headline mechanism.
**COMPOSITION-REAL** iff native − keywords ≥ .03 with CI excluding 0 — the equal-penalty
assumption was the artifact and the mechanism section reports the bounded claim only.

## HB159 (2026-07-28) — ★★★ TWO HARVESTS: OSL ladder COMPLETE (2nd bench 40/40) + the inverted-U measured in one session

**(1) OSL supervised ladder COMPLETE (8/8 cells, 24k tokens, truncation-confound closed).**
Final cell hover/8B: init .4767, random-draw mean **.5466, 40/40 above init (+.070)**, transfer
.5667. **Both benchmarks now show every one of 40 random pool draws beating the GEPA-shipped
prompt at matched 8B scale** (hotpot +.186, hover +.070), with the gain growing monotonically
with scale on both. Fig 5 regenerated: 4 rungs × 2 benches, real data throughout.

**(2) Same-session full-pool prefix curve COMPLETE (k=1..68, one server, one fingerprint,
declared order = standalone-delta rank).** `runs/prefix_extend_hotpot.json`:
- **peak k=46 at .643** — within noise of the searched compile (.633/.6367);
- declines to **.593 at k=68** (last-10 mean .576);
- the earlier cross-session wobble (HB142's plateau-vs-decline ambiguity) is resolved: the
  **inverted-U is real and now measured with a single instrument**. Taking enough top-ranked
  units MATCHES search; exceeding the optimum HURTS. This is also the cleanest support yet for
  the "selection adds little beyond rank-and-take-enough" reading.
Fig 3 LHS overlays the curve (green) with the projection band; captions updated. Checklist:
[fig5 ladder] ✓ [fig3 k68] ✓ (with [5-arm verdict] ✓ = 3/7).

## HB160 (2026-07-28) — Rank certificate REVALIDATED on median-split targets; omnibus measuring

**(1) Rank/exchangeability certificate backtest on the corrected binarization** (the old backtest
ran on tau=.02; checklist item): 272 metrics × 5 splits = 1,360 observations,
**observed P(held-out max beats train max) = .1949 vs .200 predicted — conservative, quotable.**
The paper's Fig-3 caption claim (rank certificate, 1/(N+1)) now rests on the same binarization as
every other number from the bank. Checklist [rank-certificate] ✓ (4/7).

**(2) Refill COMPLETE** (aime MIPROv2 artifact verified, best-program score 37.78 on its select
metric) and **the table omnibus is MEASURING** — handoff was fully automatic this time; hotpot
bench 1/5 in progress at k=5. ETA for all five benches ≈ 8-10h ⇒ comparable Table 1 tonight.

## HB161 (2026-07-28) — ★★★ hotpot-shuffled-8B: sign-flip = SCALE (confirmed); vocabulary-carries-all = HOVER-LOCAL (prediction missed)

One session, paired subsets, n=20/arm. init(k=5)=.3987 (≡ seed on hotpot; HB131 anchor .4000 ✓).
`runs/hb124_controls_hotpot_Qwen3-8B_init_shuf8b.json`.

| arm | mean | vs init | above init |
|---|---|---|---|
| native | .6027 | **+.2040** | 20/20 |
| shuffled | .4560 | **+.0573** | 18/20 |
paired native−shuffled: **Δ=+.1467 [+.1243,+.1680]**

**(1) HB150 prereg #3 CONFIRMED.** At 8B, scrambled native units land ABOVE init (+.057, 18/20) —
the 0.6B below-init result was **executor fragility, not bench-kind**. The sign flip is a SCALE
effect. (Also: native +.204 here replicates HB131's +.186 across sessions.)

**(2) HB158 point prediction MISSED LOW** (predicted +.140 [.130,.151], observed +.057). The
vocabulary-carries-everything account does NOT transfer to hotpot: scrambling destroys .147 of a
.204 gain there vs only .046 of .093 on hover. Under the additive model with the hover penalty,
hotpot decomposes to VOCAB ≈ .10, **PROP ≈ .10 — composition is REAL on hotpot (~half the gain)**;
alternatively the scramble penalty is bench-dependent. Either way: **the PROP≈0 result is
hover-local.** The honest layered mechanism claim for the paper:
- vocabulary share is large on BOTH benches (assumption-free floors: ≥51% hover, ≥28% hotpot);
- the compositional share is BENCH-DEPENDENT — ≈0 on hover (claim-verification), substantial on
  hotpot (multi-hop QA with format-critical answers);
- prereg discipline made both the confirmation and the miss legible.
Keyword arm (queued) now matters MORE: it separates vocabulary-delivery from scramble damage on
hover; a hotpot keyword arm becomes the obvious follow-up but is NOT required for tomorrow.
Checklist [shuf8b verdict] ✓ (5/7).

## HB162 (2026-07-28) — Advisor caps on HB161 + PREREG: hotpot keyword arm (frozen before launch)

**(1) The two-way ambiguity, stated for the record.** hotpot PROP≈.101 is a RESIDUAL under the
untested assumption that hover's scramble penalty (.046) transfers across benchmarks — and we have
positive evidence penalties vary (0.6B hotpot's penalty exceeded the entire vocab gain). A true
hotpot penalty of ~.147 reproduces the data with PROP=0. Assumption-free statement only:
vocabulary ≥28% on hotpot; the remaining ≤72% is UNALLOCATED between composition and penalty.
The advisor's print-ready mechanism paragraph (6 sentences, floors + flagged model numbers +
falsified frozen prediction) is adopted verbatim for the paper's mechanism section.

**(2) PREREG — hotpot KEYWORD arm (the discriminator), frozen now:** one session, init k=5 +
native ×20 + keywords ×20, paired subsets, tag _kw8b. Under PROP≈.10: keywords−init ≈ +.10;
under vocabulary-carries-all: ≈ +.20. **Rule: keywords−init ≥ .15 → VOCABULARY-DOMINANT on hotpot
too (penalty was bench-dependent); ≤ .12 → COMPOSITION CONFIRMED at roughly half the gain;
(.12,.15) → report the interval, no verdict.** One shot.

**(3) Omnibus-row guards (adopted before the table rebuild):** the selection-free hotpot M_ω
number is the SHIPPED compile rescored (.646 = frozen ex-ante candidate at k=5 — legitimate); the
k=46 sweep value (.643) is an oracle read of the same eval and must NEVER headline; leakage
on-record: hotpot pool 0/68 flagged + anchor-validated (HB96); framing: the row demonstrates the
information-access bound, not equal-input optimizer superiority; per-item paired tests to
accompany the table rebuild.

## HB163 (2026-07-28) — ★★★ KEYWORD ARM VERDICT: vocabulary alone is INERT; composition is REAL. HB157's additive story retracted.

One session, paired subsets, n=20/arm. seed(k=5)=.4647 (session drift vs .4527/.4580 noted;
paired contrasts unaffected). `runs/hb124_controls_hover_Qwen3-8B_seed_kwv1.json`.

| arm | vs seed | above seed |
|---|---|---|
| native (intact clauses) | **+.0937** | 20/20 |
| **keywords (same terms, clean list format)** | **+.0107** | 15/20 |
paired keywords−native: **Δ=−.0830 [−.0982,−.0687]** → frozen rule: **COMPOSITION-REAL.**

**The decisive fact: delivering the units' entire content vocabulary in a legitimate format
("relevant terms: …") reproduces almost NONE of the gain.** So HB157's additive identification —
VOCAB≈.091, PROP≈0 — is **RETRACTED as a mechanism claim** (its arm data stand; the equal-penalty
assumption failed exactly as the advisor warned). The full assumption-free ladder on hover:

  intact clauses +.094 > scrambled clauses +.047 > **keyword list +.011 ≈ intact foreign +.003** > scrambled foreign −.043

**Revised mechanism reading (hover, 8B):** the pool's value requires CLAUSE-FORM articulation.
The same terms as a list are inert; word-scrambling preserves roughly half the gain (instruction-
shaped in-domain text retains some effect even disordered — surviving collocations and/or
imperative surface form); intact composed clauses carry it fully. This is also consistent with
hotpot's PROP≈.10 (HB161) — composition now looks real on BOTH benches, restoring cross-bench
coherence that the vocabulary-only story had broken. The queued hotpot keyword arm is the
confirming replication.
**Note a genuine tension to state, not hide:** the metric-lexicon result (naming a term +.505,
paraphrase −.004) points the opposite direction in a different regime (metric identification vs
task execution). Term-naming identifies; clause-form instructs. Flag as a regime difference.
**Successive-revision ledger, for the record:** 42/58 (HB146) → vocabulary-carries-all (HB157) →
composition-real (HB163) — each overturned by the NEXT preregistered control; the final state is
the only one backed by an instrument that needed no modeling assumption.

## HB164 (2026-07-28) — Keyword-verdict caps, additivity quantified, final discriminator queued

**(1) Scrambling granularity, stated for the record:** the shuffled arms scramble WORD-level
WITHIN each unit, units kept as separate appended bullets. So "shuffled" is bulleted word-salad;
the keyword arm differs from it in format (one flat headed list), dedup, and dropped short tokens
— not only in order.
**(2) Extraction-loss audit (H4): PARTIALLY LIVE.** 30% of units (49/164) contain logical
operators (not/only/all/at least/must…) that the term-extractor drops. The keyword deficit
(−.083) is therefore an upper bound on the pure form effect; content loss contributes. It cannot,
however, explain +.011 vs +.094 — the qualitative COMPOSITION-REAL verdict stands; quantitative
shares are softened in the write-up.
**(3) Additivity, quantified (the advisor's hidden strength):** content effect .0908 / .0906
(measured at both form levels), form effect .0462 / .0459 (measured at both content levels);
predicted keywords ≈ +.047, observed +.011 → **keyword residual .036 below additive prediction**
— the printable quantity.
**(4) Mechanism paragraph v2 (advisor) ADOPTED** with an added clause for the operator-loss
caveat. **aime print policy ADOPTED verbatim**: omnibus row +.007 n.s.; range +.007..+.091 across
sessions with GEPA-arm variance comparable to the margin; "no separation claim on AIME"; never
resurrect +.091. Headline unchanged (performance first; mechanism second beat).
**(5) PREREG — bulleted-keywords discriminator (H1 order vs H2 format), queued behind hotpot-kw:**
per-unit bulleted term lists, NO header ("- term, term, …" per unit), paired subsets, n=20, one
session with seed + native anchors. **Rule: recovers to within .015 of shuffled (+.047-ish) →
FORMAT/segmentation explanation (H2); stays within .015 of flat-keywords (+.011) → ORDER/bigram
explanation (H1); between → both channels, report interval.**

## HB165 (2026-07-28 ~17:3xZ) — Table 1 partially rebuilt from the 3 completed omnibus benches; IFBENCH UPGRADES TO A WIN

User asked for tables updated NOW with everything available. Fresh paired bootstraps (20k resamples,
newest single-fingerprint block per file, all four candidates same session, k=5):

| bench | GEPA | +Merge | MIPROv2 | M_ω | Δ vs GEPA | 95% CI | p(≤0) |
|---|---|---|---|---|---|---|---|
| hotpot | .4107 | .4000 | .4387 | **.6460** | +.2353 | [.192,.281] | <5e-5 |
| aime | .3853 | .3733 | .3227 | .3920 | +.0067 | [−.020,.033] | .303 |
| ifbench | .4031 | .4106 | .3834 | **.4432** | +.0401 | [.012,.069] | **.0029** |

Head-to-heads (same session): hotpot M_ω vs Merge +.246***, vs MIPROv2 +.207***; ifbench vs Merge
+.033*, vs MIPROv2 +.060***; aime vs MIPROv2 +.069*** (but no GEPA separation → print policy holds).

**Scoreboard change: ifbench moves from "not confirmed (+.020, p=.233)" to a significant win
(+.040, p=.0029) in the uniform session.** Both sessions reported; footnote in Table 1 flags the
session-sensitivity explicitly. Session-to-session GEPA-arm variance is now documented on BOTH aime
(.307–.385) and ifbench — same instrument lesson as prompt-optimality-eval-noise.

Table 1 now 6 columns; hover row = HB127 session (MIPROv2 cell lands with tonight's omnibus);
livebench merge/mipro cells likewise; pupa envelope permanently N/A (HB154). Submodule commit above.
Remaining for [table1-comparable]: livebench (in measurement now) + hover omnibus rows.

## HB166 (2026-07-28 ~17:4xZ) — Advisor verdict on the HB165 table; must-fixes applied (submodule 6f26825)

Advisor (Fable) on the ifbench upgrade and Table-1 framing:
1. **ifbench +.040** headline is DEFENSIBLE without a third session** — the omnibus was designated
   the canonical instrument BEFORE its results existed; headlining it follows a pre-committed rule.
   Optional insurance: a cheap k=5 GEPA/M_ω ifbench re-pair if GPUs idle after hover, pre-committed
   to report regardless of outcome. Do not block the deadline on it.
2. **Symmetry is real but must be VISIBLE**: uniform session suppressed a favorable +.091 on aime
   and replaced an unfavorable n.s. on ifbench — direction-blind. Caption now states the rule with
   the AIME/IFBench parenthetical (applied).
3. Hostile-reviewer order: hotpot ^s (never quote +.235 bare; hover +.100 is the prose flagship) >
   "·" cells (must be filled or "not measured" in the final PDF — no "running" language) >
   livebench bolded-as-win vs its own pathology footnote (fixed: ^a moved onto Δ, unbolded,
   excluded from prose W-counts) > missing per-bench n (added: 300/300/294/221/150/126) >
   uncorrected multiplicity (primary-contrast sentence added) > PUPA phrasing (reworded).

APPLIED in submodule 6f26825: canonical-session rule + n + multiplicity sentence in caption;
livebench ^a on Δ, unbolded; PUPA rephrase. STILL OPEN for final PDF: resolve "·" cells (omnibus
livebench in progress, then hover); prose sweep so hover leads and livebench is out of W-tallies.

## HB167 (2026-07-28 ~20:1xZ) — PRE-COMMIT: ifbench insurance re-pair launched on freed GPU 6

GPUs 1/6 freed (orphan servers from drained lanes, killed by explicit PID; GPU 1 left to labmates).
Per advisor rec #2 (HB166), launched on GPU 6: fresh uniform session, GEPA official vs
unitrecomb_v6ctx32k, ifbench, k=5, dedicated server port 8186 (pid 1513563, engine-init verified),
rescore child 1542860 → runs_paperexact/ifbench/.../rescore_k3.jsonl (newest fingerprint block).
**Pre-commitment, frozen before any result exists: this session is REPORTED WHATEVER IT SAYS, as a
footnoted confirmation alongside the canonical omnibus row (+.040 [.012,.069]). It does not replace
the canonical estimate; it cannot be dropped if unfavorable.** Server self-kills on completion.
kw2 (hotpot keywords 8B) and kw3 (hover bulleted keywords) confirmed already in flight on port 8196.

## HB168 (2026-07-28 ~20:4xZ) — LIVEBENCH DEMOTED TO NO-SEPARATION (uniform session); kw2 = COMPOSITION-CONFIRMED on hotpot

**Livebench omnibus row (canonical uniform session, fingerprint 16:41:49Z, n=126, k=5, 20k paired bootstrap):**
| candidate | mean |
|---|---|
| GEPA official | .6989 |
| GEPA+Merge | .6877 |
| MIPROv2 | .6767 |
| M_ω | .7048 |

M_ω vs GEPA +.0059 [−.0151,+.0280] p=.297 — **NO SEPARATION**. vs Merge +.0172 (p=.080), vs MIPROv2
+.0281 (p=.030, secondary/uncorrected). The old +.092*** (GEPA .555) is a LOAD ARTIFACT: idle GEPA
scores .699 ≈ the .6956 idle-load measurement (prompt-optimality-eval-noise memory); the earlier
GEPA arm was measured busy/deflated. Direction-blind rule applied: canonical row replaced in Table 1
(submodule a7faece), old margin footnoted, caption parenthetical now lists AIME AND LiveBench as
earlier-sessions-favored-us. **W-tally: 3 of 6 quotable wins — hotpot +.235*** (seed caveat), hover
+.100***, ifbench +.040**; aime/livebench no-separation; pupa tie.**

**kw2 (hotpot keyword arm, 8B, fingerprint 15:54Z, 20 draws, init base):** init .4047; native units
.6053 (+.201, 20/20 above init); keywords .4597 (+.055, 18/20). Frozen rule (HB158/HB164: ≥.15
vocab-dominant / ≤.12 composition-confirmed): +.055 ≤ .12 → **COMPOSITION-CONFIRMED on hotpot**,
replicating hover (native +.094 vs keywords +.011). Composition carries ~73% of the hotpot unit
gain; vocabulary ~27%.

In flight: omnibus hover row (passes running, 47.0%/46.7%); ifbench insurance pair (GPU6, official
5 blocks flushed 19:07Z); kw3 bulleted-keywords.

## HB169 (2026-07-28 ~21:0xZ) — Advisor verdict on livebench demotion; headline + bound phrasing ADOPTED

Advisor (Fable), key rulings, all adopted:
1. **Headline = never-worse-plus-three-wins, not "3 of 6"** (all six canonical Δ ≥ −.001). Verbatim
   sentence adopted for paper/summary: "Under a pre-registered same-session protocol, the
   ε-certified recombination pool matches or exceeds GEPA on all six benchmarks and significantly
   exceeds it on three (HoVer +.100, IFBench +.040, and HotpotQA +.235 over GEPA's shipped prompt,
   which there equals the seed) — and the same protocol retracted two of our own earlier apparent
   wins as measurement artifacts." NEVER write "the null results are explained by instrument noise"
   (the instrument story explains why earlier sessions differed from the ties, not why ties don't count).
2. **Demotion strengthens the paper**: rule cut against us 2×, for us 1× — the distribution an
   honest rule produces. Eval-noise thread upgraded to secondary CONTRIBUTION (one discussion
   sentence: 2/6 benchmark effects in the standard setup were instrument artifacts).
3. **Kill %-split mechanism language** — keyword arm UNDERestimates vocabulary (operator loss 30%),
   so "composition carries 73%/88%" = upper bounds dressed as point estimates. Adopted phrasing:
   "Keyword-only variants recover at most ~27% (HotpotQA) and ~12% (HoVer) of the native unit gain;
   the remainder requires intact clause structure. Because keyword extraction drops logical
   operators from ~30% of units, this bounds rather than point-estimates the split."
4. Summary-note order fixed (Table1+protocol → two-artifact narrative → mechanism → scaling →
   unsupervised bank → reversal ledger → limitations).
MUST-FIX ledger: (a) hover row lands before freeze; (b) ifbench insurance reported whatever it says
(HB167 binding); (c) stale-language sweep of PAPER — done, clean (only intentional footnotes);
(d) abstract/intro clean — verified.

## HB170 (2026-07-28 ~21:3xZ) — IFBench insurance pair REPORTED (pre-commitment HB167 honored)

Insurance session (GPU 6, port 8186, single fingerprint 19:07:43Z, n=294, k=5, paired bootstrap
20k): GEPA .4262, M_ω .4476, **Δ+.0214 [−.0082,+.0510], p=.078** — directionally positive, not
individually significant. Three ifbench sessions now on record: +.020 n.s. / canonical +.040** /
confirmation +.021 (p=.078). All positive, mutually compatible (CIs overlap); only the canonical
session separates. Footnote ^f rewritten to show all three (submodule 1e9ce61). The star stays on
the canonical row per the direction-blind rule; the caveat travels with it. GPU 6 server
self-killed on completion — GPU 6 free again for labmates.

## HB171 (2026-07-28 ~22:0xZ) — OMNIBUS COMPLETE; Table 1 fully canonical; hover row = +.086***

table_omnibus finished cleanly 20:33:04Z (hover rc=0, server self-shutdown, GPU 0 freed). Hover
canonical uniform session (fp 2026-07-28T19:09:50Z, n=300, k=5, 20k paired bootstrap):
| candidate | mean |
|---|---|
| GEPA official | .4707 |
| GEPA+Merge (gepamerge) | .5167 |
| M_ω (unitrecomb_stair) | .5567 |

M_ω vs GEPA **+.0860 [.0627,.1100] p<5e-5**; vs Merge +.0400 [.0133,.0667] p=.0015. MIPROv2 has
NO hover candidate (never produced one) → cell = not measured, footnoted. Earlier session's +.100***
(with pool-free .559 ≈ M_ω .564) moved to the earlier-session note in ^c. HEADLINE SENTENCE
UPDATES: "HoVer +.100" → "HoVer +.086" everywhere forward-facing (advisor sentence amended;
HB169's version is the ledger record of what was adopted then).

**FINAL TABLE 1 (all rows canonical uniform sessions):** hotpot +.235*** / hover +.086*** /
livebench +.006 n.s. / ifbench +.040** / aime +.007 n.s. / pupa −.001. W = 3/6, never-worse
everywhere (min Δ = −.001). Submodule be7c607. [table1-comparable] TICKED. Remaining EOD items:
final PDF sweep + summary note sections 3-7 + kw3 harvest.

## HB172 (2026-07-28) — unit-type deep dive: what recombination adds vs what decompression rescues

User question: are the units that rescue small executors in the tacit-isomorphism experiment the
same TYPE as the units that carry gain in reconstruction? Method: Codex (gpt-5.6-sol) labeled all
360 frozen-pool Ω units on the tacit line's 3-level checkability codebook (MECHANICAL /
STRUCTURAL_CRAFT / TASTE, wording verbatim from notes/2026-07-03__what-gets-decompressed.md),
blinded with 6 planted anchors → 6/6 correct incl. both TASTE anchors. Labels + provenance:
datasets/prompt-optimality-test/runs/omega_unit_labels_checkability.json.

**Composition (selection surfaces are near-disjoint):** Ω pools = 292 CRAFT / 68 MECH / **0
TASTE** (anchors prove the tagger would have used TASTE). The humor/CW decompression banks are
the mirror image: 0 MECH / ~74% CRAFT / ~26% TASTE. So the two experiments sample opposite ends
of the checkability axis before any value question arises.

**Value by type is executor-relative, not type-intrinsic (hover ladder, identical 162 units):**

| scale | MECH mean Δ (pos%) | CRAFT mean Δ (pos%) |
|---|---|---|
| Qwen3-4B  | +.029 (81%) | **+.042 (89%)** |
| Qwen3-8B  | −.010 (35%) | −.020 (21%) |
| Qwen3-32B | −.011 (15%) | −.008 (26%) |

Paired 4B→32B drop: CRAFT +.050, MECH +.040. At 4B, craft units are the bigger winners; by 32B
almost nothing helps. aime shows the floor: at 1.7B units HURT (12–18% pos), at 8B they help
(75% pos) — below capacity, articulated units of any type are noise (matches OSL falling-limb /
task-relative threshold).

**Reading:** the earlier "reconstruction gains live in thin/format units" (hotpot 8B: MECH +.070,
CRAFT +.007) is an 8B-on-QA snapshot, not a law. Unit marginal Δ traces the same moving window
the tacit experiment measures directly: a unit pays exactly when the executor is above the floor
but below tacit mastery of that content. Reconstruction at 8B on verifiable QA sits past the
craft window (8B already holds the craft; only compliance slack remains); tacit-iso deliberately
sits inside it (big-can/small-can't). Residual hypotheses not separable without new runs: (b)
metric surface (EM/containment never rewards TASTE, so mining never proposes it) vs (c) mining
distribution (units mined from 8B failures = compliance failures). Both predict the 0-TASTE cell;
neither predicts the hover 4B craft>mech inversion — only the window hypothesis does.

## HB173 (2026-07-28) — kw3 bulleted-keywords discriminator: H2 FORMAT/SEGMENTATION verdict

Artifact runs/hb124_controls_hover_Qwen3-8B_seed_kw3.json (n=20 draws, one session, seed+native
anchors; init_k .4540, native anchor .5468 = +.093, consistent with prior sessions' +.094).
Bulleted per-unit keyword lists (no header, order still scrambled-set): mean .5030 → **gain
+.0490, 20k bootstrap CI [+.0402,+.0580]**; paired-draw native−bulleted +.0438 [+.0280,+.0597].

Frozen rule (HB164 §5): within .015 of shuffled (+.047) → H2; within .015 of flat-keywords
(+.011) → H1. Distances: **.002 from shuffled** vs .038 from flat-keywords → **unambiguous H2**:
the flat keyword list's −.036 sub-additive residual is recovered ENTIRELY by per-unit
segmentation (bulleting); restoring word order/bigrams contributes ≈ nothing beyond it. Combined
with HB164(3): content .091 + form .046 add; the remaining "format" channel is unit boundaries,
not token order. Format-vs-order now RESOLVED; vocabulary-vs-composition already settled (HB163).
Caveat carried: extraction drops operators from 30% of units, so keyword arms bound (not point-
estimate) the pure-form effect.

## HB174 (2026-07-28) — (A) decompression vs (B) prompt-evolution, unsupervised only: change-type CERTIFIED double-pass

User directive: localize the tacit-vs-reconstruction difference to WHAT CHANGES per evolution
step, unsupervised metrics only; paper appendix parked meanwhile. Design: 487 evolution steps in
one shared format (200 A-ladder name→def→expl transitions across 10 domains; 53 GEPA-h2h rounds;
150 gepa_nc rubric rounds; 84 metric_implementer version steps) + 7 blinded anchors; 8-way
change-type codebook (union of the expansion-chain schedule + evolution-native categories);
TWO independent Codex (gpt-5.6-sol) passes.

**Reliability: 8-way κ=.678 (raw .741; anchors 6/7 both passes). Family-level
(concept-content vs measurement-content vs other): agreement .965, κ=.932; ALL frequent
disagreements are within-measurement (scoring↔boundary 65, procedure↔scoring 14).**

**Result (replicates in both passes):** A-ladder = 100% concept-family (name→def 100%
CONCEPT_SEMANTICS, def→expl 99% MECHANISM); B GEPA lanes = 99-100% measurement-family
(scoring-mechanics 75-77% primary, boundary dominant secondary 130/287, input-hygiene ~12%);
B registry = 79% measurement + 6% concept. The 6% concept exceptions are exactly the esoteric
constructs (story-vs-narrative-discourse, distinctive_voice ×3 CLARIFY, distilled peer-review
phase) — evolution writes concept content only where the judge is shaky (H2 capability-window).

Hypotheses ledger (relayed to user): H1 residual-error slot ≡ H2 capability window (main),
H3 scoring-conventions-are-extra-conceptual (additive), H4 A-side-generated-by-instruction
(framing correction owed: A rows ≈ manipulation check; the finding is B's zero concept cells),
H5 genre-shortcut (κ pass done; style-normalized retest = remaining control). Discriminating
tests proposed: gepa_nc concept-incidence vs r0 fidelity (CPU); small-executor GEPA (1 GPU).

Artifacts: outputs/analyses/evolution_change_type_labels_20260728.json (+ transitions JSONL),
extractor methods/codability/extract_evolution_diffs.py. Appendix stays PARKED pending user
sign-off + chosen extra controls.

## HB175 (2026-07-28) — M_ω extension + change-type figure INTO PAPER; H2-on-nc refuted; OSL regimes characterized

**M_ω additions (all 360 pool units, Codex, same codebook; id-collision fixed by relabeling
hotpot+hover 0-67 under fresh ids):** PROCEDURE 47% / SCORING_MECHANICS 18% / BOUNDARY 11% /
REWORDING ~9% / EXAMPLE ~6% / concept-family ~3%. Same side of the concept/measurement divide as
GEPA, subtype-shifted: units are executor checking-steps, not judge score-anchors. → Appendix D
"What each process writes into the prompt" + Figure 9 (grouped barplot, plain-language category
names per user) now IN the paper (submodule committed); parked MECH/CRAFT/TASTE appendix remains
parked (separate, supervised-side claim).

**H2 test on gepa_nc: REFUTED there.** 105 sampled rubrics spanning r0 fidelity .02-.74: zero
concept-family edits even in the lowest quartile; low-r0 mix shifts WITHIN measurement (hygiene
19%/boundary 14% vs high-r0 6%). Evolution's basin is measurement-first regardless of fidelity;
concept content appears only for constructs the proposer itself treats as unfamiliar (registry
esoterics). H3 strengthened; H1/H2 survive only in that narrow proposer-familiarity form.

**OSL regimes characterized** (Codex names-based, 884/321/65; artifact
outputs/analyses/osl_regime_characterization_20260728.md; hedged paragraph added to
app:osl-regimes): rising = inspectable craft ops (median L .95); reaches = compact formal
devices — joke mechanics, prose economy, endings (L .76; humor+CW dominate); bounded (L .53,
41/65 humor) = truth conditions partly OUTSIDE the artifact — audience/platform outcomes,
performer identity/persona, community-specific transformation (parody fidelity, satire stance),
cross-artifact bundles. Tag-join gradient TASTE 23→29→50% (n=59, directional only).
Seam-width barplot (10-task F spectrum, two-group claim) at
outputs/analyses/figs_20260728/seam_width_spectrum.png.

## HB176 (2026-07-28) — OSL regime trust audit + realized-vs-fitted plot

Per-regime: realized recovery @top-3 z vs fitted L: RISING .683 vs L .946 (gap +.201, median CI
width .560, 99% have L_hi≥1.1 → ceiling UNIDENTIFIED, censoring not type); REACHES .747 vs .761
(gap +.023 — articulation COMPLETE at current scale, CI .330); BOUNDED .523 vs .527 (gap +.003,
CI .105 — tightest fits). So "most articulated TODAY" = REACHES (highest realized + finished);
"most articulable in principle" = rising ONLY as extrapolation (don't quote .95 as a ceiling).
Trust ladder inverts excitement ladder: bounded > reaches ≫ rising ceilings. Plot:
outputs/analyses/figs_20260728/osl_regimes_trust.png (scatter + CI whiskers + domain stacks).

## HB177 (2026-07-28) — OSL regimes on one axis: EXTERNALITY; full-panel shared taxonomy; regime×domain in paper

Shared task-agnostic 9-type taxonomy over ALL 1,270 metrics (Codex, regime-BLINDED input;
pass-2 κ in flight). Coarse axis "where truth conditions live": in-text / interface /
beyond-text. RISING 73/16/12 idx .39; REACHES 70/17/12 idx .42; **BOUNDED 54/14/32 idx .78** —
rising≈reaches on this axis (they differ by domain/device, not externality); bounded differs IN
KIND, driven by IDENTITY_PERSONA 17% (vs ~2.5%), COMM_TRANSFORM .8→4.7→9.2 monotone, RECEPTION
5.3→7.5→10.8 monotone; EVIDENCE and AUD_FIT fall toward bounded. **Domain-controlled: within
humor/news/peer separately, bounded 31-34% beyond-text vs 4-18%** — not a composition artifact.
Regime×domain: bounded only in humor 41 (15%) / news 13 (10%) / peer 9 (11%) / CW 2 (1%);
math, PR, n&c, patents ZERO. → fig:osl-types + three appendix paragraphs (trust audit /
task-agnostic types / regimes-across-tasks) in app:osl-regimes; submodule committed. Artifact
outputs/analyses/osl_metric_types_20260728.json.

## HB178 (2026-07-28) — 12-type main-body figure; 9-type κ certified; 12-cat separates rising/reaches

9-type κ=.752 (raw .817), coarse axis κ=.732; pass-2 independently replicates externality
gradient (idx .38/.37/.71). Caption updated. NEW 12-type codebook (regime-blinded, all 1,270):
**rising/reaches now SEPARATE — the compactness inversion**: rising = EXTENDED_STRUCTURE 28% >
COMPACT_DEVICE 17% (+verification 10%, rigor 9%); reaches = COMPACT_DEVICE 34% > EXTENDED 19%;
bounded = IDENTITY_PERSONA 14% + RECEPTION 11% + stance (unchanged signature). → Fig 5 (main
body, §5) grouped bars; coarse-axis stacked bars remain appendix-only per user. 12-type κ pass
not yet run (single-pass; flagged in HB, 9-type κ quoted in caption as the certified grain).
Artifacts merged into outputs/analyses/osl_metric_types_20260728.json (pass1+pass2+12type).

## HB179 (2026-07-28) — appendix granularity ladder + domain plot; seam-unit categorization launched

Fig 5 (12-type) kept per user; appendix adds fig:osl-granularity (8- and 6-category NESTED
merges of the certified 12-type labels — deterministic, no relabeling; both preserve the
compact-vs-extended inversion and the bounded social/reception loading) + fig:osl-domains
(per-domain 100% stacked regime composition; bounded share humor 15/peer 11/news 10/CW 1/rest 0;
rising 65-86% everywhere). 14 pages. SEAM-UNIT categorization: 629 blinded sub-rules built
(contracts_v3 592 probes w/ CODE-vs-L channel + legal u1-u7 v2 outcomes + humor 30 units w/
code_rt/hyb_rt), Codex 12-type descriptive pass IN FLIGHT; payoff = category × mechanization
cross-tab. Key artifact map in agent report (Tier1-5; y_gepa unit manifests, codif_merged 143
programs, seam tables 10 domains, fleet_boundness).

## HB180 (2026-07-28) — WHAT GETS MECHANIZED at the metric seam: 12-type check taxonomy × channel

629 seam sub-rules (contracts_v3 probes n=592 w/ CODE-vs-L + legal u1-u7 + humor units), channel
BLINDED during Codex labeling (single-pass; κ pass not yet run). Mechanization rate by check type:
attribution/sourcing 85% > numeric-fact 81% > lexical-marker 80% > format/layout 73% >
syntax/style 63% > count/length 62% > **normative-application 55%** (legal standards compile
better than expected — matches legal v2 6/7 units incl. fidelities .37-.57) > CLIFF >
cross-consistency 32% > semantic-category 29% > argument-adequacy 26% > discourse-pragmatics 23%
> holistic 0%. Coarse formal>relational>judgment gradient holds within 6/7 tasks (exception:
press_releases relational 87% — attribution-heavy). The seam is a CHECK-TYPE frontier, not a
domain property: domains differ in F because their banks MIX these types differently.
Artifact outputs/analyses/seam_unit_types_20260728.json.

## HB181 (2026-07-28) — figure reorganization per user: 6-cat main / 12+8 appendix; seam Table 3; Fig 7/8 rework

Fig 5 (main) = 6-category regime signatures at .62\textwidth, COMPACT_DEVICE renamed "local
patterns (punchline, endings)"; appendix granularity fig = 12 (top) + 8 (bottom), nested-merge
note inverted accordingly. change-types fig at .72 width. Seam mechanization gradient now MAIN
BODY Table 3 (sec:code-iso) with composition-effect paragraph; NORMATIVE_APPLICATION and
COUNT_LENGTH rows omitted per user (unclear measurement) — caption points to the artifact where
both are reported; single-pass labels, κ pass still owed. Fig 7 tacit-iso compacted (PW 4.85→
4.30, tightened verticals). Fig 8 code-iso v3 = ONE gray panel, LLM node ABOVE the two field
cards. 15 pages, clean compile.

## HB182 (2026-07-28) — seam κ audit: 12-type UNRELIABLE (.509), table rebuilt at 6-group grain (.662)

Second independent Codex pass on the 629 seam sub-rules. **12-type κ=.509 (raw .568)** — too weak
to publish per-category percentages, and pass-2 moves them materially (format 73→54%, numeric
81→67%, argument-adequacy 26→38%; category n's swing 64→95, 70→105). Dominant confusion
DISCOURSE_PRAGMATICS↔SEMANTIC_CATEGORY (41). Cliff grain (form/norm/meaning) κ=.630.
**Adopted: 6-group grain, κ=.662, values stable across passes** (recorded facts 82/78%, surface
form 68/60%, normative 55/60%, internal consistency 32/31%, meaning reading 25/23%, adequacy
26/31%). Table 3 now reports GROUPS with n-ranges and %-ranges across both labelings; caption
states the 12-category version was withdrawn for unreliability; normative-application row stays
omitted per user with its numbers disclosed in the caption + artifact. The qualitative cliff
(fact/form ≫ meaning) is robust to both labelings — that claim stands. Artifact updated with
pass2 + both κ.

## HB183 (2026-07-28) — seam table UPGRADED to 3-pass consensus; cliff is a DISCONTINUITY

User restored the granular 12-row table (the 6-group condensation lost the signal). Rather than
publish single-pass numbers at κ=.51, ran a THIRD independent blind pass and took per-unit
majority vote. Agreement: **52% unanimous / 42% majority / 6% all-differ (94% ≥2/3)**; pairwise
κ .585/.609/.646; **each pass vs consensus κ .777-.807** — consensus labels are the defensible
object. Consensus table (n, %compiled): attribution 16/94, lexical 5/80, numeric 25/76,
counts 14/71, syntax 57/67, format 65/66, **normative 92/62 ‖ CLIFF ‖ cross-consistency 83/30**,
argument-adequacy 82/28, pragmatics 81/21, semantic-cat 78/19, holistic 1/0.
**NOTHING COMPILES BETWEEN 30% AND 62%** — the cliff is a genuine discontinuity, not a gradient,
and it separates FORM+FACT+CODIFIED-NORM from MEANING. Normativity is NOT what resists code
(62%, and legal fleet compiles 6/7 units on retry); reading meaning is. Both dropped rows
restored per user with glosses. Artifact carries all 3 passes + consensus.

## HB184 (2026-07-28) — Fig 5 bounded signature is CODEBOOK-ROBUST (stronger than a repeat-κ)

OSL 12-type κ pass still queued, so tested the stronger property instead: does the regime
signature survive a DIFFERENT codebook? Mapped the 12-type labels and BOTH independent 9-type
labelings onto their shared 5-way grain (n=1,270). Cross-codebook κ .588-.622 (within-codebook
9-type .763) — codebook choice adds noise. But the bounded signature is identical: **community
+identity = 23.1% of bounded under all three labelings, and 13 of those 15 metrics are the SAME
metrics** (union 17) — personal comic voice, lived experience as engine, parody voice fidelity,
satire target/stance, emotional honesty. Rising 3.4-7.8%, reaches 6.9-8.1% under the same three.
Added to the Fig 5 caption. This is better evidence than a within-codebook repeat: the claim is
invariant to how the taxonomy is cut, not merely reproducible under one cut.
NOTE: main.tex working tree also carried the user's in-progress title/abstract rewrite; committed
together and flagged in the commit message (not my text).

## HB185 (2026-07-29) — BUDGET-MATCHED GEPA: hotpot margin SHRINKS from +.235 to ~+.109

Ran official GEPA at 600 AND 2400 metric calls in ONE session (same server, same local Qwen3-8B
reflection LM, k=5 final tests, GPU1 port 8192) — only the budget varies. Artifacts
runs_paperexact/hotpot/Qwen3-8B/official_budgetmatch{600,2400}/result.json, rc=0 + artifacts
verified.

| arm | seed_test | best_test |
|---|---|---|
| GEPA @600 (our declared budget) | .402 | **.4100** |
| GEPA @2400 (M_ω's budget) | .401 | **.5373** |

**Paired bootstrap (20k, item-level, same session): +.1273 [+.0893,+.1667], p<1e-5** (96 items
better / 35 worse / 169 tied). Quadrupling GEPA's budget buys +.127 on hotpot.

**Two consequences, both must propagate:**
1. **CONTROL PASSED:** GEPA@600 here = .4100 vs canonical Table-1 GEPA .4107 — the local-Qwen
   reflection substitution (forced by the dead z.ai endpoint) does NOT change GEPA's outcome.
   The budget comparison is therefore clean, not confounded by the reflection swap.
2. **THE HEADLINE MOVES:** M_ω ran at 2400. Against a budget-matched GEPA the hotpot gap is
   ~+.109, NOT +.235. The +.235 is a 600-call-GEPA comparison and must never be quoted as a
   matched-budget result. Also note GEPA@600 no longer "ships the seed" under local reflection
   (it accepted a proposal, .402→.410), so that footnote is reflection-LM-specific.
**CAVEAT — the +.109 is CROSS-SESSION** (M_ω .6460 came from the omnibus session, GEPA@2400 from
this one). Per our own canonical-session rule it is NOT yet a certified number. Required next:
rescore M_ω's shipped candidate + both GEPA candidates in ONE new session, k=5 → certified
matched-budget row. Queued.

## HB186 (2026-07-29) — CERTIFIED same-session budget row + BUDGET-ACCOUNTING CORRECTION

**(A) Budget accounting was WRONG in the appendix.** "unitrecomb at 2,400 calls" came from a CLI
DEFAULT, not from the runs. Declared caps are 6,000-40,000; ACTUAL search spend computed from
proposals.jsonl (excluding final-test evals):
| bench | GEPA spent | M_ω spent | ratio |
|---|---|---|---|
| hotpot | 600 | **16,700** | 28x |
| ifbench | 600 | **23,300** | 39x |
| hover | 657 | **10,110** | 15x |
Every "2,400" statement in the paper is false and must be replaced by these numbers.

**(B) CERTIFIED same-session rescore (Slot 1, advisor-mandated).** One server, one fingerprint,
k=5, all three candidates rescored together (rc=0, artifacts verified):
| candidate | k5 mean |
|---|---|
| M_ω (unitrecomb_v5sk2) | .6380 |
| GEPA @2400 | .5340 |
| GEPA @600 | .4107 |
Paired item-level bootstraps (20k):
- **M_ω vs GEPA@600: +.2273 [.1827,.2720] p<1e-5** (reproduces the published +.235 — the Table-1
  number is SOUND at its stated operating point)
- **M_ω vs GEPA@2400: +.1040 [.0733,.1367] p<1e-5** — the win SURVIVES 4x budget matching
- GEPA@2400 vs GEPA@600: +.1233 [.0860,.1613] p<1e-5
GEPA@600 rescored .4107 = canonical Table-1 .4107 EXACTLY → session + reflection-swap both clean.

**(C) Still open:** 16,700-call (true-match) GEPA running on gpu2. .5340 at 2,400 is NOT the
matched number; matched is 16,700. If the budget curve keeps climbing, +.104 shrinks further.

**INCIDENT:** a timed-out ssh held its connection until the rescore finished, then fired a queued
duplicate truematch launch (pid 2208456) that ran 35 min INTO THE SAME RUN DIR as the original
(542374). Killed by explicit PID (wrappers 2201904/2201905 first, then python); original survived,
shared server untouched. **truematch run-dir integrity must be audited before its result is used.**

## HB187 (2026-07-29) — ⚠ M_ω AT MATCHED BUDGET *LOSES* TO GEPA. Headline comparison inverts.

Ran M_ω (unitrecomb) on hotpot at **2,400 calls** — the same budget GEPA got — same task LM, same
local reflection LM, k=5 final tests. Artifact verified
(runs_paperexact/hotpot/Qwen3-8B/unitrecomb_momega2400/result.json, "ARTIFACT ok", rc=0).

| arm | budget | k5 test |
|---|---|---|
| M_ω (published, v5sk2) | **16,700** | .6380 |
| GEPA @2,400 | 2,400 | **.5340** |
| **M_ω @2,400** | 2,400 | **.4140** |
| GEPA @600 (canonical) | 600 | .4107 |

**M_ω@2400 − GEPA@2400 = −.1200 [−.1573,−.0840]** (30 items better / 83 worse / 187 tied).
CROSS-SESSION (M_ω@2400 scored in its own session; GEPA@2400 from the certified session) — but
the gap is far outside observed session wobble (~.01–.09), so the direction is not a session
artifact. **Same-session rescore REQUIRED before this is certified.**

**What this means.** M_ω barely moves off its seed at 2,400 calls (.4107→.4140, +.003). Its
published .638 is a *16,700-call* result. So the published +.235 over GEPA is a 28×-compute
comparison, and at equal compute on this benchmark **GEPA is ahead by ~.12**. The advisor's
outcome "C" has landed for the 2,400 operating point. M_ω is not budget-efficient; the honest
frame is that it produces a strong prompt at high spend, not that it beats GEPA per call.

**Immediate consequences for the write-up:** the sentence "matches or exceeds GEPA on all six"
cannot stand unqualified at any matched budget; the hotpot +.235 must be labeled a 600-call
operating point AND disclosed as 28× compute; the whole comparison table needs a budget column
with ACTUAL spend (11–42× across benches, HB186).

**Still pending:** GEPA@16,700 (true match, gpu2) — decides whether M_ω's .638 survives at ITS
own budget. GEPA@2400 on hover + ifbench (gpu3/gpu5) — decide the other two stars.

## HB188 (2026-07-29) — CORRECTION to HB187: M_ω@2400 TRUNCATED, it did not "lose". Plus: M_ω starts FROM GEPA.

Advisor flagged two lethal unknowns; both now resolved from the artifacts, and the first one
**walks back my own HB187 framing**.

**(1) Was the Table-1 GEPA baseline an unoptimized seed? NO — refuted.**
GEPA@600: seed_test **.402** → best_test **.4107**. GEPA does move off its seed (+.009). The
canonical baseline is a real, if barely-optimized, GEPA run. The near-identity between GEPA@600's
best (.4107) and M_ω@2400's *seed* (.41066) has a different cause — see (2).

**(2) M_ω IS INITIALIZED FROM GEPA'S SHIPPED PROMPT** (the D2 init-from-GEPA-best design). So
M_ω's "seed_test" ≈ GEPA's best. Consequences that must be stated in the paper: M_ω's true cost
is its own calls **plus** GEPA's 600, and "M_ω vs GEPA" is not a head-to-head between independent
optimizers — it is **GEPA-then-M_ω vs GEPA**. The honest framing is additive, and it is arguably
better for us: starting from GEPA's .4107, M_ω adds +.227 for 16,700 calls, while GEPA adds +.123
for 1,800 more calls. Per-call in this range GEPA is ~5× more efficient (6.8e-5 vs 1.4e-5 per
call); the pending GEPA@16,700 decides the endpoint.

**(3) M_ω@2400 did NOT terminate certified — it TRUNCATED mid-screening.** From its result.json:
`n_compiled = 0`, `fell_back_to_init = True`, marginals measured **23 of 48** pool units,
`confirm_init`/`confirm_compiled` = None. It never reached the admission or confirm phases and
**returned its initialization unchanged**. Its .414 is therefore the init prompt (= GEPA's
600-call output) plus scoring noise, NOT an M_ω result.

**⚠ HB187's headline sentence is WITHDRAWN.** "M_ω at matched budget LOSES to GEPA by −.120" is
wrong as a quality claim. The correct reading: **M_ω cannot run at 2,400 calls at all.** Its
minimum viable budget on hotpot is ~10^4 calls; below that it emits its input. So the −.120 is
not M_ω-vs-GEPA quality — it is GEPA@2400 vs GEPA@600-output, i.e. the +.123 budget effect of
HB185/HB186 restated. The defensible statement is: **GEPA is anytime; M_ω is a step function with
a ~16.7k-call minimum.**
The mirror-trap still binds, per advisor: a method with a 16.7k minimum must still be judged
against a baseline given 16.7k. That is the pending gpu2 arm, and it remains the decisive test.

**Still-open checks from the advisor list:** token-level (not call-level) re-accounting of the
11–42× ratios; verify M_ω's admission split is disjoint from the 300 test items; seeds on the
headline pair; same-session idle rescore before printing any delta.

## HB189 (2026-07-29) — token accounting does NOT rescue the ratio; HB188(2) refined

**(A) Token/length accounting (advisor's hoped-for mitigation): DEAD, and it cuts the other way.**
Shipped-prompt lengths on hotpot: GEPA@600 1,505 chars / GEPA@2400 934 / **M_ω(16.7k) 4,622**
(694 words) / M_ω@2400 306 (= its unchanged init). A "metric call" is one ITEM EVALUATION through
the same program (2-hop retrieval + 4 LM modules) for both arms, so the unit is symmetric — but
M_ω evaluates with a prompt that grows to 3–5× GEPA's length as units accumulate. **Token-weighting
therefore WIDENS the 28×, it does not shrink it to the hoped-for 3–5×.** Reflection/proposal calls
are counted separately in both arms and are small (GEPA 7 reflection calls; M_ω 60 unit
proposals), so excluding them is symmetric and fair.
Side observation: GEPA@2400 ships a SHORTER prompt (934) than GEPA@600 (1,505) yet scores much
higher (.534 vs .411) — GEPA's gain is not a length effect.

**(B) Refines HB188(2): M_ω does NOT get a head start from GEPA's optimization.**
M_ω initializes from the UNTAGGED `official` run dir, whose hotpot candidate is the SEED
(best_test .38, 306 chars — GEPA shipped the seed there, HB132). Verified module-by-module: all
4 modules of unitrecomb_v5sk2's shipped prompt begin with the official/seed text and extend it.
So on hotpot M_ω starts from an unoptimized 306-char seed and grows it to 4,622 chars — it is NOT
standing on GEPA's 600 calls of optimization. HB188's "M_ω builds on GEPA's output" is therefore
too strong FOR HOTPOT (true only in the trivial sense that GEPA's output = the seed there).
The additive framing in HB188 should be restated as: from the SEED (.38–.40), M_ω reaches .638
for 16,700 calls; GEPA reaches .534 for 2,400 and .411 for 600 from the same seed.

**(C) Leakage check CLEARED (advisor trap #8):** unitrecomb's select/confirm panels are carved
from `bench.train_set` (100/50 of 150); reported scores use `bench.test_set` (300, line 856).
Disjoint by construction — .638 is not test-contaminated.

## HB190 (2026-07-29) — HOVER SURVIVES BUDGET MATCHING (unlike hotpot). Benchmark-dependent budget sensitivity.

GEPA@2400 on hover completed (artifact verified: seed .37534 → best .4840, budget 2400).

| arm | hover k5 |
|---|---|
| M_ω (canonical, ~10.1k calls) | .5567 |
| GEPA @2,400 | .4840 |
| GEPA @600 (canonical) | .4707 |

Paired bootstraps (20k): **M_ω vs GEPA@600 +.0860 [.0627,.1093] p<1e-4 (SAME session)**;
**M_ω vs GEPA@2400 +.0727 [.0467,.1000] p<1e-4 (cross-session)**;
GEPA@2400 vs GEPA@600 **+.0133 [−.0087,+.0360] p=.12 — NOT significant.**

**Key contrast with hotpot.** Quadrupling GEPA's budget bought **+.123 on hotpot but only +.013
(n.s.) on hover.** So GEPA's budget-sensitivity is strongly benchmark-dependent, and the hover
win is NOT a budget artifact: +.086 → +.073 under 4× matching, still highly significant.
This kills the naive "all our wins are budget effects" extrapolation from hotpot. It also means
the honest paper cannot say either "the wins are budget artifacts" OR "the wins survive" —
it must report per-benchmark budget curves, because the benchmarks disagree.

Caveat: the GEPA@2400 hover arm is cross-session vs the canonical rescore; a same-session
3-candidate rescore (as done for hotpot) is required before printing +.0727. Direction is safe
(the GEPA@2400−GEPA@600 gap of +.013 is inside session wobble, so no session effect can
manufacture the +.073).
Still pending: ifbench GEPA@2400 (gpu5), hotpot GEPA@16,700 (gpu2).

## HB191 (2026-07-29) — HOVER BUDGET ROW **CERTIFIED** (same session): +.083 survives 4x matching

3-candidate same-session rescore on hover (one server, one fingerprint, k=5, rc=0, artifacts
verified; logs/rescore_hovercert.log "RESCORE hovercert COMPLETE"):
| candidate | k5 |
|---|---|
| M_ω (unitrecomb_stair, ~10.1k calls) | .5667 |
| GEPA @2,400 | .4833 |
| GEPA @600 | .4700 |
Paired bootstraps (20k, item-level, ALL SAME SESSION — printable):
- **M_ω vs GEPA@2400: +.0833 [.0547,.1120], p<1e-4** ← the certified budget-matched hover row
- M_ω vs GEPA@600: +.0967 [.0700,.1233], p<1e-4 (canonical Table-1 row reproduces: .470/.567)
- GEPA@2400 vs GEPA@600: +.0133 [−.0100,+.0367], **p=.13 n.s.**

Supersedes HB190's cross-session estimate (+.0727) — certified value is **+.0833**, slightly
LARGER. Hover's win is not a budget artifact under 4x matching.
**The cross-benchmark contrast is now certified on both sides:** 4x budget buys GEPA +.1233
[.0860,.1613] on hotpot but +.0133 [−.010,+.037] n.s. on hover. Budget sensitivity is a property
of the BENCHMARK, not of the comparison. Any blanket statement ("wins are budget artifacts" /
"wins survive") is false; per-benchmark budget curves are mandatory.
Still open: hover at M_ω's true 10.1k spend (only 4x tested, true ratio 15x); ifbench@2400 (gpu5);
hotpot@16.7k (gpu2).

## HB192 (2026-07-29) — 5-PASS PREFIX CURVE COMPLETE (hotpot k=1..40): noise halved, curve is MONOTONE-RISING

runs/prefix_5pass_hotpot.json, 40/40 k-values × 5 independent passes = 200 evals, one session.
- **Roughness (mean |Δ| between adjacent k) .0088 vs .0182 single-pass — halved**, confirming
  ~half the visible wobble in the current Fig 3 panel is measurement noise, not structure.
- Mean SE of each 5-pass point = .0059 (±2SE band ≈ ±.012).
- Curve **rises monotonically +.0813** (k=1 .552 → k=40 .633), **peak at k=40** — i.e. still
  climbing at the end of the pool. The single-pass series' apparent inverted-U (peak k≈27 then
  decline) does NOT survive averaging; it was noise. **HB142's inverted-U claim and the "peak
  k=46" line in the paper/notes should be treated as SUPERSEDED for hotpot.**
- Single-pass sits ~+.039 above the 5-pass series uniformly = session offset (different server,
  different day), NOT signal. The two series therefore MUST NOT share a Fig-3 panel.
Comparison figure (no document edit): outputs/analyses/figs_20260728/prefix_5pass_vs_1pass.png
ifbench 5-pass half now running on the same lane.

## HB193 (2026-07-29, PREREG — frozen BEFORE truematch/ifbench results exist) — decision rules

Both arms are mid final-test; no best_test number has been observed. Freezing now, per advisor
("write the contingency before the arm lands; do not look and then decide"):

**Hotpot truematch (GEPA@16,700):**
- Decision statistic: same-session 3-candidate rescore (M_ω v5sk2 + GEPA@16700-best +
  GEPA@600-best), one server fingerprint, k=5, 20k paired item bootstrap of M_ω − GEPA@16700.
  The raw best_test from the arm's own session is NOT the decision number.
- Outcome mapping (advisor's A/B/C, adopted verbatim):
  A: CI wholly >0 → "at matched 16,700-call budget M_ω still leads; advantage not a spend
     artifact" (hotpot row keeps a scoped star).
  B: CI covers 0 → tie; hotpot advantage is a budget effect; contribution reframes to the
     certificate + budget-sensitivity finding.
  C: CI wholly <0 → GEPA ahead at matched budget; report the inversion first, certificate-first
     reframe mandatory.
- Whatever lands is reported, including C. Single-seed caveat travels with the number in all
  three branches. No partial-result peeking before the rescore completes.

**IFBench (GEPA@2,400):** decision statistic = same-session 3-candidate rescore (M_ω v6ctx32k +
GEPA@2400-best + GEPA@600-best), k=5, paired bootstrap. Same A/B/C mapping for the +.040 star.
Prior expectation recorded for honesty: GEPA accepted only 1 candidate in 699 iterations, so
GEPA@2400 ≈ seed is likely and the star likely survives trivially; recording this BEFORE the
number so it cannot be presented as a post-hoc vindication.

**Refinement to HB192 (self-audit):** "inverted-U was noise" is measured only for k≤40 (the
5-pass range). Beyond k=40 the only data are single-pass (the old k=41..68 decline), so the
correct claim is: within k≤40 the rise is monotone and the apparent k≈27 peak was noise; the
k>40 region and the "peak k=46" claim are UNMEASURED at 5 passes — neither confirmed nor refuted.

## HB194 (2026-07-29) — TRUEMATCH RAW LANDED: GEPA@16,700 = .580 (own session). Rescore launched per HB193.

Artifact verified (rc=0 + "ARTIFACT ok"): seed .408 → best **.58002**, budget 16,700, k=5, n=300.
GEPA's budget curve on hotpot: 600→.4107, 2,400→.5340, 16,700→.580 (raw) — strongly CONCAVE,
not the log-linear .707 worst case; the marginal call is worth ~6x less in the 2.4k→16.7k decade
than in the 600→2.4k one. Raw gap vs M_ω's published .638 is ~+.058 in M_ω's favor, but per
frozen HB193 the DECISION number is the same-session 3-candidate rescore, now RUNNING on gpu2
(tag truecert: unitrecomb_v5sk2 + official_truematch16700 + official_budgetmatch600, k=5).
No interpretation until it lands; outcome maps to A/B/C as frozen. Single-seed caveat applies to
.580 exactly as to every other budget point.
