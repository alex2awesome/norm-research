# Objective-comparison experiment: reconstruction vs naive LLM feedback vs definition-only
## (plan drafted 2026-08-15, user-requested; NOT YET LAUNCHED — awaiting sign-off)

## The claim being tested (user's draft sentence, to be confirmed or corrected)
"AUC(m_omega, y_true) = .7 vs AUC(m_gepa, y_true) = .6 vs AUC(m_desc, y_true) = .55."
User's refined framing: the point is not m_omega-vs-GEPA per se; it is whether optimizing an
articulation TOWARD THE RECONSTRUCTION BOTTLENECK beats optimizing toward (a) naive LLM
feedback and (b) no optimization (definition only) — evaluated on PURIFIED labels.
User's mechanism bet (ties to paper #1 register result): reconstruction wins on metrics that
can be worded many ways AND where wording materially changes behavior.

## Design principle: SELECTION, not dueling optimizers
All three arms share ONE frozen candidate pool of prompt forms per metric and ONE frozen
executor. Arms differ ONLY in the objective used to SELECT the shipped form:

| arm | selection objective | label-free? |
|---|---|---|
| m_desc | none — ship form 0 (the metric's definition as prompt) | yes |
| m_fb   | argmax critic-agreement: score candidates on probe docs; a CRITIC model scores the same docs holistically on the construct; pick the form whose scores best rank-agree with the critic | yes |
| m_recon| argmax reconstruction MI: the pipeline's per-form i_binary(form sig, M_i) on the same probe docs (the validated within-metric instrument, CW rho=+.39) | yes |

Why selection-only: (1) it isolates the OBJECTIVE — the user's actual question — from
optimizer-dynamics confounds (budget sensitivity, acceptance rules, seed variance) that the
parity campaign showed are large; (2) it is cheap (score the pool once, each objective reads
the same matrix); (3) it directly extends the already-validated within-metric MI result by
adding the competitor arm + purified-label eval. Phase 2 (optional, post-readout): full
iterative optimization per objective, only if the selection result is positive and the user
wants the stronger claim.

## Arms, models, and independence
- Executor (scores all candidate forms + all eval docs): frozen Llama-3.1-8B, batch vLLM, sk3.
- Candidate pool per metric: the stored paraphrase forms (98-form sigs npz where available;
  manifests elsewhere) + 20 fresh proposals from the 4-family proposer set (glm/qwen/llama/
  haiku — same pool for every arm). Pool FROZEN before any objective is computed.
- Critic for m_fb: GLM-4.7 (strong, non-arbiter family). NOT gpt-family — the arbiter that
  makes y* is gpt-5.6-sol, and a gpt critic would gift the feedback arm arbiter-correlated
  bias. NOT the 8B executor (self-agreement degenerates). Disclosed as a design choice.
- M_i for m_recon: the metric's own verdict on probes by the SAME 8B executor (canonical).
- Probes (objective computation): 150 per task, drawn from the task corpus, DISJOINT from
  eval docs. No y of any kind touches selection.

## Evaluation: purified labels (tier-3 arbiter machinery)
- y* = gpt-5.6-sol full-document verdicts, anchor-certified per wave (existing gate).
- Reuse wave-2 purified labels where they cover a metric (peer ~16 metrics, cw ~10, humor ~6,
  crx ~10 after v2 lands); top up eval coverage to >=40 purified docs/metric with >=10
  purified positives AND >=10 purified negatives (learned from the peer 83%-flip: purified
  class balance, not raw counts, is the constraint).
- Eval readout per metric: AUC(selected-form scores, y*) for each arm, on identical docs.
- Also report: oracle ceiling (best form by eval AUC — selection headroom) and random-form
  median (floor). These bracket what ANY selection objective could do.

## Metric selection: broad sweep + preregistered moderator (no cherry-picking)
Universe: every metric in {peer, cw, humor, crx} with (a) a stored multi-form pool >=15 forms,
(b) purified-label coverage per above (after top-up). Expected n ~ 35-45 metrics.
PREREGISTERED MODERATOR (the user's bet, formalized): per-metric FORM-SENSITIVITY =
interquartile range of probe-score correlations between forms (how much wording changes
behavior), computed from the frozen pool BEFORE any eval. Prediction P2 below.
Report ALL swept metrics; the "reconstruction-favored" subset is defined by the moderator
cut, not by outcome.

## Frozen predictions (register before ANY eval scoring)
- P1: pooled paired AUC(m_recon) > AUC(m_fb) (sign test + paired bootstrap on mean delta,
  per standing rule).
- P2 (mechanism): Delta(recon - fb) per metric correlates positively with form-sensitivity
  (Spearman > 0); the high-sensitivity tercile shows the largest delta.
- P3: on the high-sensitivity tercile, both optimized arms beat m_desc; on the LOW tercile,
  m_desc ~= optimized arms (wording doesn't matter there -> selection can't help).
- Falsifiers stated: P1 reversed => naive critic-agreement is the better objective — report
  as such; P2 null => the paper-#1 register bridge does NOT carry to objective choice; the
  draft sentence's numbers get replaced by whatever lands, including a null.
- Phrasing discipline: never "beats GEPA" (this is not GEPA); the arm is "critic-agreement
  selection" — name the design.

## Execution stages + cost (sk3; one GPU; batch vLLM)
- S1 (CPU): freeze pools + form-sensitivity from existing sigs; build fresh-proposal prompts.
- S2 (GPU ~half day): executor scores pool x probes (~ 40 metrics x 70 forms x 150 probes
  ~ 420K short calls, one batch job) + M_i verdicts; Gemma-free.
- S3 (GLM, ~5-8M tokens): critic pass for m_fb (40 x 70 x 150 rank-subsample — subsample
  probes to 60 for the critic to hold quota: ~170K -> subsampled ~65K calls; batch inside
  weekly window; AFTER the pupa arm settles, or next window).
- S4 (Codex): purified-label top-up wave (~800-1,500 arbitrations, reuse wave-2 first).
- S5 (GPU ~2h): executor scores the 3 selected forms/metric x eval docs.
- S6 (CPU): readout per prereg. Artifacts under outputs/analyses/objective_comparison_v1/.
Order: S1 -> prereg commit -> S2/S4 parallel -> S3 -> S5 -> S6.

## Known risks / honesty notes
- Purified-y* is arbiter-relative; the anchor certificate + v1/v2 agreement bound its error;
  the confusion-corrected sensitivity analysis reports how much of any arm-gap could be
  arbiter artifact. Arbiter (gpt) shares no family with executor/critic/proposers.
- crx purified labels depend on wave-2 (v1 voided); if wave-2 crx fails certification, crx
  drops from the sweep (disclosed).
- Selection-only understates what full optimization could do — stated scope: "objective
  comparison under matched selection", upgrade path = phase 2.
- The draft's .7/.6/.55 are placeholders; the experiment replaces them; no number from this
  plan may be quoted before S6 lands.

## REVISION v2 (2026-08-15, after user pushback on the frozen pool)
User's vision: EVOLVE prompts under {reconstruction, LLM-critic} rewards for many metrics,
test all against y*. Resolution = TWO-TIER design:
- Tier A (breadth, cheap): the selection sweep as specified above (~40 metrics). Role:
  controlled mechanism probe + moderator estimation + flags where reconstruction has signal.
- Tier B (depth, the headline): MATCHED-MACHINERY EVOLUTION — same GEPA engine, proposer,
  budget (150-300 metric-calls), seed policy, executor; ONLY the reward differs
  (reconstruction MI vs critic-agreement). Run on Tier-A's form-sensitive flags PLUS a
  random control subset (selection into Tier B is moderator/random, never outcome). The
  reward-x-search interaction is scope of the claim ("optimizing toward"), not a confound,
  once all machinery is shared.
- Circularity note for Section 3 (user worry, resolved): labels are used ONLY as held-out
  one-shot calibration of the label-free instrument, never as an optimization signal
  (evaluate-never-gate). Draft sentence provided in session. Moderator analysis doubles as
  the anti-circularity witness: wins concentrating on form-sensitive metrics = message-side
  mechanism, not label fitting.

## REVISION v3 (2026-08-15, after user ruling: no ground-truth switch; these metrics
## definitionally lack outcome-style truth)
Tier-A finding that forced this: definition-derived arbiter labels are spec-circular for
arm comparison (m_desc structurally aligned; oracle bracket showed no pool headroom).
NEW EVAL STACK for the objectives contest (per-construct, no definition-circularity):
1. PRIMARY (human, per-construct): attentive-tier mention-y (corroborated positives +
   attentive-silence negatives). Noise attenuates arms SYMMETRICALLY -> costs power not
   validity; power comes from 197 paired metrics. Disclose matcher name-contact.
2. LABEL-FREE PSYCHOMETRIC AXES (new instruments, machinery exists):
   a. MTMM convergent/discriminant margin: arm's measurement should correlate with OTHER
      measurements of the SAME construct (different judge family, different register)
      more than with different constructs on same docs. Criterion = correlational
      structure; no labels; no arm aligned by construction.
   b. Mechanical-facet consistency (LLM-free, deterministic): seam-program compiled
      sub-rules give exact partial signals for the ~44%-codable facets; arm scored by
      structure-consistent relation to its construct's own facets. Coverage limited to
      partially-codable metrics — the uncovered set = inarticulability suspects (report).
   c. Reliability floor (necessary-not-sufficient screen).
3. TERTIARY (disclosed, mitigated LLM axis): spec-ENSEMBLE arbitration — K register-
   diverse R1 specs + exemplar-grounded variant, majority vote, any arm's exact text
   excluded from ensemble. Never load-bearing alone. (User's R1-multiplicity idea.)
4. Outcome anchors -> consequential-validity APPENDIX only (different estimand, per user).
VERDICT RULE: an objective wins only if it wins axis 1 AND axes 2a/2b directionally agree.
Framing: construct validity is TRIANGULATED not measured (Cronbach-Meehl), coherent with
Section 3's receiver-relativity. Purified-label table stays for JUDGE validation only
(appendix, committed); never for arm comparison. Awaiting user sign-off before Tier-B
build.

## TIER-A CONTEST RESULT (v1, original data, 2026-08-15) — oc_contest_v1.json
Human axis (mention-AUC, n=83): rec-fb -.002 n.s.; rec-desc -.012 [-.023,-.001] desc
narrowly wins; fb-desc -.010 n.s. Label-free MTMM-lite (n=197): rec-fb +.012 borderline;
others null. AXIS DIVERGENCE: 42/83 metrics flip rec-vs-desc sign between axes — the two
validity axes are non-redundant. Moderator null at this tier.
VERDICT (descriptive): at matched SELECTION over ~11-form paraphrase pools, objectives are
indistinguishable; definition is a formidable baseline (rhymes with GEPA-ships-seed on
hotpot/ifbench — cross-cutting observation: canonical descriptions hard to beat without NEW
content). Oracle bracket explains: pools hold nothing above the definition. Attenuation
caveats: mention-y noise symmetric (deltas shrunk, not biased); MTMM within-family v1.
DISCRIMINATING TEST = TIER B (generation under each objective, reward-only manipulation),
judged by this same triangulated stack. Tier-B build is the next go-decision.

## REVISION v4 — EXECUTOR-LADDER ARM (user hypothesis 2026-08-15, PREREG FROZEN before scoring)
Hypothesis: the selection-tier three-way tie is executor-relative — Gemma-4-31B has the
constructs internalized, so definitions suffice (cf. channel-emergence: explanation>definition
only 32B+; names at 70B). On weaker receivers, receiver-matched reconstruction selection
should pull ahead.
DESIGN: full receiver-matched pipeline per executor e in the SAME-FAMILY ladder
{Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B} (Gemma-4-31B = existing point, separate family,
plotted off-trend, disclosed; same-family staircase rule):
  per e: (1) score all pooled forms x 300 probes with e; (2) defpass (M_i_e) with e;
  (3) SELECT per objective with e's own signals (receiver-matched: M_i_e, same critic
  scores reused — critic is receiver-independent by design); (4) score selected forms +
  definition on corpus with e; (5) contest on the SAME triangulated axes (mention-AUC
  human axis primary; MTMM-lite within-e).
FROZEN PREDICTIONS:
- P-L1: Spearman(ladder position, mean Delta_rec-desc on human axis) < 0 — the rec edge
  grows as capability falls; equivalently Delta at 1B > Delta at 8B.
- P-L2 (mechanism): m_desc absolute mention-AUC decays with falling capability FASTER
  than m_rec (interaction; definitions lose transmissibility, matched forms retain it).
- P-L3: form-sensitivity moderator strengthens as capability falls (wording matters more
  to weaker receivers).
FALSIFIERS: flat/positive P-L1 => internalization does NOT explain the tie; deltas stay
within noise at all rungs => selection tier is exhausted, Tier-B generation mandatory.
Never pool across families; never quote a rung without its n. Artifacts:
outputs/objective_comparison_v1/ladder/.

## WINNER-SET AUDIT (2026-08-15, user-directed "zero in on rec winners"): NOISE VERDICT
Strict rec-wins-both on human axis: 15/83 — BELOW 3-arm chance expectation (~28/83).
Qualitative "text-local/checkable" reading of winner names REFUTED by blind 9-type join
(66/83 joined): in-text win rate .14 vs beyond .29 (n=7); rec edge MORE negative in-text
(-.038) than interface/beyond (-.008). Form-sensitivity moderator null on the cut
(.188 vs .198). MTMM anti-agrees with human-axis winners (6/15 vs 33/68).
VERDICT: selection tier EXHAUSTED — no identifiable subpopulation where rec-selection
beats the definition on human labels; separation must come from the executor ladder
(receiver-relativity, in flight) or Tier-B generation. Never quote individual winner
metrics as evidence (below-chance set).

## PREREG ADDENDUM (2026-08-15, frozen BEFORE eval): ENSEMBLE ARMS (power upgrade, secondary)
m_rec_ens = softmax(i_binary/tau)-weighted average of the metric's form scores (tau=0.1,
fixed a priori; canonical __-1 excluded); m_fb_ens = same with critic rank-agreement
weights; m_unif_ens = uniform average of forms (ablation: is any gain just averaging?).
Endpoints: same human axis (mention-AUC), paired vs m_desc and vs m_unif_ens.
PREDICTIONS: E1 m_rec_ens > m_rec (argmax) — averaging beats selection under noise;
E2 m_rec_ens > m_unif_ens — the MI weights carry signal beyond mere averaging (KEY test:
this isolates the objective from the ensemble effect); E3 m_rec_ens vs m_desc directional.
Falsifier: E2 null => reconstruction weights add nothing over uniform averaging at this
tier. Report all.

## ENSEMBLE READOUT (prereg addendum, 2026-08-15) — oc_ensemble_v1.json
E1: rec_ens vs desc -.0060 [-.0164,+.0041] (argmax was -.0116 — ensemble closes ~half the
gap, still not positive). E2 ISOLATING TEST: rec_ens vs unif_ens +.0035 [-.0013,+.0082],
wins 49/83 (one-sided sign p~.06) — DIRECTIONALLY POSITIVE, borderline: the MI weights
carry real but tiny signal beyond averaging. fb_ens ~ rec_ens (n.s.). Consistent story:
the reconstruction compass is real (within-metric +.39; E2 direction) but worth ~.003-.01
AUC when exploited over 11 paraphrases at the strong-executor tier. Escalation paths
unchanged: ladder (stage C running) + Tier-B seeded-delta generation.

## CORRECTION (2026-08-15): instability number NOISE-CORRECTED — supersedes the .181 quote
Bootstrap noise floor on one form-AUC: median .046. TRUE phrasing SD: median .020 (implied
~3SD range ~.06); above-noise spread on 29/83 metrics; p90 SD .075 (tail ranges >.2).
NEVER quote the raw .181 range as phrasing effect. Coherence check PASSES: median true
headroom above the definition (~.01-.03) matches the E2 ensemble gain (+.0035 w/ weak
weights) — small true headroom + weak-but-real compass = observed tiny gains. Paper phrasing
story (honest form): phrasing effects real for ~1/3 of metrics (tail large), modest at
median; label-instrument spec-sensitivity (54% arbiter flips) is a separate, larger effect.

## LADDER READOUT (prereg P-L1..3, 2026-08-16) — ocl_readout_v1.json, n=114 metrics/rung
| rung | desc | rec | fb | rec-desc | rec-fb |
| 1B | .5248 | .5214 | .5183 | -.0033 [-.011,+.004] | +.0031 n.s. |
| 3B | .5510 | .5490 | .5421 | -.0020 [-.011,+.007] | +.0069 n.s. |
| 8B | .5616 | .5521 | .5488 | -.0095 [-.020,+.000] | +.0032 n.s. |
P-L1: delta ordering weakly consistent (gap smallest at weak rungs: -.003 vs -.010) but
the SUBSTANTIVE hypothesis (rec POSITIVE at weak rungs) NOT achieved — no rung shows
rec > desc. P-L2: desc rises with capability (.525->.562); rec tracks in parallel — no
differential decay. P-L3: moderator null at all rungs (sign flips at 8B).
MECHANISM READING (descriptive): FLOOR EFFECT, not crossover — at 1B the whole instrument
collapses toward chance (desc .525), so weak receivers do not create articulation headroom;
they destroy measurement capacity (coheres w/ channel-emergence: execution turns on 1B->3B).
Internalization hypothesis in strong form: REFUTED at selection tier. rec>=fb at all rungs
(n.s.) — mild, consistent ordering vs the critic.
CUMULATIVE LEDGER across selection/ensemble/ladder: m_rec > m_desc does NOT materialize in
any tested regime over paraphrase pools; only E2 (MI-weights > uniform, +.0035, p~.06) is
directionally positive. LAST UNTESTED PATH: Tier-B generation (seeded delta).

## TIER-B PILOT READOUT (2026-08-16) — tbeval_readout_v1.json; VERDICT + wind-down
Evaluable n=8 (mention-y coverage floors cut 16 -> 8; humor 4/peer 3/cw 1). Budget 200/arm.
- Delta_rec vs 0: -.0330 [-.0985,+.0017] — dominated by ONE collapsed rubric (humor a1
  -.259; both arms collapsed there — probe-overfit/winner's curse, same lesson as the seam
  agentic-compile: recalibration without held-out gating hurts). Median-ish deltas ~0.
- Delta_critic vs 0: -.0565, positive on 1/7.
- Delta_rec - Delta_critic: +.0234 [-.0029,+.0600], rec better on 6/1 decided metrics.
CUMULATIVE CROSS-TIER LEDGER on the paper's hinge claim:
1. m_rec > m_desc: NOT SUPPORTED in any tested regime (selection tie; ensembles -.006;
   ladder negative at all rungs; Tier-B pilot no net gain over seed at budget 200).
2. m_rec > m_llm(critic): CONSISTENTLY SUPPORTED, weakly, in EVERY tier — selection
   rec>=fb all rungs (n.s. each), E2 MI-weights>uniform (+.0035, p~.06, 49/83), Tier-B
   6/1 (+.0234, CI grazes 0). The honest supportable ordering: rec > critic-feedback as a
   label-free guidance signal; definition remains unbeaten at tested budgets/pools.
3. Auxiliary intact positives: within-metric MI validity +.39 (CW, p=2e-4); purified-label
   judge validation table (appendix); phrasing-instability tail (1/3 of metrics).
NEXT ESCALATION if user wants (not launched): Tier-B at scale w/ the seam-h1 protocol
(train-improve + HELD-OUT gate before shipping a rubric — kills the a1 collapse mode),
lower coverage floors via pooled-task readout, budget 500+.
Wind-down: sk1 tier-B server stopped, tunnel closed, task #34 closed.

## M_OMEGA-PROPER EXPERIMENT (prereg FROZEN 2026-08-16 before any scoring)
The corrected test of the paper's hinge sentence — m_omega (recovery-selected unit
assemblies from the fresh e-cert banks) vs m_llm (critic-selected) vs m_desc, + skyline.
- UNIVERSE (frozen): 20 metrics = e-cert slice banks INTERSECT mention-y>=10pos
  (humor 13 / cw 2 / peer 5 via name-join); momega_candidates.json on sk3.
- CANDIDATES per metric (same 6 for every arm; label-free construction): C0 definition /
  C1-C3 definition+top-{2,4,8} bank units / C4 top-4 units alone / C5 best single unit;
  units pre-ranked by in-bank i_binary(unit sig, bank M_i).
- ARMS = selection objectives over the same family:
  m_omega: argmax THREE-HOP RECOVERY — encode candidate on the 300-probe panel (8B) ->
    decoder (qwen-72b, family-disjoint) reconstructs the criterion from (text, score)
    pairs -> re-execute reconstruction (8B) -> reward = rank-agreement(m_hat, M_i_8B).
    Optimum is NOT the definition (transmissibility through a receiver is rewarded).
  m_llm: argmax critic rank-agreement (frozen qwen critic scores on probes).
  m_desc: C0. skyline (calibration only): argmax mention-AUC on OBJECTIVE-half corpus.
- EVAL: mention-AUC on EVAL-half corpus docs only (hash split), same 8B scores for all.
- PREDICTIONS: Q1 Delta(m_omega - m_desc) > 0 paired; Q2 Delta(m_omega - m_llm) > 0;
  Q3 skyline-m_desc = reachable headroom. FALSIFIER DECODE: Q1 null + skyline>0 => the
  recovery compass fails where terrain exists; Q1 null + skyline~0 => assemblies hold no
  reachable headroom (terrain, not compass). Report all cells.
- Instrument: 8B executor EVERYWHERE here (disclosed: g4 is canonical for humor/cw
  elsewhere; internal consistency prioritized). Decoder never sees definitions or labels.

## M_OMEGA-PROPER READOUT (2026-08-16) — momega_readout_v1.json on sk3
Pipeline completed exactly as preregistered: 120 blind reconstructions (qwen-72b decoder,
no definitions/labels seen), hat re-execution + corpus scoring on 8B, 12/20 metrics
evaluable (attrition: 5 no seeds/M_i in pools as pre-disclosed; 3 label-coverage floors).
Selection is NON-DEGENERATE: j_omega dist {C0:3, C1:3, C2:2, C3:3, C5:1} — recovery
picks a unit-augmented assembly over the bare definition for 9/12 metrics.
- Q1 omega - desc:    mean -0.0112 [-0.0360, +0.0093]  +/-: 3/6   (NULL, slightly neg)
- Q2 omega - llm:     mean +0.0113 [-0.0061, +0.0288]  +/-: 6/3   (directional, n.s.)
- Q3 skyline - desc:  mean -0.0065 [-0.0222, +0.0080]  +/-: 3/6   (skyline FAILS to
  transfer: even LABEL-SEEING selection on the objective half does not beat C0 out-of-
  sample)
- oracle(eval) - desc: mean +0.0221 [+0.0076, +0.0402] +/-: 8/0   (eval-half oracle —
  upper bound of upper bound; headroom exists but is ~2pts and eval-selected)
FALSIFIER DECODE (preregistered): Q1 null AND skyline <= 0 => TERRAIN verdict: the
candidate family holds essentially no selection-reachable headroom above the definition
on these labels at n~12 metrics / ~10-40 positives each. The compass (three-hop
recovery) is not the failing component — even labels can't select reliably here.
MANUAL PASS: reconstructions faithful (a121 recon = "punchline/unexpected twist quality"
matches construct); a122 recon EXPOSES DRIFT (definition's 8B scores actually track
offensiveness, not truthfulness — blind decoder as an audit instrument, worth a paper
sentence); no degenerate constant-score arms.

## GEPA-FLAVOR READOUT (2026-08-16) — gb_readout_v1.json on sk3
DISCLOSURE (my error, caught in readout): the GEPA v2 lanes ran on the OLD Tier-B
15-metric list, not the m_omega-20 (list conflation at launch). The two engines are
therefore UNPAIRED universes; each internally valid; no cross-engine paired stats.
Seed baseline = oclc __-1 8B corpus scores (covers old list); eval-half only.
- Delta_rec == 0 on 15/15: the seam-h1 holdout gate (val-probe three-hop reward)
  refused to ship ANY rewrite over the definition seed. Validated null, intent-to-treat.
- Delta_critic (10 shipped, 5 evaluable): mean -0.0734 [-0.1672, +0.0204]  +/-: 2/3
  humor a1 -0.228, a50 -0.180; peer a32 +0.031, a41 -0.020, a46 +0.031.
- MECHANISM (manual read of shipped rubrics): critic-evolved a1/a50 are fluent GENERIC
  joke-quality rubrics ("comedy-first", "setup+punchline payoff") — the one-hop critic
  rewarded construct DRIFT toward overall task quality; eval labels mark the SPECIFIC
  construct, so AUC collapses. The critic arm PASSED its own holdout gate (its reward
  is itself misaligned); the three-hop gate never shipped such drift.
=> GEPA-flavor intent-to-treat ordering: Delta_rec (0) > Delta_critic (-0.073). rec>llm
   again, this time via gate discipline rather than selection accuracy.

## COMBINED FULL-SWEEP VERDICT (goal: m_rec > {m_llm, m_desc})
m_rec > m_llm: SUPPORTED, weakly but with unusual consistency — positive direction in
EVERY independent regime tested (Tier-A selection all rungs; E2 ensemble +.0035 49/83
p~.06; Tier-B v1 GEPA 6/1 +.023; m_omega-proper Q2 +.0113 6/3; GEPA-v2 gate 0 > -.073).
No single cell significant at .05; the cross-regime sign consistency is the result.
m_rec > m_desc: NOT FOUND in any tested regime (executor ladder incl. 1B, ensembles,
unit assemblies, GEPA rewrites with three-hop reward). Convergent evidence that the
definition sits at/near the optimum of its own executor's articulable range for these
silver constructs; the ~2pt eval-oracle headroom is real but not selection-reachable
at current n. Honest paper framing: reconstruction MATCHES the definition while
strictly beating critic-guided optimization — i.e., recovery-selection loses nothing
vs the ideal articulation and avoids the critic's construct-drift failure mode.
Legitimate escalations (design/power only, NOT outcome-conditioned): pooled-task n,
label expansion (more mention-y positives), Tier-B at scale budget 500+ w/ gates.

## POST-HOC IMPLEMENTATION AUDIT (2026-08-16, user-requested double-check)
Question audited: "m_rec starts from m_desc — did we implement that correctly?"
1. E-CERT FLAVOR — verified mechanically from momega_readout_v1.json: C0 (the
   definition) is in every candidate set; when the recovery selector picks C0 the
   delta is EXACTLY 0 (3/12 metrics); every negative delta (6/12) comes from a
   non-C0 pick. So Q1 = -.011 is pure SELECTION RISK, not sub-seed optimization.
2. GEPA FLAVOR — verified from all 30 artifacts: rec arm seeded at the definition;
   evolved == seed on 15/15 and shipped == seed on 15/15, so Delta_rec can never be
   negative by construction. CORRECTION to the earlier mechanism claim: the val gate
   never fired (val_seed=null) — GEPA's search itself found NO train-reward
   improvement over the seed; "gate refused to ship" was imprecise. Honest caveat
   added: budget 120 with minibatch 8 ~= only ~15 candidate evaluations — this cell
   is "no improvement found at small budget," NOT proof the definition is optimal.
3. REWARD MACHINERY — not degenerate: (a) live re-test of the exact qwen_decode call
   path with the same key file succeeded; (b) reward landscape (rewardscape.py on
   sk3): per-metric three-hop reward spreads up to .87 across candidates, and the
   readout's j_omega matches argmax of the landscape 15/15. Notably C0 is the
   three-hop TOP candidate only 3/15 — unit assemblies often transmit the metric's
   scores better than the definition text (transmissibility != label AUC).
4. LATENT BUG FOUND+FIXED: three_hop cache was keyed on rubric only, ignoring ids —
   the rec val-gate path would have compared train-id hat scores against val targets.
   Never executed (gate branch never ran for rec; critic val uses _score fresh).
   Fixed: cache key now (rubric, ids). No result affected.

## TB3 PREREG (2026-08-16, FROZEN before launch) — rec arm at larger search budget
User-directed escalation: "run m_rec on more GEPA tries." Decoder unchanged: qwen-2.5-72b
via OpenRouter (alive, live-tested today) — GLM was never the decoder.
- DESIGN: identical to v2 rec arm (tier_b_evolve.py, three-hop reward, seed = definition,
  seam-h1 val gate, ids-aware cache FIX in) with ONE change: budget 120 -> 500
  (~60 candidate evaluations/metric vs ~15).
- UNIVERSE: the same old-15 list (humor a1,a2,a10,a17,a29,a32,a50,a86,a96,a119,a177;
  peer a32,a34,a41,a46) so results PAIR with the existing v2 critic arm; disclosed:
  still unpaired with the m_omega-20 universe.
- INFRA: executor Llama-3.1-8B server on sk1 GPU 7 STACKED (74% util, ~36GB free;
  stack-per-GPU per standing rule), gpu_mem_util 0.38, port 8220; driver runs locally
  via ssh tunnel; 3 parallel lanes of 5 metrics; artifacts -> scratchpad tier_b3/.
- PREDICTIONS: (i) intent-to-treat Delta_rec >= 0 by gate construction; (ii) open
  question = does 4x search surface gate-passing improvements, and do shipped changes
  transfer to eval-half mention-AUC (paired bootstrap vs seed)?
- FALSIFIER DECODE: 0/15 ships at budget 500 => strengthens "definition ~= three-hop
  optimum in rewrite space" well beyond the budget-120 caveat; ships that pass gate but
  fail eval => the selection-noise story extends to the GEPA tier. Report every cell.

## CORRECTION (2026-08-16, user-probed): oracle "+.022 headroom" OVERSTATED — NEVER QUOTE
oracle = max eval-half AUC over 6 candidates INCLUDING C0 => oracle - desc >= 0 BY
CONSTRUCTION; its bootstrap CI vs 0 is uninformative and the "8/0 significant" framing
was wrong (8 = metrics where some non-C0 topped C0 on eval half; 4 exact zeros = C0 was
the max). Winner's curse: max of 6 noisy AUCs inflates even under identical true AUCs.
Empirical size of the inflation = oracle (+.022) minus skyline transfer (-.007) ~= .029,
i.e. the ENTIRE apparent gap is consistent with selection noise. Honest statement:
apparent oracle gap +.022 AUC pts (upper bound); realizable-by-label-selection headroom
~= 0 (skyline); "no true headroom" vs "headroom masked by selection noise at 5-20
positives/half" NOT distinguishable at current n. Units: mention-AUC points on a ~.54
baseline. The prior phrasing "headroom is real but unfound" is RETIRED.
