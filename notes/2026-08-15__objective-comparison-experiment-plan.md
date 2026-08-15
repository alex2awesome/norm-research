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
