# metric_implementer — metric validity, improvement, and articulability scaling

**Scope: every metric in the project.** Explicit online-rubric metrics, code metrics,
autometrics discoveries, metric_tree proposals, metrics_tree_infilling features,
articulation_star rationales — all of them are *measurement instruments*, and this method is
the instrument lab: it measures their validity, improves their prompts, and maps the
resource frontier of what can be articulated at all. It was abstracted out of
`methods/metrics_tree_infilling/2026-06-10__next-steps-plan.md` §C2 because none of it is
specific to the tree.

(Naming note: this folder name was previously the `score(text)` runner, since renamed to
`methods/existing_metrics_runner/`. This is a new method.)

## The one invariant

Every measure here **evaluates** metrics; none **gates** them.

| Column | Measures | Use |
|---|---|---|
| **Construct fidelity** | reliability, counterfactual validity, reconstruction, consistency | *optimizable* — instrument calibration |
| **Predictive contribution** | gap closure, coverage, deviance drop on held-out | *evaluation only* — the scientific finding |
| **Fidelity failures** | e.g. reliable+predictive but reconstruction-failing | *data* — instrument-level tacitness is part of the phenomenon |

Every Goodhart and circularity risk we identified traces to letting something cross from
column 2 or 3 into column 1. The optimizer never sees predictive performance; reconstruction
failure never removes a metric — it classifies it.

## The scorecard (one per metric, recomputed per prompt revision)

1. **Reliability** ρ — test-retest agreement, k judge passes; optionally per-region.
2. **Counterfactual validity** — accuracy on *decorrelated* planted minimal pairs: vary the
   target attribute T holding each named confound S fixed, and vice versa.
3. **Reconstruction validity** — primary: a reconstructor sees only `(text, metric-score)`
   demonstrations and selects the generating metric from a frozen, contrastively designed MCQ.
   Report per-metric target-option probability, control lift, and panel-level identity
   `I(J;Jhat)`. Re-execution agreement is a separate secondary readout. No anchor or silver label
   enters this measurement.
4. **Consistency** — invariance probes: paraphrase stability, item-order/position
   invariance, monotonicity under graded edits, format robustness.
5. **Optional external-validity report** — when naturally occurring commentary or another external
   measure happens to exist, report agreement after the prompt is frozen. It is never required by,
   optimized by, or consumed by the Reconstruction-MCQ objective or its certificate.
6. **Predictive contribution** *(evaluation only — never optimized)*.
7. **Inter-implementation agreement** — K independent implementations of the *same* metric
   (different model families, code and prompt variants written blind to each other), scored
   on a shared corpus; report mean pairwise κ/Spearman. Low agreement = the metric text is
   underspecified — a thick rule in thin-rule clothing. This is a per-metric *thickness
   measurement* (on-thesis, label-free), not just QA; report it as a result alongside the
   gaps. *Hint*: `impls = [codegen(metric, family=f) for f in FAMILIES]` +
   `itertools.combinations` → `scipy.stats.spearmanr`; persist the disagreement set —
   it is the improvement queue for the optimizer loop.
8. **Code↔judge convergence (V-claimed metrics only)** — for any metric claimed verifiable,
   corr(code score, LLM-judge score) on the same corpus. Low convergence means either the
   code is buggy or the metric isn't actually verifiable; adjudicating a sample of the
   disagreement cells tells you which, with no outcome labels touched. *Hint*: reuse the
   cells DB scores as the judge side; `df.groupby(metric_id).apply(lambda g:
   spearmanr(g.code_score, g.judge_score))`; route `|code−judge|`-top-k cells to a strong
   model for adjudication.

Validity certificates are **relative, not absolute** (underdetermination): the composed
loop — reconstruct (names suspect simpler readings) → counterfactuals decorrelating T from
each suspect → cross-model-family judge/generator/reconstructor — certifies "no articulable
simpler reading survives decorrelation," which is the strongest statement available without
label ground truth.

## Bank-level scorecard (per task, recomputed per bank version)

The per-metric scorecard defends each instrument; the bank-level scorecard defends the
*claim* — "rubrics fall short" is only persuasive against a bank that was demonstrably
given its best shot. Report per task: raw scraped count → deduplicated count (clustering
pipeline), source-diversity (n distinct expert sources per surviving cluster),
per-metric applicability rate (the judge_0p5 problem is a bank property, not just noise),
and **generic-vs-task-specific fraction** (the `project_metric_specificity` problem exists
in the scraped bank too — Halmos applies to all prose; Mathlib naming grammar only to Lean).
*Hint*: generic/specific is itself a cheap LLM tag per cluster ("could this criterion apply
verbatim to ≥3 of our 9 tasks?"); applicability rate comes free from the cells DB.

## Gap arithmetic: attenuation and conservativeness direction

The headline quantities (C−A, C−B, B−A) are differences of AUCs each measured with
instrument noise, and noise is asymmetric: a noisy code implementation biases A down and
*inflates* C−A; a weak judge biases B down and *inflates* C−B. Two rules: (1) every gap
number carries the relevant reliability ρ from the scorecard, with Spearman attenuation
correction (`r_true ≈ r_obs/√(ρ_1·ρ_2)`) or at minimum a bootstrap CI resampled over both
examples and judge seeds; (2) state the conservativeness direction explicitly — measured A
and B are lower bounds, so "C−A is at most X" is safe while "C−B is at least X" needs the
scaling defense below.

The defensible form of the headline claim is a **task-level judge-scaling curve**: C−B at
several judge strengths (8B → 70B → 122B → frontier-API subsample). A gap that asymptotes
as judge capability grows cannot be dismissed as operationalization weakness; a gap still
shrinking linearly is one. This is the task-level aggregate of the per-metric capability
axis already in the scaling section — same runs, second readout.

*Caveat (2026-06-11, per Relative Scaling Laws, arXiv 2510.24626):* scaling is not a
universal equalizer — per-distribution gaps converge, persist, or diverge heterogeneously
under the same scaling. So present per-tier curves descriptively and do NOT extrapolate an
asymptote to "no stronger judge would close this." The load-bearing bounds are the
scaling-free sandwich: dense model = *achieved* lower bound on the attainable; twin/1-NN
ceiling = upper bound with no scaling assumption.

## The improvement loop (GEPA-style, fidelity-only objective)

Candidate rubric prompts are mutated by a reviser LLM that reads the scorecard plus the
*textual* failure artifacts (the reconstructor's induced rule when it mismatches; missed
counterfactual pairs, which double as injectable few-shots with known answers; retest
flips; paraphrase instabilities). Counterfactual sets are **regenerated every round** so the
optimizer can't distill one generator's interpretation into the judge. Acceptance happens
on a frozen holdout with fresh counterfactuals and a different reconstructor family.
Predictive contribution is *reported* before/after — a fidelity-up/prediction-down outcome
is a finding (the old prompt predicted via a confound), not a regression to block.

Full interfaces, loop pseudocode, and cost model: `2026-06-10__design.md` §2.

## Articulability scaling laws

We cannot prove a metric *can't* be articulated. We **can** measure the space of metrics
articulable as a function of resources, by running the same optimizer under budget caps and
tracing the frontier. Headline readout: a **survival curve** — fraction of metrics reaching
fidelity τ as budget grows, with never-reaching metrics *right-censored* (honest: "not yet
articulated at budget B," never "tacit, period"). A metric whose frontier is flat across
*all* axes while sitting below the dense ceiling is the strongest empirical evidence of
tacitness we can construct — and the claim is falsifiable by extending any axis.

Axes: instruction tokens; few-shot count; labeled-data budget (evidence for the optimizer /
reconstructor); model capability (numerically anchored); inference-time compute (ensemble
k, thinking budget); optimizer rounds; **structural complexity of the articulation** (rubric
clause count / program size — the thin-vs-thick-rules axis). Protocol, pitfalls from the
earlier noisy attempt, and analysis plan: `2026-06-10__design.md` §3.

## Files

| File | Status | Role |
|---|---|---|
| `2026-06-10__design.md` | written | concrete design: scorecard, optimizer, scaling protocol, record-keeping (§4), trial (§5) |
| `config.py` | implemented | `ImplementerConfig` + `BudgetCaps` (the scaling axes as enforceable caps) |
| `backends.py` | implemented | role-based OpenRouter clients, cost accounting, validate-and-resample retry |
| `artifact.py` | implemented | `MetricArtifact` (prompt OR code kind) + auto complexity measurement |
| `registry.py` | implemented | **record-keeping**: immutable versions, scorecards, HEAD, append-only ledger |
| `judges.py` | implemented | `PromptJudge` (LLM) + `CodeJudge` (subprocess sandbox) |
| `measures.py` | implemented | reliability, counterfactual, reconstruction, consistency, code↔judge convergence, fidelity scalar |
| `optimizer.py` | implemented | budgeted GEPA loop, both kinds, cross-family acceptance, registry-integrated |
| `run_trial.py` | implemented | `smoke` / `scorecard` / `improve` / `scaling` on competitive code |
| `trial/` | implemented | 360-solution LC/CC/AC pool + 3 trial metrics (prompt + code seeds each) |
| `tests/test_offline.py` | 6/6 passing | fake backends, zero spend: registry, caps, sandbox, scorecard, optimizer round |
| `scaling.py` (survival fits) | planned | KM curves + per-metric saturating fits once grid data exists |

## Reconstruction-MCQ measurement (2026-07-12)

`recon_channel.py` and `experiments/run_r2_recovery.py --mode mcq` implement the primary anchor-free
reconstruction instrument. The default `--distractor contrastive` path uses executor behavior only on the
design split to reject indistinguishable options and exactly select examples that separate every retained
distractor from the target. The reconstructor itself receives only target `(text, score)` demonstrations and
the option descriptions.

The direct per-metric result is normalized target-option probability when choice logits are available, or
counterbalanced selection accuracy otherwise. `mcq_identity_channel` aggregates randomized target metrics
into `I(J;Jhat)`. No-demonstration and shuffled-label channels quantify option/semantic priors. The selected
canonical body is also replayed on untouched items, but its MI is explicitly secondary and is not how the
MCQ choice is made. These are measurements, not prompt-space upper bounds.

```bash
python -m methods.metric_implementer.experiments.run_r2_recovery \
  --mode mcq --distractor contrastive --n-options 4 \
  --mcq-examples 8 --mcq-min-disagreements 2 --R 20 \
  --task peer-review --bucket specific --n-metrics 12 \
  --target-model meta-llama/Llama-3.1-8B-Instruct \
  --reconstructor-backend zai_anthropic --reconstructor-model glm-4.7
```

Use `--mcq-choice-readout auto` (default): local vLLM reconstructors use normalized digit logits; API
reconstructors without logprobs use stochastic choices. Use a multiple of `--n-options` for exact position
counterbalancing. The per-metric JSONL persists the complete design/control/replay artifact, and the driver
also writes `<task>_<bucket>_mcq_identity_channel.json`.
This fallback applies to the comparative `run_r2_recovery` measurement only. Bound-grade CR-3 v12 requires
`--mcq-choice-readout logits` and a backend whose recorded constrained-choice protocol ID matches the run.

## Executor-indexed prompt certificates and CR-3 (2026-07-12)

Two objectives/scopes are reported and must not be conflated:

- The **auxiliary fixed-target behavioral objective** has an `all_finite_prompt_dpi_certificate` over
  **all finite prompts** `Sigma*`, with no length budget. It reports the target-indexed DPI cap `T_b`.
  For a one-form hard target defined by the same executor/readout, the canonical rubric is an exact
  self-reproduction witness. This is not the primary Reconstruction-MCQ optimum.
- CR-3 covers the bootstrapped initial pool union one declared frozen proposer mixture. It provides (a)
  a finite-sample upper bound on expected best reconstruction value after a fixed future mining budget and (b), only
  under an external exact-pattern mass floor `p_min`, an all-support exhaustion certificate for that
  proposer. It cannot lower the all-`Sigma*` DPI bound.

CR-2, ALPHA-PROBE, VALUE-CENSUS, and the legacy `cr3_certificate` split wrapper are descriptive only.
See `notes/2026-07-11__cr3-mining-loop-spec.md` and theory note §12.6b.

Ownership: `experiments/run_cr3_mining_loop.py` is the only end-to-end entrypoint;
`scripts/tools/cr3_mining_worker.py` is its GPU child process; and `experiments/cr_audit.py` is the CPU
certificate library. These add a bound-grade path. They do not delete CR-2/GEPA/value-census code. The DPI
certificate is authoritative only for the auxiliary fixed-target objective. The primary Reconstruction-MCQ
all-string result uses the exact fixed-eight state envelope described below; `1-q_no_demo(target)` remains
its coarser range cap. CR-3 is authoritative only for its declared proposer-process horizon/support scope.

For the primary anchor-free objective, add `--value-mode reconstruction_mcq`. The orchestrator then freezes
task-and-hierarchy-level option codebooks from bootstrap behavior before mining, and the worker's `value` stage applies
`experiments/cr3_reconstruction_values.py` to every pool/audit prompt. The CR mark is annotation-attributable
MCQ target-option lift over no-demo/shuffled controls, bounded by the prompt-independent global cap
`1 - q_no_demo(target)`. For every prior-passing menu, codebook v4 constructs a target-specific ordered
eight-item teaching panel from the complement of the 120-item menu-design split and its exhaustive envelope;
the final menu and associated panel are selected and frozen jointly. Every prompt uses those same selected
items and order.
The loop values all 256 binary transcripts before prompt search and certifies
`V_best <= V*_{b,E} <= U_state <= 1-q_no_demo`. Behavior signatures still define capture species. Candidate
wording is never shown to the reconstructor. The reconstruction noun, `--mcq-max-chars`, model/tokenizer/chat
template, option block, deterministic seeds, and exact rendered queries are part of the frozen evaluator
namespace. A persistent content-addressed choice-probability cache removes cross-process numeric drift, and
repeated exact patterns are required to have identical value; only then may exact-support
exhaustion be promoted to a proposer-support value ceiling. `legacy_fixed_target` remains available only for
the historical `I(M_fixed;E(p,X))` experiment.

The executor backend records normalized YES/NO probabilities, then applies the frozen hard readout `YES`
iff `pYES>0.5` (ties are `NO`). The finite state is the resulting ordered eight-bit transcript. Once its
table is frozen, bootstrap, historical, monitor, checkpoint, and confirmation `a_value` artifacts are
created by validated CPU lookup, not new Qwen calls. The table itself is evaluated directly by Qwen and
lookup-versus-direct replay is covered by tests.

The production certificate tracks the exact fixed teaching-transcript hash as its value-state partition.
The all-prompt `U_state` theorem is unconditional because all 256 states are enumerated. Separately, with a
defended `--value-p-min`, capture-recapture exhaustion of observed proposer draws proves the
proposer-support Reconstruction-MCQ value ceiling even if full 300-bit executor patterns remain diverse.

The all-string MCQ result and CR-3 answer different questions. `U_state` is the exact upper envelope over
the fixed instrument's complete binary transcript population and holds for every finite prompt without a
proposer assumption; `1-q_no_demo(target)` is its coarser range audit. CR-3 estimates unseen behavior/value-state
mass and future gain under the declared proposer mixture, so it explains and predictably tightens the
search-process result but cannot see zero- or arbitrarily-low-probability prompts. Report both scopes.

The v10 production value is MCQ target-option probability lift over the stronger frozen control. The
legacy `recon_channel.py --normalize-options` path is exploratory and always reports
`mcq_bound_grade: false`; it is not used by CR-3. `experiments/reconstruction_certificate.py` bounds a
different, secondary behavioral-replay MI estimand on a fresh lockbox. It is tested and retained for that
separate experiment, but is deliberately not imported to tighten or relabel the primary v10 certificate.

Use `--mcq-codebook-metrics` to separate the metrics being optimized from a broader frozen task-and-level
distractor bank. Each bank-only checkpoint contributes one canonical/orbit-averaged executor signature;
its historical prompts never enter the mining pool. `--reuse-mcq-codebook-root` may hard-link those
lightweight artifacts after full hash, identity, schema, and executor-namespace validation. This is the
production path for hard behaviorally related distractors; a codebook built only from the current target
subset is a different and usually easier reconstruction estimand.

V11 enumerates behaviorally hard panels and runs the exact blind no-demonstration query before any prompt
is valued. The default prior-balance gate requires maximum option probability `<=0.35`, target probability
within `0.10` of chance, and normalized prior entropy `>=0.90`. The fixed-state v12 release then enumerates
the complete T8 envelope for every prior-passing menu, retains state-live menus, and selects the largest
`U_state`; behavioral hardness and panel ID are tie-breakers only. This maximizes measurable dynamic range
and is conservative against easy certification. If no passing panel is live, the maximum-range passing
menu is formal-only; if no panel passes the prior gates, the least-violating menu is formal-only. The
calibration and value stages share the same frozen noun, rendering limit, rendered queries, seeds,
reconstructor revision, and choice cache, and the loop verifies their probability matrices agree exactly.
Each prior-passing panel costs 2,048 batched annotation/shuffled choice queries at the default four draws;
fixed panels do not rerun teaching-set MILPs per state. `--mcq-prior-max-panels-per-target` is the prospective
compute budget. Increasing it preserves executor artifacts and rendered-query cache hits but produces a new
plan/final-codebook namespace.

Historical prompts for the same target can be reused without laundering them into fresh data. First build
one immutable evidence store, then pass it to a new run:

```bash
python -m methods.metric_implementer.experiments.cr3_evidence_store \
  --source-root <old-cr3-root> --source-root <another-old-root> \
  --out-root <immutable-evidence-root>
```

`--reuse-evidence-root` installs validated content-addressed signature/choice caches and admits deduplicated
prompts as candidate-only pool rows after current-namespace rescoring and current-codebook revaluation.
They may raise the achieved lower endpoint. They are inserted before the adaptive ledger and can never be
used as checkpoint or confirmation observations.

Formal optimality is not automatically a headline articulability result. The global payload's
`instrument_quality` reports the full no-demo option prior, entropy, value headroom, selected-distractor
kappas, disagreement counts, exhaustive state capability, and operational target/orbit diagnostic. A
headline requires the blind-prior gates, coarse headroom `>=0.10`, `U_state` above the predeclared value
resolution, and at least one envelope-maximizing transcript with positive lift and a unique target-option
posterior argmax. For one-form targets the operational diagnostic is the canonical-description behavior; for
orbit targets it is the hard readout of the declared orbit average and need not be realizable by one prompt.
It is diagnostic only. The historical universal
kappa `>=0.50` threshold is retained as a descriptive near-clone diagnostic, not a headline gate.

```bash
# CPU certificate, production-path, value-census, and resume tests:
PYTHONPATH=. pytest -q \
  methods/metric_implementer/tests/test_cr_audit.py \
  methods/metric_implementer/tests/test_cr3_evidence_store.py \
  methods/metric_implementer/tests/test_cr3_mining_loop.py \
  methods/metric_implementer/tests/test_cr3_reconstruction_values.py \
  methods/metric_implementer/tests/test_value_census.py

# adaptive mine-until-bound loop (sk3, one GPU at a time):
CUDA_VISIBLE_DEVICES=<free> python methods/metric_implementer/experiments/run_cr3_mining_loop.py \
  --metrics <..._sigs.npz ...> \
  --out-root /lfs/skampere3/0/alexspan/outputs/<new-immutable-v12-fixed-t8-root> \
  --mcq-codebook-metrics <full frozen task-and-level checkpoint bank ...> \
  --reuse-bootstrap-root <verified-prior-cr3-root> \
  --reuse-mcq-codebook-root <optional prior root with the same candidate bank> \
  --reuse-evidence-root <optional immutable historical evidence store> \
  --value-mode reconstruction_mcq \
  --mcq-reconstructor Qwen/Qwen2.5-14B-Instruct \
  --mcq-choice-readout logits --mcq-n-examples 8 --mcq-value-query-batch-size 512 \
  --families microsoft/phi-4 microsoft/phi-4 Qwen/Qwen2.5-14B-Instruct meta-llama/Llama-3.1-8B-Instruct \
  --family-tags phi4_atomic phi4_holistic qwen14_atomic llama8_holistic \
  --family-modes atomic holistic atomic holistic \
  --batch-per-family 150 --confirm-per-family 300 \
  --checkpoint-per-family 300 --checkpoint-iters 0,1,2,4,8 --study-alpha 0.05 \
  --ceiling-horizon-per-family 100 \
  --target-u0 0.10 --target-value-gap 0.02 --max-iter 12 --patience 3
```

The loop first resolves the substantive hierarchy description and bootstraps the target plus all source
prompts through the same ordered probe panel and persistent content-addressed executor cache.
`--reuse-bootstrap-root` may hard-link an existing immutable bootstrap only after revalidating the source
checkpoint hash, resolved metric identity, probe/executor/readout namespace, model snapshot, scoring-code
hashes, and artifact schema. V12 rejects pre-v12 numeric roots rather than migrating them. Historical prompt
text remains reusable through `--reuse-evidence-root`, where it is re-scored in the current namespace.
For the fixed-eight release, validated e601 target bootstraps, canonical codebook-candidate bootstraps,
300-probe prompt signatures, menu rows, and exact no-demo query results remain reusable only when the frozen
reconstruction noun, rendering limit, reconstructor/readout namespace, and rendered-query identities match.
The root-bound panel-plan provenance is regenerated and prior rows are transplanted only after semantic-plan and checksum
validation. Final codebook v4, the 256-state table, and every candidate `values.npz` are new. Never resume or
mutate the e601 root, migrate an old value artifact, or hard-link a writable SQLite cache; use a private
validated snapshot when exact query-cache reuse is needed.
Use aligned `--family-modes` to distinguish `atomic` one-question criteria from `holistic` complete rubrics.
Repeat a model with distinct tags (for example `phi4_atomic` and `phi4_holistic`) when both modes are
required; they remain separate C/R strata. An atomic-family plateau cannot be reported as an unrestricted
prompt plateau when the all-prompt gap remains open.
Every proposal attempt has its own seed and provenance; duplicate texts remain valid recaptures and reuse their
cached behavior. Exact quotas, panel/model/readout hashes, immutable manifests, an ordered absorption
ledger, and a never-absorbed confirmation namespace are enforced. The final payload is
`<root>/<metric>/confirmation/certificate.json`; monitor rows select stopping and must not be quoted as
certificates. In legacy fixed-target mode, read `all_finite_prompt_certificate` for the auxiliary
unrestricted DPI/identity result. In Reconstruction-MCQ mode it reports the anchor-free global interval
`[V_best,U_state]`, the coarser `1-q_no_demo(target)` range cap, and the optimization-gap bound.
Read top-level `certified` for the process-relative CR-3 result. The source pool remains atomic, while known target-form
prompts are evaluated separately in the unrestricted certificate. This prevents an exact identity witness
from erasing the unit-discovery experiment. `assumption_dependent.exact_support` is usable only when its
stated external `p_min` premise is scientifically defended.

### Release contract

Existing v10/v11 roots remain frozen and receive no migrator. V12 adds total constrained behavior and MCQ
choice readouts, fixed-eight codebook v4 state enumeration, dual 95%/90% reporting, and
task-and-hierarchy-level matched MCQ banks. V11 added
prior-balanced menu construction and candidate-only historical evidence reuse, so it always writes a new
immutable output root. Every run
records exact code SHA-256 values and must resume from the same release overlay; even a whitespace edit is
intentionally rejected. New runs may hard-link hash-compatible bootstraps and codebook candidates and may
reuse only content-addressed caches whose release readout IDs match; old prompt texts may be rescored as
candidate-only evidence, but v11 signature/choice rows are never relabeled as v12. The orchestrator accepts
the existing R3 banks for creative writing,
humor, news homepages, press releases, code review, Math StackExchange, grant funding, peer review, and legal
outcome prediction. The nonblocking run lock fails immediately when another process owns a root.

In Reconstruction-MCQ mode every finite prompt executable by the frozen wrapper maps to one of the 256
enumerated transcripts, so `V_ann <= U_state <= 1-q_no_demo(target)`. Here `q_no_demo` is the exact mean
over the frozen complete counterbalancing block, not an estimated Bernoulli rate. `prompt_evolution_status` is issued only from
never-absorbed checkpoint/final audits and separately reports behavior `SATURATED/UNSATURATED/UNRESOLVED`
and value `RISING/PLATEAUED/UNRESOLVED`. These are fixed-executor prompt-evolution statuses, not OSL labels.
`reporting_tiers.primary_95` is the only source of process-evolution `CERTIFIED_*` labels. The same immutable audit is also
recomputed prospectively at 90% under `reporting_tiers.sensitivity_90`; conclusions unique to that tier are
spelled `SUGGESTIVE_*` and never overwrite the 95% status. The exact all-prompt frozen-control cap is not
confidence-tiered.
The same panel-quality gate also demotes process-value `RISING/PLATEAUED` labels to
`FORMAL_CERTIFICATE_ONLY` while leaving independent behavioral-coverage labels intact. A statistically
`UNRESOLVED` value axis stays unresolved; failed panel quality is never counted as resolution. Dry/fake outputs are
marked `SYNTHETIC_TEST_ONLY` and `publication_eligible=false` in certificates, trajectories, and bank
summaries.
The final `mcq_identity_final.json` also reports achieved bank-level `I(J;Jhat)` for the best pool prompt
per metric and its no-demo/shuffled controls; that companion MI is an achieved reconstruction result, not
an upper bound. Its task records state the R1/R2/R3 level and separately mark the all-panel diagnostic and
the headline-eligible subset, so weak panels cannot enter a publishable bank summary silently.

For a non-identity orbit/composite target, select and orient one candidate using discovery/calibration data,
then score it once on an untouched iid item lockbox. Feed those paired hard verdicts to
`all_finite_prompt_population_certificate`; it uses exact binomial components and the binary identity
`A* - R <= H(M_b|Z) <= h(error)` to certify the population optimization gap. Use
`zero_error_lockbox_plan(epsilon_bits)` before collection to size a zero-error design. A lockbox used for
selection, stopping, threshold choice, or prompt revision is consumed and cannot issue this certificate.

## Related

- `notes/2026-06-18__prompt-optimality-theory.md` (+ compiled `notes/prompt-optimality-whitepaper-latex/`)
  — the authoritative target-indexed theory. For a fixed target `M`,
  `R_E(p;M) <= A_E(M;P) <= I(M;X) <= H(M)`. An operational target `M_b` and ideal `M*` have separate
  ladders with no automatic ordering between them. ALPHA-PROBE and VALUE-CENSUS remain descriptive
  experiments under `experiments/`; they do not bound `M*` or the prompt optimum.
- `methods/metrics_tree_infilling/2026-06-10__next-steps-plan.md` §C2 — origin of this
  framework; the tree method becomes a *consumer* of scorecards.
- `project_tacitness_two_layers` / `project_rubric_fidelity_validation` memories — the
  three-AUC layering and the original metric-rediscovery idea (= reconstruction validity).
- Noah framing (2026-05-28): articulability ceiling is operationalization-dependent — the
  scaling frontier is that qualifier made quantitative.
