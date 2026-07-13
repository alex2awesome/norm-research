# CR-3 executor-indexed prompt-ceiling loop (authoritative v12 fixed-state release, 2026-07-13)

## Objective

For one executor/readout protocol `E`, ordered probe panel `X`, initial prompt pool `Omega_N`, and frozen
proposer families `Q_f`, bound the best **single-prompt** value under a declared anchor-free reconstruction
measurement `V`:

```
V*_{b,E} = sup_{p in Dom(E)} V_{b,E}(p).
```

The primary `V` is Reconstruction-MCQ: the candidate prompt supplies its own executor annotations; a frozen
reconstructor sees only one fixed ordered eight-item `(item, annotation)` panel and a frozen option
codebook; normalized target-option probability and annotation-attributable lift are recorded. No anchor,
silver label, human label, or outcome enters. Across randomized target metrics, the companion bank-level
quantity is identity `I(J;Jhat)`.

The v2 fixed-target value `I(M_fixed ; binarize(E(p,X)))` remains supported as a **legacy behavioral
discovery diagnostic**. It is the value used by the currently running `cr3_mining_v2` job and must not be
reported as the final Reconstruction-MCQ optimum.

Every declared value has a predeclared finite cap. For the primary annotation-attributable lift, codebook
v4 freezes `T_8` before prompt search and exactly values all `2^8` binary transcripts. Therefore
`V_best <= V*_{b,E} <= U_state <= 1-q_no_demo(target)` for every finite prompt executable by the frozen
wrapper. The final term is the coarse range cap; `U_state` is the exact finite-state upper envelope. CR-3
can further tighten the achieved-to-ceiling interval in two proposer-relative scopes:

1. a finite future mining budget: an upper confidence bound on the **expected** best prompt after fixed
   counts `m_f` of additional draws from each family;
2. the entire proposer support: only when exact-pattern missing mass is below an externally justified
   minimum support mass `p_min`.

CR-3 itself does not bound arbitrary strings, multi-prompt checklists, or the latent ideal `M*` unless those objects
are separately identified and placed inside the declared prompt/readout class. In particular, capture-
recapture broadens beyond the discovered pool to a frozen proposer process; it does not silently become an
all-strings theorem.

These are complementary bounds. The exact finite-state envelope covers every finite prompt for this fixed
instrument but contains no search-rate information. CR-3 is a discovery/gain bound that can tighten
predictably with additional fresh draws, but only for a fixed proposer horizon or, with an external mass
floor, its support. Full support over strings does not repair this distinction: a valuable unseen prompt can
have arbitrarily small proposer mass.

## Component ownership

- `experiments/cr_audit.py` is the CPU theorem-to-payload library. It computes the exact pool optimum for
  either legacy fixed-target MI or supplied bounded anchor-free reconstruction values, per-family mass
  bounds, all-draw gain bounds, finite-horizon expected-best ceiling, confirmation-only evolution status,
  and optional exact-support consequence. It **supplants `cr_horizon.py` and value/checklist census code only for
  bound claims**; those older modules remain available as historical/descriptive diagnostics.
- `scripts/tools/cr3_mining_worker.py` is a new GPU subprocess worker for this experiment. Its `propose`
  stage creates provenance-complete independent proposal occasions. Its `score` stage bootstraps or audits
  prompts through the persistent executor cache. Its `value` stage holds one frozen reconstructor resident
  and assigns the repaired MCQ/control value to every scored row. It does not replace the general vLLM
  backend, GEPA, or ordinary metric scoring.
- `experiments/cr3_reconstruction_values.py` freezes bootstrap-only codebooks and ordered teaching panels,
  enumerates and validates the complete 256-state population, and serializes hash-bound MCQ value
  transactions/envelope summaries. It supplies the exact cap and prompt values to `cr_audit.py`; it is not
  a second capture-recapture certificate engine.
- `experiments/cr3_evidence_store.py` consolidates historical prompt generations and content-addressed
  executor/MCQ caches. Imported prompts are permanently candidate-only: after current-namespace rescoring
  and current-codebook revaluation they may raise `V_Omega`, but they cannot serve as audit or confirmation
  observations.
- `experiments/run_cr3_mining_loop.py` is the user-facing orchestrator. It owns the immutable manifest,
  bootstrap transaction, monitor/absorption ledger, stopping rule, resume behavior, and isolated
  confirmation audit. It invokes the worker one model per subprocess, then calls `cr_audit.py` on CPU.
- `vllm_backend.py` remains the shared backend. CR-3 adds release-namespaced total behavior and MCQ
  readouts: vLLM masks generation to the declared exact single-token labels before log-softmax, and any
  missing or nonfinite allowed-token evidence fails closed. Legacy callers keep their existing methods.

The v12 end-to-end path is:

```
source checkpoints -> bootstrap/cache -> codebook v4 -> exhaustive 256-state value envelope
                  -> propose -> score behaviors -> measure frozen MCQ value marks
                  -> monitor -> ledger absorb -> ... repeat ...
                  -> never-absorbed checkpoints/final confirmation -> certified trajectory
```

## Frozen run contract

`run_manifest.json` is immutable and hashes the input checkpoints, resolved metric descriptions, model
families, executor, budgets, thresholds, alpha allocation, worker/orchestrator/certificate/backend code,
readout code, and the worker's startup HOME/cache environment. A run refuses a nonempty root without its
matching manifest. Every GPU subprocess receives the writable `--worker-home` before Python imports vLLM,
Triton, or Torch; assigning HOME only inside a backend method is too late for module-level cache constants.

Before mining, every metric gets `bootstrap/scored.npz`:

- resolve the hierarchy's substantive metric description; a name-only fallback is forbidden;
- load and hash the actual ordered probe texts;
- rescore the target and **all** initial-pool prompts with the declared executor/readout;
- preserve orbit averaging when the source target declares multiple forms;
- seed each item readout deterministically from the cache namespace, prompt, and probe index;
- persist a content-addressed signature under
  `(probe hash, executor revision, readout/template hash, max length, prompt text)`.

The target set and MCQ candidate bank are separate immutable inputs. `--metrics` declares prompts whose
values are optimized. `--mcq-codebook-metrics` declares the broader task-and-hierarchy-level bank from which each target's
closest behaviorally distinguishable distractors are frozen. Every bank member receives one lightweight
`mcq_codebook_candidates/<metric>/bootstrap/scored.npz` containing only its canonical/orbit-averaged target
behavior; its historical source prompts are neither rescored nor admitted into target search. A prior
candidate bank may be hard-linked with `--reuse-mcq-codebook-root` only after validating the source
checkpoint hash, metric identity, exact model snapshot, constrained readout/scoring-code namespace, and
artifact schema. Pre-v12 numeric artifacts are rejected, not migrated. Candidate prompts may still cross
versions through the evidence store because they are re-scored and revalued before entering the achieved pool.

Candidate panels are prelocked in two arms before any target-specific menu prior is observed. The primary
arm is the unchanged behavioral-hardness prefix: `--mcq-prior-max-panels-per-target` retains its historical
meaning, ordering, and panel IDs, so all matching cached queries remain reusable. The bounded fallback arm
adds at most `--mcq-centralness-fallback-panels-per-target` menus (default 64) from the 32 behaviorally
eligible metrics nearest the target under a frozen task-bank-wide blind-centralness reference.
That reference presents every bank metric in four deterministic task-wide anchor contexts and scores the
exact seed-7 four-order cyclic block. These are target-indexed centralness calibrations, not an assertion of
target-independent anchor exposure. Their scalar centered-log-probability summary is a design-only menu
search heuristic: it is not a bound, certificate premise, headline prior gate, or external label. A
connected exposure-balanced PL/BT estimate is a possible future efficiency improvement, not part of v12.
Reference plan, probabilities, model/revision/readout/noun, budgets, and hashes are frozen before either
arm's candidate-panel prior/gate outcome. Both
arms are constructed for every target regardless of whether the primary arm would pass; a failed combined
bank is retained as formal-only and is never conditionally expanded within the protocol.

Every prelocked candidate panel is then scored with the exact blind no-demonstration query before
prompt-value search. The four-option headline instrument uses the full
`S_4` block: all `4!=24` option orders exactly once. The first four rows reproduce the historical seed-7
cyclic block and the remaining 20 unique orders are appended deterministically, so existing prefix query
cache keys remain reusable. A four-row calibration artifact itself is not transplanted into the 24-row
functional; the imported private cache supplies the identical prefix and the worker fills only missing rows.
The ordered list and its hash are frozen in every relevant artifact. The default
gate requires maximum mean option probability at most `0.35`, target prior within
`0.10` of chance, and normalized menu-prior entropy at least `0.90`. At most four passing menus are kept by
the frozen order: normalized entropy descending, total variation from uniform ascending, absolute target
prior distance from chance ascending, behavioral hardness descending, then panel ID. If none passes, the
least-violating menu is carried only to diagnose a formal-only instrument.

Each retained menu receives exactly eight prospectively prehashed T8 candidates: the unchanged historical
baseline plus surface-matched, behavior-matched, nuisance-balanced, behavior-pattern-diverse, TF-IDF-diverse,
and deterministic no-good variants. They use only the sorted teaching complement's texts and the frozen
canonical/orbit-averaged option behaviors; candidate prompts and external labels are unavailable. Candidate
zero is byte-identical to the old baseline T8. Every arm enumerates all 256 transcripts under the exact
four-order prefix. This screen is selection-only and can never be quoted as a certificate. It ranks
canonical-live first (positive canonical lift and unique target mean-posterior argmax), then canonical value,
`U_4`, and stable IDs. Exactly two finalists are atomically locked before any full-24 result is read.

Those two finalists alone receive all 24 option orders. Final selection again requires canonical-live, then
maximizes `U_24`, then canonical value and stable IDs. The synthetic envelope-maximizer state is retained as
a descriptive capability diagnostic, not a gate. If neither finalist is canonically live, the exact
fixed-instrument inequality remains valid but the result is `FORMAL_CERTIFICATE_ONLY`.
The 120-item design split and 180-item teaching-candidate split are disjoint and exhaustive over the 300
probes. Every prompt uses those same eight items in the stored order. The final v4 codebook binds their
indices, item IDs, target transcript, split provenance, and instrument hash before any prompt-value search.
The executor cache stores exact normalized YES/NO probabilities; the fixed hard annotation is one iff
`pYES>0.5`, with ties mapped to zero. The 256 states are transcripts of that binary functional, not soft
probability vectors. The reconstruction noun and `--mcq-max-chars` rendering limit are frozen with the final
codebook; together with the model/tokenizer/chat template, option block, deterministic seeds, and exact
rendered queries, they define the evaluator namespace. After direct Qwen evaluation freezes the state table,
every prompt value is a validated CPU lookup by transcript. A direct-replay regression test checks lookup
equality.

The four-order state screens are exact prefixes of the full-24 queries, so a finalist reuses every screened
row from the immutable content cache. The legacy primary budget and the centralness fallback budget are bound
separately in the run and panel manifests; final codebook entries also bind the selected arm/rank and both
centralness hashes. Expanding any prospective budget regenerates selection provenance but reuses identical
executor and rendered-query cache rows. Centralness calibration runs through
the additive `scripts/tools/cr3_reconstruction_calibration_worker.py`; the load-bearing
`cr3_mining_worker.py` remains byte-identical, preserving the numeric executor-bootstrap namespace.

Historical prompts for the **same target metric** are a separate input. `--reuse-evidence-root` installs
validated cache entries and copies a deduplicated candidate manifest into the new root. Those prompts are
rescored/revalued and inserted after bootstrap but before the adaptive absorption ledger. Their role never
changes to audit evidence, even when their source file came from a historical monitor or confirmation.

**Release freeze (updated 2026-07-13).** Existing v10/v11/e601 roots remain immutable and are not migrated.
V11 added prior-balanced panel selection and candidate-only evidence reuse. V12 adds total constrained behavior/choice
readouts, new signature and choice-cache namespaces, level-matched codebook banks, and predeclared dual
95%/90% reporting; the fixed-state v12 release adds codebook v4, exhaustive `T_8` enumeration, and the exact
24-order factorial headline block. Within
each release, exact code and source hashes remain load-bearing, so a changed
release cannot resume an old root. V11 prompt texts may be imported as candidate-only evidence and rescored;
old candidate value artifacts are never promoted into the v4 namespace. Validated e601 target and codebook
bootstraps, scored 300-probe prompt signatures, semantic panel designs, and exact prior query rows are
reusable only under the same frozen noun/rendering and reconstructor/readout namespace; final codebooks,
state tables, and all prompt values are rebuilt. Writable SQLite databases are
never hard-linked across runs.
Supported existing R3 bank prefixes are creative writing, humor, news homepages, press
releases, code review, Math StackExchange, grant funding, peer review, and legal outcome prediction.

**Hierarchy population frame (added 2026-07-13).** The historical
`*_r3_expanded.json` artifact is merge-only: `merged_groups` contains R2 concepts that the
meta-merge joined to at least one other R2 concept, and omits every untouched R2 concept. It is a valid
**multi-merge R3 stratum**, including for the frozen sentinel, but is not a census of the R3 level. Population
comparisons across R1/R2/R3 must use a complete, outcome-independent frame. The complete R3 partition keeps
the historical multi-node merges in their original order and appends each untouched frozen R2 input as a
singleton carry-forward. It records `r3_membership_type` so multi-merge and singleton strata can be reported
separately. `experiments/complete_r3_census.py` constructs the partition, binds both source-file hashes, and
fails unless the result covers every frozen R2 input exactly once. Reconstruction, certificate, and external
validation outcomes are not inputs to this construction. Per-metric prompt certificates are unaffected by
the sampling frame; claims about an R-level distribution or pass rate are not.

The legacy-to-bootstrap agreement is diagnostic only. Validity comes from defining the v2 target, pool,
and every later audit signature in the same immutable namespace. Duplicate prompt strings, including
recaptures in later worker processes, reuse the exact cached signature.

## Proposal sampling

Each proposer family is one frozen tuple
`(model revision, proposal mode, metric-specific prompt, temperature, validator, max tokens)`. `atomic`
emits one short yes/no criterion; `holistic` emits a complete rubric and permits multi-sentence prompts up
to the declared executor-safe limit. Model and mode combinations receive distinct family tags, so their
capture and gain distributions are bounded separately. Every generation attempt
uses a distinct stable seed. Deterministic rejection sampling keeps the first valid outputs until the exact
family quota is filled; duplicates are retained because de-duplication would change the sampling law.
Accepted rows record model/revision, seed, attempt and accepted indices, prompt/config hashes, temperature,
and validator. Missing families, short quotas, repeated seeds, nonfinite scores, changed panels/revisions,
or inconsistent cached recaptures fail closed.

Conditional on the operational seeded-generator model, accepted draws are iid within family from the
frozen generator distribution conditional on validity. Claims never assert independence across families;
each family receives its own confidence component.

## Certificate

Let `V_Omega` be the exact best single-prompt value in the bootstrapped/absorbed pool and
`G(p)=max(0,V(p)-V_Omega)`, bounded by `B=U_state-V_Omega` in fixed-state MCQ mode (or the declared
finite cap in legacy mode). Every audit draw receives a value mark,
whether or not its behavior is novel. Therefore the finite-horizon value theorem does not assume fuzzy-
species or exact-pattern substitutability.

Total `alpha` is first split between the primary upper-bound bundle and the independent
two-sided status bundle. The upper-bound half is then split across every declared claim and across
families (four in Reconstruction-MCQ mode; three when no exact value-state partition is supplied):

1. fuzzy leader-classifier missing mass: per-family one-sided Clopper-Pearson, useful only as a mining
   diagnostic and stopping monitor;
2. exact-pattern missing mass: per-family one-sided Clopper-Pearson on the genuine partition induced by
   exact binarized executor vectors;
3. exact reconstruction-value-state missing mass: the genuine partition induced by the frozen teaching
   transcript, when the content-addressed logit path is active;
4. prompt gain: a simultaneous empirical-Bernstein mean bound and a simultaneous DKW CDF envelope.

For future counts `m_f`, two valid expected-best-gain bounds are computed:

```
U_sum = sum_f m_f U_mean,f
U_DKW = integral_0^B [1 - product_f L_f(t)^m_f] dt
U_horizon = min(B, U_sum, U_DKW).
```

The reported finite-horizon expected prompt ceiling is `V_Omega + U_horizon`. This all-draw gain mark is
the bound-grade form of value weighting: behaviorally novel zero-gain prompts contribute zero, while a
rare high-gain prompt contributes its full bounded gain. Historical `alpha_V` remains descriptive and is
not transformed into a ceiling. No species value,
substitutability, submodularity, Good-Toulmin extrapolation, or fitted asymptote enters this result.

If every exact pattern in the proposer-mixture support is externally assumed to have mass at least
`p_min`, then `U_exact < p_min` certifies **behavior-pattern** support exhaustion. It certifies the value
ceiling equals `V_Omega` only when value is proven to be a function of the exact behavior pattern. That is
true for legacy fixed-target behavioral MI. It is also true for the repaired deterministic-logit
Reconstruction-MCQ path: candidate text is hidden, teaching selection/order depends only on hard behavior,
the no-demo channel is frozen at bootstrap, and rendered choice-probability queries are content-addressed so
cross-process floating-point drift cannot change a transcript's value. Repeated exact patterns are checked
for identical value. Sampled/nondeterministic fallbacks do not get
this promotion automatically. In all modes the finite-horizon all-draw gain bound remains valid. `p_min`
is an identifying assumption, not a number estimated from the capture stream.

The value-state partition is generally coarser than the 300-bit executor-pattern partition. Under its own
external `--value-p-min`, `U_value < value_p_min` certifies that every proposer-support MCQ value is already
represented and therefore the proposer-support value ceiling equals `V_Omega`. The code fails closed if one
teaching-transcript hash ever has two values; fuzzy similarity is never used for this promotion.

### Confirmation-only status lattice

At every immutable confirmation, the v5 certificate constructs one simultaneous two-sided evidence bundle
using the other half of total `alpha`:

- behavioral missing mass interval `[L_mass,U_mass]`;
- finite-horizon expected-best-gain interval `[L_gain,U_gain]`; `L_gain` is the stronger of a
  mean-gain LCB and a DKW expected-maximum LCB, while `U_gain` is the smaller of the mean-sum and
  DKW upper bounds.

For predeclared thresholds `delta` and `epsilon`:

| axis | certified lower conclusion | unresolved | certified upper conclusion |
|---|---|---|---|
| behavior coverage | `UNSATURATED` if `L_mass>delta` | interval crosses `delta` | `SATURATED` if `U_mass<=delta` |
| prompt value | `RISING` if `L_gain>epsilon` | interval crosses `epsilon` | `PLATEAUED` if `U_gain<=epsilon` |

These are prompt-evolution labels at one fixed executor and declared proposer horizon. They are neither OSL
executor-scaling labels nor all-finite-prompts claims. Patience/max-iteration stopping never implies a
plateau by itself.

## Optional stopping and resume

Monitor batch `t` is scored against the pool frozen before that batch, then committed through one ordered,
fsynced `absorption_ledger.jsonl` row. The next iteration reconstructs its pool only from that ledger, never
by scanning directories. Stopping fires when both monitor targets are met, at `max_iter`, or after declared
patience without sufficient improvement.

Monitor numbers select the stopping time and are not reported as the final certificate. At stopping, the
loop draws a separately seeded confirmation audit in a namespace the ledger loader never reads. Its
single-shot `confirmation/certificate.json` is the only bound-grade endpoint. Completed uncommitted files
are reusable after a crash; committed artifacts and confirmations are immutable.

To certify tightening rather than merely plot adaptive monitors, predeclare `--checkpoint-iters`. Before
the next monitor at each listed absorbed-pool size, the loop draws a separately seeded `checkpoint` audit;
it is never added to the pool. Alpha is Bonferroni-allocated across checkpoint plus final cells for each
metric. Supplying `--study-alpha` additionally allocates across all declared metrics, making the complete
reported trajectory familywise simultaneous. `certified_trajectory.json` contains only these immutable
points; monitor rows are explicitly excluded.

## Artifacts

```
<root>/run_manifest.json
<root>/evidence_install.json
<root>/mcq_codebooks/<task>.json
<root>/mcq_codebooks/<task>.panel_plan.json
<root>/mcq_codebooks/<task>.prior_calibration.json
<root>/mcq_codebook_candidates/<metric>/bootstrap/scored.npz
<root>/mcq_panel_envelopes/<metric>/<panel_id>/codebook.json
<root>/mcq_panel_envelopes/<metric>/<panel_id>/states.npz
<root>/mcq_panel_envelopes/<metric>/<panel_id>/values.npz
<root>/mcq_panel_envelopes/<metric>/<panel_id>/envelope.json
<root>/mcq_state_tables/<metric>/states.npz
<root>/mcq_state_tables/<metric>/values.npz
<root>/mcq_state_tables/<metric>/envelope.json
<root>/signature_cache/<namespace>/<prompt_sha>.npz
<root>/<metric>/bootstrap/scored.npz
<root>/<metric>/bootstrap/values.npz
<root>/<metric>/historical/candidates.jsonl
<root>/<metric>/historical/scored.npz
<root>/<metric>/historical/values.npz
<root>/<metric>/historical/import.json
<root>/<metric>/monitor/iter_NNN/proposal_<family>.jsonl
<root>/<metric>/monitor/iter_NNN/scored.npz
<root>/<metric>/monitor/iter_NNN/values.npz
<root>/<metric>/absorption_ledger.jsonl
<root>/<metric>/checkpoint/iter_NNN/proposal_<family>.jsonl
<root>/<metric>/checkpoint/iter_NNN/scored.npz
<root>/<metric>/checkpoint/iter_NNN/values.npz
<root>/<metric>/checkpoint/iter_NNN/certificate.json
<root>/<metric>/confirmation/iter_000/proposal_<family>.jsonl
<root>/<metric>/confirmation/iter_000/scored.npz
<root>/<metric>/confirmation/iter_000/values.npz
<root>/<metric>/confirmation/certificate.json
<root>/<metric>/certified_trajectory.json
<root>/mcq_identity_final.json
<root>/mcq_query_cache/choice_probabilities.sqlite
```

`mcq_identity_final.json` applies the per-metric best-pool selection rule and reports achieved
`I(J;Jhat)` for annotations, no-demonstration, and shuffled-label channels. It preserves the requested
bank-level MI readout but is not itself a prompt-space upper bound.

The query cache is part of the value definition, not a speed-only convenience: identical frozen teaching
transcripts receive byte-identical choice probabilities across bootstrap, monitor, checkpoint, confirmation,
and resume. Value evaluation batches rendered queries and evaluates the prompt-independent no-demo channel
only once per metric. V12 cache keys include the exact constrained-choice protocol ID. Frozen `q_no_demo`
is an exact arithmetic mean over the finite full-factorial 24-order block, not a binomial estimate; it has
no Clopper-Pearson interval. A claim over random menus, items, or reconstructor runs is a separate estimand.

The all-prompt lower endpoint is the best value among the absorbed pool and the current fresh audit. The
fresh audit remains excluded from the pool used by CR-3 gain/missing-mass calculations; using an observed
audit prompt as an achieved global lower bound does not alter that conditioning. Every MCQ global payload
also contains `instrument_quality`. Headline gates are the blind prior, coarse headroom of at least `0.10`,
`U_state` above the predeclared resolution, positive lift plus unique target identification for the frozen
canonical/orbit target replay, and the exact
four-option 24-order block. A different frozen finite block still defines a valid formal finite-state
functional and upper envelope, but it is reported only as `FORMAL_CERTIFICATE_ONLY`. The stored
operational-target transcript is a prospective instrument gate but is neither an external anchor nor achieved
prompt-search evidence. Raw witness values/species remain in all C/R gain calculations. Prompt-evolution
status is computed from a second pool that removes preloaded target-form witnesses while retaining every raw
fresh-audit mark; an all-prompt optimum supported only by a design witness is `DESIGN_WITNESS_ONLY` and not a
scientific epsilon-optimal headline. The historical selected-distractor kappa `0.50` threshold remains hashed and reported as a descriptive
near-clone diagnostic; it does not gate
headlines. A failed gate retains a formally valid interval as `FORMAL_CERTIFICATE_ONLY`.

### Reporting tiers (declared before v11 audit results; implemented in v12)

The primary certificate and every `CERTIFIED_*` label use 95% simultaneous confidence at the scope recorded
in the payload. The same never-absorbed audit may additionally be recomputed at 90% confidence as a
predeclared sensitivity tier. A label that passes only there is written `SUGGESTIVE_*`, never `CERTIFIED_*`.
Both intervals and the unchanged point estimates are reported, so the 90% tier cannot replace an unfavorable
95% result or be selected metric by metric. This secondary computation is CPU-only and does not change the
frozen GPU run, its stopping rule, or its evidence ledger.

Instrument quality gates apply to both global and process-value conclusions. A prior-degenerate,
canonical-state-dead, low-headroom, or nonfactorial MCQ instrument retains its formal fixed-instrument
mathematics, but its value status is
`FORMAL_CERTIFICATE_ONLY` only when the statistical value axis was directionally resolved; an unresolved
axis remains `UNRESOLVED`. Independent behavioral `SATURATED/UNSATURATED` conclusions remain reportable.
Fake/dry runs set every global/process/trajectory/bank status to `SYNTHETIC_TEST_ONLY` and
`publication_eligible=false`; their diagnostic numbers are regression fixtures, never empirical results.

## Tightening levers

- **Mine more:** enlarges `Omega_N`, can raise `R_Omega`, and changes future gain marks toward zero.
- **Reuse old GPU generations:** admit them only as re-scored/revalued candidates. They can raise the
  achieved endpoint but are never spent a second time as confirmation evidence.
- **Add a value-tilted proposer stratum:** after a design stage, freeze mutations/compositions of
  high-value prompts as a separate family with its own future quota and confidence components. Never
  weight gains post hoc.
- **Audit more:** shrinks Clopper-Pearson, empirical-Bernstein, and DKW penalties; it does not move the
  underlying novelty/gain center at a fixed pool.
- **Set the horizon scientifically:** a smaller bound for 100 future prompts does not certify 10,000.
- **Broaden proposer families:** strengthens prompt-class scope but may reveal new mass and loosen the bound.
- **Change teaching-panel size only prospectively:** `k=10` has 1,024 states and may change instrument
  resolution, but it is a different estimand and release namespace, not a post-result tightening knob.
- **Increase/resample probes:** required for deployment-distribution generalization; use a new run namespace
  and an independent item-lockbox confidence layer rather than mixing panels.
- **Justify positivity externally:** only a defensible `p_min` turns exact missing mass into full-support
  exhaustion. Behavior `p_min` and value-state `value_p_min` are separate assumptions. A sensitivity curve
  is not evidence for any point on it.

## Launch

```bash
CUDA_VISIBLE_DEVICES=<free> python methods/metric_implementer/experiments/run_cr3_mining_loop.py \
  --metrics <..._sigs.npz ...> \
  --out-root /lfs/skampere3/0/alexspan/outputs/<new-immutable-v12-fixed-t8-root> \
  --mcq-codebook-metrics <full frozen task-and-level checkpoint bank ...> \
  --reuse-bootstrap-root <verified-prior-cr3-root> \
  --reuse-mcq-codebook-root <optional prior root with the same candidate bank> \
  --reuse-evidence-root <optional immutable historical evidence store> \
  --value-mode reconstruction_mcq \
  --mcq-reconstructor Qwen/Qwen2.5-14B-Instruct \
  --mcq-choice-readout logits --mcq-n-examples 8 --mcq-reconstruction-draws 24 \
  --mcq-value-query-batch-size 512 \
  --families microsoft/phi-4 microsoft/phi-4 Qwen/Qwen2.5-14B-Instruct meta-llama/Llama-3.1-8B-Instruct \
  --family-tags phi4_atomic phi4_holistic qwen14_atomic llama8_holistic \
  --family-modes atomic holistic atomic holistic \
  --batch-per-family 150 --confirm-per-family 300 --checkpoint-per-family 300 \
  --checkpoint-iters 0,1,2,4,8 --study-alpha 0.05 \
  --ceiling-horizon-per-family 100 \
  --target-u0 0.10 --target-value-gap 0.02 --max-iter 12 --patience 3
```

The July-11 live run used one shared seed for a batch of identical prompts and silently omitted some
family quotas. It was terminated. Its monitor values and all pilot-v2 intervals lacking iid provenance are
historical diagnostics, not evidence for a prompt ceiling.

The v11 breadth sentinel later failed closed during its first news checkpoint because an admissible mined
rubric elicited neither `YES` nor `NO` inside the legacy top-logprob window. The six humor checkpoint
matrices completed before that task-order failure remain immutable never-absorbed audits and may receive
per-metric v11 certificates; they cannot define a six-of-twelve population pass rate. V12 prevents recurrence
with its total two-token behavior readout. This changes the behavior functional, so v11 prompt texts are
reusable only after v12 rescoring, not by copying their cached signatures.

The `cr3_mining_v2` run launched on 2026-07-12 predates the Reconstruction-MCQ value-mark and certified-
checkpoint integration. Its fresh final audit can support an individually scoped legacy fixed-target
behavioral result. It cannot be relabeled as the primary reconstruction optimum or as a simultaneous OSL
result.
