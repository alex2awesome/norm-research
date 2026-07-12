# CR-3 executor-indexed prompt-ceiling loop (authoritative v10, 2026-07-12)

## Objective

For one executor/readout protocol `E`, ordered probe panel `X`, initial prompt pool `Omega_N`, and frozen
proposer families `Q_f`, bound the best **single-prompt** value under a declared anchor-free reconstruction
measurement `V`:

```
V*_{b,E}(P) = sup_{p in P} V_{b,E}(p),  P = Omega_N union support(Q).
```

The primary `V` is Reconstruction-MCQ: the candidate prompt supplies its own executor annotations; a frozen
reconstructor sees only contrastively selected `(item, annotation)` demonstrations and a frozen option
codebook; normalized target-option probability and annotation-attributable lift are recorded. No anchor,
silver label, human label, or outcome enters. Across randomized target metrics, the companion bank-level
quantity is identity `I(J;Jhat)`.

The v2 fixed-target value `I(M_fixed ; binarize(E(p,X)))` remains supported as a **legacy behavioral
discovery diagnostic**. It is the value used by the currently running `cr3_mining_v2` job and must not be
reported as the final Reconstruction-MCQ optimum.

Every declared value has a predeclared finite cap. For MCQ target-option probability the cap is one; for
the primary annotation-attributable lift it is the sharper frozen-control cap
`1 - q_no_demo(target)`; for legacy binary MI it is `H(M_fixed)`. CR-3 can tighten that cap in two declared
scopes:

1. a finite future mining budget: an upper confidence bound on the **expected** best prompt after fixed
   counts `m_f` of additional draws from each family;
2. the entire proposer support: only when exact-pattern missing mass is below an externally justified
   minimum support mass `p_min`.

It does not bound arbitrary strings, multi-prompt checklists, or the latent ideal `M*` unless those objects
are separately identified and placed inside the declared prompt/readout class. In particular, capture-
recapture broadens beyond the discovered pool to a frozen proposer process; it does not silently become an
all-strings theorem.

These are complementary bounds. The MCQ frozen-control cap is an assumption-light range bound covering
every finite prompt but contains no search-rate information. CR-3 is a discovery/gain bound that can tighten
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
- `experiments/cr3_reconstruction_values.py` freezes bootstrap-only codebooks and validates/serializes the
  all-row MCQ value transaction. It supplies values to `cr_audit.py`; it is not a second certificate engine.
- `experiments/run_cr3_mining_loop.py` is the user-facing orchestrator. It owns the immutable manifest,
  bootstrap transaction, monitor/absorption ledger, stopping rule, resume behavior, and isolated
  confirmation audit. It invokes the worker one model per subprocess, then calls `cr_audit.py` on CPU.
- `vllm_backend.py` remains the shared backend. CR-3 only extends it compatibly so one batch can carry a
  distinct seed per request and deterministic seeds for binary readout.

The v10 end-to-end path is:

```
source checkpoints -> bootstrap/cache -> propose -> score behaviors -> measure frozen MCQ value marks
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
values are optimized. `--mcq-codebook-metrics` declares the broader task-level bank from which each target's
closest behaviorally distinguishable distractors are frozen. Every bank member receives one lightweight
`mcq_codebook_candidates/<metric>/bootstrap/scored.npz` containing only its canonical/orbit-averaged target
behavior; its historical source prompts are neither rescored nor admitted into target search. A prior
candidate bank may be hard-linked with `--reuse-mcq-codebook-root` only after validating the source
checkpoint hash, metric identity, executor namespace, and artifact schema.

**v10 freeze (2026-07-12).** The production estimand, schemas, thresholds, and exact-code resume contract
are frozen. New task coverage uses new immutable roots and hash-compatible cache/bootstrap reuse; it does
not migrate or rewrite older roots. Exact source hashes remain load-bearing, so a changed release cannot
resume a v10 root. Supported existing R3 bank prefixes are creative writing, humor, news homepages, press
releases, code review, Math StackExchange, grant funding, peer review, and legal outcome prediction.

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
`G(p)=max(0,V(p)-V_Omega)`, bounded by `B=value_cap-V_Omega`. Every audit draw receives a value mark,
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

The reported finite-horizon expected prompt ceiling is `V_Omega + U_horizon`. No species value,
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

At every immutable confirmation, v4 constructs one simultaneous two-sided evidence bundle
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
<root>/mcq_codebooks/<task>.json
<root>/mcq_codebook_candidates/<metric>/bootstrap/scored.npz
<root>/signature_cache/<namespace>/<prompt_sha>.npz
<root>/<metric>/bootstrap/scored.npz
<root>/<metric>/bootstrap/values.npz
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
only once per metric.

The all-prompt lower endpoint is the best value among the absorbed pool and the current fresh audit. The
fresh audit remains excluded from the pool used by CR-3 gain/missing-mass calculations; using an observed
audit prompt as an achieved global lower bound does not alter that conditioning. Every MCQ global payload
also contains `instrument_quality`. A low-headroom or behaviorally easy panel retains a formally valid
fixed-instrument interval but is marked `FORMAL_CERTIFICATE_ONLY`; it cannot supply the scientific headline.
The defaults are a value cap of at least `0.10` and minimum selected-distractor kappa of at least `0.50`,
both hashed in the run manifest and configurable before data collection.

## Tightening levers

- **Mine more:** enlarges `Omega_N`, can raise `R_Omega`, and changes future gain marks toward zero.
- **Audit more:** shrinks Clopper-Pearson, empirical-Bernstein, and DKW penalties; it does not move the
  underlying novelty/gain center at a fixed pool.
- **Set the horizon scientifically:** a smaller bound for 100 future prompts does not certify 10,000.
- **Broaden proposer families:** strengthens prompt-class scope but may reveal new mass and loosen the bound.
- **Increase/resample probes:** required for deployment-distribution generalization; use a new run namespace
  and an independent item-lockbox confidence layer rather than mixing panels.
- **Justify positivity externally:** only a defensible `p_min` turns exact missing mass into full-support
  exhaustion. Behavior `p_min` and value-state `value_p_min` are separate assumptions. A sensitivity curve
  is not evidence for any point on it.

## Launch

```bash
CUDA_VISIBLE_DEVICES=<free> python methods/metric_implementer/experiments/run_cr3_mining_loop.py \
  --metrics <..._sigs.npz ...> \
  --mcq-codebook-metrics <full frozen task-level checkpoint bank ...> \
  --reuse-bootstrap-root <verified-prior-cr3-root> \
  --reuse-mcq-codebook-root <optional prior root with the same candidate bank> \
  --value-mode reconstruction_mcq \
  --mcq-reconstructor Qwen/Qwen2.5-14B-Instruct \
  --mcq-choice-readout logits --mcq-value-query-batch-size 512 \
  --families microsoft/phi-4 Qwen/Qwen2.5-14B-Instruct meta-llama/Llama-3.1-8B-Instruct \
  --family-tags phi4 qwen14 llama8 \
  --family-modes atomic holistic atomic \
  --batch-per-family 150 --confirm-per-family 300 --checkpoint-per-family 300 \
  --checkpoint-iters 0,1,2,4,8 --study-alpha 0.05 \
  --ceiling-horizon-per-family 100 \
  --target-u0 0.10 --target-value-gap 0.02 --max-iter 12 --patience 3
```

The July-11 live run used one shared seed for a batch of identical prompts and silently omitted some
family quotas. It was terminated. Its monitor values and all pilot-v2 intervals lacking iid provenance are
historical diagnostics, not evidence for a prompt ceiling.

The `cr3_mining_v2` run launched on 2026-07-12 predates the Reconstruction-MCQ value-mark and certified-
checkpoint integration. Its fresh final audit can support an individually scoped legacy fixed-target
behavioral result. It cannot be relabeled as the primary reconstruction optimum or as a simultaneous OSL
result.
