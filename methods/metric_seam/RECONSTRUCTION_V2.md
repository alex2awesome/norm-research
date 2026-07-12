# Metric-seam reconstruction v2

This is an additive confirmatory lane. Historical metric-seam programs and outputs remain
frozen and are treated as exploratory or retrospective-replay artifacts.

## Canonical terms

- **Articulability** is prompt-based. It measures whether a prompt/LLM program can recover
  the frozen LLM judgement that operationalizes an articulated criterion.
- **Verifiability** is code-based. It requires an executable, replayable, scoped certificate
  such as a symbolic derivation, program execution, type/flow invariant, date computation,
  or claim-dependency trace.
- **Reconstruction** is agreement with the frozen LLM judgement. Prompt, code, and hybrid
  channels can each be evaluated for reconstruction.
- **Isomorphism** is fidelity between the construct, input representation, executed program
  path, and frozen reference judgement. It is an evaluation property, not a synonym for
  articulability or verifiability.
- **Constructive extension** is a narrow verifier-dominant disagreement: code fails to
  reconstruct some LLM judgements but supplies a valid code-native certificate that directly
  adjudicates those cases. A higher correlation, lower error, or plausible explanation alone
  is not enough.

## Constructive asymmetry (the Collins test)

The experiment is deliberately asymmetric:

- A successful prompt/LLM reconstruction is an executable witness of **articulability**.
- A successful code-native certificate is an executable witness of **verifiability**.
- Neither success is evidence for the other channel; prompt articulation is not a code
  certificate, and a code certificate is not prompt articulation.
- A failed search establishes only bounded non-discovery: no acceptable witness was found in
  the frozen program class, capability set, compiler, data representation, and search budget.
  It does **not** establish that the construct is tacit.

This is the operational form of the claim that it is easier to show that something is
articulable (and, separately here, verifiable) than to show that it is tacit. Positive cases
have finite witnesses. A tacitness claim would require ruling out open-ended future prompts,
programs, representations, and capabilities, which this experiment does not attempt.

The historical term “ground truth” may continue to appear in frozen artifacts. In v2 reports,
use **frozen LLM reference** for those scores. Code-native certificates are reported on a
separate plane rather than substituted as a new supervised target.

## Two-plane evaluation

Every relation reports both planes even when one is unavailable:

1. **Reference plane:** held-out agreement with the frozen LLM reference. This is the
   isomorphic-reconstruction readout.
2. **Certificate plane:** success, coverage, abstention, and counterexamples for the
   executable verifier.

This produces outcomes such as `articulable_only`, `verifiable_only`,
`dual_implementation` (both channels exist but isomorphism is unavailable),
`dual_reconstruction`, `hybrid_complement`, `constructive_extension`, `proxy_mismatch`, and
`unresolved`. The executable definitions live in `reconstruction_v2.py`.

The policy is therefore **isomorphism-first, not isomorphism-only**. Isomorphic substitution
is the cleanest evidence that two media implement the same articulated relation. A verifier
that departs from the frozen LLM reference is not automatically a failure: it may be reported
as constructive extension only on the subset where a replayable certificate directly
adjudicates the disagreement. Uncertified disagreement remains reference divergence.

The primary unit may be a criterion sub-relation rather than the whole criterion. Presence,
position, attribution, entailment, execution, and functional-use relations receive separate
records because they may land on different channels. `SubrelationEvidence` records both the
construct relation and the operation actually computed; `CriterionDecomposition` deliberately
does not infer a parent outcome unless a separate aggregation rule was frozen. These records
complement rather than replace the census relation-match taxonomy.

## Blind reconstruction protocol

Development receives only:

- the frozen articulated criterion and relation contract;
- unlabeled train `ctext`;
- the declared capability catalog and allowed frozen LLM-field interface;
- construct-derived metamorphic, invariance, coverage, and abstention feedback.

Development never receives reference scores, residuals, test identifiers, dataset outcome
labels, or a retrieval index fitted on held-out text. The frozen LLM reference is read once
after candidate and manifest hashes are fixed.

## Isomorphism invariants

Within a comparison block, hold constant the criterion, relation ID, `ctext`, evidence
payload, split, compiler model, prompt, starting program, round/token budget, contract,
reference judge, and evaluation rule. A capability experiment changes only the available
operation. Compare a relation-matched operation with an equally deep mismatched operation
and with no added operation.

Typed programs must derive source provenance at runtime. Code nodes may not read LLM fields
unless an explicit typed edge permits it, and disconnected nodes cannot affect seam readouts.
New adversarial uses should use `battery/dag_schema_hardened.py`, which rejects ambient globals,
closures, hidden defaults, and stateful callable objects before rebuilding accepted functions
inside a minimal namespace. This is a callable-level provenance guard, not an OS sandbox;
capability-object internals remain a separately trusted boundary.

## Executable depth

Every selected code operation receives a relation-depth tag so “code substitution” cannot
quietly mean only keyword matching:

0. surface lexical matching;
1. parsed document structure;
2. cross-span or cross-section relation checking;
3. formal solver or evidence-graph execution;
4. environment/world execution.

Depth alone is not quality. A level-4 operation can be irrelevant or mostly indeterminate,
while a level-1 parser can exactly match a structural construct. The causal comparison is
relation-matched capability versus an equally deep mismatched capability versus no added
capability, with coverage and abstention reported at every depth.

## Manual, mocked, oracle, and replay artifacts

Expensive prior artifacts do not need to be regenerated. Every result must declare one of:

- `agentic`: created by the blind compiler in this run;
- `manual`: constructed by a researcher;
- `mock`: simulated machinery or payload;
- `oracle`: built with privileged truth/certificate access;
- `replay`: a frozen prior artifact evaluated as if proposed to the blind selector.

A replay experiment may ask whether a frozen selection rule would rediscover the value of a
manual/mock/oracle decomposition. It must not state that the original decomposition was
automatically discovered.

Original provenance and current experimental role are orthogonal. An expensive manual or
mock artifact may be marked `pipeline_status=selected` and
`selection_mode=retrospective_seed`: this means “the seeded pipeline selected this operation
and we now measure what it contributes.” It does not make the original construction agentic,
and its provenance alone does not make its measured contribution ineligible.

## Technical-domain scope

- **Code:** diffs support AST, CFG, import, exception-path, and test-to-production-symbol
  analysis. Do not claim compilation or test execution without a repository snapshot. The
  current technical replay's `f2p_mock/` and `pr_test_execution` cases are legacy prototypes,
  not the active coding census; its 800-row probe classifies frozen prior-execution telemetry
  and does not execute repositories in the current run.
- **Math:** Math.SE question/answer text supports symbolic fragments and relation-specific
  checks only after a corpus-presence screen.
- **Science:** the historical 250-item peer-review reconstruction task is abstract-based, but
  the repository also contains a newer 2,400-paper, section-targeted full-paper evidence set
  and a claim-source/body-verification set. Use the former task for frozen-reference
  reconstruction and the latter sets for code-native claim/evidence certificates. Do not use
  their accept/reject outcome as a new supervised anchor. Full statistical recomputation still
  abstains unless the represented article includes all required data and analysis details.
- **Patents:** abstract+claims support claim syntax, antecedent binding, and claim DAGs.
  Prior-art discovery and specification support are separate and any examiner-injected
  candidate set must be marked `oracle`.
