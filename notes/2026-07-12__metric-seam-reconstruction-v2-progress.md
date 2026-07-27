# Metric-seam reconstruction v2 — technical progress and claim envelope (2026-07-12)

This note records the additive v2 work performed after the independent verification review.
Historical empirical outputs remain recoverable in the prior freeze/commit; additive corrections
do not rewrite them. The new lane stays on the unsupervised reconstruction objective: the old LLM
judgement is a frozen reference for reconstruction agreement, never a newly supervised external
target. Isomorphism additionally requires construct, input, program, and reference fidelity.

## 1. Operational thesis

The experiment now keeps three questions separate:

1. **Articulability (prompt-based):** can a prompt/LLM program implement the articulated
   relation?
2. **Verifiability (code-based):** can an executable program issue a scoped, replayable
   relation certificate or honest abstention?
3. **Isomorphism:** do the prompt, code, or hybrid program and the frozen LLM reference rank
   the same represented evidence as the same construct?

The Collins asymmetry is machine-enforced. A successful prompt program is a finite witness of
articulability; a successful executable certificate is a finite witness of verifiability.
Failure establishes only bounded non-discovery within the frozen program class, capabilities,
representation, compiler, and budget. It never licenses a claim of tacitness.

The policy is **isomorphism-first, not isomorphism-only**. Isomorphic substitution is the
cleanest result. Code that adds valid evidence unavailable to the prompt may instead earn an
evidence-surface extension. A stronger verifier-dominant constructive extension requires a
replayable certificate on the exact cases where code and the frozen LLM reference disagree,
plus explicit passes for input, executed-program, and reference-instrument fidelity. The typed
record now rejects a constructive-extension flag unless reference reconstruction actually
fails under those controls. Uncertified or representation-confounded disagreement remains
reference divergence.

## 2. Instrument changes

- `methods/metric_seam/reconstruction_v2.py` now provides typed prompt/code/isomorphism axes,
  orthogonal historical provenance and current pipeline-selection status, joint outcomes, and
  executable claim permissions. Reference reconstruction is now a separate canonical readout;
  isomorphism requires construct, input, executed-program, and reference-instrument fidelity
  passes, with missing checks defaulting to unavailable. `may_claim_tacitness` is always false.
- `battery/blind_reconstruction_v2.py` and `battery/evaluate_blind_v2.py` implement a clean
  compiler/evaluator split. Development sees the contract, unlabeled TRAIN `ctext`, opaque
  aliases, and allowed capabilities; it never sees reference values, residuals, held-out IDs,
  or an index fitted on held-out text. The evaluator executes the exact frozen candidate bytes
  before opening the reference.
- `battery/contract_check_isomorphic.py` separates CODE and HYBRID gates. Missing prompt fields
  abstain rather than count as code failures; contracts, prompts, extractor provenance, and
  probe texts are SHA-bound.
- `battery/dag_schema_enforced.py` derives code/LLM/evidence taint at runtime. Nodes may read
  only declared typed inputs; disconnected or dynamically unused LLM nodes cannot move the
  measured seam.
- `battery/certify_batch_v2.py` adds immutable SHA-pinned batches, paired common support,
  minimum-effect gates, paired permutation tests, bootstrap intervals, and separate
  Benjamini-Hochberg families. It reports reference availability over the full held-out split
  separately from candidate coverage conditional on reference availability.
- `hybrids/ops_capability_v2.py` is an additive corrected wrapper over the frozen v1 library.
  The audit correctly found that the initial seven-case replay omitted three named historical
  defects. Additive v2.1 now retains invalid calendar surfaces as explicit non-checkable rows,
  resolves coordinated reporting verbs through licensed shared subjects, recognizes a narrowly
  marked named-action-beat-to-adjacent-quote relation, and removes the three-word refrain floor
  while preserving the no-adjacent-craft guard. The complete claimed-behavior ledger moves from
  0/17 correct in frozen v1 to 17/17 in v2.1 (the seventeenth case is the exact historical
  two-word a270 refrain surface and is explicitly scoped to local-relation detection, not a
  document-level a270 score). The audited 7-case artifact remains unchanged and a new
  `capability_counterexamples_v2_1.json` records the expanded replay. Historical runs
  continue to use the hashed v1 behavior.
- Executable substitutions receive a depth tag: 0 surface lexical, 1 parsed structure, 2
  cross-span/section relation, 3 formal solver/evidence graph, and 4 environment/world
  execution. Depth is reported with relation match and abstention; it is not treated as quality
  by itself.

## 3. New empirical results

### 3.1 Blind Math a144 reconstruction

A clean-room agent received only the frozen a144 contract, 150 unlabeled TRAIN `ctext` items,
and the base/math/capability catalogs. It produced a code-only structural program and ran once
with label-free feedback. The exact source was then sealed and evaluated on the 100-item
held-out complement.

- Candidate execution coverage: 100/100 held-out items.
- Frozen two-pass LLM-reference availability: 52/100; pass reliability rho = 0.8593.
- Candidate reconstruction on the 52 common items: Spearman rho = 0.0660; MAE = 0.3662.
- Historical code+LLM h0 on the same 52: rho = 0.4832.
- Existing pure-code columns on the same support: rho = 0.0106 / 0.0305 / 0.0945.
- The frozen contract had labeled all four probes `L`. The code-only candidate nevertheless
  produced positive-minus-negative deltas +0.23, +0.51, -0.05, +0.24: 3/4 clear the diagnostic
  +0.05 margin. This is an off-label channel challenge, not a CODE contract pass and not a
  retroactive relabeling.
- An independent, pre-frozen construct adversary then tested 26 novel pairs / 52 cases without
  reading any corpus item or judgement. The candidate passed only 14/26 ordering checks and
  33/52 expected ranges (frozen requirements: 24/26 and 47/52) and was **REJECTED**. It failed
  false-witness truth, quantifier scope, question/quotation isolation, and feature-laundering
  contrasts; seven critical pairs failed.

This is a useful negative reconstruction result. A code program can express several local
witness/scope contrasts yet fail both independent construct fidelity and population ranking.
The canonical outcome is therefore `proxy_mismatch`, not reference divergence or constructive
extension. Local sensitivity to a small authored contract is not sufficient evidence of either
construct fidelity or isomorphism. The historical hybrid's much higher agreement also shows why
prompt/code complementarity should remain an explicit outcome. The frozen held-out result must
not be used to tune another confirmatory a144 candidate.

The procedural claim remains positive and bounded: this was one genuinely blind, automatic
decomposition/program proposal. Automatic proposal is not the same as successful verifiability
or isomorphic reconstruction, and the machine-readable claim permissions preserve that
distinction.

Canonical artifacts:
`outputs/metric_seam_pilot/reconstruction_v2/blind_math_a144_001/sealed_eval_002/`.
Independent adversary:
`outputs/metric_seam_pilot/reconstruction_v2/blind_math_a144_001/adversary_001/`.
Canonical outcome record:
`outputs/metric_seam_pilot/reconstruction_v2/blind_math_a144_001/reconstruction_record.json`.

### 3.2 Legacy code prototype: frozen prior-execution telemetry

The selected retrospective case points to a frozen transplant artifact produced by the old
`f2p_mock/` / `pr_test_execution` prototype, not the active coding census (whose h0 fleet is
still pending). The current replay classifies stored rows; it does not launch executions.

- 800 stored execution rows across Python, Go, and Java.
- 65 behavioral transition certificates (8.125%): 29 pinned, 10 partial-pinned, 26 vacuous.
- 571 indeterminate, 145 `none` abstentions, and 19 other execution errors.
- One duplicated row ID (`chia-blockchain:496`) with conflicting non-certificate outcomes is
  disclosed and must be resolved in the next consolidation build.

This supports a narrow level-4 code-verifiability claim: execution can certify pinning or
vacuity on a minority of cases and abstains on most others. The probe ignores the source
accept/reject fields. It does not turn execution coverage into outcome prediction.

### 3.3 Full-paper science

The earlier abstract-only replay was stale as a statement about available evidence. The repo
contains 2,400 section-targeted full-paper records and 2,400 abstract-to-body verification
records; 1,957 have nonempty full-paper-derived bodies.

Two additive certificate instruments now run without reading `y`:

- A deliberately narrow recurrence probe finds 660 eligible abstract numeric relations and
  429 cases with at least one normalized number recurring in the body. Zero recurrence is
  unresolved, not unsupported.
- A deeper level-2 verifier segments abstract claims, builds a per-paper BM25 index, constructs
  a claim/evidence graph, and performs exact maximum-weight bipartite matching. The audited
  historical v2 count of 171 was contaminated by partial quantity and identifier matches and
  remains preserved only for provenance. The audit-corrected v2.2 emits **136** strong relation
  certificates across **126** papers: 99 normalized-numeric and 37
  entity/baseline/direction-checked comparative certificates. It separately reports 435 weak
  `evidence_link` witnesses across 382 papers rather than calling lexical+artifact matches
  semantic support. Status counts are 126 supported, 358 evidence-link, 1,396 insufficient,
  and 520 abstain. Exact identity comparison retains 129 old certificates, removes 42, and
  exposes 7 legitimate complete-suffix matches. A deterministic corrected strong sample is
  10/10 plausible executable relation recurrence; this remains neither scientific ground truth
  nor statistical verification.

The same-input prompt-articulability counterpart is frozen at
`methods/metric_seam/science_claims_v2/articulability_prompt.json`. It was not part of this
code-verification result; a later bounded transport smoke is reported separately in Section 3.6.
The 2,400-paper code result itself therefore remains code verifiability and evidence-surface
extension, not a prompt/code isomorphism result.

### 3.4 Seeded Math, Patents, and technical replay

Prior expensive artifacts are now `pipeline_status=selected` with
`selection_mode=retrospective_seed`; their original manual/mock/oracle/replay provenance remains
unchanged.

- Math a150: the SymPy checker distinguishes one valid from one wrong synthetic consequent,
  but reaches 0/20 real condition-to-operation licensing relations. This is a precise
  relation/corpus mismatch, not evidence that the criterion is tacit.
- Patents a34: on the same oracle-conditioned prior-art evidence surface, the evidence-aware
  hybrid reaches rho 0.745 versus 0.084 with the evidence op nulled (marginal +0.661). This is
  strong selected-pipeline utility and reconstruction agreement conditional on privileged
  evidence; it is not a full isomorphism certificate, autonomous prior-art retrieval, or
  pure-code verification.
- The seven-case technical catalog currently permits five selected-utility claims, two
  canonical code-verifiability claims, zero historical automatic-selection claims, zero
  canonical verifier-dominant disagreement extensions, and zero tacitness claims.

See `outputs/metric_seam_pilot/technical_replay_v2/REPORT.md`.

### 3.5 Additive continuation: fresh blind Math a216 and the active coding lane

These two runs preserve the experiment's three-axis vocabulary. **Articulability** means a
prompt/LLM program implementing the articulated relation; **verifiability** means a code program
executing a scoped relation check; **isomorphism** is the separate comparison of either program
with the frozen LLM judgement under additional fidelity checks. A code program can therefore
have stronger reconstruction agreement with that reference than a prompt program without
thereby establishing isomorphism or proving that code is correct on their disagreements.

#### Blind Math a216: automatic proposal, construct-fidelity preemption

The second clean-room compiler run targeted the equation-numbering/label/reference graph. Its
bundle contained only 150 opaque TRAIN `ctext` rows, the projected a216 contract, and the allowed
base/math operations: no prompt fields, labels, reference scores, held-out IDs, or held-out text.
The agentic compiler emitted and froze a code-only parser/graph score. A separately frozen
construct adversary then evaluated that exact candidate before any held-out reference was
opened.

- The candidate passed 30/34 authored ordering pairs (0.8824) and 7/10 range anchors (0.7000),
  with mean pair delta 0.4736 and output standard deviation 0.3573.
- Aggregate sensitivity was not enough. It scored 0/1 on the semantic-target category and 1/3
  on subequation grouping, making the minimum category pass rate 0.0 against the frozen 0.5
  floor. The category-floor condition therefore failed.
- Construct-fidelity failure preempted reference access. The 100-row held-out LLM reference was
  deliberately left unopened, so this run has no held-out correlation and makes no isomorphism
  claim.
- The canonical outcome is `proxy_mismatch`: the run is a genuine automatic decomposition/program
  proposal, but not a successful witness of code verifiability, isomorphic reconstruction, or
  tacitness. It establishes only bounded non-discovery in this frozen program class and budget.

An independent CPU recomputation reproduced all 34 pair decisions, 10 range decisions, summary
statistics, and the canonical preemption outcome. The unopened held-out status is strongly
supported by the artifact ordering and run contents, not represented as a hostile-process or OS
sandbox proof. Canonical record:
`outputs/metric_seam_pilot/reconstruction_v2/blind_math_a216_001/reconstruction_record.json`.

#### Code-review a104: active census, channel-corrected V4

This is the **active coding lane**, not the old `f2p_mock/` prototype summarized in Section 3.2.
The inputs are canonicalized unified PR diffs; although their raw files live under
`datasets/code-review/pr_test_execution/batch_runs`, none of the legacy prototype programs,
test-execution telemetry, or per-PR test outcomes was used. No repository checkout or submitted
PR test execution occurred.

Three executable-code poles were compared on the same 97-item common held-out support:

- The frozen prompt-generated shallow-code baseline, selected from three Claude-generated
  Python regex/tanh scorers on TRAIN only, reached rho = 0.5089.
- The pre-existing deep static/AST checker reached rho = 0.6498 (delta = +0.1409,
  `P(gate)=0.5615`, `P(beats)=0.9455`) and passes the current reconstruction gate.
- The new relation h0 reached rho = 0.6064 (delta = +0.0975, `P(gate)=0.3235`,
  `P(beats)=0.8495`). It is a positive but sub-gate retrospective seed and must not be tuned on
  this held-out readout.

V3 corrected the h0 provenance to `manual_mock_retrospective_seed` with label-unreferenced
execution. Its evaluator projects `{datapoint_id, ctext}` before scoring and delays loading the
articulated LLM judgement, but the program was manually authored after label-bearing files
existed; it was not mechanically label-inaccessible and is not a blind-discovery result. The
deep checker also predates this h0 process. An independent sanitized-input rerun reproduced the
split, all six scorer/profile outputs with zero mismatches, all correlations, and both 2,000-draw
paired bootstraps; its targeted audit passed 11/11 checks.

V4 corrects the remaining channel taxonomy without changing a number or V3 provenance field.
All three historical “prompt-compiled” baselines are executable Python programs: prompt
generation is their authoring provenance, not their runtime channel. The licensed result is
therefore **within-code-channel program-depth reconstruction**: deeper static/AST code has higher
agreement with the frozen LLM judgement than TRAIN-selected prompt-generated shallow code. It is
not a direct prompt-articulability versus code-verifiability comparison, not “code over prompt,”
and not an isomorphism certificate from rho alone. It also does not prove code substantively
correct where the programs disagree. The static evidence covers source/test presence and
balance, AST name correspondence, and assertion structure; it does not certify behavioral
intent, oracle validity, or test success. A
repo-composition sensitivity is supportive but explicitly exploratory: on 92 rows from
repositories with at least two held-out items, within-repository centered-midrank Spearman was
0.299 for the shallow program, 0.581 for the deep checker, and 0.467 for the retrospective h0.
It is neither a new gate nor a tuning result.

No model inference or GPU was used in either continuation run. The a104 comparison nevertheless
uses a pre-existing model-produced judgement as the frozen reconstruction reference and
pre-existing model-produced code programs; “no inference in this run” does not mean that no
model artifact was used. Canonical V4 report:
`outputs/metric_seam_pilot/tasks/code_review/A104_CPU_SEALED_REPORT_V4.md`. Independent V3 receipt:
`outputs/metric_seam_pilot/tasks/code_review/A104_CPU_V3_INDEPENDENT_AUDIT_V1.md`.

### 3.6 Full-paper science: strict prompt smoke and relation-corrected code invariance

The prompt counterpart now has a bounded execution receipt, but not a criterion-level result.
The request builder deserializes each source row and immediately projects it to exactly
`paper_id + abstract + body`; it never indexes, emits, or uses the label value. It SHA-binds all
2,400 rendered requests and compares prompt and code as peers rather than treating either as an
external correctness target. Versions 1--5 are preserved as failed or instrument-development
attempts. Version
6 records five successful logical request results. Because the runner may retry and did not bank
attempt counts, this does not establish exactly five physical HTTP calls. It launched neither a
local model nor a GPU.

The earlier literal-guard V4 replay correctly enforced verbatim text but incorrectly allowed two
qualitative, zero-quantity, null-comparison objects to count as strong certificates. That violated
the frozen instruction that only exact numeric/comparative relations are strong. The canonical
strict-relation V7 replay now enforces grounding, typed relation semantics, and a non-estimating
support guard. It folds only
whitespace runs needed for PDF line wrapping; case, punctuation, hyphenation, and Unicode must
otherwise match the bound source literally. On the same five raw responses:

- 1/5 passed; it was an abstention with no strong certificate or weaker evidence link. Four were
  rejected: two for ungrounded evidence spans, one for a qualitative/non-relational strong
  certificate, and one for quantity bookkeeping without a quantity payload.
- Prompt strong certificates are therefore **0**, not 2. The sole shared abstention yields only a
  bookkeeping agreement cell. The evaluator marks the support `non_estimating` because `n=1` and
  no shared paper has a strong certificate; the resulting 1/1 fractions must not be promoted.
- Reasoning was requested off, but provider telemetry reported 12,426 reasoning tokens for one
  of five responses. Hidden reasoning text was not retained, so the run is described as
  reasoning-off-requested rather than uniformly reasoning-free.

This smoke establishes validator selectivity, not full-corpus articulability or prompt/code
isomorphism. The source-addressed instrument segments the 2,400 records into 19,219 abstract and
152,750 body sentence addresses. Only 1,957 records contain a body; the other 443 form a
missing-input/abstention stratum and must not be described as full-paper semantic support. V8 fixes
the earlier schema/runner/resume gates, renders addresses rather than copied spans, hydrates exact
source text in code, and records physical attempts. The full prompt arm is prepared but remains
unexecuted because no API credential is available; preparation made no API call and used no GPU.
Its future outputs will be prompt-asserted relations, not code-verifier certificates.

The exact-address code arm separately runs the manually seeded claim selector, per-paper BM25
retrieval, relation parser, and one-to-one graph matcher over the same A/B source spans. Its first
v2.2/v8 result reproduced 136/136 outputs across representations, but a relation audit found
concrete false positives: 28% robustness matched 28% resource savings, 1000 nodes matched 1000
rounds, one nearby percentage satisfied two obligations, H.264 was treated as a quantity, and an
interrogative comparison counted as asserted support. The 136/136 result is therefore retained
only as **old-parser output invariance**, not as 136 validated verifier witnesses.

The additive v2.3/v9 correction requires one-to-one value obligations, exact units, local
entity/metric agreement, articulated numeric direction, stronger/weaker role and baseline
agreement, assertive comparisons, and filtering of codec/version, function-parameter, math-
constant, norm, and date identifiers. It emits **100 parser-accepted relation-local witnesses**
across **95 papers**: 68 numeric and 32 comparative. Continuous and addressed arms agree on all
100 after whitespace normalization and on all 95 supported-paper sets. Strict-text identity is
only 8/100 because addressed spans retain line breaks that the continuous segmenter replaces with
spaces. Paper status agrees on 2,396/2,400 rows; the four differences are weak
`evidence_link -> insufficient` transitions. Weak links are correspondingly representation-
sensitive (434 continuous, 430 addressed; normalized intersection 429).

This is a positive input-representation result for the frozen executable sub-relations, not full
semantic isomorphism, whole-paper scientific support, or external scientific truth. The program
is still a manually selected retrospective mock of a discovered decomposition, not an automatic
discovery result.

Canonical receipt:
`outputs/metric_seam_pilot/science_articulability_v6_openrouter_reasoning_off_prepared/evaluation_strict_relation_guard_v7.json`.
Report:
`outputs/metric_seam_pilot/science_articulability_v6_openrouter_reasoning_off_prepared/REPORT_STRICT_RELATION_GUARD_V7.md`.
Prepared V8 manifest:
`outputs/metric_seam_pilot/science_articulability_v8_hardened_prepared/manifest.json`.
Corrected continuous report:
`outputs/metric_seam_pilot/science_claims_v2_relation_strict_v23/REPORT.md`.
Corrected exact-address report:
`outputs/metric_seam_pilot/science_verifiability_v9_relation_strict_addressed/REPORT.md`.

### 3.7 Code-review a407: blind structural program, coverage wall, and matched seam design

The blind candidate is substantially deeper than regex: CodeScope v3 uses Tree-sitter parsers,
language-aware lexical scopes, declaration/use resolution, morpheme decomposition, and explicit
collision/shadowing graphs for Python, Go, Java, JavaScript, and TypeScript. The candidate was
authored from the articulated contract and sanitized unlabeled TRAIN input before opening the
a407 reference. It leaves `semantic_context_fit` null and exposes only a frozen structural partial
aggregate.

The sealed evaluator caught and excluded a prohibited target substitution: `items.json.judgement`
is PR merge outcome, not a407. The historical reconstruction reference is the pre-existing
two-pass a407 prompt instrument, using the active coding-lane convention `(pass1 + pass2) / 20`
on the numeric intersection. On the 99 exact-input held-out rows, 74 have both declaration
coverage and a two-pass reference. The structural aggregate reaches Spearman **0.1746**, Pearson
**0.1220**, MAE **0.1769**, and signed code-minus-reference mean **-0.0989**. Individual subscore
Spearman values range from 0.0357 to 0.2030; no leave-one-family-out result exceeds 0.2134.

Coverage, not algorithmic depth alone, is the dominant limitation. Of 75 declaration rows in the
100-row heldout bundle, 59 have parser errors/missing nodes, 46 are truncated, and only six are
strict-complete. The exact-input sensitivity is rho 0.8407 on those six versus 0.1111 on 68 partial
rows; the six-row number is explicitly exploratory and does not establish a completeness effect.
The 25 no-declaration rows are noncoverage—their neutral 0.5 never enters reconstruction—and 19
of them still contain 383 visible use events, exposing a declaration-only observation gap.
Heldout coverage is also language-skewed (61 Go, 12 Python, 2 Java, no JS/TS), so cross-language
measurement invariance is unavailable.

The audit now separates finite positive witnesses from absence claims. A locally valid detected
placeholder/collision event may be a relation-local code witness under partial input. “No event
detected” is not verified absence unless the parse and the relation observation universe are
complete; neither structural event establishes contextual inappropriateness or harmfulness. The
additive v4 policy encodes these states without rewriting v3.

The original raw/hybrid prompts were not executed: they omitted model-visible relation
definitions and changed both the evidence surface and output task. A pre-reference matched v2
addendum renders the same construct and six definitions, uses the same response schema and ctext,
and varies only whether CodeScope facts are null or present. It is still **full-graph
augmentation**, not substitution: hybrid messages are 5.0x larger at the median and include that
token/attention cost. The pre-result v4 design therefore calls for compact one-relation facts with
null, relation-matched, and equally deep relation-mismatched/token-length-matched controls, plus a
separate offline substitution lattice. Current v3 scalars are ineligible for substitution until
relation-specific construct adversaries pass.

The outcome is `DESCRIPTIVE_RECONSTRUCTION_ONLY`: whole-construct fidelity and reference-
instrument fidelity are unavailable, program fidelity fails, and full isomorphism is not
established. No new model/API/GPU operation was used. Canonical report:
`outputs/metric_seam_pilot/reconstruction_v2/a407_sealed_historical_eval_001/REPORT.md`.

### 3.8 Math a12: exact symbolic-step witnesses without a whole-proof claim

The next technical seed targets a narrow sub-relation of a12, “Precision and rigor in statements
and proofs”: an explicitly presented rational-algebra equality step should preserve the same
expression on its declared domain. It reuses the existing manually selected math-span extractor,
then parses bounded answer-side equalities with SymPy's Lark LaTeX parser, exactly reduces the
difference, and reports denominator-nonzero obligations. This is parsed symbolic execution, not a
regex or keyword score.

On 150 sanitized opaque TRAIN rows, 42 contain at least one executable rational pair and 108
abstain. The operation emits 24 exact identity certificates on 10 rows, 91 exact nonidentity
witnesses on 36 rows, one unresolved pair, and 327 parse-noncoverage pairs. It emits zero
document-level rigor defects. Nonidentity becomes a universal-identity counterexample only if a
separately frozen prompt-side scope judgment establishes that the equation was asserted
universally rather than used as a definition, special solution, assumption, or conditional step.
Ten construct-derived adversarial/metamorphic tests pass; whole-criterion fidelity and a parent
scalar remain unavailable.

This is honestly labeled
`selected_retrospective_seed_with_aggregate_train_summary_exposure`, not pristine blind
authorship: seed selection opened a legacy h0 docstring containing one aggregate TRAIN
correlation and qualitative rationale. No per-item outcome, heldout/reference value, residual, or
evaluation output was opened, and no value influenced the symbolic relation, threshold, or
weights. The reproducible preparation records SymPy/Lark and writes only aggregate coverage. No
model/API/GPU operation occurred. Canonical report:
`outputs/metric_seam_pilot/reconstruction_v2/math_a12_symbolic_step_retrospective_prepare_001/REPORT.md`.

## 4. What claims are broader now

The earlier review mainly narrowed overstatements. The v2 design also licenses positive claims
that the old single-axis reconstruction report hid:

1. **Selected machinery can be useful without being historically agentic.** Provenance and
   experimental pipeline role are orthogonal.
2. **Deep code can add a new evidence surface even without matching an LLM ranking.** Stored
   certificates from prior repository execution and corrected full-paper relation checking are
   examples; the current legacy-code replay itself performs classification, not execution.
3. **Verifiability survives unavailable isomorphism.** A valid code certificate remains a
   verifiability result when no channel-matched frozen LLM reference exists; it is reported as
   `verifiable_only`, not erased as unresolved.
4. **An authored seam annotation is a hypothesis, not a result.** The blind a144 program's 3/4
   off-label probe separations challenge the blanket all-L allocation, but the independent
   adversary rejects this particular executable witness. The allocation remains open rather
   than either proven L-only or successfully moved to code.
5. **Failure localizes the search boundary.** Zero SymPy coverage and low blind-a144 rho locate
   capability/representation/program-class limits; neither supports tacitness.
6. **Executable relations can be representation-robust without being semantically complete.**
   The corrected science relation set is identical across continuous and addressed inputs after
   whitespace normalization, while weak lexical links move and whole-paper truth remains outside
   the claim.
7. **Positive detection and verified absence have different evidence burdens.** A finite event
   can witness a code-native sub-relation under partial input; non-detection becomes an absence
   certificate only under a complete parse and observation universe.

## 5. Audit remediation and relation-local follow-up

- The capability library is now honestly v2.1-complete for the named audit set: frozen v1
  scores 0/17 and additive v2.1 scores 17/17. Invalid dates are explicit abstentions;
  shared-subject conjunct attribution, bounded action-beat-to-quote association, and 1-2 word
  refrains with intervening progression have exact historical regressions.
- The additive hardened DAG policy rejects the auditor's ambient-global smuggling program
  before side effects, plus closures, stateful callables, hidden defaults, imports, and private
  frame access. Accepted functions are rebuilt in a minimal namespace. This is callable-level
  provenance hardening, not an OS sandbox; capability internals remain a trusted boundary.
- External-review finding 5 is mechanically closed for the blind-v2 path only. A planted-marker
  run proves the bundle indexes `ctext`, excludes raw `text` and original identifiers, contains
  only five selected TRAIN rows rather than all eight fixture rows, always excludes self, and
  rejects a held-out alias. Historical pre-v2 retrieval remains unchanged.
- The channel-faithful checker now has real L-channel data. Independent GLM-4.7 extractions were
  frozen for Math a66, Math a78, and peer-review a25 without pair polarity or labels. The two
  Math hybrid gates fail (a66 1/4 plus a mode fingerprint; a78 1/4, while its CODE gate is 1/1).
  Pre-selected peer-review a25 passes CODE 2/2 and HYBRID 4/4 at 100% L coverage; its genuinely
  field-dependent L contrast moves from a code-only tie to a +0.15 separation. The corpus
  discrimination gate is `NOT_RUN`, so these are probe-local results, not criterion-level
  certification.
- Every live contract probe now emits its own `SubrelationEvidence`. Peer a25 therefore records
  two `verifiable_only/code_native` and two `articulable_only/prompt_native` witnesses while the
  parent outcome remains deliberately null. This adds the granularity missing from the initial
  whole-criterion v2 vocabulary without replacing the census taxonomy.

## 6. Next confirmatory moves

1. If an API credential becomes available, run a bounded V8 source-addressed prompt transport
   smoke before any full corpus call. Compare prompt assertions with the corrected v9 code
   witnesses as peer channels; never substitute paper acceptance labels.
2. Implement the compact, one-relation a407 null/matched/mismatched augmentation design and the
   separately gated substitution lattice. Use a new confirmatory criterion for any program
   changes motivated after the a407 reference was opened.
3. Re-author the a12 symbolic relation in a fresh context if pristine blindness is required, or
   select another untouched math criterion. In either case, pass the construct adversary before
   heldout access and do not reuse the opened a144/a407 splits for confirmation.
4. For patents, separate examiner/oracle candidate injection from autonomous retrieval and
   certify claim-element links on a non-oracle candidate set.
5. Repair the duplicated transplant row in a new immutable consolidation artifact, then compare
   relation-matched execution against an equally deep mismatched operation and a null operation.
6. Accumulate several blind technical criteria before running one immutable,
   multiplicity-controlled certification batch. The rejected a144 candidate is ineligible for
   that batch. Keep construct-adversary acceptance separate from reference-reconstruction
   inference.

## 7. Verification commands

```bash
python -m unittest \
  methods.metric_seam.test_reconstruction_v2 \
  methods.metric_seam.battery.test_blind_reconstruction_v2 \
  methods.metric_seam.battery.test_evaluate_blind_v2 \
  methods.metric_seam.battery.test_contract_check_isomorphic \
  methods.metric_seam.battery.test_build_probe_extractions_v2 \
  methods.metric_seam.battery.test_dag_schema_enforced \
  methods.metric_seam.battery.test_dag_schema_hardened \
  methods.metric_seam.battery.test_certify_batch_v2 \
  methods.metric_seam.hybrids.test_ops_capability_v2 \
  methods.metric_seam.technical_replay.test_replay

pytest -q methods/metric_seam/science_claims_v2
python -m methods.metric_seam.battery.verify_retrieval_scope_v2 --check
python -m methods.metric_seam.technical_replay.evaluate --check
python methods/metric_seam/battery/blind_reconstruction_v2.py verify \
  --bundle outputs/metric_seam_pilot/reconstruction_v2/blind_math_a144_001/compiler_bundle.json
```
