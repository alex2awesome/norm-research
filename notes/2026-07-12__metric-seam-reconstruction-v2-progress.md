# Metric-seam reconstruction v2 — technical progress and claim envelope (2026-07-12)

This note records the additive v2 work performed after the independent verification review.
Historical programs, contracts, and reported outputs remain frozen. The new lane stays on the
unsupervised reconstruction objective: the old LLM judgement is a frozen reference for
isomorphism, never a newly supervised external target.

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
replayable certificate on the exact cases where code and the frozen LLM reference disagree.
Uncertified disagreement remains reference divergence.

## 2. Instrument changes

- `methods/metric_seam/reconstruction_v2.py` now provides typed prompt/code/isomorphism axes,
  orthogonal historical provenance and current pipeline-selection status, joint outcomes, and
  executable claim permissions. `may_claim_tacitness` is always false.
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
`methods/metric_seam/science_claims_v2/articulability_prompt.json` but has not been run. The
current science result is therefore code verifiability and evidence-surface extension, not a
prompt/code isomorphism result.

### 3.4 Seeded Math, Patents, and technical replay

Prior expensive artifacts are now `pipeline_status=selected` with
`selection_mode=retrospective_seed`; their original manual/mock/oracle/replay provenance remains
unchanged.

- Math a150: the SymPy checker distinguishes one valid from one wrong synthetic consequent,
  but reaches 0/20 real condition-to-operation licensing relations. This is a precise
  relation/corpus mismatch, not evidence that the criterion is tacit.
- Patents a34: on the same oracle-conditioned prior-art evidence surface, the evidence-aware
  hybrid reaches rho 0.745 versus 0.084 with the evidence op nulled (marginal +0.661). This is
  strong selected-pipeline utility and isomorphic reconstruction conditional on privileged
  evidence; it is not autonomous prior-art retrieval or pure-code verification.
- The seven-case technical catalog currently permits five selected-utility claims, two
  canonical code-verifiability claims, zero historical automatic-selection claims, zero
  canonical verifier-dominant disagreement extensions, and zero tacitness claims.

See `outputs/metric_seam_pilot/technical_replay_v2/REPORT.md`.

## 4. What claims are broader now

The earlier review mainly narrowed overstatements. The v2 design also licenses positive claims
that the old single-axis reconstruction report hid:

1. **Selected machinery can be useful without being historically agentic.** Provenance and
   experimental pipeline role are orthogonal.
2. **Deep code can add a new evidence surface even without matching an LLM ranking.** Stored
   certificates from prior repository execution and current full-paper cross-section checking
   are examples; the current legacy-code replay itself performs classification, not execution.
3. **Verifiability survives unavailable isomorphism.** A valid code certificate remains a
   verifiability result when no channel-matched frozen LLM reference exists; it is reported as
   `verifiable_only`, not erased as unresolved.
4. **An authored seam annotation is a hypothesis, not a result.** The blind a144 program's 3/4
   off-label probe separations challenge the blanket all-L allocation, but the independent
   adversary rejects this particular executable witness. The allocation remains open rather
   than either proven L-only or successfully moved to code.
5. **Failure localizes the search boundary.** Zero SymPy coverage and low blind-a144 rho locate
   capability/representation/program-class limits; neither supports tacitness.

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

1. Run the frozen full-paper prompt counterpart on the exact same abstract+body representation;
   compare prompt/code/hybrid relations without substituting accept/reject labels.
2. Start a new blind hybrid compiler run on an untouched technical criterion and split only
   after prompt-result provenance is exactly bound at preparation time. Do not reuse the now
   opened a144 held-out split for confirmation.
3. For patents, separate examiner/oracle candidate injection from autonomous retrieval and
   certify claim-element links on a non-oracle candidate set.
4. Repair the duplicated transplant row in a new immutable consolidation artifact, then compare
   relation-matched execution against an equally deep mismatched operation and a null operation.
5. Accumulate several blind technical criteria before running one immutable,
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
