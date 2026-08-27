# Metric-seam → paper: what's left before it's proven and useful

*2026-07-03. Status after the presentation push: 7 corpora surveyed (PR / math / patents /
code_review comments / code_review diffs / code_competition / pr_exec), 160+ metrics with 2-pass
reliability + attenuation ceilings + scope channels; hybrid evolution + certified gates on PR only;
reconstruction-R sweep complete (479 channels, GLM-5.2); two external anchors (competition verdict,
pr_exec accept/days_open) with skeptical audits already applied; theory consolidated in proposal
§5bis/§5ter; certificates.py 14/14 planted unit tests green. This note is the gap list between that
state and a paper section that survives review.*

Ordering principle: **prove the instrument → kill the named confounds → complete the frontier →
tie to the PO theory → write.** Items marked ⚠ need user sign-off before starting
(scope/cost/measurement-target decisions).

---

## W1. Prove the instrument (highest priority — everything downstream inherits from this)

1. **E-S1 planted end-to-end kill-switch.** Seed the pipeline with criteria of KNOWN placement and
   run the whole thing blind (survey → codegen rungs → hybrid evolution → gates → S1–S5 → S6
   placement), then score placement accuracy:
   - pure-code plant (e.g., deterministic structural predicate),
   - pure-LLM plant (paraphrase-robust semantic criterion with no finite surface signature),
   - known-mixed plant (code skeleton + one genuinely tacit field),
   - **the two planted op-type recoveries** (one evidence-op-dependent, one computation-op-dependent;
     pipeline must separate them via the stronger-executor-without-tool test).
   Why first: without it, "a105 is A-layer" is unfalsifiable-by-construction ("maybe your codegen is
   just weak"). certificates.py's planted tests are unit-level; this is the pipeline-level version.
   The proposal already names this as the gate to "proven" — it has simply never been run.
   Cost: mostly CPU + one small Gemma batch.

2. **Second judge family.** Every number in the project is currently Gemma-4-31B-only. Re-run the
   2-pass survey on PR + math with a second family (Llama-70B BF16 recipe known on sk3; or Claude
   batch via Max subagents; Qwen excluded for validity eval per standing feedback). Claims that must
   survive: codability ordering PR > math > CR ≈ patents; scoping deconfound direction; the A-layer
   calls (a105, lede-5Ws). ~3.5h GPU per family for 4 tasks at overnight scale; 2 tasks is less.

3. **Rung-3 bootstrap on every gate + n_test scale-up where unresolved.** a110 (P(gate)=.59) and
   a80 (.31) are still undecided at n=100; ~2,500 more Gemma verdicts ≈ 10 min GPU resolves them.
   Paper rule: no gate verdict is ever quoted as a point margin.

## W2. Kill the named confounds (each is currently a footnote a referee will pull on)

1. **Diff-native codegen round for code_review_diffs.** The 0.068→0.105 code-baseline drop is
   bounded below by comments-era OOD programs. Regenerate 3-rung programs against the diff
   representation (+ diff-aware canonicalization: per-file headers + sampled hunks inside the char
   budget — the current head5000/tail2500 prose window censors 279/300 diffs). Outcome either
   rescues codability (→ representation story, like a180's .30→.754) or leaves it low (→ genuine
   A-layer on code, a stronger claim). Cheap: Claude subagent codegen + CPU scoring.
2. **Competition native round + a135 gated refinement loop.** The 2 hand-written native probes
   showed the mechanical band goes V when executor matches representation (a180 → 77% of ceiling)
   but a135 exposed judge free parameters (python −.07 vs rel₁=.92). Run the actual train-split
   gated refinement loop on a135 (+ 2–3 more mechanical aspects): "does the loop close
   judge-construct gaps?" is a paper-grade question either way.
3. **Difficulty-matched, verdict-balanced CF×Python resample** (800K candidates available).
   Upgrades the substance/surface split from "one Bonferroni-proof aspect (a153)" to a solid
   panel — or kills it. Required before the chiasmus goes in the paper. NEVER pool across
   language/platform strata (standing rule).
4. **pr_exec train round for the 3 hybrids.** Blind-v0 hybrids lose to text baselines (a104 .178
   vs .417), which invites misreading of the (immune) op-marginal result. One PR-wave-2-style
   feedback round removes the distraction.
5. **Mock→real F2P/P2F swap** when the other agent's docker/test machinery lands. Re-run
   eval_pr_exec with real `test_transition`; the mock run then reads as a pre-registration and
   replication of the a128 (+.10 certified) / a67 (certified-harmful) op-marginals. Do NOT build
   the machinery ourselves (standing directive).

## W3. Complete the frontier (the paper's money figure)

1. ⚠ **Hybrid evolution + certified gates beyond PR.** The S2 codability bracket has both arms
   (description-compiled floor AND evolved-certified upper bound) only on press releases. The
   cross-task claim "math is A-dominant, patents evidence-dominant" is currently a
   lower-bound-only claim. Needed: math fleet (AST/sympy computation ops), code_review_diffs fleet
   (diff-native, after W2.1), patents fleet (blocked on a prior-art retrieval corpus — scope
   decision: ship patents as "evidence-op-dominant, upper arm out of scope" or build the corpus).
   Cost driver: ~40 improver agents per task — sign-off required.
2. ⚠ **Add the taste pole: CW seam survey.** Bank exists (7,699 rubrics local). Completes the
   V/A/Taste frontier (E-S3 as proposed: math high-V, CW low-V, patents evidence-dominant) and
   links the seam section to the two-faces/codability results already in the paper. Standard
   overnight recipe (40 aspects × 250 × 2 passes + scope + code rungs).
3. **Consolidated cross-task figure.** Per corpus: bracket [floor, certified-upper]/ceiling, bars
   colored by dominant op type, recon-R overlaid as the independent articulability axis. Most
   ingredients exist in the notebook; the bracket upper arms are the missing data (item 1).

## W4. Theory tie-in (make "certified" mean something in the PO framework)

1. **Derive the two lemmas** currently asserted in §5bis: headroom T(m_ω)−R survives
   executor-agnostically; U₂ over Ω×{code,llm} with partition matroid + tightening ("migration
   confines the uncertified residual to the LLM share — the seam is where the certificate is
   tight"). Target: PO-note appendix.
2. **TVD-MI ↔ Spearman bridge.** §5ter flags the correspondence gap. Either a calibration slice
   (compute both on the same channels) or a formal direction-of-bound statement. Without it, seam
   certificates don't compose with the main paper's T/B_E machinery.
3. **Headroom T−R on ≥1 task** so seam numbers plug into the articulability axes (T lower-bounds
   M*; B_E upper). One task suffices for the paper.
4. (Cheaper, optional) **Recon robustness**: second recoverer family on a slice (⚠ GLM quota — be
   sparing; consider Claude batch), and resolve/report the 26 GLM-degenerate channels.

## W5. Write the section

1. **Fix the claim set** (draft, to be pruned): (i) prompt/code/tool placement is measurable and
   certifiable per criterion (S1–S6); (ii) mixedness is a spectrum whose LLM-touch share tracks
   A-ness; (iii) the evidence/computation op taxonomy is empirically decidable and op value is
   criterion-specific (a128 certified-helpful vs a67 certified-harmful, same op); (iv) chiasmus:
   codability ⊥ validity — most-codable aspects carry zero correctness signal, correctness-relevant
   core least codable, and A-channel out-signals V-channel for acceptance (two independent
   anchors); (v) reconstruction R is an independent operationalization of articulability that
   reproduces the codability ordering.
2. **Positioning paragraph** vs the agentic-workflow-optimization crowd (deep-research verified
   25/25; our cell = unsupervised fidelity objective + certificates + instrument-measurement
   purpose; never headline "reflective workflow evolution").
3. **Figures inventory**: 7-corpus codability chart, fleet anatomy (§3b), anchor/chiasmus panel,
   recon table, cross-task bracket figure (W3.3) — first four already live in the report notebook.

---

## Suggested sequence and cost profile

| step | items | cost | blocking? |
|---|---|---|---|
| 1 | W1.1 planted kill-switch | CPU + small Gemma batch | yes — credibility gate for everything |
| 2 | W2.1 diff-native codegen + W2.3 resample | subagents + CPU | kills the two loudest caveats |
| 3 | W1.2 second family (PR+math) + W1.3 gate resolution | few GPU-hours | replication requirement |
| 4 | ⚠ W3.1 hybrid fleets + W3.2 CW survey | agent fleets + 1 GPU overnight | needed for money figure |
| 5 | W4 lemmas + bridge; W5 writing | human/agent time, parallel with 3–4 | paper integration |

Open sign-offs carried from earlier sessions: scoped-gate definition (open since v1); hybrid-fleet
cost; CW as a new seam task; any GLM quota spend.
