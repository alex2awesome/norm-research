# Tacitness Battery pre-registration — convergence structure + route signatures

Date frozen: 2026-07-23, after W0 exploratory (3 domains), before any W1–W3 confirmatory
pass. Body frozen; changes via dated addenda only. Companions: battery plan (approved
2026-07-22), operationalization catalog, mechanism catalog, learning-approaches discussion.

## Frozen instruments
- Registry + probes: `methods/tacit_channels/battery/{registry,profile,gates}.py` +
  `probes/*.py` at the shas recorded in `outputs/tacit_channels/battery_prereg_shas.json`
  (generated alongside this note). Primary-statistic map = `profile.PRIMARY` as of same shas
  (incl. the SCAL-2 divergence-slope fix; the gain-level duplicate is permanently excluded).
- Store: per-domain parquet (`battery_<domain>_<tag>.parquet`); append-only; w0_v1/w0_v2
  artifacts retained as exploratory history (w0_v1 PC1=.35 is dup-inflated — never quote).
- Slices: confirmatory v1 slice = Qwen-7B/14B × humor (battery plan decision); n&c + math as
  replication slices; Llama/Gemma as family-generalization slices (exploratory until stated).

## Frozen predictions (confirmatory battery)

**Structure (from W0 exploratory, now to be confirmed with CIs):**
- P-B1 **No general factor:** PC1 share of the probe-convergence matrix < .50 in every
  domain (W0: .30/.29/.36). Failure = tacitness is closer to unidimensional than W0 implied.
- P-B2 **Metacognitive block persists:** rank-agreement(P-STAT-1, P-GEN-1) ≥ +.50 (W0: +.81).
- P-B3 **Articulation-resistance block persists:** rank-agreement(P-CHAN-core, P-SCAL-1)
  ≥ +.50 (W0: +.67).
- P-B4 **Cross-block independence:** |rank-agreement| < .35 between block representatives
  (P-CHAN-core vs P-STAT-1; W0: +.21). Jointly, P-B1–P-B4 = "tacitness is ≥3-dimensional."
- P-B5 **Proxy validity:** W1 verbalized-confidence STAT-1 correlates ≥ +.40 with the
  log-odds v0 proxy across constructs (else the free proxy is retired and W0 STAT-1 rows
  are flagged).
- P-B7 **Attenuation account:** CEIL-1 × GT-1 coupling (W0 +.54) drops by ≥ one-third after
  reliability correction of cap_oos (else the coupling is substantive, not artifact).

**Route signatures (direction frozen now; instantiated when arms exist):**
- P-B6 **Route-signature hypothesis:** at matched trained-construct ρ, tacit-route arms
  (exposure-only M16, suppression M15, §5.1 distal-selection, outcome-corpus M63) score
  higher on the tacit-profile composite {unstatability (multi-method), token-freeness
  (CoT-delta≈0), pressure-robustness, exclusion-leakage} than explicit-route arms
  (articulation, coaching-corrections, rationale-augmented distillation). Null: profile is
  content-determined; route washes out. Either outcome is a headline; the direction above is
  the behaviorist bet.
- P-B8 **Plain-distillation position:** label-only distillation (current channel B) sits
  BETWEEN the route poles on the composite (it is observational-but-unrepresentational).

## Analysis discipline
Rank statistics only; cluster bootstrap over constructs for every CI; per-domain analyses
never pooled across domains (feedback_same_family_scaling analog); every judge pass
anchor-gated; acceptance-test freshness required for any new scoring path; multi-method gate
before any "unstatable" cell; Turner-safe verdict phrasing; tier stamps on all tacit
verdicts. Exploratory rows carry `run_tag` labels distinct from confirmatory (`c1_*`).

## Stop rules
Anchor failure → batch discarded. Acceptance failure → no scoring. P-B5 failure → STAT-1
v0 rows quarantined from confirmatory profiles. Reliability floor: constructs with target
form-consistency < .30 excluded from slope claims (reported separately).

---

## Dated addenda

### 2026-07-23 — IRT analysis plan registered (user suggestion), BEFORE confirmatory runs

Three models join the analysis plan (registered now; confirmatory versions run on the
prereg'd slices):
- **Model A (dimensionality):** MIRT/graded-response over the construct × probe profile
  matrix; P-B1-IRT = a 3-factor model beats 1-factor by ΔBIC > 10 (formal version of P-B1;
  measurement error absorbed into discriminations; per-construct latent coordinates + SEs).
- **Model B (ladder scale):** executors = persons (rungs, adapters, checkpoints), constructs
  = items; 2PL/graded on rescue/gain. Deliverables: per-construct articulation-difficulty b,
  per-executor ability θ, Wright map. **Scaling-tacit-IRT definition: b unreachable at
  measured θ range OR discrimination a ≈ 0** (the de-relativized tacitness in IRT form).
- **Model C (channel-DIF):** DIF between channel groups at matched θ; DIF-flagged constructs
  = channel-tacit AT ANY ABILITY (fixes the ability-level confound in raw double
  differences). Persons: rung series (articulation channel) + adapter/checkpoint series
  (weight channel).
v0s are exploratory and labeled; confirmatory model comparisons follow the body's
discipline (cluster bootstrap over constructs, per-domain, never pooled).

### 2026-07-23 — interpretation clarification for P-B1 (user question)

PC1 < .50 is a claim about the DIMENSIONALITY of tacitness measurements, NOT about the LEVEL
of tacitness in a domain. PC1 = share of the probes' construct-ranking variance explained by
one common factor: high PC1 → probes agree which constructs are tacit → one underlying
property (tacitness would be temperature-like; one score suffices). Low PC1 → probes
disagree → "tacit" conflates several distinct properties (athleticism-like: real, measurable,
weakly correlated components; no g). LEVEL is read from statistic VALUES (e.g., humor@7B:
83/90 caps < .70 = high articulation-resistance level) — orthogonal to PC1. Noise
alternative (garbage probes also decorrelate) is excluded ONLY by the joint package: P-B2/B3
within-block ≥ .50 (convergent validity — probes can agree strongly) + P-B1/B4 across-block
(discriminant validity) = multitrait-multimethod logic. Consequence: "X is tacit" is
under-specified without naming the SENSE; the measurement object is the profile vector.

### 2026-07-23 — W1a execution declaration (frozen BEFORE any W1 scoring)

**Scope:** W1 splits into W1a (single-stage variants — this declaration) and W1b
(two-stage reason-first/confidence, elicitation passes, exemplar arm — declared separately
when built). W1a instruments, pinned here because they postdate the body's sha freeze:
- `battery/passes.py` sha256 09427c79930fe72b… (wording constants
  EXCLUSION_PREFIX / NEGATED_WRAPPER / COMPOSED_WRAPPER / HOLISTIC_PROMPT are frozen
  instrument text)
- `battery/run_variant_pass.py` sha256 2a917e6bf8204f91…
- composed pairs `outputs/tacit_channels/battery/w1_composed_pairs.json` sha256
  5f0e9cb4d7eb8464… — rule: sha256(cell_id)-ordered consecutive
  disjoint pairing; 8 A×A pairs (EXP-GTK-1 trained cells; weight-side composition) + 12
  non-A pairs (articulation-side).

**Engine loads (cost law = O(engine loads)):** sk2 freed lane, sequential: (1) Qwen-7B base,
(2) Qwen-7B + real_n8192c adapter, (3) Qwen-14B base. Variants tf+exclusion+negated+composed
(+holistic on base loads); base loads = all 90 humor cells, adapter load = A∪B1 (36 cells) +
the 8 A×A pairs. All 3 forms; all 90 items scored; analysis restricts to item-half-2 wherever
an adapter-trained cell is involved. (4) 72B target pass (composed 20 pairs + negated 90 +
holistic) runs on sk3 GPU0 — QUEUED behind the prompt-optimality live server (not killable);
scheduling constraint recorded, not a design change.

**Acceptance gate (new scoring path):** per base engine load, runner tf rows vs the frozen
executor grid, per-row ρ ≥ .999 on a 5-cell smoke — tf content IS the bank name prompt, so
this is exact-path equivalence. The adapter load inherits the 7B runner gate + the v1
LoRA-path acceptance (ρ .99998); no separate gate.

**Readout references, two versions:** until the target pass lands, negation reference =
rank-REVERSAL of the target's name vector and composition v0 reference = elementwise min of
the member target name vectors (knowing-using logical-consistency readout); when the 72B
pass lands, v1 references = the target's OWN negated/composed vectors. Both reported; v1 is
the headline. Holistic readout = unnamed-residual R² of the holistic vector on the domain's
90 named target policies (Act-3 instrument, exploratory).

**W1a directional predictions (frozen now, before any scoring; join the P-B6 composite):**
- P-W1a-1 (exclusion): leak-through under the exclusion instruction is HIGHER for
  weight-installed policy (7B+real on A-cells, half-2) than for the same cells under the
  base model's best articulation — installed-without-symbol resists instructed suppression.
- P-W1a-2 (negation): the knowing-using NOT gap (tf ρ minus negated-vs-reversed-reference ρ)
  is LARGER for adapter-installed cells than articulation-carried cells.
- P-W1a-3 (composition): at matched member-cell ρ, articulation-side pairs compose BETTER
  than weight-side A×A pairs (symbols compose, practice blends).
- P-W1a-4 (holistic, exploratory): executor unnamed-residual share shrinks with rung
  (7B > 14B); target share smallest.
Failures reported as such; these do not alter the frozen P-B1–P-B8.

**W1a instrument correction (2026-07-23, pre-data):** run_variant_pass.py sha updated to c07d9084d6c7dbd3… — argparse fix only (--out-dir now optional in acceptance mode; the wrapper's acceptance call crashed before any scoring). No statistic or wording changed; no data existed at fix time.

### 2026-07-23 — W1b execution declaration (two-stage passes; frozen BEFORE any W1b scoring)

Instrument: `battery/run_reason_first_pass.py` sha256 404314dbe189c22a… (stage wordings REASON_FIRST_INSTR
/ CONFIDENCE_QUESTION already frozen in passes.py). Execution decisions:
- **Canonical form only** for both two-stage variants (cost; the interference and confidence
  contrasts are within-form — the adverse-over-forms discipline applies to ρ estimands, not
  to these deltas).
- **Confidence answers derived from the W1a tf grids** (answer = YES iff p_yes ≥ .5); no
  second judgment pass — the confidence elicitation is anchored to the actually-scored
  behavior, one generation per (row, item), 0-100 integer parse.
- Generation budget 160 tokens (reason-first), temperature 0.0 both stages; rationales
  persisted (jsonl.gz) for articulation-quality analyses.
- Engine loads: 7B base (90 cells) → 7B+real_n8192c (A∪B1 36) → 14B base (90); chained
  behind W1a with a hard gate on the "W1A ALL DONE" marker.
- **P-W1b-1 (direction, Beilock boundary condition):** forced reason-first HURTS the
  below-floor executor (7B) on high-tacit-profile cells (reason_first ρ < tf ρ) and is
  neutral-to-positive at 14B — the fluency-mismatch interaction.
- **P-W1b-2 (instrument gate):** mean confidence parse rate ≥ .80 per engine load, else the
  verbalized-confidence instrument is VOID for that load (and P-B5 is evaluated only on
  valid loads). Gate, not a science prediction.

**W1a v0 outcome corrections (2026-07-23, post-first-readout; full readout =
notes/2026-07-23__battery-w1a-readout.md):** (1) P-W1a-1 was frozen with an unevaluable
comparison — the W1a pass scores name-arm exclusion only; articulation-arm exclusion rows
move to W1c; the computable adapter-vs-base contrast is reported with that caveat. (2)
P-W1a-3's frozen direction is NOT supported at v0 (weight-side composition retention ≈
articulation-side); recorded as-is, v1 verdict awaits the target-composed reference. (3)
P-W1a-4 has NO VERDICT: the holistic YES/NO elicitation floor-collapsed (mean p_yes ≤ .05)
— instrument defect; a graded/comparative holistic form must be declared before any
unnamed-share claim. (4) NEW instrument requirement for W1c: trivial-inversion
instruction-following g-control, to discriminate global SFT instruction-rigidity from
policy-specific automatism (the W1a exclusion gradient is confounded between the two:
adapter leak +.93 on trained AND non-trained cells alike).

**W1b outcomes (2026-07-23, later): P-B5 FAILED (+.032/−.284/−.729 vs bar +.40) → v0
log-odds proxy RETIRED; W0 STAT-1 rows quarantined from confirmatory profiles; P-B2 must be
re-scored with verbalized STAT-1. P-W1b-2 parse gate PASSED (1.0/.991/1.0). P-W1b-1 failed
as worded (interference rung-general). Full table in the W1a/W1b readout note.**

**Confidence-instrument scale-use gate (2026-07-23, prospective):** verbal-confidence loads
additionally require n_unique ≥ 8 and median within-cell std ≥ 5 (0-100 scale); parse-rate
alone (P-W1b-2) failed to catch scale degeneracy (7B base 94% constant "85"; 7B+real binary
{0,100}). Today's 7B verbal rows VOID by this gate; 14B rows VALID. P-B5 verdict unchanged.

### 2026-07-24 — Known-truth calibration program (instrument validation; user-directed)

Two-layer validation now standing for every battery statistic:

1. **Adversarial numeric tests** (`battery/stats.py` sha 2ed8989cad367bf1…,
   `tests/test_stats.py`): each statistic is attacked by synthetic agents of KNOWN profile
   — compliant-inverter vs rigid vs GENERIC-FACTOR agents for leak (the generic agent is
   the audit-#2 failure mode encoded as a permanent test), perfect-NOT vs NOT-ignorer,
   min-composer, span/degenerate holistic targets, constant/binary confidence matrices
   (both W1b degeneracies). Statistics now fail closed: holistic returns a VERDICT on
   degenerate targets; conf-acc flags constant confidence; leak always carries its
   cross-cell null.

2. **GLM known-truth calibration** (`battery/synthetic/` — constructs.py sha
   5483c295b7a2f417…): synthetic constructs with
   MECHANICAL oracles (E-tier: exclamation/length/animal/digit rules; G-tier authored;
   H-tier profile-only) × 40 fixed items, run through the REAL frozen instrument text on
   glm-4.7 (subscription endpoint, parsed YES/NO — prompt logic is the object under test;
   the logprob channel is separately acceptance-gated). Bars: E-tier tf accuracy ≥ .90 or
   the readout template itself is suspect; exclusion defect cost = acc(fixed wording) −
   acc(deployed wording) on E-tier — a DIRECT measurement of audit defect #1; negation and
   composition compliance vs NOT/AND oracles; confidence must clear the scale-use gate on
   E-tier where the stated answer is oracle-true. Cycle cadence: wordings iterate ONLY on
   synthetic data until E-tier calibrates clean; only then do wording changes get declared
   for real-construct use (W1c). Cycle-1 artifacts under
   outputs/tacit_channels/battery_calibration/.

### 2026-07-24 — Calibration CYCLE 1 RESULTS (glm-4.7, mechanical oracles; report =
### outputs/tacit_channels/battery_calibration/report_cycle1.json)

| instrument | E-tier verdict |
|---|---|
| tf template | VALID — 1.0/.92/1.0 on fair rules; the .325 miss is the word-count construct (LLMs can't count words) → E2 replaced by E5_qmark in cycle 2 |
| exclusion (deployed W1a wording) | DEFECT COST MEASURED: **+.24** (acc .59 vs .82 fixed); E1: .53→1.0, leak .48→.00 |
| exclusion (W1c fixed wording) | **VALIDATED — adopt for W1c** |
| negation (deployed wrapper) | **CONSTRUCT-DEPENDENT compliance even on a frontier judge: 1.0 (exclaim), .97 (digit) but .38 (animal), .15 (formal)** — the wrapper is ignored for semantically salient properties. Cycle 2 iterates wordings (explicit-mapping wrapper + direct-predicate upper bound). CAVEAT now attached to W1a not_gap magnitudes (ordering across configs still meaningful — same wording everywhere — but levels are wording-contaminated) |
| composed wrapper | VALIDATED — 1.0 AND-compliance on the clean pair; .70 on the pair containing the bad word-count member |
| verbalized confidence | VALID on a competent subject: scale-valid, 100/92 on oracle-true E-tier, and LOW exactly where GLM fails (52 on word-count) or has no criterion (31 charming) — instrument discriminates; subject-side degeneracy is what the scale gate catches |

Meta: one calibration cycle found one bad synthetic construct, validated two instrument
fixes, and exposed one NEW wording defect (negation) that no amount of real-data analysis
would have separated from subject incapacity. Cycle cadence continues until E-tier is
clean across all instruments.

### 2026-07-24 — Calibration CYCLE 2 RESULTS + W1c wording set CERTIFIED

Cycle 2 (glm-4.7, 864 calls; analyzer = battery/synthetic/analyze_cycle2.py):
- **Negation FIXED:** explicit-mapping wrapper (NEG_FX) calibrates .88-1.0 on ALL six
  oracled constructs (cycle-1 failures: animal .38→.88, formal .15→.90) and BEATS
  hand-negated predicates on the ambiguous register construct (.90 vs .73). Certified.
- E5_qmark replacement construct: tf 1.0, exclusion_fx 1.0. Second composed pair
  (animal∧digit): .979 — composition wrapper confirmed twice.
- **Graded holistic elicitation VALIDATED:** parse 1.0, ρ +.825 vs the known-mixture
  oracle, y std 2.7 (no floor collapse), full 0-10 scale use on the no-oracle construct
  (n_unique 10). The YES/NO holistic form is retired.
- **Estimand limitation surfaced (design constraint, not a bug):** span-DISCRIMINATION
  failed on the confounded corpus — R² on wrong-span predictors (.83) ≈ true-span (.77)
  because the corpus correlates them (excited items contain '!'). Consequence, binding on
  Act-3 claims: **unnamed-share = 1−R² is a SPAN-RELATIVE LOWER BOUND on unnamed
  structure; it cannot attribute to specific named members, and unnamed structure
  correlated with the named span is invisible to it.** Cycle-3 (optional): orthogonalized
  item corpus for discriminant validation.

**W1c wording set now ORACLE-CERTIFIED and frozen in passes.py (sha 260689d5b7eea68e…):**
EXCLUSION_FIXED_QUESTION, NEG_FX_WRAPPER + NEG_FX_QUESTION, HOLISTIC_GRADED_TEMPLATE.
W1a's deployed forms remain frozen for scored-grid provenance. W1c real-construct passes
may now be declared using only certified wordings.

---

## Addendum 2026-07-25 — adversarial test wave + Tier-1 hardening (pre-W1c, user-approved)

**Context.** GLM endpoint down → confidence-building moved to the unit layer per user
directive ("10 tests per battery arm, varied and adversarially constructed"). Seven
Sonnet agents wrote 10 adversarial tests per arm (suite 33 → 103, all green; mutation
check 4/4 killed both before and after the fixes below). Full hazard register:
notes/2026-07-25__battery-adversarial-test-hazards.md.

**Tier-1 fixes applied (user sign-off "please fix", 2026-07-25).** No wording constant
changed — passes.py stays at sha 260689d5 (wording-freeze test 459ce46f still binding).
No scored artifact is affected: W1a/W1b grids were produced under the prior shas and the
v0 tallies compute their statistics inline; these fixes bind on W1c and later.

1. run_variant_pass.py::run_acceptance — REJECTS any NaN per-row ρ (zero-variance row);
   previously `nan < floor` = False silently printed ACCEPTANCE PASSED on a fully
   degenerate scoring path.
2. run_reason_first_pass.py::tf_answers_from_grid — refuses non-finite tf scores with an
   informative ValueError; previously NaN ≥ .5 silently read as a confident "NO".
3. stats.py::leak_stats — empty cross_tf now returns leak_cross/leak_specific = None with
   n_cross=0 (fail-closed flag); previously leak_specific silently equalled the raw
   leak_self headline the module forbids.
4. stats.py::conf_acc_stats — constant AGREEMENT vector now returns conf_acc_corr=None +
   degenerate_agreement=True; previously an unguarded NaN float poisoned cross-cell means.
5. stats.py::holistic_residual — (a) rejects non-finite X up front (one NaN poisoned its
   whole z-scored column); (b) degeneracy guard also fires on empty masks and exactly-
   constant y (floor=0 bypass closed); std==floor>0 boundary semantics unchanged (strict
   <, certified by test); (c) unnamed_share clipped to [0,1] for reporting, oos_r2 stays
   raw (re-tallies of the 7B-base holistic row would report share 1.0 with oos_r2 −5.63
   instead of the nonsensical share 6.63).

**Sha re-record (provenance).** W1a ran under run_variant_pass.py c07d9084; W1b ran under
run_reason_first_pass.py 404314db (as declared). From 2026-07-25: stats.py bf7e4473,
run_variant_pass.py 39804497, run_reason_first_pass.py aeb57205. W1c and all future
passes/tallies MUST cite the new shas. Six documented-hazard tests flipped to assert the
fixed behavior (renamed accordingly); suite remains 103 tests, all green.

**Still open (Tier-2, unchanged by design):** parse_confidence biases (W1b provenance —
a v2 parser needs its own addendum before W1c's confidence rows, if re-elicited);
spearman single-NaN perturbation (channels/common.py — repo-wide primitive, separate
review); composition mode-string fallback; the two drifted reason-first splice
implementations; not_gap-requires-tf_rho>0 and the two unnamed-share estimand caveats
(span-relative + nonlinearity) remain binding interpretation constraints.
