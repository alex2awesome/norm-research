# Paper section draft — "The metric seam: certifying how much of a judged criterion is code"

*R5 deliverable, drafted 2026-07-05 overnight. Sources: results note
(notes/2026-07-01__metric-seam-pilot-results.md §R1–R7.2c), certificate lemmas note
(2026-07-03), binding/provenance theory note (2026-07-04), CAM (cam_profile.py), money figure.
Wording discipline: all gate claims are bootstrap-Spearman statements (Gap-3 rule); every
certificate is stated as a (criterion, judge-family, executor-level) property; nothing below
is label-aware. Transport-test numbers land tonight — slots marked ◻T.*

---

## 1. Setup

A *criterion* is one aspect of quality that a judge J (here Gemma-4-31B, two independent
prompt forms) scores 0–10 on documents x. The measurement target is the judge's expected
verdict M̄(x); two-pass reliability rel₁ yields an attenuation ceiling c(rel₁, K=2), and every
channel m is read as the ceiling-normalized floor **r̃ = clip₀₁(ρ_S(m, M̄)/c)** on held-out
items. An *implementation* is a program over an explicit executor level:

- **E₁** — code + computation ops (normalize, dates, sentence stats, TF-IDF retrieval);
- **E₂** — E₁ + at most two LLM-extracted fields (bounded borrowed judgment);
- **E₃** — E₂ + evidence ops (precomputed world-state lookups: test executions, prior art).

The pipeline has two arms. The *description-compiled floor*: three code flavors generated
from the criterion description alone. The *evolved-certified arm*: one blind improver round
(worst-residual pack, anti-overfit contract) producing an E₂ hybrid, gated at
G1: P[ρ_hyb ≥ max(ρ_base + .10, .60)] ≥ .5 under paired item bootstrap (B=2000), held-out,
baseline flavor frozen on train.

**Instrument validation (kill-switch).** On planted criteria with known ceilings the full
pipeline certifies 0 false positives and one blind round recovers 86–90% of the plantable
ceiling. This calibrates search: an uncertified criterion is not "not found because we barely
looked." Scoped certificates follow the four-condition rule (criterion-independent scope
predicate, frozen before gating, applied symmetrically, stamped with coverage).

**Task-level statistic.** CAM = mean r̃ of the best materialized implementation over
judge-measurable criteria; reported with its survival curve and frac ≥ .5 / ≥ .8. CAM is
one-sided and monotone under search: 1 − CAM is *uncertified residual*, never proven-tacit.

## 2. Results

**(C1) Articulable mass is real and large where communities have compiled their norms.**
Press releases: CAM .369 → .697 evolved (95% of criteria ≥ half their ceiling; 12/20
gate-certified). Title VII case facts: CAM .372 → .621 (80% ≥ half-ceiling; 9/20
gate-certified, held-out) — doctrine behaves as *institutionally pre-compiled* norm, and
the two institutional domains TIE at the description-compiled floor (.372 vs .369). What
certifies in law is exactly doctrine's *elements layer*: protected-activity presence,
neutral-practice identification, temporal order of discipline vs protected activity,
procedural exhaustion/timeliness (ρ .825 from literal 180/300-day arithmetic); what stays
uncertified is doctrinal gestalt (constructive discharge, cat's-paw). A methods lesson
rode along: two survey-zero criteria diagnosed as "evidence missing from X" (same-actor
gap, 15-employee threshold) were *recovered* by one improver round (−.05 → +.69,
−.04 → +.47) — survey floors cannot distinguish extraction shortfall from absent
evidence; the evolved arm can.

**(C2) Taste-pole tasks have thin certified mass that one search round cannot inflate.**
Creative writing CAM .131 → .466 (5/36 certified), humor .120 → .351 (4/31), math .173 → .377
(0/34). The cross-domain spectrum — **PR ≈ law ≫ math > CW ≈ humor at the compiled floor;
PR > law > CW > math > humor certified** — orders domains by how much norm-compilation
their source communities did, the paper's anthropological throughline.

**(C3) What certifies in taste domains is the compliance shell, not the craft core.** Humor:
representation ethics r̃ .78, platform standards .67, cross-cultural translatability and
topical anchoring flip *negative* code floors (−.18, −.10) to +.61 with two borrowed fields;
timing, storytelling, and incongruity-theory criteria stall or regress at judge rel1 .85–.94.

**(C4) The uncertified residual in taste tasks is genuine, not search shortfall.** A second
refinement round (h1) run with the same protocol that closes 86–90% of planted gaps promoted
0/10 near-gate criteria; regressions follow one pattern, replicated across math, CW, and
humor: improvers that *replace* the specified construct with a borrowed pointer lose;
winners *add* one discriminating field to real code signal.

**(C5) Certificates are executor- and judge-family-indexed, and gates can fail for
bar-artifact reasons.** Gates are stated per (criterion, judge-family); the absolute .60 bar
exceeds the attenuation ceiling of low-reliability criteria — 19/31 humor hybrids beat their
baselines at P ≥ .95 yet fail G1. The survival curves, not the certified counts, carry the
comparative story.

**(C6) Evidence ops obey a level-matching theorem.** For a doc-only judge, I(M̄(X); Z|X) = 0:
world-state ops cannot improve reconstruction of a target that never saw the world. Measured
(patents prior-art op, Null-ablation, held-out): marginal P = .03 (op hurts), .24, .34, and a
sign-flip artifact at .96 — a forced null predicted before the run. Evidence-dominant
criteria are thereby *unreconstructible from X by any executor, including the judge*; their
honest treatment requires a Z-aware target M̄(x, Z) (future work, needs sign-off). The
TVD-headroom check on PR runs both legs in one f-divergence: 0/29 DPI violations, and the
TVD stack independently reproduces the Spearman gate ordering.

**(C7) Coded vs articulated is two axes, not one: binding rigidity × provenance.** Libraries
leave code epistemically code (rigid, conformance-checkable reference; certificates transport
across the conformance class); learned components (LDA, CE) are *selected* — frozen binding,
ostensive semantics; LLM fields are *enculturated* — frozen snapshots of community practice,
certificates artifact-bound. Stochasticity is orthogonal (it lives in the reliability layer
the ceilings normalize). The ≤2-field contract is a budget on borrowed meaning whose marginal
is ablation-measurable. **The interpreter-swap transport test measures artifact-boundness
directly** (Llama-3.3-70B re-extraction of all 56,750 field prompts; frozen programs, frozen
splits): pooled over 120 criteria, certificate loss tracks the program's field marginal
(Spearman .59; per task .51–.67) — the borrowed content is retrieval-from-the-interpreter —
while the median transport ratio is .30: ~70% of borrowed signal survives the family swap,
so the enculturated payload is mostly *shared* training culture, not checkpoint idiosyncrasy.
Binding is nonetheless graded with real consequences: 34/120 criteria degrade at P ≥ .95 and
3 of PR's 12 certified gates fail under swap (worst: "human, humble spokesperson tone,"
ρ .813→.591, ratio ≈ 1 — fully extractor-bound, and fittingly the most taste-flavored
certified criterion); 6/120 genuinely *improve* (shared construct, better extractor; 2 further P=0 cases are inert-field ties).
Certificates are therefore stamped (criterion, judge-family, executor level, field-extractor
family), with transport_ratio as the extractor-boundness coordinate. **A third family
(Qwen3.5-122B) replicates both predictions** (pooled n=120: fm↔loss ρ .38, median ratio
.230 — and the larger extractor loses less on every task: 21 vs 35 degradations at
P≥.95). Boundness is criterion-level, not pair-level: per-criterion transport ratios
correlate across the two swaps (Spearman .295, n=101, positive within each task), and 11
criteria degrade under BOTH swaps — headed by the humble-spokesperson-tone criterion
(ratios ≈ 1 for both families), a two-family replication that its certified content is
judge-family-bound. Three criteria *improve* under both swaps (shared construct, weak
original extractor) — the improvement direction transports too.

## 3. Figure and table pointers

- Money figure (5-task brackets + CAM survival + survey floors w/ op-type diagnoses):
  outputs/metric_seam_pilot/figures/money_bracket.{png,pdf}; notebook §8.
- CAM table: cam_profile.json. Gate tables: hybrid_gate_report.json per task (incl.
  legal_title_vii) + hybrid_eval_v2.json (PR). Headroom: headroom_pr.json. Bridge:
  bridge_calibration.json. Patents op: tasks/patents_pa/pa_eval.json.
  Transport: transport_eval.json in v2/ and tasks/{creative_writing,math,humor,patents_pa}/
  (patents_pa is a design-null: verbatim-grounding checks render its fields inert, fm ≈ 0).

## 4. Threats and disciplines (for the limitations paragraph)

Survey-grade floors are full-sample (flagged; fleet tasks are held-out). Judge families are
not pooled; certificates do not transport across them (that is a finding, not a bug — C7).
Spearman gates never invoke Lemma A2; near-ceiling reads get Pearson companions. a252 (humor)
and a34 (patents_pa) gates are underpowered by judge NA-coverage. Legal is one domain
(title_vii); 12 more are drop-in. Nothing here uses outcome labels: anchors appear only as
flagged descriptive correlations.
