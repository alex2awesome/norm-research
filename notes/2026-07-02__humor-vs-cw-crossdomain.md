# Humor vs creative-writing — first cross-domain contrast (BOTH faces)

*Humor Day-0 §12.6 certificate + Face-2 decompression grid COMPLETE 2026-07-02 (sweep hit 60/60 at
15:45, grid auto-ran 15:47–15:58 via `humor_grid_chain.sh`). Both domains, SAME instrument config
(8B executor, 3×GLM-4.7 families, 300 probes, `--orbit-target 4`, `--form-invariance-n 12`; grid:
70B writer, 1B/3B/8B readers, 6 type rungs, 3 forms on verbal rungs). Catalog gate passed at 60.
Artifacts: `outputs/r3_humor/_log/cert_humor.json` (41-ckpt snapshot preserved as
`cert_humor_41snapshot.json`), `outputs/r3_humor/grid_humor_v1/report.json`,
`_log/audit_humor_60_full.json`. Executed summary notebook (tables/plots/theory/examples):
`notebooks/2026-07-02__two-faces-results-summary.ipynb` (self-contained, local data snapshot in
`notebooks/data/two_faces_20260702/`). All numbers below are AUDITED (audit_certificate: D1 H_M≥0.15
degenerate filter; D2 reference-executor(8B) excluded from grid gaps).*

## Face 1: humor rubrics are markedly LESS form-fragile than creative-writing rubrics

| domain | n (non-degen) | FORM-DOM | UNDERSAMPLED | CODIFIABLE | form gate PASS | median OPT_Ω/H |
|---|---|---|---|---|---|---|
| creative-writing (8B_v2) | 45 | 38 | 5 | 2 | 7 (16%) | ~0.50–0.90 |
| **humor (@60)** | 54 | 35 | 15 | **4** | **19 (35%)** | **0.742** |

(Raw pre-filter: humor 36/19/5 of 60, 6 degenerate; CW 39/5/2 of 46, 1 degenerate. The gate-pass
contrast was 36% vs 16% at the 41-ckpt snapshot — stable at 35% vs 16% at 60. The D1 filter again
caught the same spurious CODIFIABLE, "Audience strategy, positioning, and growth"; the 6 humor
degenerates are face-valid off-topic meta-metrics — "editorial integrity", "cross-media
adaptability", "apology quality" — near-constant on joke corpora.)

- **Form fragility is domain-specific, not universal** (~2.2×, robust to every audit): the
  "telling won't hold still in words" effect that dominates CW is much weaker for humor.
- **More UNDERSAMPLED, more CODIFIABLE for humor**: with fewer metrics blocked at the form gate,
  the certificate reaches value verdicts more often; the ε-band being wide is a sampling problem
  (curable), not a fragility wall.
- **Heads are strong in both** (humor median OPT_Ω/H 0.742): value is head-concentrated in both
  domains — pointing away from deep-tail tacitness, toward "codifiable once form/sampling handled."

### Mechanism hypothesis for the form-gate contrast (written down 2026-07-02 PM, user-requested)

**What the gate actually varies:** the instrument-seat orbit is largely NON-CONTENT presentational
edits — boilerplate wrap, statement→question, clause reorder, suffix append (form_decompose per-
transformation records) — with the judge's output protocol (free-text vs JSON, chat template) held
constant everywhere. So the finding is: *under the same presentational perturbations on the same
executor, CW criterion readouts flip ~2.2× more often than humor criterion readouts.*

**Hypothesis (boundary distance):** a criterion is a linguistic address for a decision function.
Concepts with a crisp mechanical core (setup/incongruity/target/timing) yield DECISIVE per-item
readouts — far from the executor's decision boundary — so presentational noise cannot tip them;
paraphrase moves the address, not the function. CW's evaluative concepts yield boundary-hugging
readouts, so the same noise flips them: part of the meaning lives in the wording. Supporting
evidence: form_decompose's `boundary_dist_flipped < boundary_dist_stable` (flipped items sit near
the boundary), and the calibratable main effect (uniform strictness shift, ~52%, main_effect_share
0.79–0.92 per transformation) is REMOVED before the gate — what remains is item×form entanglement.
**Conclusion as stated for the record:** form-resistance operationalizes "this concept is
well-codified in language *for this executor class*" — it is a domain-times-executor property, not
a pure model property; crispness is executor-relative (training-distribution-dependent), which is
why the claim is indexed to E and why cross-family replication (Qwen/Gemma forminv, currently
absent) and eventually human arms are the robustness path. Open cheap checks: regress flip rate on
criterion surface statistics (length/concreteness); add an output-protocol orbit (JSON vs
free-text) as a new instrument arm.

## Face 2: the decompression curves (clean 3B−1B reader gaps, 8B excluded as reference)

| rung | humor gap | CW gap | humor span_R²(3B) | CW span_R²(3B) |
|---|---|---|---|---|
| name | +.045 | +.060 | .563 | .442 |
| definition | +.104 | +.133 | .577 | .483 |
| explanation | +.091 | +.139 | .620 | .499 |
| full_rubric | **+.019** | **+.121** | .569 | .449 |
| exemplars | **+.091** | **−.006** | .394 | .211 |
| dossier | +.078 | +.106 | .472 | .392 |

Shared signature (BOTH domains): the reader gap opens at the index→content transition (name →
definition/explanation) — the core compression→capability trade replicates. 1B floors: humor 1B is
~chance everywhere (.50–.57); explanation is the peak 3B rung in both (.694/.692).

**Prediction scorecard** (predictions registered in the 41-ckpt version of this note):
1. "Humor's name→explanation gap should be SMALLER than CW's (more of humor is statable)" —
   **REFUTED**: unpacking benefit (gap_expl − gap_name) is +.089 humor vs +.072 CW. Verbal
   unpacking helps the weak reader at least as much in humor.
2. "Humor should have fewer out-of-span rungs" — **SUPPORTED**: span_R² higher in humor at every
   rung (e.g. explanation .620 vs .499) — humor rung judges are more an assembly of known species.

**The two NEW cross-domain contrasts (bigger than the predicted one):**
- **full_rubric "collapse" in humor is a SMALL-READER artifact** (⚠ CORRECTED 2026-07-02 PM when the
  70B reader landed). The 3B−1B gap at full_rubric IS small (+.019 vs CW +.121; humor 3B rubric
  .583 ≪ its own explanation .716). But the 70B reads humor's full rubric at **.742 — the single
  highest cell in the humor table** (70B−1B +.196, 70B−3B +.090). So the checklist does NOT dilute
  the signal; it is the *richest* channel and simply demands capacity the 3B lacks. "Distilled
  explanation beats the checklist" holds for weak readers and **reverses** for strong ones. The
  3B−1B number stands; its interpretation is capacity-gated, not content-dilution.
- **Exemplars transmit in humor but not CW** (3B−1B +.091 vs −.006, SAME k=2/400-char instrument).
  This reframes the CW exemplar caveat: the instrument isn't inherently broken — joke mechanics
  are visible in short excerpts; CW taste is not. Ostension works for humor at k=2; CW would need
  bigger k / longer excerpts (or is genuinely show-resistant at this grain). (70B also reads humor
  exemplars at .688, 70B−1B +.183 — showing scales with capacity too.)

**70B ladder (real dynamic range, 8B = reference excluded):** name-rung recovery climbs
monotonically with reader size in BOTH domains — humor .523→.565→.676, CW .580→.635→.675 — the
compressed-pointer / "strong reads short, weak needs unpacked" hypothesis CONFIRMED; the 70B's
name→explanation unpacking benefit is tiny (+.046 humor, +.005 CW) because it already decodes the
pointer. CW dynamic range SATURATES at 3B (70B−3B ≈0 or negative on verbal rungs: def −.032, expl
−.046 — the 3B already extracts what the 70B does); humor stays graded to 70B (70B−3B positive on
5/6 rungs). Two capacity regimes: CW verbal content is all-or-nothing at a LOW bar; humor rewards
capacity all the way up. Full ladder table + plots: notebook §3b.

Coherent overall story: humor is the more *articulable* domain on both faces — words hold still
(form gate 2.2×), the species basis spans more of each rung judge (span_R²), and even tiny
ostension works — while CW's content survives only through careful verbal articulation and
resists both paraphrase and small-k showing.

## Caveats

Single executor family (Llama) for the grid readers; 8B reference-executor targets; single
instrument family (GLM proposers); humor exemplar/dossier rungs share the CW dossier caveat
(dossier includes the exemplar block — rebuild-without-exemplars still pending). 70B-as-reader
(real dynamic range) is queued for tonight on GPU7 (Task #8).

Related: [[project_cw_grid_v1_results]], [[two-faces-theory]] (Face 1 = census read, Face 2 =
decompression), `2026-07-02__cw-grid-v1-results.md`, `2026-07-02__70b-rescore-semantics-audit.md`,
`2026-07-01__form-effects-control-plan.md`.
