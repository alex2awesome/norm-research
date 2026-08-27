# Metric-seam roadmap v2 — post-execution audit + what's left for the paper

*2026-07-04. Supersedes `notes/2026-07-03__metric-seam-paper-roadmap.md` (kept for the original
gap analysis). Two overnight execution rounds (07-03, 07-04) landed most of W1/W3; this note
audits what landed against that roadmap and re-orders the remainder. Ordering principle
unchanged: prove the instrument → kill the named confounds → complete the frontier → tie to PO
theory → write. The instrument is now proven, so the bottleneck has moved to **synthesis + theory
+ writing** — most remaining high-value items are CPU-only.*

---

## A. Audit: 07-03 roadmap status (verified on disk 2026-07-04)

| workstream | status | evidence |
|---|---|---|
| W1.1 planted kill-switch | ✅ DONE | `killswitch_report.json`, `h1_gate_boot.json`; 6/7 plants certify exactly as designed, 0 false certs in 14 arm×plant cells; h1 closed p901 (86% ceil, P=.59) + p907 (90%, P=.84); dpid fix flips p903 → CERTIFIED CODE+EVIDENCE_OP (.944 = 98% ceil); p904 = the one mis-designed plant (genuinely codable; pipeline correctly refuses MIXED) |
| W1.2 second judge family | ✅ DONE | `replication_llama/replication_report.json`; Llama-3.3-70B on verbatim prompts; codability-order replication ρ .93/.84/.79/.72 (PR-v1/v2/v3/math); disattenuated cross-judge ≈1.0 on v1 focus aspects; **a110 gate does NOT replicate** (Gemma P=.989 vs Llama P=0, same frozen everything) — real judge-dependence, new limitation |
| W1.3 bootstrap + n=500 gates | ✅ DONE | `expansion/gate_expansion_report.json`; a86 P=1.00, a110 P=.989 (Gemma), a80 scoped .93–.99 / full on-margin, a105 A-layer confirmed (P_gate=0, P_beats=.98) |
| W2.1–2.3 code-corpora confound kills | ⛔ HELD | standing directive: wait for the code-PR agent's work to land; no commits as of 07-04 |
| W2.4 pr_exec train round | ◻ actionable | not code-corpora-blocked; low priority (op-marginal result is immune; this only removes a misreading) |
| W2.5 mock→real F2P swap | ⛔ HELD | other agent owns dockers/tests |
| W3.1 hybrid fleets beyond PR | ✅ DONE (CW+math) | CW 37/37: 5 certified, median ρ .481 vs base .096 (+.327); math 35/35: 0 certified, median .369 vs .133 (+.196); diffs fleet blocked on W2.1; patents fleet blocked on prior-art corpus (⚠ scope) |
| W3.2 CW seam survey | ✅ DONE | taste pole: median rel .90 (highest), median ρ/ceiling .128 (lowest) |
| W3.3 consolidated bracket figure | ◻ **NOW UNBLOCKED** | both bracket arms exist on PR + CW + math; lower-arm-only on CR/patents |
| W4.1 lemmas | ◐ drafted, gaps | `2026-07-03__seam-certificate-lemmas.md`: A1/A2+B1/B2 proved under assumptions, 12 gaps — 2 paper-threatening (below) |
| W4.2 TVD-MI↔Spearman bridge | ◻ not started | calibration-slice data already exists |
| W4.3 headroom T−R on ≥1 task | ◻ not started | R exists (recon sweep); T needs the main-paper estimator on existing verdicts |
| W5 write the section | ◻ not started | claim set needs night-1/2 updates |

New results that change the claim set: (a) the instrument-validation claim is now earned
(kill-switch, zero false certs); (b) gates are judge-family-relative (a110) — certificates must be
stated as (criterion, judge-family) properties; (c) the codability ceiling tracks criterion CLASS:
CW hybrids close most of the effect-size gap (+.327 median) yet 5/37 clear the PR-calibrated
absolute floor, math sits between (0/35 clear, +.196, heterogeneous — LaTeX-hygiene criteria gain
.5+, weak tail regresses).

---

## B. Roadmap v2 (ordered; ★ = do next, ⚠ = needs user sign-off)

### R1. ★ The money figure (W3.3) — CPU only, all data on disk
Consolidated cross-task figure: per corpus, bracket [description-compiled floor,
evolved-certified upper] / attenuation ceiling; bars colored by dominant op type; recon-R
overlaid as the independent articulability axis. Tasks with both arms: PR, CW, math (+ pr_exec
partial). CR-comments/diffs/competition/patents ship as lower-arm-only with explicit "upper arm
held/blocked" annotation. This is the paper's central exhibit and it is now purely a synthesis
task → report notebook §8.

### R2. ★ Theory gaps that can invalidate quoted numbers (W4.1/W4.2) — CPU only
1. **Spearman-not-covered-by-A2**: the gates run Spearman but lemma A2 covers a different
   statistic. Either extend the lemma or re-state gate certificates as empirical-bootstrap-only
   (weaker but honest). Must resolve before W5 — it decides the *wording* of every gate claim.
2. **γ̂ anti-conservative → U₂ possibly invalid**: same treatment (fix or weaken).
3. **TVD-MI↔Spearman calibration slice**: compute both on the same channels (verdicts already on
   disk) → empirical correspondence curve. Unlocks composing seam certificates with the main
   paper's T/B_E machinery.

### R3. ★ Headroom T−R on one task (W4.3) — likely CPU on existing verdicts
Pick PR (richest: 2 judge families, n=500 gates, recon-R complete). T from the main-paper
transmission estimator on existing score matrices; R from recon sweep. One task suffices to plug
seam numbers into the articulability axes (T lower-bounds M*, B_E upper — standing direction of
bounds).

### R4. ⚠ h1 refinement round for near-gate aspects — Sonnet fleet + small Gemma batch
The protocol that closed p901/p907, applied to: math a132 (P=.46), a198 (.43), a42, a108, a144;
CW a153, a117, a225, a45, a261 (P .15–.37). ~10 improver agents (Sonnet per tiering rule) + one
small field-extraction batch + CPU eval. Value: tests "one more round reaches X% of ceiling" as a
*general* claim, not just on plants; may move CW 5/37→~8/37 and math 0/35→2/35. Cost is the
sign-off item. Optional extension: a110 Llama-side h1 (or just write a110 as the judge-dependence
limitation — recommended default).

### R5. ★ Write the section (W5) — can start immediately, parallel with R1–R3
Updated claim set: (i) placement is measurable and certifiable per criterion, **and the pipeline
is validated end-to-end on planted ground truth** (6/7, 0 false certs, misses conservative);
(ii) certificates are (criterion, judge-family) properties — codability *ordering* is
judge-invariant (ρ .93), individual gates need not be (a110); (iii) mixedness spectrum, LLM-touch
share tracks A-ness; (iv) op taxonomy empirically decidable, op value criterion-specific (a128
helpful vs a67 harmful, same op); (v) chiasmus (post-audit honest version: decisive-cell a153
Bonferroni-proof + within-batch a67 .235); (vi) recon-R reproduces the codability ordering as an
independent operationalization; (vii) **codability ceiling tracks criterion class** — the
three-task fleet result (PR gates readily / math intermediate-heterogeneous / CW lifts-but-
rarely-gates), tying the seam to the two-faces taste/craft/mechanical axis. Plus positioning
paragraph (already verified) + figures inventory (most live; R1 adds the bracket).

### R6. Parked (with triggers)
- W2.1–2.3 + diffs fleet: trigger = code-PR agent lands. First action then: diff-native codegen.
- W2.5 real F2P: trigger = other agent's dockers/tests.
- Patents fleet: trigger = ⚠ scope decision (build prior-art corpus vs ship "evidence-op-dominant,
  upper arm out of scope"). Recommend the latter for this paper.
- W2.4 pr_exec train round: fold into R4 if approved, else park.
- W4.4 recon second recoverer: ⚠ GLM quota; optional robustness appendix.
- Degenerate-split protocol note (a288 no-positives, a210 constant baseline): one paragraph in
  methods — resample or flag, never silently drop (guard 9).

### Open sign-offs (carried + new) — RESOLVED 2026-07-04
1. **Scoped-gate definition — ADOPTED** (user agreed): scoped certificate legitimate iff scope
   predicate is (a) criterion-independent, (b) frozen before gating, (c) applied symmetrically
   to hybrid AND baseline, (d) stamped as scoped with coverage fraction. a80 = scoped-certified.
2. **h1 fleet — APPROVED**, plus standing approval for Sonnet fan-out "across many, many tasks"
   and for expanding the analysis to NEW domains beyond the current 8 corpora.
3. **Patents scope — SUPERSEDED BY NEW FACT:** the prior-art evidence corpus already EXISTS
   (built in a parallel agent thread, complete 2026-06-29): symmetric per-claim-element evidence
   sets, 59,937 claims × K=8 candidate refs (gold planted 99.7%, leak-probe clean), Gemma-4-31B
   localize-then-verify disclosure verdicts over 479,420 references, 5-day sk3 run. Artifact:
   `datasets/patents/processed/option3_claims_gemma_scale.jsonl` (sk3 ONLY, not laptop; laptop
   derivatives: `notebooks/data/patents_va_features.csv`, `legal_va.json`). Headline: length-
   stratified disclosure gap +15.1pts pos-gold vs neg; by rejection type DoublePatent +21.6 >
   §103 +16.0 > §102 +13.3 > §101 +9.1 (legally coherent). ⇒ the patents fleet's evidence op no
   longer needs corpus-building — it needs an ops ADAPTER (pr_exec ExecOps pattern: evidence op
   = lookup into the disclosure verdicts/spans by (app_id, claim_num, element)). New R6→R7 item.
   Also: 5 legal-domain score caches on sk3 (`datasets/legal-outcome-prediction/vat_score_cache/`)
   = candidate NEW domains under sign-off #2.
4. GLM spend for recon robustness — still open (R6, optional).

### R7. NEW (2026-07-04, unlocked by sign-off resolutions)
1. **Patents evidence-op adapter + fleet** — wire `option3_claims_gemma_scale.jsonl` (sk3) as an
   evidence op in the pr_exec ExecOps mock pattern (op = disclosure verdicts/spans lookup per
   claim element; NullOps ablation twin → certified op marginal). Gives patents its upper bracket
   arm — the previous "out of scope" recommendation is retracted; the corpus exists.
2. **New-domain seam surveys** (user-approved expansion): humor first (bank + labeled data
   exist, standard overnight recipe); legal domains as candidates via the 5 sk3
   `vat_score_cache` domains (need survey-format items+judge, not just score caches);
   grant-funding / news_homepages (after its data-cleaning fix) later.
3. **CAM profile (certified articulable mass)** — adopted as the per-task complexity takeaway
   (see `methods/metric_seam/pilot/cam_profile.py`, `outputs/metric_seam_pilot/cam_profile.json`,
   results-note section): per criterion r̃ = clip[0,1](ρ_test/ceiling) of the best materialized
   implementation (one-sided, monotone-under-search); per task the survival curve + CAM = mean r̃.
   First cut: PR .369→.697 (base→certified), CW .131→.466, math .173→.377; frac≥.8: PR .25 /
   CW .03 / math .00. Feeds the R1 money figure as its second panel.

### Suggested sequence
| step | items | cost | why this order |
|---|---|---|---|
| 1 | R1 money figure + R2 lemma/bridge fixes | CPU | everything W5 quotes depends on these |
| 2 | R3 headroom T−R (PR) | CPU | PO-theory tie-in, one task |
| 3 | R5 draft the section | writing | parallel with 1–2 |
| 4 | ⚠ R4 h1 round | ~10 Sonnet agents + small Gemma batch | strengthens (vii) but not blocking |
| 5 | R6 as triggers fire | — | — |
