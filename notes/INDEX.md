# Notes Index

Themed index for the working notes in `notes/`. Notes are dated working
documents — design discussions, run logs, experiment plans, meeting prep.

## How this file is organized

- Notes are grouped by **theme**, not date. Within each theme, **most recent first**.
- Each entry: `- [YYYY-MM-DD — slug](filename.md) — 1-line hook from the note.`
- Filenames follow `YYYY-MM-DD__topic-slug.md`. Keep the convention when adding new notes.
- When adding a new note: pick a theme below (or add one), insert at the top of the theme's list, and write a 1-line hook here.
- Subdirectories (e.g. `articulability-pgm/`) are listed at the bottom.

---

## Research backlog (standing — not date-themed)

Ideas we've thought through but not yet run. Append here when an idea arrives; graduate out when executed.

- [ideas-backlog](ideas-backlog.md) — "ideas-to-run" living list. Priority pair (A) dense-gap `C−(V+A)` stratification + (B) prompt↔code V/A seam, then a full 13-section consolidated backlog (~80 deduped ideas) from a 2026-06-25 sweep of memory/notes/theory-docs/running-log: prompt-optimality certs, scaling laws, STaR/refactoring, metric-tree, E0–E6, local-explanations, rubric-corpus analyses, the 0.5 problem, per-task V/A/T fills, dataset acquisition, parked theory. Status-tagged (un-run / partial / verify / parked).

## Validity experiments — design & metric-level empirical tests

The "metrics for metrics" / E1–E6 line: how to empirically classify each rubric
as Verifiable / Articulable / Defensible / Tacit rather than introspecting.

- [2026-05-22 — metric-level-empirical-test-design](2026-05-22__metric-level-empirical-test-design.md) — Higher-level framing of *what* the L1/L2/L3/L4 test is and why available approaches don't suffice on their own.
- [2026-05-16 — validity-experiments-plan](2026-05-16__validity-experiments-plan.md) — Six experiments to give every rubric an empirical profile; profile, not prompt, decides where it lands on the articulability scale.

## Validity full pipeline — overnight runs & findings (May 24)

End-to-end paraphrase × code-gen × judge runs on peer-review and follow-ups.

- [2026-05-24 — validity-full-final-findings](2026-05-24__validity-full-final-findings.md) — Final peer-review numbers: Qwen >> Llama coder, R1 vs R2 split confirmed at scale, predictive AUCs reported.
- [2026-05-24 — validity-full-progress](2026-05-24__validity-full-progress.md) — Mid-run progress table + cross-model code-gen correlations (Llama vs Qwen vs Claude).
- [2026-05-24 — validity-pilot-smoke-and-determination](2026-05-24__validity-pilot-smoke-and-determination.md) — Smoke test of 5 R1/R2 pairs; R1 wins all code dimensions, R2 wins all judge dimensions.

## R1/R2 open coding & metric tree pipeline

The L0 → R1 → R2 hierarchy pipeline across all 11 tasks: building, refining, merging.

- [2026-05-24 — r1-r2-pipeline-and-experiments](2026-05-24__r1-r2-pipeline-and-experiments.md) — Consolidated R1/R2/Fork experiments; Fork-3 merge lifts F1 to 0.358; R2 aspects give 5.4× compression.
- [2026-05-11 — rubric-variance-analysis-plan](2026-05-11__rubric-variance-analysis-plan.md) — Original research plan for the rubric corpus (38K pages, 361K extracted rubrics, 11 tasks) and the canonical-rubric-count workflow.

## Metric taxonomy / two-axis / noun-verb thickness / structural metrics

Conceptual framing of the taxonomy and structural measurements on the locked clustering.

- [2026-05-19 — structural-metrics](2026-05-19__structural-metrics.md) — Per-task concentration, entropy, Zipf slopes, cross-task overlap on the locked tau-0.825 clustering (53K forms → 33K clusters).
- [2026-05-14 — metric-taxonomy-and-two-axis-setup](2026-05-14__metric-taxonomy-and-two-axis-setup.md) — Reference doc for the rubric taxonomy + two-axis classification: corpus, tree levels (leaf → cluster → R1 → R2 → R3), specificity buckets.
- [2026-05-14 — noun-verb-thickness](2026-05-14__noun-verb-thickness.md) — Richer thickness model: rubric = chain of noun→verb→noun; both nouns and verbs can be thick or thin.

## Articulability framing & verifiability decomposition

VAT (Verifiable / Articulable / Taste) framing — conceptual diagrams, group vs personal.

- [2026-05-28 — analysis-plan-for-noah](2026-05-28__analysis-plan-for-noah.md) — Expanded VAT hierarchy diagram for Noah meeting: verifiable → articulable → inarticulable → noise, each bisected by group vs personal consensus.

## Prompt-optimality theory (global certificates, recovery, α/B_E)

Per-metric `M_i` certificates, two-sided: lower bound = recovery `R_i ≤ T` (DPI); `T(m_ω)` is the **floor** on the ideal `M_i*`, whose upper bound is the α / `B_E` census (consensus `c_∞`, spectral `OPT′`), approached unsupervised — **no `Y`**.

- [2026-07-12 — vector scale–articulation law](2026-07-12__vector-scale-articulation-law.md) — The elegant fixed-policy `M_omega` join: exact substitution is entry into a form-quotiented identity region, while partial substitution is a coordinate vector. G6's direct signature is MAE +.366, rank -.377, flips +.319, bias +.963—confirmed component-wise transport, not a scalar law. Reuses DPI, fresh audit, CUF fingerprints/orthogonalization, and non-monotone composition; explicitly does not revive the retracted `OPT_Omega+epsilon` bridge.
- [2026-07-12 — isomorphism-first tacit-policy reconstruction](2026-07-12__isomorphism-first-tacit-policy-reconstruction.md) — ★ frozen 400-item direct-policy lockbox: 0/3 exact 3B→8B isomorphisms, but all three target-articulated humor prompts family-wise improve adverse MAE over 3B name-only and the best beats intact full text; 36.6% of raw quotient MAE and 89.4% of excess beyond the 8B form band are removed. Exact failure is localized to item order (rho .699 vs target-self .947), establishing component-wise rather than universal scale–articulation substitution without any external target.
- [2026-07-02 — two-faces-theory](2026-07-02__two-faces-theory.md) — ★ the FORMAL home for the Face-1/Face-2 framing (previously only operational labels): Face 1 = census ceiling `OPT_Ω+ε` + certified residual `Δ(E)=lowerCI(C)−[OPT_Ω+ε]` (recap of the bracket, proofs in theory doc); Face 2 = decompression, NEW formal setup — rung TYPES, estimands `R_E(r)` + reader gap `G(r)`, what each rung jump identifies (Frege/Ryle/Polanyi/Collins). THE BRIDGE: units=species, `span_R2` classifies a rung gain as in-span (better addressing) vs out-of-span (new unit); **ostension = certified census violation** (rung beats `OPT_Ω+ε` with saturated census); telling-vs-fitting = why it's not GEPA. Limits + first CW instance.

- [2026-07-01 — cw-unified-grid-roadmap](2026-07-01__cw-unified-grid-roadmap.md) — ONE grid, three deliverables on creative writing: Q1 upper-bound trichotomy over all 42 R3 metrics (CERTIFIED-BOUNDED / CERTIFIED-DEEP / NOT-YET-BOUNDABLE — CPU-free on the rescore checkpoints), Q2 codability map (20-metric message grid: TYPE rungs name/definition/explanation/full-rubric/exemplars/dossier × 4 strata — redesigned 2026-07-02, lengths dropped as axis), Q3 per-model tacit profile (decompression curves + strong−weak gap); **driver `methods/codability/run_decompression_grid.py` smoke-green 2026-07-02; §2b STATUS block: rescore level-dispatch bug fixed, v1 aligned dirs deprecated, v2 retargets + 70B v2 in flight;** all three scale on the ladder 1B(weak)/8B/31–70B/122B + frontier spot; ≈0.8–1M judge calls ≈ 7–8 sk3 nights, 1 GPU; 8 planted controls = kill gates (incl. XOR combiner-blindspot, composition, + tail-XOR breaker for CODIFIABLE); §5 addendum: Δ_comp arm + holistic adversary + order-adverse ε (all implemented); §6 EMIC layer: 691K matched community norms → emic-vs-etic OPT, articulation deficit, community-speech messages, dated norm-flux; **§2b TALK CUT (2026-07-01): minimal within-Llama tacit readout — Day-0 CPU certificates on existing 3B/8B checkpoints + one 70B fill night (3-point Δ(E) staircase) + one decompression-grid night (5 rungs × 4 readers) + Δ_comp mini-arm ≈ 2 GPU nights.**
- [2026-07-01 — codability-audit-and-proposal](2026-07-01__codability-audit-and-proposal.md) — Codability (Brown–Lenneberg → behavioral) audit: pooled probes conflate INDEXICALITY with tacitness; the one embedding vulnerability is the R2/R3 concept definitions themselves. Proposal: stratified Codability Profile — per-stratum {R_g, T_g, κ, MDL, ε-gap} + transfer matrix M[g→g'] + mixed-model decomposition (Δ_context = indexicality); ordinal levels L0 COMPILABLE → L1 UNIVERSAL → L2 INDEXICAL → L3 OSTENSIVE → L4 TACIT-within-frame + FRAGMENTED gate (feeds re-clustering, embedding-free); planted genre-indexed control mandatory. **IMPLEMENTED → `methods/codability/` (controls: `python -m methods.codability.run_codability --controls`).**
- [2026-07-01 — theory-summary-and-implementation-changes](2026-07-01__theory-summary-and-implementation-changes.md) — HANDOFF snapshot (goes stale fast): 6-line theory summary + 9 prioritized code changes (quotient partition, f₁/N gate, conditional-value certificate stream, decide() verdicts, m̄_ω, adverse-orbit reporting, scaling table, be_report leakage fix, planted control) + do-not-change list + run order. **DONE 2026-07-01 PM (103 tests green; two spec deviations recorded in the note's STATUS block: asymmetric f₁/N gate in decide(), Efron–Thisted k₀=4 in Good–Toulmin).**
- [2026-07-01 — articulability-anthropology-reframe](2026-07-01__articulability-anthropology-reframe.md) — The human-exceptionalism framing: the human never appears in the current LM-internal loop; add the ⚠ human-target bracket (certified codification gap `A_H ≥ lowerCI(C_dense) − [OPT_Ω+ε]`), the `A(m; W, Φ, E)` writer/channel/reader index with small human arms, native-rubric benchmark, α vs α_V = productivity vs depth, the direction-of-error FLIP (instrument weakness now inflates the desired gap — positive controls become the whole credibility), Collins/Polanyi/Nisbett-Wilson/Bourdieu anchors.
- [2026-07-01 — prompt-optimality-upperbound-critique](2026-07-01__prompt-optimality-upperbound-critique.md) — Review of the UPPER-bound half: the ideal-ceiling `I(M*;X)≤U` is structurally NOT certifiable (census measures mass, not impact); the fix is an ε-gap guarantee on the *reachable* ceiling `OPT_∞` via value-missing-mass `MV₀` (B–K) ∧ adversarial saturation; converge on `α_V` not count-`α`; quotient the species partition by the form group (semantic-merge) to make content-only measurement valid by construction; add a scalar strong-probe ceiling `T′`.
- [2026-06-18 — prompt-optimality-theory](2026-06-18__prompt-optimality-theory.md) — The full theory: per-metric setup (§1), the `U`-ladder (info cap → within-class → covered-`B_E`), recovery vs within-class (§11), the `B_E-ATLAS` / ALPHA-PROBE / VALUE-CENSUS (§12). Load-bearing: §1 per-metric scope, §12.3 reconstruction-based (anchor-free) value census. **§12.8 (2026-07-01): the provable core — Theorem T1 (anytime-valid ε: checkpoint×order δ-union + a-priori B-cap; 3 soundness holes closed, IMPLEMENTED in `value_certificate.py`), lemmas L1–L6 with proofs (refinement safety, Robbins direction, envelope chain, combiner separation, judge-noise directions), imports I1–I3 (OSW, support-size impossibility, Nemhauser), conjectures C1–C3 (low-degree articulability, tail-synergy decay, bracket coherence), assumption ledger A1–A4. I1 OSW horizon IMPLEMENTED (`osw_horizon_value`, c ≤ ln N, clamped+flagged) + C2's tail-XOR breaker IMPLEMENTED (pairwise-independent parity triple; CODIFIABLE at ε=.031 while .096 bits hide — 3.1× under-cover; `adv_saturated` gate = the mitigation).**
- [2026-06-26 — proof-core-and-vinformation](2026-06-26__proof-core-and-vinformation.md) — One-page companion: the core proof (convexity + DPI), why it's Shannon-DPI not V-information (V-info violates DPI; its §6 monotonicity fails), and recovery (lower) vs α/B_E (upper) as the two sides of the bracket.

- [2026-07-05 — tacit scaling + enculturation + a-priori](2026-07-05__tacit-scaling-enculturation-apriori.md) — ★ three new directions (tasks #21-23): name-sufficiency scaling on the 1B/3B/8B ladder (TASTE names come online 1B→3B .63→.33, CRAFT flat .61, MECHANICAL never — codified≠lexicalized; 70B ordinal prereg FROZEN sha 62e4b3f0); cross-family DiD math +.018 p<1e-4 @1B tier, 17/21 universal-lexicalized + 2 A-only incl "Elegance and beauty of proofs" (the user-predicted "B never taught it" cell); a-priori LODO = instructive NULL on AUC scale (tags ANTI-predict −.19; bal_acc ρ=.35 was calibration artifact). INSTRUMENT: absolute-0.5 bal_acc conflates calibration w/ tacitness (Qwen-3B exact-0.5 everywhere, AUC .649) → grid_auc_report.py threshold-free readout. + grant×peer transport 5/5 p=.083 (first powered rigor test; pooled 41/46).
- [2026-07-05 — wave-2 isomorphism expansion](2026-07-05__wave2-isomorphism-expansion.md) — ★ ALL remaining domains launched (code-review+legal GPU5, peer+grant GPU1; sweep→cert→grid chains; 2 framing bugs fixed pre-sweep: peer=PAPER not REVIEW, code-review=PULL_REQUEST not SOLUTION); band re-read ×5 domains: math 15/16 exFD→UNDERSAMPLED (CW pattern replicates — form cliff = undersampling); ★ genuine (band+H_M≥.1) verdicts INVERT: COD only expressive (CW 2, humor 7), DEEP only institutional/technical (PR 10, news 2, math 2); degeneracy census news 13/25, PR 6/42; mid-z calibration judging launched (z .56/.96/1.17).
- [2026-07-04 — cross-task isomorphism scale-out](2026-07-04__crosstask-isomorphism.md) — ★ GOAL day-1: R3 sharing matrix 5/36 task pairs FDR-significant vs other-tasks null (CW×humor z=+11.3 judge-verified κ=.69; news×PR +5.7; legal×math +5.1; grant×peer +3.7; math×peer +3.7; CW×PR ANTI −5.7); concept-TYPE transports across tasks (83% agree, p=.0017) = tacitness-class concept-intrinsic; form-pair census: definition = best rung, dossier regressive ALL sizes (construction audit owed); PR sweep GPU1 + auto-chain news.
- [2026-07-03 — what-gets-decompressed](2026-07-03__what-gets-decompressed.md) — ★ concept-type covariate (κ=.63, 106 metrics, 0 MECHANICAL in the banks): TASTE = enculturated index (75% cheap-match at L0–L1, checklists HURT it) vs CRAFT = expensive decompression (med L≈4, dominates censored) — inverts naive "taste=tacit"; exploratory (Fisher p=.086, selection-biased chain sample); contrastive>definition holds within BOTH types; v2 chains must stratify by label.
- [2026-07-03 — expansion-chain-v1-results](2026-07-03__expansion-chain-v1-results.md) — ★ planted controls WORKED: 1B instrument-INVALID (chance on "contains a question mark?", flat over 8 levels — its censoring is incapacity, not tacitness), gold-vs-view truncation artifact (fix: gold on the 4000-char view), 3B compliance ceiling ~.7. SURVIVES: grid→chain replication **36/38** (rescued/censored is a stable concept property across instruments); two regimes (rescued: 3B rises +.07 with expansion; censored: high-from-name & FLAT = prior-indexed content); 3B→70B costs (CW KM L=1/83% matched; humor L=4/65%); contrastive increments (boundary/counterexample) positive in 4/4 cells, definitions flat-to-negative; transitivity underpowered (additive where defined, CW 4/4 zero-slack) → v2 = reference rotation to 70B-orbit frees 8B as reader.
- [2026-07-02 — iso-performance-expansion-design](2026-07-02__iso-performance-expansion-design.md) — ★ the HORIZONTAL decompression measurement, designed + LAUNCHED (sk3 PID 2574425): nested monotone chains (name + 7 typed ~25-word increments) × 1B/3B/70B readers; expansion cost x*(B→A,p), δ/KM censoring, planted instruction-floor controls, type-gain matrix + reversed arm, triangle-slack transitivity test (formalism = two-faces-theory §2.4). First read off the v1 grid: humor rescued-by-expansion (62%, δ-robust), CW capacity-floored (74% censored); NEW cross-face coherence — gate-passing humor metrics are overwhelmingly the matched ones (16 vs 3). Diagram notebooks/figures/iso_performance_expansion.html.
- [2026-07-02 — 70b-rescore-semantics-audit](2026-07-02__70b-rescore-semantics-audit.md) — ★ status check turned provenance sweep: own-vs-cross M_i panel (5 negative + 3 positive controls) EXONERATES the 30 "suspect" 70B files (post-fix, not bug-era) but byte-diffs show 70b_v2/qwen_v2 are **MI-ONLY** — sigs AND forminv jsons are byte-copies of src-8B, only orbit M_i is genuine (operational face of the D/C=0.96 shared-Ω confound). 30 dupes relocated out of the full-rescore dir (skip-existing would have silently blocked native 0–36 forever), overrides corrected, pass-2 chain armed, savez/forminv patched to self-label mode+executor. Full-native-67 measured ~3h/metric ⇒ ~Jul 10 ⇒ the Q1 / Q2-head-only / Q3 fork actually needs deciding.
- [2026-07-02 — humor-vs-cw-crossdomain](2026-07-02__humor-vs-cw-crossdomain.md) — ★ COMPLETE BOTH FACES @60 (audited D1+D2): Face-1 humor 35 FD/15 US/4 COD (54 kept), form gate PASS 35% vs CW 16% (~2.2×, stable 41→60) — fragility is DOMAIN-SPECIFIC. Face-2 (clean 3B−1B gaps): shared index→content signature in both; pred-1 REFUTED (unpacking benefit +.089 humor vs +.072 CW — humor does NOT need less unpacking), pred-2 SUPPORTED (span_R² higher every rung). NEW contrasts: humor's full rubric COLLAPSES as a message (+.019 vs CW +.121; distilled explanation beats the checklist) and exemplars transmit humor but not CW at the same k=2/400ch instrument (+.091 vs −.006 — CW taste is show-resistant, the instrument isn't broken). Grid ran 15:47→15:58 on GPU7 (verified non-vacuous).
- [2026-07-02 — cw-grid-v1-results](2026-07-02__cw-grid-v1-results.md) — ★ FIRST Face-2 curves (46 metrics × name/definition/explanation/full-rubric/exemplars/dossier × 1B/3B/8B). ⚠ AUDITED: 8B-reader self-referential (=reference executor) → earlier "+.245 on rubric" RETRACTED; CLEAN gap (3B−1B) smallest at name (+.060), largest at definition/explanation (+.13/+.14), ~0 at exemplars — compression→capability trade holds, located at index→content transition; 70B−3B is the real dynamic range (evening). Messages legible, signatures not collapsed. Exemplars (k=2,400ch) LOSE to telling (instrument caveat); fragility staircase ~FLAT 3B/8B/70B (calibrated 6.5/6.2/5.3%). Face-1: 8B_v2 = 38 FD/5 US/2 COD (H_M≥0.15 filter); source now 67 metrics.
- [2026-07-02 — r3cw-data-catalog](2026-07-02__r3cw-data-catalog.md) — provenance registry + standing gate (`catalog_check.py`: C1 name↔hierarchy, C1b target_desc, shape/τ₀/forminv/orbit checks; CATALOG.json + overrides; 12 tests): llama8b_glm + *_v2 + aligned_70b = OK/valid, v1 aligned = DEPRECATED, OLD llama8b_to_{3b,70b,qwen} = SUSPECT-M (bug-era: sigs valid, M-based numbers INVALID), smoke = FIXTURE; policy = nothing consumes a non-OK dir.
- [2026-07-01 — form-effects-control-plan](2026-07-01__form-effects-control-plan.md) — CW 36/42 FORM-DOMINATED is GENUINE (gate = fixed 10% median-flip bar, 12.2% > 10%; NOT 2·τ₀ — earlier caveat corrected) but fires on the INSTRUMENT seat (Ω criteria), which `--orbit-target` never touches; two Day-3 traps (aligned dirs have no forminv → gate silently vanishes; hardcoded τ₀=0.05); fixes IMPLEMENTED on sk3 2026-07-02 (`M_i_forms`+`pairs` persistence, τ₀ carry-through, `--form-invariance-n`, `form_decompose.py`, 8B forminv copied w/ provenance; 3B turned out NOT sig-identical → needs own pass); plan = main-effect-vs-interaction calibration split, per-form attribution, fragility-vs-scale as measurand, gate→ε_form redesign (sign-off). More probes/repeats do NOT help; K>4 needs LLM-paraphrase orbit.

## Related work / landscape

- [2026-07-01 — tacit-knowledge-related-work](2026-07-01__tacit-knowledge-related-work.md) — Deep-research map of how tacit knowledge has been studied in LLMs (5 angles, 22 sources, strict verification killed 21/25 claims). Closest neighbor = Yu et al. "Grading the Unspoken" (2604.14188, tacit-step reconstruction — measures LLM capability, not human articulability, no certificate). Nearest FRAMING = "Limits of Prompt-Conditioned LMs" (2606.23668, capacity-limited channel — but claims refuted). Rubric/faithfulness cluster (Shen/Meta 2602.05125, outcome-vs-process 2603.16600) supports checklist-vs-trained gap. **Verdict: certified per-metric articulability ceilings + capture-recapture + decompression curves + scaling-flatness + human-target = uniquely ours.** Talk related-work slide.

## Noah meetings (prep, queues, follow-ups)

Meeting-specific prep notes and the overnight queues that fed into them.

- [2026-05-28 — analysis-plan-for-noah](2026-05-28__analysis-plan-for-noah.md) — VAT diagram + analysis plan to draw for May 28 Noah meeting.
- [2026-05-28 — overnight-queue](2026-05-28__overnight-queue.md) — Stage-0/1/2 wakeup-loop orchestration of overnight Qwen judge runs feeding the next morning's analyses.

## Overnight exploration logs

Autonomous-budget runs — confound hunts and other unsupervised explorations.

- [2026-05-15 — overnight-exploration-log](2026-05-15__overnight-exploration-log.md) — Autonomous overnight confound hunt on the per-task articulability classification; ≤ $30 credit budget, multiple audit subagents.

## Dataset leakage audits

Cross-dataset leakage / confounder log and fixes.

- [dataset_leakage_audit](dataset_leakage_audit.md) — Running log of every leak, confounder, and shortcut feature across the 12 modeling datasets, plus cleanup status per task.

## Patent §102/§103 supervised pairs

Patent rejection-pair construction methodology.

- [2026-06-04 — patent_supervised_pairs_methodology](2026-06-04__patent_supervised_pairs_methodology.md) — Methodology for compiling clean §102/§103 supervised pairs; data structures + pitfalls captured while debugging the v2 retriever.

---

## Subdirectories

- [`prompt-optimality-whitepaper-latex/`](prompt-optimality-whitepaper-latex/) — compiled LaTeX whitepaper + PDF (`prompt-optimality-whitepaper.tex`, `.pdf`): the polished summary of `2026-06-18__prompt-optimality-theory.md` — two-sided bracket (`R ≤ T(m_ω) ≤ I(M_i*;X) ≤ U_{B_E}`, `T` = floor on the ideal), recovery ceiling via the posterior, the `B_E` census (ALPHA-PROBE / consensus `c_∞` / spectral `OPT′`), process-relative scope, related work (Sorensen / Wenliang / Picca). Synced to the theory doc 2026-07-01; 8pp.
- [`articulability-pgm/`](articulability-pgm/) — LaTeX + PDF of the articulability probabilistic-graphical-model writeup (`articulability_pgm.tex`, `articulability_pgm.pdf`, May 9 snapshot).
