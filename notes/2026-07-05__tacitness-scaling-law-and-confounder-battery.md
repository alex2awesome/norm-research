# 2026-07-05 (night) — The tacitness scaling law + the confounder control battery

Two user directives: (1) "'Scaling laws' should say something in the absolute... if x tacit
knowledge exists in model 1, x+y in model 2, x+y+z in model 3 → model_infinity should contain
the asymptote. Design this scaling law (similar to the Chinchilla scaling laws)." (2) "I like
all of the confounder taxonomy suggestions. Note this in the notes and design tests for all
of them." Tasks #24/#25. Continues `2026-07-05__tacit-scaling-enculturation-apriori.md`.

---

## PART I — The tacitness scaling law (Chinchilla-analogue)

### I.1 The object: a two-axis transmission surface with two asymptotes

The decompression grid already measures a surface: T(N, L) = transmission of the executor's
metric to a reader of scale N given articulation depth L (rung: name < definition <
explanation < full_rubric < dossier). On the Gini scale G = 2·(AUC − ½):

    G_w(N)  =  G∞_w  −  A_w · N^(−α)          (per channel w = rung; Chinchilla's
                                                L(N) = E + A·N^(−α) with the sign flipped
                                                because transmission RISES to a ceiling)

The Chinchilla correspondence:

| Chinchilla | here |
|---|---|
| L(N) = E + A/N^α | G_w(N) = G∞_w − A_w/N^α |
| E = irreducible loss (entropy of text) | C − G∞_name = irreducible NAME-channel shortfall |
| N (params), D (data) two axes | N (reader scale), L (articulation depth) two axes |
| compute-optimal frontier N*(C) | articulation-optimal frontier L*(N): cheapest rung to hit transmission t at scale N |

**The two asymptotes are the two tacitness layers** (ties to tacitness-two-layers):
- **N → ∞ at fixed L=name**: g∞ = G∞_def − G∞_name = the part of the metric no reader EVER
  gets from the lexical pointer alone — *unlexicalized content* (the user's "model_infinity"
  asymptote). Per-family: differences in g∞ across families = enculturation differences that
  never wash out with scale ("never taught"), vs differences in A/α = "taught later"
  (arrival-time differences). **This formally separates the two readings of the DiD.**
- **L → max at any N**: the dossier/full-rubric ceiling shortfall = *unarticulable content*
  (language-tacit vs fully-tacit split).

### I.2 What "absolute" means here (and the guard rails)

The user's x, x+y, x+y+z picture: K(N) = G_name(N) is the name-unlocked knowledge at scale N;
increments are the y, z; K∞ = G∞_name is the asymptote. Per the 2026-06-13 irreducible-E lit
dive (metric_implementer/2026-06-13__irreducible-E-scaling-laws.md): **joint-fit intercepts
are EXTRAPOLATED, not identified** (Hoffmann's E is an intercept confounded with decay terms;
Henighan concedes the entropy estimate is not meaningful). So every G∞ ships with:

1. **An identified bracket**: G(biggest CLEAN reader) ≤ G∞ ≤ C(m), where C(m) = the
   executor's own full-rubric re-execution consistency (the ceiling any reader could hit).
   The parametric point estimate lives inside; the bracket is what's actually measured.
2. **Out-of-sample validation before trust**: parametric 70B predictions frozen NOW
   (prereg_scaling_law_70b.json, sha 92024275…, alongside the ordinal freeze 62e4b3f0…).
   REJECTION RULE (in the artifact): >half of cells outside their 95% CI ⇒ the parametric law
   is rejected; only ordinal claims + brackets survive.
3. Koyejo guards: within-lineage ladders only (V-info non-Lipschitz across architectures);
   AUC/Gini scale (bounded, rank-based), never raw bits; stated extrapolation domain (one
   decade beyond the largest fitted point).

### I.3 Estimation design

- **Points**: within ONE family/lineage. Llama today: 1B/3B clean + 8B SELF (executor). The
  self point gets a shared offset δ_self — and today's fits show it is UNSTABLE (+.048 in
  the cell fit, −.033 in the domain fit; CW's self point is *depressed*: 3B G .82 vs self
  .51). **Decision: exclude self points from future law fits.** The clean-ladder problem is
  solved by (a) post-Jul-10 Llama {1B, 3B, 70B} and (b) **gemma-3 {1b, 4b, 12b, 27b} — a
  4-point ladder where NO reader is the executor** (the best-identified ladder we can build;
  runnable as reader passes over byte-identical messages).
- **Hierarchical fit**: shared α per (family × domain-class) pinned by all metrics jointly;
  per-metric (G∞, A) on the clean points; weighted by probe-bootstrap SEs.
- **Per-metric hypothesis tests**: H0: g∞(m) = 0 (no irreducible name gap) via bootstrap CI /
  LRT, BH-FDR across metrics. The money table: which CONCEPTS have g∞ > 0 (irreducibly
  tacit-to-name) vs g∞ ≤ 0 (fully lexicalized in the limit) — per family.
- **Capability coordinate variant**: refit with x = κ(reader) (pseudo-concept
  definition-execution capacity, battery T1c) instead of parameter count — makes curves
  comparable across families whose parameter counts don't align (Chinchilla's compute axis
  analogue, and the T9 deconfound).

### I.4 Today's provisional fit (3-point Llama, cells ≥5 metrics; scaling_law_cells.json)

Read as a DESIGN DEMONSTRATION, not a result: α hit its 3.0 bound (the 1B→3B jump then
3B→8B flattening is not power-law-clean), rms .26 on the Gini scale, and the self point is
the known contaminant. With that caveat, the asymptote SIGNS are coherent with the survival
tables: g∞ (def−name at ∞): institutional MECHANICAL **+.093** (the spec never compresses
into the word), institutional CRAFT +.032, expressive CRAFT +.030, expressive TASTE +.016,
formal-lexicon CRAFT **−.059** (the lexicalization inversion persists at the asymptote).
Frozen 70B cell predictions: e.g. expressive craft name-AUC .867, institutional craft .773,
formal-lexicon craft .716. Evaluation lands with the 70B pass.

---

## PART II — The confounder control battery (T1–T10)

Each test: mechanism → design → decision rule → cost. FREE = computable from existing
npz/JSON. Status markers: [DONE] [RUNNING] [READY] (implemented, awaiting sk3) [DESIGN].

### T1 — Articulation-exploitation (instruction-following) ★ the operative one
**Mechanism**: a stronger instruction-follower exploits definitions better; DiD = Δdef −
Δname can be positive with NO name-side deficit. **Confirmed operative**: the significant
1B-tier DiD is Δdef-driven in both domains (math Δdef +.030 vs Δname +.012; CW +.013 vs
−.005).
- **T1a decomposition** [DONE]: always report DiD split into Δname / Δdef; only
  Δname-driven effects count toward enculturation.
- **T1b def-parity cells** [RUN, confounder_T1b_T8b_local.json]: taxonomy re-derived
  requiring |AUC_def(stranger) − AUC_def(kin)| ≤ .05. RESULT: over-strict with the current
  pairing — only 8/21 math and 1/46 CW pairs reach parity (Qwen-7B vs Llama-3B is
  capability-mismatched by construction), all parity survivors universal. The USEFUL product
  is the **direction-of-capability argument** it exposes: the stranger is the STRONGER
  reader, so capability advantage pushes its name-AUC UP ⇒ **A-only cells (strong stranger
  fails the name where the weak kin succeeds) are robust to the capability confound — it
  works against them; B-only cells are exactly what capability advantage fakes ⇒ demoted to
  suspect pending κ-matching (T9)**. Standing rule: quote A-only cells; quote B-only only
  after matched-κ replication.
- **T1c pseudo-concept ladder** [RUNNING via API; sk3 version READY]
  (methods/codability/pseudo_concept_ladder.py + api_reader_probe.py --mode pseudo):
  10 invented names ("the smorbex quality") bound to programmatic rules (threshold =
  probe-median, exact ground truth) + 2 real-word twins (conciseness, dialogue-heavy; same
  rules) → κ(reader) = mean definition-AUC = pure rule-execution capacity with the name
  prior surgically removed. Validity check inside: pseudo-NAME AUC ≈ .5 (deviation =
  leakage). Real-twin − pseudo gap = real-name priming. **Decision rule**: enculturation
  claims must survive conditioning on κ (regression or matched-κ comparison).

### T2 — Form/format-following
**Mechanism**: paraphrase brittleness ≠ concept ignorance. Partially controlled (3-form
orbits averaged into every rung score).
- **T2a scaffold orbit** [DESIGN, small GPU]: re-score a 12-metric subset under 3 harness
  scaffolds (instruction-first vs text-first template; system-vs-user placement; terse vs
  verbose framing) × both families. Variance decomposition: scaffold share vs rung share of
  variance. **Decision rule**: cross-family claims require scaffold-share < rung-share and
  DiD sign stable across scaffolds.

### T3 — Score calibration [DONE + census READY]
**Mechanism**: absolute-threshold readouts conflate P(yes) shift with signal (the Qwen-3B
lesson; feedback_threshold_free_readouts). Fixed via AUC.
- **T3a tie census** [READY, confounder_free_battery.py]: distinct-value fraction +
  mode-share per reader×rung (heavy ties attenuate AUC). **Decision rule**: flag cells with
  mode-share > .5; sensitivity re-read with midrank jitter.
- Note the API-probe readout is hard-verdict (argmax = implicit 0.5 threshold) — API panels
  are PROBES; headline numbers stay on the local soft readout. The llama-8b API-vs-local
  anchor quantifies the instrument gap.

### T4 — Context-length / attention budget
**Mechanism**: long rungs penalize small readers independent of content → fake
name-sufficiency at small N. Partial prior evidence: rungs-are-types-not-lengths (length was
tested and predicted poorly).
- **T4a padded-name** [DESIGN, small GPU]: name + neutral filler to definition length; if
  padded-name ≈ name (not ≈ definition), content not length drives the rung effect.
- **T4b truncated-dossier**: dossier hard-cut to definition length; symmetric check.
  **Decision rule**: |padded−plain| < .02 AUC across readers.

### T5 — Executor idiosyncrasy (kin shares the writer's quirks) ★ the deepest one
**Mechanism**: kin readers may match the 8B's idiosyncratic extension (shared inductive
biases), not "the concept as the culture has it"; fakes kin name-advantage.
- **T5a consensus-extension ref** [READY, confounder_free_battery.py]: restrict scoring to
  probes where executor-A (Llama-8B full-rubric) and the biggest stranger (Qwen-7B
  full-rubric), both rank-binarized at the ref base rate, AGREE — the shared extension.
  **Decision rule**: A-only cells and Δname must survive on the consensus subset.
- **T5b symmetric executor arm** [DESIGN, ~1 GPU-day]: re-run the R3 sweep with
  `run_alpha_probe --target-model Qwen/Qwen2.5-7B-Instruct` on math+CW → Qwen-authored
  metrics + refs → both reader panels again. **Decision rule**: concept-level enculturation
  predicts the A-only/B-only cells MIRROR under role swap; wholesale flip = quirk-sharing.
- **T5c**: 8B/self tier permanently excluded from cross-family claims [DONE].

### T6 — Writer-dialect of the definitions
**Mechanism**: definitions were generated by the A-side apparatus; a stranger might parse
A-phrased definitions worse. Direction check [DONE]: this deflates the stranger's def-AUC ⇒
biases DiD NEGATIVE ⇒ conservative for the observed +DiD; but it can fake B-only cells.
- **T6a definition-source cross** [DESIGN, small GPU]: for the ~30 family-specific cells,
  add Qwen-authored + neutral-authored paraphrases of each definition; all readers × all
  sources. Two-way ANOVA; the reader-family × def-source interaction = the dialect effect.
- **T6c form-variance asymmetry** [READY, battery]: per-form AUC dispersion, stranger vs
  kin, paired by metric. **Decision rule**: no significant excess stranger dispersion, or
  B-only cells discounted.

### T7 — Tokenizer binding [READY, local]
**Mechanism**: a name that fragments into many tokens under one family's tokenizer is less
"bound" for that family for reasons below culture. **Design**: token counts of each metric
name (and definition) under both tokenizers; logistic P(family-specific cell) ~
token-count asymmetry. **Decision rule**: cells whose asymmetry predicts their class are
demoted to tokenization artifacts.

### T8 — Probe-domain familiarity
**Mechanism**: the stranger reads the probe TEXTS worse (never saw the domain), attenuating
everything. Mostly nets out in within-reader d; contaminates via differential floor
censoring.
- **T8a both-measurable intersection** [DONE — all DiD/cells already require it].
- **T8b censoring asymmetry census** [RUN, confounder_T1b_T8b_local.json]: CLEAR — math
  21/21 both-measurable, CW 41 both + 5 neither, ZERO asymmetric cells. No differential
  floor censoring between families.
- **T8c probe-perplexity covariate** [DESIGN, small GPU]: mean logprob of probes under each
  family's base model; partial the DiD on it.
- **T8d bidirectionality argument** [DONE]: CW's A-only AND B-only cells coexisting already
  rules out a uniform stranger handicap.

### T9 — Capability mismatch at "size-matched" tiers [READY once T1c lands]
**Mechanism**: Qwen-1.5B ≫ Llama-1B in general capability; tier means inherit it. **Design**:
replace parameter-count tiers with κ-matched comparisons (interpolate along each family's
own ladder to equal κ). All tier analyses re-run at matched κ. **Decision rule**: DiD claims
quoted at matched-κ, not matched-N.

### T10 — Statistics: selection, dependence, regression-to-mean [READY, battery]
- **T10a probe-clustered bootstrap** (B=1000, resample probes): the honest p for tier DiDs
  (metrics share 300 probes; metric-level sign-flip overstates).
- **T10b split-half cell stability**: classify every metric's taxonomy cell on even probes,
  verify on odd. **Decision rule**: report cell persistence; only cells stable in both
  halves are quotable; expect ~ε-boundary cells to churn (rates robust, identities not).
- **T10c BH-FDR** on any per-metric asymptote/gap claims [standing].

### The gemma-2 vs gemma-3 vs gemma-4 generational series (user 2026-07-05)
Cross-GENERATION within one lineage: tokenizer/template/org-culture ≈ held, **training
vintage varies** → (a) *diachronic enculturation*: concepts gemma-2 needs defined but
gemma-4 name-transmits were taught to the lineage between generations; (b) a natural
confound-isolation series for T2/T7 (scaffold+tokenizer ~constant across the contrast);
(c) capability-vintage separation: gemma-3-4b ≈ gemma-2-9b capability at different vintages.
**Live now via OpenRouter** (gemma-2-27b, gemma-3-4b/12b/27b, gemma-4-31b all servable):
running tonight on humor (probes+refs fully local, window verified 240/240) with the
hard-verdict readout + llama-8b API-vs-local anchor. GPU (soft-readout) version queues on
sk3 for CW+math when access returns; gemma-3's 4 sizes double as the law's clean ladder
(I.3).

### Execution state (this session)
- RUNNING: OpenRouter panel (3 Llama anchors + 4-gen gemma ladder + llama-70B indicative +
  qwen3.5-9b) on humor-real + math-pseudo; GLM-5.2 (zai) sparing panels; 8/24 Sonnet
  subagent judges (wave 1) on the 12-metric humor subset with blinded directional anchors.
- READY awaiting sk3: confounder_free_battery.py (T3a/T5a/T6c/T10a/T10b), pseudo ladder GPU
  version, gemma GPU chain, T5b symmetric arm (`--target-model` switch confirmed).
- LOCAL next: T1b def-parity cells, T7 tokenizer counts (needs tokenizers — laptop HF ok),
  T8b censoring census.

## PART III — Convince verdict + E* battery cross-check (2026-07-05 night)

Cross-checked our T1–T10 against the code-seam E* battery (prompted-vs-coded fields — a
more rudimentary tacit-knowledge probe). Framing import: recast our *confounds* as competing
HYPOTHESES the way E* does, because it answers "can we be convinced" directly.

### Alternatives to enculturation, and which T kills each
- **H_follow / H_spec** (the definition is a self-contained spec; a stronger instruction-
  follower just executes it) ← T1. **CONFIRMED OPERATIVE**: DiD is Δdef-driven at 1B.
- **H_quirk / H_idio** (kin matches the 8B-writer's private extension) ← T5.
- **H_cap** (size-matched ≠ capability-matched) ← T9.
- **H_form/H_calib/H_len/H_token/H_probe/H_dialect/H_stats** ← T2/T3/T4/T7/T8/T6/T10.

### Tonight's CPU battery (confounder_free_battery.py, math+cw, llama×qwen)
- **T3 ties**: NO reader mode-share >0.5 in either domain → AUC not tie-attenuated. CONTROLLED.
- **T5a consensus-extension**: shared Llama-8B∩Qwen-7B subset = 76% (math) / 80% (cw) of
  probes. Large ⇒ signal lives on the shared extension, not private quirks. (Per-cell A-only
  survival on the subset = the remaining decision-rule read.)
- **T6c dialect**: name-rung form-variance ≈0.015 for EVERY reader, kin=stranger → no excess
  stranger dispersion. CONTROLLED.
- **T10a honest (probe-clustered) DiD**: MATH +.018/+.014/+.012 (p .000/.038/.070, shrinking).
  **CW +.036 → −.015 → −.017 (all p<.001): a SIGN-FLIP with scale.** An instruction-following
  confound is monotone in the capability gap ⇒ cannot produce the flip ⇒ the 3B/8B CW signal
  is not the T1 confound. (Replicates the math-inversion / gradient-legal −.058 pattern.)

### Verdict: can the battery FULLY convince us model diffs ⇒ tacit knowledge? NO.
1. **Eliminative, not constructive** — enculturation is the residual after killing enumerable
   alternatives; inference to best explanation, not proof (true of E6 for them too).
2. **H_follow is demonstrably LIVE** (DiD Δdef-driven at 1B), rescued only on the A-only
   subset by a direction-of-capability argument (T1b) + the CW sign-flip.
3. **The positive existence tests were missing.** Our whole battery is eliminative; E* has two
   tests that positively distinguish retrieval-key from spec — now IMPORTED:

### Imported from E* (implemented + queued tonight: methods/codability/stipulation_probe.py)
- **E1 KEY / nonce+def**: rungs name, def, named_def ('"<name>": <def>'), nonce_def
  ('"<nonce>": <def>'). named_def − nonce_def = name's retrieval value with the full spec
  held fixed. (Fixed a build bug: humor defs never restate their name (0/59), so the correct
  contrast swaps the LABEL, not text inside the def.)
- **E2 STIP / snap-back**: deviant in-prompt redefinition binding the real name to an
  orthogonal surface rule (mean |corr| 0.03 humor / 0.01 math ⇒ separable). Dual readout:
  compliance = AUC(score, deviant truth); snap_back = AUC(score, real ref). H_spec ⇒
  compliance≈1, snap≈.5; retrieval ⇒ snap stays high, compliance dragged down (name overrides
  the edited def — construct-level lexical-Stroop). Queued: Llama 1B/3B/8B + Qwen-7B +
  gemma-2-9b × {humor, math}, self-starts when gemma frees GPU 7.

### E* tests deliberately NOT imported (not relevant to our structure)
- **E5 APERTURE (coded views ν(x)) + E7 SEL (trained selector)** — specific to the code-seam
  hybrid (an LLM field feeding CODE). Our reader scores full text: no aperture, no selector.
  The leak CONCERN is partly handled by reconstruction-only refs (executor's own verdicts, not
  labels); a light surface-feature leak check (partial name-AUC on probe length / lexical
  overlap) is a reasonable lighter import.
- **E4 LOCUS (base vs instruct)** — relevant (is the "culture" pretraining or annotator
  norms?) but hard: base models don't do the soft logprob readout. Aspirational.

### Reframe worth adopting from E*: VALIDITY-gating vs INTERPRETATION-gating
E* separates tests that gate whether the effect is REAL (E6 idiosyncrasy, E5/E7 leak) from
tests that gate the STORY (E1–E4 — if they flip, T-RET retracts but certificates survive).
Ours doesn't make this split explicit. It should, because it pinpoints the convince gap:
- our VALIDITY-gaters (T3, T5a, T6c, T8b, T10a) mostly PASS ⇒ there is a real, cross-family-
  replicable model difference that is not calibration / idiosyncrasy / dialect / censoring /
  a statistical artifact;
- our INTERPRETATION-gaters (T1 operative; E1/E2 not yet read) are exactly where we are weak.
So precisely: the battery can convince us the effect is REAL; it cannot yet convince us the
effect is tacit KNOWLEDGE rather than differential spec-execution. E1/E2 are the tests that
would close that specific gap.

### E1/E2 FIRST RESULTS (2026-07-05 ~22:10; stipulation_probe.py, 5 readers × humor+math)
outputs/stipulation_{humor,math}.json (real-name stip) + stipulation2_* (adds nonce control).

**E1 (named_def − nonce_def, name value with the correct spec held fixed):** ≈ 0 everywhere
(humor +.018/+.009/+.000/−.001/−.016 across Llama-1B/3B/8B, Qwen-7B, gemma-2-9b; math similar,
−.014..+.010). ⇒ given the correct definition, the real name adds ~nothing (name and correct
def are REDUNDANT). Low power where name already saturates (humor 8B name .934).

**E2 (snap − compliance; +=name overrides the deviant redefinition):**
  Llama-1B +.124/+.112 (hum/math), Llama-3B +.204/+.076, Llama-8B +.084/+.034,
  Qwen-7B −.058/+.036, gemma-2-9b −.008/−.019. Raw pattern: LLAMA snaps back both domains,
  strongest small-scale; Qwen humor is COMPLIANCE-dominant (enculturation texture).

**BUT the nonce control (stip_nonce = same deviant rule under a nonce name) undercuts the
naive read at 1B:** Llama-1B compliance is ~chance under BOTH real (.545) and nonce (.521)
names (suppress cmp_nonce−cmp_real ≈ −.02 humor / +.01 math) ⇒ 1B cannot follow ANY deviant
rule; its "snap-back" is the instruction-following confound (defaults to real-concept
salience), NOT name-override. The AIRTIGHT snap-back signature = a CAPABLE reader that
complies well under nonce but poorly under the real name (cmp_nonce ≫ cmp_real WITH snap high).
Verdict therefore hinges on 3B/8B/Qwen/gemma (scoring; watcher stip2_watch.log). DO NOT quote
snap-back as confirmed until the stronger-reader suppress gap is in. This is precisely the
validity check the code-seam E2 needs too (their compliance .36 vs snap .46 has the same
latent confound unless nonce-controlled).

**Combined E1+E2 provisional story (pending nonce control at scale):** the construct name is
a *sufficient* key (redundant with the correct def, E1≈0) and — IF the stronger-reader nonce
gap holds — a *committed* key (a deviant def can't dislodge it, E2 snap-back). "Sufficient +
committed" = the retrieval-key reading; "sufficient + not-committed" = spec. The nonce control
at 3B/8B is the bit that decides which.

### E2 SNAP-BACK RESOLVED via nonce control at scale (2026-07-05 ~22:25) — SIGNIFICANT DISSOCIATION
suppress = compliance(nonce name) − compliance(real name), same deviant rule; >0 = the real
name RESISTS redefinition (committed meaning). Bootstrap 2000, per-metric.
- **HUMOR (expressive): SIGNIFICANT resistance, cross-family** — Llama-8B +.030 p<.001, Qwen-7B
  +.026 p<.001, gemma-2-9b +.041 p<.001; **POOLED +.032 CI[.026,.039] p<.001 (3 readers/2 fams)**.
- **MATH (formal): NO resistance** — Llama-8B −.020 (p=.034, opposite), Qwen −.004 n.s.; pooled
  −.012 p=.075.
- 1B uninformative (compliance ≈ chance under both names — can't follow any rule).

**⇒ CLEAN 2×2 dissociating the two E-tests along the isomorphism's formal/expressive axis:**
  formal (math) names = SUFFICIENT (E1 name≈def) + COMPLIANT (E2 suppress≈0) = precise lexical
  pointer / SPEC; expressive (humor) names = INSUFFICIENT (need >name) + COMMITTED (E2 suppress
  +.032) = enculturated RETRIEVAL KEY. This is the FIRST POSITIVE (non-eliminative) evidence
  that the expressive-domain effect is committed community meaning, not spec-execution — the
  name pulls back to what the culture means even when you stipulate otherwise. Magnitude small
  in absolute AUC but ~15-30% of the (small) compliance signal. Caveat: the dramatic raw
  snap−comply gaps were MOSTLY "reader defaults to real-concept salience" (snap high under both
  names); the true name-override is the modest +.032. Significance-test / more-domains = next.

### State corrections + close-out (2026-07-05 ~22:40)
- **E2 math FINAL (full data, gemma n=21):** suppress pooled −.0116 CI[−.025,+.003] p=.125 —
  cleanly NO resistance (humor +.0322 p<.001 unchanged). Dissociation stands.
- **gemma-4-31b RE-DIAGNOSED:** re-download resolved instantly against the HF manifest ⇒
  snapshot is COMPLETE; the "Following weights were not initialized" failure (vision_tower +
  q/k_norm + layernorms) is a vLLM-0.23 Gemma4 weight-MAPPING bug, NOT a broken download
  (earlier note wrong). Path forward: newer vLLM or a name-mapping patch; deprioritized.
- **gemma-3 12b/27b:** downloaded COMPLETE into /lfs/skampere3/0/shared_hf_cache (24
  safetensors; env pins HF_HUB_CACHE there — same gotcha that misled gemma-4). 4b fetch +
  3-point ladder scoring (4b/12b/27b × humor+math, same grid windows) LAUNCHED with
  ai_usage→gemma4 env fallback: outputs/gemma3_ladder_20260705.log.
- Durability: all tonight's outputs pulled to local notebooks/data/two_faces_20260702/
  (confounder_battery_{math,cw}, stipulation{,2}_{humor,math}, stip_build_*, updated
  {auc_,}report.json for humor+math grids incl. gemma-2). VLLM_BLOCK_SIZE backend patch
  ported to LOCAL vllm_backend.py (was sk3-only sync debt).
- Hardening queue for E2 before any load-bearing use: (1) probe-clustered bootstrap of
  suppress (metrics share 300 probes); (2) name-salience covariate (is suppression ∝
  name-AUC level rather than domain class? within-domain regression); (3) CW stipulation =
  3rd gradient point (PREDICTION: intermediate suppression — falsifiable wedge).

### Overnight hardening results (2026-07-05 ~23:00, CPU passes on tonight's data)
- **E2 salience covariate (e2_salience_covariate.json): DOMAIN CLAIM SURVIVES.** Within-domain
  suppress⊥name_auc (Spearman .023 humor / −.005 math); OLS suppress ~ name_auc + is_humor +
  reader dummies: **is_humor +.0516 CI[.036,.067] p<.001**, name_auc slope small & negative
  (−.068). Suppression is a domain property, not name-salience. (Cell-level resample; probe-
  clustered version unblocks when stip_sigs land overnight.)
- **T10b split-half persistence (t10b_cell_persistence.json): identities churn, rates robust**
  (as pre-registered in the decision rule). Per-metric DiD sign agreement even/odd: math
  .52/.71/.62, CW .78/.74/.74 across tiers. A-only flags stable-in-both-halves: CW-1B 9/22,
  math-3B 1/3, 8B tiers ~0; B-only: CW-3B 3/9, CW-8B 3/13. ⇒ quote AGGREGATE rates and the
  povered DiDs; per-metric cell IDs (e.g. flagship "Elegance of proofs") need split-half
  qualification.
- **T7 tokenizer (t7_token_asymmetry.json, 127 metrics): asymmetry +.124 mean, sd .037** —
  Llama names cost ~12% more tokens than Qwen UNIFORMLY; near-zero per-cell variance ⇒ can't
  explain cell-specific family effects. Morning join with cell labels to close out.

### Overnight queue (sk3 GPU 7, PID 997837, waits for gemma-3 ladder PID 903619)
1. CW stipulation E1/E2 (build check → 5 readers × 67-ref grid) → 3rd gradient point;
   PREREGISTERED PREDICTION: CW suppression INTERMEDIATE between math (≈0) and humor (+.032).
2. humor+math stipulation re-run w/ --sig-dir (3 strong readers) → probe-clustered bootstrap
   of suppress (the honest p).
3. pseudo-concept κ ladder (7 readers × humor+math) → T1c capability coordinate → T9
   κ-matched tiers → adjudicate the suspect B-only cells.
Also in flight: gemma-3 v2 ladder (4b/12b/27b, own model dirs — shared-cache lock wall
bypassed); T7 done; gemma-4 blocked on vLLM-0.23 Gemma4 weight-mapping (deprioritized).

## PART IV — Overnight harvest 2026-07-06 (~03:00): CW gradient point + honest p + κ

### E2 three-domain readout — PREREGISTERED PREDICTION REFUTED, INFORMATIVELY
Predicted CW suppression intermediate (math ≈0 < CW < humor +.032). RESULT (pooled strong
readers, metric-boot): humor **+.0322 p<.001**, CW **−.0040 p=.078**, math −.0116 p=.11 —
**CW patterns WITH math**, not between. Name-commitment is NOT graded by the lexicalization
gradient; it is humor-specific among the three. CW names are HIGHLY salient (L8B name-AUC
.933, E1≈0) yet fully COMPLIANT — craft-technical vocabulary defers to stipulation exactly
like formal math vocabulary. Sharper hypothesis (feeds a-priori #23): **evaluatively THICK
concept names (punching-up, cringe, tastefulness) resist redefinition; THIN technical craft
terms (pacing, clarity, POV) comply**. Thickness is codable a-priori from name text → new
predictive feature.
- CW small-reader positives (1B +.007 p<.001, 3B +.016 p=.006 probe-clustered) sit in the
  artifact-prone regime (humor-1B was NEGATIVE −.024): small-reader suppress signs are
  unstable; only strong-reader cells quotable.

### Probe-clustered bootstrap (the honest p; resample 300 shared probes, 1000 boots)
**Humor survives its final statistical hurdle — each strong reader individually significant:**
L8B +.0298 p=.010, Q7B +.0251 p=.034, G9B +.0410 p<.001. Math: L8B −.0196 p=.020 (mildly
NEGATIVE = real name slightly HELPS compliance — transparent-pointer reading), others n.s.
CW strong readers: all n.s. (p=.38–.89). Files: probe_clustered_suppress.json.

### κ ladder landed (pseudo_ladder_{humor,math}; T1c → T9 usable)
humor-probe κ: Llama .535/.644/.718 (1B/3B/8B), Qwen .583/.626/.711, gemma-2-9b .763.
math-probe κ lower (.50–.60; proof text hostile to surface-stat execution). Leakage: math
clean (.47–.51); humor pseudo-name leak .55–.57 (mild yes-bias correlation with text stats —
subtract as baseline; κ margin humor L8B +.17). **T9 payoffs:** Llama-3B κ .644 ≈ Qwen-3B
.626 = the 3B tier IS capability-matched (its DiD comparisons stand); Qwen-1.5B .583 ≫
Llama-1B .535 confirms the 1B-tier mismatch (1B-tier DiD stays demoted). B-only cell
re-adjudication at matched κ = morning analysis.

### Infra close-out
gemma-3 hang root-caused to init stage common to BOTH vLLM stacks (post-FlashInfer-sampler);
eager+TRITON retry (env-gated VLLM_ENFORCE_EAGER knob added to backend, local+sk3) got PAST
the hang point (compiling multimodal-bidirectional attn path) — running now behind watcher.
Overnight queue completed 02:45: stipulation_cw + stipulation3 sigs + κ ladders all landed;
zombie-free (cleanup7 kills by nvidia-smi-listed PID). All outputs pulled local.

### gemma-3 serving: CLOSED as environment-level incompatibility (2026-07-06 ~09:40)
Six strategies, three vLLM stacks, all freeze at the SAME engine-init line (immediately after
"Using FlashInfer for top-p & top-k sampling", before model load), gemma-3-12b only:
| stack | strategies | outcome |
| vLLM 0.17 (ai_usage) | default / block_size=32 / eager+TRITON | freeze |
| vLLM 0.23 (gemma4)   | TRITON / default, eager                | freeze |
| vLLM 0.24 (vllm_latest, FRESH uv env) | default / eager / VLLM_USE_FLASHINFER_SAMPLER=0 | freeze |
gemma-2 (same family), Llama, Qwen all serve fine in the same envs; gemma-4-31b fails
DIFFERENTLY (0.23 weight-name mapping). Model dirs verified complete; GPU verified clean each
attempt (cleanup7). Every timeout-guarded (no runaway hangs; GPU 7 left clean).
**Assets left behind:** /lfs/.../envs/vllm_latest (vLLM 0.24 + repo deps, reusable);
/lfs/.../models/gemma-3-{4b,12b}-it complete dirs (27b weights never downloaded — NAT64).
**Options (user decision):** (a) OpenRouter API probe for gemma-3-4b/12b/27b (~$1–2 of the
$10; hard-verdict instrument, 8B-anchored); (b) sysadmin/driver look (JIT for 262k vocab?);
(c) proceed with gemma-2 as the 3rd-family anchor (scaling law stays Llama-primary per the
same-family rule; gemma-3 clean-ladder validation deferred).

## PART V — Thickness coding adjudication + FRAME-LEVEL commitment (2026-07-06 midday)

### Coding run (name_dimension_codes.json; 267 names × 9 domains × 3 blind Sonnet coders)
Anchors: 17/17, 15/17, 16/17 (2-3 misses = defensible judgment calls, not degeneracy).
Inter-coder Spearman: safety .93–.96, thick .54–.67, practice .63–.65, crisp .25–.58 (crisp
noisiest). Merged = 3-coder mean.

### RESULT: name semantics do NOT carry commitment — the DOMAIN FRAME does
- Within humor (n=59): suppress ~ thick+safety+practice+crisp ALL null (|beta|≤.004, all
  p>.2); intercept +.032 p<.001 carries everything. NOT attenuation: per-metric suppress
  split-half reliability Qwen .74 / gemma .68 (Llama-8B .26) — real metric-level variance
  exists but name semantics don't explain it. (Nuisance check pending: suppress ~ which
  deviant_stat was assigned.)
- SAFETY double-disconfirmed (user hypothesis 2026-07-06): keyword flag (safety-flagged
  +.021 < neutral +.033; 0/19 top third; "cruelty check" +.003 ≈ fully compliant) AND coded
  regression (beta −.004 n.s., coder agreement .93+). "Punching up" itself complies.
- THICKNESS (my hypothesis) dead: cross-domain R2(dims)=.028 vs R2(domain dummies)=.327
  (12×); CW names THICKER than humor's (3.42 vs 3.14) yet CW fully compliant.
⇒ **Commitment is FRAME-LEVEL: the model defends the humor-evaluation practice as a package
(any humor criterion resists; no CW/math criterion does), not word-level thick terms.**

### PREREGISTRATION (BEFORE day-runner E2 results land; runner in flight, results unseen)
Hypothesis: humor is the panel's only VERNACULAR/folk evaluative culture; news/pr/peer/
legal/grant/cr are professional-institutional. **Prediction: pooled strong-reader suppress
≈ 0 (|mean| < .015, n.s.) in ALL SIX new domains; humor remains the sole committed domain.**
Falsifier: any institutional domain with suppress ≥ +.02 at p<.05 (probe-clustered) kills
the vernacular-culture reading and forces a humor-idiosyncratic account (e.g., pervasive
irony/non-literality in the humor register as the mechanism).

## PART VI — Performance takeaways: articulation-for-scale substitution (2026-07-06)

Question (user): what % of metrics can be fully specified (articulated) and hit iso-performance
with a SMALLER model; when can we bound that a smaller model cannot work?

### Clean estimator = same-family gemma pair (2b vs 9b; no self-recovery, no kin asymmetry)
| | humor | math |
|---|---|---|
| 9b name-sufficient | 49% | 78% |
| **2b + best rung ≥ 9b + best rung (FULL parity, 4.5× smaller)** | **32%** | **78%** |
| 2b + best rung ≥ 9b + name | 51% | 83% |
⇒ **"The price of tacitness is scale"**: in the formal domain, articulation buys back a 4.5×
size gap on ~4/5 of metrics; in the committed/vernacular domain only ~1/3 — the frame-level
tacit content is exactly what words can't substitute for.

### Pooled 9-domain rates (8B=executor column — SELF-recovery caveat; CW 100% = depressed-self
artifact, quote gemma-pair numbers instead where possible)
- **3B-BOUND (no rung in the ladder reaches .55 where the big reader works): ~1% (2/228)** —
  above 3B, articulation rescues essentially everything the 8B can do.
- **1B-BOUND: 11% pooled — but 31–38% in INSTITUTIONAL domains** (cr 34%, legal 38%, news
  31%) vs ~0% expressive/formal ⇒ small models break precisely in the codified-but-HEAVY
  domains (long instruction-following load), a different failure mode from tacitness —
  codified ≠ light.
- **Spec-HURTS (every richer rung < name−.02 at the big reader): 4% pooled; math 10%, legal
  12%, news 15%** — the name-inversion as a practical prompt warning: fuller specification
  can DEGRADE a strong judge in formal/institutional domains.

### Bounding "a smaller model just cannot work" — honest scope
Our rates are RUNG-SET-RELATIVE lower-bound failures (name…dossier; GEPA-optimized rung #14
pending could rescue some). The principled version is the upper-bound/certificate thread's
question (R'); this thread's contributions to it: (a) exportable per-metric 1B/3B-bound lists
= candidate test cases; (b) per-metric g∞ asymptotes with identified brackets + frozen 70B
predictions (rejection rule) = the scaling-law form of "no scale ever suffices at rung r."

## PART VII — Prior-art verdict on snap-back + frame-commitment (deep-research, 99 agents,
## adversarially verified, 2026-07-06)

### What EXISTS (closest relatives — the related-work section)
1. **Word/symbol-level override resistance** (closest empirical relatives, both 2026 preprints):
   logic-operator redefinition ('⊗ means OR') — 33.3% of errors are semantic override, models
   revert to pretrained operator meanings (arxiv 2602.17520); lexical-prior strength predicts
   override failure across 11 models 1B–9B, p<1e-9, incl. word-level nonce remapping
   (arxiv 2606.07555). WORD/SYMBOL level only; no criterion/rubric level.
2. **Scale-emergence of override** (Wei et al. 2023, arxiv 2303.03846 + 2025 corroboration):
   flipped-label ICL — small models can't override priors, large can; 1–12B "exactly zero"
   override. But that's contradictory-EXEMPLAR mapping, not explicit stipulative redefinition
   (different manipulandum from ours).
3. **Knowledge-conflict literature** (Xie et al. 2305.13300 Adaptive Chameleon; WikiContradict;
   EMNLP-2024 survey 2403.08319): entirely FACTUAL/propositional (entity substitution,
   proposition flipping). No concept/criterion redefinition; no nonce controls.
4. Hate-speech-definition sensitivity (Melis 2506.18576; Fasching-Lelkes): models respond to
   DIFFERENT EXPLICIT definitions / produce inconsistent moderation — adjacent (definitions
   matter) but does NOT test resistance/snap-back. (Two overreach claims about these papers
   were KILLED by the adversarial verifiers — quote them only in the weak form.)
5. Thick concepts: one 2026 preprint applying Williams to benchmarks; semantic externalism
   (Putnam/Burge/Kripke) NOT operationalized as a measured LLM quantity; enregisterment
   (Agha): no computational measurement found.

### What appears GENUINELY UNCLAIMED (verified negative space)
- **Criterion/construct-level stipulative redefinition** (evaluative criteria, not words/facts).
- **The nonce-name control** at any level above single words — i.e., isolating community-
  semantic commitment from rule-execution capacity. (Their literature measures when models CAN
  override; ours measures RESISTANCE DESPITE demonstrated capability — the nonce arm proves
  capability. No precedent found.)
- **Frame/domain-level resistance** (all criteria of one evaluative culture resist, none of
  another, invariant to name semantics): "no literature precedent."
- Operationalized semantic deference (division of linguistic labor as a measurement).
Positioning: our two claims sit in verified open territory; cite 1–4 as the word-level and
fact-level neighbors that predicted pieces of the phenomenon without the construct, the
control, or the frame-level localization.

## PART VIII — PREREGISTRATION CONFIRMED: frame-commitment is vernacular-specific (2026-07-06 eve)

Day-runner 6-domain E2 landed (verified windows 6/6 match=1.0; readers Llama 1B/3B/8B + Qwen-7B
+ gemma-2-9b where it served; gemma-2 engine FAILED on legal/grant/cr — Llama+Qwen complete
everywhere). Probe-clustered suppress, strong readers:

| domain | L8B | Q7B | G9B |
|---|---|---|---|
| humor | **+.030 (p=.014)** | **+.026 (p=.012)** | **+.041 (p<.001)** |
| cw | −.002 | −.003 | −.008 |
| math | −.019 (p=.008) | −.003 | −.012 |
| news | −.018 (p=.022) | −.025 (p<.001) | — |
| pr | −.010 | +.006 | — |
| peer | −.045 (p<.001) | −.003 | — |
| legal | +.005 | +.004 | — |
| grant | −.006 | −.017 | — |
| cr | −.010 | −.023 (p=.004) | — |

**Falsifier (any institutional domain ≥ +.02, p<.05, strong reader): NOT TRIPPED — prereg
(notes PART V, registered before these data) CONFIRMED.** Humor is the ONLY positive-suppression
domain, in all three families. Beyond confirmation, a three-way signature: VERNACULAR names
RESIST redefinition (+, humor), INSTITUTIONAL/CODIFIED names ASSIST compliance (−, math-style
transparent pointers now in peer/news/cr too), craft-neutral ≈0 (CW/pr/legal/grant). Name
commitment = enregisterment of a folk evaluative culture, not a property of names in general.
Caveats: gemma-2 missing on 3 domains (engine failures, retry cheap); magnitudes small (+.03);
1B tier remains artifact-regime (its +.013 p=.002 on cr is below the falsifier bar and the tier
is demoted).
