# CW unified grid roadmap — upper bounds × codability × tacit knowledge in one design

*2026-07-01. One task (creative writing), one grid, three deliverables. Theory:
`2026-06-18__prompt-optimality-theory.md` §12.6; `2026-07-01__codability-audit-and-proposal.md`;
backlog §1 "per-executor tacit-knowledge profile (REVISED)". Instruments: `value_certificate`,
`methods/codability/`, `recon_channel`, `run_alpha_probe --orbit-target`.*

## 0. The one-grid thesis

All three questions read off the SAME longtable of verdicts — (probe × message × reader-model ×
pass) — sliced three ways:

| question | slice | headline |
|---|---|---|
| **Q1 upper bounds** | per (metric, tier): criteria-pool signatures → certificate | which metrics have a **certified value-based bound** (`OPT_Ω + ε`) vs only a descriptive behavioral census (B_E), per §12.6.6 verdict |
| **Q2 codability** | per metric: stratified message rungs → transfer matrix → levels | the **codability map** (fraction of CW metrics at L0–L4) + Δ_context (indexicality of the community's evaluative language) |
| **Q3 tacit knowledge** | per reader: decompression curves + strong−weak gap | per-model tacit profile; **enculturation scaling** |

**All three scale with model size on the same-family primary ladder** (user asked for ≥1/3): Q1 →
Δ(E) staircase (flat = process-relative tacitness evidence); Q2 → level migration with size (does
L4 → L3 → L1?); Q3 → sparse-decompression vs size, dissociated from Δ(E). Slopes are read WITHIN the
Llama ladder only; Gemma/Qwen are replication panels (see the §1 scaling protocol). The interesting
anthropological outcome is the dissociation: **Δ(E) flat while decompression grows** = telling stays
uncompressible while showing/pointing gets cheaper with scale (Polanyi realized as two curves).

## 1. Fixed assets (freeze once, everything shares them)

**Corpus & probes.** `writingprompts_modeling_clean.csv.gz` (96K, balanced, sk3). Frozen stratified
probe pool: **4 genre strata × 250 = 1,000 items** (500 train / 500 held per stratum via
`codability.strata.stratified_split`, seed-frozen). Strata via a **local categorical judge**
(Llama-70B one-word genre classification into {horror/thriller, scifi/fantasy, romance/drama,
comedy/other} — metadata-free corpus, so categorical judge is the sanctioned route; do NOT use
topic models/embeddings; do NOT spend GLM here). Also back-tag the existing rescore probe set with
the same strata (same corpus → per-stratum certificate columns come free).

**Metrics.** Q1 runs on the FULL CW R3 bank — 42 general-bucket grandparents (371 R2 children;
`outputs/hierarchy/creative-writing_general_r3_expanded.json`) — because certificates are CPU
post-hoc on rescore checkpoints. The message grid (Q2/Q3) pilots on **15 real + 5 planted = 20**:

- 4 craft/mechanics (expect L0/L1: e.g. dialogue-punctuation/formatting side of "Dialogue craft")
- 5 generic taste (expect L3/L4 candidates: "Pacing, momentum, and suspense", voice, emotional payoff)
- 4 genre-specific (the indexicality axis: horror dread, comedic timing — expect L2)
- 2 borderline/provenance-diverse (emic scraped-rubric vs etic free-gen)
- **5 planted controls** (non-negotiable, §12.6.7 + codability §4.3 + Face-2):
  C1 planted code rule → CODIFIABLE + L1; genre-indexed rule → **L2-not-L4**; pointer-concept
  ("judge as a seasoned horror editor would") → large strong−weak reader gap; private-code
  (writer=reader vs cross-reader collapse); noise (shuffled verdicts) → NO-SIGNAL.

**Reader/executor ladder** (known-good sk3 recipes; 1 GPU, tiers sequential, HOME pinned to /lfs):

> **SCALING PROTOCOL (2026-07-01 clarification — BINDING; agents keep getting this wrong).**
> Scaling-law slopes are **within-family objects**: the classic training laws (Kaplan 2020,
> Hoffmann/Chinchilla 2022) hold architecture + data recipe fixed, and the *observational* variant
> that does span families (Ruan–Maddison–Hashimoto 2024) only works by first extracting latent
> capability axes — raw size-vs-metric curves across families are confounded by family offsets
> (data mix, tokenizer, post-training/RLHF judge-bias), and the x-axis itself is ill-defined
> cross-family (Qwen3.5-122B-A10B is MoE with **10B active params** — neither "bigger" nor
> "smaller" than dense Llama-70B on one axis). With our 3–5-point staircases, one family offset can
> flip a flat↔shrinking verdict. Therefore: **primary Δ(E) and decompression staircases are
> SAME-FAMILY; other families are REPLICATION panels (their own family-indexed points or
> mini-staircases), NEVER pooled into one slope.** The earlier mixed-family B_E scaling
> (llama-3b/8b + qwen-122b) is demoted to descriptive — do not cite it as sanctioning mixed
> ladders. (Consistent with §12.6.5's existing cross-tier-rank distrust; now also a §12.6.5 rule.)

| tier | model | role |
|---|---|---|
| weak anchor | Llama-3.2-1B | Face-2 strong−weak split; joins the primary ladder ONLY if it passes the Phase-1 T_g gate (unlikely) |
| **primary ladder** | **Llama-3.2-3B → Llama-3.1-8B → Llama-3.3-70B-FP8** (official nvidia ckpt) | the Δ(E) + decompression staircase — same family, all dense, shared tokenizer/lineage (3.1/3.2/3.3 recipe drift = stated residual caveat) |
| replication panel | Gemma-4-31B (dedicated env); Qwen3.5-122B-A10B-FP8 (`VLLM_USE_FLASHINFER_MOE_FP8=0`; subject role — the not-for-EVAL caveat doesn't apply) | family-indexed replication: does the within-Llama verdict replicate? separate points, never pooled |
| frontier spot | Claude via Max-plan subagents | 2–3 metrics × subsample only |

**Messages per metric (the channel rungs).** (a) full rubric = the rich target definition;
(b) **name-only** = the R3 grandparent name (free — the "rich concept, sparsely worded");
(c) L-capped rubrics L ∈ {32, 128} tokens (`recon_channel` detail directives; writer = Qwen-122B,
cached; second writer = Llama-70B for the private-code/cross-writer arm);
(d) rules+exemplars (k=8 balanced, from TRAIN split only);
(e) per-stratum conditional rubrics r_g (induced from stratum-g train pairs) + one pooled r_global.

**GLM budget** (quota binding): ONLY the semantic quotient judge for metrics whose certificate
verdict flips on partition choice — capped ≤300 pairs/metric, expected ≤2–3K calls total.

## 2. Phases

**Phase 0 — freeze (CPU + one local-70B batch; ~half day).**
Probe pool + strata + splits frozen; rescore probes back-tagged; 20 grid metrics picked by
provenance rule; planted-control rubrics written. QA gate: stratum sizes ≥200; label balance
within strata; `probe_balance_guard` per metric (genre-specific metrics will be UNDEFINED in some
strata — that's the design working, record it). **Certificate regime pinned (theory §12.8.0 T1,
2026-07-01): all grid certificates run `stopping='anytime'` (the code default) — the grid WILL
re-issue after UNDERSAMPLED verdicts, which is optional continuation, and only the anytime δ-spend
(doubling-checkpoint × order-statistic union, a-priori B-cap) survives that. `stopping='fixed'`
is allowed only for a one-shot run whose probe budget was declared before any data was seen.**

**Phase 1 — targets + ceilings per tier (GPU; ~1 night/tier).**
Per tier × 20 metrics × 1,000 probes: full-rubric m̄_ω with **orbit 4 forms + 1 repeat pass**
(≈100K calls/tier). Yields: M_i(E), per-stratum test–retest T_g(E), form-invariance/flip-rate.
QA gates: judge score-distribution collapse check per metric×tier (free-form+parse fallback);
`frac_near_constant_sig`. **Kill gate:** if T_g < 0.1 for most metrics on a tier, the tier is not a
usable reader — fix before spending Phase 3 on it.

**Phase 2 — certificates per tier (CPU now; fill missing tiers opportunistically).**
`value_certificate --scaling small:…,mid:…,large:… --ceiling <bits>` over the R3 rescore
checkpoint dirs (in flight) → per metric × tier: {H_M, OPT_Ω, ε, frac_H, tail_frac, γ̂, f₁/N,
verdict}. Judge-quotient rerun (GLM, capped) only where the verdict is partition-sensitive.
Targeted `adversarial_saturation` on ε-small (CODIFIABLE-candidate) metrics. Per-stratum ε_g =
same certificates on stratum-restricted probe columns (free after back-tagging).
**Q1 deliverable:** the trichotomy over all 42 R3 metrics —
(i) **CERTIFIED-BOUNDED** (CODIFIABLE), (ii) **CERTIFIED-DEEP** (bound exists, tail-heavy — the
tacitness candidates), (iii) **NOT-YET-BOUNDABLE** (UNDERSAMPLED / FORM-DOMINATED / INDETERMINATE,
each with its named cure). B_E census reported alongside as descriptive-with-error-bars only
(Lemma 12.6.0).

**Phase 3 — the message grid (GPU; ~1 night/tier).**
**RUNG REDESIGN 2026-07-02 (user decision): rungs are ARTICULATION TYPES, not lengths** — prior
result showed length alone is a poor predictor, and a length axis confounds tokens with content.
Ladder: `name` (pure index) → `definition` (intension: "which means…") → `explanation`
(mechanism/procedure: "which happens when… / recognize it by…") → `full_rubric` (the R3
merged_description, the Face-1 anchor) → `exemplars` (k± shown, no words) → `dossier`
(definition+explanation+exemplars = telling+showing ceiling). definition & explanation are
LENGTH-MATCHED (~≤50 words) so type ≠ length; token counts reported as nuisance covariate; each
verbal rung orbit-averaged over deterministic reformulations (form control). Gap readings:
name→definition = lexical/indexical; definition→explanation = knowing-that→knowing-how (Ryle);
explanation→exemplars = ostensive (Polanyi); exemplars/dossier plateau below T_g with certified
saturation = tacit-within-frame (L4).
Per tier × 20 metrics × 6 rungs × forms × 500 held ≈ comparable budget. Batch per tier resident
once (register messages as prompt versions with operators NAME/DEFINITION/EXPLANATION/RUBRIC/
EXEMPLAR/DOSSIER/STRATUM_g).
Requires the one new driver (`methods/codability/run_decompression_grid.py`, ~a day, reusing
recon_channel induction + scale batching + codability adjudication).

**Phase 4 — adjudication (CPU; ~1 day).**
- Q2: transfer matrices M[g→g'] → R_g, Δ_context, diagonal-dominance/blocks; mixed model
  (a_i, b_g, Var[(ab)] = indexicality variance); `profile_level` per metric per tier → the
  **codability map** + FRAGMENTED flags routed to the re-clustering audit.
- Q3: decompression curves R_E(L) normalized by R_E(rich); **strong−weak gap at name-only** per
  metric per tier; cross-writer κ (private-code check).
- Scaling: Δ(E) staircase + level-migration table + decompression-vs-size — trend verdicts only,
  **no fitted asymptotes** (§12.6.5); dense-CW ceiling is still climbing → treat lowerCI(C) as
  achieved floor and right-censor language accordingly.
- **Global kill gate:** any planted control off its expected cell → stop, fix, rerun before ANY
  real number is quoted (direction-of-error flip: under the thesis, instrument weakness inflates
  the desired gap — the controls are the credibility).

## 2b. TALK CUT — minimal tacit-knowledge readout, within-Llama only (added 2026-07-01; ~2 GPU nights + 2 CPU days)

*The smallest package that supports a defensible tacit-knowledge statement in a talk. Everything here
is a subset of Phases 1–4 — no new measurement targets; the full grid supersedes it row-for-row.*

**STATUS 2026-07-02 overnight:** level-dispatch BUG found+fixed (rescore_executor pulled r2_groups
descs for R3 ckpts → aligned 3b/8b M_i measured the WRONG metric; own-pair vs cross-pair spearman
0.725 vs 0.744 = indistinguishable → v1 aligned dirs DO-NOT-USE for M). Running: corrected
`aligned_8b_orbit_v2` + `aligned_3b_orbit_v2` retargets (GPU0), 70B v2 full rescore w/
`--form-invariance-n 12` (watcher-launched GPU1 00:15, ~1.5–2 nights for all 44, resumable),
10-fill (gi 20,36,37,39,42,46,50,55,56,58; GPU7) + 16-fill (GPU2). `run_decompression_grid.py`
written + fake-smoke green (TYPE rungs). Qwen relaunch + 3B forminv pass + focus-4 picks = pending
user decisions.

**Selection.** Metrics: **2 craft + 2 taste + planted rule + noise** (pick craft/taste by Day-0
verdict contrast from the §1 pilot list — e.g. dialogue-formatting vs pacing/voice). Strata: 2 × 250
(keeps the indexicality control; drops the other two). Probes: 500. Tiers: Llama-1B (weak READER
only) / 3B / 8B (existing rescore checkpoints) / 70B (one fill night). **All-Llama; replication
panel, emic layer, Q2 transfer matrices, per-stratum rubric rungs, frontier spot all OUT.**

- **Day 0 (CPU, same-day).** Phase-2 certificates on the EXISTING 3B/8B rescore checkpoints over all
  available R3 metrics (`value_certificate --scaling … --stopping anytime`; ceiling = dense-CW
  lowerCI as achieved floor). → Q1 trichotomy at 2 tiers, immediately quotable; pick the focus-4
  from it. FIRST verify checkpoint coverage (which metrics/tiers the rescore dirs actually contain)
  before selecting. In parallel: write the Phase-3 driver
  (`methods/codability/run_decompression_grid.py`, ~1 day — the one missing piece of code).
- **Night 1 (GPU, 70B).** Third same-family staircase point — the §12.6.5 three-point minimum for
  any slope verdict: focus-4 + planted rule criteria-pool sigs (~80 criteria × 300 probes × 5
  ≈ 100K calls) + m̄_ω targets w/ orbit forms + retest (~15K). CPU after: 70B certificates →
  **Δ(E) 3-point within-Llama staircase** (Face 1 complete).
- **Night 2 (GPU, all readers).** Decompression mini-grid (Face 2): 5 metrics × 6 message rungs
  (name / definition / explanation / full rubric / exemplars / dossier — TYPE ladder, see Phase 3
  redesign; NOT length-capped) × 500 probes × 2 passes × readers
  {1B, 3B, 8B, 70B} ≈ 60–70K calls (small tiers stacked; 70B chunk dominates). Plus the Δ_comp
  mini-arm: 2 metrics × 6 composed variants × 500 on 70B ≈ 6K (the "tacit lives in the SAYING"
  slide). Messages written ONCE by local 70B — not GLM (quota).
- **Day 3 (CPU).** Adjudication → the five talk artifacts: (i) trichotomy table
  (CERTIFIED-BOUNDED / CERTIFIED-DEEP / NOT-YET-BOUNDABLE); (ii) Δ(E) staircase, trend verdict only
  (flat ⇒ "no capability trend closes it — strongest process-relative evidence"; shrinking ⇒
  right-censor: "not yet articulated at 70B"); (iii) R_E(L) decompression curves craft-vs-taste +
  **strong−weak reader gap at name-only** (pointing needs a capable reader); (iv) the prize
  dissociation check — Δ(E) flat while decompression cheapens = "telling stays incompressible while
  pointing gets cheaper" (Polanyi as two curves); (v) the controls slide (planted rule → CODIFIABLE,
  noise → NO-SIGNAL, XOR under F₁ vs F₂, tail-XOR breaker's 3.1× under-cover as the honesty flank,
  Δ_comp reported alongside).

**Claim language (binding, §12.6.4/§12.6.5/§12.8):** checklist-OPT_F relative to the named process;
ε anytime-sound (T1); within-Llama trend only; right-censor if shrinking; Δ_comp never folded into ε;
C = achieved floor (dense-CW still climbing). Kill gates unchanged: T_g < 0.1 blocks a tier's Face-2
rows; any planted control off its cell blocks every quoted number.

## 3. Budget summary

| block | calls | wall (1 GPU, batched) |
|---|---|---|
| Phase 0 strata (local 70B) | ~2K | <1h |
| Phase 1 targets ×4 tiers | ~400K | ~4 nights |
| Phase 2 certificates | 0 GPU (CPU on checkpoints) | ~1 day CPU |
| Phase 3 grid ×4 tiers | ~360K | ~3–4 nights |
| GLM (quotient judge only) | ≤3K | — |
| frontier spot-check | ~5K | subagents |

≈ 0.8–1M short judge calls ≈ 7–8 sk3 nights on one B200 (tiers sequential; stack small tiers).
Code to write: the Phase-3 driver + message-induction script (~1 day); everything else exists.

## 4. Order of operations & dependencies

0 → 1 and 2 in parallel (2 depends only on rescore checkpoints) → 3 (needs 0's messages + 1's
targets) → 4. Earliest meaningful readout: **Q1 trichotomy lands as soon as the current rescore
finishes** (Phase 2 is same-day CPU); Q2/Q3 land after the grid nights. If GPU time gets tight,
degrade gracefully: **drop the replication panel first (Gemma/Qwen), then L32, then halve probes on
70B — the 3-point same-family primary ladder is untouchable** (below 3 within-family points there is
no slope verdict, only descriptive points). Replication panel runs the 10-metric core subset at half
probes (~+1 night total for both models) when it runs at all.

## 5. ADDENDUM (2026-07-01, after the Ω-composition critique) — composition-gap & combiner arms

The certificate bounds **checklist-articulability under a linear readout**, NOT prompt-space OPT:
the joint-prompt channel is not a function of unit verdicts (no DPI), prompt performance is
non-monotone in criteria, and additive-logistic CE is blind to conjunctive value (XOR). Three
additions convert the hidden assumptions into measured quantities:

1. **Composition-gap arm (Δ_comp), Phase 3:** per metric, execute the head criteria (a) as
   separate units + feature combiner (= the certified OPT_Ω object) vs (b) ~4–6 composed rubric
   prompts (order × phrasing × ±persona framing) on the same 500 held probes × tiers.
   `Δ_comp = achieved(b) − OPT_Ω`. Δ_comp ≈ 0 → the unit model is empirically adequate;
   Δ_comp > 0 on taste metrics with ≈ 0 on craft = the "tacit lives in the SAYING" finding.
   Cost: ~5 prompts × 20 metrics × 500 × 4 tiers ≈ 200K… trim to 2 tiers + 10 metrics ≈ 50K calls.
2. **Holistic adversary, Phase 2:** `adversarial_saturation` probe set must include GEPA-optimized
   WHOLE prompts + pointer/persona prompts (gepa_pr as attacker), not criteria only — converts
   "assume no prompt-space synergy" into "our best optimizer couldn't find any."
3. **XOR planted control (combiner-class limitation):** metric = parity of two pool criteria; the
   current machinery SHOULD fail it (head never picks the pair, cert says saturated) — land it as
   a documented limitation or add an interaction-capable reference combiner on the head.
4. **Tail-XOR breaker (A3's anti-conservative escape — IMPLEMENTED 2026-07-01):** M = rule ⊕
   parity(a,b,c) at ~1% flip rate, triple pairwise-independent and RECAPTURED — every census
   instrument reads zero by construction and the certificate issues CODIFIABLE at ε = 0.031 while
   0.096 bits hide jointly (3.1× under-cover). Travels with every CODIFIABLE verdict; the
   composition-covering `adversarial_saturation` gate is the only in-pipeline detector
   (`adv_saturated=False` blocks the verdict — asserted in the test).

Claim-language re-scope of §12.6.4/R12 — **APPROVED + APPLIED 2026-07-01**: theory doc now states
CHECKLIST-OPT_F with the three named escapes (zero-mass tail / composition channel / combiner class,
scorecard R16–R18); code carries `combiner=` (F₁/F₂ pair stage), `eps_bits_adv` (order-adverse band,
n_orders=8 default, `decide` reads the adverse end), `composition_gap.py` (Δ_comp +
holistic_probe_prompts), and `adversarial_saturation(probe_kinds=…)` → `covers_composition`.
Planted-control set in §1 grows by three (XOR + composition + tail-XOR breaker) — all in the test
suite (113 green). Horizon extrapolation beyond c = 1 is now certified machinery too:
`osw_horizon_value` (§12.8.7 I1, provable to c ≤ ln N) — grid certificates may report a c = ln N
column alongside c = 1 at zero extra judge cost (CPU post-hoc).

## 6. The EMIC layer — how the 690,998 matched community norms enter the grid

*(`methods/metric_implementer/norm_to_metric_matching.md`: per-corpus norm signals extracted from
real community speech — forum comments, code reviews, opinions — matched to the R2/R3 metric
catalogs via the trained CE cascade, faithfulness-audited. For CW: wp_comments + litbench signals →
the 371-R2/42-R3 catalog.)*

This is the **emic articulation record** — what the community actually says when it tells itself
what matters — against our LM-generated criteria (the etic instrument). It feeds all three
questions:

**6a. Emic Ω arm — process-independence of the upper bound (Q1; the strongest available upgrade to
the certificate's epistemics).** For each grid metric, the matched signals ARE a second,
human-authored criterion pool. Run the SAME §12.6 certificate on it: `OPT_Ω^emic + ε^emic` vs the
LM-process `OPT_Ω^etic + ε^etic`. Readings: emic ≈ etic ≪ ceiling ⇒ two maximally different
articulation processes (decades of community speech; saturated synthetic proposal) exhaust at the
same point — the wall is in the CONCEPT, not the articulator, upgrading process-*relative* tacitness
toward process-*general*. LM ≫ emic ⇒ the community under-articulates relative to what is
articulable (itself a finding about articulation practice, not tacitness). Cost: execute ~30
deduped top-matched signals × 20 metrics × 500 held probes on the reference tier ≈ 300K calls ≈ 1
night (Phase 3.5); certificate is then CPU.

**6b. The articulation deficit — density vs value (Q2 external validation; the headline
anthropological scatter).** CPU-only on existing matches: per R3 metric, community articulation
DENSITY (matched-signal count; distinct-phrasing richness via MinHash — embedding-free) vs the
metric's VALUE share (certificate head gains / dense ceiling contribution). Codability-map
prediction: L1 metrics are said often and redundantly (cheap to say → said everywhere); L4
candidates are **load-bearing but unsaid** — high value share, low stated density, or stated only
by ostension (quotes/examples). Density × level correlation = the external, human-grounded validity
check of the whole codability map; the deficit metrics are the tacitness shortlist.

**6c. Community speech as the sparse messages (Q3; Brown–Lenneberg with real utterances).** In the
decompression grid, add a message rung `emic-verbatim`: the community's own top-matched norm
statements executed as rubrics, unedited. R(emic-verbatim) = can the community's own words,
re-executed, reproduce the practice? The strong−weak reader gap ON COMMUNITY PHRASINGS = how much
enculturation the community's own shorthand presupposes — the indexicality of the community's
evaluative language measured on genuine speech, not synthetic rubrics. (+1 rung ≈ +10% Phase-3
cost.)

**6d. The community's own norm flux (the purest anthropological census).** Many corpora are dated:
run the capture-recapture machinery on the COMMUNITY signal stream over time — the discovery curve
of the community's evaluative lexicon ("has this community said everything it will say about
pacing?"). Note the inversion: the count census was demoted for LM pools (Lemma 12.6.0 — synthetic
singleton inflation), but on the human stream it IS the object of study — descriptive linguistics
of norm vocabulary growth, reported with the §12.6.6 order/quotient bands. CPU-only.

**Caveats (binding):** matching is noisy — CW recall@10 0.23 with a documented gold artifact, CE
top-10 ~67% sensible — so (i) match at **R3** (mitigates the R2 over-split), (ii) gate signals by
CE score + a small spot-check per metric before any certificate run, (iii) treat 6a/6c as
noisy-channel LOWER bounds on emic articulability (bad matches only depress emic OPT — conservative
in the direction that hurts, i.e. against "the community said it all," so flag asymmetrically),
(iv) giant-corpus tails beyond the 20k cap are base-BGE only — exclude them.
