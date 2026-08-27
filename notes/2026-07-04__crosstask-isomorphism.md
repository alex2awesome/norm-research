# Cross-task iso-morphism: scale-out day 1 (2026-07-04)

**Goal (user, 2026-07-04):** "scale up the analysis of tacit knowledge across many different
tasks, finding significant amounts of iso-morphism between pairs." Interpretation confirmed
with user: ALL THREE readings, staged — (A) pairs of TASKS (cross-task concept recurrence +
profile transport), (B) pairs of FORMS (name↔definition↔dossier interchangeability), (C) pairs
of MODELS (iso-performance chains) last. 1 GPU sanctioned alongside the 70B rescore.

## A1. R3-level concept-sharing matrix, all task pairs, permutation-tested

`methods/metric_implementer/experiments/crosstask_sharing_matrix.py` →
`notebooks/data/two_faces_20260702/crosstask/sharing_matrix_r3.json`.

Units = R3 metrics from `outputs/hierarchy/<task>_general_r3_expanded.json` (same level as
certs/grids). Embeddings bge-large ("name: description"). Statistic = symmetrized mean max-cos
coverage; null = size-matched metric sets drawn from all OTHER tasks (controls "criteria all
sound alike"); BH-FDR over 36 pairs (9 tasks kept; patents 0 R3 groups, notice-and-comment 5 —
skipped as too thin).

**5/36 pairs significant (all q=0.0036):**

| pair | z | family reading |
|---|---|---|
| creative-writing × humor | **+11.3** | narrative craft |
| news-homepages × press-releases | +5.7 | news/media communication |
| legal × math-stackexchange | +5.1 | argument/proof rigor |
| grant-funding × peer-review | +3.7 | research-merit evaluation |
| math-stackexchange × peer-review | +3.7 | rigor again |

Anti-affinities are informative too: CW × press-releases z=−5.7 (generically "writing", but
their criterion vocabularies actively avoid each other — expressive craft vs institutional
communication). Top pairs have strong face validity (legal "reader-centric signposting,
roadmaps" ↔ math "macro-organization, reader navigation, signposting" cos .83; news
"plain-language clarity" ↔ PR "plain language, minimal jargon" .92; venue-fit ↔ venue-fit;
COI-disclosure ↔ COI). NB: embedding-grade — individual pairs still need the judge protocol
before being called matches. The rigor-family shared criteria are conspicuously
mechanical/structural — the new domains will populate the empty MECHANICAL cell.

## A1b. ALL FIVE significant pairs judge-verified (same-day follow-up)

Same cascade (hierarchy-R3 texts for non-grid tasks; 2 independent judges; substitution test;
manual adjudication of 2-vs-not disagreements — umbrella criteria like peer-review's "writing
clarity, organization, audience fit" held at RELATED as subsumption):

| pair | z | judged | verified matches | % of judged | % of bank×bank |
|---|---|---|---|---|---|
| CW × humor (grid-level) | +11.3 | 370 | **30** | 8.1 | 1.09 |
| news × press-releases | +5.7 | 216 | **11** | 5.1 | 1.05 |
| legal × math-SE | +5.1 | 116 | **14** | 12.1 | **5.56** |
| grant × peer-review | +3.7 | 94 | **5** | 5.3 | 2.40 |
| math-SE × peer-review | +3.6 | 112 | **5** | 4.5 | 1.83 |

**65 verified same-criterion pairs across the 5 significant task pairs.** legal×math is the
DENSEST (5.6% of all possible pairs are the same criterion — the rigor family repeats itself
hard: signposting, modular decomposition, audience-fit all match verbatim-level). Judge
agreement κ: .69/.66/.68/.55/.44 (math×peer lowest — one judge umbrella-matched; adjudicated).
Judged files: `crosstask/<pair>_judged.json`.

**Negative control (grant×legal, matrix z=+0.20 n.s., identical protocol, κ=.74):
4/93 verified matches (4.3% of judged) — NOT zero, and all four are UNIVERSAL-CRAFT criteria**
(reviewer-skimmability↔signposting/roadmaps, mixed-audience accessibility↔audience-adapted
writing, clarity/organization↔macro-organization/flow). Honest layered reading:
1. There is a **universal communication layer** (~4% of TF-IDF-filtered candidates match in
   ANY pair) — signposting, audience-fit, clarity/organization recur in every writing task.
   This is exactly the layer the sharing-matrix null subtracts (the pooled-other-tasks null
   also contains these criteria), which is why grant×legal is z≈0 despite real matches.
2. **Significant pairs add a pair-specific layer** — their matches include domain-specific
   criteria the control lacks (misdirection-reveal, escalation discipline, COI disclosure,
   citation rigor, modular proof structure, constructive content).
3. Caveat for the paper: **raw verified-match rates do NOT separate the weakest significant
   pairs from the control** (grant×peer 5.3%, math×peer 4.5% vs control 4.3%); what separates
   them is the z (distributional affinity over ALL metrics, universal layer nulled out) and
   the CONTENT of the matches. Two distinct notions of task-pair isomorphism: shared exact
   criteria (legal×math, CW×humor excel) vs criterion-space affinity (all 5).
Control file: `crosstask/grant_funding_legal_CONTROL_judged.json`.

**Layer tagging of all 69 verified matches** (single-rater exploratory, `crosstask/
match_layer_tags.json`; UNIVERSAL = clarity/economy, organization/signposting, audience/venue
fit, mechanics, generic evidence, figure quality; DOMAIN = pair-family-specific checks):

| pair | universal | domain-specific | domain frac |
|---|---|---|---|
| CW × humor | 8 | **22** | **.73** |
| news × PR | 5 | 6 | .55 |
| legal × math | 9 | 5 | .36 |
| math × peer | 3 | 2 | .40 |
| grant × peer | 5 | 0 | .00 |
| **CONTROL grant × legal** | 4 | **0** | **.00** |

**The control's matches are 100% universal-layer; 35/65 significant-pair matches are
domain-specific — the control has ZERO.** CW×humor's iso-morphism is overwhelmingly
domain-specific (misdirection mechanics, escalation, callbacks, show-don't-tell — narrative
machinery, not generic craft). grant×peer patterns WITH the control on content (all universal)
— its z is distributional affinity, not exact-criterion sharing; treat it as the weakest of
the five. Caveats: single-rater tags; class boundary judgment calls (e.g., venue-fit counted
universal). A 2-tagger pass can harden this if it becomes a paper claim.

## A2. Flagship pair judged end-to-end: creative-writing × humor

`crosstask_match.py` cascade: TF-IDF (word+char blend) reciprocal top-5 → 370 candidates → 2
independent Claude judges on the substitution test (0/1/2), κ=0.691, 315/370 exact, ZERO 0↔2
confusions → 14 two-vs-not disagreements manually adjudicated (4 promoted) →
**30 same-criterion pairs** (20 CW × 22 humor gi's; 18 in the greedy 1-1 arm).
Files: `crosstask/cw_humor_{candidates,judged,isomorphism}.json`.

Profile transport on matched pairs (against `isomorphism_census.json` per-metric profiles):

- **Concept TYPE travels: label agreement 83% (25/30), perm p=0.0017** (1-1 arm: 72%, p=0.04).
  The taste/craft character of a criterion is concept-intrinsic, not task-idiosyncratic —
  first quantitative iso-morphism-between-tasks result.
- Continuous profile correlations (name_frac, gap_def_name, dossier): positive for definition
  (ρ=.25 all-pairs / .35 1-1) but n.s. at n=30/18. Underpowered at one task pair — this is
  precisely what more grid-bearing pairs are for (attenuation: each profile is itself an
  estimate; split-half ρ of the underlying census is .70/.96).

## B. Iso-morphism census over FORM pairs (CW+humor grids)

`methods/metric_implementer/experiments/isomorphism_census.py` → `isomorphism_census.json`.
rec(rung) = rung bits / H_self (fraction of the metric's OWN full-rubric verdict a reader
recovers from that form). Size-robust within-size shape, small-reader (1B/3B/8B) average:

| form | CW | humor |
|---|---|---|
| name | .414 | .289 |
| definition | **.465** | **.438** |
| explanation | .389 | .391 |
| exemplars | .152 | .093 |
| dossier | .154 | .159 |

- best_rung = definition for 28/46 CW, 35/57 humor; explanation second in humor.
- name↔definition iso-morphic (|gap|≤.05) for only ~27% — humor's definition gain (+.15) is
  real. The bare name carries ~.3–.4 of the metric's own decision.
- taste ≈ craft on THIS measure (MW n.s. pooled) — consistent with the earlier finding that
  taste/craft differ on match-to-70B COST, not own-verdict recovery.
- Cross-SIZE levels are confounded (each size targets its own verdict); only within-size
  shapes and same-size cross-task comparisons are licensed.

**RUNG-CONSTRUCTION AUDIT (2026-07-04, same day — claim correction):**
- `exemplars` is **ostension-BY-DESIGN**: "Judge by these examples ONLY." + k=2 pos / 2 neg
  400-ch truncated snippets (CW exemplars all exactly 1733 ch — pure template). Its collapse
  at every reader size is therefore a THEORY result — **pure ostension at tiny k fails the
  census** (two-faces: ostension = census violation) — NOT "examples don't help".
- `dossier` is CONTAMINATED: all 46/46 CW + 60/60 humor dossiers embed the exemplars block
  **including the "ONLY" instruction**, i.e. the dossier orders the reader to discard the
  definition+explanation it just provided. **The earlier "richer articulation is regressive"
  reading is RETRACTED** — the dossier rung does not measure full articulation. What stands:
  the clean name→definition→explanation shape (definition = peak).
- Owed: `dossier_v2` rung = definition+explanation+exemplars with the exemplars reframed as
  "Illustrative examples:" (no ONLY) → re-score on the small readers (cheap GPU pass, queue
  behind the domain sweeps). Only then can "does full articulation beat the definition?" be
  answered.

## Infrastructure landed today

- **Press-releases R3 sweep RUNNING on sk3 GPU1**: PID 2587370, 42 metrics, byte-comparable
  recipe (`--families glm_a/b/c glm-4.7 zai_anthropic`, M_freegen 600, n_probes 300,
  orbit 4, forminv 12, 8B executor; verified: init 6.47s, GPU1 157GB, prompts flowing).
  First default-families launch was killed at 1m29s (ensemble mismatch with CW/humor refs —
  apples-to-apples) — aborted log kept at `r3_pr/sweep.log.aborted_default_families`.
- **Chain watcher** (PID 2596989, `r3_pr/chain_next_domain.sh`): when PR exits with ≥35/42
  checkpoints → auto-launches the news-homepages sweep (25 R3 groups) on the same GPU;
  status → `r3_pr/chain_status.log`.
- Enablement edits: `config.py` TASK_PRESETS["press-releases"], `manifest.py` press_releases
  entry (deconfounded parquet, 72,315 rows; probe texts spot-checked via the driver's own
  loader — real releases, p50 3.6K chars). Synced to sk3 + hierarchy json + parquet.
- Flagged for the math domain: hierarchy/task key is `math-stackexchange`, config preset key
  is `math` — needs a key bridge before math joins the chain.
- 70B rescore untouched (PIDs 199503/200394 alive; GPU4 worker in CPU phase between engines).

## Infrastructure update (evening — fully autonomous overnight pipeline)

- News sweep DONE 25/25 (~15:59); math chain auto-fired on GPU5 (PID 2871462). User granted a
  2nd GPU mid-day, so news ran parallel on GPU5 (one GPU3 launch lost a race with the 70B
  rescore's dynamic GPU picker — killed cleanly, relaunched on GPU5 with an occupancy guard).
- **dossier_v2 IMPLEMENTED in the grid driver** (task #17): new RUNG_ORDER entry = definition +
  explanation + "Illustrative examples:" (no ONLY line); v1 rungs byte-identical; sk3 smoke
  green on real news checkpoints (7 rungs). All NEW domain grids carry it from day 0.
- Armed chains: GPU1 (PR sweep → PR cert (CPU) + PR grid → news grid), GPU5 (math sweep →
  math cert + math grid). Grids: 8B writer (comparability with CW/humor), readers 1B/3B/8B,
  forms 3, phase all. News Day-0 cert building on CPU now (behavioral quotient, no GLM).
- **Grid-launch postmortem (18:23):** first PR+news grid attempts crashed at reader-engine init
  — NOT the rescore this time: `run_decompression_grid --phase all` DOUBLE-BOOKS the GPU across
  the messages→score transition (writer engine not torn down before the reader inits), so at
  util 0.85 the writer's 151GB leaves 24GB and the reader dies ("Free memory 24.41/178.35 <
  0.85"). v1 CW/humor grids presumably ran phases as separate invocations. Fix shipped:
  `grid_run_v2.sh` = VLLM_GPU_MEM_UTIL **0.25** (8B needs ~45GB; two engines coexist, and the
  70B rescore's wandering picker prefers fully-free GPUs over a 45GB-occupied one) + retry×5
  with free-mem gate + success = report.json existence (never trust $?). Verified: PR grid
  running on GPU1, 2×47GB engines resident, scoring at full batch. Distinct from the EARLIER
  GPU3 loss, which WAS the rescore's picker (166GB footprint appeared in the 36s window).
- Concept taggers (2 independent raters × 3 domains) DONE + adjudicated (20 disagreements from
  full rubrics; κ .42/.51/.28 — lower than CW/humor's .63 because the MECHANICAL/CRAFT boundary
  is exercised for the first time; 9 borderline-flagged). `concept_tags_newdomains.json`.

**THE MECHANICAL CELL IS FILLED (13/88 vs 0/106 in CW+humor):** news 2/18/5, press-releases
**10**/30/2, math-SE 1/17/3 (MECH/CRAFT/TASTE). The mechanical criteria are exactly the
predicted checklist-convention layer: boilerplate/company-info, press-kit completeness, SEO
metadata, AP-style conformance, scannable formatting, media assets/contacts/hub-links,
news-vs-opinion LABELING, disclosure-presence, symbol-only-chain avoidance. Checkability
gradient across domains: CW 35% taste → math 81% craft → PR 24% mechanical. The
"what-gets-decompressed" analysis can now run with all three levels populated once tonight's
grids land. (Also: "banks have 0 mechanical" is hereby SCOPED — true of expressive-domain R3
banks, not institutional/technical ones.)

## News Day-0 certificate — and the SCOPE-DEGENERACY finding (evening)

First Face-1 cert for a mechanical-bearing domain (25 metrics, behavioral quotient, harvested
to `certs/news_day0_cert.json`). Raw verdicts: 10 CODIFIABLE / 12 UNDERSAMPLED / 2 DEEP / 1 FD
— but **all 10 CODIFIABLE are VACUOUS: H_M = 0.0000 exactly.** Spot-checking M_i: the
continuous own-scores vary fine (130-220 unique values) but sit entirely BELOW the binary
threshold — no homepage story ever *satisfies* "transparent corrections practice",
"anonymity-use disclosure", "verification standards", "COI disclosure". These are
**practice/apparatus criteria**: they describe editorial process, not properties a story text
can vary on. 13/25 news metrics are scope-degenerate (H_M<0.1) vs CW 0/43, humor 4/60.

Consequences (recorded before anyone quotes "news is 40% codifiable" — it is NOT):
1. **Validity filter convention: all cross-domain cert claims condition on H_M ≥ 0.1.** Under
   the filter, news has 0 genuine CODIFIABLE at n=300; live news metrics = 12. Humor's Day-0
   5 CODIFIABLE → 4 genuine (one at H_M=0); CW unaffected (0 low-H_M).
2. Expect PR worse: its 10 MECHANICAL criteria are apparatus-heavy (press kits, SEO metadata,
   media contacts, multimedia attachments — not in the release text). **The mechanical cell
   may be largely UNMEASURABLE on text-only probes** — a real design finding: mechanical
   criteria tend to live in artifacts *around* the text; only text-visible mechanical criteria
   (formatting, boilerplate-in-text, labeling, symbol-chains) can enter the tacitness
   measurement as designed. PR cert (building now) will quantify this.
3. Salvage option (NOT adopted — measurand change, needs sign-off): per-metric median-split
   binarization would measure *relative* satisfaction and revive these metrics, but changes
   the question from "is the criterion met" to "what distinguishes the top half".
4. Grid/census machinery self-protects (H_self≈0 metrics skip), so Face-2 curves and profile
   transport are unaffected; matched-pair tests will simply have both-sides-live n.

## PR grid + census — the third domain replicates, and dossier_v2 gets its verdict (night)

PR grid completed on the v2 launcher (42 metrics × 6+1 rungs × 3 readers, harvested to
`r3_pr/grid_pr_v1/`; adapter `methods/codability/grid_report_to_self.py` reshapes the report's
self-readout into census format → `grid_bits_self_pr.json`, `isomorphism_census_pr.json`).

- **Shape replicates in domain #3** (small-reader avg): name .311 → **definition .393 (peak,
  best rung for 23/42)** → explanation .297 → exemplars .212 → dossier .250. Same signature as
  CW (.414→.465→…) and humor (.289→.438→…).
- **dossier_v2 verdict: the ONLY-line was NOT the problem.** v2−v1 on 36 live metrics: mean
  −.0005, median +.0017, 19/36 positive, Wilcoxon p=.82. With the self-contradiction removed,
  telling+showing still recovers ~.25 vs definition's ~.39. The honest chain: the v1 claim was
  rightly retracted (unmeasurable as constructed) → now properly measured → **rich articulation
  genuinely underperforms a crisp definition for small readers** (PR; news/math replications
  land tonight, their grids carry dossier_v2 too). CW/humor retro re-score (#17) downgraded to
  optional confirmation.
- **First MECHANICAL decompression curves (n=7 live)**: .270→.352→.277→.225→.240 — parallel to
  craft (.306→.393→…), definition-peaked, and NO name-sufficiency advantage over craft
  (mech name .270 ≤ craft .306). Mechanical-but-articulated criteria behave like craft in
  self-recovery; taste (n=2, PR-thin) again shows name≈definition (.424=.424). Descriptive —
  cells small.
- PR Face-1 under the H_M≥0.1 filter: 36 live (22 US / 9 DEEP / 5 FD, 0 COD) — far LESS
  form-fragile than CW/humor (5/42 vs 36/43 FD) and the first sizable DEEP population.
  Degenerate 6/42, concentrated in pure-attachment criteria (SEO, boilerplate-links,
  multimedia) — 7/10 MECHANICAL are live (text-visible mechanical structure measures fine).

## Math census — the first domain where the curve INVERTS at the top (late night)

`isomorphism_census_math.json` (21 metrics): **name .452 > definition .397** = explanation
.397 → exemplars .131 → dossier .123 ≈ dossier_v2 .138. Best rung = NAME for 14/21 (vs
definition-peak in CW/humor/PR). Cross-domain name→definition gap now orders:
**humor +.149 → PR +.082 → CW +.051 → math −.055** — a lexicalization gradient: the more
standardized the field's critical vocabulary, the more the bare name carries ("notation
consistency", "citation rigor" are precise pointers; "game identification" needs unpacking).
Domain-level version of the taste-as-enculturated-index mechanism. Ostension collapse is
hardest in math (regressive 21/21); dossier_v2≈v1 replicates (ONLY-line not the cause, 2nd
domain). taste≈craft n.s. again (3 vs 17).

## CAPSTONE (night): concept-type transport replicates out-of-sample — and the 5-domain grid

**News×PR profile transport** (`crosstask/news_pr_isomorphism.json`, 11 judged matches, both
grids live): **label agreement 11/11 = 100% (perm p=.0102; 1-1 arm 7/7, p=.0228).** With
CW×humor (25/30, p=.0017) that is TWO independent significant pairs; pooled 36/41 (87.8%).
The taste/craft/mechanical class of a criterion is concept-intrinsic — the day's central
claim, now replicated on a pair sharing a different criterion family (media/accountability vs
narrative craft). Continuous profile ρ's remain n.s. at n=11/7 (scattered signs; the
continuous test accumulates power as more grid-bearing pairs land — 3 of 5 significant pairs
still lack one side's grid).

**The five-domain articulation curve** (small-reader avg, rec = self-readout / H_self):

| domain | n | name | definition | explanation | exemplars | dossier | best rung |
|---|---|---|---|---|---|---|---|
| creative-writing | 46 | .414 | **.465** | .389 | .152 | .154 | definition |
| humor | 57 | .289 | **.438** | .391 | .093 | .159 | definition |
| news-homepages | 24 | .279 | **.382** | .246 | .027 | .095 | definition |
| press-releases | 42 | .311 | **.393** | .297 | .212 | .250 | definition |
| math-SE | 21 | **.452** | .397 | .397 | .131 | .123 | **name** |

- Definition-peak in 4/5 domains; **math inverts** (name-peak, 14/21 best-rung=name).
- **Lexicalization gradient** (name→definition gap): humor +.149 > news +.103 > PR +.082 >
  CW +.051 > **math −.055**. The more formalized the field's critical vocabulary, the more
  the bare name carries — folk lexicons (humor) need unpacking; formal lexicons (math) ARE
  the unpacking. Domain-level twin of the taste-as-enculturated-index mechanism.
- Ostension collapse (exemplars ≤ .21) in ALL five; dossier_v2 ≈ dossier in all three domains
  that carry it — the ONLY-line artifact is definitively not the cause.
- News-only wrinkle (descriptive, n=5): taste name_frac .220 < craft .307 (MW p=.039) — the
  one cell where taste names carry LESS; opposite of CW's direction. Watch, don't claim.

## Math day-0 cert harvested (2026-07-05 morning)

`certs/math_day0_cert.json` (21 R3 metrics, CPU build finished overnight):

- **Scope-degeneracy: 0/21** with H_M<0.1 (news had 13/25 — degeneracy is a DOMAIN property:
  apparatus criteria live around the text; math criteria live in the answer text itself).
  gi=16, math's lone MECHANICAL ("Integrating symbols into grammatical, readable prose"), is
  LIVE: H_M=0.99, OPT_Ω=0.62 bits.
- Face-1 raw verdicts: 16/21 FORM-DOMINATED, 4 UNDERSAMPLED, 1 DEEP — CW-like (36/43 FD),
  opposite pole from PR (5/42). CAVEAT: quote only after the band-mode (eps_form) re-read —
  the form-gate redesign showed CW's FD mass mostly reclassifies to UNDERSAMPLED.
- Provisional raw-FD gradient: CW .84 ≈ math .76 ≫ news ~.3 > PR .12 — expressive/constructed
  text form-fragile, institutional text form-robust (pending band re-read).

## Next (queued) — refreshed 2026-07-05

1. ✅ PR + news grids/censuses/tags/certs; news×PR transport 11/11 p=.0102 (capstone above).
2. ✅ Rung audit → dossier_v2 built + measured (ONLY-line not the cause, p=.82).
3. **Deepen (unlocks blocked tests):** grids for legal-outcome-prediction, peer-review,
   grant-funding → transport tests for the 3 untested significant pairs (legal×math,
   grant×peer, math×peer — rigor/research-merit families; transport so far only tested in
   narrative/media families) + 8-domain curve table + powered pooled continuous-ρ.
4. Calibration judging: 2–3 mid-z non-significant pairs (currently only the z≈0 control and
   the z≥3.7 winners are judged — show match rate scales with z).
5. Band-mode (eps_form) re-read of pr/news/math certs before any cross-domain Face-1 claim.
6. Model-pair chains (Stage D, Gemma-4 panel) — user-staged last.

Related: `notes/2026-07-03__what-gets-decompressed.md`, theory §1/§2.4,
`[[project_crosstask_isomorphism]]`, `[[reference_split_half_finalizer_state]]`.
