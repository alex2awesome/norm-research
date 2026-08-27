# DESIGN (FROZEN BEFORE RUN) — What the articulability gap *means*: selection regret (Rung 1) and optimization divergence (Rung 2)

Date frozen: 2026-08-21, before any Rung-1 or Rung-2 readout was computed.
Requested by user ("I really like Rung 1 and Rung 2 ... come up with a plan to
implement them and run them ... the goal then is to show that, the larger the
gap, the more this issue presents itself"). Author: claude-fig lane.

## 0. The claim being operationalized, and the circularity analysis

The paper measures a static gap: dense models recover more of the human
preference signal than the best articulated criteria bank. Two questions
remain: (a) is the dense advantage *real human preference* rather than an
artifact of the dense channel, and (b) does the gap have *consequences* —
does anything go wrong if you act (select, optimize) using only articulated
criteria? The two rungs split these so that neither requires a circular step:

- **Rung 1** (selection regret) acts only on REAL items with REAL human
  labels. The arbiter of who was right is the human signal itself. No model
  is ever the judge of another model's selection. → answers (a) in decision
  terms and is non-circular *by construction*.
- **Rung 2** (optimization divergence) acts on GENERATED items, where no
  human label exists. Here a model must arbitrate, so the design's job is to
  make the arbitration non-circular (§3). Rung 2 borrows its license from
  Rung 1: R1 establishes that the dense-minus-bank residual tracks true human
  preference on real items; R2 then shows that this residual (i) steers
  selection among generated candidates, and (ii) is invisible to the
  articulated instrument. Together: articulated-only optimization drifts from
  human preference in ways stated criteria cannot detect.

**What "circular" would look like, and what we refuse to quote:**
"dense-selected outputs score higher under the dense model" — trivially true
(selection on a score inflates that score). Likewise "bank-selected outputs
score higher under the bank." These diagonal cells of the selection×scoring
table are NEVER quoted as evidence. All evidence lives off-diagonal (§3.3)
or in disagreement-conditioned readouts against real labels (§2.3).

Also refused: any framing where the dense arbiter shares training rows with
the dense selector (§3.2 split), and any divergence that a named-nuisance
channel explains (§3.5 — if length or format separates the two selected
sets, that is Track-B material, not tacitness; same discipline as F2).

**The headline prediction (both rungs, preregistered):** per-cell effect
sizes increase with the cell's static relative gap
(gap/(AUC_best − .5), the Fig-3 quantity). Direction is preregistered;
readouts are rank-based (threshold-free rule); n(cells) is small, so we
quote Spearman ρ with a permutation p descriptively, per-cell group-bootstrap
CIs as the primary uncertainty, and no significance theater.

---

## 1. Rung 1 — selection regret on real items (CPU, artifacts on disk)

### 1.1 Data and machinery
Reuse the frozen F2 stack verbatim (no new fits beyond the certified
Layer-1 pattern): `fusion/f2_cells.py` adapters + `f2_deconf.load_E` give,
per cell, the aligned E-frame: ids, y, grouping unit, dense scores T, and
the enriched bank matrix. Bank per-row scores = grouped-OOF predictions of
the frozen stack on [bank_enriched] — the same (a)-arm OOF vector
`f2_gapci.py` already computes. Cells: the 13 F2 adapter cells
(cap_finalist, cw_community, hashtagwars_verdict, jokes_community,
mathse_accepted_verdict, mathse_vote_score, nc_agree, nc_outcome,
nc_responded, peer_curation, peer_revealed, peer_verdict, press_verdict).
Guard: record each cell's dense train-overlap flag from its samerows/ledger
artifact; any cell whose dense scores are not clean-held-out on E is flagged
in the output and excluded from the headline correlation (the E frame was
chosen to make this pairing legitimate — this is a belt-and-suspenders
check, not a new audit).

### 1.2 Unit of analysis: decidable groups
A *decidable group* = a grouping unit (question, contest, docket, outlet-day
...) with ≥2 items and at least one positive and one negative on E. All
readouts are within-group, then aggregated over groups (composition guard —
same reason F2 bootstraps on the grouping unit). Report per cell: n_groups,
n_decidable, mass skipped.

### 1.3 Readouts (per cell)
Let b(x) = bank OOF score, t(x) = dense score, y(x) = real label.
Within each decidable group, each selector picks its argmax.

1. **Top-1 hit rate**: P(y=1 for the selector's pick), for bank, dense, and
   a random-pick baseline (analytic: pos_rate within group). Decision-flavored
   restatement of the ladder.
2. **Disagreement win rate (primary)**: among groups where bank and dense
   pick DIFFERENT items, the win ratio: P(dense pick is a winner, bank pick
   a loser) vs P(reverse). Quoted as a ratio with a group-bootstrap CI.
   This is the cleanest non-circular sentence in the study: "when the two
   instruments disagree about which item the community will prefer, the
   dense pick is right X:1 — judged by the community's actual choice."
3. **Swap rate**: fraction of decidable groups where the two selectors
   disagree (the granularity claim: how often does the extra signal change
   the decision, not just the score).
4. **Regret**: hit(dense) − hit(bank), group-bootstrap CI (2,000 draws,
   seed 12345, same bootstrap as F2).

### 1.4 Cross-cell readout (the "larger gap, larger issue" figure)
Scatter: x = static relative gap (Fig-3 table; for cells not in the Fig-3
22, compute the same quantity from the cell's ledger), y = disagreement win
rate (primary) and regret (secondary). Prereg: both increase with x
(one-sided). Note honestly in the paper: AUC and top-1 are related
functionals, so a positive relation is expected; the informative content is
its magnitude in decision units and any cells that break rank.

### 1.5 Outputs
`results/rung1_selection_regret_<cell>.json` (all readouts + counts + CIs) +
`results/rung1_summary.json`. Descriptive note in notes/; no paper text
until the user reads (report-results-not-conclusions).

---

## 2. Rung 2 — optimization divergence on generated items (sk3, batch vLLM)

### 2.1 Design in one line
Generate K candidates per real held-out prompt; select top-1 under the bank
vs under the dense selector; show (off-diagonal) that dense-selection costs
almost nothing on articulated criteria while bank-selection leaves large,
dense-visible deficits — with the asymmetry growing with the cell's static
gap.

### 2.2 Cells (need: generatable x, both instruments score raw text)
Chosen to span the relative-gap range, subject to feasibility:

| cell | rel. gap | prompt → candidate |
|---|---|---|
| cw_community (WP upvotes) | 43.5% | WP prompt → short story |
| mathse_vote_score | 25.5% | question → answer |
| jokes_community | 15.0% | joke setup/title constraint → joke |
| so_votes | 7.9% | question → answer |

(peer_revealed/citations 41.7% is attractive but generating credible
abstracts-as-papers is a validity fight we don't need; revisit only if the
4-cell trend is unclear. so_votes needs its bank scorer runnable on new
text — it does: the banks are criteria prompts, corpus-agnostic.)

### 2.3 Generation
- Generator: ONE mid-tier open model for all cells (Gemma-4-27b-class or
  the sk3 Gemma-4-31b; final pick recorded at smoke time), temperature
  ~1.0, K=16 candidates per prompt, N=150 held-out prompts per cell
  (prompts drawn from the cell's E-frame inputs; stable-hash selection, no
  seeded shuffle). ~2,400 generations/cell ≈ 10k total → offline batch
  vLLM, ONE GPU from the allowed sk3 set (0,1,2,4,5,6; 3+7 banned through
  ~08-24), chunked batches per the batch-chunking rule.
- Matched-pipeline rendering: candidates rendered through the same renderer
  as real items before any scoring; char-ngram probe before scoring runs.

### 2.4 Scoring arms
1. **Bank scorer** = the cell's frozen criteria bank, judge-scored by the
   same judge family as the bank was built with (Gemma-4-31b, guided JSON,
   anchor items in EVERY batch, judge-score-distribution check). Bank
   per-candidate score = the frozen stack applied to the criteria vector
   (same aggregation as the cell's (a) arm; no refitting on generated data).
2. **Dense arbiter, split-trained**: retrain the cell's dense model twice on
   disjoint halves of its training split (half-A model, half-B model; same
   recipe as the cell's registered T). The SELECTOR uses half-A; the
   ARBITER is half-B. Selector and arbiter share no training rows.
   Family-crossing robustness: also score with the cell's embedding-HistGB
   dense variant where one exists (different inductive family).
3. **Nuisance block**: the cell's F2 named-nuisance channels scored on all
   selected candidates (length, format, register, era-vocab...).

### 2.5 Readouts (all rank-based)
Selection policies: SEL_bank, SEL_dense(half-A), SEL_random.
Scoring: bank, dense-arbiter(half-B). NEVER quote a policy under its own
score (diagonal). Per cell:

1. **Δ_articulated** = bank(SEL_dense) − bank(SEL_bank): prereg ≈ 0⁻
   (dense-guided choices remain nearly indistinguishable to the stated
   criteria). Paired over prompts, rank statistic + bootstrap CI.
2. **Δ_dense** = arb(SEL_dense) − arb(SEL_bank): prereg > 0.
3. **Asymmetry A = Δ_dense − |Δ_articulated|** — the headline per-cell
   number. Prereg: A > 0 in every cell; A increases with static gap.
4. **Instrument blindness AUC**: can each instrument distinguish the
   SEL_dense population from the SEL_bank population? Prereg: bank ≈ .5,
   arbiter > .5. ("The two policies produce different outputs; only the
   dense channel can see the difference.")
5. **Nuisance screen (gate on 1–4)**: if any named nuisance channel
   separates SEL_dense from SEL_bank (AUC materially > .5), recompute 2–4
   on nuisance-matched prompt pairs (caliper matching, F2 Leg-2 pattern)
   and quote the matched version. If matching kills Δ_dense, the divergence
   was a named shortcut → reported as such, not as tacitness.

### 2.6 Controls & validity
- **Anchor battery**: real held-out E items with known labels planted,
  blinded, in every scoring batch (both arms); arbiter must rank real
  winners over losers at ≈ its registered AUC or the batch is void.
- **Generator-family / self-preference check**: generator ≠ judge family
  where possible; if not possible, note it — the comparison is
  between two selection policies over the SAME candidate pool, so
  generator-side preferences cancel in first order.
- **Diagonal reporting ban** (§0) and split-arbiter independence (§2.4.2).
- **Random-selection floor**: SEL_random anchors both scales.
- Judges Sonnet-or-better rule does not bind here: the bank judge must be
  the SAME family the bank was certified with (instrument identity beats
  judge upgrade); noted as a deliberate exception with the certification as
  justification.

### 2.7 Staging (freeze → smoke → run)
1. Freeze this doc (done at commit time) — predictions above are prereg.
2. Smoke (per cell, N=10 prompts, K=4): pipeline end-to-end, render probe,
   judge-distribution check, anchor sanity. NO readout of Δ's beyond
   plumbing sanity (never read smoke direction rule).
3. Full run, one cell at a time (cw_community first: largest gap, clearest
   prediction), one GPU, batch vLLM.
4. Descriptive results note; user reads before any paper text.

### 2.8 Cost envelope
~10k generations + ~10k × (bank criteria + arbiter) scorings. Generation
dominates; ≈ a few GPU-hours per cell on one B200-class GPU. No Claude
credits; judge = local Gemma batch. Codex/GLM not needed.

---

## 3. What this buys the paper
- Rung 1: the gap in decision units, non-circular, with the gap→regret
  scatter as the "so what" panel next to Fig 3.
- Rung 2: the Goodhart statement: optimizing what can be stated drifts, on
  the tacit dimension, by an amount the static gap predicts — and the
  stated criteria themselves cannot detect the drift.
- Framing hook (CreativePreferences): the gap is largest exactly where the
  field—in Csikszentmihalyi's sense—rather than a procedure confers value.

Related artifacts: FIGURES.md rounds 5–7 (ceiling audit), f2_gapci.py
(plotted-contrast CIs), notes/2026-08-20__spec_gap_by_dense_confidence_deciles.md
(localization spec, complementary: WHERE the gap lives vs WHAT it costs).

---

## ADDENDUM A (2026-08-21, recorded BEFORE running the affected cells)

Rung-1 §1.2 assumed every cell has multi-item groups. Three cells do not
(peer_curation, peer_revealed, peer_verdict: group = ntitle, one row per
group), and two cells' artifacts live on sk3 (mathse_vote_score,
mathse_accepted_verdict: their F2 ran box=sk3) — the mac sweep fails on both
kinds, correctly.

**Singleton-group cells → pairwise forced-choice mode.** Selection unit =
every (positive, negative) pair on E (deterministic, all pairs, no
sampling). Each selector picks the item it scores higher; the positive is
the right answer. Readouts unchanged in spirit: hit rate over pairs (= the
selector's AUC, restated — reported but NOT novel), swap rate, and the
disagreement-conditioned win ratio (novel, quotable), with an item-level
bootstrap CI. Output marks mode="pairwise_singleton"; grouped and pairwise
cells are plotted with distinct markers in the cross-cell figure and the
correlation is reported both pooled and grouped-only.

**sk3 cells** run with the identical script on sk3 (same frozen stack; the
artifacts are that box's).

---

## RUN LOG (descriptive; no conclusions until user reads)

2026-08-21 Rung-1 grouped sweep + Addendum-A reruns:
| cell | mode | rel gap | regret [CI95] | disagree (d:b) |
|---|---|---|---|---|
| press_verdict | grouped (8 dec) | 10.9 | +.000 [0,0] | 0:0 |
| hashtagwars | grouped (8) | 12.5 | +.125 [-.25,+.50] | 2:1 |
| jokes | grouped (10) | 15.0 | +.100 [0,+.30] | 1:0 |
| cap_finalist | grouped (46) | excl. | +.065 [-.09,+.24] | 9:6 |
| nc_responded | grouped (20) | 13.3 | +.100 [-.05,+.30] | 3:1 |
| nc_outcome | grouped (69) | 14.5 | +.101 [-.04,+.25] | 16:9 |
| nc_agree | grouped (65) | excl. | +.092 [-.06,+.25] | 16:10 |
| cw_community | grouped (523) | 43.5 | **+.061 [+.011,+.107]** | 101:69 |
| peer_curation | pairwise 460k | 36.0 | +.065 [+.019,+.107] | 1.33:1 |
| peer_verdict | pairwise 383k | 39.4 | +.109 [+.077,+.140] | 2.07:1 |
| peer_revealed | pairwise 57k | 41.7 | **+.171 [+.125,+.217]** | 4.41:1 |
| mathse ×2 | sk3 rerun in flight | 25.5/8.3 | — | — |

bbc_mostread: no mac t0 artifacts, skipped. Partial grouped-only correlation
(n=7, missing the whole high-gap flank) was ~0/negative; full-set correlation
computed only when mathse lands (collate across boxes).

2026-08-21 Rung-2 cw_community generation COMPLETE: 150 prompts x K=16 =
2,400 candidates, 0 degenerate, len p10/p50/p90 = 3162/3822/4662 chars
(cap raised 900->1500 tok after smoke; real p90 6206 — note candidates'
upper tail still thinner than real; length is in the nuisance screen).
Landmines hit + fixed: multimodal encoder-cache profiling starved KV
(limit_mm_per_prompt=0), reaped-ssh SIGTERM (setsid detach), GPU0 free-mem
squeeze (util .49). Bank-scorer build next: 144 cols = 15 programmatic V
+ 45 base (defs in score_va_gemma_banks::build_creative) + 84 mined (defs
in roundN_criteria.json x routing A); scorer = stage0_score_ext_gemma
pattern; selector model = frozen recipe fit on real E, applied to
candidates (no refit on generated rows).

---

## ADDENDUM B (2026-08-22, recorded BEFORE running): indistinguishability / tie readout

User question: "how often do V, VA fail to distinguish at all — how often are
items ranked the same (maybe under some degree of quantization)?" This is a
GRANULARITY property, complementary to accuracy: an instrument can be right
when it speaks yet unable to speak at all on much of the corpus.

Per cell and per instrument s in {V-only, VA-bank, dense}:
- scores are rank-normalized to [0,1] within the cell (instruments' raw
  scales are not comparable); quantization grid q in {0 (exact float tie),
  .01, .02, .05} — a tie at q means the instrument cannot separate two items
  by more than q of the corpus ranking.
- grouped cells: ABSTENTION rate = fraction of decidable groups whose top-1
  is not unique under q (>=2 items within q of the max).
- pairwise cells: TIE rate = fraction of pos-neg pairs with rank-score
  difference <= q.
- joint readout: mass where V ties AND VA ties but dense separates
  ("dense-only granularity"), and its label accuracy when dense speaks.
V-only arm: fit with the same frozen Layer-1 recipe on the cell's V block
(available for closure cells via round0_state['V']); cells without a clean V
block report VA + dense only. Per-row score vectors are SAVED this time
(results/rung1_scores_<cell>.npz) so readouts stop requiring refits.
Prediction (soft, exploratory — this is a descriptive readout): tie mass
falls V > VA > dense at every q; the interpretation section may use it to
decompose the regret into "articulated instrument silent" vs "speaks but
wrong". The user's "quantization of the LR weights" reading (coarsening the
linear arm's weights) is noted as an alternative; rank-scale score
quantization is implemented first as instrument-agnostic.

---

## RUNG-2 cw_community RESULT (2026-08-23) — PREREG FAIL, mechanism identified

Readout (rung2/rung2_readout_cw.json; all batteries passed: bank anchors
.552 vs certified .562, dense halves on blinded real items .778/.796 at
their registered .785/.784):
- agreement rate 12.7%; Δ_articulated = −.387 (dense pick lands mid-pool on
  the bank ranking, NOT near-top as predicted); Δ_dense = +.019 [−.036,+.080]
  (arbiter barely prefers dense picks); ASYMMETRY = −.368, prereg (A>0) FAILS.
- blindness: arbiter cannot distinguish the two selected sets (.506);
  length screen clean (.531).
- both policies beat the random floor on the arbiter (.584/.566 vs .463).

Mechanism (diagnostic, same file section as this note):
- dense halves agree strongly ACROSS prompts (ρ=.893 over 2,400 candidates;
  across-prompt sd of prompt-means .21–.23) but at ρ=.207 WITHIN a prompt's
  16-candidate pool (within-prompt sd .03–.05); each half agrees with the
  bank at ρ≈.06 within-pool. 44/129 bank criteria collapsed on generated
  text (0 collapsed on real text, same judge).
- i.e. K=16 same-generator pools are too HOMOGENEOUS: within-pool quality
  variance sits at every instrument's noise floor, so best-of-N selection
  never engages the signal whose gap the static ladder measured across
  diverse human-written stories. The split-arbiter guard did its job — a
  same-model arbiter would have manufactured a fake positive Δ_dense from
  half-A's within-pool idiosyncrasies.

NOT concluded: "the gap has no optimization consequence." Concluded: THIS
generator/K regime produces insufficient candidate diversity to test it.
Candidate design fixes (need user sign-off before running — new measurement
design): multi-generator pools; higher-temperature / persona-perturbed
generation; larger K; or pools of REAL stories mixed with generated ones
(bridges to Rung 1). Cost of a rerun ≈ one generation + one scoring pass.

---

## ADDENDUM C (2026-08-24, recorded BEFORE the re-test): floor-based readouts + certified arbiter

User-identified inconsistency in the frozen §2.5 prereg: "Δ_articulated ≈ 0
(dense picks look just as good to the bank)" presupposed a loose bank ranking
near the pool top — but Addendum B established the bank essentially never
ties ("speaks but wrong"). Under that finding the bank's own argmax MUST
stand above any alternative on its own scale, so Δ vs own-argmax readouts
are the wrong statistics (their diagonal term is also where selector/arbiter
circularity lived). Amended readouts (own-argmax comparisons dropped):

1. FLOOR-BASED ASYMMETRY: A' = [arb(SEL_dense) − arb(SEL_random)] −
   [bank_rank(SEL_bank... NO — symmetric form:
   loss_bank_channel(dense policy)  = bank_rank(SEL_dense)  − bank_rank(SEL_random)
   loss_dense_channel(bank policy)  = arb_rank(SEL_bank)    − arb_rank(SEL_random)
   A' = loss_bank_channel(dense policy) − loss_dense_channel(bank policy):
   each policy measured only under the OTHER instrument, both against the
   random floor. No own-argmax term → a SINGLE certified model may arbitrate.
2. BLINDNESS AUCs on selected sets (unchanged from §2.5.4).
3. Arbiter = the cell's CERTIFIED full dense model (wp_clean, AUC .786);
   split-half arbiters retained as agreement check (if certified agrees with
   halves within-pool, the homogeneity mechanism is confirmed as real, not a
   half-model artifact).
Prereg for the re-test (soft, given the homogeneity finding): within-pool
correlations of certified model with halves ρ < .4 → homogeneity confirmed;
floor-based A' expected ≈ 0 on THESE pools for the same reason. The decisive
Goodhart test remains gated on the pool-diversity redesign (user go/no-go).

## ADDENDUM C RESULT (2026-08-24) — certified-arbiter re-test on existing pools

Battery: certified full dense scored blinded real anchors at .785 (= its
registered .786). rung2/rung2_readout_cw_v2.json:
- FLOOR ASYMMETRY A' = +.023 [-.061, +.108], p>0 = .70 → ≈0, as the
  Addendum-C soft prereg expected on these pools. Above-floor signal on the
  cross channel: dense policy +.076 [.009,.144] on bank channel; bank policy
  +.053 [-.012,.114] on dense channel — both small, near-symmetric.
- Blindness: certified arbiter distinguishes the two selected sets at .565
  (half-B arbiter: .506); bank at .273 (mechanical).
- HOMOGENEITY MECHANISM CONFIRMED NOT A HALF-MODEL ARTIFACT: certified vs
  halves global rho .93/.89 but WITHIN-POOL .47 (A) / .22 (B); certified vs
  bank within-pool .09. The dense construct loses most of its shared
  variance inside a same-generator pool even for the certified model.
Decision standing: Goodhart test remains untested in this regime; needs the
pool-diversity redesign (user go/no-go).

## AUDIT (2026-08-24, user-requested full pass over Rung-1/Rung-2 code)

Verified: cand_id joins single-sourced; bank column order asserted from
round7 bank_names at manifest build; V scales real-vs-candidate match (raw
units both sides); score orientations via anchor batteries (every scoring
path reproduces its certified AUC on known ground truth); within_rank +
shuffle-inverse + prompt-bootstrap correct; pools are lexically diverse
(0 exact dups, Jaccard .17) — homogeneity is in quality space, not text.
DEFECTS: (1) v2 blindness AUC .565 is DIAGONAL-CONTAMINATED (certified
arbiter judges sets containing its own picks) — discount; clean statistic
is v1's .506. A' unaffected (no own-pick terms). (2) rung1_ties q=0 row
invalid (rank01 splits exact ties); q>=.01 rows unaffected. Neither defect
changes any conclusion.

---

## ADDENDUM D (2026-08-24, recorded BEFORE running): diverse-pool generation v2

User directives: inject diversity via (i) humanness instructions ("be
human" / "be very human" gradient — user hypothesis: LLMs may not natively
mimic the human language distribution, instructed humanness may recover
some), (ii) MULTIPLE generator families, not one. Plus the mixed-real
injection from the C-addendum discussion.

Pools per prompt (same 150 stable-hash prompts):
- 3 families x 6 conditions x 1 sample = 18 generated:
  families = Llama-3.1-8B-Instruct / Qwen2.5-14B-Instruct /
  Mistral-Small-24B-Instruct-2501 (all cached on sk3; sequential on ONE
  allowed GPU; judge family (Gemma) deliberately NOT a generator).
  conditions = plain | human ("write like a real human redditor, not like
  an AI") | very-human (idiosyncratic voice, imperfections, personal
  texture) | casual persona (dashing off a fun reply on a phone) |
  literary persona (practiced writer, polished and ambitious) |
  plain@temp1.3. Temp 1.0 elsewhere; seeds fixed per (family,condition).
- + ALL real stories for that prompt from the honest population (1-3,
  labels kept but blinded to all scoring).
Readouts added to §2.5/Addendum C (all rank-based, frozen now):
  R1. floor-based A' on diverse pools (the Goodhart test, now with real
      within-pool quality variance);
  R2. real-vs-generated separation per instrument (bank / certified dense)
      — tests the user's "LLMs may miss humanness entirely" hypothesis;
  R3. humanness-gradient: does dense score move plain -> human ->
      very-human toward the real-story score level? bank same question;
  R4. family/condition blindness AUCs.
Scoring: identical frozen bank scorer + certified dense (+ halves for the
agreement check). All prior batteries/screens apply.

## ADDENDUM D RESULT (2026-08-24) — diverse pools: A' POSITIVE + two surprises

Batteries: bank anchors .552 (=certified), dense .785 (=registered), bank
n_collapsed 1/129 (was 44 on v1 pools — diversity restored the instrument's
range). Pools: 150 x (18 generated [llama8b/qwen14b/phi4 x 6 conditions] +
1-6 real, 215 real rows). rung2/rung2_readout_cw_v3.json.

R1 (Goodhart asymmetry, floor-based): dense policy on bank channel +.241
[.182,.300]; bank policy on dense channel +.174 [.121,.234];
**A' = +.067 [+.002,+.129], p>0=.978** — preregistered direction, modest,
CI excludes 0 by a hair. hightemp condition is catastrophic on both
channels (rank .10/.22) — instruments have working range; v1's null was
pool homogeneity, confirmed.

R2 (real-vs-generated): bank selector .743, criteria-mean .851, certified
dense .553. NOT length (n_chars .469). Style signature: real = varied
paragraph lengths (.78), questions (.70)/exclaims (.72), HIGHER reading
ease (.83); generated = higher MATTR/hapax/longer words. THE ARTICULATED
INSTRUMENT DETECTS HUMAN-NESS FAR BETTER THAN THE DENSE PREFERENCE MODEL.
Caveat: dense .553 may be OOD-insensitivity, not real/generated equivalence
on true preference — do not over-read.

R3 (humanness gradient, user hypothesis): NOT instructable. bank plain .570
-> human .564 -> veryhuman .523 (hurts); dense flat (.547/.555/.542); real
bank rank .725 towers over every generated condition. Humanness deficit is
ARTICULABLE (criteria see it) but not SYNTHESIZABLE on demand.

R4: generator families indistinguishable on dense channel (.47-.53).

Picks: bank policy retrieves a REAL story 36% of pools (base rate 7.4%;
5x); dense policy 18% (2.4x); picked real stories skew winners (label mean
.72/.78).

Narrative note (descriptive): Rung 1 = dense > bank at ranking WITHIN real
stories (tacit residual); Addendum D = bank > dense at real-vs-LLM
discrimination (craft/register axis, articulable). Distribution membership
is articulable; within-distribution preference carries the tacit part.

---

## ADDENDUM E (2026-08-24, recorded BEFORE running): expansion + OOD dissection

User direction: expand effect size via §Q2(2,3,4) THEN iterated optimization
(1); run both OOD-dissection probes; note the OOD/humanness line drifts
toward paper #4 — results filed as paper-4-adjacent.

E1 (more cells): extend the diverse-pool design to jokes_community (gap 15.0),
mathse_vote_score (25.5), so_votes (7.9) — per-cell generation configs,
frozen-bank scorers (same manifest pattern as cw), registered dense models,
real-item injection from each cell's certified frame. Cross-cell prereg
UNCHANGED: A' increases with static relative gap.
E2 (richer real axis, cw): re-select prompts favoring those with >=3
dense-held-out real stories; target pools with 3+ real items.
E3 (variance reduction, cw): 150 -> 400 prompts; 3-seed judge averaging
reserved for the final confirmatory pass (3x scoring cost).
E4 (LATER, after E1-E3): iterated optimization — 3-5 rounds of
generate -> select-by-bank -> condition next round on winners; track
certified-dense score by round. Prereg: bank score climbs monotonically;
dense score flat/declining; divergence grows with rounds.

Q3 dissection (paper-4-adjacent):
P1 RM-representation probe: penultimate embeddings of the certified RM on
the 2,915 pool texts; linear probe real-vs-generated, grouped CV by prompt.
Prereg readout: probe AUC >> preference-head AUC (.553) = "represents but
doesn't use" (H1-strong); probe ~ .55 = axis absent from representation.
P2 capacity control: same trainer recipe, labels = real-vs-generated
(2,700 generated + matched real sample from the honest frame, stable-hash
split by prompt_id). AUC >= .9 = no capacity limit, training-signal absence
confirmed. Already-in-hand: hightemp tanking (dense rank .22) rules out
blanket OOD blindness.

## P1 RESULT (2026-08-24) — "REPRESENTS BUT DOESN'T USE" [paper-4-adjacent]

Linear probe on the certified RM's penultimate embedding (the exact vector
its preference head reads), real-vs-generated, grouped 5-fold:
**AUC = .997** (folds .990-1.000) vs the preference head's .553 on the same
axis. The humanness direction is fully present in the model's
representation; the preference readout ignores it — consistent with a head
trained purely on human-vs-human comparisons (the axis had zero variance in
its training signal). Combined with the hightemp-tanking evidence, H2
(broad OOD fragility) is REJECTED; H1 (axis-absent-from-training-signal)
confirmed in strong form. P2 same-recipe discriminator relaunched after a
split-ratio fix (greedy row-weighted 80/10/10) as the capacity control.
Artifacts: rung2/rung2_rm_penult_embs.npz; filed paper-4-adjacent per user.

## P2 RESULT (2026-08-25) — capacity control CONFIRMS

Same recipe (Llama-3.1-8B LoRA, identical trainer), labels =
real-vs-generated, prompt-grouped 80/10/10: test AUC **1.000** (epoch 2;
.996 at epoch 1), accuracy .998. Zero capacity limitation. Together with
P1 (.997 linear probe vs .553 preference head): the dense preference
model's humanness-blindness is purely a TRAINING-SIGNAL ABSENCE — the same
architecture, data pipeline and even the same internal representations
support near-perfect real-vs-LLM discrimination the moment the label asks
for it. Filed paper-4-adjacent.

---

## ADDENDUM F (2026-08-25, recorded BEFORE running): capacity-null probes + metric-tree revival + absolute-VA scoping

User challenge: is V+A a true articulability frontier, or just what ~150
metrics + a GBM head achieve against an 8B dense model? The epsilon-tail
null (many tiny articulable criteria, individually below discovery
threshold, jointly large) is NOT excluded by mining saturation or
missing-mass counts. Probes:

F-a CRITERIA-SCALING CURVE: subsample k criteria from the final bank
  (k in {8,16,32,64,96,128,144}, 3 stable-seeded subsets per k), refit the
  frozen recipe grouped-OOF, plot AUC(k), fit saturation asymptote, compare
  asymptote+CI vs dense T. Cells: cw_community + peer_verdict (mac).
  Readout: has AUC(k) flattened below T by k=144, or still rising?
F-b DISTILLATION PROBE: grouped-OOF regression of the DENSE SCORE (not the
  label) on the 144 bank features; report rank corr(predicted, dense).
  High = criteria span dense's signal (gap is estimation) ; low = dense uses
  directions orthogonal to everything named. USER NOTE, adopted: the
  orthogonal component is a candidate OPERATIONAL DEFINITION of the tacit
  part — if it exists AND predicts y beyond the bank span, tacitness is
  localized as "label-relevant text directions outside the named span."
  Compute both: orthogonality AND label-relevance of the residual.
F-c HEAD SWAP: MLP head (2-3 widths) vs GBM on identical 144 features,
  grouped-OOF; if unchanged, head capacity is not the binder.
F-d METRIC-TREE REVIVAL (user): review ALL methods/ legacy code for ideas
  to update with current VA machinery; audit + modernize metric tree
  (conditional/selective metric application to subpopulations — also a
  subcommunity-discovery instrument); attempt materially higher performance
  than the flat bank on >=1 cell. Two birds: capacity-architecture
  alternative + subcommunity structure.
F-e ABSOLUTE-VA SCOPING (user): combine (i) missing mass of DISCOVERED
  metric species (exists; appendix plots to verify/build) with (ii)
  per-metric optimal-prompt upper bounds from Paper #2's machinery
  (DPI fixed-target cap = the only certified bound per that lineage) into
  an absolute V+A estimate. Deliverable: scoping note w/ estimator sketch,
  data requirements, cost.

## ADDENDUM F RESULTS (2026-08-25 overnight) — capacity null REJECTED on all three legs; tacit residual LOCALIZED

Cells: cw_community, peer_verdict. results/addF_capacity_*.json, cbtree_*.json.

F-a SCALING CURVE: the curve has FLATTENED at the bank's size.
  cw: AUC(k) .607(k=8) -> .665(k=144); fitted asymptote A=.6675±.006 vs
  dense .792 — remaining width-headroom ≈ .002 vs gap .127 (~2%).
  peer: A=.664±.021 vs dense .777 (asymptote ≈ current bank).
  tau≈39 features both cells (marginal value halves every ~27 criteria).
  "More criteria of the same kind" cannot close the gap.
F-b DISTILLATION: bank reconstructs the dense SCORE at rho=.53 (cw)/.64
  (peer) — dense uses directions substantially outside the criteria span.
  The orthogonal residual is STRONGLY label-relevant: alone AUC .721 (cw) /
  .666 (peer); stacked on the bank it adds +.108 / +.064, reaching .773 /
  .732 (≈ dense .792/.777). THE USER'S FRAMING IS CONFIRMED AND QUANTIFIED:
  the tacit component = label-relevant text directions orthogonal to
  everything nameable, carrying ~85% of the articulability gap.
F-c HEAD SWAP: MLPs UNDERPERFORM the frozen GBM (cw .651-.659 vs .665;
  peer .626-.640 vs .668). Head capacity is not the binder.
F-d TREE (stage 1): conditional bank tree LOSES to the flat bank on the same
  folds (cw -.058, peer -.027): with the EXISTING criteria, selective
  application costs more (data splitting) than local specialization gains.
  Honest null; stage-2 node-conditional MINING remains the open question,
  and weak evidence against strong subcommunity heterogeneity at these n.
Caveat carried everywhere: all bounds are conditional on the criteria-
generating process (LLM-articulable criteria); that is now the explicitly
stated scope of "articulable."
Consequence for F-e scoping: implementation headroom on tail criteria is
irrelevant (F-a); the absolute-VA estimate simplifies to
asymptote + top-24 cap headroom + small coverage term.

## ADDENDUM E2/E3 RESULT (2026-08-25) — expanded frame REPLICATES Addendum D

402 pools (>=3 real each; 8,816 rows; batteries: bank anchors .552, dense
.785, 0/129 collapsed, prompt-hash identical). rung2_readout_cw_v3_e2.json:
- **A' = +.048 [+.011, +.083], p>0=.994** (v2 frame: +.067 [+.002,+.129]).
  Two frames, same direction; CI now comfortably clear of zero.
- R2 replicates: bank real-vs-gen .715, dense .541.
- R3: real towers on bank channel (.676); "human" +.026 over plain on bank,
  "veryhuman" NEGATIVE again (.515 vs plain .528); dense flat everywhere;
  hightemp catastrophic (.084/.173). Humanness still not instructable.
- Picks: bank policy retrieves a real story in 69% of pools (base 17.9%,
  3.9x); dense 52% (2.9x); picked real stories are winners .87/.83.
- R4: families indistinguishable on dense channel (.46-.53).

---

## ADDENDUM G (2026-08-25, design only — NOT launched): the PARALLEL-FORMS
## SURFACE — measuring "how much better would each metric be under an ideal LM"

User question (paper §2.2.3 bullet 2) + user conjecture: the marginal value
of ADDING metrics should be higher when individual metrics are suboptimally
implemented (new correlated metrics compensate for implementation noise).
Is there a principled instrument? YES — classical test theory.

### The framework
Each criterion i has a construct m*_i (what an ideal implementation would
measure) and our implementation m̂_i = m*_i + implementation error. Classical
results give both requested quantities:
- ATTENUATION (Spearman 1904): validity(m̂_i) = validity(m*_i)·sqrt(r_i),
  where r_i = reliability. So the ideal-implementation validity of metric i
  is recoverable from its measured validity and its reliability.
- SPEARMAN-BROWN: averaging J independent implementations ("parallel forms")
  of the same construct raises reliability to J·r/(1+(J-1)r) -> 1 as J->inf.
  So "ideal LM implementation" is OPERATIONALIZED as the J->infinity limit of
  averaging independent implementations — measurable by extrapolation, no
  ideal model required.
- THE USER'S CONJECTURE becomes a measurable CROSS-PARTIAL: estimate the
  response surface AUC(k, J) (k = number of metrics, J = parallel forms per
  metric). Conjecture = d²AUC/dk dJ < 0: the k-curve flattens faster at high
  J. One surface simultaneously yields: per-metric ideal bounds (J-axis),
  many-metric bound (k-axis), their interaction (cross-partial), and the
  ABSOLUTE articulated estimate A(k_max, J->inf) to compare against dense T.

### Identifiability caveat (why Paper-2 caps are still wanted)
Parallel forms remove implementation VARIANCE but not shared BIAS: if every
implementation of "narrative tension" (any prompt, same judge family) misses
part of the construct, averaging never recovers it. So:
  parallel-forms limit  <=  ideal-LM bound  <=  DPI fixed-target cap.
The surface gives the certified LOWER estimate of the ideal; Paper-2's DPI
cap gives the certified UPPER bound. Run caps on the top-3 criteria only as
a calibration spot-check (do disattenuated validities approach the caps?).
Cross-family forms (one GLM/Llama form among the Gemma forms) probe the
shared-bias magnitude directly: if cross-family reliability ≈ within-family
reliability, family bias is small.

### Experiment spec (per cell; start cw_community, then peer_verdict)
1. Criteria: top-24 by local informativeness (F-a licenses ignoring the tail).
2. Forms per criterion, J=4: (a) the frozen GEPA prompt (exists); (b) a
   blind paraphrase generated from the criterion definition alone; (c) a
   minimal definition-only prompt; (d) one cross-family form (GLM-Lite or
   Llama-70B judge, same block format). Forms b-c authored without seeing
   scores (bias independence); form d flags family bias.
3. Score the cell's E-frame on forms b-d: cw 7,008 rows x 24 x 3 ≈ 505k
   judge calls (≈ one E2-scale overnight run, TP=2); peer 1,244 x 24 x 3 ≈
   90k. Anchor batteries + collapse gates as always.
4. Readouts (all grouped-OOF, frozen recipe):
   - per-criterion reliability r_i (mean inter-form Spearman) + disattenuated
     univariate validity /sqrt(r_i);
   - AUC(k, J) for k in {4,8,16,24}, J in {1,2,3,4} (form subsets averaged;
     J=1 averaged over which single form, so J=1 isn't privileged);
   - Spearman-Brown extrapolation of the J-axis at each k -> A(k, inf);
   - the cross-partial sign test (USER CONJECTURE, preregistered);
   - A(24, inf) vs dense T vs the F-a asymptote (which was a J=1 object —
     the surface generalizes F-a).
5. Prereg: (i) r_i in [.5, .9]; (ii) AUC increases in J; (iii) cross-partial
   NEGATIVE (conjecture); (iv) A(24, inf) remains below dense T by a margin
   consistent with the F-b residual (~.10 on cw). (iv) failing = the gap was
   implementation noise all along — that would be a major, reportable
   reversal, which is exactly why the design must be frozen first.
Cost: prompt authoring (LLM, cheap) + ~600k judge calls total + CPU refits.
GLM form budget ≈ 2-3M tokens (within Lite weekly allowance).
STATUS: awaiting user go (scoring budget), then launchable in one command.

## ADDENDUM G2 (2026-08-25, user): JUDGE-SCALE LADDERS as the empirical sanity
check alongside the parallel-forms theory.
- Metrics: the top-20/24 per task AS CHOSEN UNDER Gemma-4-31b (fixed set;
  varying the chooser is a robustness arm, not primary).
- Ladders (same-family-only rule): Llama-3.2-1B -> 3.2-3B -> 3.1-8B ->
  3.3-70B(-FP8) [4 rungs]; Gemma-3-4B -> 12B -> 27B [3 rungs, one
  generation]; Gemma-4-31b = the reference instrument.
- Per user: each metric GEPA-optimized PER JUDGE (each rung at its own best
  implementation — otherwise scale effects confound with prompt fit).
  Staging: pass 1 scores all rungs with the FROZEN prompts (un-GEPA'd
  ladder); pass 2 GEPA-per-rung on top. The pass1-vs-pass2 delta BY SCALE is
  itself a readout of the user's conjecture (small judges should gain more
  from prompt optimization if implementation noise is what metrics
  compensate for).
- Readouts: per-metric validity and reliability BY JUDGE SCALE; bank AUC(k)
  by rung; saturation test (70B ≈ 31b reference?) — an independent estimate
  of "better LM" headroom to triangulate the Spearman-Brown extrapolation
  and the DPI caps.

## ADDENDUM G3 (2026-08-25): balancing GEPA cheapness vs e-cert guarantees
User question: e-cert per-metric caps are expensive; do we just asymptote
GEPA? NO — a search plateau bounds the SEARCH, not the optimum (the e-tail
worry, relocated to prompt space). Adopted strategy:
1. GEPA -> bound-like via MULTI-RESTART (R=3, different seeds/proposer
   families) + CAPTURE-RECAPTURE over high-performing prompt species
   (port the Paper-2 "mining moves level / audit sets width" machinery).
2. e-cert (DPI fixed-target cap) as CALIBRATION not coverage: 2-3 metrics
   per task, chosen where GEPA was least converged x lowest reliability x
   highest bank sensitivity; measure bracket width (cap - recapture-adjusted
   asymptote) there; transfer the width as an empirical correction to the
   cheap estimates on the remaining top-24.
3. Claim logic: if caps land ABOVE dense they are uninformative for "gap
   survives ideal implementation" — then the evidence is the TRIANGULATION
   of three independent cheap estimators saturating below dense: (a)
   multi-restart GEPA asymptote + recapture interval, (b) parallel-forms
   Spearman-Brown limit (G1), (c) judge-scale saturation (G2). Disagreement
   among them is itself diagnostic (parallel-forms << GEPA asymptote =>
   prompt search finds label-fit, not construct fidelity).
Final absolute-VA quote = bracket [triangulated asymptote, calibrated cap].
G2 pass-2 amended: GEPA-per-rung = R=3 restarts with recapture readout.

## ADDENDUM G1 RESULT (2026-08-26) — ideal implementation adds ~.006; gap intact

cw_community, top-24 criteria, forms a (frozen GEPA) / b (blind paraphrase) /
c (name-only). Battery .703/1.0 pass. rung2/g1_surface_cw.json.
- Reliability: median .728 (range .52-.84). Implementation noise moderate.
- Form validities NEARLY IDENTICAL: a .576, b .573, c .572 — the criterion
  NAME carries almost all implementable signal; per-criterion GEPA buys
  ~.004 univariate.
- Bank-level J-axis: k=24 AUC .631 (J=1) -> .635 (J=3); Spearman-Brown
  A_inf = .637. TOTAL ideal-implementation headroom ≈ +.006 AUC.
  Rough absolute-VA assembly: F-a k-asymptote .668 + J-headroom .006 ≈ .674
  vs dense .792 — the gap shrinks ~7% and survives (prediction: .01-.02
  shrink, 5-10% collapse risk — landed inside).
- Disattenuated per-criterion validity .682 vs raw .576 — the correction is
  REAL per-criterion but doesn't reach the bank because the bank's
  redundancy already averages implementation noise (the user's conjecture
  vindicated at the BANK level...)
- ...but the preregistered CROSS-PARTIAL is NOT confirmed: J-gain at k=8
  (+.0005) < at k=24 (+.0043) — mild COMPLEMENTARITY between coverage and
  fidelity, not substitution; at k=4, form-averaging even hurts. Reported
  against prereg.
Caveat: J-subsets average heterogeneous forms (a/b/c differ in strength);
J=1 is the mean of three single forms. G2 ladder = leg 2, in flight.

### RUN LOG — G2 pass-1 judge-scale ladder, cw_community (2026-08-25)
Scoring on sk3 (6/7 rungs; llama1b rung FAILED at engine level, not diagnosed —
low value, dropped). Readout local: grouped-OOF logistic bank AUC over the 24
frozen form-a criteria, mean univariate validity, mean per-criterion Spearman
agreement with the certified Gemma-4-31b reference scores.
Artifacts: rung2/g2_ladder_readout_cw.json, g2_form_scores_<rung>.npz.

| rung | battery pos/neg | bank AUC (OOF) | mean univ | agree w/ ref |
|---|---|---|---|---|
| llama3b | .599 | .5596 | .5235 | .110 |
| llama8b | .497 ORDERING FAILS | .5354 | .5090 | .079 |
| llama70b | .625 | .6381 | .5649 | .432 |
| gemma3-4b | .606 | .5735 | .5179 | .069 |
| gemma3-12b | .725 | .6190 | .5480 | .311 |
| gemma3-27b | .705 | .6221 | .5464 | .312 |
| REF gemma4-31b | .703 | .6247 | .5764 | 1.000 |

Findings (same-family staircases only, per standing rule):
1. SATURATION: gemma3 12b→27b buys +.003 bank AUC (.619→.622), and 27b sits
   .003 under the gemma4-31b certified reference (.625). Llama saturates high:
   70b hits .638, +.013 over the reference. The judge-scale curve is FLAT at the
   top — consistent with G1's parallel-forms headroom estimate (~+.006). The
   articulated frontier is not judge-capacity-limited.
2. IMPLEMENTATION DISAGREEMENT != VALIDITY LOSS: llama70b agrees with the
   reference at only rho=.43 per-criterion yet posts the best bank AUC. Two
   large judges implementing the same named criteria differently still land at
   the same bank-level validity — bank redundancy averages implementation noise
   (the G1 mechanism, reproduced across model families).
3. LLAMA8B ANOMALY: fails its anchor battery (pos-vs-neg .497, ordering
   broken) and scores BELOW llama3b — a rung-specific instrument failure, not a
   scaling trend; excluded from staircase claims. Never quote llama8b numbers.
4. Small judges (3-4B) are not at chance (.56-.57 bank AUC): criterion names
   carry signal even for weak implementers, echoing form-c (name-only) ~ form-a.

## ADDENDUM H (2026-08-26, user): SPURIOUS-FEATURE DECAY CURVE FOR THE DENSE ARM
Paper §"more metrics" flag: "spurious features decay our dense performance by
XXX with an asymptote of YYY" was never run. Design (frozen before results):
mirror of the bank-saturation curve (addF F-a), pointed at the dense readout.
- Data: per cell, F2 machinery verbatim — f2_deconf.load_E (dense readout on
  the master-ledger E rows) + f2_cells.ADAPTERS (nuisance matrix, Track-B
  spurious channels + declared STRUCT, aligned to E by the same join asserts).
- Procedure: for k in {0,1,2,4,8,16,32,...,K_all}: R=20 seeded random subsets
  of k nuisance channels; CROSS-FITTED linear residualization (GroupKFold(5)
  on the cell's grouping unit; per-fold median impute + standardize; OLS of
  dense on the k channels fit on train folds, residual taken on the held-out
  fold); AUC of the OOF residual over all E rows; mean±sd over the R subsets.
  Full-K point exact (R=1). Asymptote fit A + B*exp(-k/tau) on the means, as
  in F-a.
- Readouts: decay = AUC(k=0) − A; asymptote A; tau; nuisance-alone AUC for
  context; comparison to the bank asymptote (cw: .668).
- CAVEAT (frozen): linear partial-out removes ALL dense variance collinear
  with nuisance, including genuinely shared quality variance — so A is a
  LOWER bound on deconfounded dense performance and is anti-dense-
  conservative. The F2 stacked increment (d)-(c) stays the primary
  deconfounding readout; this curve is the descriptive decay the paper
  sentence wants.
- Cells: all 14 F2 adapters; cw_community quoted in the paper sentence.

### RUN LOG — ADDENDUM H dense decay curves, 11 local cells (2026-08-26)
Script: fusion/f2_dense_decay.py. Artifacts: results/f2_dense_decay_<cell>.json.
Plateau rule (post-hoc, applied uniformly): the exponential asymptote A is
quotable only if |A - full_K point| < .01; otherwise the curve has not
plateaued by full K and the full-K point is quoted as an UPPER bound on decay
observed / LOWER bound on the deconfounded floor is NOT certified.

| cell | K nuis | dense k0 | full-K resid | A (fit) | plateaued |
|---|---|---|---|---|---|
| cap_finalist | 38 | .612 | .521 | .518 | yes |
| cw_community | 70 | .792 | .737 | .738 | yes |
| hashtagwars_verdict | 44 | .732 | .591 | — | NO, quote full-K |
| jokes_community | 58 | .747 | .641 | .641 | yes |
| nc_agree | 25 | .603 | .568 | .569 | yes |
| nc_outcome | 22 | .624 | .569 | .571 | yes |
| nc_responded | 56 | .817 | .732 | .733 | yes |
| peer_curation | 52 | .594 | .569 | — | NO, quote full-K |
| peer_revealed | 57 | .884 | .773 | .777 | yes |
| peer_verdict | 43 | .777 | .654 | — | NO, quote full-K |
| press_verdict | 23 | .774 | .729 | .726 | yes |

Not run: bbc_mostread (no t0_rows dense ledger yet — cell still pre-scoring);
mathse_accepted_verdict + mathse_vote_score (populations sk3-side; launched on
sk3 CPU 2026-08-26, /tmp/h_mathse.log).
Descriptive notes (no verdicts): decay ranges .034 (nc_agree) to .095+
(cap_finalist, where the deconfounded dense floor sits near chance at .518 —
the one cell where the dense edge is largely nuisance-collinear). cw quoted in
paper §results: decay .054, A .738±.001, still +.070 over the bank asymptote
.668. Caveat travels in every JSON: linear partial-out also removes shared
quality variance, so floors are anti-dense-conservative; F2 (d)-(c) remains
primary.

ADDENDUM H mathse completion (2026-08-26, run on sk3 CPU, artifacts pulled):
| mathse_accepted_verdict | 36 | .644 | .579 | — | NO, quote full-K |
| mathse_vote_score | 36 | .654 | .578 | — | NO, quote full-K |
Final set = 13 cells; bbc_mostread excluded (no dense ledger). Appendix table
added to paper: main.tex app:dense-decay (tab:dense-decay), pointer from the
§results saturation paragraph.
