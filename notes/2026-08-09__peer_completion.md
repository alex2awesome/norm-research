# Peer review completion — the first field with all three preference types terminal

Date: 2026-08-07/09. Protocol: the FROZEN preregistration
`notes/2026-08-05__layer3-closure-prereg.md` (FREEZE DECLARATION 2026-08-06 +
FREEZE ADDENDUM 1 (B-side missing mass, stacked increment) + ADDENDUM 2 (Track-B
upstream-factor mode, MIXED flag) + ADDENDUM 3 (MIXED-channel decomposition pass) +
ADDENDUM 4 (position-in-container Track-B prior)).

Three jobs, in the order the brief set them:

1. **peer REVEALED topic-stratified robustness check** — does the dense edge on the
   citation-percentile cell survive *within* topic, or is it topic composition?
2. **peer REVEALED closure campaign** — conditional on (1).
3. **peer CURATION map extension to the stopping rule** — unconditional.

Code + artifacts: `methods/taste_decomposition/closure/peer_revealed/` and
`methods/taste_decomposition/closure/peer_curation_ext/`. Both directories reuse the
map-focused batch's machinery (`closure/maps_batch1/`, `closure/maps_hw_si/`) rather
than rebuilding it: the same cell adapter (`cells.py`), the same frozen fitting spec
(`closure_core.py`), the same sealed dual-track proposer harness, the same species /
Good-Turing selection, the same blind-audit and arbiter scripts, and the same
Gemma-4-31B offline-batch scorer. What is new here is listed in §0.3.

---

## 0. Terms, and what is reused vs. new

### 0.1 Terms, spelled out (standing rule)

**V** = the block of deterministic surface features; **A** = the articulated-criterion
bank, Gemma-4-31B-judged rubric scores; **VA_nl** = HistGradientBoosting aggregation of
the V+A matrix, mean over seeds {0,1,2}; **VA_lin** = its logistic counterpart;
**T** = the dense readout (Llama-3.1-8B LoRA reward model on raw text), always the
**same-rows** rescore restricted to that model's own eval/test rows;
**Δ_beyond** = T − VA_nl, the unarticulated residual; **Δ_r** = Δ_beyond after r rounds
of mining; **ε** = .005, the frozen per-round saturation threshold; **AUC** = area under
the ROC curve; **FIT+MINE / MONITOR** = the closure splits; **M** = the mining slice
(dense-held-out rows inside FIT+MINE); **HONEST population** = every dense-held-out row
(M ∪ MONITOR), the population on which T and VA are both out-of-sample;
**Track A** = the quality-criterion miner (k_A = 15 scored criteria/round);
**Track B** = the suspected-spurious-channel miner (k_B = 10 scored channels/round);
**alone-AUC** = the AUC of a single judged channel's raw 0–10 score, no fitting;
**joint B model** = HistGB on the B channels only; **stacked increment** =
AUC(logistic stack of joint-B + X) − AUC(joint-B), the stratification-free discount;
**mixed channel** = a Track-B channel whose conjectured upstream parent plausibly causes
real quality as well (ADDENDUM 2); **Good-Turing missing mass M̂** = f1/N over proposal
*species*; **LOPO** = leave-one-proposer-out jackknife; **P** = fleet size;
**GEPA** = the prompt-iteration pass required before any final quoted phrasing —
**not applied here**, so everything below carries a pre-GEPA flag.

### 0.2 What is REUSED (reuse-before-rebuild is binding)

| reused object | source | used for |
|---|---|---|
| same-rows dense predictions (T) | `closure/samerows_preds/peer_{revealed,curation}_dense_preds_slim.csv` | every T in this note; peer revealed HONEST T = **.8842** on n=478, peer curation **.5936** on n=1,571 |
| closure splits (FIT+MINE / MONITOR / mining slice) | `maps_batch1/peer_*_splits.json` | unchanged; MONITOR ⊂ dense-held-out, 50/50 inside the held-out groups |
| round-0 baseline | `maps_batch1/peer_*_r0_context.json` | Δ_beyond at r0 |
| rounds 1–2 scored criteria + routing + collapse gates | `maps_batch1/peer_*_r{1,2}_{scores.npz,routing_final.json,score_report.json,species.json}` | the bank state entering round 3, the accumulated Track-B nuisance set, and the MIXED parents the decomposition pass acts on |
| mined trend/timing channels | r1 `B07` trend/hype-keyword density, r2 `B04` Trend-Aligned Vocabulary, r2 `B10` Canonical Ecosystem Naming | JOB 1's channel-decile strata and JOB 2's discount |
| machinery | `maps_batch1/` + `maps_hw_si/` (`cells.py`, `closure_core.py`, `stage1_slice.py`, `harness_maps.py`, `run_fleet.py`, `species.py`, `audit.py`, `arbiter.py`, `decompose_round.py`, `score_gemma_maps.py`, `readout.py`) | copied, not rewritten |

Nothing was re-judged that already had a corpus-wide score, and no dense model was
re-run. The R0 refit reproduces the maps batch to 4 decimals (VA_nl HONEST .7240 vs
.72397 on file), and the R2 refit reproduces .7632 vs .763 — the reuse is verified,
not assumed.

### 0.3 What is NEW here

1. `peer_revealed/job1_topic_strat.py` — the topic-stratified robustness check (JOB 1).
2. `peer_*/mixed_parents.py` — FREEZE ADDENDUM 3 parent selection over the
   **accumulated Track-B MIXED channels** (the humor batch's `new_parents.py` selected
   bank criteria off a SHAP screen instead; this is the object Addendum 3 actually
   names). Parents ranked by |alone-AUC − .5| on **FIT+MINE only** — the M3 precedent
   that a design decision never reads MONITOR.
3. `peer_*/discount_cumulative.py` — the **cumulative** Track-B discount with
   **matched sampling** (freeze: "matched sampling once spurious-alone > .65"; peer
   revealed's joint nuisance model is already .768, so matched sampling is on from the
   first campaign round), the decile version for continuity, the ALL/STRICT mixed band,
   and the stacked increment over the whole accumulated nuisance set.
4. **Corpus-matched planted probes** (`audit.py`): four authored pairs for an ML paper
   abstract, two pairs drawn per round by stable sha256 of the round tag, so the fresh
   auditor each round never re-audits the same planted pair. This closes the maps
   batch's own carry-forward (§1d of that note).
5. **P = 6, three families.** GLM-5.2 was re-probed once at the start of this campaign
   on both Lite keys and came back LIVE (HTTP 200, thinking enabled,
   budget_tokens=2048 / max_tokens=32000) — the 5-hour cap that forced the maps batch
   down to P = 4 / two families has lifted. All 12 sealed slots returned parseable
   sets. Good-Turing numbers here are therefore at P = 6 and are **not** directly
   comparable to the maps batch's P = 4 numbers.

---

# JOB 1 — peer REVEALED topic-stratified robustness check

**Question.** Peer revealed rides a topic floor: its Track-B spurious map is headed by
trend-vocabulary channels at alone-AUC .64–.71, and the Layer-1 note carries the caveat
that citation percentile is substantially topic-predictable. If Δ_beyond = T − VA_nl is
a **topic-composition** effect — the dense model is simply better at telling which
subfield a paper is in, and subfield predicts citations — then Δ should collapse once
papers are compared only against papers on the same topic.

**Design.** CPU only, all reuse. Population = HONEST (the dense model's own held-out
rows, n=478, where T and VA_nl are both out-of-sample); MONITOR (n=244) reported
alongside. T = the same-rows rescore (.8842). VA_nl refit with the frozen spec at two
bank states: **R0** = V + A_base (83 features, HONEST .7240) and **R2** = plus the
A-routed criteria of map rounds 1–2 (109 features, HONEST .7632). R2 is primary
because it is the strongest bank on file and therefore the most conservative test of T.

Strata, three families:
- **bge topic clusters** — k-means on BAAI/bge-large-en-v1.5 abstract embeddings,
  k ∈ {5, 10, 20}, **fit on the FIT+MINE (train) side only**, HONEST/MONITOR rows
  assigned to the nearest train-side centroid;
- **mined trend-channel deciles/quintiles** — the three trend/timing channels already
  scored corpus-wide in the maps batch, and the mean of their FIT+MINE z-scores;
- **publication year** — the conjectured upstream parent of the trend channels.

Readout = n-weighted within-stratum AUC (`closure_core.stratified_auc`, min_n = 20) for
T and for VA_nl; Δ_strat = T_strat − VA_strat; group-level paired bootstrap on Δ_strat
with the strata re-read inside every replicate.

**Verdict rule, declared before running:** SURVIVES if the primary stratified Δ (k=20,
R2 bank, HONEST) keeps a CI excluding 0 AND retains ≥ 50% of pooled Δ, and the
stratification-free dense increment over topic+bank is > 0 with a CI excluding 0;
COLLAPSES if the CI covers 0 or the point estimate falls below 25% of pooled; PARTIAL
in between.

## 1.1 How big is the topic floor, actually

It is large, and it is the largest single nuisance dimension on this cell.

| topic instrument | AUC on HONEST (n=478) |
|---|---:|
| topic score alone (grouped-OOF logistic on the top-50 PCs of the abstract embedding) | **.7415** |
| mean-z of the three mined trend/timing channels, alone | **.7508** |
| Trend-Aligned Vocabulary (r2 B04), alone | .7114 |
| Trend/hype-keyword density (r1 B07), alone | .6773 |
| Canonical Ecosystem Naming (r2 B10), alone | .6407 |
| **the whole 109-feature articulated bank (VA_nl, R2)** | **.7632** |
| the dense model (T) | **.8842** |

The topic floor is not a caveat-sized effect. Raw abstract topic alone reaches .742 and
three judged trend channels reach .751, both statistically on top of the entire
articulated bank at .763. Whatever else is true, **the bank on this cell is close to a
topic-and-fashion detector**.

## 1.2 The stratified table — the edge does not merely survive, it grows

HONEST population, R2 bank, min_n = 20. Pooled Δ = .8842 − .7632 = **+.1210**.

| stratification | strata | coverage | T_strat | VA_strat | **Δ_strat** | retained | 95% CI on Δ_strat |
|---|---:|---:|---:|---:|---:|---:|---|
| bge k-means k=5 | 5 | 1.00 | .8736 | .7535 | **+.1201** | 99% | [+.073, +.166] |
| bge k-means k=10 | 10 | .97 | .8679 | .7390 | **+.1289** | 107% | [+.075, +.180] |
| **bge k-means k=20 (primary)** | 16 | .84 | .8484 | .7010 | **+.1474** | **122%** | **[+.090, +.218]** |
| bge k-means k=20, min_n=10 | 18 | .97 | .8589 | .7127 | +.1463 | 121% | — |
| Trend/hype-keyword density, deciles | 10 | 1.00 | .8687 | .7361 | +.1326 | 110% | [+.081, +.177] |
| Trend-Aligned Vocabulary, deciles | 10 | .99 | .8587 | .7293 | +.1294 | 107% | [+.083, +.178] |
| Canonical Ecosystem Naming, deciles | 10 | 1.00 | .8709 | .7430 | +.1279 | 106% | [+.082, +.174] |
| **trend-family mean-z, deciles** | 10 | .97 | .8374 | .6900 | **+.1474** | **122%** | [+.086, +.208] |
| publication year | — | .96 | .8862 | .7661 | +.1202 | 99% | [+.077, +.164] |

**Twelve of twelve stratifications keep a bootstrap CI strictly above zero, and eight of
twelve give a LARGER Δ than the pooled number.** The mechanism is visible in the
columns: going from pooled to within-k=20-topic costs the **bank** .0622 of AUC
(.7632 → .7010) and costs the **dense model** only .0358 (.8842 → .8484). Conditioning
on topic hurts the articulated scorecard nearly twice as much as it hurts the dense
model. Same story on the trend-channel deciles: bank −.073, dense −.047.

Same direction on the R0 bank (the pre-mining scorecard), where it is stronger still:
Δ_strat +.166 / +.169 / +.178 at k = 5/10/20 against a pooled +.160, and +.201 on the
trend-family deciles (126% retained).

MONITOR (n=244) agrees but is noisy, as the maps batch warned it would be: k=5 +.154,
k=10 +.135, trend-family deciles +.198, year +.144. The k=20 MONITOR cell is not
quotable — only 3 strata clear min_n = 20 and coverage falls to 25%.

## 1.3 Stratification-free control — give the bank the topic and ask again

Stratification loses rows as k grows (coverage .84 at k=20). The freeze's own
stratification-free instrument does not: build a topic score, hand it to the bank in a
grouped-OOF logistic stack, and read what the dense score still adds.

| stack, HONEST (n=478) | AUC | increment |
|---|---:|---:|
| topic alone | .7415 | — |
| topic + bank | .7883 | bank over topic **+.0467** |
| topic + dense | .8846 | dense over topic **+.1431** |
| topic + bank + dense | .8867 | **dense over topic+bank +.0984**, 95% CI [+.068, +.130], P(>0)=1.00 |
| topic + 3 trend channels + bank | .8137 | — |
| topic + 3 trend channels + bank + dense | .8869 | **dense over topic+trend+bank +.0732**, CI [+.046, +.099], P(>0)=1.00 |

On MONITOR: dense over topic+bank **+.1288** [+.076, +.182]; over topic+trend+bank
**+.0746** [+.035, +.115]. Both CIs exclude zero.

Read the two right-hand columns of the first table together: handed the same topic
representation, the **bank gains +.047 and the dense model gains +.143** — the dense
model carries three times as much information that is orthogonal to topic as the
109-feature scorecard does. Add the three judged trend channels on top of topic and the
bank climbs to .814, but the dense model *still* adds +.073 on top of all of it.

## 1.4 VERDICT — **SURVIVES** (and the caveat reverses)

Every clause of the pre-declared rule is met with room to spare: primary Δ_strat
+.1474 with CI [+.090, +.218] (retains 122%, threshold was 50%), and the
stratification-free dense increment over topic+bank +.0984 with CI [+.068, +.130].

The substantive finding is stronger than "survives". **The topic floor is real, it is
big (.742 alone), and it is a crutch of the ARTICULATED BANK, not of the dense model.**
Conditioning on topic removes about half again as much from the scorecard as from the
dense readout, so every topic control *raises* Δ_beyond. This confirms, with numbers,
the qualitative reading the maps batch reached from the discount tables
(`notes/2026-08-06__spurious_maps_batch1.md` §4.1): on peer revealed the topic-floor
caveat cuts against the bank, which is the opposite of how a topic-floor caveat is
usually read.

**Consequence for the roster: JOB 2 runs.** The caveat that travels forward is not
"Δ may be topic composition" — that is now excluded at P(>0) = 1.00 under three
independent topic instruments — but the narrower and still-real one: **peer revealed's
absolute AUC levels (all of V, A, VA and T) sit high partly because the construct leans
on topic popularity, so this cell's LEVELS are not comparable to other cells' levels;
its Δ is.**

Artifacts: `closure/peer_revealed/job1_topic_strat.py`, `job1_topic_strat.json`,
`job1.log`, embedding cache `abstract_emb_bge_large.npz`.

---

# JOB 2 — peer REVEALED full closure campaign (rounds 3–5 under the frozen protocol)

Rounds 1–2 are the maps batch's, reused verbatim. Rounds 3, 4 and 5 were run here to the
frozen protocol: MONITOR ⊂ dense-held-out; decomposition-first on the accumulated MIXED
channels each round (3 parents → 3 candidate-real + 3 surface components, counting toward
k_A = 15 / k_B = 10, so each round is 12 fleet-A species + 3 real components + 7 fleet-B
species + 3 surface components); sealed P = 6 fleet over 3 families; fresh blind auditor
per round with 2 corpus-matched planted probe pairs; frontier arbiter on every dispute;
Gemma-4-31B offline-batch corpus-wide scoring with a K = 50/class anchor battery;
ε = .005; cap 5.

## 2.1 THE CURVE

HONEST population (n=478, T and VA both out-of-sample); MONITOR (n=244) alongside.
T = .8842 throughout (the same-rows rescore; it never changes — only the bank moves).

| round | bank feats | VA_nl HONEST | gain HONEST | VA_nl MONITOR | gain MONITOR | **Δ_beyond HONEST** | Δ_beyond MONITOR |
|---|---:|---:|---:|---:|---:|---:|---:|
| r0 | 83 | .7240 | — | .6961 | — | **+.1602** | +.1667 |
| r1 | 97 | .7511 | +.0271 | .7026 | +.0066 | **+.1331** | +.1601 |
| r2 | 109 | .7632 | +.0121 | .7169 | +.0143 | **+.1210** | +.1459 |
| r3 | 123 | .7692 | +.0060 | .7319 | +.0150 | **+.1150** | +.1309 |
| r4 | 137 | .7692 | **−.0001** | .7485 | +.0166 | **+.1150** | +.1143 |
| r5 | 151 | .7717 | **+.0025** | .7423 | **−.0062** | **+.1125** | +.1205 |

**The stopping rule fires at round 5.** On the HONEST statistic rounds 4 and 5 are both
sub-ε (−.0001, +.0025) — two consecutive — and this coincides with the hard cap B = 5.
The round-4 and round-5 gain CIs both straddle zero ([−.0111, +.0106] P(>0)=.48;
[−.0080, +.0129] P(>0)=.68). The MONITOR statistic tells a slightly later story (its r5
is the first sub-ε round), so the two statistics agree on "the miner is finished by
round 5" and disagree only on whether round 4 already counted; both land at the cap.

**Plateau: Δ_beyond = +.1125 on HONEST (+.1205 on MONITOR).** Five rounds of an
actively resourced, three-family, P = 6 miner adding **68 audited quality criteria**
(83 → 151 features, an 82% enlargement of the scorecard) closed **.0477 of AUC — 30% of
the round-0 residual — and 57% of that closure happened in round 1 alone.** The
remaining +.1125 is not discoverable by this miner.

## 2.2 The discount, at the full accumulated nuisance set

The freeze makes the discount cumulative and switches to matched sampling once
spurious-alone > .65. Peer revealed was over that trigger from round 1 (joint B = .752),
so matched sampling is the primary estimator here, with deciles and the
stratification-free stacked increment reported alongside.

Cumulative set after five rounds: **56 named nuisance channels, 9 MIXED parents retired
by decomposition** (recorded in `peer_revealed_retired_channels.json`, never deleted).

| readout, HONEST n=478 | ALL B (56 ch.) | STRICT, mixed dropped (11 ch.) |
|---|---:|---:|
| spurious-alone AUC (HistGB / linear) | **.8033 / .7974** | .7694 / .7769 |
| pooled Δ | +.1125 | +.1125 |
| **matched-sampling Δ_adj** (caliper .02) | **+.2541** (122 pairs) | **+.2109** (128 pairs) |
| decile-stratified Δ_adj | +.1934 | +.1594 |
| **stacked: dense increment over B + bank** | **+.0773** [+.051, +.106] | **+.0757** [+.049, +.103] |

MONITOR agrees: matched Δ_adj +.156 / +.175, decile +.197 / +.148, stacked +.090
[+.046, +.134] / +.066 [+.030, +.102].

Two things in that table are worth stating plainly.

**(i) The named nuisance set now BEATS the articulated bank outright.** 56 judged
"predictive but not quality" channels reach **.8033**, above the 151-feature
quality-criterion bank at **.7717**, and 91% of the way from chance to the dense model's
.8842. Five rounds of quality mining could not lift the bank past its own nuisance
shadow. This is the maps batch's regime-(i) finding at full campaign strength: on
citation percentile the *scorecard* is the topic-and-resources detector, and the dense
model is the thing that is not.

**(ii) Every discount raises Δ, and the conservative one still clears zero decisively.**
Matched sampling more than doubles the residual (+.113 → +.254) because holding the
nuisance score fixed costs the bank .17 of AUC and costs the dense model .03. That is
the same asymmetry JOB 1 found for topic, now measured against 56 channels instead of
3. The stratification-free stacked increment is the number to quote when a nuisance set
this large makes conditioning estimators unstable: **the dense model adds +.077
[+.051, +.106] over the joint nuisance model AND the full bank together**, P(>0) = 1.00,
and the ALL/STRICT band is only .0016 wide there (against .043 on matched sampling).

**Quote the discounted residual as a band: Δ_adj ∈ [+.076, +.254]**, whose lower end is
the stratification-free stacked increment and whose upper end is matched sampling; the
pooled undiscounted value +.1125 sits inside it.

## 2.3 Per-round instrument health and fleet bookkeeping

| round | fleet | proposals | A/B after routing | mixed B | misroute | probes | disputes → arbiter | anchors scram | pos-vs-neg | collapsed | NA |
|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| r3 | P=6 / 3 fam | 150 | 14 / 11 | 8 | .04 | 4/4 | 1 → B (mixed) | .908 PASS | .745 | 0 | .016 |
| r4 | P=6 / 3 fam | 150 | 14 / 11 | 10 | .04 | 4/4 | 1 → B (mixed) | .931 PASS | .704 | 1 | .015 |
| r5 | P=6 / 3 fam | 150 | 14 / 11 | 8 | .04 | 4/4 | 1 → B (mixed) | .952 PASS | .691 | 0 | .010 |

All 36 sealed slots (3 rounds × 6 proposers × 2 tracks) returned parseable, distinctly
named sets: 0 parse failures, 0 retries. Corpus-matched probes passed 4/4 in every
round (the maps batch's carry-forward is closed). All three arbitrated disputes went to
B and all three were flagged mixed — the same "quality-ish surface" boundary the
prereg's parked open question (b) named, now stable across five rounds: the disputes
were a method-family label (fine-grained routing), a methodological camp
(continuous ODE/PDE derivation), and a mathematical-domain label. The auditor's
pos-vs-neg anchor AUC ran .69–.75, higher than the maps batch's .64, and the scrambled
gate cleared at .908–.952.

## 2.4 Missing mass at the plateau (both tracks)

| round | track | S_obs | f1 | M̂ | LOPO jackknife | recapture | species ≥2 families |
|---|---|---:|---:|---:|---|---:|---:|
| r3 | A | 77 | 70 | .778 | [.75, .85] | .09 | 2 |
| r3 | B | 41 | 31 | .517 | [.52, .58] | .24 | 8 |
| r4 | A | 65 | 57 | .633 | [.59, .71] | .12 | 3 |
| r4 | B | 49 | 42 | .700 | [.68, .84] | .14 | 4 |
| r5 | A | 59 | 51 | **.567** | [—] | .12 | 4 |
| r5 | B | 36 | 26 | **.433** | [—] | .28 | 7 |

Both tracks' missing mass **falls** into the plateau (A .778 → .567, B .700 → .433) and
cross-proposer recapture rises on the B side (.14 → .28), which is the species-level
signature of a search that is genuinely running out rather than being cut short. It is
not zero: at P = 6 there is still a ~57% chance a seventh independent proposer names a
quality species not yet seen, so the honest claim is **"not discoverable by this miner
at this fleet size"**, not "not articulable". Value-weighting that mass carries the
freeze's stated assumption (unfound species resemble found ones in influence) and is
not quoted as a bound.

## 2.5 Swap readout

The (C₊, C₋) pair algebra flags rounds where the bank buys rank agreement with the dense
model by inheriting its errors. Round 3: dC₊ +.0051, dC₋ +.0130 — no signature. Round 4:
dC₊ +.0015, dC₋ −.0120 — **signature present** (the round whose HONEST gain was
−.0001: the bank moved toward the dense model's ordering without gaining AUC, which is
exactly what the swap readout is for). Round 5: dC₊ +.0025, dC₋ +.0029 — no signature.
So the one round in which the closure gain vanished is also the one round in which the
bank was tracking the dense model's mistakes; the plateau is not an artifact of the
bank quietly copying T.

## 2.6 The final Track-B map (all five rounds, by |alone-AUC − .5| on HONEST)

| round | alone AUC | mixed | channel | conjectured upstream parent |
|---|---:|:--:|---|---|
| r3 | .716 | | Extent of currently fashionable subfield vocabulary | surface carrier of "Trend/hype-keyword density" |
| r2 | .711 | | Trend-Aligned Vocabulary | submission timing |
| r3 | .682 | YES | Count of enumerated evaluation settings and resource markers | surface carrier of "Evaluation breadth" |
| r5 | .680 | YES | Count of named evaluation settings and resource markers | surface carrier of the r4 enumeration channel |
| r1 | .677 | YES | Trend/hype-keyword density | submission timing / topic fashion cycle |
| r5 | .665 | YES | Extent of coverage listing and breadth-signalling phrasing | surface carrier |
| r4 | .658 | YES | Extent of coverage listing and breadth vocabulary | surface carrier of "Evaluation breadth" |
| r2 | .641 | YES | Canonical Ecosystem Naming | author network and institutional exposure |
| r1 | .640 | YES | Evaluation breadth | institutional resources and team scale |
| r1/r3/r4/r5 | **.363–.375** | YES | the formalism / mathematical-notation family (4 variants) | author training, theoretical subcommunity |
| r5 | .610 | YES | Acronym and method branding density | author's practice in the format |
| r5 | .599 | | Promotional claim density | surface-only |

Three structural facts. **(a) The trend/timing family is the strongest single channel in
the cell and it survived decomposition**: the decomposition pass split "Trend/hype-keyword
density" (.677) into a candidate-real component and a surface component, and the SURFACE
component came back at **.716** — higher than the parent. Decomposition did not dissolve
this channel; it purified it. That is a real finding and it cuts against the parent
having had merit content. **(b) The evaluation-breadth family reproduces at .64–.68
across four independent rounds** with three different proposers' phrasings, tagged to
institutional resources and team scale — this is the "who could afford to run these
experiments" fingerprint, and it is the second-strongest thing in the cell.
**(c) The formalism family is strongly NEGATIVE (.363–.375, i.e. |AUC−.5| ≈ .14).**
Mathematical formalism *anti*-predicts citation percentile in this corpus with about the
same magnitude as ecosystem naming predicts it. Four independent proposers found it
across four rounds. This is the sign-contradicting case the prereg parked: the channel
carries real signal, in the direction opposite the proposer's rationale, and it is
reported as-is.

## 2.7 The topic caveat, carried

Every number in JOB 2 carries JOB 1's finding, and JOB 1 is the reason to believe them:
peer revealed's absolute levels sit high because citation percentile leans on topic
popularity (topic alone = .742, the 56-channel nuisance set = .803), so **this cell's
LEVELS are not comparable to any other cell's**. Its **Δ** is, and the Δ survives every
topic control tried — three independent topic instruments in JOB 1, and the full
56-channel nuisance set here, all of which *raise* it. The one-line form: the topic
floor is the bank's crutch, not the dense model's, so the topic caveat strengthens the
residual rather than threatening it.

---

# JOB 3 — peer CURATION map extension to the stopping rule

Rounds 1–2 from the maps batch, reused; rounds 3, 4, 5 run here under the same frozen
protocol as JOB 2 (same P = 6 fleet, same decomposition-first composition, same
corpus-matched probes, same arbiter, same scorer). T = .5936 (same-rows, n=1,571 HONEST).

## 3.1 THE CURVE

| round | bank feats | VA_nl HONEST | gain HONEST | VA_nl MONITOR | gain MONITOR | **Δ_beyond HONEST** | Δ_beyond MONITOR |
|---|---:|---:|---:|---:|---:|---:|---:|
| r0 | 109 | .5822 | — | .5602 | — | **+.0114** | +.0364 |
| r1 | 123 | .5703 | −.0119 | .5491 | −.0111 | **+.0233** | +.0475 |
| r2 | 138 | .5774 | +.0071 | .5539 | +.0048 | **+.0163** | +.0427 |
| r3 | 153 | .5861 | +.0088 | .5593 | +.0055 | **+.0075** | +.0373 |
| r4 | 168 | .5916 | +.0055 | .5690 | +.0097 | **+.0020** | +.0276 |
| r5 | 182 | .5795 | **−.0121** | .5688 | **−.0002** | **+.0142** | +.0278 |

**The campaign stops at the hard cap B = 5, not on the saturation rule.** On the HONEST
statistic the sub-ε rounds are r1 and r5, never adjacent. On the MONITOR statistic they
are r1, r2 and r5 — so **the frozen 2-consecutive-sub-ε rule fired retrospectively at
round 2** (−.0111 then +.0048, both < ε), *and then the miner produced two supra-ε rounds
(+.0055, +.0097) and closed a further .011 of the residual.* Recorded plainly because it
is the exact failure mode AMENDMENT 2 warned about, now observed rather than
hypothesised: **the frozen stopping rule is not monotone in this cell, and applying it
at round 2 would have declared a taste bound of +.0163 that four more rounds pushed down
to +.0020.** The rule is a stopping *heuristic*, not a saturation *proof*, and any
plateau quoted off it needs the cap as a cross-check.

**The residual has effectively closed.** The best bank state is round 4:
**Δ_beyond = +.0020 on HONEST** (n=1,571), i.e. the 168-feature articulated scorecard
matches the dense model to within .002 of AUC on the same rows. Round 5's criteria were
net noise (gain −.0121, CI [−.0241, +.0004], P(>0) = .03 — the only round in this whole
programme whose added criteria significantly *hurt*), which pushes the r5 Δ back up to
+.0142. So the honest plateau statement for this cell is a two-sided one:
**Δ_beyond ∈ [+.002, +.014] depending on which of the last two bank states is taken, and
neither end is distinguishable from zero at n=1,571.** ICLR oral/spotlight curation is
the first cell in the programme where an actively mined scorecard reaches the dense model.

## 3.2 Cumulative discount (52 channels, 9 MIXED parents retired)

Spurious-alone is .5657 (HistGB) / .5846 (linear) — **below the .65 matched-sampling
trigger**, so per the freeze the decile estimator is primary here and matched sampling is
reported as a secondary.

| readout, HONEST n=1,571 | ALL B (52 ch.) | STRICT, mixed dropped (16 ch.) |
|---|---:|---:|
| spurious-alone AUC (HistGB / linear) | .5657 / .5846 | .5646 / .5630 |
| pooled Δ | +.0142 | +.0142 |
| **decile-stratified Δ_adj** (primary) | **+.0215** | **+.0202** |
| matched-sampling Δ_adj (secondary, 392 pairs) | +.0255 | −.0140 |
| stacked: dense increment over B + bank | +.0192 [+.0001, +.0391] | +.0262 [+.0052, +.0475] |

The band is tight on the decile readout (.0013 wide) and wide on matched sampling
(.0395) — at this effect size matched sampling on 392 pairs is simply too noisy to
carry a claim, which is why the freeze gates it on spurious-alone > .65. Reading the
primary: discounting the 52 named channels leaves Δ_adj ≈ +.02, and the
stratification-free stacked increment agrees (+.019 to +.026) with a CI that barely
excludes zero. **Peer curation's residual after five mined rounds is at the noise floor
of this instrument: a point estimate of one to two AUC points that no discount removes
and no further mining reaches.**

## 3.3 Instrument health, fleet, missing mass

| round | proposals | A/B | mixed B | misroute | probes | disputes | anchors scram | pos-neg | collapsed | NA | A: S_obs / M̂ | B: S_obs / M̂ |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| r3 | 150 | 15/10 | 5 | .00 | 4/4 | 0 | .998 PASS | .620 | 0 | .004 | 62 / .544 | 43 / .500 |
| r4 | 150 | 15/10 | 7 | .00 | 4/4 | 0 | .943 PASS | .661 | 0 | .027 | 65 / .578 | 51 / .717 |
| r5 | 150 | 14/11 | 10 | .04 | 4/4 | 1 → B (mixed) | .991 PASS | .618 | 0 | .001 | 78 / .778 | 48 / .667 |

The one r5 dispute is instructive and is the mirror of JOB 2's: the arbiter routed to B
a criterion that "scores contribution genre — analysis paper versus new named system",
i.e. the miner, out of quality ideas, had started proposing paper *type*. Missing mass
does **not** fall into this cell's plateau (A .544 → .778): the fleet keeps naming new
quality species right up to the cap while the measured gain goes to zero and then
negative. That combination — high species turnover, no AUC — is the cleanest evidence in
the batch that what is left on this cell is not unnamed criteria but **noise in the
outcome**: ICLR oral/spotlight selection at T = .594 is close to its own noise ceiling,
so there is very little left for either instrument to find.

## 3.4 Final Track-B map (five rounds)

| round | alone AUC | mixed | channel | conjectured upstream parent |
|---|---:|:--:|---|---|
| r2 | .561 | | Memorable title framing | surface-only |
| r3 | .547 | | Density of currently fashionable technical terms | surface carrier of "Trend-term concentration" |
| r2 | .544 | YES | Trend-term concentration | submission timing and community fashion |
| r2/r4 | .543 / .540 | YES | Abstract length / verbosity | surface-only |
| r4 | .541 | | Proprietary system and dataset branding | surface-only |
| r3 | .460 | | Abstract Length Brevity (reverse-coded) | surface-only |
| r5 | .539 | YES | Sheer text volume and density of qualifying clauses | surface carrier of the length parent |
| r3 | .537 | YES | Signalled scale of compute, data, evaluation breadth | the producing group's compute budget |
| r3 | .534 | YES | Trendy Topic Jargon Density | submission timing / audience popularity |
| r5 | .533 | YES | Contemporary LLM/agent-era vocabulary density | submission timing / which topical wave |

The map is flat — nothing clears .561 across 52 channels — and it is dominated by two
families that replicate across every round: **trend/timing vocabulary** (.533–.547, four
independent namings) and **length/verbosity** (.539–.543, three namings, one
reverse-coded at .460 which is the same channel with the sign flipped). This is the
mechanism behind Layer 2's finding that publication year was the one nuisance dimension
peer curation failed (.561 → .536), now named and replicated at campaign length. The
compute/resource-scale channel (.537) is the cell's only upstream-resource fingerprint
and it is weak — unlike peer revealed, where the same family is the second-strongest
thing in the map. **Same corpus, same abstracts, different y: the outcome decides which
nuisance channels exist.** Curation (a committee reading the paper) has almost no
resource fingerprint; revealed (the field citing the paper years later) has a large one.

---

# 4. What the two cells together say, and the three-y status

## 4.1 The three peer-review preference types, side by side

Peer review is measured against three different outcomes on the same corpus of ICLR/
NeurIPS-class abstracts. All three now have a terminal Layer-3 result.

| preference type | y | T (same-rows) | final VA_nl | **plateau Δ_beyond** | rounds | how it stopped | discounted band |
|---|---|---:|---:|---:|---:|---|---|
| **verdict** (accept/reject) | reviewer decision | .777 (n=1,244) | — | pilot, EXPLORATORY | 4 | pilot, pre-freeze | — |
| **curation** (oral/spotlight) | committee selection | .5936 (n=1,571) | .5916 (r4) | **+.0020** (r4) … +.0142 (r5) | 5 | **hard cap** (rule non-monotone) | Δ_adj +.020 [+.019, +.026] |
| **revealed** (citation pct) | the field, years later | .8842 (n=478) | .7717 (r5) | **+.1125** | 5 | **2-consecutive-sub-ε at r5** | Δ_adj [+.076, +.254] |

**Peer review is the first field in the programme with all three preference types
terminal**, and the three land in three different places:

- **curation: articulable.** An actively mined 168-feature scorecard reaches the dense
  model to within .002 AUC. There is essentially no taste residual — but there is also
  very little total signal (T = .594), and the missing-mass evidence says the ceiling
  here is outcome noise, not tacit knowledge.
- **revealed: NOT articulable, and the residual is the largest confirmed one in the
  programme.** 68 audited criteria over five rounds closed 30% of the residual and then
  saturated at +.1125, which survives a 56-channel nuisance discount that *raises* it,
  and survives topic stratification under three independent topic instruments (JOB 1)
  which also raise it.
- **verdict: exploratory only** (the pilot, pre-freeze), so the confirmatory row of the
  three-y table is curation + revealed.

## 4.2 The finding that only a three-y design can produce

The bank, the corpus, the judge and the dense architecture are held fixed across
curation and revealed. Only **whose preference** is being predicted changes. And the
answer changes completely: **Δ_beyond +.002 vs +.113, a 56× difference, on the same
text.** Whatever the residual is on revealed, it is not a property of abstracts, not a
property of the criterion bank, and not a property of the instrument — it is a property
of *the preference being modelled*. The committee's decision is articulable; the field's
long-run citation behaviour is not.

The nuisance maps say the same thing in the other direction: the resource/institution
fingerprint family is near-absent on curation (.537, the weakest channel worth naming)
and dominant on revealed (.640–.682, four rounds), so the additional thing the field
responds to and the committee does not is partly upstream circumstance — but only
partly, since discounting that whole family *increases* revealed's residual.

## 4.3 Caveats that travel with every number here

1. **Pre-GEPA.** No criterion phrasing was GEPA-iterated in any round. All alone-AUCs
   and all closure gains are lower bounds on what the same *concepts* could achieve with
   tuned phrasing, so the plateau is a lower bound on articulability and hence Δ_beyond
   is an upper bound on taste.
2. **Plateau means "not discoverable by this miner."** P = 6, three families, five
   rounds, M̂ at the plateau .567 (A) / .433 (B) on revealed and .778 (A) / .667 (B) on
   curation. A seventh proposer would very likely name new species.
3. **Δ_adj is a conditional readout, not a causal estimate.** It answers "does the gap
   survive holding the named channels fixed", not "what would the gap be without the
   upstream factor".
4. **MONITOR is small on revealed** (244 rows); every claim above is governed by the
   HONEST population and MONITOR is shown only for agreement.
5. **Closure-split levels are protocol-specific** and not comparable to Layer-1
   Δ_beyond; only round-over-round changes and the same-rows honest levels are quoted.
6. **The stopping rule is not monotone** (§3.1). Any plateau quoted off it must also
   report the cap and the full curve.
7. **Matched sampling needs the .65 gate.** On curation (spurious-alone .566) it
   produced a .0395-wide ALL-vs-STRICT band against the decile readout's .0013 — the
   freeze's trigger is doing real work and should not be lowered.

## 4.4 Compute and judge ledger

| item | count |
|---|---:|
| Gemma-4-31B judge calls, peer curation rounds 1–5 | 1,011,375 |
| Gemma-4-31B judge calls, peer revealed rounds 1–5 | 317,125 |
| of which NEW in this campaign (rounds 3–5) | ~797,000 |
| sealed proposer slots (rounds 3–5, both cells) | 72 (6 proposers × 2 tracks × 3 rounds × 2 cells), 0 parse failures |
| decomposition components authored + scored | 18 per cell (9 MIXED parents retired per cell) |
| blind auditors | 6 (fresh per round per cell) |
| arbitrated disputes | 5, all → B, all flagged mixed |
| GPU | one free sk3 GPU at a time via `gpu_runner.sh` (poll → claim → re-verify → release); GPUs 3 and 7; co-tenant devices never touched; ledger entries under `agent=claude-peer-completion` |
| CPU | laptop only for every fit, slice, readout and discount |

## 4.5 Artifacts

- **JOB 1**: `closure/peer_revealed/job1_topic_strat.py`, `job1_topic_strat.json`,
  `job1.log`, `abstract_emb_bge_large.npz`.
- **JOB 2**: `closure/peer_revealed/` — `peer_revealed_r{3,4,5}_{slice,proposals_fleet,
  species,audit_prompt,audit_key,audit_verdicts,arbiter,routing_final,scores.npz,
  score_report,results}.json`, `peer_revealed_r5_cumulative_discount.json`,
  `peer_revealed_retired_channels.json`, `peer_revealed_r{3,4,5}_{newparents,
  parents_used}.json`, logs `r{3,4,5}_*.log`, `readout_r{3,4,5}.log`, `cumdisc_r5.log`.
- **JOB 3**: `closure/peer_curation_ext/` — same file set for `peer_curation_r{3,4,5}`
  plus `peer_curation_r5_cumulative_discount.json` and
  `peer_curation_retired_channels.json`.
- New code: `mixed_parents.py`, `discount_cumulative.py`, `job1_topic_strat.py`,
  `run_round_local.sh`, `score_on_sk3.sh`, `gpu_runner.sh` (adapted), plus the
  `audit.py` corpus-matched probe pool; everything else copied unchanged from
  `maps_batch1/` and `maps_hw_si/`.
- Sealed prompts and raw proposer/auditor/arbiter/decomposer outputs:
  session scratchpad `peer_revealed/peer_revealed_r{3,4,5}/` and
  `peer_curation_ext/peer_curation_r{3,4,5}/`.
- Mirrors on sk3 under
  `/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure/peer_{revealed,curation_ext}/`.
