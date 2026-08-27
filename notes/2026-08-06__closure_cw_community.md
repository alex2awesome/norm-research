# Layer-3 articulation closure — CW community cell (writingprompts upvotes), CONFIRMATORY

Date: 2026-08-06/07. Protocol: `notes/2026-08-05__layer3-closure-prereg.md`
**FREEZE DECLARATION** + **FREEZE ADDENDUM** (B-side mass, stacked increment) +
**ADDENDUM 2** (Track-B upstream mode) + **ADDENDUM 3** (MIXED decomposition) +
**ADDENDUM 4** (position-in-container prior) — all binding. Rounds 1–5 ran under the
declaration + addenda 1–2; rounds 6–8 are a user-approved extension (2026-08-07) that
additionally carries addenda 3–4 and the re-specified two-sided sign band.
Pilot this replicates: `notes/2026-08-05__layer3_round4_peer_verdict.md` (peer verdict,
exploratory). Fleet machinery: `notes/2026-08-06__missing-mass-robustification.md`.
Layer-2 profile for this cell: `notes/2026-08-06__layer2_robustness.md`.

Artifacts: `methods/taste_decomposition/closure/cw_community/` (all provenance-tagged).

Terminology, spelled out on first mention per the standing rule:
**V** = the 15 deterministic surface features (`datasets/creative-writing/va_bank_v2/
v_features.py`); **A** = the 45-criterion articulated bank (`rubrics_initial.jsonl`,
Gemma-4-31B-scored); **VA_nl** = HistGradientBoosting aggregation of the V+A matrix,
seed-mean over {0,1,2}; **T** = the dense readout (Llama-3.1-8B LoRA on raw text,
`wp_clean_rm_out/best_model`); **Δ_beyond** = T − VA_nl; **ε** = .005 saturation
threshold; **k_A / k_B** = 15 / 10 criteria scored per round; **AUC** = area under the
ROC curve; **FIT+MINE / MONITOR / TEST** = the closure splits; **M** = the mining slice;
**Good-Turing missing mass** = probability the next independent proposal names an unseen
species; **GEPA** = the prompt-iteration pass required before a confirmatory number is
quoted; **P** = proposers in a sealed fleet round.

---

## STAGE 0 — enlarging the honest population (round 0)

### Why it was needed

The cell's matched Δ_beyond of **+.176** — the largest residual in the program — rested
on **408 rows**: the only rows of the frozen 2,000-row A/V population that were also
held out by the dense model (the other 1,592, i.e. 79.6%, are in its train split, so
their T is in-sample). 408 rows cannot support mine-plus-monitor: an 80/20 split leaves
~80 MONITOR rows, and ε = .005 is far below that readout's noise.

### What was done

Population **extension**, not a new instrument. The 45 criteria, the system prompt, the
deterministic truncation rule, the token vocabulary and the sampling parameters are
byte-identical to `datasets/va_gemma_banks/score_va_gemma_banks.py::build_creative()`.
Only the row set grew, along the frozen sample's own stable-hash prompt ordering:

- the CW bank sample is `sha256("cw-va-v2-sample|" + prompt_id)`-ordered prompt groups,
  taken until ≥2,000 rows. **Reproduction gate: all 2,000 frozen bank ids reproduce
  exactly from the local source** (`stage0_build_population.py`, asserted in code).
- the dense split is **prompt_id-GROUPED** (`split_metadata.json`, seed 42; verified
  disjoint: 56,361 + 7,046 + 7,046 = 70,453 prompt ids), so a whole extension group is
  either fully held out or not at all.
- the prefix was continued past the 2,000-row cut, keeping only groups with
  `dense_split ∈ {eval, test}`, until 6,600 new rows had been added.

| | rows | prompt groups |
|---|---:|---:|
| frozen Layer-1 population | 2,000 | 1,500 |
| …of which dense-held-out (the old honest set) | 408 | 298 |
| newly scored extension | 6,600 | 4,838 |
| **honest population (round 0)** | **7,008** | **5,136** |

Every row is dense-held-out, so the freeze's "MONITOR ⊂ dense-held-out" requirement is
satisfied by construction. Splits, stable sha256 on `prompt_id`, never a seeded
shuffle: **FIT+MINE 4,794 / MONITOR 1,114 / TEST 1,100** (<.70 / <.85 / rest).
Positive rate .5017.

Judge cost: 6,600 × 45 = **297,000** Gemma-4-31B calls plus 6,750 anchor calls,
offline-batch vLLM on one GPU, chunk-checkpointed.

### Instrument gates

| gate | result |
|---|---|
| bank ids reproduce from local source | 2,000 / 2,000 ✓ |
| criterion names match the frozen bank | asserted in code ✓ |
| NA rate | 1.84% |
| **collapse gate (per criterion)** | **0 / 45 collapsed** |
| anchor battery, K = 50/class | pos .7006 > neg .6598 > scrambled .0033, ordering holds |
| coherent vs scrambled AUC | **1.000** |
| pos vs neg AUC | **.562** |
| dense reproduction on the old 408 rows | .796693 vs .796693 on file — identical ✓ |

The **K ≥ 50 anchor rule earned its place on this cell.** The published K = 12 battery
for `creative_writing` has pos .686 **below** neg .744 — at twelve draws the known-label
contrast is not merely weak, it is inverted. At K = 50 it resolves the right way round
but stays weak (AUC .562). The scrambled gate is decisive (1.000), so the judge is
certainly reading the text; what it is not doing is separating high-scoring from
low-scoring community stories on craft criteria alone — which is the cell's whole point
and the reason its residual is large.

### Round-0 readout (the campaign's refreshed baseline)

n = 7,008 honest rows; AUCs are grouped-OOF inside FIT+MINE and refit-and-predict on
MONITOR/TEST, exactly the frozen `family1` Layer-1 spec (reproduction gate on the
2,000-row population: VA_lin .6301 and VA_nl seed-0 .6244 — identical to the Layer-1
ledger to four decimals).

| readout | FIT+MINE | MONITOR | TEST | population |
|---|---:|---:|---:|---:|
| n | 4,794 | 1,114 | 1,100 | 7,008 |
| T (dense) | .7890 | **.7950** | .8048 | **.7921** |
| VA_lin | .6450 | **.6686** | — | .6567 |
| VA_nl (seed-mean) | .6406 | **.6564** | — | .6509 |
| V-only, nonlinear | .5993 | .6275 | — | .6086 |
| **Δ_beyond = T − VA_nl** | — | **+.1386** | +.1215 | **+.1412** |
| Δ vs VA_lin | — | +.1264 | — | +.1354 |

Group-level paired bootstrap (2,000 draws, resampling prompt groups):
Δ_beyond population **+.1412 [.1276, .1548]**, P(>0) = 1.00;
MONITOR +.1324 [.0970, .1654], P(>0) = 1.00.
VA_nl seed spread on MONITOR is .0036 (.6559/.6548/.6584) — well under ε.

**Round-0 headline: the refreshed matched Δ_beyond is +.141, not +.176.** The 408-row
estimate was optimistic by ~.035; the honest population is 17× larger and its confidence
interval excludes the old point estimate. It is still comfortably the largest residual
in the program (peer verdict's comparable round-0 honest level was +.0925).

Two things to carry forward:

1. **VA_lin > VA_nl on this cell** (.6686 vs .6564 on MONITOR). The frozen readout is
   VA_nl, so Δ_beyond = +.1386 is the protocol number, but the tighter, fairer bound
   against the *best* articulated aggregation is **+.1264**. This is consistent with
   Layer-1, which recorded Δ_interact = −.0095 for this cell: on CW community the
   nonlinear stack buys nothing over the linear one. Both are reported at every round.
2. **The enlargement raised VA_nl** (population .6509 vs the Layer-1 2,000-row .6207)
   while T stayed put — 3.5× more training rows help the scorecard, not the dense
   model, which was trained on 76,867 rows either way. Part of the old +.176 was the
   scorecard being data-starved, not the residual being tacit.

## Rounds

### Round 1 — sealed fleet, selection, audit

**Mining slice.** M = FIT+MINE (4,794 rows; every population row is dense-held-out, so
the freeze's "mining slice = FIT+MINE ∩ dense-held-out" is the whole of FIT+MINE).
VA_nl OOF inside M = .6406 (seeds .6339/.6329/.6386). Slice = top |dense percentile −
VA_nl percentile|, 20 rows per direction = **40 rows**, median |rank gap| **.836**,
101 KB of story text after the frozen head+tail truncation. The prereg caps the read at
60; 40 was taken because CW stories are one to two orders of magnitude longer than the
peer-review abstracts the cap was written for, and 40 already makes a 114 KB sealed
prompt. MONITOR and TEST were not read.

**Fleet composition (degraded, recorded).**

| slot | family | model | Track A | Track B |
|---|---|---|---|---|
| claude_sonnet | claude | Claude Sonnet, sealed subagent | 15/15 | 10/10 |
| claude_opus | claude | Claude Opus, sealed subagent | 15/15 | 10/10 |
| codex_luna_a | openai | gpt-5.6-luna, `codex exec`, effort high | 15/15 | 10/10 |
| codex_luna_b | openai | gpt-5.6-luna, independent call | 15/15 | 10/10 |
| glm_a / glm_b | glm | glm-5.2 | **MISSING** | **MISSING** |

**P = 4, two families** — the freeze's degraded floor, recorded as required. Both GLM
keys returned `429 / 1308 "Usage limit reached for 5 hour"` on the Lite subscription
with a stated reset several hours out, so the third family could not be seated for this
round. Note this is a **different** GLM failure from the pilot's: the pilot hit `1302`
(request-rate, curable with backoff); this is `1308` (a five-hour usage window), which
backoff cannot cure. Retried each round.

Seal contract as in the robustified pilot: each proposer sees the slice rows and the
two percentile ranks and nothing else — no bank, no other proposer, no label — with a
per-proposer stable-hash row ordering (12 distinct order hashes recorded in the
manifests). Every Claude and luna call returned exactly k parseable, distinctly-named
criteria on the first attempt; no escalation to gpt-5.6-sol was needed.

**FREEZE ADDENDUM 2 compliance (Track B).** All four proposers met the Mode-2 quota:
each returned 5 upstream-traced channels and 5 surface-only ones, 0 untagged, with
3–5 flagged MIXED. Conjectured parents that came back include an author's established
following and cross-posting network, series momentum from earlier instalments, posting
time and thread position, whether the writer had editing help, and moderator/curation
dynamics — each proposed together with the textual fingerprint it would leave.

**Species and selection.** Blind full-recall partition by a sealed Opus judge, one pass
per track over the provenance-stripped pool (never an embedding threshold, per the
freeze): Track A **60 proposals → 35 species**, Track B **40 proposals → 18 species**.
The round's scored set is the top-k species by (number of distinct proposers naming it,
then families, then stable hash), with the representative phrasing drawn round-robin
from the least-represented proposer: 15 A (support 4/4/4/3/3 proposers at the top,
2 at the tail) + 8 B + 2 coordinator-planted probes = **25 scored criteria**.

**Blind routing audit.** Fresh Sonnet-class auditor, provenance stripped, hash-ordered:
15 quality-relevant / 10 incidental. **Misrouting rate 0.0%**, no disputes, so the
arbiter was not invoked. **Probe gate PASS**: both planted probes (a first-person
pronoun count and a body-part/sensory word count — shallow lexical counterparts of
"Sustained, distinctive narrating voice" and "Embodied Scene Presence") were routed to
the nuisance side, and both of their counterparts were routed to the bank.

**Fleet missing mass, both tracks (FREEZE ADDENDUM), round 1.** Species from the blind
full-recall partition above, so no embedding threshold enters the estimate.

| track | P | families | N proposals | S_obs | f1 | f2 | Good-Turing M̂ | cross-proposer recapture | species named by ≥2 families |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A (quality) | 4 | 2 | 60 | 35 | 18 | 12 | **.300** | **.49** | 11 |
| B (spurious) | 4 | 2 | 40 | 18 | 8 | 2 | **.200** | **.56** | 9 |

Marginal new species per added proposer (mean over random proposer orderings) —
A: 15.0 → 9.0 → 6.5 → **4.5**; B: 9.8 → 4.1 → 2.3 → **1.9**.

**This fleet is far more concordant than the peer-verdict fleet.** The robustified pilot
measured cross-proposer recapture .20–.32 and missing mass .42–.55 on ML abstracts; here
recapture is **.49 / .56** and missing mass **.30 / .20**. Independent proposers reading
creative-writing disagreement rows converge on much the same craft vocabulary, and the
Track-B space is smaller still — the nameable spurious-channel repertoire for this
community is close to exhausted at four proposers. That is a statement about the
proposal distribution, not yet about recoverable AUC; the odds-form remaining-AUC bound
needs the gain series and is reported once rounds have run.

*Caveat on the jackknife at P = 4.* Leave-one-proposer-out drops the fleet to P = 3,
which mechanically converts doubletons into singletons and pushes M̂ **up** (A: [.378,
.511], B: [.133, .300]). At this fleet size the jackknife bounds the *width*, not the
level; the point estimates M̂ = .300 / .200 are the primary figures, and a third family
would tighten both. This is why the freeze wants P ≥ 5 across ≥ 3 families.

### Round 1 — scoring and readout

Judge cost 25 × 7,008 = **175,200** calls plus 6,750 anchors. Gates: NA rate .027%,
**0 / 25 collapsed**, anchors K = 50/class pos .5012 > neg .4536 > scrambled .0668
(ordering holds), coherent-vs-scrambled AUC **.9992**, pos-vs-neg AUC **.618** — the
mined criteria separate the known-label anchors *better* than the incumbent 45-criterion
bank does (.562). **0 sign-contradicting criteria** (no A criterion has FIT+MINE
alone-AUC below .5), so the re-audit trigger did not fire.

| | round 0 | round 1 |
|---|---:|---:|
| bank columns | 60 | 75 |
| VA_nl MONITOR (seed-mean) | .6564 | **.6493** |
| per-seed MONITOR | .6559/.6548/.6584 | .6499/.6450/.6530 |
| VA_lin MONITOR | .6686 | **.6731** |
| T MONITOR | .7950 | .7950 |
| **Δ_r on MONITOR (VA_nl)** | — | **−.0071** |
| Δ_r on MONITOR (VA_lin) | — | **+.0045** |
| Δ_beyond MONITOR (vs VA_nl) | +.1386 | +.1457 |
| Δ_beyond MONITOR (vs VA_lin) | +.1264 | +.1219 |
| Δ_beyond population (vs VA_nl) | +.1412 | +.1416 |

Group-level paired bootstrap of the MONITOR gain: **−.0055, 95% CI [−.0155, +.0050],
P(gain > 0) = .15**. The strongest single mined criterion reaches alone-AUC .575 on
FIT+MINE ("Earned idea payoff from the premise" / "Earned Payoff", proposed
independently by three of the four proposers).

**Round 1 is a sub-ε round (gain −.0071 < ε = .005): saturation counter 1 of 2.**

**The two aggregators disagree about the sign, and this is the round's most useful
finding.** Fifteen fleet-selected craft criteria moved the *linear* stack **up** +.0045
and the *nonlinear* stack **down** −.0071. The frozen decision readout is VA_nl, so the
protocol records a null; but the honest reading is that the mined criteria carry a
little real signal which the gradient-boosted aggregator cannot convert at this sample
size — going from 60 to 75 columns on 4,794 FIT+MINE rows costs it more in variance than
the new columns pay in signal. This cell was already the program's one case where
Δ_interact < 0 (Layer-1: −.0095); round 1 sharpens that into a live measurement issue.
Both readouts are carried at every round, and the plateau is quoted against both.

**Swap pair.** ΔC₊ = **+.00007**, ΔC₋ = **−.0024** on the honest population. The bank
neither took on the dense model's insights nor its errors; there is no swap here, unlike
the pilot's rounds 1–3, because there is essentially no movement to decompose.

**Track-B discount, round 1** (10 declared channels; spurious-alone .5896 linear /
.5995 HistGB — below the .65 matched-sampling trigger, so decile stratification is the
estimator, as prespecified).

| readout | T | VA_nl | Δ |
|---|---:|---:|---:|
| undiscounted | .7921 | .6504 | **+.1416** |
| decile-stratified on the joint B-model, **all 10 channels** | .7816 | .6278 | **+.1538** |
| decile-stratified, **strict set (4 MIXED channels dropped)** | .7883 | .6427 | **+.1456** |

The FREEZE ADDENDUM 2 sensitivity band is therefore **+.1416 to +.1538**, and the
direction is the pilot's: conditioning on the declared nuisances leaves the residual
*unmoved or larger*, never smaller. Dropping the MIXED channels moves Δ_adj by −.008,
so the band is narrow and the MIXED question is not load-bearing on this cell — but it
is now measured rather than assumed.

**Stacked increment (stratification-free control).** Joint B-model alone .5995; stacked
with the dense score .7919 — the dense model adds **+.1924** over *all ten named
spurious channels combined*. Over the bank: .6504 → .7926, **+.1421**. The control does
not degenerate and agrees with the stratified table.

### Round 2 — sealed fleet, selection, audit

Slice regenerated against the round-1 bank (75 columns): VA_nl OOF inside FIT+MINE
.6396, 40 rows, median |rank gap| .827. Fleet again **P = 4, two families** — both GLM
keys were still inside the same five-hour `1308` usage window. Every Claude and luna
call returned k/k parseable criteria first time. Track-B Mode-2 compliance rose:
upstream-traced channels per proposer 5 / 8 / 5 / 6, MIXED flags 2 / 6 / 2 / 4.

Species (blind full-recall, fresh judge): Track A 60 → **35 species**, Track B 40 →
**18 species** — identical counts to round 1 on a regenerated slice.

Blind audit: 15 quality-relevant / 10 incidental, **misrouting 0.0%**, no disputes,
**probe gate PASS** (this round's probes were a quotation-mark count planted against
"Character-specific voice" and a final-paragraph word count planted against "Resonant
closing image").

**Fleet missing mass, round 2:** Track A M̂ = **.333** (f1 = 20, f2 = 8, recapture .43);
Track B M̂ = **.150** (f1 = 6, f2 = 4, recapture **.67**).

### Round 2 — readout, and the sign-contradiction trigger firing

Gates: NA .028%, **1 / 25 collapsed** (a contest/heat-marker channel, too rare to score),
anchors pos .414 > neg .396 > scrambled .014, coherent-vs-scrambled **1.000**,
pos-vs-neg .559.

**The freeze's sign-contradiction trigger fired for the first time.** Two criteria the
blind auditor had routed to the bank came back with FIT+MINE alone-AUC below .5. Per the
freeze these go to a fresh sealed re-auditor, which was given the criterion text and the
measured direction and asked whether the criterion is confirmed, mis-routed, or
ill-posed. Verdicts: **"Form mirrors content" → re-route to the nuisance side**;
"Moral discomfort left standing, not resolved for the reader" → **quality-relevant
confirmed** (an editor's virtue that this community happens not to reward). The re-route
was applied and round 2 recomputed; both the pre- and post-trigger results are kept
(`round2_*.PRE_REAUDIT.json`). **This is the trigger catching a real miss** — the
blind auditor's 0% misrouting rate on round 2 was, on the sign evidence, 4%.

| | round 1 | round 2 (post-trigger) |
|---|---:|---:|
| criteria kept | 15 A / 10 B | **14 A / 10 B** (1 collapsed, 1 re-routed) |
| bank columns | 75 | 89 |
| VA_nl MONITOR | .6493 | **.6578** |
| VA_lin MONITOR | .6731 | .6715 |
| **Δ_r on MONITOR (VA_nl)** | −.0071 | **+.0085** |
| bootstrap gain [95% CI], P(>0) | −.0055 [−.0155, +.0050], .15 | +.0048 [−.0072, +.0159], .77 |
| Δ_beyond MONITOR | +.1457 | **+.1372** |
| Δ_beyond population | +.1416 | +.1377 |
| swap ΔC₊ / ΔC₋ | +.0001 / −.0024 | +.0032 / +.0067 |

**Round 2 is NOT sub-ε (+.0085 > .005), so the saturation counter resets to zero.**

Read the two rounds together and the honest picture is churn, not progress: VA_nl on
MONITOR went .6564 → .6493 → .6578, a **net +.0014 over two rounds and 29 mined
criteria**, with both round-level bootstrap intervals straddling zero. The linear stack
tells the same story from the other side (.6686 → .6731 → .6715, net +.0029). The
campaign has not yet produced a movement larger than its own readout noise.

The swap pair is informative here: in round 2 **both** ΔC₊ (+.0032) and ΔC₋ (+.0067)
rose. The bank became more concordant with the truth on pairs the dense model gets right
*and* on pairs it gets wrong — i.e. it gained a little genuine ordering, rather than
inheriting the dense model's errors. That is the opposite of the pilot's swap signature.

**Track-B discount, round 2** (19 declared channels; spurious-alone .630 linear / .620
HistGB — still under the .65 matched-sampling trigger, but close):

| readout | Δ |
|---|---:|
| undiscounted | **+.1377** |
| decile-stratified, all 19 channels | **+.1683** |
| decile-stratified, strict set (7 MIXED dropped) | **+.1561** |

Stacked increment: joint B-model .6299 → stacked with dense .7927 = **+.1628** over all
nineteen named channels; over the bank .6549 → .7930 = **+.1381**.

**A planted probe out-predicted almost every mined craft criterion.** This round's
quotation-mark counter — written deliberately as a shallow typographic stand-in for
"Character-specific voice" — scored alone-AUC **.570** on the honest population, higher
than 13 of the round's 15 quality criteria and second only to "Premise-to-Plot
Transformation" (.577). The blind auditor still routed it to the nuisance side, which is
exactly what the probe is for: on this cell, *predictiveness and quality-relevance come
apart far enough that a miner optimising for AUC alone would happily bank punctuation
density*.

### Round 3 — fleet and audit

Slice regenerated against the round-2 bank: VA_nl OOF inside FIT+MINE .6454, 40 rows,
median |rank gap| .837. (Provenance note: the round-3 slice was drawn from the
pre-re-audit round-2 state, i.e. with "Form mirrors content" still in the bank. The
slice is a mining device, not a readout, so this changes only which rows the proposers
read; every reported number for round 3 is computed against the corrected state.)

Fleet again **P = 4, two families** (GLM still inside its `1308` window). Species:
Track A 60 → **38**, Track B 40 → **25**. Blind audit: 14 quality-relevant / 11
incidental, **misrouting 4.0%**, **probe gate PASS** (this round's probes: a
concrete-noun counter planted against "Concrete sensory specificity" and a question-mark
counter planted against "Interpretive Work Left to the Reader").

**Fleet missing mass, round 3:** Track A M̂ = **.383** (recapture .39), Track B M̂ =
**.325** (recapture .48). Both tracks' species pools *widen* in round 3 after
contracting in round 2 — the proposers are not running out of things to name.

### Round 3 — readout, and the trigger firing at scale

Gates: NA .12%, 0 collapsed, anchors pos .410 > neg .383 > scrambled .055,
coherent-vs-scrambled .9991, pos-vs-neg **.598**.

**The sign-contradiction trigger fired on five of fifteen A criteria.** The sealed
re-auditor confirmed two as genuine-but-unrewarded craft virtues ("Premise Explored as
Sincere Inquiry", "Psychological Contradiction"), re-routed two to the nuisance side
("Form enacts the condition it describes", "Interpretive Work Left to the Reader") and
declared one **ill-posed** and dropped it ("Non-obvious interpretive angle"). Applied and
recomputed; pre-trigger artifacts kept alongside.

The trigger is not cosmetic — it moved the headline:

| round 3 | pre-trigger | **post-trigger (governing)** |
|---|---:|---:|
| criteria kept | 14 A / 11 B | **11 A / 13 B** (2 re-routed, 1 dropped) |
| VA_nl MONITOR | .6738 | **.6672** |
| Δ_r on MONITOR | +.0160 | **+.0094** |
| bootstrap gain [95% CI], P(>0) | +.0162 [+.0024, +.0292], .99 | +.0104 [−.0003, +.0207], **.97** |
| Δ_beyond MONITOR | +.1212 | **+.1279** |

**Round 3 is not sub-ε (+.0094), so the counter stays at zero.** It is also the
campaign's first gain that is nearly clean of zero (P(gain > 0) = .97) and the first
round where the swap pair points the right way for articulation: ΔC₊ **+.0066** with
ΔC₋ **−.0013** — the bank gained concordance on pairs the dense model gets right while
very slightly *losing* it on pairs the dense model gets wrong. That is articulation, not
error-inheritance.

**Track-B discount, round 3 — the estimator switched as prespecified.** With 33 declared
channels, spurious-alone reached **.656 linear / .651 HistGB**, crossing the freeze's
.65 threshold, so **matched sampling replaced decile stratification** automatically.

| readout | Δ |
|---|---:|
| undiscounted | **+.1327** |
| matched sampling on the joint B-model, all 33 channels | **+.1775** |
| matched sampling, strict set (10 MIXED dropped) | **+.1572** |

Stacked increment: joint B-model .6559 → with dense .7933 = **+.1374**; bank .6593 →
.7933 = **+.1339**.

### Round 4

Fleet P = 4 / 2 families again — **and the GLM leg is now out for the week**: the keys
moved from `1308` (five-hour window) to **`1310` "Weekly/Monthly Limit Exhausted, resets
2026-08-13"**. No amount of backoff reaches a third family before this campaign ends;
the runner was stopped rather than left burning retries. Species: A 60 → **39**,
B 40 → **25**. Blind audit 15/10, misrouting **0.0%**, probe gate **PASS** (probes: a
proper-noun counter against "Implicit Worldbuilding", a verbatim-repeat counter against
"Controlled Repetition and Rhythm"). Gates: 0 collapsed.

Sign-contradiction trigger fired on 2 of 15; the re-auditor confirmed one and re-routed
one ("Form matches and amplifies content" — the third variant of the "form mirrors
content" idea to be re-routed in three rounds, a consistent verdict across independent
re-audits).

| round 4 | pre-trigger | **post-trigger (governing)** |
|---|---:|---:|
| criteria kept | 15 A / 10 B | **14 A / 11 B** |
| VA_nl MONITOR | .6740 | **.6737** |
| Δ_r on MONITOR | +.0068 | **+.0066** |
| bootstrap gain [95% CI], P(>0) | +.0078 [−.0041, +.0197], .90 | — |
| Δ_beyond MONITOR | +.1211 | **+.1213** |
| swap ΔC₊ / ΔC₋ | +.0076 / −.0038 | +.0071 / −.0011 |

**Round 4 is not sub-ε (+.0066).** Track-B: 43 → 45 channels cumulative, spurious-alone
**.664 linear / .660 HistGB**, matched sampling; undiscounted Δ +.1274, full-set adjusted
+.1672, strict-set +.1565. Stacked increment: dense adds **+.1297** over all named
channels and **+.1288** over the bank.

**Fleet missing mass, round 4:** Track A M̂ = .383 (recapture .41), Track B M̂ = **.425**
(recapture .32) — the B-side pool, which had contracted to M̂ = .150 in round 2, is
wider than ever by round 4.

## Spurious map

The full per-channel table is generated by `build_report.py` and lives in
`round{r}_results.json → discount.per_channel_alone_auc`. Round-1 highlights, alone-AUC
on the honest population (7,008 rows):

| channel | alone-AUC | conjectured upstream parent | MIXED |
|---|---:|---|---|
| Copy-editing regularity | **.550** | editing help / professional writing experience | **yes** |
| Markdown vertical-whitespace density | **.537** | surface-only | no |
| External audience footprint | .527 | established following, cross-posting network | **yes** |
| *(planted probe)* First-person pronoun count | .524 | surface-only | no |
| *(planted probe)* Body-part and sensory word count | .511 | surface-only | no |
| Reader-facing boilerplate | .507 | surface-only | no |
| Serial-instalment furniture and pre-assumed world lore | .507 | series momentum | **yes** |
| Transcript or log format | .499 | surface-only | no |
| Contest-compliance furniture | .496 | judged/curated challenge participation | no |
| Novice / non-native authorial disclaimer | .492 | author's novice or non-native background | **yes** |

Three things are worth naming.

1. **The strongest declared nuisance on this cell is an upstream fingerprint, not a
   surface pattern.** "Copy-editing regularity" (.550) is the textual trace of *having
   had editing help or professional experience* — exactly the class of unseen factor
   FREEZE ADDENDUM 2 was written to hunt — and it is flagged MIXED, because the same
   parent plausibly also produces better writing. Upstream reasoning found the top
   channel; surface pattern-hunting found the second.
2. **The second-strongest channel independently replicates the Layer-2 result for this
   cell.** `notes/2026-08-06__layer2_robustness.md` records CW community as one of six
   cells failing the .02 nuisance-survival tolerance, on exactly one dimension —
   **format**, pooled .624 → stratified .597. The sealed fleet, with no sight of that
   analysis, nominated markdown/vertical-whitespace density as its top surface channel
   (.537). Two instruments, one channel.
3. **The planted probes behave like probes should.** Both are weakly predictive
   (.524, .511) — enough that a naive miner might have banked them — and both were
   routed to the nuisance side by the blind auditor while their quality-relevant
   counterparts were routed to the bank. That is the audit instrument passing on a
   deliberately hard case, twice.

### Round 5 — the cap

Fleet P = 4 / 2 families. Species: A 60 → **34**, B 40 → **21**. Blind audit 15/10,
misrouting **0.0%**, probe gate **PASS** (probes: a paragraph counter against
"Elaboration beyond the inciting gimmick", a trailing-ellipsis counter against
"Recontextualizing ending"). 1 collapsed. Sign-contradiction trigger fired on 4 of 15;
the re-auditor confirmed two, re-routed one and declared one ill-posed.

**The re-auditor volunteered a protocol correction and it should be carried into the
freeze.** All four round-5 alone-AUCs sat within ~.02 of chance (.479–.499). The trigger
as written fires on `alone-AUC < .5`, which cannot distinguish *contradicting* from
*null*; the re-auditor said so unprompted and judged on the instruction text rather than
the sign. **Recommendation: re-specify the trigger as a two-sided band around .5 scaled
to the readout's noise (e.g. alone-AUC < .5 − 2·SE), so it fires on evidence rather than
on the sign of noise.** Round 5's four firings were, on that reading, three nulls and one
genuinely mis-specified instruction.

| round 5 | pre-trigger | **post-trigger (governing)** |
|---|---:|---:|
| criteria kept | 15 A / 9 B | **13 A / 10 B** |
| VA_nl MONITOR | .6739 | **.6716** |
| Δ_r on MONITOR | +.0001 | **−.0021** |
| bootstrap gain [95% CI], P(>0) | +.0006 [−.0087, +.0102], .56 | −.0014 [−.0116, +.0087], .38 |

**Round 5 is sub-ε.** Because round 4 was not, the "two consecutive sub-ε rounds"
condition was never met, and the campaign **stops at the hard cap B = 5**.

## EXTENSION — rounds 6–8 (user-approved 2026-08-07)

The cap-5 run stopped without saturating and the curve was still drifting up, so the
user authorised three further rounds under the same frozen protocol, the same fleet
floor (GLM still quota-dead until 2026-08-13 → P = 4 / 2 families, recorded), the same
audit / probe / arbiter machinery and the same readouts. The original stopping rule
still governs: two consecutive sub-ε rounds end it early, otherwise the run goes to
round 8.

**One protocol change, adopted from this campaign's own round-5 finding.** The
sign-contradiction trigger is re-specified from `alone-AUC < .5` to a **two-sided,
noise-scaled band**: a criterion trips it only if its FIT+MINE alone-AUC falls more than
2 standard errors below chance, with the standard error taken from Hanley–McNeil at
AUC = .5 on the split's own class counts (n₊ = 2,392, n₋ = 2,402 → **SE = .00834**,
band **[.4833, .5167]**). Criteria landing inside the band are recorded separately as
`sign_null_band_A` — noted, not re-audited. Rounds 1–5 keep the original one-sided rule
they were run under; rounds 6–8 use the band.

### The challenged channel — "fragmented staccato lineation"

The user challenged whether this belongs on the nuisance side at all, since line-break
rhythm and pacing are plausibly authorial technique in fiction. Full history:

| | instance 1 | instance 2 |
|---|---|---|
| cid / round | R3B08, round 3 | R4B05, round 4 |
| name | Fragmented lineation | Fragmented staccato paragraphing / line-break repetition |
| proposer | codex_luna_b (OpenAI family) | claude_sonnet (Claude family) |
| species support | 2 proposers | 2 proposers |
| tagged upstream parent | surface-only | surface-only |
| tagged MIXED | false | false |
| blind audit | incidental, conf **.75** | incidental, conf **.80** |
| audit reason | "Counts presence of fragmented formatting/short lines without judging whether the choice is effective." | "Counts a formatting pattern (staccato line breaks) without judging whether it serves the story; stylistic fashion marker." |
| disputed / arbitrated | no | no |
| alone-AUC (FIT+MINE / population) | .5424 / .5495 | **.5754 / .5816** |

Instance 2's rubric, in full: *"Count the fraction of paragraphs that are a single short
clause or sentence (under roughly eight words) set off by its own line break, and note
literal repetition of a short phrase (e.g. counting patterns, repeated single words).
HIGH = the story is built mostly from such isolated one-line beats. LOW = paragraphs are
built from ordinary multi-sentence prose units."*
Instance 1's: *"Score high when the prose uses many short standalone lines, isolated
sentences, ellipses, or abrupt visual breaks; score low when it uses conventional
paragraphs and continuous sentences."*

**Adjudication (sealed Opus, given the full history and the explicit instruction not to
split the difference to be agreeable) — `lineation_challenge_adjudication.json`:**

- **routing_verdict: `confirm_B`**, no split proposed
- **upstream_parent_verdict: `surface-only`** (unchanged)
- **mixed_verdict: `false`** (unchanged — *not* re-flagged MIXED)
- deciding phrase: *"Count the fraction of paragraphs that are a single short clause or
  sentence (under roughly eight words) set off by its own line break"*

The adjudicator's reasoning, which is the useful part: **"the reviewer is right that
lineation can be technique and wrong that this criterion measures it."** As written,
both instances count typographic density and therefore pool three different populations
— deliberate fragmentation, fashion-imitation, and careless writing. Because the
deliberate subset is never separated out, **no single latent cause is being
fingerprinted, and that is exactly what `surface-only` means**; "deliberate stylistic
control" would be the correct parent only if the instruction distinguished controlled
from uncontrolled breaks. With no parent, there is nothing that co-causes better
writing, so MIXED stays false.

Three independent corroborations sit alongside it, none of which were available to the
adjudicator as votes:

1. **The craft twin already exists on the A side.** "Controlled Repetition and Rhythm"
   (round 4) asks the judge whether patterning is *controlled* and does rhetorical work,
   and was routed to the bank. The surface/craft cut this channel falls on is one the
   study already draws, in the direction the adjudicator describes.
2. **Its shallow probe was independently caught.** The planted "Repeated-phrase
   exact-match count", written against that twin, was routed to nuisance by a blind
   auditor who did not know it was a probe.
3. **Round 6 re-ran the experiment by accident and got the same answer.** This round's
   planted probe "Mean sentence-length variance" — a purely statistical measure of
   rhythm — went to the nuisance side while its craft twin **"Deliberate sentence
   rhythm"** (R6A07) went to the bank, in the same 25-item blind pool.

**Verdict: the channel stays on the nuisance side, surface-only, not MIXED.** The
substantive point the challenge raises is real and is recorded as a limitation of the
*rubric*, not of the routing: a channel that counted only *controlled* fragmentation
would be a different, plausibly MIXED channel, and no proposer has written one.

**FREEZE ADDENDUM 3 then landed and turned that gap into an experiment** (§ below).
The channel was decomposed rather than merely re-flagged, and the missing craft half was
written and scored. It exists, it is real, and it is worth **alone-AUC .567** — so the
challenge was right that there is craft in the vicinity, and the adjudicator was right
that the parent rubric was not measuring it. Both, not either.

## Round 6 — the stopping rule fires

Fleet P = 4 / 2 families. Species A 60 → **37**, B 40 → **18**. Blind audit 15/10,
misrouting **0.0%**, probe gate **PASS**. 0 collapsed. Anchors pos .464 > neg .433 >
scrambled .048, coherent-vs-scrambled **.9998**.

**The re-specified two-sided band paid for itself immediately.** Six criteria had
FIT+MINE alone-AUC below .5; the band ([.4833, .5167], SE = .00834 on n₊ = 2,379 /
n₋ = 2,415) sent **one** to re-audit and recorded **five** as nulls. Under the old
one-sided rule all six would have been re-audited — a **6:1 over-firing rate**, which is
the quantitative version of the round-5 recommendation. The one genuine firing (R6A12)
came back `quality_relevant_confirmed`, so round 6's numbers stand unchanged.

| | round 5 | round 6 |
|---|---:|---:|
| bank columns | 127 | 142 |
| VA_nl MONITOR | .6716 | **.6731** |
| VA_lin MONITOR | .6812 | .6849 |
| **Δ_r on MONITOR** | **−.0021** | **+.0016** |
| bootstrap gain, P(>0) | −.0014, .38 | +.0031, .74 |
| Δ_beyond MONITOR | +.1234 | +.1219 |

**Rounds 5 and 6 are both sub-ε. The stopping rule — two consecutive sub-ε rounds —
fires, and closure MINING ends at round 6**, two rounds inside the authorised extension.
Saturation is now *declared by the rule*, not forced by a cap, which is the outcome the
cap-5 report said was still missing.

## Round 7 — FREEZE ADDENDUM 3 decomposition pass

Not a mining round (mining had stopped); a directed pass that decomposes two MIXED
families into components, routes each independently, scores them, and retires the
parents. Addendum 3's k-budget rule is moot here because there was no fleet share left
to take.

**Families decomposed.** Of the 20 MIXED channels standing at round 6, two families were
selected — the one the user challenged, and the strongest coherent MIXED family:

| family | parent channels retired | authored by |
|---|---:|---|
| staccato lineation | 2 (R3B08, R4B05) | coordinator, to the coordinator's explicit split |
| editing help / professional production | 5 (R1B05, R2B06, R3B02, R4B03, R5B03) | sealed Opus adjudicator-author |

The editing-help adjudicator was asked first whether the MIXED flag was even justified
and was explicitly licensed to say no. Its verdict — `warrants_decomposition: true`,
`can_components_come_apart: true` — is worth quoting because it names the distinction
the whole addendum turns on:

> "Most of what the five instructions actually name is pure orthographic fingerprint —
> typos, homophones, paragraph spacing — which a spellchecker or one proofing pass
> supplies without improving anything about the story. But the shared parent (the text
> was revised, or written by someone practised at revising) also produces changes that
> are **constitutive of quality rather than symptomatic of help**: continuity that holds,
> planted details that pay off, prose with nothing left to cut."

**Blind audit of the four components: a clean 2/2 split, no disputes, confidence .90–.97.**

| cid | role | routed | conf | alone-AUC (pop) |
|---|---|---|---:|---:|
| R7D01 One-line beats that earn their isolation | candidate-real | **A** | .90 | **.5667** |
| R7D02 Blank-line and short-paragraph density | surface | **B** | .95 | .5765 |
| R7D03 Revision depth: continuity, economy, closed setups | candidate-real | **A** | .95 | **.5856** |
| R7D04 Mechanical slip count (spellchecker-catchable) | surface | **B** | .97 | .5431 |

**Result — the single most consequential number in the extension.**

| | round 6 | round 7 (decomposition) |
|---|---:|---:|
| bank columns | 142 | 144 (**+2**) |
| VA_nl MONITOR | .6731 | **.6786** |
| **Δ on MONITOR** | +.0016 (from **15** new A columns) | **+.0055** (from **2**) |
| Δ_beyond MONITOR | +.1219 | **+.1164** — campaign minimum |
| nuisance channels live | 64 | 58 (7 retired, recorded not deleted) |

**Two decomposed components bought 3.4× the closure that the last full 25-criterion
mining round bought, at 8% of the judge cost.** R7D03 (.5856) is the **strongest single
quality-relevant criterion in the entire campaign**, beating all 67 mined A criteria
(previous best .579) — and it was not mined. It was recovered from inside a channel the
campaign had spent five rounds calling nuisance.

Read together with the lineation result, the mechanism is legible: **a MIXED channel is
a mixture, and scoring the mixture destroys both halves.** The parent "Copy-editing
regularity" family sat at .550–.558 as an undifferentiated nuisance; split, its real half
is .586 (bank) and its surface half .543 (nuisance). The parent lineation channel sat at
.582 as nuisance; split, its craft half is .567 (bank) and its surface half .577
(nuisance). In both cases the parent's predictiveness was **not** what the routing
assumed it was.

## Round 8 — FREEZE ADDENDUM 4 directed position-in-container pass — **GATE FAILED**

Addendum 4 records that across CW's 125 criteria and peer's 8 B-files **no proposer has
ever named an ordinal/position channel**, while two of the program's strongest spurious
findings are exactly that family (patents claim-ordinal .754). CW is the cell where it
should bite hardest: these are Reddit prompt threads, where early replies accumulate far
more votes than late ones largely independent of quality.

Run as a directed Track-B pass on the round-6 slice (P = 4, two families), with the
position family made mandatory and the fingerprint step spelled out.

**The blind spot is an instruction gap, not a capability gap.** Directed, all four
proposers delivered **10/10 on-family channels on the first attempt, 40/40
upstream-tagged**, naming posting earliness, time-of-day, reply order, composition
window, and reaction-to-other-replies. **32 of 40 were flagged MIXED** — far above the
31% MIXED rate of the general rounds — because "how much time the writer had" plausibly
causes real quality. Species: 40 → 22. Blind audit misrouting **30%**, the campaign's
highest; the arbiter took all three disputes and returned **B / A / B**, keeping only
"No read-through pass" on the quality side ("measures the polish a reread produced, not
the timing trace").

**And then the instrument refused the measurement.**

| gate | round 8 | every other round |
|---|---:|---:|
| coherent-vs-scrambled AUC | **.3205** | .9991–1.000 |
| ordering holds on means | **FALSE** | true |
| pos-vs-neg AUC | .523 | .555–.654 |
| NA rate | **5.9%** | .03–.19% |
| mean modal share across the 10 columns | **.831** | — |

Scrambled word salad scored **higher** (.252) than coherent stories (.222 / .211). Per
the freeze the scrambled anchor is the **gating** control, so **these ten columns are not
admitted to the nuisance set and no discounted readout uses them.** Diagnostics only:
nine of the ten sit within ±.033 of chance (max |alone-AUC − .5| = .053, and that one is
the channel the arbiter moved to the *quality* side, i.e. not a position channel at all).

**Why it failed, and what it does and does not license.** These channels ask the judge to
find paratext and framing — prefatory notes, thread references, revision-trail
disclosures — which are absent from ~83% of stories, so the columns are near-degenerate
and the word-salad anchor, which contains no narrative to fail at either, scores in the
same undifferentiated middle. The failure is **measurement-side, not proposer-side**.

The sharp point for the program: **the patents .754 came from an observed ordinal, not
from a textual fingerprint.** Addendum 4 asks proposers to reach position *through* text;
on a corpus where position leaves essentially no textual trace, that route is blocked
however good the proposer is. This cell's source table carries only `text`, `judgement`
and `prompt_id` — **there is no timestamp or reply order on file**, so the position
hypothesis cannot be tested directly here at all. Recommended follow-up: re-fetch the
Reddit `created_utc` and within-thread rank for these 7,008 rows and test position as an
*observed covariate*, exactly as patents did, rather than as a fingerprint. Until then
the honest statement is **"the position family is unmeasured on this cell", not "the
position family is null on this cell."**

A second, cheaper instrument lesson: paratext channels need a **paratext-appropriate
negative control** (a coherent story with its notes stripped), not word salad. The
scrambled anchor is the right gate for craft criteria and the wrong one here — which is
why it fired, and why it was right to fire.

## Bottom line


### The closure curve (governing, post-trigger)

| round | criteria added (post-gate) | bank cols | VA_nl MONITOR | Δ_r on MONITOR | VA_lin MONITOR | Δ_beyond MONITOR | Δ_beyond population |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | — (45 A + 15 V) | 60 | .6564 | — | .6686 | **+.1386** | +.1412 |
| 1 | +15 A / +10 B | 75 | .6493 | **−.0071** | .6731 | +.1457 | +.1416 |
| 2 | +14 A / +10 B | 89 | .6578 | **+.0085** | .6695 | +.1372 | +.1377 |
| 3 | +11 A / +13 B | 100 | .6672 | **+.0094** | .6790 | +.1279 | +.1327 |
| 4 | +14 A / +11 B | 114 | .6737 | **+.0066** | .6838 | +.1213 | +.1274 |
| 5 | +13 A / +10 B | 127 | .6716 | **−.0021** | .6812 | +.1234 | +.1283 |
| 6 | +15 A / +10 B | 142 | .6731 | **+.0016** | .6849 | +.1219 | +.1290 |
| **STOP** | *two consecutive sub-ε rounds (r5, r6) — mining ends* | | | | | | |
| 7 | +2 A / +2 B *(Addendum-3 decomposition, not mining)* | 144 | **.6786** | **+.0055** | .6838 | **+.1164** | +.1269 |
| 8 | *Addendum-4 position pass — **anchor gate FAILED**, not admitted* | — | — | — | — | — | — |

Eight passes, **164 scored criteria** (rounds 1–6: 150 mined, 71 joining the bank;
round 7: 4 decomposition components; round 8: 10 gate-failed and excluded), **~1.13 M
Gemma-4-31B judge calls** plus 303,750 for the Stage-0 population extension.

Total movement across mining (rounds 0→6): VA_nl on MONITOR **+.0167**, Δ_beyond
**−.0167**. Adding the decomposition pass: VA_nl **+.0222**, Δ_beyond **−.0222**
(+.1386 → +.1164) — **the decomposition pass alone contributed a third of the campaign's
entire closure from 2 of the 71 bank columns.**

### The number, quoted on TEST — SECOND TOUCH, FLAGGED

**TEST has now been read twice.** It was quoted once at the cap-5 stopping round as the
protocol requires; the user then authorised an extension, so it is read a second time at
the true stopping round. This is a user-approved second exposure of the held-out split
and is flagged, not hidden: the split is no longer virgin, and a third reading would not
be defensible without a fresh split.

> **Δ_plateau = +.1030 AUC on TEST** at the terminal bank state (round 7:
> T = .8048, VA_nl = .70183, n = 1,100).

The full TEST trace, which is only interpretable because the extension made it available
and which is reported in full rather than at its best point:

| round | 0 | 1 | 2 | 3 | 4 | 5 *(1st quote)* | 6 | 7 *(2nd quote, terminal)* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TEST VA_nl | .6833 | .6893 | .6896 | .6972 | .7027 | .7018 | .6999 | **.7018** |
| TEST Δ | +.1215 | +.1155 | +.1153 | +.1076 | **+.1021** | +.1030 | +.1050 | **+.1030** |

TEST closed **−.0185** from round 0, and — importantly — **flattened after round 4**,
moving only ±.003 across rounds 4–7 while MONITOR was still drifting. The two splits
agree on the level and disagree slightly on when it stopped moving; the honest reading
is that closure on this cell had substantially finished by round 4, and rounds 5–6 were
the stopping rule catching up with a curve that had already flattened on held-out data.

### Saturation verdict

**Saturation WAS reached, by the rule, at round 6.** The gain series is
−.0071, +.0085, +.0094, +.0066, **−.0021, +.0016** — rounds 5 and 6 are both sub-ε and
consecutive, so the prereg's stopping rule fired two rounds inside the authorised
extension rather than being pre-empted by the cap. This is the outcome the cap-5 report
said was still missing, and it changes the claim available from this cell.

Fitted geometric decay λ̂ = **.879**, mean gain **+.0028/round**. Every round's bootstrap
interval straddles zero except round 3's (P(gain > 0) = .97). **TEST corroborates the
stop and dates it earlier than MONITOR does**: held-out Δ moved +.1215 → +.1021 across
rounds 0–4 and then sat within ±.003 for rounds 4–7. Two independent splits, one
stopping conclusion.

So the plateau language is now licensed, with its sensitivity stated:

**Six rounds of a sealed four-proposer, two-family mining fleet closed .017 of the .139
residual (~12%) and then stopped moving on both held-out splits.** The decomposition pass
then closed a further .0055 — more than the last two mining rounds combined — from two
criteria that mining never produced.

### Remaining mass

| track | round-6 fleet | S_obs | f1 | f2 | Good-Turing M̂ | cross-proposer recapture |
|---|---:|---:|---:|---:|---:|---:|
| A (quality) | 60 proposals | 37 | 22 | 8 | **.367** | .41 |
| B (spurious) | 40 proposals | 18 | 6 | 5 | **.150** | **.67** |

Across the six mining rounds, Track-A M̂ ran .300 / .333 / .383 / .383 / .283 / .367 and
Track-B M̂ ran .200 / .150 / .325 / .425 / .250 / .150 — **no monotone contraction in
either**. The odds-form remaining-AUC bound, R̂ = [M̂/(1−M̂)] · Δ̄ · λ̂ with this cell's
own Δ̄ = +.0028 and λ̂ = .879, gives **R̂ ≈ +.0010 to +.0015 per further mining round** on
Track A. Species-form (Chao1) is computed but **not quoted**, per the freeze.

**This is the campaign's central tension and the extension sharpened it.** The species
pool never contracted — a sixth round of proposers still produced 37 distinct Track-A
species with a third of the mass unseen — while the recoverable AUC per round fell to
zero. That is the pilot's *redundancy saturation, not value exhaustion*, replicated
prospectively on a second cell: the proposers have not run out of nameable criteria, they
have run out of criteria this bank does not already span.

**But rounds 7 and 8 add a third possibility the pilot could not see.** Round 7 shows
that value can be recovered *without* new species, by re-cutting channels the campaign
already had; round 8 shows that a whole nameable family can be invisible to the miner and
then unmeasurable by the judge. Neither is redundancy and neither is exhaustion. The
honest three-way statement is: **this miner has exhausted what it can name that the bank
does not span; it has NOT exhausted what better-cut versions of the bank's own channels
can measure; and at least one nameable family lies outside what this judge can score at
all.**

Two caveats travel with the R̂ bound. First, **λ̂ < 1 only because of rounds 5–6**;
through round 4 the series was *rising* (λ̂ = 1.10), so the geometric-decay form is not
identified on six points and the bound should be read as "small", not as a precise
figure. Second, at **P = 4 the leave-one-proposer-out jackknife is upward-biased** (it
drops to P = 3 and converts doubletons to singletons), so it bounds width, not level.

### Track-B bottom line and the spurious map

Fifty-four declared nuisance channels reach **.664 alone** — 78% of the way from chance
to the dense model — yet **every discount readout leaves Δ larger, not smaller**:

| round-5 readout | T | VA_nl | Δ |
|---|---:|---:|---:|
| undiscounted (population) | .7921 | .6638 | **+.1283** |
| matched sampling on the joint B-model, all 54 channels | — | — | **+.1664** |
| matched sampling, strict set (17 MIXED dropped) | — | — | **+.1435** |
| stacked increment: dense over all 54 named channels | .6643 → .7939 | | **+.1295** |
| stacked increment: dense over the 127-column bank | .6638 → .7932 | | **+.1294** |

The FREEZE ADDENDUM 2 sensitivity band at the stopping round is **+.128 to +.166**, with
the MIXED channels accounting for +.023 of the spread. The stratification-free stacked
control lands at **+.1295**, essentially on top of the undiscounted figure, which is the
strongest single reassurance in the report: the dense model's advantage over the bank is
not an artefact of the discount estimator.

**Per-channel highlights (alone-AUC on the honest population, 7,008 rows).** Of the 54
channels, **27 are upstream-traced** (FREEZE ADDENDUM 2 Mode 2) and **17 are flagged
MIXED**.

| channel | alone-AUC | conjectured upstream parent | MIXED |
|---|---:|---|---|
| Fragmented staccato paragraphing / line-break rhythm | **.582** | surface-only | no |
| *(planted probe)* Dialogue punctuation mark count | **.570** | surface-only | no |
| Self-promotional sign-off naming the author's own subreddit | .562 | established following / cross-posting network | **yes** |
| *(planted probe)* Question-mark count | .559 | surface-only | no |
| Editorial provenance | .558 | editing help / professional experience | **yes** |
| Stock genre furniture (aliens, HFY, isekai, dragons, capes) | .557 | community genre fashion | no |
| Copyedit-level consistency | .553 | editing help / professional experience | **yes** |
| Editorial polish signature | .552 | editing help / professional experience | **yes** |
| Copy-editing regularity | .550 | editing help / professional experience | **yes** |
| Markdown vertical-whitespace density | .537 | surface-only | no |

Three findings:

1. **The nuisance side beats the quality side on this cell.** The strongest declared
   nuisance channel (.582) out-predicts *every one of the 67 mined craft criteria*, whose
   best is .579. Two deliberately shallow **planted probes** — a quotation-mark counter
   and a question-mark counter — rank second and fourth. A miner optimising AUC alone
   would bank punctuation density over craft, and only the blind routing audit (which
   sent both probes to the nuisance side, in five rounds out of five) stops it.
2. **The editing-help channel is real, recurring, and MIXED.** Four separate channels
   across four rounds — "Copy-editing regularity", "Editorial polish signature",
   "Editorial provenance", "Copyedit-level consistency" — were independently nominated by
   different proposers as fingerprints of the same unseen parent (*the writer had editing
   help or is experienced*), all at .550–.558, all flagged MIXED. Addendum 2's
   upstream mode did exactly what it was added to do: it found a *causal* nuisance
   family rather than a surface pattern, and it flagged honestly that the same parent
   plausibly improves the writing too.
3. **Layer-2 replicates from the other direction.** `notes/2026-08-06__layer2_robustness.md`
   records CW community failing the .02 nuisance-survival tolerance on exactly one
   dimension — format (.624 → .597). The sealed fleet, blind to that analysis, nominated
   markdown/whitespace density and fragmented lineation as its top *surface* channels.

### Instrument bookkeeping across the campaign

| round | scored | collapsed | misrouting | probe gate | pos/neg anchor AUC | coherent-vs-scrambled | sign-trigger fired | re-routed / dropped |
|---|---:|---:|---:|---|---:|---:|---:|---|
| 0 (bank) | 45 | 0 | — | — | .562 | 1.000 | — | — |
| 1 | 25 | 0 | 0.0% | PASS | .618 | .9992 | 0 | — |
| 2 | 25 | 1 | 0.0% | PASS | .559 | 1.000 | 2 | 1 / 0 |
| 3 | 25 | 0 | 4.0% | PASS | .598 | .9991 | 5 | 2 / 1 |
| 4 | 25 | 0 | 0.0% | PASS | .654 | .987 | 2 | 1 / 0 |
| 5 | 25 | 1 | 0.0% | PASS | .631 | 1.000 | 4 | 1 / 1 |

The blind auditor's own misrouting rate was 0–4%; the **sign-contradiction trigger caught
five more mis-routings the auditor missed**, all of the same shape (formal/structural
properties phrased as quality). "Form mirrors content" and its two paraphrases were
re-routed to the nuisance side in three separate rounds by three independent re-audits.

### Claim discipline

Quotable, with the freeze's wording:

> Six rounds of a sealed four-proposer, two-family criterion-mining fleet reading the
> dense-disagreement slice closed **.017 of the .139 AUC residual** on the CW-community
> cell and then **saturated by the prereg's own rule** (two consecutive sub-ε rounds),
> corroborated by a held-out TEST split that had stopped moving two rounds earlier. A
> subsequent decomposition pass on two MIXED nuisance families closed a further **.0055**
> from just two criteria, ending at **Δ = +.103 on TEST**. What remains is
> **"not discoverable by this miner"** at the sensitivity this fleet demonstrated —
> P = 4 across 2 families, GEPA not applied — and the remaining-mass estimate is
> **M̂ ≈ .37 (Track A) with an odds-form bound of ≈ +.001–.0015 per further mining round**.

Quotable separately, and the more transferable finding:

> **Decomposing MIXED nuisance channels beat mining.** Two criteria recovered by splitting
> the "editing help" and "staccato lineation" families moved the articulated stack
> **+.0055**, against **+.0016** from the preceding 25-criterion mining round, and
> produced the campaign's strongest single quality criterion (alone-AUC **.586**, above
> all 67 mined ones). The predictiveness of a MIXED parent is not what its routing
> assumes: split, the editing family's real half (.586) went to the bank and its surface
> half (.543) stayed nuisance.

Never quotable from this run: "the residual is tacit"; "no further nameable criteria
exist"; the pre-re-audit round figures; the +.176 matched figure from the 408-row
population; **the round-8 position channels in any scored form** (their anchor gate
failed); "the position family is null on CW" (it is *unmeasured* — see round 8).

### Caveats that travel with every number here

1. **Saturated by the rule, on a curve that was never fast.** Two consecutive sub-ε
   rounds ended it, and TEST agrees, but no single round's gain is individually
   resolvable (MONITOR n = 1,114; ε = .005 sits inside the ±.010 bootstrap half-width).
   Only the trend and the TEST endpoint are.
2. **TEST was read TWICE** — once at the cap-5 stop as the protocol requires, once at the
   true stop under user-approved extension. Flagged, not hidden. A third reading needs a
   fresh split.
3. **PRE-GEPA.** `gepa_phrasing.py` implements the bounded, label-blind phrasing pass the
   freeze requires; it was not run. Carry the pre-GEPA flag on any quoted level.
4. **P = 4, two families, every round.** GLM was rate-limited out throughout — `1308`
   (five-hour window) for rounds 1–3, then **`1310` weekly/monthly exhaustion resetting
   2026-08-13** for rounds 4–8. The freeze's P ≥ 5 / ≥ 3-family target was unreachable in
   this window; a third family can only raise the closure rate, so this is a lower bound.
5. **VA_lin > VA_nl throughout.** The frozen readout is VA_nl; the linear aggregation is
   uniformly better (terminal MONITOR .6838 vs .6786). Against the best articulated
   aggregation the residual is **+.1112 on MONITOR**, not +.1164. Both reported.
6. **The round-3/4/5/6 mining slices were drawn before that round's re-audit was
   applied.** The slice is a mining device, not a readout; every reported number is
   computed against the corrected state.
7. **The decomposition pass is n = 2 families.** Eighteen MIXED channels remain
   undecomposed (chiefly the "established following / self-promotion" family, .520–.562).
   Its parent is far less plausibly quality-causing than the editing family's, so it was
   not prioritised — but that is a judgement, not a measurement, and the +.0055 result
   argues for decomposing it too.
8. **Round 8 is a failed measurement, not a null result.** The channels exist and audited
   cleanly; the judge could not score them (scrambled anchor above coherent text, 83%
   mean modal share). No ground-truth reply order or timestamp exists on this cell's
   source table, so the position hypothesis is untested here rather than refuted.
9. **The two-sided sign band was adopted mid-campaign** (rounds 6+). Rounds 1–5 ran under
   the one-sided rule and their re-audits reflect it; the 6:1 over-firing measured in
   round 6 implies several of the earlier re-routings were probably nulls.

### Artifacts

All under `methods/taste_decomposition/closure/cw_community/`:
`RUNBOOK.md` (exact command sequence), `AGENT_PROMPTS.md` (every sealed prompt wrapper),
`stage0_*.py` + `pop_ext_manifest.json` (population extension and its gates),
`closure_lib_cw.py` (frozen Layer-1 `family1` spec, reproduction-gated),
`stage1_slice.py`, `fleet_cw.py` + `run_codex_cw.py` / `run_glm_cw.py` (sealed fleet),
`build_species_prompt.py`, `select_and_route.py`, `build_audit_prompt.py`,
`finalize_routing.py`, `score_round_gemma.py`, `round_readout.py`,
`missing_mass_cw.py`, `gepa_phrasing.py`, `build_report.py`, `gpu_runner.sh`;
extension-specific: `lineation_challenge_prompt.txt` + `lineation_challenge_adjudication.json`
(the challenged-channel ruling), `decomposition_lineation.json`,
`decomposition_editing_help.json` + `decomposition_editing_help_prompt.txt`,
`retired_parents.json` (Addendum-3 retirement registry, 7 parents),
`round7_decompositions.json`, `round8_results_GATE_FAILED.json` (the position pass's
diagnostics, excluded from every readout), `round8_arbiter.json` +
`round8_routing.PRE_ARBITER.json`;
per round `round{r}_{slice,fleet_A,fleet_B,species,probes,proposals_provenance,
audit_blind,audit_key,audit_verdicts,sign_reaudit,routing,criteria,scores,results,
state}.*` with `*.PRE_REAUDIT.json` kept wherever the trigger changed a routing.
