# HashtagWars verdict — DEEP AUDIT (independent, domain-adjacent)

Date: 2026-08-11. Auditor: the jokes_community closure agent, assigned as the independent
reviewer for this cell. User-ordered.

Cell: **HashtagWars verdict** — SemEval-2017 Task 6, @midnight Hashtag Wars. One contest
per hashtag; y = 1 if the tweet made the contest's top ten.
Audited artifacts: `methods/taste_decomposition/closure/maps_hw_si/`,
`methods/taste_decomposition/results/f2_deconf_hashtagwars_verdict.json`,
`notes/2026-08-08__maps_hw_si.md`.
New code written for this audit: `batch_audit.py`, `batch_within_class.py` (same dir);
outputs `hashtagwars_batch_audit.json`, `hashtagwars_within_class.json`.

Terms spelled out, per the standing rule. **V** = 20 programmatic surface features.
**A** = the Gemma-4-31B-judged articulated-criterion bank. **VA_nl** = HistGradientBoosting
aggregation of V+A, seed-mean over {0,1,2}. **T** = the Llama-3.1-8B LoRA dense standard.
**Δ_beyond** = T − VA_nl. **E** = the 924 dense-held-out rows the F2 fusion refits on.
**HONEST** = the same 924 rows, the only population on which T is out-of-sample.
**F2** = the deconfounded-fusion arm set. **Sweep** = the retrieval pass a tweet was
collected in. **AUC** = area under the ROC curve. **SE** = standard error.
Length readouts are in **tokens**, never characters.

---

## 0. HEADLINE

Three findings, in order of how much they change the cell's verdict.

1. **The retrieval-batch confound is real, far worse than recorded, and IRRELEVANT to the
   residual.** The label is 97.4% recoverable from tweet-id metadata — but that leak is
   invisible to text (tf-idf reads the sweep at AUC .479, chance), so it cannot be what the
   dense model is reading. The campaign's own reading of channel r1:B04 as "that batch
   difference visible in the text" is **not supported**.
2. **The residual is not distinguishable from zero under any frame.** Campaign Δ_plateau
   +.0286 has jackknife SE .0607 (t = 0.47); the F2 matched-strength increment is −.0230
   [−.063, +.029]. The F2 "starvation gap" is a diagnostic of a bank refit on 924 rows,
   not evidence about nuisance.
3. **84% of the closure came from nine rubric rewrites — and they are cleanly judged.**
   *(Revised 2026-08-11 after the gate recompute in §8; the first pass of this audit read
   the batch-level gate failure as evidence of a degraded judge. That reading was wrong and
   is retracted here.)* The decomposition batch's published coherent-vs-scrambled **.5876**
   is a **composition artifact**: on the nine A-routed components that actually joined the
   bank the same stored anchors give **.9897 PASS**. The 84% is a real measurement of
   **craft entangled with surface carriers inside the frozen bank's rubrics**, not of a
   broken instrument.

**Recommended strict-list row: NULL — no quotable taste residual.** (§4.)

---

## 1. Q1 — What the retrieval batch IS, and whether the dense edge rides on it

### 1.1 The structure, established from the raw release

The SemEval-2017 Task 6 release is one `.tsv` per hashtag contest: `tweet_id`,
`tweet_text`, `label` ∈ {0 not-top-10, 1 top-10, 2 winner}. Every contest carries exactly
one winner and nine (occasionally eight) runners-up.

Twitter ids are Snowflake — monotone in posting time. Sorting each contest by `tweet_id`:

| quantity | value |
|---|---|
| pooled AUC(within-contest tweet_id rank → top-10), 101 train contests, 11,325 tweets | **.0279** |
| contests where **max(positive id) < min(negative id)** — fully disjoint ranges | **88 / 101** |
| median fraction of negatives with an id below max(positive id) | **0.0000** |
| per-contest id-rank AUC | mean .0100, 96% below .10, **none above .5** |
| **label-free** sweep split (cut each contest at its largest id gap) → AUC on y | **.9737** |

An early-posting *effect* would produce overlapping distributions with a shifted mean.
What is actually there is **two disjoint id intervals per contest**: the top-ten tweets —
already known from the broadcast — were pulled in one API sweep and the filler negatives in
a later one. The label is recoverable from collection metadata alone at AUC **.974**, and
the label-free operationalisation (split at the largest gap, no labels used) recovers it
just as well, so this is not an artefact of how I defined the channel.

This is a more severe statement than §14.5 of the campaign note, which called it a
"downstream corpus-construction artifact" at rank-AUC .023. It is a near-deterministic
metadata leak, and it is a property of **the public dataset**, not of this campaign.

### 1.2 But the sweep is invisible in the text — so it cannot carry the dense edge

No model in this programme sees `tweet_id`. The leak can only reach a text model through a
textual fingerprint of the sweep. There is none.

| test | result |
|---|---|
| grouped-OOF tf-idf (1–2 gram) logistic, **text → sweep** | **AUC .4787** (chance) |
| the same model, text → y | AUC .5915 |
| 8 surface features → sweep (tokens, urls, @-mentions, @midnight, hashtags, ends-with-hashtag, quote char, uppercase ratio) | all in **[.440, .497]**; strongest is ends-with-hashtag at .440 |

And the sharper test, because sweep and y are collinear so a raw sweep-AUC would be
uninformative: **does any named channel track posting time WITHIN a label class**, where
the sweep is nearly constant and time still varies? A scrape/time channel must; a craft
channel must not.

Across all **44** Track-B and Track-A channels scored in the campaign:

| | max |ρ| | mean |ρ| |
|---|---:|---:|
| within negatives (n ≈ 3,489) | **.063** | .020 |
| within positives (n ≈ 367) | .218 | .057 |

Including the two channels the campaign named as the batch fingerprint:

| channel | alone-AUC on y | ρ with posting time, negatives | ρ, positives |
|---|---:|---:|---:|
| r1:B04 Boilerplate tag placed before vs. after the punchline | .374 | **+.052** | +.114 |
| r2:B10 Contest boilerplate load | .381 | **−.036** | −.125 |

**Verdict on Q1.** The dense edge is **not** carried by the retrieval batch. The batch is
undetectable from text by a tf-idf reader, by any of eight surface features, and by any of
the 44 judged channels once the label is held fixed. The campaign note's §2 interpretation
of r1:B04 — "this channel is that batch difference visible in the text" — should be
**retracted**: B04 predicts y at .374 while being flat in posting time within both classes,
so whatever it reads, it is not the sweep.

The coordinator's conditional ("if the answer is nearly all, the honest verdict is
residual = retrieval-batch artifact") therefore resolves the other way: **the residual is
not a retrieval-batch artifact.** It is also not a taste band — for the different reasons
in §2 and §5.

### 1.3 A method point that must be recorded: the sweep cannot be used as a discount channel

The coordinator asked for the level gap decomposed "stratified + matched on batch". **That
computation must not be run, and its output must not be quoted.** The sweep is 97.4%
collinear with y. Stratifying on it leaves within-stratum y nearly constant, so the AUC is
undefined on almost every stratum and whatever emerges is driven by the handful of mixed
strata. This is the same failure mode the programme already flags for decile discounting on
a strong nuisance, in its extreme form. The within-class time test in §1.2 is the
substitute, and it is the right one because it holds the collinear variable fixed by
construction.

---

## 2. Q2 — Why E is group-poor, and whether a cross-fitted rebuild is warranted

**How E was formed.** The dense arm is a single contest-grouped 80/10/10 split of the
4,228-row / 40-contest population. E is just its held-out block: **924 rows, 8 contests,
80 positives** (pos-rate .0866). Nothing pathological happened — E is group-poor because a
single split of 40 groups leaves 8, and the cell's grouping unit is the contest.

**What a cross-fitted rebuild would buy.** Five-fold cross-fitting (the SO question-hash
bucket template) makes every row honest: E becomes **4,228 rows / 40 contests / 397
positives** — 5× the containers and ~5× the positives.

**Does the verdict hinge on it? No.** Jackknife SE scales roughly as 1/√(containers):

| | containers | Δ | SE | t |
|---|---:|---:|---:|---:|
| state 0 (frozen bank) | 8 | +.0572 | .0503 | 1.14 |
| **state 4 (saturation, current)** | 8 | **+.0286** | **.0607** | **0.47** |
| state 4 after a 5-fold rebuild (projected) | 40 | +.0286 | **≈ .0271** | **≈ 1.05** |

Even with every contest honest, the plateau sits about one standard error from zero. The
rebuild costs five LoRA trainings and returns a result that is **still not significant**,
and it cannot rescue a point estimate that the matched-strength arm already puts at −.0230.

**Recommendation: do NOT build the cross-fitted arm for this cell.** No GPU claim is
warranted, and none was made. (If it is built later for cross-cell consistency, the
projection above is the number to check it against — if the realised SE is much below .027
the projection was wrong and the verdict should be revisited.)

---

## 3. Q3 — Is this cell curation-like? Yes: a fixed-quota editor pool

| quantity | value |
|---|---|
| contests in the audited population | 37 matched (40 total) |
| raw contest size (tweets) | mean **104.2**, range [12, 181] |
| **positives per contest** | mean **9.92**, sd **0.28**, range **[9, 10]** |
| pos-rate on E | .0866 |

The release fixes the number of winners per contest at ten (one winner + nine runners-up;
11 of 101 contests have eight runners-up). **y is a fixed-quota rank-within-pool label**,
not an independent per-item judgement: whether a tweet is positive depends on the other
~104 tweets in its contest, and the *number* of positives is a constant of the design
rather than a property of the entries.

That is the cap-era editor-pool shape exactly. Two consequences the cell's readouts should
carry:

* **Pool size, not quality, drives the base rate.** A contest of 12 tweets has a ~83%
  positive rate available and one of 181 has ~5.5%. Pooled AUC over contests of very
  different sizes mixes these; the within-contest readout is the one that matches the
  y-definition, and it is the tier this cell should have led with.
* **Level shift, not rank shift, is what a craft criterion buys here** — the same signature
  the cap-crowd/cap-finalist pair showed. A criterion can be genuinely predictive of
  "good tweet" and still move few pool decisions, because the quota is binding.

This also explains, without any appeal to taste, why absolute AUCs on this cell are low for
every instrument: with a binding quota and ~9% base rate, most of the ranking work is
deciding among near-ties.

---

## 4. Q4 — Reconciling the three residual claims into ONE row

The three numbers are not in tension. They are three different estimands, and once each
frame is named they agree.

| claim | frame | number | what it actually estimates |
|---|---|---|---|
| campaign Δ_plateau | **matched strength**: bank grouped-OOF on 3,712 FIT+MINE rows, T trained on 3,304 dense-train rows, both read on the 924 honest rows | **+.0286**, jackknife SE **.0607**, t = 0.47, band [+.011,+.047] | the residual after mining — *a direction, never a level* (the campaign says so itself) |
| F2 E-refit increment (d − c) | **starved**: the enriched bank is REFIT on E — 158 features on 924 rows with 80 positives | +.0297, p(>0) .944 | mostly the refit's failure. Bank on E = **.5357 ≈ chance**; the +.1393 "starvation gap" against full-strength .6751 is the size of that failure |
| "nuisance block alone .6267 BEATS the E-refit bank .5357" | same starved frame | — | **a starvation diagnostic, not a fact about nuisance.** 44 nuisance columns survive an E-refit that 158 bank columns do not. It says nothing about which block carries information at full strength |
| F2 matched-strength increment | **matched strength**, the corrected version | **−.0230 [−.063, +.029]** | **NULL** — the dense arm adds nothing over bank + nuisance once strength is equalised |

The apparent "level vs increment tension" — a +.056 level gap at full strength but no
increment from adding T — dissolves the same way. The +.056 is state 0's gap (T .7315 vs
frozen bank .6743) with **SE .0503, t = 1.14**; it was never a significant level. Mining
and rewriting halved it to +.0286 (t = 0.47), and the matched-strength increment is
negative. Nothing needs the nuisance block to carry "dense's entire edge", and §1 shows the
batch does not carry it either.

### RECOMMENDED STRICT-LIST ROW

> **hashtagwars_verdict — Δ_beyond NULL. Not quotable as a taste residual.**
> Terminal campaign Δ_plateau +.0286 on 924 honest rows / 8 contests, container jackknife
> **SE .0607 (t = 0.47)**; F2 matched-strength increment **−.0230 [−.063, +.029]**. Both
> frames are consistent with zero. Never point-quote +.0286. The F2 E-refit arms
> (bank .5357, nuisance .6267, increment +.0297) are **starvation-frame only** and must
> not be quoted as evidence about nuisance or about the residual.
> **Three caveats travel with any use of this cell:** (i) the y-label is 97.4% recoverable
> from tweet-id metadata (retrieval-sweep confound in the public dataset; invisible to text,
> so it does not bias the text arms, but it disqualifies any arm that could see collection
> metadata); (ii) 84% of the measured closure came from nine rubric rewrites rather than from
> mining — a wording/entanglement sensitivity, not a judge failure: those nine criteria
> pass the scrambled gate at **.9897** and the batch-level **.5876** is a composition
> artifact (§8); (iii) y is a fixed-quota rank-within-pool label
> (9.92 ± 0.28 positives per contest), so this cell is curation-shaped and its numbers are
> not commensurable with independent-judgement cells.

---

## 5. Q5 — "Rewrites gave 84% of bank gain": verified, and what it means

**Verified exactly.** From §12.1/§8 of the campaign note: state 0 Δ +.0572 → state 4
+.0286, so **+.0286 was closed**. The Addendum-3 decomposition pass (nine rewrites of
rubrics the bank already had, zero new concepts) contributed **+.0241**; 54 mined criteria
across four rounds contributed **+.0045**.

    +.0241 / +.0286 = 84.3%   rewrites
    +.0045 / +.0286 = 15.7%   mining

**Interpretation — REVISED 2026-08-11 (see §8).** The first pass of this audit paired the
84% with the batch's published gate failure (.5876) and concluded that the residual was
dominated by instrument degradation. **The gate recompute in §8 refutes that**: on the nine
A-routed components the same stored anchors give coherent-vs-scrambled **.9897**, with
scrambled word-salad scoring **0.15** against real negatives **4.58**. The judge is reading
those criteria emphatically well. The published .5876 is the average of that near-perfect
pass and a perfectly inverted failure on nine *surface-extent* channels (**.0000**), where
a scramble legitimately keeps its length, capitals and hashtags.

So the 84% stands, and it means something better-founded than a broken judge: the frozen
bank's rubrics had **craft entangled with surface carriers**, and Addendum-3 decomposition
is what separated them. This is the same mechanism the jokes_community campaign measured
directly — a MIXED parent whose two halves each carry more signal than the parent did —
appearing here with a *positive* closure outcome rather than as measurement only.

Two consequences worth carrying beyond this cell:

* **Wording sensitivity is a first-order instrument caveat**, not a polish step. On this
  cell rewriting nine existing rubrics was worth 5× what 54 newly mined criteria were
  worth. The jokes_community campaign independently found the same direction at smaller
  scale (its GEPA phrasing pass was worth +.0041 of +.0274 total closure, ~15%);
  HashtagWars is the extreme case. Neither is about the judge; both are about the rubrics.
* **The scrambled gate must be computed on the subset it is valid for.** A scrambled
  control tests whether the judge is *reading*, so it is only meaningful on criteria whose
  value scrambling destroys. Pooling extent-of-surface channels into it makes the gate
  measure batch composition. The campaign wrote this carry-forward itself; this audit
  confirms it quantitatively and it should be promoted to a standing rule.

## 6. What I did not do, and why

* **No cross-fitted dense rebuild** — §2 shows the verdict does not hinge on it (projected
  t ≈ 1.05). No GPU was claimed; the ledger is untouched by this audit.
* **No stratified/matched discount on the batch channel** — §1.3: undefined at 97.4%
  collinearity. Refusing to produce that number is itself an audit finding.
* **No rescore of the "failed" batch — and, after §8, none is needed.** The coordinator
  authorised the GPU job; the gate recompute made it unnecessary before a card was claimed
  (reuse-before-rebuild). No GPU was used anywhere in this audit.
* certA re-survey (strict A-mass .3667 vs τ-era .733, Z = .78 ABSORBABLE) was read and is
  consistent with everything above: the τ-era mass was inflated by under-merging, exactly
  as the jokes_community campaign found in four consecutive rounds, and the strict figure
  is the one of record. It does not bear on the residual verdict either way.

## 7. Artifacts

| file | contents |
|---|---|
| `methods/taste_decomposition/closure/maps_hw_si/batch_audit.py` | raw-release sweep reconstruction, population join, text→sweep tests, surface table |
| `.../hashtagwars_batch_audit.json` | its output |
| `.../batch_within_class.py` | within-class posting-time test over all 44 channels, pool structure, E geometry |
| `.../hashtagwars_within_class.json` | its output |

Join discipline: population → raw tweets on (contest, whitespace-normalised tweet text)
parsed out of the frozen context template — **no id-dict join**. 3,856 / 4,228 population
rows matched (91.2%); **all 924 honest rows matched**; label disagreements after join
asserted **0**; unmatched rows dropped, never imputed.


---

## 8. FOLLOW-UP (2026-08-11) — the authorised rescore was NOT run, because there is nothing to repair

The coordinator authorised a full rescore of the nine decomposition components under a
fresh passing anchor battery. Before claiming a card I checked whether the repair was
needed, since the batch's anchor scores are already on disk
(`hashtagwars_verdict_rd_scores.npz` carries `Xanchor`, 150 anchors × 23 criteria) and the
campaign's own diagnosis says the pooled gate is misapplied.

`gate_recompute.py` → `hashtagwars_gate_recompute.json`, same stored anchors, K = 50/class:

| subset | n | coherent-vs-scrambled | | scrambled mean vs negatives |
|---|---:|---:|---|---|
| ALL 23 (the published gate) | 23 | **.5876** | FAIL | 3.93 vs 3.80 |
| **A-ROUTED 9 components — the governing subset** | 9 | **.9897** | **PASS** | **0.15 vs 4.58** |
| B-routed channels | 14 | .0013 | FAIL | 6.37 vs 3.29 |
| surface-extent channels only | 9 | **.0000** | FAIL | 7.85 vs 3.40 |

The published number is the average of a near-perfect pass and a perfectly inverted
failure. On the surface-extent channels the inversion is *correct behaviour*: a word-salad
built from two tweets is about twice the length of a real tweet and keeps its capitals and
hashtags, so "extent of capitals, length and hashtag count" **should** score it high.
Scrambling does not destroy an extent. The gate is only a reading test on criteria whose
value scrambling destroys, and on those nine it is passed emphatically.

Per criterion, A-routed: **8 of 9 pass individually** (.745 to .990). The one below
threshold, A02 "Observational accuracy about a real shared behaviour" at .660, has
scrambled mean **0.00** against negatives 1.36 — the judge gives near-zero to almost
everything on it, so the AUC is depressed by ties, not by misreading. It is the
lowest-variance of the nine, not the least-read.

`delta_reproduce.py` → `hashtagwars_delta_reproduce.json` then recomputed the two Δ states
that carry the 84%, from the frozen matrices under the frozen closure protocol:

| state | features | VA_nl HONEST | published | abs diff | Δ_beyond |
|---|---:|---:|---:|---:|---:|
| state 0 (V + A_base) | 49 | .6743 | .6743 | **0.0000** | +.0572 |
| state_d (+ 9 A-routed components) | 58 | .6984 | .6984 | **0.0000** | +.0331 |

T on HONEST recomputed .7315, published .7315. Decomposition gain recomputed **+.0241**,
published +.0241. **Byte-exact reproduction.**

*Sign correction (the SI rule) is a no-op here and is recorded rather than skipped:* it
matters when criteria are combined by an unweighted mean, where an inverted criterion
cancels a correct one. VA_nl aggregates with HistGradientBoosting, which is invariant to a
monotone flip of any single feature, so no sign correction can move these numbers.

### Does +.0286 survive? — the finalising answer

**Yes as a computation, no as a claim, and the gate was never what decided it.**

* The A block needs no repair, the Δ curve reproduces exactly, and +.0286 is what the
  frozen protocol produces on this cell.
* It remains **not distinguishable from zero**: container jackknife **SE .0607, t = 0.47**,
  and the F2 matched-strength increment is **−.0230 [−.063, +.029]**. Neither of those two
  numbers involves the anchor battery, so nothing in this follow-up moves them.

**The strict-list row from §4 stands unchanged, with caveat (ii) reworded as above.** The
HW row is final: **Δ_beyond NULL, not quotable as a taste residual.**

### Two corrections this follow-up makes to the audit's own first pass

1. **Retract** "84% of the closure came from a judging batch that FAILED the scrambled
   gate" and the inference that the residual is dominated by instrument degradation. The
   nine components pass at .9897.
2. **Keep** the 84% itself and the wording-sensitivity caveat, but re-found them on
   craft/surface entanglement inside the frozen rubrics rather than on judge quality.

### Recommended standing rule (new)

> **The scrambled anchor gate is computed on the subset of criteria whose value scrambling
> destroys — in practice the A-routed subset — and never pooled over a batch containing
> extent-of-surface channels.** A pooled gate on a mixed batch measures composition, not
> reading. Both the pooled and subset figures should be recorded; the subset figure gates.

### Artifacts added by this follow-up

| file | contents |
|---|---|
| `.../gate_recompute.py`, `.../hashtagwars_gate_recompute.json` | gate on all 23 / A-routed 9 / B-routed 14 / surface-only, plus per-criterion A-routed |
| `.../delta_reproduce.py`, `.../hashtagwars_delta_reproduce.json` | state0 and state_d reproduction against published values |

No GPU was claimed for this follow-up; the ledger is untouched by the entire audit.
