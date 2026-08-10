# Journalism CURATION cell — homepage story selection — completion build

Charge: finish the journalism curation cell. Its dense side was already done; its
articulated side needed a rebuild. This note is the build record. Registry / strict-list
logging is the coordinator's; nothing under `latex/`, the strict list, the registry, or
any frozen note is touched here.

Cell slug: **`homepage_curation_storygrouped`**
(the pre-existing `homepage_curation` ledger is the OUTLET-HELD-OUT build and is left
untouched — different bank, different grouping, different numbers).

---

## 0. What was already on disk, and what it actually says

**The dense side was done and it reproduces.** First task was verification, not
rebuilding. `datasets/news-homepages/va/dense_standard_storygrouped/` (sk3) holds three
trained seeds and a scoring pass. Every one of the six AUCs in `eval_pass_results.json`
was **recomputed from the raw per-row prediction files** (`rm_out_seed*/preds_{eval,test}.csv`)
rather than copied, and all six reproduce exactly at the 4-dp precision the run reports:

| seed | eval AUC | test AUC |
|---|---:|---:|
| 42 | .7173 (.7173288599324349) | .7429 (.7428594455190201) |
| 1  | .7093 (.7092614248060289) | .7401 (.7401284885327438) |
| 2  | .7061 (.7060757508260013) | .7362 (.7361610021184489) |
| **mean** | **.7109** (spread .0113, sd .0058) | **.7397** (spread .0067, sd .0034) |

Row counts (1,313 eval / 1,318 test), snapshot counts (132 / 178) and pos-rates
(.50038 / .50076) also reproduce from the prediction files against the manifest.
Recorded formally in **`methods/taste_decomposition/results/samerows_T_homepage_storygrouped.json`**,
in the `samerows_T_*` schema, with per-file sha256 prefixes.

Two numbers are held apart in that file and must never be conflated:

* **T = .7109** — the corrected story-grouped design, 12,368 rows over the current
  frozen population, 3 seeds.
* **T = .824 (historic)** — real, but a *provisional* snapshot-grouped sweep on an older
  4,400-row split whose A side was 70B-judged. Different split, different row set. Both
  are recorded; neither is averaged, differenced, or quoted for the other.
* **T = .4322 (outlet-held-out)** — RETIRED as unpowered, not as a cell failure: k=2
  held-out outlets that disagree in sign (eval .432/.459/.439 vs test .736/.743/.753).
  Kept as a labelled secondary only.

**Weak-instrument flag, carried on every number below:** y is homepage *spatial
placement* (1 = the link renders in the top half of the capture's top-30% zone), jointly
determined with layout/ad/image constraints, not a clean editorial preference.

---

## 1. The coherence failure this rebuild exists to fix — quoted

The A side failed, and the failure is documented in three places. The registry entry
(`notes/2026-07-27__vat-run-registry.md`) states it most directly:

> **HOMEPAGE: instrument failure, terminal as documented-unbuildable-cross-outlet** —
> census bank FAILS coherence (scrambled .387: **entity detectors, not reading
> instruments**); outlet-held-out T UNUSABLE (eval .432 below chance vs test .736, two
> held-out outlets disagree in sign).

and its later correction separates the two failures:

> the census BANK's coherence failure (entity-detector criteria) stands as a separate
> A-instrument issue **needing rebuilt criteria**. Cell = live pending both.

The wave-C build manifest carries the same verdict on the bank:

> the A bank for this cell **FAILS the coherent-vs-scrambled gate (.387, below chance)**
> — a separate A-instrument issue, untouched by this build. Any Delta_beyond computed
> against it is a statement about a news-values lexical profile, not about an
> articulated-criteria reading instrument.
> — `datasets/news-homepages/va/dense_standard_storygrouped/manifest.json`, `bank_note`

and the 3×N grid note's journalism row:

> homepage: **CORRECTED 2026-08-08 — LIVE: historic story-grouped T .824 is real; only
> outlet-held-out (k=8 outlets) failed = unpowered transfer design; story-grouped T rerun
> dispatched; census bank's coherence failure stands separately (criteria rebuild
> needed)**
> — `notes/2026-08-08__vat-3xN-decomposition-grid.md`

The raw numbers behind ".387", from `homepage_curation_ledger.json` `anchor_battery`
(K=50 per class):

| anchor class | n used | mean row score |
|---|---:|---:|
| positive | 50 | .5414 |
| negative | 50 | .4980 |
| **scrambled word salad** | **34** | **.5776** |

pos-vs-neg AUC .5738; **coherent-vs-scrambled AUC .3869**; `ordering_holds_on_means`
false. Scrambled text scored *above* both real classes. 16 of the 150 anchor rows drew
NA on all 14 criteria and were dropped from the statistic.

### Why it failed — the mechanism, since the fix has to target it

Two design choices in the census bank interact, and together they produce exactly this
inversion.

1. **The criteria were entity detectors.** `a01` (Elite political actor is a central
   subject) and `a09` (Famous non-political figure or major organisation) fire on the
   *presence* of a recognisable name. Scrambling preserves every token, so it preserves
   every entity — and the campaign's `scramble()` mixes the words of a positive and a
   negative headline, so a scrambled anchor carries **both** headlines' entities and
   scores *higher* than either.
2. **NA was defined topically, and all-NA rows were dropped.** The census system prompt
   said `NA = the headline gives no evidence bearing on this criterion`. So a scrambled
   string drew NA on the criteria it could not satisfy, those cells were excluded from
   the row mean, and the surviving cells were precisely the entity detectors. The
   battery then dropped 16 all-NA rows entirely — 16 of the 50 scrambled anchors never
   reached the statistic at all.

The same NA-as-topic rule is what made 4 of 14 criteria mostly missing (crisis .616,
economic .717, legal .713, violence .491 NA) and drove `na_rate_overall` to .281.

### Per-criterion evidence, recovered from data already on disk (PRELIMINARY)

The census battery only ever reported a row mean, which is why "*which* criteria are the
entity detectors" was unanswerable. But each scaleupC shard did save its 3 blinded anchor
rows **per criterion** (`anchor_X`), so 18 rows (6 pos / 6 neg / 6 scrambled) × 14
criteria were recoverable without spending a single new judge call
(`outputs/va_gemma_banks_homepage_v2/legacy_percriterion_prelim.json`).

**Caveat, load-bearing:** `score_bank` re-draws anchors until pos > neg > scrambled
holds, so these are *the draws that passed* — the statistic is biased **toward**
coherence, and n is 18. It is preliminary. Stage 0 of this build rescores all 14 criteria
on a fresh K=50 battery through their original prompt.

| coh-AUC | pos | neg | scram | NA on scram | criterion |
|---:|---:|---:|---:|---:|---|
| 1.000 | 1.00 | 1.00 | 0.50 | .83 | Active or imminent crisis |
| .938 | 0.50 | 0.50 | 0.00 | .83 | Violence, casualties, confrontation |
| .833 | 0.75 | 0.62 | 0.25 | .67 | Economic impact reaching the reader |
| .667 | 0.50 | 0.80 | 0.33 | .50 | Famous non-political figure or major organisation |
| .604 | 0.67 | 0.75 | 0.50 | .67 | Domestic or directly reader-relevant |
| .600 | 0.50 | 0.25 | 0.25 | .67 | Concrete institutional action is reported |
| .542 | 0.92 | 0.67 | 0.67 | .50 | Large scale of people or stakes |
| .542 | 0.00 | 0.08 | 0.00 | .50 | Emotional human-interest drama |
| **.417** | 1.00 | 0.75 | **1.00** | .67 | **Hard news rather than soft** |
| **.367** | 0.33 | 0.25 | 0.50 | .50 | **Elite political actor is a central subject** |
| **.333** | 0.83 | 0.50 | **1.00** | .83 | **Part of the day's top-tier running story** |
| **.250** | 0.50 | 0.50 | **1.00** | .83 | **Legal accountability proceeding** |
| n/a | 0.83 | 0.67 | — | **1.00** | Just-happened or developing |
| n/a | 0.67 | 0.25 | — | **1.00** | Unexpected, record-breaking, novel |

Even biased toward passing, **4 of 14 criteria sit below chance**, and the reading is
sharper than "entity detectors" alone:

* **The bank's two strongest predictive columns are two of its worst readers.**
  *Hard news rather than soft* (alone-AUC **.605**, the bank's best) and *Part of the
  day's top-tier running story* (**.599**, second best) both score **scrambled word salad
  at 1.00** — higher than they score real negatives. Whatever predictive power the census
  bank had was concentrated in columns that rate nonsense as top-tier news.
* **The NA route is visible directly.** Every criterion answers NA on at least half its
  scrambled anchors, and two (*Just-happened*, *Unexpected*) answer NA on **100%** of
  them — so they contribute nothing at all to a scrambled row's mean, leaving that mean
  to be set by whichever token-presence criteria did fire. This is the drop-and-survive
  mechanism above, measured.

Both observations are what the v2 design targets. **Two of the three, and one of the
rankings, were corrected by the proper K=50 triage — see §1b, which supersedes this
table.**

### 1b. Legacy triage, measured — stage 0 (SUPERSEDES the preliminary above)

The 14 census criteria rescored on a fresh K=50 battery **through their original system
prompt** (`anchor_battery{,_percriterion}.json`, key `homepage_curation_legacy`).

**First: the headline ".387" is draw-noisy, and I am recording that rather than leaning
on it.** Same construction, same K=50, a different anchor draw (the two builders seed
their RNG differently) gives:

| | archived scaleupC draw | fresh stage-0 draw |
|---|---:|---:|
| pos / neg / scrambled row means | .5414 / .4980 / **.5776** | .5523 / .5164 / **.4941** |
| pos-vs-neg AUC | .5738 | .5565 |
| **coherent-vs-scrambled AUC** | **.3869** | **.5617** |
| scrambled rows surviving (of 50) | 34 | 37 |

A statistic that moves .387 → .562 between two draws of the same design is not a precise
instrument at K=50. **The durable finding is not the row mean — it is the per-criterion
decomposition, and that is unambiguous.**

**The all-NA drop is 13× concentrated on the scrambled class** (the statistic the old
battery hid, now reported by tag):

| anchor class | n | all-NA rows dropped |
|---|---:|---:|
| positive | 50 | 1 |
| negative | 50 | 0 |
| **scrambled** | 50 | **13** |

So the row mean for scrambled input is computed on the 37 rows where *something still
fired* — and what fires is exactly what the next table shows.

**9 of 14 census criteria are below chance on coherence.**

| coh-AUC | pos | neg | scram | NA on scram | criterion |
|---:|---:|---:|---:|---:|---|
| .774 | .37 | .30 | .00 | .96 | a11 Unexpected, record-breaking, novel |
| .708 | .74 | .82 | .46 | .74 | a12 Domestic or directly reader-relevant |
| .657 | .72 | .67 | .44 | .64 | a05 Large scale of people or stakes |
| .654 | .70 | .61 | .50 | .98 | a06 Just-happened or developing |
| .581 | .83 | .69 | .61 | .44 | a13 Hard news rather than soft |
| .499 | .16 | .12 | .13 | .62 | a10 Emotional human-interest drama |
| **.496** | .69 | .56 | .67 | .82 | a03 Active or imminent crisis |
| **.484** | .62 | .55 | .61 | .82 | a04 Violence, casualties, confrontation |
| **.475** | .33 | .52 | .47 | .70 | a09 Famous non-political figure / major org |
| **.440** | .74 | .64 | .75 | .84 | a14 Part of the day's top-tier running story |
| **.417** | .61 | .66 | .75 | .96 | a08 Economic impact reaching the reader |
| **.414** | .50 | .53 | .67 | .94 | a07 Legal accountability proceeding |
| **.401** | .43 | .44 | .62 | .68 | a01 Elite political actor is a central subject |
| **.241** | .39 | .41 | **.80** | .90 | **a02 Concrete institutional action is reported** |

Readings:

* **a02 is the single worst instrument in the bank.** It rates word salad at **.80**,
  roughly *twice* what it gives real headlines of either class (.39 / .41), for a
  coherence AUC of .241. It is an institutional-noun detector.
* **All six criteria I dropped are vindicated** — a01 .401, a09 .475, a03 .496, a04 .484,
  a07 .414, a08 .417, every one at or below chance.
* **My 18-row preliminary was directionally right about a01 and a14 but WRONG about a13.**
  The preliminary put *Hard news rather than soft* at .417 with scrambled at 1.00; on
  K=50 it is **.581, above chance**. a13 was de-genred (→ b14) because it is a *section
  label*, and that reason stands on its own — but the claim that it "scores word salad at
  ceiling" does not survive the proper battery and is withdrawn.
* **Two criteria I chose to SALVAGE are themselves below chance in the legacy
  instrument** — a14 (.440) → b09 and a02 (.241) → b11. That is not an oversight: both
  salvages are accompanied by exactly the repair their failure mode calls for (a02's
  "requires an institution" clause deleted, both NA branches folded into 0.0, and a02's
  .90 NA-on-scrambled route closed by the prompt). Whether the repair works is a
  falsifiable prediction, and §1c tests it on the same battery.
* **NA-on-scrambled is above .60 for every single criterion and above .90 for five.** The
  bank could barely be *asked* about incoherent text; it answered "not applicable" and
  the row mean was left to the detectors.

### 1c. Does the rebuild fix it? — stage 1/2 pilot, gate PASSED

Same battery, same anchor construction, v2 bank and v2 prompt (300-item pilot,
`--battery 50`):

| | census bank (stage 0) | **v2 bank (stage 1)** |
|---|---:|---:|
| **coherent-vs-scrambled AUC** | .5617 (archived draw .3869) | **.9900** |
| criteria below chance on coherence | **9 / 14** | **0 / 29** |
| scrambled row mean | .4941 | **.0000** |
| all-NA scrambled rows dropped | 13 / 50 | **0 / 50** |
| NA rate on items | .281 (full census run) | **.0000** |

Coherence backbone, individually: b01 .840, b02 .885, b03 .970, b04 .910, b05 .940 —
every one far above the .60 the gate required. **Gate PASS**; the chain proceeded to the
full 376,942-prompt run. No criteria revision round was needed, so the GEPA loop closed
at iteration 1.

The two prompt rules did exactly what they were written to do: scrambled text now scores
**0.0 on all 29 criteria** rather than drawing NA and being dropped, and the applicability
channel is empty (NA rate .0000 on 300 items × 29 criteria).

**Two cautions, recorded now rather than after the fact:**

1. **pos-vs-neg AUC on the unweighted row mean is .479 — below chance** (census: .5565).
   The v2 bank separates *coherent from incoherent* superbly and does not, as a flat
   29-criterion average, separate top-half from bottom-half placement. This is not yet a
   verdict on A: layer 1 fits weights, and an unweighted mean of 29 criteria can sit at
   chance while `A_lin` is well above it. But it is the number to watch, and it is
   read against the fact that the census bank's *apparent* pos/neg separation came from a
   bank that was 9/14 entity detectors — i.e. plausibly the genre channel itself. If
   `A_lin` also lands near chance, the honest conclusion is that this cell's articulated
   surface carries little placement signal once the genre channel is removed, and that
   conclusion is *more* informative than the census bank's .5979 was.
2. **Per-shard anchor ordering will likely fail on some shards.** `score_bank` requires
   pos > neg > scrambled within 4 re-draws; on this pilot draw neg (.6597) sat above pos
   (.5934), so the pos/neg leg is fragile even though the scrambled leg is decisive.
   Shards whose anchors never order are recorded INVALID and a leave-those-out
   sensitivity readout is reported beside the headline — never silently dropped.
3. **Five craft criteria are near-ceiling on real headlines** (b22 mean .993 / modal .99,
   b20 .977, b21 .970, b18 .973, b16 .952). Professional newsroom headlines are almost
   always free of jargon, proportionate, and cleanly attributed, so these carry little
   variance. The distribution check flags them; they are kept (dropping columns on a
   variance screen after seeing them would be selection) and their low information is a
   finding about the craft surface, not a defect. The informative spread lives in b08
   (.317), b10 (.312), b28 (.472), b13 (.525), b12 (.575), b07 (.577), b09 (.648),
   b23 (.668).

### The press lesson, which says where that NA channel ends up

`notes/2026-08-10__closure_press.md` §2.2 measured what an applicability-gated bank is
actually doing on a cell where the same rule was in force:

| block | features | VA_nl HONEST |
|---|---:|---:|
| **A_mask_only** — applicability bits, no judged level | 38 | **.7322** |
| A_levels_only — judged levels, mask erased | 37 | .6705 |
| V + mask (no judged level anywhere) | 126 | .7282 |
| the whole V + A primary bank | 126 | .7296 |

> Whatever the press A bank is measuring, it is overwhelmingly *which kind of release
> this is* — a genre fingerprint — and not *how well the release satisfies an
> articulated end.*

Forty Gemma-judged rubric levels were worth **.0014** over the bare fact of which
rubrics a judge thought applied. That is the failure mode the homepage rebuild has to
avoid, and the reason both an applicability-mask ablation and a within-story-type
readout are built into this cell's layer 1 rather than left as follow-ups.

---

## 2. What was salvaged and what was rebuilt

New bank: **`datasets/news-homepages/va/rubrics_v2.jsonl`, 29 criteria** (the census
bank had 14). Every criterion carries `origin` and `gepa_revision` fields naming the
failure mode it repairs. The census file is **not** modified.

| census criterion | disposition |
|---|---|
| a05 Large scale of people or stakes | **SALVAGED** → b06 *Scale of consequence* — bands kept in substance; NA branch folded into 0.0 |
| a06 Just-happened or developing | **SALVAGED** → b07 *Presents itself as new information now* — same, NA branch folded into 0.0 |
| a11 Unexpected / record-breaking | **SALVAGED** → b08 *Departure from expectation* — same |
| a14 Part of the day's top-tier running story | **SALVAGED** → b09 *Instalment in a larger running story* — same (was the bank's 2nd strongest column, .599) |
| a12 Domestic or reader-relevant | **SALVAGED** → b10 *Bears on the reader's own life or country* — same |
| a02 Concrete institutional action | **SALVAGED + DE-GENRED** → b11 *Definiteness of what is reported* — the taken-vs-proposed question kept, the "requires an institution" clause dropped (it was what produced a 29% NA topic bit) |
| a13 Hard news rather than soft | **SALVAGED IN SPIRIT** → b14 *Public consequence rather than private curiosity* — a13 was the bank's strongest column (.605) and also a pure **section label**; b14 keeps the stake gradation and explicitly instructs that section membership does not decide the score |
| a10 Emotional human-interest drama | **SALVAGED IN SPIRIT** → b24 *Emotional stake is legible without melodrama* — a10's "is a specific person's experience central" scored every institutional story 0.0 by definition |
| a01 Elite political actor is a central subject | **DROPPED** — entity detector, the direct cause of the coherence inversion |
| a09 Famous non-political figure or major organisation | **DROPPED** — same |
| a03 Active or imminent crisis | **DROPPED** — topic membership, 61.6% NA |
| a04 Violence, casualties, confrontation | **DROPPED** — topic membership, 49.1% NA |
| a07 Legal accountability proceeding | **DROPPED** — topic membership, 71.3% NA |
| a08 Economic impact reaching the reader | **DROPPED** — topic membership, 71.7% NA, and alone-AUC .504 |

**Rebuilt / new: 21 criteria in four families.**

* **b01–b05 assertion structure** (the coherence backbone) — an actor-action *relation*,
  truth-evaluability, headline well-formedness, self-containment, single focus.
  Scrambling preserves tokens but destroys relations, so these fail on word salad by
  construction. b01 says it explicitly: *"a name with no coherent claim attached to it
  is worth 0.0"*.
* **b06–b14 news-value gradations** — the eight salvaged axes plus b12 (stakes hard to
  undo) and b13 (consequence stated, not just the event).
* **b15–b25 editorial craft / reader pull** — news-in-the-headline, concreteness, verb
  strength, jargon, clickbait withholding, proportionate claim, attribution, quantity
  precision, scannability, legible stake, honest next question. Two of these (b19, b20)
  are **negatively oriented** on purpose: a bank in which every criterion points the same
  way is a length model in disguise (the caption-finalist lesson, where 0/32 rubrics
  survived length strata).
* **b26–b28 page-relative** — the family the census bank had no member of, and the direct
  implementation of "rank quality WITHIN story-type": each compares the focal headline
  with *the other headlines on the same capture*, i.e. against the same day's story mix,
  so a score cannot be earned by belonging to a favoured genre. b28 is an explicit
  within-page rank, chosen because the label is itself a within-capture contrast.

**The system prompt is rewritten, and that is the single most load-bearing change.**
`SYS_HOMEPAGE_V2` (in the scorer) replaces the topical NA rule with:

* `0.0 = fails the criterion, INCLUDING the case where the headline contains nothing the
  criterion could attach to`; `NA = the input is empty`;
* *"IF THE TEXT IS NOT A WELL-FORMED HEADLINE … score 0.0 on EVERY criterion. A text that
  asserts nothing satisfies nothing. Do not award credit merely because a recognisable
  name, place, number, or topic word appears in it"*;
* *"JUDGE THE HEADLINE, NOT ITS SUBJECT … a sports or entertainment headline must be able
  to score as highly as a politics one when it satisfies the criterion as well."*

This closes both halves of the mechanism in §1: the entity route (rule 2) and the
NA-as-genre route (rule 1). It also makes the applicability mask near-degenerate by
construction, which is a *prediction* the ablation in §4 measures rather than assumes.

**GEPA iteration.** Phrasing follows the census bank's own reflective-revision
discipline (every criterion records what the revision changed and which failure it
repairs), and the build adds a measured iteration loop: a 300-item pilot is scored, a
per-criterion coherence battery is computed, and a **label-blind smoke gate**
(`methods/taste_decomposition/check_homepage_smoke_gate.py`, thresholds fixed before the
first pilot) decides whether the full run may proceed. A gate failure stops the chain and
sends the criteria back for revision; each such round is an iteration and is logged here.
The gate never sees y.

---

## 3. Instrument and validity machinery

**Population — reused verbatim, not rebuilt.** `datasets/news-homepages/va/population.csv.gz`,
n = 12,998, pos-rate .5006, 1,229 snapshots, 8 outlets. Identical to the population the
wave-C bank and the corrected dense arm used.

**V** — the existing 23-feature deterministic headline surface bank
(`datasets/news-homepages/va/v_features.py`), unchanged. Computed on the HEADLINE only;
the CONTEXT field is snapshot-constant and is deliberately excluded from V. All 23
columns finite with std > 0 on the full population (checked pre-flight).

**A** — Gemma-4-31B-it, `envs/gemma4` (vLLM 0.23), **offline batch, never an HTTP
server**; temperature 0, `max_tokens` 6, one token per (item, criterion) from
{1.0, 0.5, 0.0, NA}; prefix caching on; `max_model_len` 4096 (longest assembled prompt
4,591 chars ≈ 1.3K tokens); main guard + spawn + `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
`--auto-util` sizes `gpu_memory_utilization` from free memory **at engine-init time**
(the CW-expert landmine: free-at-claim and free-at-init are different numbers), capped at
.93 with an 80 GiB floor and 6 GiB headroom.

Label-blind: y never enters a prompt, and rule 5 of the system prompt forbids inferring
placement, page position, outlet, popularity, or dataset membership.

Scoring loop, shard checkpointing, per-shard 3-row blinded anchors with re-draw, and NA
parsing are imported **verbatim** from `datasets/va_gemma_banks/score_va_gemma_banks.py`;
the K≥50 battery is imported **verbatim** from `score_scaleupC_banks.py`. Only the bank
builder, the system prompt, and the two new diagnostics are new.

**Three validity checks on every batch:**

1. **Blinded anchor battery, K = 50 per class** (positive / negative / scrambled), plus
   3 blinded anchors in every shard re-drawn up to 4× until pos > neg > scrambled. Shards
   whose anchors never order are recorded INVALID and a leave-those-out sensitivity
   readout is reported beside the headline (never silently dropped — temperature-0 item
   scores cannot change on a re-draw).
2. **Per-criterion coherence battery — NEW.** The census battery reported only a row
   mean, which is why "which criteria are entity detectors" was unanswerable. The new
   battery reports, for every criterion, the AUC separating coherent anchors (pos+neg)
   from scrambled ones, plus its NA rate on scrambled input. A criterion below .5 here
   *is* an entity detector. It also reports **all-NA anchor rows by tag** — the statistic
   the old battery hid (it dropped 16 rows without saying which class they came from).
3. **Judge score-distribution collapse check** — per-criterion mean, NA rate, modal
   share, distinct values; fails loudly on all-min collapse, NA flood, or ≥ half the
   criteria pinned to one value.

**Legacy triage (stage 0).** Before the new bank is scored, the **14 census criteria are
re-run through the same per-criterion battery using their ORIGINAL system prompt**, so
the salvage/drop table in §2 is backed by measurement rather than by reading the wording.

---

## 4. Layer 1 — and the two diagnostics built into it

Driver: `methods/taste_decomposition/homepage_v2_layer1.py`. Frozen protocol, machinery
imported from `layer1_gemma_cells.py`: linear = family1 (`SimpleImputer(median,
add_indicator)` + `StandardScaler` + `LogisticRegression(C=1, liblinear, max_iter 2000,
rs 20260728)`), GroupKFold(5) **on snapshot_id**; nonlinear = HistGradientBoosting,
frozen grid {15,31} leaves / lr .06 / max_iter 400 + early stopping, grid by inner
GroupKFold(3) inside each outer train fold, per-fold imputation identical to the linear
leg. VA_nl / V_nl = mean over seeds {0,1,2} with spread reported (FREEZE CHANGE 1);
Δ_interact CI = group-level bootstrap over snapshots (FREEZE CHANGE 3).

**Grouping unit is snapshot_id, matching the corrected dense design.** Outlet-grouped CV
is kept as a labelled descriptive secondary only.

**Same-rows Δ_beyond is ENFORCED, not asserted (FREEZE CHANGE 2).** `samerows_T_press.json`
records a Δ_beyond of +.0486 RETRACTED because T sat on 288 eval rows while VA_nl was
pooled over 2,956. Here the OOF VA vector is restricted **by row id** to the dense arm's
own held-out rows before differencing, and both the eval-only (n = 1,313) and eval+test
(n = 2,631) versions are reported. The pooled-population figure is carried only under a
`POOLED_CONTEXT_ONLY` key with the retraction warning attached.

The dense join needs its own gate, because `preds_{split}.csv` carries no id column: the
positional join to `split/{split}.csv` is admissible **only** because the `group` and
`judgement` sequences of the two files are asserted identical, row for row, in all three
seeds. Asserted in code, recorded in the ledger as `dense_alignment_gate`.

**Assembled-order gate.** The sharded matrices are independently re-assembled, the OOF
vectors are re-keyed by item id, the rows are randomly permuted, and every headline AUC
is recomputed — required to agree to < 1e-9. OOF arrays are saved **with their ids
vector** (`results/homepage_curation_storygrouped_oof.npz`: `ids`, `groups`, `y`,
`story_type`, `secondary_groups`, and every OOF column), never as a bare positional
`.npy`.

**Diagnostic A — applicability-mask ablation** (the press form, seven blocks):
`A_mask_only`, `A_levels_only_median_imputed`, `A_layer1_const05`, `A_mask_plus_levels`,
`V_only`, `V_plus_mask`, `V_plus_A_primary`, each fit linear + GBM seed 0, reported
pooled and same-rows, with `mask_alone` and `levels_worth_over_mask` broken out beside
the press values (.7322 / .0014) for direct comparison. Because the v2 prompt reserves NA
for empty input, the prediction is that the mask is near-degenerate — i.e. that the genre
channel is closed by construction. Whether it is closed is measured here, not assumed.

**Diagnostic B — story-type-stratified readout** (the within-story-type requirement,
measured). Story type is assigned by a **deterministic, label-blind keyword map over the
headline** — no judge, no y, so no circularity — into nine buckets plus `other`:

| type | n | | type | n |
|---|---:|---|---|---:|
| politics_govt | 2,758 | | sport | 504 |
| conflict_security | 2,080 | | health_science | 498 |
| crime_justice | 1,102 | | disaster_weather | 470 |
| business_econ | 672 | | culture_celebrity | 423 |
| lifestyle_service | 575 | | **other** | **3,916** |

The map is coarse on purpose — it exists to stratify, not to classify — and it was
**frozen before any A score existed**. Every readout is reported pooled *and* as a
size-weighted within-type average, plus `story_type_alone_auc` (how much of the placement
label story type buys on its own). Pooled minus stratified is the part of the readout
earned by separating story *types* rather than ranking headlines *within* one; per the
standing stratified-readout rule the stratified number is the honest one and pooled is
never the headline on its own.

---

## 5. T₀ (untrained-T) column

The T₀ arm is a **standing column** of the battery (registry scope amendment: "for all
cells", with homepage named explicitly). The 16-cell template file was frozen on
2026-08-08, before this cell's corrected dense arm existed, so this is a **post-hoc
addition under the same freeze discipline**: the template was written into
`fusion/t0_templates.json` **before any T₀ score existed for this cell** and is not
iterated afterwards.

* file sha256 **before** the addition: `50c1a5a98f8ff506033e1f1fe2ab5644b97c668ff45c86f995a9dafbfc18a080`
* file sha256 **after** (frozen for this cell): `5cc883e3f6bfd7f6ebde534aab4c9e5871f9bf2798aa539f7e15c417e1e88031`
* the pre-existing 16 templates and every frozen field are byte-unchanged; a
  `post_hoc_cell_additions` list records the addition, its reason, and the discipline.

Cell entry `homepage_curation_storygrouped`:

> **question:** "Will the following news headline be placed in the upper half of the most
> prominent zone of the news organisation's home page?"
> **document:** the dense arm's own row text, verbatim
> (`HEADLINE: … \n\nCONTEXT: …`) — byte-identical to what the trained dense T read.

Scoring reuses `fusion/t0_score_vllm.py` unchanged (base Llama-3.1-8B, zero-shot, no
LoRA, `P(Yes)` over the masked Yes/No variant set, document truncated to the trained T's
own 1024-token budget). Rows are built by a dedicated
`fusion/t0_build_rows_homepage.py` because the standing builder asserts against a
`vat_fullgrid_*` ledger this cell has never had; in its place the new builder asserts the
dense alignment gate, A-bank id containment for all 2,631 E rows, and agreement of the
recomputed T with `samerows_T_homepage_storygrouped.json`. Both deviations are recorded
in the emitted meta.

---

## 6. Artifacts

| what | path |
|---|---|
| verified dense T record | `methods/taste_decomposition/results/samerows_T_homepage_storygrouped.json` |
| rebuilt bank (29 criteria) | `datasets/news-homepages/va/rubrics_v2.jsonl` |
| A-bank scorer + per-criterion battery | `datasets/va_gemma_banks/score_homepage_v2_bank.py` |
| smoke gate (GEPA iteration checkpoint) | `methods/taste_decomposition/check_homepage_smoke_gate.py` |
| scored matrices | `outputs/va_gemma_banks_homepage_v2/homepage_curation_v2_shard{0..5}.npz`, `_meta.json` |
| anchor battery (row-mean, K=50) | `outputs/va_gemma_banks_homepage_v2/anchor_battery.json` |
| anchor battery (per criterion) | `outputs/va_gemma_banks_homepage_v2/anchor_battery_percriterion.json` |
| legacy 14-criterion triage | same two files, key `homepage_curation_legacy` |
| judge distribution check | `outputs/va_gemma_banks_homepage_v2/distribution_check.json` |
| layer-1 driver | `methods/taste_decomposition/homepage_v2_layer1.py` |
| **ledger** | `methods/taste_decomposition/results/homepage_curation_storygrouped_ledger.json` |
| OOF arrays (with ids) | `methods/taste_decomposition/results/homepage_curation_storygrouped_oof.npz` |
| T₀ template (frozen) | `methods/taste_decomposition/fusion/t0_templates.json` |
| T₀ rows builder | `methods/taste_decomposition/fusion/t0_build_rows_homepage.py` |
| T₀ rows / scores | `fusion/t0_rows/homepage_curation_storygrouped.*`, `fusion/t0_scores/homepage_curation_storygrouped.jsonl.gz` |
| run chain | `methods/dense/run_homepage_v2_chain.sh` |
| GPU claim/poll launcher | `scripts/tools/homepage_v2_gpu_claim_and_launch.sh` |
| logs | `logs/homepage_v2/{launcher.log,chain.log,stage*.log}` |

Scale: 12,998 × 29 = **376,942** judge prompts, plus 300 × 29 pilot, plus batteries
(2 × 150 × 29 for the v2 bank, 2 × 150 × 14 for the legacy triage).

---

## 7. Results

### 7a. Status at time of writing — built and validated, GPU-blocked

Everything that does not need a GPU is **done and verified**. The GPU stages are staged
and self-driving.

| check | result |
|---|---|
| dense T reproduces from raw predictions | **6/6 AUCs exact**, plus row counts, snapshot counts and pos-rates |
| `samerows_T_homepage_storygrouped.json` | written, `samerows_T_*` schema, both T values recorded and separated |
| bank builder loads (v2) | 12,998 items, 29 criteria, 1,229 snapshot groups; V dim 23, all finite, min std .023, 0 zero-variance columns |
| bank builder loads (legacy triage) | 12,998 × 14 through the ORIGINAL system prompt |
| longest assembled prompt | 4,591 chars ≈ 1.3K tokens (fits `max_model_len` 4096) |
| anchors construct | pos / neg / scrambled triples build on both banks; scramble verified as genuine word salad (word order broken *and* characters reversed) |
| census per-criterion coherence, from data already on disk | 4/14 below chance even biased toward passing; the two strongest columns score scrambled at 1.00 (§1) |
| story-type map | frozen before any A score; 9 typed buckets (423–2,758) + 30% `other` |
| **layer-1 driver, end-to-end** | synthetic-matrix dry run of the FULL driver: linear + GBM seeds {0,1,2} + seed spread + group bootstrap + 7-block mask ablation + story-type stratified readout + secondary grouping all run; **assembled-order gate PASS at max\|diff\| = 0.00e+00**; **dense alignment gate PASS**; same-rows blocks resolve to n=1,313 and n=2,631 with y identical between dense split and bank; `_oof.npz` written with ids/groups/y/story_type/secondary_groups + 10 OOF vectors |
| all four new Python files | compile clean under `envs/gemma4` |
| T₀ template | added and frozen (sha256 recorded both sides); 17 cells |

**Blocker: GPU contention.** The charge specified GPUs 0–3, claimed only after
`nvidia-smi` shows the card free. Snapshot during polling (used MiB of 183,359):

```
GPU0 77,168 (99%)   GPU1 77,168 (98%)   GPU2 77,168 (98%)   GPU3 77,168 (97%)
GPU4 115,316 (0%)   GPU5 111,280 (100%) GPU6 165,860 (99%)  GPU7 111,780 (100%)
```

GPU0–3 are the patents jobs, each holding a steady 77 GiB at ~98% util. GPU4 is another
user's parked `VLLM::EngineCore`. GPU5/6/7 are lanes A/B/C and are excluded by policy.

**The run is armed and will complete on its own** when a card frees: launcher
`scripts/tools/homepage_v2_gpu_claim_and_launch.sh` (PID logged in
`logs/homepage_v2/launcher.log`), strict mode (`ALLOW_STACK=0`), `ALLOW_GPUS=0,1,2,3`,
12 attempts over a ~24 h budget, `--auto-util` sizing at engine-init time. Stage 0/1/3
are shard- and sentinel-checkpointed and stage 4 skips a scored cell, so every retry
**resumes** rather than restarts. A smoke-gate failure exits 7 and deliberately does
*not* retry — that is a signal to revise the criteria, not to re-poll.

Note that each of GPUs 0–3 currently has ~106 GiB free, well above the 94 GiB the
Gemma-4-31B stage needs, so the box's usual `CLAIM-STACKED` pattern would start the run
now. It is **off by default here** because the charge asked for a free card; flipping
`ALLOW_STACK=1` on the launcher is a one-line, coordinator's-call unblock.

### 7b. Full-run progress

**Shard 0 certified.** 2,242 items × 29 = 65,018 prompts scored; blinded 3-row anchor
ordered **on attempt 0** — pos .897 > neg .810 > scram **.000**, `valid=True`; shard NA
rate **.000**; `homepage_curation_v2_shard0.npz` written.

Worth recording against caution 2 in §1c: the per-shard 3-row anchor ordered immediately,
even though the K=50 pilot had neg (.6597) above pos (.5934). A 3-row draw ordering is
weak evidence and a 50-row draw failing to is stronger — **the K=50 number is the one to
trust**, and the pos/neg question stays open until `A_lin`. The scrambled leg is
unambiguous at both sample sizes (.000).

Remaining: shards 1–5, then the K=50 batteries, distribution check, T₀ and layer 1.
Throughput on the shared card is ~45 prompts/s (~47 min/shard), so ≈ 4 h to the ledger —
slower than an idle card, as expected under contention.

### 7c. Headline numbers

<!--RESULTS-->

---

## 8. Deviations, and what they cost

1. **The A bank is new, not GEPA-optimised against a live scoring loop.** The campaign's
   A-bank standard is "GEPA-iterated criteria, judged label-blind by Gemma-4-31B". The
   Gemma-4-31B judge, the label-blind discipline, the K≥50 anchor battery, and the
   distribution check are all met exactly. The GEPA leg is met in the census bank's own
   reflective-revision form (per-criterion `gepa_revision` provenance) plus a **measured**
   pilot→gate→revise loop, rather than by running the GEPA optimiser: there is no
   label-free objective for a homepage-curation criterion that GEPA could optimise
   without turning the bank label-aware, which the reconstruction-only rule forbids. The
   coherence battery is the objective the loop does optimise, and it is label-blind.
2. **The story-type map is a keyword instrument, not a judge.** Coarse by construction —
   30% of items land in `other`. It exists to stratify, and a judge-assigned type would
   have re-introduced exactly the genre channel the diagnosis is testing for. Frozen
   before any A score existed.
3. **`t0_build_rows_homepage.py` skips the `vat_fullgrid_*` assertion** the standing
   builder makes, because this cell has no full-grid ledger entry. Three replacement
   assertions are listed in §5 and recorded in the emitted meta.
4. **The battery's all-NA row-drop is reported, not removed.** `C.run_battery` is imported
   verbatim so the v2 row-mean number is directly comparable with the .387 it replaces;
   the per-criterion battery reports the drops by tag beside it, so the statistic the old
   battery hid is now visible without changing the statistic itself.
5. **GPU discipline.** Lanes A/B/C hold GPUs 5/6/7 and the launcher excludes them by
   policy even when `nvidia-smi` shows them momentarily idle; GPU4 is excluded (another
   user's parked `VLLM::EngineCore`). Only GPUs 0–3 are candidates, claimed strictly at
   0 MiB / 0% util with no un-released ledger claim, claimed and released in
   `gpu_ledger.txt`, own PIDs only, co-tenants never touched.
