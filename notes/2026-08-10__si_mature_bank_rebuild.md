# Style Invitational — mature A-bank rebuild (v2)

Date: 2026-08-10. Charge: SI was promoted to **canonical humor-CURATION**
(user decision, registry 2026-08-10), but its standing verdict was
"**TERMINAL 2026-08-09 — TIE, bank = length model (0/32 rubrics survive length
strata); 'bank>dense' RETIRED**". Rebuild the A bank through the mature pipeline
(GEPA-phrased criteria, label-blind Gemma-4-31B, K≥50 anchors every batch,
collapse + distribution checks, token truncation, enforced collapse gate,
item-view consistency), design explicitly AGAINST the length failure, and report
length-stratified AUCs beside pooled for every criterion.

Status: **COMPLETE — and the rebuilt instrument FAILS.**

**Headline: the v2 bank does NOT certify and must not be used as an
instrument.** The K=50 anchor battery puts winner-vs-honorable-mention at
**AUC .483 sign-corrected / .509 raw — chance under both readings** — and
**0 of 33 criteria** clear |AUC−.5| ≥ .05 either pooled or inside length strata
(v1 managed 2 pooled). Worst of all, only **9 of 29** criteria point in their
intended direction, so the individual columns are noise and A_lin's .5647 is a
fitted composite over near-chance features, not a measurement.

**The cell is NOT declared terminal** (§7g): the 3-seed dense spread (.0351)
exceeds Δ_beyond (+.0076 eval / +.0195 test), so this design cannot resolve the
effect — the V8 underpowered signature — while T = .6241 against a .5520 length
baseline says the target is real. This is an instrument failure, not a cell
failure.

**Two findings survive and are independent of the failure** (§7h): the v1
verdict's comparison was unfair (16.3% parse artifacts; the V block it lost to
falls .6315 → .5511 once they are removed), and on clean data **A_lin .5647 now
beats V_lin .5587** with *less* length shrinkage — so "bank = length model" is
literally false, though for the deflationary reason that both sit near chance.

**The headline result does not depend on the missing pieces**: the terminal
verdict this rebuild was sent to revisit rested on a population that is
**16.3% parse artifacts** (§2), and on the cleaned population the entire
programmatic V block that beat the v1 bank falls from .6315 to **.5511** —
indistinguishable from raw character count, and .5206 inside length strata
(§7b-2). The comparison that produced "TIE, bank = length model" was not a fair
one. Whether the v2 bank clears the corrected bar is what the pending Layer-1
decides.

---

## 1. Inventory (never-repeat) — what already existed

| asset | state | used how |
|---|---|---|
| `style_invitational.jsonl` (9,637 rows, 316 weeks) | parsed 2026-07-28 from the nrars.org Book-of-Weeks archive | **reused as the source**, re-audited (§2) |
| `raw/01_text/` (332 archive text files) | present | not re-downloaded |
| `va/rubrics.jsonl` — the **v1 32-criterion bank** | frozen, scored | read as the design target to beat; **not reused** |
| `va/RESULTS_gemma.md` + `results_gemma.json` | v1 readout | the failure analysis (§2b) |
| `va/v_features.py` — 19 deterministic features | frozen | **100% reused** as the V block |
| `dense_standard/` — 3-seed T on 9,637 rows | complete | provenance verified (§4); **superseded**, see §4 |
| `closure/maps_hw_si/bank_survival.py` + `closure_core.py` | the v1 length-strata test | **stratification convention reused verbatim** so v1 and v2 are comparable |
| `build_dense_standard.py`, `parse_results.py` | present | recipe reused |

Nothing was re-scraped, re-parsed or re-downloaded. The only new compute is one
Gemma scoring pass and one 3-seed dense retrain.

## 2. THE FINDING THAT CHANGES THE VERDICT — the population was 16.3% parse artifacts

Before touching the bank, the label channel was ground-truthed (the V8 rule).
`parse_results.py` splits each week's archive text into entries heuristically,
and **1,574 of 9,637 rows carry no joke text at all**:

| artifact class | rows | example |
|---|---:|---|
| orphan byline | 1,111 | `"(Bob Zane, Woodbridge)"` |
| short list header | 226 | `"Seven deadly sins:"` |
| archive section marker | 133 | `"And last:"`, `"And Last,"` |
| cartoon / ink-blot selector | 87 | `"Cartoon B"` |
| truncated orphan | 17 | `"Takoma Park)"` |
| **total** | **1,574 (16.3%)** | mean 22 chars vs 110 for real entries |

**They are 11× concentrated in the negative class** — honorable_mention 19.1%,
runnerup 1.7%, winner 3.1% — because the parser drops a byline or a header into
its own row and the row inherits the surrounding HM tier.

The consequence is the whole story of this cell:

| readout | all 9,637 rows | fragments removed (8,063) |
|---|---|---|
| char length alone, pooled | **.6227** | **.5520** |
| char length alone, within-week | .6181 | **.5589** |

**Roughly 60% of the "length model" that killed the v1 bank was the model
detecting parse artifacts.** The v1 verdict — bank .613 losing to a length/format
block at .632 — was measured on a population where the strongest single feature
was partly "is this row a byline fragment".

The detector is deliberately conservative: shortness alone is never a fragment
signal, because genuine short entries exist (the headline contests produce real
entries like `ALIENS SIMONIZED MY CAR` and `CAPITALS WIN STANLEY CUP`, which the
detector correctly keeps). Only rows that are structurally a byline or a header
are removed. **Never delete data**: fragments stay in `va_v2/population.csv.gz`
with `is_fragment=True` and `fragment_class`, excluded from the analysis
population the same way V6's median-tied rows were.

Two other threats were checked and **ruled out**:
- **Byline format is not a leak.** Winners/runners-up are not more often
  parenthesised: the parenthesised-byline flag scores AUC **.4985** against y.
- **Byline length is not the length signal.** Stripping bylines removes ~20-24
  chars uniformly across tiers and moves length AUC only .6227 → .6258.

So on the clean population the length nuisance is real but weak (.5520 pooled /
.5589 within-week), which is what makes a content bank worth rebuilding.

### 2b. The v1 failure, restated precisely (the design target)

| v1 quantity | value |
|---|---|
| V_nl (length/format block) | .6315 |
| A_nl (32-criterion bank) | **.6131 — below V** |
| VA_nl | .6401 |
| T | .6490 |
| bank increment over joint V | +.0061, P(>0) = .80 |
| bank AUC **inside length strata** | **.5409** (T held .5889) |
| criteria with \|AUC−.5\| ≥ .05, pooled → within-V strata | 2 → **0** |
| median \|AUC−.5\|, pooled → within-V | .0120 → .0091 |
| near-constant criteria | 4 (Explanation discipline, Rhyme quality, Misdirection fairness, Phonetic/orthographic precision) |
| worst NA rates | Meter and scansion control **.966**, Verse form as comic leverage **.949** |

Three separate defects: the population (§2), elaboration-proxy criteria, and
degenerate criteria (4 near-constant, several ~95% NA because they were
conditional on the entry being verse).

## 3. The clean population

`datasets/humor/style_invitational/build_si_clean_population.py`

| | value |
|---|---|
| n (clean) | **8,063** (from 9,637) |
| weeks | 316 |
| y = top_tier (winner ∪ runnerup) pos rate | **.1883** |
| tier counts | winner 312 / runnerup 1,206 / HM 6,545 |
| week identity alone | .6595 |

Week-grouped stable-hash 80/10/10 via the patents bucketer, pos-rate matched:

| split | rows | weeks | pos rate |
|---|---|---|---|
| train | 6,456 | 236 | .18835 |
| eval | 810 | 39 | .18765 |
| test | 797 | 41 | .18821 |
| **train minority class** | **1,216** | | |

## 4. Dense T — provenance verified, then superseded

The existing `dense_standard/` arm is genuine: `style_inv_toptier`, n=9,637,
316 weeks, week-grouped 80/10/10, frozen dense-standard recipe, 3 seeds —
eval .6373 / .6137 / .6519 (mean **.6343**, spread **.0382**), test .6390 /
.6320 / .6623 (mean .6444).

It is **not usable as this rebuild's T**, for one decisive reason: it was
trained on the contaminated population, so ~16% of its training rows were parse
artifacts and part of what it learned is the fragment shortcut. Differencing a
clean-population VA against a fragment-trained T would not be a same-rows
comparison. A fresh 3-seed dense was therefore trained on the clean population
and split (`va_v2/dense_standard_si_clean/`), frozen recipe, no deviation.

**Note the seed spread on the old arm (.0382 eval) is large relative to
everything this cell measures** — the v1 bank increment was +.006. Any Δ here
must be read against the new arm's spread, reported in the ledger.

## 5. Item view and truncation

**Item-view consistency is exact and needs no sensitivity arm.** The population's
`text` column, the A judge's `ctx`, and the dense arm's training text are the
identical string: `CONTEST PROMPT: {prompt}\n\nENTRY: "{entry}"`.

**Truncation is in TOKENS** (ruling), applied with the judge's own tokenizer at a
1024-token cap. It is a guard only: the longest clean item is ~596 tokens and
the median is 60, so **truncation fires on zero rows** and nothing is silently
cut in either the judge or the 1024-token dense window.

## 6. The v2 bank — 36 criteria authored against the failure

`va_v2/build_rubrics.py` → `va_v2/rubrics.jsonl`. **32 Track A real criteria
(8 negatively-oriented) + 4 declared Track B surface probes.**

Design rules, each traceable to a specific v1 defect:

1. **Length-orthogonality is a requirement.** Every criterion carries a
   `length_orthogonality` field answering "could a writer raise this score by
   adding words?" with a reason. Ratio framings put length in the denominator
   on purpose ("every clause carries comic weight" — a longer entry has *more*
   clauses that must all pay). Positional framings are about arrangement, not
   amount. Obviousness framings are about the idea, fixed before a word is
   written.
2. **Negatively-oriented criteria are a deliberate eighth of the bank**
   (`orientation: "negative"`, 1.0 = the entry is WORSE). This is the
   length-*cancelling* family: post-punch trailing material, recycled prompt
   wording and self-signalling all grow with length while making an entry worse,
   so they enter the fitted stack with the opposite sign to the length channel.
   A bank of only positive criteria structurally cannot do this — which is why
   v1 could only ever ride length.
3. **Rank within published quality** (Wigleaf saturation lesson). Every row is a
   published entry; an HM already beat thousands of submissions. The
   obviousness-and-contestability family — would competent entrants converge on
   this joke? is the target the prompt's default? is the wordplay latent or
   forced? — exists because it is what still varies inside a publishable pool.
4. **No form-conditional criteria.** Nothing is conditional on the entry being
   verse (v1's verse criteria were 95-97% NA).
5. **The byline is not part of the joke.** Every positional or per-clause
   criterion states that the trailing `(Name, City)` is excluded — load-bearing
   for "punch word occupies the final position", since the byline sits last.
6. **Dual track.** 4 declared surface probes (parenthetical aside, exclamation
   mark, quoted material, all-caps word) are scored in the same matrix so
   A_real/A_surface split at readout without a re-score. **Raw length is
   deliberately NOT a probe** — V already carries `v_char_count`, and putting it
   in A would re-import the very confound the bank exists to escape.

### 6b. Smoke revision — validate-before-scaling, and the saturation lesson landing

The first 24-item smoke was **misleading**: `items[:24]` draws from a single
contest (the population is week-sorted), so every form-conditional criterion
looked falsely collapsed — one prompt has one register, one form, no wordplay.
The smoke was changed to spread across weeks; a 48-item / **48 distinct week**
smoke gave the usable diagnostic (NA .171 overall).

Five criteria were dead on this pool and were removed *before* the full run:

| dropped | modal | what the judge was telling us |
|---|---:|---|
| Premise had to be found | 1.00 | every contest premise is "found" |
| Commits to exactly one comic idea | .98 | an 80-char entry has one idea |
| Hedges its own premise | .98 | contest entries do not hedge |
| Explains its own joke | .96 | published entries do not self-gloss |
| Carries a second competing joke | .94 | no room for two jokes |

**This is the Wigleaf saturation lesson landing exactly where predicted, with a
refinement worth keeping**: it is *flaw-detection* that saturates on a published
pool, because those flaws were edited out before publication. Negative
orientation is not the problem — the negative criteria that survive are the ones
naming flaws published work still has (**stock joke template**, modal .50;
**era's stock referent**, modal .25). Four replacements were added in the
families the smoke showed actually split: prompt-portability
("could serve a different prompt unchanged"), obviousness ("joke is available
from the prompt alone"), inferential distance ("comic leap takes more than one
step") and found-vs-manufactured ("rests on a real-world coincidence").

## 6c. A protocol bug this bank exposed — the anchor test is invalid for a mixed-orientation bank

Shard 0's blinded 3-row anchor needed **4 draws** and passed by a margin of
**.002** (pos .517 / neg .515 / scram .269). That looks like the judge failing.
It is not — it is the shared protocol mis-measuring this bank.

`score_va_gemma_banks.score_bank` and `score_scaleupC_banks.run_battery` certify
an anchor row by the **unweighted mean of its scores across every criterion**,
then require pos > neg > scrambled. That is correct when all criteria point the
same way. This bank deliberately contains **8 negatively-oriented criteria where
1.0 marks a FLAW**, so a better entry is supposed to score LOW on them — and
averaging them in unsigned cancels the very contrast being tested.

`si_v2_anchor_recheck.py` recomputes the anchor statistics as a **quality mean**:
negative criteria enter as `1 − value`, Track B surface probes are excluded
(declared nuisance carries no quality direction). Same scores, no new judge
calls — temperature-0 outputs cannot change, so this is a re-read of evidence
already collected, not a re-run.

| shard 0 anchors | pos | neg | scram | **pos − neg** | ordering |
|---|---|---|---|---|---|
| raw (shipped protocol) | .517 | .515 | .269 | **+.0015** | PASS by .002 |
| **sign-corrected** | **.788** | **.724** | **.545** | **+.0643** | **PASS** |

The corrected margin is **43× larger**. Two consequences:

1. **Any per-shard anchor FAIL on this bank must be re-read corrected before it
   is believed** — including retries, which cost judge time chasing a
   mis-specified statistic. The retry loop fired 4 times on shard 0 for nothing.
2. **This generalises beyond SI.** Every future bank containing negatively-
   oriented criteria inherits the bug, and negative orientation is exactly what
   the program asks for when a cell needs length-cancelling criteria. The fix
   belongs upstream in `score_va_gemma_banks`, not in this cell.

One expected side effect of the correction, recorded so it is not misread:
scrambled text rises from .269 to .545, because word salad has no content and
therefore also has *no flaws*, collecting credit on every flipped negative
criterion. Coherent entries still separate from it clearly (.72–.79 vs .545),
but the coherent-vs-scrambled contrast is necessarily compressed on a
mixed-orientation bank, and the corrected number is the one to quote.

## 7. What landed, and the ledger

### 7a. Scoring pass — COMPLETE

All 7 shards of the clean 8,063-row population × 36 criteria scored with
Gemma-4-31B (offline-batch vLLM, temperature 0, one token per (item, criterion)).

| shard | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| NA rate | .181 | .183 | .180 | .183 | .185 | .182 | .182 |

**Overall NA ≈ .182, and it is uniform across shards** — a marked improvement on
v1, whose bank carried four near-constant criteria and two at 95-97% NA. Nothing
in the v2 bank is form-conditional, which is where that NA came from.

### 7b. Anchor certification — the raw protocol FAILS, the corrected one passes

The K=50 battery under the **shipped (unsigned) protocol**:

| | pos | neg | scram | pos-vs-neg AUC | coherent-vs-scram |
|---|---|---|---|---|---|
| raw K=50 | **.5138** | **.5231** | .2799 | **.509** | .938 |

**pos < neg — the ordering fails outright, and pos-vs-neg is at chance.** Taken
at face value this says the judge cannot tell a winner from an honorable
mention. It is instead the mixed-orientation bug of §6c at full scale: averaging
8 flip-signed criteria in unsigned cancels the contrast. The per-shard evidence
already showed the size of the distortion — shard 0's corrected margin is 43×
the raw one (+.0643 vs +.0015), while coherent-vs-scrambled stays strong (.938)
because scrambling degrades every criterion regardless of orientation.

The sign-corrected K=50 battery is the certification of record for this bank and
is the first thing the resume script runs. **Until it lands, this bank is scored
but NOT certified**, and no readout from it should be quoted as final.

### 7b-2. The V block on the clean population — the bar just dropped a long way

Computed locally with the frozen Layer-1 estimators (no judge, no sk3, so this
leg is final): `va_v2/v_only_clean.json`.

| quantity | clean population | v1's contaminated figure |
|---|---|---|
| **V_lin** (19 deterministic features) | **.5511** | .6227 |
| V_nl (mean GBM seeds 0/1/2, spread .0025) | **.5836** | .6315 |
| char length ALONE | .5520 | .6227 |
| **V_lin inside length deciles** | **.5206** | — |

Two things follow, and they reframe the whole cell.

1. **The V block is length, and essentially nothing else.** V_lin (.5511) is
   indistinguishable from raw character count (.5520), and holding length fixed
   collapses it to .5206. All 19 features together add ~nothing over one number.
2. **The bar the bank has to clear fell from .6315 to .5511.** v1's bank (.613)
   was judged against a length/format block that the parse artifacts had
   inflated by roughly .08 AUC. "The bank loses to length" was in large part
   "the bank loses to a fragment detector". On the clean population a bank at
   v1's own .613 would have *beaten* V by a comfortable margin.

This is now the sharpest statement of why the terminal verdict had to be
revisited, and it is established without the pending pieces: **the comparison
that produced "TIE, bank = length model" was not a fair one.** Whether the v2
bank clears the corrected bar is what the pending Layer-1 decides — this result
does not pre-empt it, it only shows the previous verdict rested on an inflated
baseline.

### 7c. CERTIFICATION VERDICT — **FAILED**

The K=50 battery, under both readings:

| reading | pos | neg | scram | **pos-vs-neg AUC** | coherent-vs-scram | ordering |
|---|---|---|---|---|---|---|
| **sign-corrected** (certification of record) | .7424 | .7572 | .5408 | **.483** | .884 | **FAIL** |
| raw (shipped protocol) | .5138 | .5231 | .2799 | **.509** | .938 | **FAIL** |

**The bank does not separate winners from honorable mentions.** With 50 per
class the standard error on that AUC is ≈.058, so .483 is *at chance*, not
inverted — the honest statement is "no signal", not "backwards".

The sign correction of §6c was a genuine bug fix and it does **not** rescue the
result: corrected and raw agree that pos-vs-neg is at chance. My earlier
shard-0-only reading (+.0643 corrected vs +.0015 raw) was a single pair and
should not have been leaned on; across all seven shards the corrected ordering
holds only **3/7** (raw held 7/7, but on margins as thin as .0015 — that 7/7 was
tiny-margin luck, not signal). The correction remains the right protocol; it
simply reveals the same failure more honestly.

**What IS certified: coherent-vs-scrambled .884.** The judge is genuinely
reading content — word salad separates cleanly from real entries. So the
instrument distinguishes text from non-text, and fails only at the fine
within-published distinction the cell actually requires.

### 7d. The ledger

n = 8,063 · pos .1883 · 316 weeks · V = 19c · A = 33c (3 dropped by the
enforced collapse gate: *Depends on contest in-group knowledge* .9823,
*Phonetic distance is tight* .9896, *Sustains one register throughout* .9906).

| quantity | v2 (clean) | v1 (contaminated) |
|---|---|---|
| V_lin / V_nl | .5587 / .5960 | .6227 / .6315 |
| **A_lin** | **.5647** | .6090 |
| VA_lin / VA_nl | .5675 / **.6011** | .6161 / .6401 |
| Δ_interact | **+.0336** [+.0115, +.0521] P(>0)=1.00 | — |
| **T** (3-seed, eval) | **.6241** (seeds .6076/.6427/.6219, spread **.0351**) | .6343 (spread .0382) |
| Δ_beyond, pooled convention | +.0230 | — |

Same-rows Δ_beyond, VA restricted to exactly the dense split's rows:

| leg | n / pos | VA_nl | T | **Δ_beyond** | T seed spread |
|---|---|---|---|---|---|
| eval | 810 / 152 | .6165 | .6241 | **+.0076** | .0351 |
| test | 797 / 150 | .6042 | .6237 | **+.0195** | .0303 |

OOF reproduction passes exactly (diff 0.00e+00); ids-carried OOF shipped.

### 7e. THE ACCEPTANCE TEST — **FAILED, and worse than v1**

| | v2 | v1 |
|---|---|---|
| criteria with \|AUC−.5\| ≥ .05, **pooled** | **0 of 33** | 2 of 32 |
| …**within length strata** | **0** | — |
| …within joint-V strata | **0** | 0 |
| median \|AUC−.5\| pooled → within-length → within-V | **.0065 → .0065 → .0046** | .0120 → — → .0091 |

The strongest criterion in the entire bank is *Comic leap takes more than one
step* at **.528**. Nothing else clears .53. The v2 criteria are individually
**weaker than v1's** (median |AUC−.5| .0065 against .0120).

Note one thing the table does get right: the criteria that do move barely shrink
under length stratification (.528 → .514, .526 → .516). They are not length
proxies. They are simply near-chance to begin with, so there is nothing for the
stratification to remove. **Designing away the length confound succeeded;
designing in signal did not.**

### 7f. Direction validity — the decisive result

Every criterion was authored with an intended direction (positive: 1.0 = better
entry; negative: 1.0 = flaw). If the bank measured what it claims, observed AUCs
should land on the intended side of .5.

**Only 9 of 29 directional criteria point the intended way** — 5 of the negative
family and 15 of the positive family point the *wrong* way, including
*Punch word occupies the final position*, *Wordplay is latent, not forced*,
*Every clause carries comic weight* and *Leans on a stock joke template*.

9/29 = 31%, i.e. **worse than a coin flip**. Combined with no criterion
exceeding |AUC−.5| = .028, the conclusion is unavoidable and is the honest
headline of this rebuild:

> **The individual criteria carry no reliable signal, and their apparent
> directions are noise.** A_lin's .5647 is a fitted composite extracting weak
> structure from 33 near-chance columns, not an aggregation of measured
> properties. The instrument does not measure what it says it measures.

### 7g. Verdict — instrument FAILS; the cell is NOT declared terminal

§14 pre-kill checklist, all five recorded, because a negative result here is
exactly where the checklist binds:

| item | value |
|---|---|
| (1) absolute minority-class count in train | **1,216** — not label-starved |
| (2) simple baseline, same split | char length .5520, V_lin .5587, **T .6241** — the TASK carries signal well above chance, so this is an **instrument** failure, not a cell failure |
| (3) historic working runs | v1 A_nl .6131, but on the contaminated population where fragments inflated everything; on clean data v1's bank is untested |
| (4) which design failed | 36 GEPA-phrased criteria, Gemma-4-31B, clean 8,063-row week-grouped population, K=50 winner-vs-HM battery |
| (5) seed spread vs claimed effect | dense spread **.0351 eval / .0303 test**, both **LARGER than Δ_beyond** (+.0076 / +.0195) — **underpowered**, the V8 signature |

**Therefore: the v2 bank FAILS certification and FAILS the acceptance test, and
must not be used as a measurement instrument. But the CELL is not terminal** —
item (5) says this design cannot resolve the effect it is being asked about, and
item (2) says the target is real (T .6241 against a .5520 length baseline).
Something in these entries is learnable; 36 articulated criteria did not capture
it.

### 7h. What the rebuild nonetheless establishes

1. **The v1 verdict's comparison was unfair, and that part stands.** The
   population was 16.3% parse artifacts and the V block it lost to falls from
   .6315 to .5511 once they are removed.
2. **"Bank = length model" is now false in its literal sense — but for a
   deflationary reason.** On clean data **A_lin (.5647) BEATS V_lin (.5587)**,
   and A shrinks *less* under length stratification than V does (A .5647→.5461,
   shrinkage .019; V .5587→.5328, shrinkage .026). The bank is less
   length-borne than the programmatic block. That reads like a win until you
   note both sit near chance: the ordering reversed because V collapsed, not
   because A rose.
3. **The dual track behaved.** The 4 declared surface probes score .5116 as a
   block (.5052 within length strata), so A_real carries essentially all of A —
   the real/surface separation is clean even though the real side is weak.
4. **Δ_interact is real and positive** (+.0336, P(>0)=1.00): whatever weak
   structure exists is nonlinear, and VA_nl (.6011) sits much closer to T
   (.6241) than any linear block does.
5. **The saturation diagnosis is confirmed twice over** — once in the smoke
   (five flaw-criteria dead because published work has no such flaws), once in
   the battery (winner-vs-HM at chance while coherent-vs-scrambled is .884). The
   judge can tell writing from noise; it cannot tell a winning joke from a
   published one.

### 7i. Recommendation

Do **not** re-mine a third bank of the same shape — that is the move this result
argues against. Two things are worth trying instead, in order:

1. **Change the readout, not the bank: pairwise within-week.** Every failure
   here is an *item-level* absolute-scoring failure. The editorial act is
   comparative (one winner chosen from ~25 entries in one contest), and the
   judge is being asked to put an absolute score on a single entry with no
   comparison class. A within-week pairwise judge (A vs B, which is funnier)
   matches the construct and removes the absolute-calibration burden that the
   .7424-vs-.7572 battery shows the judge cannot carry. The population is
   already week-grouped and the pairwise design is cheap.
2. **Check the judge, not the criteria.** Gemma-4-31B may simply lack the
   discrimination for published-vs-prizewinning humor. A small
   frontier-judge probe on a few hundred within-week pairs would separate
   "criteria are wrong" from "this judge cannot see it" — currently confounded,
   and cheap to resolve before any further bank work.

## 8. Artifacts

| what | where |
|---|---|
| contamination audit + clean population + dense bundle | `datasets/humor/style_invitational/build_si_clean_population.py` |
| population (fragments retained + flagged) | `datasets/humor/style_invitational/va_v2/population.csv.gz` |
| audit manifest (fragment classes, length nuisance before/after, splits) | `va_v2/population_manifest.json` |
| **v2 bank + every design decision** | `va_v2/build_rubrics.py` → `va_v2/rubrics.jsonl` |
| Gemma scoring driver | `datasets/humor/style_invitational/score_si_v2_bank.py` |
| scored matrix + K=50 battery | `outputs/va_gemma_banks_si_v2/` (sk3) |
| clean 3-seed dense (T) | `va_v2/dense_standard_si_clean/` |
| Layer-1 + length-survival test | `methods/taste_decomposition/si_v2_layer1.py` |
| **ledger** | `methods/taste_decomposition/results/si_v2_ledger.json` |
| ids-carried OOF | `methods/taste_decomposition/results/si_v2_oof_with_ids.npz` |
| v1 bank + its readout (retained, not deleted) | `datasets/humor/style_invitational/va/` |

## 9. Resume state (sk3 outage 2026-08-11 ~00:30)

The box began resetting SSH connections at the jump host during key exchange —
before login, so this is neither the AFS `chdir` mode nor the root-disk-full
mode in the reference notes. Detached work was unaffected: the dense wrapper was
launched `setsid --fork` with PPID 1 and continued through the outage.

`scratchpad/si_resume.sh` is armed and idempotent. On reconnect it:
1. snapshots shard and dense-seed counts;
2. runs the sign-corrected battery **only if** its output file is absent,
   choosing a GPU with <20 GB in use from the live ledger;
3. waits for dense 3/3;
4. runs Layer-1 and appends the full ledger to `scratchpad/si_resume.log`.

Every stage is checkpointed, so nothing is recomputed: the 7 scored shards are
on disk and `score_bank` skips existing shards, and the dense wrapper skips any
seed with a `RUN_DONE` sentinel.

## 10. Discipline check

No data deleted (fragments and the v1 bank both retained);
`latex/` untouched; anchors on every judging batch plus the K=50 battery;
stable-hash grouped splits, no seeded shuffle; pos-rate matched; all long jobs
`setsid --fork` with PPID 1 verified; GPUs taken from the ledger after
confirming they were free (dense GPU 5, judge GPU 0, smoke GPU 6).
