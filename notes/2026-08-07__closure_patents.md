# Layer-3 articulation closure — patents claim-fell cell: ROUND-0 AUDIT, campaign STOPPED

Date: 2026-08-07. Protocol: `notes/2026-08-05__layer3-closure-prereg.md`
**FREEZE DECLARATION** + **FREEZE ADDENDUM** + **FREEZE ADDENDUM 2** (upstream mode) +
**FREEZE ADDENDUM 3** (mixed-channel decomposition) — all binding.
Dense arm under audit: `notes/2026-08-06__dense-arms-hw-si-patents.md`.
Worked campaign example this was to replicate: `notes/2026-08-06__closure_cw_community.md`.
Artifacts: `methods/taste_decomposition/closure/patents/` (all provenance-tagged).

Terminology, spelled out on first mention per the standing rule:
**V** = the 7 deterministic lexical/structural features of `notebooks/data/patents_va_features.csv`;
**A** = the 4 Gemma-4-31B disclosure-aggregate columns of the same file;
**VA_nl** = HistGradientBoosting aggregation of the V+A matrix;
**T** = the dense readout (Llama-3.1-8B LoRA on raw text, `datasets/patents/dense_standard/`);
**Δ_beyond** = T − VA_nl, the unarticulated residual;
**ε** = .005, the per-round saturation threshold; **k_A / k_B** = 15 / 10 criteria per round;
**AUC** = area under the ROC curve; **P** = proposers in a sealed fleet round;
**FIT+MINE / MONITOR / TEST** = the closure splits; **STRUCT** = the structural
metadata block defined in §3 below; **CPC / USPC** = the international and US patent
classification schemes; **§102 / §103 / §112 / §101** = the statutory grounds for a
patent rejection (anticipation / obviousness / definiteness–enablement / patent-eligible
subject matter); **MPEP** = the USPTO's Manual of Patent Examining Procedure.

---

## VERDICT, up front

**The campaign is STOPPED at round 0 and converted to a leak post-mortem, per the task's
own stop rule.** No proposer fleet was seated; no Gemma scoring round was run; no GPU
was spent on mining.

The dense number itself is sound as an AUC. What does not survive is its
*interpretation* as an articulation residual:

> **Of the provisional Δ_beyond = +.171 — the largest residual standing in the program
> once CW community's round-0 audit revised its +.176 down to +.141 — a single non-content
> metadata column, the claim's ordinal number in its own application, claims 85% on the
> split that defines T (69% on TEST). A structural block containing no text content at
> all claims 89% / 76%. The residual beyond V + A + structure is +.020 [.001, .038] on
> EVAL and +.047 [.029, .067] on TEST.**

and, separately and more seriously for the cell:

> **The label is not a prior-art-reading label, and the dense model is not reading the
> prior art.** 17–31% of positives are §112 / §101 / double-patenting rejections, for
> which the eight retrieved prior-art references are irrelevant by construction — and
> the dense model separates *those* positives from negatives as well as it separates
> §102/§103 positives (TEST: .8345 vs .8398; §101 alone .9496, the highest of any
> family). 43–45% of positives have **zero** disclosing references and still receive a
> mean predicted probability of .72–.73 against .32–.36 on negatives. And in a text
> ablation, **deleting the claim element hurts less than deleting the references**
> (.643 vs .599) — entailment is impossible without the claim, so the half of the input
> that permits entailment contributes less on its own than the half that forbids it.

The named stop gate ("class/era alone ≥ .75") **does not fire** — every classification
and era channel sits at .52–.55. The gate that fires is the same gate's spirit applied
to the channel that actually exists here.

---

## 0. What was run, and what was not

| round-0 item (from the brief) | status |
|---|---|
| 1. dense seeds 1–2 → 3-seed T with spread | **LAUNCHED**, in flight on GPU 3 (see §7) |
| 2. score audit: distribution / per-app_id / per-class / per-era / length / manual read | **DONE**, §1–§6 |
| 3. concept census of the A bank (dedup per freeze) | **DONE**, §5 |
| 4. audited round-0 verdict before mining | **DONE** — STOP |
| rounds 1–5 (sealed fleet, k_A=15 / k_B=10, audit, probes, arbiter, Gemma scoring) | **NOT RUN** (stop rule) |

### Why running the rounds anyway would not have changed the answer

The tempting counter-argument is that a one-round campaign closing 85% of the largest
residual in the program would be a spectacular closure result. It would not, for a
reason internal to the protocol:

**the channel that closes the gap is one the blind routing audit would send to Track B,
not Track A.** "How far down the claim set this element sits" is predictive and has
nothing to do with whether the claim deserved to fall — it is the same shape as the
planted probes the CW-community audit correctly routed to the nuisance side in five
rounds out of five. So a faithful run of the frozen protocol would have produced a
**near-null Track-A curve** (Δ_r ≈ 0, two sub-ε rounds, "saturation" declared) sitting
next to a **Track-B map that absorbed the entire residual** — and the honest headline
would be exactly what round 0 already establishes: *the patents residual is a spurious
channel*. Five rounds, a sealed P = 4 fleet and roughly 750,000 Gemma-4-31B judge calls
would have bought a more expensive statement of the same finding, on a cell whose label
does not measure the construct (§4) and whose incumbent bank is a single criterion (§5).

The dual-track design is what makes this stop legible rather than a failure: the user's
framing for this cell was *"run the metric-discovery program here, both directions:
spurious as well as real"*, and the spurious direction returned a decisive answer before
the real direction needed to be paid for.

**The stop is not a feasibility stop.** Unlike CW community — which needed a Stage-0
population extension because only 408 of its 2,000 A/V rows were dense-held-out — this
cell was ready to run. The A/V population *is* the dense population (59,937 rows,
same-rows by construction per the build manifest's
`assertion_rows_subseteq_layer1_population`), and 11,988 rows (eval + test) are
dense-held-out, which is 10× the MONITOR size CW community ran on. FIT+MINE / MONITOR /
TEST would have been cut inside those 11,988 rows with room to spare. The campaign
stopped on evidence, not on instrument availability.

**Fleet floor, recorded as the freeze requires.** GLM-5.2 is quota-dead until 2026-08-13
(`1310` weekly/monthly exhaustion), so the round-1 fleet would have been **P = 4 across
2 families** (Claude ×2, gpt-5.6-luna ×2 via the Codex companion) — the freeze's degraded
floor. This is recorded for completeness; it is not the reason the campaign stopped.

**Compute discipline.** One GPU claimed (GPU 3, free at claim time: 0 MiB / 0% util,
logged to `gpu_ledger.txt` at 2026-08-07T19:46Z as
`cell=patents_claimfell … job=dense_seeds_1_2 | CLAIM`). Co-tenant GPUs 0/1/2/4/5/6/7
untouched. The ablation pass (§4) is stacked on the same claimed card. Nothing killed.
`latex/` untouched.

---

## 1. Instrument checks on the dense arm — all pass

| check | result |
|---|---|
| independent re-score of the EVAL split reproduces the filed predictions | **.7965 vs .7965 — identical** |
| collapse gate | **no collapse**: 1,352 distinct probabilities, range [.0002, .9983], bimodal |
| prediction distribution (EVAL) | mean .585, sd .363; 12.6% below .05, 20.2% above .95 |
| calibration (EVAL) | monotone reliability curve, **ECE .115**, Brier .193 — *over-confident*, no pathology |
| calibration (TEST) | ECE .084, Brier .166 |
| cross-split duplicate contamination, full text | **0 hash groups spanning splits** |
| cross-split duplicate contamination, claim element | 4 groups / 8 rows (.01% of corpus), 0 with contradictory labels |
| split integrity | app_id-grouped; train/eval/test pos-rates .60143 / .60143 / .60143 |
| row alignment jsonl ↔ split CSVs ↔ prediction files | 0 unmatched, 0 label mismatches (asserted in code) |

**One honest caveat on the level.** The checkpoint was selected on the EVAL split
(`--selection_split eval`) from two epoch-end candidates (.7820, .7966), so T = .7965
carries a max-of-two selection bias. TEST (.8389) is selection-free. Both are reported
throughout; the recipe's canonical T is clean-eval, so the EVAL column governs the
protocol number.

Reliability curve, EVAL (decile bins, mean predicted vs observed):

| bin | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean predicted | .012 | .074 | .208 | .412 | .613 | .771 | .871 | .931 | .965 | .986 |
| observed rate | .128 | .277 | .427 | .534 | .585 | .677 | .758 | .835 | .870 | .919 |

---

## 2. The named gate: class and era channels are CLEAN

All fit on the dense TRAIN split (47,949 rows), scored on the dense EVAL split
(5,994 rows) — apples-to-apples with T. Technology class and filing date join at
**100% coverage** of the 21,447 app_ids (CPC from PatentsView `pg_cpc_current`,
`cpc_type == "inventional"`, lowest `cpc_sequence`; USPC class, subclass and examiner art
unit from the PatEx `application_data` table; exact filing year from `labels.parquet`).

| channel | levels | alone-AUC (EVAL) |
|---|---:|---:|
| CPC section | 8 | **.5479** |
| CPC class | 121 | .5275 |
| CPC subclass | 537 | .5408 |
| USPC class | 386 | .5431 |
| examiner art unit | ~700 | .5348 |
| examiner tech centre (art-unit prefix) | 37 | .5203 |
| **filing year** (exact, 2010–2017) | 8 | **.5316** |
| filing year, numeric | — | .5324 |
| application series code (era proxy) | 4 | .5327 |
| CPC subclass × filing year | — | .5390 |
| examiner art unit × filing year | — | .5299 |

**Nothing reaches .60, let alone .75. The stop gate as literally written does not fire.**

The dense AUC also survives stratification on every one of these, which is the same
statement from the other side (n-weighted mean of within-stratum AUCs on EVAL):

| stratified by | CPC section | CPC subclass | filing year | tech centre | app_id | rejection_type |
|---|---:|---:|---:|---:|---:|---:|
| dense within-stratum AUC | .7954 | .8096 | .7953 | .7973 | .7925 | .7635 |

---

## 3. The gate that DOES fire: claim ordinal number

`claim_num` — the claim's position in its own application's claim set, a pure
non-content metadata integer — reaches **alone-AUC .7537 on EVAL and .7513 on TEST.**

Positive rate by claim number (population, n ≥ 200 per cell):

| claim # | 1 | 2 | 3 | 5 | 8 | 10 | 13 | 16 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P(fell) | **.946** | .825 | .768 | .712 | .641 | .603 | .528 | .454 | .391 | .310 | **.301** |
| n | 4,979 | 3,864 | 3,666 | 3,298 | 2,778 | 2,616 | 2,223 | 1,849 | 1,280 | 465 | 239 |

Mean claim number: **8.53 on positives vs 16.43 on negatives**. Independent claims fall
at .747, dependent claims at .562. Spearman(dense probability, claim number) =
**−.484 (EVAL) / −.416 (TEST)** — four times the model's correlation with the actual
disclosure evidence (+.103 / +.127 against `n_disclose`).

**The channel is not even inferred — it is printed in the text.** 82.2% of EVAL claim
elements state a claim number within their first 200 characters, almost always as the
opening dependency reference ("*The device of claim 42 wherein…*"); 77.2% match the
stricter dependent-claim pattern. Parsing that verbatim integer out and using
nothing else gives **AUC .7252 (EVAL) / .7418 (TEST)**; among dependent rows the dense
score correlates **−.564** with it.

### The ladder (all fit on the dense TRAIN split, read on EVAL and TEST)

| model | cols | EVAL | TEST |
|---|---:|---:|---:|
| V (lexical/structural) | 7 | .5925 | .6265 |
| A (disclosure aggregates) | 4 | .5687 | .5622 |
| **VA** | 11 | **.6214** | **.6434** |
| **claim_num alone** | **1** | **.7537** | **.7513** |
| STRUCT minus claim_num (dependency flag + 6 length features) | 7 | .6041 | .6530 |
| **STRUCT** (claim_num + dependency flag + lengths, no text content) | 8 | **.7626** | **.7731** |
| VA + claim_num | 12 | .7706 | .7781 |
| VA + STRUCT | 19 | .7768 | .7916 |
| **T (dense, seed 42)** | — | **.7965** | **.8389** |

### What fraction of the +.17 gap each explanation claims

| explanation | EVAL (gap = +.1751) | TEST (gap = +.1955) |
|---|---:|---:|
| **claim ordinal number, one integer column** | +.1492 → **85.2%** | +.1347 → **68.9%** |
| rest of STRUCT (dependency flag + 6 length features), marginal | +.0062 → 3.5% | +.0135 → 6.9% |
| **unexplained beyond V + A + STRUCT** | **+.0197 → 11.3%** | **+.0473 → 24.2%** |

Group-level paired bootstrap (1,000 draws, resampling app_id):
T − VA = **+.1751 [.1512, .1981]** (EVAL), **+.1955 [.1736, .2186]** (TEST);
T − (VA + STRUCT) = **+.0197 [.0010, .0380]** (EVAL), **+.0473 [.0293, .0666]** (TEST).
The residual is positive but an order of magnitude smaller than the headline, and on
EVAL its lower bound is .001.

### The prereg's own discount readouts, applied to STRUCT at round 0

| readout | EVAL: T | VA | Δ | TEST: T | VA | Δ |
|---|---:|---:|---:|---:|---:|---:|
| undiscounted | .7965 | .6214 | **+.1751** | .8389 | .6434 | **+.1955** |
| decile-stratified on the joint STRUCT score | .7200 | .5750 | +.1450 | .7829 | .5723 | +.2106 |
| matched sampling on the joint STRUCT score | .7071 | .5541 | +.1530 | .7761 | .5620 | +.2141 |
| dense matched on claim number alone (exact strata) | .7312 | — | — | .8009 | — | — |

**Read the stratified rows with care — they are the wrong instrument for this finding.**
Conditioning on STRUCT costs the *bank* proportionally more than the dense model
(VA .6214 → .5541) because V already contains length features; the discount therefore
leaves Δ_adj large. The question round 0 actually needs answering is not "what survives
conditioning" but "what survives **banking** the channel", and that is the ladder:
VA + STRUCT = .7768, residual +.0197. Both are reported; neither is cherry-picked.

**Stacked increment (FREEZE ADDENDUM, stack weights fit on the opposite split so no
increment is read in-sample):**

| base | base AUC (EVAL) | stacked with dense | increment | base (TEST) | stacked | increment |
|---|---:|---:|---:|---:|---:|---:|
| STRUCT | .7626 | .8176 | **+.0550** | .7731 | .8566 | **+.0835** |
| VA | .6214 | .7969 | +.1755 | .6434 | .8350 | +.1916 |
| VA + STRUCT | .7768 | .8215 | **+.0447** | .7916 | .8586 | **+.0670** |

The stratification-free control agrees with the ladder: over *all* named channels the
dense increment is +.045 / +.067, not +.175 / +.196.

---

## 4. The y-definition audit: this is not a prior-art-reading label

`y` = 1 iff the claim element was rejected under **some** statutory ground. The text the
dense reader receives is eight candidate **prior-art** references. Only §102
(anticipation) and §103 (obviousness) are prior-art grounds. A §112 rejection concerns
definiteness/enablement, a §101 rejection concerns patent-eligible subject matter, and a
double-patenting rejection is over the applicant's **own** earlier claims — for all of
those the eight retrieved references are irrelevant to the label *by construction*.

| positives restricted to | EVAL n_pos | EVAL dense AUC | TEST n_pos | TEST dense AUC |
|---|---:|---:|---:|---:|
| §102 / §103 (prior-art grounds) | 2,520 | .8135 | 3,012 | **.8398** |
| **§112 / §101 / double-patenting / other (NOT prior-art)** | 1,140 | .7591 | 606 | **.8345** |
| §102 only | 1,187 | .8230 | 743 | .8488 |
| §103 only | 1,333 | .8048 | 2,269 | .8369 |
| §112 only | 382 | .7678 | 111 | .8447 |
| **§101 only** (subject-matter eligibility) | 130 | .8069 | 49 | **.9496** |
| double-patenting only | 383 | .7471 | 361 | .8045 |

On TEST the model does **as well on rejections the references cannot possibly explain
(.8345) as on the ones they can (.8398)**, and its single best family is §101 — the
ground furthest from anything in the text it is given.

**Not cherry-picked: EVAL is weaker but still damning.** On EVAL the non-prior-art
families sit .054 below the prior-art ones (.7591 vs .8135) rather than at parity. A
model doing entailment would not be at **.759** on rejections whose grounds the text
cannot address — it would be near chance. A .054 gap is the size of a nuisance-channel
difference (the non-prior-art positives have a higher mean claim number, 12.05 vs 9.40),
not the size of an entailment channel.

Three further readouts point the same way:

| readout | EVAL | TEST |
|---|---:|---:|
| positives with **zero** disclosing references: n (share of positives) | 1,545 (42.9%) | 1,633 (45.3%) |
| …their mean dense probability | **.7211** | **.7316** |
| …mean dense probability on negatives | .3612 | .3157 |
| …their AUC against all negatives | .7891 | .8270 |
| among positives only, dense score predicting whether the examiner's own gold reference discloses | **.5506** | **.5548** |
| `n_disclose` alone-AUC | .5692 | .5537 |
| `gold_disclose` alone-AUC | .6589 | .6521 |

A model doing claim-to-reference entailment would be near chance on rows where nothing
discloses, and would rank the gold-disclosure question well above .55. This one does the
opposite.

### Manual read of high-confidence rows (16 dumped, `round0_manual_read.json`)

The pattern is legible without statistics.

- **Every one of the five most confident correct negatives** is a deeply dependent, late
  claim: claim 43 ("*The device of claim 42 wherein…*"), claim 81 ("*The apparatus of any
  one of claims 72-80…*"), claim 49, claim 80, claim 50.
- **The five most confident correct positives** are claims 6, 3, 21, 21 and 1 (the
  repeated claim 21 is a within-application duplicate row, one of the known 12.2%). Three
  of the five have **no disclosing reference at all** (`n_disclose` = 0,
  `gold_disclose` = 0) yet score .9977–.9978, and the single most confident row of all
  (.9983) is a **double-patenting** rejection — a ground on which the eight prior-art
  references are irrelevant by definition.
- **The most confident errors in the "should have been positive" direction** are claim 66
  and claim 60. One of them (app 12847591, §112) has **seven of eight references
  disclosing including the gold** — the easiest possible positive on an entailment
  reading — and the model gives it **.0006**.
- The confident false positives are claims 16, 13 and 1 — low-numbered claims that
  happened not to be rejected.

### `rejection_type` is definitionally the label — quarantine it

Smoothed target encoding fit on TRAIN: `NEG` → **.0011**; `102` → .9943; `103` → .9924;
`DoublePatent` → .9929; `112` → .9930; `Other` → .9931; `101` → .9895; `NONE` → .9458.
**Alone-AUC .9882.** A joint "all metadata, no text" model reaches **.9883**, entirely on
the back of this one column.

It is a *sidecar column* in `dense_standard/split/*.csv`, **not** part of the model's
input — `methods/dense/train_reward_model.py` reads only `text` and `judgement`, verified
by inspection — so it is not a dense-model leak. But it is label-equivalent, and it must
never be exposed to a Track-A or Track-B proposer, to a Gemma scoring prompt, or to any
feature block, in this or any downstream run on this corpus. Recorded here as a standing
landmine.

---

### 4b. Would fixing the label rescue the cell? No.

The obvious repair is to restrict positives to the prior-art grounds (§102 / §103) and
re-run. That was tested directly
(`round0_fixed_label_counterfactual.py` → `round0_fixed_label_counterfactual.json`):
every block refit on the restricted TRAIN split, read on the restricted EVAL and TEST
splits.

| | pooled label (all grounds) | **fixed label (§102/§103 only)** |
|---|---:|---:|
| n (EVAL / TEST) | 5,994 / 5,994 | 4,863 / 5,396 |
| positive rate | .601 / .601 | .509 / .557 |
| VA | .6214 / .6434 | .6328 / .6446 |
| claim ordinal number alone | .7537 / .7513 | **.7693 / .7737** |
| STRUCT | .7626 / .7731 | .7755 / .7918 |
| VA + STRUCT | .7768 / .7916 | .7910 / .8029 |
| T (dense) | .7965 / .8389 | .8135 / .8398 |
| gap T − VA | +.1751 / +.1955 | +.1807 / +.1952 |
| **share of the gap closed by structure** | **88.7% / 75.8%** | **87.5% / 81.1%** |
| residual beyond VA + STRUCT | +.0197 / +.0473 | **+.0225 / +.0369** |

**The picture is unchanged, and the ordinal channel is if anything slightly *stronger* on
the clean prior-art label (.769 / .774).** Fixing the label is necessary for construct
validity but is **not sufficient** to make this cell mineable: there is no hidden
entailment residual waiting behind the mixed grounds.

*Caveat carried in the artifact:* the dense model was trained on the pooled label and is
only re-read on the restricted rows, so its .8135 / .8398 here is a stand-in. A dense arm
retrained on the prior-art-only label could differ — but it would have to gain more than
.15 AUC over the structural block to change the verdict, against a structural block that
gets *better* under the restriction.

## 5. Concept census of the A bank: it is ONE concept

The freeze requires a deduplicated concept census of the incoming bank at round 0. For
this cell the census is short:

| | columns | effective concepts after dedup |
|---|---:|---:|
| **A** (`a_n_disclose`, `a_any_disclose`, `a_frac_disclose`, `a_max_disclose_overlap`) | 4 | **1** |
| **V** (7 lexical/structural) | 7 | 6 (`v_n_refs` is 8 in 99.96% of rows — degenerate) |

All four A columns are deterministic aggregations of a single Gemma-4-31B per-reference
boolean — "does this reference disclose this claim element?" Pairwise Spearman
correlations are **.853–1.000**, and `a_n_disclose` ≡ `a_frac_disclose` at **ρ = 1.000**
(exactly, because the reference count is constant at 8).

**Context, and the reason this matters more than any other finding here.** CW community
entered its campaign with a **45-criterion** bank (35 distinct species in round 1); the
peer-verdict pilot with **54 effective concepts**. Patents claim-fell enters with **one**.
Δ_beyond on this cell is therefore not "a well-articulated scorecard failed to close the
gap" — it is **"articulation was never attempted on this cell"**. A campaign that mines
15 criteria per round against a one-criterion incumbent is not measuring the same
quantity the other cells' campaigns measured, and its curve would not be comparable to
theirs.

An unscored raw rubric corpus does exist for this domain
(`datasets/patents/online-rubrics/`: 8,179 gpt-5-mini-parsed MPEP / office-action
documents), so building a real patents A bank is a tractable, separately-scoped job — see
§8.

---

## 6. Two more channels, measured and recorded (neither is the driver)

### 6a. The candidate-reference set is label-conditional

Confirms and quantifies the 2026-07-07 forensic audit
(memory `patents-prior-art-pipeline`). Each row carries K = 8 candidate references
(8 in 59,911 of 59,937 rows). For a **positive**, seven are same-CPC FAISS fillers and
the examiner's actual cited reference is **appended**; for a **negative** there is no
gold reference at all.

| | value |
|---|---:|
| gold reference present, positives | **99.66%** |
| gold reference present, negatives | **0.00%** |
| `has_gold` alone-AUC (population) | **.9983** |
| gold in the **last** slot, positives | **86.63%** |

This is severe in principle: the presence of an examiner-cited document *is* the label,
and its slot is nearly fixed. But every surrogate a model could actually key on is weak:

| surrogate of the construction channel | AUC (EVAL) |
|---|---:|
| per-position claim↔reference lexical overlap, 8 columns | .5993 |
| per-position span length, 8 columns | .5747 |
| both, 16 columns | .6122 |
| last-slot contrast features (overlap and length, last vs mean of first seven) | .6080 |
| order-invariant control (max/mean overlap, mean/total span length, n_refs, element length) | .5974 |
| `ov_last − mean(ov_first7)` alone | .5476 |
| "the highest-overlap reference is the last one" alone | .5222 |

Mean per-position overlap barely differs by slot (positives .173→.113 across slots 1–7,
then .124 at slot 8; negatives .149→.100, then .097). So the positional signature is
present but shallow *in these lexical surrogates*.

**The ablation revises this section, and the revision matters.** §7b's `references only`
condition — the claim element deleted, so the model sees nothing but the eight reference
passages — reaches **.6433**, above every lexical surrogate in the table (best: .6122).
The dense reader therefore *does* extract a reference-set provenance signal that lexical
overlap and span length cannot see, presumably the textual signature of a
retrieved-by-examiner document versus a same-CPC FAISS filler. It is not the largest
channel on this cell — claim ordinal position is — but it is real, it is larger than the
surrogates suggested, and it is **entirely an artefact of how the corpus was built**.

This is the strongest single reason the corpus must not be used for any "identify the
disclosing reference among K" evaluation without re-randomising slot order **and**
constructing negatives symmetrically, and it retrospectively justifies the earlier
registry entry's caution about this cell.

### 6b. Per-app_id decomposition: the application channel is not the driver

| | EVAL | TEST |
|---|---:|---:|
| applications | 1,118 | 5,010 |
| rows per application (mean / max) | 5.36 / 67 | 1.20 / 68 |
| share of the AUC's (pos, neg) pair population that is **within**-application | **0.10%** | **0.03%** |
| within-application AUC | .8085 | .7798 |
| cross-application AUC | .7961 | .8390 |
| application-identity oracle (leave-one-out app mean label) | **.5753** | **.6096** |
| applications that are label-pure | 25.2% | 98.4% |

The pooled AUC is essentially all cross-application, and there is no within/between
inversion. Application identity alone would only reach .58–.61.

**This also explains the EVAL/TEST gap (.7965 vs .8389) without invoking anything about
the model**: the pos-rate-matching split objective produced two very differently
*composed* held-out sets — EVAL is a small number of large applications (5.36 rows each,
only a quarter label-pure), TEST is many singleton applications (1.20 rows each, 98.4%
label-pure). Between-application variation, which the model partly captures, is a larger
share of TEST's pair population. Worth flagging for the other two V4 cells built by the
same script.

### 6c. Length and other nuisance channels

| channel | alone-AUC (EVAL) | Spearman with dense probability |
|---|---:|---:|
| total text length | .5724 | +.225 |
| total span characters | .5689 | +.207 |
| mean span characters | .5689 | +.207 |
| element characters | .5329 | +.142 |
| element words | .5289 | +.125 |
| reference count | .5000 | — (constant) |
| dependent-claim flag | .4347 (i.e. **.5653** inverted) | −.274 |
| **claim ordinal number** | **.2460** (i.e. **.7540** inverted) | **−.484** |
| joint, all of the above (no text content) | **.7626** | — |

The earlier note's reassurance — *"V itself already includes element/span-length features
and only reaches .601, so length/lexical shortcuts alone cannot explain the gap"* — was
correct about **length** and is confirmed here (length channels top out at .572). It did
not test **claim ordinal position**, which is the channel that was actually there, and
which V does not contain.

---

## 6d. Round-0 spurious map (FREEZE ADDENDUM 2 shape, measured rather than proposed)

No fleet was seated, so this map is not mined — it is the set of channels the audit
itself measured, written in the addendum's format (channel, alone-AUC, conjectured
upstream parent, MIXED flag) so it can be carried into the other cells' Track-B priors.

| channel | alone-AUC (EVAL) | conjectured upstream parent | MIXED |
|---|---:|---|---|
| **Claim ordinal position** (printed verbatim as "*of claim N*" in 82% of elements) | **.7537** | attorney claim-drafting convention: broadest claims first; examiners reject the broadest | **yes → decomposed, see §6e** |
| Parent-claim number parsed straight from the text | .7252 | same parent | yes |
| Dependency depth (dependent-claim flag) | .5653 | same parent | yes |
| Total text length | .5724 | span-extraction pipeline output length | no (surface) |
| Total / mean reference span characters | .5689 | same | no (surface) |
| USPC class | .5431 | art-unit-specific examiner leniency and prior-art density | yes |
| CPC subclass | .5408 | same | yes |
| Examiner art unit | .5348 | examiner identity and art-unit norms | yes |
| Claim element length | .5329 | claim breadth (a short limitation is a broad limitation) | yes |
| Filing year | .5316 | policy era (e.g. the §101 regime shift after mid-2014) | yes |
| Examiner tech centre | .5203 | same as art unit, coarser | yes |
| **Gold-reference presence** | **.9983** | **corpus construction** — the examiner's cited document is appended to positives only | no (construction artifact) |
| **`rejection_type` sidecar column** | **.9882** | **definitional** — derived from the same record as the label | n/a — quarantine (§4) |
| **`gold_disclose` column** in `patents_va_features.csv` | **.6589** | **construction-conditional** — see below | n/a — quarantine |

The three channels at the bottom are not model channels (none reaches the dense reader's
input); they are recorded so that no future feature block, judge prompt, or proposer
context on this corpus can pick them up by accident.

**`gold_disclose` deserves its own warning.** It is 0 for *every* negative row by
construction (negatives have no gold reference at all), so it is a strict subset
indicator of `has_gold`, and its .6589 alone-AUC is arithmetic, not validity:
½ + ½ · P(the gold reference discloses | positive) = ½ + ½ · .314 ≈ .657, matching the
2026-07-07 forensic audit's measured 31.4% gold-disclosure rate almost exactly. It sits
in `notebooks/data/patents_va_features.csv` next to the A columns and is correctly
**excluded** from `patents_verdict_layer1.py`'s `A_ONLY_COLS` — that exclusion must be
preserved, and the column must never be read as evidence that the disclosure judgment
carries signal.

## 6e. FREEZE ADDENDUM 3: decomposing the one MIXED channel

Claim ordinal position is MIXED in exactly the addendum's sense — its conjectured
parent (drafting convention) plausibly causes real merit-relevant variation, because
breadth is precisely what makes a claim vulnerable to prior art. So it was decomposed
and each component measured separately
(`round0_mixed_decomposition.py` → `round0_mixed_decomposition.json`):

- **SURFACE component** — the ordinal integer and the parent-claim number printed in the
  text; a drafting-order habit that says nothing about the claim itself.
- **CANDIDATE-REAL component** — claim **breadth**, proxied by independence, element
  brevity, limitation-marker count, comma count and numeral count.

| model | EVAL | TEST |
|---|---:|---:|
| SURFACE only | .7641 | .7740 |
| CANDIDATE-REAL (breadth) only | .6998 | .7128 |
| both | .7684 | .7813 |
| **marginal of SURFACE over breadth** | **+.0686** | **+.0685** |
| **marginal of breadth over SURFACE** | **+.0043** | **+.0073** |

Cross-stratification says the same thing twice:

| readout | EVAL | TEST |
|---|---:|---:|
| claim-number alone, stratified by element-length decile | **.7520** | **.7445** |
| element-length alone, stratified by claim-number decile | **.5098** | **.5142** |
| claim-number alone, **within independent claims only** | .8032 (n=1,364) | .7913 (n=1,646) |
| claim-number alone, within dependent claims only | .7324 (n=4,630) | .7199 (n=4,348) |

Holding brevity fixed leaves the ordinal channel essentially untouched (.754 → .752);
holding ordinal position fixed collapses the brevity channel to chance (.533 → .510).
And within independent claims alone — where the dependency distinction cannot be doing
the work — ordinal position is *stronger* than pooled (.803).

**Verdict: the components separate, and the surface side wins.** The MIXED flag is
retired in favour of a measured split: under these proxies the merit-relevant component
contributes ≈ +.005 and the drafting-order component ≈ +.069.

**Honest limit on that verdict.** The breadth proxies are deterministic and crude
(word counts, marker counts). Two readings survive the data: either ordinal position
*is* the better breadth measure — drafting convention encodes scope more faithfully
than word count does — or it carries a separate procedural regularity (examiners work
down the claim set and stop). These proxies cannot separate them. The judged criterion
that would is nameable and specific: *"is this claim element broad in scope, judged on
its own terms"*, scored blind to the claim's number and to any "of claim N" preamble.
That is the criterion a Track-A round would have had to write here, and it is recorded
as the concrete next instrument rather than left as an assumption.

## 7. Dense seeds 1–2 (in flight)

Launched on the claimed GPU 3 at 2026-08-07 12:46 PDT, exact dense-standard recipe and
the same frozen splits (`methods/dense/run_patents_seeds12.sh`, chained, `RUN_DONE`
sentinels, resumable, `nohup … & disown` so it is decoupled from any SSH session).
Verified per the server-diligence rule at launch + 45 s: PID 2207704 running
`train_reward_model.py` with the exact seed-1 arguments; GPU 3 at 28,614 MiB / 84% util
(0 MiB / 0% before launch); training log at `Epoch 1 Step 1`.

Seed 42 took 5 h 02 min; two seeds plus the 3-seed re-scoring pass land roughly
2026-08-08 03:00–05:00 PDT. Results will be written by the chain to
`datasets/patents/dense_standard/eval_pass_results.json` (all three seeds) and
`score_3seed.log`; the log is `methods/dense/logs/patents_seeds12.log`, terminating in
`PATENTS_SEEDS12_ALL_DONE`.

**The seed spread cannot change the verdict.** HashtagWars and Style Invitational showed
seed ranges of .020 and .038 at n = 4,228 and 9,637; patents is 6–14× larger, so the
expected spread is well under .01, against a structural channel that claims .13–.15 of
the gap. The 3-seed T will be appended here when the chain finishes.

## 7b. Text ablations on the trained model (partial, in flight)

Two passes re-score the EVAL split with the trained seed-42 model. Both assert, in code
and before loading the model, that their untouched variant reconstructs the split CSV's
text byte-for-byte, and both have now confirmed the numeric gate as well — the untouched
variant returns **AUC .7965** in each pass, identical to the filed predictions (swap pass:
mean input length 2,413 characters).

**Pass 1 — deletion ablation** (`round0_ablation_score.py` → `round0_ablation_eval.json`).
Deletes parts of the input: references shuffled, last reference dropped, last reference
only, claim element only, references only, gold reference dropped.

| variant | AUC | mean predicted probability | Spearman vs full |
|---|---:|---:|---:|
| full (reproduction gate) | **.7965** | .5846 | 1.000 |
| **element only** — all 8 references removed; entailment *possible* | **.5988** | **.7886** | .400 |
| **references only** — claim element removed; entailment **impossible** | **.6433** | .6409 | **.664** |

The pass was stopped after these three variants (own PID only, logged to the ledger) so
the remaining five would not keep competing with the required seed training; every
variant is checkpointed, and the two that answer the question had landed.

**Read the levels with care — the deletion pass is confounded, and the mean probability
shows how.** A model trained on ~1,024-token element-plus-eight-reference blocks is far
off-distribution when handed a bare 50-token claim element; its mean predicted
probability jumps from .585 to .789 — it is not merely less informed, it is
*miscalibrated by the shift*. So both numbers are **lower bounds**, and neither is a
decomposition.

**But the ORDERING is not confounded, and it is the finding.**

> **Removing the claim element hurts less than removing the references (.643 vs .599),
> and the claim-free scores still rank the corpus at Spearman .664 against the full-text
> scores.** Claim-to-reference entailment is *impossible* in the `references only`
> condition. So the half of the input that makes entailment possible contributes less on
> its own than the half that makes it impossible.

For scale: `element_only` = .599 sits well below the **.7537** a purpose-built model gets
from the claim's ordinal number alone — the honest, in-distribution comparison — which is
the same conclusion §3 reaches from the feature side.

**Pass 2 — swap ablation, the fair version** (`round0_swap_ablation.py` →
`round0_swap_eval.json`). Keeps the input shape and length identical and destroys only
the *correspondence*, using a deterministic stable-hash derangement within
reference-count buckets (never a seeded shuffle):

- `refs_swapped` — this claim, another row's eight references. **If the model is judging
  whether the prior art discloses the claim, this must destroy the signal.**
- `element_swapped` — another row's claim, this row's references (the mirror test).
- `both_swapped` — a coherent but wrong pair; the sanity floor.

This is the decisive form of the mechanism test, and unlike the deletion pass it carries
no distribution-shift confound: every variant is a well-formed
element-plus-eight-references block of the same length, so the model stays in
distribution and only the *correspondence* between the two halves is destroyed. Note
that a swapped half also carries its **donor's** label information, which is
uncorrelated with this row's label — so a channel that the swap destroys should fall
toward .5, and `both_swapped` is the sanity floor.

| variant | AUC | mean predicted probability | mean input chars |
|---|---:|---:|---:|
| original (reproduction gate) | **.7965** | .5846 | 2,413 |
| **`refs_swapped`** — this claim, another row's references | **.6527** | .2236 | 2,413 |
| **`element_swapped`** — another row's claim, these references | **.5983** | .2202 | 2,413 |
| `both_swapped` — sanity floor | *pending* | | |

**`refs_swapped` = .6527 is the cleanest single number in this audit.** The model keeps
this row's claim element and is given eight references drawn from an unrelated
application — references that cannot possibly disclose it, and whose own provenance
signal is uncorrelated with this row's label. Input shape and length are identical
(2,413 characters, to the character). **Destroying the entire claim-to-reference
relationship costs only .144 of the .797**, and .6527 of the AUC survives on the claim
element alone.

Two things follow.

1. **The model is not doing entailment.** An entailment reader handed mismatched prior
   art would fall to chance. This one does not: measured against AUC's actual floor of
   .5, it retains **(.6527 − .5) / (.7965 − .5) = 51.5% of its above-chance
   discrimination** with the claim-to-reference relationship completely destroyed. (The
   naive "82% of its AUC" framing overstates this and is not used.)
2. **The deletion pass understated the claim-side channel exactly as predicted.**
   In-distribution, the claim element alone carries **.653**; the off-distribution
   deletion estimate was .599. That is the distribution-shift correction §7b warned
   about, measured rather than assumed — and it is why the swap pass was built.

The mean predicted probability *does* move sharply (.585 → .224): mismatched references
make the model much less willing to say "fell". So it is sensitive to correspondence in
its *calibration* while barely losing *ranking*. For a threshold-free readout — the
program's standing rule — only the ranking counts, and the ranking is claim-side.

**Limitation of the swap design, recorded.** Donors are matched on reference count but
**not on technology class**, so a swapped reference set is usually topically obvious —
a battery claim paired with telecoms passages. The model plainly notices (hence the
calibration collapse). This makes `refs_swapped` an *easy* mismatch, and therefore a
**conservative** test of the entailment hypothesis: an entailment reader should find the
easy mismatch trivially and rank every row as non-disclosing, i.e. fall to chance. It
does not. A harder version — swapping only with same-CPC-subclass donors — would tighten
the estimate of the correspondence channel and is the obvious follow-up if this cell is
ever revived; it is not needed for the round-0 verdict.

For scale, note that .653 is **below** the .7537 a purpose-built model gets from the
claim's ordinal number alone: reading the whole claim element with an 8B model recovers
less of the label than one integer does.

Results land in `round0_swap_eval.json`, checkpointed per variant.

---

## 8. What should happen instead

1. **Do not let T = .7965 replace the registry's "NO honest dense model" decision for
   this cell without the qualifier.** `notes/2026-07-27__vat-run-registry.md` line 54 and
   `methods/taste_decomposition/patents_verdict_layer1.py`'s `special_rule` should be
   updated to point at this audit rather than simply deleted: an honest dense model now
   *exists*, and it reaches .7965, but **Δ_beyond for this cell is not interpretable as
   an articulation residual** — 76–89% of it is one un-banked structural feature and the
   label pools grounds the text cannot speak to.
2. **Report the cell as Δ_beyond ≈ +.02 to +.05 over V + A + structure**, with the
   ordinal-position channel banked, if it is reported at all. Never quote +.171 as an
   articulation residual.
3. **Fix the label before any further work on this cell — but do not expect it to
   rescue the cell.** Restrict positives to §102/§103, or split the cell by statutory
   ground: as it stands the construct ("claim later invalidated over prior art") and the
   operationalisation ("rejected on any ground") are different things. §4b measures the
   repair directly and it does not change the verdict — on the clean prior-art label the
   ordinal channel is slightly *stronger* (.769/.774) and structure still claims 81–88%
   of the gap. The label fix buys construct validity, not a mineable residual.
4. **Build a real patents A bank** from `datasets/patents/online-rubrics/` (8,179 parsed
   MPEP / office-action documents) before any closure campaign here. A campaign against a
   one-concept incumbent measures the incumbent, not the residual.
5. **Quarantine `rejection_type`** from every proposer, judge, and feature block on this
   corpus (§4).
6. **Re-examine the other two V4 cells** built by the same split script for the EVAL/TEST
   composition asymmetry documented in §6b.
7. **Carry the claim-ordinal-position finding into the other cells' Track-B maps as a
   named prior.** "Position of the item within its own parent container" is a generic
   upstream-traced nuisance in the FREEZE ADDENDUM 2 shape — parent = a drafting or
   submission convention (here: attorneys put the broadest claims first, and examiners
   reject the broadest claims).

   **Verified rather than asserted:** every criterion and routing file of the two
   completed campaigns was scanned for position-shaped channel names — CW community
   (14 `round*_criteria.json` / `round*_routing.json` files, 125 scored criteria) and the
   peer-verdict pilot (8 `round*_track_b.json` / `round*_routing_final.json` files). The
   20 name matches are all false positives of the pattern ("First-person pronoun count",
   "Late reveal reframes rather than cancels", "Dialogue-first openings", "Explicit
   priority or first-ness claims"). **No proposer on either cell named a
   container-position channel** — not "position in the thread", not "submission order",
   not "which entry number this is". It is a genuine gap in the mined maps, and this
   cell is the reason to close it. (nc_responded and maps_batch1 have no criteria/routing
   files in the repo yet and were not scannable.)

---

## 9. Claim discipline

Quotable:

> A round-0 audit of the patents claim-fell dense arm reproduced T = .7965 (clean-eval,
> seed 42) and found no class, era, application-identity, duplicate-text or
> prediction-collapse leak — every technology-classification and filing-era channel sits
> at .52–.55. It nevertheless **stopped the closure campaign before mining**, because a
> single non-content metadata column — the claim's ordinal number in its own application,
> printed verbatim in 82% of the claim texts — reaches alone-AUC .754 and claims 85% of
> the +.175 same-split residual on EVAL (69% of +.196 on TEST), leaving +.020
> [.001, .038] / +.047 [.029, .067] beyond V + A + structure. Independently, the cell's
> label pools §112 / §101 / double-patenting rejections, for which the corpus's eight
> prior-art references are irrelevant by construction, and the dense model separates
> those positives from negatives as well as it separates §102/§103 positives — while
> assigning mean probability .72 to the 43% of positives with **zero** disclosing
> references, and scoring **higher with the claim element deleted (.643) than with the
> references deleted (.599)**, though entailment is impossible in the former condition.

Never quotable from this cell: "Δ_beyond = +.171" as an articulation residual; "the
patents residual is tacit"; any patents closure curve or plateau (none was run); the
`rejection_type` or all-metadata AUCs (.988) as evidence about the dense model.

## Caveats that travel with every number here

1. **Single-seed T.** Every number above uses the seed-42 model. Seeds 1–2 are in flight
   (§7); the 3-seed T and spread will be appended.
2. **EVAL carries a max-of-two-epochs selection bias**; TEST does not. Both are reported
   at every step, and the two agree on the verdict while differing on the exact share
   (85% vs 69% for claim number alone).
3. **Same-split VA (.6214 EVAL / .6434 TEST) is not the Layer-1 protocol VA_nl (.6256)** —
   the Layer-1 number is grouped-OOF over the full 59,937-row population. The same-split
   figure is the one that makes Δ_beyond apples-to-apples with T; the two are close
   enough (.6214 vs .6256) that the headline gap reproduces either way.
4. **The structural-channel share is a *banking* statement, not a causal one.** Claim
   ordinal position is a legitimate, observable predictor, not label leakage. The finding
   is that it is trivially articulable and was simply absent from the bank — which is
   precisely why mining rounds would have been uninformative here.
5. **STRUCT's target encodings are fit on TRAIN only**, and the splits are app_id-disjoint,
   but a small optimism from encoding-on-train cannot be fully excluded for the
   high-cardinality categorical channels in §2.
6. **The deletion-ablation levels are lower bounds, not a decomposition.** Deleting part
   of the input takes a full-text-trained model off-distribution (§7b). Only the
   *ordering* of `element_only` vs `refs_only` is quoted as evidence; the fair,
   shape-preserving swap ablation was launched to settle the levels and had not finished
   at the time of writing.
7. **No GEPA phrasing pass, no fleet, no Gemma scoring** — the campaign stopped before any
   of the freeze's mining machinery was exercised on this cell.

## Artifacts

All under `methods/taste_decomposition/closure/patents/`:

| file | what |
|---|---|
| `RUNBOOK.md` | exact command sequence, input prerequisites, the standing landmines on this corpus, and the four things that must happen before the cell is revived |
| `round0_audit_cpu.py` → `round0_audit_cpu.json` | alignment gate, prediction distribution, calibration, per-app decomposition, stratified AUCs, class/era alone-AUCs, nuisance channels, construction audit, duplicate-contamination check, same-split V/A |
| `round0_gap_decomposition.py` → `round0_gap_decomposition.json`, `round0_gap_scores.npz` | the claim_num channel, the ladder, gap accounting, stacked increment, STRUCT discounts, group bootstrap CIs |
| `round0_ydef_probe.py` → `round0_ydef_probe.json` | per-statutory-ground dense AUCs, claim-number-matched AUCs, zero-disclosure positives, disclosure-evidence correlations |
| `round0_mixed_decomposition.py` → `round0_mixed_decomposition.json` | Addendum-3 proxy decomposition of the claim-ordinal-position channel into surface and breadth components, with cross-stratification |
| `round0_fixed_label_counterfactual.py` → `round0_fixed_label_counterfactual.json` | the whole ladder refit on the §102/§103-only label (§4b) |
| `round0_ablation_score.py` → `round0_ablation_eval.json`, `round0_ablation_eval_probs.npz` | deletion ablation, checkpointed per variant (partial — see §7b for the distribution-shift caveat) |
| `round0_swap_ablation.py` → `round0_swap_eval.json`, `round0_swap_eval_probs.npz` | length- and format-preserving swap ablation, the fair mechanism test (in flight) |
| `round0_manual_read.json` | the 16 high-confidence rows read in §4, with full reference lists |
| `methods/dense/run_patents_seeds12.sh` (sk3) | the seeds 1–2 chain |
