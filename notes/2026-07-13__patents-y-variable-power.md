# Patents y-variable power analysis (dual-audit implementation)

**Date:** 2026-07-13
**Provenance:** user-ordered parallel audits of the acceptance-prediction leg (Codex gpt-5.6-sol
+ Claude Fable, same three questions), then "implement any suggestions".
**Code:** `scripts/patents_event_panel.py` (dated per-app prosecution-event panel + examiner
leniency), `scripts/patents_y_power.py` (three-sense y-power analysis).
**Artifacts (sk3):** `datasets/patents/processed/prosecution_event_panel.parquet` (9.7M disposed
apps), `outputs/patents_y_power/results.json`, log `logs/ypower_chain2.log`.

## Question

Would a different y give more power/discrimination than final grant/abandon — e.g. "the patent
will get another iteration" (RCE / more OA rounds)? Split into three senses:
(a) intrinsic text-predictability, (b) label-free disclosure-metric marginal, (c) construct fit.

## Panel base rates (all disposed apps, n=9.71M)

| y (risk-set-conditioned) | n | rate |
|---|---|---|
| first-action allowance | 9.51M | .142 |
| another round (≥2 substantive OA \| ≥1) | 8.17M | .498 |
| RCE after final (\| final rejection, pre-allowance) | 3.69M | .461 |
| appeal after final (\| final rejection) | 3.69M | .152 |
| examiner withdrew at pre-appeal conf (APCA \| conf) | 93K | .103 |
| abandoned (\| ≥2 rounds, disposed) | 4.07M | .311 |
| NOA before abandonment (\| abandoned) | — | .041 |

**Cohort artifact:** the cite-bearing filter makes first-action allowance DEGENERATE on the
existing cohorts (rate .001 on the 579K balanced file, .006 on option3) — that y needs a fresh,
unfiltered cohort. It is fine on the natural sample (.130).

## Sense (c): the examiner lottery dominates (cohort A, n=579,084, 100% join)

| predictor (alone) | AUC vs final outcome |
|---|---|
| examiner LOO grant rate | **.681** |
| art-unit × year LOO | .633 |
| art-unit LOO | .607 |
| publication year | .531 |
| text length | .510 |
| leniency(3) + year, CV | **.687** |

Examiner leniency alone **beats the linear text baseline** (.654–.658) on the same cohort, and
flipped-sign it predicts the iteration y's too (another-round .648, abandon-after-2 .670,
RCE-after-final .628 flipped). Every outcome-level y is heavily lottery-loaded; none of the
prior pipeline controlled it.

## Sense (a): intrinsic text predictability (hashing TF-IDF + linear, dedup'd)

| y | cohort A grouped / out-of-time | natural grouped / out-of-time |
|---|---|---|
| final outcome (grant) | .654 / .658 | .680 / **.653** |
| first-action allowance | degenerate | .668 / **.636** |
| another round | .619 / .603 | .599 / **.577** |
| RCE after final | .597 / .595 | .578 / **.565** |
| appeal after final | .650 / .649 | .629 / **.606** |
| abandon after 2 rounds | .619 / .616 | .628 / **.605** |

(out-of-time = train ≤2016 / test ≥2017; the grouped−oot gap on the natural cohort ≈ era
inflation, ~.03 on final outcome.)

**Answer to the headline question: NO — iteration y's have LESS text signal, not more.**
"Another round" (.577) and "RCE after final" (.565) sit well below final outcome (.653).
The only alternative y's that hold up are **first-action allowance** (.636, temporally closest
to the text, least strategy-contaminated) and **appeal-filed** (.606). Both audits predicted
this direction: iteration events are applicant-money/strategy events.

## Sense (b): disclosure-metric marginal, corrected design (cohort B, n=21,447)

Fixed M (deduped 59,937→56,111 rows), honest STRUCT (unique claim counts, filing year, element
length, examiner + art-unit + AU×year leniency), logistic AND HistGradientBoosting, app-bootstrap
95% CIs:

| y | struct (log/hgb) | disclosure marginal (log) | CI |
|---|---|---|---|
| final granted | .708 / **.756** | +.0002 | [−.001, +.001] |
| another round | .674 / .679 | −.0017 | [−.003, +.000] |
| RCE after final | .576 / .570 | +.0012 | [−.003, +.006] |
| appeal after final | .602 / .589 | +.0003 | [−.005, +.005] |
| abandon after 2 | .688 / .725 | −.0002 | [−.001, +.001] |

**The #87 null REPLICATES under the corrected design** — the disclosure metric adds nothing to
any outcome-level y, and that is no longer attributable to stale M row-weighting or thin
controls. (HGB marginals go *negative*: 12 noisy features dilute it.)

**Kicker:** metadata alone (leniency + structure, no text at all) reaches **.756** on final
outcome — above the orphaned ".70 dense ceiling" the old VAT table treated as the text
recoverability bound. The old "1% V vs 70% dense" arithmetic is dead in both directions.

## Consolidated conclusions

1. Patents outcome-level low-V **stands, and is now properly grounded**: the metric marginal is
   ≈0 with honest controls, for every y construction including iteration y's.
2. The right restatement of the ceiling: outcome y's are ~half examiner-lottery; honest linear
   text signal is .58–.68 depending on y; any future dense probe must be compared against the
   **.756 metadata baseline**, not chance.
3. If a cleaner outcome y is ever wanted: first-action allowance (needs a non-cite-bearing
   cohort) or appeal outcomes (APCA/APCP, n=93K, true examiner-error signal) — NOT RCE/rounds.
4. Doc fixes applied to `build_final_outcome_with_rejections.py`, `03_parse_labels.py`,
   `assemble_vat_table_patents.py` (dense-dict do-not-quote caveat),
   `patents_alty_recovery.py` (SUPERSEDED header).

## Not implemented (GPU/new-extraction, decision pending)

- Honest dense retrain on the real cohort (1 GPU, days) — value reduced now that the metadata
  baseline is .756 and linear text is characterized; would only sharpen sense-(a).
- M rebuild with semantic-delta propagation (GPU decompose+verify) — sense-(b) null replicated
  with deduped M; expected flip probability low.
- Claim-level survival y (new OARD/IFW extraction + text matching) — best construct in the
  space, highest cost; supersedes Y3's claim-number matching artifact.
