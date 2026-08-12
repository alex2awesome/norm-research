# Layer-3 articulation closure — BBC most-read (journalism community #2)

Cell build note: `notes/2026-08-10__bbc_mostread_build.md`.
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` + FREEZE DECLARATION 2026-08-06 +
ADDENDA 1–4. Campaign dir: `methods/taste_decomposition/closure/bbc_mostread/`.
Agent: `claude-v9-journalism-tweets` (journalism discovery lane).

**Why this cell first.** It carries the best-powered residual in the journalism column
— Layer-1 same-rows Δ_beyond **+.0864 eval / +.0690 test** against a dense 3-seed
spread of .0021 — so it is the column's genuine taste question. The homepage cell
(+.0068/+.0109) and the tweets cell run after it.

## Terms, spelled out on first mention (standing rule)

| term | what it means here |
|---|---|
| **V** | the 23 deterministic headline surface features (`v_*`) |
| **A** | the 14-criterion Gemma-4-31B news-values bank, reused verbatim from the homepage curation cell |
| **VA_lin / VA_nl** | the articulated instrument: V+A fit linearly / by HistGradientBoosting on the frozen grid; VA_nl = mean over fit seeds {0,1,2} |
| **T** | the dense arm: Llama-3.1-8B LoRA, T = MEAN OVER DENSE SEEDS of the held-out AUC (never the AUC of the seed-averaged prediction) |
| **Δ_r** | the closure curve: T − VA_nl after r rounds of active mining |
| **FIT+MINE / MONITOR** | the closure split; MONITOR lives inside the dense-held-out rows and is never read by any proposer |
| **M (mining slice)** | FIT+MINE ∩ dense-held-out — the only rows where dense scores are honest |
| **Track A / Track B** | A proposes quality-relevant criteria that could close the gap; B proposes suspected-spurious channels used only to DISCOUNT |
| **swap pair (C₊, C₋)** | P(bank orders a discordant pair correctly \| dense does) and \| dense does not) |
| **ε** | .005, the frozen saturation threshold on the MONITOR VA_nl round-over-round gain |

---

## 1. Round 0 — hard gates, run before anything else

### 1.1 DENSE JOIN GATE — **PASS**

This cell's dense predictions carry **no row_id**: `preds_{eval,test}.csv` are
`(judgement, prob, group)` only. The join to rows is therefore **by order**, which is
precisely the registry's alignment landmine. It is proven rather than assumed — the
`(judgement, group)` sequence must match the split file element-wise on every row for
every seed, and a shuffled counterfactual must destroy the AUC:

| leg | n | sequence match, all 3 seeds | AUC seed 42 | shuffled counterfactual |
|---|---|---|---|---|
| eval | 5,075 | ✓ | .8239 | **.5038** |
| test | 5,072 | ✓ | .8095 | **.4956** |

The script raises and refuses to report if any of these fail.

### 1.2 SPLITS — **PASS**

Stable-hash `sha256("bbc-mostread-closure-v1|" + capture_day) < .20`, never a seeded
shuffle, and — per prereg AMENDMENT 1 — MONITOR is taken strictly **inside the
dense-held-out rows**.

| | value |
|---|---|
| MONITOR | 2,060 rows / 88 days |
| FIT+MINE | 48,701 rows |
| mining slice M = FIT+MINE ∩ dense-held-out | 8,087 rows |
| MONITOR ⊂ dense-held-out | **true** (asserted) |
| no day spans both | **true** (asserted) |
| pos rate MONITOR / FIT+MINE | .4340 / .4407 |

T over the dense-held-out rows: per-seed **.8167 / .8149 / .8174**, mean **.8164**,
spread **.0025**.

### 1.3 ITEM-VIEW ASSERTION (the SO lesson) — trivially satisfied here

`notes/2026-08-11__so_votes_audit.md` showed that a cell can produce a spurious
bank-beats-dense verdict when the dense arm and the A judge read different documents,
and that truncation must be measured in **tokens on the real tokenizer**, never in
characters. On this cell V, A and the dense arm all read the **byte-identical** string
`"HEADLINE: " + anchor headline`; headlines are ≤22 tokens (measured), so nothing
truncates and no view asymmetry is possible. Recorded because the assertion is
mandatory, not because it was in doubt.

### 1.4 OBSERVED-ORDINAL COVARIATES (FREEZE ADDENDUM 4) — the position channel is **structurally unavailable**, and this is a finding

Addendum 4 requires the position-in-container channel be considered, because it has
produced two of the programme's strongest spurious findings. On this cell it **cannot
be measured**, and the reason is in the scraper source rather than in the data:

```
rec["others"] = harvest_other_headlines(soup, mr_hrefs)
```

The negative pool is built by **excluding every most-read href by construction**.
Measured consequence: of **33,400** most-read entries across the morph captures,
**0** also appear in `others`. So page position exists only for negatives and is
perfectly confounded with y — using it as a covariate would be a leak, not a control.

This has a direct consequence for the standing worry the taxonomy note filed against
this y ("lists reflect placement… instrument w/ homepage data: placement-adjusted"):
**that worry cannot be addressed on this build.** Doing so would require a re-parse of
the raw captures retaining page position for most-read items. Recorded as a limitation
of the cell, not mined, and not silently skipped.

The two ordinals that *are* legitimate:
- **most-read rank (1–10)** — defined only for positives, i.e. post-outcome. It is a
  within-winner ordering readout (Layer-1 already reports Spearman +.144 for VA_nl),
  never a covariate for y.
- **capture-day ordinal** — available and legitimate; carried as this cell's observed
  ordinal (era index; Layer-1 per-year VA_nl .706–.770 across 2017–2023).

### 1.5 ε-RESOLVABILITY POWER CHECK — run BEFORE round 0's curve, per the SO lesson

A sub-ε round is only interpretable if the design can resolve a change of ε = .005 on
MONITOR. The noise floor is estimated from a comparison whose true change is known to
be **zero**: two VA_nl fits differing only by GBM seed, with the paired AUC difference
group-bootstrapped over MONITOR days (2,000 resamples, day-level, matching every other
CI in the campaign). If the mean paired SD ≥ ε, the saturation rule is not
interpretable as written and the campaign cross-fits (averages over fold-seeds) until
it is, recording the depth.

**RESULT — resolvable, no cross-fitting needed, but with a caveat that matters.**

| known-zero comparison | point | boot SD | 95% width |
|---|---|---|---|
| seed0 vs seed1 | −.0025 | .0023 | .0088 |
| seed0 vs seed2 | +.0009 | .0028 | .0108 |
| seed1 vs seed2 | **+.0034** | .0025 | .0096 |
| **mean paired SD** | | **.00252** | |

Mean paired SD **.00252 < ε = .005**, so the design resolves ε and the campaign runs
**without cross-fitting** (depth 0, recorded).

The caveat, stated because it constrains how any single round may be read: the 95%
width of a comparison whose true change is **zero** is .009–.011, i.e. roughly **2ε**.
And one such comparison actually reads **+.0034 — 68% of ε — from seed noise alone.**
So a single round gain of ~.003 is *not* distinguishable from nothing on this cell.
This is exactly why the frozen rule requires **two consecutive** sub-ε rounds rather
than one, and it is the quantitative form of the SO lesson for this campaign: the
saturation rule is interpretable here, but no individual round's gain near ε is
quotable on its own.

---

## 2. Round 0 — the anchor

The curve is measured from the **closure protocol's own round-0 anchor** (VA fit on
FIT+MINE only, read on MONITOR), never from the Layer-1 number: closure-split levels
are protocol-specific (prereg AMENDMENT 1). *(This cell runs sklearn 1.8.0, the same
build the Layer-1 ledger used, so unlike the math.SE campaign there is no GroupKFold
version drift here — recorded, but the protocol-specific rule still governs.)*

| quantity | MONITOR (n = 2,060, 88 days) |
|---|---|
| VA_lin | .7176 |
| VA_nl per seed | .7477 / .7502 / .7468 |
| **VA_nl (mean of seed AUCs)** | **.7482** |
| T per dense seed | .8229 / .8215 / .8249 |
| **T** | **.8231** |
| **Δ₀** | **+.0749** |

(VA_lin OOF within FIT+MINE: .7052. The seed-averaged-prediction AUC is .7502 and is
*not* the reported VA_nl — the frozen definition is the mean of per-seed AUCs.)

### 2.1 GATE — **PASS, rounds run**

Δ₀ = **+.0749** against the frozen threshold of .02. For context only (never
differenced against the closure number): Layer-1 same-rows Δ_beyond was +.0864 eval /
+.0690 test, so the closure-split anchor sits between the two Layer-1 legs, which is
the expected behaviour and a reassuring sign that the closure split is not doing
anything strange.

### 2.2 Swap baseline (C₊, C₋)

| | value |
|---|---|
| C₊ = P(bank correct \| dense correct) | **.8227** |
| C₋ = P(bank correct \| dense wrong) | **.4079** |
| dense concordance on sampled pairs | .8278 |
| pairs sampled | 400,000 |

The round-0 asymmetry is large: where the dense model gets a discordant pair right the
bank agrees 82% of the time, but where the dense model gets it wrong the bank is at
**.41 — below the coin flip**. The adverse signature to watch in later rounds (per the
swap algebra) is C₊ rising while C₋ **falls**, which would mean the bank is buying rank
agreement by inheriting dense errors rather than by getting independently better.
Starting from C₋ < .5, there is ample room for a genuine uniform improvement, and that
is what a real closure should look like.

## 3. Fleet

Per the coordinator's standing instruction, fleets run at **P = 8 across 2 families**
(gpt-5.6-luna via the Codex companion, GLM-5.2) — **the Claude legs are unavailable
this session because the subagent cap is exhausted (500/500)**, so the 3-family target
degrades to the freeze's 2-family floor. This is recorded per round rather than
silently absorbed, as the freeze requires.

## 4. Compute discipline

Creative writing has first claim on free GPUs (coordinator, 2026-08-11). This lane runs
**one card at a time**; round 0 is CPU-only and holds no card. Gemma scoring for rounds
≥ 1 is shard-checkpointed (`score_bank` skips completed shards), so the card can be
yielded mid-round and re-claimed without losing work. All jobs launch under
`setsid --fork` with **ppid = 1 asserted** — the V9 build lost ~10 minutes of GPU to a
chain that waited on `pgrep -f` and matched its own launching shell, so chains here
wait on a PID via `kill -0`.

## 5. Artifacts

| what | path |
|---|---|
| round-0 driver | `methods/taste_decomposition/closure/bbc_mostread/round0_bbc.py` |
| round-0 results | `.../closure/bbc_mostread/round0_results.json` |
| splits | `.../closure/bbc_mostread/splits.csv.gz` |
| cell population | `datasets/bbc-mostread/va/population.csv.gz` |
| A/V matrices | `outputs/va_gemma_banks_bbc_mostread/` |
| dense arm | `datasets/bbc-mostread/va/dense_standard_bbc_mostread/` |
| Layer-1 ledger | `methods/taste_decomposition/results/bbc_mostread_ledger.json` |
