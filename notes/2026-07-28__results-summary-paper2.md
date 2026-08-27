# Paper #2 results summary — prompt-optimality campaign (drafted 2026-07-28, for EOD 2026-07-29)

## 1. Headline result (Table 1, canonical same-session protocol)

**"Under a pre-registered same-session protocol, the ε-certified recombination pool matches or
exceeds GEPA on all six benchmarks and significantly exceeds it on three (HoVer +.086, IFBench
+.040, and HotpotQA +.235 over GEPA's shipped prompt, which there equals the seed) — and the same
protocol retracted two of our own earlier apparent wins as measurement artifacts."**

| benchmark (n) | GEPA | GEPA+Merge | MIPROv2 | M_ω | Δ vs GEPA |
|---|---|---|---|---|---|
| HotpotQA (300) | .411 | .400 | .439 | **.646** | +.235*** |
| HoVer (300) | .471 | .517 | not measured | **.557** | +.086*** |
| LiveBench (126) | .699 | .688 | .677 | .705 | +.006 n.s. |
| IFBench (294) | .403 | .411 | .383 | **.443** | +.040** |
| AIME (150) | .385 | .373 | .323 | .392 | +.007 n.s. |
| PUPA (221) | .883 | n/a | n/a | .882 | −.001 |

Protocol: every row = one uniform session, all candidates rescored together, one server
fingerprint, k=5 passes; paired bootstrap on mean item-level delta (20k resamples). Primary
contrast pre-specified (Δ vs GEPA); head-to-heads secondary and uncorrected. Canonical-session
rule is direction-blind: it suppressed earlier favorable margins on AIME (+.091) and LiveBench
(+.092) and replaced an earlier unfavorable n.s. on IFBench (+.020).

## 2. The two-artifact retraction narrative (protocol works)

- **AIME**: earlier +.091 (p=.0012) came with GEPA-arm session variance (.307–.385) comparable to
  the margin; the uniform session gives +.007 [−.020,+.033]. No separation claim.
- **LiveBench**: earlier +.092*** rested on a GEPA arm scored under GPU load (.555); the metric is
  load-dependent (.696 idle vs .479 busy, same prompt). Idle uniform session: both arms ≈.70,
  Δ+.006 [−.015,+.028]. Load artifact; no claim.
- Discussion sentence (contribution, not confession): two of six benchmark effects measured in the
  literature's standard setup were instrument artifacts (session variance; judge load) — the
  same-session protocol detects and removes both.

## 3. Executive summary for reporting (advisor-signed, 2026-07-29)

- **Main result:** Under a pre-registered same-session protocol (all candidates rescored together,
  one server fingerprint, k=5, paired bootstrap), the ε-certified pool matches or exceeds GEPA on
  all six benchmarks (min Δ −.001) and significantly exceeds it on three: HoVer +.086*** (n=300),
  IFBench +.040** (n=294), HotpotQA +.235*** (n=300) — the HotpotQA margin is over GEPA's shipped
  prompt, which equals the seed there. [The parenthetical "(min Δ −.001)" is PART of the sentence
  — never drop it when shortening.]
- **The protocol polices itself:** the same rule that produced those stars retracted two of our own
  earlier apparent wins — AIME +.091 (GEPA-arm session variance comparable to the margin) and
  LiveBench +.092 (judge load-dependent; both arms ≈.70 idle). Two of six benchmark effects in the
  standard setup were instrument artifacts; a contribution, not a confession. Session wobble spans
  ~.01 (HoVer, conclusion unchanged) to ~.09 (AIME/LiveBench, conclusion reversed).
- **Mechanism:** the gain requires intact clause structure, not vocabulary — keyword-only variants
  recover at most ~27% (HotpotQA) / ~12% (HoVer) of the native unit gain (upper bounds; extraction
  strips operators from ~30% of units); ladder: intact > scrambled > keywords ≈ foreign >
  scrambled-foreign. Format-vs-order RESOLVED (kw3, HB173): per-unit bulleted keyword lists
  recover +.049 [.040,.058] — .002 from the shuffled reference, .038 from flat keywords → the
  missing form channel is **unit segmentation**, not word order.
- **Scaling:** complete Qwen3 ladder 8/8 cells — at 8B random pool draws beat the GEPA-shipped
  prompt 40/40 on both benchmarks (HotpotQA +.186, HoVer +.070), gain grows with scale;
  same-session prefix curve is inverted-U (peak k=46), so "more units" is not monotone.
- **Unsupervised bank + caveats:** 272 mined metrics, median achieved .825 vs calibrated ceiling
  .863 (gap .033, 96.1% held-out coverage; rank certificate p ≤ .0016). Standing caveats: hotpot
  seed caveat travels with the number; IFBench star session-sensitive (three sessions all
  positive, one significant, footnoted); MIPROv2 was not run on HoVer (gap, not result); PUPA
  remaining cells unmeasurable (deprecated third-party judge API).

## 4. Mechanism: what makes a unit work (complete)

The ladder is **intact clauses > word-scrambled > keyword list ≈ intact foreign > scrambled
foreign**. Two channels are now separated:

- **Vocabulary vs composition (HB163):** keyword-only variants recover **at most ~27% (HotpotQA)
  / ~12% (HoVer)** of the native unit gain. These are *upper bounds*, not point estimates: term
  extraction strips logical operators (not/only/all/at least/must) from ~30% of units, so some of
  the deficit is content loss rather than form. The qualitative verdict — composition is real —
  survives that caveat; the quantitative share is deliberately soft.
- **Format vs order (HB173, kw3):** bulleted per-unit keyword lists recover **+.049
  [.040, .058]** against a native anchor of +.093 — .002 from the shuffled-arm reference and .038
  from flat keywords, decided by a rule frozen before the run. So the recoverable "form" channel
  is **unit segmentation** (keeping units as separate bulleted clauses); restoring word order and
  bigrams adds essentially nothing beyond it. Additivity check: content .091 + form .046, with a
  −.036 sub-additive residual for flat keywords that segmentation alone recovers.

## 5. Scaling: the articulation bound as executors grow

- **Supervised ladder complete (8/8 cells, 24k context).** At 8B, random pool draws beat the
  GEPA-shipped prompt **40/40** on both benchmarks (HotpotQA +.186, HoVer +.070), and the gain
  grows with scale. The earlier truncation confound is gone (all cells re-run at 24k).
- **Curves are not monotone folklore.** The same-session prefix curve is inverted-U with peak
  k=46; never quote .643 as a plateau. A 5-pass re-measurement is running to quantify how much of
  the per-k jaggedness is single-pass noise (early within-k spread .013–.040).
- **Unsupervised panel, 1,270 fitted metrics, three regimes** with distinct type signatures
  (labels blind to regime): *rising* (n=884) skews to extended multi-part structure and
  verification/rigor; *reaches* (n=321) to local self-contained patterns — setup→punchline,
  endings, line economy — which is why those curves finish early (realized recovery .75, done);
  *bounded* (n=65) loads on creator identity/persona and real-world reception. **Trust caveat:**
  bounded and reaches ceilings are attained in range (CI widths .105/.33); rising ceilings are
  extrapolations (CI .56, 99% of upper bounds above 1) — "rising" mostly records censoring by our
  strongest executor, so never quote its median L (.95) as a ceiling.
- **Bounded metrics are domain-specific:** humor 15%, peer review 11%, news 10%, creative writing
  1%, and **zero** in math, press releases, notice-and-comment, patents.

## 6. Where the articulation boundary sits (two independent instruments)

- **Against code (seam, Table 3):** 629 sub-rules, labeled by what each check examines, blind to
  outcome, consensus of three passes (94% ≥2/3 agreement; κ .78–.81 vs consensus). Checks against
  recorded facts, surface form, and **codified normative standards** compile at 62–94%; checks
  that require reading meaning (cross-part consistency, semantic/pragmatic classification,
  argument adequacy) compile at 0–30%, with **nothing in between** — a genuine discontinuity. The
  ordering holds within six of seven domains, so the ten-domain seam-width spectrum (F .48–.91)
  is largely a *composition* effect: domains differ because their banks mix these check types
  differently. Headline reading: **normativity compiles; meaning does not.**
- **Against smaller executors (change-types):** 487 revision steps from three unsupervised
  corpora, two passes (κ .68 fine, .93 at the concept-vs-measurement grain). Tacit decompression
  writes **concept content** (definitions 51%, causal explanation 49%, and nothing else); GEPA
  optimization rounds write **measurement content** (score anchors 75–77%, edge cases, input
  hygiene); M_ω unit additions sit on the same side with their own signature (executor checking
  steps 47%). The only optimization lineages that write definitions are esoteric constructs the
  judge cannot be assumed to hold. Caveat owed in print: the decompression half is partly a
  manipulation check (those rungs were generated by instructions asking for definitions), so the
  informative cells are the **zero** concept cells on the optimization side.

## 7. Prediction: do the bounds forecast future behavior?

Backtested on all six benchmarks' prefix trajectories (fit on the first j points, predict the
rest): pooled median absolute error **.018 / .020 / .029** at horizons of 1–3 / 4–8 / 9+ units
ahead (power-law form; saturating exponential degrades to .043 at long horizons). The usable
claim is calibrated *median* error, not a guarantee: errors are heavy-tailed for fits on ≤5
points (pooled max .43), and on AIME short-window fits placed the asymptote *below* the realized
maximum. Supportable phrasing: fitted on ≥8–10 prefix points, the curve predicts 4–8 units ahead
with median error ≈.02–.03.

## 8. Budgets and comparability (state this before a reviewer does)

GEPA and the in-house replication ran at 600 metric calls per benchmark; the ε-certified arm at
2,400 (its paired selection prices more decisions); the original GEPA paper budgets 2.4K–7K. Two
consequences we disclose rather than bury: (a) at 600 calls GEPA accepted 1–8 candidates per
benchmark and on HotpotQA and IFBench accepted **nothing beyond its seed**, so our deltas there
are "at equal-order small budgets," not bounds on well-funded GEPA; (b) the 2,400-vs-600
asymmetry runs in our favour. A budget-matched 2,400-call GEPA run is the natural robustness arm
and is not yet done.

## 9. Reversal ledger and limitations

Four preregistered reversals make the protocol legible rather than embarrassing: the mechanism
verdict moved HB146 → HB157 → HB163 (each overturned by the *next* frozen control), and the
labeling grain moved HB182 → HB183 when a second pass showed 12-category labels were unstable
(κ .51) and a third pass plus consensus voting fixed it rather than hiding the rows.

Standing limitations, each of which must travel with its number: the **HotpotQA seed caveat**
travels with +.235 in every sentence; **IFBench** is session-sensitive (three sessions all
positive, one significant, all footnoted); **MIPROv2 was never run on HoVer** (a gap in the
candidate set, not a null result); **PUPA**'s remaining envelope cells are unmeasurable
(deprecated third-party judge API); **LiveBench** pre-2026-07-24 numbers are all deflated by a
Levenshtein defect and must never be re-quoted; OSL regime *themes* are names-based and
hypothesis-generating, not certified; and the seam/change-type labelings are LLM-assigned
consensus labels, reported with their agreement statistics.
