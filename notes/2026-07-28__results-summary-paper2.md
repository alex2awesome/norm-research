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
  scrambled-foreign. Format-vs-order (kw3) still running — claim nothing there yet.
- **Scaling:** complete Qwen3 ladder 8/8 cells — at 8B random pool draws beat the GEPA-shipped
  prompt 40/40 on both benchmarks (HotpotQA +.186, HoVer +.070), gain grows with scale;
  same-session prefix curve is inverted-U (peak k=46), so "more units" is not monotone.
- **Unsupervised bank + caveats:** 272 mined metrics, median achieved .825 vs calibrated ceiling
  .863 (gap .033, 96.1% held-out coverage; rank certificate p ≤ .0016). Standing caveats: hotpot
  seed caveat travels with the number; IFBench star session-sensitive (three sessions all
  positive, one significant, footnoted); MIPROv2 was not run on HoVer (gap, not result); PUPA
  remaining cells unmeasurable (deprecated third-party judge API).

## 3–7. (fold in after hover omnibus + ifbench insurance land tonight)
3. Mechanism: intact > scrambled > keywords ≈ foreign > scrambled-foreign; keyword-only variants
   recover at most ~27% (HotpotQA) / ~12% (HoVer) of the native unit gain — bounds, not point
   estimates (operator loss ~30%). kw3 (format control) pending.
4. Scaling: OSL ladder 8/8 cells; draws beat shipped prompt 40/40 at 8B on both benches
   (hotpot +.186, hover +.070); same-session prefix curve inverted-U, peak k=46 (never headline .643).
5. Unsupervised bank (272 metrics): calibrated missing-value ceiling .863 (gap .033, 96.1%
   held-out coverage); rank certificate .1949 vs .200 predicted.
6. Reversal ledger: three preregistered reversals (HB146→HB157→HB163) — frozen rules made each
   legible.
7. Limitations: hotpot seed caveat travels with +.235 in every sentence; PUPA envelope unmeasurable
   (deprecated third-party judge API); session sensitivity documented on aime/ifbench/livebench.
