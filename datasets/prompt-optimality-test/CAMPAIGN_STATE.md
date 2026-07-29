# CAMPAIGN STATE — prompt-optimality / Paper #2

## ⏰ EOD-2026-07-29 READINESS CHECKLIST (hourly monitor 0e6cd647 drives this)
- [x] Table 1 FULLY COMPARABLE (HB171): all rows canonical uniform sessions; hover +.086***; MIPROv2-hover not measured (no candidate); daggers gone
- [x] Five-arm decomposition verdict (HB157: vocab ≥51% assumption-free; PROP bounded [0,.046]; advisor caps adopted HB158)
- [x] hotpot-shuffled-8B verdict (HB161: sign flip = SCALE confirmed; vocab-carries-all is hover-local — composition real on hotpot)
- [x] Fig 5 ladder complete (8/8 cells; BOTH benches 40/40 at 8B — HB159)
- [x] Fig 3 LHS same-session k=1..68 overlay (inverted-U measured: peak k=46 — HB159)
- [x] Rank certificate revalidated on median-split targets (.1949 vs .200 — conservative, HB160)
- [x] kw3 bulleted-keywords discriminator harvested (HB173): H2 FORMAT/SEGMENTATION — gain +.049 [.040,.058], .002 from shuffled ref; format-vs-order RESOLVED
- [ ] Final PDF + one-page results summary the user can report from (summary note sections 3-7 still stubs)
**Single source of truth. Updated 2026-07-28 00:05Z. Rewrite this file rather than adding another notes file.**

Written because the campaign had drifted across three boxes and several parallel plans. A census
(2026-07-27) showed the sprawl was mostly imaginary — see "Where things actually run".

---

## 1. Where things actually run

| box | our campaign work | verdict |
|---|---|---|
| **sk1** | **NONE.** Only `host_factory_loop.py`, which belongs to a different project. Its busy GPUs (2, 7) belong to other users — no compute apps are ours. | **not a campaign box** |
| **sk2** | **Everything.** All live lanes, all servers, all results. | **THE campaign box** |
| sk3 | Only other lineages (`src.search_loop`, `crx_r2_scores`, `score_va_gemma_captions`, patents/news `search_scores_oob`) plus two orphaned vLLM servers of ours. The unsupervised-bank *inputs* live here (`cr3-v12/inputs`) and are read-only. | **data source only** |

**Rule going forward: launch campaign jobs on sk2 only.** sk1 additionally kills orphaned processes
on ssh exit (logind) and runs a vLLM that rejects `--disable-log-requests` and dies in Triton
codegen — it cost hours for zero results. Do not use it without a specific reason.

## 2. Live lanes on sk2 (verified by pgrep, not assumed)

| lane | pid | state |
|---|---|---|
| ~~lane_t1_v3.sh~~ | — | DRAINED 2026-07-28: 8 of 10 cells done; the 2 pupa cells are UNRUNNABLE (GLM-wired metric judge, HB154) and were killed after burning 2.2h |
| `lane2.sh osl` — OSL ladder | 1311630 | RUNNING, on Qwen3-4B (final rung) |
| `mipro_refill.sh` | 398357 | WAITING for the T1 lane (re-runs the 2 cells burned by the optuna bug) |
| `table_omnibus.sh` (HB150 L1) | 2592654 | WAITING for T1+refill, then per-bench ALL-candidate k=5 rescore → retires every † in Table 1 |
| `controls_v2.sh` (HB150 L2) | 2592656 | RUNNING gpu2: hover 5-arm PAIRED (adds foreign_shuffled + shuffled_entity), then hotpot-shuffled-8B |
| `hover_ceiling.sh` (HB128) | — | **COMPLETE** |
| `hb124_full.sh` (4-arm) | — | **COMPLETE** |
| `paperexact_arms.py ifbench --arm unitrecomb --run-tag solwide` | 50630 | **NOT LAUNCHED BY THIS LINEAGE.** budget 60000. Leave alone; do not quote its output without provenance. |

## 3. Result scoreboard — what is quotable TODAY

Same-session paired, k≥3, bootstrap on mean item-level delta. **Nothing single-pass is quotable.**

**Paper sync status: Table 1, Fig 3 captions, appendix figure updated in main.tex as of 23:05Z (submodule commit 5f2c921). PDF = 7 pages.**

| bench | M_ω vs GEPA | status |
|---|---|---|
| hotpot | **+.235** [+.192,+.281] uniform session (also beats Merge +.246***, MIPROv2 +.207***) | WIN — GEPA shipped the SEED here |
| livebench | +.006 [−.015,+.028] p=.297 uniform session (GEPA .699 / M_ω .705) | NO SEPARATION — old +.092*** was a LOAD ARTIFACT (HB168); never re-quote |
| hover | **+.086** [+.063,+.110] canonical uniform session (vs Merge +.040**; MIPROv2 not measured) | WIN — earlier session +.100 now the footnote |
| aime | +.007 [−.020,+.033] uniform session | NO SEPARATION (print policy HB162: range +.007..+.091 across sessions, never headline +.091) |
| ifbench | **+.040** [+.012,+.069] p=.0029 uniform session | WIN (session-sensitive: earlier session +.020 n.s.; both reported, HB165) |
| pupa | −.001 [−.032,+.031] | TIE, final |

**W = 3 of 6 (hotpot, hover, ifbench); aime + livebench no-separation (uniform sessions); pupa tie.**

### The mechanism result (hover, all arms appended to the bare SEED)
| arm | mean | vs seed |
|---|---|---|
| seed | .4580 | — |
| **native units** | **.5535** (20/20 above seed) | **+.0955** |
| shuffled (word-scrambled) | .4977 | +.0396 |
| foreign (count-matched) | .4568 | −.0012 |

Decomposition: **vocabulary priming +.0396**, **propositional content +.0558**, generic text **0**.
Same-bench anchors (different session): GEPA .4640 · inhouse .5587 · M_ω .5640.

### Unsupervised bank (272 metrics)
Median achieved **.825**; **calibrated** missing-value ceiling **.863** (gap .033, 96.1% hold-out
coverage); rank certificate P(fresh beats best) ≤ 1/(N+1) ≈ .0016.

## 4. Retracted — do NOT re-quote

| claim | why |
|---|---|
| "zero search" | the pool proposer is shown REAL TRAINING EXAMPLES (HB144) |
| "units GEPA mined" | the pool is **88% LLM-proposed**, 12% trajectory (HB133) |
| "screening panel is circular" | `won_8b` is greedy-assembly membership, NOT a threshold on `delta_8b` (HB148) |
| "the pool sets the ceiling" | `inhouse` reaches the same band with **no pool at all** (HB127b) |
| ceiling gap .0167 | failed hold-out (93.4%); calibrated value is **.0333** (HB145) |
| "units are not individually causal" | battery was underpowered, not null (HB120b) |
| any tau=0.02 bank number | global threshold sat below many metrics' minimum (HB138) |
| max draw .5933 · single-pass `best_test` | max over noisy single passes = winner's curse (HB121) |

## 5. Open, ranked

1. **Table 1 comparability — IN FLIGHT** (`table_omnibus.sh`): per-bench all-candidate k=5
   same-session rescore; when it lands, replace every † cell and drop the daggers.
2. **scrambled-foreign + shuffled_entity — IN FLIGHT** (`controls_v2.sh`, paired subsets;
   preregs frozen in HB150).
3. **hotpot-shuffled-at-8B — IN FLIGHT** (same chain, second stage).
4. ~~pool↔test entity-overlap audit~~ — **DONE (HB149): leakage REFUTED** (native 3% bigram
   overlap vs foreign 0%, word-overlap .31 vs .27 — cannot produce +.097).
5. **selection-regret** — GEPA's explored candidates were never retained, so closing this needs a
   fresh GEPA run with candidate logging.
6. ~~Appendix missing-mass plots~~ — **DONE**: `gen_fig_appendix_mass.py`, all five benches at two
   declared species granularities; in the PDF as fig:appendix-mass. (An LLM-clustered semantic
   version would need a Sonnet+ judge per standing rules — the deterministic version is the
   conservative stand-in.)

## 6. Standing methodology rules earned the hard way

- **Verify ARTIFACTS, never exit codes.** Three separate failures today returned `rc=0` while
  producing nothing (vLLM in the wrong venv; a relocated `EM/F1` import; missing `optuna`).
- **A shared control needs ~√K× the passes of each treatment arm** — its error enters every
  comparison and never averages down (HB130).
- **Never compare across sessions.** One invocation, one server, one fingerprint.
- **A settled cell is re-measured only if the INSTRUMENT changed**, never because the result is
  inconvenient.
- **Check the code before asserting how a variable was computed.** Two retractions today came from
  guessing (HB144, HB148).
