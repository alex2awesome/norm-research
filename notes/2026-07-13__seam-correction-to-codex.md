# Correction to Codex — metric seam, code-review codability lane

*2026-07-13. Paste this to Codex alongside `notes/2026-07-13__codability-verifier-roadmap.md`.*

---

Three of your conclusions on this lane are wrong, and one of mine is. Corrections first, then what to do.

## 1. RETRACTED: the preregistered ρ<0.40 "instrument limit" branch

I authored that branch, and it fired on an invalid basis. **Do not write it up. Do not cite it.**

The ceiling arm (`full_executable_contract`) discloses the complete program source and asks the model to
"simulate this program on the diff and report the score it would produce." But these programs compute
`math.exp(-mean_density * 25.0)`, `n_cosmetic / len(eligible)`, `1.0 - l1 / 2.0`. Simulating them requires
exact enumerate-count-arithmetic over 200-line diffs, accurate enough that the *rank across 125 items*
survives. **That arm measures arithmetic execution, not articulation transport.** A low ρ on it says nothing
about whether the seam is crossable. It is not an upper anchor and must not be read as one.

My error, not yours. But it means the branch it triggered has no evidentiary standing.

## 2. RETRACTED: ρ = 0.149 as a codability estimate

It is an invalid statistic, for a reason that has nothing to do with the model.

**The targets are constants.** Ten of the eighteen cells have code targets whose top tercile *equals* their
bottom tercile — a131 mode 0.97, a401 0.95, a1 0.94. `a1_simplicity_yagni` returns exactly **1.0 on 102 of
125** held-out items. Spearman on a 94%-tied vector is a tie-density statistic, not a reconstruction statistic.

The cause is structural, not a bug: the coded "metric parts" are **violation detectors** — absence of TODOs,
absence of YAGNI flags, absence of `foo`/`bar`/`data` names — and they were run on a corpus of **merged,
reviewed PRs**, where review has already removed the violations. A violation detector applied to post-review
code is a constant function. *You cannot measure the codability of a constant.*

Note that the only program with real held-out variance — `a8_small_focused_changes` (mode fraction 0.02, 116
distinct values, scores all 125 items) — measures diff **size**, and is not among the 10 mapped programs.

## 3. The signal was there and the readout buried it

Arithmetic-free **tercile AUC** on the 8 cells that do have spread:

| cell | tercile AUC |
|---|---|
| a0 | 0.720 |
| a37 | 0.711 |
| a92 | 0.710 |
| *median over the 8* | **0.573** |
| descriptive item-bootstrap 95% CI | **[0.502, 0.678]** |
| AUC below 0.5 | **2/8** |

The earlier 0.547/zero-inversions summary does not reproduce under the same estimator and is superseded by
the table above. Report tercile AUC (threshold-free, rank-based) with the target's tie structure disclosed —
mode fraction, distinct-value count, n per tercile. **Never Spearman on a tied target**, and never a ρ without
the tie structure printed next to it.

## 4. NOT TRUSTED: the 50/90 → 27/90 → 18/90 funnel, as a codability claim

Every step of that funnel gated on **coverage**. No step gated on **discrimination**. A funnel that selects
on coverage and not on variance selects *for constants* — which is exactly what it delivered.

The repo already contains the gate this lane needed and did not run:
`methods/metric_seam/battery/contract_check.py` — planted pos/neg probe separation ≥75%, zero inversions,
`min_std ≥ 0.05`, `max_frac_at_mode ≤ 0.85`, TRAIN-only execution, ≥90% completeness, plus a sentinel guard
against candidates gaming probe mode. The lane instead used `hierarchy_train_gate.py:27`, whose bar is
`min_coverage: 0.05, min_unique_scores: 2` — which admits a program returning 1.0 on 124 of 125 items.

Wiring `contract_check.py` into this lane kills roughly 4 of 16 programs **before any model call**.

## What IS trusted

Raw data and provenance. I independently re-parsed all 4,500 responses and reproduced the analyzer's median ρ
to four decimals (0.14643). Items are label-free, held-out is sealed, sources are digest-bound, there is no
outcome leakage. The artifacts are sound; the conclusions drawn from them are not.

Also standing, from the earlier fix: the "GLM-5.2 output-contract failure" was a **fence in the parser**
(`json.loads` with no fence unwrap, plus `strict=True` rejecting literal tabs in evidence spans — which drops
rows *non-randomly*, selecting against tab-indented languages). 4,442/4,500 recovered on CPU, zero model calls.
ρ is **defined and low**, not undefined. Fix at the parser, never at `SYSTEM_PROMPT` — changing the prompt
changes `request_sha256` and invalidates every response already collected.

---

## What to do

Follow **`notes/2026-07-13__codability-verifier-roadmap.md`**. It supersedes the "Continuation priority order"
section of `notes/2026-07-13__seam-ceiling-arm-handoff-to-codex.md` (that doc's *corrections to the record*
still stand; its *forward plan* does not).

The short version: components become **verifiers, not scorers** — `(applies: bool, verdict: satisfied|violated,
witness: spans)`, **no floats**, so there is no weight to hand-set and no feature-extraction to guess at.
Verifiability gets an operational gate in the RLVR sense: two *independently authored* implementations (AST-
deterministic and schema-constrained LLM) must agree on held-out at chance-corrected **κ ≥ 0.80** *and* point
at the **same witness spans** — agreement with disjoint witnesses is coincidence, not verification, and counts
as a certificate failure. A unit that fails the certificate is **uncertified for the frozen corpus, verifier
class, and budget**, which is a result to report rather than a program to tune indefinitely. It does not by
itself establish tacitness.

Immediate order of work — steps 1–4 are CPU-only and cost **zero GLM**:

1. Retract the instrument-limit branch in the runbook (`notes/2026-07-10__seam-agentic-program-runbook.md`,
   the 2026-07-13 entry) and in the notebook. State the three defects above. You own that file; I did not
   write to it, to avoid a concurrent-write race.
2. Re-score the existing 18 cells with tercile AUC + tie structure. Data is already on disk.
3. Replace `hierarchy_train_gate.py` with the roadmap's §4 gate; wire the lane into `contract_check.py`;
   re-run the funnel.
4. Join the surviving cells onto the Certified Unit Framework units (§Stage 0).

Then, and only then, spend model calls.

## Two process lessons, recorded

- **A smoke test that fails 100% is a STOP signal, not a row to exclude.** The 12:52 transport smoke returned
  2/2 contract errors and was "excluded from analysis"; 4,500 production calls launched 73 minutes later.
- **Audit the harness and the statistic before blaming the instrument.** Both confident verdicts on this lane
  in 24 hours were wrong in the same direction — each blamed the external thing (the model, then the
  instrument) rather than the internal one (the parser, then the target's variance). When a result says "the
  model can't do this," the next move is to check whether *the target has variance*.
