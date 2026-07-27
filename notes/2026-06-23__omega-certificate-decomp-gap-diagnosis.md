# Ω-certificate decomposition-gap diagnosis — the R>T "anomaly" fully resolved (2026-06-23, overnight)

**TL;DR.** The creative-writing run on sk3 (PID 52990) **finished cleanly** at 02:31:29 — the z.ai
credit outage only killed the laptop session that was *polling* it; the job ran to completion on local
Llama-8B. The `I(M, M_ω)=0.263 > T` anomaly is **not a DPI violation**. It was three stacked issues, now
all diagnosed; two are fixed in code, the third is a property of the run (weak executor), not a bug.

All numbers below were recalibrated **from the saved npz, no re-scoring** (the persistence fix worked —
`P_cont`/`P_bin`/`prose_prompt` are now in the npz).

---

## The corrected panel (creative-writing, Llama-8B)

| quantity | value | ceiling | verdict |
|---|---|---|---|
| **Decomposition gap** `I(M, M_ω)` | 0.039 | `T_ω`=0.136 (DPI-valid) → R ≤ T_ω ✓ | **UNTRUSTWORTHY** — prose channel collapsed |
| **Selection gap** `I(M_ω, M_s)` greedy/OPT | 0.105 / 0.122 = **0.862** | `T_ω`=0.136; full-set R=0.128 = **94%** of T_ω | **trustworthy & near-saturated** |

The exact within-class certificate is sound: greedy = 0.862·OPT_Ω, OPT brute-forced over all 63 subsets,
every subset R ≤ T_ω.

---

## What the anomaly actually was (three layers)

1. **Scale (fixed in the earlier session).** `I(M,M_ω)` was computed with `vinfo.tvd_mi`
   (2×2-contingency TVD, cap-normalized to [0,1]) and compared against `tvd_transmission.tvd_t` (raw
   per-item-MAD on [0,½]). Different rulers → the 0.263-vs-0.107 mismatch (≈2.46×). Swapped to
   `tvd_recovery` (per-item-MAD, raw scale) → 0.263 collapsed to **0.039**.

2. **Wrong DPI leg (fixed this session).** `tvd_recovery(M_cont, P)` keeps `M_cont` **soft** and uses `P`
   only as a **binary grouping label**. The termwise-Jensen-guaranteed leg is therefore
   `R ≤ T(recovered side) = T_ω = tvd_transmission(M_cont)` — exactly what `tvd_guardrail` is built
   around — and it **holds** (0.039 ≤ 0.136). The leg `R ≤ T_prose` is **not** a valid check: R is built
   on the *binarized* `P` while `T_prose` is the *soft* `P_cont` transmission — two different
   representations of the prose channel, so the soft `T_prose` cannot bound a binarized-label R. The old
   `dpi_ok = R ≤ min(T_prose, T_ω)` fired a **spurious** violation on exactly this mismatch.
   **Fix:** `dpi_ok = R ≤ T_ω`; `T_prose` is kept as a diagnostic only.

3. **Degeneracy — the real signal (now flagged).** The deeper reason `T_prose` is tiny: the GEPA prose
   prompt scores **~0.97 for essentially every item** (mean 0.973, MAD 0.013). Median-splitting a
   near-constant channel manufactures an **arbitrary balanced 50/50 label**, so any downstream R against
   it is spurious. **Fix:** flag `prose_collapsed`/`omega_collapsed` when a channel's soft transmission is
   below the noise floor, and mark the decomposition number `trustworthy=False` rather than reporting a
   clean R.

   *The 0.02 cutoff is principled, not arbitrary* — it agrees with the permutation-floor test:
   - prose: raw_T=0.0130, perm_floor=0.0101, **sig=0.0029** → at the noise floor (degenerate)
   - ω: raw_T=0.1363, perm_floor=0.0969, **sig=0.0394** → significantly above floor (real signal)

---

## The substantive finding (worth your read)

The degenerate prose prompt GEPA produced is a **3-point mechanical checklist over near-universal
features**: *"Evaluate using ONLY these mechanical checks… Step 1: if fewer than 3 sentences, score=0 and
stop. Step 2: 2+ adjectives → 1pt. Step 3: a subject-verb sentence → 1pt. Step 4: any one of
character/emotion/location/time → 1pt."* Almost every creative-writing excerpt clears all three → ~0.97
for all. The prose prompt is degenerate **by design**.

But the **decomposed criteria `C(Ω)` discriminate far more** than the prose they came from:

| channel | MAD across items | range |
|---|---|---|
| prose `P_cont` | **0.013** | [0.915, 0.993] |
| all-criteria `M_ω` | **0.136** | [0.166, 0.967] |

So decomposition into 6 explicit atomic checks **did not lose signal — it ADDED discrimination** the
"mechanical-checks-only" prose suppressed (e.g. the emotion/location/temporal checks vary across short
excerpts; recompiled separately they fire variably instead of collapsing to 3/3).

**Implication for the theory:** "decomposition gap `I(M, M_ω)` = how much of the prose survives
atomization" is **mis-framed when the prose is the weaker channel**. Here prose ⟂ criteria are just two
different behavioral objects and the prose one is near-dead. The two-quantity design (§6.7a′) still holds,
but the *interpretation* of `I(M, M_ω)` assumes prose is the richer standard — false under a weak executor.
Worth a sentence in §6.7a′ noting the decomposition gap is only interpretable when **both** channels clear
the degeneracy floor.

---

## Why this run can't give "better numbers" yet — and what would

Not an estimator problem anymore: the valid DPI leg holds and the selection certificate is clean. The
blocker is the **executor**. Under Llama-8B the prose prompt collapses to ~0.97 for everything (T_prose at
the noise floor) — the same 8B-collapse the `recovery_trial` REPORT documented on math.SE. Even `M_ω`'s
signal (sig=0.039 over a 0.097 floor) is only modestly above noise.

**To get trustworthy decomposition-gap numbers:** a stronger executor (70B / a real frontier judge) so the
prose prompt actually discriminates across items. No 70B is cached on sk3 and OpenRouter/z.ai are down, so
this is gated on either a model download or credits — **your call; I did not launch anything.**

---

## Code changes made this session (all low-risk, reviewable)

- `experiments/omega_certificate.py` — decomposition-gap block: `dpi_ok` now uses the DPI-valid `T_ω`
  ceiling (not `min(T_prose,T_ω)`); added `prose_collapsed`/`omega_collapsed`/`trustworthy` flags + a
  noise-floor cutoff; `T_prose` demoted to a diagnostic; heavy comment block explaining all three layers.
- Tests: `test_real_test.py` 5/5 pass; full vinfo+orthogonalize suite 23/23 pass (no regressions).
- The selection-gap path and the exact certificate were already correct — untouched.

## Open follow-ups (for when you're back)
1. Approve the `T_ω`-only ceiling + degeneracy-flag change (or adjust the floor).
2. Decide on a stronger executor for a re-run (70B download vs wait for credits).
3. One-sentence §6.7a′ caveat: decomposition gap interpretable only when both channels clear the floor.
4. (Optional) have the certificate **skip / loudly warn** the decomposition gap when `prose_collapsed`,
   instead of printing a number at all.
