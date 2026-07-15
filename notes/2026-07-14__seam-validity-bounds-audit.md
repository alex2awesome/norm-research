# Audit of the validity-bounds implementation — what stands, what retracts

*2026-07-14. Fable audit of the run reported complete in `outputs/metric_seam_pilot/validity_bounds_v1/`.
Method: three independent audit passes (matched-info harness; Shapley + DAG-cut code; tests/notebook/git)
plus direct verification of the a12 context arm. Every number below was recomputed from artifacts on
disk, not taken from the report. Paste to Codex.*

---

## Scorecard

| result | verdict |
|---|---|
| a12 context arm (design + data) | **SOUND** — headline κ's are the wrong readout, conditional matrices stand |
| a12 G2 proxy-trap gate | **SOUND** — behaves exactly as designed |
| a34 exact Shapley (evidence .801 / computation .118) | **SOUND** — bit-exact reproduction, pinning artifact ruled out |
| a34 matched-information C>B | **NOT RESOLVED** — Spearman-on-tied-target; tie-robust CI crosses zero |
| a34 DAG-cut "2.338-bit bound" | **VACUOUS — RETRACT** — the number is exactly H(target) for any cut |
| pipeline-inversion replay (6/6 kills) | **SOUND** (retrospective, as labeled) |
| K=3 capture–recapture (13.8) | arithmetic verified; width pilot only, as its claim_limits say |
| "notebook end-to-end + 28 focused tests" | **FALSE as stated** — 8 focused tests; core modules have zero |

## 1. a12 context arm — sound, but report the matrices, not the κ's

Verified directly: `compile_request` embeds the full line-numbered document
(`verifiers/math_a12_context_contract.py:46-55`), the compiler enforces a `{ctext, item_key}` allowlist
so no symbolic labels can leak (`compile_math_a12_context_requests.py:37`), and the prediction was
frozen in the manifest before any call. The smoking gun reproduces: `train_0001.pair-004`
(`B² =?= SXS⁻¹SXS⁻¹`), which the document-blind arm called `violated` for lack of context, is now
`asserted_identity_step / not violated`.

**Do not headline κ = −.052.** The symbolic verifier's `not_applicable` means "the bounded parser
cannot decide" (`math_a12_symbolic.py:8`) — a coverage abstention, not an individuation judgment. The
applicability κ therefore mixes two unrelated things: role reclassification (the finding) and SymPy's
parse coverage (156 pairs Sonnet calls asserted steps that SymPy simply can't read). Likewise
κ = .082 for polarity is a marginal-skew artifact (context says `violated` only 2/47 times).

**Report instead:**
- Of 111 symbolically-applicable pairs, **64 (57.7%) reclassified** by full context as
  definition (21) / equation-to-solve (15) / hypothesis (11) / other (17).
- Polarity matrix on the 47 joint asserted steps: **23/23 closed-form tautologies transport
  perfectly; 22/24 symbolic "violations" are context-resolved; 0 flips in the reverse direction;
  2 both-violated** (`train_0062.pair-003`, `train_0067.pair-003` — candidate genuine errors, worth
  a manual look).

**Two gaps to close (cheap):**
- **G2 controls were never run through the context arm.** `run_math_a12_g2_controls.py` exercises only
  the symbolic verifier. Run the same 6 plants through the Sonnet context arm (6 calls): passing the
  4 traps while catching the 2 true violations is the positive control the occasion claim needs.
- **13/443 rows dropped non-randomly** (11 "multiple JSON fences" + 2 empty; visibly skewed toward
  complex LaTeX). Retry those 13 with a fresh seed — do not touch the parser or `SYSTEM_PROMPT`.

## 2. Matched-information C>B — not certifiable as read; power fix is on disk

The harness is clean: joins digest-checked, bootstrap correctly paired, numbers reproduce exactly
(A .278 / B .661 / C .740). The problem is the statistic. **54/98 items are tied at the target's
ceiling** (both Gemma passes 10/10). Recomputed both ways, 5,000 paired draws:

| C−B statistic | observed | 95% CI | 90% CI | P(Δ≤0) |
|---|---|---|---|---|
| Spearman ρ | +0.079 | [+0.013, +0.152] | [+0.024, +0.139] | .011 |
| c-index (tie-robust) | +0.028 | [−0.012, +0.070] | [−0.005, +0.063] | .083 |

Under our own standing rule (threshold-free statistics for any cross-family comparison — prompt vs
code is cross-family), the c-index row is primary, and it says **not resolved**. On the 44 non-ceiling
items alone the sign flips (B .40 > C .32, noise-level n). The readout's `reading` field should change
from "algorithmic execution advantage" to its own third branch ("B and C are not resolved").

Secondary disclosures for the eventual writeup: B is single-pass with no reliability estimate against
a deterministic C (attenuation mechanically favors C); C additionally consumes the two precomputed
LLM fields (`closest_art`/`distinguishing`) B never sees (ablation-worth ≈ .0001, so not the driver);
the target judge saw evidence capped at 2,200 chars / top-2×2 while B and C see the full ~9.6k
`pa_features` record.

**Fix (resolves it either way):** the target and features exist for all 250 items but only the
100-item test split was used. Extend the B arm to all 250 × 2 passes (~300–400 Sonnet calls),
pre-register the c-index as primary, stratify ceiling vs non-ceiling. Keep the current run as arm 1.

## 3. DAG-cut certificate — retract the number, keep the idea

Verified independently: `I(cut; target) = H(target) = 2.3382868708389988 bits` **exactly, for every
one of the 8 cuts**. Cause: `_plugin_mi` (`ws4/patents_pa__a34/build_readouts.py:276-281`) applies no
discretization — raw node outputs are JSON-canonicalized into categorical keys, and `norm` (the ~5.8k-char
normalized document, which `final_combine` needs directly, so it sits in every minimal cut) is unique per
item. Any cut is item-injective at n=149, so H(target|cut)=0 and the "bound" is just the target's entropy.
The "DPI margin .831" is H(target|score) — a statement about score quantization (39 distinct values), not
about information flow. The existing caveat ("descriptive, not a population bound") addresses
generalization, not this in-sample saturation.

This is the **third degenerate-input incident** in this stream (constant targets, tied vectors, now
injective cut variables). Strike the "empirical minimum-information bound 2.338 bits / DPI margin 0.831"
language from `validity_bounds_v1/report.md`, `readout.json`, and notebook §24. A meaningful redo needs
(a) equal-mass binning with bins ≪ n for continuous outputs, (b) `norm` excluded or its cardinality
flagged, (c) the bound quoted **only if it lands strictly below H(target)**, with H(target) printed
beside it. Cut enumeration itself is correct (8 inclusion-minimal cuts independently re-derived; spot
checks pass) — the graph machinery survives, the MI estimator does not.

## 4. Shapley — survives adversarial audit; quote freely

Independent re-derivation reproduced `v(full)=0.735525427387259`, `φ(prior_art_lookup)=0.5720010367129067`,
and both op-class masses bit-exactly. The pinning concern is refuted with measurement: 216/2048 coalitions
go constant and get pinned, but 83% of prior_art_lookup's φ comes from transitions between genuinely
non-constant coalitions. Exclusions (sink, root `norm`) are structurally forced and don't bias the
evidence/computation split. One caveat to carry in prose: the 0.801 "evidence mass" cannot include the
sink's own (definitionally unmeasurable) contribution.

## 5. Process debts

1. **"28 focused tests" is false.** 8 focused tests exist, all on adjacent modules; the 13 core new
   modules (`validity_bounds.py`, both a12-context files, all three matched-info files, the G2/capture/
   replay runners, `build_validity_bounds_summary.py`) have **zero dedicated tests**. 28 is the count of
   Jul-14 `.py` files. Minimum bar: regression tests pinning the recomputed numbers above.
2. **Nothing is committed.** All Jul-14 code is untracked; `outputs/` is git-ignored (fine), but the
   code must land in git before any further iteration.
3. `build_validity_bounds_summary.py` takes no argparse — any invocation runs `main()` and overwrites
   its outputs. An audit agent tripped this; regeneration was verified byte-identical (deterministic
   builder, inputs unchanged), so no harm done — but add an argparse + `--output` guard.
4. Notebook §24 prose discusses only the old retracted κ = .445/1.0 and never surfaces the corrected
   conditional numbers; update alongside the retraction edits.

## Order of work

1. CPU-only, zero model calls: strike the cut-bound language; flip the matched-info `reading` to
   "not resolved" with the c-index table; replace κ headlines with the conditional matrices in
   readout/report/notebook §24; commit all code; add the pinned-number regression tests.
2. ~6 calls: G2 controls through the context arm.
3. ~13 calls: retry the dropped a12 rows.
4. ~300–400 calls: matched-info power fix (all 250 items × 2 B-passes, c-index primary, pre-registered).
5. Only after 1–4: replication of the context/occasion design on a second technical criterion, and the
   corrected cut certificate if we still want it.
