# Battery adversarial test expansion — hazard register (2026-07-25)

Context: user directive "10 tests per battery arm, varied and adversarially constructed"
(GLM endpoint down, so confidence-building moved from synthetic calibration to the unit
layer). Seven Sonnet agents wrote one file per arm; every finding below was verified by
the authoring agent against the live code and is ENCODED as a passing documented-hazard
test (no source file was modified — passes.py sha 260689d5 and run_reason_first_pass.py
sha 404314db verified unchanged after the wave; stats.py 2ed8989c, run_variant_pass.py
c07d9084 recorded here as the audited baseline).

## Suite state

103 tests green (was 33): test_adv_{tf_planner,exclusion,negation,composition,holistic,
confidence,reason_first}.py = 10 each, plus test_battery.py 10 / test_stats.py 15 /
test_instruments.py 8. Mutation check 4/4 KILLED (leak median→mean; not_gap reversed-
target sign dropped; composition adverse min→max; acceptance floor < → <=), sources
restored sha-identical. Runner: `python -m pytest methods/tacit_channels/battery/tests -q`.

## Tier 1 — fix before W1c (touches the gate or future scoring; needs sign-off + dated addendum)

| # | Hazard | Location | Failure mode |
|---|---|---|---|
| 1 | Acceptance gate PASSES when every matched tf row is zero-variance (all ρ NaN; `nan < floor` is False) | run_variant_pass.py:97 | a fully degenerate scoring path prints "ACCEPTANCE PASSED" |
| 2 | `tf_answers_from_grid`: NaN score → `>= .5` is False → "confidently NO" | run_reason_first_pass.py:77 | unscored items silently become NO answers for the confidence elicitation (W1a's fail-closed write guard makes this latent, not active, for our own grids) |
| 3 | `leak_stats` with empty `cross_tf` returns leak_specific == raw leak_self with no flag — exactly the headline the audit banned | stats.py:51 | single-cell or misconfigured tallies silently un-correct the generic-factor confound |
| 4 | `conf_acc_stats`: constant AGREEMENT vector → conf_acc_corr = NaN while degenerate_confidence stays False | stats.py:139-143 | tally-level mean over cells silently poisoned |
| 5 | `holistic_residual` guard bypasses: all-False fit mask or y_std_floor=0.0 skip the degeneracy guard (NaN/0 < floor is False); one NaN in X poisons its whole column via mean(0)/std(0); unnamed_share unclipped (can be >1 or ±inf under verdict "ok") | stats.py:95,98,113-116 | garbage numbers under an "ok" verdict |

## Tier 2 — interpretation constraints (bind on tallies/prose; no code change required)

- **not_gap ≈ 0 is ambiguous**: a pure-noise agent matches a perfect NOT-applier. Gap is
  only interpretable when tf_rho is materially positive — tallies must report tf_rho
  alongside (they do; now enforced by test).
- **unnamed_share conflates nonlinearity with unnamed structure**: y = interaction of two
  named constructs reads as ~100% unnamed (linear ridge). NEW caveat, same family as the
  span-relative lower-bound caveat; both bind on Act-3 claims.
- **parse_confidence biases** (W1b provenance frozen — document, don't patch): question-echo
  "0-100? ... 90" parses 0; "9.5 out of 10" → 9/100; leading minus dropped ("-5" → 5);
  4+ contiguous digits → NaN (no clamp). A v2 parser for W1c+ needs a dated addendum.
- **composition_rho footguns**: any mode string ≠ "min_z" silently uses member_refs[0] as
  the whole reference (order-dependent); a constant member ref clamps the soft-AND blend
  (zrank(const)=0 vector); partial-NaN composed vectors slip the filter and return
  plausible wrong values.
- **spearman (channels/common.py:91-97)**: a single NaN element gets an ordinary extreme
  rank — silently perturbs ANY battery statistic. Shared primitive; blast radius beyond
  the battery. Candidate for a repo-wide guard, separately reviewed.
- **two-stage splice**: str.replace hits ALL occurrences of the answer-instruction sentinel
  (item text or rationale containing it corrupts the prompt; truncation-dependent);
  passes.py::assemble_reason_first_tf vs runner's reason_first_tf_prompt are DRIFTED
  duplicate implementations — the runner never imports the passes.py version.
- **confidence_scale_valid**: orientation-sensitive (transposed matrix flips the verdict);
  n_unique inflatable by float noise (median_cell_std is the load-bearing check).
- **plumbing**: build_single_stage_rows accepts a bare string as `variants` (substring
  containment footgun); load_composed_pairs crashes on explicit JSON null;
  run_confidence zip-truncates when the answers vector is shorter than texts and silently
  drops rows whose (cell,form) key is missing.

## Disposition

**Tier-1 FIXED 2026-07-25 (user sign-off).** All five items patched; six documented-hazard
tests flipped to assert the fixed behavior; suite 103/103 green; mutation check re-run
4/4 killed. New shas: stats.py bf7e4473, run_variant_pass.py 39804497,
run_reason_first_pass.py aeb57205; passes.py UNCHANGED 260689d5 (no wording touched).
Sha re-record + full fix list: dated addendum in notes/2026-07-23__battery-prereg.md
(2026-07-25 entry). W1a/W1b scored artifacts unaffected (produced under prior shas; v0
tallies compute inline). Tier-2 stays as documented constraints enforced by the suite.
