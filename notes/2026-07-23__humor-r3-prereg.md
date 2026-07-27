# Humor R3 four-universe readout — prereg (2026-07-23)

Written BEFORE any R3 corpus scoring (scoring launched immediately after this note;
readout auto-chains on completion). Purpose: make the humor headline confirmatory and
stratum-consistent with the rest of the cross-task table, per the stratum-conflation
audit (block `HUMOR_STRATUM_CORRECTED_AUDIT_20260723`).

## Frozen inputs
- **Universes** (averaged-AUC axis: per-form AUC within each universe, ≥10 pos, then
  averaged; doc universes NEVER pooled): old 5,343 / R1-new 7,996 / R2 8,702 /
  **R3 17,306** = 11,309 quote-verified pos threads + 6,000 negatives sampled by
  md5("r3neg"+thread_id) from the 12,865-thread negpool, pool post-only text,
  len≥30, 6,000-char cap — the exact R2 recipe (R2 rebuild validated byte-identical
  before R3 was built; `humor_r3_prep.py`).
- **y** = ypos_v6 = ypos_v5 ∪ humor_r3_ypos (R3 labels quote-verified at source).
- **x** = frozen MI channels (rel_MI .816), 2,053 full-coverage channels
  (1,142 old-form + 911 new-half), Gemma-4-31B instrument, frozen estimator
  (≥5 forms/metric, Fisher-z pooling, 5,000-perm null, md5 doc split-half rel_auc).

## Canonical headline (declared in advance)
**Detectable stratum** = metrics with averaged auc_mean ≥ .55 (threshold carried
unchanged from three prior independent runs; membership is invariant under the
permutation null since auc_mean is unchanged by within-metric permutation of AUCs).
Report: raw pooled ρ, stratum split-half rel_auc, corrected = raw / sqrt(.816 × rel_auc),
within-stratum 5,000-perm p.
Secondary (reported, not headline): pooled-all (with perm p), multi-universe (≥2),
triple+ (≥3), quad-support (=4), low tail (<.55). Methods footnote: Spearman-Brown
stepped-up corrected.

## Predictions (stated in advance)
1. Detectable raw ρ ∈ [.37, .45].
2. Detectable stratum rel_auc rises above the 3-universe .643.
3. Detectable corrected ∈ [.48, .58] with correction factor ≤ 1.45.
4. Low stratum stays < .10.

## Decision rule
If predictions 1+3 hold → humor enters the cross-task table AT THE DETECTABLE STRATUM
(stratum-consistent quoting). If not → report as-is; no further humor GPU spend.

## Runbook
`humor_r3_prep.py` → `humor_r3_gpu.sh N GPU` (s0→GPU2 s1→GPU3 s2→GPU4 s3→GPU5, exit-code
checked) → `chain_r3_readout.sh` (gates on all 8 score files) → `humor_r3_readout.py` →
`humor_r3_readout_result.json` (+ writes `humor_ypos_v6.json`). All under
sk3:/lfs/skampere3/0/alexspan/mention_auc/.
