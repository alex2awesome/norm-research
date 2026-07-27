# 70B rescore semantics audit — status check turned provenance sweep

*2026-07-02 afternoon. Trigger: user asked "how is the current process going, any errors, how much
left?" The humor sweep and grid chain checked out healthy; auditing the 70B leg surfaced a
semantics mix inside `outputs/r3_cw/aligned_70b_orbit` and led to a full provenance panel. All
paths on sk3 under `/lfs/skampere3/0/alexspan/outputs/r3_cw/` unless noted.*

## The behavioral provenance test (own-vs-cross M_i spearman)

Per aligned dir: own = spearman(aligned `M_i`_k, src-8B `M_i`_k); cross = median over j≠k.
Level-dispatch-bug files measure the WRONG metric → own ≈ cross (coin-flip win rate). Clean files
→ own decisively > cross.

| dir | n | own med | cross med | gap | own>cross | verdict |
|---|---|---|---|---|---|---|
| aligned_3b_orbit (v1) | 41 | .541 | .546 | −.005 | 20/41 | bug-era (neg control ✓) |
| aligned_8b_orbit (v1) | 41 | .730 | .735 | −.002 | 20/41 | bug-era ✓ |
| llama8b_to_llama3b | 41 | .507 | .505 | +.001 | 21/41 | bug-era ✓ |
| llama8b_to_llama70b_fp8 | 8 | .316 | .321 | +.016 | 5/8 | bug-era ✓ |
| llama8b_to_qwen122b | 14 | .330 | .351 | −.005 | 6/14 | bug-era ✓ |
| aligned_8b_orbit_v2 | 46 | .976 | .755 | +.221 | 46/46 | clean, native |
| aligned_3b_orbit_v2 | 41 | .641 | .548 | +.081 | 37/41 | clean, native |
| aligned_qwen_orbit_v2 | 57 | .476 | .343 | +.123 | 55/57 | clean, MI-ONLY |
| aligned_70b_orbit_v2 | 54 | .619 | .430 | +.163 | 53/54 | clean, MI-ONLY |
| **aligned_70b_orbit, 30 pre-10:21 files** | 30 | .623 | .438 | +.160 | 29/30 | **clean — NOT bug-era** |

The 30 files in the running full-rescore dir were suspected bug-era (they lack `target_desc`); the
panel exonerates them — written post-level-fix, pre-`target_desc` patch. Byte-level diffs then
showed what they actually are:

## What each dir actually contains (byte-diff vs src)

- `aligned_70b_orbit_v2` (54) and those 30 files: **sigs byte-identical to src-8B**
  (`--retarget-mi-only`, the user-approved Option 1) → 8B signatures + genuine 70B orbit M_i.
  **AND all their `*_forminv.json` are byte-copies of the src 8B jsons** (all stamped 02:53) — the
  dirs carry NO 70B form-fragility data. Ditto all 57 qwen_v2 forminv jsons.
- `aligned_70b_orbit/metric37` + whatever the 10:21 job writes: TRUE full rescore — sigs
  re-executed on 70B (corr vs src ≈ .47), real 70B forminv. Measured ~3 h/metric
  (VLLM_GPU_MEM_UTIL=0.55, GPU2, ~775 criteria × 300 probes each).
- `aligned_8b_orbit_v2`: retarget-mi-only but src executor == target ⇒ copies are semantically
  correct (the sanctioned same-executor orbit upgrade). CW cert numbers untouched.
- `aligned_3b_orbit_v2`: native full rescore (sigs corr .61) but **0 forminv jsons** → the 3B
  forminv pass is still owed (Task #8).
- metric26 (the one own<cross case): v1-copy vs v2-copy M_i spearman **+1.000** (greedy decoding ⇒
  deterministic replicas) → its low src-agreement is a real 70B↔8B disagreement, not corruption.
- CW src forminv jsons: 57/67 are summary-only (pre-`pairs` patch); gate fields present, so the
  8B form gate stands; per-pair decomposition only exists for the 10 newest.

This is the operational face of the measured shared-Ω confound
(`2026-07-02__shared-omega-retarget-validity-review.md`: D/C = 0.96, cross OPT_Ω ≈ 0.4× native):
**MI-ONLY dirs are valid for the M_i staircase only — never for B_E / certs / form-fragility.**
(Supersedes the `r3cw-data-catalog` note's "aligned_70b = OK/valid" line.)

## The hazard fixed

The 30 MI-only files sat inside the FULL-rescore out-dir: `--skip-existing` would have permanently
blocked native versions of metrics 0–36 and left one dir holding two semantics with no marker.
Actions taken:

1. **Relocated** the 30 to `aligned_70b_orbit/_mi_only_dupes/` — lossless: all 30 are
   byte-duplicated in `aligned_70b_orbit_v2` (which also carries `target_desc`).
2. **`CATALOG_OVERRIDES.json`**: 70b_v2/qwen_v2 → MI-ONLY with exact valid/invalid uses;
   `aligned_70b_orbit` → FULL-partial; `llama8b_to_*` → restored the precise SUSPECT-M nuance
   (sigs/B_E valid, M_i wrong) now backed by the own-vs-cross proof.
3. **Armed `rescore70b_chain.sh`** (PID 1121580 on sk3): waits for pass-1 (PID 199503) to exit →
   GPU2 ≥150 GB free gate → pass-2 fills 0–36 with the captured env (`env_70b.sh`).
   **Disarm it if the fork below lands on Q1 / Q3 / head-only.**
4. **Provenance patches** (sk3 `rescore_executor.py`): `savez` now records `retarget_mi_only`;
   forminv json now embeds `executor`. Future copies are self-labeling. (The laptop copy of this
   file is STALE relative to sk3 — sync debt; `catalog_check.py` md5-identical both sides.)

## The open fork (user decision — refreshed costs)

The running full rescore is Q2-maximal — the path the shared-Ω review already called infeasible.
It was resurrected by the wedge-recovery watcher, not chosen at the fork. Measured rate ~3 h/metric
⇒ pass-1 (metrics 38–66) ≈ 3.5 d more, pass-2 (0–36) ≈ +4.6 d ⇒ native-67 lands ~Jul 10–11.

- **Q1** consistent shared-8B-Ω ladder — cheap; answers "reach of the 8B articulation process";
  peaks at 8B, not a capability monotone.
- **Q2 / Option B** head-only native (top ~120 census criteria, 70B + Qwen, ~25–40 h total) — the
  measured confound justifies it; needs a small `--head-only` subsetting patch to
  `rescore_executor`. Full-rescore metrics finished by decision time become free calibration
  anchors (head-only OPT_Ω vs full OPT_Ω on the same metrics).
- **Q3** accept the lower bound — free; 70B/Qwen OPT_Ω stays loose; M_i/recovery emphasized.

Recommendation: **Q2/Option B on GPU2** (stop pass-1 then, keep its files as calibration, disarm
the chain). Until decided, pass-1 keeps running — its output is valid under every branch.

## Status at write time (~14:55 PT)

Humor Face-1 sweep 52/60, ~8 min/metric, exits ~16:00 → `humor_grid_chain.sh` auto-fires (catalog
gate → 70B-writer messages → 1B/3B/8B readers → report; GPU7; ~4–7 h). **No errors anywhere**: the
"3 err/429 hits" in the sweep log were `…429.28 toks/s` substrings in progress bars. Watchers
armed: local sweep-exit poll (3-min cadence) + grid-report poll (20-min). Humor cert re-run at 60
+ `audit_certificate` D1, and grid D2 with `--ref-executor 8B`, both queued on wake.
