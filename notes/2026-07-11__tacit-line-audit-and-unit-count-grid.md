# 2026-07-11 — Tacit-scaling line: audit, 70B prereg execution, and the unit-count grid

User asks (this session): (1) audit the tacit measurement line (prompt isomorphism small->large
models), (2) enumerate remaining work, (3) ADDENDUM: unit-level analysis borrowing M_omega/CUF
theory — "how many MORE units, on average, do we need when we switch from a bigger model to a
smaller model?"

## I. Audit deltas (vs the 2026-07-05/06 notes)

1. **70B prereg (sha 62e4b3f0 + law 92024275) was OVERDUE and had NO eval code path.**
   `name_sufficiency.py` only *writes* the freeze artifacts; nothing anywhere read them back.
   Built `methods/codability/eval_prereg_70b.py` (persistence; rank-AUC + prefix violations +
   permutation p; parametric-law CI rule with the frozen >half-cells rejection protocol).
2. **A 70B pass existed all along for cw+humor** (`grid_fde04ee…npz` = nvidia Llama-3.3-70B-FP8,
   sk3 `r3_{cw,humor}/grid_*_v1`, scored 2026-07-02 for the Face-2 curves). It PREDATES the
   Jul-5 freeze — the freeze text "before any 70B reader pass" is wrong for those domains.
   → cw/humor are CONTAMINATED for prereg purposes; the honest eval = the 7 clean domains.
   The evaluator reports `clean7` and `cw_humor_pre_freeze` scopes separately.
3. **gemma-4-31b grids landed Jul-6 for humor+math but were never harvested** (auc_report.json
   predated them). Harvested tonight. First look (math): gemma-4-31b name .630 / def .610 —
   BELOW gemma-2-2b; treat the checkpoint/serving env as SUSPECT (cf. Qwen2.5-3B anomaly),
   do not use as a family point without a sanity pass.
4. **cw grid dir mixes M_i scorings** (Llama ladder → `aligned_8b_orbit_v2`; 70B npz →
   `llama8b_glm`). Its auc_report must never be refreshed with a single --ref-dir; excluded
   from tonight's refresh.
5. `name_sufficiency_scaling.json` was regenerated Jul-11 01:54 by a bulk notebook re-execution
   (content unchanged in scope: 1B/3B/8B only) — no longer provably byte-identical to the
   artifact behind the Jul-5 note numbers.

Remaining-work queue (priority order): 70B prereg eval (RUNNING tonight); band per-metric join
in build_master (broken, band 0); a-priori §3 items (within-domain eval, split-half reliability
ceiling, LM zero-shot baseline, wave-3 held-outs); kappa-matched B-only cell re-adjudication;
E2 gemma-2 retry legal/grant/cr; T2a/T4/T5b/T6a/T8c (code exists Jul-6 — `scaffold_orbit_probe`,
`length_control_probe`, `probe_ppl`, `def_source_cross` — run status unverified); notebook debt
(PART VIII capstone table prose-only; crosstask notebook prints stale "awaiting 70B" line);
coordination with the z×a ladder spec (separate program, do not conflate).

## II. Unit-count grid (NEW instrument, addendum)

Question: articulation QUANTITY (in Omega-units) needed per reader scale, complementing the
rung grid's articulation TYPE axis.

Design (`methods/codability/unit_count_grid.py`):
- Units = CUF leaf partition (address_lattice regexes: `_segment_sentences` + `_CLAUSE_SPLIT`,
  min_words 3) of the metric's VERBAL dossier content — full_rubric + definition + explanation
  leaves from the byte-identical grid `messages.json`, deduped, ordered rubric-first.
  humor: ~6.6 units/metric (max 10); math: ~9.3 (max 13). Direct bank join rejected: CUF bank
  is R2-granularity, name-matches only 17/60 (humor), 3/21 (math) of grid R3 metrics.
- Rungs: u0 = name (byte-identical to rung grid `name`); uk = name + first k units;
  fk = name + length-matched inert filler (CUF FILLER_BANK) — mechanical length control.
- Same instrument everywhere: same 300 probes, same 8B-executor M_i refs (llama8b_glm), forms=3
  orbit on u-rungs, Mann-Whitney AUC readout (grid_auc_report conventions), exemplar mask.
- One reader per PROCESS (in-process engine #2 OOMs against engine #1's unreleased memory —
  reproduced tonight, fixed; npz-exists skip guard added).

Readout (`methods/codability/unit_deficit_report.py`): per (reader, metric) best-so-far
envelope of AUC(k); k*(tau) = min units to reach tau; Delta-k(small,big;tau); horizontal shift
= mean displacement in units over the shared reachable AUC range [0.52, min(max_s, max_b)];
filler checks; by-tag breakdown; metric-level bootstrap CIs. 8B flagged SELF throughout.

## III. Interim results (Stage A: Llama 1B/3B/8B; 70B pending)

Artifact: `notebooks/data/two_faces_20260702/unit_deficit_report.json` (+ per-domain raw
`r3_{humor,math}/unitgrid_v1/unit_auc_report.json`, local copies pulled).

| | humor | math |
|---|---|---|
| h-shift 1B vs 3B | **+1.32 units** [1.11, 1.57] | +0.08 [-0.04, 0.20] |
| h-shift 3B vs 8B(SELF) | +0.19 | +0.19 |
| mean Delta-k @ tau=.55/.60/.65 (1B vs 3B) | 0.42 / 0.63 / **1.08** | 0.00 / −0.24 / 0.50 |
| 1B censored @ .65 (units never suffice, 3B reaches) | **18/57** | 5-ish/21 (mixed both dirs) |
| content − filler (AUC) | +.02..+.06 | **≈ 0.00** |
| mean AUC: u0 -> all units, 1B | .614 -> .696 | (names already clear floor) |

Reading: switching 3B->1B in humor costs ~+1.3 units per metric on average, RISING with the
quality bar, with a hard enculturation ceiling (31% of metrics unreachable at .65 for 1B at ANY
unit count; 1B full-dossier ceiling .70 < 3B name-only .86). In math the name IS the unit:
zero deficit at floor, content beyond the name adds nothing over length-matched filler (clean
domain dissociation), and extra articulation sometimes HURTS the smaller reader (negative
Delta-k at tau=.60) — the unit-level face of the name-peak inversion. TASTE vs CRAFT unit-costs
in humor are similar (1.29 vs 1.33; censoring directionally TASTE-heavier 43% vs 28%, small n)
— cost is FRAME-LEVEL, echoing the E2 frame-commitment finding.

Guards: ordinal claims only; 8B = self-recovery; unit ORDER fixed (rubric->def->expl, document
order) — greedy/salience-ordered curves would be steeper (k* here upper-bounds the minimal
count); metric-level (not probe-clustered) CIs.

## IV. 70B pass (Stage B/C, running overnight)

sk3 GPU4 chain `outputs/unit_chain.sh` → `outputs/unitgrid_chain/STATUS`; one engine per
process; Llama-3.3-70B-FP8 snapshot fde04ee… scores the byte-identical rung messages for the
7 clean domains (prereg protocol), then the unit grids (humor+math 70B point).
First landed: math 70B name .707 > def .642 — the formal-lexicon inversion REPLICATES and
DEEPENS at 70B (8B-self deficit was −.021). Partial evaluator smoke (math+news only):
persistence 16/22; parametric law trending to its own pre-specified rejection (9/10 cells
outside CI — the 8B self-point strain the freeze itself flagged). DO NOT QUOTE until all 7
domains land; final numbers → `notebooks/data/two_faces_20260702/prereg_70b_evaluation.json`.

## V. FINAL RESULTS (2026-07-12 morning — chain ALL-DONE, 70B anchor landed)

**Prereg closure (prereg_70b_evaluation.json, clean-7 domains, both frozen hashes verified):**
persistence 34/51 = **FALSIFIED** (losses include real reversals: legal +.174, pr +.167, cr +.11 —
not only eps=0 noise); literal prefix = **FALSIFIED** (288 violations); BUT frozen deficit-ranking
carries real ordinal signal (rank-AUC .689, perm p=.008). Parametric law **REJECTED** by its own
rule (6/8 cells outside CI). Quote as: "failed as frozen; ranking predictive." Tainted cw/humor
scope (70B pre-freeze): persistence 25/37, rank-AUC .569 — weaker, reported separately.

**Baseline-gated rescue rates (unit_deficit_report.json v2; target = 8B executor M_i throughout —
all "big" anchors are recovery-of-8B-policy, so 70B pairs carry a target-kinship caveat and 8B
pairs a SELF flag):**

| pair | humor: gaps rescued | math: gaps rescued | humor local shift (descriptive) |
|---|---|---|---|
| 1B->3B | 2/55 (3.6%) | 0/10 | +1.32 segments |
| 1B->70B | 7/51 (13.7%) | 0/14 | +1.19 |
| 3B->8B [SELF] | 15/52 (29%) | 0/20 | +0.19 |
| 3B->70B | 10/13 (77%, only 13 gaps) | 2/13 | -0.16 (3B closer to 8B-policy than 70B) |

Filler control (form_matched=False on these legacy artifacts; canonical-only recompute confirms):
humor articulation-content real at every scale incl 70B (+.02..+.06); math ~0 at every scale.

**ANSWER to the driving question ("how many MORE units when switching big->small"):** the honest
answer is mostly "no finite number." The +1.3-segment displacement is a local overlap statistic;
under baseline-gating, added segments close only 3.6-13.7% of genuine humor gaps at 1B and none of
math's, while one scale step (3B->8B-policy) is partially articulable (29%). Quantity of
articulation does not substitute for scale; what it buys is partial, domain-dependent, and in math
zero beyond the name itself. Converges with the 2nd-agent fixed-target result (0 full
methodological substitutions on its stricter gates).

## VI. Classic scaling readout extended to 70B (2026-07-12, user request)

Artifact: `name_sufficiency_scaling_70b.json` (new file; original scaling json untouched).
CAVEATS: 8B col = SELF; 70B col = non-self recovery of the 8B policy; cw 70B cell INVALID
(mixed M_i scorings) and excluded from tag pooling; **grant 70B INVALID — probe corpus drifted
since Jul-5 (70B npz scored 150 probes vs original 200, misaligned) → grid_auc_report crash left
Jul-5 auc_report in place; grant stays "no_70B_data" in the prereg eval. Byte-identical re-score
impossible until the grant corpus drift is resolved (dataset-first check).**
Headlines: math 70B = 0/21 still-deficient (every name sufficient, deficit −.065 = deepest
inversion); humor deficiency 3B .632 / 70B .509; news/pr/cr flat-or-worse at 70B (where prereg
losses live). Tags ex-cw: TASTE onset at 3B (.77→.40) then PLATEAU (.56 at both 8B and 70B);
CRAFT slow decline (.75→.51); MECH high throughout (.71→.57).
