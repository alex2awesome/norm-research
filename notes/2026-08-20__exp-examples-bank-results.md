# ★★★ RETRACTED 2026-08-20 (same day) — WRONG OBJECTIVE, DO NOT QUOTE
All numbers below are reference-key balanced-accuracy readouts (LOFO 11-panel / 2-voter
consensus). User directive: the only sanctioned objective is the RECONSTRUCTION OBJECTIVE
(full decoder round trip, I(m; m')). Key artifacts quarantined in
data/DO_NOT_USE__reference_keys/. Successor: bank_examples_mi_v1.json (recon rescore,
exp_examples_bank_recon.py, launched 2026-08-20). Kept for the audit trail only.

# EXP-EXAMPLES-BANK-1 — results (2026-08-20)

Prereg c42a8f54db2e6f54 + Amendment 1 (49dd908dacce36f9) + Amendment 2 (a9d410335e879840).
Selection 236/236 metrics (llama8b, sk3, Aug 17); evaluation llama70b + qwen25-72b (BF16 TP=2,
sk3, Aug 20, per-item P(YES) vectors saved); score stage run LOCALLY (sk3 lacks the crowd_panels
dir — the first sk3 score pass silently produced empty LOFO tables; local rerun with
--legacy-dir is the artifact of record).

## Headline (LOFO family-balanced 11-panel key, PRIMARY)
- examples − definition ≈ 0 EVERYWHERE: all 6 criterion types within ±.007, every metric-level
  bootstrap CI includes zero; all 8 tasks within −.038..+.027 (large-n cells ±.002).
- OLS (delta ~ task + category, metric bootstrap): no coefficient separates from zero except
  task:news_homepages −.037 [−.053, −.000] at n=6 (boundary; news = flagged task, standing
  landmine).
- Bounded-camp / tacit-transmission hypothesis NOT supported: identity/social +.007
  [−.004, +.021].

## Diagnosis (two-voter selection key, sensitivity)
- Same sets under the key selection optimized: −.017..−.044 by category; three categories'
  CIs exclude zero on the NEGATIVE side (8B-selected sets overfit their selection key; at
  70B-class judges they subtract).
- L1 null gate: PASSES under LOFO (+.022 < .03); fails under two-voter (+.094) — the leak is
  in the selection key, not the symmetric readout. L2 agreement: all 8 tasks ≥ .80 (pass).

## Legacy 38/56 rescore (legacy38_rescored_v1.json, 448 rows)
Old slate-scale readings (+.07 verifiable / +.05 local patterns, dossier-anchored key) collapse
to ≈0–.02 under the LOFO key. ★ NEVER quote the old-key flips category medians without naming
the key; the symmetric-key numbers supersede them.

## Paper landings
- Figure 4 right panel: figs/gen_examples_bank.tex (gen_fig_examples_bank.py), dot+CI per
  category, both keys side by side. Caption updated; pending-box fallback retained in
  gen_fig_message_forms.py.
- §4 ¶"Definitions, Explanations, Reasoning or Examples?" stubs filled (main.tex).

Artifacts: outputs/exp_examples_bank/{bank_flips_v1,legacy38_rescored_v1}.json (laptop =
artifact of record, mirrored to sk3); eval vectors ebank_eval_*.jsonl + ebank_legacy_eval_*.jsonl
(both boxes). Queued next: EXP-EXAMPLES-BOUNDED-1 (absolute-level + covariate design — note the
bank null makes its prior low but the covariate test is what the prereg promised); notebook
4.2g-b rebuild from the primary key.

===============================================================================
# RECONSTRUCTION-OBJECTIVE RESULTS — FINAL, both judges (2026-08-21)

Pipeline: exp_examples_bank_recon.py; llama70b on sk3 (+2 rows sk2, eager), qwen25-72b
102 rows sk2 (eager) + 134 sk3 (compiled). 236/236 per judge. Artifact:
outputs/exp_examples_bank/bank_examples_mi_v1.json (laptop = record, mirrored sk3).

## Headline (pooled judges)
- No task mean ΔI separates from 0 (humor +.011 [−.005,+.026] top; peer_review −.005 low).
- Categories: only LOCAL PATTERNS clears 0: +.010 [+.002,+.019] — 1 of 6 at 95%,
  SUGGESTIVE not survive-correction.
- Null-selected sets: noise bound ±.06 bits (p90, n=158). BOTH-JUDGE survivors: 3 pos /
  1 neg of 236 vs ~2.4 expected each — count ≈ chance, but the 3 positives are thematically
  coherent local-pattern craft: humor delivery&timing (+.096/+.069), humor incongruity
  mechanics (+.071/+.104), news conflict/tension (+.108/+.081).
- Cross-judge per-metric delta correlation ρ=+.148 (p=.023): mostly noise, whisper of signal.
- NON-replications to never headline: novelty&positioning +.17→−.07 at qwen; callback craft,
  Australian conventions, format-innovation negative — all die at the second judge. The
  anti-inductive novelty story did NOT replicate.
- Channel level: pooled MI ~.05 bits/arm; MI strongly entropy-coupled (ρ=.73 with H(m));
  report H(m) alongside or normalize.

## Paper landings (2026-08-21)
Fig 4 RHS = 4-task boxplots (humor/n&c vs creative_writing/peer_review), callouts = the 2
humor both-judge survivors only. ¶Examples final prose: calibrated null + the thin
local-patterns slice, hedged. Calibration levers if pursued: null arm on all metrics,
R=3 decoder seeds on tail metrics, demo/MI fold rotation (EXP-EXAMPLES-BOUNDED-1 still queued).
