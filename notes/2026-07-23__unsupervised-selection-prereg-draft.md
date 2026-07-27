# Unsupervised / reconstruction-objective selection — prereg DRAFT (NOT FROZEN)

2026-07-23, Fable. Status: DRAFT for user review — nothing confirmatory launches until this is
revised + frozen (SHA recorded per prereg discipline). Origin: user directive "expand all of our
baselines to the unsupervised metrics and the reconstruction objective as well."

## Goal
Extend the prompt-optimality head-to-head (seed / GEPA / GEPA+Merge / MIPROv2-Heavy / M_ω) to the
regime where NO labels drive selection: the selection signal is the reconstruction/recovery
objective (C(R(Ω)) = I(M_ω;·) readout; MCQ mode, hard distractors — standing defaults). Labels are
used ONLY at final test (evaluation, never construction — reconstruction-only rule).

## Key design move
Per-item MCQ recovery correctness (0/1) is a per-example scalar → it plugs directly into GEPA's
and MIPROv2's metric interface with NO adapter surgery. (The corpus-level recovery readout stays
the headline REPORTED metric per feedback_report_recovery_metric_only; the per-item version is the
optimization signal.)

## Arms (per bench, Qwen3-8B, same splits/serving as supervised arms — apples-to-apples rule)
1. seed (shared with supervised table)
2. GEPA(rec) — official GEPA, metric = per-item recovery
3. MIPROv2-Heavy(rec) — paper-literal config, metric = per-item recovery
4. M_ω(rec) — unitrecomb, selection + confirm gate on recovery objective (frozen 8B pools)
5. (optional) GEPA+Merge(rec)

## Readouts (all three, always)
- R1: recovery objective achieved, C(R(Ω)) — headline
- R2: downstream labeled test score (labels at evaluation only)
- R3: rank consistency recovery↔labels across each run's candidate trail (threshold-free:
  Spearman + pairwise AUC; per-bench)

## Hypotheses to freeze (directional — MUST be frozen before confirmatory runs)
- U-i (capture): recovery-selected prompts capture a fraction ρ_b of the label-selected gain
  (best supervised arm − seed) per bench b; report ρ_b with CIs. Directional expectation:
  ρ substantially > 0 on high-articulability benches.
- U-ii (rank validity): R3 rank consistency predicts ρ_b across benches.
- U-iii (pool width): wider pools shrink (1−ρ) — mining moves the unsupervised level too.
- U-iv (articulability link): benches with higher V/A (program's other legs) show higher ρ.

## Guards
- Freeze-before-eval: this note revised → frozen SHA recorded → only then confirmatory runs.
- Stable hash splits (never seeded-shuffle); identical splits to supervised arms.
- Anchor-test annotation passes in any judge batch (pupa).
- Threshold-free readouts for all cross-family comparisons.
- No label leakage into selection anywhere (audit: grep metric wiring per arm before launch).
- Judge = Sonnet-or-better / GLM only (standing rule) for recovery MCQ grading.

## Schedule
1. Tonight: this draft → user review/edits.
2. Validation (validate-before-scaling): ONE bench — aime (verifiable metric, no judge dep) —
   M_ω(rec) + GEPA(rec) only; inspect extraction samples + R3 before fanning out.
3. Fan out to 6 benches AFTER the supervised baseline queue (MIPROv2-Heavy, official_merge)
   completes on freed servers.
4. EVT/rescore integration: unsupervised arms join the offline-batch rescore pass → EVT columns
   on matched populations.

## Open questions for user
- ρ target/threshold for "substantial capture" (suggest reporting continuous, no threshold).
- Include GEPA+Merge(rec) in v1 or defer?
- MCQ distractor source for benches without existing recovery banks (livebench/ifbench):
  synthesize per standing exemplar-synthesis rule, or restrict v1 to benches with banks?
