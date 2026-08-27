# EXP-EXAMPLES-BANK-1 — AMENDMENT 1 (evaluation reference), 2026-08-17

Amends the frozen prereg (sha c42a8f54db2e6f54) BEFORE the evaluation stage has run
(selection stage in flight, unaffected — selected sets are prompts; scoring them under a
differently-anchored key only hardens the test for the functional arm; asymmetry disclosed).

## Reason (user objection, 2026-08-17)
The frozen reference (2-voter consensus under the metric's BANK RUBRIC) is generated under one
of the compared arms' own formulation (the bank rubric IS the definition), and the lineage
reference (dossier-arm consensus) is generated under the MOST explicit formulation. Either way
the key's errors are not neutral: they concentrate against less-explicit arms, and the bias
magnitude may co-vary with a metric's explicitness — poisoning the task x category comparison.
No formulation-neutral true labels exist in the unsupervised regime; the least-biased available
key is the formulation-symmetric panel consensus.

## Changes (all evaluation-stage only)
1. PRIMARY key = LOFO family-balanced consensus of the full 11-executor crowd panel (the
   paper's recovery objective; each executor labels under the same bank rubric, but the key is
   the panel consensus, leave-one-family-out w.r.t. the evaluating judge — no single arm or
   voter privileged). Ties/undecided -> -1 as before; decided-fraction reported per metric.
2. The originally frozen 2-voter bank key is DEMOTED to a sensitivity readout (reported side by
   side; divergences between keys reported per category).
3. Per-item arm labelings are SAVED (not just holdout scalars) so any future key can rescore
   without re-running judges.
4. External validity check: on tasks with sound task-level silver (humor, creative_writing,
   peer_review, code_review per the silver-matching audits), report each arm's AUC against
   silver as a no-LLM-key cross-check (task-level labels; per-metric readout, labeled as such).
5. Disclosure: selection optimized against the 2-voter bank key (already in flight); primary
   evaluation uses the symmetric key. This asymmetry biases AGAINST the functional arm and is
   accepted as conservative.

No other change to sample, arms, gates, or decision readouts.
