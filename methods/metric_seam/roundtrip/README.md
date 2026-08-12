# Code round-trip experiment (2026-08-12)

Question (user): is a construct as articulable in CODE as in prose? Fair test = symmetric
home-modality round trips through the reconstruction bottleneck:
- prompt arm (exists): LLM labels -> decoder prose m_hat -> LLM re-executes -> R (stored,
  channel="judge" in reconstruction/recon_results.jsonl)
- code arm (this pipeline): code labels -> decoder prose m_hat (stored, channel="code_*")
  -> BLIND recompile to Python -> deterministic re-execution -> R_home

Pipeline (run in order; every step deterministic/resumable; eval items frozen throughout):
1. build_jobs.py          — job packs from recon_results + detail splits (seed frozen)
2. run_compile_trials.py  — N independent blind Codex compilations per rule (--fresh
                            threads; RT_INSTRUCTIONS.md is the whole protocol the
                            compiler sees). Trial t0 provenance note: the 2026-08-12 t0
                            arm was produced by interactive Sonnet crews given the SAME
                            RT_INSTRUCTIONS verbatim (before this script existed);
                            reproduce t0-style arms by running more Codex trials.
3. run_gepa_program.py    — reflective program optimization, TRAIN-side feedback only
                            (worst rank-disagreements shown to the reviser); greedy keep
                            by train spearman; eval never loaded.
4. eval_roundtrip.py      — the ONLY step touching the 40-item eval splits; scores every
                            arm + train-selected best-of + gepa; calibration = 6 planted
                            rules vs reference implementations.

Interpretation guardrails: one-shot trials measure "does prose suffice as the carrier";
gepa arm measures "prose + behavioral feedback"; the delta = what the prose bottleneck
loses but execution feedback recovers. Target labels are themselves program outputs, so
the true generator is in the search space — failures are attributable to the channel.
Arm selection (best-of, gepa keep) is ALWAYS by train spearman, never eval.
