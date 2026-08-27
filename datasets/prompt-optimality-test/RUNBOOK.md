# Runbook — official-GEPA vs in-house-GEPA + prompt-optimality estimators

Read README.md isolation rules first. Every phase writes ONLY under this folder.

## Phase 0 — pin & data
1. `bash setup_gepa.sh` → PIN.txt (gepa pip version + repo sha; quote both in the paper).
2. `source .venv/bin/activate && python download_datasets.py` → data/{hotpotqa,hover,aime2025}/.
   Fixed hash-stable train(150)/val(300) JSONL; both arms read the same files.

## Phase 1 — Arm A: official GEPA
Per dataset: `gepa.optimize(seed_candidate=SEED[ds], trainset, valset, max_metric_calls=600,
task_lm=<local vLLM offline batch adapter>, reflection_lm=<GLM subscription>)`.
- One shared SEED prompt per dataset, committed to `runs/<ds>/seed.txt` BEFORE any run.
- Wrap the metric so every candidate evaluation appends to `runs/<ds>/official/proposals.jsonl`
  (`{ts, candidate_text, parent, score}` — raw draws, accepted AND rejected; rule #4).
- Task LM = GLM over the z.ai subscription HTTP endpoint (documented deviation from the
  offline-batch-vLLM rule: these arms are 0-GPU API runs; the paperexact harness additionally
  needs an HTTP endpoint because DSPy cannot do offline batch). Reflection via GLM.

## Phase 2 — Arm B: in-house GEPA
Same seed, same budget (600 metric calls), same task/reflection LMs, same logging schema →
`runs/<ds>/inhouse/proposals.jsonl`. Use the norm-research GEPA loop code by IMPORT, but point
every output path here (⚠ remember the GEPA_CORPUS-style env traps — set every corpus/target env
explicitly; defaults silently point at main-repo corpora).

## Phase 3 — optimizer sanity comparison
Table: final val accuracy per arm per dataset + trajectory best-so-far curves from the proposal
logs. Purpose: show in-house ≈ official (within noise), so Phase 4 conclusions are not an
artifact of a nonstandard optimizer. Threshold-free readout (rank/AUC style), no cherry-picked
checkpoints.

## Phase 4 — prompt-optimality estimators on both trajectories
For each arm × dataset, from `proposals.jsonl` (TRUE draw order, no survivor bias):
1. Discovery: paraphrase-collapse candidate texts (collide τ consistent with main study) →
   rarefaction + `fit_power_law` + `heaps_unit_bootstrap_ci` (real draw sequence ⇒ the unit
   bootstrap is finally fully licensed).
2. Value: execute each distinct criterion/prompt on the val items (GLM HTTP, threaded) → binary
   behavior matrix; y = gold label. Run `exchangeable_joint_value` + saturating fit
   (y_inf/H(y), τ) + `value_frontloading_stat` D with probe bootstrap.
3. Report the saturation pair + D per dataset; prediction: discovery Heaps-linear, value
   saturates in a handful of criteria, both arms, all datasets.

## Deliverable
`runs/summary.json` + one notebook `analysis.ipynb` INSIDE this folder. Nothing copied into main
outputs; the paper cites this folder's PIN.txt + JSONs.

## Phase 5 — PAPER-EXACT evaluation harness (2026-07-19)

**Requirement (user):** the end evaluation must be exactly the original GEPA paper's, for
comparison purposes. Phases 1-4 used simplified evaluators (containment / claim-only
classification / ad-hoc AIME split) — those results are internally valid for arm-vs-arm
comparison but are NOT paper-comparable.

**Source of truth:** `vendor/gepa-artifact` (github.com/gepa-ai/gepa-artifact) — the paper's own
experiment code. Everything below is used verbatim from it, not reimplemented:
- Programs: `HotpotMultiHop` (2 retrieval hops + answer; 4 LM modules), `HoverMultiHop` (3 hops,
  4 LM modules, k=7), AIME/LiveBench-Math single CoT.
- Retrieval: **local BM25S over wiki.abstracts.2017** (the paper's actual retriever —
  `hover_program.initialize_bm25s_retriever_and_corpus`; the dspy.ColBERTv2 line in
  hotpot_program is vestigial and the public server is dead anyway). Corpus →
  `data/wiki17/wiki.abstracts.2017.jsonl`.
- Metrics: hotpot `answer_exact_match_with_feedback` (EM), hover `discrete_retrieval_eval`
  (gold titles ⊆ retrieved), AIME answer match.
- Splits: `Benchmark` base class trims — train 150 / val 300 / test 300 (seeded); AIME:
  seed-0 shuffle of AI-MO/aimo-validation-aime 45/45, test = MathArena/aime_2025 ×5 (=150).
  Verified loading: AIME 45/45/150 ✓.
- Task LM (paper): Qwen3-8B temp .6/top-p .95/top-k 20, ctx 8192/16384, or GPT-4.1-mini temp 1.
  Plan: Qwen3-8B via vLLM on sk2, dspy.LM against it. GLM is NOT paper-comparable → GLM runs
  are labeled explicitly as extra-model columns if kept.
- Budgets (paper): GEPA max_metric_calls = MIPROv2-Heavy rollout count (2.4K-7K/benchmark).
  Our three-arm comparison runs at a declared smaller budget (internally valid); paper numbers
  at paper budgets quoted alongside as reference, never as same-budget comparisons.

**Deps installed in .venv:** dspy 3.2.1 (litellm pinned <1.93 — 1.93 sdist needs Rust/cargo
edition2024, mac cargo too old), bm25s 0.2.12, PyStemmer, diskcache, ujson.

Status (2026-07-20): corpus downloaded; paperexact_arms.py written (all three arms, M_ω-v2
geometry in unitrecomb, per-task-LM run dirs); AIME arms RUNNING on Qwen3-8B (sk2:8077);
hover/hotpot still need the BM25S wiki index built before their programs can run.
