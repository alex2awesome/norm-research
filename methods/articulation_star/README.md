# articulation_star

## Purpose

A STaR-style loop that trains a small generator (Llama-3.1-8B + LoRA) to
**articulate** community-agreeable reasons for a binary label on a
per-datapoint basis, without learning the label-prediction shortcut.
"Articulation" here means: generate POSITIVE_ASPECTS / NEGATIVE_ASPECTS
bullets grounded in the artifact such that an information-bottleneck judge
reading the rationale alone (no artifact) recovers the true label more often
than chance. The trained articulator is the deliverable, not a classifier.

## Algorithm sketch

Per iter (orchestrated by `loop.py`):

1. **Generate** K rationales per artifact with the current (LoRA-adapted)
   generator. Generator never sees the label — anti-leakage mechanism (1).
2. **Filter** with a bottleneck judge that reads **rationale only**, no
   artifact. Two modes:
   - `single`: keep iff strong_judge_margin > 0 on the true label.
   - `contrastive`: keep iff strong prefers correct AND weak does not —
     drops rationales whose label is decodable from cheap surface cues.
   Logprob-based scoring: margin = `logP(y_true) − logP(y_other)` on the
   first generated token after `ANSWER:`.
3. **SFT** the current LoRA on kept rationales. Loss masked to the
   assistant turn (rationale only); the label word is never a training
   target — anti-leakage mechanism (2). TRL `assistant_only_loss=True`.

Optional fallbacks (see `project_articulation_star_fallback_defenses.md`):
metric-seeded cold-start, anti-template filter, distilbert leakage probe,
counterfactual-swap probe, rStar-style process reward.

## Key files

- `loop.py` — top-level orchestrator (`run(cfg)`); calls
  `generate_rationales → judge_filter → train_sft` per iter.
- `config.py` — `LoopConfig` dataclass + `TASKS` registry
  (peer_review, press_releases, creative_writing). Holds generator /
  weak-judge / strong-judge model paths, decoding params, LoRA hyperparams,
  output root.
- `generate_rationales.py` — vLLM generation; supports
  `shard_idx / n_shards` for data-parallel generation across GPUs and
  `teacher_model` for iter-0 cold-start from a bigger model.
- `prompts.py` — `render_gen`, `render_judge` chat templates.
- `judge_filter.py` — logprob scoring of rationales; `--mode predict
  --judge {weak,strong}` runs a single judge to avoid two vLLM engines on
  one GPU; `--mode combine --balanced_k_per_label K` builds
  `rationales_kept.jsonl`.
- `merge_shards.py` — merges per-shard `rationales.shard*.jsonl` after DP
  generation.
- `train_sft.py` — TRL `SFTTrainer` LoRA training; patches the
  Llama-3 chat template to add `{% generation %}` markers so
  `assistant_only_loss` works.
- `test_eval.py` — held-out evaluation: `build_split`, `generate`,
  `score`, `summarize` modes across stages (`base`, `iter_00`,
  `iter_01`, `iter_02`).
- `audit_rationales.py` — diversity, mode-collapse, format-compliance
  audit on a single iter.
- `leakage_detect.py` — auto proxies (specificity, n-gram overlap,
  sentiment polarity, template hits) + LLM-judged leakage rubric
  (gpt-5-mini).
- `distilbert_leakage.py` — small frozen text classifier as the
  canonical per-iter leakage measurement.

Shell entrypoints live in
`/Users/spangher/Projects/stanford-research/norm-research/scripts/articulation_star/`:
`smoke_test.sh`, `run_iter.sh`, `run_overnight.sh`, `run_overnight_v2.sh`,
`run_test_eval.sh`, `run_distilbert_probe.sh`, `eval_compare.sh`,
`run_explore_contrastive.sh`.

## How to run

End-to-end loop (in-process — fine for smoke, not for real runs):

```bash
python -m methods.articulation_star.loop \
    --task creative_writing --run_name v0 --n_iters 3 \
    --n_train_subsample 4000 --n_rationales_per_input 4
```

Production overnight run (subprocess-per-step so vLLM tears down between
generator / weak / strong / train):

```bash
TASK=creative_writing RUN_NAME=v2_weak1b_logprob \
    bash scripts/articulation_star/run_overnight_v2.sh
```

Held-out test eval (one stage at a time):

```bash
bash scripts/articulation_star/run_test_eval.sh
```

vLLM env quirks for this sk3 environment (see smoke status memo):

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
# Qwen-3 reasoning models: pass enable_thinking=False in chat template
```

## Dependencies

- Generator: `meta-llama/Llama-3.1-8B-Instruct` (LoRA r=16).
- Strong judge: `Qwen/Qwen3.5-122B-A10B-FP8` (cached on sk3 at
  `/lfs/skampere3/0/shared_hf_cache/...`).
- Weak judge: `meta-llama/Llama-3.2-1B-Instruct` (FIXED across iters —
  training the weak judge defeats its bottleneck purpose).
- Data: `datasets/<task>/...csv.gz` with `text` and `judgement` columns;
  task paths in `config.py`.
- Python: `vllm`, `transformers`, `peft`, `trl>=1.5`, `datasets`, `pandas`.
- GPU: per `feedback_minimize_gpus`, the loop is designed for a single GPU
  (subprocess-per-step). Data-parallel generation uses up to 4 GPUs in the
  overnight scripts.

## Current state

In progress. Last full run: `v2_weak1b_logprob` on creative_writing
(2026-05-31) — the contrastive filter inverted (1B weak hit 68% acc vs.
Qwen-122B strong 53% in logprob mode), so iter 0 combine kept 0 negatives
and the run died. Did not retry; the inversion was itself the signal.

The earlier `v1_overnight_logprob` run completed clean: train loss dropped
0.76 → 0.64 → 0.50 across 3 iters, base→iter_02 test acc 48% → 56.2% (+8.2pp
held-out), no mode collapse, modest template-driven leakage growth.

See:
- `project_articulation_star_smoke_status.md` — judge sentiment bias diagnosis.
- `project_articulation_star_overnight_run.md` — v1 completed run details.
- `project_articulation_star_v2_run.md` — v2 inversion + leakage findings.

Proposed next experiments are listed at the bottom of the v2 memo:
CoT-strong + logprob-weak; dedicated distilbert leakage classifier;
anti-template keep rule.

## Outputs

```
outputs/articulation_star/<task>/<run_name>/
  iter_{00,01,02}/
    rationales.shard*.jsonl   # per-DP-shard generations
    rationales.jsonl          # merged
    judge_preds_weak.jsonl
    judge_preds_strong.jsonl
    judge_diagnostics.jsonl
    rationales_kept.jsonl     # SFT input for this iter
    lora/                     # LoRA adapter, fed to next iter
  test_eval/
    test_artifacts.jsonl
    rationales_<stage>.jsonl
    scores_<stage>.jsonl
    leakage_auto_<stage>.jsonl
    leakage_llm_<stage>.jsonl
```

Logs: `logs/articulation_star/<run_name>/{master,iter*_*}.log`.

## Related

- `methods/local_explanations/` — the rationale-extraction approach that
  motivated articulation-STaR; goal (a) was extract-then-cluster, goal (b)
  is articulation-STaR (model itself IS the articulation).
- `methods/dense/` — the dense reward-model baseline. STaR test-set acc is
  compared against the dense ceiling for the same task.
- Memory:
  - `project_star_algorithm_brainstorm.md` — bottom-up z₊/z₋ extraction,
    Haupt level-set hierarchy, thin/thick per node.
  - `project_articulation_star_rstar_followup.md` — rStar process-reward
    extension once basic STaR works.
  - `project_articulation_star_fallback_defenses.md` — ordered fallback
    options (contrastive judge, metric-seeded cold-start, etc.).
  - `project_articulation_star_smoke_status.md` /
    `project_articulation_star_overnight_run.md` /
    `project_articulation_star_v2_run.md` — run logs and findings.
