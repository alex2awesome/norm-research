# ctree — location & live path (2026-06-26)

*Moved out of a sibling project's memory (the prior session launched from the
`regulations-demo` repo, so this note landed in the wrong project). Kept as a package doc.*

"ctree" = the user's shorthand for the **metric-tree** work, which lives in a *different* repo: `~/Projects/stanford-research/norm-research/methods/`. Specifically `methods/metrics_tree_infilling/` (the newest gap-detecting MOB tree; the literal `c_tree` substring in `metric_tree` is why it greps as "ctree"). Older sibling: `methods/metric_tree/` (algorithm 2/3). Today's active work is `methods/metric_implementer/` (metric validity/codegen).

**Live LLM path runs through the z.ai anthropic proxy** (`ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic`, `ANTHROPIC_MODEL=glm-5.2`) — the "anthropic" backend resolves to GLM models. `claude-sonnet-4-20250514` and `glm-5.2` are accepted by the proxy; `glm-4.7-flash` is not. The default `materialize_backend="vllm"` / `materialize_model="Qwen/..."` won't work on the laptop — override to `anthropic`+`glm-5.2` for any local run. Proxy is heavily rate-limited (shared with the Claude Code session) — add outer retries.

**Fixed 2026-06-26 (was THE blocker — "the --live path had never been executed"):** `verification_library/client.py`'s `LLMClient` is constructed once with an `asyncio.Semaphore`, but `io_metrics`/`feature_gen` drive it via `asyncio.run()` per batch (each a fresh event loop) → `RuntimeError: Semaphore bound to a different event loop`. Fix (now in `client.py`): rebind the semaphore to the current loop when it changes. No-LLM tests (10) unaffected.

**Live smoke harness:** `norm-research/smoke_infill_live.py` — builds a trimmed (n=500) synthetic creature corpus, runs the full live loop via glm-5.2, self-contained (adds `methods/` to sys.path). Run: `python smoke_infill_live.py` from the norm-research root. Judge calls cache to `outputs/.../smoke_live/judge_cache/judge.jsonl` (resumable). The full 2,400-item version is `tests/test_scenario/test_discovery.py::_run_live()` (run `python -m ...test_discovery --live`).

Smoke completed end-to-end 2026-06-26 (203s) but discovery quality was poor on the trim: proposed a redundant feature (dropped, R²=0.92), did not rediscover the planted song/glow norms — ~~likely needs the full 2400-item corpus (+ possibly a stronger proposer)~~.
  - **CORRECTED 2026-06-26 (oracle bisect, `tests/test_scenario/diag_bisect.py`):** corpus size
    and cfg are *exonerated*. The oracle (no-LLM) proposer+judge rediscovers song+glow at **n=500
    with the exact smoke cfg** (`min_node_size=30, max_depth=4, max_outer_rounds=2`). The live
    smoke failed purely on the **glm-5.2 LLM seam** — it needs a stronger proposer/judge, NOT more
    data. Bisecting proposer-vs-judge needs a live run that persists proposer outputs (the smoke
    only prints); blocked on a non-z.ai backend (OpenRouter/Gemma, see distillation pathway).
  - The full 2400-item `_run_live` could NOT finish on z.ai — rate limits killed it at 55 min. **z.ai is not viable for multi-thousand-call batch runs.**

**Scale path (decided 2026-06-26):** move off z.ai to **OpenRouter** (key at `~/.openrouter-api-key.txt`) for prototyping + **sk3** (reachable non-interactively, B200 GPUs) for vLLM scale. **User wants Gemma 4 specifically** (`google/gemma-4-31b-it`; tracked in `metric_implementer/manifest.py`, blocked for sk3-local load on current transformers but served by OpenRouter) — not Gemma 3.

**Distillation pathway (plan approved 2026-06-26, plan file `~/.claude/plans/splendid-percolating-wolf.md`):** build a GEPA+Gemma4 pre-task stage that turns base rubric metrics into cheap scorers so the tree runs at scale. Builds on `methods/metric_implementer/` (already has `optimizer.improve()` GEPA, `measures.py` 7 label-free fidelity measures, `registry.py`, OpenRouter `backends.py`). Missing pieces to build: batch `run_distillation.py`, `export.py` (registry→rubrics JSON / `.py`), `--rubrics-dir` in infilling `run.py`, OpenRouter-key read in `make_vllm_judge_scorer`. Primary product = GEPA-optimized prompts judged by Gemma 4; code-distillation (rubric→`score(text)`) is Phase 4. In-loop judge+reviser=Gemma4 (same-family OK); acceptance reconstructor/generator = Qwen/Llama (cross-family, anti-Goodhart).

Real datasets for `run.py`: peer-review=70K rows (needs subsampling), press-release is a sharded dir (`load_items` handles it), code-review/notice-and-comment dataset files are missing. See the package's `2026-06-10__next-steps-plan.md`.
