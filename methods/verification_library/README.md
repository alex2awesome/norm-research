# verification_library

## Purpose

Bottom-up discovery of programmatic evaluation criteria. For each training example, an LLM writes a Python `predict(text) -> float` program that scores the artifact; a refactoring pass periodically distils repeated logic into a shared library; an ensemble of programs predicts the label on held-out data. This is the **Direction 2** pipeline — articulation first, then prediction.

## Algorithm sketch

Generate → execute → refactor → evaluate loop:

1. **Extract** structured fields from raw text (`extractor.py`); each field is regex, stdlib program, or LLM-thick. The schema evolves as the library matures (thick → thin migration).
2. **Generate** per-example programs in batches of `batch_size` via the cheap "generation" client (Qwen3-Coder-Next-FP8 by default).
3. **Execute** each program in a sandbox (`sandbox.py`, subprocess + timeout) on the training example to confirm it returns a usable float.
4. **Refactor** after every `refactor_size` new successful programs: the expensive "refactoring" client (Claude Sonnet 4 by default) merges duplicates, extracts shared helpers into the library, and prunes dead programs.
5. **Evaluate** the program ensemble on the test set; combine outputs via `vote`, `mean`, or `logistic`.
6. **Check convergence** over a sliding window of `convergence_window` refactor rounds.

Four approach variants are wired through `approach` in `VerificationLibraryConfig`:

| `approach` | Behaviour |
|---|---|
| `v1` | basic per-example raw-text programs |
| `v2` | extraction-aware (uses `extractor.py` schema) |
| `approach1` | full-example agnostic programs with optional z-guidance |
| `approach2` | per-z program synthesis from STaR rationales (default) |

## Key files

```
methods/verification_library/
├── runner.py          # run_verification_library / _v2 — outer generate→refactor→eval loop
├── config.py          # VerificationLibraryConfig, DatasetConfig, DATASET_REGISTRY
├── client.py          # LLMClient (Anthropic / OpenAI-compatible) + MultiClient
├── generator.py       # per-example program generation, ``` parsing, hypothesis extraction
├── refactorer.py      # library refactor pass (Claude) + similarity routing above 300 fns
├── sandbox.py         # subprocess execution with timeout, source validation
├── extractor.py       # schema field extraction (regex / program / llm)
├── program_store.py   # GeneratedProgram, ProgramStore, LibraryStore
├── evaluator.py       # ensemble combination (vote / mean / logistic) + AUC
├── combiner.py        # combination-method helpers
├── convergence.py     # edit-distance + accuracy-stability convergence
└── prompts.py         # SYSTEM_PROMPT_GENERATION, render_* templates
```

Top-level driver: `scripts/run_verification_library.py`.

## How to run

```bash
python scripts/run_verification_library.py \
    --dataset peer-review \
    --approach approach2 \
    --generation-model qwen/qwen3-coder-next \
    --generation-base-url https://openrouter.ai/api/v1 \
    --refactoring-model claude-sonnet-4-20250514 \
    --batch-size 20 \
    --refactor-size 3 \
    --max-examples 200 \
    --concurrency 30 \
    --sandbox-timeout 5.0 \
    --test-set-size 500 \
    --combination-method logistic
```

Supported `--dataset`: `peer-review`, `code-review`, `notice-and-comment`, `press-release` (extras `creative-writing` and `humor` are wired in `DATASET_REGISTRY` but not yet in the script's `DATASET_CONFIGS`). Point `--generation-base-url` at `http://localhost:<port>/v1` for local vLLM serving.

## Dependencies

- `anthropic`, `openai` (async clients).
- `pandas`, `numpy`, `scikit-learn` (logistic ensemble + AUC).
- A code-embedding model for similarity routing once library size exceeds `library_similarity_threshold = 300` functions (`jinaai/jina-code-embeddings-0.5b` local, `nomic-ai/nomic-embed-code` server).
- For local generation: vLLM with Qwen3-Coder-Next-FP8 (see `reference_fp8_vllm_sk3.md`, `reference_qwen35_vllm_sk3.md`).

## Current state

**Experimental, still being tuned.** Per `project_verification_pipeline_recipe.md` (peer review, 20K examples):

- Llama STaR extraction: 133K features over 20K examples, 94% parse rate.
- Dedup via `all-MiniLM-L6-v2` + K-means → 12,485 canonical features.
- Qwen coding (forced `def check(text):` stub prompt): **42% single-pass success**; multi-pass retry recovers an additional ~30–40% of remaining failures per round.
- Llama self-assessment: 28% thin / 72% thick; Qwen wrote programs for 1,874 features Llama had called "thick" → the empirical thin/thick boundary is wider than self-report.
- v1 ensemble AUC hit **0.500 on peer_review across 7+ runs** because the prompt restricted programs to stdlib-only regex heuristics — this is the failure mode that motivated `methods/existing_metrics_runner/coded/` as a per-norm complement.
- Hierarchy building (Step 4) and program evaluation on test set (Step 5) are not yet implemented in the script.

## Outputs

- `outputs/verification_library/<dataset>_<tag>/`
  - `programs.jsonl` / `feature_programs_merged.jsonl` — generated programs with `source_code`, `code_classification` (thin/thick/unknown), execution status.
  - `library/` — refactored shared helpers.
  - `convergence_history.json`, `summary.json` — per-round metrics, final ensemble accuracy + AUC + library size + converged flag.

Concrete known runs: `outputs/verification_library/peer_review_100*`, `peer_review_approach1_100`, `peer_review_a1_100_v2`, `peer_review_v1_100_incremental`, `peer_review_smoke`, `peer_review_diff_test`, `direction1_20k/programs.jsonl` (20K peer-review predict-programs), `direction2_20k/coded_features_v2/feature_programs_reparsed.jsonl` (12.5K per-canonical-feature programs: 2,618 thin / 15 thick / 5,994 failed / 3,634 syntax-error), `math_20k/star_features.jsonl` (150K math STaR features).

## Related

- **Sibling per-aspect programs at `runs/validity_full/v2/{task}/codegen_claude/`** — pre-existing per-task corpora produced by an earlier codegen pass. Per `reference_codegen_per_aspect_programs.md`, files follow `a{ID}_v{0,1,2}_{keyword,structure,holistic}.py`. Per-task counts on this checkout: **peer_review 654, creative_writing 1104, code_review 1182, humor 1098, patents 748, news_homepages 621, notice_and_comment 597, press_releases 972, math 732**. These are regex/structural heuristics on raw text and are the empirical reason `methods/existing_metrics_runner/coded/` was started: their text-level pattern matching collapsed to comment-length proxies on code_review.
- `project_verification_pipeline_recipe.md` — current working recipe + tuning notes.
- `project_code_review_verifiability_plan.md` — what's built vs. what's missing for code_review's V+A+Taste layer; this method is the "Python predict-program" rung.
- `project_verifiability_explainability_gaps.md` — overarching `Outcome = Verifiable + Articulable + Taste` decomposition.
- `methods/existing_metrics_runner/coded/` — complementary per-norm conformance scorer (this method predicts the label end-to-end; that one measures one norm at a time using real tools).
- `methods/local_explanations/` — same bottom-up philosophy but produces natural-language features + linear classifier instead of Python programs.
