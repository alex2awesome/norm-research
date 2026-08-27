# existing_metrics_runner

## Purpose

Run a fixed catalog of evaluation aspects against a dataset to obtain per-(example, aspect) scores. The catalog can be measured either by **per-aspect Python programs that call real verification tools** (`coded/`) or by **LLM-as-judge prompts scored via vLLM** (`judge/`). Each backend produces a (N × M) feature matrix that downstream classifiers consume to predict the task label.

## Algorithm sketch

Both backends share the same I/O contract — `(text, aspect) -> (applies, score)` — and differ only in how the score is computed:

- **`coded/`** — One Python module per aspect with `applies(diff_text) -> bool` and `score(diff_text) -> float | None`. Each module is required to lean on the strongest tool available (linters, AST parsers, complexity analyzers); regex on code is allowed only with an explicit `# REGEX_OK:` annotation. Aspects that genuinely cannot be measured deterministically are marked `CLASSIFICATION = "THICK"` and abstain. The runner builds a `(N × 2M)` matrix of `(score, applied)` columns and trains an RF / L1 LR on it.
- **`judge/`** — A v2 LLM judge (Llama-3.3-70B-FP8 by default) is fed pre-rendered prompt files, returns a JSON verdict per (text, aspect), and the parser distils them into a flat `judge_scores.jsonl`.

## Key files

```
methods/existing_metrics_runner/
├── coded/
│   ├── GUIDE.md                       # authoritative spec for metric authors
│   ├── runner.py                      # apply all metrics to fixtures, dump stats
│   ├── eval_combined.py               # build (score, applied) matrix → RF/LR CV AUC
│   ├── sandbox.py                     # subprocess + tool whitelist (lizard, ruff, eslint, …)
│   ├── check_no_regex_on_code.py      # soft check: flag re.* without REGEX_OK
│   ├── audit_regex_when_degenerate.py # diagnose constant-output metrics
│   ├── debug_per_fixture.py           # single-metric × single-fixture trace
│   ├── metrics/a{ID}_{slug}.py        # 123 per-aspect modules (a0…a410+)
│   └── fixtures/sample_prs.json       # 22 (diff, label) tuples for local iteration
└── judge/
    ├── sk3_v2_judge_runner.py         # vLLM batch judge driver
    └── v2_parse_judge_responses.py    # raw JSON → judge_scores.jsonl
```

## How to run

**Coded backend, local sanity check:**
```bash
python -m methods.existing_metrics_runner.coded.runner
python -m methods.existing_metrics_runner.coded.eval_combined
python methods/existing_metrics_runner/coded/check_no_regex_on_code.py
```

**Judge backend, sk3 vLLM:**
```bash
export PROMPT_DIR=runs/validity_full/v2/<task>/judge_prompts
export RESPONSE_DIR=runs/validity_full/v2/<task>/judge_responses
export MODEL_DIR=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/<sha>
export TP_SIZE=2 GPU_MEM_UTIL=0.93 MAX_MODEL_LEN=16384 BATCH_FLUSH=200
python methods/existing_metrics_runner/judge/sk3_v2_judge_runner.py

# Then flatten to JSONL:
python methods/existing_metrics_runner/judge/v2_parse_judge_responses.py \
    --run-dir runs/validity_full/v2/<task>
```

Optional env vars for the judge runner: `KV_CACHE_DTYPE`, `TEMPERATURE`, `MAX_TOKENS`, `PROMPT_LIST_FILE` (ordered subset).

## Dependencies

- `coded/`: `whatthepatch`, `unidiff`, `pandas`, `scikit-learn`, `numpy`, plus tools on `$PATH` declared in `sandbox.ALLOWED_TOOLS` (`lizard`, `ruff`, `radon`, `eslint`, `mypy`, `pylint`, `bandit`, `semgrep`, `vulture`, `jscpd`, `cppcheck`, `clang-tidy`, `cpplint`, `flawfinder`, `yamllint`, `detect-secrets`, `hadolint`, `sqlfluff`, `gofmt`, `prettier`, `google-java-format`).
- `judge/`: vLLM (sk3 BF16 / FP8 recipes in `feedback_sk3_afs_tokens.md`, `reference_sk3_vllm_bf16.md`), an FP8 / BF16 Llama-70B or Qwen snapshot under `/lfs/skampere3/0/shared_hf_cache/`.

## Current state

- **`coded/`** — actively in progress. 123 metric modules under `coded/metrics/` (mix of `a0…a410+` IDs). Designed as the principled successor to the `runs/validity_full/v2/{task}/codegen_claude/` regex programs, which collapsed to text-length proxies (AUC ≈ 0.59 on code_review). Per `GUIDE.md`, a Tier-3 `lizard` features pilot moved code_review AUC from 0.616 → 0.627; per-norm coverage is the goal. Fixtures are 22 PRs, so AUCs here are directional only — the real eval is sk3's 4978-row dataset.
- **`judge/`** — working and used in production. Drives the v2 judge sweep that populates `outputs/v2_db/cells_v1/`.

## Outputs

- **`coded/`** — stdout per-metric table from `runner.py`; combined RF/LR CV AUC from `eval_combined.py` (no file artifacts by default; intended to feed `outputs/v2_db/` cells once wired in).
- **`judge/`** — one JSON per (text × aspect bundle) under `RESPONSE_DIR`, flattened into `runs/validity_full/<run>/judge_scores.jsonl` (`key`, `bundle_id`, `paraphrase_idx`, `chunk_idx`, `aspect_id`, `datapoint_id`, `applicable`, `score`, `reason`). These flow into `outputs/v2_db/cells_v1/task=<task>/judge=<judge>/data.parquet`.

## Related

- `methods/existing_metrics_runner/coded/GUIDE.md` — metric authoring contract (tiers, REGEX_OK annotations, THICK semantics).
- `runs/validity_full/v2/{task}/codegen_claude/` — the legacy regex-program corpus this method is meant to replace (per-task counts: peer_review 654, creative_writing 1104, code_review 1182, humor 1098, patents 748, news_homepages 621, notice_and_comment 597, press_releases 972, math 732).
- `reference_codegen_per_aspect_programs.md`, `project_code_review_verifiability_plan.md`, `project_verifiability_explainability_gaps.md` — design context.
- `outputs/v2_db/cells_v1/` — canonical store for judge-cell counts/coverage (`reference_cells_db.md`).
- `methods/verification_library/` — sibling method that writes one program per **example** to predict the label directly (this method writes one program per **norm** that returns a conformance score).
