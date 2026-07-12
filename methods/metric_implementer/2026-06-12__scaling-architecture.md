# Scaled E7 — batch architecture (2026-06-12)

Goal (user): scale articulability measurement across **many metrics × many datasets × many
models**, batch-processing with **vLLM offline** (batches of metrics × datapoints), keeping an
output keyed by *(judge, item, (prompt, iteration))* plus the GEPA operator labels
(INIT / CLARIFY / MECHANIZE / FEWSHOT+ / ANCHOR / EDGE / PRUNE / DECOMPOSE), and **saving every
prompt and iteration**. sk3-only, no OpenRouter ([[feedback_metric_implementer_sk3_only]]).

## The decoupling that makes it scale on 1 GPU

GEPA per-tier optimization needs the judge tier AND a stronger reviser resident *simultaneously*
— two big models, won't fit one GPU. So we **decouple search from measurement**, keeping exactly
one large model resident at a time:

```
SEARCH (scale.improve_all)         MEASURE (scale.score_all)
─────────────────────────          ─────────────────────────
resident: STRONG model only        resident: ONE judge tier at a time
per (dataset×metric×seed×cap):      per tier:
  GEPA loop mints a lineage of        union ALL (metric × prompt-version × item × pass)
  prompt iterations, each             across every dataset → ONE vLLM generate() (mega-batch)
  persisted IMMUTABLY to Registry   → stream the long table (LONGTABLE_COLUMNS)
  (every prompt + iteration saved)  → per-tier articulability frontier falls out, no 2nd model
```

The per-tier frontier is recovered in MEASURE (each tier scores every iteration), so we do not
need per-tier GEPA — a strict simplification of the E7 plan that trades the H1 "per-tier
optimization beats fixed rubric" test for 1-GPU tractability. (Per-tier GEPA remains available
when two models fit, e.g. 8B judge + 70B-FP8 reviser on a B200.)

## Modules (all under `methods/metric_implementer/`)

| module | role |
|---|---|
| `vllm_backend.py` | `OfflineVLLM` (resident-model singleton, drop-in for `LLMBackend`; sk3 env: HOME→/lfs, FLASHINFER check off, dtype auto, prefix caching, kv-cache auto-not-fp8) + `FakeVLLM` (role-routing, GPU-free, deterministic) |
| `batch_scoring.py` | `LONGTABLE_COLUMNS` schema, `LongTableWriter` (parquet parts, checkpointed, no gzip-append), `ScoreJob`, `batch_score_many` (mega-batch demux → long rows) |
| `manifest.py` | `RunManifest` / `DatasetEntry`; metric loaders (trial ladders, **online-rubrics** `extracted.rubrics_metrics`), corpus loader; `pilot_manifest` + `full_manifest` (9 local datasets × online-rubric banks) |
| `scale.py` | `improve_all` (SEARCH) + `score_all` (MEASURE) orchestrator + CLI |
| `run_scale_sk3.sh` | nohup launcher with the sk3 env |

## The long table (the deliverable)

One row per **(judge_model × metric × prompt-version × item × pass)**:

```
run_id, dataset, task, judge_model, metric_id, version_id, operator, round,
token_cap, pass, item_id, score, applicable, prompt_hash, ts
```

`version_id` = the prompt (full body + lineage in the Registry); `operator` = the GEPA op that
minted it; `round` = the iteration; `judge_model` = the capability-axis tier; `pass` = test-retest
replicate. The E7 brackets / frontier / words_share are computed from this table offline
(`e7_brackets.py`, to be extended to read the long table directly).

## What the inventory found (2026-06-12 sweep) — the scale-out fuel

- **~69K parsed online-rubric metrics** across 11 task dirs (`datasets/*/online-rubrics/
  gpt-parsed/gpt-5-mini/*.json`), uniform schema `{name, description, guidance}` →
  `manifest._online_rubric_metrics` loads them; per-task bank size is a knob.
- **9 locally-available labeled datasets** (text col `text`, label `judgement`; legal uses
  `facts`/`binary_label`). creative_writing + code_review are sk3-only; press_releases corrupted.
- **Canonical vLLM recipe** baked into `OfflineVLLM`: BF16/FP8 via `dtype='auto'`,
  `kv_cache_dtype='auto'` (fp8 → `!!!!` garbage), `enable_prefix_caching`, HOME pin, FlashInfer
  version check off, Qwen `enable_thinking=False`.

## Status / verification

GPU-free dry-run (`scale all --fake`) runs the full SEARCH→MEASURE path: mints operator-labeled
iterations (INIT/CLARIFY/MECHANIZE/ANCHOR/DECOMPOSE across rounds 0–2) and streams a long table
with all tiers/operators/rounds. 15/15 offline tests pass. `--fake` without `--out-root` redirects
to `/tmp` scratch so dry-runs never touch real registries.

## Follow-ups before the real sk3 run
1. Wire a `full` manifest name into `scale._manifest`; tune `metrics_per_task` / `metric_files_cap`.
2. Add a `(version_id, doc_id, tier)` **score cache** (metric_tree has `LabelCache`; we re-score) —
   big saving when re-measuring after adding tiers.
3. Decide the reviser model for SEARCH on sk3 (default Qwen2.5-72B resident); per-tier GEPA only
   where two models co-fit.
4. Extend `e7_brackets.py` to read the parquet long table (currently reads the registry/triad).
5. Per-task config presets are in place for all 9 tasks (correct judge framing — guards the
   cross-task prompt bug).
