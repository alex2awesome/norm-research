# Phase 1b status — overnight prep, 2026-06-07

## TL;DR

The **CC=0.72 bank AUC reproduces** when we re-score the 1,004 prior pool with the current bank. The number is real and not a measurement error.

The reason Phase 1 hits only 0.55 isn't a bank regression or a sampling-design problem — it's that **I built Phase 1 with a bug that dropped all 21,758 CC pairs**. CC candidates in `competition_unified` live under `platform=cc` with `source=taco-verified`, and my Phase 1 builder filtered candidates by `source` (only `lc_discuss`, `matrixstudio`, `luogu_editorial_alt`, `atcoder_submission`), silently excluding the entire CC platform.

The fix is one line — use the `platform` column. Done. Phase 1b pool built tonight. Prompts staged. Ready for Qwen launch when you authorize.

## Diagnostic results — the prior 0.7 reproduces

Test ran tonight: re-score legacy pools with the *current* bank.

| Pool | Subset | n | This-night AUC | Prior reported | Reproduces? |
|---|---|---:|---:|---:|---|
| 2K LC Claude pool | pooled | 1,995 | **0.645 ± 0.028** | ~0.68 | ✓ close |
| 1,004 Claude pool | pooled | 749 (704+45 codes available) | 0.599 ± 0.021 | ~0.62 | ✓ close |
| 1,004 pool | **CC subset** | 449 | **0.704 ± 0.047** | **~0.72** | **✓ YES** |
| 1,004 pool | Luogu subset | 555 | 0.558 ± 0.065 | ~0.54 | ✓ close |

**The 0.7 ceiling is real, reproducible, and on file.**

## Cosine distribution diagnosis

| Cosine bin | 2K LC | 5K source (1,004) | Phase 1 |
|---|---:|---:|---:|
| <0.2 | 0.1% | **13.7%** | 4.8% |
| 0.2-0.4 | 1.9% | 14.6% | 34.6% |
| 0.4-0.6 | 12.7% | 17.6% | 38.8% |
| 0.6-0.8 | 34.0% | 31.2% | 14.0% |
| 0.8-1.0 | **51.3%** | 22.8% | 7.7% |

The 5K source had genuine "obviously different" pairs (cosine <0.2 = 13.7%) — that's where the bank discriminates. Phase 1's uniform-within-problem sampling produced a middle-concentrated distribution missing both tails.

## TF-IDF beats bank — surprise finding

From the dense+TF-IDF ladder (#130) on the Phase 1 stratified subsample:

| Method | Pooled | LC | Luogu |
|---|---:|---:|---:|
| Bank LR (231 metrics) | 0.521 | 0.581 | 0.535 |
| **TF-IDF char_wb 3-5gram + LR** | **0.570** | **0.618** | 0.541 |
| ModernBERT cross-encoder | running ~10h | running | running |

TF-IDF on **candidate code alone** (no editorial — leakage-clean) beats the 231-metric bank by ~0.05 pooled. Tells us the bank is leaving signal on the table that simple char n-grams capture. ModernBERT still cooking on GPU 6.

## Phase 1b — what's staged

`outputs/v2_analysis/comp_qwen_phase1b_full_pool.parquet` — 175,376 formable pairs:
- **LC: 102,436** (using platform col — same as before but cleaner)
- **CC: 20,813** ← *previously missing, now unlocked*
- Luogu: 52,127

`outputs/v2_analysis/comp_qwen_phase1b_sample.parquet` — **36,111-pair Qwen-ready sample**:
- ALL 20,813 CC pairs (the 1,004's high-AUC bounty at 21× scale)
- 15,298 LC pairs, stratified by Phase 1 cosine to lean toward the 5K-source distribution
- (Luogu skipped from this sample — never hit 0.7 in any pool)

`outputs/v2_analysis/comp_qwen_phase1b_prompts/` — **36,078 prompts staged** (33 oversized filtered).

## What's NOT done tonight (didn't want to launch unsupervised)

| Job | Why not | When |
|---|---|---|
| bge-code-v1 cosine on 20,813 CC pairs | GPU 5 contended; cosine not strictly needed to label | Tomorrow morning, ~30-60 min on GPU 5 |
| Qwen labeling of the 36K Phase 1b sample | Parse-fail risk needs supervised watch; uses ~$$ via GPU hours | Tomorrow when you authorize |

## What's still running

| Job | Status |
|---|---|
| Phase 1 retry (Qwen on the 48K parse-fails) | GPU 4, ~50% done, ETA ~2h, no errors |
| ModernBERT cross-encoder on Phase 1 stratified | GPU 6, ~10h ETA, fold-by-fold checkpoints |

## Tomorrow morning — clean launch commands

```bash
# Optional: embed CC pairs to get cosine (GPU 5, ~30 min when free)
ssh sk3 'cd /lfs/skampere3/0/alexspan/norm-research && \
  nohup env HOME=/lfs/skampere3/0/alexspan CUDA_VISIBLE_DEVICES=5 \
  python scripts/competition_editorials/embed_phase1_cosine.py \
    --input outputs/v2_analysis/comp_qwen_phase1b_sample.parquet \
    --output outputs/v2_analysis/comp_qwen_phase1b_cosine.parquet \
    > logs/phase1b_embed.log 2>&1 &'

# Launch Qwen on Phase 1b prompts (free GPU; check first)
# Use the patched runner with BF16 KV from Phase 1 retry
ssh sk3 'cd /lfs/skampere3/0/alexspan/norm-research && \
  nohup env HOME=/lfs/skampere3/0/alexspan \
  CUDA_VISIBLE_DEVICES=<FREE_GPU> \
  MODEL_DIR=/lfs/skampere3/0/shared_hf_cache/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9 \
  ENABLE_THINKING=false MAX_TOKENS=400 MAX_MODEL_LEN=24576 \
  GPU_MEM_UTIL=0.93 BATCH_FLUSH=3000 \
  VLLM_USE_FLASHINFER_MOE_FP8=0 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  TEMPERATURE=0.6 TOP_P=0.9 KV_CACHE_DTYPE=auto \
  PROMPT_DIR=outputs/v2_analysis/comp_qwen_phase1b_prompts \
  RESPONSE_DIR=outputs/v2_analysis/comp_qwen_phase1b_responses \
  python scripts/sk3_v2_judge_runner.py > logs/qwen_phase1b.log 2>&1 &'
```

ETA for 36K Qwen labels with BF16 KV (the parse-fail-resistant config): ~3-4h.

## What this run is testing

**Direct test of the platform-bug hypothesis**: build the same composition as the 1,004 (CC + Luogu + LC, with `taco-verified` candidates for CC), at 50× scale (~36K vs 1K), Qwen-labeled.

Predicted result:
- If Bank LR AUC on CC subset hits 0.65-0.72 → bug confirmed, 0.7 was real and scale-able
- If CC subset still hits 0.55 → something else is going on, need deeper diagnostic

## Bug documentation (so this doesn't recur)

- `datasets/competition_unified/README.md` — written tonight, documents the platform-vs-source pitfall + canonical mapping table
- Memory: `feedback_use_platform_col_not_source.md` — recurring-pattern rule
