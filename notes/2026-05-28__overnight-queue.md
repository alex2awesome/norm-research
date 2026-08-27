# Overnight Qwen queue (auto-orchestrated by wakeup loop)

## Current stage: S0 (running)

**Stage 0: top-500 dps × all aspects × p0 (in flight)**
- Launched 22:50 UTC May 27 with 47,600 prompts (partial sync of 107K top-1000 dps)
- Note: Pool was built from partially-synced top-1000 prompts. Since claude-prioritized
  dps come first in dp ordering for most tasks, ~500 dps worth of p0 prompts are in
  the pool. Some tasks fully covered, others partial.
- 6 Qwens, BATCH_FLUSH=200, GPU_MEM_UTIL=0.93, MAX_MODEL_LEN=64000
- ETA ~12-15h on 6 GPUs at 47 prompts/min observed throughput
- Should complete around midday tomorrow

## Stage 1: S0 mop-up (after S0 dies / completes)

When `pgrep sk3_v2_judge_runner | wc -l == 0`:
1. SSH sk3: `python3 scripts/build_cells_db.py --judges qwen_thinking_fp8_20x1_r2post` (ingest S0 responses)
2. SSH sk3: `python3 scripts/v2_build_20x1_r2post.py --top-k 500 --bundle-size 15 --paraphrase 0 --out-subdir judge_prompts_15x1_s1_p0` (builds cells STILL MISSING for top-500 panel)
3. If prompt count > 100: rebuild pool, shard 7 ways (use GPU 1 too), launch 7 Qwens
4. If prompt count ≤ 100: skip stage 1, go to stage 2 (already complete enough)

## Stage 2: 100-dp panel × all aspects × p1

After S1 (or skipped):
1. Ingest S1 to cells DB
2. `python3 scripts/v2_build_20x1_r2post.py --top-k 100 --bundle-size 15 --paraphrase 1 --out-subdir judge_prompts_15x1_s2_p1`
3. Build pool, shard 7 ways, launch 7 Qwens

## Stage 3: 100-dp panel × all aspects × p2

After S2:
1. Same as S2 but `--paraphrase 2 --out-subdir judge_prompts_15x1_s3_p2`

## Stage 4: dps 501-1000 × all aspects × p0

After S3:
1. `python3 scripts/v2_build_20x1_r2post.py --dp-start 500 --dp-end 1000 --bundle-size 15 --paraphrase 0 --out-subdir judge_prompts_15x1_s4_p0`
2. Build pool, shard 7 ways, launch 7 Qwens

## State file

`outputs/v2_analysis/queue_state.txt` on local + sk3:
- Line 1: current stage (S0, S1_running, S2_running, ...)
- Line 2: stage-start-timestamp UTC

The wakeup checks this file + Qwen PIDs to decide what to do.

## Failure modes to watch

1. **Qwen processes die mid-stage**: lose in-flight work since last 200-flush. Restart same stage with the dynamic builder (it'll skip already-scored cells).
2. **Sk3 cluster maintenance**: same as #1.
3. **Truncation patch fails on a prompt**: would have caught all but extreme edge cases. v8 patch is enable-thinking aware.

## Constraints / sanity checks

- Never exceed 7 GPUs (GPU 0 has unrelated user process)
- Never set GPU_MEM_UTIL > 0.93 (we saw cascades at 0.95)
- Never set BATCH_FLUSH > 200 (we lose too much on death)
- Always verify processes via the 3 checks before reporting "running"