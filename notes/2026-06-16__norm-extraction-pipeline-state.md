# Norm-extraction pipeline state — 2026-06-16 02:00 UTC

## Goal recap

Per `/goal` set 2026-06-15:
> we want to process and extract explicit feedback, or anchor metrics, given by humans when they either review their preferences or return their verdicts. ... extract supersets of free-text norms ... go through all datasets you have on disk

## Pipeline architecture

- **Runner**: `/lfs/skampere3/0/alexspan/scripts/llama_norm_extraction/run_sk3_batch.py` (patched 2026-06-15)
- **Model**: Qwen3.5-122B-A10B-FP8 on B200 via vLLM offline batch
- **Output**: per-batch chunked gzipped JSONL (`<base>.chunks/chunk_<ts>_<N>.jsonl.gz`) — fixes the long-running `gzip.open("at")` corruption (`feedback_no_long_running_gzip_append`)
- **Validator**: lenient mode — retries only on structural breaks + loop strings, not on OOV rubric IDs or invalid enums (was eating 65% throughput before patch)

## Tasks defined in `TASK_SOURCES` (23 total)

### Already extracted (chunks dir exists)
| Task | Records (pre-resume) | Source | Status |
|---|---:|---|---|
| peer_review | 89,529 / 258K | unified academic reviews | **active in current queue** (170K to go) |
| code_review | 92,997 / 141,644 PRs | GitHub PR comments | queued |
| humor_multi | 27,366 / 49,762 threads | reddit jokes+standup+AST+McSweeney | queued |
| press_releases_full | 82,600 / 100K pairs | editorial PR↔article | queued |
| notice_and_comment | 3,644 / 3,642 RTCs | agency responses (v1) | **done** |
| notice_and_comment_v2 | 7,902 / 7,902 RTCs | agency responses (backfill) | **done** |

### New corpora — input.jsonl + config + few-shots staged
| Task | Records | Source | Wave | In current queue? |
|---|---:|---|:---:|:---:|
| crse | 14,302 | CR.SE propensity-balanced answers | 1 | yes |
| law_se | 146,281 | Law.SE Comments.xml + PostHistory | 1 | yes |
| reddit_supremecourt | 58,062 | r/supremecourt v2 author-group balanced | 1 | yes |
| competition_editorials | 27,994 | LC+CF+CC editorials | 2 | yes |
| wp_comments | 20,366 | r/WritingPrompts depth≥1 feedback | 2 | yes |
| legaladvice_uk | 50,000 | LegalAdviceUK + offtopic | 2 | yes |
| math_se | 100,000 | Math.SE Comments+PostHistory sampled | 2 | yes |
| nc_public_comments | 35,258 | distinct from RTCs | 3a | **no — next wave** |
| litbench_rationales | 43,736 | LLM-written craft rationales | 3a | **no — next wave** |
| courtlistener_opinions | 30,000 | judicial reasoning (TVII/FLSA/SS) | 3a | **no — next wave** |
| aops_forum | 30,000 | math olympiad discussion | 3a | **no — next wave** |
| nlrb_decisions | 5,604 | ALJ + Board labor-law decisions | 3b | **no — next wave** |
| ttab_inter_partes | 2,054 | trademark adjudication | 3b | **no — next wave** |
| bva_opinions | 15,000 | veterans-law sampled | 3b | **no — next wave** |
| cavc_decisions | 15,000 | appellate veterans-law sampled | 3b | **no — next wave** |
| ptab_fwd | 4,863 | patent final written decisions | 3b | **no — next wave** |
| dol_arb | 1,742 | DOL whistleblower/wage-hour | 3b | **no — next wave** |

## Current launch (active)

```
ssh sk3 # PID 3776684
HOME=/lfs/skampere3/0/alexspan \
CUDA_VISIBLE_DEVICES=1 \
VLLM_USE_FLASHINFER_MOE_FP8=0 \
nohup /lfs/skampere3/0/alexspan/envs/ai_usage/bin/python -u /lfs/skampere3/0/alexspan/scripts/llama_norm_extraction/run_sk3_batch.py \
  --tasks peer_review,code_review,humor_multi,press_releases_full,crse,competition_editorials,wp_comments,legaladvice_uk,reddit_supremecourt,math_se,law_se \
  --batch-size 2000 \
  --gpu-mem-util 0.93 \
  --max-model-len 32768 \
  > /lfs/skampere3/0/alexspan/logs/all_tasks_chunked.log 2>&1 &
```

Throughput: **22.4/min** (batch 1 done at 5347s / 2000 prompts; 59% retry rate persists despite lenient validator — retries are JSON parse failures + loop strings). ETA for full 11-task queue: **~10 days**.

## Next launch (queue this when current PID 3776684 dies)

```
ssh sk3
HOME=/lfs/skampere3/0/alexspan \
CUDA_VISIBLE_DEVICES=1 \
VLLM_USE_FLASHINFER_MOE_FP8=0 \
nohup /lfs/skampere3/0/alexspan/envs/ai_usage/bin/python -u /lfs/skampere3/0/alexspan/scripts/llama_norm_extraction/run_sk3_batch.py \
  --tasks nc_public_comments,litbench_rationales,courtlistener_opinions,aops_forum,nlrb_decisions,ttab_inter_partes,bva_opinions,cavc_decisions,ptab_fwd,dol_arb \
  --batch-size 2000 \
  --gpu-mem-util 0.93 \
  --max-model-len 32768 \
  > /lfs/skampere3/0/alexspan/logs/all_tasks_wave2_3.log 2>&1 &
```

Total records: ~184K. ETA: **~6 days**.

## Grand total when done

- Current queue: ~480K records
- Next queue: ~184K records
- Plus already-extracted: ~303K (post-migration)
- **Grand total norm extractions: ~970K records across 23 distinct corpora**

## Pending audits / deferred

- **#134 CourtListener full opinion-clusters bulk** — `/lfs/.../bulk_data/` snapshot 2025-07-02 IS complete (57GB); slice_opinions.jsonl.gz already used. Newer 2026-03-31 snapshot is partial (no opinions text file). Defer broader-corpus extraction until needed.
- **#137 RoyalRoad reviews** — chapter text scraped; READER REVIEWS need separate scrape (not blocked, just lower priority)
- **#138 humor contest blurbs** — sparse per user's table; defer
- **CR.SE expansion** — current input is 14,302 (propensity-balanced); full pool 80K available if we want larger sample later
- **SO/dba/codegolf SE answers** — not commentary-on-others but technical answers; if expanded, would be ~756K more records, multi-week additional run

## Patches deployed today (2026-06-15→16)

1. `run_sk3_batch.py:284` — `gzip.open(output_path, "at")` → chunked write per batch (root-cause fix for `feedback_no_long_running_gzip_append`)
2. `is_bad_output()` — removed enum strict-check + OOV rubric_matches check (root-cause fix for "validator over-strict" 65% retry rate noted in prior session)
3. `migrate_legacy_to_chunks()` helper added — recovered 303,038 records from 6 corrupted legacy `.jsonl.gz` files (peer_review 88,529 / code_review 92,997 / humor_multi 27,366 / press_releases_full 82,600 / N&C v1 3,644 / N&C v2 7,902)
4. Original `extracted_qwen.jsonl.gz` files renamed `.legacy_corrupted` (per `feedback_never_delete_data` — preserved)
5. Backup of original runner at `_archive/run_sk3_batch_pre_chunked.py.bak`

## Open questions for user

- After this completes (~16 days serial on 1 GPU), do we want to:
  - (a) Expand CR.SE / Math.SE / N&C from sample → full corpus (10× longer runs)
  - (b) Scrape RoyalRoad reviews + add as task
  - (c) Re-run earlier extractions with v2 rubric vocabulary (bootstrap from this round's signal_text)
  - (d) Move to downstream analysis: cluster signal_texts into rubrics, score each as "anchor metric m" candidate
