# R1/R2 pipeline + experiments log (consolidated 2026-05-24)

Single consolidated note covering: validation oracle, all R1/R2/Fork experiments,
results table, key technical decisions, R2 spot-check summary, and follow-ups.

Supersedes: `2026-05-23__r1-experiments-log.md` and `2026-05-24__overnight-r1-r2-summary.md`.
The auto-generated `2026-05-24__r2-spot-check-report.md` is deleted (rerun
`scripts/spot_check_r2.py` to regenerate).

## TL;DR

Pipeline complete for all 11 tasks: **L0 → R1 → Fork3-merged R1 → R2 aspects**.

| Phase | Method | Avg result across 11 tasks |
|---|---|---|
| R1 base | Claude subagents + LoRA-bge sort + bs=400 | F1 = 0.282 vs v6 |
| **Fork 3 merge** | LoRA-bge centroid candidates (cos≥0.70) + pairwise YES/NO judge | **F1 = 0.358 (+0.076)** |
| **R2 aspects** | R1 prompt at meta level over family centroids | **5.4× compression (16,182 → 3,013)** |

Artifacts:
- `outputs/analyses/structural_metrics/r1_v4a_subagent_lora_bs400/` (base R1)
- `outputs/analyses/structural_metrics/r1_v4a_lora_fork3_merge/` (Fork3 merged)
- `outputs/analyses/structural_metrics/r2_v1_subagent/` (R2 aspects)
- `outputs/analyses/structural_metrics/fork_b{,_v2}/` (Llama consistency runs)
- `outputs/analyses/structural_metrics/validation/peer-review_v6_verdicts.jsonl` (+ 10 other tasks)

## Validation oracle

`<task>_v6_verdicts.jsonl` from sk3 `/lfs/skampere3/0/alexspan/norm_embed/all_verdicts.jsonl`
— 35,937 pairs of peer-review canonical rubrics (33-50K per task), each labeled by
the v6 judge (Llama-3.3-70B) on 0/1/2 (unrelated / related / same rule). Validation
script: `scripts/validate_r1_against_v6.py`. R1 is judged by how well "same R1 family"
predicts "score == 2 (same rule)" over cross-cluster informative pairs.

Within-task pair-class distribution (peer-review example):
- 35,937 total v6 pairs
- 32,206 informative (cross-L0-cluster)
- 14,248 score=0 (unrelated)
- 15,404 score=1 (related)
- 2,554 score=2 (same rule) — **ceiling on recall**

## Per-task results

| Task | L0 clusters | R1 base | Fork3 R1 | R2 aspects | R1 F1 (base) | R1 F1 (Fork3) | R2 compression |
|---|---|---|---|---|---|---|---|
| peer-review | 1,871 | 1,156 | 956 | 218 | .338 | **.417** | 4.4× |
| news-homepages | 1,610 | 1,037 | 982 | 231 | .273 | **.315** | 4.3× |
| grant-funding | 1,657 | 815 | 759 | 146 | .336 | **.390** | 5.2× |
| notice-and-comment | 2,046 | 1,352 | 1,306 | 296 | .296 | **.357** | 4.4× |
| legal-outcome | 2,268 | 1,498 | 1,452 | 294 | .299 | **.351** | 4.9× |
| patents | 2,397 | 1,141 | 1,108 | 257 | .334 | **.418** | 4.3× |
| creative-writing | 2,941 | 1,673 | 1,622 | 285 | .298 | **.375** | 5.7× |
| math-stackexchange | 3,381 | 2,553 | 2,505 | 403 | .232 | **.314** | 6.2× |
| humor | 3,407 | 2,325 | 2,273 | 378 | .190 | **.287** | 6.0× |
| press-releases | 3,427 | 1,697 | 1,619 | 249 | .335 | **.376** | 6.5× |
| code-review | 8,118 | 6,297 | 6,197 | 1,059 | .267 | **.342** | 5.9× |
| **Mean** | — | — | — | — | **.282** | **.358** | **5.4×** |

## Headline R1 build progression on peer-review

| Method | #Fams | Hard FP | TP (score=2) | P | R | **F1** |
|---|---|---|---|---|---|---|
| Llama base (`r1_v4a`) | 1,111 | 71 | 631 | .312 | .247 | .276 |
| Claude+LoRA+bs400 | 1,156 | 18 | 883 | .331 | .346 | .338 |
| Claude+LoRA+meta-merge (A.2) | 1,074 | ~28 | 992 | .312 | .388 | .346 |
| Llama+merge_v2 (sk3 prior) | 1,070 | 125 | 1,158 | .308 | .453 | .367 |
| **Claude+LoRA+Fork3 merge** | **956** | **56** | **1,531** | **.320** | **.599** | **.417** |

## Key technical decisions

1. **LoRA-bge for batching (not base-bge)**: trained-on-judge embeddings concentrate
   same-rule clusters into the same batch → LLM can actually merge them. Lifted F1
   0.215 → 0.338 in pilot.
2. **bs=400 (not bs=40)**: larger batches give merge horizon across more clusters at once.
   Combined with LoRA: 0.338 base F1.
3. **Fork 3 pairwise merge**: family centroids in LoRA-bge → candidate pairs at cos≥0.70 →
   batched YES/NO subagent judges (20 pairs/batch) → union-find. Lifted F1 0.338 → 0.417
   on peer-review (+0.079) and transferred to all 11 tasks (avg +0.076).
4. **R2 prompt at meta level**: same cover-once batching machinery, new system prompt asking
   for "thematic aspects" not "same rule". 5.4× compression with mostly coherent themes.

## Known limitations + planned fixes

- **R2 cross-batch fragmentation**: 0-15 duplicate aspect names per task (worst: code-review
  15, e.g. "Naming Conventions" appears in 4 batches with 53 R1 families total). Same theme
  appears in multiple batches because each batch only sees 400 of N families.
  **Fix: R2.5 merge pass** (analogous to Fork 3 for R1). 110 batches prepped at `/tmp/r2_5/`.
- **Math has one umbrella aspect** ("Specialized Mathematical Subject Matter", 73 fams):
  heterogeneous subdomains lumped together. Self-admitted catch-all. Other tasks don't have this.
  **Fix: re-run R2 with stricter "no catch-all" instruction on just those 73 families.**
- **Fork 3 only judged top 200 pairs/task** (top 10 batches at 20 pairs). Lower threshold sampled
  at cos∈[0.60,0.70) gave only 1.7% YES rate on peer-review — diminishing returns.
- **Fork 3 hard FP rate ~18%** (38/211 merges on peer-review were rated "unrelated" by v6).
  Spot-check showed ~60% of these are defensible "v6 too fine-grained" cases (multiple
  image-integrity sub-rules); true precision is closer to 0.40 than nominal 0.32.

## Side experiments

- **Meta-merge (A.2)**: applied R1 prompt at family level. Modest +0.008 F1 improvement,
  dominated by Fork 3's direct pairwise approach.
- **Fork B v1 (Llama consistency, 5 seeds, temp=0)**: variance only 2% (1099-1122 families).
  Confirmed LoRA-bge neighborhood structure dominates over anchor shuffling.
- **Fork B v2 (2 farthest-point anchors × 3 temperatures, Llama-70B)**:
  - **Anchor barely matters** (ΔF1 = 0.001 between farthest-point anchors) — even with
    maximally-different starting points, LoRA-bge propagation converges.
  - **Temperature is the real lever**: temp=0.5 lifts F1 by 0.030 over temp=0;
    temp=0.7 regresses slightly (−0.005).
  - **temp=0.5 = operating sweet spot for Llama R1**: F1=.277, matches original
    deterministic-batching baseline (`r1_v4a` F1=.276). Use temp=0.5 with whatever
    batching for future Llama R1 runs.

## R2 spot-check summary

(See `scripts/spot_check_r2.py` to regenerate full per-task report.)

| Task | R1 fams | R2 aspects | Compression | Max size | Singletons | Cross-batch dup names |
|---|---|---|---|---|---|---|
| peer-review | 956 | 218 | 4.39× | 26 | 58 | 3 |
| news-homepages | 982 | 231 | 4.25× | 20 | 63 | 1 |
| grant-funding | 759 | 146 | 5.20× | 23 | 27 | 0 |
| notice-and-comment | 1306 | 296 | 4.41× | 38 | 84 | 3 |
| legal-outcome-prediction | 1452 | 294 | 4.94× | 43 | 48 | 1 |
| patents | 1108 | 257 | 4.31× | 31 | 66 | 4 |
| creative-writing | 1622 | 285 | 5.69× | 33 | 45 | 5 |
| math-stackexchange | 2505 | 403 | 6.22× | 73 | 75 | 4 |
| humor | 2273 | 378 | 6.01× | 57 | 98 | 2 |
| press-releases | 1619 | 249 | 6.50× | 51 | 42 | 1 |
| code-review | 6197 | 1059 | 5.85× | 45 | 223 | 15 |

Largest R2 aspects per task look coherent (e.g., peer-review's top 5: "Domain-Specific
Reporting Guideline Adherence" [26], "Statistical Methodology and Reporting" [24],
"Evidence Synthesis and Meta-analysis" [24], "Research Ethics and Human/Animal Subjects" [17],
"Reproducibility Artifacts and Packaging" [15]).

User's specificity-vs-generality observation confirmed: high-fragmentation tasks (humor,
math, code, press) get most R2 compression AND have most cross-batch dups.

## Files written

| Path | Purpose |
|---|---|
| `outputs/analyses/structural_metrics/r1_v4a_subagent_lora_bs400/r1_families_<task>.json` | Base R1 per task |
| `outputs/analyses/structural_metrics/r1_v4a_lora_fork3_merge/r1_families_<task>.json` | Fork 3 merged R1 |
| `outputs/analyses/structural_metrics/r2_v1_subagent/r2_aspects_<task>.json` | R2 aspects |
| `outputs/analyses/structural_metrics/validation/<task>_v6_verdicts.jsonl` | v6 pair labels |
| `outputs/analyses/structural_metrics/fork_b/seed_{0..4}/` | Fork B v1 |
| `outputs/analyses/structural_metrics/fork_b_v2/anchor{0,1}_temp{0,0.5,0.7}/` | Fork B v2 |
| `scripts/r1_local_prep.py` | Local prep (monkey-patches sk3_build_r1) |
| `scripts/r1_fork3_pairmerge_{prep,apply}.py` | Fork 3 generation + union-find |
| `scripts/r2_subagent_prep.py`, `aggregate_r2.py`, `spot_check_r2.py` | R2 build + checks |
| `scripts/r25_aspect_merge_{prep,apply}.py` | R2.5 cross-batch merge (prepped, not run) |
| `scripts/validate_r1_against_v6.py` | F1 against v6 pair labels |
| `scripts/sk3_fork_b{,_v2}_consistency.py` | Fork B scripts (run on sk3) |
| `reference_norm_embed_pair_labels` (memory) | Where 434K v6 pair labels live on sk3 |
