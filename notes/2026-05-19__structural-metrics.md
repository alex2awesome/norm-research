# Structural / cross-task metrics on the locked rubric clustering

Date: 2026-05-19 (overnight autonomous run)

## Setup

Computed on the **locked leaf clustering** — the canonicalize → judge → distill
→ blend → average-linkage pipeline, cut at **tau 0.825**:

- 53,413 canonical rubric forms (11 tasks × general/specific/hyper_specific)
- → **33,123 clusters** (distinct concepts), held-out FP ~10.6% / FN ~9.8%
- Artifacts: `match_out/clusters_<task>.json` on sk3; metrics JSON in
  `outputs/analyses/structural_metrics/`.

Scripts: `sk3_structural_metrics.py`, `sk3_cross_task_concepts.py`.

## 1. Per-task concentration

| task | forms | clusters | %singleton | compression | entropy(norm) | Zipf slope | Gini | top-10 cov |
|---|---|---|---|---|---|---|---|---|
| code-review | 12,262 | 8,118 | 79% | 1.51× | 0.96 | −0.44 | 0.30 | 3.5% |
| creative-writing | 4,950 | 2,941 | 74% | 1.68× | 0.95 | −0.52 | 0.35 | 5.4% |
| grant-funding | 2,836 | 1,657 | 79% | 1.71× | 0.93 | −0.52 | 0.38 | 10.2% |
| humor | 5,885 | 3,407 | 75% | 1.73× | 0.95 | −0.53 | 0.37 | 6.5% |
| legal-outcome-prediction | 3,303 | 2,268 | 81% | 1.46× | 0.96 | −0.41 | 0.28 | 7.5% |
| math-stackexchange | 5,257 | 3,381 | 77% | 1.55× | 0.96 | −0.47 | 0.31 | 4.6% |
| news-homepages | 3,022 | 1,610 | 75% | 1.88× | 0.92 | −0.57 | 0.42 | 12.5% |
| notice-and-comment | 2,881 | 2,046 | 84% | 1.41× | 0.96 | −0.40 | 0.27 | 6.2% |
| patents | 4,130 | 2,397 | 76% | 1.72× | 0.93 | −0.51 | 0.38 | 11.6% |
| peer-review | 3,257 | 1,871 | 76% | 1.74× | 0.94 | −0.54 | 0.38 | 7.6% |
| press-releases | 5,630 | 3,427 | 77% | 1.64× | 0.95 | −0.50 | 0.35 | 6.2% |

**Findings.**
- **The rubric space is a long tail, not concentrated.** Normalised entropy is
  0.92–0.96 (≈uniform) and the top-10 clusters cover only 3.5–12.5% of forms.
  No task is dominated by a handful of mega-concepts — expert evaluation
  criteria are diffuse.
- **news-homepages is the most concentrated** (top-10 = 12.5%, Gini 0.42, Zipf
  −0.57) — a few criteria ("be accurate", "be balanced") dominate journalism.
  **notice-and-comment, legal, code-review are the flattest** (Gini 0.27–0.30)
  — their criteria are uniformly diffuse, many one-off.
- Compression is modest (1.4–1.9×) because 74–84% of forms are singletons. The
  per-bucket breakdown (§2) shows this is mostly by design.

## 2. Specificity profile

Per-bucket forms → clusters (compression, % singleton):

| task | general | specific | hyper_specific |
|---|---|---|---|
| code-review | 2519→1592 (1.6×, 42%s) | 8350→6187 (1.4×, 57%s) | 1393→771 (1.8×, 35%s) |
| creative-writing | 4950→2941 (1.7×, 44%s) | — | — |
| grant-funding | 1389→772 (1.8×, 39%s) | 1227→838 (1.5×, 52%s) | 220→168 (1.3×, 55%s) |
| humor | 4158→2431 (1.7×, 40%s) | 1639→1204 (1.4×, 49%s) | 88→74 (1.2×, 54%s) |
| legal | 1578→983 (1.6×, 43%s) | 1654→1357 (1.2×, 67%s) | 71→58 (1.2×, 56%s) |
| math | 2378→1340 (1.8×, 34%s) | 2184→1729 (1.3×, 61%s) | 695→560 (1.2×, 61%s) |
| news-homepages | 2497→1319 (1.9×, 38%s) | 453→340 (1.3×, 48%s) | 72→56 (1.3×, 52%s) |
| notice-and-comment | 1350→888 (1.5×, 48%s) | 1494→1215 (1.2×, 69%s) | 37→34 (1.1×, 78%s) |
| patents | 132→89 (1.5×, 34%s) | 3846→2256 (1.7×, 44%s) | 152→123 (1.2×, 59%s) |
| peer-review | 1431→811 (1.8×, 39%s) | 1746→1122 (1.6×, 46%s) | 80→69 (1.2×, 58%s) |
| press-releases | 3743→2196 (1.7×, 41%s) | 1400→1130 (1.2×, 59%s) | 487→360 (1.4×, 50%s) |

**Findings.**
- **The `general` bucket compresses most** (1.5–1.9×, lowest singleton rate
  34–48%) — broad rubrics genuinely repeat across guides. **`specific` and
  `hyper_specific` stay near-singletons** (50–78% singleton) — they are
  particular by construction, and canonicalization was explicitly told to
  preserve specificity. The clustering behaves exactly as the bucket design
  intends.
- **Tasks differ sharply in their general/specific mix.** patents is 93%
  *specific* (regulatory/technical criteria are particular); creative-writing
  is 100% *general*; notice-and-comment is ~half specific. This mix — not a
  clustering artifact — drives the apparent compression rate.

## 3. Most-redundant concept per task (largest cluster)

The single most-repeated piece of expert advice, per task:

- code-review — "consistent indentation" (73 forms)
- creative-writing — "show, don't tell" (42)
- grant-funding — "well-justified, reasonable budget" (44)
- humor — "incongruity / unexpected combinations" (114)
- legal — "clear, simple language, avoid jargon" (97)
- math-stackexchange — "identify and consider the audience" (35)
- news-homepages — "the story should be accurate" (96)
- notice-and-comment — "support evaluation with empirical evidence" (23)
- patents — "enablement: a person of ordinary skill can make/use it" (110)
- peer-review — "include appropriate references" (31)
- press-releases — "target a specific, well-understood audience" (51)

## 4. Cross-task universal concepts

Method: each multi-member cluster (7,451 of them) → representative text →
**leading "<subject> should/must" clause stripped** (removes the task-noun
confound: "the *code* should be clear" vs "the *writing* should be clear") →
bge-large embedding of the bare predicate → complete-linkage across tasks.

At cosine ≥ 0.86: 6,561 meta-clusters, of which **58 span ≥ 3 tasks**
(108 at ≥ 0.82; 30 at ≥ 0.90).

**Finding: only a thin universal layer; >99% of concepts are domain-specific.**
The cross-domain concepts are exactly the generic communication-quality
dimensions:

- **concise / brief** — 9 tasks (the single most universal concept)
- **clear** — 7 tasks
- **well-organized** — 6 tasks
- **accurate** — 6 tasks
- **complete** — 6 tasks
- **specific** — 5 tasks; **states a clear main idea** — 5 tasks
- 4-task: **consistent style, appropriate sample size, original/creative,
  tailored to audience, free of grammar/spelling errors, precise**
- 3-task: documented, naming conventions, internally consistent, effective word
  choice, truthful/accurate, relevant, attention to detail, correct citation…

Everything else — the other ~7,400 multi-member concepts and ~25,700 singletons
— is domain-specific. Expert evaluation criteria are overwhelmingly
task-particular; the shared layer is a small set of writing-quality basics.

## 5. Task-family structure (shared-concept matrix)

Counting concepts (meta-clusters, cos ≥ 0.86) shared between each task pair
reveals three groups plus two outliers:

- **Communication / journalism cluster** — press-releases ↔ news-homepages share
  **40** concepts (by far the highest pair); press also links creative (19),
  humor (11). press-releases is the most *central* task — it shares with
  everything.
- **Technical / academic cluster** — math ↔ code-review (23), code ↔ peer-review
  (22), math ↔ peer-review (21), math/code/peer/patents all inter-linked.
- **Creative pair** — creative-writing ↔ humor (22).
- **legal** sits between the two clusters (links press 20, math 14, creative 13).
- **Outliers** — grant-funding and notice-and-comment share very little with any
  task (row sums lowest). Their evaluation criteria are the most idiosyncratic /
  domain-bound of the eleven.

## 6. Cross-source consensus (strength of norms)

A cluster of 5 forms from one expert guide is intra-document repetition; 5
forms from 5 guides is a genuine shared norm. Each form's provenance key
(`task::source_dir::source_file::idx`) gives the source, so distinct source
files per cluster separates the two.

**Finding: 98% of multi-member clusters are cross-source.** Only 153 of 7,451
multi-member clusters (2%) draw all their forms from a single document. The
clustering is overwhelmingly capturing concepts that *independent* expert
sources arrive at separately — genuine shared norms, not one guide repeating
itself. This is also a strong validation of the clustering: it is not gluing
together within-document boilerplate.

Per task, multi-member clusters by source spread:

| task | multi-cl | ≥3 sources | ≥5 sources | max sources |
|---|---|---|---|---|
| code-review | 1,730 | 652 | 226 | 70 |
| creative-writing | 761 | 332 | 120 | 40 |
| grant-funding | 346 | 162 | 70 | 42 |
| humor | 861 | 376 | 151 | 88 |
| legal | 419 | 176 | 57 | 87 |
| math-stackexchange | 787 | 329 | 112 | 34 |
| news-homepages | 406 | 196 | 87 | 93 |
| notice-and-comment | 333 | 148 | 60 | 23 |
| patents | 567 | 253 | 89 | 98 |
| peer-review | 455 | 221 | 97 | 31 |
| press-releases | 786 | 334 | 133 | 49 |

`n_distinct_sources` is a usable **norm-strength** measure. The strongest norms
— criteria stated by the most independent expert sources:

- **patent enablement** ("a person of ordinary skill can make/use it") — 98 sources
- **news accuracy** ("the story should be accurate") — 93 sources
- **humor incongruity** ("unexpected combinations / contrasts") — 88 sources
- **legal plain language** ("clear, simple, avoid jargon") — 87 sources
- **code indentation** ("proper and consistent indentation") — 70 sources
- code max-line-length (64), patent claim clarity (60), news balance (56),
  patent industrial application (56), press-release audience targeting (49)…

These are the field-defining criteria of each domain — what virtually every
expert guide independently insists on.

## 8. Data and artifact locations (overall online-fetching pipeline)

End-to-end flow:

```
raw HTML/PDF  →  LLM extraction  →  rubrics.parquet  →  canonicalize (Llama-3.3-70B)
   →  canon_all_real_forms.jsonl  →  judge-distilled LoRA + ModernBERT-CE
   →  kNN + blend affinity  →  average-linkage @ tau 0.825  →  clusters_<task>.json (L0)
   →  per-batch LLM grouping  →  r1_families_<task>.json (R1)
   →  (R2/R3 to come)
```

### Raw online-fetching per task — local

Per-task directories under `datasets/<task>/online-rubrics/`:

- `raw/` — cached source pages (HTML / PDF), one file per visited URL.
- `gpt-parsed/` and `claude-parsed/` — LLM-extracted rubrics per page (JSON
  with `name`, `description`, `guidance`).
- `urls-visited.csv` (or `rej_log.csv` for legal / notice-and-comment) —
  visit / rejection log.

11 tasks: code-review, creative-writing, grant-funding, humor,
legal-outcome-prediction, math-stackexchange, news-homepages,
notice-and-comment, patents, peer-review, press-releases.

### Aggregated leaves / rubrics — local

- `notebooks/_explore_cache/pages.parquet` (~38K pages) — page metadata only
  (source, URL, etc.). **No full text.** Loads in a notebook in seconds.
- `notebooks/_explore_cache/rubrics.parquet` (~72K raw extracted rubrics) —
  name + description + guidance + provenance.
- **Provenance key** format throughout: `task::source_dir::source_file::idx`
  (the `idx` is the rubric index within the source file).
- `outputs/analyses/canon_all_real_forms.jsonl` — **53,413** canonical leaves
  (mirror at sk3 `/lfs/skampere3/0/alexspan/norm_embed/canon_all_real_forms.jsonl`).
  Each line: `{task, bucket, idx, key, canonical}`. **This file is the entry
  point to the clustering pipeline.**

### Judges and training pairs — sk3 `/lfs/skampere3/0/alexspan/norm_embed/`

- `all_verdicts.jsonl` — **434K** judged pairs (`{task, key_a, key_b,
  canonical_a, canonical_b, score (0/1/2), split (train/eval)}`).
- `all_train_pairs.jsonl` — full training pairs.
- `judge_prompt.py` — v6 graded judge prompt (used by sk3_judge_pairs.py).

### Distilled judges per task — sk3 `norm_embed/`

- `lora_adapters/<task>/` — LoRA-bge adapter (CoSENT, judge-distilled).
- `ce_models/<task>/` — ModernBERT-base cross-encoder (soft-label BCE on
  `score/2`, ρ≈0.83 with judge).

### Embeddings — sk3 `norm_embed/out/`

- `emb_bge_<bucket>_<task>.npy` — base-bge embeddings of canonical text,
  pooled order matches `keys_<task>.json`. **Used for cross-task analysis
  and R1 batching.**
- `emb_bge_canon_<task>.npy` — alternate per-task layout.
- `emb_lora_<bucket>_<task>.npy` — LoRA-adapted bge (per-task, not
  cross-task comparable).

### L0 clustering artifacts — sk3 `norm_embed/match_out/`

- `clusters_<task>.json` — `{key: cluster_id}` for the locked tau-0.825
  clustering. **The L0 = "same rubric" partition.**
- `keys_<task>.json` — the pooled row order (matches the emb arrays).
- `Z_avg_<task>.npy` — the locked average-linkage dendrogram (cut at any tau
  → relabel via scipy `fcluster`).
- `Z_blend_<task>.npy`, `Z_ce_<task>.npy` — earlier complete-linkage variants
  retained for ablation.
- `scored_<task>.npz` — `{ii, jj, ce}` cached CE scores for candidate pairs
  (re-blending / re-cutting needs no GPU).

### R1 hierarchy — sk3 `norm_embed/match_out/r1/`

- `r1_families_<task>.json` — `{families: [{family_id, name, description,
  cluster_ids}], cluster_to_family: {cluster_id: family_id}}`. **The R1
  partition.**
- `r1_raw_<task>.jsonl` — per-batch raw LLM JSON for debugging / regrading.

### Structural / cross-task metrics — local

`outputs/analyses/structural_metrics/`:

- `concentration.json`, `specificity.json`, `top_clusters.json`,
  `consensus.json`, `consensus_top_concepts.json`,
  `universal_concepts_v2.json`, `task_pair_sharing.json`.
- Mirrored `clusters_<task>.json` and `r1_families_<task>.json` from sk3 for
  local inspection.

### Models on sk3 shared cache

`/lfs/skampere3/0/shared_hf_cache/`:

- `models--meta-llama--Llama-3.3-70B-Instruct/snapshots/<hash>/` — BF16 70B
  for canonicalization and R1/R2/R3 grouping.
- `models--BAAI--bge-large-en-v1.5/snapshots/<hash>/` — base embeddings.
- `models--answerdotai--ModernBERT-base/snapshots/<hash>/` — CE base.

### Build scripts (local `scripts/`, synced to sk3 `norm_embed/`)

- `sk3_canonicalize_vllm.py` — leaf canonicalization with Llama-3.3-70B.
- `sk3_judge_pairs.py` — v6 judge on training pairs.
- `sk3_train_lora.py`, `sk3_train_ce.py` — distillation.
- `sk3_match_pipeline.py` — kNN candidates → CE re-rank → blend affinity →
  complete-linkage. Produces `scored_<task>.npz`, `Z_ce_<task>.npy`.
- `sk3_finalize_clusters.py` — average-linkage at locked tau, writes
  `Z_avg_<task>.npy` and `clusters_<task>.json`.
- `sk3_blend_sweep.py`, `sk3_linkage_test.py`, `sk3_operator_sweep.py` —
  design sweeps that locked the recipe.
- `sk3_structural_metrics.py`, `sk3_cross_task_concepts.py`,
  `sk3_consensus_metrics.py` — structural / cross-task metrics.
- `sk3_build_r1.py` — R1 family build (current).

## Files

- `outputs/analyses/structural_metrics/concentration.json`
- `outputs/analyses/structural_metrics/specificity.json`
- `outputs/analyses/structural_metrics/top_clusters.json`
- `outputs/analyses/structural_metrics/universal_concepts_v2.json` (58 concepts)
- `outputs/analyses/structural_metrics/task_pair_sharing.json` (11×11 matrix)
- `outputs/analyses/structural_metrics/consensus.json` (per-task source spread)
- `outputs/analyses/structural_metrics/consensus_top_concepts.json` (top 200 by source count)

## 7. Hierarchy plan (4-level, in progress 2026-05-20)

The old build_hierarchy/meta_merge R1/R2/R3 design was muddled and not cleanly
defined by levels of generality. Locking a clean 4-level grain. Each level
asks a slightly different "same?" question, and the granularity follows:

- **L0 Cluster** — *same rubric, restated*. Output of the locked clustering
  pipeline. Validated against the v6 judge: pairs **within** a cluster would
  mostly score 2 (same); pairs **across** clusters mostly 1 (related-but-
  different). Example: code-review's 73-member "consistent indentation"
  cluster. **~33K clusters across 11 tasks.**
- **R1 rule** — *different rubrics, same underlying principle*. Bundles L0
  clusters whose pairwise judge score would mostly be 1, but which all enforce
  one principle. Example: ten distinct L0 clusters ("results shouldn't exceed
  the evidence", "tone matches evidence strength", "moderate unsupported
  claims", "no causation language without support" …) → one R1 rule
  "match claims to evidence strength." Compression observed in test: **~2–3×
  per batch** (the data has many genuinely distinct rules). Target: ~12–15K R1.
- **R2 criterion family** — *different principles, same aspect of evaluation*.
  Merges R1 rules that address one facet. Example: "match-claims-to-evidence" +
  "critical-appraisal-of-sources" + "rebuttal-evidence" + "evidence-thresholds"
  → R2 "evidence-grounded reasoning." Target: ~1.5–3K R2 (~5–10× from R1).
- **R3 theme** — *different aspects, same broad concern*. Example:
  "evidence-grounded reasoning" + "logical coherence" + "audience appropriate-
  ness" → R3 "argument quality." Target: ~200–300 R3 (~10× from R2).
- *(Optional R4 — top-level domain areas. ~20–30 per task. The "headline
  dimensions" each task is evaluated on.)*

Each level gets its own prompt asking *its* same-? question (same rubric, same
principle, same aspect, same concern). Levels are produced with batched LLM
calls (anchor + kNN candidates, average-linkage / consensus across overlapping
batches). R1 build is in progress on sk3 with Llama-3.3-70B BF16.

## Next steps (not done — flagged for review)

- **R1 build** on sk3 with Llama-3.3-70B — in progress. R2/R3 to follow once
  R1 is validated.
- **Two-axis classification** (articulability × surface) re-run on the new
  clusters — LLM + a calibration loop that wants user review.
- **E6 — inarticulability corpus analysis** — blocked: needs full source-page
  *text*; `pages.parquet` holds only metadata, page text not located in the
  explore cache.
