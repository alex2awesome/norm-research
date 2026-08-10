# Press releases

A press-release "newsworthiness" task: given the body of a press release, predict whether it was picked up by a top-domain news outlet. The label is derived from observed press-release ↔ news-article links across two complementary scrapes.

## 1. Task

**Binary classification.** Input is the press-release body text. Target column `judgement ∈ {0, 1}` indicates whether the release was covered by at least one tracked news article from a sufficiently common outlet.

The canonical column layout (see [`evals.py`](evals.py)) carries:

- `press_release_id`, `text` (PR body), `press_release_url`, `press_release_date`, `press_release_company`
- `news_article_id` (list), `news_article_domain` (list), `judgement`

Eval slices are defined per company, per news domain (top domains with ≥ 50 rows), and per year — see [`evals.py`](evals.py).

## 2. Sources

Two scraped link-graphs are merged in [`build_press_release_dataset.py`](build_press_release_dataset.py):

1. **Forward** (`Press Releases -> News Articles`): a CSV `article-to-pr-mapper.csv.gz` of `(news_url, press_release_url, date_first_seen, company_name, news_url_domain)`, joined against an on-disk HuggingFace dataset `all-coref-resolved/` to recover article and PR body text via URL match.
2. **Backward** (`Press Releases <- News Articles`): a SQLite mirror `article_to_press_release_data.db` whose tables (`article_to_href`, `press_release_data`, `article_data`, `article_map`) yield additional `(press_release, news_article)` pairs where the href on a news article was flagged `is_press_release = 1`.

Both raw artifacts (plus a `full-source-scored-data.jsonl.gz` and a demo notebook) live under [`raw_data/`](raw_data/).

URL normalization, SURT → standard URL conversion, and `tldextract`-based domain extraction are done up front so that the same PR or article from different scrapes deduplicates correctly. See `normalize_url`, `surt_to_standard_url`, and `extract_domain_and_subdomain` in [`build_press_release_dataset.py`](build_press_release_dataset.py).

## 3. Collection / processing scripts

| Script | Purpose |
|---|---|
| [`build_press_release_dataset.py`](build_press_release_dataset.py) | Merges forward + backward scrapes, dedupes into normalized `press_releases.csv`, `news_articles.csv`, `press_release_news_mappings.csv`. |
| [`build_pr_article_pairs_full.py`](build_pr_article_pairs_full.py) | Builds positive PR ↔ article pairs (`pr_article_pairs_full.jsonl`) from the clean modeling splits: explodes `news_article_id` lists, applies PR length filter (200–12 000 chars), article length filter (300–15 000 chars), drops near-duplicate article/PR (`overlap_ratio ≥ 0.85` or exact equality), and optionally stratified-downsamples to ~200 K by outlet. |
| [`../../scripts/process_press_releases_vllm.py`](../../scripts/process_press_releases_vllm.py) | VLLM offline pass that extracts the **clean press-release body** out of raw scraped pages (strips chrome, returns `{news_release_found, result}` JSON). |
| [`../../scripts/postprocess_press_release_extraction.py`](../../scripts/postprocess_press_release_extraction.py) | Parses the VLLM output (markdown fences, partial truncations) into clean `extracted_text` + `news_release_found` flags. |
| [`evals.py`](evals.py) | Per-company / per-domain / per-year slices and macro-AUC summary metrics. Consumed by the global eval runner (`scripts/eval_model.py`). |

## 4. File layout

```
datasets/press-releases/
├── README.md                                            (this file)
├── build_press_release_dataset.py                       canonical merge of forward + backward scrapes
├── build_pr_article_pairs_full.py                       positive PR↔article pair JSONL builder
├── evals.py                                             slice / summary metric definitions
├── raw_data/                                            (gitignored) source CSVs, sqlite, HF dataset
│   ├── article-to-pr-mapper.csv.gz                      forward mapper
│   ├── article_to_press_release_data.db.tar.gz          backward sqlite
│   ├── all-coref-resolved/                              HF dataset for article/PR body text
│   └── full-source-scored-data.jsonl.gz
├── press_releases.csv.zip                               (gitignored) normalized PR entity table
├── news_articles.csv  (+ .zip)                          (gitignored) normalized news entity table
├── press_release_news_mappings.csv                      (gitignored) (pr_id, article_id, was_covered)
├── press_release_modeling_dataset.csv.gz                (gitignored) original v1 row-level modeling CSV
├── press_release_modeling_dataset.csv/                  (gitignored) train/eval/test split CSVs
│   ├── train.csv, eval.csv, test.csv
├── press_release_modeling_dataset_clean.csv.gz          ← canonical (see §5)
├── press_release_clean.jsonl.gz                         (gitignored) cleaned VLLM-extracted PR bodies
├── pr_article_pairs.jsonl                               (gitignored) initial positive pairs (~53 MB)
├── pr_article_pairs_full.jsonl                          (gitignored) full positive pairs build (~2 GB)
├── rubrics.jsonl                                        rubric vocab over PR-article pairs
├── validation_sample_100.jsonl / _clean.jsonl           hand-validation samples
├── press_release_modeling_dataset.csv__doc_to_topic.csv topic-model assignments
├── press_release_modeling_dataset.csv__topic_to_words.csv
├── online-rubrics/                                      AAAS / AP / Bernays / boorstin / etc.
│   ├── raw/                                             scraped style-guide markdown
│   ├── claude-parsed/                                   Claude-cleaned versions
│   ├── gpt-parsed/gpt-5-mini/                           GPT-extracted rubric metric JSON
│   └── urls-visited.csv, waveh*_log.csv, waveh*_seen.txt
└── sheldon_results/lasso_nested_cv_results_20260329_114302.json
```

The repo-root `.gitignore` excludes all the bulk CSV/JSONL files (`raw_data/`, `*.csv`, `*.csv.gz`, `*.jsonl`, `press_release_modeling_dataset*`, `news_articles*`, `press_releases.csv.zip`, etc.), so only build scripts, the rubric files, validation samples, and topic outputs are checked in.

## 5. Canonical dataset file

| Use this | Path | Stats |
|---|---|---|
| ✅ Primary | `press-releases/press_release_modeling_dataset_clean.csv.gz` (118 MB, Apr 8) | 128 131 rows, 53 780 positive / 74 351 negative (from running-research-notes §2.4) |

Per `reference_clean_datasets_per_task.md` and §2.4 of `running-research-notes.md`, this is the current canonical "clean" build used for all modeling. **No group-split rebuild has been done yet** — leakage audit (`notes/dataset_leakage_audit.md`, row "press-releases") flags outlet / industry / length / year confounders as **not yet end-to-end audited**, and there is no group key chosen (`reference_clean_datasets_per_task.md` table marks group-split as `✗ (TODO)`).

The positive-pair JSONL built from this file is `pr_article_pairs_full.jsonl` (~2 GB, Jun 2), which is the input to the in-flight norm-extraction pipeline (`project_norm_extraction_overnight_2026_06_02.md`).

For sk3 mirroring, the expected path is `/lfs/skampere3/0/alexspan/norm-research/datasets/press-releases/press_release_modeling_dataset_clean.csv.gz`.

## 6. Modeling state

### Dense reward models (Llama-8B LoRA, sk3, `runs/press_release_*`)

From `project_dense_model_sweeps.md` and `project_press_release_results.md`:

| Subset | Rows | #Trials | med AUC | max AUC | min AUC |
|---|---|---|---|---|---|
| 0.1 | ~6 K | 13 | 0.653 | 0.667 | 0.529 |
| 0.2 | ~12 K | 13 | 0.677 | 0.690 | 0.588 |
| 0.3 | ~18 K | 3 | 0.691 | 0.693 | 0.615 |
| 0.5 | ~30 K | 1 | 0.687 | 0.687 | 0.687 |
| 0.6 | ~36 K | 3 | 0.701 | 0.708 | 0.701 |
| 0.7 | ~42 K | 3 | 0.707 | 0.709 | 0.705 |
| 0.8 | ~48 K | 3 | 0.705 | 0.717 | 0.699 |
| 1.0 | ~60 K | 3 | 0.711 | 0.712 | 0.505 |

**Saturated around subset 0.6–0.7 (~36–42 K rows), plateau ~0.71 AUC.** Variance is high — some trials at 0.1/0.4/0.9/1.0 fail and collapse to ~0.50 min AUC.

- **Llama-70B** (`runs/press_release_sweep_llama-70b`): only subset 0.1 run; med AUC 0.649 → **no improvement** over 8B at the same size.
- **Bradley–Terry pairwise variant** (`runs/press_release_sweep_bradley_terry_llama-8b`): only subset 0.1, med AUC 0.615 → **underperforms** pointwise.

Other run directories under `runs/`: `press_release_clean_sweep_llama8b`, `press_release_sweep_llama-8b`, `press_releases_8b`.

### Rubric-based methods

From `project_press_release_results.md`:

- **Iterative Autometrics** — 25 iterations, best test ROC-AUC ~**0.585** (iter 17), final ~**0.534**. Eval gate plateaus ~0.56. 26 metrics accumulated; test AUC frequently dips below 0.5 (overfitting).
- **Metric Tree** — only grew the root node (no exception children). Test accuracy 0.531; no articulability gap. Five metrics: Evidence_Quality, Societal_Impact_Scope, Source_Credibility, Novelty_and_Timeliness, Audience_Relevance.

**Gap.** Rubric ≈ 0.53–0.58 vs dense ≈ 0.71 — a substantial articulability gap (the task's signal is real but hard to verbalize). Dense ceiling at 0.71 is consistent with either a label-quality cap or a genuine task-difficulty cap (April 2026 validation of cleaned VLLM-extracted PR bodies found some `judgement` labels are likely wrong, supporting the label-noise theory).

## 7. Key decisions

- **Label = top-domain news pickup.** Binary `judgement` is derived from whether any tracked news article links to the PR. Eval slices threshold news domains at ≥ 50 rows (`evals.py`); per-domain macro-AUC reported alongside the global metric.
- **Forward + backward merge.** Two independent scrapes (URL-mapper CSV + sqlite href graph) are concatenated and entity-deduplicated by `press_key` / `news_key` (URL → normalized URL → SHA1 of body text fallback) so the same PR found from both directions counts once.
- **URL normalization + SURT decoding** before deduping (otherwise CommonCrawl-style SURT entries fail to match standard URLs).
- **Body-text cleanup via VLLM** (`process_press_releases_vllm.py`) — strips nav, footers, cookie banners, social links from scraped HTML before modeling. This produced the `clean` variants.
- **Pair-builder filters** (`build_pr_article_pairs_full.py`): PR length 200–12 000 chars, article length 300–15 000 chars, drop pairs where article and PR are identical or overlap ≥ 0.85 token-Jaccard, optional stratified down-sample to ~200 K by outlet.
- **Group-split: not yet applied.** Currently row-level random splits. `reference_clean_datasets_per_task.md` flags this as a TODO, and a group key has not been chosen (candidates: `press_release_company`, time bucket).

The user prompt mentioned XBRL / duplicate filtering / `go_factset` exclusion as suspected decisions — these strings do not appear in the build scripts or notes searched (`grep` for `XBRL`, `go_factset`, `top.?domain` returned nothing in `datasets/press-releases/`). If these filters are applied, they live elsewhere (likely sk3-side preprocessing) and should be documented when located.

## 8. Open questions / next steps

- **Group-split rebuild** (per `reference_clean_datasets_per_task.md`): pick a group key (`press_release_company`? year bucket?) and rebuild splits so the same company / time period does not appear in both train and test. Until then, the 0.71 dense ceiling may be partially inflated by company-level memorization.
- **Outlet / industry / length / year confound audit** flagged but not done (`notes/dataset_leakage_audit.md` row 2, status `❓`).
- **Label noise.** Validation of VLLM-cleaned bodies (April 2026) suggested some `judgement = 0/1` labels are likely wrong. Path forward to push past 0.71: relabel a held-out validation set or reframe target (e.g., predicted-count of pickups, or specific outlet pickup), not collect more data.
- **Norm extraction in flight** (per `project_norm_extraction_overnight_2026_06_02.md`): 2 366-pair sample with rubric vocab from the R2 hierarchy, building toward an articulable-norm library. Inputs are `pr_article_pairs_full.jsonl`; rubrics in [`rubrics.jsonl`](rubrics.jsonl) and the online style guides under [`online-rubrics/`](online-rubrics/).
- **Online-rubrics expansion**: ~13 wave logs of AAAS/AP/Bernays/Boorstin/etc. style-guide scrapes parsed by Claude and GPT-5-mini, intended as priors for the rubric-method extraction (`online-rubrics/gpt-parsed/gpt-5-mini/*.json`).
- **Llama-70B** has only been tried at subset 0.1 with no gain — open whether scaling base model would help at saturation size, but cheap dense-AUC evidence so far says no.
- **Sheldon LASSO baseline**: nested-CV result stored at [`sheldon_results/lasso_nested_cv_results_20260329_114302.json`](sheldon_results/lasso_nested_cv_results_20260329_114302.json) — useful as a linear-feature baseline against the dense Llama numbers.
