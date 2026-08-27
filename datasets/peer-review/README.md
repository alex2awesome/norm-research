# Peer Review

The most-developed dataset in `norm-research/`. A multi-venue corpus pairing
academic manuscripts with their reviews and accept/reject decisions, used as
the primary task for verifiability / articulability / taste decomposition,
dense reward modeling, and the verification-library pipeline.

## 1. Task

- **Primary target**: binary accept / reject of a submitted paper given
  its review text (and optionally the abstract / full PDF text).
- **Secondary target**: review-score regression (reviewer ratings,
  normalized to `[0, 1]` per venue scale in `unify_datasets.py`).
- **Canonical labeled split**: `splits/train.csv.gz` (56,153 rows), columns
  `paper_id, source, venue, year, domain, text, judgement, id`. The
  `judgement` column is the binary label at the user's preferred threshold
  and must be used as-is. Companion files `splits/eval.csv.gz`,
  `splits/test.csv.gz`.

## 2. Sources and coverage

Per `PEER_REVIEW_DATABASE.md` (the SQLite snapshot dated March 2026) and the
`unify_datasets.py` merge logic:

| Venue | Years | Papers | Reviews | PDFs | Decision labels | Notes |
|-------|-------|-------:|--------:|-----:|-----------------|-------|
| ICLR | 2020–2025 | ~27,800 | ~100K | ~28K | accept/reject | **Load-bearing source.** Shows every submission, including rejects. |
| NeurIPS | 2021–2025 | ~18,800 | ~70K | ~13K | accept/reject | Rejected papers are opt-in only (~5%). |
| ICML | 2023–2025 | ~7,900 | ~28K | ~6K | accept/reject | Same opt-in caveat as NeurIPS. |
| COLM | 2024–2025 | ~720 | ~2.7K | ~700 | accept/reject | |
| TMLR | rolling | ~6,400 | ~23K | ~12K | accept/reject | Rolling journal; `year` may be NULL. |
| EMNLP | 2023–2025 | ~6,000 | ~21K | ~6K | accept/reject | |
| eLife | 2022–2026 | ~4,300 | ~15K | ~8K | **assessment only** | No binary accept/reject; only eLife assessment text. |
| F1000Research | — | ~11,369 | — | — | 3-level decision | From JATS XML public corpus. |
| PeerRead legacy | ACL 2017, CoNLL 2016, ICLR 2017 | — | — | — | accept/reject | NLP coverage. |

**Unified snapshot (Mar 2026)**: 82,460 papers / 299,961 reviews /
43,332 accept / 21,880 reject across ICLR / NeurIPS / ICML / TMLR / eLife /
F1000 / COLM / ACL / CoNLL.

Also present but **excluded from `unify_datasets.py`**:

- `orb-dataset/` — SciPost physics + small workshops, metadata only, no review text.
- `berenslab-iclr-dataset/` — overlaps with the OpenReview ICLR crawl.
- `Seafoodair-Openreview/` — ICLR 2021–2022 analysis artifacts, not raw text.

## 3. Collection and preprocessing scripts

### Crawlers / fetchers

- `PeerRead/code/data_prepare/crawler/openreview_crawl.py` — OpenReview v1 + v2
  API. Supports ICLR, NeurIPS, ICML, COLM (year-based) and TMLR (rolling).
  v2 returns empty replies for pre-2024 venues (ICLR 2022–2023); a v1
  per-paper forum fetch fallback was added in `crawl_venue()` on 2026-03-18.
  Credentials use `OPENREVIEW_USERNAME=spangher@usc.edu`.
- `PeerRead/code/data_prepare/crawler/neurips_crawl.py`,
  `NIPS_crawl.py` — NeurIPS-specific paths.
- `PeerRead/code/data_prepare/crawler/backfill_revision_pdfs.py` — pulls
  revision PDFs via OpenReview Edit-API attachment references (22,625 papers
  have multiple PDF versions).
- `elife/fetch_elife_reviews.py` — eLife public REST API; 4,286 reviewed
  preprints with assessments. No API key required.
- `elife/download_pdfs.py`, `elife/parse_pdfs.py` — PDFs pulled from
  `elifesciences/enhanced-preprints-data` GitHub mirror; text parsed with
  pymupdf.
- `f1000research/fetch_f1000_reviews.py` — parses JATS XML from the F1000
  public corpus; 11,369 articles with 3-level review decisions.

### Unification / extraction

- `unify_datasets.py` — merges OpenReview, PeerRead legacy, eLife, F1000
  into `unified_papers.csv.gz` and `unified_reviews.csv.gz`. Domain
  classification + per-venue score normalization to `[0, 1]` lives here.
- `extract_unified.py` — extracts the `/tmp/peer_review.db` SQLite into
  `extracted/peer_review_unified.parquet` (and a 100-row sample). Drops
  reviews shorter than 50 chars, normalizes the `OPENREVIEW` venue label
  (re-tagging TMLR), parses leading numeric scores.
- `/scripts/link_reviews_to_peer_review_dataset.py` — joins non-meta
  reviews onto `peer_review_modeling_dataset.csv.gz` and writes
  `_with_reasoning.csv.gz` (concatenated reviews per paper).
- `analyze_all_data.py`, `evals.py` — descriptive stats and quick eval
  helpers.

## 4. File layout

### Local Mac

```
datasets/peer-review/
├── PEER_REVIEW_DATABASE.md           # schema doc for peer_review_pdfs.db
├── peer_review_pdfs.db.gz            # 13 GB SQLite (1.9 GB gzipped)
├── peer_review_modeling_dataset.csv.gz  # 32 MB pre-rewrite modeling file
├── unified_papers.csv.gz             # 43 MB, one row per paper
├── unified_reviews.csv.gz            # 241 MB, one row per review
├── unify_datasets.py
├── extract_unified.py
├── analyze_all_data.py
├── evals.py
├── splits/{train,eval,test}.csv.gz   # canonical labeled splits (judgement col)
├── extracted/
│   ├── peer_review_unified.parquet
│   ├── sample_100.parquet
│   ├── smoke_v1/
│   └── smoke_v2/
├── online-rubrics/                   # scraped venue review guidelines
│   ├── raw/         (~2,770 HTML/PDF docs)
│   ├── claude-parsed/ (~167 parsed JSONs)
│   └── gpt-parsed/
├── berenslab-iclr-dataset/           # excluded (overlaps OpenReview)
├── casimir/                          # article pair / mapping JSONLs
├── elife/{data, fetch, download, parse}.py
├── f1000research/{data, fetch}.py
├── orb-dataset/                      # excluded (metadata only)
├── PeerRead/                         # legacy code + ACL/CoNLL/ICLR2017 data
└── Seafoodair-Openreview/            # excluded (analysis artifacts)
```

### sk3 mirror

`/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/` carries the
same layout. Norm extraction output lands at
`/lfs/skampere3/0/alexspan/norm-research/data/peer_review/norm_extracted/extracted_qwen.jsonl.gz`.

## 5. Canonical dataset file

For all v2 validity / modeling pipelines, use:

```
datasets/peer-review/splits/train.csv.gz
```

- 56,153 rows, columns `paper_id, source, venue, year, domain, text,
  judgement, id`.
- Already balanced via stratified sample; `judgement` is the binary label
  set by the user during dense reward model training (per
  `reference_v2_task_datasets.md`).
- **Do not re-threshold from `review_score`** — keep `judgement` consistent
  with the dense sweep numbers below.
- **Known caveat**: this file has **not yet been rebuilt with `paper_id` as
  the group key**. Train/test leakage across papers is flagged in
  `notes/dataset_leakage_audit.md` and in `reference_clean_datasets_per_task.md`.
  Production runs should wait for the group-split rebuild; use `paper_id`
  for the regroup.

## 6. Modeling state

### Dense reward model (Llama-8B + LoRA)

From `runs/peer_review_sweep_llama8b/subset_<frac>/trial_<N>/training_history.json`,
with 5 trials per subset (`project_dense_model_sweeps.md`):

| Subset | Rows | med AUC | max AUC |
|--------|-----:|--------:|--------:|
| 0.1 | ~5.6K | 0.745 | 0.758 |
| 0.2 | ~11K  | 0.755 | 0.766 |
| 0.3 | ~17K  | 0.685 | 0.768 |
| 0.4 | ~22K  | 0.747 | 0.775 |
| 0.5 | ~28K  | 0.768 | 0.776 |
| 0.6 | ~34K  | 0.775 | 0.781 |
| 0.7 | ~39K  | 0.768 | 0.780 |
| 0.8 | ~45K  | 0.776 | 0.783 |
| 0.9 | ~51K  | 0.694 | 0.760 |
| 1.0 | ~56K  | 0.770 | 0.780 |

**Saturated at subset 0.5–0.6 (~28–34K rows) with plateau AUC ~0.77–0.78.**
Strongest dense-model signal of the four primary tasks. Adding more data
won't help; gains must come from label quality or text quality.

### Verification library (per-aspect Python predict-programs)

`runs/validity_full/v2/peer_review/codegen_claude/` holds **571 / ~654 .py
files** of the form `a{ID}_v{0,1,2}_{keyword,structure,holistic}.py`
(`reference_codegen_per_aspect_programs.md`). Each file exposes a pure
`score(text: str) -> float` that returns one of `{0.0, 0.5, 1.0}`. Three
flavors per aspect (`v0_keyword`, `v1_structure`, `v2_holistic`) let
downstream code pick the best version or ensemble them.

This is currently the **only task with codegen_claude programs populated**
end-to-end (all other tasks have either zero files or different generation
recipes). Used as the Tier-3 deterministic layer in the V+A+T plan and as
the basis for the Direction-2 hierarchy work.

Also present:

- `runs/validity_full/v2/peer_review/aspects.json`,
  `aspect_to_bundle.json` — the aspect catalogue.
- `claude_judge_batch_*.json` — Claude-as-judge cell outputs.

### Direction-2 verification pipeline (experimental)

Per `project_verification_pipeline_recipe.md`, on 20K peer-review examples:

- Llama-3.3-70B-FP8 STaR extraction → **133K features, 94% parse rate**.
- Dedup with all-MiniLM-L6-v2 + k-means → **12,485 canonical features**.
- Qwen3-Coder-Next-FP8 codes all features → 42% success single-pass,
  ~60–70% after 3 passes (`feature_programs_merged.jsonl`).
- Llama self-assessment: 28% thin / 72% thick.
- Hierarchy building, evaluation, and thin/thick annotation are designed
  but **not yet implemented**.

Outputs on sk3 under
`/lfs/skampere3/0/alexspan/norm-research/outputs/verification_library/direction2_20k/`.

### Local explanations (STaR-Local) clustering

Per `project_local_explanations_clustering_findings.md`:

- 429,486 unique features extracted from the 56,152 training abstracts by
  Llama-3.3-70B-FP8 with LoRA-merged BGE-large embeddings.
- **Locked operating point**: UMAP (`n_neighbors=30, n_components=20,
  min_dist=0.0, cosine`) + HDBSCAN (`min_cluster_size=100,
  cluster_selection_method="eom"`), `target_weight=0.0`,
  `cluster_selection_epsilon=0.0`, plus the LLM dedup judge.
- Supervised UMAP fragments same-concept-different-label and was dropped.
- Per-feature P(y=1|feature) is near-unanimous (99.7% pure-pos or pure-neg);
  pre-feature label signal carries the geometry.

### Online rubrics

`online-rubrics/raw/` holds **~2,770 scraped venue review guidelines**
(AAAI, ACL/ACL Rolling Review, ICML, NeurIPS, AAUP, journal policies)
crawled across waves `waveh3–waveh7` (see `waveh*_log.csv`).
`claude-parsed/` (~167 JSONs) and `gpt-parsed/` (1 JSON so far) contain
extracted rubric metrics. These feed
`methods/metrics_tree_infilling/` as `extracted.rubrics_metrics` — ~75K raw
entries for peer-review; caller caps to M.

### Norm extraction (in flight)

Qwen-122B batch-mode extraction on sk3 (switched from OpenAI-server runs
2026-06-04 per `feedback_never_openai_server_for_bulk.md`). Output stream:
`/lfs/.../data/peer_review/norm_extracted/extracted_qwen.jsonl.gz`. ~48K
rows preserved across resume; first batch on GPU 1, `batch_size=1000`.

### Local-explanation AUC (Apr 17)

`project_nc_pipeline_state.md`: baseline test AUC 0.5892 → 0.6141 sweep-best
(Trial 11) on peer-review. Optuna scaling-law sweep designed and queued
(`project_local_explanations_hyperparam_sweep.md`); Step 1+2 extraction is
cached so Step 3+ trials are cheap.

## 7. Key decisions

- **ICLR is the load-bearing source** because it publishes every
  submission, including rejects. NeurIPS / ICML only expose ~5% of rejects
  (opt-in), so they skew positive on their own.
- **eLife and TMLR have no binary decision** — eLife uses assessment text
  and `accepted` is NULL; TMLR is a rolling Yes/No / no-score field. Both
  are kept in the corpus but excluded from binary-AUC modeling cells.
- **PDF + revision tracking**: `pdf_versions` stores both the original
  submission (`version=0`) and revisions (`version=1+`). 22,625 papers have
  both, enabling pre-/post-revision text comparisons.
- **Score normalization** is venue-specific (1–10 for ICLR/NeurIPS/ICML/COLM/TMLR;
  1–5 for ACL/CoNLL; Yes/No for some TMLR) and normalized to `[0, 1]` in
  `unify_datasets.py`.
- **Group split is mandatory** — multiple reviews per paper means
  per-review row splits leak the label. The canonical split needs to be
  rebuilt with `paper_id` as the group key.
- **Excluded datasets**: orb (no review text), berenslab (overlaps
  OpenReview ICLR), Seafoodair (analysis artifacts, not raw text).

## 8. Open questions and next steps

1. **Rebuild canonical splits with `paper_id` group key.** Currently
   blocking trustworthy production runs (`notes/dataset_leakage_audit.md`,
   `reference_clean_datasets_per_task.md`).
2. **Finish the Direction-2 verification pipeline** — hierarchy building,
   per-program evaluation on a test set, empirical thin/thick annotation.
3. **Run the Optuna scaling-law sweep for local explanations** (peer-review
   operating point is locked; sweep is staged but GPU-blocked).
4. **Decide whether eLife / TMLR contribute to V+A+T** — neither has a
   binary label, but assessment / rolling-Yes-No text might serve as a
   regression target.
5. **Reconcile dense-model plateau (~0.78 AUC) with verification-library
   ceiling.** The dense AUC is the empirical articulability ceiling for
   strong models; the gap to 1.0 is the taste residual
   (`project_tacitness_two_layers.md`).
6. **Online-rubrics parsing** — only 167 of ~2,770 raw rubric documents
   are parsed; expand to feed more `rubrics_metrics` into
   `metrics_tree_infilling`.
