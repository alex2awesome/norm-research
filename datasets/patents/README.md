# Patents

Patent application outcome prediction and §102 anticipation retrieval. The dataset is built from the USPTO Open Data Portal, PatentsView bulk products, Google Patents scraping, and the USPTO Office Action Research Dataset (OARD). It supports several prediction targets in parallel, with the active push (June 2026) being a §102 novelty-destruction retrieval pipeline that indexes full-spec passages, not just claim 1.

Local working dir: `datasets/patents/`. Canonical storage and all heavy artifacts live on sk3 at `/lfs/skampere3/0/alexspan/norm-research/datasets/patents/`.

> **2026-07-10:** evidence-aware judging flips the prior-art-op null — doc-only LLM judges produce a structural false null on retrieval-dependent criteria; op-marginals go +.21/+.61/+.66 once the judge sees the search record. Details + reuse pointers: [`2026-07-10__evidence_aware_judge_ws3.md`](2026-07-10__evidence_aware_judge_ws3.md).

---

## 1. Tasks

| ID | Label | Prediction time | Allowed features | Forbidden features |
|---|---|---|---|---|
| Task A — first-draft approval | `first_draft_approved` (granted with zero office actions) | t = filing | draft text, applicant IDS cites | examiner cites, OA text (leaky — examiner cites only exist if there was an OA, which by definition means not first-draft approved) |
| Task B — final approval | granted vs abandoned | any time pre-outcome | draft text, applicant cites, examiner cites | (absence-of-examiner-cites is weakly leaky — use `--require-oa` for strict) |
| Task C — §102/§103 novelty destruction | per-claim retrieval target: identify cited prior art that anticipates | post-filing | examined claim text + prior-art corpus | — |
| Task D — patent quality / examiner cites | downstream signals such as forward-citation count | post-grant | full patent | — |

See `project_patents_first_draft_prediction.md` for Tasks A/B; `notes/2026-06-04__patent_supervised_pairs_methodology.md` for Task C.

---

## 2. Sources

| Source | What | Used for |
|---|---|---|
| **USPTO Open Data Portal (ODP)** — PVGPATTXT / PVPGPUBTXT | Long Text bulk: granted (1976-2025) and pre-grant pub (2001-2025) detailed-description + brief-summary + claims, year-split TSV zips | Task C spec-passage corpus (~228 GB) |
| **PatentsView bulk** (legacy S3) | `pg_published_application`, `pg_cpc_current`, `g_cpc_current`, `pg_claims`, `pg_brf_sum_text_*`, `pg_detail_desc_text_*`, granted equivalents | CPC join, draft text, fallback bulk |
| **Google Patents scrape** | Claim 1 (legacy) + full text (current) for pre-2001 grants, design and plant patents not in PV | Filling corpus gaps; ~1.7M target pool |
| **OARD** (`patents-public-data.uspto_oce_office_actions` on BigQuery) | `office_actions`, `rejections`, `citations` — structured per-OA records | Rejection flags, §102/§103 ground truth |
| **PatEx** (USPTO bulk application + transaction data) | `application_data.csv`, transaction events | Label derivation (CTNF/CTFR/abandonment), pgpub_id ↔ application_number crosswalk |
| **Google Patents Public Data** (BigQuery) | `patents-public-data.patents.publications` | Fallback for pgpub ↔ appnum crosswalk |

Important: **OARD does not contain free-form examiner reasoning text.** It is the Lu/Myers/Beliveau (2017) regex extraction of structured fields from office action prose — the original text was discarded. To get the articulated rationale we would need the USPTO Office Action Weekly Zips (OACT, 2020-01-06+). See `oard_deployment_plan.md` for the full investigation.

---

## 3. Collection scripts

All scripts are in the repo-level `scripts/` directory unless noted. The first six (`01_*` … `06_*`) live under `datasets/patents/scripts/` and are the original 2026-04 dataset build pipeline; the rest are the 2026-05/06 expansions.

### Original pipeline (`datasets/patents/scripts/`)
1. `01_download_patex.sh` — pull PatEx (event log + metadata)
2. `02_download_patentsview.sh` — pull PV pre-grant and grant tables
3. `03_parse_labels.py` — derive labels from PatEx events
4. `04_build_dataset.py` — join labels + draft + final text → baseline CSVs
5. `05_build_citation_lookup.py` — patent_id → claim 1 lookup
6. `06_build_augmented_datasets.py` — add cited claim text to draft
7. `07_balance_datasets.py` — basic balance
8. `08_download_oa_citations.sh` — pull OARD bulk (DEAD URLs, superseded by BigQuery)
9. `09_query_oard_bigquery.py` — pull OARD citations from BigQuery
10. `10_get_pgpub_app_mapping.py` — pgpub_id → application_number map

### Local builders (`datasets/patents/`)
- `build_first_draft_cpc_balanced.py` — per-(CPC × length × label) balancing, filing-year ≤ 2021 to avoid prosecution survivorship bias, canceled-claim regex filter
- `build_first_draft_with_rejections.py` — same CPC-balanced setup + OARD rejection flags joined in
- `build_final_outcome_with_rejections.py` — Task B variant; requires ≥1 examiner citation, uses pg_claims (symmetric across grant/abandon)

### OARD expansion (repo `scripts/`)
- `download_oard_rejection_flags.py` — BigQuery aggregation to per-app flags (101/102/103/112a/b/d/f/DP)
- `query_oard_action_subtypes.py`, `query_oard_rejection_counts.py` — diagnostic queries

### §102 retrieval pipeline (repo `scripts/`) — currently active
- `download_patentsview_long_text.py` — pulls PVGPATTXT / PVPGPUBTXT detailed-description and brief-summary, year-split (~228 GB total)
- `download_patentsview_specs.sh` — wget fallback for PV S3 spec bulks
- `paragraph_chunk_specs.py` — splits spec docs into paragraph chunks, streams to parquet (`processed/spec_chunks/<source>_<year>.parquet`)
- `embed_spec_chunks.py` — BGE-M3-anticipation-v2 embeddings, fp16 npy + line-aligned meta.jsonl
- `build_spec_faiss_index.py` — IVF_FLAT FAISS index over spec chunk embeddings
- `silver_label_spec_chunks.py` — LLM YES/NO/partial labeling of (claim, spec_chunk) pairs for fine-tuning
- `extract_clean_102_pairs.py` — filters `oard_citations.csv` to `action_type='102'`, joins to text; **the correct §102 ground truth**
- `validate_retriever_on_oard_pairs.py` — retrieval eval on the clean pairs
- `build_retriv_claim_index.py` — claim-1-only retriv index (legacy; superseded by spec index)
- `fetch_full_text_from_google_patents.py`, `fetch_missing_from_google_patents.py` — corpus-gap scraping
- `autorun_spec_pipeline.sh` — **the active orchestrator (2026-06-05)**: waits for the PV download to exit, sanity-checks year files, then chains chunk → embed → index → extract pairs

### Misc / analysis
- `analyze_patents_year_drift.py`, `analyze_patents_leakage.py`, `analyze_patents_citation_count.py`
- `inspect_*`, `manual_check_102_pairs.py`, `investigate_claims_canceled.py`
- `extract_pgpub_corpus.py`, `extract_granted_patent_corpus.py`
- `v2_cluster_aspects_to_bundles.py`, `v2_per_aspect_trust_table.py`, `r2_aspect_year_mentions.py`, `r25_aspect_merge_*.py`, `migrate_patent_metrics.py`, `export_spec_leaves.py`, `draft_patent_exemplars.py` — V/A/T track integration

---

## 4. File layout

### Local (`datasets/patents/`)
```
build_first_draft_cpc_balanced.py
build_first_draft_with_rejections.py
build_final_outcome_with_rejections.py
oard_deployment_plan.md
patents_final_outcome_balanced.csv.gz       (523 MB local mirror)
online-rubrics/                              expert online rubrics scrape
  raw/                                       4,259 fetched rubric pages
  claude-parsed/, gpt-parsed/                LLM-extracted rubric JSON
  urls-visited.csv, waveh*_log.csv
replications/                                vendored prior-art replications
  claim-compare/                             ClaimCompare (SIGIR/PatentSemTech 2024)
  FLAN-Graph/                                FLAN Graph (Gao 2024)
  pedantic-patentsemtech/                    PEDANTIC §112(b) definiteness
scripts/                                     01_* … 10_* original pipeline
```

### sk3 (`/lfs/skampere3/0/alexspan/norm-research/datasets/patents/`)
```
patents_dataset.jsonl.gz                     6.3 GB, 4.69M rows — rich JSONL
patents_first_draft.csv.gz                   4.5 GB, 4.69M — Task A baseline
patents_final_outcome.csv.gz                 3.8 GB, 3.99M — Task B baseline
patents_first_draft_with_applicant_cites.csv.gz
patents_final_outcome_with_examiner_cites.csv.gz
patents_first_draft_cpc_balanced.csv.gz
patents_first_draft_cpc_balanced_with_rejections.csv.gz
patents_final_outcome_cpc_balanced_with_rejections.csv.gz
processed/
  labels.parquet                             10.57M per-app labels
  claim1_lookup.parquet                      6.92M legacy claim-1 lookup
  granted_patents_claim1.parquet             6.20M granted (2001-2025)
  pgpub_claims1.parquet                      8.30M pgpub claim-1
  google_patents_supplement.parquet          ~1.4M pre-2001 / design
  google_patents_full_text.parquet           full-text scrape (active)
  pgpub_to_appnum.parquet                    crosswalk
  spec_chunks/<source>_<year>.parquet        paragraph chunks (active build)
  spec_embeddings/embeddings.fp16.npy        BGE-M3 vectors
  spec_embeddings/meta.jsonl
  missing_after_local_sources.csv
  silver_labels.jsonl.gz                     LLM-labeled (claim, chunk) pairs
raw/
  patex/application_data.csv                 PatEx bulk
  patentsview_grant/                         g_*.tsv.zip
  patentsview_pg/                            pg_*.tsv.zip
  oard/oard_citations.csv                    ~60M per-OA-per-cite-per-rejection
  oard/oard_rejections_by_app.csv            2.19M per-app rejection flags
indexes/spec_chunks_v1/                      FAISS IVF_FLAT (separate path on sk3)
```

---

## 5. Canonical dataset files (modeling pool)

| Task | File | Rows | Label column |
|---|---|---:|---|
| A (first-draft) | `patents_first_draft_cpc_balanced.csv.gz` | balanced subset | `judgement` |
| A + rejections | `patents_first_draft_cpc_balanced_with_rejections.csv.gz` | balanced subset | `judgement` + `rejected_10[1-3]`, `rejected_112{a,b,d,f}`, `rejected_dp` |
| B (final outcome) | `patents_final_outcome_cpc_balanced_with_rejections.csv.gz` | filtered to ≥1 examiner cite | `judgement` |
| C (§102 pairs) | `processed/clean_102_pairs.jsonl.gz` | (in progress) | implicit positive: `(anchor, positive_pgpub_id)` from `action_type='102'` rows |
| C silver labels | `processed/silver_labels.jsonl.gz` | (in progress) | `llm_label: yes/no/partial` |

Per-aspect V/A/T plumbing lives under `runs/validity_full/v2/patents/codegen_claude/` on sk3 with extended `_patent_utils.py` (662 lines, 26 helpers).

---

## 6. Modeling state

### Task A (first-draft) — Llama-8B dense sweep
`runs/patents_first_draft_baseline_sweep_llama8b/subset_0p{1,2,3,4}/validation_metrics.csv`. At subset 0.4 the validation AUC progresses from 0.71 → 0.74 across the logged epochs (see `validation_metrics.csv`). Variant with applicant cites: `runs/patents_first_draft_with_applicant_cites_sweep_llama8b/subset_0p{1,2}`.

### Task B (final outcome) — Llama-8B dense sweep
`runs/patents_final_outcome_baseline_sweep_llama8b/subset_0p{1,2,3}` and `…_with_examiner_cites_sweep_llama8b/…` parallel directories.

### Code-AUC (V/A/T) on patents
**Headline (2026-06-02):** V3-codegen ensemble + V0 LR reached **CV AUC 0.6040**, breaking the prior 0.563 ceiling for the first time. 23 v3 codegen files written; novelty caches built (`outputs/v2_analysis/patents_claim1_{bigrams,emb,tfidf}.*`). Per-aspect summary at `outputs/v2_analysis/patents_rebuild_v3_summary.parquet`. Source: `project_patents_v3_rebuild_result_2026_06_02.md`.

### Task C (§102 retrieval) — in progress
The v2 retriever validation (`notes/2026-06-04__patent_supervised_pairs_methodology.md`) confirmed v2 was unusable: Recall@1 = 2.4 %, Recall@10 = 10.9 %, MRR 0.052, with 79 % of "positives" not even present in the FAISS index (it covered 4.7M docs while training pairs spanned ~20M). v3 pipeline is the active push (see § 7).

### Cross-task judge bug (must be re-scored)
The v2 judge system prompt was a peer-review template applied to every task. Patents judge marked rejected 0.914 / approved 0.922, AUC 0.507 (random). **Any V/A/T number predating the patent-specific judge prompt must be discarded.** Source: `feedback_judge_prompt_cross_task_bug.md`.

### HUPD baselines to beat
Suzgun et al. 2022 published best ~64 % accuracy on a 50/50 balanced subset; Mistral-7B QLoRA (2024 follow-up) at 64.4 % did not beat DistilBERT. Sarkar 2023 AUC 0.57 on the unbalanced eval. Neither used citation features.

---

## 7. Current active pipeline — §102 anticipation retrieval (2026-06-05)

Goal: replace claim-1-only retrieval with full-spec passage retrieval, using **clean** §102 supervision (per-row `action_type='102'` in OARD, not per-app `rejected_102` flag).

Sequence (orchestrated by `scripts/autorun_spec_pipeline.sh`):
1. `download_patentsview_long_text.py` — pulls `g_detail_desc_text_<year>.tsv.zip` (1976-2025) and `pg_detail_desc_text_<year>.tsv.zip` (2001-2025), ~228 GB
2. `paragraph_chunk_specs.py` — paragraph splits → `processed/spec_chunks/<source>_<year>.parquet`
3. `embed_spec_chunks.py` — BGE-M3-anticipation-v2 fp16 vectors
4. `build_spec_faiss_index.py` — IVF_FLAT, `nlist ≈ sqrt(N)` (~14k at 200M chunks)
5. `extract_clean_102_pairs.py` — pull rows with `action_type='102'` from `oard_citations.csv`, join to anchor+positive text
6. `silver_label_spec_chunks.py` — Qwen3.5 / Claude YES/NO/partial on `(claim, chunk)` for hard-positive and hard-negative pairs
7. `validate_retriever_on_oard_pairs.py` — honest recall/MRR

**Status 2026-06-05:** 60 spec_chunks shards completed; ParquetWriter fix landed (it was holding writers open across years, causing OOM); `autorun_spec_pipeline.sh` resumed in background. Running log: `logs/patents_codegen_exec.log` and the autorun-emitted timestamps.

**Decision (2026-06-05):** drop training pairs whose cited prior art text cannot be found in our corpus, instead of padding with random or weak negatives. Once PVGPATTXT/PVPGPUBTXT (1976-2025) are fully indexed, coverage is projected at ~95 %+. Residual unfindable pool is mostly design patents (~7 %) and foreign references (~0.2 %).

---

## 8. Replications (`replications/`)

Vendored implementations of prior-art systems for direct comparison / dataset reuse.

| Subdir | Paper | Use to us |
|---|---|---|
| `claim-compare/` | Parikh & Dori-Hacohen, **ClaimCompare: A Data Pipeline for Evaluation of Novelty Destroying Patent Pairs** (SIGIR PatentSemTech 2024). 1,045 base × 25 related labeled electrochemical pairs | Independent §102 evaluation set; pipeline blueprint for novelty-destruction labels |
| `FLAN-Graph/` | Gao et al. 2024, **Beyond Scaling: Predicting Patent Approval with Domain-specific Fine-grained Claim Dependency Graph** (arXiv 2404.14372). PatentAP dataset on HF | Approval-prediction baselines + claim-dependency graph features |
| `pedantic-patentsemtech/` | **PEDANTIC: A Dataset for the Automatic Examination of Definiteness in Patent Claims** (arXiv 2505.21342). 14k NLP-domain claims annotated with §112(b) indefiniteness reasons | §112 articulation track — annotated examiner reasoning (rare); LR baseline + LLM-as-judge eval |

---

## 9. Key decisions

- **Leakage rules.** Applicant IDS cites are non-leaky for any task. Examiner cites are leaky for Task A (they exist only if an OA happened). For Task B they are non-leaky, but absence-of-cites is weakly leaky → `--require-oa` for the strict variant.
- **Year drift.** Filter `date_published ≤ 2021-12-31` to avoid prosecution-survivorship bias — recent filings (2022-24) had 32-44 % positive rate vs the 50 % baseline because young apps haven't had time to be abandoned yet.
- **CPC balancing.** Per `(cpc_section × length_bucket × label)` cell cap, sampled to `2 × min(pos, neg)`. Single-CPC join uses lowest `cpc_sequence` of `cpc_type=inventional` rows.
- **Canceled-claim filter.** Regex in `build_first_draft_cpc_balanced.py` matches `1 . (canceled)`, `1-18. (Canceled)`, `1. (Cancelled)` — must require an explicit claim number/range before the parenthesized "canceled" to avoid false matches on the word in claim bodies.
- **§102 vs §103 distinction.** OARD `oard_citations.csv` is per-OA-per-cite-per-rejection. `action_type` field gives the split: 4 % `'102'`, 17 % `'103'`, 79 % empty (IDS/Form 892/1449, not tied to a rejection). Clean §102 supervision = filter to `action_type=='102'`; the per-app `rejected_102=True` flag conflates §102, §103, and IDS noise — do **not** use it for pair extraction.
- **Drop unfindable prior-art pairs.** If we cannot locate the cited reference's text (claim or spec) in our indexed corpus, drop the training pair rather than padding it. Document the drop count in the run notes.
- **PDF vs MS_WORD for office actions.** USPTO ODP returns scanned-image PDFs for older OAs (pypdf returns ≤30 chars from a 29-page doc). MS_WORD `.docm` is the real text source, but the download URL returns a plain-text message with a 1-hour-valid redirect to `data-documents.uspto.gov` — extract with regex, then follow.
- **Patent-number normalization.** OARD stores citations as bare digits (`20060242362`, `6807434`); examiner OA text uses formatted variants (`US 2006/0242362 A1`, `US 6,807,434 B2`, `Smith et al. (US 6,807,434)`). Generate format variants from the OARD ID OR extract patent-number-shaped tokens from OA text and normalize them.
- **Judge prompt.** Cross-task peer-review template made any v2 patents code-AUC random. The task-specific judge must be wired into `runs/validity_full/v2/patents/judge_system.txt` and any older V/A/T number must be re-run.

---

## 10. Open questions / next steps

- Finish the spec-chunk index (`spec_chunks_v1`) and rerun `validate_retriever_on_oard_pairs.py` for an honest Recall@k baseline.
- Build a limitation-level query decomposer so a §102 hit requires all examined-claim limitations to match within a single prior-art doc.
- OA-text ground-truth validation: download MS_WORD OAs via redirect, extract the §102 paragraph, verify retriever recovers the actual cited reference. Use this as the honest eval metric (not the silver labels).
- Per-aspect code generation for patents: extend the V/A/T programmatic library beyond peer_review (where 571 files exist) to patents, using the V3 ensemble work as the seed.
- §112 indefiniteness as a separate articulability target — PEDANTIC gives 14k annotated NLP-domain claims; treat as a sister task to §102.
- Decide whether to surface the OACT 2020+ weekly zips as a future text source for actual examiner reasoning (OARD does not provide this).
- Re-score all v2 patents cells with the patent-specific judge prompt; recompute V/A/T tables.
