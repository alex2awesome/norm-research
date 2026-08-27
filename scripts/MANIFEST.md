# `scripts/` Manifest

Inventory of the ~339 files in `scripts/` so the user (and future Claude) can tell
**which scripts are load-bearing right now**, which have been replaced, which were
one-off probes, and which are pure tooling. **Nothing here has been deleted** —
this is a roadmap, not a cleanup.

## How to use this file

- Hunting for "the canonical script for X"? Find X's pipeline section and look
  for **Active** entries.
- Touching an old script? If it's marked **Superseded**, follow the pointer to
  its replacement before editing.
- Investigating a one-off analysis result? **Exploratory** scripts are usually
  paired with a note under `notes/YYYY-MM-DD__*.md` — search by date.
- Need GPU launchers / sync utilities? See **Tooling** at the bottom.

## Legend

- **Active** — current canonical script for an ongoing pipeline. Touch with care.
- **Superseded** — a newer version exists; pointer to replacement given.
- **Exploratory** — a dated one-off probe / audit / spot-check. Findings already
  consumed elsewhere (usually a `notes/` file or memory entry). Keep for the
  audit trail, don't rebuild on top of it.
- **Tooling** — utility, launcher, helper, or shared module imported by others.

Approx. dates after each entry are file mtimes.

---

## 1. Verification Library / V+A+Taste validity ladder (v2 pipeline)

The current canonical "code + judge across paraphrases" pipeline lives under the
`v2_*` and `validity_full_*` prefixes. Per-task results are at
`runs/validity_full/v2/<task>/`.

### v2 pipeline (current — May–June 2026)

- `v2_task_registry.py` — **Active** Single source of truth for per-task paths/configs. Imported by every other `v2_*` script. (May 25)
- `v2_setup_task.py` — **Active** End-to-end "add a new task to v2" bootstrap (aspects, sample dps, bundle clusters, exemplar pool). (May 24)
- `v2_sample_datapoints.py` — **Active** Stratified DP sampling per task. (May 24)
- `v2_build_exemplar_pool.py` — **Active** Top/bottom-score exemplar pool per aspect. (May 24)
- `v2_cluster_aspects_to_bundles.py` — **Active** k-means clustering of R2 aspects into ~30 judge-bundles. (May 24)
- `v2_paraphrase_enrichments.py` — **Active** Per-aspect paraphrased enrichment generation. (May 24)
- `v2_assemble_judge_prompt.py` — **Active** Build multi-rubric judge prompt files (bundle × paraphrase × chunk). (May 24)
- `v2_build_judge_jobs.py` — **Active** Enumerate (bundle × paraphrase × chunk) jobs + manifest. (May 26)
- `v2_smoke_judge_openrouter.py` — **Tooling** Local smoke before kicking off sk3. (May 24)
- `v2_exec_codes.py` — **Active** In-process exec of all aspect code variants on 5K dps. (May 24)
- `v2_qwen_audit.py` — **Active** Health check on Qwen-judge response parsing. (May 27)
- `v2_claude_analyses.py` — **Active** Aspect-aspect clustering + discriminativeness on Claude p0. (May 27)
- `v2_build_feature_matrices.py` — **Active** Per-task feature matrices from judge responses. (May 27)
- `v2_per_aspect_trust_table.py` — **Active** ρ(llama, claude) + ρ(code, claude) per aspect. (May 27)
- `v2_analyze.py` — **Active** Per-aspect stats + lasso for code-only / judge-only / combined / stacked. (May 24)
- `v2_build_r2_post.py` — **Active** Build R2_post (reconciled cluster verdicts + collapse map). (May 27)
- `v2_build_r2_post_prompts.py` — **Active** Build R2_post bundles + enrichments + prompts. (May 27)
- `v2_build_20x1_r2post.py` — **Active** 1-dp × 20-aspect prompts on R2_post aspects (Qwen). (May 27)
- `v2_build_20x1_test.py` — **Active** Per-task 1-dp × 20-aspect Qwen prompts. (May 27)
- `v2_build_targeted_prompts.py` — **Active** Targeted Claude prompts to maximize per-dp coverage. (May 26)
- `build_mopup_lists.py` — **Active** Greedy mop-up prompt ordering. (May 27)
- `build_cells_db.py` — **Active** Build the unified `cells_v1` judge-cells DB. Single source of truth for counts/coverage (per memory). (May 29)
- `sync_cells_db.sh` — **Tooling** Bidirectional cells DB sync between laptop and sk3. (May 27)

### v2 relaxed-applicability (creative writing pivot — June 1)

- `cw_norm_categorization.py` — **Active** Manual RELAX/STRICT/BORDERLINE categorization of 368 CW aspects. (Jun 1)
- `build_relaxed_appl_prompts.py` — **Active** Build relaxed-applicability re-scoring prompts. (Jun 1)
- `build_relaxed_v2_prompts.py` — **Active** Task-specific relaxed prompts for 260 RELAX+BORDERLINE CW aspects. (Jun 1)
- `build_v2relax_cells.py` — **Active** Ingest v2relax responses into cells_v1 as `qwen_relaxed_v2_2026_06_01`. (Jun 1)
- `v2relax_intermediate_checks.py` — **Exploratory** Mid-run sanity checks. (Jun 1)
- `v2relax_headline_compare.py` — **Exploratory** Legacy qwen vs relaxed_v2 headline comparison. (Jun 1)
- `audience_norm_scan.py` — **Exploratory** Keyword scan across tasks for audience-appeal aspects. (Jun 1)

### validity_full (R1+R2 paraphrase pipeline — predecessor of v2)

- `validity_full_data_prep.py` — **Active** R2 aspects + top-K R1 sub-families + stratified peer-review dps. (May 24)
- `validity_full_codegen_prep.py` — **Active** Per-(R1 metric × 5 paraphrases) code-gen prompts. (May 24)
- `validity_full_judge_prep.py` — **Active** R2-level judge prompts. (May 24)
- `validity_full_judge_r1_prep.py` — **Active** R1-level judge prompts (parallel to R2). (May 24)
- `validity_full_paraphrase_prep.py` — **Active** Paraphrase generation prompts. (May 24)
- `validity_full_qwen_codegen.py` — **Superseded by** `validity_full_qwen_codegen_all.py`. (May 24)
- `validity_full_qwen_codegen_all.py` — **Active** Qwen code-gen on ALL R1 prompts. (May 24)
- `validity_full_exec_codes.py` — **Active** Run generated codes on dps. (May 24)
- `validity_full_exec_inproc.py` — **Active** In-process exec variant. (May 24)
- `validity_full_analyze.py` — **Active** Aggregate + lasso analyses. (May 24)
- `validity_full_code_only_report.py` — **Exploratory** Code-only mini-report (May 24)
- `validity_full_compare_levels.py` — **Active** R1↔R1 vs R2↔R2 comparison across models/methods. (May 24)
- `sk3_validity_full_runner.py` — **Tooling** sk3 vLLM runner for validity_full prompt dirs. (May 24)
- `sk3_validity_full_runner_streamed.py` — **Tooling** Same with incremental flush. (May 24)

### validity_pilot (predecessor — superseded by validity_full)

- `validity_pilot_data_prep.py` — **Superseded by** `validity_full_data_prep.py`. (May 24)
- `validity_pilot_codegen_prep.py` — **Superseded** (May 24)
- `validity_pilot_judge_prep.py` — **Superseded** (May 24)
- `validity_pilot_code_exec.py` — **Superseded** (May 24)
- `validity_pilot_code_exec_llama.py` — **Superseded** (May 24)
- `validity_pilot_analyze.py` — **Superseded** (May 24)
- `validity_pilot_analyze_multimodel.py` — **Superseded** (May 24)
- `validity_pilot_analyze_judge_multimodel.py` — **Superseded** (May 24)
- `validity_pilot_r1_within_r2_prep.py` — **Superseded** (May 24)
- `validity_pilot_r1_within_r2_analyze.py` — **Superseded** (May 24)
- `sk3_validity_pilot_llama.py` — **Superseded** sk3 runner for pilot prompts. (May 24)

### Validity analyses (Lasso, ranking, recoverability)

- `validity_lasso.py` — **Superseded by** `validity_lasso_combined_v2.py`. (May 24)
- `validity_lasso_combined_v2.py` — **Active** Wider Cs grid combined L1. (May 24)
- `validity_within_method_ranking.py` — **Active** Paraphrase σ + label Pearson rankings. (May 24)
- `validity_recoverability_obfuscate.py` — **Active** Strip docstrings/comments/identifiers for recoverability test. (May 24)
- `validity_recoverability_run.py` — **Active** Multiple-choice "which aspect generated this code" probe. (May 24)
- `build_recoverability_prompts.py` — **Active** Build recoverability smoke-test prompts. (May 26)
- `score_recoverability_guesses.py` — **Active** LLM judge for recoverability guesses. (May 26)

### Dataset-benchmark judge sub-pipeline

- `build_dataset_judge_input.py` — **Active** JSONL inputs for dataset/benchmark judge. (May 24)
- `dataset_benchmark_judge_prompt.py` — **Active** "Is this paper's primary contribution a dataset/benchmark?" prompt. (May 24)
- `sk3_dataset_benchmark_judge.py` — **Active** sk3 vLLM Llama runner. (May 24)
- `aggregate_dataset_judge.py` — **Active** Aggregate by venue/year/label. (May 24)
- `plot_dataset_judge.py` — **Active** Plot outputs. (May 24)

---

## 2. Cross-task norm analyses (June 1 cells_v1 sweeps)

Once cells_v1 stabilized, several thin aggregate analyses ran across all tasks.

- `lift_all_tasks.py` — **Active** Per-task aspect-label lift across all judges. (Jun 1)
- `norm_logreg_all_tasks.py` — **Active** L2 LogReg on aspect score+applicable features per task/judge. (Jun 1)
- `norm_rf_all_tasks.py` — **Active** RF on union of qwen+claude judge cells. (Jun 1)
- `norm_error_deepdive.py` — **Exploratory** Worst confident errors + near-boundary for CW RF. (Jun 1)
- `norm_error_multitask.py` — **Exploratory** Same across tasks. (Jun 1)

---

## 3. Code review pipeline (`cr_*`, `crse_*`, `dense_4096tok_*`)

CR has both a v2 (aspect/judge) track and a dense-source track. Verifiability
ladder (Tiers 1–4 + codegen) is the current focus.

### Tier ladder + codegen (current — June 1–5)

- `cr_tier1_baseline.py` — **Active** Tier 1: 5 metadata features. (Jun 1)
- `cr_tier2_diff_features.py` — **Active** Tier 2: diff-parsing features. (Jun 1)
- `cr_tier34_features.py` — **Active** Tier 3 (lizard) + Tier 4 (test-signal text). (Jun 1)
- `cr_final_ladder.py` — **Active** Combines all tiers + codegen, trains RF + LR. (Jun 1)
- `cr_run_codegen.py` — **Active** Run 1182 per-aspect Python predict-programs on v2 dps. (Jun 1)
- `cr_run_codegen_on_diff.py` — **Active** Re-run codegen on dense_4096tok DIFF text. (Jun 1)
- `cr_codegen_diff_preview.py` — **Exploratory** Sample preview before full run. (Jun 2)
- `cr_codegen_diagnostic.py` — **Exploratory** "Why does codegen hurt RF AUC?" diagnostic. (Jun 1)
- `cr_mi_feature_select.py` — **Active** Univariate MI filter + L1 LR over MI matrix. (Jun 2)
- `cr_run_metric_implementer.py` — **Active** Score metric_implementer programs on 4978 v2 dps (with DENSE diff). (Jun 5)
- `code_review_dive.py` — **Exploratory** Survey of CR aspects + artifacts + cells coverage. (Jun 1)

### CR dataset cleaning + audits (June 2)

- `cr_dataset_cleaning_v2.py` — **Active** Deep cleaning audit for code_review_dense_4096tok. (Jun 2)
- `cr_per_project_merge_rates.py` — **Active** Project-level merge-rate analysis. (Jun 2)
- `cr_audit_dense_for_leakage.py` — **Exploratory** Spurious-feature audit. (Jun 2)

### dense_4096tok single-file reconstruction (June 5)

- `dense_4096tok_reconstruct_single_file.py` — **Active** Reconstruct single-file new-side view per PR. (Jun 5)
- `dense_4096tok_single_file_bank_score.py` — **Active** Score metric_implementer bank on reconstructions. (Jun 5)
- `dense_4096tok_single_file_mi_ladder.py` — **Active** 4-cell MI ladder on reconstructed view. (Jun 5)

### Code Review Stack Exchange (CR.SE) quality labels

- `crse_quality_label_shard.py` — **Active** One shard of CR.SE answer quality labels via Claude. (Jun 3)
- `crse_quality_aggregate.py` — **Active** Aggregate Claude labels + correlations. (Jun 3)

### Linking comment text → modeling datasets

- `link_comments_to_code_review.py` — **Active** Link PR review comments to CR dataset. (May 9)
- `link_reviews_to_peer_review_dataset.py` — **Active** Link review text to peer-review dataset. (May 9)

---

## 4. Patents pipeline (rejection prediction + §102 anticipation retrieval)

Largest single subsystem. Multiple parallel sub-tracks; some text-extract paths
got many iterations.

### Text corpus building (current canonical = v2/v3 BGE-M3)

- `extract_granted_patent_corpus.py` — **Active** Extract claim-1 from all granted patents (PatentsView). (Jun 3)
- `extract_pgpub_corpus.py` — **Active** Extract claim-1 from all pre-grant pubs. (Jun 3)
- `paragraph_chunk_specs.py` — **Active** Chunk PatentsView spec text into paragraphs. (Jun 5)
- `embed_spec_chunks.py` — **Active** Embed spec chunks with v2 BGE-M3. (Jun 5)
- `silver_label_spec_chunks.py` — **Active** LLM silver labeling for (claim, spec_chunk). (Jun 5)
- `build_spec_faiss_index.py` — **Active** FAISS IVF over spec chunk embeddings. (Jun 5)

### Anticipation pair extraction + fine-tune

- `extract_anticipation_training_pairs.py` — **Superseded by** `_v2.py`. (Jun 3)
- `extract_anticipation_training_pairs_v2.py` — **Active** Uses both pgpub + granted text sources. (Jun 3)
- `finetune_bge_m3_anticipation.py` — **Active** Fine-tune BGE-M3 on anticipation pairs (MNR loss). (Jun 3)

### FAISS / retriv indexes

- `build_retriv_claim_index.py` — **Superseded** retriv index (crashed in autofaiss float32 JSON). (Jun 3)
- `build_faiss_index_direct.py` — **Superseded by** `build_faiss_index_v3.py`. (Jun 3)
- `build_faiss_index_v3.py` — **Active** v3 = v2 corpus + Google Patents supplement. (Jun 4)
- `bake_top_k_retrievals.py` — **Active** Bake top-K retrievals into patents file as columns. (Jun 3)
- `validate_retriever_on_oard_pairs.py` — **Active** Retriever validation on held-out OARD §102 pairs. (Jun 4)

### Missing-text fetchers

- `fetch_missing_from_bigquery.py` — **Superseded by** `_batched.py`. (Jun 3)
- `fetch_missing_from_bigquery_batched.py` — **Active** Batched BQ supplement. (Jun 3)
- `fetch_missing_from_google_patents.py` — **Active** Google Patents HTML scrape for pre-2001 patents. (Jun 3)
- `fetch_full_text_from_google_patents.py` — **Active** Re-scrape for full text (all claims + spec). (Jun 4)
- `fetch_uspto_bulk_pre2002.py` — **Active** USPTO bulk grant data 1976–2001. (Jun 3)
- `fetch_office_action_text.py` — **Active** Office action text from USPTO ODP. (Jun 4)
- `enumerate_prior_art_union.py` — **Active** Enumerate prior-art union + coverage. (Jun 3)
- `measure_citation_coverage.py` — **Active** Coverage after all text sources loaded. (Jun 3)

### OARD rejection flags

- `download_oard_rejection_flags.py` — **Active** Per-app rejection flags from OARD. (Jun 3)
- `query_oard_rejection_counts.py` — **Active** Per-ground rejection counts. (Jun 3)
- `query_oard_action_subtypes.py` — **Exploratory** Action-subtype counts. (Jun 3)
- `download_patentsview_long_text.py` — **Active** Pull PV long-text bulks from USPTO ODP. (Jun 5)
- `download_patentsview_specs.sh` — **Active** Shell driver for spec bulk download. (Jun 4)

### Patent classifier / aspect features

- `migrate_patent_metrics.py` — **Active** Migrate `metric_implementer_patents/metrics/p*.py` into validity_full/v2/patents codegen format. (Jun 2)
- `draft_patent_exemplars.py` — **Active** Draft calibration exemplars for new aspects a190–a254. (Jun 2)

### Patents analyses & leakage audits

- `analyze_patents_leakage.py` — **Exploratory** Length/marker/cite leakage audit. (Jun 2)
- `analyze_patents_year_drift.py` — **Exploratory** Year drift on balanced file. (Jun 2)
- `analyze_patents_citation_count.py` — **Exploratory** applicant_citations length distribution. (Jun 2)
- `analyze_balanced_cite_distribution.py` — **Exploratory** Cite-count distribution within balanced file. (Jun 2)
- `analyze_cite_count_vs_rejection.py` — **Exploratory** Cite count vs rejection rate. (Jun 3)
- `test_patents_cites_value.py` — **Exploratory** Do applicant cites add signal? (Jun 2)
- `test_patents_cpc_balanced_auc.py` — **Exploratory** AUC test on CPC-balanced file. (Jun 2)
- `test_patents_cpc_top_features.py` — **Exploratory** Top LR features on balanced file. (Jun 2)
- `investigate_claims_canceled.py` — **Exploratory** "claims canceled" leak check. (Jun 2)

### Manual §102 pair deep-dives (June 4 audit run)

- `extract_clean_102_pairs.py` — **Exploratory** Clean §102 pairs by `action_type='102'`. (Jun 5)
- `inspect_10_clean_102_pairs.py` — **Exploratory** Pull 10 clean §102 pairs side-by-side. (Jun 4)
- `manual_check_102_pairs.py` — **Exploratory** Print pairs for manual inspection. (Jun 4)
- `deep_dive_10_pairs.py` — **Exploratory** OA-text + cited-ref full claims/abstract dive. (Jun 4)
- `deep_inspect_10_clean_with_specs.py` — **Exploratory** Above + full spec from Google Patents. (Jun 4)
- `dump_10_pairs_full_text.py` — **Exploratory** Dump pair full text to per-pair files. (Jun 4)
- `peek_oa_text.py` — **Exploratory** Quick peek at OA text. (Jun 4)

---

## 5. Peer review pipeline

Mostly absorbed into v2 + validity_full now. Peer-review-specific items:

- `link_reviews_to_peer_review_dataset.py` — **Active** See §3.
- All R1/R2/aspect work for peer-review now flows through v2/validity_full.

---

## 6. Press releases pipeline

- `process_press_releases_vllm.py` — **Active** Extract clean PR body text via VLLM offline. (Apr 1)
- `postprocess_press_release_extraction.py` — **Active** Parse raw VLLM JSON output. (Apr 4)

---

## 7. News homepages

- `add_snapshot_id_homepages.py` — **Active** Add `snapshot_id` to homepage_newsworthiness_topic_balanced. (May 12)

---

## 8. Notice & Comment (NC)

NC clustering tracks. Driven through the local-explanations / clustering
pipeline (see §15).

- `queue_nc_extractions.sh` — **Active** Sequential STaR-Local extraction across NC overall + 5 agencies. (Apr 15)
- `queue_nc_clustering.sh` — **Active** Full clustering pipeline for all NC tasks. (Apr 16)
- `queue_nc_agency_runs.sh` — **Active** Sequential local_explanations for 5 smallest agencies. (Apr 14)
- `analyze_nc_leakage.py` — **Exploratory** Leakage audit on `notice_and_comment_len_balanced`. (Jun 2)

---

## 9. LeetCode (`lc_*`, `leetcode_*`)

V/A/T pivot to LeetCode for code_review-style tasks (memory: `project_leetcode_push_2026_06_02`).

### LC editorial labeling (June 2)

- `lc_editorial_claude_label.py` — **Active** 15-shard Claude labeling of 1480-pair sample. (Jun 2)
- `lc_editorial_claude_aggregate.py` — **Active** Parse shards → labeled parquet + analysis. (Jun 2)
- `leetcode_editorials/build_editorial_lookup.py` — **Active** Per-(slug,lang) canonical-code lookup. (Jun 5)
- `leetcode_editorials/test_a408_on_balanced.py` — **Exploratory** Test a408 on synthetic LC diffs. (Jun 5)

### LC multi-dim labeling (June 2)

- `lc_multidim_build_sample.py` — **Active** Build 2000-pair multi-dim sample on sk3. (Jun 2)
- `lc_multidim_claude_label.py` — **Active** Parallel Claude labeling across N shards. (Jun 2)
- `lc_multidim_aggregate_and_correlate.py` — **Active** Aggregate + correlate against bank. (Jun 2)

### LC approach triples (June 2)

- `label_lc_approach_triples.py` — **Active** Claude labels approach-matched triples. (Jun 2)
- `analyze_lc_approach_triples.py` — **Active** Bank vs Claude agreement on triples. (Jun 2)

### LC Python MI ladder (June 3)

- `label_lc_python_2k_shard.sh` — **Active** Per-shard launcher. (Jun 3)
- `mi_ladder_lc_python_2k.py` — **Active** MI ladder against Claude binary label on 2K Python LC. (Jun 3)
- `run_lc_python_2k_mi_ladder.py` — **Active** Bank vs bank+a478–a482 ladder. (Jun 3)
- `run_lc_python_runnability.py` — **Active** Compute a480/a481/a482 (pass/runtime/memory). (Jun 3)
- `leetcode_time_matched/build_time_matched_1v1.py` — **Active** Build time-matched 1v1 LC label. (Jun 2)
- `leetcode_time_matched/mi_ladder_time_matched.py` — **Active** MI ladder on time-matched pairs. (Jun 2)

---

## 10. Creative writing pipeline

- `dedup_creative_writing.py` — **Active** MinHash dedup + LDA on LitBench. (May 8)
- `analyze_creative_writing_leakage.py` — **Exploratory** Prompt-level leak + near-dup audit. (May 12)

(NB: `writingprompts_*` and `build_writingprompts_*` scripts live under
`datasets/creative-writing/`, not in `scripts/`.)

---

## 11. Articulation-STaR (rationale-bootstrapping training loop)

Method package; orchestrated by shell scripts here, Python code lives at
`methods/articulation_star/`. See memory entries `project_articulation_star_*`.

- `articulation_star/smoke_test.sh` — **Active** 1-iter / 20-artifact smoke. (sk3)
- `articulation_star/run_iter.sh` — **Active** One full STaR iter (gen → judge → train). (sk3)
- `articulation_star/run_overnight.sh` — **Superseded by** `run_overnight_v2.sh`. (May 30)
- `articulation_star/run_overnight_v2.sh` — **Active** v2 overnight: WEAK=Llama-1B + test eval + leakage probe. (May 30)
- `articulation_star/run_test_eval.sh` — **Active** Held-out test eval across LoRA stages. (sk3)
- `articulation_star/eval_compare.sh` — **Active** Side-by-side base vs LoRA generation. (sk3)
- `articulation_star/run_distilbert_probe.sh` — **Active** DistilBERT leakage probe per stage. (sk3)
- `articulation_star/run_explore_contrastive.sh` — **Exploratory** Audit-only weak/strong contrastive run. (sk3)

---

## 12. Llama norm extraction (`llama_norm_extraction/`)

- `llama_norm_extraction/run_openrouter.py` — **Active** Iteration harness on small batches via OpenRouter. (Jun 2)
- `llama_norm_extraction/run_sk3.py` — **Active** Bulk vLLM Qwen3.5-122B on sk3 (resumable). (Jun 2)

---

## 13. AutoMetrics + Metric Tree (proposer/router methods)

See memory: `autometrics_architecture`, `project_three_algorithms`, `project_metrics_tree_infilling`.

- `run_autometrics_vllm.py` — **Active** Iterative AutoMetrics, single GPU vLLM. (Apr 14)
- `run_metric_tree.py` — **Active** Partitioned metric tree with VLLM backend. (Apr 14)
- `resume_metric_tree_inference.py` — **Active** Resume inference after a saved tree. (Apr 14)
- `test_metric_tree.py` — **Exploratory** Small end-to-end smoke test. (Mar 15)
- `metric_test_pilot.py` — **Exploratory** Empirical V/A/T level-assignment pilot. (May 22)
- `tools/run_metric_tree_sk3.sh` — **Tooling** sk3 launcher.

---

## 14. Verification Library (current canonical Active version)

- `run_verification_library.py` — **Active** Verification library discovery algorithm. (Apr 26)

### STaR-Local feature extraction (Apr 27–28)

- `batch_generate_programs.py` — **Active** Phase 1: generate predict-programs per example. (Apr 27)
- `batch_star_features.py` — **Active** Direction 2: STaR-like feature extraction. (Apr 28)
- `batch_code_features.py` — **Active** Code every canonical feature with Qwen. (Apr 27)
- `multipass_code_features.py` — **Active** Re-run failures, keep best per cluster. (Apr 28)
- `reparse_coded_features.py` — **Active** Re-parse raw_responses with improved code extraction. (Apr 28)
- `dedup_star_features.py` — **Active** Dedup 133K STaR features → ~13K canonicals via kmeans. (Apr 27)
- `build_code_library.py` — **Active** Embedding-sorted incremental code refactoring. (Apr 28)
- `build_text_taxonomy.py` — **Active** Embedding-sorted LLM hierarchy over canonical features. (Apr 28)

---

## 15. Local explanations + clustering pipeline

The April clustering driver; many auxiliaries.

- `run_local_explanations.py` — **Active** Rationalization or STaR-local driver. (Apr 28)
- `extract_only.py` — **Active** Step 1 only (cache extractions). (Apr 15)
- `run_clustering_only.py` — **Active** Steps 3–6 only. (Apr 14)
- `run_full_clustering_pipeline.py` — **Active** All steps 1–6. (Apr 16)
- `optuna_sweep_local_explanations.py` — **Active** Optuna sweep over pipeline hyperparams. (Apr 15)
- `run_best_sweep_config.py` — **Active** Run best Optuna config. (Apr 15)
- `select_canonical_via_dedup.py` — **Active** Dedup-first canonical feature selection. (Apr 15)
- `run_dedup_then_step45.sh` — **Active** Chain: dedup → Steps 4+5. (Apr 15)
- `refine_features_from_misclassifications.py` — **Active** Re-extract on mispredicted training docs. (Apr 14)
- `sweep_cluster_selection_epsilon.py` — **Exploratory** HDBSCAN epsilon + LLM dedup sweep. (Apr 14)
- `sweep_umap_target_weight.py` — **Exploratory** UMAP target_weight sweep. (Apr 14)
- `compute_topk_discrimination.py` — **Tooling** Post-hoc top-K discrimination metric. (Apr 14)

---

## 16. Hierarchy / R1 / R2 / R2.5 / R3 (cluster→family→aspect pipeline)

Heavy May 12–28 work on building the rule (R1) → aspect (R2) hierarchy across
tasks. Some scripts are run via watcher shell scripts; many cells were redone
multiple times.

### Clustering primitives (May 12)

- `find_dedup_candidates.py` — **Active** Embedding-based dedup candidates. (May 12)
- `find_all_candidates.py` — **Active** All within-bucket candidate pairs. (May 12)
- `cluster_pairs.py` — **Active** Anchored complete-linkage with bridge tracking. (May 12)
- `cluster_embeddings_only.py` — **Active** LLM-free embedding clustering. (May 12)
- `cluster_complete_linkage.py` — **Active** Hybrid single→complete linkage on 12K rubrics. (May 12)
- `purify_clusters.py` — **Active** Post-hoc LLM purification of complete-linkage clusters. (May 12)
- `dedup_rubrics_v1.py` — **Active** v1 pairwise dedup via gpt-5-mini. (May 12)
- `run_dedup_2k_eval.py` — **Exploratory** Stratified 2K-pair eval of the dedup classifier. (May 12)
- `embed_correlation_nvembed.py` — **Exploratory** nv-embed correlation cross-check. (May 12)

### Hierarchy building / refinement (May 12–13)

- `build_hierarchy.py` — **Active** Iterative LLM-driven hierarchy builder. (May 12)
- `meta_merge.py` — **Active** Round 2+ meta-merge of LLM-proposed parents. (May 12)
- `refine_parent.py` — **Active** Within-parent refinement (drop misfits). (May 12)
- `run_all_refinements.sh` — **Active** Parallel `refine_parent` over all R1 cells. (May 13)
- `run_r2_watcher.sh` — **Active** Per-cell R2 meta-merge watcher. (May 13)
- `run_r3_watcher.sh` — **Active** R2 → R3 watcher. (May 13)
- `r2_to_r3_input.py` — **Active** Convert R2 expanded → R3 input. (May 12)
- `generate_few_shots.py` — **Active** Generate few-shots for the dedup classifier. (May 12)
- `scan_r2_timeouts.py` — **Exploratory** Scan R2 outputs for timeouts. (May 13)
- `reset_timed_out_cells.sh` — **Tooling** Reset timed-out cells so watchers re-fire. (May 13)
- `inspect_r1.py` / `inspect_r2.py` / `inspect_r3.py` — **Exploratory** Sanity inspectors. (May 13)
- `compression_by_bucket.py` — **Exploratory** Compression ratios per (task, bucket). (May 13)

### sk3 R1/R2 LLM batched build (May 21–24)

- `sk3_build_r1.py` — **Active** Llama-70B vLLM R1 family builder per task. Big file. (May 21)
- `r1_local_prep.py` — **Active** Local prep for R1-via-subagents. (May 23)
- `aggregate_subagent_r1.py` — **Active** Aggregate per-batch subagent R1 → r1_families. (May 23)
- `validate_r1.py` — **Exploratory** Manual R1 validation. (May 21)
- `validate_r1_against_v6.py` — **Exploratory** Validate against v6 judge pair labels. (May 23)
- `test_r1_prompt.py` — **Exploratory** R1 prompt test on peer-review sample. (May 19)
- `r1_metrics.py` — **Tooling** Aggregate R1 metrics for one or more output dirs. (May 21)

### Fork / merge passes (May 23)

- `r1_merge_pass_prep.py` — **Active** Step 1 of LoRA-bge post-hoc merge pass. (May 23)
- `r1_meta_merge_prep.py` — **Active** Approach A.2 meta-level R1 prompt. (May 23)
- `r1_meta_merge_apply.py` — **Active** Apply meta-merge verdicts via union-find. (May 23)
- `r1_fork3_pairmerge_prep.py` — **Active** Fork 3 pairwise post-hoc merge prep. (May 23)
- `r1_fork3_pairmerge_apply.py` — **Active** Apply Fork 3 verdicts via union-find. (May 23)
- `spot_check_fork3_hard_fps.py` — **Exploratory** Inspect 56 hard FPs from Fork 3. (May 23)
- `spot_check_r1_diff.py` — **Exploratory** Subagent R1 vs sk3-Llama R1 spot-check. (May 23)
- `spot_check_bs40_vs_bs200.py` — **Exploratory** Batch size sensitivity check. (May 23)
- `sk3_fork_b_consistency.py` — **Superseded by** `sk3_fork_b_v2_consistency.py`. (May 23)
- `sk3_fork_b_v2_consistency.py` — **Active** Variance from real sources (not just anchor shuffle). (May 24)

### R2 (theme aspects)

- `r2_subagent_prep.py` — **Active** Theme-grouping R1 families into R2 aspects. (May 24)
- `r2_aggregate_labels.py` — **Active** Aggregate 11 R2 attribute-labeling outputs. (May 28)
- `build_r2_labeling_input.py` — **Active** Per-task labeling inputs. (May 28)
- `r2_aspect_year_mentions.py` — **Active** Per-mention table for time plots. (May 28)
- `r2_attr_time_join.py` — **Active** Join R2 attributes + time profile. (May 28)
- `r2_time_analysis.py` — **Active** Per-R2-aspect source-year stats. (May 28)
- `aggregate_r2.py` — **Tooling** R2 aggregator. (May 24)
- `spot_check_r2.py` — **Exploratory** Extensive R2 spot-check. (May 24)
- `r2_subagent_prep.py` — see above.

### R2.5 / R2 post

- `r25_aspect_merge_prep.py` — **Active** Cross-batch merge prep for R2 aspects. (May 24)
- `r25_aspect_merge_apply.py` — **Active** Apply R2.5 verdicts. (May 24)

### Open coding

- `build_open_coding_samples.py` — **Active** Sample N R2_post aspects per task. (May 28)
- `consolidate_open_coding.py` — **Active** Consolidate codes across 11 tasks. (May 28)
- `build_open_coding_samples_r1.py` — **Active** R1 open-coding sample. (May 28)
- `consolidate_open_coding_r1.py` — **Active** R1 consolidator. (May 28)

---

## 17. Canonicalization + clustering of leaf rubric forms (sk3)

- `canonicalize_leaves.py` — **Active** OpenRouter canonicalization driver. (May 18)
- `sk3_canonicalize_vllm.py` — **Active** sk3 vLLM Llama-70B-FP8 sharded canonicalization. (May 18)
- `build_canon_stress_sample.py` — **Active** Calibration stress sample. (May 18)
- `cluster_canon.py` — **Active** Re-cluster canonical forms across thresholds. (May 18)
- `examine_tau_bands.py` — **Exploratory** Tau decision zone families. (May 18)

### Locked clustering recipe (canonical FP/FN — see memory `project_rubric_clustering_pipeline`)

- `build_judge_pool.py` — **Active** Stratified pair pool for sameness judge. (May 18)
- `build_judge_calib.py` — **Active** Calibration set for sameness judge. (May 18)
- `calib_judge.py` — **Active** Calibration runner via OpenRouter. (May 18)
- `judge_prompt.py` — **Active** Shared 0–3 graded sameness prompt. (May 18)
- `sk3_judge_pairs.py` — **Active** sk3 vLLM judge over pair pool. (May 18)
- `tau_fpfn.py` — **Active** FP/FN vs agglomeration tau. (May 18)
- `build_train_pairs.py` — **Active** Per-task LoRA train pairs from v6 judge. (May 18)
- `build_more_train_pairs.py` — **Active** +300K more pairs. (May 18)
- `sk3_train_lora.py` — **Active** Per-task LoRA on bge-large with CoSENT. (May 18)
- `sk3_train_ce.py` — **Active** Per-task cross-encoder (ModernBERT-base) on judge pairs. (May 18)
- `sk3_eval_ce.py` — **Active** Held-out CE eval. (May 18)
- `eval_lora.py` — **Active** FP/FN: baseline vs LoRA bge. (May 18)
- `sk3_match_pipeline.py` — **Active** CE re-rank → hybrid affinity → complete linkage. (May 18)
- `sk3_blend_sweep.py` — **Active** Sweep CE/cos blend weight. (May 18)
- `sk3_linkage_test.py` — **Active** Compare linkage methods. (May 18)
- `sk3_inspect_clusters.py` — **Exploratory** Manual cluster inspection at tau. (May 18)
- `sk3_finalize_clusters.py` — **Active** Final average-linkage cut at tau 0.92. (May 18)
- `sk3_tau_compression.py` — **Exploratory** Compression vs tau curves. (May 19)
- `sk3_verify_singletons.py` — **Exploratory** Exhaustive singleton nearest-neighbour search. (May 19)
- `sk3_operator_sweep.py` — **Exploratory** Sweep CE/cos combination operators. (May 19)
- `sk3_structural_metrics.py` — **Active** Structural / cross-task metrics on locked clustering. (May 19)
- `sk3_consensus_metrics.py` — **Active** Cross-source consensus metrics. (May 19)
- `sk3_cross_task_concepts.py` — **Active** Cross-task universal-concept analysis (confound-stripped). (May 19)

### Embeds for clustering

- `sk3_embed.py` — **Superseded** Generic re-embed driver. (May 16)
- `sk3_embed_leaves.py` — **Active** bge-large embed of leaf names. (May 16)
- `sk3_embed_canon.py` — **Active** bge-large embed of canonical leaf forms. (May 18)
- `sk3_embed_nemotron.py` — **Superseded by** `sk3_embed_bge_all.py` (memory: nemotron instr collapsed). (May 18)
- `sk3_embed_bge_all.py` — **Active** bge-large per (bucket, task). (May 18)
- `export_leaf_names.py` — **Active** Export leaf names for sk3 embed. (May 16)
- `export_canon_forms.py` — **Active** Export real on-topic canonical forms. (May 18)
- `export_spec_leaves.py` — **Active** Spec + hyper_spec leaf names export. (May 18)
- `export_all_real_forms.py` — **Active** Combine general + spec canonical forms. (May 18)
- `export_cluster_texts.py` — **Active** Export cluster + subtask texts for sk3 re-embed. (May 16)

---

## 18. Dedup / classifier exploration on rubrics (May 11–12 batch)

Big extractor/classifier prompt set; most heavy lifting now lives in
batch_extract_vllm + batch_classify_vllm.

### Current canonical (Active)

- `batch_extract_vllm.py` — **Active** Offline batch extractor (Llama-3.3-70B BF16 vLLM). (May 12)
- `batch_classify_vllm.py` — **Active** Offline batch classifier over 361K rubrics. (May 12)
- `extract_rubric_features_v5_prompt.py` — **Active** v5 per-task system prompt. (May 12)
- `classify_rubric_llama_prompt.py` — **Active** Llama-3.3-70B classifier prompt. (May 12)
- `task_taxonomy.py` — **Active** Shared per-task taxonomy (imported by extract + classify). (May 12)
- `classify_rubric_v2_prompt.py` — **Active** gpt-5-mini v2 classifier prompt (shares taxonomy). (May 11)

### Superseded

- `extract_rubric_features.py` — **Superseded** Original async OpenAI extractor. (May 11)
- `chunked_extract.py` — **Superseded** Chunked variant for huge PDFs. (May 11)
- `extract_rubric_features_v4_prompt.py` — **Superseded by** v5. (May 11)
- `extract_rubric_features_v5verbose_prompt.py` — **Superseded** Verbose v5 variant. (May 12)
- `classify_rubric_v1_prompt.py` — **Superseded by** v2. (May 11)

### Exploratory tests around these prompts

- `test_classifier_v1.py` — **Exploratory** (May 11)
- `test_classifier_v2.py` — **Exploratory** (May 11)
- `test_classifier_llama.py` — **Exploratory** (May 11)
- `test_classifier_verifiability_openrouter.py` — **Exploratory** (May 12)
- `test_llama_v4_openrouter.py` — **Exploratory** (May 11)
- `test_llama_v5_openrouter.py` — **Exploratory** (May 11)
- `test_llama_v5verbose_openrouter.py` — **Exploratory** (May 12)
- `scale_classifier_v2.py` — **Exploratory** Scale-out check. (May 11)
- `audit_classifier_100.py` — **Exploratory** 100-rubric audit. (May 12)
- `analyze_tractability_errors.py` — **Exploratory** Per-axis disagreement printer. (May 12)

### Misc rubric-side tooling

- `capsolver_book_downloader.py` — **Tooling** Captcha-aware book downloader. (May 11)
- `organize_rubrics.sh` — **Tooling** Organize `online-rubrics/` into claude-parsed / raw subdirs. (May 10)

---

## 19. 2-axis articulability classification (May 14–15)

Output of an audit + correction cycle on cluster articulability.

- `classify_clusters_2axis.py` — **Active** Two-axis classification of rubric clusters. (May 15)
- `aggregate_cluster_2axis.py` — **Active** Per-task distribution. (May 15)
- `confound_analysis.py` — **Exploratory** Structural-confound hunt. (May 15)
- `confound_correction.py` — **Exploratory** Quantify + correct L3 jargon inflation. (May 15)
- `aggregate_v7_corrected.py` — **Active** Final per-task distribution after v7 L3 re-judge. (May 15)
- `sample_audit.py` — **Exploratory** Build audit samples for subagent re-assessment. (May 15)
- `inspect_cluster_quality.py` — **Exploratory** Singleton coherence check. (May 15)
- `embedding_diversity.py` — **Exploratory** Threshold-free diversity metrics. (May 15)
- `extract_noun_verb_chains.py` — **Exploratory** Noun/verb thickness chains over R2. (May 14)
- `extract_source_years.py` — **Active** Per-source year via gpt-5-mini. (May 14)
- `concentration_metrics.py` — **Exploratory** Per-cell concentration metrics. (May 14)
- `dispersion_crosscheck.py` — **Exploratory** Cross-check dispersion vs bge re-embed. (May 16)
- `dispersion_transforms.py` — **Exploratory** Expand dispersion dynamic range. (May 16)
- `singleton_audit.py` — **Exploratory** Near-neighbour pairs for singleton adjudication. (May 16)
- `leaf_name_clusters.py` — **Exploratory** Under-merge measurement via leaf-NAME embeddings. (May 16)

---

## 20. Dense reward-model training & queue (sk3)

See memory: `project_dense_model_sweeps`, `reference_sk3_queue_supervisor`.

- `queue_supervisor.sh` — **Active** Cron-driven idempotent supervisor for queue workers. (May 18)
- `queue_gpu1.sh` — **Active** Per-GPU queue worker (math|humor|codereview). (May 18)
- `tools/train_sweep.sh` — **Tooling** Dense reward-model data-fraction sweep. (sk3)
- `wait_and_launch_classifier.sh` — **Tooling** Polls nvidia-smi to launch extra classifier workers. (May 12)
- `wait_and_launch_classifier_gpu7.sh` — **Tooling** GPU-7-specific variant. (May 12)
- `eval_model.py` — **Active** Global evaluation runner for dense reward models. (Mar 19)

---

## 21. Topic modeling / dataset inspection / generic helpers

- `inspect_datasets.py` — **Active** Cleanliness audit across modeling datasets. (May 3)
- `inspect_datasets_v2.py` — **Active** Fast variant. (May 3)
- `run_topic_modeling.py` — **Active** LDA k=100 across 12 datasets. (May 3)
- `run_topic_model.py` — **Superseded** Single-CSV LDA driver from March. (Mar 19)
- `openrouter.py` — **Tooling** Shared async OpenRouter client (imported across validity_pilot). (May 23)
- `code_gen_phase.py` — **Active** Code-gen phase for validity pilot (multi-model). (May 23)

---

## 22. Patents anticipation chained launchers (sk3 shell)

- `autorun_spec_pipeline.sh` — **Active** Wait-for-download → chunk → embed → index spec passages. (Jun 5)
- `run_anticipation_pipeline.sh` — **Active** Chain: extract pairs → finetune → retriv index. (Jun 3)
- `run_v2_anticipation_pipeline.sh` — **Active** v2 variant after granted parquet lands. (Jun 3)
- `run_v2_finetune_then_index.sh` — **Active** v2 fine-tune + retriv index. (Jun 3)
- `run_v2_finetune_fast.sh` — **Active** 1M sampled pairs, 1 epoch fast variant. (Jun 3)
- `run_v3_when_gp_done.sh` — **Active** v3 FAISS + re-bake top-K. (Jun 4)
- `run_index_then_bake.sh` — **Active** FAISS index → bake top-K. (Jun 3)
- `loop_until_coverage.sh` — **Active** Master loop: build → measure → BQ fetch → re-measure. (Jun 3)

---

## 23. Tooling (`scripts/tools/`)

All entries here are **Tooling**.

- `tools/launch_when_gpus_free.sh` — Wait for N free GPUs, then run a command.
- `tools/run_metric_tree_sk3.sh` — Launcher for Metric Tree on sk3.
- `tools/sync_with_sk.py` — Sync local norm-research dir with sk hosts.
- `tools/train_sweep.sh` — Dense reward-model data-fraction sweep.

## 24. Assets

- `assets/preference-modeling-tasks.png` — **Tooling** Figure asset.

---

## Approximate counts

| Category | Approx count |
|---|---|
| **Active** (current canonical) | ~155 |
| **Superseded** (clearly replaced) | ~25 |
| **Exploratory** (dated one-offs, findings consumed) | ~115 |
| **Tooling** (helpers, launchers, shared modules) | ~45 |

When in doubt, search the corresponding `notes/2026-MM-DD__*.md` file by mtime, or
the memory entries listed in `MEMORY.md` for the relevant `project_*` namespace.
