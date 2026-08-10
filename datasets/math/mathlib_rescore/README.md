# Mathlib A-bank rescore

This directory contains the audit-prescribed v2 articulated-criteria pipeline for
mathlib pull-request title+diffs. It is an authoring artifact: no scoring job was
launched while these files were prepared.

## Standing rules

- Rubric authoring and scoring are label-blind. The scorer projects parquet input
  onto `title` and `diff` only, and output shards contain no outcome or partition
  metadata.
- The judge is **Gemma-4-31B** at the pinned local snapshot in
  `score_mathlib_gemma.py`.
- Scoring uses **offline-batch vLLM**, never an HTTP server.
- On sk3, vLLM runs with multiprocessing `spawn` and
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The script sets both
  `VLLM_WORKER_MULTIPROC_METHOD=spawn` and Python's start method before vLLM is
  imported.
- `gpu_memory_utilization` must stay in the `.93-.95` range; the default is `.94`.
- Generation preserves the reference protocol exactly in structure:
  `1.0`/`0.5`/`0.0`/`NA`, temperature `0`, `max_tokens=6`, and prefix caching.
- Every vLLM shard/batch includes **exactly three blinded anchor PRs**:
  a merged trivial-fix PR, a rejected PR that visibly adds `sorry` (or `admit`),
  and a deterministic shuffled-diff PR. This follows the fixed-RNG
  seed-anchor-plus-scramble pattern in
  `datasets/humor/caption_multiy/score_va_gemma_captions.py`.

The anti-compression instruction is part of the judge prompt. It explicitly asks
the model to discriminate, use the full token range, avoid mode collapse, and
justify its choice internally before returning only the score token.

## Run order

1. Use `rubrics_v2.jsonl` as the A-bank. It contains the ten fidelity-rewritten
   base criteria plus eight additions.
2. Optionally run `retrieval_context.py` to produce
   `toscore_with_context.jsonl`.
3. Run `score_mathlib_gemma.py` on either the original parquet (context-free) or
   the enriched JSONL. The context-free path is independent of retrieval, so it
   is safe and useful to score both variants into separate output directories.

In deliverable numbering, that is **1 -> optionally 3 -> 2**.

## Prepare the blinded anchors

An operator or data custodian must prepare
`mathlib_rescore/anchors_blinded.jsonl` separately. It must contain exactly two
seed rows and only the fields shown here:

```json
{"anchor_kind":"merged_trivial_fix","title":"<title>","diff":"<diff>"}
{"anchor_kind":"rejected_sorry","title":"<title>","diff":"<diff that visibly adds sorry>"}
```

Selection is performed outside this pipeline, then all outcome/partition
metadata is stripped before the file reaches the scorer. The scorer validates
the two anchor kinds, verifies that the second diff visibly adds `sorry` or
`admit`, constructs the shuffled third anchor with `random.Random(0)`, and
appends the same three blinded records to every shard. In each `.npz`,
`is_anchor` and `anchor_kind` identify those rows so downstream analysis can
exclude them from estimates while retaining their sanity-check scores.

## Commands on sk3

After copying this directory to
`/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib_rescore`, use the
Gemma environment and one B200-class GPU:

```bash
export MATHLIB_DATA=/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib
export RESCORE_DIR=/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib_rescore
export GEMMA_PY=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
```

Optional retrieval enrichment:

```bash
"$GEMMA_PY" "$RESCORE_DIR/retrieval_context.py" \
  --input "$MATHLIB_DATA/accept_reject_clean.parquet" \
  --index "$MATHLIB_DATA/library_decl_index.jsonl" \
  --output "$RESCORE_DIR/toscore_with_context.jsonl" \
  --neighbors 3
```

Context-free scoring:

```bash
"$GEMMA_PY" "$RESCORE_DIR/score_mathlib_gemma.py" \
  --input "$MATHLIB_DATA/accept_reject_clean.parquet" \
  --rubrics "$RESCORE_DIR/rubrics_v2.jsonl" \
  --anchors "$RESCORE_DIR/anchors_blinded.jsonl" \
  --outdir "$RESCORE_DIR/scores_context_free" \
  --util 0.94 --max-model-len 8192 --num-shards 8
```

Context-enriched scoring:

```bash
"$GEMMA_PY" "$RESCORE_DIR/score_mathlib_gemma.py" \
  --input "$RESCORE_DIR/toscore_with_context.jsonl" \
  --rubrics "$RESCORE_DIR/rubrics_v2.jsonl" \
  --anchors "$RESCORE_DIR/anchors_blinded.jsonl" \
  --outdir "$RESCORE_DIR/scores_with_context" \
  --util 0.94 --max-model-len 8192 --num-shards 8
```

Each invocation is one offline vLLM process and writes
`nc_scores_shard000.npz`, `nc_scores_shard001.npz`, and so on. It skips existing
shards unless `--overwrite` is explicitly supplied.

The pinned BF16 Gemma-4-31B snapshot fits on one B200-class GPU with tensor
parallel size 1. For multi-GPU throughput, launch one process per GPU with a
different `CUDA_VISIBLE_DEVICES` and `--shard-id`, keeping `--num-shards`
identical across processes. Do not point two processes at the same shard.

## Output layout

Each shard contains:

- `X`: rows (real PRs followed by three anchors) by rubric columns;
- `doc_id`: stable title+diff hashes for real PRs and reserved IDs for anchors;
- `rubric_ids` and `a_names`: column metadata;
- `is_anchor` and `anchor_kind`: blinded sanity-check metadata;
- `na_rate`, `truncated_pairs`, and `has_retrieval_context`: run diagnostics.

The scorer token-truncates the combined title+diff separately for each rubric
after accounting for the rendered Gemma chat template, criterion text, optional
retrieval evidence, and six output tokens. Every rendered prompt therefore fits
`max_model_len=8192`.
