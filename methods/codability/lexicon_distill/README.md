# Gemma-4 similarity distillation

This package turns the persisted lexicon-pipeline similarity judgments into
hash-frozen R1/R2/R3 datasets and trains constrained three-way Gemma-4-31B
LoRA classifiers. It makes no API calls and never treats string matching or a
lexical heuristic as semantic truth.

## Frozen dataset

Build locally with:

```bash
PYTHONPATH=. python -m methods.codability.lexicon_distill.dataset --replace
```

The July 13 v1 freeze contains 199,100 canonical text-backed pairs: 171,503
R1, 17,585 R2, and 10,012 R3. The builder:

- joins terse votes to the exact persisted judge payloads;
- excludes L0 repair judgments;
- balances Sonnet and GPT-5 after averaging within teacher family;
- retains attributable Opus/GLM-only pairs at quarter weight;
- keeps legacy R2, R2-v2, and R2-v2.1 protocol-conditioned and distinct;
- reserves audit identities across protocol variants;
- lets attributable reserved labels override earlier build labels;
- produces pair-disjoint and cold-concept test views; and
- quarantines rows whose teacher, task, level, or displayed pair is ambiguous.

`manifest.json` binds every source and generated artifact by SHA-256.
`inventory.json` contains the exact split counts and powered task-level cells.

## sk2 execution

Training is hard-guarded to sk2. The frozen DAG uses the sk2 Gemma snapshot,
an isolated Transformers 5.12.1 / PEFT 0.18.1 environment, and H200 GPUs. No
sk3 path is accepted.

```bash
PYTHONPATH=. python -m methods.codability.lexicon_distill.freeze_sk2_jobs
python -m methods.codability.lexicon_distill.run_sk2_jobs \
  --plan outputs/lexicon/similarity_distill_v1/sk2_jobs.json
```

Where auxiliary teachers exist, the headline pooled adapter first sees all of
their distributions at one quarter of the primary learning rate, then receives
a Sonnet/GPT-5 refinement. A matched primary-only adapter supplies the
auxiliary-data ablation. Primary fits use `2e-5`; the auxiliary curriculum uses
`5e-6`. Levels without auxiliary targets skip that meaningless extra
comparison. Every powered task-level cell continues on primary labels from its
pooled adapter. Cells with some task-local training data but insufficient
weighted power also receive a descriptive continuation; their reports can
diagnose heterogeneity but are never eligible for automatic promotion. Tasks
with no training pairs remain pooled-only out-of-domain generalization tests.

Gemma-4's BF16 backward pass can produce a non-finite gradient on a rare,
deterministic accumulation window even when the loss is finite. Trainable LoRA
weights and optimizer state therefore remain FP32. Batches are shuffled in
length-local buckets, and explicit position IDs make left padding invariant to
the other prompts in a batch. A fit may quarantine at most 1% of its residual
non-finite accumulation windows; every skipped example ID and order view is
written into the training report, and exceeding that scale-aware ceiling fails
the fit. No non-finite update is ever applied or silently counted as a
completed optimizer step.

Evaluation reports three-way macro-F1, Cohen's kappa, SAME precision/recall/F1,
ordinal error, Brier score, calibration, input-order consistency, protocol
breakdowns, teacher-family breakdowns, and cold-concept performance. A task
adapter is promoted only if it clears the predeclared paired-bootstrap and
non-regression gates.
