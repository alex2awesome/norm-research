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

Where auxiliary teachers exist, the headline pooled adapter first learns their
distributions at quarter weight, then receives a lower-learning-rate
Sonnet/GPT-5 refinement. A matched primary-only adapter supplies the
auxiliary-data ablation. Levels without auxiliary targets skip that meaningless
extra comparison. Every powered task-level cell continues on primary labels
from its pooled adapter.

Evaluation reports three-way macro-F1, Cohen's kappa, SAME precision/recall/F1,
ordinal error, Brier score, calibration, input-order consistency, protocol
breakdowns, and cold-concept performance. A task adapter is promoted only if
it clears the predeclared paired-bootstrap and non-regression gates.
