# Context-aware content-task Nemotron LoRAs

Run root:
`/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context`

The `teacher_sets/<task>/teacher_build_report.json` files are authoritative for
input/output hashes and leakage checks.  Independent human dev/test rows are in
`external_dev_test.jsonl`; every source group represented there is blocked
from the high-volume Sonnet distillation set.  Human train labels override a
conflicting Sonnet label.  Trainer-internal source-hash splits select epochs;
external dev determines promotion.  External test is run once only after the
dev decision is frozen.

## Current decisions (2026-07-12)

* `creative-writing`: the raw-Sonnet adapter is rejected.  External dev
  recall@50 fell from 0.8143 to 0.2286 (n=70), so its frozen test has not been
  touched and the adapter must not be used for production retrieval.  The
  Sonnet teachers are concentrated in `creative_writing`, whereas the external
  evaluation is mostly LitBench.
  A dev-selected BGE hybrid is currently the strongest baseline at exact
  recall@50 0.9000 (n=70); base Nemotron comparison is still required before
  freezing it.  Its provisional production K=50 captures 1,394/2,060 Sonnet
  proposals naturally; 666 (32.33%) are explicitly injected for judging.
* `press-releases`: the context-aware Nemotron adapter is promoted.  External
  dev recall@50 rose from 0.5741 to 0.8148 (n=54).  Its frozen test was invoked
  exactly once and recall@50 rose from 0.7143 to 0.8571 (n=56).  The immutable
  dense promotion record is
  `promotions/press-releases.SEALED.json`; never rerun this test.
* `press-releases` fusion: dev selected dense rank 2/3 plus character rank 1/3.
  It reaches recall@50 0.8889 on dev.  It is intentionally untested because the
  dense test was already consumed; see
  `promotions/press-releases.FUSION_DEV_ONLY_UNTESTED.SEALED.json`.
* `humor`: do not train the raw-Sonnet adapter described by the historical
  command below.  First select the final baseline retriever/fusion on external
  dev, verify proposals twice with order-perturbed Gemma-4 on final K=50
  slates, and independently hand-audit a stratified train-only sample of
  retained, corrected, and rejected proposals.  Only verified exact matches
  plus human-train labels may enter a new Humor LoRA.
  The dev-selected BGE hybrid currently has exact recall@50 0.6038 over 53
  exact-match labels; base Nemotron comparison is still pending.

The historical first-pass command is retained for reproducibility, not as the
current recipe:

All training commands use physical GPU 4 (`CUDA_VISIBLE_DEVICES=4`) and:

```bash
cd /lfs/skampere3/0/alexspan/norm-research
export PYTHONPATH=/lfs/skampere3/0/alexspan/norm-research
export HF_MODULES_CACHE=/lfs/skampere3/0/alexspan/.cache/huggingface/modules
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
ROOT=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context
FAITH=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful

for TASK in creative-writing press-releases; do
  "$PY" -m scripts.tools.silver_match_v3.train_nemotron_lora \
    --task "$TASK" --manifest "$FAITH/manifest.json" \
    --teachers "$ROOT/teacher_sets/$TASK/teacher_train.jsonl" \
    --output-root "$ROOT/adapters_lr2e5_e2" --device cuda \
    --attention eager --max-seq-length 512 --epochs 2 --batch-size 8 \
    --gradient-accumulation-steps 4 --eval-batch-size 32 \
    --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.1 \
    --margin 0.15 --hard-negative-pool 24 --negatives-per-positive 2 \
    --lora-rank 32 --lora-alpha 64 --lora-dropout 0.05 \
    --min-dev-recall-gain 0.01 --no-enforce-promotion-gate
done
```

For any internally promotable adapter, external dev is evaluated with:

```bash
"$PY" -m scripts.tools.silver_match_v3.evaluate_nemotron_adapter \
  --manifest "$FAITH/manifest.json" \
  --labels "$ROOT/teacher_sets/$TASK/external_dev_test.jsonl" \
  --task "$TASK" --split dev \
  --adapter "$ROOT/adapters_lr2e5_e2/$TASK/adapter" \
  --output "$ROOT/external_evals/$TASK.dev.json" --device cuda
```

Only an adapter retained on external dev receives the analogous `--split
test` invocation.  A rejected adapter remains an auditable diagnostic and is
never used for production retrieval.

## Production teacher verification

Sonnet labels are proposals, not exact-match truth.  Production verification
must use the same final K=50 retrieval variant that will be deployed.  If a
proposal is missing from that K=50, `prepare_teacher_verification compact`
may inject it for adjudication, but the injection flag must remain explicit and
must never be reported as natural retrieval capture.  For press releases,
813/7,869 proposals (10.33%) required such injection.

Run the dev-selected GEPA verifier twice with different alternative ordering.
`finalize_teacher_verifications.py` retains an exact teacher only when both
runs return `CONFIRM_MATCH` at medium/high confidence and at least one run is
high confidence.  Stable corrections are hard-contrast evidence only; they do
not become positive teachers.  Test data must never be used for prompt
selection, teacher filtering, fusion selection, or threshold calibration.

Point precision from a tiny verifier dev set is not sufficient.  Power-aware
GEPA selections record Wilson intervals, retained sample size, and whether an
independent audit is required.  Underpowered selections cause finalized rows
to be written with `gradient_eligible: false`.  Auditors receive a blinded,
stratified packet that omits Gemma outcomes, reasons, injection flags, rarity,
and stratum identity.  `promote_audited_teachers.py` is the only promotion path:
by default it requires at least 30 audited retains, design-weighted precision
at least 0.90, and an approximate 95% interval lower bound at least 0.80 before
emitting a new immutable `gradient_eligible: true` teacher file.
