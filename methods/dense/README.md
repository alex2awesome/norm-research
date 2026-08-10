# dense

## Purpose

Train a **dense reward model** (Llama-3.1-8B or 70B with LoRA / QLoRA) as the
classifier baseline for each task: read the full artifact, predict the binary
`judgement` label with a pointwise BCE head, optionally also a pairwise
Bradley-Terry head. This is the ceiling we compare every articulable /
metric-based method against — if a transparent rubric pipeline matches the
dense model, the task is articulable; if it lags, the gap is the
"taste residual."

## Algorithm sketch

1. Load a CSV(.gz) with `text` and `judgement` columns; persist an
   80/10/10 train/eval/test split on disk (`split_metadata.json`).
2. Tokenize, build `RewardDataset` (pointwise) or
   `BradleyTerryPairDataset` (sampled (pos, neg) pairs/epoch).
3. Load a HF causal model with `AutoModelForSequenceClassification`,
   optional 4-bit BnB quantization (QLoRA), gradient checkpointing,
   FSDP via `accelerate`.
4. Wrap with PEFT LoRA on all attention + MLP modules
   (`q,k,v,o,gate,up,down`).
5. Train with BCE-with-logits (`--class_weight_auto` for imbalanced
   tasks) or Bradley-Terry pairwise loss (`--bradley-terry`).
6. Evaluate `EVALS_PER_EPOCH = 5` times per epoch; track AUC, F1,
   precision, recall, accuracy, loss; optionally run dataset-specific
   sliced eval via `evals.py` discovered next to `data_path`.
7. Save best checkpoint by `eval_auc`; emit `training_history.json` +
   `validation_metrics.csv`.
8. Optionally hyperparameter-sweep via `--use_optuna`.

## Key files

- `train_reward_model.py` — the only entrypoint. ~1200 lines. Handles:
  `parse_args`, `get_or_create_fixed_split`, `build_model`, `train`,
  `run_optuna`, `evaluate`, `score_texts` (load saved adapter →
  score arbitrary texts), `pairwise_accuracy` (BT pairs CSV).
- `eval_utils.py` — `compute_metrics()` canonical AUC/F1/etc.
  implementation; `discover_dataset_evals()` finds `evals.py` next to
  the dataset for sliced metrics; `run_dataset_evals()` runs per-slice
  + summary metrics.
- `fsdp_configs/b200_fsdp.yaml`, `fsdp_configs/h200_fsdp.yaml` —
  `accelerate launch` configs for B200 and H200 nodes (matched FSDP
  wrap policy + mixed precision).
- `requirements.txt` — `torch transformers datasets peft accelerate
  bitsandbytes scikit-learn pandas optuna flash-attn`.
- `reward_model_lora_training_guide.md` — long-form training guide
  (covers QLoRA, FSDP, hyperparameter ranges).

## How to run

Single GPU, 8B base, pointwise:

```bash
python train_reward_model.py \
    --data_path datasets/peer-review/peer_review_modeling_dataset.csv.gz \
    --model_name meta-llama/Llama-3.1-8B \
    --epochs 3 --batch_size 8 --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 --lora_r 16 --lora_alpha 32 \
    --class_weight_auto \
    --output_dir runs/peer_review_sweep_llama8b/subset_1.0/trial_0
```

70B QLoRA on B200 with FSDP:

```bash
accelerate launch --config_file fsdp_configs/b200_fsdp.yaml \
    train_reward_model.py \
    --data_path datasets/press-releases/press_release_modeling_dataset.csv.gz \
    --model_name meta-llama/Llama-3.1-70B \
    --quantize --gradient-checkpointing \
    --epochs 3 --batch_size 2 --gradient_accumulation_steps 16 \
    --output_dir runs/press_release_sweep_llama-70b/...
```

Bradley-Terry variant:

```bash
python train_reward_model.py ... --bradley-terry
```

Data-scaling sweep slice (subset of train at 0.1, 0.2, ... 1.0):

```bash
python train_reward_model.py ... --train_subset_percentage 0.3
```

Optuna HP sweep:

```bash
python train_reward_model.py ... --use_optuna --optuna_trials 20
```

Score arbitrary texts with a saved adapter (importable):

```python
from train_reward_model import score_texts
probs = score_texts("runs/.../best_model", texts=["..."], max_length=1024)
```

## Dependencies

- Data: any CSV(.gz) with `text` and `judgement` columns. Canonical
  per-task paths in
  `/Users/spangher/.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/reference_v2_task_datasets.md`
  and
  `reference_clean_datasets_per_task.md`. Persistent on-disk split is
  reused if it exists; otherwise created with `random_state=42`.
- Optional `evals.py` next to the dataset for sliced eval
  (e.g. per-venue AUC for peer-review).
- Models: `meta-llama/Llama-3.1-8B` (default) or `Llama-3.1-70B`
  (needs `--quantize` + FSDP).
- GPU: 1× A100/B200/H200 for 8B; 2-4× B200/H200 with FSDP for 70B.
  Per `feedback_minimize_gpus`, prefer 1 GPU when feasible.
- `requirements.txt` deps; `flash-attn` only needed for 70B.

## Current state

Working / actively used. Established per-task saturation curves
(see `project_dense_model_sweeps.md`, 2026-04):

| task | Llama-8B saturation rows | plateau AUC |
|---|---:|---:|
| peer_review | ~30K | ~0.77 |
| press_release | ~36K | ~0.71 |
| code_review | ~57K (1.0 incomplete) | ~0.78 |
| creative_writing / litbench | not saturated at 70K | ~0.90 max |

Notes:
- Variance across trials is high; report median + max over 3-5 trials.
- Llama-70B at subset 0.1 matched 8B on press_release — no evidence
  larger base helps at tested sizes.
- Bradley-Terry pairwise underperformed pointwise on press_release 0.1.

## Outputs

`<output_dir>/` contains:
- `best_model/` — best checkpoint by eval_auc (LoRA adapter +
  tokenizer).
- `training_history.json` — per-eval-step metrics.
- `validation_metrics.csv` — flat CSV of the same.
- `training_run.log` — file-handler log.
- `split_metadata.json` (in the data split dir, not the run dir) —
  records the 80/10/10 split sizes and seeds.
- If Optuna: `optuna.db` + per-trial subdirs `trial_<N>/`.
- If dataset-specific evals: `dataset_eval_results.json` with overall
  + per-slice + summary metrics.

Canonical sk3 run roots:
`/lfs/skampere3/0/alexspan/norm-research/runs/<task>_sweep_llama8b/subset_<frac>/trial_<N>/`.

## Related

- `methods/autometrics/` — flat-feature LLM-judge baseline; dense is
  the upper bound it tries to approximate.
- `methods/articulation_star/` — STaR-trained articulator; the
  rationale-only judge's held-out accuracy is compared to dense AUC
  to measure how much of the dense signal is articulable.
- `methods/metric_tree/` — tree-structured prediction (Algorithm 2/3
  in `project_three_algorithms.md`).
- `methods/local_explanations/` — extract-then-cluster rationale
  pipeline whose downstream predictor is benchmarked against dense.
- Memory:
  - `project_dense_model_sweeps.md` — data-scaling curves per task.
  - `reference_norm_embed_pair_labels.md` — 434K v6 pair labels used
    by adjacent rubric-clustering work (not by this script directly).
  - `reference_sk3_queue_supervisor.md` — overnight sweep queue.
  - `reward_model_lora_training_guide.md` — long-form how-to.
