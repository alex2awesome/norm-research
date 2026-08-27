# fsdp_configs

## Purpose

Reference `accelerate` + PyTorch FSDP YAML configs for sharding 70B-class
LLaMA checkpoints across 2 GPUs during reward-model / LoRA training. Used by
`methods/train_reward_model.py` (and anything else launched through
`accelerate launch --config_file ...`). Originally written for the
QLoRA + LoRA reward-model recipe documented in
`../reward_model_lora_training_guide.md`.

## What's here

| File | GPU target | Memory / GPU | Distinguishing knobs |
|---|---|---|---|
| `h200_fsdp.yaml` | 2× H200 | ~141 GB | aggressive activation checkpointing, `fsdp_forward_prefetch: false`, `fsdp_activation_prefetch: false`, `fsdp_cpu_ram_efficient_loading: true`, `fsdp_min_num_params: 100_000_000` |
| `b200_fsdp.yaml` | 2× B200 | ~192 GB | looser wrapping (`fsdp_min_num_params: 75_000_000`), `fsdp_forward_prefetch: true`, `fsdp_activation_prefetch: true`, `fsdp_cpu_ram_efficient_loading: false` |

Common to both: `FULL_SHARD`, `TRANSFORMER_BASED_WRAP` on `LlamaDecoderLayer`,
bf16 mixed precision, `fsdp_use_orig_params: true`,
`fsdp_activation_checkpointing: true`, `fsdp_limit_all_gathers: true`,
`fsdp_gradient_clipping: 1.0`, `num_processes: 2`.

## When to use which

- **H200 (141 GB):** memory is tight for a 70B QLoRA run. The H200 config
  keeps activation prefetch off and turns on `cpu_ram_efficient_loading` so
  the shards stage through CPU on init. Pick this when you're staying under
  ~120 GB/GPU.
- **B200 (192 GB):** more headroom and more memory bandwidth. The B200
  config trades the safety knobs for prefetch (forward + activation) and a
  smaller `min_num_params` threshold so more sub-modules get wrapped — meant
  to take advantage of the extra bandwidth.

If you move to a different machine class, copy the closer config and tweak
`fsdp_min_num_params`, the prefetch flags, and `num_processes` rather than
starting from scratch.

## How to use

```bash
accelerate launch \
  --config_file methods/fsdp_configs/h200_fsdp.yaml \
  methods/train_reward_model.py \
  --data_path path/to/dataset.csv \
  --model_name meta-llama/Llama-3.1-70B \
  --quantize \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir runs/l70b_h200
```

Swap in `b200_fsdp.yaml` on B200 nodes. See
`../reward_model_lora_training_guide.md` (§"FSDP-ready Accelerate configs")
for the full QLoRA recipe these configs were tuned against.

## Current state

Stable reference configs; checked in 2026-02-17. Not run in CI — verify your
sequence length / batch size against actual GPU headroom.

## Related

- `methods/train_reward_model.py` — the primary consumer.
- `methods/reward_model_lora_training_guide.md` — the full training recipe.
- `methods/dense/reward_model_lora_training_guide.md` — duplicate of the
  guide kept alongside the dense-model code.
