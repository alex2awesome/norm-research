# tacit_channels — the tacit-knowledge program

Program note (theory, hypotheses, prereg gates):
`notes/2026-07-21__adding-tacit-knowledge-installation-channels.md`
Refactor plan of record: `~/.claude/plans/recursive-swimming-unicorn.md` (approved 2026-07-22).

## Layout

| dir | contents |
|---|---|
| `_apparatus.py` | the ONLY bridge to the frozen parts-1–2 scorer (currently `methods/codability/experiments/`); stage-2 move flips one constant here |
| `isomorphism/` | placeholder — receives `methods/codability/experiments/` in the gated stage-2 move (see isomorphism/README.md) |
| `channels/frontier_probe/` | Phase 0: are articulation rescues recombination of known vocabulary or novel formalization? CPU/API only |
| `channels/distill/` | Channel B: soft-label distillation of the target's name-invoked policy into a small executor (LoRA) |
| `channels/eval/` | adapter-aware teacher-forced scoring + the 2-D exchange-rate tally |
| `channels/peer_review/` | §5.1 downstream-reward design: metric battery (DRAFT pending G1), reliability ceilings, residualizer w/ stop-rule |
| `channels/assay/` | §4 tacitness assay: elicit self-articulation from a trained executor |
| `channels/gepa_bridge/` | C(c1): GEPA fitness = downstream reconstruction ρ |
| `tests/` | CPU-safe unit tests |

## Non-negotiable disciplines

- **Reconstruction-only**: every training signal is the *target model's own* name-invoked
  judgment; no human labels anywhere (`feedback_reconstruction_only_no_labels`).
- **Never edit the frozen apparatus**: `methods/codability/experiments/` is hash-pinned by live
  frozen manifests. Adapter support lives in `metric_implementer/vllm_backend.py` (additive,
  no-op without `cfg.vllm_lora_path`) and in channels-side forks — never in frozen files.
- **Zero-adapter acceptance test** (`channels/eval/score_with_adapter.py --acceptance-test`):
  a no-adapter run through the new path must reproduce the frozen scorer's vectors at ρ > .999
  on a 200-item slice before ANY intervention result is reportable.
- **Training data**: open `tacit_breadth_search` partition only; stable-hash splits; low-N regime
  (the exchange rate is the estimand — saturation cloning is a failure mode, not a result).
- **Prereg gates**: G1 = user signs off the peer-review metric battery; G2 = predictions frozen
  before confirmatory runs; G3 = residual≈noise stop-rule in the residualizer.

## vLLM LoRA caveat

`methods/articulation_star/` is old — used only as a shape reference. The LoRARequest API is
verified against the installed sk2/sk3 vLLM at acceptance-test time; if LoRA × `prompt_logprobs`
is broken on our versions, fall back to `channels/distill/merge_adapter.py` (merge-and-score).
