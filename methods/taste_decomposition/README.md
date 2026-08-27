# taste_decomposition

Tightening the taste residual Δ = T − (V+A) into named parts.

**Design spec (read first):** `notes/2026-08-05__taste-decomposition-design.md`
— §0 fixes the quantity ledger (`VA_lin`, `VA_nl`, `Δ_interact`, `Δ_beyond`, …),
§1 freezes the Layer-1 protocol, §4 lists the task matrix, §6 lists what must be
prereg-frozen after the pilot and before any confirmatory cell.

## Contents

| path | what |
|---|---|
| `layer1_stack.py` | Layer 1 — nonlinear (HistGradientBoosting) re-aggregation of an existing V/A score matrix on the task's own grouped OOF folds, plus the linear reproduction gate, seed spread, overfit gap and SHAP interaction screen. CPU only. |
| `results/peer_verdict_layer1.json` | pilot result, peer-review verdict cell |
| `results/peer_verdict_va_nl_oof_seed0.npy` | OOF predictions of the nonlinear V+A model (seed 0), for paired comparisons |

Layer 2 (`layer2_robustness.py`) and Layer 3 (`layer3_closure/`) are not written yet.

## Running Layer 1

```bash
python methods/taste_decomposition/layer1_stack.py --cell verdict --T 0.753
```

The script refuses to be meaningful unless the linear gate reproduces the cell's
published V / A / V+A numbers — check the `gate` block of the JSON before reading
anything else.

## Pilot status

Peer-review verdict (2026-08-05, exploratory): gate reproduces to machine
precision; `Δ_interact = −.002` (null), `Δ_beyond = +.065`. Write-up:
`notes/2026-08-05__layer1_peer_verdict_pilot.md`.
