# Real-corpus scaling shapes (search1 campaign) — 2026-06-18

First analysis of the **existing** real metric-implementer corpus (no new GPU). Source:
`sk3:outputs/metric_implementer_scale/search1` (registry + scorecards) and `.../longtable`
(per-tier 5-pass scoring). All GEPA-optimized; judge = Llama-3.1-8B for the *optimization*,
re-scored across 6 tiers for the *measurement*.

## Corpus inventory (real)
- **988 GEPA prompt versions**, 7 tasks (law 336, humor 112, news-homepages 112, math 110,
  n&c 109, patents 106, peer-review 103), 56 metric-task groups, all with scorecards.
- Longtable: **2.99M rows** = 988 versions × 56 metrics × **6 judge tiers** × 60 items × 5 passes.
- Tiers: Llama-3.2-3B, Qwen2.5-3B, Qwen2.5-7B, Qwen3-8B, Llama-3.1-8B, Mixtral-8x7B.

## Finding 1 — prompt features do NOT predict recovery (E, N fixed)
Stage-0 feature analysis on the 988 prompts (within-metric, rank-normalized, apples-to-apples
vs `recon_behavioral`): **every prompt-content feature has |pooled_rank_rho| < 0.09.** The only
positive signals are *process* variables — `lineage_depth` (0.155) and `optimizer_round`
(0.112) — and both are ~half-explained by the GEPA operator (operator_eta2 ≈ 0.57).
- Features vary widely (instruction_tokens 17–499, char_count 97–2540), so this is NOT lack of
  variation. At fixed judge tier and fixed N, **prompt surface = a flat axis.**
- Deliberate axes are unexercised here: `data_budget_n` CONSTANT (60), `n_fewshots` CONSTANT (0),
  `token_cap` only 2 levels. So N and K were never swept in search1.
- Caveat: the population is **all-GEPA**, so "no feature predicts recovery" may be GEPA-specific;
  needs optimizer diversity to generalize.
- Read: evidence *for* the positioning (lead with E, demote L/M); the scaling lives in E and N.

## Finding 2 — E-axis recovery curve (the lead scaling law)
Consistency-channel transmission I_V (5-pass, binary: `H(p̄) − mean_i H(p_i)`, normalized by
`H(p̄)`), per (tier, metric, version); frontier = median-over-metrics of max-over-versions.

| tier (≈ capability →) | frontier I_V | median I_V | mean bits | % degenerate |
|---|---|---|---|---|
| Llama-3.2-3B | 0.654 | 0.408 | 0.225 | 19% |
| Qwen2.5-3B | 0.627 | 0.000 | 0.010 | 86% |
| Qwen2.5-7B | 0.893 | 0.556 | 0.318 | 31% |
| Qwen3-8B | 0.897 | 0.627 | 0.370 | 22% |
| Llama-3.1-8B | 0.706 | 0.510 | 0.252 | 19% |
| Mixtral-8x7B | 1.000 | 0.895 | 0.378 | 15% |

- **Broadly monotone in capability** (7B→8B→Mixtral) — Xu-monotonicity prediction holds.
- **Degenerate floor**: Qwen2.5-3B collapses to constant scores on 86% of metrics → ~0 bits.
  Below some capability the judge cannot transmit at all.
- **Capability ≠ param count**: Llama-3.1-8B (0.51) < Qwen2.5-7B (0.56); family matters.
  → the E-axis x-coordinate should be a measured capability, not parameter count.

## Gaps → what new generation would add
- **N axis (novel)**: never swept (N=60 const). Need GEPA runs varying data_budget.
- **K axis**: never swept (n_fewshots=0 const).
- **Optimizer diversity**: corpus is all-GEPA; feature analysis can't generalize across
  mechanisms without EvoPrompt/ProTeGi/APE/etc.

## Phase 2 — real multi-optimizer N×K corpus (launched 2026-06-18)
New optimizers implemented (`optimizers.py`): EvoPrompt (population GA), ProTeGi (textual-gradient
beam), APE/OPRO (induction) — distinct mechanisms, same unsupervised `fidelity_scalar` objective,
registry-tagged by `optimizer`. Validated offline on the planted `is_scary` judge (all 4 accept,
$0); driver `sweep_nk.py` validated on real `law`/`humor` corpora with `--fake` (0 failures).

First real run launched on sk3 GPU 6 (PID 2646465, resumable, `HOME=/lfs`):
`Llama-3.1-8B judge · humor · 2 metrics · N{10,30,60} × K{0,2} × {gepa,evoprompt,protegi,ape}` = 48 cells.
Verified live: real fidelity improvements (GEPA 0.325→0.382, EvoPrompt 0.313→0.464; real cf/recon/rho).
Output: `outputs/metric_implementer_scale/nksweep/`. Next: N- and K-axis recovery curves per optimizer.

## Repro (zero GPU, on sk3)
```
python -m methods.metric_implementer.analyze_features outputs/metric_implementer_scale/search1 --outcome=recon_behavioral
# E-axis I_V: group longtable sampled_*.parquet by (judge_model, metric_id, version_id), 5-pass binary I_V
```
