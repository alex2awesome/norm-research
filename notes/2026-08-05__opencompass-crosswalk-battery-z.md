# OpenCompass (CompassRank) crosswalk vs our battery-z (2026-08-05)

Data: rank.opencompass.org.cn closed-set LLM leaderboard, all 13 quarterly editions 24-02→26-04,
pulled via their CDN JSONs (gateway /gw/opencompass-be, manifest listRankTableAvailableMonths).
Cached: outputs/analyses/opencompass_crosswalk_20260805/ (242 distinct models). Dimensions per
edition: Average/Language/Knowledge/Reasoning/Math/Code/Agent.

## Crosswalk (our 14 local + frontier executors)

EXACT checkpoint matches (6, co-present in editions 24-09/24-11/25-01): llama3b=Llama3.2-3B-
Instruct, llama8b=Llama3.1-8B-Instruct, qwen25-7b, qwen25-72b, gemma2-9b=Gemma-2-9B-it,
gemma2-27b=Gemma-2-27B-it. NEAR (2): llama70b (ours 3.3 vs OC 3.1-70B), mistral-24b (ours
Small-24B-2501 vs OC Small-22B-2409). MISSING locals (6): llama1b, qwen25-3b/14b/32b, phi4,
mistral7b(v-mismatch; ours gate-excluded anyway). Frontier: GLM-4.5-Air/4.5 (25-07), GLM-4.6
(25-10), GLM-4.7 (26-01), Kimi-K2.5 (26-01), Qwen3-235B-A22B-Instruct-2507 (25-07, non-thinking
= matches our reasoning-off usage), DeepSeek-V3/V3-0324; GLM-5.2 absent (OC has 5.1, 26-04);
qwen3-32b/next-80b only as (Thinking) = mode mismatch; hermes-405b absent (Nous, not base Llama).

## Correlations with battery-z (within-edition; z = logit battery AUC, our serving stack)

| edition | set | Average r/ρ | best dim | Reasoning r/ρ |
|---|---|---|---|---|
| 24-09 | exact n=6 | +.66/+.54 | Knowledge +.77/+.77 | +.60/+.43 |
| 24-11 | exact n=6 | +.67/+.54 | Language +.71 | +.55/+.37 |
| 25-01 | exact n=6 | +.64/+.54 | Language +.69 | +.66/+.54 |
| 24-09 | exact MINUS gemma2-27b n=5 | **+.87/+.90** | | |

**The entire disagreement is gemma2-27b** — our battery scores it BELOW gemma2-9b (z 1.102 vs
2.043, the inversion that got the gemma2 ladder excluded from our fits), while OC ranks 27b above
9b in every edition. OC therefore sides with the size prior and suggests the inversion is our
battery's artifact for that family (consistent with gemma2-27b's recorded form-fragility). With
that one executor removed, leaderboard-z and battery-z agree at r≈.87/ρ≈.90 on n=5.

## Caveats that gate any use as a z-axis (tell the friends)

1. **Cross-edition scores are NOT comparable** — same checkpoint drops as the closed set is
   re-hardened (Qwen2.5-72B 70.3→57.3→49.9; Llama3.1-8B 44.0→33.8→27.2). Any axis pooled across
   editions mixes instrument versions. Consequence for us: the GLM rungs sit in FOUR different
   editions (4.5:25-07, 4.6:25-10, 4.7:26-01, 5.1:26-04) → OC cannot supply the single frontier
   axis we hoped for without noisy cross-edition bridging (bridge models disagree in direction:
   Qwen3-235B-Thinking 63.8→68.9→64.9).
2. Frontier entries are mostly (Thinking) variants; our probes run reasoning-off.
3. n=6 exact locals only; the small rungs that anchor our scaling fits are absent.

Verdict: useful as an external VALIDATION of battery-z (moderate agreement, one localized
artifact identified) and as a within-edition sensitivity axis; NOT usable as a replacement
x-axis for the local ladder (coverage) or for cross-edition frontier ladders (instrument drift).
