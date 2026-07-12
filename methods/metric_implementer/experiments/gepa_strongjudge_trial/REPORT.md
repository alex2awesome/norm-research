# GEPA trial — strong reviser (Sonnet) + Llama-70B judge, UNCAPPED — 2026-06-21

**Question:** do MECHANIZE/DECOMPOSE dominate GEPA because the *judge is weak* (Llama-3.1-8B) and the
*reviser is non-frontier* (Llama-3.3-70B)? Test by upgrading both and removing the budget cap.

**Setup (all real, verified):**
- Reviser + reconstructor = **Claude Sonnet** subagent (≫ Llama-70B). Judge/executor = **Llama-3.3-70B via
  OpenRouter** (≫ 8B). **Uncapped** (no 600-token / 4-fewshot limit). 4 iterations.
- New, deliberately taste-heavy metric: **ELEGANCE/INSIGHT** ("reveals WHY via a minimal clever idea;
  penalize brute-force grinding even when correct"). 50 math.SE answers.
- 500 real Llama-70B judge calls (10 passes × 50), all logged with raw JSON. Artifacts: `rubric_v*.{txt,json}`,
  `scores_v*.jsonl`, `log.jsonl`.

## Operator sequence
`DECOMPOSE → FEWSHOT+ → DECOMPOSE → PRUNE+ANCHOR`

| iter | op | #criteria | flip | score hist 0/.25/.5/.75/1 |
|---|---|---|---|---|
| v0 seed | — | 1 | 0.14 | 4/1/1/19/25 |
| v1 | DECOMPOSE | 4 | 0.06 | 1/4/2/27/16 |
| v2 | FEWSHOT+ | 4 | 0.06 | 1/8/3/18/20 |
| v3 | DECOMPOSE | 6 | 0.10 | 1/5/4/18/22 |
| v4 | PRUNE+ANCHOR | 4 | 0.12 | 3/6/3/20/18 |

## Findings
1. **MECHANIZE vanished.** The weak prior runs were MECHANIZE-heavy (23/30 versions). The strong setup used
   **zero MECHANIZE**. So the MECHANIZE dominance WAS an artifact of the weak 8B judge (everything tagged
   JUDGE_LIMITATION → routed to MECHANIZE). Confirmed.
2. **But the winning operator was FEWSHOT+ (calibration), not "ambiguity ops" broadly.** The single most
   impactful move was adding worked examples (v2): it broke the seed's bimodality and pulled the inflated
   1.0-cluster down correctly. DECOMPOSE was still used twice — the bottleneck for a 70B judge on a
   qualitative dimension is **calibration** ("how to apply the check against real examples"), not knowing
   *what* to check. ANCHOR/FEWSHOT+ address that; more criteria do not.
3. **Bigger Ω HURT (counter to the "Ω should be very large" intuition).** Even uncapped + strong reviser,
   the rubric converged to **4 working criteria**; the union across all versions is **Ω=12**. v3's extra
   DECOMPOSE (→6 criteria) **regressed** — more abstract sub-criteria gave the weak judge more positive
   anchors → upward drift. v4 PRUNED back to 4. For a single metric + single-verdict judge, *fewer concrete
   criteria + calibrated examples* beat *more criteria*. (Consistent with PRUNE: the single verdict can't
   juggle many criteria.) "Very large Ω" must come from finer granularity / many seeds / pooling across
   metrics, NOT from elaborating one metric — and it pays off only with a non-single-verdict aggregator.
4. **Elegance SURVIVED** (the weak GEPA evolved it away). Stable correct high-scorers throughout: lower-
   triangular eigenvalue obs, generating-functions, Cantor diagonalization reframe, parity factorization.
   Good recovery: GAP-computer-code answer 1.0→0.0 (penalized as brute-force); bare-link 0.0 throughout.
5. **Judge pathologies (Llama-70B):** persistent upward bias (38/50 ≥0.75 even at best — math.SE answers are
   generally competent); worked-example *blindness* (id=2 I-E walkthrough correctly 0.25 in v2 but regressed
   to 0.75 in v3-4 — the judge generalizes the example as a style guide but loses it as the prompt grows);
   most unstable region = the 0.5/0.75 boundary.

## Takeaway
A stronger judge+reviser **does** diversify operators away from MECHANIZE — but toward **calibration**
(FEWSHOT+/ANCHOR), and it does **not** want a large criterion set: more criteria regressed. This sharpens the
research picture: the lever for articulability is calibrated application of a FEW concrete criteria, plus an
aggregator that isn't a single bottlenecked verdict — not a sprawling Ω.
