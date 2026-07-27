# What actually gets decompressed? Concept-type analysis (2026-07-03)

**Question (user, 2026-07-03):** when expansion/definition-adding produces iso-performance, WHICH
concepts are being clarified — taste-based or mechanical? Parallel to the code-vs-prompt thread.

**Protocol.** All 106 grid metrics (46 CW + 60 humor) tagged by two independent Claude taggers on
a 3-level checkability scale — MECHANICAL (surface rule) / STRUCTURAL_CRAFT (objective-ish
compositional property) / TASTE (graded aesthetic judgment where competent judges legitimately
disagree) — plus a content-domain tag. Raw agreement 89/106 (84%), Cohen's κ = 0.63; all 17
disagreements manually adjudicated from the full rubrics (doubles as the spot-check; 15/17 were
craft↔taste boundary cases). Tags + provenance: `notebooks/data/two_faces_20260702/concept_tags.json`;
cross-tabs: `concept_crosstabs.json`.

**Composition finding:** the R3 banks contain essentially NO mechanical criteria (0–2/106 across
taggers; adjudicated final = 0). Distilled human rubric clusters live entirely on the craft–taste
axis; the planted controls are our only mechanical anchors. Final: CW 30 craft / 16 taste,
humor 46 craft / 14 taste.

## Result 1 — expansion cost by concept type (chain, 3B→70B family-top anchor, δ=0)

| cell | n | at name (L0) | expanded (L1–7) | censored | med. level if expanded |
|---|---|---|---|---|---|
| CW craft | 12 | 5 | 4 | 3 | 4.5 |
| CW taste | 6 | 3 | 3 | 0 | 1.0 |
| humor craft | 14 | 5 | 4 | 5 | 4.0 |
| humor taste | 6 | 3 | 1 | 2 | 1.0 |

Pooled 2×2 (cheap = match at L0–L1): **taste 75% cheap (9/12) vs craft 42% cheap (11/26)**,
Fisher p = 0.086 — *exploratory*: n=38 and the chain items were selected by grid behavior
(rescued/censored), not by concept type. Robust to dropping borderline tags (70% vs 38%).

## Result 2 — grid corroboration (all 106 metrics, better powered)

3B rubric−name gap (what full articulation adds over the bare name, anchored bal_acc): CW taste
**+0.057** vs CW craft **+0.019** (excluding borderline: **+0.095** vs **+0.012**); humor ≈ 0 for
both. Pooled MW z = 0.82 (n.s.). 70B self-bits from the bare NAME: taste 0.31 vs craft 0.20 (CW) —
taste names index more of the 70B's own judgment than craft names do.

## Result 3 — increment-type gains hold within both concept types

Boundary (+0.027 craft / +0.020 taste) and counterexample (+0.017 / +0.024) are the positive
increments in BOTH labels; definition is negative for craft (−0.029) and flat for taste (−0.006);
checklist ~0 for craft, negative for taste (−0.014). The contrastive-beats-definition finding is
not an artifact of one concept type.

## Descriptive reading (hypothesis, not verdict)

The naive mapping "taste = tacit, mechanical = articulable" does not describe these data. Within
human-articulated criterion banks:

- **Taste concepts behave like enculturated indices.** The name alone (or one definitional
  increment) gets a 3B to the 70B's read — the vocabulary ("purple prose", "surprising-yet-
  inevitable", "tonal balance") points into shared prior; telling more adds little, and
  compressing taste into imperative checklists mildly HURTS.
- **Craft concepts are where decompression is expensive.** They dominate the never-matchers and,
  when they do match, need deep expansion (median L≈4 — procedure/boundary/worked-example
  territory). The knowledge is transmissible but costly-to-say — multi-step compositional checks a
  small reader can't execute from a name.

Alternative explanations to keep live: (i) anchored yardstick — H_self(70B) is similar across
labels (0.87–0.92) so target-noise differences don't obviously drive it, but humor taste H_self is
lower (0.80); (ii) chain-item selection bias (fix in v2 below); (iii) small taste cells (12–16 per
domain).

## Design consequence for chain v2

Stratify chain-item selection by concept label (all taste items + matched craft sample, or full
coverage with the 3B-only reader), so the taste/craft contrast is powered rather than incidental.
Concept tags are now a standing covariate for every Face-2 analysis and for the new domains
(news-homepages / press-releases / math will add genuinely MECHANICAL-leaning criteria, filling
the empty cell of the checkability axis).

Related: `notes/2026-07-03__expansion-chain-v1-results.md` (v1 chain), theory §2.4 (family-top
anchor resolution), `[[project_isoperf_expansion_experiment]]`.
