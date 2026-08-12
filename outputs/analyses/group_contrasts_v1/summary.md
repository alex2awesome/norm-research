# Group contrasts v1 (2026-08-11): what can we say about each group of metrics?

Master table: 1,347 metrics x {task, %codable units, %taste units, 9-type (n=1,270),
truth-condition axis, OSL fit verdict, staircase satgroup (n=1,032), rung (n=1,032),
concreteness (audited instrument), humor channel winners (n=203), qwen exemplar content
(n=203), blind category (n=203)}. All contrasts descriptive; MW = Mann-Whitney.

## A. Codable vs rest (any MECHANICAL unit: 503 vs 844)
- Codability is TYPE-determined: EXTERNAL_BUNDLE 93% / MECHANICAL_CHECK 84% /
  EVIDENCE_RIGOR 57% codable vs HOLISTIC_TASTE 3% / COMMUNITY_TRANSFORM 7% /
  IDENTITY_PERSONA 18%. And TASK-determined (patents 86% ... humor 12%).
- Codable metrics: LOWER fitted ceilings (L med .768 vs .839, p=6e-4), heavily
  R4-listener-bound (34% vs 9%), rarely R1-name (7% vs 20%), less TASTE (0 vs 25 med).
- Codability does NOT predict scaling verdict (all verdict shares ~flat) — the seam and
  the scaling boundary are different boundaries (replicates the 4.1b null).

## B. Best-channel groups (humor bank, name/def/examples holdout)
- definition-best is the default (175/203).
- examples-best (n=24): 88% blind-category MECHANICS, 0% norm-boundary; enriched
  CRAFT_OPERATION (71%) + COMMUNITY_TRANSFORM (17%: meme culture, parody); REACHES 71%
  vs 49%; RISING depleted (12% vs 36%). E.g. Button/punchline landing, Meme culture,
  Surprise/misdirection variants, Parody craft.
- CONTRAST WITH THE CONTENT LEDGER: certified cross-receiver exemplar CONTENT (fun-mm at
  qwen) concentrates in NORM-BOUNDARY constructs — two different senses of "does well
  with examples": mechanics constructs WIN with examples because their definitions
  underperform (show-don't-tell devices); norm-boundary constructs carry unique
  example-content but definitions still win the arm comparison.

## C. Scaling groups
- RISING (884): norm-boundary 34% vs 5% (humor-cat subset), EVIDENCE_RIGOR-enriched,
  R2-statement-heavy, TASTE-depleted (0 vs 18 med, p=4e-4), examples arm WEAKER
  (ch_ex .864 vs .929, p=8e-6).
- REACHES (321): compact craft mechanics (68% mechanics), R1-name-enriched (35% vs 10%),
  the home of examples-best metrics.
- BOUNDED (65): beyond-text 26% vs 10%, IDENTITY_PERSONA 14% vs 2%, TASTE-enriched
  (25 vs 0, p=9e-4), L med .527; ALSO R1-name-enriched (35%) — name-carried conventions
  that cap (the name transmits the convention; the construct still isn't recoverable).
- Staircase rising (69): in-text 84%, R2-statement 75%, LOWER concreteness (p=.02).
- Staircase plateaued (362): EVIDENCE_RIGOR-enriched, HIGHER concreteness (p=.004).

## The smoking-gun test (LOO majority-in-cell prediction of OSL verdict)
base rate .696; task .696; type9 .702; task+type9 .694; +codable .692; +rung .724
(rung partially circular). Humor-only with blind category: base .517 -> .611;
norm-boundary -> RISING 23/30 (77%). VERDICT: content features are richly DESCRIPTIVE
of the groups but only weakly PREDICTIVE of rising-vs-reaches; the strong predictable
signature is (a) BOUNDED <- beyond-text/identity/taste and (b) norm-boundary -> RISING
(humor-scale, blind labels). No wording-level feature (concreteness, codability)
separates rising from reaches.

## Provenance answer (examples HTML near-duplicates)
Bank grain = R2 merged groups (humor: 285 groups; each merges 2-9 source clusters /
4-44 leaf rubrics mined from distinct web documents). The surprise variants are SIBLING
R2 groups under two R3 parents (Setup-misdirection-reframe mechanics: 8 R2 children;
Misdirection-and-reveal: 2). The 18 examples-winners collapse to ~13 R3 concepts; 6/18
are the misdirection family. Known-instrument context: mined banks are 54-68% degenerate
at R2 grain (A-bank audit); R2/R3 dialect census showed R2/R3 SATURATED vs open R1 tail.
