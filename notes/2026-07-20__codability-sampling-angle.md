# Codability as a sampling problem — paper notes

*2026-07-20. User-approved framing: "the ultimate codability story needs a sampling angle."
Scripts: `methods/codability/lexicon/agreement_vs_missing_mass.py`,
`methods/codability/lexicon/codability_sampling_model.py`. Results:
`outputs/lexicon/agreement_vs_missing_mass_20260720.json`,
`outputs/lexicon/codability_sampling_20260720.json`.*

## E1 — Framing unification (agreement ≡ name-space coverage at shallow sampling)

At the census's sampling depth (median 4 namings per construct), Brown–Lenneberg naming
agreement (modal share) and Good-Turing missing mass (f1/N over the naming distribution) are
**determinate re-expressions of each other**: partial Spearman given logN = −.89…−.92 in all
four domains, and given (N, agreement) the missing mass has 0–3% residual variance (humor:
0/103 concepts ambiguous). Consequences for the paper:

1. Do **not** present missing mass as an "error source" on codability — it adds nothing at
   this grain (it IS the same statistic). We verified this before deciding, not after.
2. Do present the identity as a bridge between two literatures: psycholinguistic codability
   and ecological coverage estimation collapse into one measurement at shallow N. "These
   constructs have low codability" and "these naming distributions are far from saturated"
   are one claim. Low codability = an **unsaturated naming process**, which motivates
   modeling the process (E3) rather than reporting point agreement.

## E2 — Three-way novelty decomposition (what happens when the next author speaks)

Pooled Turing estimators over author-named records (head_term present), concept grain:

| task | n named | P(novel concept) | P(known concept, novel name) | P(known concept, known name) |
|---|---|---|---|---|
| humor | 1,471 | .419 | .433 | **.147** |
| creative-writing | 1,748 | .368 | .482 | **.150** |
| news-homepages | 990 | .342 | .442 | **.215** |
| math-stackexchange | 1,013 | .508 | .343 | **.149** |

Headline: **conventionalized reuse is ~15–21% everywhere.** When an author articulates and
names an evaluation criterion, only about one time in six or seven do they use an existing
name for an existing concept. The evaluative register is spoken almost entirely in fresh
words — most of the "new species" mass is *lexical*, not conceptual. (Mirror-collapsed arm
in the JSON; direction unchanged.)

## E3 — The naming process is Pitman-Yor, not Dirichlet (and codability gets a size-free number)

Per-concept naming partitions fit by shared-parameter EPPF (concepts with ≥2 namings):

| task | PY d | PY θ | LRT PY vs DP | coincidence (1−d)/(1+θ) | 95% CI |
|---|---|---|---|---|---|
| humor | .78 | .69 | 119 | **.128** | [.092, .164] |
| creative-writing | .89 | −.30 | 308 | **.158** | [.125, .192] |
| news-homepages | .84 | −.12 | 274 | **.185** | [.132, .236] |
| math-stackexchange | .80 | .03 | 95 | **.195** | [.138, .254] |

- Discounts d ≈ .8–.9: naming distributions are **heavy-tailed power laws** — a Dirichlet
  process is decisively rejected in every domain (LRT 95–308 on 1 df). The lexicon never
  stops minting names; there is no finite name inventory being sampled.
- The **coincidence parameter** — asymptotic P(two independent authors use the same name for
  the same construct) — is the size-free codability number the raw agreement statistic was
  trying to be. Ordering: humor .128 < CW .158 < news .185 < math .195, matching the census
  conventionalization ordering (math most conventionalized) but now free of the size confound
  that killed the earlier headline.
- Posterior-predictive checks pass: simulated mean #distinct-names and singleton fractions
  track observed within 2–6% and 0.3–3.5pp respectively.
- Same CRP/PY machinery as the corpus-expansion instrument — the Chinese restaurant is both
  the field procedure (seating new criteria) and the measurement model (naming diversity).

## E5 — Held-out prediction: the fair in-population GT point test PASSES

Stable-hash 20% doc holdout, prequential scoring.

**Concept level** (answers "is Heaps/GT predicting our new-item rate correctly?" —
in-population, unlike the CRP waves): Turing f1/N computed on train vs realized new-concept
rate in test: humor .393 pred / .387 actual; CW .396/.366; news .336/.357; math .446/.465.
**Within 1–3 points in all four domains.** The GT machinery is calibrated; the CRP waves'
excess over census GT is register shift, not estimator failure.

**Name level**: PY predictive (θ+Kd)/(θ+N) vs plug-in singleton fraction: model wins Brier in
all four (.17–.22 vs .21–.28) and crushes log-loss (0.53–0.64 vs 2.1–3.0); predicted new-name
rates within 1–7pp of actual (humor: .764 pred vs .760 actual).

## E4 — TASTE-vs-CRAFT: EXCLUDED from the confirmatory suite

User challenge (2026-07-20) sustained. Status and what it would take to run it properly:

- **The distinction**: per-R1-construct type codes (`outputs/lexicon/codability/types_*.jsonl`,
  humor+CW only): TASTE = hedonic/response-terminating constructs (wit, charm, funniness);
  CRAFT = artifact-checkable execution constructs (structure, pacing, mechanics). Codes were
  LLM-assigned from construct names/glosses in the 2026-07-07 leg.
- **The measurement would be**: separate PY fits per type, compare coincidence, construct-label
  permutation null.
- **The two-faces prediction would be**: TASTE > CRAFT coincidence (taste names are index
  words pointing at shared response; craft gets decompressed novel phrasing). The raw-agreement
  version of this claim was already killed once as a size artifact.
- **Why excluded**: (a) type codes were assigned from text that includes the construct's NAME
  — the label is not blind to the naming behavior being measured; (b) coverage is 2/4 domains
  with many "?" codes; (c) no surviving preregistered direction — re-running a killed headline
  on a new estimator without prereg is a forking path.
- **To resurrect (if the taste/craft axis earns its place in the paper)**: (1) preregister the
  direction and the single test (PY coincidence gap, permutation null, mirror-guarded);
  (2) re-code types BLIND — definitions only, names redacted, two model families + adjudication;
  (3) then run once.

## Caveats that ride along

Conditional-on-pool (census sources); author-articulated criteria only (unnamed ≠ tacit);
GLM-4.7 extraction instrument (certified via 8/8 anchors); junk-source and mirror caveats
inherit from the census; concept grain = Jul-6 repaired partition.
