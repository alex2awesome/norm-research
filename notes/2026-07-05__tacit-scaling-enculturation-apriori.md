# 2026-07-05 — Tacit-knowledge scaling, cross-family enculturation, a-priori prediction

Three user directions (afternoon): (1) "tacit knowledge scaling laws — can we say anything
absolute from our relative positioning of models?", (2) "cross-model family distillation —
concepts that need defining A→B (B never taught it) but not A→A′ (smaller kin still has it)?",
(3) "can we summarize metrics a-priori in a way that predicts tacit knowledge required,
between tasks?" Tasks #21/#22/#23. Code: `methods/codability/name_sufficiency.py`
(subcommands scaling / family / prereg / apriori) + `methods/codability/grid_auc_report.py`.
Data: `notebooks/data/two_faces_20260702/{name_sufficiency_scaling, family_enculturation,
prereg_70b_name_sufficiency, apriori_tacitness_lodo}.json`.

## 0. Instrument first: the absolute-0.5 threshold conflated calibration with tacitness

`report.json`'s bal_acc thresholds orbit-averaged reader scores at an ABSOLUTE 0.5.
Qwen2.5-3B on math scored bal_acc == 0.5 on ALL 21 metrics (its P(yes) sits globally below
0.5 → all-negative predictions) while its 1.5B and 7B siblings cleared floors — yet its
rank signal is healthy (mean articulation-AUC 0.649, 21/21 measurable). **Calibration is an
instrument property, not tacitness**; cross-family comparisons need rank statistics.
`grid_auc_report.py` (numpy-only, mirrors report()'s masking/orbit-averaging) emits
threshold-free AUC + Spearman vs the executor's continuous M_i per (reader, gi, rung); ran
for all 9 Llama grids + math-Qwen on sk3 (CPU). AUC also reveals far more signal everywhere
(mean rung AUC: humor .83, grant .80, peer .80). All results below on the AUC scale
(executor-verdict recovery = cross-executor transmission — deliberately NOT the census
self_bits readout, which is the reader-internal decompression shape; both are kept in the
master table).

## 1. Name-sufficiency scaling (Direction 1) — the ladder is A→A′→A(self)

Definitions (per metric m, reader r): name-deficit d = AUC_definition − AUC_name;
measurable = max articulation-rung AUC ≥ 0.55; **name-sufficient = AUC_name ≥ 0.55 AND
d ≤ ε** (conjunction matters: near the floor def≈name≈chance makes d≤0 a coin flip — this
faked non-monotone survival in the first pass). S*(m) = smallest scale where NS holds and
persists up-ladder. 8B reader == executor ⇒ that rung of the ladder is SELF-recovery.

**Survival (fraction of measurable metrics still name-deficient, ε=0):**

| group | 1B | 3B | 8B(self) |
|---|---|---|---|
| TASTE | .63 (35) | **.33** (39) | .49 (39) |
| STRUCTURAL_CRAFT | .67 (144) | .61 (169) | .61 (171) |
| MECHANICAL | .71 (14) | .61 (18) | .72 (18) |
| expressive | .65 | .54 | .62 |
| institutional | .80 | .64 | .62 |
| formal-lexicon | **.35** | .38 | .41 |

- **Three-way dissociation**: taste names "come online" between 1B and 3B (.63→.33 —
  enculturated indices, cheap once the culture is absorbed); craft stays
  articulation-dependent at every scale (~.61 flat to 8B); mechanical names never suffice
  (the operational content lives in the definition — "copy-editing" the word doesn't carry
  the spec). Naive expectation (mechanical = most codified = most name-transmissible)
  INVERTS: codified ≠ lexicalized.
- **Lexicalization gradient replicates on the transmission scale** (def−name AUC at 3B):
  peer +.044 > pr +.031 > cr +.025 > humor +.020 ≈ grant +.020 > cw −.000 > math −.023 >
  legal −.051. Formal-lexicon domains invert; all institutional domains positive. (News 3B
  −.078 excluded from headlines: 3B reader ≈ chance on news — known data-quality domain.)
- Absolute-claim shape delivered: empirical name-transmissible fractions at each absolute
  scale + the S* ordering; NO continuous extrapolation (V-info non-Lipschitz guard).
- **Prereg frozen** (`prereg_70b_name_sufficiency.json`, sha256 62e4b3f0…): 92 metrics
  predicted to persist name-sufficient at 70B; 136 ranked by 8B deficit with the flip-set
  predicted to be a prefix of the ranking. Evaluate with a 70B reader pass after the ~Jul 10
  rescore frees GPUs. (Supersedes a same-day bal_acc-scale freeze, sha ccf4b806…, retired
  with the threshold-free readout — noted inside the artifact.)
- Caveats: per-scale survival cells at 1B are selection-noisy near the floor (persistence
  guard handles S*, not the per-scale table); 8B column is self-recovery (qualitatively
  different — the executor reading its own articulation); taste 8B uptick (.33→.49) not
  over-read.

## 2. Cross-family enculturation (Direction 2) — math panel (n=21)

Byte-identical rung messages; target = Llama-8B executor verdicts; readers Llama 1B/3B/8B
(kin) vs Qwen2.5 1.5B/3B/7B (stranger), size-matched tiers. Within-reader d = def−name is
capability-self-normalizing; **DiD = d_stranger − d_kin**.

| tier | mean DiD | n | p (sign-flip) |
|---|---|---|---|
| 1B | **+0.018** | 21 | **<1e-4** |
| 3B | +0.014 | 21 | .33 |
| 8B/7B | +0.011 | 21 | .13 |

Positive at every tier (stranger needs the definition more), significant at the smallest —
where enculturation should show most. Taxonomy (kin=Llama-3B vs stranger=Qwen-7B,
def-recovery conditioning separates missing-binding from missing-capability):
**17/21 universal-lexicalized** (math's formalized lexicon is family-independent — names
work for everyone), 2 craft-everywhere, and **2 A-only-lexicalized (the user-predicted
"B was never taught it" cell): "Elegance and beauty of proofs" (the bank's TASTE metric;
kin name BEATS its definition d=−.061 while the stranger needs the definition +.038) and
"Notation and terminology"**. Existence proof with the canonical tacit-aesthetic concept as
the example. Sign-flip treats metrics as independent but they share 300 probes —
probe-clustered bootstrap is the hardening step.

**CW panel landed (same day, n=46): the 1B-tier DiD REPLICATES (+.017, p=.0001)** — two
domains, same tier, same direction. But the taxonomy is BIDIRECTIONAL in the expressive
domain: 28 universal + **5 A-only** (pacing/rhythm [TASTE], flash-fiction compression,
opening hooks, core-conflict stakes, audience clarity) + **6 B-only** (macro plot structure,
POV control, genre conventions, sustained tension, setting-as-force [str d=−.137!],
pitch/query) + 2 craft-everywhere + 5 unmeasurable. So the sharper statement than the
original hypothesis: **formal lexicons are family-INVARIANT (math 2 A-only/0 B-only, 11%
family-specific); expressive lexicons are family-VARIANT and bidirectional (CW 5+6/41 = 27%)
— each family enculturated its own partial inventory: dialects of craft culture.** CW
8B-tier DiD flips sign (−.016, p=.03) but that tier's kin side is SELF-recovery (executor
reading its own generated definitions) — flagged artifact-prone, not "stranger knows CW
better". Individual cells sit near ε=.02 with se≈.03-.04: rates are robust, single-cell
identities are not. gemma-2 triangulation = next (A-only cells that are also gemma-deficient
⇒ Llama-specific enculturation; deficient only for Qwen ⇒ Qwen-specific gap).

## 3. A-priori prediction (Direction 3) — first pass is an instructive null

LODO (train 8 domains, predict the 9th), outcome = mean name-deficit over 1B+3B readers;
features from metric TEXT only. On the **bal_acc scale** tags+class predicted ρ=.35 /
AUC=.65 — **that was calibration structure, not tacitness**: on the AUC scale the same
models are null (tags+class ρ=−.07 n.s.) and concept-type tags ALONE significantly
ANTI-predict (ρ=−.19, p=.005) — the pooled type→tacitness relation reverses out-of-domain
(type effects are domain-contextual; Simpson structure). Metric-level tacitness is NOT yet
predictable a-priori on the honest scale. Scoped next steps: (a) evaluate WITHIN the
held-out domain (kill between-domain mean shifts); (b) probe split-half reliability ceiling
of per-metric deficits (effects ±.03 vs se≈.04 — may be reliability-limited); (c) richer
features: definition embeddings, zipf name-frequency, LM zero-shot predicted-articulability
baseline; (d) domain-LEVEL prediction (class → gradient sign, visibly real) tested on
wave-3 domains as genuine held-outs.

## 4. Wave-2 closure: grant census + grant×peer transport (#18 DONE)

Grant grid (probe-window repair) landed 12:58; census built (n=16;
isomorphism_census_grant.json). **Grant×peer transport: 5/5 label agreement, p_perm=.083**
— the first POWERED rigor-family test (mixed marginals: A-side 3 CRAFT + 2 MECH; B bank
7C/1T/5M), unlike legal×math's degenerate p=1.0. Includes the first two MECHANICAL
transports (copy-editing↔English-correctness, solicitation-scope↔venue-fit). Gap transport
ρ=.775 (n=4, n.s., accumulates). Pooled label transport now **41/46 across 4 pairs**
(CW×humor 25/30 p=.0017, news×PR 11/11 p=.0102, grant×peer 5/5 p=.083, legal×math 6/6
degenerate). `crosstask/grant_peer_isomorphism.json`.

## 5. In flight / next

- ~~CW Qwen panel~~ LANDED same day (see §2): 1B-tier DiD replicated; expressive lexicon
  family-variant + bidirectional. gemma-2 2b/9b = third family (downloads) → triangulate the
  11 CW family-specific cells.
- 70B reader pass on the two pole domains after Jul 10 → prereg evaluation.
- Notebook section for the three directions (after CW panel lands).
- Band per-metric join in name_sufficiency.py build_master is broken (band 0) — fix when
  band verdicts are needed as outcomes.

## 6. DiD decomposition (confounder audit, user-prompted — evening)

DiD = Δdef − Δname (Δ = stranger − kin, per tier). Decomposition from existing AUCs:

| tier | math DiD = Δdef − Δname | cw DiD = Δdef − Δname |
|---|---|---|
| 1B | +.018 = **+.030** − (+.012) | +.018 = **+.013** − (−.005) |
| 3B | +.014 = −.007 − (**−.021**) | +.002 = −.276 − (−.278) (Qwen-3B level-collapse on CW) |
| 8B | +.011 = −.045 − (−.056) | −.016 = −.043 − (−.028) (kin=SELF both) |

- **The significant 1B-tier DiD is Δdef-driven in BOTH domains** (stranger exploits definitions
  better — Qwen small models are stronger instruction-followers), NOT Δname-driven. The
  1B-tier DiD is therefore CONFOUND-LIABLE (articulation-exploitation ≠ enculturation) and is
  demoted from headline status.
- The enculturation-consistent (Δname-driven) component lives at the **3B tier on math**
  (Δname −.021, per-metric corr(DiD, −Δname)=+.59) — n.s. alone.
- Qwen2.5-3B collapses ~.28 AUC on CW at BOTH rungs (checkpoint-level outlier; the within-
  reader d nets it out but the tier is uninformative). Second Qwen-3B anomaly (after the
  calibration one) — treat that checkpoint as suspect generally.
- 8B tier: kin=self → Δname strongly negative = self-recovery inflation on the name side, as
  expected; excluded from cross-family claims.
- **The robust objects are the def-CONDITIONED taxonomy cells** (A-only already requires
  stranger def-AUC ≥ floor), not tier-mean DiDs. Hardening: re-derive cells under def-PARITY
  matching (|Δdef| ≤ .05) so name comparisons happen at matched articulation-exploitation.
