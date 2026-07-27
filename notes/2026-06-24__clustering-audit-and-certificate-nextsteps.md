# Peer-review L0 clustering audit + certificate-pipeline next steps
**Date:** 2026-06-24 · **Scope:** L0 metric-clustering audit (peer-review) → decision → R1/R2 + Ω-certificate roadmap

---

## 0. Decision (read this first)

**Ship the existing v6/Llama L0 labels for now.** v6 is precision-extreme (0.999 — essentially never wrongly merges two distinct criteria) with recall ~0.58 vs the true boundary (between the two arbiters). It is *moderately* undermerged, not catastrophically. We revisit after the independent Opus adjudication lands. **GLM-4.7 group-tuned is the better recall/precision tradeoff and the fallback** if we decide undermerging matters downstream.

---

## 1. What we set out to resolve

1. Is the existing v6/Llama L0 clustering **undermerged**? (standing suspicion: yes)
2. Which method is **best** — grouping (G) vs pairwise (P); baseline vs GEPA-tuned?
3. **Who is the arbiter?** v6 is circular as gold (it built the partition) → need an independent, family-neutral reference.

---

## 2. Methodology

- **Arbiters (independent gold):** GLM-5.2 (conservative) and Opus (liberal) — different family from the candidate (GLM-4.7) and from the existing partition (Llama).
- **Eval sample:** 3000-pair *spectrum* sample — ~150 pairs per 0.05 cosine-sim bin (OpenAI text-embedding-3-small), so the full sim range is covered.
- **Two pipelines, kept distinct:**
  - **Group (G):** GLM-4.7 groups 30 statements → union-find reconcile (min_votes gate). Prompt tunable. → partition.
  - **Pairwise (P):** GLM judges pairs same/different → (correlation) clustering. → partition.
- **Two accuracy notions, kept distinct:**
  - *Partition accuracy* (opus_rescore.py): partition → pairwise (same-cluster = same) vs arbiter.
  - *Pairwise accuracy* (pairwise_accuracy.py): direct pair label vs arbiter.

---

## 3. Findings

### 3a. Arbiter calibration gap — the central finding
- Opus calls **27%** of pairs SAME; GLM-5.2 calls **19%** SAME; binary agreement **0.836**.
- The disagreement is **concentrated at high similarity**: 62% of Opus's extra SAME-calls are at sim ≥ 0.7; 86% of GLM-5.2's extra SAME-calls are at sim ≥ 0.7. Only **19/359 (5%)** of Opus's extra SAME-calls are below sim 0.5 (the genuinely indefensible ones).
- **Conclusion:** Opus is a *liberal-but-legitimate* arbiter, **not** a disqualified/sloppy one. The two judges make **defensible but different calibration choices on the high-sim borderline.** The true boundary sits **between** them.
- *Correction logged:* an earlier stratified-20 eyeball falsely suggested "Opus was sloppy." The stratified sample over-weighted the low-sim tail (only 5% of disagreements live there). Retracted.

### 3b. Partition scores (full 3000 pairs unless noted)
| partition | recall vs Opus | recall vs GLM-5.2 | precision |
|---|---|---|---|
| **v6/Llama (existing)** | 0.469 | 0.707 | **0.999** |
| GLM-4.7 group-tuned + tfidf (mv=1) | 0.763 | 0.806 | 0.88 |
| GLM-4.7 group-tuned + openai (mv=1) | 0.763 | 0.677 | 0.75 |
| GLM-4.7 baseline group | 0.211 | 0.226 | 1.00 |

- v6's *true* recall ≈ between 0.47 and 0.71 → **~0.58**. Moderate undermerging; near-perfect precision.
- GLM-4.7 group-tuned: **robust** (0.76–0.81 recall against *both* arbiters) at a still-good 0.88 precision → best overall tradeoff.
- ⚠ GLM-4.7 partition numbers are on only **61** among-300-forms pairs (noisy). Need a larger eval before treating them as final.

### 3c. GEPA tuning outcomes
- **Grouping-prompt GEPA** (`gepa_cluster_prompt.py`): per-batch recall 0.544 → 0.982; reconciled recall lifted substantially (esp. with min_votes=1). **Worked.** Produces the group-tuned rule above.
- **Pairwise-prompt GEPA** (`pairwise_gepa.py`): baseline won (F1 0.745); every mutation reverted (over-merged). **Did NOT help** ⇒ "GLM-4.7 tuned pairwise" == "GLM-4.7 baseline pairwise."
- **Lesson:** GEPA improves *grouping* prompts (one model, one objective) but did not improve *pairwise* prompts toward a different-model arbiter. Not distillation.

### 3d. Reconcile coverage gate
- min_votes=2 starves recall (coverage gap). min_votes=1 is the operating point for the group pipeline. Offline sweep (`sweep_minvotes.py`) confirmed; no extra GLM cost.

---

## 4. Open items (not yet resolved)
1. **Independent Opus adjudication** of the 484 disputed pairs — brief written (`opus_adjudicate_disagreement.py`); not yet run. Will pin the boundary and tell us if careful-Opus drifts toward GLM-5.2 (→ original Opus pass sloppy) or holds (→ genuine calibration gap).
2. **Full pairwise matrix** (7-entry): GLM-4.7 spectrum labeling at 1900/3000 (PID 43696); ~30 min out. Partial matrix is low-sim-biased — do not read yet.
3. **corr_cluster G-vs-P head-to-head** (`corr_cluster.py`): built, not yet run end-to-end on real edges.

---

## 5. Ω≈30 extraction — did we land on good methods? **YES**

The certificate pipeline (`run_real_test.py` + `harvest_gepa_omega.py` + `orthogonalize.py` + `omega_certificate.py`) is mature:

- **Where Ω comes from:** harvest the **GEPA prompt-optimization lineage** — every accepted mutation's *added leaf criterion* (semantic diffs), dedup by normalized text. This is richer than pure "rephrasings": it's the criterion set the optimizer actually discovered.
- **The principled filter (the part that matters):** **behavioral orthogonalization** (`orthogonalize.py`) — score each candidate's per-item signal X_e and keep only units **not already explained** by Ω (Shannon-CMI / submodular-tail filter). This is what turns a bag of near-duplicate strings into an atomic, non-redundant Ω.
- **Scale dispatch** (`omega_certificate.py`):
  - K ≤ 15 → **exact** brute-force subset lattice (`small_omega_brute_force.py`): exact OPT_Ω, exact γ, detects non-monotonicity (PRUNE). R(C(S)) = I_TVD(M; M̂_S), no-anchor.
  - K > 15 → **large-Ω fallback** (`large_omega.py`): U2 per-instance additive upper bound + double_greedy (Buchbinder et al., 1/3-deterministic / ½-randomized for non-monotone f).
- **⇒ Ω≈30 routes to the large-Ω path** (U2 + double_greedy), **not** exact enumeration.

**Caveat:** Ω extraction is **coupled to a GEPA Phase-A run** — you grow Ω *from* the prompt-optimization lineage of the target model X. For R1 clusters you therefore need Phase A to run on each cluster. Ω is not a free-standing "generate 30 rephrasings" call; it's the discovered criterion set, behaviorally de-duplicated.

---

## 6. Next steps (the map)

### 6a. Finish L0 clustering
1. Run `corr_cluster.py` G-vs-P head-to-head on the same arbiter edges (task #38).
2. Land the Opus adjudication → pin the boundary → final v6-vs-GLM-4.7-tuned call.
3. Scale the chosen L0 method to **all 9 tasks**.
4. **R1:** cluster L0 representatives (same primitive). **R2:** cluster R1 reps.

### 6b. Certificate pipeline (the prompt-optimization core)
- **#29 Held-out recovery:** train/test item split; GEPA on train, certify R on test (guards prompt overfit).
- **#31 Channel-cleanliness certificate (validity layer):** does each metric measure its claimed aspect or a confound (length/topic/identity)?
- **#32 Grow Ω + measure missing-impact:** per metric, harvest Ω≈30, measure the unrecovered R residual (the "taste" tail).
- **#30 Prompt-transfer generalization:** disjoint GEPA/test split — does the optimized prompt transfer?

### 6c. Prompt scaffolding / optimization (the pivot — next)
- Return to the prompt-construction + GEPA-optimization design now that L0 is "good enough" on v6.

---

## 7. Artifacts
- Eval data: `outputs/analyses/spectrum_pairs.jsonl`, `spectrum_opus_labels.jsonl`, `spectrum_glm52_labels.jsonl`, `spectrum_glm47_labels.jsonl`
- Partitions: `outputs/analyses/glm_cluster_peer_tuned{,_openai}.json`, `glm_cluster_peer_pilot.json`, `structural_metrics/clusters_peer-review.json` (v6)
- Disagreements: `outputs/analyses/arbiter_disagreements.json` (484), eyeball set `eyeball_overmerge_candidates.md`
- Code: `opus_adjudicate_disagreement.py`, `opus_rescore.py`, `pairwise_accuracy.py`, `corr_cluster.py`, `sweep_minvotes.py`
