# Subtask-pair same/different annotation (codability instrument)

Hand-grade (Sonnet) annotation of **peer-review subtask pairs** as SAME / RELATED / DIFFERENT,
to (a) measure subtask-level codability free of the kNN-candidate selection bias that caps the
rubric-level curve, and (b) seed a cross-encoder (CE) that recognizes same evaluation-aim across
surface wording.

## Files
- `subtask_pairs_peer_review_v1.jsonl` — 1000 labeled pairs. Fields: `pair_id, stratum, cos_label,
  A{subtask,breadth,n_rubrics,example_rubrics[]}, B{...}, label, confidence, reason`.
- `build_subtask_pairs.py` — samples 1000 pairs from the 3,813 peer-review `subtask_short` values in
  `notebooks/_explore_cache/rubrics.parquet`. Stratified by frozen MiniLM cosine of the subtask label+
  first rubric: 300 uniform-random (base-rate) + 280 high(≥0.55)/210 mid(0.35–0.55)/210 low(<0.35).
- `analyze_subtask_labels.py` — joins labels to strata; distribution, codability curve, low-surface cases.

## Annotation scheme (v6-aligned: SAME=2 / RELATED=1 / DIFFERENT=0)
- **SAME**: substantially the same evaluation aim — an evaluator would apply largely the same criteria.
- **RELATED**: same broad domain, partially overlapping criteria, distinct scope/object/angle.
- **DIFFERENT**: largely disjoint criteria.
Judged from the *example rubrics* (the real aim), not the terse label.

## Result (v1, 2026-06-15; workflow wnmoah253, 20 Sonnet agents)
Overall SAME 192 / RELATED 308 / DIFFERENT 500. Random-stratum base rate 0.7 / 16 / 83.
High quality: 771 high-confidence, 0 low-confidence.

## ⚠️ Codability interpretation RETRACTED (same day)
An earlier draft read this as evidence of *low* peer-review codability (`corr(cos_S,label)=0.82`,
"SAME floors at cos≈0.35"). That was wrong, for two reasons:
1. **No genuine surface axis.** `cos_S` was a MiniLM *semantic* embedding, not surface, so it and the
   70B judge are two deep-meaning estimators — agreement is expected, not a finding. A real lexical
   axis (token Jaccard) inverts it: partial corr(label, semantic | jaccard)=**0.758** vs
   partial corr(label, jaccard | semantic)=**0.093**. The judge tracks *meaning across different
   words*, not surface → peer-review is NOT surface-bound/low-codability.
2. **Not a recovery measurement.** Independently-authored rubrics → no shared source `s`, no decoder,
   nothing "recovered"; only a similarity verdict. `I(s; m_recovered)` and the DPI ceiling don't apply.

Codability needs a recovery loop with a *measured* object (functional/behavioral re-scoring of a fixed
item set, or paraphrase round-trip) — parked, not built.

## What this dataset IS good for
A legitimate **subtask-relatedness resource** — graded labels that carry real meaning-signal beyond
lexical overlap (the partial-corr result), suitable for a CE / cleaner subtask taxonomy. NOT a
codability measurement. The useful surface-orthogonal training cases: ~50 high-Jaccard DIFFERENT
(lexical twins, distinct aim) + the low-Jaccard SAME/RELATED pairs. Scale to ~3–5K if training a CE.
