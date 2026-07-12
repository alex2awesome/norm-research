# Science claim verifier v2 (audit-corrected current: v2.2)

This additive lane preserves the frozen `programs_peer_review/cv1_*`, `cv2_*`, and
`cv3_*` artifacts. It treats their abstract-claim → full-paper-evidence decomposition as
a retrospectively seeded pipeline decision (`pipeline_status=selected`) and evolves it
into a fully executable, auditable verifier.

The current verifier uses rule-aware sentence segmentation, complete-token normalized quantities
and units, local entity binding for ambiguous bare counts,
document-local BM25, exact bipartite claim/evidence matching, comparison entity roles and
direction, and explicit support/contradiction/abstention certificates. It never reads the
dataset's `y` field and makes no supervised, external-ground-truth, or original-discovery
claim.

`core.py` and `evaluate.py` remain frozen with the audit-reproduced historical 171-certificate
result. The additive `core_corrected.py` / `evaluate_corrected.py` path is current: it fixes the
quantity-prefix, index, superscript, named-version, and small-count entity-collision defects. Its
versioned output reports 136 strong relation certificates across 126 papers, with 435 weaker
evidence links across 382 papers. See
`outputs/metric_seam_pilot/science_claims_v2_corrected_v2/REPORT.md`.

The output has two strength tiers. Exact normalized numeric matches and entity/baseline/
direction-checked comparisons are strong `relation_certificate`s. Empirical, theoretical,
and qualitative lexical-plus-artifact matches are weaker `evidence_link`s: they locate
potential evidence but are not reported as semantic support.

`articulability_prompt.json` freezes a future prompt/LLM counterpart over exactly the
same `paper_id + abstract + body` inputs and certificate/abstention semantics. It is not
run here: the present result is certificate-plane code verifiability only.

Run:

```bash
python -m pytest methods/metric_seam/science_claims_v2/test_science_claims_v2.py -q
python -m pytest methods/metric_seam/science_claims_v2/test_science_claims_corrected.py -q
python -m methods.metric_seam.science_claims_v2.evaluate_corrected
```
