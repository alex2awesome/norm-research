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
run in the code-verification artifact: that result remains certificate-plane code
verifiability only. The additive `articulability_pipeline.py` now prepares a sealed,
2,400-request same-input counterpart. It hashes the prompt, allowlisted inputs, rendered
requests, exact model contract, and source instruments; supports resumable independent
API-result ingestion; rejects unbound and textually ungrounded witnesses; and computes a
descriptive prompt↔code witness comparison without treating either channel as ground truth.
The prepared, not-run bundle is under
`outputs/metric_seam_pilot/science_articulability_v1_prepared/`.

The bounded transport work remains additive: v1--v5 are preserved as failed or
instrument-development attempts. The current v6 OpenRouter bundle requested JSON mode and
`reasoning={"effort":"none"}` and executed only a five-request serial smoke; it did not launch
the 2,400-request batch and used no local GPU. An earlier v3 replay incorrectly described
lowercased, punctuation-stripped token containment as exact grounding; that artifact is
preserved but superseded. The v4 guard permits only whitespace-run folding for PDF/layout line
wraps and otherwise preserves case, punctuation, hyphenation, and Unicode literally.

Under v4, two of five responses passed binding, schema, and verbatim whitespace-canonical
grounding. Three were rejected: one paraphrased certificate evidence absent from the body, one
used a body-only sentence as an abstract claim, and one dehyphenated/paraphrased weaker evidence
instead of copying the body span. On the two valid papers, prompt/code status and
strong-certificate presence each agreed on 1/2. There were zero matched strong witnesses
(2 prompt, 0 code); neither valid paper had a weaker link in either channel. These are partial
descriptive isomorphism measurements, not a criterion-level result.

Provider telemetry also prevents calling the run uniformly reasoning-free: although reasoning
was explicitly requested off, one of five responses reported 12,426 reasoning tokens. Hidden
reasoning text was never retained. The prepared manifest remains immutable and records the
source hashes used at preparation; requests and raw results remain bound to it. Because the
validator was hardened after the calls, the canonical v4 additive replay records both the
historical preparation-time pipeline SHA and the exact final ingest/evaluation pipeline SHA.
It is not presented as an exact source replay of preparation. The bounded receipt and
accepted/rejected partition are:

- `outputs/metric_seam_pilot/science_articulability_v6_openrouter_reasoning_off_prepared/evaluation_literal_guard_v4.json`
- `outputs/metric_seam_pilot/science_articulability_v6_openrouter_reasoning_off_prepared/REPORT_LITERAL_GUARD_V4.md`
- `outputs/metric_seam_pilot/science_articulability_v6_openrouter_reasoning_off_prepared/normalized_results_literal_guard_v4.jsonl`
- `outputs/metric_seam_pilot/science_articulability_v6_openrouter_reasoning_off_prepared/rejected_results_literal_guard_v4.jsonl`

Run:

```bash
python -m pytest methods/metric_seam/science_claims_v2/test_science_claims_v2.py -q
python -m pytest methods/metric_seam/science_claims_v2/test_science_claims_corrected.py -q
python -m pytest methods/metric_seam/science_claims_v2/test_articulability_pipeline.py -q
python -m methods.metric_seam.science_claims_v2.evaluate_corrected
```
