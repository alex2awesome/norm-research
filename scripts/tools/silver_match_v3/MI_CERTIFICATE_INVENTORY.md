# Frozen MI-certificate inventory for v3 silver validation

These label-free per-metric certificates predate v3 final silver assignments
and are analysis-only. They may never tune retrieval, GEPA, verification,
rescue, thresholds, or banks. `silver_mi_validation_v3.py` validates the exact
bank join again at analysis time.

| Task | Remote frozen certificate | SHA-256 | Exact bank join | Note |
|---|---|---|---:|---|
| notice-and-comment | `analysis_inputs/mi_certificates/notice-and-comment.json` | `4dacd71da39a29a2a43971358d4706084ed471a228263b147c4fee2554a3fbaf` | 18/88 | Extracted immutably from the pre-existing combined curve table; lower-power task result |
| humor | `analysis_inputs/mi_certificates/humor.json` | `c18a765d68529a9986a02c41012d54f4672deac790898d27dac57010b1b014c3` | 285/285 | Full R2 certificate; primary 12-hour target |
| press-releases | `analysis_inputs/mi_certificates/press-releases.json` | `254ad20bd97603ae6eb3ad819ef2760c6fb5f6bdd2ede47bf194620b4897c56a` | 221/221 | Matcher verifier currently fails dev gate; certificate readiness does not authorize analysis |
| math-stackexchange | `analysis_inputs/mi_certificates/math-stackexchange.json` | `46ac755e28224adb67d644c75a22b542ea2af20d378a822339cfffd50113217b` | 141/141 | Matcher currently requires fresh boundary labels |
| peer-review | `analysis_inputs/mi_certificates/peer-review.json` | `4214e334e39c5bf3fa9baa9535c6168de0e6dce4ab73c926e79d94d645a051f7` | 88/88 | Full production matching deferred behind smaller tasks |
| code-review | `analysis_inputs/mi_certificates/code-review.json` | `b3ce76b25cadbee5a98e677d2876423d03d806e59b72cbed4b3fd990112a0b6a` | 133/133 | Full production matching deferred due 611,785-row size |
| creative-writing | `analysis_inputs/mi_certificates/creative-writing.json` | `8ff7c1088309a0183d3a393d2610155df6c3f9d99926072d0247abb03ca0bea6` | 368/371 | Three current-bank leaves lack a certificate; report 99.19% coverage |

The remote root is
`/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful/`.
No complete compatible certificate was found for Legal.
