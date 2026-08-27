#!/usr/bin/env python3
"""Upsert the audited 60-metric family-scale structural result."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NB = ROOT / "notebooks/2026-07-02__metric-seam-certificates-and-overnight-report.ipynb"

MD = """## §25 · Family-scale structural reconstruction (2026-07-14)

The scaling unit is now a reusable relation family rather than a bespoke metric scorer. A frozen,
outcome-blind sample contains 60 technical metrics: 15 each from Code, Math, Patents, and full-article
Science, balanced across R1/R2/R3 generation rounds. Three independent Sonnet fleets decomposed only the
metric construct and description; they received no corpus, program, output, prior judgment, child rubric,
or held-out item.

All 60 metrics completed. Exact normalized string overlap was 0/60, demonstrating that lexical equality is
not a usable structural-stability instrument. A separate one-call blinded semantic aligner—opaque unit and
source IDs, one-to-one-per-fleet clusters, unmatched units allowed—produced an overall median metric-level
mean pairwise semantic Jaccard of **.603**, metric-bootstrap 95% CI **[.524,.667]**. Domain medians were Code
.587, Math .576, Patents .667, and full-article Science .524. R1/R2/R3 medians were .603/.598/.603 and are
descriptive only because those rounds are not a certified ancestry partition.

This is an articulability/decomposition-stability result, not code-based verifiability or behavioral
isomorphism. The semantic aligner is itself one unsupervised Sonnet pass per metric, not external ground
truth, and its reliability has not yet been replicated. Width is also censored: 148/180 decompositions hit
the five-relation maximum, so capture–recapture is not interpreted. The next stage is to induce recurring
families across metrics and run corpus base-rate probes before authoring or importing code units.
"""

CODE = """family_root = ROOT / "outputs/metric_seam_pilot/family_scale_v1"
family = json.load(open(family_root / "structural_readout.json"))
print("complete:", family["coverage"])
print("overall:", family["overall"])
display(pd.DataFrame(family["by_domain"]).T)
display(pd.DataFrame(family["by_hierarchy_round_descriptive"]).T)
print("instrument diagnostics:", family["instrument_diagnostics"])
display(pd.DataFrame(family["records"])[
    ["domain", "level", "construct", "mean_pairwise_semantic_jaccard", "unmatched_unit_count"]
].sort_values(["domain", "mean_pairwise_semantic_jaccard"]))
"""


def cell(kind, cid, source):
    value = {"cell_type": kind, "id": cid, "metadata": {}, "source": [line + "\n" for line in source.splitlines()]}
    if kind == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


nb = json.loads(NB.read_text())
replacements = {
    "seam-20260714-family-scale-md": cell("markdown", "seam-20260714-family-scale-md", MD),
    "seam-20260714-family-scale-code": cell("code", "seam-20260714-family-scale-code", CODE),
}
seen = set()
new = []
for existing in nb["cells"]:
    cid = existing.get("id")
    if cid in replacements:
        new.append(replacements[cid]); seen.add(cid)
    else:
        new.append(existing)
for cid, value in replacements.items():
    if cid not in seen:
        new.append(value)
nb["cells"] = new
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(NB)
