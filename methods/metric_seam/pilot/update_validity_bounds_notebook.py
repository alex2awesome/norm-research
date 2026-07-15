#!/usr/bin/env python3
"""Upsert the 2026-07-14 validity/bounds result cells into the seam notebook."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
NB=ROOT/"notebooks/2026-07-02__metric-seam-certificates-and-overnight-report.ipynb"

MD="""## §24 · Validity bounds and occasion-sensitive verification (2026-07-14)

**Retraction, with both defects stated.** The old Math a12 headline is not an adjudication result. The
pair-only Sonnet arm never received the document, so applicability κ=.445 asked a context-free model to
guess whether an equation was an occasion for the norm. Conditional polarity κ=1.0 then measured two
symbolic-identity checkers agreeing on symbolic identity. Independently, the construct treated definitions,
hypotheses, constraints, and equations-to-solve as rigor violations. Those κ values remain provenance for
the context-free control and must not be quoted as validity or whole-metric isomorphism.

The canonical G2 controls make the distinction executable: the frozen SymPy verifier detects both true
algebra errors but fires on all four construct-satisfied proxy traps. Its algebra capability survives; its
old occasion selector fails.

**a34 structural result.** Exact Shapley enumerates all 2,048 coalitions and survives independent
reproduction. Report contribution by operation class—not node count. The former 2.338-bit DAG-cut
certificate is retracted: raw `norm` text is item-injective and belongs to every minimal cut, so every
plugin MI equals H(target). The cut enumeration survives only as a graph description.

Three blind authoring fleets proposed 27 raw a34 nodes. One unsupervised reconciliation produced 13
semantic node types; bias-corrected incidence Chao2 estimates 13.8 total (94.2% estimated coverage). This
restores a width estimator at the node-type level, but K=3 is small and the estimate says nothing about
construct validity or performance.

**Tie-robust correction.** The a34 matched-information C>B claim is unresolved. Although Spearman gives
C−B=.079, 54/98 target values are tied at the ceiling. The primary pairwise-concordance difference is .028
with audit 95% CI [−.012,.070]. A larger, representation-matched, two-pass B arm is required.

**Code-review correction.** The historical full-source median ρ=.149 is not a codability estimate: most
code targets are near constants on merged post-review PRs. It is retained as provenance only. New code
families must use AST/dataflow relations supported by diffs and disclose tie/mode structure before prompt
calls.
"""

CODE="""from pathlib import Path
import json

vb_root = ROOT / "outputs/metric_seam_pilot"
g2 = json.load(open(vb_root / "hierarchy_r123/results/math_a12_g2_validity_v1/readout.json"))
a34 = json.load(open(vb_root / "battery/effort_ladder/ws4/patents_pa__a34/readouts.json"))
shapley = a34["exact_shapley_op_class_decomposition"]
cuts = a34["dag_cut_information_certificate"]
ceiling = json.load(open(vb_root / "hierarchy_r123/results/code_review_glm52_ceiling_v1/readout.json"))
inversion = json.load(open(vb_root / "hierarchy_r123/results/pipeline_inversion_replay_v1/readout.json"))

print("a12 G2:", g2["g2_pass"], g2["negative_proxy_traps"], g2["positive_true_violations"])
print("a34 exact coalitions:", shapley["coalition_count"], "efficiency residual:", shapley["efficiency_residual"])
display(pd.DataFrame(shapley["node_values"])[["node","op_class","phi","applies_rate","per_node_gate"]])
display(pd.DataFrame.from_dict(shapley["op_class_mass"], orient="index"))
print("DAG minimal cuts:", cuts["n_inclusion_minimal_cuts"], "information-bound reading: RETRACTED (raw text makes cuts item-injective)")
print("full-source execution median rho / CI:", ceiling["aggregate"]["median_raw_rho"], ceiling["aggregate"]["ci95"])
display(pd.DataFrame(inversion["rows"])[["task","unit","n_occurs","n","occurrence_rate","decision"]])

context_path = vb_root / "hierarchy_r123/results/math_a12_context_train_v1/readout.json"
if context_path.exists():
    context = json.load(open(context_path))
    print("a12 context roles:", context["role_distribution"])
    print("a12 individuation gap:", context["role_conditioned_individuation_gap"])
    print("a12 residual polarity:", context["joint_asserted_identity_polarity_agreement"])
matched_path = vb_root / "hierarchy_r123/results/patents_a34_matched_info_v1/readout.json"
if matched_path.exists():
    matched = json.load(open(matched_path))
    print("a34 matched information Spearman sensitivity:", matched["spearman"])
    print("primary tie-robust result: unresolved; c-index delta=.028, audit CI [-.012,.070]")
capture_path = vb_root / "hierarchy_r123/results/patents_a34_capture_recapture_k3_v1/readout.json"
if capture_path.exists():
    capture = json.load(open(capture_path))["estimate"]
    print("K=3 node-type capture-recapture:", capture["observed_unique_types"],
          capture["estimated_total_types"], capture["estimated_coverage"])
"""

def cell(kind,cid,source):
    base={"cell_type":kind,"id":cid,"metadata":{},"source":[x+"\n" for x in source.splitlines()]}
    if kind=="code": base.update({"execution_count":None,"outputs":[]})
    return base

nb=json.load(open(NB)); replacements={
    "seam-20260714-validity-bounds-md":cell("markdown","seam-20260714-validity-bounds-md",MD),
    "seam-20260714-validity-bounds-code":cell("code","seam-20260714-validity-bounds-code",CODE),
}
seen=set(); new=[]
for c in nb["cells"]:
    if c.get("id") in replacements: new.append(replacements[c["id"]]); seen.add(c["id"])
    else: new.append(c)
for cid,c in replacements.items():
    if cid not in seen: new.append(c)
nb["cells"]=new; NB.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+"\n")
print(NB)
