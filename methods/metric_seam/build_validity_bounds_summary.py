#!/usr/bin/env python3
"""Build the consolidated 2026-07-14 validity/bounds experimental report."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/"outputs/metric_seam_pilot"
OUT=B/"validity_bounds_v1"
def load(path): return json.load(open(B/path))
def main():
    g2=load("hierarchy_r123/results/math_a12_g2_validity_v1/readout.json")
    a34=load("battery/effort_ladder/ws4/patents_pa__a34/readouts.json"); sh=a34["exact_shapley_op_class_decomposition"]; cuts=a34["dag_cut_information_certificate"]
    inv=load("hierarchy_r123/results/pipeline_inversion_replay_v1/readout.json")
    ceil=load("hierarchy_r123/results/code_review_glm52_ceiling_v1/execution_reframe.json")
    optional={}
    for key,path in {
        "math_a12_context":"hierarchy_r123/results/math_a12_context_train_v1/readout.json",
        "patents_a34_matched_info":"hierarchy_r123/results/patents_a34_matched_info_v1/readout.json",
        "patents_a34_capture_recapture":"hierarchy_r123/results/patents_a34_capture_recapture_k3_v1/readout.json"}.items():
        p=B/path
        if p.exists(): optional[key]=json.load(open(p))
    status="complete" if len(optional)==3 else "partial_active_runs"
    out={"schema":"metric-seam.validity-bounds-summary.v2","status":status,"completed_core":{"a12_g2":g2,"a34_shapley":sh,"a34_dag_cuts_retracted":cuts,"pipeline_inversion":inv,"code_review_execution_reframe":ceil},"experimental_arms":optional,"audit_corrections":{"dag_cut_information_bound":{"status":"retracted_vacuous","reason":"raw norm text is item-injective and belongs to every minimal cut, so every plugin MI equals H(target); cut enumeration remains descriptive"},"patents_a34_matched_information":{"status":"unresolved","primary_tie_robust_c_index_delta":0.028,"ci95":[-0.012,0.070],"p_delta_le_zero":0.083,"reason":"54/98 target values are tied at the ceiling; Spearman is retained only as a sensitivity"}},"central_reading":"Verifiability is code-based and articulability is prompt-based; isomorphism is a separate reconstruction estimate. Prompt context repaired occasion individuation that the symbolic verifier omitted. Exact Shapley describes the frozen program, while neither a matched-information algorithmic advantage nor a nontrivial DAG information bound has been established.","scope":["No supervised external ground-truth anchor was introduced.","The patent retrieval/disclosure operation is precomputed/mocked machinery retained from the existing pipeline.","Failure finds no witness within a frozen program class and budget; it never establishes tacitness."]}
    OUT.mkdir(exist_ok=True);(OUT/"readout.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    mass=sh["op_class_mass"]; lines=["# Metric-seam validity and bounds — audit-corrected result","",f"Status: **{status}**.","",f"- Math a12 G2: **FAIL**; {g2['negative_proxy_traps']['n']-g2['negative_proxy_traps']['passed']}/{g2['negative_proxy_traps']['n']} proxy traps fired, while {g2['positive_true_violations']['passed']}/{g2['positive_true_violations']['n']} true violations were detected.",f"- Patent a34: 2,048 exact coalitions; evidence absolute φ-mass {mass['evidence']['absolute_phi_mass']:.3f}, computation {mass['computation']['absolute_phi_mass']:.3f}; efficiency residual {sh['efficiency_residual']:.3g}.",f"- DAG cuts: {cuts['n_inclusion_minimal_cuts']} minimal cuts enumerated correctly; the 2.338-bit information-bound reading is **retracted as vacuous** because every raw-text cut is item-injective.",f"- Pre-authoring replay: {inv['killed']}/{inv['n']} known dead units would be killed before authorship.",f"- Full-source execution: median ρ={ceil['median_raw_spearman']:.3f}, CI [{ceil['clustered_bootstrap_ci95'][0]:.3f},{ceil['clustered_bootstrap_ci95'][1]:.3f}]; this historical readout is target-resolution-confounded and not a codability estimate."]
    if "math_a12_context" in optional:
        x=optional["math_a12_context"]; gap=x["role_conditioned_individuation_gap"]; matrix=x["joint_asserted_identity_polarity_matrix"];lines.append(f"- a12 context arm: {x['valid']}/443 valid; {gap['reclassified_non_asserted_n']}/{gap['symbolic_applicable_n']} symbolically-applicable pairs reclassified as non-asserted roles; {matrix['both_satisfied']}/23 tautologies transported and {matrix['symbolic_violated_context_satisfied']}/24 symbolic violations context-resolved, with {matrix['symbolic_satisfied_context_violated']} reverse flips.")
    if "patents_a34_matched_info" in optional:
        x=optional["patents_a34_matched_info"];lines.append(f"- a34 matched information: **unresolved**. Spearman sensitivity B={x['spearman']['B_prompt_xZ']:.3f}, C={x['spearman']['C_program_xZ']:.3f}; primary tie-robust c-index delta=.028, audit CI [-.012,.070].")
    if "patents_a34_capture_recapture" in optional:
        x=optional["patents_a34_capture_recapture"]["estimate"];lines.append(f"- K=3 node-type capture–recapture: {x['observed_unique_types']} observed, estimated total {x['estimated_total_types']:.2f}, estimated coverage {x['estimated_coverage']:.1%}.")
    lines += ["","These are unsupervised reconstruction and executable-artifact results, not external correctness or a proof of tacitness."]
    (OUT/"report.md").write_text("\n".join(lines)+"\n");print(OUT/"report.md")
if __name__=="__main__":main()
