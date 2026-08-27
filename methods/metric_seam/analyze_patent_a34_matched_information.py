#!/usr/bin/env python3
"""Compare a34 matched-information arms A/B/C to one frozen LLM target."""
from __future__ import annotations
import importlib.util,json,math,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TASK=ROOT/"outputs/metric_seam_pilot/tasks/patents_pa"; RES=ROOT/"outputs/metric_seam_pilot/hierarchy_r123/results/patents_a34_matched_info_v1"
REQ=ROOT/"outputs/metric_seam_pilot/hierarchy_r123/requests/patents_a34_matched_info_v1/requests.jsonl"
sys.path[:0]=[str(ROOT/"methods/metric_seam"),str(ROOT/"methods/metric_seam/f2p_mock")]
from certificates import spearman
from ops_pa import PriorArtOps
from eval_patents_pa import load_all
def rho(x,y,ids): return spearman([x[d] for d in ids],[y[d] for d in ids])
def main():
    responses=[json.loads(x) for x in (RES/"responses.jsonl").read_text().splitlines()]; B={x["datapoint_id"]:x["score"]/10 for x in responses if x["valid"]}
    request_ids={json.loads(x)["datapoint_id"] for x in REQ.read_text().splitlines()}
    # Frozen evidence-aware target.
    p1={};p2={}
    for line in open(TASK/"ws3_evidence_results.jsonl"):
        r=json.loads(line)
        if r["aspect_id"]=="a34" and r["channel"] in ("evidence_pass1","evidence_pass2") and isinstance(r["score"],int): (p1 if r["channel"].endswith("pass1") else p2)[r["datapoint_id"]]=r["score"]/10
    target={d:(p1[d]+p2[d])/2 for d in p1 if d in p2}
    target_rel1=spearman([p1[d] for d in target],[p2[d] for d in target])
    target_ceiling=math.sqrt((2*target_rel1)/(1+target_rel1))
    doc,_,_,fields=load_all(); A=doc["a34"]
    spec=importlib.util.spec_from_file_location("a34dag",ROOT/"outputs/metric_seam_pilot/battery/effort_ladder/ws4/patents_pa__a34/dag_program.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    texts={x["datapoint_id"]:x["ctext"] for x in json.load(open(TASK/"items.json"))}; ops=PriorArtOps(TASK/"pa_features.json"); fmap={d:{f.split("__",1)[1]:fields[f].get(d,"") for f in fields if f.startswith("a34__")} for d in texts}; C={d:mod.score_with_dpid(texts[d],fmap[d],ops,d) for d in texts}
    ids=sorted(set(B)&set(C)&set(target)); ids_a=sorted(request_ids&set(A)&set(target))
    obs={"A_prompt_x":rho(A,target,ids_a),"B_prompt_xZ":rho(B,target,ids),"C_program_xZ":rho(C,target,ids)}
    rng=random.Random(20260714); deltas=[]
    for _ in range(5000):
        samp=[ids[rng.randrange(len(ids))] for _ in ids]; deltas.append(rho(C,target,samp)-rho(B,target,samp))
    ds=sorted(x for x in deltas if x==x); ci=[ds[int(.025*len(ds))],ds[int(.975*len(ds))-1]]
    delta=obs["C_program_xZ"]-obs["B_prompt_xZ"]
    reading="algorithmic execution advantage" if ci[0]>0 else "program is lossy relative to informed prompt" if ci[1]<0 else "B and C are not resolved; information access remains sufficient explanation"
    out={"schema":"metric-seam.patents-a34-matched-info-readout.v1","status":"complete" if len(B)==100 else "partial","n_common":{"B_vs_C":len(ids),"A_descriptive":len(ids_a)},"target":"frozen Gemma M-bar(x,Z), unsupervised reconstruction anchor","spearman":obs,"reliability":{"C_program":1.0,"B_prompt":"one pass; not estimated","target_single_pass_rel1":target_rel1,"target_two_pass_attenuation_ceiling":target_ceiling},"ceiling_normalized_spearman":{"B_prompt_xZ":obs["B_prompt_xZ"]/target_ceiling,"C_program_xZ":obs["C_program_xZ"]/target_ceiling},"C_minus_B":delta,"C_minus_B_bootstrap_ci95":ci,"bootstrap_draws":5000,"reading":reading,"claim_limits":["This compares reconstruction of a frozen LLM target, not external correctness.","The evidence record is mocked/precomputed machinery retained from the existing pipeline.","A versus C remains information-confounded and has smaller doc-only target support; the capability contrast is B versus C on the 98 contract-valid matched-information items."]}
    (RES/"readout.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); (RES/"report.md").write_text(f"# Patent a34 matched-information result\n\nA ρ={obs['A_prompt_x']:.3f}; B ρ={obs['B_prompt_xZ']:.3f}; C ρ={obs['C_program_xZ']:.3f}. C−B={delta:.3f}, 95% paired bootstrap CI [{ci[0]:.3f}, {ci[1]:.3f}].\n\n**Reading:** {reading}.\n"); print(RES/"readout.json")
if __name__=="__main__": main()
