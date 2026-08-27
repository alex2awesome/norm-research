#!/usr/bin/env python3
"""Counterfactual base-rate-first replay on a34 and four code-review units."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"methods/metric_seam/f2p_mock"))
from eval_patents_pa import load_all
from methods.metric_seam.verifiers.validity_bounds import base_rate_probe
OUT=ROOT/"outputs/metric_seam_pilot/hierarchy_r123/results/pipeline_inversion_replay_v1"
def main():
    _,_,_,fields=load_all(); items=json.load(open(ROOT/"outputs/metric_seam_pilot/tasks/patents_pa/items.json")); ids=[x["datapoint_id"] for x in items]
    rows=[]
    for field in ("a34__closest_art","a34__distinguishing"):
        obs=[bool(str(fields.get(field,{}).get(d,"")).strip()) for d in ids]
        rows.append({"task":"patents_pa","unit":field,**base_rate_probe(obs)})
    cr=json.load(open(ROOT/"outputs/metric_seam_pilot/hierarchy_r123/results/code_review_ast_train_v2/readout.json"))
    for unit in cr["real_units"]:
        natural=[r for r in unit["natural"] if r.get("verdict") is not None]
        # The proposed detector's target event is the named violation, not the
        # broader syntactic occasion on which it could in principle apply.
        obs=[r["verdict"]["applies"] and r["verdict"]["violated"] for r in natural]
        rows.append({"task":"code_review","unit":unit["aspect_id"],"implemented_relation":unit["implemented_relation"],**base_rate_probe(obs)})
    out={"schema":"metric-seam.pipeline-inversion-replay.v1","status":"complete","rule":"kill before authoring if occurrence <.10 or >.90","rows":rows,"killed":sum(not r["passed"] for r in rows),"n":len(rows),"reading":"Retrospective counterfactual only: the features already existed. The same probe is prospective for new units."}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"readout.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); (OUT/"report.md").write_text("# Pipeline inversion replay\n\n"+"\n".join(f"- {r['task']} {r['unit']}: occurrence={r['occurrence_rate']:.3f} → {r['decision']}" for r in rows)+"\n"); print(OUT/"readout.json")
if __name__=="__main__":main()
