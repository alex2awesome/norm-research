#!/usr/bin/env python3
"""Reconcile blind a34 fleet paraphrases and compute node-type Chao2."""
import json,re,subprocess
from pathlib import Path
from methods.metric_seam.verifiers.validity_bounds import capture_recapture_node_types
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"outputs/metric_seam_pilot/hierarchy_r123/results/patents_a34_capture_recapture_k3_v1"
SYSTEM="""You reconcile independently proposed verifier node types. Cluster two proposals only when they have the same op_class and express the same operational relation, allowing paraphrase. Keep genuinely different relations separate. Return exactly one JSON object {"clusters":[{"canonical":{"op_class":string,"witness_kind":string,"relation":string},"members":[{"fleet":integer,"index":integer}]}]}. Every supplied member must occur exactly once. No rationale or Markdown."""
FENCE=re.compile(r"```(?:json)?\s*(.*?)```",re.S|re.I)
def main():
    fleets=[json.load(open(BASE/f"fleet_{i}.json")) for i in (1,2,3)]
    payload={"fleets":{str(i+1):rows for i,rows in enumerate(fleets)}}
    cp=subprocess.run(["claude","--model","claude-sonnet-4-5-20250929","--output-format","text","--system-prompt",SYSTEM,"-p",json.dumps(payload,sort_keys=True)],capture_output=True,text=True,timeout=240,check=False)
    m=FENCE.search(cp.stdout); value=json.loads(m.group(1) if m else cp.stdout)
    expected={(fi,idx) for fi,rows in enumerate(fleets,1) for idx in range(len(rows))}; seen=set(); canonical=[[] for _ in fleets]
    for cluster in value["clusters"]:
        c=cluster["canonical"]
        if set(c)!={"op_class","witness_kind","relation"}: raise ValueError("bad canonical schema")
        for member in cluster["members"]:
            key=(member["fleet"],member["index"])
            if key not in expected or key in seen: raise ValueError("bad/duplicate member")
            if fleets[key[0]-1][key[1]]["op_class"]!=c["op_class"]: raise ValueError("cross-op-class cluster")
            seen.add(key);canonical[key[0]-1].append(c)
    if seen!=expected: raise ValueError(f"unassigned members: {expected-seen}")
    estimate=capture_recapture_node_types(canonical)
    out={"schema":"metric-seam.patents-a34-node-capture-recapture.v1","status":"complete","raw_fleet_sizes":[len(x) for x in fleets],"reconciliation":{"model":"claude-sonnet-4-5-20250929","unsupervised":True,"cluster_count":len(value["clusters"]),"contract_valid":True},"estimate":estimate,"claim_limits":["K=3 bounds cost but leaves a wide small-list estimator.","Semantic overlap is reconstructed by one unsupervised model, not externally validated.","This estimates authoring width, not construct validity, performance, or codability in general."]}
    (BASE/"reconciliation_raw.txt").write_text(cp.stdout);(BASE/"clusters.json").write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");(BASE/"readout.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    (BASE/"report.md").write_text(f"# Patent a34 node-type capture–recapture\n\nThree blind fleets proposed {sum(map(len,fleets))} raw nodes. Semantic reconciliation yielded {estimate['observed_unique_types']} observed types; bias-corrected Chao2 estimates {estimate['estimated_total_types']:.2f} total and {estimate['estimated_coverage']:.1%} coverage.\n\nThis is an authoring-width estimate, not validity or a codability ceiling.\n");print(BASE/"readout.json")
if __name__=="__main__":main()
