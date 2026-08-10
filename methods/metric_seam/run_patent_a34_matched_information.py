#!/usr/bin/env python3
"""Execute the frozen 100-call a34 matched-information prompt arm."""
from __future__ import annotations
import concurrent.futures, json, re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REQ=ROOT/"outputs/metric_seam_pilot/hierarchy_r123/requests/patents_a34_matched_info_v1"
OUT=ROOT/"outputs/metric_seam_pilot/hierarchy_r123/results/patents_a34_matched_info_v1"
FENCE=re.compile(r"```(?:json)?\s*(.*?)```",re.S|re.I)
def one(r):
    try:
        cp=subprocess.run(["claude","--model",r["model"],"--output-format","text","--system-prompt",r["system_prompt"],"-p",r["user_prompt"]],capture_output=True,text=True,timeout=240)
        raw=cp.stdout; cand=FENCE.search(raw); value=json.loads(cand.group(1) if cand else raw)
        if not isinstance(value,dict) or set(value)!={"score"} or type(value["score"]) is not int or not 0<=value["score"]<=10: raise ValueError("bad score contract")
        return {"request_sha256":r["request_sha256"],"datapoint_id":r["datapoint_id"],"valid":True,"score":value["score"],"raw_response":raw,"returncode":cp.returncode,"stderr":cp.stderr}
    except Exception as e:
        return {"request_sha256":r["request_sha256"],"datapoint_id":r["datapoint_id"],"valid":False,"error":f"{type(e).__name__}: {e}"}
def main():
    rs=[json.loads(x) for x in (REQ/"requests.jsonl").read_text().splitlines()]; OUT.mkdir(parents=True,exist_ok=False)
    smoke=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: smoke=list(ex.map(one,rs[:5]))
    rows=list(smoke); stopped=sum(x["valid"] for x in smoke)!=5
    if not stopped:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: rows+=list(ex.map(one,rs[5:]))
    (OUT/"responses.jsonl").write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in rows))
    (OUT/"manifest.json").write_text(json.dumps({"status":"stopped_after_smoke" if stopped else "complete","smoke_valid":sum(x["valid"] for x in smoke),"executed":len(rows),"valid":sum(x["valid"] for x in rows),"retries":0,"gpu_used":False},indent=2)+"\n")
    print(OUT/"manifest.json"); raise SystemExit(2 if stopped else 0)
if __name__=="__main__": main()
