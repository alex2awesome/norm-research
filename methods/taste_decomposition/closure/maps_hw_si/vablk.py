import json, numpy as np
from pathlib import Path
import cells as C, closure_core as L
from readout import stack_oof
HERE=Path(".")
out={}
for cell in C.CELLS:
    d=C.load(cell)
    sp=json.loads((HERE/f"{cell}_splits.json").read_text())
    split=np.array([r["split"] for r in sp["rows"]])
    y,g,dense=d["y"],d["groups"],d["dense"]
    fitm,monm=split=="fit_mine",split=="monitor"
    held=np.isin(d["dense_split"],["eval","test"])
    res={}; preds={}
    for nm,blocks in (("V",[d["V"]]),("A",[d["A"]]),("VA",[d["V"],d["A"]])):
        r=L.fit_block(blocks,fitm,monm,y,g)
        v=np.full(len(y),np.nan); v[fitm]=r["oof_nl_fitmine"]; v[monm]=r["nl_mon"]
        res[nm]={"n_feat":r["n_features"],"nl_HONEST":L.auc(y[held],v[held]),
                 "lin_MONITOR":L.auc(y[monm],r["lin_mon"]),"nl_MONITOR":L.auc(y[monm],r["nl_mon"])}
        preds[nm]=v
    yh=y[held]; gh=g[held]
    aV=preds["V"][held]; aA=preds["A"][held]
    s=stack_oof([aV,aA],yh,gh)
    res["A_increment_over_V_HONEST"]=L.auc(yh,s)-L.auc(yh,aV)
    res["ci_A_over_V"]=L.group_boot_ci(yh,s,aV,gh)
    res["T_HONEST_ensemble"]=L.auc(yh,dense[held])
    out[cell]=res
    print(cell, json.dumps(res,default=float))
Path("va_block_ledger.json").write_text(json.dumps(out,indent=1,default=float))
