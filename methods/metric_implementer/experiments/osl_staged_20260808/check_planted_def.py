import sys, json
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")
import numpy as np
from methods.metric_implementer.experiments.osl_sweep import planted_metrics
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer import config as cfgmod

def balanced(pred, ref, min_per=8):
    m = np.isfinite(pred) & np.isfinite(ref)
    if m.sum() < 30: return float("nan")
    p, r = pred[m], ref[m]
    accs = [float(np.mean(p[r==v]==v)) for v in (0,1) if (r==v).sum() >= min_per]
    return float(np.mean(accs)) if len(accs)==2 else float("nan")

for task in ("humor", "math"):
    frz = json.load(open(f"/lfs/skampere3/0/alexspan/outputs/osl_multi/freeze_zxa_{task}_v1.json"))
    meta = frz["meta"]
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), meta["task"])
    n = int(meta["n_probes"])
    texts, _ = _load_texts(meta["task"], 60+n, cfg)
    probes = [t[:cfg.max_text_chars] for t in texts[60:60+n]]
    truth = {m["name"]: np.asarray(m["truth"], float) for m in planted_metrics(probes, int(meta["k_med_words"]))}
    z = np.load(f"/lfs/skampere3/0/alexspan/outputs/osl_multi/mbar_zxaglm_{task}_glm-52.npz", allow_pickle=True)
    names = [str(x) for x in z["names"]]
    mb = z["m_bar"]
    accs=[]
    for i,nm in enumerate(names):
        base, arm = nm.rsplit("||",1)
        if arm=="definition" and base in truth:
            b = balanced(mb[i], truth[base])
            accs.append((base,b))
    print(task, "PLANTED definition-arm balanced acc:", accs)
    finite=[b for _,b in accs if np.isfinite(b)]
    print(task, "mean:", np.mean(finite) if finite else float("nan"))
