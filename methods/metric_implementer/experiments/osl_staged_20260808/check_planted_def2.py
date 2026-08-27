import sys, json
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")
import numpy as np
from methods.metric_implementer.experiments.osl_sweep import planted_metrics
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer import config as cfgmod

task = "math"
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
for i,nm in enumerate(names):
    base, arm = nm.rsplit("||",1)
    if arm=="definition" and base in truth:
        pred, ref = mb[i], truth[base]
        m = np.isfinite(pred) & np.isfinite(ref)
        p, r = pred[m], ref[m]
        for v in (0,1):
            sel = r==v
            n_v = int(sel.sum())
            acc_v = float(np.mean(p[sel]==v)) if n_v else float("nan")
            print(f"{base:25s} class={v} n={n_v} acc={acc_v:.3f}")
        print(f"   truth mean={np.mean(r):.3f} pred mean(yes-rate)={np.mean(p):.3f} nan_rate={1-m.mean():.3f}")
