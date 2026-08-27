#!/usr/bin/env python
"""z×a degeneracy audit: WHERE and WHY do executor x task cells go degenerate.

Per task: probe length stats + truncation. Per exec: per-ARM nan/const/yes rates,
per-probe corr(shown-len, nan) and corr(shown-len, yes). Also dumps, per task, the
per-probe predictions of every FRONTIER exec at name+dossier arms for TACIT/DIALECT
metrics (-> laptop subtask-kappa decomposition), keyed by probe sha1.
"""
import glob, hashlib, json, os, sys
import numpy as np

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
sys.path.insert(0, f"{B}/norm-research")
os.chdir(f"{B}/norm-research")
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer import config as cfgmod

ARMS = ["name", "definition", "explanation", "dossier", "dossier_mismatched", "definition_padded"]
FRONTIER = ["glm-47", "glm-52", "llama70b", "qwen25-72b", "hermes405b", "qwen25-32b"]
TASKS = ["humor", "creative_writing", "peer_review", "math"]

dump = {}
for task in TASKS:
    fz = f"{OM}/freeze_zxa_{task}_v1.json"
    if not os.path.exists(fz):
        continue
    frz = json.load(open(fz)); meta = frz["meta"]
    arm_of = {m["name"]: m["zxa"]["arm"] for m in frz["metrics"]}
    cls_of = {m["name"]: m["zxa"]["class"] for m in frz["metrics"]}
    nw_of = {m["name"]: m["zxa"]["n_words"] for m in frz["metrics"]}
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), meta["task"])
    n = int(meta["n_probes"])
    texts, ids = _load_texts(meta["task"], 60 + n, cfg)
    raw = texts[60:60 + n]
    probes = [t[:cfg.max_text_chars] for t in raw]
    plen = np.array([len(p) for p in probes], float)
    rlen = np.array([len(t) for t in raw], float)
    print(f"\n===== {task}: n={n} shown-chars med={int(np.median(plen))} p90={int(np.percentile(plen,90))} "
          f"raw-med={int(np.median(rlen))} truncated={float(np.mean(rlen > cfg.max_text_chars)):.0%} "
          f"cap={cfg.max_text_chars}")
    dump[task] = {"probe_sha": [hashlib.sha1(t.encode()).hexdigest() for t in raw],
                  "probe_len": plen.tolist(), "execs": {}}

    files = sorted(glob.glob(f"{OM}/mbar_zxa_{task}_*.npz")) + \
            sorted(glob.glob(f"{OM}/mbar_zxaglm_{task}_*.npz"))
    for fp in files:
        bn = os.path.basename(fp)
        if "prenanfix" in bn or bn.endswith(".bak.npz"):
            continue
        zz = np.load(fp, allow_pickle=True)
        if "alias_skip" in zz.files and int(np.atleast_1d(zz["alias_skip"])[0]):
            continue
        is_hard = bn.startswith("mbar_zxaglm")
        ex = bn.replace(f"mbar_zxaglm_{task}_", "").replace(f"mbar_zxa_{task}_", "").replace(".npz", "")
        ex = {"glm-45air": "glm-47"}.get(ex, ex)
        mb = np.asarray(zz["m_bar"], float)
        names = [str(x) for x in zz["names"]]
        keep = [i for i, nm in enumerate(names) if nm in arm_of]
        if not keep:
            continue
        Y = mb[keep]
        A = [arm_of[names[i]] for i in keep]
        NW = np.array([nw_of[names[i]] for i in keep], float)
        Yb = Y if is_hard else np.where(np.isfinite(Y), (Y > 0.5).astype(float), np.nan)
        parts = []
        for arm in ARMS:
            idx = [j for j, a in enumerate(A) if a == arm]
            if not idx:
                continue
            S = Yb[idx]
            nanr = float(np.mean(~np.isfinite(S)))
            fr = [r for r in S if np.isfinite(r).sum() >= 30]
            const = float(np.mean([float(np.nanstd(r)) == 0.0 for r in fr])) if fr else float("nan")
            yes = float(np.nanmean(S)) if np.isfinite(S).any() else float("nan")
            parts.append(f"{arm[:6]}:n{nanr:.2f}/c{const:.2f}/y{yes:.2f}")
        colnan = np.mean(~np.isfinite(Yb), axis=0)
        with np.errstate(all="ignore"):
            colyes = np.nanmean(Yb, axis=0)
        m = min(len(plen), Yb.shape[1])

        def cor(a, b):
            a = np.asarray(a, float); b = np.asarray(b, float)
            k = np.isfinite(a) & np.isfinite(b)
            if k.sum() < 10 or np.std(a[k]) == 0 or np.std(b[k]) == 0:
                return float("nan")
            return float(np.corrcoef(a[k], b[k])[0, 1])

        # entry-level: does degeneracy track the ARM text length? (row const vs n_words)
        rc = np.array([1.0 if (np.isfinite(r).sum() >= 30 and float(np.nanstd(r)) == 0.0) else 0.0
                       for r in Yb])
        print(f"  {ex:14s} r(len,nan)={cor(plen[:m], colnan[:m]):+.2f} "
              f"r(len,yes)={cor(plen[:m], colyes[:m]):+.2f} r(armwords,const)={cor(NW, rc):+.2f} | "
              + " ".join(parts))
        if ex in FRONTIER:
            sel = {}
            for j, i in enumerate(keep):
                nm = names[i]
                if cls_of[nm] in ("TACIT-CANDIDATE", "DIALECT-SUSPECT") and \
                        arm_of[nm] in ("name", "dossier"):
                    row = Yb[j][:m]
                    sel[nm] = [None if not np.isfinite(v) else float(v) for v in row]
            if sel:
                dump[task]["execs"][ex] = sel

out = f"{OM}/zxa_degen_dump.json"
json.dump(dump, open(out, "w"))
print(f"\nwrote {out} ({os.path.getsize(out)//1024} KB)")
