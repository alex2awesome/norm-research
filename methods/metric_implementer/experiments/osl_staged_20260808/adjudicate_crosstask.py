"""Cross-task GLM step-down test (2026-07-10).
Ports the humor frozen-crowd recipe (qwen3_adjudicate.cons_for): LOCAL_MID crowd,
hard-binarized majority consensus, family exclusion, agreement per metric then median.
Readout per task: agreement(glm-47), agreement(glm-52), delta -- vs humor's -0.18 step.
Also each mid rung's own leave-one-out agreement for the crowd-competence band.
"""
import numpy as np, glob, re, json, collections

LOCAL_MID = {"llama1b","llama3b","llama8b","llama70b","qwen25-3b","qwen25-7b",
             "qwen25-14b","qwen25-32b","qwen25-72b","gemma2-9b","gemma2-27b",
             "mistral7b","mistral-24b","phi4"}
FAM = {}
for e in LOCAL_MID:
    FAM[e] = ("llama" if e.startswith("llama") else "qwen" if e.startswith("qwen")
              else "gemma" if e.startswith("gemma") else "mistral" if e.startswith("mistral") else "phi")

TASKS = ["creative_writing", "math", "peer_review"]
out = {}
for T in TASKS:
    glm = {}
    for rung in ["glm-47", "glm-52"]:
        f = f"mbarglm_{T}_{rung}.npz"
        d = np.load(f, allow_pickle=True)
        glm[rung] = {"names": [str(x) for x in d["names"]], "M": d["m_bar"].astype(float),
                     "kinds": [str(x) for x in d["kinds"]]}
    names18 = [n for n in glm["glm-47"]["names"] if n in glm["glm-52"]["names"]]
    n47, n52 = len(glm["glm-47"]["names"]), len(glm["glm-52"]["names"])
    print(f"{T}: rung-name intersection {len(names18)} (47:{n47} 52:{n52})")
    for rung in glm:
        sel = [glm[rung]["names"].index(n) for n in names18]
        glm[rung] = {"names": names18, "M": glm[rung]["M"][sel],
                     "kinds": [glm[rung]["kinds"][i] for i in sel]}
    nan47 = float(__import__("numpy").isnan(glm["glm-47"]["M"]).mean())
    nan52 = float(__import__("numpy").isnan(glm["glm-52"]["M"]).mean())
    print(f"  NaN rate 47={nan47:.3f} 52={nan52:.3f}")

    mids = {}
    for f in glob.glob(f"mbar2_{T}_*.npz"):
        e = re.match(rf"mbar2_{re.escape(T)}_(.+)\.npz$", f.split("/")[-1]).group(1)
        if e not in LOCAL_MID: continue
        d = np.load(f, allow_pickle=True)
        nm = [str(x) for x in d["names"]]
        idx = [nm.index(n) for n in names18 if n in nm]
        matched = [n for n in names18 if n in nm]
        if len(matched) < len(names18):
            pass  # partial battery; align on matched subset below
        mids[e] = {"names": matched, "M": d["m_bar"].astype(float)[idx]}

    # common metric set across ALL mids + glm
    common = [n for n in names18 if all(n in v["names"] for v in mids.values())]
    ci = {n: i for i, n in enumerate(common)}
    H = {}  # hard verdicts per executor: (n_common, 300)
    for e, v in mids.items():
        sel = [v["names"].index(n) for n in common]
        H[e] = (v["M"][sel] > 0.5)
    gsel = [names18.index(n) for n in common]
    for rung in glm:
        H[rung] = (glm[rung]["M"][gsel] > 0.5)
    kinds = [glm["glm-47"]["kinds"][names18.index(n)] for n in common]

    def agreement(target, exclude_fam=None):
        crowd = [e for e in mids if e != target and (exclude_fam is None or FAM[e] != exclude_fam)]
        stack = np.stack([H[e] for e in crowd])           # (n_crowd, m, 300)
        cons = stack.mean(0) > 0.5                         # majority
        per_metric = (H[target] == cons).mean(1)           # (m,)
        return float(np.median(per_metric)), per_metric

    res = {"n_metrics": len(common), "n_crowd": len(mids), "kinds": dict(collections.Counter(kinds))}
    for rung in ["glm-47", "glm-52"]:
        med, pm = agreement(rung)
        res[rung] = round(med, 4)
        res[rung + "_by_kind"] = {k: round(float(np.median(pm[[i for i, kk in enumerate(kinds) if kk == k]])), 4)
                                  for k in set(kinds)}
    res["delta_52_minus_47"] = round(res["glm-52"] - res["glm-47"], 4)
    band = {}
    for e in sorted(mids):
        med, _ = agreement(e, exclude_fam=FAM[e])
        band[e] = round(med, 4)
    res["mid_band_LOO"] = band
    out[T] = res
    print(f"\n=== {T} (n_metrics={len(common)}, crowd={len(mids)}) ===")
    print(f"  glm-47 agreement: {res['glm-47']}   glm-52: {res['glm-52']}   DELTA: {res['delta_52_minus_47']:+.3f}")
    print(f"  by kind 47: {res['glm-47_by_kind']}")
    print(f"  by kind 52: {res['glm-52_by_kind']}")
    print(f"  mid band: min={min(band.values())} max={max(band.values())}")

json.dump(out, open("crosstask_glm_stepdown.json", "w"), indent=1)
print("\nsaved crosstask_glm_stepdown.json")
