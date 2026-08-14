#!/usr/bin/env python3
"""ERA-VOCABULARY AS SPURIOUS FEATURE (user directive 2026-08-13: "test this as a
spurious feature, it could be a leak for topic").

Channel construction: PREDICTED YEAR from text embeddings — a ridge regression on the
50 bge-large topic PCs + full 1024-d embedding, grouped-OOF (no row predicts its own
year) — i.e., the text-recoverable "era voice" of the abstract. If the dense model's
pre-2022 edge rides era-recognition (which proxies topic-era interactions), then
conditioning on predicted-year should collapse the band residual.

Arms on the 478 E-rows (frozen fit_arm):
  (c)   bank + nuis(57) + identity(4) + NLL + true-year + topicPC(50)  [= prior kitchen sink]
  (c*)  (c) + predicted_year (era-vocabulary channel)
  (d*)  (c*) + T
Readouts: overall residual + per-year-band residuals for BOTH (c) and (c*) stacks
(OOFs saved this time). Descriptive.
"""
import hashlib, importlib.util, gzip, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
TD = HERE.parent
def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m
    spec.loader.exec_module(m); return m
F2 = _mod(TD / "fusion/f2_deconf.py", "f2m_ev")
spec = importlib.util.spec_from_file_location("f2id3", TD / "fusion/f2_identity_arm.py")
fi = importlib.util.module_from_spec(spec); sys.modules["f2id3"] = fi
spec.loader.exec_module(fi)

meta, ids_E, y, groups, dense, t0col = F2.load_E("peer_revealed")
ad = F2.F2C.ADAPTERS["peer_revealed"]()
bank, nuis, join = F2.align("peer_revealed", ad, ids_E, y, groups)
ID, _, _ = fi.identity_columns(ids_E)
nll_map = {json.loads(l)["ntitle"]: json.loads(l)["mean_nll"] for l in open(HERE / "peer_recognition_nll.jsonl")}
NLL = np.array([nll_map.get(str(nt), np.nan) for nt in ids_E]); NLL = np.where(np.isnan(NLL), np.nanmedian(NLL), NLL)
rows = [json.loads(l) for l in open(TD.parents[1] / "datasets/peer-review/vat_3y/revealed.jsonl")]
yrm = {r["ntitle"]: int(r["year"]) for r in rows}
YR = np.array([yrm.get(str(nt), 2020) for nt in ids_E], dtype=float)

# embeddings on E rows (same join as topic arm)
z = np.load(TD / "closure/peer_revealed/abstract_emb_bge_large.npz", allow_pickle=True)
cache = {str(k): v for k, v in zip(z["keys"], z["vecs"])}
texts_by_nt = {}
for line in gzip.open(TD / "fusion/t0_rows/peer_revealed.texts.jsonl.gz", "rt"):
    r = json.loads(line); texts_by_nt.setdefault(r["id"], []).append(r["text"])
for rr in rows:
    texts_by_nt.setdefault(rr["ntitle"], []).append(rr["text"])
E = np.zeros((len(ids_E), 1024)); hit = 0
for i, nt in enumerate(ids_E):
    for t in texts_by_nt.get(str(nt), []):
        k = hashlib.sha256(t.encode()).hexdigest()
        if k in cache: E[i] = cache[k]; hit += 1; break
assert hit == len(ids_E)
from sklearn.decomposition import PCA
PC = PCA(n_components=50, random_state=0).fit_transform(E)

# predicted year: grouped-OOF ridge on the full embedding
pred_year = np.zeros(len(y))
for tr, te in GroupKFold(5).split(E, groups=groups):
    m = Ridge(alpha=10.0).fit(E[tr], YR[tr])
    pred_year[te] = m.predict(E[te])
from scipy.stats import spearmanr
rho_year = float(spearmanr(pred_year, YR).statistic)
print(f"era-recoverability: spearman(pred_year, true_year) = {rho_year:.3f}", flush=True)

fam = meta["family"]
base = np.column_stack([bank, nuis, ID, NLL, YR, PC])
withev = np.column_stack([base, pred_year])
r_c = F2.fit_arm(fam, base, dense, y, groups)
r_e = F2.fit_arm(fam, withev, dense, y, groups)

out = {"era_recoverability_rho": rho_year,
       "pred_year_alone_auc": F2.alone_auc(y, pred_year)}
for tag, r in (("kitchen_sink", r_c), ("plus_era_vocab", r_e)):
    oc, od = r["_oof_VA_nl0"], r["_oof_VAT_nl0"]
    prim = F2.gboot(y, od, oc, groups, n_boot=2000)
    bands = {}
    for name, m in {"2013-2019": (YR <= 2019), "2020-2021": (YR >= 2020) & (YR <= 2021),
                    "2022-2023": YR >= 2022}.items():
        mm = np.where(m)[0]
        bands[name] = {"n": int(len(mm)),
                       "c": float(roc_auc_score(y[mm], oc[mm])),
                       "d": float(roc_auc_score(y[mm], od[mm])),
                       "residual": float(roc_auc_score(y[mm], od[mm]) - roc_auc_score(y[mm], oc[mm]))}
    out[tag] = {"c": r["VA_nl_mean"], "d": r["VAT_nl_mean"], "primary": prim, "bands": bands}
    print(tag, "| overall", f"{prim['estimate']:+.4f}", "| bands",
          {k: round(v["residual"], 4) for k, v in bands.items()}, flush=True)

json.dump(out, open(TD / "results/f2_eravocab_peer_revealed.json", "w"), indent=1, default=float)
print("ERAVOCAB_DONE", flush=True)
