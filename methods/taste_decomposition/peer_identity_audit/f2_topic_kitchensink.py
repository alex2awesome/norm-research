#!/usr/bin/env python3
"""FINAL unified conditioning arm for peer_revealed: bank + nuisance(57) +
identity(4) + recognition-NLL + year + TOPIC (top-50 PCs of the cached
BAAI/bge-large-en-v1.5 abstract embeddings — same instrument as the closure-time
job1 topic control). User question 2026-08-13: "did we do topic de-correlation as
well? some fields just get more citations."  PCA is unsupervised, fit on the
E-rows themselves (no label enters). NEW registered arm."""
import hashlib, importlib.util, gzip, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TD = HERE.parent
def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m
    spec.loader.exec_module(m); return m
F2 = _mod(TD / "fusion/f2_deconf.py", "f2m5")
spec = importlib.util.spec_from_file_location("f2ident2", TD / "fusion/f2_identity_arm.py")
fi = importlib.util.module_from_spec(spec); sys.modules["f2ident2"] = fi
spec.loader.exec_module(fi)

meta, ids_E, y, groups, dense, t0col = F2.load_E("peer_revealed")
ad = F2.F2C.ADAPTERS["peer_revealed"]()
bank, nuis, join = F2.align("peer_revealed", ad, ids_E, y, groups)
ID, id_names, joined = fi.identity_columns(ids_E)

nll_map = {json.loads(l)["ntitle"]: json.loads(l)["mean_nll"] for l in open(HERE / "peer_recognition_nll.jsonl")}
NLL = np.array([nll_map.get(str(nt), np.nan) for nt in ids_E]); NLL = np.where(np.isnan(NLL), np.nanmedian(NLL), NLL)
rows = [json.loads(l) for l in open(TD.parents[1] / "datasets/peer-review/vat_3y/revealed.jsonl")]
yrm = {rr["ntitle"]: int(rr["year"]) for rr in rows}
YR = np.array([yrm.get(str(nt), 2020) for nt in ids_E], dtype=float)

# topic embeddings: cache keyed by sha256(text); map ntitle -> text via both sources
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
        if k in cache:
            E[i] = cache[k]; hit += 1; break
print(f"embedding hit {hit}/{len(ids_E)}", flush=True)
assert hit >= 0.95 * len(ids_E), "embedding join too sparse — rebuild via job1 embed path"
from sklearn.decomposition import PCA
PC = PCA(n_components=50, random_state=0).fit_transform(E)

fam = meta["family"]
bn = np.column_stack([bank, nuis])
bnt = np.column_stack([bank, nuis, ID, NLL, YR, PC])
r_ref = F2.fit_arm(fam, bn, dense, y, groups)
r_top = F2.fit_arm(fam, bnt, dense, y, groups)
prim_ref = F2.gboot(y, r_ref["_oof_VAT_nl0"], r_ref["_oof_VA_nl0"], groups, n_boot=2000)
prim_top = F2.gboot(y, r_top["_oof_VAT_nl0"], r_top["_oof_VA_nl0"], groups, n_boot=2000)
out = {"arm": "f2_topic_kitchensink", "cell": "peer_revealed",
       "conditioning": "bank + nuisance(57) + identity(4) + NLL + year + topic PCs(50, bge-large)",
       "embedding_hit": hit,
       "c_ref": r_ref["VA_nl_mean"], "d_ref": r_ref["VAT_nl_mean"],
       "c_full": r_top["VA_nl_mean"], "d_full": r_top["VAT_nl_mean"],
       "primary_ref": prim_ref, "primary_full": prim_top}
(TD / "results/f2_topic_kitchensink_peer_revealed.json").write_text(json.dumps(out, indent=1))
print(f"(c) {out['c_ref']:.4f} (d) {out['d_ref']:.4f} | +topic (c3) {out['c_full']:.4f} "
      f"(d3) {out['d_full']:.4f} | ref {prim_ref['estimate']:+.4f} | "
      f"FULL+TOPIC {prim_top['estimate']:+.4f} {prim_top['ci95']} P={prim_top['p_gt_0']:.3f}", flush=True)
print("TOPIC_KITCHENSINK_DONE", flush=True)
