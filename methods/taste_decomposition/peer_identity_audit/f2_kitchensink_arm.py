#!/usr/bin/env python3
"""FULL-CONDITIONING arm for peer_revealed: nuisance(57) + identity(4) + base-model
recognition NLL + year. If the dense residual survives this, no named suspicion
channel (spurious text, author/institution identity, pretraining seen-ness, era)
absorbs it. NEW registered arm; frozen F2 row untouched."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TD = HERE.parent
def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m
    spec.loader.exec_module(m); return m
F2 = _mod(TD / "fusion/f2_deconf.py", "f2m4")
IDA = _mod(TD / "fusion/f2_identity_arm.py", "f2ida") if False else None

meta, ids_E, y, groups, dense, t0col = F2.load_E("peer_revealed")
ad = F2.F2C.ADAPTERS["peer_revealed"]()
bank, nuis, join = F2.align("peer_revealed", ad, ids_E, y, groups)

# identity block (same construction as f2_identity_arm)
sys.path.insert(0, str(TD / "fusion"))
spec = importlib.util.spec_from_file_location("f2ident", TD / "fusion/f2_identity_arm.py")
fi = importlib.util.module_from_spec(spec); sys.modules["f2ident"] = fi
spec.loader.exec_module(fi)
ID, id_names, joined = fi.identity_columns(ids_E)

nll_map = {json.loads(l)["ntitle"]: json.loads(l)["mean_nll"]
           for l in open(HERE / "peer_recognition_nll.jsonl")}
NLL = np.array([nll_map.get(str(nt), np.nan) for nt in ids_E])
NLL = np.where(np.isnan(NLL), np.nanmedian(NLL), NLL)
rows = [json.loads(l) for l in open(TD.parents[1] / "datasets/peer-review/vat_3y/revealed.jsonl")]
yrm = {rr["ntitle"]: int(rr["year"]) for rr in rows}
YR = np.array([yrm.get(str(nt), 2020) for nt in ids_E], dtype=float)

nuis_full = np.column_stack([nuis, ID, NLL, YR])
bn = np.column_stack([bank, nuis])
bnf = np.column_stack([bank, nuis_full])
fam = meta["family"]
r_ref = F2.fit_arm(fam, bn, dense, y, groups)
r_full = F2.fit_arm(fam, bnf, dense, y, groups)
prim_ref = F2.gboot(y, r_ref["_oof_VAT_nl0"], r_ref["_oof_VA_nl0"], groups, n_boot=2000)
prim_full = F2.gboot(y, r_full["_oof_VAT_nl0"], r_full["_oof_VA_nl0"], groups, n_boot=2000)
out = {"arm": "f2_kitchensink", "cell": "peer_revealed",
       "conditioning": "bank + nuisance(57) + identity(4) + recognition NLL + year",
       "c_ref": r_ref["VA_nl_mean"], "d_ref": r_ref["VAT_nl_mean"],
       "c_full": r_full["VA_nl_mean"], "d_full": r_full["VAT_nl_mean"],
       "primary_ref": prim_ref, "primary_full_conditioning": prim_full,
       "nll_alone_auc": F2.alone_auc(y, NLL)}
(TD / "results/f2_kitchensink_peer_revealed.json").write_text(json.dumps(out, indent=1))
print(f"(c) {out['c_ref']:.4f} (d) {out['d_ref']:.4f} | full (c'') {out['c_full']:.4f} "
      f"(d'') {out['d_full']:.4f} | ref {prim_ref['estimate']:+.4f} | "
      f"FULL {prim_full['estimate']:+.4f} {prim_full['ci95']} P={prim_full['p_gt_0']:.3f} | "
      f"NLL-alone {out['nll_alone_auc']:.3f}", flush=True)
print("KITCHENSINK_DONE", flush=True)
