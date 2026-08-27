#!/usr/bin/env python3
"""PREREG'D READOUT — peer_revealed r6-ERA probe (notes/2026-08-14__prereg_peer_r6era_probe.md,
committed ae7b7c329 BEFORE the slice was drawn).

PRIMARY (declared): pre-2022 BAND residual (d)-(c) on the E rows with the r6-era
A-ROUTED criteria appended to (c)'s bank — paired grouped bootstrap — against the
frozen baselines +.1298 (200-PC kitchen-sink frame, f2_eravocab2) and +.168
[+.090,+.251] (plain bank+nuis frame, boot_bands).  SECONDARY: the 2022-23 band
must NOT move.  VERDICTS (declared): H-vocab SUPPORTED if the pre-2022 residual
falls >= .04 with 2022-23 stable; H-configural STRENGTHENED if it falls < .02;
intermediate reported as split, no verdict.

Exactly ONE thing changes per frame vs the frozen baseline artifacts: the r6
A-routed columns are appended to the bank block.  Nothing else (no r6 B columns,
same extras, same fit_arm family, same folds via the shared machinery).

DEVIATION RECORD: the prereg declared a 2-family P=8 fleet; the executed fleet was
3-family P=12 (claude_opus/claude_sonnet legs added per the r5 precedent, still
sealed fresh sessions) — strictly MORE family diversity, recorded not hidden.

Run on sk3 after peer_revealed_r6_scores.npz lands.  CPU only.
"""
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TD = HERE.parent


def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


F2 = _mod(TD / "fusion/f2_deconf.py", "f2m_r6")
fi = _mod(TD / "fusion/f2_identity_arm.py", "f2id_r6")

meta, ids_E, y, groups, dense, t0col = F2.load_E("peer_revealed")
ad = F2.F2C.ADAPTERS["peer_revealed"]()
bank, nuis, join = F2.align("peer_revealed", ad, ids_E, y, groups)
fam = meta["family"]

# ---- r6 A-routed columns on the E rows --------------------------------------
d6 = TD / "closure/peer_revealed"
z6 = np.load(d6 / "peer_revealed_r6_scores.npz", allow_pickle=True)
cids6 = [str(c) for c in z6["crit_ids"]]
routing6 = json.load(open(d6 / "peer_revealed_r6_routing_final.json"))
track6 = {c["blind_id"]: c["final_route"] for c in routing6["final"]}  # ARBITER-FINAL
iA6 = [i for i, c in enumerate(cids6) if track6.get(c) == "A"]
namesA6 = [str(z6["crit_names"][i]) for i in iA6]
pop_ids6 = [str(x) for x in z6["row_id"]]
pos6 = {}
for k, r in enumerate(pop_ids6):
    pos6.setdefault(r, k)
assert len(pos6) == len(pop_ids6), "r6 scores: duplicate row ids"
idx6 = np.array([pos6[str(i)] for i in ids_E])
A6 = z6["X"][idx6][:, iA6].astype(float)
print(f"r6 A-routed columns on E: {A6.shape} ({len(iA6)} criteria)", flush=True)

# ---- year + kitchen-sink extras (identical joins to f2_eravocab2_arm) -------
ID, _, _ = fi.identity_columns(ids_E)
nll_map = {json.loads(l)["ntitle"]: json.loads(l)["mean_nll"]
           for l in open(HERE / "peer_recognition_nll.jsonl")}
NLL = np.array([nll_map.get(str(nt), np.nan) for nt in ids_E])
NLL = np.where(np.isnan(NLL), np.nanmedian(NLL), NLL)
rows = [json.loads(l) for l in open(TD.parents[1] / "datasets/peer-review/vat_3y/revealed.jsonl")]
yrm = {r["ntitle"]: int(r["year"]) for r in rows}
YR = np.array([yrm.get(str(nt), 2020) for nt in ids_E], dtype=float)

z = np.load(TD / "closure/peer_revealed/abstract_emb_bge_large.npz", allow_pickle=True)
cache = {str(k): v for k, v in zip(z["keys"], z["vecs"])}
texts_by_nt = {}
for line in gzip.open(TD / "fusion/t0_rows/peer_revealed.texts.jsonl.gz", "rt"):
    r = json.loads(line)
    texts_by_nt.setdefault(r["id"], []).append(r["text"])
for rr in rows:
    texts_by_nt.setdefault(rr["ntitle"], []).append(rr["text"])
E = np.zeros((len(ids_E), 1024))
hit = 0
for i, nt in enumerate(ids_E):
    for t in texts_by_nt.get(str(nt), []):
        k = hashlib.sha256(t.encode()).hexdigest()
        if k in cache:
            E[i] = cache[k]
            hit += 1
            break
assert hit == len(ids_E)
PC = PCA(n_components=200, random_state=0).fit_transform(E)

BANDS = {"2013-2019": (YR <= 2019), "2020-2021": (YR >= 2020) & (YR <= 2021),
         "2022-2023": YR >= 2022}


def band_read(r):
    oc, od = r["_oof_VA_nl0"], r["_oof_VAT_nl0"]
    out = {"overall": F2.gboot(y, od, oc, groups, n_boot=2000), "bands": {}}
    for name, m in BANDS.items():
        mm = np.where(m)[0]
        b = {"n": int(len(mm)),
             "c": float(roc_auc_score(y[mm], oc[mm])),
             "d": float(roc_auc_score(y[mm], od[mm])),
             "residual": float(roc_auc_score(y[mm], od[mm]) - roc_auc_score(y[mm], oc[mm]))}
        b["boot"] = F2.gboot(y[mm], od[mm], oc[mm], groups[mm], n_boot=2000)
        out["bands"][name] = b
    return out


FRAMES = {
    "plain": np.column_stack([bank, nuis]),
    "kitchen_sink": np.column_stack([bank, nuis, ID, NLL, YR, PC]),
}
BASELINE = {"plain": {"pre2022_2013_2019": 0.168, "ci": [0.090, 0.251]},
            "kitchen_sink": {"pre2022_2013_2019": 0.1298}}

res = {"prereg": "notes/2026-08-14__prereg_peer_r6era_probe.md (ae7b7c329, frozen pre-slice)",
       "r6_A_criteria": namesA6, "n_r6_A": len(iA6),
       "fleet_deviation": ("prereg declared 2-family P=8; executed 3-family P=12 "
                           "(claude legs added per r5 precedent, sealed fresh sessions)"),
       "frames": {}}
for tag, base in FRAMES.items():
    withr6 = np.column_stack([base, A6])
    r_c = F2.fit_arm(fam, base, dense, y, groups)
    r_e = F2.fit_arm(fam, withr6, dense, y, groups)
    fr = {"baseline_quoted": BASELINE[tag],
          "without_r6": band_read(r_c), "with_r6A": band_read(r_e)}
    fr["fall_2013_2019"] = (fr["without_r6"]["bands"]["2013-2019"]["residual"]
                            - fr["with_r6A"]["bands"]["2013-2019"]["residual"])
    fr["move_2022_2023"] = (fr["with_r6A"]["bands"]["2022-2023"]["residual"]
                            - fr["without_r6"]["bands"]["2022-2023"]["residual"])
    res["frames"][tag] = fr
    print(f"[{tag}] pre-2022(13-19) residual {fr['without_r6']['bands']['2013-2019']['residual']:+.4f}"
          f" -> {fr['with_r6A']['bands']['2013-2019']['residual']:+.4f}"
          f" (fall {fr['fall_2013_2019']:+.4f}) | 2022-23 move {fr['move_2022_2023']:+.4f}",
          flush=True)

# ---- declared verdict (kitchen-sink frame is the stated PRIMARY comparison) --
falls = {t: res["frames"][t]["fall_2013_2019"] for t in FRAMES}
moves = {t: res["frames"][t]["move_2022_2023"] for t in FRAMES}
modern_stable = all(abs(m) < 0.04 for m in moves.values())
if all(f >= 0.04 for f in falls.values()) and modern_stable:
    verdict = "H-vocab SUPPORTED (pre-2022 residual falls >= .04 in both frames, modern band stable)"
elif all(f < 0.02 for f in falls.values()):
    verdict = "H-configural STRENGTHENED (pre-2022 residual falls < .02 in both frames)"
else:
    verdict = "SPLIT (intermediate/frame-divergent falls) — reported, no verdict per prereg"
res["verdict"] = verdict
res["modern_band_stable"] = modern_stable
print("VERDICT:", verdict, flush=True)

json.dump(res, open(TD / "results/r6era_band_readout_peer_revealed.json", "w"),
          indent=1, default=float)
print("R6ERA_READOUT_DONE", flush=True)
