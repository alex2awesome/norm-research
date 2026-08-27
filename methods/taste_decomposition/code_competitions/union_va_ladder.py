#!/usr/bin/env python3
"""Full-union code-competitions VA ladder (user directive 2026-08-13: 'make sure you
are using ALL the data'). Population = all 6,353 labeled four-platform rows; A = 154
coded metrics scored corpus-wide; V = the 27 deterministic features. Frozen Layer-1
stack (HistGB {15,31} lr .06, seeds 0-2, grouped OOF), folds by platform-prefixed
canonical_pid. READOUT = WITHIN-PLATFORM (pooled-across-platform is composition-
dominated, pos rates .19-.84 — never quoted). Dense joins when the union chain lands."""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
R = HERE.parents[2]
sys.path.insert(0, str(HERE))
from build_v_and_readout import v_features, prep, gbm_oof  # frozen pieces

u = pd.read_parquet(R / 'outputs/v2_analysis/comp_fourplatform_cells/union_bank_scores.parquet')
sc = [c for c in u.columns if c.endswith('_score')]
A = u[sc].astype(float).values
lang = u.language.values
V = pd.DataFrame([v_features(t, l) for t, l in zip(pd.read_parquet(R / 'outputs/v2_analysis/comp_fourplatform_cells/union_bank_input.parquet').set_index('pair_id').loc[u.pair_id, 'candidate_text'], lang)]).values.astype(float)
y = u.label.astype(int).values
g = (u.platform + ':' + u.canonical_pid.astype(str)).values
plat = u.platform.values

def leg(M, name):
    X = prep(M)
    oofs = [gbm_oof(X, y, g, s) for s in (0, 1, 2)]
    o = np.mean(oofs, axis=0)
    res = {"pooled_NEVER_QUOTE": float(roc_auc_score(y, o))}
    per, nw, tot = {}, 0.0, 0
    for p in ("ac", "cf", "lc", "cc"):
        m = plat == p
        a = float(roc_auc_score(y[m], o[m]))
        per[p] = {"n": int(m.sum()), "auc": a}
        nw += a * m.sum(); tot += m.sum()
    res["per_platform"] = per
    res["within_platform_nwtd"] = nw / tot
    print(f"[{name}] within-platform {res['within_platform_nwtd']:.4f} | "
          + " ".join(f"{p}:{per[p]['auc']:.3f}" for p in per), flush=True)
    return res, o

out = {"n": int(len(y)), "n_groups": int(len(set(g))), "A_cols": len(sc)}
out["V"], oV = leg(V, "V")
out["A"], oA = leg(A, "A")
out["VA"], oVA = leg(np.column_stack([V, A]), "VA")
np.savez_compressed(HERE / 'union_va_oof.npz', pair_id=u.pair_id.values.astype(str),
                    y=y, groups=g, platform=plat, V_oof=oV, A_oof=oA, VA_oof=oVA)
json.dump(out, open(HERE / 'union_va_ladder.json', 'w'), indent=1)
print("UNION_VA_DONE", flush=True)
