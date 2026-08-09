#!/usr/bin/env python3
"""T0 (UNTRAINED-T) ARM, step 3: VAT0 = the frozen Layer-1 stack with the trained
dense column swapped for the untrained-base-model column.

Frozen design: notes/2026-07-27__vat-run-registry.md ("2026-08-08 -- FROZEN
DESIGN (before any scoring): UNTRAINED-T FUSION ARM").

The stack machinery is NOT reimplemented: `direction1_mirror.fit_arm` and
`direction1_mirror.group_paired_boot` are imported and called verbatim.  For a
cell, `fit_arm` is called TWICE on the identical (family, VA_raw, y, groups) --
once with the trained T column and once with the T0 column.  `L1.outer_folds`
is a pure function of (n, groups), so both calls share byte-identical folds and
the VA arm they produce is asserted equal, which is what makes the paired
bootstrap VAT_nl - VAT0_nl legitimate.

Readouts per cell (all threshold-free):
    T0 alone on E, T alone on E
    VA_lin / VA_nl        (recomputed; gated against the master ledger)
    VAT_lin / VAT_nl      (recomputed; gated against the master ledger)
    VAT0_lin / VAT0_nl
    group paired bootstraps, 2,000 draws, on the cell's grouping unit:
        VAT0_nl - VA_nl     (does fusion help at all WITHOUT training?)
        VAT_nl  - VAT0_nl   (what is label-training worth on top of T0?)
        T0      - T

PLATFORM NOTE (the ledger's own landmine): GroupKFold fold MEMBERSHIP depends on
sklearn version AND on the sort kernel, i.e. on the architecture.  This script
therefore records platform/versions and a reproduction gate per cell, and is run
on BOTH boxes; the merger keeps, per cell, the box that reproduces that cell's
published VA_nl / VAT_nl.

CPU only.  Usage:  python3 t0_fuse.py --box mac      (or --box sk3)
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TD = HERE.parent
RESULTS = TD / "results"
ROWS = HERE / "t0_rows"
SCORES = HERE / "t0_scores"
OUT = HERE / "t0_results"
OUT.mkdir(exist_ok=True)


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


D1 = load_module(HERE / "direction1_mirror.py", "d1_fuse_t0")
fit_arm = D1.fit_arm
group_paired_boot = D1.group_paired_boot

TOL = 1e-4          # reproduction tolerance for "this box produced the ledger"


def env_block():
    import numpy as _np
    import scipy as _sp
    return {"platform": f"{platform.system()}-{platform.machine()}",
            "python": sys.version.split()[0], "sklearn": sklearn.__version__,
            "numpy": _np.__version__, "scipy": _sp.__version__}


def load_cell(cell):
    z = np.load(ROWS / f"{cell}.npz", allow_pickle=True)
    meta = json.loads((ROWS / f"{cell}.meta.json").read_text())
    uids = [str(u) for u in z["uids"]]
    y = z["y"].astype(int)
    groups = np.array([str(g) for g in z["groups"]], dtype=object)
    dense = z["dense"].astype(float)
    VA = z["VA_raw"].astype(float)

    sp = SCORES / f"{cell}.jsonl.gz"
    if not sp.exists():
        raise FileNotFoundError(sp)
    pm = {}
    with gzip.open(sp, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            pm[r["uid"]] = float(r["p_yes"])
    assert len(pm) == len(uids), f"{cell}: score rows {len(pm)} != E rows {len(uids)}"
    assert set(pm) == set(uids), f"{cell}: uid sets differ between rows and scores"
    t0col = np.array([pm[u] for u in uids], dtype=float)
    assert np.isfinite(t0col).all(), f"{cell}: non-finite T0 score"
    smeta = json.loads((SCORES / f"{cell}.meta.json").read_text())
    return meta, uids, y, groups, dense, VA, t0col, smeta


def _code_v3_within_repo(uids, y, groups, dense, t0col, r_T, r_0):
    """code_v3's canonical readout is WITHIN-REPO (the pooled AUC is marked
    POOLED_DO_NOT_QUOTE in the master ledger because its two dense-held-out
    halves contradict each other).  Reuses closure/code_v3/cells_code.py's own
    within_repo_auc / within_repo_delta.  Available only where that closure
    directory and the v3 population live (sk3)."""
    try:
        CC = load_module(TD / "closure" / "code_v3" / "cells_code.py", "cells_code_t0fuse")
        d = CC.load()
        pos = {f"{i:06d}|{str(x)}": i for i, x in enumerate(d["ids"])}
        idx = np.array([pos[u] for u in uids])
        split = np.asarray([str(s) for s in d["split"]], dtype=object)[idx]
    except Exception as e:                                    # pragma: no cover
        return {"status": f"unavailable on this box: {type(e).__name__}: {e}"}
    blocks = {}
    for sp in ("eval", "test"):
        m = split == sp
        if m.sum() < 50:
            continue
        blocks[sp] = {
            "n": int(m.sum()),
            "T0": CC.within_repo_auc(y[m], t0col[m], groups[m]),
            "T": CC.within_repo_auc(y[m], dense[m], groups[m]),
            "VA_nl_seed0": CC.within_repo_auc(y[m], r_T["_oof_VA_nl0"][m], groups[m]),
            "VAT_nl_seed0": CC.within_repo_auc(y[m], r_T["_oof_VAT_nl0"][m], groups[m]),
            "VAT0_nl_seed0": CC.within_repo_auc(y[m], r_0["_oof_VAT_nl0"][m], groups[m]),
            "delta_VAT0_minus_VA": CC.within_repo_delta(y[m], r_0["_oof_VAT_nl0"][m],
                                                        r_0["_oof_VA_nl0"][m], groups[m]),
            "delta_VAT_minus_VAT0": CC.within_repo_delta(y[m], r_T["_oof_VAT_nl0"][m],
                                                         r_0["_oof_VAT_nl0"][m], groups[m]),
        }
        for k, v in list(blocks[sp].items()):
            if isinstance(v, dict):
                v.pop("per_repo", None)
                v.pop("per_repo_delta", None)
        blocks[sp]["pooled_within_split_caveat"] = (
            "per-split E-refit; the ledger's own readout for this cell is within-repo")
    return blocks


def run_cell(cell, box):
    t_start = time.time()
    meta, uids, y, groups, dense, VA, t0col, smeta = load_cell(cell)
    family = meta["family"]
    led = json.loads((RESULTS / f"vat_fullgrid_{cell}.json").read_text())

    r_T = fit_arm(family, VA, dense, y, groups)
    r_0 = fit_arm(family, VA, t0col, y, groups)
    # same folds, same VA matrix -> the VA arm must be bit-identical
    assert np.allclose(r_T["_oof_VA_nl0"], r_0["_oof_VA_nl0"]), \
        f"{cell}: VA arm differs between the two fit_arm calls -- folds not shared"
    assert abs(r_T["VA_lin"] - r_0["VA_lin"]) < 1e-12

    T_E = float(roc_auc_score(y, dense))
    T0_E = float(roc_auc_score(y, t0col))

    def gate(name, got, pub):
        return {"published": pub, "reproduced": got,
                "abs_diff": None if pub is None else abs(got - pub),
                "pass": None if pub is None else abs(got - pub) <= TOL}

    out = {
        "cell": cell, "box": box, "env": env_block(),
        "n_E": int(len(y)), "n_groups_E": int(len(set(groups))),
        "pos_rate_E": float(y.mean()),
        "family": family, "group_column": meta["group_column"],
        "n_features_VA_raw": int(VA.shape[1]),
        "n_features_VA_screened": r_T["n_features_VA"],
        "ids_sha256": meta["ids_sha256"], "texts_sha256": meta["texts_sha256"],
        "templates_sha256": smeta["templates_sha256"],

        "T0": T0_E,
        "T": T_E,
        "VA_lin": r_T["VA_lin"], "VA_nl": r_T["VA_nl_mean"],
        "VA_nl_seeds": r_T["VA_nl_seeds"], "VA_nl_spread": r_T["VA_nl_spread"],
        "VAT_lin": r_T["VAT_lin"], "VAT_nl": r_T["VAT_nl_mean"],
        "VAT_nl_seeds": r_T["VAT_nl_seeds"], "VAT_nl_spread": r_T["VAT_nl_spread"],
        "VAT0_lin": r_0["VAT_lin"], "VAT0_nl": r_0["VAT_nl_mean"],
        "VAT0_nl_seeds": r_0["VAT_nl_seeds"], "VAT0_nl_spread": r_0["VAT_nl_spread"],

        "boot_VAT0_nl_minus_VA_nl": group_paired_boot(y, r_0["_oof_VAT_nl0"],
                                                      r_0["_oof_VA_nl0"], groups),
        "boot_VAT_nl_minus_VAT0_nl": group_paired_boot(y, r_T["_oof_VAT_nl0"],
                                                       r_0["_oof_VAT_nl0"], groups),
        "boot_T0_minus_T": group_paired_boot(y, t0col, dense, groups),
        "boot_VAT0_nl_minus_T0": group_paired_boot(y, r_0["_oof_VAT_nl0"], t0col, groups),

        "ledger_gate": {
            "tol": TOL,
            "T": gate("T", T_E, led.get("T")),
            "VA_nl": gate("VA_nl", r_T["VA_nl_mean"], led.get("VA_nl")),
            "VAT_nl": gate("VAT_nl", r_T["VAT_nl_mean"], led.get("VAT_nl")),
            "VA_lin": gate("VA_lin", r_T["VA_lin"], led.get("VA_lin")),
            "VAT_lin": gate("VAT_lin", r_T["VAT_lin"], led.get("VAT_lin")),
        },
        "t0_score_distribution": {k: smeta[k] for k in
                                  ("p_yes_min", "p_yes_p05", "p_yes_median", "p_yes_p95",
                                   "p_yes_max", "p_yes_mean", "n_distinct_p_yes",
                                   "argmax_yes_fraction", "COLLAPSE_FLAG") if k in smeta},
        "ledger_VA_nl_fullfit_at_E": led.get("VA_nl_fullfit_at_E"),
        "ledger_V3": led.get("V3"),
        "POOLED_DO_NOT_QUOTE": bool(led.get("POOLED_DO_NOT_QUOTE") or
                                    meta.get("POOLED_DO_NOT_QUOTE")),
        "runtime_sec": None,
    }
    if cell == "code_v3":
        out["within_repo"] = _code_v3_within_repo(uids, y, groups, dense, t0col,
                                                  r_T, r_0)

    g = out["ledger_gate"]
    out["ledger_gate"]["pass"] = bool(
        (g["VA_nl"]["pass"] is not False) and (g["VAT_nl"]["pass"] is not False))
    out["runtime_sec"] = time.time() - t_start
    (OUT / f"{cell}.{box}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"  [{cell}/{box}] T0 {T0_E:.4f} T {T_E:.4f} | VA_nl {r_T['VA_nl_mean']:.4f} "
          f"(led {led.get('VA_nl')}) | VAT0_nl {r_0['VAT_nl_mean']:.4f} | "
          f"VAT_nl {r_T['VAT_nl_mean']:.4f} (led {led.get('VAT_nl')}) | "
          f"gate={'PASS' if out['ledger_gate']['pass'] else 'DRIFT'} "
          f"({out['runtime_sec']:.0f}s)", flush=True)
    return out


ALL_CELLS = ["peer_verdict", "peer_curation", "peer_revealed",
             "nc_responded", "nc_outcome", "nc_agree", "cw_community",
             "hashtagwars_verdict", "cap_finalist", "cap_crowd", "jokes_community",
             "mathse_accepted_verdict", "mathse_vote_score", "aops_curation",
             "code_v3", "press_verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True)
    ap.add_argument("--cell", action="append", default=None)
    args = ap.parse_args()
    cells = args.cell or ALL_CELLS
    ok, bad = [], {}
    for c in cells:
        try:
            run_cell(c, args.box)
            ok.append(c)
        except Exception as e:
            bad[c] = f"{type(e).__name__}: {e}"
            print(f"  [{c}/{args.box}] FAILED: {bad[c]}", flush=True)
    print("OK:", ok)
    print("FAILED:", json.dumps(bad, indent=2) if bad else "none")


if __name__ == "__main__":
    main()
