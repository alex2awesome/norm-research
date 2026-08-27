#!/usr/bin/env python3
"""RUNG 1 — selection regret on real items (design frozen 2026-08-21:
notes/2026-08-21__rung12_design_gap_consequences.md §1; run AFTER freeze).

Per F2 cell, on the same aligned E-frame and frozen Layer-1 stack as the
certified ladder: within each *decidable group* (>=2 items, both classes),
compare the item picked by the articulated-bank selector vs the dense
selector, judged by the REAL human label. No model arbitrates another model.

Readouts (design §1.3): top-1 hit rates (bank / dense / random floor),
disagreement win ratio (primary), swap rate, regret with group-bootstrap CI.
Convention inherited from f2_deconf/f2_gapci: decisions use the SEED-0 OOF
bank vector; the ledger's 3-seed mean is an AUC convention, not a per-row one.

CPU only.  Usage: python3 rung1_selection_regret.py [--cell C ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


F2 = _mod(HERE / "f2_deconf.py", "f2_deconf_r1")
fit_arm = F2.fit_arm

# static RELATIVE gap (Fig-3 convention: (best_any - best_articulated)/(best_any - .5)),
# from make_fig5_ranges.py V list, 2026-08-21 state. Cells absent from the
# Fig-3 22 are EXCLUDED from the headline correlation (flag why).
REL_GAP = {
    "cw_community":           43.5,
    "mathse_vote_score":      25.5,
    "jokes_community":        15.0,
    "hashtagwars_verdict":    12.5,
    "mathse_accepted_verdict": 8.3,
    "nc_responded":           13.3,
    "nc_outcome":             14.5,
    "peer_verdict":           39.4,
    "peer_curation":          36.0,
    "peer_revealed":          41.7,
    "press_verdict":          10.9,
    # cap_finalist: EXCLUDED (user 2026-08-20 dropped NYer caption cells from Fig 3)
    # nc_agree: EXCLUDED (design-conditional cell; no Fig-3 gap defined)
}


def group_indices(groups):
    idx = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    return idx


def pair_stats(y, b, t):
    """Closed-form over ALL pos-neg pairs; strict > wins, ties excluded."""
    p, n = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    Bd = b[p][:, None] - b[n][None, :]
    Td = t[p][:, None] - t[n][None, :]
    ok = (Bd != 0) & (Td != 0)
    npairs = int(ok.sum())
    bw, tw = Bd > 0, Td > 0                 # selector picked the true positive
    return dict(
        mode="pairwise_singleton", n_pairs=npairs,
        tie_mass=float(1 - npairs / (len(p) * len(n))),
        hit_bank=float((bw & ok).sum() / npairs),
        hit_dense=float((tw & ok).sum() / npairs),
        hit_random=.5,
        regret=float(((tw & ok).sum() - (bw & ok).sum()) / npairs),
        swap_rate=float(((bw != tw) & ok).sum() / npairs),
        dense_wins_on_disagree=int((tw & ~bw & ok).sum()),
        bank_wins_on_disagree=int((bw & ~tw & ok).sum()),
    )


def run_pairwise(cell, meta, y, b, t, r_bank, n_boot, seed, t0):
    point = pair_stats(y, b, t)
    rng = np.random.default_rng(seed)
    regs, ratios = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))       # item-level bootstrap
        if y[ix].min() == y[ix].max():
            continue
        s = pair_stats(y[ix], b[ix], t[ix])
        regs.append(s["regret"])
        ratios.append(s["dense_wins_on_disagree"] / max(s["bank_wins_on_disagree"], 1))
    out = {
        "cell": cell,
        "design": "notes/2026-08-21__rung12_design_gap_consequences.md §1 + ADDENDUM A",
        "n_E": int(len(y)), "n_groups": int(len(y)), "n_decidable": 0,
        "group_column": meta.get("group_column"),
        "rel_gap_fig3": REL_GAP.get(cell),
        "excluded_from_correlation": cell not in REL_GAP,
        "point": point,
        "regret_ci95": [float(np.percentile(regs, 2.5)), float(np.percentile(regs, 97.5))],
        "regret_p_gt_0": float(np.mean(np.array(regs) > 0)),
        "win_ratio_ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
        "bank_arm_auc_3seed": float(r_bank["VA_nl_mean"]),
        "convention": ("ADDENDUM A pairwise mode: all pos-neg pairs, strict wins, "
                       "ties excluded; item-level bootstrap; hit rates = pairwise AUC "
                       "(reported, NOT novel) -- disagreement ratio is the quotable stat"),
        "runtime_sec": time.time() - t0,
    }
    p = RESULTS / f"rung1_selection_regret_{cell}.json"
    p.write_text(json.dumps(out, indent=2))
    wr = f"{point['dense_wins_on_disagree']}:{point['bank_wins_on_disagree']}"
    print(f"  [{cell}] PAIRWISE n_pairs={point['n_pairs']} swap={point['swap_rate']:.2f} "
          f"hit b/d={point['hit_bank']:.3f}/{point['hit_dense']:.3f} "
          f"regret={point['regret']:+.3f} CI[{out['regret_ci95'][0]:+.3f},{out['regret_ci95'][1]:+.3f}] "
          f"disagree {wr} | {out['runtime_sec']:.0f}s", flush=True)
    return out


def run_cell(cell, n_boot=2000, seed=12345):
    t0 = time.time()
    meta, ids_E, y, groups, dense, t0col = F2.load_E(cell)
    a = F2.F2C.ADAPTERS[cell]()
    bank, nuis, join = F2.align(cell, a, ids_E, y, groups)
    r_bank = fit_arm(meta["family"], bank, dense, y, groups)
    b = r_bank["_oof_VA_nl0"]          # articulated selector (grouped-OOF, seed 0)
    t = np.asarray(dense, float)       # dense selector (same-rows T scores)
    y = np.asarray(y)

    gidx = group_indices(np.asarray(groups))
    dec = [np.array(ix) for ix in gidx.values()
           if len(ix) >= 2 and 0 < y[ix].sum() < len(ix)]

    if len(dec) <= 1:
        # ADDENDUM A (design doc, 2026-08-21): (near-)singleton-group cell
        # (peer cells: ntitle, e.g. peer_verdict 1239 groups / 1244 rows) ->
        # pairwise forced choice over ALL pos-neg pairs, deterministic.
        return run_pairwise(cell, meta, y, b, t, r_bank, n_boot, seed, t0)

    def stats(dec_groups):
        hb = hd = hr = swaps = d_win = b_win = 0
        for ix in dec_groups:
            pb, pd = ix[np.argmax(b[ix])], ix[np.argmax(t[ix])]
            hb += y[pb]; hd += y[pd]; hr += y[ix].mean()
            if pb != pd:
                swaps += 1
                if y[pd] == 1 and y[pb] == 0: d_win += 1
                if y[pb] == 1 and y[pd] == 0: b_win += 1
        n = len(dec_groups)
        return dict(n_decidable=n, hit_bank=hb / n, hit_dense=hd / n,
                    hit_random=hr / n, regret=(hd - hb) / n,
                    swap_rate=swaps / n, n_swaps=swaps,
                    dense_wins_on_disagree=d_win, bank_wins_on_disagree=b_win)

    point = stats(dec)

    rng = np.random.default_rng(seed)
    regs, ratios = [], []
    for _ in range(n_boot):
        bs = [dec[i] for i in rng.integers(0, len(dec), len(dec))]
        s = stats(bs)
        regs.append(s["regret"])
        dw, bw = s["dense_wins_on_disagree"], s["bank_wins_on_disagree"]
        ratios.append(dw / max(bw, 1))
    out = {
        "cell": cell, "design": "notes/2026-08-21__rung12_design_gap_consequences.md §1",
        "n_E": int(len(y)), "n_groups": len(gidx), "n_decidable": point["n_decidable"],
        "group_column": meta.get("group_column"),
        "rel_gap_fig3": REL_GAP.get(cell),
        "excluded_from_correlation": cell not in REL_GAP,
        "point": point,
        "regret_ci95": [float(np.percentile(regs, 2.5)), float(np.percentile(regs, 97.5))],
        "regret_p_gt_0": float(np.mean(np.array(regs) > 0)),
        "win_ratio_ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
        "bank_arm_auc_3seed": float(r_bank["VA_nl_mean"]),
        "convention": "seed-0 OOF bank vector for decisions; bootstrap over decidable groups",
        "runtime_sec": time.time() - t0,
    }
    p = RESULTS / f"rung1_selection_regret_{cell}.json"
    p.write_text(json.dumps(out, indent=2))
    wr = (f"{point['dense_wins_on_disagree']}:{point['bank_wins_on_disagree']}")
    print(f"  [{cell}] dec={point['n_decidable']} swap={point['swap_rate']:.2f} "
          f"hit b/d/r={point['hit_bank']:.3f}/{point['hit_dense']:.3f}/{point['hit_random']:.3f} "
          f"regret={point['regret']:+.3f} CI[{out['regret_ci95'][0]:+.3f},{out['regret_ci95'][1]:+.3f}] "
          f"disagree {wr} | {out['runtime_sec']:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    cells = args.cell or list(F2.F2C.ADAPTERS)
    rows = []
    for c in cells:
        print(f"=== RUNG1 {c} ===", flush=True)
        try:
            rows.append(run_cell(c, args.n_boot))
        except Exception as e:
            print(f"  [{c}] FAILED: {e}", flush=True)
    # cross-cell prereg readout: effect vs static relative gap
    ok = [r for r in rows if not r["excluded_from_correlation"]]
    if len(ok) >= 4:
        from scipy.stats import spearmanr
        x = [r["rel_gap_fig3"] for r in ok]
        for key, get in (("regret", lambda r: r["point"]["regret"]),
                         ("win_ratio", lambda r: (r["point"]["dense_wins_on_disagree"] /
                                                  max(r["point"]["bank_wins_on_disagree"], 1)))):
            rho, p = spearmanr(x, [get(r) for r in ok])
            print(f"cross-cell {key} vs rel_gap: rho={rho:.3f} p={p:.3f} n={len(ok)}")
    (RESULTS / "rung1_summary.json").write_text(json.dumps(rows, indent=2))
    print("wrote rung1_summary.json")


if __name__ == "__main__":
    main()
