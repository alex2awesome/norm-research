#!/usr/bin/env python3
"""F2 DECONFOUNDED-FUSION battery — per-cell matrix adapters.

Frozen spec: notes/2026-08-09__full_sweep_queue.md §F2.  Each adapter returns, on
the cell's MASTER-LEDGER E rows and in the master ledger's row order:

    bank_enriched  the TERMINAL bank incl. promoted Track-A criteria (campaign's
                   own assembly code, imported -- never reimplemented)
    nuisance       the cell's Track-B spurious channels (Gemma-scored columns)
                   + declared STRUCT/observed-covariate columns where it has them
    names          block provenance for the results JSON

Row identity is established by ID against fusion/t0_rows/<cell>.npz (which was
itself assert-matched to results/vat_fullgrid_<cell>.json on n_E / n_groups_E / T),
and y is asserted elementwise.  No adapter may reorder E.

CPU only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TD = HERE.parent
CLOSURE = TD / "closure"


def _mod(path: Path, alias: str, extra_syspath=None):
    if extra_syspath:
        for p in extra_syspath:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


# ============================================================ jokes_community
def jokes_community():
    d_ = CLOSURE / "jokes_community"
    C = _mod(d_ / "cells.py", "f2_jokes_cells", [d_])
    TL = _mod(d_ / "terminal_ledger.py", "f2_jokes_tl", [d_])

    d = C.load("jokes_community")
    ids = [str(x) for x in d["ids"]]
    y = np.asarray(d["y"]).astype(int)
    groups = np.array([str(g) for g in d["groups"]], dtype=object)
    dsplit = np.asarray(d["dense_split"], dtype=object)

    # ---- terminal (enriched) bank: EXACTLY terminal_ledger.main()'s gated blocks
    tgt = json.loads((d_ / "jokes_community_gepa_targets.json").read_text())
    collapsed = {(c["round"], c["blind_id"]) for c in tgt["criteria"] if c["COLLAPSED"]}
    winners = json.loads((d_ / "jokes_community_gepa_selection.json").read_text())["winners"]
    blocks, names, dropped = [d["V"], d["A"]], ["V", "A_base"], []
    for r in ("1", "2", "3", "4", "5"):
        Xg, keep, drop = TL.load_round_A("jokes_community", r, collapsed, winners)
        dropped += [f"r{r}:{b}" for b in drop]
        if Xg.shape[1]:
            blocks.append(Xg)
            names.append(f"A_round{r}({Xg.shape[1]})")
    bank = np.column_stack([np.asarray(b, dtype=float) for b in blocks])

    # ---- nuisance: every Track-B routed column, all rounds, + created_utc STRUCT
    nb, nn, strict_flags = [], [], []
    for r in ("1", "2", "3", "4", "5"):
        rt = d_ / f"jokes_community_r{r}_routing_final.json"
        sc = d_ / f"jokes_community_r{r}_scores.npz"
        if not (rt.exists() and sc.exists()):
            continue
        rows = json.loads(rt.read_text())["final"]
        z = np.load(sc, allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        B = [x for x in rows if x["final_route"] == "B" and x["blind_id"] in cids]
        if not B:
            continue
        nb.append(z["X"][:, [cids.index(x["blind_id"]) for x in B]].astype(float))
        nn += [f"r{r}:{x['blind_id']}:{x.get('name', '')[:60]}" for x in B]
        strict_flags += [not bool(x.get("mixed", False)) for x in B]
    nuis_gemma = np.column_stack(nb) if nb else np.zeros((len(y), 0))
    struct = np.asarray(d["created_utc"], dtype=float).reshape(-1, 1)
    nuis = np.column_stack([nuis_gemma, struct])
    nn_all = nn + ["STRUCT:created_utc (observed covariate, 86.1% matched, NaN never imputed)"]
    strict_all = strict_flags + [True]

    return dict(
        cell="jokes_community", ids=ids, y=y, groups=groups,
        E=np.isin(dsplit, ["eval", "test"]),
        bank=bank, bank_names=names, nuis=nuis, nuis_names=nn_all,
        nuis_strict=np.array(strict_all, dtype=bool),
        n_nuis_gemma=int(nuis_gemma.shape[1]), n_struct=1,
        collapse_gate_dropped=dropped,
        provenance={
            "bank_enriched": ("closure/jokes_community/terminal_ledger.py load_round_A, "
                              "collapse-gated + GEPA winners swapped, rounds 1-5 "
                              "(the TERMINAL_LEDGER bank verbatim)"),
            "nuisance": ("all Track-B routed criteria from "
                         "jokes_community_r{1..5}_routing_final.json, columns from the "
                         "matching *_scores.npz, plus the cell's declared observed "
                         "covariate created_utc (FREEZE-ADDENDUM-4 position-in-container)"),
            "terminal_ledger_ref": "closure/jokes_community/jokes_community_TERMINAL_LEDGER.json",
        },
    )


ADAPTERS = {"jokes_community": jokes_community}


# ============================================================ generic adapter
# Every closure campaign after jokes_community follows one contract:
#   cells.py  load(...) -> dict(ids, y, groups, A, V, dense_split)
#   <prefix>_routing_final.json per round, entries carrying final_route in {A,B}
#     (and, on most cells, a `mixed` flag on B-routed channels)
#   <prefix>_scores.npz per round, keys X / crit_ids
# The terminal (enriched) bank is [V, A_base] + every A-ROUTED round block, with the
# cumulative collapse gate and the GEPA winner swap applied ONLY where that campaign
# produced those artifacts (recorded per cell in `conventions`).

GENERIC = {
    "press_verdict": dict(dir="press_verdict", tag="press_verdict",
                          pat="press_verdict_r%s", rounds=("1", "2")),
    "peer_curation": dict(dir="peer_curation_ext", tag="peer_curation",
                          pat="peer_curation_r%s", rounds=("1", "2", "3", "4", "5"),
                          loader_arg="peer_curation"),
    "peer_revealed": dict(dir="peer_revealed", tag="peer_revealed",
                          pat="peer_revealed_r%s", rounds=("1", "2", "3", "4", "5"),
                          loader_arg="peer_revealed"),
    "hashtagwars_verdict": dict(dir="maps_hw_si", tag="hashtagwars_verdict",
                                pat="hashtagwars_verdict_r%s", rounds=("1", "2", "3", "4"),
                                loader_arg="hashtagwars_verdict"),
    "mathse_accepted_verdict": dict(dir="mathse_accepted", tag="mathse_accepted",
                                    pat="mathse_accepted_r%s", rounds=("1", "2"),
                                    loader_arg="mathse_accepted"),
    "mathse_vote_score": dict(dir="mathse_vote", tag="mathse_vote",
                              pat="mathse_vote_r%s", rounds=("1", "2", "3"),
                              loader_arg="mathse_vote"),
    "cap_finalist": dict(dir="cap_finalist", tag="cap_finalist",
                         pat="cap_finalist_r%s", rounds=("1", "2", "4", "5"),
                         loader_arg="cap_finalist",
                         gepa_targets="cap_finalist_gepa_targets.json"),
    "nc_responded": dict(dir="nc_responded", tag="nc_responded", pat="round%s",
                         rounds=("1", "2", "3", "4", "5"), loader="lib"),
}


def _round_blocks(d_, pat, rounds, collapsed, n_rows):
    """A-routed and B-routed column blocks per round, in round order."""
    Ab, An, Bb, Bn, Bstrict, dropped = [], [], [], [], [], []
    for r in rounds:
        rt = d_ / f"{pat % r}_routing_final.json"
        sc = d_ / f"{pat % r}_scores.npz"
        if not (rt.exists() and sc.exists()):
            continue
        rows = json.loads(rt.read_text())["final"]
        z = np.load(sc, allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        assert z["X"].shape[0] == n_rows, f"{rt.name}: {z['X'].shape[0]} rows != {n_rows}"
        A = [x for x in rows if x["final_route"] == "A" and x["blind_id"] in cids
             and (r, x["blind_id"]) not in collapsed]
        dropped += [f"r{r}:{x['blind_id']}" for x in rows
                    if x["final_route"] == "A" and (r, x["blind_id"]) in collapsed]
        B = [x for x in rows if x["final_route"] == "B" and x["blind_id"] in cids]
        if A:
            Ab.append(z["X"][:, [cids.index(x["blind_id"]) for x in A]].astype(float))
            An.append(f"A_round{r}({len(A)})")
        if B:
            Bb.append(z["X"][:, [cids.index(x["blind_id"]) for x in B]].astype(float))
            Bn += [f"r{r}:{x['blind_id']}:{str(x.get('name', ''))[:60]}" for x in B]
            Bstrict += [not bool(x.get("mixed", False)) for x in B]
    return Ab, An, Bb, Bn, Bstrict, dropped


def _generic(cell):
    cfg = GENERIC[cell]
    d_ = CLOSURE / cfg["dir"]
    if cfg.get("loader") == "lib":
        L = _mod(d_ / "nc_closure_lib.py", f"f2_{cell}_lib", [d_])
        pop = L.load_population()
        _s, _sp, dsplit, _m, _mf = L.load_splits()
        ids = [str(x) for x in pop["doc_id"]]
        y = np.asarray(pop["y"]).astype(int)
        groups = np.array([str(g) for g in pop["docket"]], dtype=object)
        A, V = np.asarray(pop["A"], dtype=float), np.asarray(pop["V"], dtype=float)
        dsplit = np.asarray(dsplit, dtype=object)
    else:
        C = _mod(d_ / "cells.py", f"f2_{cell}_cells", [d_])
        d = C.load(cfg["loader_arg"]) if cfg.get("loader_arg") else C.load()
        ids = [str(x) for x in d["ids"]]
        y = np.asarray(d["y"]).astype(int)
        groups = np.array([str(g) for g in d["groups"]], dtype=object)
        A, V = np.asarray(d["A"], dtype=float), np.asarray(d["V"], dtype=float)
        dsplit = np.asarray(d["dense_split"], dtype=object)

    collapsed, conv = set(), []
    gt = cfg.get("gepa_targets")
    if gt and (d_ / gt).exists():
        j = json.loads((d_ / gt).read_text())
        if isinstance(j, dict) and "criteria" in j:
            collapsed = {(c["round"], c["blind_id"]) for c in j["criteria"] if c.get("COLLAPSED")}
            conv.append(f"cumulative collapse gate applied from {gt} ({len(collapsed)} excluded)")
    if not collapsed:
        conv.append("no cumulative collapse-gate artifact for this campaign -- none applied")
    conv.append("no GEPA winner-column swap artifact for this campaign -- incumbents stand"
                if not (d_ / f"{cfg['tag']}_gepa_selection.json").exists() else
                "GEPA winner swap available (not applied by the generic path)")

    Ab, An, Bb, Bn, Bstrict, dropped = _round_blocks(
        d_, cfg["pat"], cfg["rounds"], collapsed, len(y))
    bank = np.column_stack([V, A] + Ab)
    names = ["V", "A_base"] + An
    nuis = np.column_stack(Bb) if Bb else np.zeros((len(y), 0))
    return dict(
        cell=cell, ids=ids, y=y, groups=groups,
        E=np.isin(dsplit, ["eval", "test"]),
        bank=bank, bank_names=names, nuis=nuis, nuis_names=Bn,
        nuis_strict=np.array(Bstrict, dtype=bool) if Bstrict else np.zeros(0, dtype=bool),
        n_nuis_gemma=int(nuis.shape[1]), n_struct=0,
        collapse_gate_dropped=dropped,
        provenance={
            "bank_enriched": (f"closure/{cfg['dir']}: [V, A_base] + every A-ROUTED round "
                              f"block from {cfg['pat'] % 'N'}_routing_final.json, "
                              f"rounds {','.join(cfg['rounds'])}"),
            "nuisance": (f"every Track-B routed criterion from the same routing files, "
                         f"columns from {cfg['pat'] % 'N'}_scores.npz"),
            "conventions": conv,
            "struct": "no declared STRUCT/observed-covariate column for this cell",
        },
    )


for _c in GENERIC:
    ADAPTERS[_c] = (lambda c=_c: _generic(c))


# ========================================== cw_community / peer_verdict (special)
# On these two cells the MASTER LEDGER's own bank is ALREADY the terminal enriched
# bank (cw's ledger row is the round8=round7 144-column bank; peer_verdict's is the
# round-4 227-column bank), so `bank_enriched` is taken from fusion/t0_rows/<cell>.npz
# verbatim -- identical matrix, no re-derivation, no risk of a second convention.
# Only the Track-B nuisance block has to be assembled here.

def _bank_from_t0_rows(cell):
    z = np.load(HERE / "t0_rows" / f"{cell}.npz", allow_pickle=True)
    return ([str(i) for i in z["ids"]], z["y"].astype(int),
            np.array([str(g) for g in z["groups"]], dtype=object),
            z["VA_raw"].astype(float))


def cw_community():
    d_ = CLOSURE / "cw_community"
    z0 = np.load(d_ / "round0_state.npz", allow_pickle=True)
    ids = [str(x) for x in z0["ids"]]
    y = z0["y"].astype(int)
    groups = np.array([str(g) for g in z0["groups"]], dtype=object)
    z7 = np.load(d_ / "round7_state.npz", allow_pickle=True)
    bank = z7["VA"].astype(float)          # terminal bank = the ledger's own arm
    nb, nn = [], []
    for r in range(1, 9):
        sc = d_ / f"round{r}_scores.npz"
        if not sc.exists():
            continue
        z = np.load(sc, allow_pickle=True)
        tr = [str(s) for s in z["tracks"]]
        names = [str(s) for s in z["names"]]
        cids = [str(s) for s in z["cids"]]
        assert list(str(i) for i in z["ids"]) == ids, f"cw round{r}: row order differs"
        jj = [k for k, s in enumerate(tr) if s == "B"]
        if jj:
            nb.append(z["X"][:, jj].astype(float))
            nn += [f"r{r}:{cids[k]}:{names[k][:60]}" for k in jj]
    nuis = np.column_stack(nb) if nb else np.zeros((len(y), 0))
    return dict(
        cell="cw_community", ids=ids, y=y, groups=groups,
        E=np.ones(len(y), dtype=bool),      # whole honest population is dense-held-out
        bank=bank, bank_names=["VA_round8(=round7 terminal, 144)"],
        nuis=nuis, nuis_names=nn,
        nuis_strict=np.ones(nuis.shape[1], dtype=bool),
        n_nuis_gemma=int(nuis.shape[1]), n_struct=0, collapse_gate_dropped=[],
        provenance={
            "bank_enriched": "closure/cw_community/round7_state.npz['VA'] -- the master "
                             "ledger's own terminal arm (round8 anchor gate FAILED, 0 "
                             "criteria admitted, so round8 bank IS round7's)",
            "nuisance": "every track=='B' criterion in round{1..8}_scores.npz",
            "conventions": ["no `mixed` flag on this campaign's routing -- all B treated strict"],
            "struct": "cw position covariates exist (cw_position_covariates.csv) but are "
                      "NOT declared STRUCT for the fused ledger; excluded and recorded",
        },
    )


def peer_verdict():
    d_ = CLOSURE
    ids, y, groups, bank = _bank_from_t0_rows("peer_verdict")   # round-4 terminal, 227 cols
    # the closure population is the 6,030-row union; B columns are pulled in its order
    sys.path.insert(0, str(d_))
    L = _mod(d_ / "closure_lib.py", "f2_pv_lib", [d_])
    pop = L.load_population()
    pop_ids = [str(s) for s in pop["ntitle"]]
    nb, nn, strict = [], [], []
    for r in (1, 2, 3, 4):
        rt = d_ / f"round{r}_routing_final.json"
        sc = d_ / f"round{r}_scores.npz"
        if not (rt.exists() and sc.exists()):
            continue
        rows = json.loads(rt.read_text())["final"]
        z = np.load(sc, allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        B = [x for x in rows if x["final_route"] == "B" and x["blind_id"] in cids]
        if B:
            nb.append(z["X"][:, [cids.index(x["blind_id"]) for x in B]].astype(float))
            nn += [f"r{r}:{x['blind_id']}:{str(x.get('name', ''))[:60]}" for x in B]
            strict += [not bool(x.get("mixed", False)) for x in B]
    nuis_pop = np.column_stack(nb) if nb else np.zeros((len(pop_ids), 0))
    assert nuis_pop.shape[0] == len(pop_ids), "peer_verdict: B rows != population rows"
    pos = {d: i for i, d in enumerate(pop_ids)}
    idx = np.array([pos[i] for i in ids])
    return dict(
        cell="peer_verdict", ids=ids, y=y, groups=groups,
        E=np.ones(len(y), dtype=bool),       # t0_rows is already E
        bank=bank, bank_names=["V+A+mined_rounds1-4 (227, master-ledger terminal arm)"],
        nuis=nuis_pop[idx], nuis_names=nn,
        nuis_strict=np.array(strict, dtype=bool),
        n_nuis_gemma=int(nuis_pop.shape[1]), n_struct=0, collapse_gate_dropped=[],
        provenance={
            "bank_enriched": "fusion/t0_rows/peer_verdict.npz VA_raw -- the master ledger's "
                             "round-4 terminal arm verbatim ([V, A, XA1..XA4])",
            "nuisance": "every Track-B routed criterion in closure/round{1..4}_routing_final"
                        ".json, columns from round{r}_scores.npz, joined to E by ntitle",
            "conventions": ["no `mixed` flag on this campaign's routing -- all B treated strict"],
            "struct": "no declared STRUCT column for this cell",
        },
    )


ADAPTERS["cw_community"] = cw_community
ADAPTERS["peer_verdict"] = peer_verdict




if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="jokes_community")
    a = ap.parse_args()
    d = ADAPTERS[a.cell]()
    E = d["E"]
    print(f"{d['cell']}: n_pop={len(d['y'])} n_E={E.sum()} groups_E={len(set(d['groups'][E]))} "
          f"bank={d['bank'].shape} nuis={d['nuis'].shape} "
          f"(gemma {d['n_nuis_gemma']} + struct {d['n_struct']}) "
          f"strictB={int(d['nuis_strict'].sum())} collapse_dropped={len(d['collapse_gate_dropped'])}")
    print("bank blocks:", d["bank_names"])
    print("bank NaN frac:", round(float(np.isnan(d['bank']).mean()), 4),
          "| nuis NaN frac:", round(float(np.isnan(d['nuis']).mean()), 4))
