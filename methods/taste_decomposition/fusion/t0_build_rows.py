#!/usr/bin/env python3
"""T0 (UNTRAINED-T) ARM, step 1: export the E rows of every master-ledger cell.

Frozen design: notes/2026-07-27__vat-run-registry.md, entry
"2026-08-08 -- FROZEN DESIGN (before any scoring): UNTRAINED-T FUSION ARM".

For each of the 16 cells in the master fused ledger this script rebuilds the
EXACT E population the ledger's VAT arm used -- by importing the ledger's own
loaders (`direction1_mirror.py`, `direction1_mirror2.py`, `direction1_stack.py`
and the closure cell adapters), never by reimplementing them -- and writes:

    fusion/t0_rows/<cell>.npz        ids, y, groups, dense (trained T), VA_raw
    fusion/t0_rows/<cell>.texts.jsonl.gz   {"id": ..., "text": ...} per E row
    fusion/t0_rows/<cell>.meta.json  family, group_column, n_E, checksums

n_E is asserted against results/vat_fullgrid_<cell>.json for every cell.

TEXT = the document the trained dense model T read (same source, same
pre-formatting).  Where the dense chain's own split CSV carries the text it is
used verbatim; otherwise the closure adapter's `texts` field (which the cell
docstrings state is "the SAME context block the Gemma A-judge and the dense
reader saw") is used.  Two cells' texts live only on sk3 (aops_curation,
code_v3) -- run with --cell on that box, or supply --text-only.

CPU only, no GPU, read-only w.r.t. every existing result file.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TD = HERE.parent
REPO = TD.parents[1]
CLOSURE = TD / "closure"
RESULTS = TD / "results"
OUT = HERE / "t0_rows"
OUT.mkdir(exist_ok=True)


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


D1 = load_module(HERE / "direction1_mirror.py", "d1_t0")
_E_mask = D1._E_mask
CELLS_BATCH1 = D1.CELLS_BATCH1
CELLS_HWSI = D1.CELLS_HWSI


# ------------------------------------------------------------------ writer --
def emit(cell, ids, y, groups, dense, VA_raw, texts, family, group_column,
         population, extra=None):
    ids = [str(x) for x in ids]
    y = np.asarray(y).astype(int)
    groups = np.asarray([str(g) for g in groups], dtype=object)
    dense = np.asarray(dense, dtype=float)
    VA_raw = np.asarray(VA_raw, dtype=float)
    texts = [("" if t is None else str(t)) for t in texts]
    n = len(ids)
    assert len(y) == n and len(groups) == n and len(dense) == n and VA_raw.shape[0] == n \
        and len(texts) == n, (cell, n, len(y), len(groups), len(dense), VA_raw.shape, len(texts))
    assert np.isfinite(dense).all(), f"{cell}: non-finite trained-T column on E"
    # `ids` are the cell's NATURAL row keys and are not always unique (peer_verdict
    # has 5 repeated ntitles among its 1,244 E rows -- n_groups_E 1,239 in the
    # ledger).  `uid` = position-prefixed id is the unique key every downstream
    # join (prompts, scores) uses; row ORDER is the contract.
    n_dup_ids = n - len(set(ids))
    uids = [f"{i:06d}|{d}" for i, d in enumerate(ids)]
    n_empty = sum(1 for t in texts if not t.strip())

    ledger = json.loads((RESULTS / f"vat_fullgrid_{cell}.json").read_text())
    assert n == ledger["n_E"], f"{cell}: n_E {n} != ledger {ledger['n_E']}"
    ng = int(len(set(groups)))
    assert ng == ledger["n_groups_E"], f"{cell}: n_groups {ng} != ledger {ledger['n_groups_E']}"
    T_here = _auc(y, dense)
    T_led = ledger.get("T")

    np.savez_compressed(OUT / f"{cell}.npz", ids=np.array(ids, dtype=object),
                        uids=np.array(uids, dtype=object),
                        y=y, groups=groups, dense=dense, VA_raw=VA_raw)
    with gzip.open(OUT / f"{cell}.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for u, i, t in zip(uids, ids, texts):
            fh.write(json.dumps({"uid": u, "id": i, "text": t}) + "\n")

    meta = {
        "cell": cell, "n_E": n, "n_groups_E": ng, "pos_rate_E": float(y.mean()),
        "family": family, "group_column": group_column, "population": population,
        "n_features_VA_raw": int(VA_raw.shape[1]),
        "T_recomputed_on_E": T_here, "T_ledger": T_led,
        "T_abs_diff": None if T_led is None else abs(T_here - T_led),
        "n_duplicate_natural_ids": int(n_dup_ids),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "ids_sorted_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "texts_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
        "n_empty_texts": n_empty,
        "text_char_len": {"min": int(min(len(t) for t in texts)),
                          "median": float(np.median([len(t) for t in texts])),
                          "max": int(max(len(t) for t in texts))},
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        meta.update(extra)
    (OUT / f"{cell}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"  [{cell}] n_E={n} groups={ng} pos={y.mean():.3f} VA={VA_raw.shape[1]} "
          f"T={T_here:.4f} (ledger {T_led}) empty_texts={n_empty} "
          f"medlen={meta['text_char_len']['median']:.0f}", flush=True)
    return meta


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _split_texts(split_dir, id_col="row_id"):
    """id -> text from a dense-chain split dir's eval.csv + test.csv."""
    out = {}
    for sp in ("eval", "test"):
        d = pd.read_csv(Path(split_dir) / "split" / f"{sp}.csv")
        for i, t in zip(d[id_col].astype(str), d["text"].astype(str)):
            out[i] = t
    return out


# =========================================================== maps_batch1 (6)
BATCH1_FAMILY = {"peer_curation": "clean_once", "peer_revealed": "clean_once",
                 "nc_outcome": "clean_once", "nc_agree": "clean_once",
                 "cap_crowd": "clean_once", "cap_finalist": "clean_once"}
BATCH1_GROUPCOL = {"peer_curation": "ntitle", "peer_revealed": "ntitle",
                   "nc_outcome": "docket", "nc_agree": "docket",
                   "cap_crowd": "contest", "cap_finalist": "contest"}


def do_batch1(cell):
    d = CELLS_BATCH1.load(cell)
    dense = np.asarray(d["dense"], dtype=float)
    dsplit = np.asarray(d["dense_split"], dtype=object)
    E = _E_mask(dsplit) & np.isfinite(dense)
    V = np.asarray(d["V"], dtype=float)[E]
    A = np.asarray(d["A"], dtype=float)[E]
    VA = np.column_stack([V, A]) if V.shape[1] and A.shape[1] else (V if V.shape[1] else A)
    ids = [str(x) for x in np.asarray(d["ids"], dtype=object)[E]]
    texts = [d["texts"][i] for i in np.flatnonzero(E)]
    pop = ("dense-held-out rows (dense_split in {eval,test}) of the "
           f"{cell} Layer-1 matrix; closure/maps_batch1/cells.py adapter")
    return emit(cell, ids, np.asarray(d["y"])[E], np.asarray(d["groups"], dtype=object)[E],
                dense[E], VA, texts, BATCH1_FAMILY[cell], BATCH1_GROUPCOL[cell], pop,
                extra={"text_source": "closure/maps_batch1/cells.py `texts` "
                                      "(the item text the Gemma judge and dense reader saw)",
                       "n_features_raw": {"V": int(V.shape[1]), "A": int(A.shape[1])}})


# ============================================================== hashtagwars
def do_hashtagwars_verdict():
    d = CELLS_HWSI.load("hashtagwars_verdict")
    dsplit = np.asarray(d["dense_split"], dtype=object)
    E = _E_mask(dsplit)
    V = np.asarray(d["V"], dtype=float)[E]
    A = np.asarray(d["A"], dtype=float)[E]
    VA = np.column_stack([V, A])
    ids = [str(x) for x in np.asarray(d["ids"], dtype=object)[E]]
    texts = [d["texts"][i] for i in np.flatnonzero(E)]
    dense = np.asarray(d["dense"], dtype=float)[E]     # 3-seed mean = the ledger row
    return emit("hashtagwars_verdict", ids, np.asarray(d["y"])[E],
                np.asarray(d["groups"], dtype=object)[E], dense, VA, texts,
                "impute_perfold", "hashtag contest",
                "dense-held-out rows (dense_split in {eval,test}); maps_hw_si/cells.py adapter",
                extra={"text_source": "closure/maps_hw_si/cells.py `texts` "
                                      "(CONTEST HASHTAG header + TWEET, verbatim as the "
                                      "dense reader saw it)",
                       "dense_column": "mean of dense seeds {1,2,42} (the ledger's ensemble row)",
                       "n_features_raw": {"V": int(V.shape[1]), "A": int(A.shape[1])}})


# ============================================================== cw_community
def do_cw_community():
    cw = CLOSURE / "cw_community"
    z0 = np.load(cw / "round0_state.npz", allow_pickle=True)
    z7 = np.load(cw / "round7_state.npz", allow_pickle=True)
    assert (z0["y"] == z7["y"]).all() and (z0["ids"] == z7["ids"]).all()
    assert np.allclose(z0["T"], z7["T"])
    ids = [str(x) for x in z0["ids"]]
    VA8 = z7["VA"].astype(float)         # terminal arm: round8 gate FAILED == round7 bank
    popcsv = pd.read_csv(cw / "cw_honest_population.csv")
    tmap = dict(zip(popcsv["id"].astype(str), popcsv["text"].astype(str)))
    missing = [i for i in ids if i not in tmap]
    assert not missing, f"cw_community: {len(missing)} ids missing from cw_honest_population.csv"
    texts = [tmap[i] for i in ids]
    return emit("cw_community", ids, z0["y"].astype(int),
                [str(g) for g in z0["groups"]], z0["T"].astype(float), VA8, texts,
                "impute_perfold", "prompt_id (story prompt)",
                "cw_honest_population (n=7008), ALL rows dense_split in {eval,test}",
                extra={"text_source": "closure/cw_community/cw_honest_population.csv `text`",
                       "arm": "round8 (== round7 bank: round8 anchor gate FAILED, 0 criteria admitted)",
                       "n_features_raw": {"VA_round8": int(VA8.shape[1])}})


# ============================================================== peer_verdict
def do_peer_verdict():
    sys.path.insert(0, str(CLOSURE))
    import stage4_readout as SR4        # noqa: E402
    import stage4_round4 as S4R4        # noqa: E402

    pop, split, dsplit, XA1, XB1, a1_ids, b1_ids, summary = SR4.build_blocks()
    XA2, _, _, _, _ = S4R4.load_round_blocks(2)
    XA3, _, _, _, _ = S4R4.load_round_blocks(3)
    XA4, _, _, _, _ = S4R4.load_round_blocks(4)
    y_full, nt_full = pop["y"], pop["ntitle"]
    dp = pd.read_csv(CLOSURE / "peer_verdict_dense_preds.csv")
    assert len(dp) == len(y_full)
    assert (dp["dense_split"].values == dsplit).all()
    assert (dp["judgement"].astype(int).values == y_full).all()
    dense_full = dp["dense_prob"].astype(float).values

    E = np.isin(dsplit, ["eval", "test"])
    V, A = pop["V"][E], pop["A"][E]
    VA4 = np.column_stack([V, A, XA1[E], XA2[E], XA3[E], XA4[E]])
    ids = [str(s) for s in nt_full[E]]
    texts = [pop["texts"][i] for i in np.flatnonzero(E)]
    return emit("peer_verdict", ids, y_full[E], [str(s) for s in nt_full[E]],
                dense_full[E], VA4, texts, "clean_once", "ntitle",
                "dense-held-out rows (dense_split in {eval,test}); "
                "closure/peer_verdict_dense_preds.csv join by row order (asserted)",
                extra={"text_source": "closure/closure_lib.py load_population() `texts` "
                                      "(= datasets/peer-review/vat_3y/verdict.jsonl `text`)",
                       "arm": "round4 (V + A + 56 mined criteria, rounds 1-4)",
                       "n_features_raw": {"V": int(V.shape[1]), "A": int(A.shape[1]),
                                          "mined": int(VA4.shape[1] - V.shape[1] - A.shape[1])}})


# ============================================================= nc_responded
def do_nc_responded():
    ncr = CLOSURE / "nc_responded"
    sys.path.insert(0, str(ncr))
    import nc_closure_lib as NCL        # noqa: E402
    import readout as NCR               # noqa: E402

    pop = NCL.load_population()
    summary, split, dsplit, mining, monitor_full = NCL.load_splits()
    y_full = pop["y"]
    docket_full = np.array([str(s) for s in pop["docket"]], dtype=object)
    dp = pd.read_csv(ncr / "nc_responded_dense_preds_aligned.csv")
    pr = dict(zip(dp["doc_id"].astype(str), dp["dense_prob"].astype(float)))
    ds = dict(zip(dp["doc_id"].astype(str), dp["dense_split"].astype(str)))
    doc_id_full = pop["doc_id"]
    dense_full = np.array([pr.get(str(d), np.nan) for d in doc_id_full])
    assert (np.array([ds.get(str(d), "unmapped") for d in doc_id_full]) == dsplit).all()

    Xr, names_r = NCR.load_round_scores([1, 2, 3, 4, 5])
    E = np.isin(dsplit, ["eval", "test"])
    V, A = pop["V"][E], pop["A"][E]
    VA5 = np.column_stack([V, A, Xr[E]]) if Xr is not None else np.column_stack([V, A])
    ids = [str(d) for d in np.asarray(doc_id_full, dtype=object)[E]]
    texts_full = pop["texts"]
    texts = [str(texts_full[i]) for i in np.flatnonzero(E)]
    return emit("nc_responded", ids, y_full[E], docket_full[E], dense_full[E], VA5, texts,
                "clean_once", "docket",
                "dense-held-out rows (dense_split in {eval,test}); "
                "nc_responded_dense_preds_aligned.csv join by doc_id (asserted)",
                extra={"text_source": "closure/nc_responded/nc_closure_lib.py "
                                      "load_population() `texts` (comment body)",
                       "arm": "round5 (V + A + rounds 1-5 mined criteria)",
                       "n_features_raw": {"V": int(V.shape[1]), "A": int(A.shape[1]),
                                          "mined": int(VA5.shape[1] - V.shape[1] - A.shape[1])}})


# ========================================================= scale-up-wave-C (4)
D2 = None


def d2():
    global D2
    if D2 is None:
        D2 = load_module(HERE / "direction1_mirror2.py", "d2_t0")
    return D2


AOPS_DENSE_SPLIT = REPO / "runs/aops_same_approach_dense_llama8b/split_full"


def _aops_texts(pop_csv):
    """row_id -> the EXACT text the reused AoPS dense arm read.

    population.csv.gz carries (statement, body) but not the assembled text; the
    dense arm's own split_full/{eval,test}.csv carries `text` but no row_id.  The
    assembly is verified to be "Problem: {statement}\\n\\nSolution: {body}"
    (4,270/5,202 byte-exact, 5,202/5,202 after whitespace normalisation -- the
    remainder differ only by CSV whitespace round-tripping).  Each row_id is
    therefore mapped to the dense arm's byte-exact string via a whitespace-
    normalised key, falling back to the assembled string if absent.
    """
    import re
    p = pd.read_csv(pop_csv)
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    exact = {}
    if AOPS_DENSE_SPLIT.exists():
        for sp in ("eval", "test"):
            d = pd.read_csv(AOPS_DENSE_SPLIT / f"{sp}.csv")
            for t in d["text"].astype(str):
                exact.setdefault(norm(t), t)
    out, n_exact = {}, 0
    for rid, s, b in zip(p["row_id"].astype(str), p["statement"].astype(str),
                         p["body"].astype(str)):
        asm = "Problem: " + s + "\n\nSolution: " + b
        hit = exact.get(norm(asm))
        if hit is not None:
            n_exact += 1
        out[rid] = hit if hit is not None else asm
    print(f"    [aops text] {n_exact}/{len(out)} rows recovered byte-exact from the "
          f"dense arm's split_full; rest assembled from (statement, body)", flush=True)
    return out


SCC_TEXT = {
    "jokes_community": ("split_dir", REPO / "datasets/humor/reddit_jokes/dense_standard"),
    "mathse_accepted_verdict": ("split_dir",
                                REPO / "datasets/math/stackexchange/v2_va/dense_standard_mathse_accepted_verdict"),
    "mathse_vote_score": ("split_dir",
                          REPO / "datasets/math/stackexchange/v2_va/dense_standard_mathse_vote_score"),
    "aops_curation": ("population_csv", REPO / "datasets/math/aops/va/population.csv.gz"),
}


def do_scaleupC(cell):
    M2 = d2()
    SC = M2.scaleupC()
    d = SC.CELLS[cell]()
    A, V, y, groups, ids = d["A"], d["V"], d["y"], d["groups"], d["ids"]
    VA_raw_full = np.column_stack([V, A])
    meta = M2.SCALEUPC_META[cell]
    join = M2.load_join(meta["join_cell"])
    if cell == "aops_curation":
        probs, dsplit, ja = M2.align_join(cell, ids, y, groups, join, group_col="problem",
                                          prob_cols=["dense_prob"])
        ens = probs["dense_prob"]
    else:
        cols = [f"dense_prob_{s}" for s in meta["seeds"]]
        probs, dsplit, ja = M2.align_join(cell, ids, y, groups, join, prob_cols=cols)
        ens = np.mean(np.column_stack([probs[c] for c in cols]), axis=1)
    E = M2._E_mask(dsplit) & np.isfinite(ens)

    kind, src = SCC_TEXT[cell]
    if kind == "split_dir":
        tmap = _split_texts(src)
    else:
        tmap = _aops_texts(src)
    ids_E = [str(x) for x in np.asarray(ids, dtype=object)[E]]
    miss = [i for i in ids_E if i not in tmap]
    assert not miss, f"{cell}: {len(miss)}/{len(ids_E)} E ids missing text (e.g. {miss[:3]})"
    texts = [tmap[i] for i in ids_E]
    return emit(cell, ids_E, np.asarray(y)[E], np.asarray(groups, dtype=object)[E],
                ens[E], VA_raw_full[E], texts, "impute_perfold", meta["group_column"],
                f"dense-held-out rows of the {cell} scale-up-wave-C bank; "
                "joined to the dense chain BY ID (row_id)",
                extra={"text_source": str(src),
                       "dense_column": ("mean of seeds " + ",".join(meta["seeds"])),
                       "n_features_raw": {"V": int(V.shape[1]), "A": int(A.shape[1])}})


# ============================================================= press_verdict
def do_press_verdict():
    M2 = d2()
    sys.path.insert(0, str(TD))
    PV = load_module(TD / "press_verdict_layer1.py", "press_l1_t0")
    ids, y, comp, topic, levels, applicable = PV.load_population()
    y = np.asarray(y).astype(int)
    Vraw, v_names_raw = PV.build_v_matrix(ids)
    A_imp = np.where(applicable, levels, 0.5)
    VA_raw_full = np.column_stack([Vraw, A_imp])
    join = M2.load_join("press_verdict")
    seeds = ["seed42", "seed1", "seed2"]
    probs, dsplit, ja = M2.align_join("press_verdict", ids, y, comp, join,
                                      prob_cols=[f"dense_prob_{s}" for s in seeds])
    ens = np.mean(np.column_stack([probs[f"dense_prob_{s}"] for s in seeds]), axis=1)
    E = M2._E_mask(dsplit) & np.isfinite(ens)
    tmap = _split_texts(REPO / "datasets/press-releases/dense_standard_k3")
    ids_E = [str(x) for x in np.asarray(ids, dtype=object)[E]]
    miss = [i for i in ids_E if i not in tmap]
    assert not miss, f"press_verdict: {len(miss)} E ids missing text"
    texts = [tmap[i] for i in ids_E]
    return emit("press_verdict", ids_E, y[E], np.asarray([str(c) for c in comp], dtype=object)[E],
                ens[E], VA_raw_full[E], texts, "clean_once", "company",
                "dense-held-out rows of the 2,956-row k>=3 press-verdict population",
                extra={"text_source": "datasets/press-releases/dense_standard_k3/split/{eval,test}.csv",
                       "dense_column": "mean of seeds 42,1,2",
                       "n_features_raw": {"V": int(Vraw.shape[1]), "A": int(A_imp.shape[1])}})


# =================================================================== code_v3
def do_code_v3():
    CC = load_module(CLOSURE / "code_v3" / "cells_code.py", "cells_code_t0")
    d = CC.load()
    ids = np.asarray([str(x) for x in d["ids"]], dtype=object)
    groups = np.asarray([str(g) for g in d["groups"]], dtype=object)
    y = np.asarray(d["y"]).astype(int)
    split = np.asarray([str(s) for s in d["split"]], dtype=object)
    A = np.asarray(d["A"], dtype=float)
    V = np.asarray(d["V"], dtype=float)
    n_exec = len(d["v_exec"])
    dense42 = np.asarray(d["dense_seed42"], dtype=float)
    A_with_ind = np.column_stack([A, (~np.isnan(A)).astype(float)])
    VA_raw_full = np.column_stack([V[:, :n_exec], V[:, n_exec:], A_with_ind])
    E = _E_mask(split) & np.isfinite(dense42)
    assert E.all(), "code_v3: expected every row dense-held-out"

    sd = REPO / "datasets/code-review/dense_standard_v3"
    tmap = {}
    for sp in ("eval", "test"):
        df = pd.read_csv(sd / "split" / f"{sp}.csv")
        key = df["repo"].astype(str) + "/" + df["pr_number"].astype(str)
        for k, t in zip(key, df["text"].astype(str)):
            tmap[k] = t
    ids_E = [str(x) for x in ids[E]]
    miss = [i for i in ids_E if i not in tmap]
    assert not miss, f"code_v3: {len(miss)}/{len(ids_E)} E ids missing text (e.g. {miss[:3]})"
    texts = [tmap[i] for i in ids_E]
    return emit("code_v3", ids_E, y[E], groups[E], dense42[E], VA_raw_full[E], texts,
                "clean_once", "repository",
                "ALL 11,452 rows of the code_v3 closure population (all dense-held-out)",
                extra={"text_source": str(sd / "split/{eval,test}.csv"),
                       "dense_column": "seed42 (only seed with per-row preds)",
                       "split": {sp: int((split[E] == sp).sum()) for sp in ("eval", "test")},
                       "POOLED_DO_NOT_QUOTE": True,
                       "n_features_raw": {"V_exec": n_exec, "V_text": V.shape[1] - n_exec,
                                          "A_scores": A.shape[1],
                                          "A_applied_indicators": A.shape[1]}})


# ====================================================================== main
CELL_FNS = {
    "peer_verdict": do_peer_verdict,
    "peer_curation": lambda: do_batch1("peer_curation"),
    "peer_revealed": lambda: do_batch1("peer_revealed"),
    "nc_responded": do_nc_responded,
    "nc_outcome": lambda: do_batch1("nc_outcome"),
    "nc_agree": lambda: do_batch1("nc_agree"),
    "cw_community": do_cw_community,
    "hashtagwars_verdict": do_hashtagwars_verdict,
    "cap_finalist": lambda: do_batch1("cap_finalist"),
    "cap_crowd": lambda: do_batch1("cap_crowd"),
    "jokes_community": lambda: do_scaleupC("jokes_community"),
    "mathse_accepted_verdict": lambda: do_scaleupC("mathse_accepted_verdict"),
    "mathse_vote_score": lambda: do_scaleupC("mathse_vote_score"),
    "aops_curation": lambda: do_scaleupC("aops_curation"),
    "code_v3": do_code_v3,
    "press_verdict": do_press_verdict,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", default=None, choices=list(CELL_FNS))
    args = ap.parse_args()
    cells = args.cell if args.cell else list(CELL_FNS)
    ok, bad = [], {}
    for cell in cells:
        t0 = time.time()
        print(f"=== {cell} ===", flush=True)
        try:
            CELL_FNS[cell]()
            ok.append(cell)
        except Exception as e:
            bad[cell] = f"{type(e).__name__}: {e}"
            print(f"  [{cell}] FAILED: {bad[cell]}", flush=True)
        print(f"  [{cell}] {time.time()-t0:.0f}s\n", flush=True)
    print("OK:", ok)
    print("FAILED:", json.dumps(bad, indent=2) if bad else "none")


if __name__ == "__main__":
    main()
