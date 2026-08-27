#!/usr/bin/env python3
"""Harvest one RAW-MATCHED displacement-control cell into
results/v3_rawmatched_<slug>.json.

THE EXPERIMENT
--------------
On four long-document cells the k=20 criterion block displaces document text:
prepending it pushes truncation from ~2% to .28-.45 of rows at max_len 1024.
The V3 and raw/T arms therefore did NOT read the same amount of document, and
the confound runs in the direction that makes V3 look worse -- which is exactly
where V3 landed NEGATIVE against its bank on the two N&C cells.

The raw-matched arm removes that confound: same rows, same splits, same recipe,
no block, document truncated to 1024 - block_tokens.  Three numbers are emitted
on E = eval + test pooled (the same E the V3 arm used):

    V3 - raw_matched      the headline -- the criterion block's contribution at
                          a MATCHED document budget
    raw_matched - T_orig  what the lost text cost on its own
    V3 - VA_nl            re-stated verbatim from results/v3_grid_<slug>.json

Both bootstraps are group-level paired, 2,000 draws, resampling the cell's
canonical grouping unit (docket for N&C, company for press), using the SAME
estimator as harvest_v3_grid_cell.group_paired_boot.

GATES (a null with a stated reason beats an ungated CI)
-------------------------------------------------------
Every per-row vector that gets differenced must first pass an alignment gate:
  * preds row count == split row count, and judgement equal ELEMENTWISE
    (per split, for both the raw-matched arm and the V3 arm)
  * the raw-matched arm's split rows are byte-identical to the V3 arm's:
    same did in the same order, same group, same judgement
  * T is joined by did and asserted judgement-equal elementwise
  * row counts are re-asserted after every join (the `.loc`-on-duplicated-index
    defect silently EXPANDS row sets)
Any failure => that bootstrap is emitted as null with `*_reason` set.

Usage:
    python3 harvest_v3_rawmatched_cell.py --slug nc_agree
    python3 harvest_v3_rawmatched_cell.py --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
FUS = HERE.parent
TD = FUS.parent
RESULTS = TD / "results"
DATA = FUS / "dense_data"

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("v3h", HERE / "harvest_v3_grid_cell.py")
_v3h = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v3h)
group_paired_boot = _v3h.group_paired_boot      # identical estimator, 2000 draws

CELLS = ("nc_agree", "nc_outcome", "nc_responded", "press_verdict")

SLIM = TD / "closure" / "samerows_preds"
JOINS = FUS / "dense_joins"


def t_vector_gated(slug, dids, y_ref):
    """Per-row dense-T prediction aligned to `dids`, plus a gate dict.

    Returns (vals, judgement, gate).  `vals` is None whenever the gate fails.

    WHY NOT harvest_v3_grid_cell.t_vector DIRECTLY
    ----------------------------------------------
    For the N&C cells that function routes through
    closure/maps_batch1/cells.py::load(), which reconstructs the whole Layer-1
    cell (NCData + V-features over every text) purely to end up reading
    `closure/samerows_preds/<slug>_dense_preds_slim.csv` and indexing
    `dense_prob` by doc_id.  That costs tens of minutes and >24 GB of RAM.  We
    read the same CSV directly and then PROVE the shortcut is exact by
    reproducing the published `T_same_rows_at_E` that the adapter path produced
    in results/v3_grid_<slug>.json (tol 1e-6).  press_verdict already used the
    fast dense_joins path inside t_vector, so it is reproduced here unchanged.

    Gates applied before any vector is returned:
      * key present for every E row, unambiguous (duplicate keys dropped)
      * row count preserved by the reindex (the `.loc`-on-duplicated-index
        defect silently EXPANDS row sets)
      * no NaN probabilities
      * judgement equal ELEMENTWISE to the arm's own y
      * no E row was a dense-TRAIN row (`in_dense_train`), which would inflate T
      * AUC reproduces the published T_same_rows_at_E where one exists
    """
    gate = {"attempted": True, "slug": slug}
    n = len(dids)

    if slug == "press_verdict":
        jf = JOINS / "press_verdict_join.csv.gz"
        if not jf.exists():
            gate.update(passed=False, reason=f"{jf.name} not found")
            return None, None, gate
        j = pd.read_csv(jf)
        j["row_id"] = j.row_id.astype(str)
        cols = [c for c in j.columns if c.startswith("dense_prob")]
        j = j.drop_duplicates("row_id", keep=False).set_index("row_id")
        sub = j.reindex(list(dids))
        vals = np.nanmean(sub[cols].to_numpy(dtype=float), axis=1)
        yy = sub["judgement"].to_numpy(dtype=float)
        gate["source"] = (f"fusion/dense_joins/press_verdict_join.csv.gz, mean of "
                          f"{','.join(cols)} (same vector and seed-ensemble as "
                          f"harvest_v3_grid_cell.t_vector)")
        gate["n_seeds_ensembled"] = len(cols)
        train_flag = None
    else:
        f = SLIM / f"{slug}_dense_preds_slim.csv"
        if not f.exists():
            gate.update(passed=False, reason=f"{f.name} not found")
            return None, None, gate
        b = pd.read_csv(f)
        b["doc_id"] = b.doc_id.astype(str)
        n_dup = int(b.doc_id.duplicated().sum())
        b = b.drop_duplicates("doc_id", keep=False).set_index("doc_id")
        sub = b.reindex(list(dids))
        vals = sub["dense_prob"].to_numpy(dtype=float)
        yy = sub["judgement"].to_numpy(dtype=float)
        train_flag = sub["in_dense_train"]
        gate["source"] = (f"closure/samerows_preds/{f.name} dense_prob indexed by "
                          f"doc_id -- the exact column "
                          f"closure/maps_batch1/cells.py::load('{slug}') reads")
        gate["n_duplicate_keys_dropped"] = n_dup

    # ---- structural gates -------------------------------------------------
    if len(sub) != n:
        gate.update(passed=False,
                    reason=f"reindex changed the row count {n} -> {len(sub)}")
        return None, None, gate
    n_miss = int(np.isnan(vals).sum())
    if n_miss:
        gate.update(passed=False,
                    reason=f"{n_miss} of {n} E rows have no unambiguous dense "
                           f"probability")
        return None, None, gate
    if not (np.nan_to_num(yy, nan=-1).astype(int) == np.asarray(y_ref)).all():
        gate.update(passed=False,
                    reason="T join UNSAFE: judgement disagrees elementwise "
                           "between the arm's split rows and the T source")
        return None, None, gate
    if train_flag is not None:
        n_train = int(train_flag.fillna(False).astype(bool).sum())
        gate["n_E_rows_that_were_dense_train"] = n_train
        if n_train:
            gate.update(passed=False,
                        reason=f"{n_train} of {n} E rows were TRAIN rows for the "
                               f"dense T arm -- T is not comparable on this row set")
            return None, None, gate

    # ---- reproduction gate against the published adapter-path number ------
    obs = float(roc_auc_score(y_ref, vals))
    gate["observed_T_same_rows_at_E"] = obs
    vg = RESULTS / f"v3_grid_{slug}.json"
    if vg.exists():
        pub = json.loads(vg.read_text()).get("T_same_rows_at_E")
        if pub is not None:
            d = abs(obs - float(pub))
            gate.update(published_T_same_rows_at_E=float(pub),
                        reproduction_delta=d, tol=1e-6,
                        reproduced_published_T=bool(d < 1e-6))
            if d >= 1e-6:
                gate.update(passed=False,
                            reason=(f"fast T path does not reproduce the published "
                                    f"T_same_rows_at_E ({obs:.8f} vs {pub}) -- "
                                    f"refusing to difference against a vector that "
                                    f"is not the published dense arm"))
                return None, None, gate
        else:
            gate["reproduction_gate"] = ("results/v3_grid_%s.json has no "
                                         "T_same_rows_at_E to gate against" % slug)
    else:
        gate["reproduction_gate"] = (f"no results/v3_grid_{slug}.json yet; "
                                     f"see second_source_check")

    # ---- nc_responded has no published V3 harvest: use a 2nd source -------
    alt = TD / "closure" / "nc_responded" / "nc_responded_dense_preds_aligned.csv"
    if slug == "nc_responded" and alt.exists():
        a = pd.read_csv(alt)
        a["doc_id"] = a.doc_id.astype(str)
        a = a.drop_duplicates("doc_id", keep=False).set_index("doc_id")
        sa = a.reindex(list(dids))
        av = sa["dense_prob"].to_numpy(dtype=float)
        mx = float(np.nanmax(np.abs(av - vals))) if not np.isnan(av).all() else None
        gate["second_source_check"] = {
            "source": ("closure/nc_responded/nc_responded_dense_preds_aligned.csv "
                       "-- the DEDICATED file harvest_v3_grid_cell.t_vector uses "
                       "for this cell"),
            "max_abs_elementwise_diff": mx,
            "auc_second_source": (float(roc_auc_score(y_ref, av))
                                  if not np.isnan(av).any() else None),
            "agrees": bool(mx is not None and mx < 1e-9),
        }
        if not gate["second_source_check"]["agrees"]:
            gate.update(passed=False,
                        reason="the two registered T sources for nc_responded "
                               "disagree elementwise")
            return None, None, gate

    gate["passed"] = True
    return vals, yy, gate


def load_arm(dirname: Path, tag: str):
    """(DataFrame[did, group, y, <tag>, split], notes) for one trained arm.

    Enforces the preds/split alignment gate per split.
    """
    run = dirname / "rm_out_seed42"
    frames = []
    for sp in ("eval", "test"):
        pr = pd.read_csv(run / f"preds_{sp}.csv")
        sf = pd.read_csv(dirname / "split" / f"{sp}.csv")
        if len(pr) != len(sf):
            raise AssertionError(
                f"{dirname.name}/{sp}: preds {len(pr)} vs split {len(sf)} rows")
        if not (pr.judgement.values == sf.judgement.values).all():
            raise AssertionError(
                f"{dirname.name}/{sp}: preds/split judgement mismatch elementwise")
        frames.append(pd.DataFrame({
            "did": sf["did"].astype(str).values,
            "group": sf["group"].astype(str).values,
            "y": sf["judgement"].astype(int).values,
            tag: pr["prob"].astype(float).values,
            "split": sp,
        }))
    return pd.concat(frames, ignore_index=True)


def harvest(slug: str, write: bool = True) -> dict:
    rm_dir = DATA / f"v3grid_{slug}_rawmatched"
    v3_dir = DATA / f"v3grid_{slug}"
    rman = json.loads((rm_dir / "manifest.json").read_text())
    vman = json.loads((v3_dir / "manifest.json").read_text())

    RM = load_arm(rm_dir, "rawmatched")
    n_E = len(RM)

    out = {
        "cell": slug,
        "arm": f"v3grid_{slug}_rawmatched",
        "role": rman["role"],
        "max_length": rman["max_length"],
        "budget_derivation": rman["budget_derivation"],
        "budget_equivalence": rman["budget_equivalence"],
        "selection_split": rman["selection_split"],
        "group_column": rman["group_column"],
        "class_weight_auto": False,
        "recipe": rman["recipe"],
        "only_difference_from_v3": rman["only_difference_from_v3"],
        "n_E": int(n_E),
        "n_eval": int((RM.split == "eval").sum()),
        "n_test": int((RM.split == "test").sum()),
        "n_groups_E": int(RM.group.nunique()),
        "pos_rate_E": float(RM.y.mean()),
        "auc_E": float(roc_auc_score(RM.y, RM.rawmatched)),
        "auc_eval": float(roc_auc_score(RM[RM.split == "eval"].y,
                                        RM[RM.split == "eval"].rawmatched)),
        "auc_test": float(roc_auc_score(RM[RM.split == "test"].y,
                                        RM[RM.split == "test"].rawmatched)),
        "matching_assertions": {},
    }

    # ================= gate 1: raw-matched rows == V3 rows =================
    match = {"attempted": True}
    V3 = None
    try:
        V3 = load_arm(v3_dir, "v3")
    except (AssertionError, FileNotFoundError) as e:
        match.update(passed=False, reason=f"could not load the V3 arm: {e}")
    if V3 is not None:
        problems = []
        if len(V3) != len(RM):
            problems.append(f"n_E differs: V3 {len(V3)} vs raw-matched {len(RM)}")
        else:
            if list(V3.did) != list(RM.did):
                problems.append("did vectors differ (ids and/or order)")
            if list(V3.split) != list(RM.split):
                problems.append("split assignment differs")
            if not (V3.y.values == RM.y.values).all():
                problems.append("judgement differs elementwise")
            if list(V3.group) != list(RM.group):
                problems.append("group differs elementwise")
        match.update(
            passed=not problems,
            n_E_v3=int(len(V3)), n_E_rawmatched=int(len(RM)),
            checked=["n_E", "did (ids and order)", "split", "judgement elementwise",
                     "group elementwise"],
            source_byte_assertions=rman["byte_assertions"],
        )
        if problems:
            match["reason"] = "; ".join(problems)
    out["matching_assertions"]["rows_identical_to_v3"] = match

    # ================= headline: V3 - raw_matched ==========================
    if not match.get("passed"):
        out["boot_v3_minus_rawmatched"] = None
        out["boot_v3_minus_rawmatched_reason"] = match.get("reason")
        out["v3_auc_E"] = None
    else:
        J = RM.merge(V3[["did", "v3"]], on="did", how="inner", validate="one_to_one")
        assert len(J) == n_E, (
            f"{slug}: join changed the row count {n_E} -> {len(J)}")
        out["v3_auc_E"] = float(roc_auc_score(J.y, J.v3))
        out["boot_v3_minus_rawmatched"] = group_paired_boot(
            J.y.values, J.v3.values, J.rawmatched.values, J.group.values)
        out["boot_v3_minus_rawmatched_note"] = (
            "positive = the criterion block helps AT A MATCHED DOCUMENT BUDGET. "
            "Both arms read the same rows and the same amount of document, so "
            "this isolates the block's contribution from the displacement it "
            "causes.")

    # ================= raw_matched - T_original ============================
    tv, tY, tgate = t_vector_gated(slug, RM.did.tolist(), RM.y.values)
    out["T_gate"] = tgate
    out["T_source"] = tgate.get("source")
    if tv is None:
        out["boot_rawmatched_minus_T"] = None
        out["boot_rawmatched_minus_T_reason"] = tgate.get("reason")
        out["T_same_rows_at_E"] = None
    else:
        assert len(tv) == n_E, f"{slug}: T vector length {len(tv)} != n_E {n_E}"
        out["T_same_rows_at_E"] = float(roc_auc_score(RM.y, tv))
        out["n_E_for_T"] = int(n_E)
        out["boot_rawmatched_minus_T"] = group_paired_boot(
            RM.y.values, RM.rawmatched.values, tv, RM.group.values)
        out["boot_rawmatched_minus_T_note"] = (
            "T_original is the cell's full-budget dense arm (max_len 1024, raw "
            "text, IDENTICAL rows). NEGATIVE = the document text the block "
            "displaced was carrying signal; this prices the lost text on its own.")

    # ================= re-state V3 - VA_nl ================================
    vg = RESULTS / f"v3_grid_{slug}.json"
    if vg.exists():
        g = json.loads(vg.read_text())
        out["restated_from_v3_grid"] = {
            "source": str(vg),
            "v3_auc_E_published": g.get("auc_E"),
            "n_E_published": g.get("n_E"),
            "T_same_rows_at_E_published": g.get("T_same_rows_at_E"),
            "VA_nl_oof_at_E_observed": g.get("VA_nl_oof_at_E_observed"),
            "boot_v3_minus_VA_nl": g.get("boot_v3_minus_VA_nl"),
            "boot_v3_minus_VA_nl_reason": g.get("boot_v3_minus_VA_nl_reason"),
            "boot_v3_minus_T": g.get("boot_v3_minus_T"),
        }
        if out.get("v3_auc_E") is not None and g.get("auc_E") is not None:
            d = abs(out["v3_auc_E"] - g["auc_E"])
            out["restated_from_v3_grid"]["v3_auc_reproduction_delta"] = round(d, 8)
            out["restated_from_v3_grid"]["v3_auc_reproduces"] = bool(d < 1e-6)
    else:
        out["restated_from_v3_grid"] = {
            "source": None,
            "reason": f"results/v3_grid_{slug}.json not present on this machine"}

    out["k_caveat"] = _v3h.K_CAVEAT
    out["truncation"] = rman["truncation"]

    if write:
        p = RESULTS / f"v3_rawmatched_{slug}.json"
        p.write_text(json.dumps(out, indent=2, default=str))
        print("wrote", p)
    return out


def fmt_boot(b):
    if not b:
        return "null"
    return (f"{b['estimate']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] "
            f"P={b['p_gt_0']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    slugs = list(CELLS) if a.all else a.slug
    if not slugs:
        raise SystemExit("give --slug or --all")
    rows = []
    for s in slugs:
        try:
            o = harvest(s, write=not a.no_write)
        except FileNotFoundError as e:
            print(f"[{s}] SKIP (not ready): {e}")
            continue
        rows.append(o)
        print(f"\n=== {s}  n_E={o['n_E']} groups={o['n_groups_E']} "
              f"max_len={o['max_length']}")
        print(f"  T_original     {o.get('T_same_rows_at_E')}")
        print(f"  raw_matched    {o['auc_E']:.4f} "
              f"(eval {o['auc_eval']:.4f} / test {o['auc_test']:.4f})")
        print(f"  V3             {o.get('v3_auc_E')}")
        print(f"  V3 - rawmatched  {fmt_boot(o.get('boot_v3_minus_rawmatched'))}"
              + ("" if o.get("boot_v3_minus_rawmatched")
                 else "  REASON: " + str(o.get("boot_v3_minus_rawmatched_reason"))))
        print(f"  rawmatched - T   {fmt_boot(o.get('boot_rawmatched_minus_T'))}"
              + ("" if o.get("boot_rawmatched_minus_T")
                 else "  REASON: " + str(o.get("boot_rawmatched_minus_T_reason"))))
        r = o.get("restated_from_v3_grid") or {}
        print(f"  V3 - VA_nl       {fmt_boot(r.get('boot_v3_minus_VA_nl'))}")
        print(f"  rows==V3 gate    {o['matching_assertions']['rows_identical_to_v3'].get('passed')}")

    if len(rows) > 1:
        print("\n| cell | n_E | T_original | raw_matched | V3 | (V3-raw) [CI] P | "
              "(raw-T) [CI] P | V3-VA_nl |")
        print("|---|---:|---:|---:|---:|---|---|---|")
        for o in rows:
            r = o.get("restated_from_v3_grid") or {}
            t = o.get("T_same_rows_at_E")
            v = o.get("v3_auc_E")
            print(f"| {o['cell']} | {o['n_E']} | "
                  f"{('%.4f' % t) if t is not None else 'null'} | "
                  f"{o['auc_E']:.4f} | "
                  f"{('%.4f' % v) if v is not None else 'null'} | "
                  f"{fmt_boot(o.get('boot_v3_minus_rawmatched'))} | "
                  f"{fmt_boot(o.get('boot_rawmatched_minus_T'))} | "
                  f"{fmt_boot(r.get('boot_v3_minus_VA_nl'))} |")


if __name__ == "__main__":
    main()
