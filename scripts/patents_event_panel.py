#!/usr/bin/env python3
"""Prosecution-event panel: dated per-app event history -> corrected outcome y's + examiner leniency.

Built from the 2026-07-13 dual audit (Codex gpt-5.6-sol + Fable) of the acceptance-prediction leg.
Both auditors' rank-1/2 recommendation: replace the undated boolean labels (ever-RCE, raw
n_office_actions counting advisory/Quayle rows) with one dated pass over PatEx transactions,
and join the examiner/art-unit fields that no script in the pipeline had ever read.

Event vocabulary (verified against raw/patex/event_codes.csv):
  substantive OA   CTNF (non-final), CTFR (final)          <- the only codes counted as "rounds"
  excluded from rounds: CTAV (advisory: exists only if applicant files after final = strategy
                    event), CTEQ (Quayle: substance already allowable) — counted separately
  allowance        MN/=., N/=., N/=A, N/=F
  abandonment      prefix ABN / MABN
  RCE              RCEX, BRCE, FRCE
  appeal           N/AP (Notice of Appeal Filed), AP.B (brief), APCA (pre-appeal decision:
                   rejection WITHDRAWN = examiner backed down), APCP (proceed to PTAB)

Panel y's (all dated, denominators = explicit risk sets):
  Y_first_action_allow  among apps w/ >=1 substantive event: allowance strictly before any
                        CTNF/CTFR (cleaner than labels.first_draft_approved, which counted
                        CTAV/CTEQ as OAs and required no substantive event at all)
  n_oa_rounds           deduped (app,code,date) CTNF+CTFR count — ordinal
  Y_another_round       among >=1 round: >=2 rounds ("will this get another iteration")
  Y_rce_after_final     among apps w/ CTFR: RCE dated after it and before any allowance
                        (excludes pre-final RCEs and post-NOA IDS-consideration RCEs that
                        polluted alty Y1 "ever-RCE")
  Y_appeal_after_final  among apps w/ CTFR: N/AP after it
  appeal_examiner_withdrew  among apps w/ a pre-appeal conference decision: APCA vs APCP
                        (near-ground-truth for "examiner was wrong")
  Y_abandon_after_2     among disposed apps w/ >=2 rounds: abandoned
  noa_before_abandon    abandoned apps whose first allowance PRECEDES abandonment — examiner
                        said yes, applicant walked (label-contamination flag for final_outcome;
                        measured ~3.3%% of cohort negatives in the audit)
  time_to_disposition_days / censored

Examiner leniency (leave-one-out, disposed apps only — the "patent lottery" control that was
absent from the whole pipeline): exm_loo_grant / au_loo_grant / auyr_loo_grant + group sizes.

Stages (sk3, CPU):
  build : one chunked pass over the 13GB transactions.csv -> event_cache parquet of per-app
          first-dates + deduped counts (no full event list retained)
  panel : merge application_data (examiner fields read for the FIRST time in this pipeline),
          derive y's + leniency -> processed/prosecution_event_panel.parquet
  report: base rates on (a) all disposed apps, (b) the 579K balanced final-outcome cohort,
          (c) the 21,447-app option3 extraction cohort
"""
import argparse, csv, gzip, json, os, sys
import numpy as np
import pandas as pd

csv.field_size_limit(2**31 - 1)   # cohort-A rows carry full claim texts

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
TXN = f"{BASE}/raw/patex/transactions.csv"
APPDATA = f"{BASE}/raw/patex/application_data.csv"
PROC = f"{BASE}/processed"
CACHE = f"{PROC}/event_panel_cache"
PANEL = f"{PROC}/prosecution_event_panel.parquet"
COHORT_A = f"{BASE}/patents_final_outcome_cpc_balanced_with_rejections.csv.gz"
COHORT_B = f"{PROC}/option3_claims_gemma_scale.jsonl"

SUBSTANTIVE = {"CTNF", "CTFR"}
OTHER_OA = {"CTAV", "CTEQ"}
ALLOW = {"MN/=.", "N/=.", "N/=A", "N/=F"}
RCE = {"RCEX", "BRCE", "FRCE"}
NAP = {"N/AP"}
EXACT = SUBSTANTIVE | OTHER_OA | ALLOW | RCE | NAP | {"AP.B", "APCA", "APCP"}
PREFIXES = ("ABN", "MABN")

FIRSTDATE_FAMILIES = {
    "ctnf": lambda c: c == "CTNF",
    "ctfr": lambda c: c == "CTFR",
    "allow": lambda c: c in ALLOW,
    "abn": lambda c: c.startswith(PREFIXES),
    "rce": lambda c: c in RCE,
    "nap": lambda c: c in NAP,
    "apbrief": lambda c: c == "AP.B",
    "apca": lambda c: c == "APCA",
    "apcp": lambda c: c == "APCP",
}


FAMILY_MASKS = {
    "ctnf": lambda c: c == "CTNF",
    "ctfr": lambda c: c == "CTFR",
    "allow": lambda c: c.isin(ALLOW),
    "abn": lambda c: c.str.startswith(PREFIXES),
    "rce": lambda c: c.isin(RCE),
    "nap": lambda c: c.isin(NAP),
    "apbrief": lambda c: c == "AP.B",
    "apca": lambda c: c == "APCA",
    "apcp": lambda c: c == "APCP",
}


def _merge_min(dst, ser):
    for an, d in ser.items():
        cur = dst.get(an)
        if cur is None or d < cur:
            dst[an] = d


def cmd_build(_):
    os.makedirs(CACHE, exist_ok=True)
    first = {k: {} for k in FAMILY_MASKS}
    other_counts = {"ctav": {}, "cteq": {}}
    seen_round = set()          # global (an, code, date) dedup for substantive rounds
    n_rows = n_kept = 0
    for chunk in pd.read_csv(TXN, chunksize=5_000_000, dtype=str):
        n_rows += len(chunk)
        code = chunk["event_code"].fillna("")
        keep = code.isin(EXACT) | code.str.startswith(PREFIXES)
        sub = chunk.loc[keep & chunk["recorded_date"].str.len().ge(8).fillna(False)]
        n_kept += len(sub)
        c = sub["event_code"]
        for fam, pred in FAMILY_MASKS.items():
            m = pred(c)
            if m.any():
                _merge_min(first[fam],
                           sub[m].groupby("application_number")["recorded_date"].min())
        subst = sub[c.isin(SUBSTANTIVE)]
        seen_round.update(zip(subst["application_number"], subst["event_code"],
                              subst["recorded_date"]))
        for key, cc in (("ctav", "CTAV"), ("cteq", "CTEQ")):
            cnt = sub[c == cc].groupby("application_number").size()
            dst = other_counts[key]
            for an, n in cnt.items():
                dst[an] = dst.get(an, 0) + int(n)
        if n_rows % 50_000_000 < 5_000_000:
            print(f"  scanned {n_rows:,} rows, kept {n_kept:,}", flush=True)
    # substantive round counts from the globally-deduped (an, code, date) set
    counts = {"ctnf": {}, "ctfr": {}, **other_counts}
    for an, code, _d in seen_round:
        k = "ctnf" if code == "CTNF" else "ctfr"
        counts[k][an] = counts[k].get(an, 0) + 1
    apps = set()
    for d in first.values():
        apps.update(d)
    for d in counts.values():
        apps.update(d)
    apps = sorted(apps)
    df = pd.DataFrame({"application_number": apps})
    for fam in FAMILY_MASKS:
        df[f"first_{fam}"] = df["application_number"].map(first[fam])
    for k in counts:
        df[f"n_{k}"] = df["application_number"].map(counts[k]).fillna(0).astype(int)
    df.to_parquet(f"{CACHE}/app_events.parquet")
    print(f"[build] {n_rows:,} txn rows scanned, {n_kept:,} relevant, "
          f"{len(df):,} apps with >=1 event", flush=True)
    print("PANEL_BUILD_DONE", flush=True)


def _loo(df, group_cols, ycol, prefix):
    g = df.groupby(group_cols)[ycol].agg(["sum", "count"])
    g.columns = [f"{prefix}_sum", f"{prefix}_n"]
    df = df.merge(g, left_on=group_cols, right_index=True, how="left")
    n = df[f"{prefix}_n"]
    df[f"{prefix}_loo_grant"] = np.where(
        n > 1, (df[f"{prefix}_sum"] - df[ycol]) / (n - 1), np.nan)
    return df.drop(columns=[f"{prefix}_sum"])


def cmd_panel(_):
    ev = pd.read_parquet(f"{CACHE}/app_events.parquet")
    for c in ev.columns:
        if c.startswith("first_"):
            ev[c] = pd.to_datetime(ev[c], errors="coerce")
    print(f"[panel] events for {len(ev):,} apps", flush=True)

    ad = pd.read_csv(APPDATA, usecols=[
        "application_number", "filing_date", "patent_number", "patent_issue_date",
        "appl_status_desc", "examiner_full_name", "examiner_art_unit",
        "application_invention_type"], dtype=str, on_bad_lines="skip", low_memory=False)
    ad = ad[ad["application_invention_type"].fillna("").str.upper().isin(
        ["UTILITY", "REISSUE", ""])]
    ad["is_granted"] = ad["patent_number"].notna() & (ad["patent_number"].str.strip() != "")
    status = ad["appl_status_desc"].fillna("").str.lower()
    ad["status_abandoned"] = status.str.contains("abandon", na=False)
    ad["filing_date"] = pd.to_datetime(ad["filing_date"], errors="coerce")
    ad["patent_issue_date"] = pd.to_datetime(ad["patent_issue_date"], errors="coerce")
    ad["filing_year"] = ad["filing_date"].dt.year
    print(f"[panel] application_data {len(ad):,} utility/reissue apps", flush=True)

    p = ad.merge(ev, on="application_number", how="left")
    for k in ("n_ctnf", "n_ctfr", "n_ctav", "n_cteq"):
        p[k] = p[k].fillna(0).astype(int)
    p["n_oa_rounds"] = p["n_ctnf"] + p["n_ctfr"]
    p["disposed"] = p["is_granted"] | p["status_abandoned"]
    p["final_outcome"] = np.select(
        [p["is_granted"], p["status_abandoned"]], ["granted", "abandoned"], "pending")

    first_sub = p[["first_ctnf", "first_ctfr"]].min(axis=1)
    has_sub = p["first_allow"].notna() | first_sub.notna()
    p["Y_first_action_allow"] = np.where(
        ~has_sub, np.nan,
        (p["first_allow"].notna() &
         (first_sub.isna() | (p["first_allow"] < first_sub))).astype(float))
    p["Y_another_round"] = np.where(
        p["n_oa_rounds"] >= 1, (p["n_oa_rounds"] >= 2).astype(float), np.nan)
    has_final = p["first_ctfr"].notna()
    p["Y_rce_after_final"] = np.where(
        ~has_final, np.nan,
        (p["first_rce"].notna() & (p["first_rce"] >= p["first_ctfr"]) &
         (p["first_allow"].isna() | (p["first_rce"] < p["first_allow"]))).astype(float))
    p["Y_appeal_after_final"] = np.where(
        ~has_final, np.nan,
        (p["first_nap"].notna() & (p["first_nap"] >= p["first_ctfr"])).astype(float))
    has_preappeal = p["first_apca"].notna() | p["first_apcp"].notna()
    p["appeal_examiner_withdrew"] = np.where(
        ~has_preappeal, np.nan,
        (p["first_apca"].notna() &
         (p["first_apcp"].isna() | (p["first_apca"] <= p["first_apcp"]))).astype(float))
    p["Y_abandon_after_2"] = np.where(
        p["disposed"] & (p["n_oa_rounds"] >= 2),
        p["status_abandoned"].astype(float), np.nan)
    p["noa_before_abandon"] = (p["status_abandoned"] & p["first_allow"].notna() &
                               (p["first_abn"].isna() |
                                (p["first_allow"] <= p["first_abn"])))
    disp_date = np.where(p["is_granted"], p["patent_issue_date"], p["first_abn"])
    p["time_to_disposition_days"] = (pd.to_datetime(pd.Series(disp_date)) -
                                     p["filing_date"]).dt.days
    p.loc[~p["disposed"], "time_to_disposition_days"] = np.nan

    # examiner / art-unit leniency, leave-one-out, disposed apps only
    d = p[p["disposed"]].copy()
    d["y_grant"] = d["is_granted"].astype(int)
    d = _loo(d, ["examiner_full_name"], "y_grant", "exm")
    d = _loo(d, ["examiner_art_unit"], "y_grant", "au")
    d["fy_str"] = d["filing_year"].fillna(-1).astype(int).astype(str)
    d = _loo(d, ["examiner_art_unit", "fy_str"], "y_grant", "auyr")
    len_cols = ["exm_loo_grant", "exm_n", "au_loo_grant", "au_n",
                "auyr_loo_grant", "auyr_n"]
    p = p.merge(d[["application_number"] + len_cols], on="application_number", how="left")

    keep = (["application_number", "filing_date", "filing_year", "final_outcome",
             "disposed", "is_granted", "status_abandoned",
             "examiner_full_name", "examiner_art_unit",
             "n_ctnf", "n_ctfr", "n_ctav", "n_cteq", "n_oa_rounds"]
            + [f"first_{f}" for f in FIRSTDATE_FAMILIES]
            + ["Y_first_action_allow", "Y_another_round", "Y_rce_after_final",
               "Y_appeal_after_final", "appeal_examiner_withdrew", "Y_abandon_after_2",
               "noa_before_abandon", "time_to_disposition_days"] + len_cols)
    p[keep].to_parquet(PANEL)
    print(f"[panel] wrote {len(p):,} apps -> {PANEL}", flush=True)
    print("PANEL_DONE", flush=True)


def _rates(p, name):
    ys = ["Y_first_action_allow", "Y_another_round", "Y_rce_after_final",
          "Y_appeal_after_final", "appeal_examiner_withdrew", "Y_abandon_after_2"]
    print(f"\n=== {name} (n={len(p):,}) ===", flush=True)
    for y in ys:
        v = p[y].dropna()
        print(f"  {y:26s} n={len(v):9,}  rate={v.mean():.3f}" if len(v) else
              f"  {y:26s} n=0", flush=True)
    if "noa_before_abandon" in p:
        ab = p[p["final_outcome"] == "abandoned"]
        if len(ab):
            print(f"  noa_before_abandon | abandoned: {ab['noa_before_abandon'].mean():.3f}",
                  flush=True)


def cmd_report(_):
    _csv = csv
    p = pd.read_parquet(PANEL)
    p["an_norm"] = p["application_number"].str.lstrip("0")
    _rates(p[p["disposed"]], "all disposed apps")
    a_ids = set()
    with gzip.open(COHORT_A, "rt") as f:
        for r in _csv.DictReader(f):
            a_ids.add(r["app_id"].lstrip("0"))
    _rates(p[p["an_norm"].isin(a_ids)], "cohort A: 579K balanced final-outcome")
    b_ids = set()
    with open(COHORT_B) as f:
        for ln in f:
            b_ids.add(str(json.loads(ln)["app_id"]).lstrip("0"))
    _rates(p[p["an_norm"].isin(b_ids)], "cohort B: option3 extraction (21,447 apps)")
    print("PANEL_REPORT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "panel", "report"):
        sub.add_parser(c)
    a = ap.parse_args()
    {"build": cmd_build, "panel": cmd_panel, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
