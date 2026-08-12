"""Step 4 — evaluate all arms on the FROZEN 40-item eval splits (the only step that
touches eval items).

Arms (auto-discovered from output files in roundtrip/):
  t0        output_rt_c<k>.py            (Sonnet crew one-shot, 2026-08-12 wave)
  t1,t2,... output_rt_c<k>_codex_t<t>.py (Codex one-shot trials)
  best-of   per rule, the trial arm with best TRAIN spearman (never eval-selected)
  gepa_r<r> output_gepa_r<r>_c<k>.py     (feedback-optimized; falls back per rule to
                                          best-of where a rule wasn't revised)
R_home = spearman(arm scores, original code-channel scores) on the eval ids stored in
reconstruction/detail/. Comparator: stored channel="judge" R (prompt arm's home trip).
Calibration rules are scored against reference implementations (60 PR texts).
Output: roundtrip/roundtrip_results.json + printed summary.
"""
import json
import re
import statistics as st

from common import (RECON, WORK, channel_scores_full, load_functions, load_items,
                    spearman)


def cal_refs():
    def c1(t): d = sum(c.isdigit() for c in t); return 10 if d >= 3 else (5 if d >= 1 else 0)
    def c2(t): return 10 if "?" in t else 0
    def c3(t): w = len(t.split()); return 10 if w > 150 else (5 if w >= 50 else 0)
    def c4(t): return 10 if any(ln.startswith("#") for ln in t.split("\n")) else 0
    def c5(t):
        n = sum(1 for w in re.findall(r"[A-Za-z]{3,}", t) if w.isupper())
        return 10 if n >= 5 else (5 if n >= 1 else 0)
    def c6(t):
        n = len(re.findall(r"\bthe\b", t.lower()))
        return 10 if n > 10 else (5 if n >= 3 else 0)
    return {"CAL1": c1, "CAL2": c2, "CAL3": c3, "CAL4": c4, "CAL5": c5, "CAL6": c6}


def eval_fn_on(fn, ids, items, target):
    got, want = [], []
    for d in ids:
        if d not in items or target.get(d) is None:
            continue
        try:
            s = float(fn(items[d]))
        except Exception:
            continue
        if s == s:
            got.append(s)
            want.append(float(target[d]))
    if len(got) < 25 or len(set(got)) <= 1 or len(set(want)) <= 1:
        return None
    # near-degeneracy screen (user concern 2026-08-12): a channel whose modal output
    # covers >85% of items has its rank statistic driven by a handful of off-modal
    # items — spuriously winnable on n=40. Screened, not scored.
    from collections import Counter
    if Counter(got).most_common(1)[0][1] / len(got) > 0.85:
        return "NEAR_DEGENERATE"
    return spearman(got, want)


def main():
    jobs = json.load(open(WORK / "jobs_full.json"))
    RJ = {}
    for line in open(RECON / "recon_results.jsonl"):
        r = json.loads(line)
        if "R" in r and r["channel"] == "judge":
            RJ[(r["task"], r["aspect"])] = r["R"]

    arms = {"t0": load_functions("output_rt_c[0-9].py")}
    t = 1
    while list(WORK.glob(f"output_rt_c*_codex_t{t}.py")):
        arms[f"t{t}"] = load_functions(f"output_rt_c*_codex_t{t}.py")
        t += 1
    r = 1
    gepa = {}
    while list(WORK.glob(f"output_gepa_r{r}_c*.py")):
        gepa.update(load_functions(f"output_gepa_r{r}_c*.py"))
        r += 1
    print("arms:", {a: len(f) for a, f in arms.items()}, "| gepa revisions:", len(gepa))

    items_cache = {}
    refs = cal_refs()
    cal_rows, results = [], []
    for j in jobs:
        jid = j["job_id"]
        if j["task"] == "CALIBRATION":
            texts = items_cache.setdefault("press_releases", load_items("press_releases"))
            sample = [v for _, v in sorted(texts.items())[:60]]
            want = [float(refs[j["aspect"]](x)) for x in sample]
            row = {"cal": j["aspect"]}
            for a, funcs in arms.items():
                fn = funcs.get(jid)
                if fn is None:
                    continue
                try:
                    got = [float(fn(x)) for x in sample]
                    rr = spearman(got, want) if len(set(want)) > 1 else float("nan")
                    row[a] = round(rr, 3) if rr == rr else None
                except Exception as e:
                    row[a] = f"ERR {str(e)[:40]}"
            cal_rows.append(row)
            continue
        items = items_cache.setdefault(j["task"], load_items(j["task"]))
        det = json.load(open(RECON / "detail" /
                             f"{j['task']}__{j['aspect']}__{j['channel']}.json"))
        eval_target = {d: det[d][1] for d in det}
        train_target, _ = channel_scores_full(j)
        row = {"task": j["task"], "aspect": j["aspect"], "channel": j["channel"],
               "R_mixed_code": j["R_mixed"], "R_judge": RJ.get((j["task"], j["aspect"]))}
        train_r = {}
        for a, funcs in arms.items():
            fn = funcs.get(jid)
            if fn is None:
                continue
            v = eval_fn_on(fn, sorted(det), items, eval_target)
            if v == "NEAR_DEGENERATE":
                row[f"neardeg_{a}"] = True
            else:
                row[f"R_home_{a}"] = round(v, 3) if v is not None else None
            tr = eval_fn_on(fn, sorted(train_target), items, train_target)
            if tr is not None and tr != "NEAR_DEGENERATE":
                train_r[a] = tr
        if train_r:                     # best-of selected on TRAIN, scored on eval
            ba = max(train_r, key=train_r.get)
            row["best_arm_by_train"] = ba
            row["R_home_best"] = row.get(f"R_home_{ba}")
        if jid in gepa:
            gr = eval_fn_on(gepa[jid], sorted(train_target), items, train_target)
            keep = gr is not None and gr >= max(train_r.values(), default=-2)
            row["gepa_kept_by_train"] = bool(keep)
            if keep:
                v = eval_fn_on(gepa[jid], sorted(det), items, eval_target)
                if v == "NEAR_DEGENERATE":
                    row["neardeg_gepa"] = True
                else:
                    row["R_home_gepa"] = round(v, 3) if v is not None else None
        results.append(row)

    json.dump({"calibration": cal_rows, "results": results},
              open(WORK / "roundtrip_results.json", "w"), indent=1)
    print("\n=== CALIBRATION (spearman vs reference implementation) ===")
    for c in cal_rows:
        print(" ", c)
    nd = sum(1 for r in results for k in r if str(k).startswith("neardeg_"))
    print(f"\nnear-degenerate channels screened (modal output >85%): {nd}")
    print("\n=== HOME-vs-HOME (code round trip vs stored judge round trip) ===")
    for arm in [f"R_home_{a}" for a in arms] + ["R_home_best", "R_home_gepa"]:
        ok = [r for r in results if r.get(arm) is not None and r.get("R_judge") is not None]
        if len(ok) < 10:
            continue
        wins = sum(1 for r in ok if r[arm] >= r["R_judge"])
        med = st.median(r[arm] for r in ok)
        print(f"{arm:16s} n={len(ok):3d}  med={med:+.3f}  code>=judge: "
              f"{wins}/{len(ok)} ({100*wins/len(ok):.0f}%)")
        for tk in sorted({r["task"] for r in ok}):
            sub = [r for r in ok if r["task"] == tk]
            w = sum(1 for r in sub if r[arm] >= r["R_judge"])
            print(f"    {tk:16s} n={len(sub):3d} med={st.median(r[arm] for r in sub):+.3f} "
                  f"wins {w}/{len(sub)}")
    print("\nsaved ->", WORK / "roundtrip_results.json")


if __name__ == "__main__":
    main()
