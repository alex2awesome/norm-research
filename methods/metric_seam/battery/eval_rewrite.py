"""REWRITE intervention eval (2026-07-08): does checklist re-articulation move the
description-compiled code floor?

Three arms per criterion (18 near-miss criteria, rewrite_sample.json):
  orig    — stored codegen_claude flavors (code_scores.json, as in the fleet)
  ctl     — fresh Sonnet recompile of the ORIGINAL description (compiler-version control)
  rewrite — fresh Sonnet compile of the checklist REWRITE
Flavor selected on TRAIN within arm; TEST Spearman reported; paired bootstrap on the
test items for rewrite-vs-ctl (the description effect) and ctl-vs-orig (compiler drift).

-> outputs/metric_seam_pilot/battery/rewrite_eval.json
"""
import importlib.util, json, pathlib, signal, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, BASE  # noqa: E402
from eval_hybrids_task import paired_boot, FLAVORS  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import certificates  # noqa: E402
spearman = certificates.spearman

RW = BASE / "battery/rewrite"


def _alarm(sig, frame):
    raise TimeoutError()


def run_plain(path, items):
    spec = importlib.util.spec_from_file_location(path.stem + "_rw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    col = {}
    for dpid, t in items.items():
        try:
            signal.alarm(10)
            col[dpid] = float(mod.score(t))
        except Exception:
            col[dpid] = None
        finally:
            signal.alarm(0)
    return col


def rho_on(ids, col, judge):
    s = [d for d in ids if col.get(d) is not None and d in judge]
    return (spearman([col[d] for d in s], [judge[d] for d in s]), len(s)) \
        if len(s) >= 20 else (float("nan"), len(s))


def best_arm(cols, train, test, judge):
    """Train-select a flavor, return (test_rho, flavor, test_col)."""
    best_fl, best_tr = None, -2
    for fl, col in cols.items():
        if col is None:
            continue
        r, n = rho_on(train, col, judge)
        if r == r and r > best_tr:
            best_fl, best_tr = fl, r
    if best_fl is None:
        return None, None, None
    r_te, _ = rho_on(test, cols[best_fl], judge)
    return (round(r_te, 3) if r_te == r_te else None), best_fl, cols[best_fl]


def main():
    signal.signal(signal.SIGALRM, _alarm)
    sample = json.load(open(BASE / "battery/rewrite_sample.json"))
    ctxs, out = {}, {}
    for s in sample:
        t, aid = s["task"], s["aid"]
        if t not in ctxs:
            ctxs[t] = load_ctx(t)
        ctx = ctxs[t]
        judge = ctx["judge"].get(aid, {})
        train, test = sorted(ctx["train"]), sorted(ctx["test"])

        cs_path = ctx["outdir"] / ("code_scores_v2.json" if t == "press_releases"
                                   else "code_scores.json")
        stored = json.load(open(cs_path)) if cs_path.exists() else {}
        cols_orig = {fl: stored.get(f"{aid}_{fl}") for fl in FLAVORS}
        cols_ctl = {fl: (run_plain(RW / "programs_ctl" / f"{t}__{aid}_{fl}.py",
                                   ctx["items"])
                         if (RW / "programs_ctl" / f"{t}__{aid}_{fl}.py").exists()
                         else None) for fl in FLAVORS}
        cols_rw = {fl: (run_plain(RW / "programs" / f"{t}__{aid}_{fl}.py", ctx["items"])
                        if (RW / "programs" / f"{t}__{aid}_{fl}.py").exists()
                        else None) for fl in FLAVORS}

        r_o, fl_o, col_o = best_arm(cols_orig, train, test, judge)
        r_c, fl_c, col_c = best_arm(cols_ctl, train, test, judge)
        r_r, fl_r, col_r = best_arm(cols_rw, train, test, judge)

        row = {"y_code_cam": s["y_code"], "y_hyb_cam": s["y_hyb"],
               "orig": {"test": r_o, "flavor": fl_o},
               "ctl": {"test": r_c, "flavor": fl_c},
               "rewrite": {"test": r_r, "flavor": fl_r}}
        if col_r is not None and col_c is not None:
            sel = [d for d in test if d in judge and col_r.get(d) is not None
                   and col_c.get(d) is not None]
            _, p, _ = paired_boot(sel, col_r, col_c, judge)
            row["P_rewrite_gt_ctl"] = p
        if col_c is not None and col_o is not None:
            sel = [d for d in test if d in judge and col_c.get(d) is not None
                   and col_o.get(d) is not None]
            _, p, _ = paired_boot(sel, col_c, col_o, judge)
            row["P_ctl_gt_orig"] = p
        out[f"{t}.{aid}"] = row
        print(f"{t}.{aid}: orig={r_o} ctl={r_c} rewrite={r_r} "
              f"P_rw>ctl={row.get('P_rewrite_gt_ctl')} P_ctl>orig={row.get('P_ctl_gt_orig')}")

    ds = [(v["rewrite"]["test"] or 0) - (v["ctl"]["test"] or 0) for v in out.values()
          if v["rewrite"]["test"] is not None and v["ctl"]["test"] is not None]
    ds.sort()
    med = ds[len(ds) // 2] if ds else None
    n_sig = sum(1 for v in out.values() if (v.get("P_rewrite_gt_ctl") or 0) >= .95)
    summary = {"n": len(out), "median_delta_rewrite_minus_ctl": round(med, 3) if med is not None else None,
               "n_P95_rewrite_gt_ctl": n_sig}
    json.dump({"summary": summary, "per_criterion": out},
              open(BASE / "battery/rewrite_eval.json", "w"), indent=1)
    print("\nSUMMARY", summary)
    print(f"-> {BASE / 'battery/rewrite_eval.json'}")


if __name__ == "__main__":
    main()
