"""Close the two holes the advisor found in the leakage audit, plus the free conditional readout.

HOLE A - the no-ship verification was ANSWER-GRADE ONLY. We verified that no shipped candidate
contains an `answer` unit, but 19 of hover's 20 flags are `item_hint`, and 4 flagged units have
content located in the TEST split. An item_hint whose content is in TEST is functionally leakage.
So: recompute TEST_HIT status for every flagged unit, then check EVERY shipped candidate on EVERY
bench for TEST_HIT units of ANY grade.

CONDITIONAL READOUT (free, and stronger than any disclosure paragraph). The rank certificate's
generator already includes each unit independently with p=.5, so ~half the draws contain any given
flagged unit — a built-in randomized experiment. Condition the stored draw scores on flagged-unit
presence: if flagged-present draws do not outscore flagged-absent draws, that is a CAUSAL NULL on
the flagged content's contribution. Costs zero GPU because the draws are already scored and the
mask sequence is reproducible (np.random.default_rng(seed) advanced once per draw).

The guiding principle, stated once: LEAKAGE REQUIRES BOTH answer-like content AND a test item it
answers. An answer-grade unit with no test-split referent is memorized TRAINING content — that is
the memorization finding, not test contamination.

Usage:  python3 audit_leakage_holes.py --benches livebench aime ifbench     # boxes w/o hover
        python3 audit_leakage_holes.py --benches hover --cert-seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

import paperexact_arms as px

HERE = Path(__file__).parent


def probes(u):
    q = (u.get("quote") or "") + " " + u["unit"]
    caps = re.findall(r"[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){1,5}", q)
    nums = re.findall(r"\b\d{3}\b", q)
    return [c.strip() for c in caps if len(c.strip()) > 8][:6] + nums[:3]


def norm(s):
    return " ".join(str(s).lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="+", required=True)
    ap.add_argument("--lm-tag", default="Qwen3-8B")
    ap.add_argument("--audit", default="runs/unit_leakage_audit.json")
    ap.add_argument("--out", default="runs/leakage_holes.json")
    ap.add_argument("--cert-seed", type=int, default=0)
    ap.add_argument("--include-p", type=float, default=0.5)
    a = ap.parse_args()

    aud = json.loads((HERE / a.audit).read_text())
    out_path = HERE / a.out
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for bench in a.benches:
        rec = aud.get(bench)
        if not rec:
            print(f"[{bench}] not in audit"); continue
        flagged = [u for u in rec["units"] if u.get("category") in ("answer", "item_hint")]
        try:
            b, _, _, _ = px.load_bench(bench)
        except Exception as e:
            print(f"[{bench}] LOAD FAILED {type(e).__name__} - skipping"); continue

        def blob(split):
            out = []
            for x in split:
                try: out.append(json.dumps(x.toDict()))
                except Exception: out.append(str(getattr(x, "__dict__", x)))
            return " ".join(out).lower()
        tr, te = blob(list(b.train_set) + list(b.val_set)), blob(list(b.test_set))

        test_hits = []
        for u in flagged:
            ps = probes(u)
            hit = [p for p in ps if p.lower() in te]
            if hit:
                test_hits.append({"unit": u["unit"], "category": u["category"],
                                  "test_spans": hit[:3],
                                  "also_in_train": any(p.lower() in tr for p in ps)})
        print(f"\n=== {bench} === flagged={len(flagged)} TEST_HIT={len(test_hits)}")

        # ---- HOLE A: any TEST_HIT unit (ANY grade) inside any shipped candidate? ----
        shipped = {}
        for f in sorted(glob.glob(str(HERE / "runs_paperexact" / bench / a.lm_tag / "*" / "result.json"))):
            name = os.path.basename(os.path.dirname(f))
            try: d = json.loads(Path(f).read_text())
            except Exception: continue
            cand = d.get("best_candidate") or {}
            cblob = norm(" ".join(str(v) for v in cand.values()))
            hits = [t for t in test_hits if norm(t["unit"])[:110] in cblob]
            shipped[name] = {"best_test": d.get("best_test"),
                             "n_test_hit_units": len(hits),
                             "categories": [t["category"] for t in hits]}
            flag = "  <-- TEST_HIT IN SHIPPED" if hits else ""
            print(f"  {name:36s} best={str(d.get('best_test')):8s} test_hit_units={len(hits)}{flag}")

        # ---- CONDITIONAL READOUT on the stored certificate draws ----
        cond = None
        run_json = HERE / "runs_paperexact" / bench / a.lm_tag / "rank_certificate" / "running.json"
        pool_file = HERE / "pools" / f"{bench}_{a.lm_tag}_frozen.json"
        if run_json.exists() and pool_file.exists() and test_hits:
            cert = json.loads(run_json.read_text())
            scores = cert["scores"]
            seed_cand = px.get_instructions(px.load_bench(bench)[1])
            units = [(d["module"], d["unit"]) for d in json.loads(pool_file.read_text())["units"]
                     if d["module"] in seed_cand]
            rng = np.random.default_rng(a.cert_seed)
            masks = [rng.random(len(units)) < a.include_p for _ in range(len(scores))]
            idxs = [i for i, u in enumerate(units)
                    if any(norm(u[1])[:110] == norm(t["unit"])[:110] for t in test_hits)]
            if idxs:
                pres = [s for s, m in zip(scores, masks) if s is not None and any(m[i] for i in idxs)]
                absn = [s for s, m in zip(scores, masks) if s is not None and not any(m[i] for i in idxs)]
                if pres and absn:
                    p, q = np.array(pres), np.array(absn)
                    cond = {"n_flagged_present": len(p), "n_flagged_absent": len(q),
                            "mean_present": float(p.mean()), "mean_absent": float(q.mean()),
                            "delta": float(p.mean() - q.mean()),
                            "sd_present": float(p.std()), "sd_absent": float(q.std())}
                    # permutation test on the difference of means
                    allx = np.concatenate([p, q]); n1 = len(p)
                    rng2 = np.random.default_rng(12345)
                    obs = p.mean() - q.mean()
                    perm = sum(1 for _ in range(10000)
                               if (lambda z: z[:n1].mean() - z[n1:].mean())(rng2.permutation(allx)) >= obs)
                    cond["perm_p_one_sided"] = perm / 10000
                    cond["verdict"] = ("CAUSAL NULL: flagged-present draws do not outscore "
                                       "flagged-absent" if cond["delta"] <= 0 or cond["perm_p_one_sided"] > .05
                                       else "flagged content shows a positive conditional effect -> PURGE")
                    print(f"  CONDITIONAL: present n={len(p)} mean={p.mean():.4f} | "
                          f"absent n={len(q)} mean={q.mean():.4f} | delta={obs:+.4f} "
                          f"perm_p={cond['perm_p_one_sided']:.4f}")
                    print(f"  -> {cond['verdict']}")

        results[bench] = {"n_flagged": len(flagged), "test_hits": test_hits,
                          "shipped_scan": shipped, "conditional_draw_readout": cond}
        out_path.write_text(json.dumps(results, indent=1))

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
