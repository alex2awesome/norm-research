"""Checkpoint provenance catalog + invariant checker — the standing guard against
level/identity mix-ups (born from the 2026-07-02 level-dispatch bug, where R3 checkpoints were
silently paired with R2 rubrics and the aligned dirs measured the wrong metrics).

Walks a root of checkpoint dirs and, per *_sigs.npz, machine-verifies:

  C1 name_match   npz `name` == hierarchy[level][gi] merged_name  (the bug-catcher: the ckpt IS
                  the metric its filename claims, at the level its filename claims)
  C2 shape        len(M_i) == sigs.shape[1]  (target aligned with probe axis)
  C3 tau0_stale   tau0 == 0.05 literal -> WARN (pre-fix rescore hardcode era)
  C4 forminv      *_forminv.json present? has per-pair records? (form gate has data / vanishes)
  C5 orbit        orbit_forms>1 => M_i_var_phi & M_i_flip_rate present; M_i_forms probe axis match
  C6 rescoring    rescored ckpts record source_ckpt + target_model

Writes <root>/CATALOG.json (per-dir: counts, executor, levels, check failures, status) and prints
a table. Manual dir verdicts (e.g. DEPRECATED with reason) live in <root>/CATALOG_OVERRIDES.json
as {"dirname": {"status": "...", "reason": "..."}} and are carried into the catalog verbatim.

Policy: no certificate/grid consumes a dir that isn't OK (or explicitly acknowledged) here.

Usage: python -m methods.metric_implementer.experiments.catalog_check \
           --root <outputs>/r3_cw --task creative-writing [--json <path>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

CKPT_RE = re.compile(r"_(R[123])_metric(\d+)_sigs\.npz$")


def check_ckpt(path: str, groups_for) -> dict:
    """All C1-C6 checks for one checkpoint. `groups_for(level) -> list[dict]` is injectable."""
    m = CKPT_RE.search(os.path.basename(path))
    out = {"file": os.path.basename(path), "fails": [], "warns": []}
    if not m:
        out["fails"].append("unparseable filename (no _R?_metric<N>_sigs.npz)")
        return out
    lvl, gi = m.group(1), int(m.group(2))
    out.update(level=lvl, gi=gi)
    z = np.load(path, allow_pickle=True)
    # C1 -- identity
    try:
        groups = groups_for(lvl)
        want = str(groups[gi].get("merged_name", "")).strip()
        got = str(z["name"]).strip() if "name" in z.files else ""
        if want and got and want != got:
            out["fails"].append(f"C1 name_match: npz={got[:40]!r} != hierarchy={want[:40]!r}")
        elif not got:
            out["warns"].append("C1: npz has no name field")
    except Exception as e:
        out["warns"].append(f"C1 unverifiable: {e}")
    # C1b -- rubric-TEXT identity (field written by rescore_executor after 2026-07-02; the v1
    # aligned dirs predate it, which is exactly why the bug was invisible to artifacts alone)
    if "target_desc" in z.files:
        try:
            want_d = str(groups_for(lvl)[gi].get("merged_description", "")).strip()
            if want_d and str(z["target_desc"]).strip() != want_d:
                out["fails"].append("C1b target_desc != hierarchy merged_description")
        except Exception as e:
            out["warns"].append(f"C1b unverifiable: {e}")
    # C2 -- shapes
    if "M_i" in z.files and "sigs" in z.files:
        n_m, n_s = len(np.atleast_1d(z["M_i"])), np.asarray(z["sigs"]).shape[-1]
        if n_m != n_s:
            out["fails"].append(f"C2 shape: len(M_i)={n_m} != sigs probes={n_s}")
    # C3 -- stale noise floor
    if "tau0" in z.files and float(z["tau0"]) == 0.05:
        out["warns"].append("C3 tau0==0.05 literal (pre-fix rescore hardcode)")
    # C4 -- form gate data
    fip = path.replace("_sigs.npz", "_forminv.json")
    if os.path.exists(fip):
        out["forminv"] = "pairs" if json.load(open(fip)).get("pairs") else "summary-only"
    else:
        out["forminv"] = "MISSING"
    # C5 -- orbit consistency
    if "orbit_forms" in z.files and int(z["orbit_forms"]) > 1:
        for req in ("M_i_var_phi", "M_i_flip_rate"):
            if req not in z.files:
                out["fails"].append(f"C5 orbit: {req} missing")
        if "M_i_forms" in z.files and np.asarray(z["M_i_forms"]).shape[-1] != len(np.atleast_1d(z["M_i"])):
            out["fails"].append("C5 orbit: M_i_forms probe axis mismatch")
    # C6 -- rescore provenance
    if "rescoring" in z.files and bool(z["rescoring"]):
        for req in ("source_ckpt", "target_model"):
            if req not in z.files:
                out["warns"].append(f"C6: rescored ckpt missing {req}")
        out["target_model"] = str(z["target_model"]) if "target_model" in z.files else "?"
    return out


def check_dir(d: str, groups_for) -> dict:
    recs = [check_ckpt(f, groups_for) for f in sorted(glob.glob(os.path.join(d, "*_sigs.npz")))]
    fails = [r for r in recs if r["fails"]]
    warns = [r for r in recs if r["warns"] and not r["fails"]]
    execs = sorted({r.get("target_model", "") for r in recs if r.get("target_model")})
    return {"dir": os.path.basename(d), "n_ckpts": len(recs),
            "levels": sorted({r.get("level", "?") for r in recs}),
            "executors": execs,
            "forminv": {k: sum(1 for r in recs if r.get("forminv") == k)
                        for k in ("pairs", "summary-only", "MISSING")},
            "n_fail": len(fails), "n_warn": len(warns),
            "status": "FAIL" if fails else ("WARN" if warns else "OK"),
            "failures": [{"file": r["file"], "fails": r["fails"]} for r in fails[:10]]}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--r2-bucket", default="general")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    from methods.metric_implementer.experiments.mine_clusters import r2_groups, r3_groups
    cache = {}

    def groups_for(level):
        if level not in cache:
            cache[level] = (r3_groups if level == "R3" else r2_groups)(a.task, a.r2_bucket)
        return cache[level]

    overrides = {}
    op = os.path.join(a.root, "CATALOG_OVERRIDES.json")
    if os.path.exists(op):
        overrides = json.load(open(op))

    cat = {}
    for d in sorted(glob.glob(os.path.join(a.root, "*"))):
        if not os.path.isdir(d) or not glob.glob(os.path.join(d, "*_sigs.npz")):
            continue
        rec = check_dir(d, groups_for)
        if rec["dir"] in overrides:
            rec["status"] = overrides[rec["dir"]].get("status", rec["status"])
            rec["reason"] = overrides[rec["dir"]].get("reason", "")
        cat[rec["dir"]] = rec

    out = a.json or os.path.join(a.root, "CATALOG.json")
    json.dump(cat, open(out, "w"), indent=1)
    print(f"{'dir':<26} {'n':>4} {'status':<11} {'lvl':<5} forminv(pairs/summ/miss)  notes")
    for k, r in cat.items():
        fi = r["forminv"]
        note = r.get("reason") or (r["failures"][0]["fails"][0][:60] if r["failures"] else "")
        print(f"{k:<26} {r['n_ckpts']:>4} {r['status']:<11} {'/'.join(r['levels']):<5} "
              f"{fi['pairs']}/{fi['summary-only']}/{fi['MISSING']:<18} {note}")
    print(f"\nwrote {out}")
    if any(r["status"] == "FAIL" and "reason" not in r for r in cat.values()):
        raise SystemExit(2)   # gate: unacknowledged failures block downstream consumers


if __name__ == "__main__":
    main()
