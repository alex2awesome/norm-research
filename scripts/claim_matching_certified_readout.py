#!/usr/bin/env python3
"""Certified-core readout: re-slice EXISTING claim-matching scores on the certified-span subset.

The certified core (built by claim_matching_certify.py) = probe claims whose gold span matches the
examiner's OWN cited paragraph (span-level examiner testimony) and whose element passes provenance
checks (no orphan misassignment, no placeholder/boilerplate). On this subset the labels carry no
pipeline-localization or LLM-extraction noise, and no LLM-consensus filtering — so a high bank
score here establishes "matching works on faithful labels" without judge circularity.

Reads (no rescoring):
  outputs/claim_matching/certified_core.json          {"certified_uids": [...], "clean_uids": [...]}
  outputs/claim_matching/scores_<tag>.jsonl           bank scores per ladder model
  outputs/claim_matching/scores_glm_holistic.jsonl    frontier holistic arm (optional)
  datasets/claim-matching/testbed/pair_testbed_v2.jsonl

  python scripts/claim_matching_certified_readout.py --tags gemma3_4b gemma3_12b gemma3_27b \
      --patch gemma3_12b=outputs/claim_matching/scores_gemma3_12b_v2negs.jsonl
"""
import argparse, json, re, hashlib, os, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE = "/lfs/skampere3/0/alexspan/norm-research"
TESTBED = f"{BASE}/datasets/claim-matching/testbed/pair_testbed_v2.jsonl"
OUTDIR = f"{BASE}/outputs/claim_matching"
CORE = f"{OUTDIR}/certified_core.json"
WORD = re.compile(r"[a-z]{3,}")


def toks(s):
    return set(WORD.findall((s or "").lower()))


def within(uids, y, s, keep=None):
    byu = collections.defaultdict(dict)
    for u, yy, ss in zip(uids, y, s):
        if keep is None or u in keep:
            byu[u][yy] = ss
    acc = n = 0
    for d in byu.values():
        if 1 in d and 0 in d:
            n += 1
            acc += 1.0 if d[1] > d[0] else 0.5 if d[1] == d[0] else 0.0
    return acc / max(1, n), n


def cv_combined(Mraw, y, apps, k=5):
    folds = np.array([int(hashlib.md5(f"cv::{a}".encode()).hexdigest(), 16) % k for a in apps])
    oof = np.zeros(len(y))
    for f in range(k):
        te = folds == f; tr = ~te
        if len(set(y[tr])) < 2 or te.sum() == 0:
            continue
        med = np.nanmedian(Mraw[tr], axis=0)
        med[np.isnan(med)] = 0.0
        Xtr = np.where(np.isnan(Mraw[tr]), med, Mraw[tr])
        Xte = np.where(np.isnan(Mraw[te]), med, Mraw[te])
        keep = Xtr.std(axis=0) > 0
        if not keep.any():
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(Xtr[:, keep], y[tr]); oof[te] = clf.predict_proba(Xte[:, keep])[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--patch", nargs="*", default=[],
                    help="TAG-SCOPED: tag=path (Codex #2 — never apply one model's patch to another tag)")
    a = ap.parse_args()
    patches = {}
    for p in a.patch:
        if "=" not in p:
            raise SystemExit(f"--patch must be tag=path (got {p!r})")
        t, path = p.split("=", 1)
        patches.setdefault(t, []).append(path)

    core = json.load(open(CORE))
    cert = set(core["certified_uids"])
    clean = set(core.get("clean_uids") or core["certified_uids"])
    tb, u2app = {}, {}
    for ln in open(TESTBED):
        r = json.loads(ln)
        tb[(r["uid"], r["y"])] = r
        u2app[r["uid"]] = str(r.get("app_id") or r["uid"])
    print(f"[core] certified {len(cert)} uids; clean(non-cert incl.) {len(clean)}", flush=True)

    summary = {}
    for tag in a.tags:
        fp = f"{OUTDIR}/scores_{tag}.jsonl"
        if not os.path.exists(fp):
            print(f"[{tag}] no scores file, skip", flush=True); continue
        rows = [json.loads(l) for l in open(fp)]
        over = {}
        for pf in patches.get(tag, []):
            for ln in open(pf):
                r = json.loads(ln)
                over[(r["uid"], r["y"], r["metric_id"])] = r["score"]
        for r in rows:
            k = (r["uid"], r["y"], r["metric_id"])
            if k in over:
                r["score"] = over[k]
        mids = sorted({r["metric_id"] for r in rows})
        pairkeys = sorted({(r["uid"], r["y"]) for r in rows})
        pidx = {k: i for i, k in enumerate(pairkeys)}
        mcol = {m: j for j, m in enumerate(mids)}
        uids = np.array([k[0] for k in pairkeys])
        y = np.array([k[1] for k in pairkeys])
        apps = np.array([u2app.get(u, u) for u in uids])
        Mraw = np.full((len(pairkeys), len(mids)), np.nan)
        for r in rows:
            if r["score"] is not None:
                Mraw[pidx[(r["uid"], r["y"])], mcol[r["metric_id"]]] = r["score"]

        oof = cv_combined(Mraw, y, apps)
        lex = np.array([len(toks(tb[k]["element"]) & toks(tb[k]["span"])) /
                        max(1, len(toks(tb[k]["element"]))) if k in tb else 0.0 for k in pairkeys])
        strata = {"full_probe": None, "certified_core": cert,
                  "clean_noncert": clean - cert, "uncertified_rest": set(uids) - clean}
        res = {}
        print(f"\n===== {tag} =====", flush=True)
        for name, keep in strata.items():
            wb, nb = within(uids, y, oof, keep)
            wl, _ = within(uids, y, lex, keep)
            res[name] = {"n_claims": nb, "bank_within": wb, "lexical_within": wl}
            print(f"  {name:16s} n={nb:4d}  bank_combined={wb:.3f}  lexical={wl:.3f}", flush=True)
        # subset-refit check: CV fit ONLY on certified pairs (guards against the combined model
        # being carried by noisy strata's weights)
        mask = np.array([u in cert for u in uids])
        if mask.sum() >= 40 and len(set(y[mask])) == 2:
            oofc = cv_combined(Mraw[mask], y[mask], apps[mask])
            wc, nc = within(uids[mask], y[mask], oofc)
            res["certified_refit"] = {"n_claims": nc, "bank_within": wc}
            print(f"  certified_refit  n={nc:4d}  bank_combined={wc:.3f}  (CV fit within core only)",
                  flush=True)
        summary[tag] = res

    # frontier holistic arm, if present
    ghf = f"{OUTDIR}/scores_glm_holistic.jsonl"
    if os.path.exists(ghf):
        gu, gy, gs = [], [], []
        for ln in open(ghf):
            r = json.loads(ln)
            if r.get("score") is not None:
                gu.append(r["uid"]); gy.append(r["y"]); gs.append(r["score"])
        for name, keep in (("full_probe", None), ("certified_core", cert)):
            w, n = within(gu, gy, gs, keep)
            print(f"[glm_holistic] {name:16s} n={n:4d} within={w:.3f}", flush=True)

    json.dump(summary, open(f"{OUTDIR}/certified_readout.json", "w"), indent=1)
    print("\nCERTIFIED_READOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
