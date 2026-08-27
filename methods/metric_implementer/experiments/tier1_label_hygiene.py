"""Tier-1 label-hygiene dose-response (peer mention-AUC) — prereg frozen 2026-08-14 in
notes/2026-08-05__direction12-battery-plan.md BEFORE this script ran.

Fixed instrument: peer_p_scores.json (frozen 8B judge). Only labels vary.
Positive tiers P0-P3 (nested), negative tiers N0-N2 (attention-controlled), per prereg.
Name->a-id mapping replicates build_gold_labels.py (bank file name2id).
Output: tier1_label_hygiene_result.json + printed grid. Runs on sk3 (CPU).
"""
import json
import re
from collections import defaultdict

import numpy as np

MD = "/lfs/skampere3/0/alexspan/mention_auc"
BANK = "/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful/banks/peer-review.json"
MIN_POS, MIN_NEG = 10, 30


def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def main():
    bank = json.load(open(BANK))["metrics"]
    name2id = {}
    for i, m in enumerate(bank):
        nm = m["name"] if isinstance(m, dict) else str(m)
        name2id[nm.strip().lower()] = f"a{i}"

    # join rows -> per (doc, aid): list of (polarity, confidence, review_id)
    mentions = defaultdict(list)
    doc_review_mentions = defaultdict(lambda: defaultdict(int))   # doc -> review -> n real mentions
    for line in open(f"{MD}/mention_join_peer_20260716.jsonl"):
        r = json.loads(line)
        ch = str(r.get("choice", "")).strip().lower()
        if ch in ("noise", "abstain", "none", ""):
            continue
        aid = name2id.get(ch)
        sid = str(r.get("source_id", ""))
        mm = re.match(r"(.+)_r(\d+)$", sid)
        doc, rev = (mm.group(1), mm.group(2)) if mm else (sid, "0")
        doc_review_mentions[doc][rev] += 1
        if aid is None:
            continue
        mentions[(doc, aid)].append((r.get("polarity"), r.get("confidence"), rev))

    unmapped = sum(1 for line in open(f"{MD}/mention_join_peer_20260716.jsonl")
                   if (c := str(json.loads(line).get("choice", "")).strip().lower())
                   not in ("noise", "abstain", "none", "") and c not in name2id)
    print(f"name-mapping: {len(name2id)} bank names; unmapped real-mention rows: {unmapped}")

    p = json.load(open(f"{MD}/peer_p_scores.json")); ids = p["post_ids"]; S = p["scores"]
    y_pos_canon = json.load(open(f"{MD}/peer_y_pos.json"))
    canon = defaultdict(set)                                # doc -> set(aid)
    k0 = next(iter(y_pos_canon))
    if re.fullmatch(r"a\d+", k0):
        for m, docs in y_pos_canon.items():
            for d in docs:
                canon[d].add(m)
    else:
        for d, ms in y_pos_canon.items():
            canon[d] = set(ms)

    def pos_tier(tier):
        out = defaultdict(set)
        if tier == "P0":
            return canon
        for (doc, aid), ms in mentions.items():
            pos = [(pol, conf, rev) for pol, conf, rev in ms if pol == "pos"]
            if tier in ("P1", "P2", "P3"):
                pos = [x for x in pos if x[1] == "high"]
            if tier in ("P2", "P3") and any(pol in ("neg", "mixed") for pol, _, _ in ms):
                continue
            if tier == "P3" and len({rev for _, _, rev in pos}) < 2:
                continue
            if pos:
                out[doc].add(aid)
        return out

    def neg_ok(doc, tier):
        if tier == "N0":
            return True
        revs = doc_review_mentions.get(doc, {})
        total = sum(revs.values())
        need = 5 if tier == "N1" else 10
        return len([r for r, n in revs.items() if n >= 1]) >= 2 and total >= need

    grid = {}
    for P in ("P0", "P1", "P2", "P3"):
        ymap = pos_tier(P)
        for N in ("N0", "N1", "N2"):
            aucs = []
            for mid, ps in S.items():
                pa = np.asarray(ps, float)
                rows = []
                for i, doc in enumerate(ids):
                    if not np.isfinite(pa[i]):
                        continue
                    if mid in ymap.get(doc, ()):
                        rows.append((1, pa[i]))
                    elif neg_ok(doc, N):
                        rows.append((0, pa[i]))
                yv = np.array([r[0] for r in rows]); pv = np.array([r[1] for r in rows])
                if yv.sum() >= MIN_POS and (len(yv) - yv.sum()) >= MIN_NEG:
                    aucs.append(auc(yv, pv))
            grid[f"{P}x{N}"] = {"n_metrics": len(aucs),
                                "median_auc": round(float(np.median(aucs)), 4) if aucs else None,
                                "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None}
    json.dump(grid, open(f"{MD}/tier1_label_hygiene_result.json", "w"), indent=1)
    print(f"{'cell':8s} {'n_met':>5s} {'median':>7s} {'mean':>7s}")
    for k, v in grid.items():
        print(f"{k:8s} {v['n_metrics']:5d} {str(v['median_auc']):>7s} {str(v['mean_auc']):>7s}")


if __name__ == "__main__":
    main()
