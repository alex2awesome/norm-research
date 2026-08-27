"""Structural / cross-task metrics on the locked tau-0.825 clustering.

From match_out/clusters_<task>.json (key -> cluster id):

  1. Per-task concentration -- n_clusters, singletons, Zipf slope, normalised
     entropy, Gini, top-k coverage. How redundant / concentrated is each
     task's rubric space.
  2. Specificity profile -- per (task, bucket): forms, clusters, compression
     ratio, singleton rate.
  3. Cross-task universal concepts -- centroid each multi-member cluster in
     the shared base-bge space, complete-linkage across tasks; meta-clusters
     spanning >=3 tasks are universal concepts.

Writes JSON to match_out/metrics/ and prints readable tables.
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
FORMS = WORK / "canon_all_real_forms.jsonl"
EMB = WORK / "out"
MATCH_OUT = WORK / "match_out"
OUT = MATCH_OUT / "metrics"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2 * np.sum(idx * x) / (n * x.sum()) - (n + 1) / n)


def concentration(sizes, n_forms):
    sizes = np.array(sorted(sizes, reverse=True), dtype=float)
    n_cl = len(sizes)
    p = sizes / sizes.sum()
    H = -np.sum(p * np.log(p))
    rank = np.arange(1, n_cl + 1)
    slope = float(np.polyfit(np.log(rank), np.log(sizes), 1)[0]) if n_cl > 2 else 0.0
    cs = np.cumsum(sizes)
    cov = {k: float(cs[min(k, n_cl) - 1] / n_forms) for k in (1, 3, 10, 50)}
    return dict(n_forms=int(n_forms), n_clusters=n_cl,
                n_singletons=int((sizes == 1).sum()),
                pct_singleton=round(float((sizes == 1).mean()), 3),
                largest=int(sizes[0]), mean_size=round(float(sizes.mean()), 2),
                compression=round(n_forms / n_cl, 2),
                entropy_norm=round(float(H / np.log(n_cl)) if n_cl > 1 else 0.0, 3),
                zipf_slope=round(slope, 3), gini=round(gini(sizes), 3),
                top1=round(cov[1], 3), top3=round(cov[3], 3),
                top10=round(cov[10], 3), top50=round(cov[50], 3))


def load_task(task):
    by_bucket = defaultdict(list)
    for r in FORMS_BY_TASK[task]:
        by_bucket[r["bucket"]].append(r)
    rows, mats, ok = [], [], True
    for bucket in sorted(by_bucket):
        rs = sorted(by_bucket[bucket], key=lambda r: r["idx"])
        p = EMB / f"emb_bge_{bucket}_{task}.npy"
        rows += rs
        if p.exists():
            e = np.load(p).astype(np.float32)
            if len(e) == len(rs):
                mats.append(e)
            else:
                ok = False
        else:
            ok = False
    emb = np.vstack(mats) if (ok and mats) else None
    return rows, emb


FORMS_BY_TASK = defaultdict(list)


def main():
    OUT.mkdir(exist_ok=True)
    for line in FORMS.open():
        r = json.loads(line)
        FORMS_BY_TASK[r["task"]].append(r)

    conc, spec = {}, {}
    multi = []  # (task, cluster_id, size, rep_text, centroid)

    print(f"{'=' * 78}\n1. PER-TASK CONCENTRATION (tau-0.825 clustering)\n{'=' * 78}")
    hdr = (f"{'task':<26}{'forms':>7}{'clust':>7}{'%sing':>7}{'compr':>7}"
           f"{'entH':>7}{'zipf':>7}{'gini':>7}{'top10':>7}")
    print(hdr)
    for task in TASKS:
        rows, emb = load_task(task)
        cl = json.loads((MATCH_OUT / f"clusters_{task}.json").read_text())
        members = defaultdict(list)
        for gi, r in enumerate(rows):
            members[cl[r["key"]]].append(gi)
        sizes = [len(v) for v in members.values()]
        c = concentration(sizes, len(rows))
        conc[task] = c
        print(f"{task:<26}{c['n_forms']:>7}{c['n_clusters']:>7}"
              f"{c['pct_singleton']:>7.2f}{c['compression']:>7.2f}"
              f"{c['entropy_norm']:>7.2f}{c['zipf_slope']:>7.2f}"
              f"{c['gini']:>7.2f}{c['top10']:>7.3f}")

        # per-bucket specificity
        for r in rows:
            pass
        bk = defaultdict(lambda: {"forms": 0, "cl": set(), "sing": 0})
        cl_size = {k: len(v) for k, v in members.items()}
        for gi, r in enumerate(rows):
            c2 = cl[r["key"]]
            b = bk[r["bucket"]]
            b["forms"] += 1
            b["cl"].add(c2)
            if cl_size[c2] == 1:
                b["sing"] += 1
        spec[task] = {bb: dict(forms=v["forms"], clusters=len(v["cl"]),
                               compression=round(v["forms"] / len(v["cl"]), 2),
                               pct_singleton=round(v["sing"] / v["forms"], 3))
                      for bb, v in bk.items()}

        # multi-member cluster centroids for cross-task
        if emb is not None:
            en = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            for cid, idxs in members.items():
                if len(idxs) >= 2:
                    cen = en[idxs].mean(0)
                    txt = Counter(rows[i]["canonical"] or "" for i in idxs)
                    multi.append((task, cid, len(idxs),
                                  txt.most_common(1)[0][0], cen))

    print(f"\n{'=' * 78}\n2. SPECIFICITY PROFILE (forms / clusters / compression"
          f" / %singleton)\n{'=' * 78}")
    for task in TASKS:
        parts = "  ".join(f"{b}:{d['forms']}->{d['clusters']}"
                           f"({d['compression']}x,{int(d['pct_singleton']*100)}%s)"
                           for b, d in sorted(spec[task].items()))
        print(f"  {task:<26} {parts}")

    print(f"\n{'=' * 78}\n3. TOP-5 LARGEST CLUSTERS PER TASK\n{'=' * 78}")
    top_per_task = {}
    for task in TASKS:
        rows, _ = load_task(task)
        cl = json.loads((MATCH_OUT / f"clusters_{task}.json").read_text())
        members = defaultdict(list)
        for gi, r in enumerate(rows):
            members[cl[r["key"]]].append(gi)
        big = sorted(members.values(), key=len, reverse=True)[:5]
        top_per_task[task] = []
        print(f"\n  {task}:")
        for idxs in big:
            txt = Counter(rows[i]["canonical"] or "" for i in idxs)
            rep = txt.most_common(1)[0][0]
            top_per_task[task].append({"size": len(idxs), "rep": rep})
            print(f"    [{len(idxs):>3}] {rep[:96]}")

    # 4. cross-task universal concepts
    print(f"\n{'=' * 78}\n4. CROSS-TASK UNIVERSAL CONCEPTS\n{'=' * 78}")
    cens = np.array([m[4] for m in multi])
    print(f"  {len(multi)} multi-member clusters across "
          f"{len(set(m[0] for m in multi))} tasks (with bge embeddings)")
    universal = {}
    if len(cens) > 2:
        Z = linkage(pdist(cens, metric="cosine"), method="complete")
        for thr in (0.86, 0.90, 0.93):
            lab = fcluster(Z, t=1.0 - thr, criterion="distance")
            meta = defaultdict(list)
            for i, c in enumerate(lab):
                meta[c].append(i)
            uni = [m for m in meta.values()
                   if len({multi[i][0] for i in m}) >= 3]
            print(f"  cos>={thr}: {len(meta)} meta-clusters, "
                  f"{len(uni)} span >=3 tasks")
            if thr == 0.90:
                ranked = sorted(uni, key=lambda m: -len({multi[i][0] for i in m}))
                for m in ranked[:40]:
                    tasks_in = sorted({multi[i][0] for i in m})
                    big = max(m, key=lambda i: multi[i][2])
                    universal[str(multi[big][1]) + "@" + multi[big][0]] = {
                        "n_tasks": len(tasks_in), "tasks": tasks_in,
                        "n_clusters": len(m),
                        "n_forms": sum(multi[i][2] for i in m),
                        "rep": multi[big][3]}
                print(f"\n  universal concepts (cos>=0.90, >=3 tasks), "
                      f"top 40 by task-coverage:")
                for m in ranked[:40]:
                    tin = sorted({multi[i][0] for i in m})
                    big = max(m, key=lambda i: multi[i][2])
                    print(f"    [{len(tin):>2} tasks] {multi[big][3][:80]}")

    (OUT / "concentration.json").write_text(json.dumps(conc, indent=1))
    (OUT / "specificity.json").write_text(json.dumps(spec, indent=1))
    (OUT / "top_clusters.json").write_text(json.dumps(top_per_task, indent=1))
    (OUT / "universal_concepts.json").write_text(json.dumps(universal, indent=1))
    print(f"\nwrote concentration/specificity/top_clusters/universal_concepts"
          f" JSON -> {OUT}")


if __name__ == "__main__":
    main()
