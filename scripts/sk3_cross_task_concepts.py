"""Cross-task universal-concept analysis, with the task-noun confound removed.

Canonical rubric texts are templated -- "the CODE should be clear" vs "the
WRITING should be clear" -- so the task-specific subject noun drags down
cross-task cosine and undercounts shared concepts. This strips the leading
"<subject> should/must" clause, embeds only the task-neutral predicate with
bge-large, then meta-clusters the multi-member cluster representatives across
tasks.

Outputs: universal concepts (meta-clusters spanning >=3 tasks) and an 11x11
task-pair concept-sharing matrix.
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
os.environ["HF_HOME"] = "/lfs/skampere3/0/alexspan/hf_cache"
os.environ["HF_MODULES_CACHE"] = "/lfs/skampere3/0/alexspan/hf_cache/modules"
os.environ["XDG_CACHE_HOME"] = "/lfs/skampere3/0/alexspan/.cache"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
FORMS = WORK / "canon_all_real_forms.jsonl"
MATCH_OUT = WORK / "match_out"
OUT = MATCH_OUT / "metrics"
BGE_BASE = "/lfs/skampere3/0/shared_hf_cache/models--BAAI--bge-large-en-v1.5/snapshots"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]

PREFIX = re.compile(
    r"^(the|a|an|each|every|all|its)\s+[a-z][\w\s,'()/-]*?\s+"
    r"(should not|must not|should|must|shall|needs? to|is expected to|"
    r"are expected to|ought to)\s+", re.I)


def predicate(t):
    m = PREFIX.match(t or "")
    return (t[m.end():] if m else (t or "")).strip() or (t or "")


def main():
    forms_by_task = defaultdict(list)
    for line in FORMS.open():
        r = json.loads(line)
        forms_by_task[r["task"]].append(r)

    # multi-member clusters: (task, size, rep_text)
    multi = []
    for task in TASKS:
        rows = sorted(forms_by_task[task], key=lambda r: (r["bucket"], r["idx"]))
        cl = json.loads((MATCH_OUT / f"clusters_{task}.json").read_text())
        members = defaultdict(list)
        for r in rows:
            members[cl[r["key"]]].append(r["canonical"] or "")
        for texts in members.values():
            if len(texts) >= 2:
                rep = Counter(texts).most_common(1)[0][0]
                multi.append((task, len(texts), rep))
    print(f"{len(multi)} multi-member clusters across {len(TASKS)} tasks")

    preds = [predicate(rep) for _, _, rep in multi]
    print("sample subject-stripped predicates:")
    for rep, p in list(zip([m[2] for m in multi], preds))[:6]:
        print(f"  '{rep[:60]}' -> '{p[:60]}'")

    from sentence_transformers import SentenceTransformer
    bge = BGE_BASE + "/" + sorted(os.listdir(BGE_BASE))[0]
    model = SentenceTransformer(bge, device="cuda")
    emb = model.encode(preds, batch_size=256, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True)
    emb = emb.astype(np.float64)

    Z = linkage(pdist(emb, metric="cosine"), method="complete")
    print(f"\n{'=' * 74}\nCROSS-TASK UNIVERSAL CONCEPTS "
          f"(subject-neutralised predicates)\n{'=' * 74}")
    universal = {}
    pair = np.zeros((len(TASKS), len(TASKS)), dtype=int)
    ti = {t: i for i, t in enumerate(TASKS)}
    for thr in (0.82, 0.86, 0.90):
        lab = fcluster(Z, t=1.0 - thr, criterion="distance")
        meta = defaultdict(list)
        for i, c in enumerate(lab):
            meta[c].append(i)
        uni = [m for m in meta.values()
               if len({multi[i][0] for i in m}) >= 3]
        print(f"\ncos>={thr}: {len(meta)} meta-clusters, {len(uni)} span >=3 tasks")
        if thr == 0.86:
            ranked = sorted(uni, key=lambda m: -len({multi[i][0] for i in m}))
            for m in ranked:
                tin = sorted({multi[i][0] for i in m})
                big = max(m, key=lambda i: multi[i][1])
                universal[multi[big][2][:90]] = {
                    "n_tasks": len(tin), "tasks": tin, "n_clusters": len(m),
                    "n_forms": sum(multi[i][1] for i in m)}
            for m in ranked[:45]:
                tin = sorted({multi[i][0] for i in m})
                big = max(m, key=lambda i: multi[i][1])
                print(f"  [{len(tin):>2}t {sum(multi[i][1] for i in m):>4}f] "
                      f"{multi[big][2][:78]}")
            for m in meta.values():
                tin = sorted({multi[i][0] for i in m})
                for a, b in combinations(tin, 2):
                    pair[ti[a], ti[b]] += 1
                    pair[ti[b], ti[a]] += 1

    print(f"\n{'=' * 74}\nTASK-PAIR SHARED-CONCEPT COUNTS (cos>=0.86)\n{'=' * 74}")
    print("     " + " ".join(f"{t[:4]:>5}" for t in TASKS))
    for i, t in enumerate(TASKS):
        print(f"{t[:4]:>5}" + " ".join(f"{pair[i, j]:>5}" for j in range(len(TASKS))))

    (OUT / "universal_concepts_v2.json").write_text(json.dumps(universal, indent=1))
    (OUT / "task_pair_sharing.json").write_text(json.dumps(
        {"tasks": TASKS, "matrix": pair.tolist()}, indent=1))
    print(f"\nwrote universal_concepts_v2 + task_pair_sharing -> {OUT}")


if __name__ == "__main__":
    main()
