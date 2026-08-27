"""Chain-proof repair of the L0 singleton tail using existing v6 judge labels.

Diagnosis (2026-06-12): 33-68% of judge-confirmed "same rule" (score=2) pairs
are split across L0 clusters, so a large share of the singleton tail is
unmerged duplicates. But score=2 is NOT transitive (P(A=C | A=B=2, B=C=2) =
0.78-0.93 by judged triangles), so union-find over score=2 edges snowballs
(creative-writing: 1,228 clusters chained into one 3,034-form blob).

Repair operator: ONE-ROUND STAR ADOPTION — no transitive growth possible.
  - Only tail clusters (size <= TAIL_MAX) are absorbed.
  - Candidate target: any cluster connected by >=1 score=2 edge and NO
    score=0 edge; support = (#score2 edges, -#score<=1 edges, target size).
  - Tail -> non-tail: adopt directly (stars around head clusters).
  - Tail -> tail: only MUTUAL-best pairs merge (no chains).

Upward propagation is deterministic bookkeeping (no LLM re-runs):
  - L0->R1: absorbed cluster's forms inherit the TARGET's R1 family; the
    absorbed cluster id is removed from its old family; emptied families drop.
  - R1->R2: dropped families are removed from their aspects; emptied aspects drop.
  - Cross-family adoptions are logged as CANDIDATE R1 merge edges
    (candidate_r1_merges_<task>.json) for a future Fork3-style pass -- they
    are evidence the two families are related, but are NOT applied.

Outputs under outputs/analyses/structural_metrics/adopt_v1/:
  clusters_<task>.json        repaired L0 (key -> new cluster id)
  r1_families_<task>.json     propagated R1
  r2_aspects_<task>.json      propagated R2
  candidate_r1_merges_<task>.json
  adoptions_<task>.jsonl      audit log: every adoption with texts + support
  summary.md

The locked tau-0.825 artifacts are untouched. Known precision of a single
score=2 edge is ~60-80% (spot-checked), so treat adopt_v1 as a variant for
sensitivity analysis; v2 should re-adjudicate adoptions with fresh judge
calls (majority over seeds) on sk3.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SM = ROOT / "outputs/analyses/structural_metrics"
import os
MIN_SUPPORT = int(os.environ.get("MIN_SUPPORT", "1"))
OUT = SM / os.environ.get("OUT_NAME", "adopt_v1")
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]

TAIL_MAX = 2


def load_canon():
    canon = {}
    for line in open(ROOT / "outputs/analyses/canon_all_real_forms.jsonl"):
        r = json.loads(line)
        canon[r["key"]] = r["canonical"]
    return canon


def run_task(task, canon):
    cl = json.load(open(SM / f"clusters_{task}.json"))
    members = defaultdict(list)
    for k, c in cl.items():
        members[c].append(k)
    size = {c: len(ks) for c, ks in members.items()}

    # judged evidence between cluster pairs
    pair_scores = defaultdict(list)  # frozenset({ca,cb}) -> [scores]
    for line in open(SM / f"validation/{task}_v6_verdicts.jsonl"):
        r = json.loads(line)
        ca, cb = cl.get(r["key_a"]), cl.get(r["key_b"])
        if ca is None or cb is None or ca == cb or r["score"] is None:
            continue
        pair_scores[frozenset((ca, cb))].append(r["score"])

    # candidate targets per tail cluster
    cand = defaultdict(dict)  # tail -> {target: (n2, n_le1)}
    for pair, scores in pair_scores.items():
        a, b = tuple(pair)
        n2 = sum(1 for s in scores if s == 2)
        n0 = sum(1 for s in scores if s == 0)
        n_le1 = sum(1 for s in scores if s <= 1)
        if n2 < MIN_SUPPORT or n0 > 0:
            continue
        for t, o in ((a, b), (b, a)):
            if size[t] <= TAIL_MAX:
                cand[t][o] = (n2, n_le1)

    def best_target(t, pool):
        cs = [(o, v) for o, v in cand[t].items() if o in pool]
        if not cs:
            return None
        # most score2 edges, fewest contradicting, then larger target
        cs.sort(key=lambda x: (-x[1][0], x[1][1], -size[x[0]]))
        return cs[0][0]

    nontail = {c for c in size if size[c] > TAIL_MAX}
    tails = {c for c in size if size[c] <= TAIL_MAX}

    adopt = {}  # absorbed -> target
    # round A: tail -> non-tail stars
    for t in sorted(tails):
        tgt = best_target(t, nontail)
        if tgt is not None:
            adopt[t] = tgt
    # round B: remaining tail -> tail, mutual best only
    remaining = sorted(tails - set(adopt))
    rem = set(remaining)
    for t in remaining:
        if t not in rem:
            continue
        tgt = best_target(t, rem - {t})
        if tgt is not None and best_target(tgt, rem - {tgt}) == t:
            adopt[t] = tgt  # t absorbed into tgt; tgt stays
            rem.discard(t)
            rem.discard(tgt)

    # optional filter: only adoptions in ALLOWED_PAIRS (jsonl of
    # {task, absorbed, target}) survive — used to apply fresh-judge verdicts
    ap = os.environ.get("ALLOWED_PAIRS")
    if ap:
        allowed = {(r["task"], r["absorbed"], r["target"])
                   for r in map(json.loads, open(ap))}
        adopt = {t: g for t, g in adopt.items()
                 if (task, t, g) in allowed}

    # apply (single hop by construction: targets are never absorbed —
    # nontail targets can't be tails; mutual-best targets are removed from rem)
    assert not (set(adopt) & set(adopt.values())), "chain detected"
    newmem = defaultdict(list)
    for c, ks in members.items():
        newmem[adopt.get(c, c)].extend(ks)

    # audit log
    with open(OUT / f"adoptions_{task}.jsonl", "w") as f:
        for t, tgt in sorted(adopt.items()):
            n2, n_le1 = cand[t][tgt]
            f.write(json.dumps({
                "absorbed": t, "target": tgt, "n_score2": n2,
                "n_score_le1": n_le1, "absorbed_size": size[t],
                "target_size": size[tgt],
                "absorbed_text": canon.get(members[t][0], ""),
                "target_text": canon.get(members[tgt][0], "")}) + "\n")

    new_cl = {}
    for c, ks in newmem.items():
        for k in ks:
            new_cl[k] = int(c)
    json.dump(new_cl, open(OUT / f"clusters_{task}.json", "w"))

    # ---- propagate to R1 ----
    r1 = json.load(open(SM / f"r1_v4a_lora_fork3_merge/r1_families_{task}.json"))
    fams = r1["families"]
    c2f = {}
    for fi, fam in enumerate(fams):
        for cid in fam["cluster_ids"]:
            c2f[cid] = fi
    cand_merges = Counter()
    dropped_fams = set()
    for t, tgt in adopt.items():
        ft, fg = c2f.get(t), c2f.get(tgt)
        if ft is None or fg is None or ft == fg:
            continue
        # absorbed cluster moves to target's family
        fams[ft]["cluster_ids"] = [c for c in fams[ft]["cluster_ids"] if c != t]
        fams[fg]["cluster_ids"].append(t)
        cand_merges[(min(ft, fg), max(ft, fg))] += 1
        if not fams[ft]["cluster_ids"]:
            dropped_fams.add(ft)
    new_fams = [f for i, f in enumerate(fams) if i not in dropped_fams]
    old2new = {}
    j = 0
    for i in range(len(fams)):
        if i not in dropped_fams:
            old2new[i] = j
            j += 1
    r1_out = dict(r1, families=new_fams,
                  n_merged_families=len(new_fams),
                  method=r1.get("method", "") + "+adopt_v1")
    json.dump(r1_out, open(OUT / f"r1_families_{task}.json", "w"))
    json.dump([{"fam_a": a, "fam_b": b, "n_adoption_edges": n,
                "name_a": fams[a]["name"], "name_b": fams[b]["name"]}
               for (a, b), n in cand_merges.most_common()],
              open(OUT / f"candidate_r1_merges_{task}.json", "w"), indent=1)

    # ---- propagate to R2 ----
    r2 = json.load(open(SM / f"r2_v1_subagent/r2_aspects_{task}.json"))
    new_aspects = []
    for a in r2["aspects"]:
        fids = [old2new[f] for f in a["family_ids"]
                if f in old2new]
        if fids:
            new_aspects.append(dict(a, family_ids=fids, n_families=len(fids)))
    r2_out = dict(r2, aspects=new_aspects, n_r1_families=len(new_fams),
                  n_r2_aspects=len(new_aspects))
    json.dump(r2_out, open(OUT / f"r2_aspects_{task}.json", "w"))

    sizes1 = sorted((len(v) for v in newmem.values()), reverse=True)
    return dict(
        task=task, n_adopted=len(adopt),
        adopt_to_head=sum(1 for t, g in adopt.items() if size[g] > TAIL_MAX),
        clusters_before=len(members), clusters_after=len(newmem),
        singl_before=sum(1 for v in members.values() if len(v) == 1),
        singl_after=sum(1 for v in newmem.values() if len(v) == 1),
        max_before=max(size.values()), max_after=sizes1[0],
        r1_before=len(fams), r1_after=len(new_fams),
        r2_before=len(r2["aspects"]), r2_after=len(new_aspects),
        cand_r1_merges=len(cand_merges))


def main():
    canon = load_canon()
    rows = [run_task(t, canon) for t in TASKS]
    with open(OUT / "summary.md", "w") as f:
        f.write("# adopt_v1: one-round star adoption of tail clusters "
                "(judge score=2 edges, no chaining)\n\n")
        f.write("| task | adopted | ->head | L0 before->after | singl% b->a | "
                "max b->a | R1 b->a | R2 b->a | cand R1 merges |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['task']} | {r['n_adopted']} | {r['adopt_to_head']} | "
                    f"{r['clusters_before']}->{r['clusters_after']} | "
                    f"{100*r['singl_before']/r['clusters_before']:.0f}->"
                    f"{100*r['singl_after']/r['clusters_after']:.0f} | "
                    f"{r['max_before']}->{r['max_after']} | "
                    f"{r['r1_before']}->{r['r1_after']} | "
                    f"{r['r2_before']}->{r['r2_after']} | "
                    f"{r['cand_r1_merges']} |\n")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
