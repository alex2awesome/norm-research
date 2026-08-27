"""Build R2_post — reconciled cluster verdicts + collapse map + merged aspects + new enrichments.

Pipeline:
 1. Load my classifications + subagent classifications
 2. Reconcile verdicts (conservative rule; honor subagent sub-DUPs)
 3. Build collapse map per task: list of merge groups
 4. Pick canonical aspect per group (highest n_dps_seen, fallback longest description)
 5. Write aspects_r2_post.json per task (new aspect inventory)
 6. Write enrichments_r2_post/{aid}_p{0,1,2}.json per task (canonical's enrichments copied)
"""
import argparse, json
from pathlib import Path
from collections import defaultdict

import pandas as pd


def load_my_verdicts(repo: Path):
    out = {}
    for line in (repo / "outputs/v2_analysis/my_classifications.tsv").open():
        if line.startswith("cluster_id"): continue
        cid, v = line.strip().split("\t")
        out[cid] = v
    return out


def load_subagent(repo: Path):
    verdicts, sub_dups, rats = {}, {}, {}
    for f in sorted((repo / "outputs/v2_analysis/subagent_classifications").glob("*_results.jsonl")):
        for line in f.open():
            line = line.strip()
            if not line: continue
            try:
                o = json.loads(line)
            except: continue
            cid = o.get("cluster_id")
            if not cid: continue
            verdicts[cid] = o.get("verdict")
            sub_dups[cid] = [sd for sd in o.get("sub_dups", []) if len(sd.get("members", [])) >= 2]
            rats[cid] = o.get("rationale", "")
    return verdicts, sub_dups, rats


def reconcile(my_v, sa_v, sa_sub_dups):
    """Returns per-cluster reconciled verdict + list of merge_groups.

    merge_groups is a list of dicts: {members: [aspect_ids], source: 'cluster' or 'sub_dup', concept: str}
    """
    out = {}
    for cid in set(my_v) | set(sa_v):
        m = my_v.get(cid, "DIFF")
        s = sa_v.get(cid, "DIFF")
        merge_groups = []
        # Same: use that verdict
        if m == s:
            verdict = m
        # Conservative on disagreement: pick closer-to-DIFF
        else:
            order = {"DUP": 2, "OVERLAP": 1, "DIFF": 0}
            verdict = m if order[m] < order[s] else s
        # If reconciled to DUP, collapse the whole cluster
        if verdict == "DUP":
            # Members come from sub_dup if subagent provided one covering all; else need cluster member list
            sds = sa_sub_dups.get(cid, [])
            if sds:
                # Use the largest sub_dup as the collapse group
                largest = max(sds, key=lambda sd: len(sd["members"]))
                merge_groups.append({"members": largest["members"], "concept": largest.get("concept", "")})
            # else: we'll fill in members from the cluster data later (in build_r2_post)
        elif verdict == "OVERLAP":
            # Honor subagent's sub-DUPs (collapse only those sub-groups)
            for sd in sa_sub_dups.get(cid, []):
                merge_groups.append({"members": sd["members"], "concept": sd.get("concept", "")})
        # DIFF: no merges
        out[cid] = {"verdict": verdict, "merge_groups": merge_groups}
    return out


def build_r2_post_for_task(task: str, repo: Path, reconciled: dict, out_root: Path):
    task_dir = repo / "runs/validity_full/v2" / task
    aspects_json = json.loads((task_dir / "aspects.json").read_text())
    aspect_meta = {a["aspect_id"]: a for a in aspects_json}
    clusters_df = pd.read_parquet(repo / f"outputs/v2_analysis/{task}__aspect_clusters.parquet")
    cluster_members = clusters_df.groupby("cluster_id")["aspect_id"].apply(list).to_dict()
    cluster_nscored = clusters_df.set_index("aspect_id")["n_dps_seen"].to_dict()

    # Gather all merge groups for this task
    task_merges = []
    for cid, rec in reconciled.items():
        if not cid.startswith(f"{task}/"): continue
        cluster_num = int(cid.split("/c")[-1])
        # If verdict is DUP and merge_groups empty (no sub_dup from subagent), use full cluster member list
        if rec["verdict"] == "DUP" and not rec["merge_groups"]:
            members = cluster_members.get(cluster_num, [])
            if members:
                rec["merge_groups"].append({"members": members, "concept": ""})
        for mg in rec["merge_groups"]:
            task_merges.append({
                "cluster_id": cid,
                "verdict": rec["verdict"],
                "members": mg["members"],
                "concept": mg.get("concept", ""),
            })

    # Now build the collapse map: old aspect_id -> new aspect_id
    # New aspect_id = canonical = member with highest n_dps_seen (fallback: lex smallest)
    collapse_map = {}  # old_id -> canonical_new_id
    new_aspect_groups = []  # list of {canonical, merged_from, concept}
    used_canonical = set()
    for merge in task_merges:
        members = [m for m in merge["members"] if m in aspect_meta]
        if len(members) < 2: continue
        # Skip if any member is already in another merge (cluster overlap)
        if any(m in collapse_map for m in members): continue
        canonical = max(members, key=lambda m: (cluster_nscored.get(m, 0), -int(m[1:]) if m[1:].isdigit() else 0))
        if canonical in used_canonical: continue
        used_canonical.add(canonical)
        for m in members:
            collapse_map[m] = canonical
        new_aspect_groups.append({
            "canonical": canonical,
            "canonical_name": aspect_meta[canonical].get("name", ""),
            "merged_from": members,
            "concept": merge["concept"],
            "source_cluster": merge["cluster_id"],
        })

    # Build new aspects.json (R2_post inventory)
    aspects_r2_post = []
    for a in aspects_json:
        aid = a["aspect_id"]
        if aid in collapse_map:
            if collapse_map[aid] == aid:
                # This is a canonical that absorbed others
                grp = next(g for g in new_aspect_groups if g["canonical"] == aid)
                a_new = dict(a)
                a_new["r2_post_merged_from"] = grp["merged_from"]
                a_new["r2_post_merge_concept"] = grp["concept"]
                aspects_r2_post.append(a_new)
            # Else: this aspect was absorbed; drop it
        else:
            aspects_r2_post.append(a)

    # Write outputs
    out_dir = out_root / task
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aspects_r2_post.json").write_text(json.dumps(aspects_r2_post, indent=2))
    (out_dir / "collapse_map.json").write_text(json.dumps({
        "task": task,
        "n_aspects_before": len(aspects_json),
        "n_aspects_after": len(aspects_r2_post),
        "n_collapse_groups": len(new_aspect_groups),
        "collapse_map": collapse_map,
        "groups": new_aspect_groups,
    }, indent=2))
    return len(aspects_json), len(aspects_r2_post), len(new_aspect_groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="outputs/v2_analysis/r2_post")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out_root = repo / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    my_v = load_my_verdicts(repo)
    sa_v, sa_sub_dups, sa_rats = load_subagent(repo)
    reconciled = reconcile(my_v, sa_v, sa_sub_dups)

    # Save reconciled verdicts as TSV
    with (out_root / "reconciled_verdicts.tsv").open("w") as f:
        f.write("cluster_id\tmy_verdict\tsa_verdict\treconciled_verdict\tn_merge_groups\n")
        for cid in sorted(reconciled):
            f.write(f"{cid}\t{my_v.get(cid,'')}\t{sa_v.get(cid,'')}\t{reconciled[cid]['verdict']}\t{len(reconciled[cid]['merge_groups'])}\n")

    TASKS = ["peer_review", "math", "notice_and_comment", "press_releases",
             "humor", "news_homepages", "patents", "code_review", "creative_writing"]
    print(f"{'task':<22} {'before':>7} {'after':>7} {'merges':>7} {'saved':>7}")
    print("-" * 55)
    total_b, total_a = 0, 0
    for t in TASKS:
        b, a, m = build_r2_post_for_task(t, repo, reconciled, out_root)
        print(f"{t:<22} {b:>7} {a:>7} {m:>7} {b-a:>7}")
        total_b += b; total_a += a
    print("-" * 55)
    print(f"{'TOTAL':<22} {total_b:>7} {total_a:>7} {'':>7} {total_b-total_a:>7}")


if __name__ == "__main__":
    main()
