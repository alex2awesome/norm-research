"""Build R2_post bundles + enrichments + prompts for each task.

Per task:
  1. Read collapse_map.json from r2_post/<task>/
  2. Build judge_bundles_r2_post.json by remapping aspect_ids in existing bundles
  3. Build judge_enrichment_paraphrased_r2_post/ — copy canonical's p0/p1/p2 enrichments
     (placeholder; subagents will rewrite later to synthesize merged definitions)
  4. Generate prompts via build_prompt, write to judge_prompts_r2_post/
  5. Generate manifest_r2_post.json

The placeholder enrichments use the canonical aspect's existing enrichment unchanged.
A subsequent subagent step will rewrite them with merged content.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, "scripts")
from v2_assemble_judge_prompt import build_prompt, SYSTEM_PROMPT

TASKS = ["peer_review", "math", "notice_and_comment", "press_releases",
         "humor", "news_homepages", "patents", "code_review", "creative_writing"]


def build_for_task(task, repo):
    task_dir = repo / "runs/validity_full/v2" / task
    r2p = repo / "outputs/v2_analysis/r2_post" / task
    if not (r2p / "collapse_map.json").exists():
        print(f"  {task}: no collapse_map; skip"); return
    cmap_data = json.loads((r2p / "collapse_map.json").read_text())
    cmap = cmap_data["collapse_map"]
    groups = cmap_data["groups"]
    canonical_ids = {g["canonical"] for g in groups}
    canonical_to_merged = {g["canonical"]: g["merged_from"] for g in groups}

    # 1. Load existing bundles and remap
    bundles = json.loads((task_dir / "judge_bundles.json").read_text())
    new_bundles = []
    for b in bundles:
        new_aids = []
        seen = set()
        for aid in b["aspect_ids"]:
            new_aid = cmap.get(aid, aid)
            if new_aid not in seen:
                seen.add(new_aid)
                new_aids.append(new_aid)
        if new_aids:
            new_b = dict(b)
            new_b["aspect_ids"] = new_aids
            new_bundles.append(new_b)
    (r2p / "judge_bundles.json").write_text(json.dumps(new_bundles, indent=2))

    # 2. Build enrichments dir — copy canonical's p0/p1/p2 enrichments for canonicals;
    #    keep originals for non-collapsed aspects
    src_enr_dir = task_dir / "judge_enrichment_paraphrased"
    dst_enr_dir = r2p / "judge_enrichment_paraphrased"
    dst_enr_dir.mkdir(parents=True, exist_ok=True)
    # All aspect IDs that survive in new bundles
    surviving = set()
    for b in new_bundles:
        surviving.update(b["aspect_ids"])
    n_copied = 0
    n_missing = 0
    for aid in surviving:
        for p_idx in (0, 1, 2):
            src = src_enr_dir / f"{aid}_p{p_idx}.json"
            dst = dst_enr_dir / f"{aid}_p{p_idx}.json"
            if src.exists():
                # Add r2_post merge metadata if this is a canonical that absorbed others
                enr = json.loads(src.read_text())
                if aid in canonical_to_merged:
                    enr["r2_post_merged_from"] = canonical_to_merged[aid]
                    enr["r2_post_placeholder"] = True  # to be rewritten by subagent
                dst.write_text(json.dumps(enr, indent=2))
                n_copied += 1
            else:
                n_missing += 1

    # 3. Generate prompts
    datapoints = json.loads((task_dir / "datapoints.json").read_text())
    prompt_dir = r2p / "judge_prompts"
    prompt_dir.mkdir(exist_ok=True, parents=True)
    (r2p / "judge_system.txt").write_text(SYSTEM_PROMPT)
    chunks = [datapoints[i:i + 10] for i in range(0, len(datapoints), 10)]
    manifest = []
    n_skipped = 0
    for bundle in new_bundles:
        per_paraphrase_enrs = []
        bundle_ok = True
        for p_idx in range(3):
            enrs = []
            for aid in bundle["aspect_ids"]:
                ep = dst_enr_dir / f"{aid}_p{p_idx}.json"
                if ep.exists(): enrs.append(json.loads(ep.read_text()))
            if not enrs: bundle_ok = False; break
            per_paraphrase_enrs.append(enrs)
        if not bundle_ok:
            n_skipped += 1; continue
        for p_idx, enrs in enumerate(per_paraphrase_enrs):
            actual_aids = [e["aspect_id"] for e in enrs]
            for c_idx, chunk in enumerate(chunks):
                key = f"{bundle['bundle_id']}__p{p_idx}__c{c_idx}"
                texts = [(d["datapoint_id"], d["text"]) for d in chunk]
                system, user = build_prompt(bundle, enrs, texts, text_max_chars=4000)
                (prompt_dir / f"{key}.txt").write_text(system + "\n=== USER ===\n" + user)
                manifest.append({
                    "key": key, "bundle_id": bundle["bundle_id"],
                    "aspect_ids": actual_aids, "paraphrase_idx": p_idx,
                    "chunk_idx": c_idx,
                    "datapoint_ids": [d["datapoint_id"] for d in chunk],
                })
    (r2p / "judge_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"  {task}: bundles {len(bundles)}→{len(new_bundles)}, "
          f"enrichments copied {n_copied} (missing {n_missing}), "
          f"prompts {len(manifest)} (skipped {n_skipped} bundles)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    print("Building R2_post bundles + enrichments + prompts per task:")
    for t in args.tasks:
        build_for_task(t, repo)


if __name__ == "__main__":
    main()
