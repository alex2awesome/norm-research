"""Build TARGETED Claude judge prompts for the missing (aspect, dp) cells.

Strategy:
1. For each task, find the 40 datapoints that already have the MOST aspects
   cleanly scored (maximize current coverage as the starting basis).
2. Compute missing aspects per dp = aspects not yet scored on that dp.
3. Group the 40 dps into 4 chunks of 10.
4. Per chunk: union of missing aspects across the 10 dps → split into
   small bundles of N_BUNDLE_SIZE aspects each.
5. For each (chunk × ad-hoc bundle), build a prompt using the existing
   build_prompt() with the same system+enrichment scaffolding.

Output:
  runs/validity_full/v2/<task>/judge_prompts_targeted/<key>.txt
  runs/validity_full/v2/<task>/targeted_manifest.json   — list of all keys
  runs/validity_full/v2/<task>/claude_judge_batch_100..NN.json  — batches for subagent dispatch

Naming: keys are "t<chunk_idx>__bX" where X enumerates ad-hoc bundles within the chunk.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from v2_assemble_judge_prompt import build_prompt, SYSTEM_PROMPT

REPO = Path("runs/validity_full/v2")


def parse_resp(raw):
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    if not raw.startswith("{"):
        s, e = raw.find("{"), raw.rfind("}")
        if s >= 0 and e > s: raw = raw[s:e+1]
    return json.loads(raw)


def build_for_task(task: str, n_dps: int = 40, bundle_size: int = 5,
                    chunk_size: int = 10, batch_size: int = 15,
                    new_batch_start: int = 100):
    asps = json.loads((REPO/task/"aspects.json").read_text())
    aid_set = set(a["aspect_id"] for a in asps)
    dps = json.loads((REPO/task/"datapoints.json").read_text())
    dp_by_id = {d["datapoint_id"]: d for d in dps}

    # Aggregate current cleanly-scored (dp, aspect) pairs
    cd = REPO/task/"judge_responses_claude"
    dp_aspect = collections.defaultdict(set)
    for f in cd.glob("*.json"):
        try:
            obj = parse_resp(f.read_text())
            for tr in obj.get("results", []):
                tid = tr.get("text_id")
                if not tid: continue
                for sc in tr.get("scores", []):
                    aid = sc.get("aspect_id")
                    if aid in aid_set and sc.get("score") is not None:
                        dp_aspect[tid].add(aid)
        except Exception:
            pass

    # Top n_dps datapoints by aspect coverage
    ranked = sorted(dp_aspect.items(), key=lambda x: -len(x[1]))[:n_dps]
    chosen_dps = [d for d, _ in ranked]
    missing_per_dp = {d: aid_set - dp_aspect.get(d, set()) for d in chosen_dps}

    # Load enrichments (one per aspect)
    enr_dir = REPO/task/"judge_enrichment"
    enr_by_id = {}
    for f in enr_dir.glob("*.json"):
        try:
            obj = json.loads(f.read_text())
            enr_by_id[obj["aspect_id"]] = obj
        except Exception:
            pass

    # Group chosen_dps into chunks
    chunks = [chosen_dps[i:i+chunk_size] for i in range(0, len(chosen_dps), chunk_size)]

    out_dir = REPO/task/"judge_prompts_targeted"
    out_dir.mkdir(exist_ok=True, parents=True)
    # Clear existing targeted prompts to avoid stale collisions
    for f in out_dir.glob("*.txt"):
        f.unlink()

    manifest = []
    for chunk_idx, chunk in enumerate(chunks):
        # Union of missing aspects across this chunk's dps
        union_missing = set()
        for dp in chunk:
            union_missing |= missing_per_dp.get(dp, set())
        # Stable order
        union_missing = sorted(union_missing)
        # Split into bundles of bundle_size
        bundles = [union_missing[i:i+bundle_size]
                   for i in range(0, len(union_missing), bundle_size)]

        for bundle_idx, bundle_aids in enumerate(bundles):
            # Skip bundle if any aspect lacks an enrichment
            enrs = [enr_by_id[aid] for aid in bundle_aids if aid in enr_by_id]
            if len(enrs) < len(bundle_aids):
                # Fall back to a "minimal enrichment" object for missing ones
                for aid in bundle_aids:
                    if aid not in enr_by_id:
                        # Get name+description from aspects.json
                        a_obj = next((a for a in asps if a["aspect_id"] == aid), None)
                        if a_obj:
                            enrs.append({
                                "aspect_id": aid,
                                "name": a_obj.get("name", "?"),
                                "description": a_obj.get("description", "?"),
                                "what_to_look_for": [],
                                "applicability_note": "",
                                "calibration_exemplars": [],
                            })
            # Build the prompt
            fake_bundle = {"bundle_id": f"t{chunk_idx}__b{bundle_idx}",
                            "aspect_ids": bundle_aids}
            texts = [(d, dp_by_id[d]["text"]) for d in chunk if d in dp_by_id]
            system, user = build_prompt(fake_bundle, enrs, texts)
            key = f"t{chunk_idx}__b{bundle_idx}"
            (out_dir / f"{key}.txt").write_text(system + "\n=== USER ===\n" + user)
            manifest.append({
                "key": key,
                "chunk_idx": chunk_idx,
                "bundle_idx": bundle_idx,
                "aspect_ids": bundle_aids,
                "datapoint_ids": chunk,
            })

    (REPO/task/"targeted_manifest.json").write_text(json.dumps(manifest, indent=1))

    # Build claude_judge_batch_<NN>.json files (size = batch_size)
    keys = [m["key"] for m in manifest]
    n_batches = 0
    for i in range(0, len(keys), batch_size):
        batch = keys[i:i+batch_size]
        bn = new_batch_start + i // batch_size
        bf = REPO/task/f"claude_judge_batch_{bn:02d}.json"
        bf.write_text(json.dumps(batch))
        n_batches += 1
    return len(manifest), n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["peer_review","math","notice_and_comment","press_releases",
                              "humor","news_homepages","patents","creative_writing"])
    ap.add_argument("--n-dps", type=int, default=40)
    ap.add_argument("--bundle-size", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--batch-start", type=int, default=100)
    args = ap.parse_args()

    print(f"{'task':<22} {'prompts':>9} {'batches':>9}")
    print("-" * 45)
    total = 0
    for task in args.tasks:
        n_prompts, n_batches = build_for_task(task, n_dps=args.n_dps,
                                                bundle_size=args.bundle_size,
                                                chunk_size=args.chunk_size,
                                                batch_size=args.batch_size,
                                                new_batch_start=args.batch_start)
        print(f"{task:<22} {n_prompts:>9} {n_batches:>9}")
        total += n_prompts
    print(f"\nTotal prompts: {total}, batches: {(total + args.batch_size - 1) // args.batch_size}")


if __name__ == "__main__":
    main()
