"""End-to-end setup for a NEW task in v2:

  1. Build aspects.json by concatenating R2 expansions across buckets.
  2. Sample n=N datapoints from the task's source CSV (50/50 balanced).
  3. Cluster aspects into ~30 bundles via embeddings.
  4. Sample N_TEXT random datapoints + dump as 'exemplar candidate sample'
     for subagents to read & either pick from or synthesize from.
  5. Split aspects into subagent batches.

After this, run the subagent waves (code + judge), then paraphrase,
then build_judge_jobs, then sk3 run.

Usage:
  python scripts/v2_setup_task.py --task creative_writing --n-dps 5000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from v2_task_registry import task_config


def _slugify_aspect(s: str) -> str:
    return re.sub(r'\W+', '_', s.strip().lower())[:40]


def step1_aspects(cfg):
    """Concatenate R2 expansions across (general/specific/hyper_specific)
    into a single aspects.json with stable aspect_ids."""
    task_dir = cfg["task_dir"]
    out_path = task_dir / "aspects.json"
    if out_path.exists():
        print(f"  [aspects.json exists, skipping]")
        return json.loads(out_path.read_text())

    aspects = []
    seen = set()
    for src in cfg["aspects_sources"]:
        data = json.loads(src.read_text())
        bucket = src.stem.split("_")[1]  # general / specific / hyper_specific
        # Format may be list of merged_groups OR dict with 'merged_groups'
        if isinstance(data, dict) and "merged_groups" in data:
            groups = data["merged_groups"]
        elif isinstance(data, list):
            groups = data
        else:
            groups = list(data.values())
        for g in groups:
            name = g.get("merged_name") or g.get("name") or g.get("title", "")
            desc = g.get("merged_description") or g.get("description", "")
            if not name: continue
            # Dedup by name across buckets
            key = name.lower()
            if key in seen: continue
            seen.add(key)
            aspect_id = f"a{len(aspects)}"
            child_metric_ids = []
            for child in (g.get("children") or g.get("members") or []):
                cn = child.get("name") if isinstance(child, dict) else child
                if cn: child_metric_ids.append(cn)
            aspects.append({
                "aspect_id": aspect_id,
                "name": name,
                "description": desc,
                "bucket": bucket,
                "n_children": len(child_metric_ids),
                "child_examples": child_metric_ids[:8],
            })
    out_path.write_text(json.dumps(aspects, indent=1))
    print(f"  wrote {out_path}: {len(aspects)} aspects "
          f"(across {len(cfg['aspects_sources'])} R2 files)")
    return aspects


def step2_sample_datapoints(cfg, args):
    """Sample n=N stratified datapoints from the task's CSV."""
    task_dir = cfg["task_dir"]
    out_path = task_dir / "datapoints.json"
    if out_path.exists() and not args.force_resample:
        print(f"  [datapoints.json exists, skipping; "
              f"--force-resample to redo]")
        return json.loads(out_path.read_text())

    print(f"  loading {cfg['datapoints_csv']}...")
    df = pd.read_csv(cfg["datapoints_csv"], low_memory=False)
    print(f"  full shape: {df.shape}  cols: {list(df.columns)}")

    # Drop short texts (< 200 chars)
    text_col, label_col = cfg["text_col"], cfg["label_col"]
    df[text_col] = df[text_col].fillna("")
    df = df[df[text_col].str.len() >= 200].reset_index(drop=True)
    print(f"  after >=200 char filter: {len(df):,}")

    # Stratified sample: 50/50 by label
    rng = random.Random(args.seed)
    n_per_label = args.n_dps // 2
    sampled_frames = []
    for label in (0, 1):
        sub = df[df[label_col] == label]
        if len(sub) <= n_per_label:
            sampled_frames.append(sub)
        else:
            sampled_frames.append(sub.sample(n=n_per_label, random_state=args.seed))
    out_df = pd.concat(sampled_frames).sample(frac=1, random_state=args.seed
                                                ).reset_index(drop=True)
    print(f"  sampled {len(out_df)} balanced (target {args.n_dps})")
    print(f"  label balance: {out_df[label_col].value_counts().to_dict()}")
    print(f"  text-len mean={int(out_df[text_col].str.len().clip(upper=args.text_trunc).mean())} "
          f"max={int(out_df[text_col].str.len().clip(upper=args.text_trunc).max())}")

    records = []
    for i, row in out_df.iterrows():
        text = (row[text_col] or "")[:args.text_trunc]
        records.append({
            "datapoint_id": f"d{i:05d}",
            "judgement": int(row[label_col]),
            "text": text,
        })
    out_path.write_text(json.dumps(records, indent=1))
    print(f"  wrote {out_path}")
    return records


def step3_exemplar_sample(cfg, datapoints, args):
    """Dump a small RANDOM sample of datapoints per aspect for subagents to
    read. Subagents will either pick exemplars from these or synthesize."""
    task_dir = cfg["task_dir"]
    out_dir = task_dir / "exemplar_sample"
    out_dir.mkdir(exist_ok=True, parents=True)
    aspects = json.loads((task_dir / "aspects.json").read_text())

    # We don't have v1 judge scores for this task, so we just give each
    # subagent a SHARED random sample of ~45 datapoints. The subagent
    # reads them and decides which (if any) clearly demonstrate the rubric;
    # synthesizes if needed.
    rng = random.Random(args.seed)
    pool = rng.sample(datapoints,
                       min(args.exemplar_pool_size, len(datapoints)))
    pool_path = task_dir / "exemplar_random_pool.json"
    pool_records = [{"datapoint_id": d["datapoint_id"],
                      "label": d["judgement"],
                      "text_preview": (d["text"] or "")[:800]}
                     for d in pool]
    pool_path.write_text(json.dumps(pool_records, indent=1))
    print(f"  wrote shared random pool of {len(pool_records)} datapoints "
          f"to {pool_path}")
    print(f"  (subagents will read this & either pick or synthesize)")
    return pool


def step4_cluster_bundles(cfg, args):
    """Cluster aspects into ~30 bundles via embeddings + k-means."""
    task_dir = cfg["task_dir"]
    out_path = task_dir / "judge_bundles.json"
    if out_path.exists() and not args.force_recluster:
        print(f"  [judge_bundles.json exists, skipping]")
        return json.loads(out_path.read_text())

    aspects = json.loads((task_dir / "aspects.json").read_text())
    texts = [f"{a['name']}: {a['description']}" for a in aspects]
    print(f"  embedding {len(texts)} aspects via OpenAI text-embedding-3-small...")
    if "OPENAI_API_KEY" not in os.environ:
        kpath = Path.home() / ".openai-salt-lab-key.txt"
        if kpath.exists():
            os.environ["OPENAI_API_KEY"] = kpath.read_text().strip()

    import openai
    client = openai.OpenAI()
    embs = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts[i:i+100])
        embs.extend([d.embedding for d in resp.data])
    embs = np.array(embs)

    from sklearn.cluster import KMeans
    target_bundle_size = 7
    k = max(8, min(40, len(aspects) // target_bundle_size))
    print(f"  k-means k={k}")
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(embs)
    by_cluster = defaultdict(list)
    for a, lab in zip(aspects, km.labels_):
        by_cluster[int(lab)].append(a["aspect_id"])

    # Split clusters >10 into halves
    final = []
    for ids in by_cluster.values():
        if len(ids) <= 10:
            final.append(ids)
        else:
            for i in range(0, len(ids), 8):
                final.append(ids[i:i+8])

    aspect_by_id = {a["aspect_id"]: a for a in aspects}
    bundles = []
    for bid, ids in enumerate(final):
        bundles.append({
            "bundle_id": f"b{bid}",
            "aspect_ids": ids,
            "aspects": [{"aspect_id": aid,
                          "name": aspect_by_id[aid]["name"],
                          "description": aspect_by_id[aid]["description"]}
                         for aid in ids],
        })
    sizes = sorted(len(b["aspect_ids"]) for b in bundles)
    print(f"  {len(bundles)} bundles  sizes: min={sizes[0]} "
          f"median={sizes[len(sizes)//2]} max={sizes[-1]}")
    out_path.write_text(json.dumps(bundles, indent=1))
    return bundles


def step5_batch_aspects_for_subagents(cfg, args):
    """Split aspects into N batches for parallel subagent processing."""
    task_dir = cfg["task_dir"]
    aspects = json.loads((task_dir / "aspects.json").read_text())
    aids = [a["aspect_id"] for a in aspects]
    n_batches = args.n_subagent_batches
    batches = {f"batch_{i:02d}": aids[i::n_batches] for i in range(n_batches)}
    out = task_dir / "subagent_batches.json"
    out.write_text(json.dumps(batches, indent=1))
    print(f"  split {len(aids)} aspects into {n_batches} batches → {out}")
    for k, b in batches.items():
        print(f"    {k}: {len(b)} aspects")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n-dps", type=int, default=5000)
    ap.add_argument("--text-trunc", type=int, default=12000)
    ap.add_argument("--exemplar-pool-size", type=int, default=80,
                    help="random datapoints shared with subagents for exemplar selection")
    ap.add_argument("--n-subagent-batches", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-resample", action="store_true")
    ap.add_argument("--force-recluster", action="store_true")
    args = ap.parse_args()

    cfg = task_config(args.task)
    print(f"\n=== v2 setup: {args.task} ===")
    print(f"  task_dir: {cfg['task_dir']}")

    print("\n--- step 1: build aspects.json ---")
    aspects = step1_aspects(cfg)

    print("\n--- step 2: sample datapoints ---")
    datapoints = step2_sample_datapoints(cfg, args)

    print("\n--- step 3: shared exemplar pool ---")
    step3_exemplar_sample(cfg, datapoints, args)

    print("\n--- step 4: cluster bundles ---")
    bundles = step4_cluster_bundles(cfg, args)

    print("\n--- step 5: subagent batches ---")
    step5_batch_aspects_for_subagents(cfg, args)

    print(f"\n=== setup done for {args.task} ===")
    print(f"  task_dir: {cfg['task_dir']}")
    print(f"  next: fan out subagent waves (code + judge enrichment)")


if __name__ == "__main__":
    main()
