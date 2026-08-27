"""Enumerate all judge jobs: (bundle × paraphrase × text-chunk) records.

For each job, materialize the full prompt and write it to disk for the sk3
runner to consume. Also build a manifest of all jobs.

Job key: b<bundle_id>__p<paraphrase_idx>__c<chunk_idx>

Output:
  runs/validity_full/full_v2/judge_prompts/<key>.txt   (full user prompt)
  runs/validity_full/full_v2/judge_system.txt          (shared system prompt)
  runs/validity_full/full_v2/judge_manifest.json       (list of all jobs)

Manifest entry:
  {"key": "b0__p0__c0", "bundle_id": "b0",
   "aspect_ids": [...], "paraphrase_idx": 0, "chunk_idx": 0,
   "datapoint_ids": ["d00000", ...]}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
from v2_assemble_judge_prompt import build_prompt, SYSTEM_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paraphrases", type=int, default=3,
                    help="paraphrases per bundle (p0..pN-1)")
    ap.add_argument("--texts-per-chunk", type=int, default=10)
    ap.add_argument("--text-max-chars", type=int, default=4000)
    ap.add_argument("--task", default=None,
                    help="task name; reads from runs/validity_full/v2/<task>/. "
                         "If unset, uses legacy runs/validity_full/full_v2/.")
    args = ap.parse_args()

    if args.task:
        v2 = Path(f"runs/validity_full/v2/{args.task}")
    else:
        v2 = Path("runs/validity_full/full_v2")
    bundles = json.loads((v2 / "judge_bundles.json").read_text())
    datapoints = json.loads((v2 / "datapoints.json").read_text())
    enr_dir = v2 / "judge_enrichment_paraphrased"
    out_dir = v2 / "judge_prompts"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Save the shared system prompt once
    (v2 / "judge_system.txt").write_text(SYSTEM_PROMPT)

    # Chunk the datapoints
    chunks = [datapoints[i:i + args.texts_per_chunk]
              for i in range(0, len(datapoints), args.texts_per_chunk)]
    print(f"datapoints: {len(datapoints)}  chunks: {len(chunks)} "
          f"(texts_per_chunk={args.texts_per_chunk})")

    manifest = []
    n_skipped_bundles = 0
    for bundle in bundles:
        # For each paraphrase, load enrichment for each aspect in bundle
        per_paraphrase_enrs = []
        bundle_ok = True
        for p_idx in range(args.n_paraphrases):
            enrs = []
            for aid in bundle["aspect_ids"]:
                ep = enr_dir / f"{aid}_p{p_idx}.json"
                if ep.exists():
                    enrs.append(json.loads(ep.read_text()))
            if len(enrs) == 0:
                bundle_ok = False
                break
            per_paraphrase_enrs.append(enrs)
        if not bundle_ok:
            n_skipped_bundles += 1
            continue

        for p_idx, enrs in enumerate(per_paraphrase_enrs):
            actual_aspect_ids = [e["aspect_id"] for e in enrs]
            for c_idx, chunk in enumerate(chunks):
                key = f"{bundle['bundle_id']}__p{p_idx}__c{c_idx}"
                texts = [(d["datapoint_id"], d["text"]) for d in chunk]
                system, user = build_prompt(bundle, enrs, texts,
                                             text_max_chars=args.text_max_chars)
                # Write in v1's combined format for the sk3 runner:
                # <system>\n=== USER ===\n<user>
                (out_dir / f"{key}.txt").write_text(
                    system + "\n=== USER ===\n" + user)
                manifest.append({
                    "key": key,
                    "bundle_id": bundle["bundle_id"],
                    "aspect_ids": actual_aspect_ids,
                    "paraphrase_idx": p_idx,
                    "chunk_idx": c_idx,
                    "datapoint_ids": [d["datapoint_id"] for d in chunk],
                })

    (v2 / "judge_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} judge prompts to {out_dir}")
    print(f"  skipped {n_skipped_bundles} bundles without any enrichments")
    print(f"  manifest: {v2 / 'judge_manifest.json'}")

    # Stats
    sample_prompt = (out_dir / manifest[0]["key"]).with_suffix(".txt").read_text()
    sys_chars = len(SYSTEM_PROMPT)
    user_chars = len(sample_prompt)
    print(f"\nsample prompt stats (first job):")
    print(f"  system chars: {sys_chars}  (~{sys_chars//4} tokens)")
    print(f"  user chars:   {user_chars}  (~{user_chars//4} tokens)")
    print(f"  total input tokens (est): {(sys_chars + user_chars)//4}")


if __name__ == "__main__":
    main()
