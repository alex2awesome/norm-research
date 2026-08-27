"""Build 1-dp × 20-aspect prompts for Qwen across all tasks.

Per task: top-K best-covered Claude dps × all aspects partitioned into bundles of ~20.
Each prompt = 1 dp × 20 aspects.

Output: runs/validity_full/v2/<task>/judge_prompts_20x1/{key}.txt
        where key = b20_<N>__d<dp_id>
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, "scripts")
from v2_assemble_judge_prompt import build_prompt, SYSTEM_PROMPT

N_ASPECTS_PER_BUNDLE = 20


def build_for_task(task: str, repo: Path, top_k_dps: int):
    task_dir = repo / "runs/validity_full/v2" / task
    enr_dir = task_dir / "judge_enrichment_paraphrased"
    aspects_json = json.loads((task_dir / "aspects.json").read_text())
    aspect_ids = sorted([a["aspect_id"] for a in aspects_json])
    aspect_ids = [aid for aid in aspect_ids if (enr_dir / f"{aid}_p0.json").exists()]

    bundles = []
    for i in range(0, len(aspect_ids), N_ASPECTS_PER_BUNDLE):
        chunk = aspect_ids[i:i + N_ASPECTS_PER_BUNDLE]
        bid = f"b20_{i // N_ASPECTS_PER_BUNDLE}"
        bundles.append({"bundle_id": bid, "aspect_ids": chunk})

    from v2_claude_analyses import load_claude_matrix, build_dp_aspect_matrix, find_best_covered_dps
    df = load_claude_matrix(task, repo)
    wide = build_dp_aspect_matrix(df)
    top_dps = find_best_covered_dps(wide, k=top_k_dps)

    datapoints = json.loads((task_dir / "datapoints.json").read_text())
    dp_map = {d["datapoint_id"]: d for d in datapoints}

    out_dir = task_dir / "judge_prompts_20x1"
    out_dir.mkdir(exist_ok=True, parents=True)
    (task_dir / "judge_system_20x1.txt").write_text(SYSTEM_PROMPT)

    manifest = []
    for bundle in bundles:
        enrs = []
        for aid in bundle["aspect_ids"]:
            p = enr_dir / f"{aid}_p0.json"
            if p.exists():
                enrs.append(json.loads(p.read_text()))
        if len(enrs) < N_ASPECTS_PER_BUNDLE * 0.5:
            continue
        for dp_id in top_dps:
            d = dp_map.get(dp_id)
            if d is None:
                continue
            texts = [(d["datapoint_id"], d["text"])]
            try:
                system, user = build_prompt(bundle, enrs, texts, text_max_chars=4000)
            except Exception:
                continue
            key = f"{bundle['bundle_id']}__d{dp_id}"
            (out_dir / f"{key}.txt").write_text(system + "\n=== USER ===\n" + user)
            manifest.append({
                "key": key, "bundle_id": bundle["bundle_id"],
                "aspect_ids": [e["aspect_id"] for e in enrs],
                "datapoint_id": dp_id,
            })
    (task_dir / "judge_manifest_20x1.json").write_text(json.dumps(manifest, indent=1))
    print(f"  {task:<22} {len(bundles)} bundles × {len(top_dps)} dps → {len(manifest)} prompts")
    return len(manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--tasks", nargs="+", default=[
        "peer_review", "math", "notice_and_comment", "press_releases",
        "humor", "news_homepages", "patents", "code_review", "creative_writing"])
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    total = 0
    for t in args.tasks:
        total += build_for_task(t, repo, args.top_k)
    print(f"\nTOTAL prompts: {total}")


if __name__ == "__main__":
    main()
