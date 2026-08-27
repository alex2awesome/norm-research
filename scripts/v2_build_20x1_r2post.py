"""Build 1-dp × 20-aspect prompts using R2_post aspects (post-collapse).

Skip (dp, canonical_aspect) cells where any merged_from R2 member already has
Qwen scoring (qwen_thinking_fp8 or qwen_thinking_v1), since cluster members are
highly correlated and we trust the existing coverage.

Per task: top-K best-covered Claude dps × R2_post canonical aspects partitioned
into bundles of ~20. Each prompt = 1 dp × 20 aspects.

Output: runs/validity_full/v2/<task>/judge_prompts_20x1_r2post/{key}.txt
        where key = b20p_<N>__d<dp_id>
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "scripts")
from v2_assemble_judge_prompt import build_prompt, SYSTEM_PROMPT

N_ASPECTS_PER_BUNDLE_DEFAULT = 20
DB = Path("outputs/v2_db/cells_v1")
TRUST_TABLE = Path("outputs/v2_analysis/per_aspect_trust.csv")
TRUST_THRESHOLD = 0.5  # if ρ(llama, claude) >= this OR ρ(code, claude) >= this, skip the aspect


def load_trustable_aspects(task: str, threshold: float):
    """Return set of aspect_ids that should be SKIPPED (already trustworthy via llama/code)."""
    if not TRUST_TABLE.exists():
        return set()
    df = pd.read_csv(TRUST_TABLE)
    df = df[df["task"] == task]
    # Either llama or code agrees with claude above threshold
    mask = (df["rho_llama_vs_claude"].fillna(0) >= threshold) | (df["rho_code_vs_claude"].fillna(0) >= threshold)
    return set(df.loc[mask, "aspect_id"])


QWEN_JUDGES_FOR_SKIP = [
    "qwen_thinking_fp8", "qwen_thinking_v1",
    "qwen_thinking_fp8_20x1", "qwen_thinking_fp8_20x1_r2post",
]

def load_qwen_coverage(task):
    """Return set of (dp_id, aspect_id) already validly scored by any Qwen at p0.

    Only counts cells where the score is not null (i.e., parsed AND applicable=True,
    or applicable=False with non-null applicable flag). This means parse-failed cells
    are NOT considered covered — they'll be re-attempted.
    """
    cells = set()
    for judge in QWEN_JUDGES_FOR_SKIP:
        for ext in ["data.parquet", "data.csv.gz"]:
            p = DB / f"task={task}/judge={judge}" / ext
            if p.exists():
                df = pd.read_parquet(p) if ext.endswith("parquet") else pd.read_csv(p, compression="gzip")
                if "paraphrase_idx" in df.columns:
                    df = df[df["paraphrase_idx"] == 0]
                # Skip rows where parse failed (would have no score AND no applicable)
                if "parse_status" in df.columns:
                    df = df[df["parse_status"] == "ok"]
                for _, r in df.iterrows():
                    cells.add((r["datapoint_id"], r["aspect_id"]))
                break
    return cells


def build_for_task(task: str, repo: Path, top_k_dps: int,
                   n_aspects_per_bundle: int = N_ASPECTS_PER_BUNDLE_DEFAULT,
                   out_subdir: str = "judge_prompts_20x1_r2post",
                   paraphrase_idx: int = 0,
                   dp_start: int = 0,
                   dp_end: int = None):
    task_dir = repo / "runs/validity_full/v2" / task
    r2p_dir = repo / "outputs/v2_analysis/r2_post" / task

    # Load R2_post collapse map and merged_from sets per canonical
    cmap_data = json.loads((r2p_dir / "collapse_map.json").read_text())
    canonical_to_merged = {g["canonical"]: set(g["merged_from"]) for g in cmap_data["groups"]}
    collapse_map = cmap_data["collapse_map"]  # old_id → canonical_id

    # R2_post aspect inventory (canonicals + un-collapsed)
    aspects_r2p = json.loads((r2p_dir / "aspects_r2_post.json").read_text())
    aspect_ids = [a["aspect_id"] for a in aspects_r2p]

    # Use R2_post enrichments at the requested paraphrase
    enr_dir = r2p_dir / "judge_enrichment_paraphrased"
    enr_suffix = f"_p{paraphrase_idx}.json"
    aspect_ids = [aid for aid in aspect_ids if (enr_dir / f"{aid}{enr_suffix}").exists()]

    # Skip aspects where llama or code already agrees with Claude well
    trustable = load_trustable_aspects(task, TRUST_THRESHOLD)
    # An R2_post canonical is trustable if it OR any of its merged_from is in `trustable`
    n_before = len(aspect_ids)
    aspect_ids = [aid for aid in aspect_ids
                  if aid not in trustable
                  and not any(m in trustable for m in canonical_to_merged.get(aid, set()))]
    n_skipped_trust = n_before - len(aspect_ids)

    datapoints = json.loads((task_dir / "datapoints.json").read_text())
    dp_map = {d["datapoint_id"]: d for d in datapoints}

    # Build target dp set
    if dp_end is not None:
        # Explicit range: take dps[dp_start:dp_end] in datapoint order
        all_dp_ids = [d["datapoint_id"] for d in datapoints]
        top_dps = all_dp_ids[dp_start:dp_end]
        print(f"  {task}: explicit dp range [{dp_start}:{dp_end}] = {len(top_dps)} dps")
    else:
        # Default: best-covered Claude dps, then extend
        from v2_claude_analyses import load_claude_matrix, build_dp_aspect_matrix, find_best_covered_dps
        df_cl = load_claude_matrix(task, repo)
        wide = build_dp_aspect_matrix(df_cl)
        claude_top = find_best_covered_dps(wide, k=top_k_dps)
        claude_set = set(claude_top)
        extra = [d["datapoint_id"] for d in datapoints if d["datapoint_id"] not in claude_set]
        extra = extra[:max(0, top_k_dps - len(claude_top))]
        top_dps = list(claude_top) + extra
        print(f"  {task}: target {top_k_dps} dps = {len(claude_top)} claude-scored + {len(extra)} extension")

    # Existing Qwen coverage
    qwen_covered = load_qwen_coverage(task)

    # For each canonical aspect, the "equivalent R2 set" = canonical itself + merged_from
    aspect_equiv = {aid: {aid} | canonical_to_merged.get(aid, set()) for aid in aspect_ids}

    # Bundle aspects into N-packs
    bundles = []
    bundle_prefix = f"b{n_aspects_per_bundle}p"
    for i in range(0, len(aspect_ids), n_aspects_per_bundle):
        chunk = aspect_ids[i:i + n_aspects_per_bundle]
        bid = f"{bundle_prefix}_{i // n_aspects_per_bundle}"
        bundles.append({"bundle_id": bid, "aspect_ids": chunk})

    out_dir = task_dir / out_subdir
    out_dir.mkdir(exist_ok=True, parents=True)
    (task_dir / f"judge_system_{out_subdir}.txt").write_text(SYSTEM_PROMPT)

    manifest = []
    n_skipped_all = 0
    for bundle in bundles:
        enrs = []
        for aid in bundle["aspect_ids"]:
            p = enr_dir / f"{aid}{enr_suffix}"
            if p.exists(): enrs.append(json.loads(p.read_text()))
        if len(enrs) < n_aspects_per_bundle * 0.5: continue
        for dp_id in top_dps:
            d = dp_map.get(dp_id)
            if d is None: continue
            # Filter aspects: skip canonical if ANY merged_from member already Qwen-scored for this dp
            filtered_enrs = []
            for e in enrs:
                canonical = e["aspect_id"]
                equiv = aspect_equiv.get(canonical, {canonical})
                if any((dp_id, m) in qwen_covered for m in equiv):
                    n_skipped_all += 1
                    continue
                filtered_enrs.append(e)
            if len(filtered_enrs) == 0: continue
            sub_bundle = {"bundle_id": bundle["bundle_id"], "aspect_ids": [e["aspect_id"] for e in filtered_enrs]}
            texts = [(d["datapoint_id"], d["text"])]
            try:
                system, user = build_prompt(sub_bundle, filtered_enrs, texts, text_max_chars=4000)
            except Exception:
                continue
            key = f"{bundle['bundle_id']}__p{paraphrase_idx}__d{dp_id}"
            (out_dir / f"{key}.txt").write_text(system + "\n=== USER ===\n" + user)
            manifest.append({
                "key": key, "bundle_id": bundle["bundle_id"],
                "aspect_ids": [e["aspect_id"] for e in filtered_enrs],
                "datapoint_id": dp_id,
                "paraphrase_idx": paraphrase_idx,
            })
    (task_dir / f"judge_manifest_{out_subdir}.json").write_text(json.dumps(manifest, indent=1))
    print(f"  {task:<22}  bsz={n_aspects_per_bundle}  asp_skipped_trust={n_skipped_trust:>3}  bundles={len(bundles):>3}  prompts={len(manifest):>5}  cells_skipped_already_qwen={n_skipped_all}")
    return len(manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--tasks", nargs="+", default=[
        "peer_review", "math", "notice_and_comment", "press_releases",
        "humor", "news_homepages", "patents", "code_review", "creative_writing"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--bundle-size", type=int, default=N_ASPECTS_PER_BUNDLE_DEFAULT)
    ap.add_argument("--out-subdir", type=str, default="judge_prompts_20x1_r2post",
                    help="output dir name under each task dir (avoid clobbering existing prompts)")
    ap.add_argument("--paraphrase", type=int, default=0, choices=[0, 1, 2],
                    help="which paraphrase to use (0=canonical, 1, 2)")
    ap.add_argument("--dp-start", type=int, default=0,
                    help="for explicit ranges (e.g., 500 to get dps 501+)")
    ap.add_argument("--dp-end", type=int, default=None,
                    help="if set, takes dps[dp_start:dp_end] from datapoints.json in sequential order — overrides --top-k claude-prioritized selection")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    total = 0
    for t in args.tasks:
        try:
            total += build_for_task(t, repo, args.top_k, args.bundle_size,
                                    args.out_subdir, args.paraphrase,
                                    args.dp_start, args.dp_end)
        except Exception as e:
            print(f"  {t}: ERR {e}")
    print(f"\nTOTAL prompts: {total}")


if __name__ == "__main__":
    main()
