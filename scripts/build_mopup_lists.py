"""Build greedy mop-up prompt order for tasks with heavy random coverage.

For each task: at paraphrase p0, walk over all (bundle, chunk) prompts in greedy
order, picking the one that flips the most dps from partial -> fully covered.
After flip-positive prompts exhausted, fall back to depth-first chunk-major order.

Output: runs/validity_full/v2/<task>/prompt_order_mopup.txt — one prompt key per line.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

DEFAULT_TASKS = ["math", "press_releases", "notice_and_comment", "humor", "creative_writing"]
N_BUNDLES = 30  # bundles per task (standard)


def build_for_task(task: str, repo_root: Path, response_dir_name: str):
    task_dir = repo_root / "runs" / "validity_full" / "v2" / task
    manifest = json.loads((task_dir / "judge_manifest.json").read_text())
    resp_dir = task_dir / response_dir_name

    # Index manifest entries: only p0 for the mop-up objective; restrict to bundles seen
    p0 = [m for m in manifest if m["paraphrase_idx"] == 0]
    bundles_seen = sorted({int(m["bundle_id"].replace("b", "")) for m in p0})
    n_bundles = len(bundles_seen)
    bundle_set = set(bundles_seen)

    # dp -> set of bundles already covered (for its chunk) at p0
    # Each manifest entry has text_ids = list of dp ids in the chunk
    dp_bundles = defaultdict(set)  # (chunk_idx, dp_id) -> set(bundle_ids)
    # Also need: for each (bundle, chunk) prompt, what dp ids it would add
    prompt_dps = {}  # key -> list[(chunk_idx, dp_id)]

    for m in p0:
        b = int(m["bundle_id"].replace("b", ""))
        c = m["chunk_idx"]
        # field name is 'datapoint_ids' in v2 manifest
        dps = m.get("datapoint_ids") or m.get("text_ids") or m.get("dp_ids") or []
        if not dps:
            # Fallback: read the prompt file to enumerate text_ids — too expensive; rely on manifest
            continue
        prompt_dps[m["key"]] = [(c, d) for d in dps]
        if (resp_dir / f"{m['key']}.json").exists():
            for d in dps:
                dp_bundles[(c, d)].add(b)

    n_dps = len(dp_bundles) or sum(len(v) for v in prompt_dps.values()) // 1
    # Ensure dp_bundles has an entry for every dp (even if 0 cached) so greedy can count
    for key, pairs in prompt_dps.items():
        for cd in pairs:
            dp_bundles.setdefault(cd, set())

    # Greedy: iteratively pick prompt with max flip-count
    remaining = {k for k, pairs in prompt_dps.items()
                 if not (resp_dir / f"{k}.json").exists()}
    fully = {cd for cd, bs in dp_bundles.items() if len(bs) >= n_bundles}
    print(f"  {task}: {len(dp_bundles)} dps, {len(fully)} already fully covered, "
          f"{len(remaining)}/{len(prompt_dps)} prompts to run", file=sys.stderr)

    ordered = []
    while remaining:
        best_key, best_score = None, -1
        for k in remaining:
            b_id = int(k.split("__")[0].replace("b", ""))
            score = 0
            for cd in prompt_dps[k]:
                if cd in fully:
                    continue
                bs = dp_bundles[cd]
                # would flip if this is the only missing bundle (or covers last needed)
                if b_id not in bs and len(bs) == n_bundles - 1:
                    score += 1
            if score > best_score:
                best_score, best_key = score, k
            if best_score >= 10:  # max possible per prompt; can short-circuit
                break
        if best_score <= 0:
            break  # exhausted flip-positive prompts
        # Apply pick
        ordered.append(best_key)
        b_id = int(best_key.split("__")[0].replace("b", ""))
        for cd in prompt_dps[best_key]:
            dp_bundles[cd].add(b_id)
            if len(dp_bundles[cd]) >= n_bundles:
                fully.add(cd)
        remaining.discard(best_key)

    print(f"  {task}: greedy flips selected {len(ordered)} prompts; "
          f"{len(fully)} fully-covered dps after", file=sys.stderr)

    # Now: for the rest, fall back to a softer greedy by partial coverage gain
    # (rank by how many dps are close to fully covered that this prompt touches)
    # Simpler: walk by (smallest gap remaining) per chunk first
    while remaining:
        best_key, best_score = None, -1
        for k in remaining:
            b_id = int(k.split("__")[0].replace("b", ""))
            score = 0
            for cd in prompt_dps[k]:
                if cd in fully:
                    continue
                bs = dp_bundles[cd]
                if b_id not in bs:
                    # closer to fully covered = higher score
                    gap = n_bundles - len(bs)
                    score += max(0, n_bundles - gap)  # bigger when bs is already large
            if score > best_score:
                best_score, best_key = score, k
        if best_key is None:
            break
        ordered.append(best_key)
        b_id = int(best_key.split("__")[0].replace("b", ""))
        for cd in prompt_dps[best_key]:
            dp_bundles[cd].add(b_id)
            if len(dp_bundles[cd]) >= n_bundles:
                fully.add(cd)
        remaining.discard(best_key)

    # Then: append any remaining p1/p2 prompts in depth-first order
    extras = sorted([(m["chunk_idx"], int(m["bundle_id"].replace("b", "")),
                      m["paraphrase_idx"], m["key"])
                     for m in manifest if m["paraphrase_idx"] > 0
                        and not (resp_dir / f"{m['key']}.json").exists()])
    ordered.extend(k for _, _, _, k in extras)

    out_path = task_dir / "prompt_order_mopup.txt"
    out_path.write_text("\n".join(ordered) + "\n")
    print(f"  {task}: wrote {len(ordered)} keys to {out_path}", file=sys.stderr)
    print(f"  {task}: final fully-covered = {len(fully)}", file=sys.stderr)
    return len(ordered), len(fully)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="repo root")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--response-dir", default="judge_responses_llama_bf16",
                    help="response dir name relative to task dir (use judge_responses_claude for Claude pool)")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    print(f"Repo: {repo}")
    print(f"Response dir: {args.response_dir}")
    for t in args.tasks:
        try:
            build_for_task(t, repo, args.response_dir)
        except FileNotFoundError as e:
            print(f"  {t}: SKIP ({e})", file=sys.stderr)


if __name__ == "__main__":
    main()
