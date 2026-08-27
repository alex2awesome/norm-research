"""STAGE 1 builder for the refreshed code_review E2-ladder task.

Item = one GitHub pull request from the pr_test_execution execution-labeled
corpus (datasets/code-review/pr_test_execution/batch_runs/<repo>/), identified
by (repo, pr_number). text/ctext = the full unified diff (title/description
are not available for this corpus locally -- see BUILD_PLAN.md). judgement =
1 if merged ("accepted" in the manifest, i.e. merged_at is non-null), else 0.

Source of truth for the labeled pool: /tmp/final_ladder_table.parquet (the
44,751-PR ladder table also used by the coding A-bank degeneracy audit,
project_a_bank_degeneracy_audit.md / datasets/code-review/gepa_revive_dead_pr.py).
Local diffs only exist for a 594-repo subset of that table; of those, 22
repos + 3,816 PRs have the actual .diff file present on this laptop.

Sampling: STABLE HASH (sha256 of "repo#pr_number"), never a seeded shuffle.
datapoint_id = "crb" + first 10 hex chars of that hash (stable regardless of
future pool growth/shrinkage).

Usage: python3 build_code_review_items.py
Writes outputs/metric_seam_pilot/tasks/code_review/items.json
"""
import hashlib
import json
import os
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
BATCH_RUNS = ROOT / "datasets/code-review/pr_test_execution/batch_runs"
LADDER = pathlib.Path("/tmp/final_ladder_table.parquet")
OUT = ROOT / "outputs/metric_seam_pilot/tasks/code_review"

N_ITEMS = 250
MIN_BYTES = 300          # drop near-trivial diffs (single-line/whitespace-only)
READ_CAP = 300_000       # cap raw "text" read (bytes) -- guards a 483MB outlier
HEAD, TAIL = 5000, 2500  # match methods/metric_seam/pilot/build_task.py canonical()


def canonical(text):
    if len(text) <= HEAD + TAIL + 500:
        return text
    return text[:HEAD] + "\n[...]\n" + text[-TAIL:]


def read_capped(path, cap=READ_CAP):
    size = os.path.getsize(path)
    if size <= cap:
        return open(path, errors="replace").read()
    # keep head + tail within the cap for outsized diffs (matches canonical()'s
    # own head/tail logic one level up, just at a coarser byte budget)
    half = cap // 2
    with open(path, "rb") as f:
        head = f.read(half)
        f.seek(max(size - half, half), os.SEEK_SET)
        tail = f.read()
    return head.decode("utf-8", "replace") + "\n[...RAW-TRUNCATED...]\n" + tail.decode("utf-8", "replace")


def main():
    df = pd.read_parquet(LADDER, columns=["repo", "pr_number", "judgement"])
    df = df.dropna(subset=["pr_number", "judgement"])
    df["pr_number"] = df["pr_number"].astype(int)
    df = df[df["judgement"].isin(["accepted", "rejected"])]

    def diff_path(r):
        return BATCH_RUNS / r.repo / "diffs" / f"pr_{r.pr_number}.diff"

    df["path"] = df.apply(diff_path, axis=1)
    df["exists"] = df["path"].apply(lambda p: p.exists())
    pool = df[df["exists"]].copy()
    pool["size"] = pool["path"].apply(lambda p: p.stat().st_size)
    pool = pool[pool["size"] >= MIN_BYTES]
    assert not pool.duplicated(subset=["repo", "pr_number"]).any(), "dupe (repo,pr_number) in pool"

    def stable_id(r):
        h = hashlib.sha256(f"{r.repo}#{r.pr_number}".encode()).hexdigest()
        return h, "crb" + h[:10]

    ids = pool.apply(stable_id, axis=1)
    pool["hash"] = [h for h, _ in ids]
    pool["datapoint_id"] = [d for _, d in ids]
    pool = pool.sort_values("hash")

    print(f"eligible pool: {len(pool)} PRs across {pool.repo.nunique()} repos "
          f"(judgement: {pool.judgement.value_counts().to_dict()})")

    sample = pool.head(N_ITEMS).copy()
    assert sample["datapoint_id"].is_unique
    assert len(sample) == N_ITEMS, f"only {len(sample)} eligible, need {N_ITEMS}"

    items = []
    for r in sample.itertuples():
        text = read_capped(r.path)
        items.append({
            "datapoint_id": r.datapoint_id,
            "judgement": 1 if r.judgement == "accepted" else 0,
            "text": text,
            "ctext": canonical(text),
            "repo": r.repo,           # kept as metadata for confound auditing;
            "pr_number": r.pr_number,  # NOT shown to the judge (see prompts builder)
        })

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(OUT / "items.json", "w"))

    n_pos = sum(it["judgement"] for it in items)
    print(f"wrote {len(items)} items -> {OUT / 'items.json'}")
    print(f"judgement distribution: accepted={n_pos} ({n_pos/len(items):.1%})  "
          f"rejected={len(items)-n_pos} ({1-n_pos/len(items):.1%})")
    print("repo composition:")
    print(sample.repo.value_counts())
    lens = [len(it["text"]) for it in items]
    print(f"text length chars: min={min(lens)} p50={sorted(lens)[len(lens)//2]} max={max(lens)}")


if __name__ == "__main__":
    main()
