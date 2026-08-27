#!/usr/bin/env python3
"""
Cross-link extracted peer_review norms (sk3) to the oral_spotlight and
best_papers label sets to build two norm-rich subsets.

Pipeline:
  1. Load oral_spotlight labels (train+eval+test) -> set of OpenReview paper_ids
     (after stripping venue prefix). Filter to tier in {oral, spotlight}.
  2. Load best_papers labels (train+eval+test) -> normalized titles -> paper_ids
     via title fuzzy-match against unified_papers.csv.gz (and against the
     sk3 peer_review_unified.parquet titles).
  3. Resolve paper_ids -> review_ids using peer_review_unified.parquet on sk3
     (review_id is the chunk unit_id).
  4. Stream the sk3 extraction chunks (gzipped jsonl) line-by-line, emit each
     record whose unit_id is in the matched review_id set.
  5. Write two gzipped jsonl subsets + a JSON manifest with counts.
  6. Sample 5 records from each subset for eyeballing.

Run modes:
  * --mode remote (default): does the heavy join on sk3 via ssh ControlMaster,
    then pulls back the output files. Recommended on laptop.
  * --mode local : assumes you already have the parquet and chunks under
    --remote-root (i.e. running from inside sk3). No ssh.

Usage (from laptop):
    python3 datasets/scripts/subset_openreview_awards.py

Outputs (laptop, after remote run):
    datasets/peer-review/extracted_norms_oral_spotlight.jsonl.gz
    datasets/peer-review/extracted_norms_best_papers.jsonl.gz
    datasets/peer-review/openreview_awards_subset_manifest.json
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shlex
import subprocess
import sys
import unicodedata
from pathlib import Path

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "datasets" / "peer-review"

ORAL_FILES = [
    DATASETS / "oral_spotlight" / "train.csv.gz",
    DATASETS / "oral_spotlight" / "eval.csv.gz",
    DATASETS / "oral_spotlight" / "test.csv.gz",
]
BEST_FILES = [
    DATASETS / "best_papers" / "built" / "best_papers_v2_train.csv.gz",
    DATASETS / "best_papers" / "built" / "best_papers_v2_eval.csv.gz",
    DATASETS / "best_papers" / "built" / "best_papers_v2_test.csv.gz",
]
UNIFIED_PAPERS = DATASETS / "unified_papers.csv.gz"
UNIFIED_REVIEWS = DATASETS / "unified_reviews.csv.gz"  # not strictly needed (parquet has review_ids), used as a cross-check.

OUT_ORAL = DATASETS / "extracted_norms_oral_spotlight.jsonl.gz"
OUT_BEST = DATASETS / "extracted_norms_best_papers.jsonl.gz"
OUT_MANIFEST = DATASETS / "openreview_awards_subset_manifest.json"
OUT_SAMPLES = DATASETS / "openreview_awards_subset_samples.json"

# sk3 paths
SK3_PARQUET = "/lfs/skampere3/0/alexspan/data/peer_review/peer_review_unified.parquet"
SK3_CHUNK_GLOB = "/lfs/skampere3/0/alexspan/data/peer_review/norm_extracted/extracted_qwen.chunks/*.jsonl.gz"
SK3_PYTHON = "/lfs/skampere3/0/alexspan/miniconda3/bin/python3"
SK3_HOST = "sk3"
SK3_SCRATCH = "/lfs/skampere3/0/alexspan/data/peer_review/norm_extracted/awards_subset_scratch"

SSH_OPTS = [
    "-o", "ControlMaster=auto",
    "-o", f"ControlPath={os.path.expanduser('~/.ssh/cm-%r@%h:%p')}",
    "-o", "ControlPersist=10m",
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def norm_title(t: str) -> str:
    """Aggressive normalize: NFKD-strip-accents, lowercase, strip non-alnum.
    Stable enough for exact-match join after both sides go through it."""
    if t is None:
        return ""
    t = unicodedata.normalize("NFKD", str(t))
    t = t.encode("ascii", "ignore").decode("ascii")
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t


def strip_venue_prefix(pid: str) -> str:
    """oral_spotlight uses 'iclr_S1uxsye0Z'; parquet has 'S1uxsye0Z'."""
    if not isinstance(pid, str):
        return ""
    if "_" in pid:
        head, rest = pid.split("_", 1)
        if head in {"iclr", "neurips", "icml", "tmlr", "colm", "emnlp", "elife"}:
            return rest
    return pid


def load_oral_spotlight_paper_ids() -> tuple[set[str], dict]:
    """Returns (set of bare openreview paper_ids for oral|spotlight, breakdown)."""
    import pandas as pd
    dfs = [pd.read_csv(f) for f in ORAL_FILES]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["tier"].isin(["oral", "spotlight"])].copy()
    df["paper_id_short"] = df["id"].apply(strip_venue_prefix)
    breakdown = {
        "n_papers_total": int(len(df)),
        "n_oral": int((df["tier"] == "oral").sum()),
        "n_spotlight": int((df["tier"] == "spotlight").sum()),
        "by_venue_year": df.groupby(["venue_key", "year", "tier"]).size().to_dict(),
    }
    # serialize tuple keys
    breakdown["by_venue_year"] = {
        f"{v}|{int(y)}|{t}": int(n)
        for (v, y, t), n in breakdown["by_venue_year"].items()
    }
    return set(df["paper_id_short"].astype(str)), breakdown, df[["paper_id_short", "tier", "venue_key", "year"]].drop_duplicates()


def load_best_papers_titles() -> tuple[list[dict], dict]:
    """Returns list of dicts with normalized title + meta for label=1 best papers."""
    import pandas as pd
    dfs = [pd.read_csv(f) for f in BEST_FILES]
    df = pd.concat(dfs, ignore_index=True)
    df_pos = df[df["label"] == 1].copy()
    df_pos["title_norm"] = df_pos["title"].apply(norm_title)
    records = df_pos[["dblp_key", "venue", "year", "title", "title_norm", "split"]].to_dict(orient="records")
    breakdown = {
        "n_total_rows": int(len(df)),
        "n_label1": int(len(df_pos)),
        "by_venue": df_pos["venue"].value_counts().to_dict(),
    }
    return records, breakdown


# ----------------------------------------------------------------------------
# Remote join (runs on sk3)
# ----------------------------------------------------------------------------
REMOTE_JOIN_PY = r"""
import gzip, io, json, os, re, sys, glob, time, unicodedata
import pandas as pd

PARQUET   = sys.argv[1]
CHUNK_GLOB = sys.argv[2]
ORAL_IDS_FILE  = sys.argv[3]  # newline-delimited bare openreview ids
BEST_TITLES_FILE = sys.argv[4]  # newline-delimited normalized titles
OUT_ORAL  = sys.argv[5]
OUT_BEST  = sys.argv[6]
OUT_MAP   = sys.argv[7]  # paper_id -> review_ids mapping (for manifest debug)

def nt(t):
    if t is None: return ""
    t = unicodedata.normalize("NFKD", str(t))
    t = t.encode("ascii","ignore").decode("ascii").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t

with open(ORAL_IDS_FILE) as f:
    oral_pids = {ln.strip() for ln in f if ln.strip()}
with open(BEST_TITLES_FILE) as f:
    best_titles = {ln.strip() for ln in f if ln.strip()}

print(f"[remote] loaded {len(oral_pids)} oral/spotlight paper_ids, {len(best_titles)} best-paper titles",
      file=sys.stderr, flush=True)

t0 = time.time()
df = pd.read_parquet(PARQUET)
df["review_id"] = df["review_id"].astype(str)
df["paper_id"]  = df["paper_id"].astype(str)
df["title_norm"] = df["title"].apply(nt)
print(f"[remote] parquet loaded: {len(df)} rows, {df['paper_id'].nunique()} papers in {time.time()-t0:.1f}s",
      file=sys.stderr, flush=True)

# ----- oral/spotlight: paper_id -> review_ids -----
oral_rows = df[df["paper_id"].isin(oral_pids)]
oral_review_ids = set(oral_rows["review_id"].tolist())
oral_paper_ids_matched = set(oral_rows["paper_id"].unique())
print(f"[remote] oral/spotlight matched {len(oral_paper_ids_matched)} papers -> {len(oral_review_ids)} reviews",
      file=sys.stderr, flush=True)

# ----- best papers: title_norm -> review_ids -----
title_rows = df[df["title_norm"].isin(best_titles)]
best_review_ids = set(title_rows["review_id"].tolist())
best_titles_matched = set(title_rows["title_norm"].unique())
best_paper_ids_matched = set(title_rows["paper_id"].unique())
print(f"[remote] best_papers matched {len(best_titles_matched)} titles -> "
      f"{len(best_paper_ids_matched)} papers -> {len(best_review_ids)} reviews",
      file=sys.stderr, flush=True)

# ----- write paper_id -> review_ids map for the manifest -----
map_obj = {
    "oral_spotlight": {
        "paper_ids_in_labels": len(oral_pids),
        "paper_ids_matched_in_parquet": len(oral_paper_ids_matched),
        "review_ids_total": len(oral_review_ids),
    },
    "best_papers": {
        "titles_in_labels": len(best_titles),
        "titles_matched_in_parquet": len(best_titles_matched),
        "paper_ids_matched": len(best_paper_ids_matched),
        "review_ids_total": len(best_review_ids),
    },
}
with open(OUT_MAP, "w") as f:
    json.dump(map_obj, f, indent=2)

# ----- stream chunks; filter by unit_id -----
n_total_extracts = 0
n_oral_extracts = 0
n_best_extracts = 0
n_oral_extracts_ok = 0
n_best_extracts_ok = 0
fo_oral = gzip.open(OUT_ORAL, "wt")
fo_best = gzip.open(OUT_BEST, "wt")
files = sorted(glob.glob(CHUNK_GLOB))
print(f"[remote] streaming {len(files)} chunk files", file=sys.stderr, flush=True)
for cf in files:
    with gzip.open(cf, "rt") as fi:
        for line in fi:
            n_total_extracts += 1
            # tiny prefix check first to skip JSON parsing on most lines
            try:
                rec = json.loads(line)
            except Exception:
                continue
            uid = rec.get("unit_id")
            if uid is None: continue
            uid = str(uid)
            if uid in oral_review_ids:
                fo_oral.write(line if line.endswith("\n") else line + "\n")
                n_oral_extracts += 1
                if rec.get("ok"): n_oral_extracts_ok += 1
            if uid in best_review_ids:
                fo_best.write(line if line.endswith("\n") else line + "\n")
                n_best_extracts += 1
                if rec.get("ok"): n_best_extracts_ok += 1
fo_oral.close()
fo_best.close()

summary = {
    "n_total_extractions_scanned": n_total_extracts,
    "n_oral_spotlight_extractions": n_oral_extracts,
    "n_oral_spotlight_extractions_ok": n_oral_extracts_ok,
    "n_best_papers_extractions": n_best_extracts,
    "n_best_papers_extractions_ok": n_best_extracts_ok,
    **map_obj,
}
print(json.dumps({"_summary": summary}), flush=True)
"""


def run_remote(oral_ids: set[str], best_titles: list[str]) -> dict:
    """Send the remote driver + the id lists to sk3, run it, pull outputs."""
    # 1. stage input files locally then scp.
    local_tmp = REPO / ".cache" / "subset_awards"
    local_tmp.mkdir(parents=True, exist_ok=True)
    oral_ids_local = local_tmp / "oral_ids.txt"
    best_titles_local = local_tmp / "best_titles.txt"
    driver_local = local_tmp / "remote_join.py"
    oral_ids_local.write_text("\n".join(sorted(oral_ids)) + "\n")
    best_titles_local.write_text("\n".join(sorted(best_titles)) + "\n")
    driver_local.write_text(REMOTE_JOIN_PY)

    # 2. ensure remote scratch exists and ship files
    subprocess.run(
        ["ssh", *SSH_OPTS, SK3_HOST, f"mkdir -p {shlex.quote(SK3_SCRATCH)}"],
        check=True,
    )
    for local_path in [oral_ids_local, best_titles_local, driver_local]:
        dest = f"{SK3_HOST}:{SK3_SCRATCH}/{local_path.name}"
        subprocess.run(["scp", "-q", *SSH_OPTS, str(local_path), dest], check=True)

    remote_oral_ids = f"{SK3_SCRATCH}/oral_ids.txt"
    remote_best_titles = f"{SK3_SCRATCH}/best_titles.txt"
    remote_driver = f"{SK3_SCRATCH}/remote_join.py"
    remote_out_oral = f"{SK3_SCRATCH}/extracted_norms_oral_spotlight.jsonl.gz"
    remote_out_best = f"{SK3_SCRATCH}/extracted_norms_best_papers.jsonl.gz"
    remote_out_map = f"{SK3_SCRATCH}/map.json"

    # 3. run the driver remotely; pin HOME to /lfs to avoid AFS token issues
    cmd_str = (
        f"export HOME=/lfs/skampere3/0/alexspan && "
        f"{SK3_PYTHON} {remote_driver} "
        f"{shlex.quote(SK3_PARQUET)} "
        f"{shlex.quote(SK3_CHUNK_GLOB)} "
        f"{shlex.quote(remote_oral_ids)} "
        f"{shlex.quote(remote_best_titles)} "
        f"{shlex.quote(remote_out_oral)} "
        f"{shlex.quote(remote_out_best)} "
        f"{shlex.quote(remote_out_map)}"
    )
    print("[local] launching remote join...", file=sys.stderr, flush=True)
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, SK3_HOST, cmd_str],
        capture_output=True, text=True
    )
    print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        print("REMOTE STDOUT:\n", proc.stdout, file=sys.stderr)
        raise RuntimeError(f"remote join failed rc={proc.returncode}")
    # parse last line of stdout (the JSON summary)
    summary = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith('{"_summary"'):
            summary = json.loads(line)["_summary"]
    if summary is None:
        print("REMOTE STDOUT:\n", proc.stdout, file=sys.stderr)
        raise RuntimeError("no summary returned from remote")

    # 4. pull outputs back
    subprocess.run(
        ["scp", "-q", *SSH_OPTS,
         f"{SK3_HOST}:{remote_out_oral}", str(OUT_ORAL)],
        check=True,
    )
    subprocess.run(
        ["scp", "-q", *SSH_OPTS,
         f"{SK3_HOST}:{remote_out_best}", str(OUT_BEST)],
        check=True,
    )
    return summary


# ----------------------------------------------------------------------------
# Post-process: samples + manifest
# ----------------------------------------------------------------------------
def sample_records(path: Path, n: int = 5, ok_only: bool = True, seed: int = 0) -> list[dict]:
    """Read all OK records and sample n with a fixed seed (reservoir for memory)."""
    import random
    rng = random.Random(seed)
    rs = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if ok_only and not rec.get("ok"):
                continue
            # reservoir sample
            if len(rs) < n:
                rs.append(rec)
            else:
                k = rng.randint(0, i)
                if k < n:
                    rs[k] = rec
    return rs


def count_ok(path: Path) -> tuple[int, int]:
    n = 0; ok = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            n += 1
            try:
                rec = json.loads(line)
                if rec.get("ok"): ok += 1
            except Exception:
                pass
    return n, ok


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-remote", action="store_true",
                    help="Skip running the remote join; reuse existing output files.")
    args = ap.parse_args()

    # 1. labels
    print("[local] loading oral_spotlight labels...", file=sys.stderr)
    oral_pids, oral_breakdown, _ = load_oral_spotlight_paper_ids()
    print(f"  -> {len(oral_pids)} unique oral|spotlight paper_ids", file=sys.stderr)

    print("[local] loading best_papers labels...", file=sys.stderr)
    best_records, best_breakdown = load_best_papers_titles()
    best_titles = [r["title_norm"] for r in best_records]
    print(f"  -> {len(best_records)} best papers (label=1), {len(set(best_titles))} unique normalized titles", file=sys.stderr)

    # 2. cross-link on sk3
    if not args.skip_remote:
        summary = run_remote(oral_pids, best_titles)
    else:
        summary = {"_skip_remote": True}

    # 3. local sanity counts
    print("[local] counting output files...", file=sys.stderr)
    oral_n, oral_ok = count_ok(OUT_ORAL) if OUT_ORAL.exists() else (0, 0)
    best_n, best_ok = count_ok(OUT_BEST) if OUT_BEST.exists() else (0, 0)

    # 4. samples
    print("[local] sampling records...", file=sys.stderr)
    oral_samples = sample_records(OUT_ORAL, n=5, ok_only=True) if OUT_ORAL.exists() else []
    best_samples = sample_records(OUT_BEST, n=5, ok_only=True) if OUT_BEST.exists() else []
    with open(OUT_SAMPLES, "w") as f:
        json.dump({"oral_spotlight": oral_samples, "best_papers": best_samples}, f, indent=2)

    manifest = {
        "label_breakdown": {
            "oral_spotlight": oral_breakdown,
            "best_papers": best_breakdown,
        },
        "remote_join_summary": summary,
        "output_files": {
            "oral_spotlight": str(OUT_ORAL),
            "best_papers": str(OUT_BEST),
            "samples": str(OUT_SAMPLES),
        },
        "output_counts": {
            "oral_spotlight_total_lines": oral_n,
            "oral_spotlight_ok_lines": oral_ok,
            "best_papers_total_lines": best_n,
            "best_papers_ok_lines": best_ok,
        },
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[local] wrote manifest: {OUT_MANIFEST}", file=sys.stderr)
    print(json.dumps(manifest["output_counts"], indent=2))
    print(json.dumps(manifest["remote_join_summary"], indent=2))


if __name__ == "__main__":
    main()
