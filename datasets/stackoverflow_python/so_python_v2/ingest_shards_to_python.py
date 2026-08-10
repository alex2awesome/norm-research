#!/usr/bin/env python3
"""Phase 1 ingest: stream all 58 shards -> two parquets
(so_python_questions.parquet, so_python_answers.parquet) containing every
python-tagged question and every answer to one.

Strategy (memory-safe; each shard <= ~600MB compressed, ~3-4GB inflated):
  Pass 1: scan all shards, collect python question Ids (a set).
  Pass 2: re-scan all shards, write
            - questions: rows where PostTypeId==1 AND Id in py_qids
            - answers:   rows where PostTypeId==2 AND ParentId in py_qids
          using pyarrow ParquetWriter (per-shard row groups).

Checkpoints:
  - manifests/ingest_pass1_done.json after Pass 1 (py_qids count, per-shard
    counts). Pass 2 resumes by checking the per-shard sentinel file
    manifests/pass2/shard_{i:02d}.done — if present, skip.
  - Per-shard progress line on stdout AND appended to logs/ingest_progress.log.

KEEP Body raw (HTML). The CR.SE pilot was burned by pre-stripped HTML; we strip
later at pool-build time only.

Columns retained in OUTPUT:
  questions: Id, Score, AcceptedAnswerId, CreationDate, Body, Title,
             OwnerUserId, ViewCount, Tags
  answers  : Id, ParentId, Score, CreationDate, Body, OwnerUserId
  (accepted flag + position are derived later from questions.AcceptedAnswerId
  and CreationDate, so we don't pre-compute them here.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PY_QUESTION_COLS = [
    "Id", "Score", "AcceptedAnswerId", "CreationDate", "Body", "Title",
    "OwnerUserId", "ViewCount", "Tags",
]
PY_ANSWER_COLS = [
    "Id", "ParentId", "Score", "CreationDate", "Body", "OwnerUserId",
]


def is_py_strict(tags) -> bool:
    if tags is None:
        return False
    try:
        for t in tags:
            if isinstance(t, str) and (t == "python" or t.startswith("python-")):
                return True
    except TypeError:
        return False
    return False


def collect_py_qids(shards: list[str], manifest_path: Path,
                    log_path: Path) -> set[int]:
    """Pass 1: union of python question Ids across all shards."""
    py_qids: set[int] = set()
    per_shard: list[dict] = []
    for i, s in enumerate(shards):
        t0 = datetime.now()
        df = pd.read_parquet(s, columns=["Id", "PostTypeId", "Tags"])
        qs = df[df.PostTypeId == 1]
        mask = qs.Tags.apply(is_py_strict)
        ids = qs.loc[mask, "Id"].astype(int).tolist()
        n_new = len(set(ids) - py_qids)
        py_qids.update(ids)
        dur = (datetime.now() - t0).total_seconds()
        line = (f"[pass1 shard {i+1:02d}/{len(shards)}] {os.path.basename(s)} "
                f"qs={len(qs):,} py_qs={len(ids):,} new={n_new:,} "
                f"running={len(py_qids):,}  ({dur:.1f}s)")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")
        per_shard.append({
            "shard": os.path.basename(s),
            "qs": int(len(qs)),
            "py_qs": int(len(ids)),
            "secs": round(dur, 1),
        })
    manifest = {
        "build_date": datetime.now().isoformat(timespec="seconds"),
        "n_shards": len(shards),
        "n_py_qids": len(py_qids),
        "per_shard": per_shard,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n[pass1 DONE] total python Qs = {len(py_qids):,}", flush=True)
    return py_qids


def emit_pass2(shards: list[str], py_qids_path: Path, out_dir: Path,
               log_path: Path) -> None:
    """Pass 2: for each shard, write
       _pass2_chunks/shard_{i:02d}_q.parquet  (one per shard)
       _pass2_chunks/shard_{i:02d}_a.parquet
    Then concat into the two final parquets at the end. Per-shard sentinels
    make this resumable.
    """
    print(f"[pass2] loading py_qids from {py_qids_path}", flush=True)
    with open(py_qids_path) as f:
        py_qids_list = json.load(f)["py_qids"]
    py_qids = set(int(x) for x in py_qids_list)
    print(f"[pass2] loaded {len(py_qids):,} python question Ids", flush=True)

    chunk_dir = out_dir / "_pass2_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    sentinel_dir = out_dir / "manifests" / "pass2"
    sentinel_dir.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(shards):
        sentinel = sentinel_dir / f"shard_{i:02d}.done"
        if sentinel.exists():
            print(f"[pass2 shard {i+1:02d}/{len(shards)}] SKIP (sentinel exists)", flush=True)
            continue
        t0 = datetime.now()
        # Read just what we need (faster + cheaper)
        df_q = pd.read_parquet(s, columns=PY_QUESTION_COLS + ["PostTypeId"])
        # questions
        qs = df_q[(df_q.PostTypeId == 1) & df_q.Id.astype(int).isin(py_qids)]
        qs = qs[PY_QUESTION_COLS].copy()
        # answers (separate read with the answer columns)
        df_a = pd.read_parquet(s, columns=PY_ANSWER_COLS + ["PostTypeId"])
        ans = df_a[(df_a.PostTypeId == 2)
                   & df_a.ParentId.fillna(-1).astype(int).isin(py_qids)]
        ans = ans[PY_ANSWER_COLS].copy()

        q_out = chunk_dir / f"shard_{i:02d}_q.parquet"
        a_out = chunk_dir / f"shard_{i:02d}_a.parquet"
        pa_q = pa.Table.from_pandas(qs, preserve_index=False)
        pa_a = pa.Table.from_pandas(ans, preserve_index=False)
        pq.write_table(pa_q, q_out, compression="zstd")
        pq.write_table(pa_a, a_out, compression="zstd")
        sentinel.write_text(json.dumps({"qs": int(len(qs)), "as": int(len(ans)),
                                        "ts": datetime.now().isoformat(timespec="seconds")}))
        dur = (datetime.now() - t0).total_seconds()
        line = (f"[pass2 shard {i+1:02d}/{len(shards)}] {os.path.basename(s)} "
                f"qs={len(qs):,} ans={len(ans):,}  ({dur:.1f}s)")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    # Final concat: write so_python_questions.parquet + so_python_answers.parquet
    print("[pass2 concat] writing final parquets", flush=True)
    q_chunks = sorted(chunk_dir.glob("shard_*_q.parquet"))
    a_chunks = sorted(chunk_dir.glob("shard_*_a.parquet"))

    # Stream into a single ParquetWriter
    def stream_write(chunks, out_path):
        writer = None
        total = 0
        for c in chunks:
            t = pq.read_table(c)
            if writer is None:
                writer = pq.ParquetWriter(out_path, t.schema, compression="zstd")
            writer.write_table(t)
            total += t.num_rows
        if writer is not None:
            writer.close()
        return total

    n_q = stream_write(q_chunks, out_dir / "so_python_questions.parquet")
    n_a = stream_write(a_chunks, out_dir / "so_python_answers.parquet")
    print(f"[pass2 DONE] questions={n_q:,} answers={n_a:,}", flush=True)
    with open(log_path, "a") as f:
        f.write(f"[pass2 DONE] questions={n_q:,} answers={n_a:,}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shards-glob",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/_raw_parquet/stackoverflow-posts-*.parquet",
    )
    ap.add_argument(
        "--out-dir",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python",
    )
    ap.add_argument("--phase", choices=["pass1", "pass2", "both"], default="both")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.shards_glob))
    if not shards:
        raise SystemExit(f"No shards found at {args.shards_glob}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "manifests").mkdir(exist_ok=True)
    log_path = out_dir / "logs" / "ingest_progress.log"

    py_qids_path = out_dir / "manifests" / "py_qids.json"

    if args.phase in ("pass1", "both"):
        manifest_path = out_dir / "manifests" / "ingest_pass1.json"
        py_qids = collect_py_qids(shards, manifest_path, log_path)
        # Persist set as JSON list (fast to reload by Pass 2 even if it dies)
        with open(py_qids_path, "w") as f:
            json.dump({"n": len(py_qids), "py_qids": sorted(py_qids)}, f)
        print(f"[pass1] wrote py_qids set ({len(py_qids):,}) -> {py_qids_path}", flush=True)

    if args.phase in ("pass2", "both"):
        if not py_qids_path.exists():
            raise SystemExit(f"Cannot run pass2: missing {py_qids_path}; run pass1 first.")
        emit_pass2(shards, py_qids_path, out_dir, log_path)


if __name__ == "__main__":
    main()
