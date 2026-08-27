#!/usr/bin/env python3
"""Chunk PatentsView detailed-description specs into paragraphs for indexing.

Reads:
  raw/patentsview_grant/g_detail_desc_text_*.tsv.zip
  raw/patentsview_pg/pg_detail_desc_text_*.tsv.zip
  (also brf_sum if present — smaller, prepended)

Outputs streaming parquet shards (one per year file) to:
  processed/spec_chunks/<source>_<year>.parquet
  with columns: source ('g'|'pg'), doc_id (str), chunk_idx (int), text (str)

Paragraph splitting heuristic:
  1. Split on double-newlines first
  2. If a chunk > MAX_CHARS, split on sentence boundaries (". " before capital)
  3. Skip chunks < MIN_CHARS

Designed to be streamable: handles one zip at a time, never loads everything.
"""
import csv
import glob
import io
import os
import re
import sys
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
G_DIR = f"{BASE}/raw/patentsview_grant"
PG_DIR = f"{BASE}/raw/patentsview_pg"
OUT_DIR = f"{BASE}/processed/spec_chunks"

MAX_CHARS = 1200   # ~250 tokens
MIN_CHARS = 200    # drop tiny fragments

os.makedirs(OUT_DIR, exist_ok=True)


def chunk_text(text: str) -> list:
    if not text:
        return []
    text = re.sub(r"\s+\n\s*\n\s*", "\n\n", text)  # normalize blank lines
    out = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for p in paragraphs:
        if len(p) <= MAX_CHARS:
            if len(p) >= MIN_CHARS:
                out.append(p)
            continue
        # Split long paragraph on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", p)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 <= MAX_CHARS:
                buf = (buf + " " + s).strip() if buf else s
            else:
                if len(buf) >= MIN_CHARS:
                    out.append(buf)
                buf = s
        if len(buf) >= MIN_CHARS:
            out.append(buf)
    return out


def detect_text_column(rdr_iter):
    """Peek first row, return name of the longest column = description text."""
    for r in rdr_iter:
        if not isinstance(r, dict):
            continue
        best_col = max(r, key=lambda k: len(str(r[k] or "")))
        return best_col, r
    return None, None


def detect_id_column(row):
    for cand in ("patent_id", "pgpub_id"):
        if cand in row:
            return cand
    return None


SCHEMA = pa.schema([
    pa.field("source", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("chunk_idx", pa.int32()),
    pa.field("text", pa.string()),
])


def process_zip(zip_path: str, source: str, year: str):
    out_path = f"{OUT_DIR}/{source}_{year}.parquet"
    sentinel = out_path + ".done"
    if os.path.exists(sentinel):
        return out_path, 0, 0
    # Existing parquet without sentinel = completed in old code path; mark and skip
    if os.path.exists(out_path) and not os.path.exists(sentinel):
        # Heuristic: any parquet >5MB likely came from a completed pre-fix run; trust it.
        # Otherwise drop and redo.
        if os.path.getsize(out_path) > 5_000_000:
            open(sentinel, "w").close()
            return out_path, 0, 0
        os.remove(out_path)

    print(f"  processing {zip_path} → {out_path}", file=sys.stderr, flush=True)
    n_docs = 0; n_chunks = 0
    buffer = []
    BUFFER_SIZE = 200_000
    writer_holder = {"w": None}

    def flush_buffer():
        if not buffer:
            return
        tbl = pa.Table.from_pylist(buffer, schema=SCHEMA)
        if writer_holder["w"] is None:
            writer_holder["w"] = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")
        writer_holder["w"].write_table(tbl)
        buffer.clear()

    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            tr = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            rdr = csv.DictReader(tr, delimiter="\t")
            first = next(rdr, None)
            if first is None:
                return out_path, 0, 0
            id_col = detect_id_column(first)
            if id_col is None:
                print(f"    no patent_id/pgpub_id column in {zip_path}", file=sys.stderr)
                return None, 0, 0
            text_col = max(first, key=lambda k: len(str(first[k] or "")))
            print(f"    using id_col={id_col} text_col={text_col}", file=sys.stderr)

            def emit(row):
                nonlocal n_docs, n_chunks
                doc_id = str(row.get(id_col, "")).strip()
                text = str(row.get(text_col, "") or "")
                if not doc_id or not text: return
                n_docs += 1
                for i, chunk in enumerate(chunk_text(text)):
                    buffer.append({"source": source, "doc_id": doc_id,
                                   "chunk_idx": i, "text": chunk})
                    n_chunks += 1

            emit(first)
            for n, row in enumerate(rdr, 1):
                emit(row)
                if n % 50_000 == 0:
                    print(f"      {n:,} docs scanned, {n_chunks:,} chunks",
                          file=sys.stderr, flush=True)
                if len(buffer) >= BUFFER_SIZE:
                    flush_buffer()
    flush_buffer()
    if writer_holder["w"] is not None:
        writer_holder["w"].close()
    open(sentinel, "w").close()
    return out_path, n_docs, n_chunks


def main():
    files = []
    for pat, src in [
        (f"{G_DIR}/g_detail_desc_text_*.tsv.zip", "g"),
        (f"{PG_DIR}/pg_detail_desc_text_*.tsv.zip", "pg"),
    ]:
        for p in sorted(glob.glob(pat)):
            m = re.search(r"_(\d{4})\.tsv\.zip$", p)
            year = m.group(1) if m else "0"
            files.append((p, src, year))
    print(f"Found {len(files)} year zips to chunk")

    total_docs = 0; total_chunks = 0
    for zip_path, src, year in files:
        out, nd, nc = process_zip(zip_path, src, year)
        if out is not None:
            print(f"  {os.path.basename(zip_path)}: {nd:,} docs, {nc:,} chunks → {out}")
            total_docs += nd; total_chunks += nc

    print(f"\nDone. {total_docs:,} docs total, {total_chunks:,} chunks total")


if __name__ == "__main__":
    main()
