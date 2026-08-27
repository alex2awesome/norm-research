#!/usr/bin/env python3
"""
Build patents_first_draft_cpc_balanced.csv.gz with three confound fixes:

  1. Join `patentsview_pg/pg_cpc_current.tsv.zip` on `pgpub_id` → cpc_section.
     Use only `cpc_type=inventional` rows. If pgpub_id has multiple, keep
     the lowest cpc_sequence (primary classification).
  2. Filter `date_published <= 2021-12-31` to avoid prosecution-survivorship
     bias on recent filings (2022-24 had 32-44% pos rate vs 50% baseline).
  3. Balance per (cpc_section × length_bucket × label) cell → 2·min(pos,neg).

Input:  patents_dataset.jsonl.gz (4.69M rows, full metadata + text)
        patentsview_pg/pg_cpc_current.tsv.zip (CPC classifications)
Output: patents_first_draft_cpc_balanced.csv.gz with columns:
        text, judgement, cpc_section, year, length_bucket
"""
import csv
import gzip
import io
import json
import random
import re
import sys
import zipfile
from collections import defaultdict

csv.field_size_limit(2**31 - 1)

# USPTO pre-grant publication format for canceled claims:
#   "1 . (canceled)"          (single claim)
#   "1 - 12 . (canceled)"     (range of claims)
#   "1-18. (Canceled)"        (no spaces, capital C)
#   "1. (Cancelled)"          (British spelling)
# Require a claim number (or range) immediately before the parenthesized
# "canceled" — this avoids matching random uses of the word in claim bodies.
PRELIM_AMEND_RE = re.compile(
    r"(?:^|\n)\s*\d+\s*[.,]?\s*"               # claim number, optional punctuation
    r"(?:[-–—]\s*\d+\s*[.,]?\s*)?"             # optional range
    r"\(\s*[Cc]ancel(?:l)?ed\s*\)",            # (canceled) / (Canceled) / (Cancelled)
)


def has_prelim_amend(text):
    return bool(PRELIM_AMEND_RE.search(text or ""))

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"
CPC_ZIP = f"{BASE}/raw/patentsview_pg/pg_cpc_current.tsv.zip"
OUTPUT = f"{BASE}/patents_first_draft_cpc_balanced.csv.gz"

YEAR_MAX = 2021
SEED = 42
LENGTH_BUCKET_EDGES = [0, 3000, 6000, 10000, 16000, 999999]
# Cap per CPC section. Without this, G+H dominate at ~62% of dataset.
# Capped at 100K, G and H drop to ~100K each, total ~545K.
PER_SECTION_CAP = 100_000


def length_bucket(n):
    for i in range(len(LENGTH_BUCKET_EDGES) - 1):
        if LENGTH_BUCKET_EDGES[i] <= n < LENGTH_BUCKET_EDGES[i + 1]:
            return i
    return len(LENGTH_BUCKET_EDGES) - 2


def load_cpc_section():
    """Returns dict pgpub_id (str) -> cpc_section (single letter)."""
    print(f"Loading CPC sections from {CPC_ZIP} ...", file=sys.stderr)
    out = {}
    seq = {}  # pgpub_id -> kept cpc_sequence (lowest wins)
    with zipfile.ZipFile(CPC_ZIP) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            tr = io.TextIOWrapper(fh, encoding="utf-8")
            rdr = csv.DictReader(tr, delimiter="\t")
            for n, r in enumerate(rdr, 1):
                if r.get("cpc_type") != "inventional":
                    continue
                pid = r["pgpub_id"]
                section = r.get("cpc_section", "").strip()
                if not section or len(section) != 1:
                    continue
                try:
                    s = int(r["cpc_sequence"])
                except (KeyError, ValueError):
                    s = 999
                if pid not in seq or s < seq[pid]:
                    seq[pid] = s
                    out[pid] = section
                if n % 2_000_000 == 0:
                    print(f"  CPC rows scanned: {n:,}, mapped {len(out):,}", file=sys.stderr)
    print(f"  Total mapped pgpub_ids: {len(out):,}", file=sys.stderr)
    return out


def first_pass_metadata(cpc_map):
    """Stream JSONL, capture per-row metadata. Returns list of dicts:
       pgpub_id, year, label, length_bucket, cpc_section.
    """
    print(f"First pass over {JSONL} (metadata only) ...", file=sys.stderr)
    keep = []  # eligible rows
    n_total = 0
    n_missing_cpc = 0
    n_year_drop = 0
    n_missing_text = 0
    n_missing_label = 0
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            n_total += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            pid = str(d.get("pgpub_id", "")).strip()
            if not pid:
                continue
            section = cpc_map.get(pid)
            if not section:
                n_missing_cpc += 1
                continue
            date_pub = (d.get("date_published") or "")[:4]
            if not date_pub.isdigit():
                continue
            year = int(date_pub)
            if year > YEAR_MAX:
                n_year_drop += 1
                continue
            abstract = d.get("pg_abstract") or ""
            claims = d.get("pg_claims") or ""
            if not claims.strip():
                n_missing_text += 1
                continue
            label = d.get("first_draft_approved")
            if label is None:
                n_missing_label += 1
                continue
            label = int(bool(label))
            text_len = len(abstract) + len(claims) + 30  # 30 = headers
            bucket = length_bucket(text_len)
            keep.append({
                "pgpub_id": pid,
                "year": year,
                "label": label,
                "length_bucket": bucket,
                "cpc_section": section,
                "has_prelim_amend": int(has_prelim_amend(claims)),
            })
            if n_total % 500_000 == 0:
                print(f"  JSONL rows scanned: {n_total:,}  kept: {len(keep):,}", file=sys.stderr)
    print(f"  Total scanned: {n_total:,}", file=sys.stderr)
    print(f"  Dropped (no CPC):       {n_missing_cpc:,}", file=sys.stderr)
    print(f"  Dropped (year>{YEAR_MAX}):     {n_year_drop:,}", file=sys.stderr)
    print(f"  Dropped (no claims):    {n_missing_text:,}", file=sys.stderr)
    print(f"  Dropped (no label):     {n_missing_label:,}", file=sys.stderr)
    print(f"  Eligible rows: {len(keep):,}", file=sys.stderr)

    # Quick stats on preliminary-amendment flag
    n_flag = sum(r["has_prelim_amend"] for r in keep)
    pos_flag = sum(r["has_prelim_amend"] * r["label"] for r in keep)
    n_pos = sum(r["label"] for r in keep)
    print(f"  has_prelim_amend=1: {n_flag:,} ({n_flag/len(keep)*100:.1f}%)", file=sys.stderr)
    if n_flag > 0:
        print(f"    pos rate within flag=1: {pos_flag/n_flag*100:.1f}%", file=sys.stderr)
    if (len(keep) - n_flag) > 0:
        pos_no_flag = n_pos - pos_flag
        print(f"    pos rate within flag=0: {pos_no_flag/(len(keep)-n_flag)*100:.1f}%", file=sys.stderr)
    return keep


def balance(rows):
    """Per (cpc_section, length_bucket, label) -> 2*min(pos,neg), then cap per section."""
    print("Balancing per (cpc_section, length_bucket, label) ...", file=sys.stderr)
    rng = random.Random(SEED)
    cells = defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        cells[(r["cpc_section"], r["length_bucket"])][r["label"]].append(r)

    # Pass 1: per-cell balance to 2*min(pos,neg).
    by_section_cells = defaultdict(list)  # section -> list of (bucket, pos_balanced, neg_balanced)
    cell_report = []
    for key in sorted(cells):
        section, bucket = key
        pos = cells[key][1]
        neg = cells[key][0]
        n = min(len(pos), len(neg))
        cell_report.append((section, bucket, len(pos), len(neg), n * 2))
        if n == 0:
            continue
        rng.shuffle(pos)
        rng.shuffle(neg)
        by_section_cells[section].append((bucket, pos[:n], neg[:n]))

    # Pass 2: cap per section. If section total > PER_SECTION_CAP, downsample
    # each cell proportionally to preserve per-cell 50/50 ratio and the
    # relative length-bucket mix within the section.
    chosen = []
    section_report = []
    for section in sorted(by_section_cells):
        cells_in_sec = by_section_cells[section]
        section_total = sum(2 * len(p) for _, p, _ in cells_in_sec)
        if section_total <= PER_SECTION_CAP:
            scale = 1.0
        else:
            scale = PER_SECTION_CAP / section_total
        kept = 0
        for bucket, pos_list, neg_list in cells_in_sec:
            n_keep_per_class = max(0, int(round(len(pos_list) * scale)))
            chosen.extend(pos_list[:n_keep_per_class])
            chosen.extend(neg_list[:n_keep_per_class])
            kept += 2 * n_keep_per_class
        section_report.append((section, section_total, kept, scale))
    rng.shuffle(chosen)

    print(f"\n  Per-cell counts (before per-section cap):", file=sys.stderr)
    print(f"  {'sect':4s} {'bkt':3s} | {'pos':>8s} {'neg':>8s} -> {'balanced':>10s}", file=sys.stderr)
    for s, b, p, n, t in cell_report:
        print(f"  {s:4s} {b:>3d} | {p:>8d} {n:>8d} -> {t:>10d}", file=sys.stderr)
    print(f"\n  Per-section after cap ({PER_SECTION_CAP:,}):", file=sys.stderr)
    print(f"  {'sect':4s} | {'pre_cap':>10s} {'post_cap':>10s} {'scale':>7s}", file=sys.stderr)
    for s, pre, post, sc in section_report:
        print(f"  {s:4s} | {pre:>10d} {post:>10d} {sc:>6.3f}", file=sys.stderr)
    print(f"\n  Total balanced rows: {len(chosen):,}", file=sys.stderr)
    return set(r["pgpub_id"] for r in chosen)


def second_pass_write(keep_pids, kept_meta):
    """Stream JSONL again, write rows whose pgpub_id is in keep_pids."""
    print(f"\nSecond pass: writing {OUTPUT} ...", file=sys.stderr)
    meta_by_pid = {m["pgpub_id"]: m for m in kept_meta}
    n_written = 0
    with gzip.open(OUTPUT, "wt", newline="") as out:
        w = csv.DictWriter(
            out, fieldnames=["text", "judgement", "cpc_section", "year",
                             "length_bucket", "has_prelim_amend"]
        )
        w.writeheader()
        with gzip.open(JSONL, "rt") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                pid = str(d.get("pgpub_id", "")).strip()
                if pid not in keep_pids:
                    continue
                m = meta_by_pid.get(pid)
                if m is None:
                    continue
                abstract = d.get("pg_abstract") or ""
                claims = d.get("pg_claims") or ""
                text = f"ABSTRACT:\n{abstract}\n\nCLAIMS:\n{claims}"
                w.writerow({
                    "text": text,
                    "judgement": m["label"],
                    "cpc_section": m["cpc_section"],
                    "year": m["year"],
                    "length_bucket": m["length_bucket"],
                    "has_prelim_amend": m["has_prelim_amend"],
                })
                n_written += 1
                if n_written % 50_000 == 0:
                    print(f"  written: {n_written:,}", file=sys.stderr)
    print(f"  Total written: {n_written:,}", file=sys.stderr)


def main():
    cpc_map = load_cpc_section()
    meta = first_pass_metadata(cpc_map)
    # Free the CPC map now that meta carries cpc_section
    del cpc_map
    keep_pids = balance(meta)
    # Filter meta to chosen rows so second pass can write metadata.
    kept_meta = [m for m in meta if m["pgpub_id"] in keep_pids]
    del meta
    second_pass_write(keep_pids, kept_meta)


if __name__ == "__main__":
    main()
