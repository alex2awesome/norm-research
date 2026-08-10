#!/usr/bin/env python3
"""
Build patents_first_draft_cpc_balanced_with_rejections.csv.gz:
  Same per-(CPC × length × label) balancing as build_first_draft_cpc_balanced.py,
  but with extra columns joined in:
    - pgpub_id, app_id  (from PatEx)
    - rejected_101, rejected_102, rejected_103, rejected_112a, rejected_112b,
      rejected_112d, rejected_112f, rejected_dp  (from OARD)

Label=1 (first_draft_approved=True) rows by definition had zero office actions,
so all rejection flags = 0 for them.

Inputs:
  - patents_dataset.jsonl.gz (text + labels + pgpub_id)
  - raw/patentsview_pg/pg_cpc_current.tsv.zip (cpc_section)
  - raw/patex/application_data.csv (pgpub_id → application_number)
  - raw/oard/oard_rejections_by_app.csv (app_id → rejection flags)
"""
import csv
import gzip
import io
import json
import os
import random
import re
import sys
import zipfile
from collections import defaultdict

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"
CPC_ZIP = f"{BASE}/raw/patentsview_pg/pg_cpc_current.tsv.zip"
PATEX_CSV = f"{BASE}/raw/patex/application_data.csv"
OARD_CSV = f"{BASE}/raw/oard/oard_rejections_by_app.csv"
OUTPUT = f"{BASE}/patents_first_draft_cpc_balanced_with_rejections.csv.gz"

YEAR_MAX = 2021
SEED = 42
LENGTH_BUCKET_EDGES = [0, 3000, 6000, 10000, 16000, 999999]
PER_SECTION_CAP = 100_000

PRELIM_AMEND_RE = re.compile(
    r"(?:^|\n)\s*\d+\s*[.,]?\s*"
    r"(?:[-–—]\s*\d+\s*[.,]?\s*)?"
    r"\(\s*[Cc]ancel(?:l)?ed\s*\)",
)
PGPUB_STRIP_RE = re.compile(r"^US|[A-Z]\d?$")

REJ_COLS = [
    "rejected_101", "rejected_102", "rejected_103",
    "rejected_112a", "rejected_112b", "rejected_112d", "rejected_112f",
    "rejected_dp",
]


def length_bucket(n):
    for i in range(len(LENGTH_BUCKET_EDGES) - 1):
        if LENGTH_BUCKET_EDGES[i] <= n < LENGTH_BUCKET_EDGES[i + 1]:
            return i
    return len(LENGTH_BUCKET_EDGES) - 2


def has_prelim_amend(text):
    return bool(PRELIM_AMEND_RE.search(text or ""))


def normalize_pgpub(s):
    """'US20010023252A1' -> '20010023252'. Returns '' if not parseable."""
    if not s:
        return ""
    s = s.strip()
    # Strip US prefix and any kind/year suffix like A1, B2, P3
    return re.sub(r"^US|[A-Z]\d$", "", s)


def load_cpc_section():
    print(f"Loading CPC sections from {CPC_ZIP} ...", file=sys.stderr)
    out, seq = {}, {}
    with zipfile.ZipFile(CPC_ZIP) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            tr = io.TextIOWrapper(fh, encoding="utf-8")
            rdr = csv.DictReader(tr, delimiter="\t")
            for r in rdr:
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
    print(f"  Total mapped pgpub_ids: {len(out):,}", file=sys.stderr)
    return out


def load_pgpub_to_app():
    """Stream PatEx, return dict pgpub_id (str, normalized) -> app_id (str)."""
    print(f"Loading PatEx pgpub→app from {PATEX_CSV} ...", file=sys.stderr)
    out = {}
    n = 0
    with open(PATEX_CSV) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            n += 1
            pgpub = row.get("earliest_pgpub_number", "").strip()
            app = row.get("application_number", "").strip()
            if not pgpub or not app:
                continue
            key = normalize_pgpub(pgpub)
            if key:
                out[key] = app
            if n % 1_000_000 == 0:
                print(f"  PatEx scanned {n:,}  mapped {len(out):,}", file=sys.stderr)
    print(f"  Total PatEx rows: {n:,}, pgpub→app mappings: {len(out):,}", file=sys.stderr)
    return out


def load_oard_flags():
    """Read OARD per-app rejection flags CSV. Returns dict app_id -> dict of flags."""
    print(f"Loading OARD flags from {OARD_CSV} ...", file=sys.stderr)
    out = {}
    with open(OARD_CSV) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            app = r["app_id"].strip()
            out[app] = {c: int(r[c]) for c in REJ_COLS}
    print(f"  OARD per-app rows: {len(out):,}", file=sys.stderr)
    return out


def first_pass(cpc_map, pgpub_to_app, oard_flags):
    print(f"First pass over {JSONL} ...", file=sys.stderr)
    keep = []
    n_total = 0
    drops = defaultdict(int)
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            n_total += 1
            try:
                d = json.loads(line)
            except Exception:
                drops["json_err"] += 1
                continue
            pid = str(d.get("pgpub_id", "")).strip()
            if not pid:
                drops["no_pgpub"] += 1
                continue
            section = cpc_map.get(pid)
            if not section:
                drops["no_cpc"] += 1
                continue
            date_pub = (d.get("date_published") or "")[:4]
            if not date_pub.isdigit():
                drops["bad_date"] += 1
                continue
            year = int(date_pub)
            if year > YEAR_MAX:
                drops["year"] += 1
                continue
            claims = d.get("pg_claims") or ""
            if not claims.strip():
                drops["no_claims"] += 1
                continue
            label = d.get("first_draft_approved")
            if label is None:
                drops["no_label"] += 1
                continue
            label = int(bool(label))
            abstract = d.get("pg_abstract") or ""
            text_len = len(abstract) + len(claims) + 30
            bucket = length_bucket(text_len)

            app_id = pgpub_to_app.get(pid, "")
            flags = oard_flags.get(app_id, {c: 0 for c in REJ_COLS}) if app_id else {c: 0 for c in REJ_COLS}

            keep.append({
                "pgpub_id": pid,
                "app_id": app_id,
                "year": year,
                "label": label,
                "length_bucket": bucket,
                "cpc_section": section,
                "has_prelim_amend": int(has_prelim_amend(claims)),
                **flags,
            })
            if n_total % 500_000 == 0:
                print(f"  scanned {n_total:,}  kept {len(keep):,}", file=sys.stderr)
    print(f"  Total scanned: {n_total:,}", file=sys.stderr)
    for k, v in drops.items():
        print(f"  Dropped ({k}): {v:,}", file=sys.stderr)
    print(f"  Eligible: {len(keep):,}", file=sys.stderr)

    # Stats
    n_have_appid = sum(1 for r in keep if r["app_id"])
    print(f"  Rows with app_id mapping: {n_have_appid:,} ({n_have_appid/len(keep)*100:.1f}%)", file=sys.stderr)
    for c in REJ_COLS:
        n = sum(r[c] for r in keep)
        print(f"  {c}: {n:,} positive ({n/len(keep)*100:.2f}%)", file=sys.stderr)
    return keep


def balance(rows):
    print("Balancing per (cpc_section, length_bucket, label) + per-section cap ...", file=sys.stderr)
    rng = random.Random(SEED)
    cells = defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        cells[(r["cpc_section"], r["length_bucket"])][r["label"]].append(r)

    by_section_cells = defaultdict(list)
    for key in sorted(cells):
        section, bucket = key
        pos, neg = cells[key][1], cells[key][0]
        n = min(len(pos), len(neg))
        if n == 0: continue
        rng.shuffle(pos); rng.shuffle(neg)
        by_section_cells[section].append((bucket, pos[:n], neg[:n]))

    chosen = []
    section_report = []
    for section in sorted(by_section_cells):
        cells_in_sec = by_section_cells[section]
        section_total = sum(2 * len(p) for _, p, _ in cells_in_sec)
        scale = min(1.0, PER_SECTION_CAP / section_total) if section_total > PER_SECTION_CAP else 1.0
        kept = 0
        for bucket, pos_list, neg_list in cells_in_sec:
            n_keep = max(0, int(round(len(pos_list) * scale)))
            chosen.extend(pos_list[:n_keep])
            chosen.extend(neg_list[:n_keep])
            kept += 2 * n_keep
        section_report.append((section, section_total, kept, scale))
    rng.shuffle(chosen)

    print(f"\n  Per-section after cap ({PER_SECTION_CAP:,}):", file=sys.stderr)
    print(f"  {'sec':4s} | {'pre':>10s} {'post':>10s} {'scale':>6s}", file=sys.stderr)
    for s, pre, post, sc in section_report:
        print(f"  {s:4s} | {pre:>10d} {post:>10d} {sc:>5.3f}", file=sys.stderr)
    print(f"  Total balanced: {len(chosen):,}", file=sys.stderr)
    return chosen


def write_output(chosen):
    print(f"\nWriting {OUTPUT} ...", file=sys.stderr)
    meta_by_pid = {r["pgpub_id"]: r for r in chosen}
    n = 0
    with gzip.open(OUTPUT, "wt", newline="") as out:
        fields = (
            ["text", "judgement", "cpc_section", "year", "length_bucket",
             "has_prelim_amend", "pgpub_id", "app_id"] + REJ_COLS
        )
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()
        with gzip.open(JSONL, "rt") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                pid = str(d.get("pgpub_id", "")).strip()
                m = meta_by_pid.get(pid)
                if m is None:
                    continue
                abstract = d.get("pg_abstract") or ""
                claims = d.get("pg_claims") or ""
                text = f"ABSTRACT:\n{abstract}\n\nCLAIMS:\n{claims}"
                row = {
                    "text": text,
                    "judgement": m["label"],
                    "cpc_section": m["cpc_section"],
                    "year": m["year"],
                    "length_bucket": m["length_bucket"],
                    "has_prelim_amend": m["has_prelim_amend"],
                    "pgpub_id": m["pgpub_id"],
                    "app_id": m["app_id"],
                }
                for c in REJ_COLS:
                    row[c] = m[c]
                w.writerow(row)
                n += 1
                if n % 50_000 == 0:
                    print(f"  written {n:,}", file=sys.stderr)
    print(f"  Total written: {n:,}", file=sys.stderr)

    # Final summary of rejection-flag positives within the balanced output
    print(f"\n--- Rejection flag positives in output (label=0 only by construction) ---", file=sys.stderr)
    for c in REJ_COLS:
        cnt = sum(m[c] for m in chosen)
        print(f"  {c}: {cnt:,} ({cnt/len(chosen)*100:.2f}%)", file=sys.stderr)


def main():
    if not os.path.exists(OARD_CSV):
        print(f"ERROR: OARD CSV not found: {OARD_CSV}", file=sys.stderr)
        print("Run scripts/download_oard_rejection_flags.py first.", file=sys.stderr)
        sys.exit(1)
    cpc_map = load_cpc_section()
    pgpub_to_app = load_pgpub_to_app()
    oard_flags = load_oard_flags()
    rows = first_pass(cpc_map, pgpub_to_app, oard_flags)
    del cpc_map, pgpub_to_app, oard_flags
    chosen = balance(rows)
    del rows
    write_output(chosen)


if __name__ == "__main__":
    main()
