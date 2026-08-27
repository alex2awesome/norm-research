#!/usr/bin/env python3
"""
Build patents_final_outcome_cpc_balanced_with_rejections.csv.gz.

Same pipeline as build_first_draft_with_rejections.py but:
  - Label: final_outcome (granted=1, abandoned=0)
  - Filter: ≥1 examiner citation (cite-bearing pool only)
  - Input text: pg_claims (original draft) — symmetric across grant/abandon

Drops the zero-cite shortcut. Result is a clean §102/§103 verifiability
dataset: of applications substantive enough to receive examiner art search,
did the claims survive (granted) or did the applicant give up (abandoned).

SCOPE CORRECTIONS (Codex audit 2026-07-10):
- "cite-bearing" admits ANY oard_citations row — action_type is NOT checked, and ~79% of
  citation rows have empty action_type (can be IDS/892/1449, not examiner-rejection-linked).
  The cohort is "has any OARD citation record", weaker than the docstring's original claim.
- The exact CPC×length-bucket outcome balancing DELETES mechanically-predictable V by
  construction. Downstream recovery numbers are RESIDUAL-V after forcing CPC and coarse
  length non-predictive — not natural-population V, and PPV/threshold stats are
  unrecoverable from the artificial 50/50 base rate.

DUAL-AUDIT FINDINGS (Codex gpt-5.6-sol + Fable, 2026-07-13 — measured on the frozen artifact;
behavior here unchanged, consumers must handle these downstream):
- YEAR IS NOT IN THE BALANCE CELLS: grant rate drifts .461 (2011) -> .592 (2017) inside the
  "balanced" file; publication year alone = AUC .531. Any model evaluated with a random split
  earns part of its AUC from era. Use out-of-time or year-stratified splits.
- DUPLICATE TEXTS SURVIVE: writer keys by pgpub_id but emits one row per matching JSONL line
  (app-level multiplicity upstream, no A1/A2 kind-code filter) — 579,084 rows / 577,430 unique
  ids; 3,427 exact-duplicate texts (533 cross-label). Dedup before any split. Patent FAMILIES
  (continuations, near-identical specs) are not grouped anywhere.
- LABEL CONSTRUCT: ~3.3% of cohort negatives received a Notice of Allowance before abandoning
  (examiner said yes; applicant walked) — "abandoned" is not "examiner rejected". Abandonment
  in favor of a continuation is also labeled 0 with no family credit. See
  scripts/patents_event_panel.py (noa_before_abandon) for the per-app flag.
- X IS PUBLISHED TEXT: pg_claims can reflect preliminary amendments ("(canceled)" markers are
  detected -> has_prelim_amend flag, but the markers REMAIN in text) — not strictly at-filing.
- YEAR_MAX applies to PUBLICATION year, not filing year or a common disposition horizon.
- The .70-.74 "dense ceiling" numbers quoted alongside this dataset were trained on the NAIVE
  globally-balanced file (07_balance), not this one — cross-cohort comparisons are invalid
  (see scripts/assemble_vat_table_patents.py header).
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
OARD_REJ = f"{BASE}/raw/oard/oard_rejections_by_app.csv"
OARD_CITES = f"{BASE}/raw/oard/oard_citations.csv"
OUTPUT = f"{BASE}/patents_final_outcome_cpc_balanced_with_rejections.csv.gz"

YEAR_MAX = 2021
SEED = 42
LENGTH_BUCKET_EDGES = [0, 3000, 6000, 10000, 16000, 999999]
PER_SECTION_CAP = 100_000

PRELIM_AMEND_RE = re.compile(
    r"(?:^|\n)\s*\d+\s*[.,]?\s*"
    r"(?:[-–—]\s*\d+\s*[.,]?\s*)?"
    r"\(\s*[Cc]ancel(?:l)?ed\s*\)",
)

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
    if not s: return ""
    s = s.strip()
    return re.sub(r"^US|[A-Z]\d$", "", s)


def load_cpc_section():
    print(f"Loading CPC sections ...", file=sys.stderr)
    out, seq = {}, {}
    with zipfile.ZipFile(CPC_ZIP) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            rdr = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"), delimiter="\t")
            for r in rdr:
                if r.get("cpc_type") != "inventional":
                    continue
                pid = r["pgpub_id"]
                section = r.get("cpc_section", "").strip()
                if not section or len(section) != 1:
                    continue
                try: s = int(r["cpc_sequence"])
                except (KeyError, ValueError): s = 999
                if pid not in seq or s < seq[pid]:
                    seq[pid] = s
                    out[pid] = section
    print(f"  {len(out):,} pgpub_ids", file=sys.stderr)
    return out


def load_pgpub_to_app():
    print(f"Loading PatEx pgpub→app ...", file=sys.stderr)
    out = {}
    with open(PATEX_CSV) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            pgpub = normalize_pgpub(r.get("earliest_pgpub_number", ""))
            app = r.get("application_number", "").strip()
            if pgpub and app:
                out[pgpub] = app
    print(f"  {len(out):,} pgpub→app mappings", file=sys.stderr)
    return out


def load_oard_flags():
    print(f"Loading OARD rejection flags ...", file=sys.stderr)
    out = {}
    with open(OARD_REJ) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            out[r["app_id"].strip()] = {c: int(r[c]) for c in REJ_COLS}
    print(f"  {len(out):,} OARD per-app rows", file=sys.stderr)
    return out


def load_cite_bearing_apps():
    """Set of app_ids that have ≥1 examiner citation in OARD."""
    print(f"Loading cite-bearing apps from OARD citations ...", file=sys.stderr)
    apps = set()
    with open(OARD_CITES) as f:
        rdr = csv.DictReader(f)
        for n, r in enumerate(rdr, 1):
            app = r.get("app_id", "").strip()
            if app:
                apps.add(app)
            if n % 10_000_000 == 0:
                print(f"  scanned {n:,}", file=sys.stderr)
    print(f"  {len(apps):,} apps with ≥1 examiner cite", file=sys.stderr)
    return apps


def first_pass(cpc_map, pgpub_to_app, oard_flags, cite_bearing_apps):
    print(f"First pass over JSONL ...", file=sys.stderr)
    keep = []
    n_total = 0
    drops = defaultdict(int)
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            n_total += 1
            try: d = json.loads(line)
            except Exception:
                drops["json"] += 1; continue
            pid = str(d.get("pgpub_id", "")).strip()
            if not pid:
                drops["no_pgpub"] += 1; continue
            section = cpc_map.get(pid)
            if not section:
                drops["no_cpc"] += 1; continue
            year = (d.get("date_published") or "")[:4]
            if not year.isdigit():
                drops["bad_date"] += 1; continue
            year = int(year)
            if year > YEAR_MAX:
                drops["year"] += 1; continue
            claims = d.get("pg_claims") or ""
            if not claims.strip():
                drops["no_claims"] += 1; continue
            final = d.get("final_outcome")
            if final not in ("granted", "abandoned"):
                drops["pending_or_unknown"] += 1; continue
            label = 1 if final == "granted" else 0

            app_id = pgpub_to_app.get(pid, "")
            if not app_id or app_id not in cite_bearing_apps:
                drops["no_cite"] += 1; continue

            abstract = d.get("pg_abstract") or ""
            text_len = len(abstract) + len(claims) + 30
            bucket = length_bucket(text_len)
            flags = oard_flags.get(app_id, {c: 0 for c in REJ_COLS})

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
                print(f"  scanned {n_total:,}, kept {len(keep):,}", file=sys.stderr)
    print(f"  Total scanned: {n_total:,}", file=sys.stderr)
    for k, v in drops.items():
        print(f"  Dropped ({k}): {v:,}", file=sys.stderr)
    n_pos = sum(r["label"] for r in keep)
    print(f"  Eligible: {len(keep):,}  pos rate (granted): {n_pos/len(keep)*100:.1f}%", file=sys.stderr)
    return keep


def balance(rows):
    print("Balancing per (cpc, length, label) + per-section cap ...", file=sys.stderr)
    rng = random.Random(SEED)
    cells = defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        cells[(r["cpc_section"], r["length_bucket"])][r["label"]].append(r)

    by_section = defaultdict(list)
    for key in sorted(cells):
        section, bucket = key
        pos, neg = cells[key][1], cells[key][0]
        n = min(len(pos), len(neg))
        if n == 0: continue
        rng.shuffle(pos); rng.shuffle(neg)
        by_section[section].append((bucket, pos[:n], neg[:n]))

    chosen = []
    report = []
    for section in sorted(by_section):
        total = sum(2 * len(p) for _, p, _ in by_section[section])
        scale = min(1.0, PER_SECTION_CAP / total) if total > PER_SECTION_CAP else 1.0
        kept = 0
        for bucket, pos_list, neg_list in by_section[section]:
            n_keep = max(0, int(round(len(pos_list) * scale)))
            chosen.extend(pos_list[:n_keep])
            chosen.extend(neg_list[:n_keep])
            kept += 2 * n_keep
        report.append((section, total, kept, scale))
    rng.shuffle(chosen)
    print(f"\n  Per-section after cap ({PER_SECTION_CAP:,}):", file=sys.stderr)
    print(f"  {'sec':4s} | {'pre':>10s} {'post':>10s} {'scale':>6s}", file=sys.stderr)
    for s, pre, post, sc in report:
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
                try: d = json.loads(line)
                except Exception: continue
                pid = str(d.get("pgpub_id", "")).strip()
                m = meta_by_pid.get(pid)
                if m is None: continue
                abstract = d.get("pg_abstract") or ""
                claims = d.get("pg_claims") or ""
                text = f"ABSTRACT:\n{abstract}\n\nCLAIMS:\n{claims}"
                row = {
                    "text": text, "judgement": m["label"],
                    "cpc_section": m["cpc_section"], "year": m["year"],
                    "length_bucket": m["length_bucket"],
                    "has_prelim_amend": m["has_prelim_amend"],
                    "pgpub_id": m["pgpub_id"], "app_id": m["app_id"],
                }
                for c in REJ_COLS: row[c] = m[c]
                w.writerow(row)
                n += 1
                if n % 50_000 == 0:
                    print(f"  written {n:,}", file=sys.stderr)
    print(f"  Total written: {n:,}", file=sys.stderr)


def main():
    cpc = load_cpc_section()
    pgpub_to_app = load_pgpub_to_app()
    oard = load_oard_flags()
    cite_apps = load_cite_bearing_apps()
    rows = first_pass(cpc, pgpub_to_app, oard, cite_apps)
    del cpc, pgpub_to_app, oard, cite_apps
    chosen = balance(rows)
    del rows
    write_output(chosen)


if __name__ == "__main__":
    main()
