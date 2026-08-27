#!/usr/bin/env python3
"""Build the balanced N&C VAT sample from nc_v42_training.jsonl.gz.

Primary y (y_out): rule-change outcome, priority-collapsed across a row's labels
(MADE > CONSIDERED > NONE). Rows collapsing to MADE (y=1) or NONE (y=0) are eligible;
CONSIDERED/OTHER rows are excluded. Secondary y (y_eng): any label engagement ==
substantive_response (readout-only; not balanced on).

Per-agency balanced sampling (equal MADE/NONE), exact-text dedup, per-docket-per-class
cap to stop mass-comment dockets from dominating. Docket kept for group-aware CV;
year parsed from docket id for the time axis.

Output: nc_vat_sample.jsonl (one row per comment: doc_id, agency, docket, year, org,
split, y_out, y_eng, n_labels, text[:4000]).
"""
import gzip, json, random, re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "nc_v42_training.jsonl.gz"
OUT = BASE / "nc_vat_sample.jsonl"

# agencies with >=200 per class get n=400 (200/200); smaller balanced otherwise
BIG = ["FDA","CMS","ED","FWS","FS","NOAA","EPA","APHIS","ICEB","AMS","USCIS",
       "CDC","NHTSA","DOT","FAA","FSIS"]
SMALL = ["IRS","OSHA","DHS","BLM","USCBP","FEMA","DOD","MSHA","DOJ","USDA"]
AGENCIES = set(BIG + SMALL)
N_PER_CLASS_BIG = 200
MIN_PER_CLASS = 50
DOCKET_CAP = 10          # per (docket, class); relaxed x2 then removed if short
TEXT_TRUNC = 4000

YEAR_RE = re.compile(r"(?:^|[-_])((?:19|20)\d{2})(?:[-_]|$)")
YEAR2_RE = re.compile(r"(?:^|[-_])(\d{2})(?:[-_])")

def docket_year(docket):
    m = YEAR_RE.search(docket or "")
    if m:
        return int(m.group(1))
    m = YEAR2_RE.search(docket or "")
    if m:
        y = int(m.group(1))
        if 0 <= y <= 30:
            return 2000 + y
        if 90 <= y <= 99:
            return 1900 + y
    return None

def collapse_outcome(labels):
    ocs = {l.get("outcome_collapsed") for l in labels}
    if "MADE" in ocs:
        return "MADE"
    if "CONSIDERED" in ocs:
        return "CONSIDERED"
    if "NONE" in ocs:
        return "NONE"
    return "OTHER"

def main():
    random.seed(0)
    pools = defaultdict(list)   # (agency, y) -> rows
    seen_text = set()
    n_read = n_kept = 0
    with gzip.open(SRC, "rt", errors="ignore") as fh:
        for line in fh:
            n_read += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            ag = d.get("agency")
            if ag not in AGENCIES:
                continue
            labels = d.get("labels") or []
            if not labels:
                continue
            oc = collapse_outcome(labels)
            if oc == "MADE":
                y = 1
            elif oc == "NONE":
                y = 0
            else:
                continue
            txt = (d.get("text") or "").strip()
            if len(txt) < 200:
                continue
            key = re.sub(r"\s+", " ", txt[:2000]).lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            y_eng = int(any(l.get("engagement") == "substantive_response" for l in labels))
            pools[(ag, y)].append(dict(
                doc_id=d.get("doc_id"), agency=ag, docket=d.get("docket") or "",
                year=docket_year(d.get("docket")), org=(d.get("org") or "")[:200],
                split=d.get("split"), y_out=y, y_eng=y_eng, n_labels=len(labels),
                text=txt[:TEXT_TRUNC]))
            n_kept += 1
    print(f"read {n_read} rows, kept {n_kept} eligible (deduped)")

    out_rows = []
    summary = []
    for ag in BIG + SMALL:
        pos, neg = pools.get((ag, 1), []), pools.get((ag, 0), [])
        n_class = min(N_PER_CLASS_BIG, len(pos), len(neg))
        if n_class < MIN_PER_CLASS:
            summary.append((ag, len(pos), len(neg), 0, "SKIP(<%d/class)" % MIN_PER_CLASS))
            continue
        picked = []
        for pool, cls in ((pos, 1), (neg, 0)):
            random.shuffle(pool)
            for cap in (DOCKET_CAP, DOCKET_CAP * 2, 10**9):
                per_docket = defaultdict(int)
                sel = []
                for r in pool:
                    if per_docket[r["docket"]] >= cap:
                        continue
                    per_docket[r["docket"]] += 1
                    sel.append(r)
                    if len(sel) >= n_class:
                        break
                if len(sel) >= n_class:
                    break
            picked.extend(sel)
        out_rows.extend(picked)
        summary.append((ag, len(pos), len(neg), len(picked), "ok"))

    random.shuffle(out_rows)
    with open(OUT, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{'agency':8} {'pos_pool':>8} {'neg_pool':>8} {'sampled':>8}")
    for ag, p, n, s, note in summary:
        print(f"{ag:8} {p:>8} {n:>8} {s:>8}  {note}")
    print(f"TOTAL sampled: {len(out_rows)} -> {OUT}")
    yrs = [r["year"] for r in out_rows if r["year"]]
    print(f"year coverage: {len(yrs)}/{len(out_rows)} rows, range "
          f"{min(yrs) if yrs else '-'}-{max(yrs) if yrs else '-'}")

if __name__ == "__main__":
    main()
