#!/usr/bin/env python3
"""Convert OACT bulk weekly zips -> office_actions_v3-compatible part files.

OACT (USPTO ODP "Office Actions Weekly Archives") turned out to be backfilled
to 2007 with full oaText — covering the OARD-era cohort the per-app API
crawler was fetching at 1.1 app/s. This converter replaces that crawl:

  datasets/patents/raw/oact/*.zip  (981 weekly JSONL zips)
    -> filter to cohort app_ids + CTNF/CTFR
    -> emit {app_id, ifw_number, document_code, document_date, text,
             rejections} records
    -> datasets/patents/processed/office_actions_v3/office_actions_part_bulk_p{N}.jsonl
       (PRIORITY tier: app_level_v sample + phase2 dataset_v0 apps)
       office_actions_part_bulk_r{N}.jsonl (rest of the 1.1M cohort)

The extraction cron's snapshot glob (office_actions_part_*.jsonl) picks these
up automatically; extract_oa_102_vllm.py dedupes on (app_id, ifw_number).
`rejections` (structured 102/103 types from OACT) is carried through for the
future §103 round.
"""
import glob
import gzip
import json
import os
import zipfile

BASE = os.path.expanduser("~/norm-research/datasets/patents")
OACT = f"{BASE}/raw/oact"
V3 = f"{BASE}/processed/office_actions_v3"
TODO = f"{BASE}/processed/oa_102_app_ids_todo.txt"
PH2 = f"{BASE}/processed/phase2_dataset_v0/dataset_v0.jsonl.gz"
ALV = f"{BASE}/processed/phase2_dataset_v0/app_level_v/sample.jsonl"
ROTATE = 20000  # records per part file

cohort = {l.strip() for l in open(TODO) if l.strip()}
prio = set()
for line in open(ALV):
    d = json.loads(line)
    if d.get("app_id"):
        prio.add(str(d["app_id"]))
with gzip.open(PH2, "rt") as f:
    for line in f:
        d = json.loads(line)
        if d.get("app_id"):
            prio.add(str(d["app_id"]))
prio &= cohort
print(f"cohort {len(cohort):,}  priority {len(prio):,}", flush=True)

seen = set()  # (app_id, ifw) within this conversion
counts = {"p": 0, "r": 0}
fhs = {}


REST_DIR = f"{BASE}/processed/office_actions_bulk_rest"  # staged, NOT live:
os.makedirs(REST_DIR, exist_ok=True)  # move into V3 when ready to extract


def writer(tier):
    idx = counts[tier] // ROTATE + 1
    root = V3 if tier == "p" else REST_DIR
    path = f"{root}/office_actions_part_bulk_{tier}{idx:03d}.jsonl"
    if fhs.get(tier) and fhs[tier].name != path:
        fhs[tier].close()
        fhs[tier] = None
    if not fhs.get(tier):
        fhs[tier] = open(path, "a")
    return fhs[tier]


zips = sorted(glob.glob(f"{OACT}/*.zip"))
print(f"zips: {len(zips)}", flush=True)
n_recs = 0
for zi, zp in enumerate(zips, 1):
    try:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        app = str(d.get("patentApplicationNumber") or "")
                        if app not in cohort:
                            continue
                        if d.get("documentCode") not in ("CTNF", "CTFR"):
                            continue
                        ifw = str(d.get("obsoleteDocumentIdentifier") or "")
                        if not d.get("oaText") or (app, ifw) in seen:
                            continue
                        seen.add((app, ifw))
                        tier = "p" if app in prio else "r"
                        rec = {"app_id": app, "ifw_number": ifw,
                               "document_code": d["documentCode"],
                               "document_date":
                                   (d.get("submissionDate") or "")[:10],
                               "page_count": None,
                               "text": d["oaText"],
                               "rejections": d.get("rejections"),
                               "source": "oact_bulk"}
                        writer(tier).write(json.dumps(rec) + "\n")
                        counts[tier] += 1
                        n_recs += 1
    except Exception as e:
        print(f"  zip error {zp}: {e}", flush=True)
    if zi % 50 == 0:
        print(f"  {zi}/{len(zips)} zips  priority={counts['p']:,} "
              f"rest={counts['r']:,}", flush=True)
for fh in fhs.values():
    if fh:
        fh.close()
print(f"BULK-PARTS-DONE priority={counts['p']:,} rest={counts['r']:,} "
      f"total={n_recs:,}", flush=True)
