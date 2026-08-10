"""True-cites claim-level testbed v1 (task #51).

Purpose: development/validation sandbox for V and A implementations (Alex
2026-06-10). NOT the end prediction task — claim-level labels calibrate V
metrics (claim-element coverage); only aggregates feed the real app-level task.

Unit: one record per (app, OA). Per claim of the app: fell_102 = claim was
102-rejected over the attached art (from Llama OA extraction); negatives =
claims of the same app NOT 102-targeted in that OA (caveat: may be 103/112-
rejected — flagged for later audit via raw OA text).

Inputs:
  oa_102_extractions.jsonl.gz + oa_102_extractions_v2/*.jsonl
    {app_id, ifw_number, extraction: [{target_claim, claim_element,
     prior_art_pgpub_id, prior_art_location, ...}]}
  patents_first_draft_cpc_balanced_with_rejections.csv.gz  (claims text, labels)
  paragraph_keyed_specs.jsonl.gz + paragraph_keyed_specs_v2/*.jsonl
    {pgpub_id, paragraphs: {"0034": text, ...}}
Output: processed/truecite_testbed_v1/testbed.jsonl.gz  + stats printed
"""
import csv, glob, gzip, json, os, re, sys
csv.field_size_limit(sys.maxsize)

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
APPS_CSV = f"{BASE}/datasets/patents/patents_first_draft_cpc_balanced_with_rejections.csv.gz"
OUT_DIR = f"{PROC}/truecite_testbed_v1"
os.makedirs(OUT_DIR, exist_ok=True)

norm = lambda x: "".join(c for c in str(x) if c.isdigit()).lstrip("0")
log = lambda m: print(m, flush=True)


def loc_type(loc):
    if not loc: return "none"
    l = str(loc).lower()
    if re.search(r"\[?00\d{2,}\]?|para", l): return "paragraph"
    if re.search(r"col|line", l): return "col_line"
    if re.search(r"fig", l): return "figure"
    if re.search(r"abstract|claim", l): return "abstract/claim"
    return "other"


def parse_anchors(loc):
    """'[0034]-[0036]', 'para. 34', 'paragraphs 0041, 0044' -> ['0034', ...]"""
    out = []
    for m in re.findall(r"\[(\d{3,4})\]|para(?:graph)?s?\.?\s*\[?(\d{1,4})", str(loc).lower()):
        n = m[0] or m[1]
        out.append(n.zfill(4))
    return list(dict.fromkeys(out))


# ---------- 1. extraction records grouped by (app, ifw) ----------
log("Loading extraction records ...")
oa = {}  # (app, ifw) -> list of element dicts
def eat(path, op):
    with op(path) as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            els = [e for e in (r.get("extraction") or [])
                   if isinstance(e.get("target_claim"), int) and e.get("prior_art_pgpub_id")]
            if els:
                oa.setdefault((r["app_id"], r.get("ifw_number")), []).extend(els)
eat(f"{PROC}/oa_102_extractions.jsonl.gz", lambda p: gzip.open(p, "rt"))
for fn in sorted(os.listdir(f"{PROC}/oa_102_extractions_v2")):
    eat(f"{PROC}/oa_102_extractions_v2/{fn}", open)
apps_needed = {a for a, _ in oa}
log(f"  {len(oa):,} OAs with valid elements, {len(apps_needed):,} apps")

# ---------- 2. app claims via PatEx -> patents_dataset chain ----------
# The balanced csv covers only ~1% of extraction apps (different universe).
# Chain: app_id -> PatEx application_data (earliest_pgpub_number, 100%/96%
# coverage) -> patents_dataset.jsonl.gz (pg_claims full text + both labels).
# Cached to app_record_cache.jsonl; reruns only scan for missing apps.
PATEX_CSV = f"{BASE}/datasets/patents/raw/patex/application_data.csv"
PT_DS = f"{BASE}/datasets/patents/patents_dataset.jsonl.gz"
CACHE = f"{OUT_DIR}/app_record_cache.jsonl"

cache = {}
if os.path.exists(CACHE):
    with open(CACHE) as f:
        for line in f:
            try:
                r = json.loads(line)
                cache[r["app_id"]] = r
            except Exception:
                continue
missing = apps_needed - set(cache)
log(f"  app cache: {len(cache):,} cached, {len(missing):,} to resolve")
if missing:
    log("  scanning PatEx for app -> pgpub ...")
    app2pg = {}
    with open(PATEX_CSV) as f:
        for row in csv.DictReader(f):
            a = row["application_number"].replace("/", "").strip()
            if a in missing:
                # 'US20010023252A1' -> '20010023252' (year4+serial7; drop kind code)
                m = re.search(r"(20\d{2}\d{7})", row["earliest_pgpub_number"])
                if m:
                    app2pg[a] = m.group(1)
    pg2app = {}
    for a, pg in app2pg.items():
        pg2app.setdefault(pg, []).append(a)
    log(f"  {len(app2pg):,} apps have pgpub; scanning patents_dataset ...")
    found = 0
    with gzip.open(PT_DS, "rt") as f, open(CACHE, "a", buffering=1) as out:
        for line in f:
            r = json.loads(line)
            pg = norm(r.get("pgpub_id"))
            for a in pg2app.get(pg, []):
                rec = {"app_id": a, "pgpub_id": pg,
                       "pg_claims": r.get("pg_claims"),
                       "first_draft_approved": r.get("first_draft_approved"),
                       "final_outcome": r.get("final_outcome"),
                       "n_office_actions": r.get("n_office_actions")}
                cache[a] = rec
                out.write(json.dumps(rec) + "\n")
                found += 1
    log(f"  resolved {found:,} new app records")

log("Parsing app claims ...")
CLAIM_SPLIT = re.compile(r"\n(?=\d{1,3}\s*\.\s)")
app_info = {}
n_no_claims = 0
for a, rec in cache.items():
    t = rec.get("pg_claims") or ""
    claims = {}
    for chunk in CLAIM_SPLIT.split("\n" + t.strip()):
        m = re.match(r"\s*(\d{1,3})\s*\.\s*(.+)", chunk, re.S)
        if m:
            body = m.group(2).strip()
            # drop amendment-status stubs: "5. (canceled)" etc. — pgpubs
            # publish post-preliminary-amendment; stubs are not real claims
            if re.match(r"^\(\s*(canceled|cancelled|withdrawn|not entered)\b",
                        body, re.I) or len(body) < 60:
                continue
            claims[int(m.group(1))] = body
    if not claims:
        n_no_claims += 1
        continue
    app_info[a] = {
        "claims": claims,
        "judgement": rec.get("first_draft_approved"),
        "final_outcome": rec.get("final_outcome"),
        "n_office_actions": rec.get("n_office_actions"),
        "cpc_section": None, "year": None, "has_prelim_amend": None,
        "flags": {},
    }
log(f"  claims parsed for {len(app_info):,} apps (no-claims: {n_no_claims:,})")

# ---------- 3. GP paragraph corpus ----------
log("Loading GP paragraph corpus ...")
gp = {}
def eat_gp(path, op):
    with op(path) as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            if r.get("paragraphs"):
                gp[norm(r["pgpub_id"])] = r["paragraphs"]
eat_gp(f"{PROC}/paragraph_keyed_specs.jsonl.gz", lambda p: gzip.open(p, "rt"))
for fn in sorted(glob.glob(f"{PROC}/paragraph_keyed_specs_v2/*.jsonl")):
    eat_gp(fn, open)
log(f"  {len(gp):,} docs with paragraphs")

# ---------- 4. emit ----------
log("Assembling testbed ...")
stats = {"records": 0, "claims_pos": 0, "claims_neg": 0, "elements": 0,
         "el_para_resolved": 0, "el_para_anchor": 0, "docs_with_gp": 0, "docs_total": 0}
out_path = f"{OUT_DIR}/testbed.jsonl.gz"
with gzip.open(out_path, "wt") as out:
    for (a, ifw), els in oa.items():
        info = app_info.get(a)
        if info is None:
            continue
        fell = {e["target_claim"] for e in els}
        claims = [{"num": n, "text": txt, "fell_102": n in fell}
                  for n, txt in sorted(info["claims"].items())]
        # group elements by doc
        art = {}
        for e in els:
            dn = norm(e["prior_art_pgpub_id"])
            if not dn:
                continue
            lt = loc_type(e.get("prior_art_location"))
            el = {"target_claim": e["target_claim"],
                  "claim_element": e.get("claim_element"),
                  "location": e.get("prior_art_location"), "loc_type": lt,
                  "paragraph_text": None}
            if lt == "paragraph":
                stats["el_para_anchor"] += 1
                paras = gp.get(dn)
                if paras:
                    hits = [paras[x] for x in parse_anchors(e.get("prior_art_location"))
                            if x in paras]
                    if hits:
                        el["paragraph_text"] = "\n".join(hits)
                        stats["el_para_resolved"] += 1
            art.setdefault(dn, {"doc_id": dn, "in_gp_corpus": dn in gp,
                                "elements": []})["elements"].append(el)
            stats["elements"] += 1
        if not art:
            continue
        rec = {"app_id": a, "ifw_number": ifw,
               "judgement": info["judgement"],
               "final_outcome": info["final_outcome"],
               "n_office_actions": info["n_office_actions"],
               "cpc_section": info["cpc_section"],
               "year": info["year"], "has_prelim_amend": info["has_prelim_amend"],
               **info["flags"],
               "claims": claims, "art": list(art.values())}
        out.write(json.dumps(rec) + "\n")
        stats["records"] += 1
        stats["claims_pos"] += sum(c["fell_102"] for c in claims)
        stats["claims_neg"] += sum(not c["fell_102"] for c in claims)
        stats["docs_total"] += len(art)
        stats["docs_with_gp"] += sum(d["in_gp_corpus"] for d in art.values())

log(json.dumps(stats, indent=2))
log(f"Wrote {out_path}")

# ---------- 5. validation sample (manual inspection) ----------
log("\n=== VALIDATION SAMPLES ===")
import random
random.seed(0)
recs = []
with gzip.open(out_path, "rt") as f:
    for line in f:
        recs.append(line)
for line in random.sample(recs, min(2, len(recs))):
    r = json.loads(line)
    pos = [c for c in r["claims"] if c["fell_102"]][:1]
    neg = [c for c in r["claims"] if not c["fell_102"]][:1]
    log(f"\napp {r['app_id']} ifw {r['ifw_number']} label={r['judgement']} "
        f"claims={len(r['claims'])} art_docs={len(r['art'])}")
    for c in pos + neg:
        log(f"  claim {c['num']} fell={c['fell_102']}: {c['text'][:160]}")
    d = r["art"][0]
    e = d["elements"][0]
    log(f"  art {d['doc_id']} (gp={d['in_gp_corpus']}) el->claim {e['target_claim']}: "
        f"{str(e['claim_element'])[:100]} @ {e['location']}")
    if e["paragraph_text"]:
        log(f"    resolved para: {e['paragraph_text'][:180]}")
log("DONE")
