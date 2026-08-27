"""Gold-paragraph 102/103 mixture testbed builder (task #61). STAGED — CPU-validate only.

Research question: how much does claim-level prior-art verification (V) improve
when the evidence set includes the EXAMINER'S OWN cited paragraphs (gold) vs
purely retrieved paragraphs?

Unit: (app_id, ifw_number, claim_num).
  - label_fell: claim was targeted by the examiner's element->art mapping in
    this OA (from the Llama OA extraction).  Negatives = untargeted claims of
    the SAME OA, so every unit (both labels) comes from an app/OA that has
    examiner citations — "has examiner citation" cannot leak the label at the
    app level.  (Caveat inherited from truecite v1: untargeted claims may be
    103/112-rejected; filter on oa_primary_rejection_type downstream.)
  - gold: examiner-cited paragraphs resolved from the GP pgpub-paragraph
    corpus (paragraph-anchor locations only; col/line/figure locations are
    recorded but unresolvable against pgpub paragraph keys).
  - retrieved evidence: NOT built here.  Each unit records the OA-level
    cited-doc pool (`cited_docs`); the scoring script plugs in v6a/xenc
    retrieval.  See condition_spec.json for the three conditions.

Conditions supported downstream (condition_spec.json):
  (a) gold-only          — gold paragraphs, padded to K with distribution-
                           matched fillers if |gold| < K
  (b) retrieved-only     — top-K retrieved (control)
  (c) mixture            — gold ∪ distribution-matched retrieved fillers,
                           same fixed total K per claim, SYMMETRIC across
                           labels (negatives get all-K retrieved/filler sets
                           assembled identically; Phase-2 design rules:
                           fixed K, distribution-matched fillers).

Leakage rules enforced here:
  - evidence text comes ONLY from the GP paragraph corpus (pgpub specs,
    ex-ante by construction); no OA text ever enters a unit record;
  - the extraction's own rejection language (claim_element strings, location
    strings) is stored under `gold_provenance` for AUDIT ONLY — scoring
    scripts must never feed it to the verifier;
  - claim text = the app's published pgpub claims (public pre-OA).

Inputs:
  datasets/patents/processed/oa_102_extractions_v2/extractions_round_*.jsonl
  datasets/patents/processed/office_actions_v3/office_actions_part_000_bulk_p*.jsonl
      (document_code CTNF/CTFR, document_date, primary rejection type 102/103)
  datasets/patents/raw/patex/application_data.csv   (outcome + app->pgpub)
  datasets/patents/patents_dataset.jsonl.gz          (pgpub claims text)
  datasets/patents/processed/paragraph_keyed_specs.jsonl.gz
  datasets/patents/processed/paragraph_keyed_specs_v2/*.jsonl

Output (default datasets/patents/processed/gold_mixture_testbed_v1/):
  units.jsonl          one record per (app, ifw, claim)
  build_stats.json     all join rates
  condition_spec.json  downstream condition definitions
  patex_cache.json / app_claims_cache.jsonl   (reusable caches)

Usage:
  python scripts/build_gold_mixture_testbed.py --max-records 3000 \
      --spot-check 3 --no-write          # CPU validation
  python scripts/build_gold_mixture_testbed.py                      # full (DO NOT run yet)
"""
import argparse, csv, glob, gzip, json, os, random, re, sys
from collections import Counter, defaultdict

csv.field_size_limit(sys.maxsize)

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
PATEX_CSV = f"{BASE}/datasets/patents/raw/patex/application_data.csv"
PT_DS = f"{BASE}/datasets/patents/patents_dataset.jsonl.gz"

norm = lambda x: "".join(c for c in str(x) if c.isdigit()).lstrip("0")
log = lambda m: print(m, flush=True)


def loc_type(loc):
    if not loc:
        return "none"
    l = str(loc).lower()
    if re.search(r"\[\s*0\d{2,3}\s*\]|para(?:graph)?|¶|\b0\d{3}\b", l):
        return "paragraph"
    if re.search(r"col(?:umn)?\b|col\.|\blines?\b", l):
        return "col_line"
    if re.search(r"fig", l):
        return "figure"
    if re.search(r"\bpage\b|\bpg\.", l):
        return "page"
    if re.search(r"abstract", l):
        return "abstract"
    return "other"


def parse_anchors(loc, max_range=30):
    """'[0034]-[0036]', 'para. 34', '0136-0142', 'paragraphs 0041, 0044'
    -> ['0034','0035','0036', ...].  Expands short ranges."""
    l = str(loc).lower()
    nums = []
    # bracketed or para-prefixed or bare 0xxx anchors, with optional range tail
    pat = re.compile(
        r"(?:\[\s*(\d{3,4})\s*\]|para(?:graph)?s?\.?\s*\[?(\d{1,4})\]?|\b(0\d{3})\b)"
        r"(?:\s*[-–]+\s*\[?(\d{3,4})\]?)?")
    for m in pat.finditer(l):
        a = m.group(1) or m.group(2) or m.group(3)
        b = m.group(4)
        a_i = int(a)
        nums.append(a_i)
        if b:
            b_i = int(b)
            if a_i < b_i <= a_i + max_range:
                nums.extend(range(a_i + 1, b_i + 1))
    out = [str(n).zfill(4) for n in nums]
    return list(dict.fromkeys(out))


def id_style(doc_norm):
    return "pgpub" if len(doc_norm) >= 9 else "grant"


# ---------------------------------------------------------------- loaders
def load_extractions(globpat, max_records):
    oa = {}  # (app, ifw) -> list of element rows
    n_rec = n_fail = 0
    for fn in sorted(glob.glob(globpat)):
        for line in open(fn):
            if max_records and n_rec >= max_records:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_rec += 1
            ext = r.get("extraction")
            if not ext:
                n_fail += 1
                continue
            rows = [e for e in ext
                    if isinstance(e.get("target_claim"), int) and e.get("prior_art_pgpub_id")]
            if rows:
                oa.setdefault((r["app_id"], r.get("ifw_number")), []).extend(rows)
    return oa, n_rec, n_fail


def load_bulk_oa_index(keys_needed):
    """(app,ifw) -> {document_code, document_date, primary rejection type}."""
    idx = {}
    for fn in sorted(glob.glob(f"{PROC}/office_actions_v3/office_actions_part_000_bulk_p*.jsonl")):
        for line in open(fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = (r["app_id"], r.get("ifw_number"))
            if k not in keys_needed:
                continue
            rej = r.get("rejections") or []
            idx[k] = {"document_code": r.get("document_code"),
                      "document_date": r.get("document_date"),
                      # OACT bulk metadata carries ONE (primary) rejection per OA
                      "primary_rejection_type": rej[0].get("type") if rej else None}
    return idx


def load_patex(apps_needed, cache_path):
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    missing = apps_needed - set(cache)
    if missing:
        log(f"  PatEx scan for {len(missing):,} apps ...")
        with open(PATEX_CSV) as f:
            for row in csv.DictReader(f):
                a = row["application_number"].replace("/", "").strip()
                if a in missing:
                    m = re.search(r"(20\d{2}\d{7})", row["earliest_pgpub_number"])
                    cache[a] = {"appl_status_desc": row["appl_status_desc"],
                                "granted": bool(row["patent_number"].strip()),
                                "pgpub_id": m.group(1) if m else None}
        json.dump(cache, open(cache_path, "w"))
    return cache


CLAIM_SPLIT = re.compile(r"\n(?=\d{1,3}\s*\.\s)")


def parse_claims(pg_claims):
    claims = {}
    for chunk in CLAIM_SPLIT.split("\n" + (pg_claims or "").strip()):
        m = re.match(r"\s*(\d{1,3})\s*\.\s*(.+)", chunk, re.S)
        if not m:
            continue
        body = m.group(2).strip()
        if re.match(r"^\(\s*(canceled|cancelled|withdrawn|not entered)\b", body, re.I) \
                or len(body) < 60:
            continue
        claims[int(m.group(1))] = body
    return claims


def load_app_claims(app2pg, cache_path):
    """app -> pg_claims text, via patents_dataset scan keyed on pgpub. Cached."""
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            try:
                r = json.loads(line)
                cache[r["app_id"]] = r["pg_claims"]
            except Exception:
                continue
    pg2app = defaultdict(list)
    for a, pg in app2pg.items():
        if a not in cache and pg:
            pg2app[norm(pg)].append(a)
    if pg2app:
        log(f"  patents_dataset scan for {len(pg2app):,} pgpubs ...")
        with gzip.open(PT_DS, "rt") as f, open(cache_path, "a", buffering=1) as out:
            for line in f:
                r = json.loads(line)
                pg = norm(r.get("pgpub_id"))
                for a in pg2app.get(pg, []):
                    cache[a] = r.get("pg_claims")
                    out.write(json.dumps({"app_id": a, "pg_claims": r.get("pg_claims")}) + "\n")
    return cache


def load_gp_corpus(docs_needed):
    gp = {}

    def eat(path, op):
        with op(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                d = norm(r.get("pgpub_id", ""))
                if d in docs_needed and r.get("paragraphs"):
                    gp[d] = r["paragraphs"]
    eat(f"{PROC}/paragraph_keyed_specs.jsonl.gz", lambda p: gzip.open(p, "rt"))
    for fn in sorted(glob.glob(f"{PROC}/paragraph_keyed_specs_v2/*.jsonl")):
        eat(fn, open)
    return gp


CONDITION_SPEC = {
    "unit": "(app_id, ifw_number, claim_num)",
    "label": "label_fell (claim targeted by examiner element->art mapping in this OA)",
    "fixed_K": "single K for ALL units and conditions; choose after retrieval hook lands",
    "conditions": {
        "gold_only": "gold paragraphs; if |gold|<K pad with distribution-matched fillers",
        "retrieved_only": "top-K retrieved paragraphs from cited_docs pool (v6a/xenc hook); control",
        "mixture": "gold UNION distribution-matched retrieved fillers, total exactly K",
    },
    "symmetry": ("negatives are untargeted claims of OAs that DO carry examiner citations; "
                 "their evidence sets are assembled with the identical K and filler "
                 "distribution so evidence-set shape never leaks the label"),
    "leakage": ("evidence text only from GP pgpub paragraphs; never OA text; "
                "gold_provenance (examiner element/location strings) is audit-only"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction-glob",
                    default=f"{PROC}/oa_102_extractions_v2/extractions_round_*.jsonl")
    ap.add_argument("--out-dir", default=f"{PROC}/gold_mixture_testbed_v1")
    ap.add_argument("--max-records", type=int, default=0,
                    help="cap on extraction records (CPU validation)")
    ap.add_argument("--spot-check", type=int, default=0,
                    help="print N fully-resolved examples for manual reading")
    ap.add_argument("--no-write", action="store_true",
                    help="skip writing units.jsonl (validation runs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stats = Counter()

    # 1. extractions
    log("Loading extraction records ...")
    oa, n_rec, n_fail = load_extractions(args.extraction_glob, args.max_records)
    stats["extraction_records"] = n_rec
    stats["extraction_parse_failures"] = n_fail
    stats["oas_with_rows"] = len(oa)
    log(f"  {n_rec:,} records ({n_fail:,} parse failures), {len(oa):,} OAs with usable rows")

    # 2. OACT bulk join (document_code + primary rejection type)
    log("Joining OACT bulk metadata ...")
    bulk = load_bulk_oa_index(set(oa))
    stats["oa_bulk_joined"] = len(bulk)
    log(f"  {len(bulk):,}/{len(oa):,} OAs join to OACT bulk "
        f"({len(bulk)/max(len(oa),1):.1%})")

    # 3. outcomes + app->pgpub via PatEx
    apps = {a for a, _ in oa}
    log("Resolving app outcomes via PatEx ...")
    patex = load_patex(apps, f"{args.out_dir}/patex_cache.json")
    stats["apps_total"] = len(apps)
    stats["apps_in_patex"] = sum(a in patex for a in apps)
    stats["apps_granted"] = sum(patex.get(a, {}).get("granted", False) for a in apps)

    # 4. claims text
    log("Resolving app claims ...")
    app2pg = {a: patex[a]["pgpub_id"] for a in apps if patex.get(a, {}).get("pgpub_id")}
    claims_raw = load_app_claims(app2pg, f"{args.out_dir}/app_claims_cache.jsonl")
    app_claims = {a: parse_claims(t) for a, t in claims_raw.items() if t}
    app_claims = {a: c for a, c in app_claims.items() if c}
    stats["apps_with_claims"] = len(app_claims)
    log(f"  claims for {len(app_claims):,}/{len(apps):,} apps")

    # 5. GP paragraph corpus for cited docs
    log("Loading GP paragraph corpus (cited docs only) ...")
    cited_all = {norm(e["prior_art_pgpub_id"]) for rows in oa.values() for e in rows}
    cited_all.discard("")
    gp = load_gp_corpus(cited_all)
    stats["cited_docs_unique"] = len(cited_all)
    stats["cited_docs_in_gp"] = len(gp)
    log(f"  {len(gp):,}/{len(cited_all):,} unique cited docs have paragraphs "
        f"({len(gp)/max(len(cited_all),1):.1%})")

    # 6. emit units
    log("Assembling units ...")
    rng = random.Random(args.seed)
    out_path = f"{args.out_dir}/units.jsonl"
    out = None if args.no_write else open(out_path, "w")
    spot_pool = []
    for (a, ifw), rows in oa.items():
        info = app_claims.get(a)
        meta = bulk.get((a, ifw))
        px = patex.get(a)
        if info is None or px is None:
            stats["oa_skipped_no_claims_or_patex"] += 1
            continue
        oa_year = int(meta["document_date"][:4]) if meta and meta.get("document_date") else None

        # OA-level cited-doc pool (retrieval hook for ALL claims of this OA)
        pool, seen = [], set()
        for e in rows:
            d = norm(e["prior_art_pgpub_id"])
            if not d or d in seen:
                continue
            seen.add(d)
            ev_year = int(d[:4]) if id_style(d) == "pgpub" and d[:4].isdigit() else None
            pool.append({"doc_id": d, "id_style": id_style(d),
                         "in_gp_corpus": d in gp,
                         "doc_year": ev_year,
                         "ex_ante_ok": (ev_year is None or oa_year is None
                                        or ev_year <= oa_year)})
        stats["pool_docs"] += len(pool)
        stats["pool_docs_temporal_anomaly"] += sum(not p["ex_ante_ok"] for p in pool)

        # gold per targeted claim
        by_claim = defaultdict(list)
        for e in rows:
            by_claim[e["target_claim"]].append(e)
        targeted = set(by_claim)

        for cnum, ctext in sorted(info.items()):
            fell = cnum in targeted
            gold_by_doc, prov = defaultdict(dict), []
            if fell:
                for e in by_claim[cnum]:
                    d = norm(e["prior_art_pgpub_id"])
                    lt = loc_type(e.get("prior_art_location"))
                    stats[f"el_loc_{lt}"] += 1
                    prov.append({"doc_id": d, "loc_type": lt,
                                 "claim_element": e.get("claim_element"),
                                 "location_raw": e.get("prior_art_location")})
                    if lt != "paragraph":
                        continue
                    stats["el_paragraph_anchor"] += 1
                    paras = gp.get(d)
                    if not paras:
                        stats["el_paragraph_doc_missing"] += 1
                        continue
                    anchors = parse_anchors(e.get("prior_art_location"))
                    hits = {x: paras[x] for x in anchors if x in paras}
                    if hits:
                        stats["el_paragraph_resolved"] += 1
                        gold_by_doc[d].update(hits)   # dedup (doc, anchor)
                    else:
                        stats["el_paragraph_anchor_miss"] += 1
            gold = [{"doc_id": d, "anchors": sorted(hits), "paragraphs": hits}
                    for d, hits in sorted(gold_by_doc.items())]
            unit = {
                "app_id": a, "ifw_number": ifw, "claim_num": cnum,
                "claim_text": ctext,
                "label_fell": fell,
                "oa_document_code": meta.get("document_code") if meta else None,
                "oa_date": meta.get("document_date") if meta else None,
                "oa_primary_rejection_type": meta.get("primary_rejection_type") if meta else None,
                "app_granted": px["granted"],
                "appl_status_desc": px["appl_status_desc"],
                "app_pgpub_id": px["pgpub_id"],
                "cited_docs": pool,            # retrieval hook (per-OA pool)
                "gold": gold,                  # resolved examiner paragraphs
                "gold_n_paras": sum(len(g["paragraphs"]) for g in gold),
                "has_gold": bool(gold),
                "gold_provenance": prov,       # AUDIT ONLY — never feed to verifier
            }
            stats["units"] += 1
            stats["units_pos" if fell else "units_neg"] += 1
            if fell and gold:
                stats["units_pos_with_gold"] += 1
            if out:
                out.write(json.dumps(unit) + "\n")
            if args.spot_check and fell and gold and len(spot_pool) < args.spot_check * 20:
                spot_pool.append(unit)
    if out:
        out.close()
        json.dump(CONDITION_SPEC, open(f"{args.out_dir}/condition_spec.json", "w"), indent=2)
        json.dump(dict(stats), open(f"{args.out_dir}/build_stats.json", "w"), indent=2)
        log(f"Wrote {out_path}")

    # 7. report
    log("\n=== BUILD STATS ===")
    log(json.dumps(dict(sorted(stats.items())), indent=2))
    u = stats["units_pos"]
    if u:
        log(f"\npos units with >=1 resolved gold paragraph: "
            f"{stats['units_pos_with_gold']:,}/{u:,} ({stats['units_pos_with_gold']/u:.1%})")

    # 8. spot check
    if args.spot_check:
        log(f"\n=== SPOT CHECK ({args.spot_check}) ===")
        for unit in rng.sample(spot_pool, min(args.spot_check, len(spot_pool))):
            log(f"\napp {unit['app_id']} ifw {unit['ifw_number']} claim {unit['claim_num']} "
                f"fell={unit['label_fell']} oa={unit['oa_document_code']} "
                f"rej_type={unit['oa_primary_rejection_type']} granted={unit['app_granted']}")
            log(f"CLAIM: {unit['claim_text'][:400]}")
            for g in unit["gold"]:
                for anc, txt in list(g["paragraphs"].items())[:3]:
                    log(f"  GOLD {g['doc_id']} [{anc}]: {txt[:300]}")
            for p in unit["gold_provenance"][:4]:
                log(f"  (prov: {str(p['claim_element'])[:70]} @ {str(p['location_raw'])[:50]} [{p['loc_type']}])")
    log("DONE")


if __name__ == "__main__":
    main()
