"""Phase 2 step 4: assemble the count-balanced classifier dataset (v0).

Per app (first-draft abstract+claims), attach exactly K=10 prior-art claim texts:
  - real:   resolved examiner 102/103 cites (fidelity-first: true cited docs)
  - filler: from step-3 top-200 candidates, filtered (year < app year, not self,
            not a real cite) and similarity-distribution-MATCHED to real-cite sims
            (global real-sim distribution for apps with zero resolved reals).
Count carries zero information by construction. n_real_attached lives in the
sidecar manifest ONLY, never in model input.

Stages (resumable, CPU-only — real-cite sims via pool-vector lookup):
  A. pool metadata: norm-id -> pool_row map; pool_row -> year array
  B. real-attachment manifest: per-app resolved cites + sims (pool lookup)
  C. filler selection over step-3 shards -> manifest parquet
  D. text join + final JSONL assembly + splits + summary stats

Inputs:
  processed/phase2_real_cites.jsonl.gz            {app_id, cites_102, cites_103}
  processed/phase2_cited_doc_text.parquet         cited_id, claim_text, year, source
  processed/phase2_filler_candidates/             app_claim_emb_fp16.npy, app_ids.txt,
                                                  top200_shard_*.parquet (app_id, pool_row, sim)
  indexes/patent_claims_v3/pool_fp16.npy + pgpub_ids.txt
  patents_first_draft_cpc_balanced_with_rejections.csv.gz
Outputs (processed/phase2_dataset_v0/):
  attachment_manifest.parquet   app_id, kind, doc_norm, sim, doc_year, action
  dataset_v0.jsonl.gz           one record per app (one-shot write)
  summary.json
"""
import csv, gzip, hashlib, json, os, sys, time
csv.field_size_limit(sys.maxsize)
import numpy as np
import pandas as pd

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
IDX_DIR = f"{BASE}/indexes/patent_claims_v3"
CAND_DIR = f"{PROC}/phase2_filler_candidates"
OUT_DIR = f"{PROC}/phase2_dataset_v0"
APPS_CSV = f"{BASE}/datasets/patents/patents_first_draft_cpc_balanced_with_rejections.csv.gz"
K = 10
SEED = 0

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)


def norm(x):
    return "".join(ch for ch in str(x) if ch.isdigit()).lstrip("0")


def log(msg):
    print(msg, flush=True)


# ---------- load app universe ----------
log("Loading app universe ...")
apps = {}  # app_id -> row dict
with gzip.open(APPS_CSV, "rt") as f:
    for row in csv.DictReader(f):
        a = (row.get("app_id") or "").strip()
        if a:
            apps[a] = row
log(f"  {len(apps):,} apps")

pool_ids = [l.strip() for l in open(f"{IDX_DIR}/pgpub_ids.txt")]
n_pool = len(pool_ids)

# ---------- Stage A: pool metadata ----------
YEAR_NPY = f"{OUT_DIR}/pool_years.npy"
if not os.path.exists(YEAR_NPY + ".done"):
    log("Stage A: pool norm-id map + years ...")
    pool_norm = {}
    for i, p in enumerate(pool_ids):
        pn = norm(p)
        if pn and pn not in pool_norm:
            pool_norm[pn] = i
    years = np.zeros(n_pool, dtype=np.int16)
    for src, idcol in [("pgpub_claims1.parquet", "pgpub_id"),
                       ("granted_patents_claim1.parquet", "patent_id")]:
        df = pd.read_parquet(f"{PROC}/{src}", columns=[idcol, "claim_year"])
        for did, yr in zip(df[idcol].astype(str), df["claim_year"]):
            r = pool_norm.get(norm(did))
            if r is not None and years[r] == 0 and pd.notna(yr):
                years[r] = int(yr)
        del df
        log(f"  {src}: years filled so far {int((years > 0).sum()):,}/{n_pool:,}")
    np.save(YEAR_NPY, years)
    with open(f"{OUT_DIR}/pool_norm_map.tsv", "w") as f:
        for pn, i in pool_norm.items():
            f.write(f"{pn}\t{i}\n")
    open(YEAR_NPY + ".done", "w").write("ok")
    log("  stage A done")
else:
    log("Stage A: cached")
    pool_norm = {}
    with open(f"{OUT_DIR}/pool_norm_map.tsv") as f:
        for line in f:
            pn, i = line.rstrip("\n").split("\t")
            pool_norm[pn] = int(i)
years = np.load(YEAR_NPY)

# ---------- Stage B: real-attachment manifest ----------
REAL_PQ = f"{OUT_DIR}/real_manifest.parquet"
if not os.path.exists(REAL_PQ):
    log("Stage B: real cites + sims via pool lookup ...")
    cited = pd.read_parquet(f"{PROC}/phase2_cited_doc_text.parquet")
    cited["dnorm"] = cited["cited_id"].map(norm)
    resolved = dict(zip(cited["dnorm"], zip(cited["year"].fillna(0).astype(int),
                                            cited["source"])))
    app_emb_ids = [l.split("\t")[0].strip() for l in open(f"{CAND_DIR}/app_ids.txt")]
    app_row = {a: i for i, a in enumerate(app_emb_ids)}
    qemb = np.load(f"{CAND_DIR}/app_claim_emb_fp16.npy", mmap_mode="r")
    pool = np.load(f"{IDX_DIR}/pool_fp16.npy", mmap_mode="r")

    rows, n_no_emb, n_sim_miss = [], 0, 0
    with gzip.open(f"{PROC}/phase2_real_cites.jsonl.gz", "rt") as f:
        for li, line in enumerate(f):
            rec = json.loads(line)
            a = rec["app_id"]
            if a not in apps:
                continue
            qi = app_row.get(a)
            if qi is None:
                n_no_emb += 1
            actions = {}  # dnorm -> set of actions (dedupe 102+103 overlap)
            for act in ("102", "103"):
                for c in rec.get(f"cites_{act}", []):
                    dn = norm(c)
                    if dn:
                        actions.setdefault(dn, set()).add(act)
            qv = qemb[qi].astype(np.float32) if qi is not None else None
            for dn, acts in actions.items():
                if dn not in resolved:
                    continue  # unresolved -> not attachable (control-set logic)
                yr, _src = resolved[dn]
                pr = pool_norm.get(dn)
                if qv is not None and pr is not None:
                    sim = float(qv @ pool[pr].astype(np.float32))
                else:
                    sim = np.nan
                    n_sim_miss += 1
                rows.append((a, "real", dn, sim, yr, "+".join(sorted(acts))))
            if li % 20000 == 0:
                log(f"  {li:,} apps scanned, {len(rows):,} real attachments")
    real_df = pd.DataFrame(rows, columns=["app_id", "kind", "doc_norm", "sim",
                                          "doc_year", "action"])
    real_df.to_parquet(REAL_PQ, index=False)
    log(f"  stage B done: {len(real_df):,} resolved real attachments; "
        f"apps w/o emb: {n_no_emb:,}; sim misses: {n_sim_miss:,}")
else:
    log("Stage B: cached")
real_df = pd.read_parquet(REAL_PQ)

# ---------- Stage C: filler selection ----------
FILL_PQ = f"{OUT_DIR}/filler_manifest.parquet"
if not os.path.exists(FILL_PQ):
    log("Stage C: filler selection (filter + distribution-match) ...")
    # per-app real info
    real_by_app = {}
    for a, g in real_df.groupby("app_id"):
        real_by_app[a] = (set(g["doc_norm"]), g["sim"].dropna().to_numpy())
    global_sims = real_df["sim"].dropna().to_numpy()
    log(f"  global real-sim pool: {len(global_sims):,} "
        f"(p10={np.percentile(global_sims,10):.3f} p50={np.percentile(global_sims,50):.3f} "
        f"p90={np.percentile(global_sims,90):.3f})")

    # full real-cite id sets per app (incl. UNRESOLVED, for exclusion)
    excl_by_app = {}
    with gzip.open(f"{PROC}/phase2_real_cites.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            ex = {norm(c) for c in rec.get("cites_102", []) + rec.get("cites_103", [])}
            excl_by_app[rec["app_id"]] = {e for e in ex if e}

    shards = sorted(f for f in os.listdir(CAND_DIR) if f.startswith("top200_shard_"))
    log(f"  {len(shards)} candidate shards")
    out_rows, short_apps = [], 0
    t0 = time.time()
    for si, sh in enumerate(shards):
        df = pd.read_parquet(f"{CAND_DIR}/{sh}")
        rp = f"{CAND_DIR}/{sh.replace('top200_shard_', 'rand_shard_')}"
        if os.path.exists(rp):  # step-3b sim-stratified randoms: full-range targets
            df = pd.concat([df, pd.read_parquet(rp)], ignore_index=True)
        for a, g in df.groupby("app_id", sort=False):
            row = apps.get(a)
            if row is None:
                continue
            app_year = int(float(row["year"])) if row.get("year") else 0
            self_norm = norm(row.get("pgpub_id") or "")
            real_norms, real_sims = real_by_app.get(a, (set(), np.array([])))
            excl = excl_by_app.get(a, set()) | {self_norm}
            n_real = min(len(real_norms), K)
            n_fill = K - n_real
            if n_fill <= 0:
                continue
            # filter candidates
            pr = g["pool_row"].to_numpy()
            sims = g["sim"].to_numpy()
            cand_norm = np.array([norm(pool_ids[r]) for r in pr])
            cand_year = years[pr]
            ok = (cand_year > 0) & (cand_year < app_year)
            ok &= ~np.isin(cand_norm, list(excl)) if excl else ok
            # dedupe same doc appearing twice (pgpub + grant rows)
            seen, keep = set(), []
            for i in np.where(ok)[0]:
                if cand_norm[i] not in seen:
                    seen.add(cand_norm[i]); keep.append(i)
            keep = np.array(keep, dtype=int)
            if len(keep) == 0:
                short_apps += 1
                continue
            csims = sims[keep]
            order = np.argsort(csims)
            sorted_sims = csims[order]
            # target sims: ALWAYS the global real-sim distribution. Per-app
            # targets concentrate fillers around the app's few real sims ->
            # low per-app spread for rejected apps = std/texture leak
            # (v0.1 measured AUC(label~sim_std)=0.60). Reals are themselves
            # draws from the global distribution, so global targets make
            # per-app sim marginals label-invariant by construction.
            src = global_sims
            targets = rng.choice(src, size=n_fill, replace=True)
            used = np.zeros(len(keep), dtype=bool)
            chosen = []
            for t in targets:
                j = int(np.searchsorted(sorted_sims, t))
                # nearest unused around insertion point
                best, bestd = -1, 1e9
                for d in range(len(keep)):
                    for jj in (j - d, j + d):
                        if 0 <= jj < len(keep) and not used[jj]:
                            dd = abs(sorted_sims[jj] - t)
                            if dd < bestd:
                                best, bestd = jj, dd
                    if best >= 0 and d > 5:  # local window is enough once found
                        break
                if best < 0:
                    break
                used[best] = True
                ci = keep[order[best]]
                chosen.append((a, "filler", cand_norm[ci], float(sims[ci]),
                               int(cand_year[ci]), ""))
            if len(chosen) < n_fill:
                short_apps += 1
            out_rows.extend(chosen)
        log(f"  shard {si+1}/{len(shards)}: {len(out_rows):,} fillers "
            f"({(time.time()-t0)/60:.1f} min)")
    fill_df = pd.DataFrame(out_rows, columns=["app_id", "kind", "doc_norm", "sim",
                                              "doc_year", "action"])
    fill_df.to_parquet(FILL_PQ, index=False)
    log(f"  stage C done: {len(fill_df):,} fillers; short apps: {short_apps:,}")
else:
    log("Stage C: cached")
fill_df = pd.read_parquet(FILL_PQ)

# ---------- Stage D: text join + assembly ----------
FINAL = f"{OUT_DIR}/dataset_v0.jsonl.gz"
if not os.path.exists(FINAL):
    log("Stage D: assembling final dataset ...")
    # real text from step-2 parquet
    cited = pd.read_parquet(f"{PROC}/phase2_cited_doc_text.parquet")
    cited["dnorm"] = cited["cited_id"].map(norm)
    text_map = dict(zip(cited["dnorm"], cited["claim_text"]))
    # filler text from pool sources (norm-keyed, only needed ids)
    need = set(fill_df["doc_norm"]) - set(text_map)
    log(f"  filler texts needed: {len(need):,}")
    for src, idcol in [("pgpub_claims1.parquet", "pgpub_id"),
                       ("granted_patents_claim1.parquet", "patent_id")]:
        df = pd.read_parquet(f"{PROC}/{src}", columns=[idcol, "claim_text"])
        dn = df[idcol].astype(str).map(norm)
        m = dn.isin(need)
        for k, v in zip(dn[m], df["claim_text"][m]):
            if k not in text_map:
                text_map[k] = v
        del df
        log(f"  after {src}: {len(need - set(text_map)):,} still missing")

    # cap reals at K (prefer 102s, then random 103s) and build manifest
    manifest = []
    att_by_app = {}
    for a, g in pd.concat([real_df, fill_df]).groupby("app_id", sort=False):
        reals = g[g["kind"] == "real"]
        if len(reals) > K:
            r102 = reals[reals["action"].str.contains("102")]
            r103 = reals[~reals["action"].str.contains("102")]
            take = pd.concat([r102.head(K),
                              r103.sample(min(len(r103), K - min(len(r102), K)),
                                          random_state=SEED)])
            reals = take.head(K)
        sel = pd.concat([reals, g[g["kind"] == "filler"]]).head(K)
        att_by_app[a] = sel
        manifest.append(sel)
    manifest = pd.concat(manifest)
    manifest.to_parquet(f"{OUT_DIR}/attachment_manifest.parquet", index=False)

    n_written, n_dropped_noatt, n_text_miss = 0, 0, 0
    label_counts, nreal_by_label = {}, {0: [], 1: []}
    with gzip.open(FINAL, "wt") as out:
        for a, row in apps.items():
            sel = att_by_app.get(a)
            if sel is None or len(sel) < K:
                n_dropped_noatt += 1
                continue
            atts = []
            for _, r in sel.iterrows():
                t = text_map.get(r["doc_norm"])
                if t is None or not str(t).strip():
                    n_text_miss += 1
                    t = None
                atts.append({"text": t, "year": int(r["doc_year"]) or None})
            if any(x["text"] is None for x in atts):
                n_dropped_noatt += 1
                continue
            # deterministic per-app shuffle (hide real-vs-filler order)
            h = int(hashlib.md5(a.encode()).hexdigest(), 16)
            prm = np.random.default_rng(h % (2**32)).permutation(K)
            atts = [atts[i] for i in prm]
            y = int(float(row["judgement"]))
            split_h = int(hashlib.md5(("split:" + a).encode()).hexdigest(), 16) % 100
            split = "train" if split_h < 80 else ("val" if split_h < 90 else "test")
            rec = {
                "app_id": a, "text": row["text"], "attachments": atts,
                "judgement": y, "split": split,
                "cpc_section": row.get("cpc_section"),
                "year": row.get("year"),
            }
            for c in row:
                if c.startswith("rejected_"):
                    rec[c] = row[c]
            out.write(json.dumps(rec) + "\n")
            n_written += 1
            label_counts[y] = label_counts.get(y, 0) + 1
            nreal_by_label[y].append(int((sel["kind"] == "real").sum()))

    summary = {
        "K": K, "written": n_written, "dropped_no_full_K": n_dropped_noatt,
        "text_misses": n_text_miss, "label_counts": label_counts,
        "mean_n_real_by_label": {str(k): float(np.mean(v)) if v else None
                                 for k, v in nreal_by_label.items()},
        "note": "every written record has exactly K attachments; "
                "n_real lives only in attachment_manifest.parquet",
    }
    json.dump(summary, open(f"{OUT_DIR}/summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log("Stage D done.")
else:
    log("Stage D: cached")

log("ALL DONE")
