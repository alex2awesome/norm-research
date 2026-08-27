#!/usr/bin/env python3
"""U1 COMMUNITY cell — RoyalRoad reader engagement (unified-X triple, CW field).

Plan of record (registry 2026-08-16): X = fiction DESCRIPTIONS from the deep
pages; y from the listing engagement stats (community channel).

Design (mirrors the other community cells' within-stratum relative split —
jokes_community within-stratum quartile, journalism tweets within-outlet-day
median; here: MEDIAN split, the task ruling):

  POPULATION  the stable-hash RANDOM deep-fetch sample ONLY, recomputed
              deterministically with the same rule as scrape_deep_metrics.py
              targets(): drop STUB fictions, drop top-5000 by followers, order
              the rest by md5("deep::<fid>") and take the first 5000. The
              top-5000 stratum is popularity-SELECTED and the stubs are the
              verdict cell — both excluded so the community y is a site-
              representative contrast, not a range-restricted one.
  X           description_text (ld+json blurb, one shared renderer:
              unescape -> strip tags -> collapse whitespace, done in
              parse_deep_pages.py for every row identically).
              Gate: >= 200 chars (drops empty/placeholder blurbs).
  y           followers (listing snapshot, single 2026-08-12 scrape window)
              MEDIAN SPLIT within stratum = (primary genre, created-year).
              Strata with < MIN_STRATUM rows fall back to (year,) pooled
              strata; still-small strata dropped. Rows tied at the stratum
              median dropped (no arbitrary side).
  SPLITS      stable sha256("rr_community::<fid>") % 1000 -> <800/<900/rest
              (feedback_stable_hash_splits).

Output: royalroad_community_cell/rr_community_population.jsonl.gz + manifest.
Never deletes; output is versioned on collision.
"""
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "royalroad_expansion")
OUT_DIR = os.path.join(HERE, "royalroad_community_cell")
MIN_DESC_CHARS = 200
MIN_STRATUM = 8


def load_jsonl_gz(path):
    return [json.loads(l) for l in gzip.open(path, "rt")]


def random_sample_ids():
    """Deterministic recompute of scrape_deep_metrics.targets() 'rnd' leg."""
    stubs = {r["fiction_id"] for r in load_jsonl_gz(os.path.join(EXP, "listings_stub.jsonl.gz"))}
    seen, rows = set(), []
    for d in load_jsonl_gz(os.path.join(EXP, "listings_all.jsonl.gz")):
        if d["fiction_id"] in seen:
            continue
        seen.add(d["fiction_id"])
        rows.append((d["fiction_id"], d.get("followers") or 0))
    top = {fid for fid, _ in sorted(rows, key=lambda r: -r[1])[:5000]}
    rest = [fid for fid, _ in rows if fid not in top]
    rnd = sorted(rest, key=lambda f: hashlib.md5(f"deep::{f}".encode()).hexdigest())[:5000]
    return [f for f in rnd if f not in stubs], stubs, top


def split_of(fid):
    h = int(hashlib.sha256(f"rr_community::{fid}".encode()).hexdigest(), 16) % 1000
    return "train" if h < 800 else ("eval" if h < 900 else "test")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "rr_community_population.jsonl.gz")
    if os.path.exists(out_path):
        i = 2
        while os.path.exists(out_path.replace(".jsonl.gz", f"_v{i}.jsonl.gz")):
            i += 1
        out_path = out_path.replace(".jsonl.gz", f"_v{i}.jsonl.gz")

    rnd_ids, stubs, top = random_sample_ids()
    print(f"random-sample universe: {len(rnd_ids)} (stubs excluded {len(stubs)}, top-5000 excluded)")

    listing = {}
    for d in load_jsonl_gz(os.path.join(EXP, "listings_all.jsonl.gz")):
        listing.setdefault(d["fiction_id"], d)
    deep = {r["fiction_id"]: r for r in load_jsonl_gz(os.path.join(EXP, "rr_deep_parsed.jsonl.gz"))}

    cand, drops = [], Counter()
    for fid in rnd_ids:
        dp, li = deep.get(fid), listing.get(fid)
        if dp is None or dp.get("is_404"):
            drops["no_deep_page"] += 1
            continue
        desc = dp.get("description_text") or ""
        if len(desc) < MIN_DESC_CHARS:
            drops["short_desc"] += 1
            continue
        followers = li.get("followers")
        if followers is None:
            drops["no_followers"] += 1
            continue
        genres = dp.get("genres") or []
        genre = genres[0] if genres else (li.get("tags")[0] if li.get("tags") else "unknown")
        year = (dp.get("date_created") or "")[:4] or "unknown"
        cand.append(dict(
            fiction_id=fid, text=desc, followers=int(followers),
            views=li.get("views"), rating_pct=li.get("rating_pct"),
            pages=li.get("pages"), chapters=li.get("chapters"),
            genre=genre, year=year, genres_all=genres,
            title=dp.get("title"), date_created=dp.get("date_created"),
        ))
    print(f"candidates after gates: {len(cand)}  drops: {dict(drops)}")

    # strata: (genre, year) if big enough, else pooled (year,)
    by_gy = defaultdict(list)
    for r in cand:
        by_gy[(r["genre"], r["year"])].append(r)
    strata = defaultdict(list)
    for key, rows in by_gy.items():
        if len(rows) >= MIN_STRATUM:
            for r in rows:
                strata[f"{key[0]}::{key[1]}"].append(r)
        else:
            for r in rows:
                strata[f"__pooled__::{r['year']}"].append(r)
    small = [k for k, v in strata.items() if len(v) < MIN_STRATUM]
    n_small = sum(len(strata[k]) for k in small)
    for k in small:
        del strata[k]
    print(f"strata: {len(strata)} kept; {len(small)} dropped ({n_small} rows)")

    rows_out, ties = [], 0
    for sk, rows in strata.items():
        fol = sorted(r["followers"] for r in rows)
        n = len(fol)
        med = (fol[n // 2 - 1] + fol[n // 2]) / 2 if n % 2 == 0 else fol[n // 2]
        for r in rows:
            if r["followers"] == med:
                ties += 1
                continue
            rr = dict(r)
            rr["judgement"] = int(r["followers"] > med)
            rr["stratum"] = sk
            rr["stratum_median_followers"] = med
            rr["row_id"] = f"rrc_{r['fiction_id']}"
            rr["group"] = str(r["fiction_id"])
            rr["split"] = split_of(r["fiction_id"])
            rows_out.append(rr)
    print(f"rows: {len(rows_out)}  median-tie drops: {ties}")

    pos = sum(r["judgement"] for r in rows_out)
    by_split = Counter((r["split"], r["judgement"]) for r in rows_out)
    print(f"pos rate {pos / len(rows_out):.4f}")
    for s in ("train", "eval", "test"):
        print(f"  {s}: {by_split[(s, 1)] + by_split[(s, 0)]} rows — "
              f"{by_split[(s, 1)]} pos / {by_split[(s, 0)]} neg")

    with gzip.open(out_path, "wt") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    manifest = dict(
        cell="cw_royalroad_community", built_utc=__import__("datetime").datetime.utcnow().isoformat(),
        n=len(rows_out), pos=pos, pos_rate=pos / len(rows_out),
        universe="stable-hash random deep-fetch sample (stubs + top-5000-followers excluded)",
        x="deep-page ld+json description_text, single renderer (parse_deep_pages.py)",
        y="followers > within-stratum median; stratum=(primary_genre, created_year), "
          f"min {MIN_STRATUM}, small strata pooled by year; median ties dropped ({ties})",
        engagement_stat="followers (listing snapshot 2026-08-12); views/rating_pct carried as alternates",
        split_rule="sha256('rr_community::'+fiction_id)%1000 <800/<900/rest",
        splits={s: by_split[(s, 1)] + by_split[(s, 0)] for s in ("train", "eval", "test")},
        splits_pos={s: by_split[(s, 1)] for s in ("train", "eval", "test")},
        min_desc_chars=MIN_DESC_CHARS, drops=dict(drops), strata_kept=len(strata),
        confounds_declared=["fiction age (stratified)", "genre (stratified)",
                            "length/chapters accumulated (carried as covariates, NOT in y)",
                            "listing snapshot = one scrape window (no era drift within cell)"],
    )
    mpath = out_path.replace(".jsonl.gz", "_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"WROTE {out_path}\n      {mpath}")


if __name__ == "__main__":
    main()
