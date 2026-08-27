#!/usr/bin/env python3
"""V8 -- build the N&C CO-SIGNING cell (the field's VOTE/REVEALED column).

y-definition (rationale in notes/2026-08-08__v8_cosigning_build.md S1):
  Regulations.gov has no upvote. The only act by which the public endorses
  ANOTHER person's comment is ADOPTION: an organisation (or individual) writes
  a comment and N other people submit that same text under their own names.

  y_cosign_count(c) = the number of documents in c's OWN DOCKET whose
  normalised canonical_text (lowercase, whitespace-collapsed) is identical to
  c's -- i.e. how many people signed on. Recomputed from the current
  public_submission_all_text.csv over all 6,803,623 documents in the 1,814
  dockets our population touches (recount_cosign.py; 9,521/9,524 located).

  PRIMARY binarisation  y_cosign   = 1{count >= 2}  ("did anyone else sign on")
  SECONDARY             y_campaign = 1{count >= 10} ("organised-campaign scale")
  SENSITIVITY           y_nearby   = 1{shipped MinHash near-dup family >= 2}

TWO REJECTED CHANNELS, both measured, not assumed:
  (1) the agency-populated `Duplicate Comments` metadata field -- DEAD: >1 on
      6,514 of 11,698,149 comment rows (0.06%). Agencies do not fill it in.
  (2) the shipped dedup mappers from
      regulations-demo/.../scripts/minhash_comment_deduping.py, as the PRIMARY
      count. Ground-truthed on AMS-NOP-17-0031 (2026-08-08): the docket holds
      47,108 documents including one 8,258-member byte-identical text family,
      but the authoritative `*__dedup_mapper.csv.gz` covers only 30,661 of them
      and its largest cluster is 1,879 -- partial coverage that varies by
      directory, so any threshold on it is contaminated by which directories
      the dedup job finished. Its stale `.csv` sibling is worse: pre-exact-text
      -expansion, and MinHash transitive chaining gives it an 8,501-member
      cluster for a text with exactly ONE copy. The gz channel is retained
      ONLY as the y_nearby sensitivity arm, with its partial coverage stated.
  The exact channel is uniform-coverage across every row, which is what makes
  it the primary; it undercounts adoptions that personalise the template
  (exact-pos 343 vs gz-near-pos 386, overlap 154) -- hence the sensitivity arm.

Unit of analysis = ONE ROW PER (docket, normalised text). Co-signers of one
template are duplicate texts carrying an identical y; keeping several would be
duplicate rows. Collisions are de-duplicated by stable hash (lowest sha1
doc_id wins) and counted.

Reuse: the population is the EXACT 9,521-row A/V-scored N&C universe
(nc_scores_shard0..4.npz + nc_scores_unmatched.npz, 198-rubric pre-GEPA Gemma
bank). No new judging -- this script attaches a new y to frozen scores.

Outputs (datasets/notice-and-comment/cosigning/):
  cosign_population.jsonl   one row per unit: doc_id, docket, agency, text,
                            cosign_count, y_cosign, y_campaign, y_nearby,
                            split, docket_n_docs, docket_n_unique_texts
  cosign_build_stats.json   funnel + distribution + split audit
  dense_llama/cosign/{data.csv,split/*.csv}   dense-standard bundle
                            (docket-grouped 80/10/10 stable hash, pos-rate
                            matched)
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NC = REPO / "datasets" / "notice-and-comment" / "v4"
HERE = Path(__file__).resolve().parent
JOIN = HERE / "scored_cosign_join_v2.json"          # gz near-dup (sensitivity)
EXACT = HERE / "cosign_counts_exact.json"           # exact co-sign count (primary)

sys.path.insert(0, str(NC))
sys.path.insert(0, str(REPO / "datasets" / "patents"))
from aggregate_nc_multiy import (MATCHED_SHARDS, UNMATCHED_NPZ, SAMPLE_JSONL,  # noqa: E402
                                 UNMATCHED_JSONL, load_shard_scores, load_jsonl_texts)
from build_dense_standard_claimfell import stable_hash_bucket_map  # noqa: E402

CAMPAIGN_THRESHOLD = 10


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def main():
    funnel = {}
    join = json.loads(JOIN.read_text())
    exact = json.loads(EXACT.read_text())
    funnel["gz_join_rows"] = len(join)
    funnel["exact_count_rows"] = len(exact)

    # ---- the frozen A/V-scored universe (identical construction to
    # nc_layer1_stack.NCData: anchors already dropped by load_shard_scores,
    # matched-vs-unmatched overlap resolved in favour of matched) ----------
    X_m, docket_m, agency_m, _ = load_shard_scores(MATCHED_SHARDS)
    X_u, docket_u, agency_u, _ = load_shard_scores([str(UNMATCHED_NPZ)])
    overlap = set(X_m) & set(X_u)
    for d in overlap:
        del X_u[d], docket_u[d], agency_u[d]
    text_m = load_jsonl_texts(SAMPLE_JSONL)
    text_u = load_jsonl_texts(UNMATCHED_JSONL)

    universe = {}
    for d in X_m:
        universe[d] = (docket_m[d], agency_m[d], text_m.get(d, ""), 1)
    for d in X_u:
        universe[d] = (docket_u[d], agency_u[d], text_u.get(d, ""), 0)
    funnel["scored_universe"] = len(universe)
    funnel["matched_unmatched_overlap_dropped"] = len(overlap)

    # ---- admit rows with an exact co-sign count -------------------------
    rows, drop = [], Counter()
    for did, (dk, ag, text, is_matched) in universe.items():
        e = exact.get(did)
        if e is None:
            drop["not_located_in_all_text"] += 1
            continue
        if not text or len(text) < 50:
            drop["no_text"] += 1
            continue
        j = join.get(did, {})
        gz = int(j.get("cluster_size", -1)) if j.get("mapper_src") == "gz" else -1
        rows.append({
            "doc_id": did, "docket": dk, "agency": ag, "text": text,
            "cosign_count": int(e["cosign_count"]),
            "gz_near_size": gz,
            "is_matched": is_matched,
            "posted": j.get("posted", ""), "org": j.get("org", ""),
            "category": j.get("category", ""),
            "docket_n_docs": int(e["docket_n_docs"]),
            "docket_n_unique_texts": int(e["docket_n_unique_texts"]),
        })
    funnel["dropped"] = dict(drop)
    funnel["admitted_documents"] = len(rows)

    # ---- one row per (docket, normalised text) --------------------------
    # co-signers of one template are duplicate rows with an identical y.
    # Two of OUR rows collide only if the sample drew two signers of the same
    # template; keep the stable-hash-lowest doc_id.
    by_text = defaultdict(list)
    for r in rows:
        by_text[(r["docket"], sha1(" ".join(r["text"].lower().split())))].append(r)
    units, n_collisions = [], 0
    for _, rs in by_text.items():
        if len(rs) > 1:
            n_collisions += len(rs) - 1
        units.append(sorted(rs, key=lambda r: sha1(r["doc_id"]))[0])
    funnel["intra_template_collisions_dropped"] = n_collisions
    funnel["units"] = len(units)

    for u in units:
        u["y_cosign"] = int(u["cosign_count"] >= 2)
        u["y_campaign"] = int(u["cosign_count"] >= CAMPAIGN_THRESHOLD)
        u["y_nearby"] = int(u["gz_near_size"] >= 2) if u["gz_near_size"] >= 1 else -1

    # ---- docket-grouped, pos-rate-matched 80/10/10 stable-hash split -----
    y_by_docket = defaultdict(list)
    for u in units:
        y_by_docket[u["docket"]].append(u["y_cosign"])
    bmap = stable_hash_bucket_map(dict(y_by_docket))
    for u in units:
        u["split"] = bmap[u["docket"]]

    units.sort(key=lambda u: (u["docket"], u["doc_id"]))
    with open(HERE / "cosign_population.jsonl", "w") as fh:
        for u in units:
            fh.write(json.dumps(u) + "\n")

    # ---- audit -----------------------------------------------------------
    def stat(sel):
        n = len(sel)
        return {"n": n,
                "pos_rate_cosign": (sum(u["y_cosign"] for u in sel) / n) if n else None,
                "pos_rate_campaign": (sum(u["y_campaign"] for u in sel) / n) if n else None,
                "n_dockets": len({u["docket"] for u in sel})}

    audit = {"overall": stat(units),
             "by_split": {s: stat([u for u in units if u["split"] == s])
                          for s in ("train", "eval", "test")}}
    size_dist = Counter(u["cosign_count"] for u in units)
    audit["cosign_count_dist_head"] = {str(k): size_dist[k] for k in sorted(size_dist)[:20]}
    audit["cosign_count_quantiles_pos"] = sorted(
        u["cosign_count"] for u in units if u["y_cosign"])[::max(1, sum(u["y_cosign"] for u in units) // 20)]
    audit["y_nearby"] = {
        "n_defined": sum(1 for u in units if u["y_nearby"] >= 0),
        "n_pos": sum(1 for u in units if u["y_nearby"] == 1),
        "agreement_with_exact": dict(Counter(
            f"exact{u['y_cosign']}_near{u['y_nearby']}" for u in units if u["y_nearby"] >= 0))}
    audit["by_agency"] = {a: stat([u for u in units if u["agency"] == a])
                          for a in sorted({u["agency"] for u in units})}
    dk_pos = Counter(u["docket"] for u in units if u["y_cosign"])
    dk_all = Counter(u["docket"] for u in units)
    mixed = [d for d in dk_all if 0 < dk_pos.get(d, 0) < dk_all[d]]
    audit["mixed_dockets"] = {"n_dockets": len(mixed),
                              "n_rows": sum(dk_all[d] for d in mixed),
                              "n_pos": sum(dk_pos[d] for d in mixed)}
    # cross-y contrast with the verdict column (responded = is_matched)
    both = [(u["is_matched"], u["y_cosign"]) for u in units]
    tab = Counter(both)
    audit["cosign_x_responded"] = {f"resp{a}_cos{b}": v for (a, b), v in sorted(tab.items())}
    funnel["audit"] = audit

    # ---- dense-standard bundle ------------------------------------------
    dd = HERE / "dense_llama" / "cosign"
    (dd / "split").mkdir(parents=True, exist_ok=True)
    cols = ["text", "judgement", "group"]
    allrows = [{"text": u["text"], "judgement": u["y_cosign"], "group": u["docket"]}
               for u in units]
    with open(dd / "data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(allrows)
    for s, fn in (("train", "train.csv"), ("eval", "eval.csv"), ("test", "test.csv")):
        sub = [{"text": u["text"], "judgement": u["y_cosign"], "group": u["docket"]}
               for u in units if u["split"] == s]
        with open(dd / "split" / fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(sub)
    funnel["dense_bundle"] = str(dd)

    json.dump(funnel, open(HERE / "cosign_build_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in funnel.items() if k != "audit"}, indent=1))
    print("overall", audit["overall"])
    for s in ("train", "eval", "test"):
        print(" ", s, audit["by_split"][s])
    print("mixed dockets", audit["mixed_dockets"])
    print("cosign x responded", audit["cosign_x_responded"])


if __name__ == "__main__":
    main()
