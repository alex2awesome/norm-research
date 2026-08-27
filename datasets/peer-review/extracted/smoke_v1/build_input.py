"""Build stratified 20-review input set for peer-review smoke test v1.

Strata:
- Venue: 5 ICLR + 5 NeurIPS + 5 TMLR + 3 eLife + 2 EMNLP
- Length: 1000 <= len(review_text) <= 8000 chars
- is_meta_review: 16 False + 4 True
- Mix accept/reject decisions when distinguishable

Output: smoke_v1/input_20.jsonl
"""
import json, random, re
import pandas as pd

SRC = "/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/extracted/peer_review_unified.parquet"
OUT = "/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/extracted/smoke_v1/input_20.jsonl"
SEED = 13

VENUE_QUOTA = {"ICLR": 5, "NeurIPS": 5, "TMLR": 5, "eLife": 3, "EMNLP": 2}
META_TARGET = 4   # of 20

ACCEPT_PAT = re.compile(r"accept", re.I)
REJECT_PAT = re.compile(r"reject", re.I)


def decision_class(d):
    if not isinstance(d, str):
        return "unknown"
    if REJECT_PAT.search(d) and "withdrawn" not in d.lower():
        return "reject"
    if ACCEPT_PAT.search(d):
        return "accept"
    if "withdrawn" in d.lower():
        return "withdrawn"
    return "unknown"


def main():
    random.seed(SEED)
    df = pd.read_parquet(SRC)
    df = df.copy()
    df["len"] = df["review_text"].str.len()
    df = df[(df["len"] >= 1000) & (df["len"] <= 8000)]
    df["dclass"] = df["decision"].map(decision_class)

    picks = []
    meta_count = 0

    for venue, n in VENUE_QUOTA.items():
        sub = df[df["venue"] == venue]
        # Try to balance accept/reject and meta when possible
        sub_a = sub[sub["dclass"] == "accept"]
        sub_r = sub[sub["dclass"] == "reject"]

        # Allocate meta to NeurIPS/ICLR/TMLR/eLife where present
        n_meta = 0
        if venue == "ICLR":
            n_meta = 1
        elif venue == "NeurIPS":
            n_meta = 1
        elif venue == "eLife":
            n_meta = 0
        elif venue == "EMNLP":
            n_meta = 2  # EMNLP in length window is all meta-reviews
        # TMLR: 0 meta

        # Split remaining quota across accept/reject
        n_non_meta = n - n_meta
        n_accept = (n_non_meta + 1) // 2
        n_reject = n_non_meta - n_accept

        rng = random.Random(SEED + hash(venue) % 9999)

        sub_meta_a = sub[(sub["is_meta_review"] == True) & (sub["dclass"] == "accept")]
        sub_meta_r = sub[(sub["is_meta_review"] == True) & (sub["dclass"] == "reject")]
        sub_meta_other = sub[sub["is_meta_review"] == True]
        sub_norm_a = sub[(sub["is_meta_review"] == False) & (sub["dclass"] == "accept")]
        sub_norm_r = sub[(sub["is_meta_review"] == False) & (sub["dclass"] == "reject")]

        chosen = []

        # Meta-review picks: try accept then reject
        meta_rows = []
        n_meta_a = (n_meta + 1) // 2
        n_meta_r = n_meta - n_meta_a
        if n_meta_a > 0 and len(sub_meta_a) > 0:
            meta_rows += sub_meta_a.sample(min(n_meta_a, len(sub_meta_a)), random_state=rng.randint(0, 99999)).to_dict("records")
        if n_meta_r > 0 and len(sub_meta_r) > 0:
            meta_rows += sub_meta_r.sample(min(n_meta_r, len(sub_meta_r)), random_state=rng.randint(0, 99999)).to_dict("records")
        # Fill remaining meta with anything
        if len(meta_rows) < n_meta and len(sub_meta_other) > 0:
            already_ids = {r["review_id"] for r in meta_rows}
            extras = sub_meta_other[~sub_meta_other["review_id"].isin(already_ids)]
            need = n_meta - len(meta_rows)
            if len(extras) > 0:
                meta_rows += extras.sample(min(need, len(extras)), random_state=rng.randint(0, 99999)).to_dict("records")
        chosen += meta_rows

        # Non-meta picks
        norm_rows = []
        if n_accept > 0 and len(sub_norm_a) > 0:
            norm_rows += sub_norm_a.sample(min(n_accept, len(sub_norm_a)), random_state=rng.randint(0, 99999)).to_dict("records")
        if n_reject > 0 and len(sub_norm_r) > 0:
            norm_rows += sub_norm_r.sample(min(n_reject, len(sub_norm_r)), random_state=rng.randint(0, 99999)).to_dict("records")
        # Fill remaining with any non-meta
        if len(norm_rows) < n_non_meta:
            already_ids = {r["review_id"] for r in norm_rows + meta_rows}
            extras = sub[(sub["is_meta_review"] == False) & (~sub["review_id"].isin(already_ids))]
            need = n_non_meta - len(norm_rows)
            if len(extras) > 0:
                norm_rows += extras.sample(min(need, len(extras)), random_state=rng.randint(0, 99999)).to_dict("records")
        chosen += norm_rows

        # If still short, just sample any from venue
        if len(chosen) < n:
            already_ids = {r["review_id"] for r in chosen}
            extras = sub[~sub["review_id"].isin(already_ids)]
            need = n - len(chosen)
            if len(extras) > 0:
                chosen += extras.sample(min(need, len(extras)), random_state=rng.randint(0, 99999)).to_dict("records")

        picks += chosen[:n]
        print(f"{venue}: picked {len(chosen[:n])} (meta={sum(1 for r in chosen[:n] if r['is_meta_review'])})")

    # Drop to 20 exactly if over
    picks = picks[:20]
    actual_meta = sum(1 for r in picks if r["is_meta_review"])
    print(f"Total picks: {len(picks)} meta={actual_meta}")

    # Write JSONL
    keep_fields = ["paper_id", "venue", "year", "title", "abstract", "decision",
                   "review_id", "review_text", "review_score", "confidence",
                   "recommendation", "is_meta_review"]
    with open(OUT, "w") as f:
        for r in picks:
            rec = {}
            for k in keep_fields:
                v = r.get(k)
                # Convert numpy/pandas types
                if hasattr(v, "item"):
                    try:
                        v = v.item()
                    except Exception:
                        v = str(v)
                if pd.isna(v) if not isinstance(v, (str, bool, list, dict)) else False:
                    v = None
                rec[k] = v
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
