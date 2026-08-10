"""Assemble the final RoyalRoad-stubs dataset jsonl.

- Archive-availability filter is symmetric by construction: a fiction
  (either side) enters only with >=1 Wayback-recovered chapter.
- Greedy 1:1 matching of stubs to controls on listing metadata.
- One row per recovered chapter; raw_text + presentation-normalized text;
  stable md5 split on fiction id (0-7 train / 8 eval / 9 test).

Usage: python build_dataset.py --data-dir DIR [--min-words 200]
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rr_common import append_jsonl, display, read_jsonl, stable_split
from prioritize_controls import load_pools, metric_dist


def load_chapters(raw, side, min_words):
    """-> {fid: {chapter_id: row}} deduped keep-last, word filter applied."""
    by_fic = defaultdict(dict)
    for row in read_jsonl(os.path.join(raw, f"{side}_chapters.jsonl")):
        if row.get("n_words", 0) >= min_words:
            by_fic[row["fiction_id"]][row["chapter_id"]] = row
    return by_fic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--min-words", type=int, default=200)
    args = ap.parse_args()
    raw = os.path.join(args.data_dir, "raw")
    built = os.path.join(args.data_dir, "built")
    os.makedirs(built, exist_ok=True)

    stubs_meta, controls_meta = load_pools(args.data_dir)
    stub_ch = load_chapters(raw, "stub", args.min_words)
    ctrl_ch = load_chapters(raw, "control", args.min_words)
    stub_fids = [f for f in stub_ch if f in stubs_meta]
    ctrl_fids = [f for f in ctrl_ch if f in controls_meta]
    print(f"recovered fictions: {len(stub_fids)} stubs, {len(ctrl_fids)} controls")

    # greedy 1:1 matching, best pairs first
    pairs = []
    for sf in stub_fids:
        for cf in ctrl_fids:
            pairs.append((metric_dist(stubs_meta[sf], controls_meta[cf]), sf, cf))
    pairs.sort(key=lambda x: x[0])
    used_s, used_c, match = set(), set(), {}
    for d, sf, cf in pairs:
        if sf in used_s or cf in used_c:
            continue
        match[sf] = (cf, d)
        used_s.add(sf)
        used_c.add(cf)

    out_path = os.path.join(built, "royalroad_stubs_v1.jsonl")
    if os.path.exists(out_path):
        os.rename(out_path, out_path + ".bak")  # never delete data
    n_rows = {1: 0, 0: 0}
    n_fic = {1: 0, 0: 0}
    for sf, (cf, d) in sorted(match.items()):
        for fid, y, meta_pool, ch_pool in ((sf, 1, stubs_meta, stub_ch),
                                           (cf, 0, controls_meta, ctrl_ch)):
            meta = meta_pool[fid]
            n_fic[y] += 1
            for cid in sorted(ch_pool[fid]):
                row = ch_pool[fid][cid]
                append_jsonl(out_path, {
                    "fiction_id": fid,
                    "chapter_id": cid,
                    "chapter_rank": row["chapter_rank"],
                    "y": y,
                    "title": meta.get("title"),
                    "chapter_title": row.get("chapter_title"),
                    "tags": meta.get("tags"),
                    "labels": meta.get("labels"),
                    "followers": meta.get("followers"),
                    "rating": meta.get("rating"),
                    "pages": meta.get("pages"),
                    "views": meta.get("views"),
                    "chapters_listed": meta.get("chapters_listed"),
                    "last_update_ts": meta.get("last_update_ts"),
                    "wayback_ts": row["wayback_ts"],
                    "x_source": "wayback",
                    "match_fiction_id": cf if y == 1 else sf,
                    "match_dist": round(d, 4),
                    "n_words": row["n_words"],
                    "raw_text": row["raw_text"],
                    "text": display(row["raw_text"]),
                    "split": stable_split(fid),
                })
                n_rows[y] += 1

    print(f"matched pairs: {len(match)}")
    print(f"fictions: pos={n_fic[1]} neg={n_fic[0]}")
    print(f"chapter rows: pos={n_rows[1]} neg={n_rows[0]}")
    rows = read_jsonl(out_path)
    from collections import Counter
    print("splits:", Counter((r["split"], r["y"]) for r in rows))
    for f in ("followers", "rating", "tags"):
        nn = sum(1 for r in rows if r.get(f) not in (None, []))
        print(f"  {f} non-null: {nn}/{len(rows)} ({nn/max(len(rows),1):.1%})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
