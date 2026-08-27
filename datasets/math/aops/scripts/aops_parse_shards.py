"""
Parse raw AoPS jsonl.gz shards → structured records.

Input:  raw/shards/*.jsonl.gz   (one topic per line, {topic_id, response})
Output: parsed/topics.jsonl.gz  (one topic per line, normalized schema)

Schema we emit per topic (style + correctness signals preserved):
{
  "topic_id":         int,
  "topic_title":      str | null,
  "category_path":    [str] | null,
  "n_posts":          int,
  "first_post": {
    "post_id":         int, "user": str, "user_id": int,
    "ts":              int,
    "canonical":       str,   # LaTeX/Markdown source
    "rendered_html":   str,
    "thanks":          int, "nothanks": int,
    "num_edits":       int, "last_edit_reason": str | null,
  },
  "solutions": [   # every NON-first post in order
    {
      "post_number":  int, "post_id": int,
      "user":         str, "user_id": int,
      "ts":           int,
      "canonical":    str,  "rendered_html": str,
      "thanks":       int,  "nothanks": int,
      "num_edits":    int,
      "char_len":     int,
      "is_accepted":  null,   # AoPS has no accepted-answer concept
    }, ...
  ],
  "n_solutions": int,
  "max_thanks":  int,
  "sum_thanks":  int,
  "earliest_ts": int,
  "latest_ts":   int,
}
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHARDS = REPO / "raw" / "shards"
PARSED = REPO / "parsed"
PARSED.mkdir(parents=True, exist_ok=True)


def parse_topic(rec: dict) -> dict | None:
    tid = rec.get("topic_id")
    if tid is None:
        return None
    resp = rec.get("response", {})
    if not isinstance(resp, dict):
        return None
    if resp.get("error_code"):
        return None
    inner = resp.get("response", {})
    if not isinstance(inner, dict):
        return None
    posts = inner.get("posts") or []
    if not posts:
        return None

    def _post(p, include_first_only_fields=False):
        return {
            "post_number":   p.get("post_number"),
            "post_id":       p.get("post_id"),
            "user":          p.get("username"),
            "user_id":       p.get("poster_id"),
            "ts":            p.get("post_time"),
            "canonical":     p.get("post_canonical") or "",
            "rendered_html": p.get("post_rendered") or "",
            "thanks":        int(p.get("thanks_received") or 0),
            "nothanks":      int(p.get("nothanks_received") or 0),
            "num_edits":     int(p.get("num_edits") or 0),
            "last_edit_reason": p.get("last_edit_reason"),
            "char_len":      len(p.get("post_canonical") or ""),
        }

    first = _post(posts[0])
    solutions = [_post(p) for p in posts[1:]]

    timestamps = [p["ts"] for p in [first] + solutions if isinstance(p["ts"], (int, float))]
    return {
        "topic_id":     tid,
        "topic_title":  inner.get("title") or inner.get("topic_title"),
        "category_path": inner.get("category_path"),
        "n_posts":      len(posts),
        "first_post":   first,
        "solutions":    solutions,
        "n_solutions":  len(solutions),
        "max_thanks":   max((s["thanks"] for s in solutions), default=0),
        "sum_thanks":   sum(s["thanks"] for s in solutions),
        "earliest_ts":  min(timestamps) if timestamps else None,
        "latest_ts":    max(timestamps) if timestamps else None,
    }


def run(shard_glob: str, out_path: Path, min_solutions: int = 0) -> None:
    shard_files = sorted(SHARDS.glob(shard_glob))
    if not shard_files:
        print(f"No shards matching {shard_glob} in {SHARDS}")
        return
    print(f"Reading {len(shard_files)} shard file(s)")

    total = 0
    kept = 0
    out_f = gzip.open(out_path, "wt", encoding="utf-8")
    try:
        for sf in shard_files:
            with gzip.open(sf, "rt", encoding="utf-8") as f:
                for line in f:
                    total += 1
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    parsed = parse_topic(rec)
                    if parsed is None:
                        continue
                    if parsed["n_solutions"] < min_solutions:
                        continue
                    out_f.write(json.dumps(parsed, ensure_ascii=False))
                    out_f.write("\n")
                    kept += 1
    finally:
        out_f.close()
    print(f"Total topics read: {total}; kept (n_solutions >= {min_solutions}): {kept}")
    print(f"Wrote → {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", default="*.jsonl.gz", help="Glob in raw/shards/")
    ap.add_argument("--out", default=str(PARSED / "topics.jsonl.gz"))
    ap.add_argument("--min-solutions", type=int, default=0,
                    help="Only keep topics with at least this many solutions")
    args = ap.parse_args()
    run(args.shard_glob, Path(args.out), args.min_solutions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
