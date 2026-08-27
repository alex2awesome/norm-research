"""Post-build sanity checks for contest_corpus.jsonl.

Prints: per-contest/cycle counts, tier distributions, text coverage and
length stats, normalization invariants (no curly quotes / nbsp in `text`),
and flags suspiciously short texts (possible teasers/extraction failures).
"""
import json
import os
from collections import Counter

from fetch_util import STAGING

PATH = os.path.join(STAGING, "contest_corpus.jsonl")


def main():
    rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]
    print(f"total rows: {len(rows)}")
    print(f"rows with text: {sum(1 for r in rows if r['text_available'])}")

    print("\nper contest:")
    for c, n in Counter(r["contest"] for r in rows).items():
        nt = sum(1 for r in rows if r["contest"] == c and r["text_available"])
        print(f"  {c:18s} rows={n:4d} with_text={nt:4d}")

    print("\nnormalization invariants on `text`:")
    bad = [r for r in rows if r["text"] and any(ch in r["text"] for ch in "‘’“” ")]
    print(f"  rows with curly quotes/nbsp remaining: {len(bad)}")
    for r in bad[:5]:
        print("   !!", r["contest"], r["cycle"], r["title"])

    print("\nshortest texts (check for teasers/extraction failures):")
    with_text = sorted((r for r in rows if r["text"]), key=lambda r: len(r["text"]))
    for r in with_text[:10]:
        print(f"  {len(r['text']):5d} ch | {r['contest']} {r['cycle']} {r['rank_tier']:18s} | {r['title']!r:50s} | {r['url']}")

    print("\nmissing fields:")
    for f in ["title", "author", "rank_tier", "cycle"]:
        n = sum(1 for r in rows if not r.get(f))
        if n:
            print(f"  {f}: {n} rows missing")
            for r in [x for x in rows if not x.get(f)][:5]:
                print("   ", r["contest"], r["cycle"], r["url"])

    print("\nduplicate (contest, cycle, title, author):")
    dups = [k for k, n in Counter((r["contest"], r["cycle"], (r["title"] or "").lower(), r["author"].lower()) for r in rows).items() if n > 1]
    print(f"  {len(dups)} duplicated keys")
    for k in dups[:10]:
        print("   ", k)


if __name__ == "__main__":
    main()
