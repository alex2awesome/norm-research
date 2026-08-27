#!/usr/bin/env python3
"""Rust release-highlight curation cell.

Pool: RELEASES.md entries per version (exhaustive, each with PR links).
Curated: the version's blog announcement (prose highlights a small subset).
Stores: parsed pool entries + raw announcement HTML per version.
"""
import functools
import gzip
import json
import os
import re
import time
import urllib.request

print = functools.partial(print, flush=True)
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_highlights", "rust")
UA = {"User-Agent": "norm-research/0.1 (academic; contact alex2awesome@gmail.com)"}


def main():
    md = open(os.path.join(HERE, "RELEASES.md")).read()
    sections = re.split(r"(?=^Version \d+\.\d+)", md, flags=re.M)
    out = gzip.open(os.path.join(HERE, "pool_entries.jsonl.gz"), "wt")
    nv = ne = 0
    for s in sections:
        m = re.match(r"Version (\d+\.\d+(?:\.\d+)?) \(([\d-]+)\)", s)
        if not m:
            continue
        ver, date = m.group(1), m.group(2)
        nv += 1
        cat = None
        for line in s.splitlines():
            h = re.match(r"^([A-Z][A-Za-z /]+)\n?-*$", line.strip())
            if h and not line.startswith("-"):
                cat = line.strip()
                continue
            if line.startswith("- "):
                prs = re.findall(r"rust-lang/rust/pull/(\d+)", line)
                text = re.sub(r"\[([^\]]*)\]\[?\(?[^)\]]*\)?\]?", r"\1", line[2:]).strip()
                out.write(json.dumps({"version": ver, "date": date,
                                      "category": cat, "text": text,
                                      "pr_ids": prs}) + "\n")
                ne += 1
    out.close()
    print(f"pool: {nv} versions, {ne} entries")

    idx = urllib.request.urlopen(urllib.request.Request(
        "https://blog.rust-lang.org/releases/", headers=UA), timeout=60).read().decode(errors="ignore")
    urls = sorted(set(re.findall(r'href="(https://blog\.rust-lang\.org/\d{4}/\d{2}/\d{2}/Rust-[\d.]+/?)"', idx)))
    print(len(urls), "announcement urls")
    adir = os.path.join(HERE, "announcements")
    os.makedirs(adir, exist_ok=True)
    for u in urls:
        ver = re.search(r"Rust-([\d.]+)", u).group(1).rstrip(".")
        f = os.path.join(adir, f"{ver}.html.gz")
        if os.path.exists(f):
            continue
        try:
            h = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read().decode(errors="ignore")
        except Exception as e:
            print(ver, "ERR", str(e)[:60])
            continue
        with gzip.open(f, "wt") as fh:
            fh.write(f"<!-- {u} -->\n" + h)
        time.sleep(2)
    print("announcements saved:", len(os.listdir(adir)))


if __name__ == "__main__":
    main()
