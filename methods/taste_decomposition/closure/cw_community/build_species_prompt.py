#!/usr/bin/env python3
"""Emit the blind full-recall SPECIES-PARTITION prompt for a round's fleet pool.

The freeze forbids embedding-threshold concept identity, so species come from a
sealed judge reading the whole pool at once.  Provenance is stripped and the pool is
hash-ordered so the judge cannot infer who wrote what.

Usage: python build_species_prompt.py --round 1 --track A  > /tmp/species_A.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

HEAD = """You are given a pool of {n} candidate scoring criteria for short stories
posted to an online creative-writing community, in a fixed arbitrary order, with all
authorship stripped. Several different proposers wrote them independently and some of
them are the SAME underlying concept expressed differently.

Partition the pool into distinct CONCEPT SPECIES. Two criteria belong to the same
species when an independent judge scoring a story against them would be measuring the
same property -- not merely when they share vocabulary or a theme. Different facets of
one broad theme are DIFFERENT species if a story could clearly satisfy one and clearly
fail the other.

Use full recall: consider every pairing; do not shortlist by surface similarity.
Do NOT read any file and do NOT search anything; work only from this message.

Output exactly one JSON object and nothing else:
{{"species": {{"<short species label>": ["<pid>", ...], ...}}}}
Every pid below must appear in exactly one species.

--- POOL ---
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--track", required=True, choices=["A", "B"])
    a = ap.parse_args()
    pool = json.loads(
        (HERE / f"round{a.round}_fleet_{a.track}.json").read_text())["proposals"]
    pool = sorted(pool, key=lambda p: hashlib.sha256(
        f"{a.round}|{p['pid']}".encode()).hexdigest())
    print(HEAD.format(n=len(pool)))
    for p in pool:
        print(f"pid: {p['pid']}\n  name: {p['name']}\n  instruction: {p['instruction']}\n")


if __name__ == "__main__":
    main()
