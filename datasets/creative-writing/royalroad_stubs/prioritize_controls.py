"""Order the control pool for Wayback recovery by match priority.

For each stub, find its K nearest controls on listing metadata
(log followers, log pages, rating, tag Jaccard); emit unique control ids
round-robin (every stub's 1st choice, then 2nd choices, ...) so that a
partial recovery run still gives every stub a candidate match.

Usage: python prioritize_controls.py --data-dir DIR [--k 8]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rr_common import read_jsonl


def metric_dist(a, b):
    def lg(x):
        return math.log10((x or 0) + 1)
    d = abs(lg(a.get("followers")) - lg(b.get("followers")))
    d += 0.5 * abs(lg(a.get("pages")) - lg(b.get("pages")))
    ra, rb = a.get("rating"), b.get("rating")
    d += 0.3 * (abs(ra - rb) if ra is not None and rb is not None else 0.5)
    ta, tb = set(a.get("tags") or []), set(b.get("tags") or [])
    jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    d += 0.5 * (1 - jac)
    return d


def load_pools(data_dir):
    raw = os.path.join(data_dir, "raw")
    stubs = {}
    for c in read_jsonl(os.path.join(raw, "stubs_listing.jsonl")):
        stubs[c["fiction_id"]] = c
    controls = {}
    for c in read_jsonl(os.path.join(raw, "controls_listing.jsonl")):
        if any("STUB" in l.upper() for l in c.get("labels", [])):
            continue
        if c["fiction_id"] in stubs:
            continue
        controls[c["fiction_id"]] = c
    return stubs, controls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    stubs, controls = load_pools(args.data_dir)
    print(f"{len(stubs)} stubs, {len(controls)} eligible controls")
    ctrl_list = list(controls.values())

    per_stub = []
    for s in stubs.values():
        ranked = sorted(ctrl_list, key=lambda c: metric_dist(s, c))[: args.k]
        per_stub.append([c["fiction_id"] for c in ranked])

    seen, order = set(), []
    for rank in range(args.k):
        for choices in per_stub:
            if rank < len(choices) and choices[rank] not in seen:
                seen.add(choices[rank])
                order.append(choices[rank])

    out = os.path.join(args.data_dir, "raw", "control_priority.txt")
    with open(out, "w") as f:
        f.write("\n".join(map(str, order)) + "\n")
    print(f"wrote {len(order)} prioritized control ids -> {out}")


if __name__ == "__main__":
    main()
