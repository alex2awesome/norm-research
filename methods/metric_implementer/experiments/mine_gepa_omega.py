"""Mine a GEPA-DISCOVERED criterion ground set Ω from a registry metric lineage (Discovery-to-
Selection, §6.5). Harvest the distinct criterion statements the optimizer evolved across ALL versions
of a task's metrics, dedup, and emit a rubric file that `small_omega_brute_force` consumes UNCHANGED. The
criteria are what GEPA found, not hand-written — the only change vs the earlier run is the source of Ω.

  python -m methods.metric_implementer.experiments.mine_gepa_omega --task code-review --max-k 12 \
      --out /tmp/gepa_omega_code.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import re

_PREFIX = re.compile(r"^\s*(score|rate|judge|evaluate|assess)\b.*?\b(whether|how much|how well|this|the)\b",
                     re.I)
_SCALE = re.compile(r"\b\d(\.\d)?\s*=", re.I)          # cut at "1.0 =" scale anchors


def _criterion_from_body(body: str):
    """Pull the lead criterion clause out of a rubric body, dropping the Score-prefix + scale anchors."""
    # take text up to the first scale anchor ("1.0 =")
    head = _SCALE.split(body, 1)[0]
    head = head.replace("\n", " ").strip()
    head = _PREFIX.sub("", head).strip(" :.-")
    # normalize whitespace; keep it a single clause
    head = re.sub(r"\s+", " ", head)
    return head


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def harvest(task):
    pool = []
    for vf in sorted(glob.glob(f"outputs/metric_implementer/{task}/registry/metrics/*/versions/v*__prompt.json")):
        b = json.load(open(vf)).get("body") or ""
        c = _criterion_from_body(b)
        if 12 < len(c) < 200:
            pool.append(c)
    # dedup by normalized form; keep first (shortest-ish) surface form per concept-key
    seen, out = {}, []
    for c in pool:
        k = _norm(c)
        # collapse near-dupes: same first 6 normalized words
        key = " ".join(k.split()[:6])
        if key not in seen:
            seen[key] = c
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="code-review")
    ap.add_argument("--max-k", type=int, default=12)
    ap.add_argument("--out", default="/tmp/gepa_omega.txt")
    a = ap.parse_args()
    crits = harvest(a.task)
    print(f"harvested {len(crits)} distinct GEPA-evolved criteria for {a.task}:")
    for i, c in enumerate(crits):
        print(f"  [{i}] {c}")
    crits = crits[: a.max_k]
    with open(a.out, "w") as f:
        f.write(f"A high-quality {a.task} item should:\n")
        for c in crits:
            f.write(f"- {c}\n")
    print(f"\nwrote {len(crits)} criteria to {a.out}")


if __name__ == "__main__":
    main()
