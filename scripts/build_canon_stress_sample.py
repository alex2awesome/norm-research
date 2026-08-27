"""Assemble a SEEDED stress-test sample for calibrating the leaf-canonicalization
prompt.

Not random: deliberately loaded with cases that break a naive rewrite, so each
calibration round can check all three failure modes at once.

  CONVERGE  groups -- same concept, many phrasings. A good prompt rewrites every
            member to near-identical text.  (show-don't-tell, omit-needless-
            words, unit-testing, active-voice, point-of-view, reproducibility)
  TRAP      groups -- lexically near-identical, conceptually DISTINCT. A good
            prompt keeps them clearly separated. (Sanderson's 1st/2nd/3rd Law,
            Extract Class/Method/Superclass/Subclass, Replace Type Code variants)
  JARGON    terse domain terms -- a good prompt restates faithfully and does NOT
            invent a definition. (Lazy Class, TooManyFields, ...)
  RANDOM    a plain slice for general coverage.

Output: outputs/analyses/canon_stress_sample.jsonl
        {task, idx, key, name, seed_group, kind}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
LEAF = OUT / "_sk3_leaf_input.jsonl"

# (seed_group, kind, task-filter or None, regex, cap)
SEEDS = [
    ("show-not-tell", "CONVERGE", "creative-writing",
     r"show.{0,12}(tell|telling)|showing.{0,8}(vs|versus|not).{0,8}tell|show,?\s*don", 11),
    ("omit-needless-words", "CONVERGE", "creative-writing",
     r"omit.{0,8}needless|needless words|cut.{0,12}(unnecessary|excess)|trim.{0,10}word", 8),
    ("active-voice", "CONVERGE", "creative-writing", r"active voice|passive voice", 7),
    ("point-of-view", "CONVERGE", "creative-writing",
     r"^point of view|^perspective|element: point of view", 8),
    ("unit-testing", "CONVERGE", "code-review",
     r"unit test|write tests|test coverage|automated test", 11),
    ("reproducibility", "CONVERGE", "peer-review", r"reproducib|replicab", 9),
    ("sanderson-laws", "TRAP", "creative-writing", r"sanderson", 8),
    ("extract-refactor", "TRAP", "code-review", r"^extract (class|method|superclass|subclass)", 8),
    ("replace-type-code", "TRAP", "code-review", r"replace type code", 4),
    ("nist-functions", "TRAP", "code-review", r"^function: (recover|detect|identify|protect|respond)", 5),
    ("jargon-terms", "JARGON", "code-review",
     r"^(lazy class|toomanyfields|substitute algorithm|crates|speculative generality|shotgun surgery)", 6),
]


def main():
    rows = [json.loads(l) for l in LEAF.open()]
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    rng = np.random.default_rng(7)

    picked = []
    seen_keys = set()
    for group, kind, task, pat, cap in SEEDS:
        p = re.compile(pat, re.I)
        pool = by_task.get(task, [])
        hits = [r for r in pool if p.search(r["name"])]
        rng.shuffle(hits)
        n = 0
        for r in hits:
            if r["key"] in seen_keys:
                continue
            seen_keys.add(r["key"])
            picked.append({**{k: r[k] for k in ("task", "idx", "key", "name")},
                           "seed_group": group, "kind": kind})
            n += 1
            if n >= cap:
                break
        print(f"  {group:<22} {kind:<9} {n} leaves")

    # random slice across all tasks
    allrows = [r for r in rows if r["key"] not in seen_keys]
    ridx = rng.choice(len(allrows), size=28, replace=False)
    for i in ridx:
        r = allrows[i]
        picked.append({**{k: r[k] for k in ("task", "idx", "key", "name")},
                       "seed_group": "random", "kind": "RANDOM"})
    print(f"  {'random':<22} {'RANDOM':<9} 28 leaves")

    with (OUT / "canon_stress_sample.jsonl").open("w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(picked)} leaves -> {OUT/'canon_stress_sample.jsonl'}")


if __name__ == "__main__":
    main()
