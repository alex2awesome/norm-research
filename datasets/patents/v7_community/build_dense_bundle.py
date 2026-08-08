#!/usr/bin/env python3
"""V7 patents forward-citation cell: dense-standard bundle (data.csv + split/).

Same convention as `datasets/stackoverflow-votes/build_dense_bundle.py` and
`datasets/patents/build_dense_standard_claimfell.py`: columns text / judgement /
group / row_id, plus split/{train,eval,test}.csv, plus manifest.json. The splits
are NOT recomputed -- they are carried over verbatim from population.csv.gz, so
the dense arm, the V matrix and the A bank all sit on identical rows (FREEZE
CHANGE 2: T is a same-rows readout by construction).

`text` is title + abstract + claim 1 and NOTHING else. Asserted below.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEF_POP = ("/lfs/skampere3/0/alexspan/norm-research/datasets/patents/"
           "v7_community/population.csv.gz")
DEF_OUT = ("/lfs/skampere3/0/alexspan/norm-research/datasets/patents/"
           "v7_community/dense_standard")

BANNED_SUBSTR = ["examiner", "art unit", "assignee", "application number",
                 "attorney docket"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default=DEF_POP)
    ap.add_argument("--out", default=DEF_OUT)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "split").mkdir(parents=True, exist_ok=True)

    pop = pd.read_csv(a.pop)
    pop = pop[pop.y_fwd5.notna()].copy()
    d = pd.DataFrame({"text": pop.text.astype(str),
                      "judgement": pop.y_fwd5.astype(int),
                      "group": pop.family_group.astype(str),
                      "row_id": pop.row_id.astype(str)})

    # the claim-fell killer was metadata reaching the model input; assert it cannot
    low = d.text.str.lower()
    for b in BANNED_SUBSTR:
        hits = float(low.str.contains(b, regex=False).mean())
        assert hits < 0.01, f"{b!r} appears in {hits:.3%} of dense text"
    assert not d.text.str.contains(r"\b(?:19|20)\d{2}-\d{2}-\d{2}").any(), \
        "ISO date leaked into dense text"

    d.to_csv(out / "data.csv", index=False)
    sizes = {}
    for s in ["train", "eval", "test"]:
        sub = d[pop.split.values == s]
        sub.to_csv(out / "split" / f"{s}.csv", index=False)
        sizes[s] = {"n": int(len(sub)), "pos_rate": float(sub.judgement.mean()),
                    "n_groups": int(sub.group.nunique())}
        print(f"  {s:6s} n={len(sub):6d} pos={sub.judgement.mean():.4f} "
              f"groups={sub.group.nunique()}")

    straddle = d.assign(sp=pop.split.values).groupby("group").sp.nunique()
    assert int(straddle.max()) == 1, "a family group straddles splits"

    man = {"cell": "patents_fwdcites", "source_population": a.pop,
           "n": int(len(d)), "pos_rate": float(d.judgement.mean()),
           "n_groups": int(d.group.nunique()), "splits": sizes,
           "group_column": "family_group (near-duplicate / continuation cluster)",
           "text_fields": ["title", "abstract", "claim1"],
           "splits_carried_from_population": True,
           "recipe": ("dense standard: Llama-3.1-8B LoRA r16/a32 lr5e-5 batch16 "
                      "len1024 2ep --gradient-checkpointing --selection_split eval"),
           "metadata_assertions_passed": BANNED_SUBSTR + ["ISO dates"]}
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print("wrote", out / "data.csv", "and", out / "split")


if __name__ == "__main__":
    main()
