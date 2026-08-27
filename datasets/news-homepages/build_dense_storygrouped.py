#!/usr/bin/env python3
"""STORY-GROUPED dense split for the homepage-curation cell (user-directed correction
2026-08-08; registry entry "MATHLIB AND HOMEPAGE TERMINAL VERDICTS RETRACTED").

WHY THIS BUILD EXISTS
---------------------
Scale-up wave C trained this cell OUTLET-HELD-OUT: 8 outlets, one held out for eval and
one for test. Three seeds gave eval .4322/.4590/.4393 (below chance) and test
.7361/.7429/.7534 -- the two held-out outlets disagree in SIGN. With k=2 held-out groups
that is grouping variance, not evidence of no signal, so no T was quotable. The historic
story-grouped dense arm for this cell reached T .824 (registry Journalism/curation row,
"T .824 groupsplit prov") on a different, larger population.

Fix: keep the scale-up-C population (12,998 scored rows) and re-split it by STORY BLOCK
instead of by outlet, so the design is powered and the outlet-held-out arm becomes a
labelled unpowered-transfer secondary.

GROUPING UNIT -- what "story-grouped" means here, and why
--------------------------------------------------------
The population has no story/article ID. Two candidate keys exist:
  * snapshot_id -- one outlet's homepage capture at one moment; a DATE-BLOCK key. This is
    the historic ".824 groupsplit" unit and the wave-C secondary readout's unit.
  * the headline itself -- an ARTICLE key; 11,592 distinct normalised headlines over
    12,998 rows, and 800 headlines recur across more than one snapshot (2,199 rows),
    because a story sits on the homepage across successive captures.
Neither key alone is sufficient:
  * grouping ONLY by article splits every snapshot across folds. Each item's text carries
    a CONTEXT field that is byte-identical for all rows of a snapshot, and the label is a
    WITHIN-snapshot contrast (top vs bottom half of the top-30% zone), so an article-only
    split would let the model see ~80% of each eval snapshot's labelled rows in training.
    That is a direct leak and would inflate T.
  * grouping ONLY by snapshot leaks the 800 recurring articles across folds.
  * unioning the two (connected components of the snapshot-headline bipartite graph)
    removes both leaks but CHAINS: persistent wire stories merge snapshots until the
    largest component is 3,186 rows (24.5% of the corpus) spanning three outlets, which
    forces most of four outlets into a single fold and destroys outlet balance.
So this build uses the snapshot (date-block) as the grouping unit -- the historic and
comparable choice -- and closes the article leak by DE-DUPLICATION on the train side:
any train row whose normalised headline also appears in eval or test is dropped. Removing
from train rather than from eval/test keeps the readout folds intact and representative.

Split: deterministic greedy + hill-climb bucket packer targeting 80/10/10 by row count
AND matched per-bucket pos-rate, verbatim stable_hash_bucket_map from
datasets/humor/hashtagwars/build_dense_standard.py (the same function every other
dense-standard build in this campaign uses). No seeded shuffle.

Usage (CPU only):
  python3 build_dense_storygrouped.py
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
POP_CANDIDATES = [
    HERE / "va" / "population.csv.gz",
    Path("/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/va/population.csv.gz"),
]
OUT = HERE / "va" / "dense_standard_storygrouped"

EXPECTED_N = 12998
EXPECTED_POS_RATE = 0.5006154793045083

HEADLINE_RE = re.compile(r"^HEADLINE: (.*?)(?:\n\nCONTEXT:|$)", re.S)
NONALNUM_RE = re.compile(r"[^a-z0-9 ]")


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def norm_headline(text: str) -> str:
    m = HEADLINE_RE.match(str(text))
    h = m.group(1) if m else str(text)
    return NONALNUM_RE.sub("", h.lower()).strip()


def stable_hash_bucket_map(y_by_group: dict, targets=None, lam: float = 2.5) -> dict:
    """Verbatim from datasets/humor/hashtagwars/build_dense_standard.py."""
    targets = targets or {"train": .8, "eval": .1, "test": .1}
    sizes = {g: len(v) for g, v in y_by_group.items()}
    pos = {g: sum(v) for g, v in y_by_group.items()}
    total = sum(sizes.values())
    overall_rate = sum(pos.values()) / total
    order = sorted(sizes, key=lambda g: (-sizes[g], sha1(g)))
    filled = {b: 0 for b in targets}
    filled_pos = {b: 0 for b in targets}
    bmap = {}

    def obj():
        o = sum((filled[b] / total - targets[b]) ** 2 for b in targets)
        o += lam * sum(((filled_pos[b] / max(filled[b], 1)) - overall_rate) ** 2 for b in targets)
        return o

    for g in order:
        best_b, best_o = None, None
        for b in targets:
            filled[b] += sizes[g]; filled_pos[b] += pos[g]
            o = obj()
            if best_o is None or o < best_o:
                best_o, best_b = o, b
            filled[b] -= sizes[g]; filled_pos[b] -= pos[g]
        bmap[g] = best_b
        filled[best_b] += sizes[g]; filled_pos[best_b] += pos[g]

    improved, n_iter = True, 0
    while improved and n_iter < 20:
        improved = False
        n_iter += 1
        for g in order:
            cur = bmap[g]
            best_b, best_o = cur, obj()
            for b in targets:
                if b == cur:
                    continue
                filled[cur] -= sizes[g]; filled_pos[cur] -= pos[g]
                filled[b] += sizes[g]; filled_pos[b] += pos[g]
                o = obj()
                if o < best_o - 1e-12:
                    best_b, best_o = b, o
                filled[b] -= sizes[g]; filled_pos[b] -= pos[g]
                filled[cur] += sizes[g]; filled_pos[cur] += pos[g]
            if best_b != cur:
                filled[cur] -= sizes[g]; filled_pos[cur] -= pos[g]
                filled[best_b] += sizes[g]; filled_pos[best_b] += pos[g]
                bmap[g] = best_b
                improved = True
    return bmap


def main():
    src = next((p for p in POP_CANDIDATES if p.exists()), None)
    assert src is not None, f"none of {POP_CANDIDATES} exist"
    df = pd.read_csv(src)
    n = len(df)
    pos_rate = float(df["judgement"].mean())
    print(f"population {src}: n={n} pos_rate={pos_rate!r} snapshots={df.snapshot_id.nunique()} "
          f"outlets={df.outlet.nunique()}")
    assert n == EXPECTED_N, f"n mismatch {n} != {EXPECTED_N}"
    assert abs(pos_rate - EXPECTED_POS_RATE) < 1e-12, f"pos_rate mismatch {pos_rate!r}"
    print("ASSERTION PASS: rows are exactly the scale-up-C scored A/V population "
          "(homepage_curation_ledger.json: n=12,998, pos_rate=.5006)")

    df["hn"] = df["text"].map(norm_headline)

    # a snapshot is one outlet's capture, so snapshots are outlet-pure -- verified, then the
    # packer is run WITHIN each outlet so every fold keeps the population's outlet mix
    # (a single global pack lands eval/test at ~36% guardian, because guardian's captures are
    # the smallest and the packer fills the small buckets with small groups).
    assert (df.groupby("snapshot_id")["outlet"].nunique() == 1).all(), "snapshot spans outlets"
    bmap = {}
    for outlet, sub in df.groupby("outlet"):
        y_by_group = defaultdict(list)
        for snap, y in zip(sub["snapshot_id"], sub["judgement"]):
            y_by_group[snap].append(int(y))
        bmap.update(stable_hash_bucket_map(y_by_group))
    df["split"] = df["snapshot_id"].map(bmap)

    pre_counts = df["split"].value_counts().to_dict()

    # article de-duplication: drop TRAIN rows whose headline also occurs in eval or test
    heldout_hn = set(df.loc[df["split"] != "train", "hn"])
    dup_mask = (df["split"] == "train") & (df["hn"].isin(heldout_hn))
    n_dup = int(dup_mask.sum())
    dropped = df[dup_mask]
    df = df[~dup_mask].copy()
    print(f"article de-dup: dropped {n_dup} TRAIN rows whose normalised headline also occurs "
          f"in eval/test ({dropped['hn'].nunique()} distinct headlines)")
    assert not (set(df.loc[df["split"] == "train", "hn"]) & heldout_hn), "article leak remains"
    for a, b in (("train", "eval"), ("train", "test")):
        ga = set(df.loc[df["split"] == a, "snapshot_id"])
        gb = set(df.loc[df["split"] == b, "snapshot_id"])
        assert not (ga & gb), f"snapshot leak between {a} and {b}"
    print("ASSERTION PASS: zero snapshot overlap and zero train<->heldout article overlap")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split").mkdir(exist_ok=True)
    cols = ["text", "judgement", "group", "row_id"]
    df["group"] = df["snapshot_id"]
    n_tot = len(df)
    by_split = {}
    for s in ("train", "eval", "test"):
        sub = df[df["split"] == s]
        by_split[s] = sub
        sub[cols].to_csv(OUT / "split" / f"{s}.csv", index=False)
    df[cols].to_csv(OUT / "data.csv", index=False)

    fractions = {s: len(by_split[s]) / n_tot for s in by_split}
    for s, f in fractions.items():
        tgt = 0.8 if s == "train" else 0.1
        assert abs(f - tgt) <= 2e-2, (
            f"{s} fraction {f:.4f} outside the trainer's 80/10/10 +-2pp requirement")
    print(f"ASSERTION PASS: post-dedup fractions {fractions} inside the trainer's +-2pp gate")

    outlet_comp = {s: by_split[s]["outlet"].value_counts().sort_index().to_dict() for s in by_split}
    outlet_share = {s: {k: round(v / len(by_split[s]), 4) for k, v in outlet_comp[s].items()}
                    for s in by_split}

    manifest = {
        "cell": "homepage_curation (STORY-GROUPED)",
        "why": "the registry's outlet-held-out design is unpowered (8 outlets, k=2 held out; the two "
               "held-out outlets disagree in sign: eval .4322/.4590/.4393 vs test .7361/.7429/.7534), "
               "so no T was quotable. The historic story-grouped dense arm reached .824 on a different "
               "population. User-directed correction 2026-08-08.",
        "source": str(src),
        "population": {"n_scored": EXPECTED_N, "pos_rate": EXPECTED_POS_RATE,
                       "snapshots": 1229, "outlets": 8},
        "group_column": "snapshot_id (one outlet's homepage capture at one moment = a DATE-BLOCK key; "
                        "the population carries no story/article ID), packed WITHIN outlet so every "
                        "fold keeps the population's outlet mix",
        "grouping_rationale": (
            "article-only grouping would split every snapshot across folds, and each item's text "
            "carries a snapshot-shared CONTEXT field while the label is a within-snapshot contrast -- "
            "a direct leak. Unioning snapshot+article into connected components removes both leaks but "
            "chains through persistent wire stories into a 3,186-row (24.5%) component spanning three "
            "outlets, destroying outlet balance. So: group by snapshot, and close the article leak by "
            "dropping the offending TRAIN rows."
        ),
        "article_dedup": {
            "rule": "drop any TRAIN row whose normalised headline also appears in eval or test",
            "rows_dropped_from_train": n_dup,
            "distinct_headlines": int(dropped["hn"].nunique()),
            "pre_dedup_split_counts": pre_counts,
        },
        "split_row_counts": {s: len(by_split[s]) for s in by_split},
        "split_fractions": fractions,
        "split_pos_rates": {s: float(by_split[s]["judgement"].mean()) for s in by_split},
        "split_group_counts": {s: int(by_split[s]["snapshot_id"].nunique()) for s in by_split},
        "outlet_composition_counts": outlet_comp,
        "outlet_composition_share": outlet_share,
        "y_definition": "1 = link rendered in the TOP half of the capture's top-30% zone, 0 = bottom half",
        "weak_instrument_flag": "y is spatial placement, jointly determined with layout/ad/image "
                                "constraints, not a clean editorial preference. Carried from the "
                                "wave-C manifest; every number from this cell keeps it.",
        "bank_note": "the A bank for this cell FAILS the coherent-vs-scrambled gate (.387, below "
                     "chance) -- a separate A-instrument issue, untouched by this build. Any "
                     "Delta_beyond computed against it is a statement about a news-values lexical "
                     "profile, not about an articulated-criteria reading instrument.",
        "recipe": "Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs, "
                  "gradient-checkpointing, select-on-eval, 3 seeds (42,1,2); population is balanced "
                  "(pos_rate .50) so no class weighting.",
    }
    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps({k: manifest[k] for k in
                      ("split_row_counts", "split_fractions", "split_pos_rates",
                       "split_group_counts", "outlet_composition_share", "article_dedup")}, indent=2))
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
