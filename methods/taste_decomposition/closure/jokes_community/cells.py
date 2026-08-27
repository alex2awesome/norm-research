#!/usr/bin/env python3
"""Cell adapter for the Layer-3 articulation-closure campaign on the REDDIT-JOKES
COMMUNITY cell (r/Jokes posts, y = crowd upvote quartile).

Same contract as mathse_vote/cells.py and press_verdict/cells.py: return the cell's
Layer-1 A/V population in a REPRODUCIBLE row order together with ids / groups / y /
raw A,V blocks / texts / dense.

POPULATION (frozen 2026-08-07 by notes/2026-08-08__scaleupC_builds.md BUILD 1; never
rebuilt here).  16,000 r/Jokes posts drawn by stable hash
sha256("jokes-va-v1|" + sha1(text)[:20]) from the 383,786-row deduplicated modelling
corpus.  Row order = `outputs/va_gemma_banks_scaleupC/jokes_community_meta.json`
["item_ids"], verbatim -- every y is finite on this cell, so there is no kept-subset
filter (unlike the two math.SE cells).

GROUPING UNIT = LDA TOPIC (50 topics, `group` = t00..t49).  r/Jokes posts have no
natural container; near-duplicate reposts were removed upstream by MinHash LSH
(Jaccard >= .8).  Pos-rate is ~.50 inside every topic BY CONSTRUCTION of the labeller
(range .422-.566), so topic identity carries almost no label information: the grouped
split is a lexical-domain control, not a leakage fix.  Recorded because it is the
reason a 5-topic MONITOR is tolerable here and would not be on a container-grouped cell.

Y DEFINITION, and why it matters to TRACK B.  y = 1 for the top 25% by raw score
INSIDE its (length_bin x format x topic) stratum, 0 for the bottom 25%; the middle 50%
was dropped upstream.  Post LENGTH, coarse FORMAT and TOPIC are therefore already
partly matched out of the label by construction.  This is a fact about the instrument,
not a hint to give a proposer: the sealed fleet is NEVER told it (telling it would be a
design steer outside the freeze).  It is used only when the map is INTERPRETED -- a
length or format channel that still carries alone-AUC here is carrying residual
within-stratum variation, and its discount must be read against that.

ALIGNMENT GATE (MANDATORY, registry 2026-08-10).  `*_va_nl_oof_*.npy` are keyed in bank
item_ids order, NOT population/join order.  `oof_alignment_gate.py` asserts
    AUC(y, jokes_community_va_nl_oof_seed0.npy in THIS row order)
      == jokes_community_ledger.json nonlinear.VA["0"].auc  (= .7321856098790323)
to < 1e-9 before any readout.  `load()` runs it unless JOKES_SKIP_GATE=1.

A BLOCK.  47 Gemma-4-31B judged criteria (a01-a47), scored on the bank's
{1.0, 0.5, 0.0, NA} scale, NaN where the judge marked the criterion inapplicable
(overall NA ~.24 at smoke; five criteria are honest conditional branches above 70% NA).
Layer-1 median-imputes with a missingness indicator inside each fold, the SAME
convention `closure_core.clean_fit` uses, so there is no const-0.5 / median-impute fork
on this cell.

V BLOCK.  27 hand-coded surface features (`v_*`).  NOTE for Track B: v_char_count /
v_token_count / v_sentence_count / v_linebreak_count / v_uppercase_letter_ratio /
v_emoji_count / v_flesch_reading_ease and friends are ALREADY IN THE BANK.  A Track-B
channel that is a monotone function of one of them is a channel the articulated
instrument already owns; `readout.py` annotates every mapped channel with its strongest
rank correlation against the V block so that is visible instead of double-counted.

DENSE (T).  `datasets/humor/reddit_jokes/dense_standard/` on sk3: Llama-3.1-8B LoRA
dense-standard (r16/a32, lr 5e-5, batch 16, max_len 1024, 2 epochs, select-on-eval),
TOPIC-grouped 80/10/10 via the frozen bin-packer (train 40 topics / 12,837 rows, eval 5
topics / 1,663, test 5 topics / 1,500).  All three seeds {42, 1, 2} are on disk.
`rm_out_seed*/preds_{eval,test}.csv` carry no row key and are row-aligned with
`split/{eval,test}.csv`, which do; `fetch_dense.py` performs that join with a
judgement+group positional assertion.

T CONVENTION.  T = MEAN OVER DENSE SEEDS OF THE AUC (the programme's VA_nl convention
applied to the dense side), never the AUC of the averaged prediction.  `d["dense"]` is
the per-row seed-mean probability, used only where a single score VECTOR is required
(swap algebra, stacks, strata).

OBSERVED COVARIATES (never features in any bank).  `created_utc` -- the post's position
in the subreddit's own posting timeline, recovered by `build_covariates.py` from the raw
scrape (86.1% matched; unmatched carried as NaN, never imputed).  This is this cell's
FREEZE-ADDENDUM-4 position-in-container covariate; `era_line.py` audits it.  The raw
`score` column is deliberately NOT carried: it is what y is defined from.

SKLEARN.  The Layer-1 ledger was produced under scikit-learn 1.8.0.  Fold ASSIGNMENTS
move across sklearn releases, so Layer-1 LEVELS are not byte-reproducible under a
different release; every number this campaign quotes is recomputed under ONE version,
asserted here and recorded in every results JSON.  The alignment gate is
version-independent (it reads a stored OOF vector).

CPU only.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
TD = CLOSURE.parent
REPO = TD.parents[1]
sys.path.insert(0, str(TD))
sys.path.insert(0, str(CLOSURE))
sys.path.insert(0, str(HERE))

RESULTS = TD / "results"
BANK_DIR = REPO / "outputs" / "va_gemma_banks_scaleupC"
BANK = "jokes_community"
POP = REPO / "datasets" / "humor" / "reddit_jokes" / "va" / "population.csv.gz"
COVARIATES = HERE / "jokes_community_covariates.csv"
DENSE_CSV = HERE / "jokes_community_dense_preds.csv"
LEDGER = RESULTS / "jokes_community_ledger.json"
OOF_SEED0 = RESULTS / "jokes_community_va_nl_oof_seed0.npy"
OOF_MEAN3 = RESULTS / "jokes_community_va_nl_oof_mean3.npy"

DENSE_SEEDS_ALL = (42, 1, 2)
Y_KEY = "crowd_top_quartile"

CELL_META = {
    "jokes_community": dict(
        group_column="LDA topic (50)", item="joke",
        corpus="short jokes posted to a large public joke-sharing forum, where readers "
               "vote on what they find funny",
        construct="whether the forum's readers voted this joke into the top quarter of "
                  "its comparison pool rather than the bottom quarter -- a crowd verdict "
                  "on how funny the joke is",
        text_trunc=3600,          # corpus max is 3,508 chars; nothing is actually cut
        layer1="jokes_community_ledger.json",
    ),
}
CELLS = ["jokes_community"]

# The V block is the cell's own surface bank; these are the columns a Track-B proposer
# would most plausibly re-propose.  Named here so the discount readouts can say
# "already articulated" instead of double-counting.
V_ALREADY_ARTICULATED_SURFACE = [
    "v_char_count", "v_token_count", "v_avg_token_length", "v_type_token_ratio",
    "v_sentence_count", "v_avg_sentence_tokens", "v_final_beat_char_share",
    "v_uppercase_letter_ratio", "v_all_caps_token_count", "v_question_count",
    "v_exclamation_count", "v_ellipsis_count", "v_quote_mark_count",
    "v_linebreak_count", "v_list_marker_count", "v_url_count", "v_emoji_count",
    "v_digit_count", "v_repeated_char_run_count", "v_automated_readability_index",
    "v_flesch_reading_ease",
]


def sklearn_guard():
    import sklearn
    return sklearn.__version__


# ------------------------------------------------------------------- bank ----
def _load_bank():
    meta = json.loads((BANK_DIR / f"{BANK}_meta.json").read_text())
    Xs, Vs, ids, shard_of = [], [], [], []
    si = 0
    while (BANK_DIR / f"{BANK}_shard{si}.npz").exists():
        z = np.load(BANK_DIR / f"{BANK}_shard{si}.npz", allow_pickle=True)
        Xs.append(z["X"]); Vs.append(z["V"])
        ids += [str(s) for s in z["ids"]]
        shard_of += [si] * len(z["ids"])
        si += 1
    assert Xs, f"no shards for {BANK} under {BANK_DIR}"
    X, V = np.vstack(Xs), np.vstack(Vs)
    order = {d: i for i, d in enumerate(ids)}
    idx = np.array([order[str(d)] for d in meta["item_ids"]])
    return (meta, X[idx], V[idx],
            np.array([str(g) for g in meta["item_groups"]], dtype=object),
            np.array(shard_of)[idx],
            np.array([str(s) for s in meta["item_ids"]], dtype=object))


# ----------------------------------------------------- population covariates --
def _population(want):
    csv.field_size_limit(10 ** 9)
    texts, cov = {}, {}
    with gzip.open(POP, "rt") as fh:
        for row in csv.DictReader(fh):
            rid = row["row_id"]
            if rid not in want:
                continue
            texts[rid] = row["text"]
            cov[rid] = {"group": row["group"], "topic": row["topic"]}
    return texts, cov


def _created_utc(idl):
    if not COVARIATES.exists():
        return np.full(len(idl), np.nan)
    by = {}
    with open(COVARIATES) as fh:
        for r in csv.DictReader(fh):
            by[r["row_id"]] = r["created_utc"]
    out = []
    for i in idl:
        v = by.get(i, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out, dtype=float)


# ------------------------------------------------------------------- dense ----
def _dense(ids):
    assert DENSE_CSV.exists(), f"missing {DENSE_CSV}; run fetch_dense.py"
    with open(DENSE_CSV) as fh:
        rows = list(csv.DictReader(fh))
    seeds = [s for s in DENSE_SEEDS_ALL if f"p{s}" in rows[0] and rows[0][f"p{s}"] != ""]
    by = {r["row_id"]: r for r in rows}
    P = np.full((len(ids), len(seeds)), np.nan)
    dsplit = []
    for k, i in enumerate(ids):
        r = by.get(i)
        if r is None:
            dsplit.append("train")
            continue
        dsplit.append(r["dense_split"])
        for j, s in enumerate(seeds):
            P[k, j] = float(r[f"p{s}"])
    return P, np.array(dsplit), seeds


# -------------------------------------------------------------------- load ----
def load(cell="jokes_community", gate=None):
    assert cell == "jokes_community", cell
    meta, A, V, groups, shard, ids = _load_bank()
    yraw = np.array(meta["ys"][Y_KEY], dtype=float)
    assert np.isfinite(yraw).all(), "this cell has no undefined y rows; a filter appeared"
    y = yraw.astype(int)

    idl = [str(s) for s in ids]
    texts_by, cov_by = _population(set(idl))
    missing = [i for i in idl if i not in texts_by]
    assert not missing, f"{len(missing)} row_ids missing from {POP}"
    texts = [texts_by[i] for i in idl]

    P, dsplit, dseeds = _dense(idl)
    # dense probabilities exist only on the dense-held-out rows; the all-NaN train rows
    # are expected, so the seed-mean is taken without numpy's empty-slice warning.
    nfin = np.isfinite(P).sum(axis=1)
    dense_mean = np.where(nfin > 0,
                          np.nansum(np.where(np.isfinite(P), P, 0.0), axis=1)
                          / np.maximum(nfin, 1),
                          np.nan)

    grp = np.array([cov_by[i]["group"] for i in idl], dtype=object)
    assert (grp == groups).all(), "population group != bank item_groups"

    d = dict(tag="jokes_community", ids=idl, y=y, groups=groups, shard_of=shard,
             A=np.where(np.isfinite(A), A, np.nan), V=V,
             a_names=[str(s) for s in meta["a_names"]],
             v_names=[str(s) for s in meta["v_names"]],
             texts=texts,
             topic=np.array([cov_by[i]["topic"] for i in idl], dtype=object),
             created_utc=_created_utc(idl),
             dense=dense_mean, dense_seeds=P,
             dense_seed_ids=list(dseeds), dense_split=dsplit)
    d["meta"] = CELL_META[cell]
    led = json.loads(LEDGER.read_text())
    d["layer1"] = led
    assert len(y) == led["n"], f"n={len(y)} != ledger n={led['n']}"
    assert len(set(groups.tolist())) == led["n_groups"]

    if gate is None:
        gate = os.environ.get("JOKES_SKIP_GATE", "") != "1"
    if gate:
        from oof_alignment_gate import assert_aligned
        d["alignment_gate"] = assert_aligned(d)
    return d


LOADERS = {"jokes_community": load}


def T_by_seed(d, mask):
    """T = mean over available dense seeds of AUC on `mask`."""
    from sklearn.metrics import roc_auc_score
    y = d["y"][mask]
    per = [float(roc_auc_score(y, d["dense_seeds"][mask, j]))
           for j in range(d["dense_seeds"].shape[1])]
    return {"T": float(np.mean(per)),
            "per_seed": {str(s): p for s, p in zip(d["dense_seed_ids"], per)},
            "spread": float(max(per) - min(per)) if len(per) > 1 else 0.0,
            "n_seeds": len(per),
            "T_seed_ensemble_NOT_QUOTED": float(roc_auc_score(y, d["dense"][mask])),
            "n": int(mask.sum())}


if __name__ == "__main__":
    print("sklearn", sklearn_guard())
    d = load()
    held = np.isin(d["dense_split"], ["eval", "test"])
    print(f"jokes_community n={len(d['y'])} topics={len(set(d['groups'].tolist()))} "
          f"pos={d['y'].mean():.4f} A={d['A'].shape[1]} V={d['V'].shape[1]} "
          f"A_na={np.isnan(d['A']).mean():.4f} "
          f"held-out={held.sum()} ({len(set(d['groups'][held].tolist()))} topics) "
          f"dense_seeds={d['dense_seed_ids']} "
          f"created_utc={np.isfinite(d['created_utc']).mean():.3f} matched")
    print("alignment gate:", json.dumps(d["alignment_gate"], indent=1))
    print("T HONEST:", json.dumps(T_by_seed(d, held), indent=1))
