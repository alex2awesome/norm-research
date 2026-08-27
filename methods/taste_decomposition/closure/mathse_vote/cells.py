#!/usr/bin/env python3
"""Cell adapter for the Layer-3 articulation-closure campaign on the
math.StackExchange VOTE-SCORE cell (within-question median split of the raw
answer score, un-binarised v2 rebuild).

Same contract as press_verdict/cells.py and peer_revealed/cells.py: return the
cell's Layer-1 A/V population in a REPRODUCIBLE row order together with
ids / groups / y / raw A,V blocks / texts / dense.

POPULATION (frozen, never rebuilt here).  The bank meta
`outputs/va_gemma_banks_scaleupC/mathse_multiy_meta.json` IS the population
definition: 13,001 scored answers over 4,960 questions, of which the 11,629 with
a FINITE `ys["vote_score"]` form this cell (rows whose score ties their
question's median are undefined and dropped -- see scaleupC_layer1._mathse).
Row order = `meta["item_ids"]` filtered by that finite mask, verbatim.  This is
the "kept-subset order" named in the 2026-08-10 registry landmine entry.

SHARED MATRIX.  The A/V matrix is shared with the math.SE ACCEPTED-VERDICT cell
(one Gemma pass, two y's).  The two y's are never merged and never differenced.

ALIGNMENT GATE (MANDATORY, registry 2026-08-10).  `*_va_nl_oof_*.npy` are keyed
in bank item_ids order (here: the kept-subset order above), NOT population/join
order.  `oof_alignment_gate.py` asserts
    AUC(y, mathse_vote_score_va_nl_oof_seed0.npy in THIS row order)
      == mathse_vote_score_ledger.json nonlinear.VA["0"].auc   (= .624849045069194)
to < 1e-9 before any readout.  `load()` runs it unless MATHSE_SKIP_GATE=1.

A BLOCK.  32 Gemma-4-31B judged criteria, NaN where the judge marked the
criterion inapplicable (na_rate .2396 overall).  Unlike press, this cell's
Layer-1 pipeline (`layer1_gemma_cells` family-1 linear leg) uses
SimpleImputer(median, add_indicator) inside each fold, i.e. it median-imputes --
the SAME convention `closure_core.clean_fit` uses.  There is therefore no
const-0.5 / median-impute fork on this cell: the closure standard IS the Layer-1
convention.  Recorded so the absence of a sensitivity arm is deliberate.

V BLOCK.  28 hand-coded lint features (`v_*`), already in the shard npz.
NOTE for Track B: v_log_len / v_word_count / v_latex_density / v_n_display_math
are ALREADY IN THE BANK.  Any Track-B channel that is a monotone function of
those is a channel the articulated instrument already owns, and the discount
readouts must say so rather than double-counting it.

DENSE (T).  `datasets/math-stackexchange/v2_va/dense_standard_mathse_vote_score/`
on sk3: Llama-3.1-8B LoRA dense-standard, QUESTION-grouped stable-hash 80/10/10,
select-on-eval (no deviation).  `rm_out_seed*/preds_{eval,test}.csv` carry no row
key and are row-aligned with `split/{eval,test}.csv`, which do carry `row_id`;
`fetch_dense.py` performs that join with a judgement+group positional assertion
and writes `mathse_vote_dense_preds.csv`.  Seed 42 is on disk; seeds 1 and 2 are
produced by the scaleupC dense chain (GATE -- see notes/2026-08-10__closure_mathse_vote.md).

T CONVENTION.  T = MEAN OVER DENSE SEEDS OF THE AUC (the programme's VA_nl
convention applied to the dense side), never the AUC of the averaged prediction.
`d["dense"]` is the per-row seed-mean probability, used only where a single score
VECTOR is required (swap algebra, stacks, strata).

OBSERVED COVARIATES (never features in any bank; FREEZE ADDENDUM 4 position line):
`answer_position` (0-based order of the answer under its question),
`n_answers`, `answer_year`, `primary_tag`, plus the raw `score` and `accepted`
flags that define / neighbour y.

SKLEARN.  The Layer-1 ledger was produced under scikit-learn 1.8.0.  Fold
ASSIGNMENTS move across sklearn releases, so Layer-1 LEVELS are not byte-
reproducible under a different release; every number this campaign quotes is
recomputed under ONE version, asserted here and recorded in every results JSON.
The alignment gate is version-independent (it reads a stored OOF vector).

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
BANK = "mathse_multiy"
POP = REPO / "datasets" / "math-stackexchange" / "v2_va" / "population.csv.gz"
DENSE_CSV = HERE / "mathse_vote_dense_preds.csv"
LEDGER = RESULTS / "mathse_vote_score_ledger.json"
OOF_SEED0 = RESULTS / "mathse_vote_score_va_nl_oof_seed0.npy"
OOF_MEAN3 = RESULTS / "mathse_vote_score_va_nl_oof_mean3.npy"

DENSE_SEEDS_ALL = (42, 1, 2)
Y_KEY = "vote_score"

CELL_META = {
    "mathse_vote": dict(
        group_column="question_id", item="answer",
        corpus="answers to mathematics questions on math.StackExchange, each shown "
               "with the (truncated) question it answers",
        construct="whether the site's voters scored this answer ABOVE the median "
                  "answer on its own question -- a within-question crowd preference "
                  "over answers to the SAME question",
        text_trunc=3500,
        layer1="mathse_vote_score_ledger.json",
    ),
}
CELLS = ["mathse_vote"]

# The V block is the cell's own lint bank; these columns are the ones a Track-B
# proposer would most plausibly re-propose.  Named here so the discount readouts
# can say "already articulated" instead of double-counting.
V_ALREADY_ARTICULATED_SURFACE = [
    "v_log_len", "v_word_count", "v_sentence_count", "v_avg_sentence_words",
    "v_n_display_math", "v_inline_math_delims", "v_latex_cmd_count",
    "v_latex_density", "v_numeral_density", "v_linebreak_count",
    "v_paragraph_count", "v_list_marker_count", "v_type_token_ratio",
    "v_uppercase_letter_ratio", "v_alpha_share",
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
COV_COLS = ("question_id", "answer_id", "score", "accepted", "answer_position",
            "n_answers", "answer_year", "primary_tag")


def _population(want):
    csv.field_size_limit(10 ** 9)
    texts, cov = {}, {}
    with gzip.open(POP, "rt") as fh:
        for row in csv.DictReader(fh):
            rid = row["row_id"]
            if rid not in want:
                continue
            texts[rid] = row["text"]
            cov[rid] = {c: row[c] for c in COV_COLS}
    return texts, cov


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
def load(cell="mathse_vote", gate=None):
    assert cell == "mathse_vote", cell
    meta, A, V, groups, shard, ids = _load_bank()
    yraw = np.array(meta["ys"][Y_KEY], dtype=float)
    keep = np.isfinite(yraw)
    A, V, groups, shard, ids = A[keep], V[keep], groups[keep], shard[keep], ids[keep]
    y = yraw[keep].astype(int)

    idl = [str(s) for s in ids]
    texts_by, cov_by = _population(set(idl))
    missing = [i for i in idl if i not in texts_by]
    assert not missing, f"{len(missing)} row_ids missing from {POP}"
    texts = [texts_by[i] for i in idl]

    def _col(name, cast=float, fill=np.nan):
        out = []
        for i in idl:
            v = cov_by[i][name]
            try:
                out.append(cast(v))
            except (TypeError, ValueError):
                out.append(fill)
        return np.array(out, dtype=object if cast is str else float)

    P, dsplit, dseeds = _dense(idl)

    # question_id from the population must reproduce the bank's grouping column
    qid = np.array([cov_by[i]["question_id"] for i in idl], dtype=object)
    assert (qid == groups).all(), "population question_id != bank item_groups"

    d = dict(tag="mathse_vote", ids=idl, y=y, groups=groups, shard_of=shard,
             A=np.where(np.isfinite(A), A, np.nan), V=V,
             a_names=[str(s) for s in meta["a_names"]],
             v_names=[str(s) for s in meta["v_names"]],
             texts=texts,
             answer_position=_col("answer_position"), n_answers=_col("n_answers"),
             answer_year=_col("answer_year"), score_raw=_col("score"),
             accepted=_col("accepted"),
             primary_tag=np.array([cov_by[i]["primary_tag"] for i in idl], dtype=object),
             answer_id=np.array([cov_by[i]["answer_id"] for i in idl], dtype=object),
             dense=np.nanmean(P, axis=1), dense_seeds=P,
             dense_seed_ids=list(dseeds), dense_split=dsplit)
    d["meta"] = CELL_META[cell]
    led = json.loads(LEDGER.read_text())
    d["layer1"] = led
    assert len(y) == led["n"], f"n={len(y)} != ledger n={led['n']}"
    assert len(set(groups.tolist())) == led["n_groups"]

    if gate is None:
        gate = os.environ.get("MATHSE_SKIP_GATE", "") != "1"
    if gate:
        from oof_alignment_gate import assert_aligned
        d["alignment_gate"] = assert_aligned(d)
    return d


LOADERS = {"mathse_vote": load}


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
    print(f"mathse_vote n={len(d['y'])} questions={len(set(d['groups'].tolist()))} "
          f"pos={d['y'].mean():.4f} A={d['A'].shape[1]} V={d['V'].shape[1]} "
          f"A_na={np.isnan(d['A']).mean():.4f} "
          f"held-out={held.sum()} ({len(set(d['groups'][held].tolist()))} questions) "
          f"dense_seeds={d['dense_seed_ids']}")
    print("alignment gate:", json.dumps(d["alignment_gate"], indent=1))
    print("T HONEST:", json.dumps(T_by_seed(d, held), indent=1))
