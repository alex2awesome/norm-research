#!/usr/bin/env python3
"""Cell adapter for the Layer-3 articulation-closure campaign on the
AoPS CURATION cell (y = 1 iff the forum solution takes substantially the SAME
APPROACH as the editorial / AoPS-wiki solution to that competition problem).

Contract identical to mathse_accepted/cells.py and press_verdict/cells.py: return
the cell's Layer-1 A/V population in a REPRODUCIBLE row order together with
ids / groups / y / raw A,V blocks / texts / dense.

POPULATION (frozen, never rebuilt here).  `outputs/va_gemma_banks_scaleupC/
aops_curation_meta.json` IS the population definition: 5,202 forum solutions over
606 competition problems.  `ys["same_approach"]` is finite on every row, so the
kept-subset filter is a no-op and the row order is `meta["item_ids"]` VERBATIM.
`datasets/math/aops/va/population.csv.gz` is written in exactly that order (the
adapter asserts it) and carries the statement, the solution body, y, and the
dense probability.

THE STRUCTURAL FACT THAT SETS THIS CELL APART FROM EVERY OTHER CLOSURE CELL.
The A/V population IS the dense arm's held-out set.  `build_va_population.py`
defined the population as `split_full/eval` union `split_full/test` of the REUSED
`runs/aops_same_approach_dense_llama8b` arm, precisely so that T would be
same-rows by construction at zero GPU cost.  Consequences, all recorded before
any readout:
  * every one of the 5,202 rows is dense-held-out, so HONEST = the FULL
    population = the master ledger's E rows (n_E 5,202, 606 groups, pos .6734);
  * the prereg's "MONITOR must live inside the dense-held-out rows" is satisfied
    by ANY cut, so the split reverts to the prereg's base 80/20 rule rather than
    the .50-within-held-out rule the math.SE cells needed;
  * the mining slice M = FIT+MINE in full (dense scores are honest everywhere),
    so there is no M-vs-FIT+MINE distinction to draw on this cell.

DENSE (T).  ONE dense arm, ONE seed, reused not retrained.  `dense_prob` sits in
the population file itself, so there is no fetch/join step and no fetch_dense.py
on this cell.  Because there is exactly one dense score vector, the
"T = mean of per-seed AUCs" vs "AUC of the seed-mean" distinction that governs
the math.SE cells COLLAPSES here: the two are the same number, .7806334165434924
pooled (eval .7739 / test .7879).  Every Delta on this cell is therefore on one
convention and the sibling cells' "never difference T-based against
ensemble-based" caveat does not arise.

A BLOCK.  44 Gemma-4-31B judged criteria on the bank's 0.0 / 0.5 / 1.0 + NA
scale, NaN where the judge marked the criterion inapplicable (overall NA rate
.2285 -- the applicability gate firing on a corpus where many criteria simply do
not bear on a three-line solution).  Mined criteria are scored 0-10 by the same
judge, the programme standard on every closure cell (the math.SE, press and CW
banks are 0/0.5/1 too); the mismatch is a monotone rescaling absorbed by the
StandardScaler / GBM and is recorded, not corrected.

V BLOCK.  24 deterministic lint features (`v_*`) computed on the SOLUTION BODY
ONLY -- `build_va_population.py` strips the problem statement first, on purpose,
so statement length and statement LaTeX cannot leak into a "solution style"
feature.  Named in V_ALREADY_ARTICULATED_SURFACE so the Track-B discount readouts
say "already articulated" instead of double-counting.

ITEM VIEW.  The bank's own view, reproduced byte-for-byte
(`datasets/va_gemma_banks/score_scaleupC_banks.py::build_aops_curation`):

    PROBLEM: <statement truncated to 1500 chars>

    FORUM SOLUTION:
    <body, deterministic HEAD-3000 + TAIL-2000 middle omission at 5000 chars>

Note the truncation applies to the BODY ONLY and the statement prefix is added
after -- a whole-view HEAD/TAIL cut (what the sibling cells' score_gemma_maps.py
does) would show the mined criteria a different document from the bank's.
`item_view()` here is the single definition; build_splits.py writes it into
`aops_curation_population.csv` and score_gemma_maps.py consumes it unchanged.

OBSERVED COVARIATES (never features in any bank; FREEZE ADDENDUM 4 position
line).  Recovered by `build_position_covariates.py` from
`datasets/math/aops/forum_solutions.parquet` -- the raw AoPS crawl behind this
corpus, which carries the TRUE ordinal: `post_number` (position of the post in
its AoPS topic thread), `post_time`, `topic_id`, `num_edits`,
`topic_num_views`, `poster_id`, and the contest year implied by the problem key.
`thanks_received` is carried too but is a NEIGHBOURING OUTCOME (crowd approval),
never a feature and never in the joint position model.

ALIGNMENT GATE (MANDATORY, registry 2026-08-10).  `*_va_nl_oof_*.npy` are keyed
in bank item_ids order.  `oof_alignment_gate.py` asserts
    AUC(y, aops_curation_va_nl_oof_seed0.npy in THIS row order)
      == aops_curation_ledger.json nonlinear.VA["0"].auc  (= .7689588189522913)
to < 1e-9 before any readout.  `load()` runs it unless AOPS_SKIP_GATE=1.

SKLEARN.  The Layer-1 ledger was produced under scikit-learn 1.8.0.  GroupKFold
fold assignments move across releases, so Layer-1 LEVELS are not byte-reproducible
under a different release and this campaign's own round-0 anchor is the baseline
the curve is measured from.  The alignment gate is version-free (it reads a
stored vector).

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
BANK = "aops_curation"
POP = REPO / "datasets" / "math" / "aops" / "va" / "population.csv.gz"
COV_CSV = HERE / "aops_curation_position_covariates.csv"
LEDGER = RESULTS / "aops_curation_ledger.json"
OOF_SEED0 = RESULTS / "aops_curation_va_nl_oof_seed0.npy"
OOF_MEAN3 = RESULTS / "aops_curation_va_nl_oof_mean3.npy"

Y_KEY = "same_approach"

# ---- the bank's item view, reproduced exactly (see module docstring) ----------
TRUNC_SRC, TRUNC_HEAD, TRUNC_TAIL = 5000, 3000, 2000
TRUNC_MARK = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"
STATEMENT_CHARS = 1500


def body_trunc(s: str) -> str:
    s = (s or "")
    return s if len(s) <= TRUNC_SRC else s[:TRUNC_HEAD] + TRUNC_MARK + s[-TRUNC_TAIL:]


def item_view(statement: str, body: str) -> str:
    return (f"PROBLEM: {(statement or '')[:STATEMENT_CHARS]}\n\n"
            f"FORUM SOLUTION:\n{body_trunc(body)}")


CELL_META = {
    "aops_curation": dict(
        group_column="problem", item="forum solution",
        corpus="solutions posted to the Art of Problem Solving competition forums, each "
               "shown with the competition problem it solves",
        construct="whether the posted solution takes substantially the SAME SOLUTION "
                  "APPROACH as the canonical editorial / wiki write-up of that problem -- "
                  "a match against a reference the model never sees, not a quality "
                  "preference and not a vote",
        text_trunc=4000,
        layer1="aops_curation_ledger.json",
    ),
}
CELLS = ["aops_curation"]

# V columns a Track-B proposer would most plausibly re-propose.  Named so the
# discount readouts can say "already articulated" instead of double-counting.
V_ALREADY_ARTICULATED_SURFACE_HINT = ("length", "latex", "math", "line", "word", "char",
                                      "sentence", "token", "para", "list", "digit",
                                      "numeral", "upper", "punct")

# Observed covariates carried from build_position_covariates.py.  NEVER features.
COV_NUM = ("post_number", "sol_rank", "n_sols_group", "position_pct", "n_posts_topic",
           "thread_age_days", "post_year", "contest_year", "years_after_contest",
           "num_edits", "topic_num_views", "poster_n_posts", "thanks_received",
           "nothanks_received", "match_sim")
COV_STR = ("topic_id", "poster_id", "username", "contest", "match_kind")


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


# ----------------------------------------------------- population + dense ----
def _population(want):
    csv.field_size_limit(10 ** 9)
    rec = {}
    with gzip.open(POP, "rt") as fh:
        for row in csv.DictReader(fh):
            rid = row["row_id"]
            if rid in want:
                rec[rid] = row
    return rec


def _covariates(idl):
    """Observed AoPS thread covariates.  Absent until build_position_covariates.py
    has run; the loader degrades to all-NaN so round 0's bank state can be built
    before the position line exists."""
    if not COV_CSV.exists():
        return None
    with open(COV_CSV, newline="") as fh:
        by = {r["row_id"]: r for r in csv.DictReader(fh)}
    missing = [i for i in idl if i not in by]
    assert not missing, f"{len(missing)} row_ids missing from {COV_CSV}"
    out = {}
    for c in COV_NUM:
        vals = []
        for i in idl:
            v = by[i].get(c, "")
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[c] = np.array(vals, dtype=float)
    for c in COV_STR:
        out[c] = np.array([str(by[i].get(c, "")) for i in idl], dtype=object)
    return out


# -------------------------------------------------------------------- load ----
def load(cell="aops_curation", gate=None):
    assert cell == "aops_curation", cell
    meta, A, V, groups, shard, ids = _load_bank()
    yraw = np.array(meta["ys"][Y_KEY], dtype=float)
    keep = np.isfinite(yraw)
    assert keep.all(), (
        "same_approach must be finite on all 5,202 rows; a non-trivial keep mask "
        "means the wrong y / wrong bank meta was loaded")
    A, V, groups, shard, ids = A[keep], V[keep], groups[keep], shard[keep], ids[keep]
    y = yraw[keep].astype(int)

    idl = [str(s) for s in ids]
    rec = _population(set(idl))
    missing = [i for i in idl if i not in rec]
    assert not missing, f"{len(missing)} row_ids missing from {POP}"

    statements = [rec[i]["statement"] for i in idl]
    bodies = [rec[i]["body"] for i in idl]
    texts = [item_view(s, b) for s, b in zip(statements, bodies)]

    ypop = np.array([int(rec[i]["judgement"]) for i in idl])
    assert (ypop == y).all(), "population judgement != bank ys['same_approach']"
    gpop = np.array([str(rec[i]["problem"]) for i in idl], dtype=object)
    assert (gpop == groups).all(), "population problem != bank item_groups"

    dense = np.array([float(rec[i]["dense_prob"]) for i in idl])
    dsplit = np.array([str(rec[i]["dense_split"]) for i in idl], dtype=object)
    assert np.isfinite(dense).all(), "dense_prob has holes"
    assert set(dsplit.tolist()) <= {"eval", "test"}, (
        "every population row must be dense-HELD-OUT on this cell; a 'train' row means "
        "the population was rebuilt off the frozen split")

    d = dict(tag="aops_curation", ids=idl, y=y, groups=groups, shard_of=shard,
             A=np.where(np.isfinite(A), A, np.nan), V=V,
             a_names=[str(s) for s in meta["a_names"]],
             v_names=[str(s) for s in meta["v_names"]],
             texts=texts, statements=statements, bodies=bodies,
             dense=dense, dense_seeds=dense.reshape(-1, 1),
             dense_seed_ids=["reused"], dense_split=dsplit)
    cov = _covariates(idl)
    d["cov"] = cov
    d["has_covariates"] = cov is not None
    d["meta"] = CELL_META[cell]
    led = json.loads(LEDGER.read_text())
    d["layer1"] = led
    assert len(y) == led["n"], f"n={len(y)} != ledger n={led['n']}"
    assert len(set(groups.tolist())) == led["n_groups"]

    if gate is None:
        gate = os.environ.get("AOPS_SKIP_GATE", "") != "1"
    if gate:
        from oof_alignment_gate import assert_aligned
        d["alignment_gate"] = assert_aligned(d)
    return d


LOADERS = {"aops_curation": load}


def T_by_seed(d, mask):
    """T on `mask`.  ONE dense seed on this cell, so the mean-over-seeds
    convention and the seed-ensemble convention are the same number; both are
    reported so the field names match the sibling cells."""
    from sklearn.metrics import roc_auc_score
    y = d["y"][mask]
    per = [float(roc_auc_score(y, d["dense_seeds"][mask, j]))
           for j in range(d["dense_seeds"].shape[1])]
    return {"T": float(np.mean(per)),
            "per_seed": {str(s): p for s, p in zip(d["dense_seed_ids"], per)},
            "spread": float(max(per) - min(per)) if len(per) > 1 else 0.0,
            "n_seeds": len(per),
            "T_seed_ensemble_NOT_QUOTED": float(roc_auc_score(y, d["dense"][mask])),
            "single_seed_note": "one reused dense arm: T and the ensemble figure are "
                                "identical on this cell, so no cross-convention caveat "
                                "applies to any Delta reported here",
            "n": int(mask.sum())}


def already_articulated(v_names):
    return [n for n in v_names
            if any(h in n.lower() for h in V_ALREADY_ARTICULATED_SURFACE_HINT)]


if __name__ == "__main__":
    print("sklearn", sklearn_guard())
    d = load()
    print(f"aops_curation n={len(d['y'])} problems={len(set(d['groups'].tolist()))} "
          f"pos={d['y'].mean():.4f} A={d['A'].shape[1]} V={d['V'].shape[1]} "
          f"A_na={np.isnan(d['A']).mean():.4f} "
          f"dense_split={ {s: int((d['dense_split'] == s).sum()) for s in ('eval', 'test')} } "
          f"covariates={d['has_covariates']}")
    print("item view chars: max", max(len(t) for t in d["texts"]),
          "median", int(np.median([len(t) for t in d["texts"]])))
    print("V surface columns already articulated:", already_articulated(d["v_names"]))
    print("alignment gate:", json.dumps(d["alignment_gate"], indent=1))
    print("T HONEST (= full population = E):",
          json.dumps(T_by_seed(d, np.ones(len(d["y"]), bool)), indent=1))
