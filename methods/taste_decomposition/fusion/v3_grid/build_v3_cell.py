#!/usr/bin/env python3
"""Generic V3 "criteria-in-prompt" dataset builder for the 13-cell grid.

Production recipe frozen by notes/2026-08-09__v3_audit_fable.md §5 and
notes/2026-08-07__vat_fusion_directions.md "Direction 3":

  * frozen Llama-3.1-8B LoRA recipe downstream (r16/a32, lr 5e-5, bs16,
    max_len 1024, 2 epochs, seed 42) -- this script only builds data.
  * k = 20 criteria (default; --k overrides), NAMES ONLY (no definitions, no
    importance weights, no score re-rendering).
  * ranking = TRAINING-FOLD-ONLY grouped permutation importance:
    GroupKFold(3) within the cell's own dense TRAIN split, frozen HistGB
    (max_leaf_nodes=31, lr=.06, max_iter=400, early stopping) seed 0,
    permutation_importance(scoring="roc_auc", n_repeats=5) on the inner
    held-out fold, mean over the 3 folds.  Importance NEVER sees eval/test.
  * >=95%-modal columns are dropped from the top-k candidate pool BEFORE
    extending k (audit §5: "prefer dropping >=95%-modal columns").
  * block rendered as "<name>: <score>" lines under a "VA metrics:" header.
  * SHORT-text cells APPEND the block after the text (caption format,
    build_direction23_data.py / build_v3_audit_data.py);
    LONG-text cells PREPEND it (build_cw_mirror_data.py) because the trainer
    and scorer truncate the RIGHT side at max_len=1024 and an appended block
    is silently deleted on exactly the long documents.
  * criterion scores are label-blind Gemma-4-31B judge outputs -> safe on all
    splits; y NEVER appears in a prompt; splits come from the cell's OWN
    dense-standard split CSVs so text and splits are the cell's own.

Emits:
  methods/taste_decomposition/fusion/dense_data/v3grid_<slug>/
      data.csv  split/{train,eval,test}.csv  manifest.json
  (and v3grid_<slug>_k<J>/ for each --also-k J)

and records skipped cells in
  methods/taste_decomposition/fusion/v3_grid/build_blockers.json

CPU only.  No GPU.

Usage:
  python3 build_v3_cell.py --cell nc_agree [--k 20] [--also-k 10]
  python3 build_v3_cell.py --list
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent            # fusion/v3_grid
FUS = HERE.parent                                  # fusion
TD = FUS.parent                                    # taste_decomposition
REPO = TD.parents[1]                               # repo root
CLOSURE = TD / "closure"
OUT_ROOT = FUS / "dense_data"
BLOCKERS_PATH = HERE / "build_blockers.json"

DEFAULT_K = 20
MODAL_DROP = 0.95
MAX_LEN = 1024
SPLITS = ("train", "eval", "test")

# Local mirror of the meta-llama/Llama-3.1-8B tokenizer (gated on the hub; the
# repo's own copy lives on sk3 under $HF_HOME).  Override with V3_TOKENIZER.
TOKENIZER = os.environ.get(
    "V3_TOKENIZER", str(Path.home() / ".cache" / "llama31_8b_tokenizer"))

K_CAVEAT = (
    "k=20 is the audit's §5 recommendation but it is UNCONFIRMED: the +.0113 "
    "k20-over-k10 lift was established on the cap_crowd SANDBOX cell and the "
    "designated CONFIRM cell (cap_finalist) came back -.0276 [-.058,+.003] "
    "P(>0)=.043 when its k20 checkpoint was finally harvested (2026-08-08). "
    "k in {10,20} is a wash with cell-specific sign -- do NOT read a "
    "V3-vs-bank gap smaller than ~.03 as real."
)


# ------------------------------------------------------------------ utils ---
def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(v) -> str:
    """Score rendering, byte-identical to the caption/CW builders."""
    v = float(v)
    if np.isnan(v):
        return "NA"
    if v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _fit_gbm(seed=0):
    """layer1_gemma_cells.GRID[1] verbatim (leaves 31, lr .06, 400 iters)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_leaf_nodes=31, learning_rate=0.06, max_iter=400,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=seed)


# ======================================================== cell adapters =====
# Every adapter returns the same contract:
#   split_dir        Path -- holds split/{train,eval,test}.csv (the cell's OWN
#                    dense-standard splits; `text` is what the dense model read)
#   group_column     canonical grouping unit name (for the manifest)
#   long_text        True -> PREPEND the block; False -> APPEND (caption format)
#   selection_split  "eval"/"test" -- matches the cell's ORIGINAL dense chain
#   orig_chain       provenance string for the manifest
#   bank_source      provenance string for the manifest
#   bank_keys        list[str], one join key per bank row
#   bank_VA          (n_bank, n_cols) float array, NaNs preserved
#   bank_names       list[str] column names, len == n_cols
#   bank_y           (n_bank,) int array (join assertion)
#   bank_ids         list[str] stable bank row ids (written as `did`)
#   split_key_fn     callable(DataFrame) -> list[str] join keys, one per row

_CACHE = {}


def _maps_batch1():
    if "b1" not in _CACHE:
        _CACHE["b1"] = load_module(CLOSURE / "maps_batch1" / "cells.py", "v3g_cells_b1")
    return _CACHE["b1"]


def _maps_hw_si():
    if "hw" not in _CACHE:
        _CACHE["hw"] = load_module(CLOSURE / "maps_hw_si" / "cells.py", "v3g_cells_hw")
    return _CACHE["hw"]


def _va_from(d):
    V = np.asarray(d["V"], dtype=float)
    A = np.asarray(d["A"], dtype=float)
    names = [str(s) for s in d["v_names"]] + [str(s) for s in d["a_names"]]
    VA = np.column_stack([V, A]) if V.shape[1] and A.shape[1] else (V if V.shape[1] else A)
    assert VA.shape[1] == len(names), (VA.shape, len(names))
    return VA, names


# ---- peer cells: dense split `group` IS the bank id (ntitle) ---------------
def _peer_cell(slug, dl_name):
    d = _maps_batch1().load(slug)
    VA, names = _va_from(d)
    ids = [str(x) for x in d["ids"]]
    return dict(
        slug=slug,
        split_dir=REPO / "datasets/peer-review/vat_3y/dense_llama" / dl_name,
        group_column="ntitle",
        long_text=True,
        selection_split="test",
        orig_chain="methods/dense/run_dense_standard.sh (no --selection_split "
                   "-> trainer default 'test'), seed 42, out dir rm_out/",
        bank_source="closure/maps_batch1/cells.py load('%s') -- Layer-1 V+A "
                    "matrix (the bank vat_stack_%s.json's `layer1_bank` arm "
                    "used)" % (slug, slug),
        bank_keys=ids, bank_VA=VA, bank_names=names,
        bank_y=np.asarray(d["y"], dtype=int), bank_ids=ids,
        split_key_fn=lambda df: [str(g) for g in df["group"]],
        join_desc="dense split `group` column == bank id == ntitle",
    )


# ---- N&C cells: join on (text, docket) ------------------------------------
def _nc_cell(slug, dl_name):
    d = _maps_batch1().load(slug)
    VA, names = _va_from(d)
    ids = [str(x) for x in d["ids"]]
    keys = ["%s\x1f%s" % (str(g), t) for g, t in zip(d["groups"], d["texts"])]
    return dict(
        slug=slug,
        split_dir=REPO / "datasets/notice-and-comment/v4/dense_llama" / dl_name,
        group_column="docket",
        long_text=True,
        selection_split="test",
        orig_chain="methods/dense/run_dense_standard.sh (no --selection_split "
                   "-> trainer default 'test'), seed 42, out dir rm_out/",
        bank_source="closure/maps_batch1/cells.py load('%s') -- Layer-1 V+A "
                    "matrix (the bank vat_stack_%s.json's `layer1_bank` arm "
                    "used)" % (slug, slug),
        bank_keys=keys, bank_VA=VA, bank_names=names,
        bank_y=np.asarray(d["y"], dtype=int), bank_ids=ids,
        split_key_fn=lambda df: ["%s\x1f%s" % (str(g), str(t))
                                 for g, t in zip(df["group"], df["text"])],
        join_desc="(docket, comment text) composite key; both sides built from "
                  "the same nc_vat_sample.jsonl text field",
    )


# ---- nc_responded: TERMINAL bank = round0 V+A + rounds 1-5 mined criteria --
def _nc_responded():
    ncr = CLOSURE / "nc_responded"
    sys.path.insert(0, str(ncr))
    NCL = load_module(ncr / "nc_closure_lib.py", "v3g_nc_closure_lib")
    NCR = load_module(ncr / "readout.py", "v3g_nc_readout")
    pop = NCL.load_population()
    V = np.asarray(pop["V"], dtype=float)
    A = np.asarray(pop["A"], dtype=float)
    names = [str(s) for s in pop["v_names"]]
    a_names = [str(s) for s in pop["a_names"]]
    Xr, names_r = NCR.load_round_scores([1, 2, 3, 4, 5])
    blocks, allnames = [V, A], list(names) + list(a_names)
    if Xr is not None:
        blocks.append(np.asarray(Xr, dtype=float))
        # load_round_scores labels mined columns "r{round}:{blind_id}:{name}";
        # the prompt gets the human name only (names-only recipe).
        allnames += [str(s).split(":", 2)[-1] for s in names_r]
    VA = np.column_stack(blocks)
    assert VA.shape[1] == len(allnames), (VA.shape, len(allnames))
    ids = [str(x) for x in pop["doc_id"]]
    texts = [str(t) for t in pop["texts"]]
    dockets = [str(g) for g in pop["docket"]]
    keys = ["%s\x1f%s" % (g, t) for g, t in zip(dockets, texts)]
    return dict(
        slug="nc_responded",
        split_dir=REPO / "datasets/notice-and-comment/v4/dense_llama/responded",
        group_column="docket",
        long_text=True,
        selection_split="test",
        orig_chain="methods/dense/run_dense_standard.sh (no --selection_split "
                   "-> trainer default 'test'), seed 42, out dir rm_out/",
        bank_source="closure/nc_responded: nc_closure_lib.load_population() V+A "
                    "(round0) + readout.load_round_scores([1..5]) mined "
                    "criteria = the TERMINAL round5 bank (vat_stack_"
                    "nc_responded.json's `round5` arm)",
        bank_keys=keys, bank_VA=VA, bank_names=allnames,
        bank_y=np.asarray(pop["y"], dtype=int), bank_ids=ids,
        split_key_fn=lambda df: ["%s\x1f%s" % (str(g), str(t))
                                 for g, t in zip(df["group"], df["text"])],
        join_desc="(docket, comment text) composite key",
    )


# ---- peer_verdict: TERMINAL bank = round0 V+A + rounds 1-4 mined ----------
def _peer_verdict():
    sys.path.insert(0, str(CLOSURE))
    SR4 = load_module(CLOSURE / "stage4_readout.py", "v3g_stage4_readout")
    S4R4 = load_module(CLOSURE / "stage4_round4.py", "v3g_stage4_round4")
    pop, split, dsplit, XA1, XB1, a1_ids, b1_ids, summary = SR4.build_blocks()
    XA2, _, a2_ids, _, _ = S4R4.load_round_blocks(2)
    XA3, _, a3_ids, _, _ = S4R4.load_round_blocks(3)
    XA4, _, a4_ids, _, _ = S4R4.load_round_blocks(4)

    def _named(r, ids):
        """blind_id -> the criterion's human name (names-only recipe)."""
        routing = json.loads((CLOSURE / f"round{r}_routing_final.json").read_text())
        by = {x["blind_id"]: x.get("name", x["blind_id"]) for x in routing["final"]}
        return [str(by.get(i, i)) for i in ids]

    V, A = np.asarray(pop["V"], dtype=float), np.asarray(pop["A"], dtype=float)
    names = [str(s) for s in pop["v_names"]] + [str(s) for s in pop["a_names"]]
    mined = [(XA1, _named(1, a1_ids)), (XA2, _named(2, a2_ids)),
             (XA3, _named(3, a3_ids)), (XA4, _named(4, a4_ids))]
    blocks = [V, A]
    for X, nm in mined:
        blocks.append(np.asarray(X, dtype=float))
        names += [str(s) for s in nm]
    VA = np.column_stack(blocks)
    assert VA.shape[1] == len(names), (VA.shape, len(names))
    ids = [str(s) for s in pop["ntitle"]]
    return dict(
        slug="peer_verdict",
        split_dir=REPO / "datasets/peer-review/vat_3y/dense_llama/verdict",
        group_column="ntitle",
        long_text=True,
        selection_split="test",
        orig_chain="methods/dense/run_dense_standard.sh (no --selection_split "
                   "-> trainer default 'test'), seed 42, out dir rm_out/",
        bank_source="closure/stage4_readout.build_blocks() V+A (round0) + "
                    "stage4_round4.load_round_blocks(2..4) + round1 XA1 = the "
                    "TERMINAL round4 bank (vat_stack_peer_verdict.json `round4`)",
        bank_keys=ids, bank_VA=VA, bank_names=names,
        bank_y=np.asarray(pop["y"], dtype=int), bank_ids=ids,
        split_key_fn=lambda df: [str(g) for g in df["group"]],
        join_desc="dense split `group` column == bank id == ntitle",
    )


# ---- hashtagwars_verdict (short text) -------------------------------------
def _hashtagwars_verdict():
    d = _maps_hw_si().load("hashtagwars_verdict")
    VA, names = _va_from(d)
    ids = [str(x) for x in d["ids"]]
    texts = [str(t) for t in d["texts"]]
    return dict(
        slug="hashtagwars_verdict",
        split_dir=REPO / "datasets/humor/hashtagwars/dense_standard",
        group_column="hashtag contest",
        long_text=False,
        selection_split="eval",
        orig_chain="methods/dense/run_dense_standard_v4.sh "
                   "(--selection_split eval), seeds 42/1/2, rm_out_seed*/",
        bank_source="closure/maps_hw_si/cells.py load('hashtagwars_verdict') -- "
                    "outputs/va_gemma_banks/hashtagwars_{meta.json,shard*.npz} "
                    "Layer-1 V+A matrix",
        bank_keys=ids, bank_VA=VA, bank_names=names,
        bank_y=np.asarray(d["y"], dtype=int), bank_ids=ids,
        split_key_fn=lambda df: [str(r) for r in df["row_id"]],
        join_desc="split CSV `row_id` column == bank item_ids (sha1); the bank's "
                  "`texts` are the SAME context block the Gemma judge and the "
                  "dense reader saw",
        _fallback_key_fn=lambda df: [str(t) for t in df["text"]],
        _fallback_bank_keys=texts,
        _fallback_desc="tweet text",
    )


# ---- scale-up wave C cells: split CSVs carry row_id == bank item_ids -------
def _scaleupC(slug, bank_name, y_key, split_dir, group_column, long_text):
    SC = _CACHE.get("scaleupC")
    if SC is None:
        SC = _CACHE["scaleupC"] = load_module(TD / "scaleupC_layer1.py", "v3g_scaleupC")
    meta, A, V, groups, shard, ids = SC.load_scaleupC_bank(bank_name)
    y = np.array(meta["ys"][y_key], dtype=float)
    keep = np.isfinite(y)
    A, V, ids = np.asarray(A, float)[keep], np.asarray(V, float)[keep], np.asarray(ids)[keep]
    y = y[keep].astype(int)
    names = [str(s) for s in meta["v_names"]] + [str(s) for s in meta["a_names"]]
    VA = np.column_stack([V, A])
    assert VA.shape[1] == len(names), (VA.shape, len(names))
    sids = [str(x) for x in ids]
    return dict(
        slug=slug,
        split_dir=split_dir,
        group_column=group_column,
        long_text=long_text,
        selection_split="eval",
        orig_chain="methods/dense/run_dense_standard_scaleupC.sh "
                   "(--selection_split eval), seed 42 (seed 1/2 incomplete), "
                   "rm_out_seed42/",
        bank_source="outputs/va_gemma_banks_scaleupC/%s_{meta.json,shard*.npz} "
                    "via scaleupC_layer1.load_scaleupC_bank (y=%s)" % (bank_name, y_key),
        bank_keys=sids, bank_VA=VA, bank_names=names,
        bank_y=y, bank_ids=sids,
        split_key_fn=lambda df: [str(r) for r in df["row_id"]],
        join_desc="split CSV `row_id` column == bank meta item_ids",
    )


# ---- press_verdict: round-0 (terminal) A cache + V-88 npz ------------------
def _press_verdict():
    zA = np.load(TD / "results" / "press_verdict_pr_A_k3_scores_CACHE.npz",
                 allow_pickle=True)
    zV = np.load(CLOSURE / "press_verdict" / "press_v88.npz", allow_pickle=True)
    a_ids = [str(x) for x in zA["ids"]]
    v_ids = [str(x) for x in zV["ids"]]
    assert a_ids == v_ids, "press_verdict: A-cache and V-88 row order differ"
    # Layer-1 fill: inapplicable criteria render as the neutral 0.5 level
    A = np.where(zA["applicable"], zA["levels"].astype(float), 0.5)
    V = np.asarray(zV["V"], dtype=float)
    names = [str(s) for s in zV["names"]] + [str(s) for s in zA["names"]]
    VA = np.column_stack([V, A])
    assert VA.shape[1] == len(names), (VA.shape, len(names))
    return dict(
        slug="press_verdict",
        split_dir=REPO / "datasets/press-releases/dense_standard_k3",
        group_column="company",
        long_text=True,
        selection_split="eval",
        orig_chain="methods/dense/run_dense_standard_scaleupA.sh "
                   "(--selection_split eval, --max_length 1024, no "
                   "class_weight_auto), seeds 42/1/2, rm_out_seed*/",
        bank_source="TERMINAL (round-0; no round>=1 mining state exists): "
                    "results/press_verdict_pr_A_k3_scores_CACHE.npz A(40, "
                    "applicable-gated with 0.5 fill) + closure/press_verdict/"
                    "press_v88.npz V(88)",
        bank_keys=a_ids, bank_VA=VA, bank_names=names,
        bank_y=np.asarray(zA["y"], dtype=int), bank_ids=a_ids,
        split_key_fn=lambda df: [str(r) for r in df["row_id"]],
        join_desc="split CSV `row_id` (cast to str) == A-cache `ids` == V-88 "
                  "`ids`; the split `group` column is the same company key as "
                  "the A-cache `comp`",
    )


MATHSE_DIR = REPO / "datasets/math/stackexchange/v2_va"

ADAPTERS = {
    "nc_agree":                lambda: _nc_cell("nc_agree", "agree"),
    "peer_curation":           lambda: _peer_cell("peer_curation", "curation"),
    "hashtagwars_verdict":     _hashtagwars_verdict,
    "nc_outcome":              lambda: _nc_cell("nc_outcome", "outcome"),
    "jokes_community":         lambda: _scaleupC(
        "jokes_community", "jokes_community", "crowd_top_quartile",
        REPO / "datasets/humor/reddit_jokes/dense_standard", "LDA topic (50)", False),
    "mathse_accepted_verdict": lambda: _scaleupC(
        "mathse_accepted_verdict", "mathse_multiy", "accepted_verdict",
        MATHSE_DIR / "dense_standard_mathse_accepted_verdict", "question_id", True),
    "mathse_vote_score":       lambda: _scaleupC(
        "mathse_vote_score", "mathse_multiy", "vote_score",
        MATHSE_DIR / "dense_standard_mathse_vote_score", "question_id", True),
    "aops_curation":           None,   # BLOCKED, see BLOCKED_CELLS
    "press_verdict":           _press_verdict,
    "peer_verdict":            _peer_verdict,
    "nc_responded":            _nc_responded,
    "peer_revealed":           lambda: _peer_cell("peer_revealed", "revealed"),
    "code_v3":                 None,   # BLOCKED, see BLOCKED_CELLS
}

# Cells that cannot be built and why (written into build_blockers.json).
BLOCKED_CELLS = {
    "aops_curation": {
        "reason": "The AoPS V+A bank population (5,202 rows, 606 problems) IS "
                  "exactly the dense arm's HELD-OUT rows: "
                  "datasets/math/aops/va/population_manifest.json records "
                  "T_provenance = 'grouped Llama-3.1-8B LoRA arm trained "
                  "upstream on runs/aops_same_approach_dense_llama8b/"
                  "split_full/train.csv; these are its held-out rows' "
                  "(dense_split column: eval 2,510 / test 2,692, TRAIN 0). "
                  "The cell therefore has ZERO bank-covered rows in its own "
                  "dense TRAIN split, so a V3 arm cannot be trained on the "
                  "cell's own splits without Gemma-judging the ~22,725 "
                  "split_full train rows (new judging = out of scope).",
        "decoy_warning": "datasets/math/aops/va/dense_standard/split/ exists "
                         "(10,457/1,307/1,307) but is a STALE ORPHAN of a "
                         "superseded 13,071-row draw -- no rm_out_seed*, no "
                         "eval_pass_results.json, nothing was ever trained on "
                         "it. It shares the sha1(problem|body)[:20] id scheme "
                         "with the bank, so a naive merge on row_id silently "
                         "matches 2,307/5,202 bank rows and mislabels 1,933 of "
                         "them 'train'. DO NOT USE.",
        "what_would_unblock": "Either (a) score the AoPS dense-train rows with "
                              "the frozen Gemma-4-31B A-bank, or (b) redefine "
                              "the cell with a fresh grouped 80/10/10 split "
                              "over the 5,202 scored rows and retrain the raw "
                              "dense control at matched n (a NEW cell, not the "
                              "registry cell).",
    },
    "code_v3": {
        "reason": "The code_v3 A-bank was only ever scored on the dense arm's "
                  "EVAL+TEST rows: closure/code_v3/abank_rescore/score_meta.json "
                  "records text_source = "
                  "'datasets/code-review/dense_standard_v3/split/{eval,test}.csv' "
                  "and the 16 scores_shard*.npz stack to exactly 11,452 rows = "
                  "eval 5,822 + test 5,630. The dense TRAIN split (47,659 rows / "
                  "973 repos) has ZERO scored rows, and the split is "
                  "repo-grouped with 0 repo overlap across buckets, so no "
                  "train-row criterion scores can be borrowed. A V3 arm cannot "
                  "be trained on this cell's own splits, and the importance "
                  "ranking (which must be TRAIN-fold-only) cannot be computed "
                  "at all.",
        "secondary_deviations": "code_v3's own dense chain also deviates from "
                                "the frozen recipe (--max_length 2048 and "
                                "--class_weight_auto ON, "
                                "methods/dense/run_pr_dense_v3.sh on sk3), so "
                                "even a rebuilt arm would not be recipe-matched "
                                "to the other 12 cells.",
        "what_would_unblock": "Gemma-4-31B-score the 47,659 dense-train rows "
                              "with the same 83-criterion A bank (new judging = "
                              "out of scope), or define a new cell on the "
                              "11,452 scored rows with a fresh repo-grouped "
                              "80/10/10 split plus a matched-n raw-text control.",
    },
}


# ================================================================= engine ===
def read_splits(split_dir: Path):
    sd = split_dir / "split"
    out = {}
    for s in SPLITS:
        p = sd / f"{s}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        out[s] = pd.read_csv(p)
    return out


def build_bank_index(spec):
    """key -> bank row index, with a degeneracy guard on duplicate keys.

    Duplicate keys are tolerated ONLY when every bank row sharing the key has
    the same y and a numerically identical VA row (NaN-aware) -- then the first
    row is used.  Otherwise the key is AMBIGUOUS and every dataset row carrying
    it is dropped (never fabricate an alignment).
    """
    keys = spec["bank_keys"]
    VA, y = spec["bank_VA"], spec["bank_y"]
    first, dupes = {}, {}
    for i, k in enumerate(keys):
        if k in first:
            dupes.setdefault(k, [first[k]]).append(i)
        else:
            first[k] = i
    ambiguous = set()
    for k, idx in dupes.items():
        a = VA[idx[0]]
        ok = all(int(y[j]) == int(y[idx[0]]) and
                 np.array_equal(np.nan_to_num(VA[j], nan=-9e18),
                                np.nan_to_num(a, nan=-9e18)) for j in idx[1:])
        if not ok:
            ambiguous.add(k)
    return first, dupes, ambiguous


def rank_criteria(VA, y, groups, k, seed=0):
    """TRAIN-FOLD-ONLY grouped permutation importance (frozen protocol)."""
    n_splits = min(3, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("need >=2 groups in the train split for GroupKFold")
    col_med = np.nanmedian(VA, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    VAi = np.where(np.isnan(VA), col_med[None, :], VA)
    imps = np.zeros(VA.shape[1])
    nf = 0
    for itr, ite in GroupKFold(n_splits=n_splits).split(VAi, y, groups):
        if len(np.unique(y[itr])) < 2 or len(np.unique(y[ite])) < 2:
            continue
        m = _fit_gbm(seed)
        m.fit(VAi[itr], y[itr])
        r = permutation_importance(m, VAi[ite], y[ite], scoring="roc_auc",
                                   n_repeats=5, random_state=seed, n_jobs=-1)
        imps += r.importances_mean
        nf += 1
    if nf == 0:
        raise RuntimeError("no usable importance folds (single-class folds)")
    imps /= nf
    return imps, nf


def modal_share(col):
    """Share of the modal value, NaN treated as its own category."""
    s = pd.Series(col)
    vc = s.value_counts(dropna=False)
    return float(vc.iloc[0] / len(s)) if len(vc) else 1.0


def render(text, crits, VA_row, long_text):
    lines = ["VA metrics:"] + [f"    {nm}: {fmt(VA_row[j])}" for nm, j in crits]
    if long_text:
        # CW mirror format: block FIRST (right-side truncation safety)
        return "\n".join(lines) + "\n" + text
    # caption format: text first, block appended
    return "\n".join(["full text:", f"    {text}"] + lines)


def truncation_stats(tok, texts, max_len=MAX_LEN):
    lens = []
    B = 512
    for i in range(0, len(texts), B):
        enc = tok(list(texts[i:i + B]), add_special_tokens=True)["input_ids"]
        lens.extend(len(x) for x in enc)
    lens = np.asarray(lens)
    return {"n": int(len(lens)), "rate_over_max_len": float((lens > max_len).mean()),
            "tok_median": int(np.median(lens)), "tok_p95": int(np.percentile(lens, 95)),
            "tok_max": int(lens.max())}


def build_cell(slug, k=DEFAULT_K, also_k=(), verbose=True):
    t0 = time.time()
    if slug in BLOCKED_CELLS:
        record_blocker(slug, BLOCKED_CELLS[slug])
        print(f"[{slug}] SKIPPED (recorded blocker) -> {BLOCKERS_PATH}")
        return None
    if ADAPTERS.get(slug) is None:
        raise KeyError(f"no adapter for {slug}")

    spec = ADAPTERS[slug]()
    split_dir = Path(spec["split_dir"])
    splits = read_splits(split_dir)
    orig_sha = {s: sha256_file(split_dir / "split" / f"{s}.csv") for s in SPLITS}
    n_orig = {s: int(len(splits[s])) for s in SPLITS}
    if verbose:
        print(f"[{slug}] dense-standard splits {split_dir} n={n_orig}")

    # ---------------- join --------------------------------------------------
    # Fall back to the adapter's secondary key if the primary column is absent
    # (never silently: the choice is recorded in the manifest).
    try:
        spec["split_key_fn"](splits["train"].head(2))
    except KeyError:
        if "_fallback_key_fn" not in spec:
            raise
        print(f"[{slug}] primary join key unavailable in the split CSV -> "
              f"falling back to {spec['_fallback_desc']}")
        spec["split_key_fn"] = spec["_fallback_key_fn"]
        spec["bank_keys"] = spec["_fallback_bank_keys"]
        spec["join_desc"] = spec["_fallback_desc"] + " (FALLBACK key)"
    key_to_row, dupes, ambiguous = build_bank_index(spec)
    y_bank = spec["bank_y"]
    kept, join_report = {}, {}
    for s in SPLITS:
        df = splits[s]
        keys = spec["split_key_fn"](df)
        rows = np.array([key_to_row.get(kk, -1) if kk not in ambiguous else -1
                         for kk in keys])
        ok = rows >= 0
        # ASSERT the join: y must agree elementwise on every matched row
        if ok.sum():
            ymatch = (y_bank[rows[ok]] == df["judgement"].to_numpy()[ok])
            if not ymatch.all():
                bad = int((~ymatch).sum())
                raise AssertionError(
                    f"{slug}/{s}: join UNSAFE -- y disagrees on {bad}/{int(ok.sum())} "
                    f"matched rows between the dense split CSV and the bank")
        kept[s] = (np.flatnonzero(ok), rows[ok])
        join_report[s] = {
            "n_orig": n_orig[s], "n_matched": int(ok.sum()),
            "coverage": round(float(ok.mean()), 4),
            "n_unmatched": int((~ok).sum()),
            "n_dropped_ambiguous_key": int(sum(
                1 for kk in np.asarray(keys)[~ok] if kk in ambiguous)),
        }
    total_orig = sum(n_orig.values())
    total_kept = sum(len(v[0]) for v in kept.values())
    coverage = total_kept / total_orig
    row_set_identical = all(len(kept[s][0]) == n_orig[s] for s in SPLITS)
    if total_kept == 0:
        raise AssertionError(f"{slug}: bank/dense join matched 0 rows")
    if verbose:
        print(f"[{slug}] join coverage {coverage:.4f} "
              f"(train {join_report['train']['coverage']}, "
              f"eval {join_report['eval']['coverage']}, "
              f"test {join_report['test']['coverage']}) "
              f"identical_row_sets={row_set_identical}")

    VA_all = spec["bank_VA"]
    names = spec["bank_names"]

    # ---------------- column screen (TRAIN rows only) -----------------------
    tr_idx, tr_rows = kept["train"]
    Xtr = VA_all[tr_rows]
    ytr = splits["train"]["judgement"].to_numpy()[tr_idx].astype(int)
    gtr = np.array([str(g) for g in splits["train"]["group"].to_numpy()[tr_idx]], dtype=object)

    allnan = np.isnan(Xtr).all(axis=0)
    const = np.array([len(np.unique(Xtr[np.isfinite(Xtr[:, j]), j])) <= 1
                      for j in range(Xtr.shape[1])])
    usable = ~(allnan | const)
    usable_cols = np.flatnonzero(usable)
    mshare = {int(j): modal_share(Xtr[:, j]) for j in usable_cols}
    candidates = [j for j in usable_cols if mshare[int(j)] < MODAL_DROP]
    if len(candidates) < k:
        # relax rather than fabricate: keep the least-modal columns available
        candidates = sorted(usable_cols, key=lambda j: mshare[int(j)])[:max(k, 1)]

    # ---------------- importance (TRAIN FOLDS ONLY) -------------------------
    imps_sub, nfolds = rank_criteria(Xtr[:, usable_cols], ytr, gtr, k)
    imps = np.full(VA_all.shape[1], -np.inf)
    imps[usable_cols] = imps_sub
    order = [int(j) for j in usable_cols[np.argsort(-imps_sub)]]
    cand_set = set(int(j) for j in candidates)
    ranked_candidates = [j for j in order if j in cand_set]

    kmax = max([k] + list(also_k))
    dropped_modal = [
        {"name": names[j], "col": int(j), "modal_share": round(mshare[int(j)], 4),
         "importance": float(imps[j]),
         "rank_if_kept": order.index(j) + 1}
        for j in order[:kmax * 3] if j not in cand_set]

    def topk(kk):
        picks = ranked_candidates[:kk]
        return [(names[j], int(j)) for j in picks]

    # ---------------- tokenizer -------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    manifests = {}
    for kk in [k] + [x for x in also_k if x != k]:
        crits = topk(kk)
        suffix = "" if kk == k else f"_k{kk}"
        outdir = OUT_ROOT / f"v3grid_{slug}{suffix}"
        (outdir / "split").mkdir(parents=True, exist_ok=True)

        # --- placement: declared by the adapter, with an automatic escalation.
        # Total token count is placement-INVARIANT, so the append-vs-prepend
        # choice only decides WHAT the right-side truncation deletes.  If more
        # than ESCALATE_AT of rows would overrun max_len, an appended block
        # would be silently deleted on exactly those rows (audit §1.2) -> force
        # PREPEND.  Measured, never assumed.
        ESCALATE_AT = 0.01
        all_raw = [str(t) for s in SPLITS for t in splits[s].iloc[kept[s][0]]["text"]]
        all_rows = [r for s in SPLITS for r in kept[s][1]]
        probe = [render(t, crits, VA_all[r], True) for t, r in zip(all_raw, all_rows)]
        pre_stats = truncation_stats(tok, probe)
        long_text = bool(spec["long_text"])
        escalated = False
        if not long_text and pre_stats["rate_over_max_len"] > ESCALATE_AT:
            long_text, escalated = True, True
            print(f"[{slug}] ESCALATED to PREPEND: "
                  f"{pre_stats['rate_over_max_len']:.4f} of rows overrun max_len "
                  f"with the k={kk} block, an appended block would be deleted there")

        frames, raw_texts, aug_texts = {}, [], []
        for s in SPLITS:
            idx, rows = kept[s]
            df = splits[s].iloc[idx].reset_index(drop=True)
            texts = [str(t) for t in df["text"]]
            aug = [render(t, crits, VA_all[r], long_text)
                   for t, r in zip(texts, rows)]
            out = pd.DataFrame({
                "text": aug,
                "judgement": df["judgement"].astype(int).to_numpy(),
                "group": [str(g) for g in df["group"]],
                "did": [spec["bank_ids"][r] for r in rows],
            })
            # ---- byte-level assertions against the ORIGINAL split rows ----
            assert list(out["judgement"]) == list(df["judgement"].astype(int)), \
                f"{slug}/{s}: judgement column drifted"
            assert list(out["group"]) == [str(g) for g in df["group"]], \
                f"{slug}/{s}: group column drifted"
            for a, t in zip(aug, texts):
                if long_text:
                    assert a.endswith("\n" + t), f"{slug}/{s}: prepend corrupted text"
                else:
                    assert a.startswith("full text:\n    " + t + "\nVA metrics:"), \
                        f"{slug}/{s}: append corrupted text"
            out.to_csv(outdir / "split" / f"{s}.csv", index=False)
            frames[s] = out
            raw_texts.extend(texts)
            aug_texts.extend(aug)
        pd.concat([frames[s] for s in SPLITS], ignore_index=True).to_csv(
            outdir / "data.csv", index=False)

        n_kept = {s: int(len(frames[s])) for s in SPLITS}
        tot = sum(n_kept.values())
        fracs = {s: n_kept[s] / tot for s in SPLITS}
        stock_gate_ok = (abs(fracs["train"] - .8) <= 2e-2 and
                         abs(fracs["eval"] - .1) <= 2e-2 and
                         abs(fracs["test"] - .1) <= 2e-2)

        block_only = ["\n".join(["VA metrics:"] +
                                [f"    {nm}: {fmt(VA_all[kept['train'][1][0]][j])}"
                                 for nm, j in crits])]
        trunc = {
            "max_len": MAX_LEN,
            "raw_text_only": truncation_stats(tok, raw_texts),
            "with_block": truncation_stats(tok, aug_texts),
            "block_tokens": int(len(tok(block_only[0], add_special_tokens=False)["input_ids"])),
        }

        man = {
            "cell": slug, "arm": f"v3grid_{slug}{suffix}", "k": kk,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recipe": "V3 criteria-in-prompt, names only, no definitions, no "
                      "importance weights, no score re-rendering "
                      "(notes/2026-08-09__v3_audit_fable.md §5)",
            "k_caveat": K_CAVEAT,
            "group_column": spec["group_column"],
            "selection_split": spec["selection_split"],
            "orig_chain": spec["orig_chain"],
            "dense_standard_split_dir": str(split_dir),
            "orig_split_sha256": orig_sha,
            "n_orig": n_orig, "n": n_kept, "split_fractions":
                {s: round(fracs[s], 4) for s in SPLITS},
            "trainer_entry": ("methods/dense/train_reward_model.py" if stock_gate_ok
                              else "methods/taste_decomposition/fusion/train_grown_split.py"),
            "trainer_entry_reason": ("split fractions inside the stock 80/10/10 "
                                     "+-2%% gate" if stock_gate_ok else
                                     "split fractions outside the stock 80/10/10 "
                                     "+-2%% gate -> relaxed-gate launcher (gate only, "
                                     "recipe unchanged)"),
            "bank_source": spec["bank_source"],
            "bank_n_rows": int(VA_all.shape[0]),
            "bank_n_cols": int(VA_all.shape[1]),
            "join": {
                "description": spec["join_desc"],
                "per_split": join_report,
                "overall_coverage": round(coverage, 4),
                "n_bank_keys_duplicated": int(len(dupes)),
                "n_bank_keys_ambiguous_dropped": int(len(ambiguous)),
                "assertion": "y compared elementwise between the dense split CSV "
                             "and the joined bank row; build aborts on any mismatch",
            },
            "eval_test_row_sets": {
                "identical_to_original": bool(row_set_identical),
                "status": ("IDENTICAL" if row_set_identical else
                           "SUBSET_OF_ORIGINAL (bank does not cover every dense row)"),
                "eval": {"n_original": n_orig["eval"], "n_kept": n_kept["eval"]},
                "test": {"n_original": n_orig["test"], "n_kept": n_kept["test"]},
                "byte_assertions": [
                    "every emitted row's judgement equals the original split CSV row's",
                    "every emitted row's group equals the original split CSV row's",
                    ("every emitted text ENDS WITH '\\n' + the original text verbatim"
                     if long_text else
                     "every emitted text STARTS WITH 'full text:\\n    ' + the "
                     "original text verbatim + '\\nVA metrics:'"),
                    "original split CSV sha256 recorded in orig_split_sha256",
                ],
            },
            "block_placement": "PREPEND" if long_text else "APPEND",
            "block_placement_reason": (
                ("long-text cell: the trainer and score_eval_dense_v4.py truncate "
                 "the RIGHT side at max_len=1024, so an appended block would be "
                 "silently deleted on exactly the long documents (audit §1.2)"
                 + (" [AUTO-ESCALATED from APPEND: measured %.4f of rows overrun "
                    "max_len with this block]" % pre_stats["rate_over_max_len"]
                    if escalated else ""))
                if long_text else
                "short-text cell: measured %.5f of rows overrun max_len with the "
                "block (below the %.2f escalation threshold), so the caption "
                "format (text first, block appended) is reproduced verbatim"
                % (pre_stats["rate_over_max_len"], ESCALATE_AT)),
            "block_placement_auto_check": {
                "escalation_threshold": ESCALATE_AT,
                "measured_rate_over_max_len_with_block": pre_stats["rate_over_max_len"],
                "escalated": escalated,
                "note": "total token count is placement-invariant; placement only "
                        "decides whether the right-side truncation eats the block "
                        "or the document tail",
            },
            "importance_protocol":
                "TRAIN-ONLY GroupKFold(%d) inside the dense train split, frozen "
                "HistGB (max_leaf_nodes=31, lr=.06, max_iter=400, early stopping) "
                "seed 0, permutation_importance roc_auc n_repeats=5 on the inner "
                "held-out fold, mean over folds" % nfolds,
            "importance_n_train_rows": int(len(tr_idx)),
            "importance_n_groups": int(len(np.unique(gtr))),
            "top_k_criteria": [
                {"rank": i + 1, "name": nm, "col": j,
                 "train_fold_importance": float(imps[j]),
                 "modal_share_train": round(mshare[j], 4)}
                for i, (nm, j) in enumerate(crits)],
            "columns_screened": {
                "n_bank_cols": int(VA_all.shape[1]),
                "n_all_nan_on_train": int(allnan.sum()),
                "n_constant_on_train": int((const & ~allnan).sum()),
                "n_usable": int(len(usable_cols)),
                "modal_threshold": MODAL_DROP,
                "n_dropped_modal_ge_threshold": int(len(usable_cols) - len(cand_set)),
                "dropped_modal_that_would_have_ranked_high": dropped_modal,
            },
            "truncation": trunc,
            "leakage_rules": [
                "importance ranked on the dense TRAIN split only (GroupKFold(3) "
                "inside it); eval/test rows never enter the ranking",
                "criterion scores are label-blind Gemma-4-31B judge outputs, so "
                "they are safe to render on every split",
                "y NEVER appears in a prompt",
                "splits are the cell's own dense-standard split CSVs, read verbatim",
            ],
            "example_prompts": [aug_texts[0][:1500],
                                aug_texts[len(aug_texts) // 2][:1500]],
        }
        (outdir / "manifest.json").write_text(json.dumps(man, indent=2))
        manifests[kk] = man
        if verbose:
            print(f"[{slug}] wrote {outdir}  k={kk}  n={n_kept}  "
                  f"trunc raw {trunc['raw_text_only']['rate_over_max_len']:.3f} "
                  f"-> block {trunc['with_block']['rate_over_max_len']:.3f}  "
                  f"({time.time()-t0:.0f}s)")
    return manifests


def record_blocker(slug, info):
    cur = {}
    if BLOCKERS_PATH.exists():
        cur = json.loads(BLOCKERS_PATH.read_text())
    info = dict(info)
    info["recorded_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cur[slug] = info
    BLOCKERS_PATH.write_text(json.dumps(cur, indent=2))


GRID_ORDER = ["nc_agree", "peer_curation", "hashtagwars_verdict", "nc_outcome",
              "jokes_community", "mathse_accepted_verdict", "mathse_vote_score",
              "aops_curation", "press_verdict", "peer_verdict", "nc_responded",
              "peer_revealed", "code_v3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", default=None)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--also-k", type=int, action="append", default=[])
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for c in GRID_ORDER:
            print(c, "BLOCKED" if c in BLOCKED_CELLS else "")
        return
    cells = args.cell or GRID_ORDER
    for c in cells:
        print(f"=== {c} ===", flush=True)
        try:
            build_cell(c, k=args.k, also_k=tuple(args.also_k))
        except Exception as e:
            print(f"[{c}] FAILED: {type(e).__name__}: {e}", flush=True)
            record_blocker(c, {"reason": f"{type(e).__name__}: {e}",
                               "stage": "build", "auto_recorded": True})
            raise


if __name__ == "__main__":
    main()
