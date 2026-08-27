#!/usr/bin/env python3
"""RoyalRoad cross-fit folds with the A JUDGE'S EXACT VIEW.

Mandated audit for the bank>dense flag (Delta_beyond -.0732 on the honest set).
Prime suspect is view asymmetry, the SO-qtrunc pattern repeating: the dense arm
read the FIRST 1,024 tokens while the A judge read head 960 + tail 640 -- i.e. the
judge SEES THE ENDING and the dense arm never does, on chapters with a median
2,918 source tokens (dense saw ~35% of the chapter, never the close).

This build changes exactly ONE thing: the text view. Fold membership is
byte-identical to dense_crossfit (same ids, same tenths, same seed), and the text
is cut by the SAME CODE PATH the judge used -- score_cw_expert_banks.token_trunc,
1600 source / 960 head / 640 tail, TOKENS not characters.

Trained at max_length 1600 so the model can actually receive the 1,600-token view;
that is a deliberate second difference from the frozen 1024 recipe and is reported
as such. If this arm moves T, a follow-up head-614+tail-410 arm at max_length 1024
separates "sees the ending" from "sees more tokens". Not run unless this one moves.

  python datasets/creative-writing/build_royalroad_judgeview_folds.py
"""
import os, sys, json
from pathlib import Path
import pandas as pd

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
CELL = REPO / "datasets/creative-writing/royalroad_stubs"
SRC = CELL / "dense_crossfit"
OUT = CELL / "dense_crossfit_judgeview"
sys.path.insert(0, str(REPO / "datasets/va_gemma_banks"))
import score_cw_expert_banks as SC          # the judge's own truncation code path

tok = SC._tokenizer()
pop = pd.read_csv(CELL / "va/population.csv.gz")
pop["row_id"] = pop["row_id"].astype(str)
view, ntrunc = {}, 0
for r in pop.itertuples():
    t, _, was = SC.token_trunc(str(r.text), tok)
    view[str(r.row_id)] = t
    ntrunc += int(was)
print(f"[view] token_trunc {SC.TRUNC_TOKENS_SOURCE} (head {SC.TRUNC_TOKENS_HEAD} / "
      f"tail {SC.TRUNC_TOKENS_TAIL}); {ntrunc}/{len(pop)} truncated")

cols = ["text", "judgement", "group", "row_id"]
man = {"design": "cross-fit folds, A-judge view (head 960 + tail 640 TOKENS)",
       "fold_membership": "byte-identical to dense_crossfit", "folds": {}}
for k in range(5):
    d = OUT / f"fold{k}"; (d / "split").mkdir(parents=True, exist_ok=True)
    for sp in ("train", "eval", "test"):
        s = pd.read_csv(SRC / f"fold{k}" / "split" / f"{sp}.csv")
        s["row_id"] = s["row_id"].astype(str)
        assert s.row_id.isin(view).all(), f"fold{k} {sp}: unknown row_id"
        s["text"] = s["row_id"].map(view)
        s[cols].to_csv(d / f"split/{sp}.csv", index=False)
    allrows = pd.read_csv(SRC / f"fold{k}" / "data.csv")
    allrows["row_id"] = allrows["row_id"].astype(str)
    allrows["text"] = allrows["row_id"].map(view)
    allrows[cols].to_csv(d / "data.csv", index=False)
    man["folds"][f"fold{k}"] = {sp: int(len(pd.read_csv(d / f"split/{sp}.csv")))
                                for sp in ("train", "eval", "test")}
    print(f"[fold{k}] {man['folds'][f'fold{k}']}")
# integrity: identical ids per fold/split
for k in range(5):
    for sp in ("train", "eval", "test"):
        a = set(pd.read_csv(SRC / f"fold{k}/split/{sp}.csv").row_id.astype(str))
        b = set(pd.read_csv(OUT / f"fold{k}/split/{sp}.csv").row_id.astype(str))
        assert a == b, f"fold{k} {sp} id mismatch"
print("[check] fold membership byte-identical to dense_crossfit: PASS")
(OUT / "manifest.json").write_text(json.dumps(man, indent=2))
print("BUILD_JUDGEVIEW_FOLDS_DONE")
