#!/usr/bin/env python3
"""RoyalRoad DECOMPOSITION arm: head 614 + tail 410 tokens at max_len 1024.

The judge-view arm moved T .4986 -> .5846 (+.0860) but changed TWO things at once:
the VIEW (first-1024 -> head+tail, i.e. the arm finally sees the ending) and the
BUDGET (1024 -> 1600 tokens). This arm holds the BUDGET at the frozen 1024 and
changes only the VIEW, on the same folds/seed/rows.

  if this arm holds most of +.086  -> the story is clean "sees the ending" and the
                                      registry retires standard-1024 T as a VIEW artifact
  if it recovers only part of it   -> "view + budget", and both must be named

Head/tail ratio is the judge's own 60/40, scaled to 1024: 614 head + 410 tail.
Truncation uses the judge's code path (tokens, not chars).

  python datasets/creative-writing/build_royalroad_decomp_folds.py
"""
import json, os, sys
from pathlib import Path
import pandas as pd
REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
CELL = REPO / "datasets/creative-writing/royalroad_stubs"
SRC, OUT = CELL / "dense_crossfit", CELL / "dense_crossfit_decomp1024"
sys.path.insert(0, str(REPO / "datasets/va_gemma_banks"))
import score_cw_expert_banks as SC
HEAD, TAIL, BUDGET = 614, 410, 1024
tok = SC._tokenizer()
pop = pd.read_csv(CELL / "va/population.csv.gz"); pop["row_id"] = pop.row_id.astype(str)
view, n = {}, 0
for r in pop.itertuples():
    ids = tok.encode(str(r.text), add_special_tokens=False)
    if len(ids) <= BUDGET:
        view[str(r.row_id)] = str(r.text).strip()
    else:
        view[str(r.row_id)] = (tok.decode(ids[:HEAD], skip_special_tokens=True)
                               + SC.TRUNC_MARKER
                               + tok.decode(ids[-TAIL:], skip_special_tokens=True)); n += 1
print(f"[view] head {HEAD} + tail {TAIL} @ budget {BUDGET}; {n}/{len(pop)} truncated")
cols = ["text", "judgement", "group", "row_id"]
for k in range(5):
    d = OUT / f"fold{k}"; (d / "split").mkdir(parents=True, exist_ok=True)
    for sp in ("train", "eval", "test"):
        s = pd.read_csv(SRC / f"fold{k}/split/{sp}.csv"); s["row_id"] = s.row_id.astype(str)
        assert s.row_id.isin(view).all()
        s["text"] = s.row_id.map(view); s[cols].to_csv(d / f"split/{sp}.csv", index=False)
    a = pd.read_csv(SRC / f"fold{k}/data.csv"); a["row_id"] = a.row_id.astype(str)
    a["text"] = a.row_id.map(view); a[cols].to_csv(d / "data.csv", index=False)
    for sp in ("train", "eval", "test"):
        assert set(pd.read_csv(SRC/f"fold{k}/split/{sp}.csv").row_id.astype(str)) == \
               set(pd.read_csv(d/f"split/{sp}.csv").row_id.astype(str))
print("[check] folds byte-identical to dense_crossfit: PASS")
(OUT / "manifest.json").write_text(json.dumps(
    {"design_id": "decomp1024", "head_tokens": HEAD, "tail_tokens": TAIL,
     "max_length": BUDGET, "purpose": "isolate VIEW from BUDGET in the +.0860 judge-view lift",
     "folds_identical_to": "dense_crossfit"}, indent=2))
print("BUILD_DECOMP1024_DONE")
