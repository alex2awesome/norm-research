#!/usr/bin/env python3
"""V3 criteria-in-prompt audit: build the caption sweep datasets (2026-08-08).

Reuses build_direction23_data.py's exact machinery (same module loads, same
frozen importance protocol: TRAIN-ONLY GroupKFold(3) permutation importance,
HistGB GRID[1] seed 0, roc_auc n_repeats=5, mean over folds). Determinism gate:
the recomputed top-10 must equal dense_data/build_manifest.json's aug_top.

Arms built (cap_crowd sandbox + cap_finalist confirm candidates):
  {cell}_k20 / {cell}_k40   count sweep, arm-a rendering (name: score)
  {cell}_k10defs            top-10 + one-line rubric definition in parentheses
  cap_crowd_augmd           k10 aug_a rendering x Direction-2 grown train pool
                            (extras get the SAME cleaned VA columns; NaN cols
                            imputed with the cell population median, identical
                            to clean_cols semantics)

Leakage rules unchanged: importance never sees eval/test rows; criterion
scores are label-blind judge outputs; y never in the prompt; eval/test row
sets byte-identical to the original cell splits (except augmd whose eval/test
equal aug_a's).

CPU only. Usage: python3 build_v3_audit_data.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
FUS = HERE.parent
TD = FUS.parent
REPO = TD.parents[1]
CAP = REPO / "datasets/humor/caption_multiy"
DD = FUS / "dense_data"
OUT = FUS / "dense_data"  # datasets land next to the existing arms
MIN_VOTES_EXTRA = 50


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


L1 = load_module(TD / "layer1_gemma_cells.py", "l1_v3_audit")
capagg = sys.modules["capagg_taste_decomp"]


def fmt(v):
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def one_line_def(desc: str, maxlen=160) -> str:
    d = " ".join(str(desc).split())
    # first sentence, capped
    m = re.split(r"(?<=[.;])\s", d, maxsplit=1)
    s = m[0].rstrip(".;")
    if len(s) > maxlen:
        s = s[:maxlen].rsplit(" ", 1)[0] + "..."
    return s


V_DEFS = {
    "v_char_len": "number of characters in the caption",
    "v_word_len": "number of words in the caption",
    "v_digit": "count of digit characters",
    "v_dash": "count of dash characters",
    "v_speech_verb": "count of speech verbs (say/ask/tell...)",
    "v_exclam": "count of exclamation marks",
    "v_question": "count of question marks",
    "v_quote": "count of quotation marks",
    "v_comma": "count of commas",
    "v_period": "count of periods",
    "v_colon": "count of colons",
    "v_ellipsis": "count of ellipses",
    "v_upper_frac": "fraction of uppercase characters",
    "v_ttr": "type-token ratio (lexical diversity)",
    "v_mean_word_chars": "mean word length in characters",
    "v_first_person": "count of first-person pronouns",
}


def main():
    manifest = json.loads((DD / "build_manifest.json").read_text())
    rubdefs = {}
    for line in (HERE / "standup_rubrics.jsonl").open():
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("name"):
                rubdefs[r["name"]] = one_line_def(r.get("description", ""))

    c = L1._caption_pools()
    meta = capagg.load_pool()
    report = {}

    med_by_contest = {}
    by_contest = defaultdict(list)
    for did in c["X_by_id"]:
        m = meta.get(did)
        if m is None:
            continue
        if m.get("crowd_mean") is not None and (m.get("crowd_votes") or 0) >= capagg.MIN_VOTES:
            by_contest[m["contest"]].append(m["crowd_mean"])
    med_by_contest = {k: float(np.median(v)) for k, v in by_contest.items() if len(v) >= 6}

    for cell, dl_name, id_key, y_key in (
            ("cap_crowd", "crowd", "crowd_ids", "y_crowd"),
            ("cap_finalist", "finalist", "hardneg_ids", "y_fin")):
        dl = CAP / "dense_llama" / dl_name
        data = pd.read_csv(dl / "data.csv")
        splits = {s: pd.read_csv(dl / "split" / f"{s}.csv") for s in ("train", "eval", "test")}

        by_ct = {}
        for did in c[id_key]:
            if did in c["X_by_id"]:
                by_ct[(str(c["contest_by_id"][did]), meta[did]["text"])] = did
        for nm, df in [("data", data)] + list(splits.items()):
            dids = [by_ct.get((str(g), t)) for t, g in zip(df.text, df.group)]
            assert all(d is not None for d in dids), f"{cell}/{nm}: unmatched rows"
            df["did"] = dids
        bucket_of_contest = {}
        for s in ("train", "eval", "test"):
            for g in splits[s].group.astype(str).unique():
                bucket_of_contest.setdefault(g, s)

        # ---- cleaned VA matrix, identical to the original build
        ids_all = sorted(d for d in c[id_key] if d in c["X_by_id"])
        A = np.array([c["X_by_id"][d] for d in ids_all], dtype=float)
        V = np.array([c["V_by_id"][d] for d in ids_all], dtype=float)
        Ac, a_keep = capagg.clean_cols(A)
        Vc, v_keep = capagg.clean_cols(V)
        names = [c["v_names"][j] for j in v_keep] + [c["a_names"][j] for j in a_keep]
        VA = np.column_stack([Vc, Ac])
        col_medians = np.median(VA, axis=0)
        n_v = len(v_keep)
        row_of = {d: i for i, d in enumerate(ids_all)}
        y_by_id = c[y_key]

        # ---- full train-only importance ranking (same protocol, all columns)
        tr_ids = [d for d in splits["train"].did]
        tr_idx = np.array([row_of[d] for d in tr_ids])
        Xtr, ytr = VA[tr_idx], np.array([y_by_id[d] for d in tr_ids])
        gtr = np.array([str(c["contest_by_id"][d]) for d in tr_ids])
        imps = np.zeros(VA.shape[1])
        nf = 0
        for itr, ite in GroupKFold(n_splits=3).split(Xtr, ytr, gtr):
            m = L1._fit_gbm(L1.GRID[1], seed=0)
            m.fit(Xtr[itr], ytr[itr])
            r = permutation_importance(m, Xtr[ite], ytr[ite], scoring="roc_auc",
                                       n_repeats=5, random_state=0, n_jobs=-1)
            imps += r.importances_mean
            nf += 1
        imps /= nf
        order = np.argsort(-imps)

        # determinism gate vs the shipped top-10
        old_top = [t["name"] for t in manifest[cell]["aug_top"]]
        new_top = [names[j] for j in order[:10]]
        assert new_top == old_top, f"{cell}: importance no longer reproduces\n{new_top}\nvs\n{old_top}"
        print(f"[{cell}] importance reproduces shipped top-10 exactly")

        full_rank = [{"rank": r + 1, "name": names[j], "col": int(j),
                      "importance": float(imps[j]),
                      "is_v": bool(j < n_v),
                      "modal_share": float(np.mean(VA[:, j] == pd.Series(VA[:, j]).mode()[0]))}
                     for r, j in enumerate(order)]
        (HERE / f"importance_full_{cell}.json").write_text(json.dumps({
            "n_cols_total": int(VA.shape[1]), "n_v": n_v,
            "protocol": "TRAIN-ONLY GroupKFold(3), HistGB GRID[1] seed0, "
                        "permutation_importance roc_auc n_repeats=5, mean over folds",
            "ranking": full_rank}, indent=2))

        def topk(k):
            return [{"name": names[j], "col": int(j)} for j in order[:k]]

        def aug_text_for(did, crits, with_defs=False):
            i = row_of[did]
            lines = ["full text:", f"    {meta[did]['text']}", "VA metrics:"]
            for t in crits:
                v = fmt(float(VA[i, t["col"]]))
                if with_defs:
                    nm = t["name"]
                    d = rubdefs.get(nm) or V_DEFS.get(nm)
                    assert d, f"no definition found for criterion {nm!r}"
                    lines.append(f"    {nm} ({d}): {v}")
                else:
                    lines.append(f"    {t['name']}: {v}")
            return "\n".join(lines)

        def write_arm(arm_name, crits, with_defs=False):
            ad = OUT / f"{cell}_{arm_name}"
            (ad / "split").mkdir(parents=True, exist_ok=True)
            for nm, df in [("data", data)] + list(splits.items()):
                out = df.copy()
                out["text"] = [aug_text_for(d, crits, with_defs) for d in out.did]
                out = out[["text", "judgement", "group", "did"]]
                if nm == "data":
                    out.to_csv(ad / "data.csv", index=False)
                else:
                    out.to_csv(ad / "split" / f"{nm}.csv", index=False)
            (ad / "manifest.json").write_text(json.dumps({
                "cell": cell, "arm": arm_name, "k": len(crits), "with_defs": with_defs,
                "criteria": [t["name"] for t in crits],
                "rows_identical_to_original_split": True,
                "importance_protocol": "same frozen protocol as dense_data/build_manifest.json",
            }, indent=2))
            print(f"[{cell}] wrote {arm_name} (k={len(crits)}, defs={with_defs})")

        write_arm("k20", topk(20))
        write_arm("k40", topk(40))
        write_arm("k10defs", topk(10), with_defs=True)

        report[cell] = {"n_cols": int(VA.shape[1]),
                        "top40": [names[j] for j in order[:40]]}

        # ---- cap_crowd only: augmd = k10 aug rendering x grown train pool
        if cell == "cap_crowd":
            crits10 = topk(10)
            extras = []
            for did, ro in c["role_by_id"].items():
                m = meta.get(did)
                if m is None or did in c["y_crowd"] or did not in c["X_by_id"]:
                    continue
                cm, cv = m.get("crowd_mean"), (m.get("crowd_votes") or 0)
                if cm is None or not (MIN_VOTES_EXTRA <= cv < capagg.MIN_VOTES):
                    continue
                if bucket_of_contest.get(str(m["contest"])) not in (None, "train"):
                    continue
                med = med_by_contest.get(m["contest"])
                if med is None or cm == med:
                    continue
                extras.append({"did": did, "judgement": int(cm > med), "group": m["contest"]})
            # cross-check against the shipped Direction-2 extras
            md_extras = pd.read_csv(DD / "cap_crowd_moredata" / "extras_manifest.csv")
            assert sorted(e["did"] for e in extras) == sorted(md_extras.did), \
                "augmd extras differ from shipped Direction-2 extras"

            # VA rows for extras: same kept columns, NaN -> population median
            def extra_va_row(did):
                a = np.asarray(c["X_by_id"][did], dtype=float)[a_keep]
                v = np.asarray(c["V_by_id"][did], dtype=float)[v_keep]
                row = np.concatenate([v, a])
                nan = np.isnan(row)
                row[nan] = col_medians[nan]
                return row

            def extra_aug_text(did):
                row = extra_va_row(did)
                lines = ["full text:", f"    {meta[did]['text']}", "VA metrics:"]
                for t in crits10:
                    lines.append(f"    {t['name']}: {fmt(float(row[t['col']]))}")
                return "\n".join(lines)

            ad = OUT / "cap_crowd_augmd"
            (ad / "split").mkdir(parents=True, exist_ok=True)
            aug_a = OUT / "cap_crowd_aug_a"
            tr = pd.read_csv(aug_a / "split" / "train.csv")
            ex_rows = pd.DataFrame([{
                "text": extra_aug_text(e["did"]), "judgement": e["judgement"],
                "group": e["group"], "did": e["did"]} for e in extras])
            grown = pd.concat([tr, ex_rows], ignore_index=True)
            grown.to_csv(ad / "split" / "train.csv", index=False)
            for s in ("eval", "test"):
                src = (aug_a / "split" / f"{s}.csv").read_text()
                (ad / "split" / f"{s}.csv").write_text(src)  # byte-identical to aug_a
            pd.concat([pd.read_csv(aug_a / "data.csv"), ex_rows], ignore_index=True) \
                .to_csv(ad / "data.csv", index=False)
            (ad / "manifest.json").write_text(json.dumps({
                "cell": cell, "arm": "augmd",
                "recipe": "aug_a k10 rendering x Direction-2 grown train pool",
                "n_train": int(len(grown)), "n_extras": int(len(ex_rows)),
                "extras_pos_rate": float(ex_rows.judgement.mean()),
                "eval_test_byte_identical_to_aug_a": True}, indent=2))
            print(f"[cap_crowd] wrote augmd (train {len(tr)} -> {len(grown)})")

    (HERE / "build_report.json").write_text(json.dumps(report, indent=2))
    print("BUILD_V3_AUDIT_DONE")


if __name__ == "__main__":
    main()
