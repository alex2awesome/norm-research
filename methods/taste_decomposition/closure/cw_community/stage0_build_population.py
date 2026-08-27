#!/usr/bin/env python3
"""STAGE 0 for the CW-community (writingprompts upvote) confirmatory closure cell.

Enlarges the honest population from 408 rows to >=2,500 by EXTENDING the frozen
CW A-bank sample along its own stable-hash prompt ordering, restricted to prompt
groups that are held out for the dense model.

Provenance rules obeyed:
  * SAME salt / SAME hash order as datasets/va_gemma_banks/score_va_gemma_banks.py
    :: build_creative()  (SALT = "cw-va-v2-sample", order = sha256(f"{SALT}|{pid}")).
    We simply continue past the original 2,000-row cut.
  * SAME row id scheme: f"{prompt_id}_{sha1(text)[:10]}".
  * dense split is prompt_id-GROUPED (split_metadata.json group_split_column =
    prompt_id, seed 42), verified: 56,361 + 7,046 + 7,046 = 70,453 disjoint pids.
    So a whole extension group is either fully held out or not at all.
  * criteria are UNCHANGED (the 45 rubrics_initial.jsonl entries) -- this is a
    population extension, not a new instrument.

Outputs (closure/cw_community/):
  pop_ext_manifest.json     counts + provenance
  cw_ext_to_score.csv       NEW rows needing Gemma scoring (id, prompt_id, prompt,
                            story, judgement, dense_split)  [uploaded to sk3]
  cw_honest_population.csv  the full honest population (old held-out + new)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CW = REPO / "datasets" / "creative-writing"
SOURCE = CW / "va_bank_v2" / "writingprompts_modeling_clean_reconstructed.csv.gz"
VA_OUT = REPO / "outputs" / "va_gemma_banks"

SALT = "cw-va-v2-sample"
TARGET_NEW_ROWS = int(os.environ.get("CW_TARGET_NEW_ROWS", "2500"))


def sha(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def rid(prompt_id, text) -> str:
    return f"{prompt_id}_{hashlib.sha1(str(text).encode()).hexdigest()[:10]}"


def cw_split(text):
    parts = str(text).split("\n\nSTORY: ", 1)
    if len(parts) == 2:
        return parts[0].removeprefix("PROMPT: ").strip(), parts[1]
    return "", str(text)


def main():
    df = pd.read_csv(SOURCE)
    df["prompt_id"] = df["prompt_id"].astype(str)
    df["rid"] = [rid(p, t) for p, t in zip(df["prompt_id"], df["text"])]
    assert df["rid"].nunique() == len(df), "source id collisions"
    print(f"[src] n={len(df)} pids={df.prompt_id.nunique()}")

    # --- gate: the frozen 2,000-row bank population must be reproducible -------
    meta = json.loads((VA_OUT / "creative_writing_meta.json").read_text())
    bank_ids = list(meta["item_ids"])
    by_rid = dict(zip(df["rid"], range(len(df))))
    n_match = sum(1 for i in bank_ids if i in by_rid)
    print(f"[gate] bank ids matched in source: {n_match}/{len(bank_ids)}")
    assert n_match == len(bank_ids), "bank ids do not reproduce from local source"

    order = sorted(df["prompt_id"].unique(), key=lambda p: sha(f"{SALT}|{p}"))
    # reproduce the original prefix cut (>=2000 rows)
    counts = df.groupby("prompt_id").size().to_dict()
    n, orig_pids = 0, []
    for pid in order:
        orig_pids.append(pid)
        n += counts[pid]
        if n >= 2000:
            break
    orig_pid_set = set(orig_pids)
    orig_rows = df[df.prompt_id.isin(orig_pid_set)]
    print(f"[gate] reproduced original prefix: {len(orig_rows)} rows, "
          f"{len(orig_pids)} pids (expected 2000 / 1500)")
    assert len(orig_rows) == len(bank_ids), "prefix size mismatch"
    assert set(orig_rows["rid"]) == set(bank_ids), "prefix id-set mismatch"

    # --- dense split map ------------------------------------------------------
    smap = pd.read_csv(HERE / "cw_promptid_dense_split.csv")
    smap["prompt_id"] = smap["prompt_id"].astype(str)
    split_of = dict(zip(smap["prompt_id"], smap["dense_split"]))
    df["dense_split"] = df["prompt_id"].map(split_of)
    unmapped = int(df["dense_split"].isna().sum())
    print(f"[split] unmapped rows: {unmapped}")

    heldout_pid = {p for p, s in split_of.items() if s in ("eval", "test")}

    # --- extension: continue the SAME order, keep held-out groups only ---------
    already = set(orig_pids)
    ext_pids, n_new = [], 0
    for pid in order:
        if pid in already or pid not in heldout_pid:
            continue
        ext_pids.append(pid)
        n_new += counts[pid]
        if n_new >= TARGET_NEW_ROWS:
            break
    ext_rows = df[df.prompt_id.isin(set(ext_pids))].copy()
    print(f"[ext] {len(ext_rows)} new rows across {len(ext_pids)} new prompt groups")

    # NB: re-slice from df (which now carries dense_split), not from orig_rows
    old_held = df[df.prompt_id.isin(orig_pid_set & heldout_pid)].copy()
    print(f"[old] already-scored held-out rows: {len(old_held)} "
          f"({old_held.prompt_id.nunique()} pids)")

    honest = pd.concat([old_held, ext_rows], ignore_index=True)
    honest = honest.sort_values("rid").reset_index(drop=True)
    print(f"[honest] population n={len(honest)} pids={honest.prompt_id.nunique()} "
          f"pos_rate={honest.judgement.mean():.4f}")

    # split prompt/story for the Gemma scoring job
    ps = [cw_split(t) for t in ext_rows["text"]]
    out_ext = pd.DataFrame({
        "id": ext_rows["rid"].values,
        "prompt_id": ext_rows["prompt_id"].values,
        "prompt": [p for p, _ in ps],
        "story": [s for _, s in ps],
        "judgement": ext_rows["judgement"].astype(int).values,
        "dense_split": ext_rows["dense_split"].values,
    })
    out_ext.to_csv(HERE / "cw_ext_to_score.csv", index=False)

    hps = [cw_split(t) for t in honest["text"]]
    pd.DataFrame({
        "id": honest["rid"].values,
        "prompt_id": honest["prompt_id"].values,
        "text": honest["text"].values,
        "prompt": [p for p, _ in hps],
        "story": [s for _, s in hps],
        "judgement": honest["judgement"].astype(int).values,
        "dense_split": honest["dense_split"].values,
        "is_new": (~honest["rid"].isin(set(old_held["rid"]))).values,
    }).to_csv(HERE / "cw_honest_population.csv", index=False)

    man = {
        "salt": SALT,
        "source": str(SOURCE),
        "n_source_rows": int(len(df)),
        "n_source_pids": int(df.prompt_id.nunique()),
        "gate_bank_ids_reproduced": True,
        "orig_population": {"n": int(len(orig_rows)), "n_pids": len(orig_pids)},
        "orig_heldout": {"n": int(len(old_held)),
                         "n_pids": int(old_held.prompt_id.nunique())},
        "extension": {"n": int(len(ext_rows)), "n_pids": len(ext_pids),
                      "target_new_rows": TARGET_NEW_ROWS,
                      "pos_rate": float(ext_rows.judgement.mean())},
        "honest_population": {
            "n": int(len(honest)), "n_pids": int(honest.prompt_id.nunique()),
            "pos_rate": float(honest.judgement.mean()),
            "dense_split_counts": honest.dense_split.value_counts().to_dict()},
        "rule": ("prompt groups with dense_split in {eval,test}, taken in "
                 "sha256('cw-va-v2-sample|'+prompt_id) order, prefix extended "
                 "past the original 2,000-row cut until >=2,500 NEW rows"),
    }
    (HERE / "pop_ext_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
