#!/usr/bin/env python3
"""Style Invitational MATURE REBUILD: clean population + dense bundle.

WHY THIS EXISTS. SI's A bank was declared TERMINAL 2026-08-09 as
"TIE, bank = length model (0/32 rubrics survive length strata)". That verdict
was reached on a population that is **16.3% parse artifacts**, and the artifacts
are what most of the "length model" was made of.

THE CONTAMINATION (measured, not assumed -- the V8 ground-truth lesson):
`parse_results.py` splits each week's archive text into entries heuristically.
1,574 of the 9,637 rows carry NO joke text at all:

  | class                              | rows  |
  |------------------------------------|-------|
  | orphan byline, e.g. "(Bob Zane, Woodbridge)"      | 1,111 |
  | short list header ending in ':'                    |   428 |
  | archive section marker "And last:" / "And Last,"   |   133 |
  | cartoon / ink-blot selector "Cartoon B"            |    93 |
  | truncated orphan "Takoma Park)"                    |    17 |

These are 11x over-represented in the negative class -- honorable_mention
19.1%, runnerup 1.7%, winner 3.1% -- because the parser drops a byline or a
header into its own row and the row inherits the surrounding HM tier. They
average 22 chars against 110 for real entries.

CONSEQUENCE, and the reason the terminal verdict has to be revisited:

  | readout                         | all rows | fragments removed |
  |---------------------------------|----------|-------------------|
  | char length alone, pooled       |  .6227   |  **.5520**        |
  | char length alone, within-week  |  .6181   |  **.5589**        |

Roughly 60% of the length signal that beat the old bank was the model learning
"is this row a parse artifact". On the clean population length is a much weaker
nuisance, which is what makes a content bank worth rebuilding at all.

NEVER DELETE DATA: fragments are retained in the emitted population with
`is_fragment=True` and `fragment_class`, and are excluded from the analysis
population (and from the dense bundle) exactly the way V6's median-tied rows
were -- flagged, kept, and excluded with the reason on the record.

ITEM VIEW. `text` is byte-identical to the A judge's ctx and to the ORIGINAL
dense bundle's text: 'CONTEST PROMPT: {prompt}\\n\\nENTRY: "{entry}"'. Item-view
consistency therefore holds by construction and needs no sensitivity arm.

  python3 datasets/humor/style_invitational/build_si_clean_population.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SALT = "si-clean-v1|"

# ---- fragment detector (precise; each class is a row with NO joke content) ---
F_BYLINE = re.compile(r"^\(\s*[^()]*\)\s*\.?$")
F_ANDLAST = re.compile(r"(?i)^[+*\s]*and\s+(even\s+more\s+)?last\s*[:,.]?$")
F_CARTOON = re.compile(r"(?i)^\(?\s*(cartoon|ink\s*blot)\s+[a-z]\s*\)?\s*[:,.]?$")
F_ORPHAN = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)?\)\s*\.?$")
F_COLON = re.compile(r"^.{0,45}:$")
HAS_SENT = re.compile(r"[.!?]")


def fragment_class(s: str):
    """Return the artifact class, or None if the row carries joke text.

    Deliberately CONSERVATIVE: shortness alone is NOT a fragment signal, because
    genuine short entries exist (the headline-writing contests produce real
    entries like 'ALIENS SIMONIZED MY CAR' and 'CAPITALS WIN STANLEY CUP'). Only
    rows that are structurally a byline or a header are removed.
    """
    s = (s or "").strip()
    if not s:
        return "empty"
    if F_BYLINE.match(s):
        return "orphan_byline"
    if F_ANDLAST.match(s):
        return "section_marker"
    if F_CARTOON.match(s):
        return "cartoon_selector"
    if F_ORPHAN.match(s):
        return "truncated_orphan"
    if F_COLON.match(s) and not HAS_SENT.search(s):
        return "list_header"
    return None


def load_bucketer(repo: Path):
    import importlib.util
    p = repo / "datasets/patents/build_dense_standard_claimfell.py"
    spec = importlib.util.spec_from_file_location("cf_build", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["cf_build"] = m
    spec.loader.exec_module(m)
    return m.stable_hash_bucket_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    repo = Path(a.repo)
    si = repo / "datasets/humor/style_invitational"
    out = Path(a.out_dir) if a.out_dir else si / "va_v2"
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(si / "style_invitational.jsonl") if l.strip()]
    df = pd.DataFrame(rows)
    df["entry_text"] = df.entry_text.astype(str).str.strip()
    df["group"] = df.week_id.astype(str)
    # stable row id: same recipe as the original bank build, so the two
    # populations can be joined row-for-row if ever needed
    df["row_id"] = [hashlib.sha1(f"{r.week_id}\0{i}\0{r.entry_text}".encode())
                    .hexdigest()[:20] for i, r in enumerate(df.itertuples())]
    df["y_top_tier"] = df.tier.isin(["winner", "runnerup"]).astype(int)
    df["y_winner"] = (df.tier == "winner").astype(int)
    df["fragment_class"] = df.entry_text.map(fragment_class)
    df["is_fragment"] = df.fragment_class.notna()
    df["char_len"] = df.entry_text.str.len()
    df["text"] = ('CONTEST PROMPT: ' + df.contest_prompt.astype(str)
                  + '\n\nENTRY: "' + df.entry_text + '"')

    stats = {"build_date": datetime.now().isoformat(timespec="seconds"),
             "salt": SALT,
             "source": "datasets/humor/style_invitational/style_invitational.jsonl",
             "n_raw": int(len(df)),
             "fragment_classes": {k: int(v) for k, v in
                                  Counter(df.fragment_class.dropna()).items()},
             "n_fragments": int(df.is_fragment.sum()),
             "fragment_rate": float(df.is_fragment.mean()),
             "fragment_rate_by_tier": {k: float(v) for k, v in
                                       df.groupby("tier").is_fragment.mean().items()},
             "mean_charlen_fragment": float(df[df.is_fragment].char_len.mean()),
             "mean_charlen_clean": float(df[~df.is_fragment].char_len.mean())}

    clean = df[~df.is_fragment].copy()
    bucketer = load_bucketer(repo)
    y_by_group = {g: d.y_top_tier.tolist() for g, d in clean.groupby("group")}
    bmap = bucketer(y_by_group)
    df["split"] = df.group.map(bmap)
    clean = df[~df.is_fragment].copy()

    stats["population_clean"] = {
        "n": int(len(clean)), "n_weeks": int(clean.group.nunique()),
        "y_top_tier_pos_rate": float(clean.y_top_tier.mean()),
        "y_winner_pos_rate": float(clean.y_winner.mean()),
        "tier_counts": {k: int(v) for k, v in Counter(clean.tier).items()}}
    stats["splits"] = {s: {"rows": int((clean.split == s).sum()),
                           "weeks": int(clean[clean.split == s].group.nunique()),
                           "pos_rate": float(clean[clean.split == s].y_top_tier.mean())}
                       for s in ["train", "eval", "test"]}

    # the nuisance the rebuild is designed against, before and after cleaning
    from sklearn.metrics import roc_auc_score

    def wq(d, col):
        tot = w = 0.0
        for _, dd in d.groupby("group"):
            if dd.y_top_tier.nunique() < 2:
                continue
            n = int(dd.y_top_tier.sum() * (len(dd) - dd.y_top_tier.sum()))
            tot += n * roc_auc_score(dd.y_top_tier, dd[col]); w += n
        return float(tot / w)
    stats["length_nuisance"] = {
        "all_rows_pooled": float(roc_auc_score(df.y_top_tier, df.char_len)),
        "clean_pooled": float(roc_auc_score(clean.y_top_tier, clean.char_len)),
        "all_rows_within_week": wq(df, "char_len"),
        "clean_within_week": wq(clean, "char_len"),
        "week_identity_alone_clean": float(roc_auc_score(
            clean.y_top_tier, clean.groupby("group").y_top_tier.transform("mean")))}

    cols = ["row_id", "group", "split", "text", "entry_text", "contest_prompt",
            "tier", "y_top_tier", "y_winner", "char_len", "is_fragment",
            "fragment_class", "week_id"]
    df[cols].to_csv(out / "population.csv.gz", index=False, compression="gzip")

    # ---- dense bundle on the CLEAN population (same-rows T) ------------------
    dd = out / "dense_standard_si_clean"
    (dd / "split").mkdir(parents=True, exist_ok=True)
    c = clean.copy()
    c["judgement"] = c.y_top_tier
    dcols = ["text", "judgement", "group", "row_id"]
    c[dcols].to_csv(dd / "data.csv", index=False)
    for s in ["train", "eval", "test"]:
        c[c.split == s][dcols].to_csv(dd / "split" / f"{s}.csv", index=False)
    man = {"cell": "style_inv_toptier_clean", "n": int(len(c)),
           "pos_rate": float(c.judgement.mean()),
           "n_groups": int(c.group.nunique()), "group_column": "week_id",
           "y_definition": '1 iff tier in {"winner","runnerup"}, else 0 (top_tier)',
           "population_note": "PARSE-ARTIFACT-FREE: 1,574 byline/header rows "
                              "(16.3% of the archive parse, 11x concentrated in "
                              "the HM class) excluded; see population_manifest.json",
           "item_view": 'CONTEST PROMPT: {prompt}\\n\\nENTRY: "{entry}" -- '
                        "byte-identical to the A judge ctx and to the original "
                        "dense bundle, so item-view consistency holds",
           "recipe": "Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, "
                     "2 epochs, gradient-checkpointing, select-on-eval "
                     "(dense-standard, no deviation)",
           "split_row_counts": {s: int((c.split == s).sum())
                                for s in ["train", "eval", "test"]},
           "split_pos_rates": {s: float(c[c.split == s].judgement.mean())
                               for s in ["train", "eval", "test"]},
           "train_minority_count": int(min((c[c.split == "train"].judgement == 0).sum(),
                                           (c[c.split == "train"].judgement == 1).sum()))}
    (dd / "manifest.json").write_text(json.dumps(man, indent=1))
    stats["dense_manifest"] = man
    (out / "population_manifest.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in
                      ["n_raw", "n_fragments", "fragment_rate", "fragment_classes",
                       "fragment_rate_by_tier", "population_clean", "splits",
                       "length_nuisance"]}, indent=1))
    print("train minority:", man["train_minority_count"])


if __name__ == "__main__":
    main()
