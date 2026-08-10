#!/usr/bin/env python3
"""Tag-stratified math bank scoring (sub-community heterogeneity, task #60).

Scores the clean math answer-quality bank (medoid-bank-clean, 48 rubrics) on a sample of
math-SE answers stratified over the top primary_tag communities, with the SAME offline-vLLM
judge recipe as the CW arm runs (Llama-3.3-70B-FP8, executor-closed, post-audit prompt
semantics via io_metrics._JUDGE_PROMPT_HEADER).

Design:
- Sampling is stable-hash by question_id (md5), whole questions kept together (no pair leak
  in later grouped CV) — never a seeded shuffle of a growing list.
- ALL bank metrics are scored (no pooled viability pre-drop): a metric that is non-viable
  pooled may be viable within one tag — that asymmetry is part of the measurement.
- One parquet per tag (resume = skip existing), final concat + metric key JSON.

Output: <out>/scores__<tag>.parquet, <out>/math_tag_bank_scores.parquet, <out>/metric_key.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "methods")

import numpy as np
import pandas as pd

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.io_metrics import load_rubric_metrics_from_dir, make_vllm_judge_scorer

DEFAULT_TAGS = ("real-analysis,calculus,linear-algebra,abstract-algebra,probability,"
                "algebra-precalculus,general-topology,combinatorics,sequences-and-series,"
                "complex-analysis,geometry,integration")


def stable_tag_sample(df: pd.DataFrame, per_tag: int) -> pd.DataFrame:
    """Whole-question sampling ordered by md5(question_id) until >= per_tag items."""
    order = df.question_id.astype(str).map(
        lambda q: hashlib.md5(f"mathtag::{q}".encode()).hexdigest())
    out, n = [], 0
    for qid, g in df.assign(_h=order).sort_values("_h").groupby("question_id", sort=False):
        out.append(g)
        n += len(g)
        if n >= per_tag:
            break
    return pd.concat(out).drop(columns=["_h"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/math/stackexchange/math_se_v3_position_matched.csv.gz")
    ap.add_argument("--bank-dir", default="datasets/math/stackexchange/medoid-bank-clean")
    ap.add_argument("--tags", default=DEFAULT_TAGS)
    ap.add_argument("--per-tag", type=int, default=550)
    ap.add_argument("--out", default="outputs/ctree/math_tag_bank")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--executor-label", default="llama-3.3-70b-fp8")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    bank = load_rubric_metrics_from_dir(args.bank_dir)
    print(f"bank: {len(bank)} metrics from {args.bank_dir}", flush=True)
    key = {f"m{j:02d}": {"metric_id": m.metric_id, "name": m.name, "description": m.description}
           for j, m in enumerate(bank)}
    json.dump({"executor": args.executor_label, "judge_model": args.judge_model,
               "bank_dir": args.bank_dir, "metrics": key},
              open(out / "metric_key.json", "w"), indent=1)

    df = pd.read_csv(args.data, low_memory=False).dropna(subset=["text", "judgement", "primary_tag"])
    df["judgement"] = pd.to_numeric(df["judgement"], errors="coerce")
    df = df.dropna(subset=["judgement"])
    df["judgement"] = df["judgement"].astype(int)

    cfg = InfillConfig(
        materialize_backend="vllm_offline", materialize_model=args.judge_model,
        max_text_tokens=700, verbose=False,
        cache_dir="outputs/ctree/B_tree/judge_cache",
        output_dir=str(out))
    judge = make_vllm_judge_scorer(cfg)

    done = []
    for tag in tags:
        f = out / f"scores__{tag}.parquet"
        if f.exists():
            print(f"[{tag}] exists, skip", flush=True)
            done.append(f)
            continue
        sub = stable_tag_sample(df[df.primary_tag == tag], args.per_tag).reset_index(drop=True)
        print(f"[{tag}] n={len(sub)} base={sub.judgement.mean():.3f} "
              f"questions={sub.question_id.nunique()}", flush=True)
        lv, apl = judge(bank, sub["text"].astype(str).tolist())
        rec = sub[["question_id", "answer_id", "primary_tag", "judgement"]].copy()
        for j in range(len(bank)):
            rec[f"m{j:02d}_score"] = lv[:, j]
            rec[f"m{j:02d}_applied"] = apl[:, j]
        rec.to_parquet(f, index=False)
        sc = lv[apl] if apl.any() else np.array([np.nan])
        print(f"[{tag}] wrote {f.name}; applied {apl.mean():.2f}, "
              f"score mean {np.nanmean(sc):.3f} std {np.nanstd(sc):.3f}", flush=True)
        done.append(f)

    full = pd.concat([pd.read_parquet(f) for f in done], ignore_index=True)
    full.to_parquet(out / "math_tag_bank_scores.parquet", index=False)
    print(f"FINAL {len(full)} rows x {len(bank)} metrics -> {out}/math_tag_bank_scores.parquet",
          flush=True)


if __name__ == "__main__":
    main()
