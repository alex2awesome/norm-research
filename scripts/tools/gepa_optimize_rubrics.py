#!/usr/bin/env python3
"""GEPA re-pass over stage-2 candidates: iterate each candidate's DENSE rubric against
label-free instrument diagnostics (retest + MI-recovery + distribution) on a calibration
slice of its home community, emitting an optimized {name: rubric} map for the rep4 confirm.

  python scripts/tools/gepa_optimize_rubrics.py --stage2 outputs/ctree/stage2/w3-cw/stage2_ledger.json \
      --dense outputs/ctree/foundry/dense_cw.json --data-template ... --leg-prefix cw-genre- \
      --id-col prompt --judge-model <path> --out .../dense_cw_gepa.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "methods")

import pandas as pd

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.io_metrics import make_vllm_judge_scorer
from metrics_tree_infilling.operationalize import operationalize_rubric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--dense", required=True)
    ap.add_argument("--data-template", required=True)
    ap.add_argument("--leg-prefix", required=True)
    ap.add_argument("--leg-suffix", default="")
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--proposer-model", default="glm-5.2")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-cal", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cands = [r for r in json.load(open(args.stage2))
             if (r.get("rep_p_auc") or 1) < args.alpha or (r.get("rep_p_bits") or 1) < args.alpha
             or r.get("stage2_status") in ("KEPT", "degenerate")]
    dense = json.load(open(args.dense)) if Path(args.dense).exists() else {}
    if not cands:
        json.dump({}, open(args.out, "w"))
        print("no candidates; wrote empty map", flush=True)
        return

    cfg = InfillConfig(materialize_backend="vllm_offline", materialize_model=args.judge_model,
                       max_text_tokens=700, verbose=False,
                       cache_dir="outputs/ctree/B_tree/judge_cache",
                       output_dir=str(Path(args.out).parent))
    judge = make_vllm_judge_scorer(cfg)

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                 base_url=os.environ.get("ANTHROPIC_BASE_URL"))

    def proposer(prompt):
        import time
        for wait in (0, 20, 60):
            if wait:
                time.sleep(wait)
            try:
                r = client.messages.create(model=args.proposer_model, max_tokens=900,
                                           temperature=0.4,
                                           messages=[{"role": "user", "content": prompt}])
                return "".join(b.text for b in r.content if hasattr(b, "text"))
            except Exception:
                continue
        return None

    out_map = json.load(open(args.out)) if Path(args.out).exists() else {}
    for c in cands:
        if c["name"] in out_map:
            continue
        community = c["leg"].replace(args.leg_prefix, "")
        if args.leg_suffix and community.endswith(args.leg_suffix):
            community = community[: -len(args.leg_suffix)]
        df = pd.read_csv(args.data_template.format(community=community)).dropna(subset=["text"])
        cal = df.sample(min(args.n_cal, len(df)), random_state=99).text.astype(str).tolist()
        rubric0 = dense.get(c["name"], c.get("rubric", ""))
        res = operationalize_rubric(c["name"], c.get("description", ""), rubric0,
                                    cal, judge, proposer, cfg)
        out_map[c["name"]] = res.rubric
        json.dump(out_map, open(args.out, "w"), indent=1)
        print(f"[{c['name'][:48]:48s}] iters={res.iterations} retest={res.retest:.2f} "
              f"recovery={res.recovery:.2f}", flush=True)
    print(f"WROTE {len(out_map)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
