#!/usr/bin/env python3
"""Style Invitational v2: SIGN-CORRECTED anchor certification.

WHY THIS EXISTS. The shared anchor protocol in `score_va_gemma_banks.score_bank`
and `score_scaleupC_banks.run_battery` certifies a shard by taking the
UNWEIGHTED MEAN of an anchor row's scores across every criterion and requiring
pos > neg > scrambled. That is correct for a bank whose criteria all point the
same way. **It is wrong for this bank**, which by design contains 8
negatively-oriented criteria where 1.0 marks a FLAW: a better entry is supposed
to score LOW on those, so averaging them in unsigned partially cancels the very
contrast the anchor test is trying to see.

The symptom was visible immediately: shard 0 needed 4 anchor draws and passed by
a margin of .002 (pos .517 / neg .515 / scram .269). That is the protocol
mis-measuring a mixed-orientation bank, not the judge failing.

THE CORRECTION. Score each anchor row as a QUALITY mean with negative criteria
flipped:  contribution = value if orientation is positive, else (1 - value).
Surface (Track B) criteria are excluded from the quality mean entirely -- they
are declared nuisance and carry no quality direction.

TWO PASSES, and the first needs no GPU:
  1. `--shards`  recomputes every shard's 3-row blinded anchor ordering from the
     `anchor_X` array already saved inside each shard npz. Zero new judge calls;
     temperature-0 scores cannot change, so this is a pure re-read of evidence
     already collected.
  2. `--battery K`  runs a fresh K>=50-per-class extended battery with the same
     sign correction (150 rows x 36 criteria ~ 5.4K prompts, about a minute).

Both the corrected and the raw uncorrected statistics are reported side by side,
so the record shows what the standard protocol said and why it is superseded here.

  python3 datasets/humor/style_invitational/si_v2_anchor_recheck.py --shards
  CUDA_VISIBLE_DEVICES=N python3 ... --battery 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_HERE = Path(__file__).resolve()
REPO_GUESS = _HERE.parents[3]
sys.path.insert(0, str(REPO_GUESS / "datasets/va_gemma_banks"))
sys.path.insert(0, str(_HERE.parent))

OUT_DEFAULT = REPO_GUESS / "outputs/va_gemma_banks_si_v2"
V2 = REPO_GUESS / "datasets/humor/style_invitational/va_v2"


def orientations():
    rub = [json.loads(l) for l in open(V2 / "rubrics.jsonl") if l.strip()]
    return [(r["name"], r["track"], r["orientation"]) for r in rub]


def quality_mean(X, orient):
    """Row-wise quality mean with negative criteria flipped and Track B dropped."""
    X = np.asarray(X, dtype=float)
    cols, signs = [], []
    for j, (_, track, o) in enumerate(orient):
        if track != "A":
            continue
        cols.append(j)
        signs.append(-1.0 if o == "negative" else 1.0)
    sub = X[:, cols].copy()
    for k, s in enumerate(signs):
        if s < 0:
            sub[:, k] = 1.0 - sub[:, k]
    with np.errstate(invalid="ignore"):
        return np.nanmean(sub, axis=1)


def raw_mean(X):
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.asarray(X, dtype=float), axis=1)


def do_shards(out: Path):
    orient = orientations()
    rows = []
    si = 0
    while (out / f"si_v2_shard{si}.npz").exists():
        z = np.load(out / f"si_v2_shard{si}.npz", allow_pickle=True)
        aX = z["anchor_X"]                      # (3, n_criteria): pos, neg, scram
        q = quality_mean(aX, orient)
        r = raw_mean(aX)
        rows.append({
            "shard": si,
            "corrected": {"pos": float(q[0]), "neg": float(q[1]),
                          "scram": float(q[2]),
                          "ordering_holds": bool(q[0] > q[1] > q[2]),
                          "pos_minus_neg": float(q[0] - q[1])},
            "raw_uncorrected": {"pos": float(r[0]), "neg": float(r[1]),
                                "scram": float(r[2]),
                                "ordering_holds": bool(r[0] > r[1] > r[2]),
                                "pos_minus_neg": float(r[0] - r[1])},
            "shipped_report": json.loads(str(z["anchor_json"].item())),
        })
        si += 1
    n_ok_c = sum(1 for x in rows if x["corrected"]["ordering_holds"])
    n_ok_r = sum(1 for x in rows if x["raw_uncorrected"]["ordering_holds"])
    res = {"n_shards": len(rows),
           "n_ordering_holds_CORRECTED": n_ok_c,
           "n_ordering_holds_raw": n_ok_r,
           "mean_pos_minus_neg_CORRECTED": float(np.mean(
               [x["corrected"]["pos_minus_neg"] for x in rows])),
           "mean_pos_minus_neg_raw": float(np.mean(
               [x["raw_uncorrected"]["pos_minus_neg"] for x in rows])),
           "shards": rows}
    p = out / "anchor_recheck_shards.json"
    p.write_text(json.dumps(res, indent=1))
    print(f"per-shard 3-row anchors: ordering holds {n_ok_c}/{len(rows)} CORRECTED "
          f"vs {n_ok_r}/{len(rows)} raw")
    print(f"  mean (pos - neg): {res['mean_pos_minus_neg_CORRECTED']:+.4f} corrected "
          f"/ {res['mean_pos_minus_neg_raw']:+.4f} raw")
    for x in rows:
        c, r = x["corrected"], x["raw_uncorrected"]
        print(f"  shard {x['shard']}: corrected {c['pos']:.3f}/{c['neg']:.3f}/"
              f"{c['scram']:.3f} {'PASS' if c['ordering_holds'] else 'FAIL'}   "
              f"raw {r['pos']:.3f}/{r['neg']:.3f}/{r['scram']:.3f} "
              f"{'PASS' if r['ordering_holds'] else 'FAIL'}")
    return res


def do_battery(out: Path, k: int, util: float, max_model_len: int):
    import multiprocessing as _mp
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    import score_va_gemma_banks as S
    from sklearn.metrics import roc_auc_score
    import score_si_v2_bank as B
    from transformers import AutoTokenizer

    orient = orientations()
    tok = AutoTokenizer.from_pretrained(S.GEMMA4)
    bank = B.build_si_v2(tok)

    rows, tags = [], []
    for j in range(k):
        for r in bank["anchors"](900_000 + j):
            rows.append(r)
            tags.append(r["anchor_tag"])
    from vllm import LLM, SamplingParams
    llm = LLM(model=S.GEMMA4, dtype="bfloat16", gpu_memory_utilization=util,
              max_model_len=max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=512)
    sp = SamplingParams(temperature=0.0, max_tokens=6)
    convs = []
    for r in rows:
        c = bank["ctx"](r)
        for blk in bank["blocks"]:
            convs.append([{"role": "user",
                           "content": f"{bank['sys']}\n\n{c}\n\n{blk}"}])
    print(f"[battery] {len(rows)} anchors x {len(bank['blocks'])} = {len(convs)} prompts",
          flush=True)
    outs = llm.chat(convs, sp)
    X = np.array([S.parse_tok(o.outputs[0].text) for o in outs],
                 dtype=float).reshape(len(rows), len(bank["blocks"]))
    tags = np.array(tags)

    res = {"k_per_class": k}
    for label, fn in (("corrected", lambda M: quality_mean(M, orient)),
                      ("raw_uncorrected", raw_mean)):
        m = fn(X)
        ok = np.isfinite(m)
        mm, tt = m[ok], tags[ok]
        d = {"n_all_NA_dropped": int((~ok).sum())}
        for t in ("anchor_pos", "anchor_neg", "anchor_scram"):
            v = mm[tt == t]
            d[t] = {"n": int(len(v)), "mean": float(np.mean(v)) if len(v) else float("nan"),
                    "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")}
        pv, nv = mm[tt == "anchor_pos"], mm[tt == "anchor_neg"]
        sv = mm[tt == "anchor_scram"]
        d["pos_vs_neg_auc"] = float(roc_auc_score(
            [1] * len(pv) + [0] * len(nv), np.concatenate([pv, nv])))
        d["coherent_vs_scrambled_auc"] = float(roc_auc_score(
            [1] * (len(pv) + len(nv)) + [0] * len(sv), np.concatenate([pv, nv, sv])))
        d["ordering_holds_on_means"] = bool(
            d["anchor_pos"]["mean"] > d["anchor_neg"]["mean"] > d["anchor_scram"]["mean"])
        res[label] = d
        print(f"[battery:{label}] pos {d['anchor_pos']['mean']:.4f} / "
              f"neg {d['anchor_neg']['mean']:.4f} / scram {d['anchor_scram']['mean']:.4f} "
              f"| pos-vs-neg AUC {d['pos_vs_neg_auc']:.3f} "
              f"| coherent-vs-scram {d['coherent_vs_scrambled_auc']:.3f} "
              f"| ordering {d['ordering_holds_on_means']}", flush=True)
    p = out / "anchor_battery_signcorrected.json"
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p)
    print("SI_V2_BATTERY_DONE", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--shards", action="store_true")
    ap.add_argument("--battery", type=int, default=0)
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    a = ap.parse_args()
    out = Path(a.out)
    if a.shards:
        do_shards(out)
    if a.battery:
        do_battery(out, a.battery, a.util, a.max_model_len)


if __name__ == "__main__":
    main()
