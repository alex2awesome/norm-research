#!/usr/bin/env python3
"""Layer-3 closure round 2, Stage 3: score the 25 new criteria over the FULL
6,030-row peer-VERDICT population with Gemma-4-31B, offline-batch vLLM on sk3.

Reuses the vat_3y scoring machinery (datasets/peer-review/vat_3y/score_va_gemma_3y.py):
same model snapshot, same offline `llm.chat` batch call, same temperature-0 single-token
readout, same 5,000-char abstract truncation.  Two deliberate changes:
  * scale is 0-10 (round-1 criteria are authored on a 0-10 scale) rather than 0/.5/1;
  * a blinded anchor battery (pos / neg / scrambled, K per class) is appended to the
    SAME batch, per the standing every-batch anchor rule.

Both splits are scored; MONITOR rows are simply never read by a proposer.
Anchor labels are used only inside this script -- the proposer context never saw them.

GPU0 only.  Run:
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=0 nohup $HOME/envs/gemma4/bin/python score_round1_gemma.py > score_round1.log 2>&1 &
"""
import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import numpy as np  # noqa: E402

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
HERE = ROOT / "methods" / "taste_decomposition" / "closure"
GEMMA4 = (
    "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/"
    "snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
)

SYS = (
    "You are an expert academic peer reviewer. You are given a paper's ABSTRACT and ONE "
    "criterion. Decide how strongly the abstract, on its own evidence, exhibits that "
    "criterion. Answer with EXACTLY ONE token:\n"
    "  an integer from 0 to 10, where 0 = not at all and 10 = to the fullest degree\n"
    "  NA = the abstract gives no evidence bearing on this criterion\n"
    "Judge the criterion as literally described, not whether the paper will be accepted. "
    "Output only the token."
)

NUM = re.compile(r"\d+")
K_ANCHOR = 12
SEED = 20260807


def parse_tok(t):
    t = (t or "").strip()
    low = t.lower()
    if low.startswith("na") or "n/a" in low:
        return np.nan
    m = NUM.search(t)
    if not m:
        return np.nan
    v = float(m.group(0))
    return v if 0.0 <= v <= 10.0 else np.nan


def scramble(texts, rng):
    words = " ".join(texts).split()
    rng.shuffle(words)
    return " ".join(words[:220])


def main():
    from vllm import LLM, SamplingParams

    blind = json.loads((HERE / "round3_proposals_blinded.json").read_text())
    crits = blind["criteria"]
    blocks = [
        f"CRITERION: {c['name']}\nINSTRUCTION: {c['instruction']}\n\nAnswer with one token:"
        for c in crits
    ]
    cids = [c["id"] for c in crits]

    import csv

    rows = []
    with open(HERE / "peer_verdict_population.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    texts = [r["text"] for r in rows]
    print(f"[closure-r3] {len(rows)} abstracts x {len(crits)} criteria = "
          f"{len(rows) * len(crits)} prompts", flush=True)

    # ---- blinded anchor battery (pos / neg / scrambled), K per class -----------
    rng = random.Random(SEED)
    pos_pool = [r["text"] for r in rows if r["judgement"] == "1"]
    neg_pool = [r["text"] for r in rows if r["judgement"] == "0"]
    a_texts, a_tags = [], []
    for j in range(K_ANCHOR):
        p, n = rng.choice(pos_pool), rng.choice(neg_pool)
        a_texts += [p, n, scramble([p, n], rng)]
        a_tags += ["anchor_pos", "anchor_neg", "anchor_scram"]
    print(f"[closure-r3] anchors {len(a_texts)} rows x {len(crits)} = "
          f"{len(a_texts) * len(crits)} prompts", flush=True)

    all_texts = texts + a_texts
    convs = []
    for t in all_texts:
        f = t[:5000]
        for b in blocks:
            convs.append([{"role": "user", "content": f"{SYS}\n\nABSTRACT:\n{f}\n\n{b}"}])

    util = float(os.environ.get("GEMMA_UTIL", "0.85"))
    print(f"[closure-r3] gpu_memory_utilization={util}", flush=True)
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=util,
              max_model_len=4096, enable_prefix_caching=True, trust_remote_code=True,
              max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=6)
    print(f"[closure-r3] scoring {len(convs)} prompts ...", flush=True)
    outs = llm.chat(convs, sp)
    raw = [o.outputs[0].text for o in outs]
    X = np.array([parse_tok(t) for t in raw], dtype=float).reshape(len(all_texts), len(crits))

    Xpop, Xanc = X[: len(texts)], X[len(texts):]
    np.savez_compressed(
        HERE / "round3_scores.npz",
        X=Xpop,
        crit_ids=np.array(cids, dtype=object),
        crit_names=np.array([c["name"] for c in crits], dtype=object),
        i=np.array([int(r["i"]) for r in rows]),
        ntitle=np.array([r["ntitle"] for r in rows], dtype=object),
        Xanchor=Xanc,
        anchor_tags=np.array(a_tags, dtype=object),
        scale="0-10",
    )

    # ---- collapse check + anchor readout --------------------------------------
    rep = {"n_rows": int(len(texts)), "n_criteria": len(crits), "per_criterion": {}}
    for k, cid in enumerate(cids):
        col = Xpop[:, k]
        ok = col[~np.isnan(col)]
        vals, counts = (np.unique(ok, return_counts=True) if len(ok) else (np.array([]), np.array([])))
        rep["per_criterion"][cid] = {
            "name": crits[k]["name"],
            "na_rate": float(np.isnan(col).mean()),
            "mean": float(np.mean(ok)) if len(ok) else None,
            "std": float(np.std(ok)) if len(ok) else None,
            "n_distinct": int(len(vals)),
            "modal_value": float(vals[np.argmax(counts)]) if len(vals) else None,
            "modal_frac": float(counts.max() / len(ok)) if len(ok) else None,
            "value_counts": {str(v): int(c) for v, c in zip(vals, counts)},
            "collapsed": bool(len(ok) == 0 or len(vals) <= 1 or counts.max() / len(ok) > 0.98),
        }
    from sklearn.metrics import roc_auc_score

    tags = np.array(a_tags)
    item = np.nanmean(Xanc, axis=1)
    anc = {"k_per_class": K_ANCHOR}
    for t in ("anchor_pos", "anchor_neg", "anchor_scram"):
        v = item[tags == t]
        anc[t] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1))}
    pv, nv, sv = item[tags == "anchor_pos"], item[tags == "anchor_neg"], item[tags == "anchor_scram"]
    anc["pos_vs_neg_auc"] = float(roc_auc_score([1] * len(pv) + [0] * len(nv), np.concatenate([pv, nv])))
    anc["coherent_vs_scrambled_auc"] = float(
        roc_auc_score([1] * (len(pv) + len(nv)) + [0] * len(sv), np.concatenate([pv, nv, sv]))
    )
    anc["pass_scrambled"] = bool(anc["coherent_vs_scrambled_auc"] >= 0.70)
    rep["anchors"] = anc
    rep["n_collapsed"] = int(sum(v["collapsed"] for v in rep["per_criterion"].values()))
    rep["overall_na_rate"] = float(np.isnan(Xpop).mean())
    (HERE / "round3_score_report.json").write_text(json.dumps(rep, indent=2))
    print("ANCHORS " + json.dumps(anc), flush=True)
    print("COLLAPSED " + str(rep["n_collapsed"]) + " NA " + f"{rep['overall_na_rate']:.3f}", flush=True)
    print("ROUND3_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
