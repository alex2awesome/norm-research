#!/usr/bin/env python3
"""Stage 3 of any round: score the round's 25 criteria over the FULL 9,521-row
N&C RESPONDED population with Gemma-4-31B, offline-batch vLLM on sk3.

Reuses the cell's own scoring machinery (datasets/notice-and-comment/v4/
score_va_gemma_nc.py): same model snapshot, same offline `llm.chat` batch call,
same temperature-0 single-token readout, same 4,000-char comment truncation, same
regulatory-analyst system framing.  Two deliberate changes, both frozen:
  * scale is 0-10 (closure criteria are authored on a 0-10 scale) rather than
    1.0/0.5/0.0;
  * a blinded anchor battery (pos / neg / scrambled) with K >= 50 per class is
    appended to the SAME batch (freeze: "anchors K>=50/class").

Both splits are scored; MONITOR rows are simply never read by a proposer.
Anchor labels are used only inside this script.

Run (see run_round_when_free.sh):
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=<gid> GEMMA_UTIL=<u> ROUND=<r> python score_round_gemma.py
"""
import csv
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
HERE = ROOT / "methods" / "taste_decomposition" / "closure" / "nc_responded"
GEMMA4 = (
    "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/"
    "snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
)

SYS = (
    "You are an expert regulatory analyst reviewing PUBLIC COMMENTS submitted on proposed "
    "federal rules. You are given one public comment and ONE criterion. Decide how strongly "
    "the comment, on its own evidence, exhibits that criterion. Answer with EXACTLY ONE "
    "token:\n"
    "  an integer from 0 to 10, where 0 = not at all and 10 = to the fullest degree\n"
    "  NA = the comment gives no evidence bearing on this criterion\n"
    "Judge the criterion as literally described, not whether the comment's position is "
    "correct and not whether the agency will respond to it. Output only the token."
)

NUM = re.compile(r"\d+")
K_ANCHOR = 50          # freeze: anchors K >= 50 per class
SEED = 20260806
TRUNC = 4000


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

    r = int(os.environ["ROUND"])
    blind = json.loads((HERE / f"round{r}_proposals_blinded.json").read_text())
    crits = blind["criteria"]
    blocks = [
        f"CRITERION: {c['name']}\nINSTRUCTION: {c['instruction']}\n\nAnswer with one token:"
        for c in crits
    ]
    cids = [c["id"] for c in crits]

    rows = []
    with open(HERE / "nc_responded_population.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    texts = [row["text"] or "" for row in rows]
    print(f"[nc-closure-r{r}] {len(rows)} comments x {len(crits)} criteria = "
          f"{len(rows) * len(crits)} prompts", flush=True)

    rng = random.Random(SEED)
    pos_pool = [row["text"] for row in rows if row["y"] == "1"]
    neg_pool = [row["text"] for row in rows if row["y"] == "0"]
    a_texts, a_tags = [], []
    for _ in range(K_ANCHOR):
        p, n = rng.choice(pos_pool), rng.choice(neg_pool)
        a_texts += [p, n, scramble([p, n], rng)]
        a_tags += ["anchor_pos", "anchor_neg", "anchor_scram"]
    print(f"[nc-closure-r{r}] anchors {len(a_texts)} rows x {len(crits)} = "
          f"{len(a_texts) * len(crits)} prompts", flush=True)

    all_texts = texts + a_texts
    convs = []
    for t in all_texts:
        f = (t or "")[:TRUNC]
        for b in blocks:
            convs.append([{"role": "user", "content": f"{SYS}\n\nPUBLIC COMMENT:\n{f}\n\n{b}"}])

    util = float(os.environ.get("GEMMA_UTIL", "0.85"))
    print(f"[nc-closure-r{r}] gpu_memory_utilization={util}", flush=True)
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=util,
              max_model_len=4096, enable_prefix_caching=True, trust_remote_code=True,
              max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=6)
    print(f"[nc-closure-r{r}] scoring {len(convs)} prompts ...", flush=True)
    outs = llm.chat(convs, sp)
    raw = [o.outputs[0].text for o in outs]
    X = np.array([parse_tok(t) for t in raw], dtype=float).reshape(len(all_texts), len(crits))

    Xpop, Xanc = X[: len(texts)], X[len(texts):]
    np.savez_compressed(
        HERE / f"round{r}_scores.npz",
        X=Xpop,
        crit_ids=np.array(cids, dtype=object),
        crit_names=np.array([c["name"] for c in crits], dtype=object),
        i=np.array([int(row["i"]) for row in rows]),
        doc_id=np.array([row["doc_id"] for row in rows], dtype=object),
        Xanchor=Xanc,
        anchor_tags=np.array(a_tags, dtype=object),
        scale="0-10",
    )

    rep = {"round": r, "n_rows": int(len(texts)), "n_criteria": len(crits), "per_criterion": {}}
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
    # per-criterion anchor separation (the K>=50 battery the pilot could not afford)
    per_crit_anchor = {}
    for k, cid in enumerate(cids):
        cp, cn, cs = Xanc[tags == "anchor_pos", k], Xanc[tags == "anchor_neg", k], Xanc[tags == "anchor_scram", k]
        try:
            pn = float(roc_auc_score([1] * len(cp) + [0] * len(cn),
                                     np.nan_to_num(np.concatenate([cp, cn]), nan=-1)))
            cvs = float(roc_auc_score([1] * (len(cp) + len(cn)) + [0] * len(cs),
                                      np.nan_to_num(np.concatenate([cp, cn, cs]), nan=-1)))
        except ValueError:
            pn = cvs = float("nan")
        per_crit_anchor[cid] = {"pos_vs_neg_auc": pn, "coherent_vs_scrambled_auc": cvs}
    anc["per_criterion"] = per_crit_anchor
    rep["anchors"] = anc
    rep["n_collapsed"] = int(sum(v["collapsed"] for v in rep["per_criterion"].values()))
    rep["overall_na_rate"] = float(np.isnan(Xpop).mean())
    (HERE / f"round{r}_score_report.json").write_text(json.dumps(rep, indent=2))
    print("ANCHORS " + json.dumps({k: v for k, v in anc.items() if k != "per_criterion"}), flush=True)
    print("COLLAPSED " + str(rep["n_collapsed"]) + " NA " + f"{rep['overall_na_rate']:.3f}", flush=True)
    print(f"ROUND{r}_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
