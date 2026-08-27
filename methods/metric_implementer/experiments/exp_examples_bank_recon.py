"""EXP-EXAMPLES-BANK-1 rescore under the RECONSTRUCTION OBJECTIVE — full round trip
(2026-08-20, user directive: no reference keys; the objective is I(m; m') through a decoder).

Per metric x arm x judge:
  m  = the judge's saved holdout verdicts under the arm message (ebank_eval_<judge>.jsonl,
       p_yes — the transmitter side; already collected, NOT redone).
  DECODER: the same model sees ONLY labeled excerpts (a 60-item demonstration slice of the
       holdout, formatted by recon_channel._balanced_examples) and writes a rule p'
       (recon_channel.induce_free / _RECON_FREE — the Section-3 machinery, verbatim).
  RE-EXECUTE: p' labels the REMAINING holdout items (never shown to the decoder) -> m'.
  SCORE (score mode, CPU): vinfo.binary_soft_channel_mi(m, m') Shannon bits, the paper's
       fixed-target recovery object. delta_mi = MI(functional) - MI(definition);
       functional_null = the reconstruction-side noise floor.

Modes:
  run    (sk3, --judge llama70b|qwen25-72b, TP=2): -> <out>/ebank_recon_<judge>.jsonl
         (p' text + m + m' vectors per arm; resumable per metric; flush per row)
  score  (CPU, laptop): -> outputs/exp_examples_bank/bank_examples_mi_v1.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

from .exp_examples_bank import _task_setup, _Scorer, TASKS, load_npz  # noqa: F401
from ..recon_channel import induce_free, _balanced_examples

ARMS = ["name", "definition", "functional", "functional_null"]
DEMO_N = 60          # holdout items shown to the decoder (excluded from MI)
L_CAP = 450          # articulation budget for the reconstructed rule (induce_free default)
MIN_MI_N = 60        # minimum scoreable MI items per arm
NOUN = {t: t.replace("_", " ") + " excerpt" for t in
        ["humor", "creative_writing", "math", "peer_review", "press_releases",
         "news_homepages", "notice_and_comment", "patents"]}


def _seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2**31)


def run(a):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from .. import config as cfgmod
    from ..vllm_backend import make_judge_backend
    from .osl_sweep import EXECUTORS
    ev = [json.loads(l) for l in open(os.path.join(a.out, f"ebank_eval_{a.judge}.jsonl"))]
    rf = os.path.join(a.out, f"ebank_recon_{a.judge}.jsonl")
    done = {(json.loads(l)["task"], json.loads(l)["name"])
            for l in open(rf)} if os.path.exists(rf) else set()
    cfg = cfgmod.ImplementerConfig()
    cfg.vllm_tp_size = int(os.environ.get("VOICE_TP", "1"))
    backend = make_judge_backend(EXECUTORS[a.judge][0], cfg, temperature=None)
    fout = open(rf, "a")
    for task in TASKS:
        todo = [r for r in ev if r["task"] == task and (task, r["name"]) not in done]
        if not todo:
            continue
        probes, mx, IA, IB, IH = _task_setup(task, a.root)
        sc = _Scorer(backend, probes, mx)
        for r in todo:
            ih = np.asarray(r["ih_items"], int)
            masked = set(r["masked_items"])
            kept = [k for k, i in enumerate(ih) if i not in masked]
            demo_pos, mi_pos = kept[:DEMO_N], kept[DEMO_N:]
            mi_items = [int(ih[k]) for k in mi_pos]
            out = {"task": task, "name": r["name"], "category": r["category"],
                   "judge": a.judge, "mi_items": mi_items, "arms": {}}
            # decode all arms in one batch, then re-execute all p' in one scorer call
            arm_list = [arm for arm in ARMS if arm in r["p_yes"]]
            dem, prompts_meta = [], []
            for arm in arm_list:
                pv = np.asarray([np.nan if v is None else v for v in r["p_yes"][arm]], float)
                dfin = [k for k in demo_pos if np.isfinite(pv[k])]
                if len(dfin) < 20 or sum(np.isfinite(pv[k]) for k in mi_pos) < MIN_MI_N:
                    continue
                ex_str = _balanced_examples([probes[int(ih[k])] for k in dfin],
                                            pv[dfin], k=30, max_chars=600)
                dem.append((arm, pv, ex_str))
            for arm, pv, ex_str in dem:
                p_prime = induce_free(backend, NOUN[task], ex_str,
                                      _seed(task, r["name"], arm, a.judge),
                                      max_tokens=L_CAP)["rubric"]
                if not p_prime:
                    continue
                prompts_meta.append((arm, pv, p_prime))
            if prompts_meta:
                mats = sc.score_rubrics([p for _, _, p in prompts_meta], np.asarray(mi_items))
                for (arm, pv, p_prime), mprime in zip(prompts_meta, mats):
                    out["arms"][arm] = {
                        "p_prime": p_prime,
                        "m": [None if not np.isfinite(pv[k]) else round(float(pv[k]), 4)
                              for k in mi_pos],
                        "m_prime": [None if v != v else round(float(v), 4) for v in mprime]}
            fout.write(json.dumps(out) + "\n")
            fout.flush()
            print(f"[recon|{a.judge}|{task}] {r['name'][:44]:44s} "
                  f"arms={list(out['arms'])} cum_calls={sc.n_calls:,}", flush=True)
    fout.close()
    print(f"[recon|{a.judge}] DONE", flush=True)


def score(a):
    from ..vinfo import binary_soft_channel_mi
    per = []
    for judge in ("llama70b", "qwen25-72b"):
        f = os.path.join(a.out, f"ebank_recon_{judge}.jsonl")
        if not os.path.exists(f):
            continue
        for line in open(f):
            r = json.loads(line)
            row = {"task": r["task"], "name": r["name"], "category": r["category"],
                   "judge": r["judge"], "mi": {}, "h_m": {}, "n": {}}
            for arm, d in r["arms"].items():
                m = np.asarray([np.nan if v is None else v for v in d["m"]], float)
                mp = np.asarray([np.nan if v is None else v for v in d["m_prime"]], float)
                ok = np.isfinite(m) & np.isfinite(mp)
                if ok.sum() < MIN_MI_N:
                    continue
                mi = binary_soft_channel_mi(m[ok], mp[ok])["shannon"]
                qb = float(np.mean(m[ok]))
                hm = -(qb * np.log2(qb) + (1 - qb) * np.log2(1 - qb)) if 0 < qb < 1 else 0.0
                row["mi"][arm] = round(mi, 4)
                row["h_m"][arm] = round(hm, 4)
                row["n"][arm] = int(ok.sum())
            row["delta_mi"] = (round(row["mi"]["functional"] - row["mi"]["definition"], 4)
                               if "functional" in row["mi"] and "definition" in row["mi"]
                               else None)
            row["delta_mi_null"] = (round(row["mi"]["functional_null"] - row["mi"]["definition"], 4)
                                    if "functional_null" in row["mi"] and "definition" in row["mi"]
                                    else None)
            per.append(row)

    def tab(field, dkey):
        groups = {}
        for r in per:
            if r.get(dkey) is not None:
                groups.setdefault(r[field], {}).setdefault(r["name"], []).append(r[dkey])
        out = {}
        rng = np.random.default_rng(0)
        for g, by in sorted(groups.items()):
            vals = np.array([np.mean(v) for v in by.values()])
            boots = [float(np.mean(rng.choice(vals, len(vals)))) for _ in range(2000)]
            out[g] = {"mean": round(float(vals.mean()), 4),
                      "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                               round(float(np.percentile(boots, 97.5)), 4)],
                      "n_metrics": len(vals)}
        return out

    art = {"experiment": "EXP-EXAMPLES-BANK-1 reconstruction-objective rescore (full round trip)",
           "objective": "I(m; m') Shannon bits via binary_soft_channel_mi; m = judge verdicts "
                        "under the arm message; m' = verdicts under the decoder's reconstruction "
                        "p' (induce_free on a 60-item demonstration slice, MI on the disjoint "
                        "remainder); NO reference keys anywhere",
           "keys_quarantined": "data/DO_NOT_USE__reference_keys/ (2026-08-20)",
           "n_rows": len(per),
           "mi_by_arm_pooled": {arm: round(float(np.mean([r["mi"][arm] for r in per
                                                          if arm in r["mi"]])), 4)
                                for arm in ARMS},
           "delta_mi_by_category": tab("category", "delta_mi"),
           "delta_mi_by_task": tab("task", "delta_mi"),
           "null_floor_by_category": tab("category", "delta_mi_null"),
           "per_metric": per}
    dst = os.path.join(a.root, "outputs/exp_examples_bank/bank_examples_mi_v1.json")
    json.dump(art, open(dst, "w"), indent=1)
    print(f"[score] {len(per)} rows -> {dst}")
    print(" delta_mi by category:", {c: v["mean"] for c, v in art["delta_mi_by_category"].items()})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "score"])
    p.add_argument("--root", default=".")
    p.add_argument("--out", required=True)
    p.add_argument("--judge")
    a = p.parse_args(argv)
    if a.mode == "run" and not a.judge:
        sys.exit("run needs --judge")
    {"run": run, "score": score}[a.mode](a)


if __name__ == "__main__":                                   # spawn-safety (sk3 vLLM rule)
    main()
