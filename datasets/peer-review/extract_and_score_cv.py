#!/usr/bin/env python3
"""Claim-verification: extract fields from ABSTRACT (Gemma-4), then verify against BODY (code).

Loads cv1/cv2/cv3 modules, extracts their LLM_FIELDS from each abstract, then scores each module
against the body (pure-Python verification: do the claim's numbers/baselines appear in results?).
Reports per-cv AUC vs accept/reject + saves the score matrix.

Usage (gemma4 env, 1 GPU):
  CUDA_VISIBLE_DEVICES=2 python extract_and_score_cv.py --evidence peer_review_cv_evidence.jsonl
"""
import argparse, importlib.util, json, pathlib, re, sys
import numpy as np
from vllm import LLM, SamplingParams
from sklearn.metrics import roc_auc_score

BASE = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
PROG = BASE / "methods/metric_seam/hybrids/programs_peer_review"
sys.path.insert(0, str(BASE / "methods/metric_seam/hybrids"))
from ops import Ops  # noqa
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
CVIDS = ["cv1", "cv2", "cv3"]
CVFILES = {"cv1": "cv1_supported_h0", "cv2": "cv2_beats_baselines_h0", "cv3": "cv3_has_evidence_h0"}
LABELS = {"cv1": "whether the headline claim is supported in the body",
          "cv2": "whether the claim of beating baselines is substantiated",
          "cv3": "whether the body carries evidence for the claim"}

def load_mod(aid):
    spec = importlib.util.spec_from_file_location(aid, PROG / f"{CVFILES[aid]}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

SYS = ("You extract structured fields from a research-paper ABSTRACT. Give the shortest faithful "
       "answer, or NONE if the abstract gives no evidence. Reply with ONE compact JSON object.")
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
def parse_json(raw):
    if not raw: return {}
    m = _OBJ_RE.search(raw)
    if not m: return {}
    try: return json.loads(m.group(0))
    except:
        try: return json.loads(re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", m.group(0))))
        except: return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="peer_review_cv_evidence.jsonl")
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--out", default=str(BASE / "datasets/peer-review/peer_review_cv_scores.npz"))
    a = ap.parse_args()

    mods = {aid: load_mod(aid) for aid in CVIDS}
    rows = [json.loads(l) for l in open(a.evidence) if l.strip()]
    print(f"[cv] {len(rows)} papers x {len(CVIDS)} cv-metrics (extract from abstract, verify vs body)", flush=True)

    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util, max_model_len=4096,
              enable_prefix_caching=True, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=160)
    convs, key = [], []
    for r in rows:
        ab = r["abstract"][:5000]
        for aid in CVIDS:
            qs = "\n".join(f'"{k}": {v}' for k, v in mods[aid].LLM_FIELDS.items())
            convs.append([{"role": "user", "content":
                f"{SYS}\n\nABSTRACT:\n{ab}\n\nAssessing {LABELS[aid]}. Extract:\n{qs}\n\nReply JSON only."}])
            key.append((r["paper_id"], aid))
    print(f"[cv] extracting {len(convs)} on Gemma-4-31B ...", flush=True)
    outs = llm.chat(convs, sp)
    by_id = {}
    for (pid, aid), o in zip(key, outs):
        obj = parse_json(o.outputs[0].text)
        for f in mods[aid].LLM_FIELDS:
            obj.setdefault(f, "NONE")
        by_id.setdefault(pid, {})[aid] = obj

    # score: verify against BODY (ops used only for normalize; no retrieval needed here)
    ops = Ops()
    y = np.array([r["y"] for r in rows], dtype=int)
    X = np.full((len(rows), len(CVIDS)), np.nan)
    for j, aid in enumerate(CVIDS):
        fn = mods[aid].score
        for i, r in enumerate(rows):
            try:
                X[i, j] = float(fn(r["body"], by_id.get(r["paper_id"], {}).get(aid, {}), ops))
            except Exception:
                X[i, j] = np.nan

    print(f"[cv] na={np.isnan(X).mean():.3f}")
    print(f"{'cv':5s} {'mean':>6s} {'std':>6s} {'AUC':>6s} {'n':>5s}")
    for j, aid in enumerate(CVIDS):
        col = X[:, j]; mask = ~np.isnan(col)
        auc = roc_auc_score(y[mask], col[mask]) if mask.sum() > 30 and len(set(y[mask].tolist())) == 2 else float("nan")
        print(f"{aid:5s} {col[mask].mean() if mask.sum() else float('nan'):6.3f} "
              f"{col[mask].std() if mask.sum() else float('nan'):6.3f} {auc:6.3f} {int(mask.sum()):5d}")
    np.savez_compressed(a.out, X=X, y=y, ids=np.array([r["paper_id"] for r in rows], dtype=object),
                        cv_names=np.array(CVIDS, dtype=object))
    print(f"[cv] saved -> {a.out}", flush=True); print("CV_DONE", flush=True)

if __name__ == "__main__":
    main()
