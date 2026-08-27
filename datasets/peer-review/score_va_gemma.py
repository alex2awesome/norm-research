#!/usr/bin/env python3
"""Peer-review V/A scorer (Gemma-4-31B offline batch vLLM).

Expert-verdict Y = accept/reject (judgement) on paper ABSTRACTS (splits/*.csv.gz).

  V metrics = programmatic thin/surface features of the abstract (no GPU).
  A metrics = 154 merged quality-criterion rubrics (data/peer_review/rubrics.jsonl),
              each scored by Gemma-4-31B as one token 1.0/0.5/0.0/NA.

Saves the RAW score matrix to peer_review_scores.npz so any future V vs V+A cut
never needs a GPU re-score. AUC aggregation is done separately (aggregate_va.py,
base env with sklearn).

Usage (gemma4 env, 1 GPU):
  source /lfs/skampere3/0/alexspan/envs/gemma4/bin/activate
  CUDA_VISIBLE_DEVICES=2 python score_va_gemma.py --split eval --sample 2000 --util 0.85 \
      --out /lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/peer_review_scores.npz
"""
import argparse, csv, gzip, json, os, random, re, sys
from pathlib import Path
import numpy as np
from vllm import LLM, SamplingParams

BASE   = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review")
RUBRICS = Path("/lfs/skampere3/0/alexspan/data/peer_review/rubrics.jsonl")
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"

SYS = ("You are an expert academic peer reviewer. You are given a paper's ABSTRACT and ONE "
       "quality criterion. Decide how strongly the abstract, on its own evidence, satisfies "
       "that criterion. Answer with EXACTLY ONE token:\n"
       "  1.0 = clearly satisfies the criterion\n"
       "  0.5 = partially / weakly / borderline\n"
       "  0.0 = fails / cuts against the criterion\n"
       "  NA = the abstract gives no evidence bearing on this criterion\n"
       "Judge the paper's quality, not whether it will be accepted. Output only the token.")

# ---------------- A metrics (rubric bank) ----------------
def load_rubrics():
    ms = []
    with open(RUBRICS) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("name"):
                ms.append(r)
    return ms

def metric_block(m):
    return (f"CRITERION: {m['name']}\n"
            f"DESCRIPTION: {m.get('description','')}\n\nAnswer with one token:")

def parse_tok(t):
    t = (t or "").strip().lower()
    # NA first (avoid the '0' inside 'na' being mis-hit)
    if t.startswith("na") or "n/a" in t or t == "na": return np.nan
    if "0.5" in t or t.startswith("0.5"): return 0.5
    if re.search(r"\b1(\.0)?\b", t) or t.startswith("1"): return 1.0
    if re.search(r"\b0(\.0)?\b", t) or t.startswith("0"): return 0.0
    return np.nan

# ---------------- V metrics (programmatic, stdlib) ----------------
NUM_RE   = re.compile(r"\d")
NUMTOK   = re.compile(r"\b\d[\d,\.]*\b")
SENT_RE  = re.compile(r"[.!?]+")
KW = {
    "v_kw_baseline":   re.compile(r"baseline|state[- ]of[- ]the[- ]art|\bsota\b|outperform|benchmark|compared? with|compared? to", re.I),
    "v_kw_ablation":   re.compile(r"ablation|ablate", re.I),
    "v_kw_dataset":    re.compile(r"dataset|corpus|benchmark", re.I),
    "v_kw_novel":      re.compile(r"\bnovel\b|first\b|new\b|propose|introduce|present a", re.I),
    "v_kw_theory":     re.compile(r"theorem|proof|prove|\bbound\b|guarantee|convergence|optimal", re.I),
    "v_kw_code":       re.compile(r"github|open[- ]source|code (?:is )?(?:available|released)|release our", re.I),
    "v_kw_hedge":      re.compile(r"\bmay\b|\bmight\b|\bcould\b|suggest|potential|possibly|appears? to", re.I),
    "v_kw_superlative":re.compile(r"\bbest\b|superior|significant|substantial|dramatic|remarkable|state[- ]of[- ]the[- ]art", re.I),
    "v_kw_cite":       re.compile(r"\bSection\b|\bFigure\b|\bTable\b|\bEq\.|\bAppendix\b", re.I),
}
def v_features(text):
    t = text or ""
    words = t.split()
    nw = max(len(words), 1)
    sents = [s for s in SENT_RE.split(t) if s.strip()]
    ns = max(len(sents), 1)
    feats = {
        "v_char_len":     float(len(t)),
        "v_word_len":     float(nw),
        "v_sent_count":   float(ns),
        "v_avg_word_len": float(sum(len(w) for w in words) / nw),
        "v_avg_sent_len": float(nw / ns),
        "v_num_density":  float(100.0 * len(NUMTOK.findall(t)) / nw),
        "v_pct_count":    float(t.count("%")),
        "v_question":     float(t.count("?")),
    }
    for name, rgx in KW.items():
        feats[name] = float(len(rgx.findall(t)))
    return feats

V_NAMES = ["v_char_len","v_word_len","v_sent_count","v_avg_word_len","v_avg_sent_len",
           "v_num_density","v_pct_count","v_question"] + list(KW.keys())

# ---------------- data ----------------
def load_rows(split, sample, venue_substr=None, source=None):
    """Rows = (paper_id, text, y, venue, source). venue_substr/source filter for
    confound control (e.g. venue_substr='ICLR', source='peerread_openreview' isolates
    the clean expert-verdict subpopulation where all submissions incl. rejects are observed)."""
    p = BASE / "splits" / f"{split}.csv.gz"
    rows = []
    with gzip.open(p, "rt", errors="ignore") as fh:
        for d in csv.DictReader(fh):
            txt = (d.get("text") or "").strip()
            ven = d.get("venue") or "?"
            src = d.get("source") or "?"
            try:
                y = int(d.get("judgement"))
            except Exception:
                continue
            if y not in (0, 1) or len(txt) < 50:
                continue
            if venue_substr and venue_substr.lower() not in ven.lower():
                continue
            if source and source != src:
                continue
            rows.append((d.get("paper_id") or d.get("id") or "", txt, y, ven, src))
    if sample and len(rows) > sample:
        random.seed(0)
        pos = [r for r in rows if r[2] == 1]
        neg = [r for r in rows if r[2] == 0]
        k = sample // 2
        rows = random.sample(pos, min(k, len(pos))) + random.sample(neg, min(k, len(neg)))
        random.shuffle(rows)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--venue-substr", default=None, help="e.g. ICLR (confound control)")
    ap.add_argument("--source", default=None, help="e.g. peerread_openreview")
    ap.add_argument("--out", default=str(BASE / "peer_review_scores.npz"))
    a = ap.parse_args()

    metrics = load_rubrics()
    blocks  = [metric_block(m) for m in metrics]
    a_names = [m["name"] for m in metrics]
    rows    = load_rows(a.split, a.sample, venue_substr=a.venue_substr, source=a.source)
    ids     = [r[0] for r in rows]
    y       = np.array([r[2] for r in rows])
    venues  = [r[3] for r in rows]

    Vf = np.array([[v_features(r[1])[n] for n in V_NAMES] for r in rows], dtype=float)

    bal = dict(zip(*[x.tolist() for x in np.unique(y, return_counts=True)]))
    print(f"[peer_review/{a.split}] {len(rows)} abstracts x {len(metrics)} A-rubrics "
          f"= {len(rows)*len(metrics)} prompts; {len(V_NAMES)} V-feats; balance {bal}", flush=True)

    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    convs = []
    for r in rows:
        txt = r[1]
        f = txt[:5000]
        for b in blocks:
            # Gemma chat template has no system role -> fold SYS into the user turn
            convs.append([{"role": "user", "content": f"{SYS}\n\nABSTRACT:\n{f}\n\n{b}"}])
    print(f"[peer_review] scoring {len(convs)} prompts on Gemma-4-31B ...", flush=True)
    outs = llm.chat(convs, sp)
    vals = [parse_tok(o.outputs[0].text) for o in outs]
    X = np.array(vals, dtype=float).reshape(len(rows), len(metrics))
    na = float(np.isnan(X).mean())
    print(f"[peer_review] A-score NA rate {na:.3f}", flush=True)

    np.savez_compressed(a.out, X=X, y=y, V=Vf, a_names=np.array(a_names, dtype=object),
                        v_names=np.array(V_NAMES, dtype=object), ids=np.array(ids, dtype=object),
                        venues=np.array(venues, dtype=object), split=a.split, na_rate=na,
                        venue_substr=str(a.venue_substr), source=str(a.source))
    print(f"[peer_review] saved raw scores -> {a.out}", flush=True)
    print("SCORE_DONE", flush=True)

if __name__ == "__main__":
    main()
