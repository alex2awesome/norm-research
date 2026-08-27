#!/usr/bin/env python3
"""T0 (UNTRAINED-T) ARM, step 2: score every cell's E rows with the BASE
Llama-3.1-8B checkpoint, zero-shot, offline batch vLLM.

Frozen design: notes/2026-07-27__vat-run-registry.md ("2026-08-08 -- FROZEN
DESIGN (before any scoring): UNTRAINED-T FUSION ARM").  Everything about the
elicitation is read from fusion/t0_templates.json -- this script contains NO
prompt text of its own and never edits the templates.

  prompt  = "{question}\\n\\n{document}\\n\\nAnswer Yes or No.\\n"
  document truncated to 1024 tokens (the trained T's --max_length), right side
  score   = P(Yes) over the masked {Yes,No}-variant token set, first token only

One vLLM session, all cells batched together.  Resumable: a cell whose output
file already exists is skipped unless --force.

sk3 only.  Usage (from the repo root, HOME pinned to /lfs for nohup):
    CUDA_VISIBLE_DEVICES=<one card> python3 t0_score_vllm.py --gpu-frac 0.93
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

HERE = Path(__file__).resolve().parent
ROWS = HERE / "t0_rows"
OUT = HERE / "t0_scores"
TEMPLATES = HERE / "t0_templates.json"

DEFAULT_MODEL_GLOB = ("/lfs/skampere3/0/alexspan/.cache/huggingface/hub/"
                      "models--meta-llama--Llama-3.1-8B/snapshots/*")


def resolve_model(spec):
    from glob import glob
    if spec and Path(spec).exists():
        p = Path(spec)
        if (p / "config.json").exists():
            return str(p)
        cands = sorted(glob(str(p / "snapshots" / "*")))
        if cands:
            return cands[-1]
        raise SystemExit(f"no snapshot under {p}")
    cands = sorted(glob(DEFAULT_MODEL_GLOB))
    if not cands:
        raise SystemExit(f"base Llama-3.1-8B not found under {DEFAULT_MODEL_GLOB}")
    return cands[-1]


def build_variant_ids(tok, spec):
    """Single-token surface forms of the Yes/No variants, per the frozen list."""
    got = {"pos": {}, "neg": {}}
    for side in ("pos", "neg"):
        for s in spec[side]:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                got[side][s] = ids[0]
    pos = sorted(set(got["pos"].values()))
    neg = sorted(set(got["neg"].values()))
    assert pos and neg, f"no single-token Yes/No variants found: {got}"
    assert not (set(pos) & set(neg)), "Yes and No variants share a token id"
    return pos, neg, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--gpu-frac", type=float, default=0.93)
    ap.add_argument("--max-model-len", type=int, default=1280)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    T = json.loads(TEMPLATES.read_text())
    tpl = T["prompt_format"]["template"]
    trunc = T["prompt_format"]["document_truncation"]
    maxlen = int(trunc["max_length"])
    cells = args.cell or list(T["cells"])

    model = resolve_model(args.model)
    print(f"[t0] model  = {model}", flush=True)
    print(f"[t0] freeze = sha256 {hashlib.sha256(TEMPLATES.read_bytes()).hexdigest()}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    pos_ids, neg_ids, realised = build_variant_ids(tok, T["score"]["yes_no_variants"])
    allowed = sorted(set(pos_ids) | set(neg_ids))
    print(f"[t0] POS ids {pos_ids} NEG ids {neg_ids} "
          f"(realised {json.dumps(realised)})", flush=True)

    # ---------------------------------------------------------- build prompts
    todo, prompts, index = [], [], []
    for cell in cells:
        outp = OUT / f"{cell}.jsonl.gz"
        if outp.exists() and not args.force:
            print(f"[t0] {cell}: already scored, skipping", flush=True)
            continue
        tp = ROWS / f"{cell}.texts.jsonl.gz"
        if not tp.exists():
            print(f"[t0] {cell}: MISSING {tp} -- skipping", flush=True)
            continue
        q = T["cells"][cell]["question"]
        n_trunc = 0
        with gzip.open(tp, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                ids = tok.encode(r["text"], add_special_tokens=False)
                if len(ids) > maxlen:
                    ids = ids[:maxlen]
                    n_trunc += 1
                doc = tok.decode(ids)
                prompts.append(tpl.format(question=q, document=doc))
                index.append((cell, r["uid"]))
        todo.append(cell)
        print(f"[t0] {cell}: {sum(1 for c, _ in index if c == cell)} prompts "
              f"({n_trunc} truncated at {maxlen} tok)", flush=True)
    if not prompts:
        print("[t0] nothing to do")
        return

    plens = [len(tok.encode(p)) for p in prompts[:2000]]
    print(f"[t0] TOTAL {len(prompts)} prompts across {len(todo)} cells; "
          f"prompt token len (first 2k) min/med/max "
          f"{min(plens)}/{sorted(plens)[len(plens)//2]}/{max(plens)}", flush=True)

    # ------------------------------------------------------------------ vLLM
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_frac,
              max_model_len=args.max_model_len,
              tensor_parallel_size=1, enforce_eager=False, trust_remote_code=False)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=len(allowed),
                        allowed_token_ids=allowed)
    print(f"[t0] engine up in {time.time()-t0:.0f}s; generating ...", flush=True)
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"[t0] generate done in {time.time()-t1:.0f}s", flush=True)

    # --------------------------------------------------------------- readout
    import math
    by_cell = {c: [] for c in todo}
    n_bad = 0
    for (cell, uid), o in zip(index, outs):
        lp = o.outputs[0].logprobs[0]
        sp_, sn = 0.0, 0.0
        seen = {}
        for tid, obj in lp.items():
            v = getattr(obj, "logprob", obj)
            if not math.isfinite(v):
                continue
            seen[int(tid)] = float(v)
            if int(tid) in pos_ids:
                sp_ += math.exp(v)
            elif int(tid) in neg_ids:
                sn += math.exp(v)
        tot = sp_ + sn
        if tot <= 0:
            n_bad += 1
            p_yes = float("nan")
        else:
            p_yes = sp_ / tot
        by_cell[cell].append({"uid": uid, "p_yes": p_yes,
                              "top_token": int(o.outputs[0].token_ids[0]),
                              "n_ids_returned": len(seen)})
    assert n_bad == 0, f"{n_bad} rows had no finite Yes/No mass -- FAIL CLOSED"

    meta_common = {
        "model": model, "checkpoint": "meta-llama/Llama-3.1-8B (BASE, no LoRA)",
        "templates_sha256": hashlib.sha256(TEMPLATES.read_bytes()).hexdigest(),
        "pos_token_ids": pos_ids, "neg_token_ids": neg_ids,
        "variant_surface_forms_realised": realised,
        "sampling": {"temperature": 0.0, "max_tokens": 1,
                     "allowed_token_ids": allowed, "logprobs": len(allowed)},
        "max_model_len": args.max_model_len, "gpu_memory_utilization": args.gpu_frac,
        "doc_max_tokens": maxlen,
        "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for cell in todo:
        rows = by_cell[cell]
        with gzip.open(OUT / f"{cell}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        vals = [r["p_yes"] for r in rows]
        vs = sorted(vals)
        n = len(vs)
        yes_frac = sum(1 for r in rows if r["top_token"] in pos_ids) / n
        m = dict(meta_common)
        m.update({"cell": cell, "n": n,
                  "p_yes_min": vs[0], "p_yes_p05": vs[int(.05 * n)],
                  "p_yes_median": vs[n // 2], "p_yes_p95": vs[int(.95 * n)],
                  "p_yes_max": vs[-1],
                  "p_yes_mean": sum(vs) / n,
                  "n_distinct_p_yes": len(set(round(v, 9) for v in vals)),
                  "argmax_yes_fraction": yes_frac,
                  "COLLAPSE_FLAG": bool(len(set(round(v, 6) for v in vals)) < max(10, n * 0.01)
                                        or yes_frac in (0.0, 1.0) and vs[-1] - vs[0] < 1e-3)})
        (OUT / f"{cell}.meta.json").write_text(json.dumps(m, indent=2))
        print(f"[t0] {cell}: n={n} p_yes med={m['p_yes_median']:.4f} "
              f"[{vs[0]:.4f},{vs[-1]:.4f}] distinct={m['n_distinct_p_yes']} "
              f"argmaxYes={yes_frac:.3f} collapse={m['COLLAPSE_FLAG']}", flush=True)
    print("[t0] DONE", flush=True)


if __name__ == "__main__":
    main()
