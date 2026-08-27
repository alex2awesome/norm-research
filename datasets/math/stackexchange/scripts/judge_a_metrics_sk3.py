#!/usr/bin/env python3
"""Math.SE A-metric LLM judge (a01-a14) over the v3.3 propensity-balanced pool.

For each (Question + Answer) in the eval split, an offline-vLLM judge
(Qwen3.5-122B-A10B-FP8) rates the answer 1-5 on each of the 14 articulability
metrics defined in METRICS.md (a01-a14) and classifies answer_type. This is the
A-layer counterpart to the already-measured V layer (mathse_lint, combined LR
AUC 0.567 vs floor 0.461): it measures how much of community answer-quality
judgement is recoverable from LLM-judged-but-not-mechanizable criteria.

The judge system prompt + output schema live in a sibling spec JSON
(a_judge_spec.json), produced by the essay-grounded design workflow, so the
rubric can be iterated without touching this harness.

Conventions copied from aops/scripts/judge_approach_sk3.py:
  * env pinned BEFORE importing torch/vllm (HOME on /lfs; shared HF cache;
    VLLM_USE_FLASHINFER_MOE_FP8=0 for Qwen3.5 MoE FP8)
  * offline LLM.chat over thousands of prompts per call, never HTTP
  * token-fit guard renders tokenize=False then tok.encode()
    (transformers>=5 BatchEncoding trap)
  * retry-with-a-different-seed on invalid output, NEVER repetition_penalty
  * append-only plain JSONL, fsync per chunk, resume-safe via done answer_ids
  * per-row isolation fallback if a chunk chat() raises

Usage (sk3, GPU 3 only):
  HOME=/lfs/skampere3/0/alexspan CUDA_VISIBLE_DEVICES=3 nohup \
    /lfs/skampere3/0/alexspan/miniconda3/bin/python3.11 judge_a_metrics_sk3.py \
    --pool .../math_se_v3_3_propensity_balanced.csv.gz --split eval \
    --spec a_judge_spec.json --output a_metric_verdicts_eval.jsonl \
    > a_judge.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

# --- env BEFORE torch/vllm ---------------------------------------------------
if os.path.isdir("/lfs/skampere3/0/alexspan"):
    os.environ.setdefault("HOME", "/lfs/skampere3/0/alexspan")
    os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
    os.environ.setdefault("TRITON_CACHE_DIR",
                          "/lfs/skampere3/0/alexspan/.cache/triton")
    os.environ.setdefault("VLLM_CACHE_ROOT",
                          "/lfs/skampere3/0/alexspan/.cache/vllm")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                          "/lfs/skampere3/0/alexspan/.cache/torchinductor")
    os.environ.setdefault("TMPDIR", "/lfs/skampere3/0/alexspan/tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")

DEFAULT_MODEL = ("/lfs/skampere3/0/shared_hf_cache/hub/"
                 "models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/"
                 "fb53b9f3bdaab287c597d4e943783153ec527e06")

METRIC_IDS = [f"a{n:02d}" for n in range(1, 15)]  # a01..a14
TEXT_CHAR_CAP = 9000  # Q+A text cap before token-fit guard (p90 ~2731 chars)

FALLBACK_SYSTEM = (
    "You are an expert mathematical-writing referee. Rate the Math Stack "
    "Exchange answer on 14 axes (a01-a14), each an integer 1-5, and classify "
    "answer_type. Output ONLY one JSON object with integer keys a01..a14 and "
    "string answer_type. (Spec file missing — using fallback prompt.)"
)
ANSWER_TYPES = {"proof", "computation", "conceptual_explanation",
                "counterexample", "reference_or_pointer", "other"}


def load_spec(spec_path):
    if spec_path and Path(spec_path).exists():
        spec = json.loads(Path(spec_path).read_text())
        sysp = spec.get("system_prompt") or FALLBACK_SYSTEM
        atypes = set(spec.get("answer_type_values") or ANSWER_TYPES)
        return sysp, atypes
    print(f"!! spec {spec_path} not found; using fallback prompt", flush=True)
    return FALLBACK_SYSTEM, ANSWER_TYPES


def trunc(s, cap):
    s = (s or "").strip()
    return s[:cap] + " [...truncated]" if len(s) > cap else s


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
BAD_ESCAPE_RE = re.compile(r'\\(?![\\"/bfnrtu])')


def _coerce_score(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        v = v.strip()
        if v.isdigit():
            v = int(v)
        else:
            m = re.match(r"([1-5])", v)
            v = int(m.group(1)) if m else None
    if isinstance(v, float) and v == int(v):
        v = int(v)
    if isinstance(v, int) and 1 <= v <= 5:
        return v
    return None


def _salvage_scores(raw):
    """Field-wise regex extraction when JSON never closes."""
    d = {}
    for mid in METRIC_IDS:
        m = re.search(rf'"{mid}"\s*:\s*"?([1-5])"?', raw)
        if m:
            d[mid] = int(m.group(1))
    m = re.search(r'"answer_type"\s*:\s*"([a-z_]{2,40})"', raw)
    if m:
        d["answer_type"] = m.group(1)
    if all(mid in d for mid in METRIC_IDS):
        d["_salvaged"] = True
        return d
    return None


def parse_verdict(raw, atypes):
    if not raw or not raw.strip():
        return None, "empty_output"
    raw = (raw.replace("“", '"').replace("”", '"')
              .replace("‘", "'").replace("’", "'"))
    d = None
    m = JSON_RE.search(raw)
    if m:
        for cand in (m.group(0), BAD_ESCAPE_RE.sub(r"\\\\", m.group(0))):
            try:
                d = json.loads(cand)
                break
            except Exception:
                continue
    if not isinstance(d, dict):
        d = _salvage_scores(raw)
    if d is None:
        return None, "no_json_object" if not m else "json_unrecoverable"

    out, missing = {}, []
    for mid in METRIC_IDS:
        sv = _coerce_score(d.get(mid))
        if sv is None:
            missing.append(mid)
        else:
            out[mid] = sv
    if missing:
        return None, f"bad_scores:{','.join(missing[:4])}"
    at = d.get("answer_type")
    at = at.strip().lower() if isinstance(at, str) else "other"
    if at not in atypes:
        at = "other"
    out["answer_type"] = at
    out["salvaged"] = bool(d.get("_salvaged", False))
    return out, None


def load_pool(pool_path, split, limit=None):
    import pandas as pd
    df = pd.read_csv(pool_path)
    if split and split != "all":
        df = df[df["split"] == split]
    df = df.drop_duplicates(subset=["answer_id"])
    # held-out splits first so the V-vs-A head-to-head can be analyzed while
    # the large train split is still scoring
    order = {"eval": 0, "test": 1, "train": 2}
    df = df.assign(_o=df["split"].map(lambda s: order.get(s, 3))) \
           .sort_values(["_o", "question_id", "answer_id"])
    rows = [{"answer_id": int(r.answer_id),
             "question_id": int(r.question_id),
             "judgement": int(r.judgement),
             "split": r.split,
             "text": r.text or ""} for r in df.itertuples()]
    if limit:
        rows = rows[:limit]
    return rows


def base_rec(row):
    return {"answer_id": row["answer_id"], "question_id": row["question_id"],
            "judgement": row["judgement"], "split": row["split"]}


def load_done(out_path, retry_failed=False):
    done, failed = set(), set()
    if not Path(out_path).exists():
        return done
    with open(out_path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            aid = d.get("answer_id")
            if aid is None:
                continue
            done.add(aid)
            (failed.add if d.get("judge_failed") else failed.discard)(aid)
    return done - failed if retry_failed else done


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--spec", default="a_judge_spec.json")
    ap.add_argument("--output", default="a_metric_verdicts_eval.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    system_prompt, atypes = load_spec(args.spec)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_pool(args.pool, args.split, args.limit)
    done = load_done(out_path, retry_failed=args.retry_failed)
    pending = [r for r in rows if r["answer_id"] not in done]
    print(f"=== {len(rows):,} {args.split} rows, {len(done):,} done, "
          f"{len(pending):,} pending ===", flush=True)
    if not pending:
        print("nothing to do")
        return

    from vllm import LLM, SamplingParams
    print(f"=== loading vLLM model: {args.model} ===", flush=True)
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, kv_cache_dtype="auto",
              dtype="auto", tensor_parallel_size=args.tp,
              max_num_seqs=args.max_num_seqs, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    input_limit = args.max_model_len - args.max_tokens

    def build_convo(row, cap):
        user = ("Rate this Math Stack Exchange answer. The text contains the "
                "question then the answer.\n\n" + trunc(row["text"], cap)
                + "\n\nOutput ONLY the JSON object.")
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user}]

    def n_tokens(convo):
        s = tok.apply_chat_template(convo, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
        return len(tok.encode(s, add_special_tokens=False))

    def build_fitted(row):
        cap = TEXT_CHAR_CAP
        convo = build_convo(row, cap)
        while n_tokens(convo) > input_limit and cap > 400:
            cap = max(400, cap // 2)
            convo = build_convo(row, cap)
        return convo if n_tokens(convo) <= input_limit else None

    fout = out_path.open("a")
    t_start = time.time()
    n_done = n_ok = n_failed = 0
    from collections import Counter
    try:
        queue = pending
        for attempt in range(args.retries + 1):
            if not queue:
                break
            if attempt == 0:
                sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
            else:
                sp = SamplingParams(temperature=0.7, top_p=0.95,
                                    seed=args.seed + attempt,
                                    max_tokens=args.max_tokens)
            print(f"\n=== pass {attempt}: {len(queue):,} rows "
                  f"(temp={sp.temperature}) ===", flush=True)
            next_queue = []
            for ci in range(0, len(queue), args.chunk_size):
                batch_all = queue[ci:ci + args.chunk_size]
                batch, convos = [], []
                for r in batch_all:
                    convo = build_fitted(r)
                    if convo is None:
                        fout.write(json.dumps({**base_rec(r),
                                   "judge_failed": "overlong",
                                   "attempts": attempt + 1}) + "\n")
                        n_done += 1
                        n_failed += 1
                        continue
                    batch.append(r)
                    convos.append(convo)
                if not batch:
                    continue
                t0 = time.time()
                try:
                    outs = llm.chat(convos, sp, use_tqdm=False,
                                    chat_template_kwargs={
                                        "enable_thinking": False})
                except Exception as e:
                    print(f"  chunk chat failed ({type(e).__name__}: "
                          f"{str(e)[:160]}); isolating", flush=True)
                    outs = []
                    for c_one in convos:
                        try:
                            outs.extend(llm.chat([c_one], sp, use_tqdm=False,
                                        chat_template_kwargs={
                                            "enable_thinking": False}))
                        except Exception:
                            outs.append(None)
                dt = time.time() - t0
                chunk_ok = chunk_retry = 0
                err_counter, err_samples = Counter(), []
                for row, out in zip(batch, outs):
                    if out is None:
                        fout.write(json.dumps({**base_rec(row),
                                   "judge_failed": "render_failed",
                                   "attempts": attempt + 1}) + "\n")
                        n_done += 1
                        n_failed += 1
                        continue
                    raw = out.outputs[0].text if out.outputs else ""
                    verdict, err = parse_verdict(raw, atypes)
                    final = (attempt == args.retries)
                    if verdict is None:
                        err_counter[err] += 1
                        if len(err_samples) < 3:
                            snip = raw if len(raw) <= 400 else \
                                raw[:200] + " <...> " + raw[-200:]
                            err_samples.append((row["answer_id"], err,
                                                snip.replace("\n", "\\n")))
                    if verdict is None and not final:
                        next_queue.append(row)
                        chunk_retry += 1
                        continue
                    if verdict is None:
                        fout.write(json.dumps({**base_rec(row),
                                   "judge_failed": f"bad_output:{err}",
                                   "attempts": attempt + 1,
                                   "raw_head": raw[:300]}) + "\n")
                        n_failed += 1
                    else:
                        fout.write(json.dumps({**base_rec(row), **verdict,
                                   "attempts": attempt + 1},
                                   ensure_ascii=False) + "\n")
                        n_ok += 1
                    n_done += 1
                    chunk_ok += 1
                fout.flush()
                os.fsync(fout.fileno())
                elapsed = time.time() - t_start
                rate = max(n_done, 1) / max(elapsed, 1)
                print(f"  pass{attempt} chunk {ci // args.chunk_size + 1}: "
                      f"{len(batch)} prompts in {dt:.0f}s | accepted={chunk_ok}"
                      f" retry={chunk_retry} | done={n_done:,} ok={n_ok:,} "
                      f"failed={n_failed:,} rate={rate * 60:.1f}/min",
                      flush=True)
                if err_counter:
                    print(f"    errors: {dict(err_counter.most_common(8))}",
                          flush=True)
                    for aid_, e_, s_ in err_samples:
                        print(f"    sample aid={aid_} err={e_} raw: {s_}",
                              flush=True)
            queue = next_queue
    finally:
        fout.flush()
        try:
            os.fsync(fout.fileno())
        except Exception:
            pass
        fout.close()

    wall = time.time() - t_start
    print(f"\n=== DONE in {wall / 3600:.2f}h: judged={n_done:,} ok={n_ok:,} "
          f"failed={n_failed:,} ===\noutput: {out_path}")


if __name__ == "__main__":
    main()
