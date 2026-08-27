#!/usr/bin/env python3
"""Batch claim extraction on sk3 via OFFLINE vLLM (Qwen3.5-122B-A10B-FP8).

Reads math_se_v3_position_matched.csv.gz, filters to --split (default eval),
splits each row's text into Question/Answer, and extracts checkable claims
with the prompt in extraction_prompt.py. OFFLINE batch mode only
(LLM.chat over thousands of prompts per call) — never an HTTP server.

Conventions copied from the repo's sk3 batch scripts
(scripts/batch_extract_vllm.py, scripts/sk3_judge_pairs.py,
scripts/queue_nc_agency_runs.sh):
  * env pinned BEFORE importing torch/vllm (HOME on /lfs so nohup jobs do
    not touch AFS; shared HF cache; FLASHINFER_DISABLE_VERSION_CHECK=1;
    VLLM_USE_FLASHINFER_MOE_FP8=0 for the Qwen3.5 MoE FP8)
  * default model = the shared-cache Qwen3.5-122B-A10B-FP8 snapshot
  * GPU_MEM_UTIL 0.93, MAX_MODEL_LEN with margin over prompt+output
  * chunked llm.chat() calls (thousands of prompts), append-only JSONL
    checkpointing with fsync after every chunk, resume from existing output
  * retry-with-a-different-seed on invalid output (loops, bad JSON, failed
    fidelity checks) — NEVER repetition_penalty

Output: claims_{split}.jsonl — one line per claim (flat, harness-ready),
row ids carried on every line. Answers whose extraction failed after all
retries get a single {"claim_type": "EXTRACTION_FAILED", ...} record so
resume logic never re-queues them. Verification is a separate CPU step:
run_verification.py.

Usage (on sk3 — do NOT launch from a laptop):
  CUDA_VISIBLE_DEVICES=4 nohup python3 run_extraction_sk3.py --split eval \
      > extraction_eval.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- env BEFORE torch/vllm (see scripts/batch_extract_vllm.py) -------------
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
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")  # Qwen3.5 MoE FP8

sys.path.insert(0, str(HERE))
from extraction_prompt import (SYSTEM_PROMPT, build_user_prompt,  # noqa: E402
                               validate_extraction)

# Exact snapshot used by the other sk3 Qwen3.5 runs (queue_nc_agency_runs.sh)
DEFAULT_MODEL = ("/lfs/skampere3/0/shared_hf_cache/hub/"
                 "models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/"
                 "fb53b9f3bdaab287c597d4e943783153ec527e06")
DEFAULT_DATA = HERE.parent / "math_se_v3_position_matched.csv.gz"

ANSWER_MARKER = "\n\nAnswer: "


def split_question_answer(text: str):
    """text format: 'Question: <q>\\n\\nAnswer: <a>' (100% of v3 rows)."""
    if ANSWER_MARKER in text:
        q, a = text.split(ANSWER_MARKER, 1)
    else:  # defensive: treat the whole row as the answer
        q, a = "", text
    if q.startswith("Question: "):
        q = q[len("Question: "):]
    return q.strip(), a.strip()


def load_rows(data_path, split, limit=None):
    import pandas as pd
    df = pd.read_csv(data_path)
    if split != "all":
        df = df[df["split"] == split]
    if limit:
        df = df.iloc[:limit]
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "row_id": int(r["answer_id"]),
            "question_id": int(r["question_id"]),
            "split": str(r["split"]),
            "judgement": int(r["judgement"]),
            "text": r["text"],
        })
    return rows


def load_done_row_ids(out_path: Path, retry_failed: bool = False):
    """With retry_failed, EXTRACTION_FAILED rows do not count as done, so a
    recovery pass (e.g. with a larger --max-tokens) re-attempts exactly them.
    Downstream readers tolerate the resulting duplicate row_ids: the :fail
    row carries no claims, the recovery rows carry the real ones."""
    done, failed = set(), set()
    if not out_path.exists():
        return done
    with out_path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
                if "row_id" in d:
                    done.add(d["row_id"])
                    if d.get("claim_type") == "EXTRACTION_FAILED":
                        failed.add(d["row_id"])
                    else:
                        failed.discard(d["row_id"])
            except Exception:
                continue
    return done - failed if retry_failed else done


def write_claims(fout, row, claims, errors, attempt):
    """Flatten one answer's validated claims into harness-ready JSONL lines."""
    meta = {"row_id": row["row_id"], "question_id": row["question_id"],
            "split": row["split"], "judgement": row["judgement"]}
    n = 0
    for i, c in enumerate(claims):
        rec = {**meta, "claim_id": f"{row['row_id']}:{i}", **c}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    if n == 0:
        fout.write(json.dumps({**meta, "claim_id": f"{row['row_id']}:fail",
                               "claim_type": "EXTRACTION_FAILED",
                               "errors": errors[:8],
                               "attempts": attempt + 1}) + "\n")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--split", default="eval",
                    choices=["train", "eval", "test", "all"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output", default=None,
                    help="default: <this dir>/claims_{split}.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=2000,
                    help="prompts per llm.chat() call (submit thousands)")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.93)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--max-answer-chars", type=int, default=18000)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt rows whose only output is "
                         "EXTRACTION_FAILED (recovery pass)")
    ap.add_argument("--retries", type=int, default=2,
                    help="extra passes with a different seed on invalid output")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    out_path = Path(args.output) if args.output else \
        HERE / f"claims_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data, args.split, args.limit)
    done = load_done_row_ids(out_path, retry_failed=args.retry_failed)
    pending = [r for r in rows if r["row_id"] not in done]
    print(f"=== split={args.split}: {len(rows):,} rows, "
          f"{len(done):,} already done, {len(pending):,} pending ===",
          flush=True)
    if not pending:
        print("nothing to do")
        return

    from vllm import LLM, SamplingParams
    print(f"=== loading vLLM model: {args.model} ===", flush=True)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        kv_cache_dtype="auto",
        dtype="auto",
        tensor_parallel_size=args.tp,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,  # shared system prompt across all rows
    )

    def build_convo(row, cap=None):
        q, a = split_question_answer(row["text"])
        cap = cap or args.max_answer_chars
        if len(a) > cap:
            a = a[:cap] + " [...truncated]"
        row["_answer"] = a
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(q, a)}]

    # char truncation alone is not enough: LaTeX-heavy text can hit ~1.5
    # chars/token, and an input over max_model_len is a FATAL engine error
    # in vLLM (killed the 2026-06-10 run at 16,385 tokens). Fit each convo
    # under max_model_len - max_tokens by halving the answer cap as needed.
    tok = llm.get_tokenizer()
    input_limit = args.max_model_len - args.max_tokens

    def n_tokens(convo):
        # render to text, then encode: apply_chat_template(tokenize=True)
        # returns a 2-key BatchEncoding in transformers>=5 (len() == 2!),
        # which silently defeated the length check on 2026-06-10
        s = tok.apply_chat_template(convo, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
        return len(tok.encode(s, add_special_tokens=False))

    def build_convo_fitted(row):
        cap = args.max_answer_chars
        convo = build_convo(row, cap)
        while n_tokens(convo) > input_limit and cap > 1000:
            cap //= 2
            convo = build_convo(row, cap)
        if n_tokens(convo) > input_limit:
            return None  # question alone blows the budget; skip the row
        return convo

    t_start = time.time()
    n_done = n_claims_total = n_failed = 0
    fout = out_path.open("a")

    try:
        # pass 0 = greedy; retry passes resample with a fresh seed (never
        # repetition_penalty — see feedback_no_repetition_penalty_retry_instead)
        queue = pending
        for attempt in range(args.retries + 1):
            if not queue:
                break
            if attempt == 0:
                sp = SamplingParams(temperature=0.0,
                                    max_tokens=args.max_tokens)
            else:
                sp = SamplingParams(temperature=0.7, top_p=0.95,
                                    seed=args.seed + attempt,
                                    max_tokens=args.max_tokens)
            print(f"\n=== pass {attempt}: {len(queue):,} answers "
                  f"(temperature={sp.temperature}, seed={getattr(sp, 'seed', None)}) ===",
                  flush=True)
            next_queue = []
            for ci in range(0, len(queue), args.chunk_size):
                batch_all = queue[ci:ci + args.chunk_size]
                batch, convos = [], []
                for r in batch_all:
                    convo = build_convo_fitted(r)
                    if convo is None:
                        write_claims(fout, r, [], ["overlong_prompt"], attempt)
                        n_done += 1
                        n_failed += 1
                        continue
                    batch.append(r)
                    convos.append(convo)
                t0 = time.time()
                try:
                    outs = llm.chat(convos, sp, use_tqdm=False,
                                    chat_template_kwargs={"enable_thinking": False})
                except Exception as e:
                    # client-side render/validation error: the engine is still
                    # alive, so isolate the poison row(s) one by one instead
                    # of dying (a single overlong prompt killed two runs)
                    print(f"  chunk chat failed ({type(e).__name__}: "
                          f"{str(e)[:160]}); isolating row-by-row", flush=True)
                    outs = []
                    for c_one in convos:
                        try:
                            outs.extend(llm.chat(
                                [c_one], sp, use_tqdm=False,
                                chat_template_kwargs={"enable_thinking": False}))
                        except Exception:
                            outs.append(None)
                dt = time.time() - t0
                chunk_ok = chunk_retry = 0
                for row, out in zip(batch, outs):
                    if out is None:  # row failed even in isolation
                        write_claims(fout, row, [], ["render_failed"], attempt)
                        n_done += 1
                        n_failed += 1
                        chunk_ok += 1
                        continue
                    raw = out.outputs[0].text if out.outputs else ""
                    claims, errors = validate_extraction(
                        raw, source_text=row["_answer"])
                    final = (attempt == args.retries)
                    if errors and not final:
                        # retry the whole answer with a different seed
                        row["_last_errors"] = errors
                        next_queue.append(row)
                        chunk_retry += 1
                        continue
                    # success, or final attempt: accept the valid subset
                    n = write_claims(fout, row, claims, errors, attempt)
                    n_done += 1
                    if n and not (len(claims) == 1
                                  and claims[0]["claim_type"] == "NONE"):
                        n_claims_total += n
                    if not claims:
                        n_failed += 1
                    chunk_ok += 1
                fout.flush()
                os.fsync(fout.fileno())  # checkpoint after every chunk
                elapsed = time.time() - t_start
                rate = max(n_done, 1) / elapsed
                print(f"  pass{attempt} chunk {ci // args.chunk_size + 1}: "
                      f"{len(batch)} prompts in {dt:.0f}s | accepted={chunk_ok} "
                      f"retry={chunk_retry} | total done={n_done:,} "
                      f"claims={n_claims_total:,} failed={n_failed:,} "
                      f"rate={rate * 60:.1f}/min", flush=True)
            queue = next_queue
    finally:
        fout.flush()
        try:
            os.fsync(fout.fileno())
        except Exception:
            pass
        fout.close()

    wall = time.time() - t_start
    print(f"\n=== DONE in {wall / 3600:.2f}h: answers={n_done:,} "
          f"claims={n_claims_total:,} extraction_failed={n_failed:,} ===")
    print(f"output: {out_path}")
    print("next (CPU, anywhere): python3 run_verification.py "
          f"--claims {out_path}")


if __name__ == "__main__":
    main()
