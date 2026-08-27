#!/usr/bin/env python3
"""LLM approach-classification for AoPS forum solutions vs wiki editorials.

For each forum solution that joins to a wiki problem, an offline-vLLM judge
(Qwen3.5-122B-A10B-FP8) decides which wiki editorial solution (if any) uses
the same mathematical approach, names the approach, and rates elegance.
This is the LLM-judge parallel to the LeetCode/CodeContests
editorial-similarity analysis.

Conventions copied from datasets/math/stackexchange/verification/
run_extraction_sk3.py (the just-finished Math.SE run):
  * env pinned BEFORE importing torch/vllm (HOME on /lfs; shared HF cache;
    VLLM_USE_FLASHINFER_MOE_FP8=0 for the Qwen3.5 MoE FP8)
  * offline LLM.chat over thousands of prompts per call, never HTTP
  * token-fit guard renders with tokenize=False then tok.encode()
    (transformers>=5: apply_chat_template(tokenize=True) returns a 2-key
    BatchEncoding, len()==2, which silently defeats length checks)
  * chunk llm.chat wrapped in try/except with per-row isolation fallback
  * retry-with-a-different-seed on invalid output — NEVER repetition_penalty
  * append-only plain JSONL (never gzip-append), fsync per chunk,
    resume-safe via done post_ids

Output: one JSON row per post_id in approach_verdicts.jsonl. Rows that
cannot be judged carry {"judge_failed": "<reason>"} so resume never
re-queues them (unless --retry-failed).

Usage (on sk3, GPU 3 only):
  HOME=/lfs/skampere3/0/alexspan CUDA_VISIBLE_DEVICES=3 nohup \
    /lfs/skampere3/0/alexspan/miniconda3/bin/python3.11 judge_approach_sk3.py \
    > /lfs/skampere3/0/alexspan/aops/judge_approach.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
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
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")  # Qwen3.5 MoE FP8

# Exact snapshot used by run_extraction_sk3.py / queue_nc_agency_runs.sh
DEFAULT_MODEL = ("/lfs/skampere3/0/shared_hf_cache/hub/"
                 "models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/"
                 "fb53b9f3bdaab287c597d4e943783153ec527e06")

AOPS_DIR = Path("/lfs/skampere3/0/alexspan/aops")
DEFAULT_FORUM = AOPS_DIR / "forum_solutions.parquet"
DEFAULT_WIKI_SOLS = AOPS_DIR / "wiki" / "solutions.parquet"
DEFAULT_WIKI_PROBS = AOPS_DIR / "wiki" / "problems.parquet"
DEFAULT_OUT = AOPS_DIR / "approach_verdicts.jsonl"

MAX_WIKI_SOLS = 6          # wiki solutions shown per prompt (note when capped)
WIKI_SOL_CHAR_CAP = 2500   # per wiki solution, before token-fit guard
FORUM_SOL_CHAR_CAP = 4000  # forum solution, before token-fit guard
PROBLEM_CHAR_CAP = 4000

SYSTEM_PROMPT = """\
You are an expert competition-mathematics editor. You will be given a contest \
problem, the official wiki's editorial solutions (numbered), and one solution \
posted on a community forum. Your job is to classify the forum solution's \
mathematical APPROACH against the wiki solutions.

Judge by the core mathematical method (e.g. coordinate geometry vs synthetic, \
generating functions vs direct counting, casework structure, key substitution \
or invariant), NOT by writing style, level of detail, or final answer.

Respond with STRICT JSON only — a single JSON object, no markdown fences, no \
commentary — with exactly these keys:
{
  "matched_solution": <int 1-based index of the wiki solution that uses the \
same core approach as the forum solution, or null if none does>,
  "same_approach": <bool — true iff matched_solution is not null>,
  "approach_label": "<2-5 word name of the forum solution's method, e.g. \
'coordinate bash', 'casework on units digit', 'vieta jumping', 'complementary \
counting'. Be specific to this solution's actual technique; never a generic \
label like 'algebra' or 'clever manipulation'>",
  "novel_approach": <bool — true iff NO listed wiki solution uses this \
method>,
  "elegance": <int 1-5, your own rating of the forum solution's elegance: \
1 = brute-force grind, 3 = standard competent solution, 5 = strikingly \
elegant insight>,
  "reason": "<one sentence justifying the match/no-match decision>"
}

Rules:
- matched_solution must be one of the listed wiki solution numbers, or null.
- If the forum solution matches several wiki solutions, pick the closest one.
- If the forum solution is not actually a solution (e.g. just an answer, a \
question, or off-topic), use matched_solution null, approach_label \
"not a solution", novel_approach false, elegance 1.
- novel_approach must be false whenever same_approach is true.
"""


def trunc(s: str, cap: int) -> str:
    s = (s or "").strip()
    if len(s) > cap:
        return s[:cap] + " [...truncated]"
    return s


def build_user_prompt(prob: dict, wiki_sols: list, forum_text: str,
                      wiki_cap: int, forum_cap: int) -> str:
    parts = [
        f"## Problem ({prob['variant'].replace('_', ' ')} "
        f"Problem {prob['problem_n']})",
        trunc(prob["problem_statement"], PROBLEM_CHAR_CAP),
        "",
        f"## Wiki editorial solutions ({len(wiki_sols)} shown)",
    ]
    for i, ws in enumerate(wiki_sols, 1):
        heading = (ws.get("solution_heading") or "").strip()
        title = f"### Wiki Solution {i}"
        if heading and heading.lower() not in (f"solution {i}", "solution"):
            title += f" — {heading}"
        parts.append(title)
        parts.append(trunc(ws["solution_text"], wiki_cap))
        parts.append("")
    parts.append("## Forum solution to classify")
    parts.append(trunc(forum_text, forum_cap))
    parts.append("")
    parts.append("Classify the forum solution. STRICT JSON only.")
    return "\n".join(parts)


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
# LaTeX in string fields produces invalid JSON escapes (e.g. "$r\cos(x+a)$");
# double any backslash not starting a legal JSON escape
BAD_ESCAPE_RE = re.compile(r'\\(?![\\"/bfnrtu])')


def _salvage_fields(raw: str):
    """Field-wise extraction for outputs whose JSON never closes (the known
    Qwen-FP8 repetition loop usually degenerates inside the final "reason"
    string, leaving the five core fields intact)."""
    d = {}
    m = re.search(r'"matched_solution"\s*:\s*(null|"?\d+"?|"none")', raw,
                  re.IGNORECASE)
    if m:
        v = m.group(1).strip('"').lower()
        d["matched_solution"] = None if v in ("null", "none") else int(v)
    for key in ("same_approach", "novel_approach"):
        m = re.search(rf'"{key}"\s*:\s*"?(true|false)"?', raw, re.IGNORECASE)
        if m:
            d[key] = m.group(1).lower() == "true"
    m = re.search(r'"approach(?:es)?_label"\s*:\s*"([^"\n]{2,80})"', raw)
    if m:
        d["approach_label"] = m.group(1)
    m = re.search(r'"elegance"\s*:\s*"?([1-5])"?', raw)
    if m:
        d["elegance"] = int(m.group(1))
    m = re.search(r'"reason"\s*:\s*"([^"]{0,500})', raw)
    if m:
        d["reason"] = m.group(1)
    required = ("same_approach", "novel_approach", "approach_label",
                "elegance")
    if all(k in d for k in required) and "matched_solution" in d:
        d["_salvaged"] = True
        return d
    return None


def parse_verdict(raw: str, n_sols: int):
    """Return (verdict_dict, None) or (None, error_string)."""
    if not raw or not raw.strip():
        return None, "empty_output"
    # degenerate decoding sometimes emits Unicode curly quotes for JSON keys
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
        d = _salvage_fields(raw)
    if d is None:
        return None, "no_json_object" if not m else "json_unrecoverable"
    # coerce string booleans
    for key in ("same_approach", "novel_approach"):
        v = d.get(key)
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            d[key] = v.strip().lower() == "true"

    ms = d.get("matched_solution")
    if isinstance(ms, str):  # coerce "2" / "null" / "none"
        s = ms.strip().lower()
        if s in ("null", "none", ""):
            ms = None
        elif s.isdigit():
            ms = int(s)
        else:
            return None, "matched_solution_not_int_or_null"
    if isinstance(ms, float) and ms == int(ms):
        ms = int(ms)
    if ms is not None:
        if isinstance(ms, bool) or not isinstance(ms, int):
            return None, "matched_solution_not_int_or_null"
        if ms == 0:
            ms = None  # 0 used as "no match"
        elif not (1 <= ms <= n_sols):
            return None, f"matched_solution_out_of_range:{ms}"
    sa = d.get("same_approach")
    if not isinstance(sa, bool):
        return None, "same_approach_not_bool"
    if sa and ms is None:
        return None, "same_approach_true_but_no_match"
    if (not sa) and ms is not None:
        # model: "closest is #k but approach differs" -> treat as no match
        ms = None
    na = d.get("novel_approach")
    if not isinstance(na, bool):
        return None, "novel_approach_not_bool"
    if sa and na:
        na = False  # same approach exists in wiki, so by definition not novel
    label = d.get("approach_label")
    if not isinstance(label, str) or not (2 <= len(label.strip()) <= 80):
        return None, "bad_approach_label"
    el = d.get("elegance")
    if isinstance(el, str) and el.strip().isdigit():
        el = int(el.strip())
    if isinstance(el, float) and el == int(el):
        el = int(el)
    if isinstance(el, bool) or not isinstance(el, int) or not (1 <= el <= 5):
        return None, "bad_elegance"
    reason = d.get("reason")
    if not isinstance(reason, str):
        reason = str(reason or "")
    return {
        "matched_solution": ms,
        "same_approach": sa,
        "approach_label": label.strip(),
        "novel_approach": na,
        "elegance": el,
        "reason": reason.strip()[:500],
        "salvaged": bool(d.get("_salvaged", False)),
    }, None


def load_rows(forum_path, wiki_sols_path, wiki_probs_path, limit=None):
    import pandas as pd
    key = ["variant", "problem_n"]

    probs = pd.read_parquet(wiki_probs_path)
    probs = probs[~probs["is_redirect"]].drop_duplicates(subset=key)
    sols = pd.read_parquet(wiki_sols_path)
    sols = sols[~sols["is_redirect"]]
    sols = sols.drop_duplicates(subset=key + ["solution_index"])
    sols = sols.sort_values(key + ["solution_index"])

    sols_by_key = {}
    for (variant, pn), g in sols.groupby(key):
        sols_by_key[(variant, int(pn))] = [
            {"solution_heading": r.solution_heading,
             "solution_text": r.solution_text}
            for r in g.itertuples()
        ]
    probs_by_key = {}
    for r in probs.itertuples():
        probs_by_key[(r.variant, int(r.problem_n))] = {
            "variant": r.variant, "problem_n": int(r.problem_n),
            "contest": r.contest, "year": int(r.year),
            "problem_statement": r.problem_statement,
        }

    fs = pd.read_parquet(forum_path)
    fs = fs.drop_duplicates(subset=["post_id"])
    # group all forum solutions of the same problem adjacently so prefix
    # caching shares the (system + problem + wiki solutions) prefix
    fs = fs.sort_values(["variant", "problem_n", "topic_id", "post_id"])

    rows = []
    for r in fs.itertuples():
        k = (r.variant, int(r.problem_n))
        rows.append({
            "post_id": int(r.post_id), "topic_id": int(r.topic_id),
            "contest": r.contest, "year": int(r.year),
            "variant": r.variant, "problem_n": int(r.problem_n),
            "forum_text": r.post_canonical or r.body_noquote or "",
            "prob": probs_by_key.get(k),
            "wiki_sols": sols_by_key.get(k, []),
        })
    if limit:
        rows = rows[:limit]
    return rows


def load_done_post_ids(out_path: Path, retry_failed: bool = False):
    done, failed = set(), set()
    if not out_path.exists():
        return done
    with out_path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            pid = d.get("post_id")
            if pid is None:
                continue
            done.add(pid)
            if d.get("judge_failed"):
                failed.add(pid)
            else:
                failed.discard(pid)
    return done - failed if retry_failed else done


def base_rec(row):
    return {"post_id": row["post_id"], "topic_id": row["topic_id"],
            "contest": row["contest"], "year": row["year"],
            "variant": row["variant"], "problem_n": row["problem_n"],
            "n_wiki_solutions": len(row["wiki_sols"]),
            "n_wiki_shown": min(len(row["wiki_sols"]), MAX_WIKI_SOLS),
            "wiki_capped": len(row["wiki_sols"]) > MAX_WIKI_SOLS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forum", default=str(DEFAULT_FORUM))
    ap.add_argument("--wiki-solutions", default=str(DEFAULT_WIKI_SOLS))
    ap.add_argument("--wiki-problems", default=str(DEFAULT_WIKI_PROBS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=2000,
                    help="prompts per llm.chat() call (submit thousands)")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.93)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt rows whose only output is a "
                         "judge_failed sentinel (recovery pass)")
    ap.add_argument("--retries", type=int, default=4,
                    help="extra passes with a different seed on bad output "
                         "(this Qwen3.5-FP8 setup degenerates on ~half of "
                         "math-heavy prompts per pass; re-rolling converges)")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.forum, args.wiki_solutions, args.wiki_problems,
                     args.limit)
    done = load_done_post_ids(out_path, retry_failed=args.retry_failed)
    pending = [r for r in rows if r["post_id"] not in done]
    print(f"=== {len(rows):,} forum rows, {len(done):,} already done, "
          f"{len(pending):,} pending ===", flush=True)

    # rows that cannot be judged at all -> sentinel, no GPU needed
    fout = out_path.open("a")
    judgeable = []
    for r in pending:
        if r["prob"] is None or not str(
                r["prob"].get("problem_statement") or "").strip():
            fout.write(json.dumps({**base_rec(r),
                                   "judge_failed": "no_problem_statement"})
                       + "\n")
        elif not r["wiki_sols"]:
            fout.write(json.dumps({**base_rec(r),
                                   "judge_failed": "no_wiki_solutions"})
                       + "\n")
        else:
            judgeable.append(r)
    fout.flush()
    os.fsync(fout.fileno())
    print(f"=== {len(pending) - len(judgeable):,} sentinel rows "
          f"(no wiki problem/solutions), {len(judgeable):,} to judge ===",
          flush=True)
    if not judgeable:
        print("nothing to do")
        fout.close()
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
        enable_prefix_caching=True,  # problem+wiki prefix shared across posts
    )

    tok = llm.get_tokenizer()
    input_limit = args.max_model_len - args.max_tokens

    def build_convo(row, wiki_cap, forum_cap):
        sols = row["wiki_sols"][:MAX_WIKI_SOLS]
        user = build_user_prompt(row["prob"], sols, row["forum_text"],
                                 wiki_cap, forum_cap)
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]

    def n_tokens(convo):
        # render to text, then encode (transformers>=5 BatchEncoding trap)
        s = tok.apply_chat_template(convo, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
        return len(tok.encode(s, add_special_tokens=False))

    def build_convo_fitted(row):
        wiki_cap, forum_cap = WIKI_SOL_CHAR_CAP, FORUM_SOL_CHAR_CAP
        convo = build_convo(row, wiki_cap, forum_cap)
        while n_tokens(convo) > input_limit and (wiki_cap > 300
                                                 or forum_cap > 300):
            wiki_cap = max(300, wiki_cap // 2)
            forum_cap = max(300, forum_cap // 2)
            convo = build_convo(row, wiki_cap, forum_cap)
        if n_tokens(convo) > input_limit:
            return None  # unfittable even at minimum caps
        return convo

    t_start = time.time()
    n_done = n_ok = n_failed = 0
    try:
        queue = judgeable
        for attempt in range(args.retries + 1):
            if not queue:
                break
            if attempt == 0:
                sp = SamplingParams(temperature=0.0,
                                    max_tokens=args.max_tokens)
            else:
                # different seed, never repetition_penalty
                sp = SamplingParams(temperature=0.7, top_p=0.95,
                                    seed=args.seed + attempt,
                                    max_tokens=args.max_tokens)
            print(f"\n=== pass {attempt}: {len(queue):,} rows "
                  f"(temperature={sp.temperature}, "
                  f"seed={getattr(sp, 'seed', None)}) ===", flush=True)
            next_queue = []
            for ci in range(0, len(queue), args.chunk_size):
                batch_all = queue[ci:ci + args.chunk_size]
                batch, convos = [], []
                for r in batch_all:
                    convo = build_convo_fitted(r)
                    if convo is None:
                        fout.write(json.dumps(
                            {**base_rec(r), "judge_failed": "overlong",
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
                    # client-side validation error: engine survives, so
                    # isolate the poison row(s) one by one instead of dying
                    print(f"  chunk chat failed ({type(e).__name__}: "
                          f"{str(e)[:160]}); isolating row-by-row",
                          flush=True)
                    outs = []
                    for c_one in convos:
                        try:
                            outs.extend(llm.chat(
                                [c_one], sp, use_tqdm=False,
                                chat_template_kwargs={
                                    "enable_thinking": False}))
                        except Exception:
                            outs.append(None)
                dt = time.time() - t0
                chunk_ok = chunk_retry = 0
                from collections import Counter
                err_counter = Counter()
                err_samples = []
                for row, out in zip(batch, outs):
                    if out is None:  # failed even in isolation
                        fout.write(json.dumps(
                            {**base_rec(row), "judge_failed": "render_failed",
                             "attempts": attempt + 1}) + "\n")
                        n_done += 1
                        n_failed += 1
                        continue
                    raw = out.outputs[0].text if out.outputs else ""
                    n_shown = min(len(row["wiki_sols"]), MAX_WIKI_SOLS)
                    verdict, err = parse_verdict(raw, n_shown)
                    final = (attempt == args.retries)
                    if verdict is None:
                        err_counter[err] += 1
                        if len(err_samples) < 3:
                            snip = raw if len(raw) <= 400 else \
                                raw[:200] + " <...> " + raw[-200:]
                            err_samples.append(
                                (row["post_id"], err,
                                 snip.replace("\n", "\\n")))
                    if verdict is None and not final:
                        row["_last_error"] = err
                        next_queue.append(row)
                        chunk_retry += 1
                        continue
                    if verdict is None:  # terminal sentinel
                        fout.write(json.dumps(
                            {**base_rec(row), "judge_failed": f"bad_output:"
                             f"{err}", "attempts": attempt + 1,
                             "raw_head": raw[:300]}) + "\n")
                        n_failed += 1
                    else:
                        fout.write(json.dumps(
                            {**base_rec(row), **verdict,
                             "attempts": attempt + 1},
                            ensure_ascii=False) + "\n")
                        n_ok += 1
                    n_done += 1
                    chunk_ok += 1
                fout.flush()
                os.fsync(fout.fileno())  # checkpoint after every chunk
                elapsed = time.time() - t_start
                rate = max(n_done, 1) / max(elapsed, 1)
                print(f"  pass{attempt} chunk {ci // args.chunk_size + 1}: "
                      f"{len(batch)} prompts in {dt:.0f}s | "
                      f"accepted={chunk_ok} retry={chunk_retry} | "
                      f"total done={n_done:,} ok={n_ok:,} "
                      f"failed={n_failed:,} rate={rate * 60:.1f}/min",
                      flush=True)
                if err_counter:
                    print(f"    errors: {dict(err_counter.most_common(8))}",
                          flush=True)
                    for pid_, e_, s_ in err_samples:
                        print(f"    sample post={pid_} err={e_} raw: {s_}",
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
          f"failed={n_failed:,} ===")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
