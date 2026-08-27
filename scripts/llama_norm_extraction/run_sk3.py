#!/usr/bin/env python3
"""
Bulk norm-adjacent extraction via local vLLM (Qwen3.5-122B-A10B-FP8) on sk3.
Mirrors run_openrouter.py prompt/template/few-shots, but talks to local OpenAI-compatible vLLM.

Two modes:
  --mode benchmark --n 100      : process N reviews, print throughput
  --mode bulk                   : full sweep over peer_review_unified.parquet, resumable jsonl.gz

Usage:
  python run_sk3.py --task peer_review --mode benchmark --n 100 --concurrency 64 --run-name bench100
  python run_sk3.py --task peer_review --mode bulk --concurrency 64 --output /lfs/.../extracted_qwen.jsonl.gz
"""
import argparse, gzip, json, os, pathlib, random, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

ROOT = pathlib.Path(__file__).parent
DATA_ROOT = pathlib.Path(os.environ.get("NORM_DATA_ROOT", "/lfs/skampere3/0/alexspan/data/peer_review"))
SMOKE_INPUT = DATA_ROOT / "input_20.jsonl"
RUBRICS_JSONL = DATA_ROOT / "rubrics.jsonl"
PARQUET_PATH = DATA_ROOT / "peer_review_unified.parquet"

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8001/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.5-122b-a10b-fp8")


def load_template():
    return (ROOT / "template.txt").read_text()


def load_config(task):
    return json.load(open(ROOT / "configs" / f"{task}.json"))


def load_few_shots(task):
    return json.load(open(ROOT / "few_shots" / f"{task}_examples.json"))


def load_rubrics():
    return [json.loads(l) for l in open(RUBRICS_JSONL)]


def fmt_rubric_block(rubrics):
    lines = []
    for r in rubrics:
        rid = r["rubric_id"]
        name = r.get("name") or r.get("merged_name") or "?"
        desc = (r.get("description") or "")[:160].replace("\n", " ")
        lines.append(f"[{rid}] {name} — {desc}")
    return "\n".join(lines)


def fmt_worked_examples(demos):
    chunks = []
    for i, d in enumerate(demos, 1):
        in_text = d["input"]["text"]
        if len(in_text) > 1500:
            in_text = in_text[:1500] + "... [truncated]"
        out_json = json.dumps(d["output"], ensure_ascii=False)
        chunks.append(
            f"## Example {i} ({d['label']}, venue={d['venue']})\n\n"
            f"INPUT (unit_id={d['input']['unit_id']}):\n{in_text}\n\n"
            f"OUTPUT (one line of JSON):\n{out_json}"
        )
    return "\n\n".join(chunks)


def build_prompt(unit_id, input_text, template, config, rubric_block, worked):
    return template.format(
        role=config["role"],
        input_unit=config["input_unit"],
        speaker_subject=config["speaker_subject"],
        speaker_possessive=config["speaker_possessive"],
        target=config["target"],
        norm_term=config["norm_term"],
        norm_term_plural=config["norm_term_plural"],
        inline_evidence_example=config["inline_evidence_example"],
        polarity_hint=config.get("polarity_hint", ""),
        rubric_block=rubric_block,
        worked_examples=worked,
        unit_id=unit_id,
        input_text=input_text,
    )


def call_vllm(prompt, max_tokens=8000, temperature=0.0, timeout=900):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Anti-loop: Qwen3.5-FP8 greedy mode at high concurrency enters degenerate
        # token-repetition cycles (~20% of requests). repetition_penalty=1.05 is
        # multiplicative on logits — barely affects normal exact-substring extraction
        # but strongly breaks single-token loops. Even with it, ~15-20% of requests
        # still loop; we accept this as noise (259K * 0.8 = 207K extractions is plenty).
        "repetition_penalty": 1.05,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        # vLLM-specific: route via extra fields. OpenAI server accepts chat_template_kwargs at top-level.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        VLLM_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# --- parse + snap (copied from run_openrouter.py) ---
def parse_output(text):
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip("` \n")
    s_stripped = s.rstrip()
    is_truncated = not s_stripped.endswith("}")
    try:
        return json.loads(s), None
    except Exception:
        pass
    if is_truncated:
        return None, "truncated (no closing brace)"
    try:
        import json_repair
        repaired = json_repair.loads(s)
        if isinstance(repaired, dict):
            return repaired, "repaired"
    except Exception:
        pass
    start = s.find("{")
    if start == -1:
        return None, "no opening brace"
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                substr = s[start:i + 1]
                try:
                    return json.loads(substr), None
                except Exception:
                    try:
                        import json_repair
                        return json_repair.loads(substr), "repaired"
                    except Exception as e:
                        return None, f"parse failed: {e}"
    return None, "no balanced object"


import re as _re
_WS = _re.compile(r"\s+")


def _norm(s):
    return _WS.sub(" ", s).strip()


def snap_to_source(span, source, min_len=20):
    if not span:
        return span, "empty"
    if span in source:
        return span, "exact"
    nspan = _norm(span)
    nsrc = _norm(source)
    if nspan in nsrc:
        i = nsrc.find(nspan)
        src_idx, n_idx = 0, 0
        start = None
        while src_idx < len(source) and n_idx < i + len(nspan):
            if start is None and n_idx == i:
                start = src_idx
            c = source[src_idx]
            if c.isspace():
                if src_idx == 0 or not source[src_idx - 1].isspace():
                    n_idx += 1
            else:
                n_idx += 1
            src_idx += 1
        if start is not None:
            return source[start:src_idx].rstrip(), "ws_normalized"
    if len(span) >= min_len:
        anchor = span[:min_len]
        nanchor = _norm(anchor)
        idx = nsrc.find(nanchor)
        if idx != -1:
            return span, "anchor_only"
    return span, "no_match"


def validate_and_snap(parsed, source_text):
    issues = []
    snaps = {"exact": 0, "ws_normalized": 0, "no_match": 0, "anchor_only": 0, "empty": 0}
    if not isinstance(parsed, dict):
        return ["not a dict"], snaps
    for required in ("unit_id", "passages"):
        if required not in parsed:
            issues.append(f"missing key: {required}")
    passages = parsed.get("passages", [])
    if not isinstance(passages, list):
        issues.append("passages not a list")
        return issues, snaps
    for pi, p in enumerate(passages):
        if not isinstance(p, dict):
            issues.append(f"passage[{pi}] not a dict")
            continue
        ptext_orig = p.get("passage_text", "")
        ptext, kind = snap_to_source(ptext_orig, source_text)
        snaps[kind] += 1
        p["passage_text"] = ptext
        p["_snap_kind"] = kind
        if kind in ("no_match", "anchor_only"):
            issues.append(f"passage[{pi}] snap={kind}")
        for si, s in enumerate(p.get("signals", [])):
            if not isinstance(s, dict):
                issues.append(f"passage[{pi}].signal[{si}] not a dict")
                continue
            stext_orig = s.get("signal_text", "")
            stext, skind = snap_to_source(stext_orig, ptext)
            snaps[skind] = snaps.get(skind, 0) + 1
            s["signal_text"] = stext
            s["_snap_kind"] = skind
            if skind in ("no_match", "anchor_only"):
                issues.append(f"passage[{pi}].signal[{si}] snap={skind}")
            if s.get("signal_type") not in ("complaint", "praise", "observation", "suggestion"):
                issues.append(f"passage[{pi}].signal[{si}] bad signal_type")
            if s.get("polarity") not in ("positive", "negative", "neutral"):
                issues.append(f"passage[{pi}].signal[{si}] bad polarity")
    return issues, snaps


def _clean(v):
    """Convert pandas NA / numpy types to JSON-safe scalars."""
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except Exception:
        pass
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def process_one(prompt, source_text, unit_id, meta):
    t0 = time.time()
    try:
        resp = call_vllm(prompt)
    except Exception as e:
        return {"unit_id": unit_id, "ok": False, "error": f"http: {e}", "elapsed": time.time() - t0, **meta}
    if "error" in resp:
        return {"unit_id": unit_id, "ok": False, "error": str(resp["error"]), "elapsed": time.time() - t0, **meta}
    try:
        msg = resp["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning") or ""
        if not raw:
            return {"unit_id": unit_id, "ok": False, "error": "empty content", "elapsed": time.time() - t0, **meta}
    except Exception:
        return {"unit_id": unit_id, "ok": False, "error": "no choices", "elapsed": time.time() - t0, **meta}
    parsed, perr = parse_output(raw)
    if parsed is None:
        return {"unit_id": unit_id, "ok": False, "error": f"parse: {perr}", "raw_text": raw[:2000],
                "elapsed": time.time() - t0, **meta}
    try:
        issues, snaps = validate_and_snap(parsed, source_text)
    except Exception as e:
        return {"unit_id": unit_id, "ok": False, "error": f"validate: {e}",
                "raw_text": raw[:2000], "elapsed": time.time() - t0, **meta}
    return {
        "unit_id": unit_id,
        "ok": len(issues) == 0,
        "issues": issues,
        "snap_stats": snaps,
        "parsed": parsed,
        "usage": resp.get("usage"),
        "elapsed": time.time() - t0,
        **meta,
    }


# --- I/O helpers ---
def load_inputs_benchmark(n, seed, exclude_ids, max_text_chars=60000):
    """Stratified sample of N reviews from the parquet, excluding few-shot ids."""
    import pandas as pd
    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["review_text"].notna() & (df["review_text"].str.len() > 50)]
    df = df[~df["review_id"].astype(str).isin({str(x) for x in exclude_ids})]
    # Stratified-ish: sample equal-ish counts per venue, take min(n_per_v, available)
    venues = list(df["venue"].unique())
    per_v = max(1, n // len(venues))
    parts = []
    for v in venues:
        parts.append(df[df["venue"] == v].sample(n=min(per_v, (df["venue"] == v).sum()),
                                                  random_state=seed))
    sample = pd.concat(parts).sample(frac=1, random_state=seed)
    sample = sample.iloc[:n]
    out = []
    for _, row in sample.iterrows():
        text = row["review_text"]
        if len(text) > max_text_chars:
            text = text[:max_text_chars]
        out.append({
            "review_id": str(row["review_id"]),
            "review_text": text,
            "venue": _clean(row["venue"]),
            "decision": _clean(row.get("decision")),
        })
    return out


def load_inputs_bulk(parquet_path, seed, max_text_chars=60000):
    """Stream reviews from the unified parquet. Stratify across venues."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    # Stratified shuffle: interleave venues so failure modes surface early
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df[df["review_text"].notna() & (df["review_text"].str.len() > 50)]
    for _, row in df.iterrows():
        text = row["review_text"]
        if len(text) > max_text_chars:
            text = text[:max_text_chars]
        yield {
            "review_id": str(row["review_id"]),
            "review_text": text,
            "venue": _clean(row["venue"]),
            "decision": _clean(row.get("decision")),
            "paper_id": str(row.get("paper_id", "")),
        }


def load_done_ids(output_path):
    done = set()
    if not pathlib.Path(output_path).exists():
        return done
    try:
        with gzip.open(output_path, "rt") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "unit_id" in d:
                        done.add(d["unit_id"])
                except Exception:
                    pass
    except Exception as e:
        print(f"warn: error reading existing output: {e}", file=sys.stderr)
    return done


# --- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer_review")
    ap.add_argument("--mode", choices=["benchmark", "bulk"], default="benchmark")
    ap.add_argument("--n", type=int, default=100, help="benchmark mode: number of reviews")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--run-name", default="bench100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="/lfs/skampere3/0/alexspan/data/peer_review/norm_extracted/extracted_qwen.jsonl.gz")
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--max-text-chars", type=int, default=60000)
    args = ap.parse_args()

    template = load_template()
    config = load_config(args.task)
    demos = load_few_shots(args.task)
    rubrics = load_rubrics()
    rubric_block = fmt_rubric_block(rubrics)
    worked = fmt_worked_examples(demos)

    fs_ids = {d["input"]["unit_id"] for d in demos}

    if args.mode == "benchmark":
        inputs = load_inputs_benchmark(args.n, args.seed, fs_ids)
        run_dir = ROOT / "runs" / f"{args.task}_{args.run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = run_dir / "results.jsonl"
        print(f"=== BENCHMARK MODE: {len(inputs)} reviews, concurrency={args.concurrency} ===")
        print(f"=== model={MODEL}  url={VLLM_URL} ===")
        print(f"=== rubric block: {len(rubric_block):,} chars ({len(rubrics)} rubrics) ===")
        print(f"=== output: {out_jsonl} ===\n")

        results = []
        t_start = time.time()
        total_in_tok = 0
        total_out_tok = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex, open(out_jsonl, "w") as fo:
            futs = {}
            for r in inputs:
                text = r["review_text"]
                if len(text) > args.max_text_chars:
                    text = text[:args.max_text_chars]
                prompt = build_prompt(r["review_id"], text, template, config, rubric_block, worked)
                meta = {"venue": r.get("venue"), "decision": r.get("decision")}
                futs[ex.submit(process_one, prompt, text, r["review_id"], meta)] = r
            done_n = 0
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                done_n += 1
                u = res.get("usage") or {}
                total_in_tok += u.get("prompt_tokens", 0)
                total_out_tok += u.get("completion_tokens", 0)
                fo.write(json.dumps(res) + "\n")
                fo.flush()
                ok = "Y" if res["ok"] else "N"
                p = res.get("parsed")
                n_pass = n_sig = 0
                if isinstance(p, dict):
                    passages = p.get("passages", [])
                    if isinstance(passages, list):
                        n_pass = len(passages)
                        for pp in passages:
                            if isinstance(pp, dict):
                                sigs = pp.get("signals", [])
                                if isinstance(sigs, list):
                                    n_sig += len(sigs)
                print(f"[{done_n:3d}/{len(inputs)}] {ok} {res['unit_id']:>10} pass={n_pass:>2} sig={n_sig:>3} {res['elapsed']:5.1f}s issues={len(res.get('issues',[]))}")

        wall = time.time() - t_start
        n_parsed = sum(isinstance(r.get("parsed"), dict) for r in results)
        n_ok = sum(r["ok"] for r in results)
        snap_exact = sum((r.get("snap_stats") or {}).get("exact", 0) for r in results)
        snap_ws = sum((r.get("snap_stats") or {}).get("ws_normalized", 0) for r in results)
        snap_no = sum((r.get("snap_stats") or {}).get("no_match", 0) for r in results)
        snap_anchor = sum((r.get("snap_stats") or {}).get("anchor_only", 0) for r in results)
        total_pass = 0; total_sig = 0
        for r in results:
            parsed = r.get("parsed")
            if not isinstance(parsed, dict): continue
            passages = parsed.get("passages", [])
            if not isinstance(passages, list): continue
            total_pass += len(passages)
            for p in passages:
                if isinstance(p, dict):
                    sigs = p.get("signals", [])
                    if isinstance(sigs, list):
                        total_sig += len(sigs)
        mean_sig = total_sig / max(1, n_parsed)

        print(f"\n=== BENCHMARK SUMMARY ===")
        print(f"wall clock: {wall:.1f}s   throughput: {len(results)/wall*60:.1f} reviews/min")
        print(f"input tokens: {total_in_tok:,}  output tokens: {total_out_tok:,}")
        print(f"out tok/s: {total_out_tok/wall:.0f}   in tok/s: {total_in_tok/wall:.0f}")
        print(f"parsed: {n_parsed}/{len(results)} ({100*n_parsed/len(results):.0f}%)")
        print(f"valid (all snaps clean, all enums valid): {n_ok}/{len(results)} ({100*n_ok/len(results):.0f}%)")
        denom_snaps = snap_exact + snap_ws + snap_no + snap_anchor
        print(f"snap stats: exact={snap_exact}  ws_norm={snap_ws}  anchor_only={snap_anchor}  no_match={snap_no}")
        if denom_snaps:
            print(f"substring validity: {100*(snap_exact+snap_ws)/denom_snaps:.0f}% (target >=60%)")
        print(f"mean signals/review: {mean_sig:.1f} (target ~16-20)")
        print(f"total signals: {total_sig}  total passages: {total_pass}")
        print(f"results: {out_jsonl}")
        return

    # ============ BULK MODE ============
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(out_path)
    print(f"=== BULK MODE ===")
    print(f"=== output: {out_path} ===")
    print(f"=== resuming: {len(done_ids):,} already-done ===")

    # Skip few-shot IDs too
    skip = set(done_ids) | {str(x) for x in fs_ids}

    pending = []
    print(f"=== loading parquet... ===")
    for r in load_inputs_bulk(PARQUET_PATH, args.seed, args.max_text_chars):
        if r["review_id"] not in skip:
            pending.append(r)
    print(f"=== {len(pending):,} reviews to process ===")
    print(f"=== concurrency={args.concurrency}  checkpoint_every={args.checkpoint_every} ===\n")

    t_start = time.time()
    completed_since_ckpt = 0
    total_completed = 0
    n_ok_total = 0
    n_parsed_total = 0
    fout = gzip.open(out_path, "at")  # append text mode

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            # Submit in chunks to bound memory of pending futures
            chunk_size = args.concurrency * 8
            for chunk_start in range(0, len(pending), chunk_size):
                chunk = pending[chunk_start:chunk_start + chunk_size]
                futs = {}
                for r in chunk:
                    text = r["review_text"]
                    if len(text) > args.max_text_chars:
                        text = text[:args.max_text_chars]
                    prompt = build_prompt(r["review_id"], text, template, config, rubric_block, worked)
                    meta = {"venue": r.get("venue"), "decision": r.get("decision"), "paper_id": r.get("paper_id")}
                    futs[ex.submit(process_one, prompt, text, r["review_id"], meta)] = r
                for fut in as_completed(futs):
                    try:
                        res = fut.result()
                    except Exception as e:
                        r = futs[fut]
                        res = {"unit_id": r["review_id"], "ok": False, "error": f"exc: {e}",
                               "venue": r.get("venue"), "decision": r.get("decision")}
                    # Drop heavy raw text on success to keep output lean
                    if res.get("ok") and "raw_text" in res:
                        del res["raw_text"]
                    fout.write(json.dumps(res) + "\n")
                    completed_since_ckpt += 1
                    total_completed += 1
                    if isinstance(res.get("parsed"), dict):
                        n_parsed_total += 1
                    if res.get("ok"):
                        n_ok_total += 1
                    if completed_since_ckpt >= args.checkpoint_every:
                        fout.flush()
                        os.fsync(fout.fileno())
                        completed_since_ckpt = 0
                        elapsed = time.time() - t_start
                        rate = total_completed / elapsed
                        eta_sec = (len(pending) - total_completed) / max(rate, 0.01)
                        print(f"[{total_completed:>6}/{len(pending):>6}] "
                              f"ok={n_ok_total} parsed={n_parsed_total} "
                              f"rate={rate*60:.1f}/min "
                              f"eta={eta_sec/3600:.1f}h "
                              f"elapsed={elapsed/3600:.2f}h",
                              flush=True)
    finally:
        fout.flush()
        try:
            os.fsync(fout.fileno())
        except Exception:
            pass
        fout.close()
        wall = time.time() - t_start
        print(f"\n=== BULK DONE ===")
        print(f"wall: {wall/3600:.2f}h  completed: {total_completed:,}  ok: {n_ok_total:,}  parsed: {n_parsed_total:,}")
        print(f"output: {out_path}")


if __name__ == "__main__":
    main()
