#!/usr/bin/env python3
"""Patch for /lfs/skampere3/0/alexspan/scripts/llama_norm_extraction/run_sk3_batch.py.

Replaces the single-file gzip-append writer (line 284) with chunked output
to avoid long-running gzip-append corruption (feedback_no_long_running_gzip_append).

CHANGES:
- `output_path` (e.g. .../extracted_qwen.jsonl.gz) is now interpreted as the
  CHUNKS DIR: `<output_path>.chunks/` becomes the dir, one chunk file per batch.
- Each chunk file: `chunk_{ts}_{counter:06d}.jsonl.gz`, opened in "wt" mode,
  closed cleanly via context manager. No long-lived file handles.
- `read_done` scans all chunks AND the legacy file (if present) for back-compat.
- A migration helper (`migrate_legacy_to_chunks`) peels readable lines off any
  existing legacy file into `chunk_legacy_<sha>.jsonl.gz` inside the chunks
  dir, then renames the legacy file to `<path>.legacy_corrupted` so the
  reader skips it on subsequent runs.

This is the FULL replacement of run_sk3_batch.py. Drop in place after backing up.
"""
import argparse, gzip, hashlib, json, os, pathlib, random, sys, time, uuid, re as _re
from collections import defaultdict

ROOT = pathlib.Path(os.environ.get("NORM_SCRIPTS_ROOT",
                                    "/lfs/skampere3/0/alexspan/scripts/llama_norm_extraction"))
DATA_BASE = pathlib.Path("/lfs/skampere3/0/alexspan/data")
MODEL_DIR = "/lfs/skampere3/0/shared_hf_cache/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"

TASK_SOURCES = {
    "peer_review": {
        "input_path": str(DATA_BASE / "peer_review/peer_review_unified.parquet"),
        "input_loader": "parquet",
        "id_field": "review_id",
        "text_field": "review_text",
        "meta_fields": ["venue", "year", "decision"],
        "rubrics_jsonl": str(DATA_BASE / "peer_review/rubrics.jsonl"),
        "default_output": str(DATA_BASE / "peer_review/norm_extracted/extracted_qwen.jsonl.gz"),
    },
    "code_review": {
        "input_path": str(DATA_BASE / "code_review/code_review_pr_aggregated.csv.gz"),
        "input_loader": "csv_gz",
        "id_field": "paper_id",
        "text_field": "text",
        "meta_fields": ["language", "judgement", "n_comments"],
        "rubrics_jsonl": str(DATA_BASE / "code_review/rubrics.jsonl"),
        "default_output": str(DATA_BASE / "code_review/extracted_qwen.jsonl.gz"),
    },
    "notice_and_comment": {
        "input_path": str(DATA_BASE / "notice_and_comment/rtc_sections.parquet"),
        "input_loader": "parquet",
        "id_field": "rule_id",
        "text_field": "rtc_text",
        "meta_fields": ["agency", "rule_title", "rtc_text_len"],
        "rubrics_jsonl": str(DATA_BASE / "notice_and_comment/rubrics.jsonl"),
        "default_output": str(DATA_BASE / "notice_and_comment/extracted/extracted_qwen.jsonl.gz"),
    },
    "notice_and_comment_v2": {
        "input_path": str(DATA_BASE / "notice_and_comment/rtc_sections_backfill.parquet"),
        "input_loader": "parquet",
        "id_field": "rule_id",
        "text_field": "rtc_text",
        "meta_fields": ["agency", "rule_title", "rtc_text_len"],
        "rubrics_jsonl": str(DATA_BASE / "notice_and_comment/rubrics.jsonl"),
        "default_output": str(DATA_BASE / "notice_and_comment/extracted/extracted_qwen_v2backfill.jsonl.gz"),
    },
    "press_releases_full": {
        "input_path": str(DATA_BASE / "press_releases/pr_article_pairs_full.jsonl"),
        "input_loader": "jsonl",
        "id_field": "pair_id",
        "text_field": "text",
        "meta_fields": ["pr_company", "outlet"],
        "rubrics_jsonl": str(DATA_BASE / "press_releases/rubrics.jsonl"),
        "default_output": str(DATA_BASE / "press_releases/extracted/extracted_qwen_full.jsonl.gz"),
    },
    "humor_multi": {
        "input_path": str(DATA_BASE / "humor/standup_multi/filtered_threads.jsonl"),
        "input_loader": "jsonl",
        "id_field": "thread_id",
        "text_field": "text",
        "meta_fields": ["source", "post_title", "n_comments_kept"],
        "rubrics_jsonl": str(DATA_BASE / "humor/standup_reddit/rubrics.jsonl"),
        "default_output": str(DATA_BASE / "humor/standup_multi/extracted_qwen.jsonl.gz"),
    },
    "crse": {
        "input_path": str(DATA_BASE / "crse/input.jsonl"),
        "input_loader": "jsonl",
        "id_field": "unit_id",
        "text_field": "text",
        "meta_fields": ["score", "accepted", "primary_tag", "answer_year", "judgement", "split"],
        "rubrics_jsonl": str(DATA_BASE / "crse/rubrics.jsonl"),
        "default_output": str(DATA_BASE / "crse/extracted_qwen.jsonl.gz"),
    },
    "law_se": {
        "input_path": str(DATA_BASE / "law_se/input.jsonl"),
        "input_loader": "jsonl",
        "id_field": "unit_id",
        "text_field": "text",
        "meta_fields": ["source", "post_id", "score", "post_type", "creation_year"],
        "rubrics_jsonl": str(DATA_BASE / "law_se/rubrics.jsonl"),
        "default_output": str(DATA_BASE / "law_se/extracted_qwen.jsonl.gz"),
    },
    "reddit_supremecourt": {
        "input_path": str(DATA_BASE / "reddit_supremecourt/input.jsonl"),
        "input_loader": "jsonl",
        "id_field": "unit_id",
        "text_field": "text",
        "meta_fields": ["score", "post_title", "flair", "judgement", "weekday"],
        "rubrics_jsonl": str(DATA_BASE / "reddit_supremecourt/rubrics.jsonl"),
        "default_output": str(DATA_BASE / "reddit_supremecourt/extracted_qwen.jsonl.gz"),
    },
    "courtlistener_opinions": {
        "input_path": str(DATA_BASE / "courtlistener_opinions/input.jsonl"),
        "input_loader": "jsonl",
        "id_field": "unit_id",
        "text_field": "text",
        "meta_fields": ["court_id", "date_filed", "case_name", "posture", "slices", "cluster_id", "docket_id", "n_chars_orig"],
        "rubrics_jsonl": str(DATA_BASE / "courtlistener_opinions/rubrics.jsonl"),
        "default_output": str(DATA_BASE / "courtlistener_opinions/extracted_qwen.jsonl.gz"),
    },
}

# --- snap-to-source ---
_WS = _re.compile(r"\s+")
def _norm(s): return _WS.sub(" ", s).strip()

def snap(span, source):
    if not span: return span, "empty"
    if span in source: return span, "exact"
    if _norm(span) in _norm(source): return span, "ws_normalized"
    return span, "no_match"

# --- shared loaders ---
def load_template():
    return (ROOT / "template.txt").read_text()

def load_config(task):
    return json.load(open(ROOT / "configs" / f"{task}.json"))

def load_few_shots(task):
    return json.load(open(ROOT / "few_shots" / f"{task}_examples.json"))

def load_rubrics(task):
    return [json.loads(l) for l in open(TASK_SOURCES[task]["rubrics_jsonl"])]

def load_inputs(task, exclude_ids):
    src = TASK_SOURCES[task]
    loader = src["input_loader"]
    rows = []
    if loader == "jsonl":
        for line in open(src["input_path"]):
            r = json.loads(line)
            uid = str(r[src["id_field"]])
            if uid in exclude_ids: continue
            entry = {"unit_id": uid, "text": str(r[src["text_field"]])}
            for m in src["meta_fields"]:
                if m in r: entry[m] = r[m]
            rows.append(entry)
    elif loader in ("csv_gz", "parquet"):
        import pandas as pd
        if loader == "csv_gz":
            cols = [src["id_field"], src["text_field"]] + list(src["meta_fields"])
            df = pd.read_csv(src["input_path"], usecols=cols)
        else:
            df = pd.read_parquet(src["input_path"])
        df = df[~df[src["id_field"]].astype(str).isin(exclude_ids)]
        for _, r in df.iterrows():
            entry = {"unit_id": str(r[src["id_field"]]), "text": str(r[src["text_field"]])}
            for m in src["meta_fields"]:
                if m in r.index:
                    v = r[m]
                    if hasattr(v, "item"): v = v.item()
                    if v is not None and str(v) != "nan": entry[m] = v
            rows.append(entry)
    return rows

def fmt_rubric_block(rubrics):
    out = []
    for r in rubrics:
        rid = r["rubric_id"]
        name = r.get("name") or r.get("merged_name") or "?"
        desc = (r.get("description") or r.get("merged_description") or "")[:160].replace("\n", " ")
        out.append(f"[{rid}] {name} — {desc}")
    return "\n".join(out)

def fmt_worked_examples(demos):
    chunks = []
    for i, d in enumerate(demos, 1):
        in_text = d["input"]["text"]
        if len(in_text) > 1500: in_text = in_text[:1500] + "... [truncated]"
        ctx = ", ".join(f"{k}={d[k]}" for k in ("venue","language","judgement","agency") if k in d)
        chunks.append(
            f"## Example {i} ({d['label']}{', ' + ctx if ctx else ''})\n\n"
            f"INPUT (unit_id={d['input']['unit_id']}):\n{in_text}\n\n"
            f"OUTPUT (one line of JSON):\n{json.dumps(d['output'], ensure_ascii=False)}"
        )
    return "\n\n".join(chunks)

def build_prompt(template, config, rubric_block, worked, unit_id, input_text):
    return template.format(
        role=config["role"], input_unit=config["input_unit"],
        speaker_subject=config["speaker_subject"],
        speaker_possessive=config["speaker_possessive"],
        target=config["target"], norm_term=config["norm_term"],
        norm_term_plural=config["norm_term_plural"],
        inline_evidence_example=config["inline_evidence_example"],
        polarity_hint=config.get("polarity_hint",""),
        rubric_block=rubric_block, worked_examples=worked,
        unit_id=unit_id, input_text=input_text,
    )

def parse_output(text):
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("```",2)[1]
        if s.startswith("json"): s = s[4:]
        s = s.strip("` \n")
    if not s.rstrip().endswith("}"):
        return None, "truncated"
    try: return json.loads(s), None
    except Exception: pass
    try:
        import json_repair
        r = json_repair.loads(s)
        return (r, "repaired") if isinstance(r, dict) else (None, "not dict")
    except Exception as e:
        return None, f"parse: {e}"

def validate_and_snap(parsed, src):
    issues = []
    if not isinstance(parsed, dict): return ["not a dict"]
    passages = parsed.get("passages", [])
    if not isinstance(passages, list): return ["passages not a list"]
    for pi, p in enumerate(passages):
        if not isinstance(p, dict):
            issues.append(f"passage[{pi}] not dict")
            continue
        ptext = p.get("passage_text", "")
        if not isinstance(ptext, str): ptext = str(ptext)
        snapped, kind = snap(ptext, src)
        p["passage_text"] = snapped; p["_snap"] = kind
        if kind == "no_match": issues.append("passage no_match")
        for s in p.get("signals", []) if isinstance(p.get("signals", []), list) else []:
            if not isinstance(s, dict):
                issues.append(f"signal not dict"); continue
            stext = s.get("signal_text", "")
            if not isinstance(stext, str): stext = str(stext)
            snapped, kind = snap(stext, p["passage_text"])
            s["signal_text"] = snapped; s["_snap"] = kind
            if kind == "no_match": issues.append("signal no_match")
    return issues

# --- bad-output detection for retry-on-loop ---
VALID_TYPES = {"observation", "complaint", "praise", "suggestion"}
VALID_POLS  = {"positive", "negative", "neutral", "mixed"}
_LOOP_RE = _re.compile(r"(.{4,40})\1{8,}")

def is_loop_string(s):
    if not isinstance(s, str): return True
    if len(s) > 800: return True
    return bool(_LOOP_RE.search(s))

def is_bad_output(parsed, vocab_ids):
    """LENIENT validator: only retry on STRUCTURAL breaks or loop garbage.
    OOV rubric IDs and invalid enums are kept as-is (just data quality artifacts,
    not corruption — the signal_text is what matters). Avoids 65% retry rate
    when LLM tags rubric IDs outside vocab (especially for tasks with empty rubric vocab).
    """
    if not isinstance(parsed, dict): return True
    passages = parsed.get("passages", [])
    if not isinstance(passages, list): return True
    for p in passages:
        if not isinstance(p, dict): return True
        if is_loop_string(p.get("passage_text", "")): return True
        sigs = p.get("signals", [])
        if not isinstance(sigs, list): return True
        for s in sigs:
            if not isinstance(s, dict): return True
            # NOTE: enum and rubric_matches checks REMOVED — too strict, caused 2-3x retry overhead
            # If signal_type/polarity is non-standard, it survives downstream; if rubric_matches
            # has OOV ints, they survive too (filterable post-hoc by readers).
            if is_loop_string(s.get("signal_text", "")): return True
    return False

# ============================================================================
# CHUNKED OUTPUT (this is the fix)
# ============================================================================

def chunks_dir_for(output_path):
    """Derive chunks dir from legacy single-file output_path."""
    p = pathlib.Path(str(output_path).rstrip("/"))
    # If callers already pass a dir, use it directly
    if p.is_dir() or not p.suffix:
        return p
    # Otherwise: <parent>/<stem_without_extensions>.chunks/
    name = p.name
    # strip both .jsonl.gz and .jsonl
    if name.endswith(".jsonl.gz"): name = name[:-len(".jsonl.gz")]
    elif name.endswith(".jsonl"):  name = name[:-len(".jsonl")]
    return p.parent / f"{name}.chunks"

def legacy_marker_for(output_path):
    """Where we move the original (corrupted) legacy file after migration."""
    return pathlib.Path(str(output_path) + ".legacy_corrupted")

def read_done_from_chunks(chunks_dir, legacy_path=None):
    """Scan all chunks + (optionally) a legacy single file. Return set of unit_ids done."""
    done = set()
    # Chunks
    if chunks_dir.exists():
        for cf in sorted(chunks_dir.glob("chunk_*.jsonl.gz")):
            try:
                with gzip.open(cf, "rt") as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            if "unit_id" in d: done.add(str(d["unit_id"]))
                        except Exception:
                            pass
            except Exception as e:
                print(f"warn reading {cf}: {e}", file=sys.stderr)
    # Legacy (back-compat — only matters during migration, after migration this is renamed)
    if legacy_path is not None and pathlib.Path(legacy_path).exists():
        try:
            with gzip.open(legacy_path, "rt") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if "unit_id" in d: done.add(str(d["unit_id"]))
                    except Exception:
                        pass
        except Exception as e:
            print(f"warn reading legacy {legacy_path}: {e}", file=sys.stderr)
    return done

def migrate_legacy_to_chunks(legacy_path, chunks_dir):
    """
    If a legacy single .jsonl.gz exists, peel its readable lines into a single
    chunk file inside chunks_dir, then rename the legacy file to .legacy_corrupted.
    Idempotent: if legacy_path doesn't exist, no-op.
    """
    legacy_path = pathlib.Path(legacy_path)
    if not legacy_path.exists():
        return 0
    marker = legacy_marker_for(legacy_path)
    if marker.exists() and not legacy_path.exists():
        return 0  # already migrated
    chunks_dir.mkdir(parents=True, exist_ok=True)
    # Distinct chunk file name based on legacy file hash so re-runs don't overwrite
    h = hashlib.sha1(str(legacy_path).encode()).hexdigest()[:10]
    chunk_path = chunks_dir / f"chunk_legacy_{h}.jsonl.gz"
    if chunk_path.exists():
        print(f"  legacy chunk already exists: {chunk_path}", file=sys.stderr)
        n_records = sum(1 for _ in gzip.open(chunk_path, "rt"))
    else:
        n_records = 0
        with gzip.open(chunk_path, "wt") as fo:
            try:
                with gzip.open(legacy_path, "rt") as fi:
                    for line in fi:
                        line = line.rstrip("\n")
                        if not line: continue
                        # validate it's parseable JSON before keeping
                        try:
                            json.loads(line)
                        except Exception:
                            continue
                        fo.write(line + "\n")
                        n_records += 1
            except Exception as e:
                # gzip corruption mid-stream — fine, we keep what we got
                print(f"  legacy read stopped at line {n_records}: {e}", file=sys.stderr)
    # Rename legacy to marker so future runs ignore it
    if legacy_path.exists():
        legacy_path.rename(marker)
        print(f"  migrated legacy: {legacy_path} -> {marker} ({n_records} records preserved as {chunk_path.name})", file=sys.stderr)
    return n_records

# --- core batch loop with bad-output retry ---
def _llm_chat(llm, messages, sampling):
    try:
        return llm.chat(messages, sampling_params=sampling,
                        chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    except Exception as e:
        return e

def process_task_batched(llm, sampling_first, sampling_retries, template, task, output_path,
                          vocab_ids, max_text_chars=20000, batch_size=1000, max_units=0):
    """Process one task end-to-end via batched LLM.chat. Resume-safe + retry-on-loop + chunked write."""
    config = load_config(task)
    demos = load_few_shots(task)
    rubrics = load_rubrics(task)
    rubric_block = fmt_rubric_block(rubrics)
    worked = fmt_worked_examples(demos)

    chunks_dir = chunks_dir_for(output_path)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    # First: migrate any legacy single-file output into a chunk
    migrate_legacy_to_chunks(output_path, chunks_dir)
    done = read_done_from_chunks(chunks_dir)
    print(f"=== {task} BATCH (chunked) ===", flush=True)
    print(f"=== chunks_dir: {chunks_dir} ===", flush=True)
    print(f"=== resuming: {len(done)} already-done ===", flush=True)

    inputs = load_inputs(task, exclude_ids=done)
    if max_units > 0: inputs = inputs[:max_units]
    print(f"=== {len(inputs)} units to process ===", flush=True)
    if not inputs:
        return

    t_start = time.time()
    n_done = n_ok = n_err = n_retried = 0
    run_ts = time.strftime("%Y%m%dT%H%M%S")
    chunk_counter = 0

    for chunk_start in range(0, len(inputs), batch_size):
        chunk = inputs[chunk_start:chunk_start + batch_size]
        messages_batch = []
        chunk_texts = []
        for r in chunk:
            text = r["text"]
            if max_text_chars and len(text) > max_text_chars:
                text = text[:max_text_chars] + "\n\n... [truncated]"
            chunk_texts.append(text)
            prompt = build_prompt(template, config, rubric_block, worked, r["unit_id"], text)
            messages_batch.append([{"role": "user", "content": prompt}])

        t_batch = time.time()
        outputs = _llm_chat(llm, messages_batch, sampling_first)

        # Open a FRESH chunk file for this batch. Close cleanly via context manager.
        chunk_path = chunks_dir / f"chunk_{run_ts}_{chunk_counter:06d}.jsonl.gz"
        chunk_counter += 1
        with gzip.open(chunk_path, "wt") as fo:
            if isinstance(outputs, Exception):
                print(f"  BATCH FAILED chunk {chunk_start}: {outputs}", flush=True)
                for r in chunk:
                    fo.write(json.dumps({"unit_id": r["unit_id"], "ok": False, "error": f"batch_exception: {outputs}"}) + "\n")
                    n_done += 1; n_err += 1
                continue  # file closes via context manager on next iter

            row_results = [None] * len(chunk)
            need_retry_idx = []
            for i, (r, out) in enumerate(zip(chunk, outputs)):
                raw = out.outputs[0].text if out.outputs else ""
                parsed, perr = parse_output(raw)
                if parsed is None or is_bad_output(parsed, vocab_ids):
                    need_retry_idx.append(i)
                else:
                    row_results[i] = (parsed, None)

            for retry_pass, retry_sp in enumerate(sampling_retries):
                if not need_retry_idx: break
                n_retried += len(need_retry_idx)
                retry_msgs = [messages_batch[i] for i in need_retry_idx]
                retry_outs = _llm_chat(llm, retry_msgs, retry_sp)
                if isinstance(retry_outs, Exception):
                    print(f"  RETRY {retry_pass} FAILED: {retry_outs}", flush=True)
                    break
                new_need = []
                for ix, out in zip(need_retry_idx, retry_outs):
                    raw = out.outputs[0].text if out.outputs else ""
                    parsed, perr = parse_output(raw)
                    if parsed is None or is_bad_output(parsed, vocab_ids):
                        new_need.append(ix)
                    else:
                        row_results[ix] = (parsed, None)
                need_retry_idx = new_need

            for i, r in enumerate(chunk):
                meta = {k: r[k] for k in r if k not in ("unit_id", "text")}
                if row_results[i] is None:
                    res = {"unit_id": r["unit_id"], "ok": False, "error": "bad_after_retries", **meta}
                    n_err += 1
                else:
                    parsed, _ = row_results[i]
                    issues = validate_and_snap(parsed, chunk_texts[i])
                    res = {"unit_id": r["unit_id"], "ok": len(issues) == 0, "issues": issues,
                           "parsed": parsed, **meta}
                    if res["ok"]: n_ok += 1
                fo.write(json.dumps(res) + "\n")
                n_done += 1
        # chunk file closes cleanly here

        batch_elapsed = time.time() - t_batch
        total_elapsed = time.time() - t_start
        rate_min = n_done / max(total_elapsed/60, 1e-9)
        eta_h = (len(inputs) - n_done) / max(rate_min, 1e-9) / 60
        print(f"  [{n_done:>6}/{len(inputs)}] batch={len(chunk)} batch_s={batch_elapsed:.1f} "
              f"ok={n_ok} err={n_err} retried={n_retried} rate={rate_min:.1f}/min ETA={eta_h:.1f}h "
              f"chunk={chunk_path.name}", flush=True)

    total_elapsed = time.time() - t_start
    print(f"=== {task} DONE: n_done={n_done} ok={n_ok} err={n_err} elapsed={total_elapsed/60:.1f}min ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="single task (use --tasks for multi)")
    ap.add_argument("--tasks", help="comma-separated tasks to process sequentially through one LLM")
    ap.add_argument("--output", help="output path (defaults from TASK_SOURCES; only valid with --task)")
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--max-text-chars", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem-util", type=float, default=0.93)
    ap.add_argument("--max-units", type=int, default=0, help="limit per task (0=all)")
    args = ap.parse_args()

    if args.task and args.tasks:
        sys.exit("--task XOR --tasks, not both")
    tasks = [args.task] if args.task else (args.tasks.split(",") if args.tasks else None)
    if not tasks:
        sys.exit("specify --task or --tasks")

    for t in tasks:
        if t not in TASK_SOURCES:
            sys.exit(f"unknown task: {t}")

    print(f"=== loading LLM {MODEL_DIR.split('/')[-1]} ===", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL_DIR,
        dtype="auto",
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        max_model_len=args.max_model_len,
    )
    sampling_first = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, seed=0)
    sampling_retries = [
        SamplingParams(max_tokens=args.max_tokens, temperature=0.0, seed=12345),
        SamplingParams(max_tokens=args.max_tokens, temperature=0.1, seed=67890),
        SamplingParams(max_tokens=args.max_tokens, temperature=0.2, seed=99999),
    ]
    template = load_template()

    for task in tasks:
        out = args.output or TASK_SOURCES[task]["default_output"]
        vocab_ids = {r["rubric_id"] for r in load_rubrics(task)}
        process_task_batched(llm, sampling_first, sampling_retries, template, task, out,
                              vocab_ids=vocab_ids,
                              max_text_chars=args.max_text_chars,
                              batch_size=args.batch_size,
                              max_units=args.max_units)

if __name__ == "__main__":
    main()
