"""
Offline batch classifier for the full 361K rubrics parquet using vLLM.

Features:
  - vLLM `LLM` class (offline, continuous batching) — much higher throughput
    than the api_server. Single GPU is plenty for 361K rubrics in 30-60 min.
  - Per-task prompts assembled via classify_rubric_llama_prompt
  - guided_json constraint with xgrammar — minItems:2 on inputs enforced at
    decode time, JSON shape guaranteed.
  - Chunked processing with one JSONL FILE PER CHUNK in `--chunks-dir`. Each
    chunk N is the deterministic slice df[N*chunk_size:(N+1)*chunk_size] of the
    SORTED parquet — so different workers compute identical chunk boundaries.
  - O_CREAT|O_EXCL lock files (chunk_NNNNNN.lock) coordinate multiple workers:
    a chunk is claimed atomically. Workers skip chunks whose final .jsonl
    already exists or whose .lock file is fresh. Stale locks (mtime older than
    --lock-stale-min, default 30 min) can be re-claimed by another worker.
    => Safe to launch a second worker on another GPU at any time; it'll pick
       up whatever chunks the first hasn't claimed yet.
  - Two-pass retry runs PER CHUNK (T=0 → T=0.3 on errors only) so each
    chunk_NNNNNN.jsonl is fully resolved before being committed.
  - Atomic write: each chunk is written to chunk_NNNNNN.jsonl.tmp first, then
    os.replace()'d to its final name. Partial files never poison a resume.
  - Sort by task so vLLM's prefix cache maximizes hits within each chunk.
  - At end (if not --jsonl-only), merges all chunk JSONLs (+ optional legacy
    monolithic JSONL) into a parquet.

Inputs:
  --input         : parquet with (page_id, task, subtask_short, rubric_idx,
                    rubric_name, rubric_description, rubric_guidance).
  --output        : parquet to write at the end.
  --chunks-dir    : directory holding chunk_NNNNNN.jsonl + .lock files.
  --cache-jsonl   : legacy single-file JSONL (read-only; rows here count as
                    already cached so they aren't re-classified).
  --lock-stale-min: minutes after which a lock file is considered abandoned.

Usage:
  # Single worker, full run, resumable
  CUDA_VISIBLE_DEVICES=4 python3 batch_classify_vllm.py

  # Add a second worker on another free GPU later — it'll pick up unclaimed chunks
  CUDA_VISIBLE_DEVICES=2 python3 batch_classify_vllm.py
"""

from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
from tqdm.auto import tqdm

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(ROOT / "scripts"))

# Set env BEFORE importing torch/vllm
os.environ.setdefault("HOME", "/lfs/skampere3/0/alexspan")
os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/alexspan/.cache/huggingface")
os.environ.setdefault("TRITON_CACHE_DIR", "/lfs/skampere3/0/alexspan/.cache/triton")
os.environ.setdefault("VLLM_CACHE_ROOT", "/lfs/skampere3/0/alexspan/.cache/vllm")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/lfs/skampere3/0/alexspan/.cache/torchinductor")
os.environ.setdefault("TMPDIR", "/lfs/skampere3/0/alexspan/tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import re
from classify_rubric_llama_prompt import build_prompt_for_task, JSON_SCHEMA_LLAMA, SCHEMA_HINT

# Using BF16 Meta Llama-3.3-70B-Instruct instead of NVIDIA's FP8 version, because
# the FP8 checkpoint ships without calibrated attention scales (q_scale / prob_scale)
# and vLLM 0.17's FP8 attention path saturates without them, producing "!!!!!"
# degenerate output. BF16 is ~2x larger but B200's 183GB has headroom.
MODEL_PATH = "/lfs/skampere3/0/shared_hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"


def build_messages(row) -> list[dict]:
    sys_prompt = build_prompt_for_task(row.task)
    user_msg = (
        f"PAGE CONTEXT:\n"
        f"  task: {row.task}\n"
        f"  page_id: {row.page_id}\n"
        f"  subtask_short: {getattr(row, 'subtask_short', '') or ''}\n\n"
        f"RUBRIC TO CLASSIFY:\n"
        f"  name: {row.rubric_name}\n"
        f"  description: {row.rubric_description}\n"
        f"  guidance: {getattr(row, 'rubric_guidance', '') or ''}\n\n"
        + SCHEMA_HINT
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_msg},
    ]


def salvage_json(raw: str):
    """Best-effort JSON extraction from a possibly-prose-wrapped response."""
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    # Strip markdown fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # Find the first balanced { ... } block that parses
    start = s.find("{")
    if start < 0:
        return None
    for end in range(len(s), start, -1):
        if s[end-1] == "}":
            try:
                return json.loads(s[start:end])
            except Exception:
                continue
    return None


# Valid enum values (mirror JSON_SCHEMA_LLAMA)
VALID_TARGETS = {"work","production_process","submission_form","evaluation_judgment",
                 "selection_criterion","meta_artifact","actor_attribute","service_or_logistics"}
VALID_ACTORS  = {"producer","evaluator","gatekeeper","consumer","platform"}
VALID_ACTIONS = {"produce","constrain","judge","select","transact","distribute","describe"}
VALID_VERIFS  = {"computational","factual","consistency","procedural","statistical",
                 "causal","completeness","pragmatic","normative"}
VALID_TRACTS  = {"programmatic_check","llm_judge","expert_judgment","intractable"}
VALID_SPECIFS = {"vague","general","specific","hyper_specific"}
VALID_KEEPS   = {"keep","drop","borderline"}


def validate_record(parsed: dict) -> tuple[bool, str | None]:
    """Check that the parsed JSON has all enum values within the valid set + inputs non-empty."""
    if not isinstance(parsed, dict): return False, "not_a_dict"
    for k, valid in [
        ("target", VALID_TARGETS), ("actor", VALID_ACTORS),
        ("action", VALID_ACTIONS),
        ("verifiability_type", VALID_VERIFS),
        ("tractability", VALID_TRACTS),
        ("specificity", VALID_SPECIFS),
        ("keep", VALID_KEEPS),
    ]:
        v = parsed.get(k)
        if v not in valid:
            return False, f"invalid_{k}={v}"
    if not isinstance(parsed.get("requires_lookup"), bool):
        return False, "requires_lookup_not_bool"
    inputs = parsed.get("inputs")
    if not isinstance(inputs, list) or len(inputs) < 2:
        return False, "inputs_too_short"
    return True, None


def key_of(row) -> str:
    """Stable key for cache deduplication: (page_id, rubric_idx)."""
    return f"{row.page_id}::{row.rubric_idx}"


def load_cached_keys(jsonl_path: Path) -> set[str]:
    """Read existing JSONL cache; return set of keys with a SUCCESSFUL classification.
    Error records are NOT treated as cached — they'll be re-tried on resume."""
    if not jsonl_path.exists():
        return set()
    # Walk the file: latest record per key wins. Only the latest ok=True counts as cached.
    last_ok = {}
    with jsonl_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                key = f"{rec['page_id']}::{rec['rubric_idx']}"
                last_ok[key] = bool(rec.get('cls_ok'))
            except Exception:
                pass
    return {k for k, ok in last_ok.items() if ok}


# ---- Concurrent chunk claim / release ----
#
# Each chunk is gated by a blank `chunk_NNNNNN.processing` file. The file is
# created atomically via O_CREAT|O_EXCL; only one worker can win the race.
# Other workers see the file (or the final .jsonl) and move on. If a worker
# crashes mid-chunk the .processing file remains; another worker reclaims it
# once mtime exceeds --lock-stale-min.

def chunk_paths(chunks_dir: Path, ci: int):
    return (chunks_dir / f"chunk_{ci:06d}.jsonl",
            chunks_dir / f"chunk_{ci:06d}.jsonl.tmp",
            chunks_dir / f"chunk_{ci:06d}.processing")


def try_claim_chunk(ci: int, chunks_dir: Path, stale_seconds: int) -> str:
    """Returns:
       "done"    — chunk already finished (.jsonl exists)
       "busy"    — another worker holds a fresh .processing file
       "claimed" — we now own the .processing marker and should process
    """
    chunk_path, _, proc_path = chunk_paths(chunks_dir, ci)
    if chunk_path.exists():
        return "done"
    try:
        # Blank marker file — its mere existence indicates "in progress".
        fd = os.open(str(proc_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return "claimed"
    except FileExistsError:
        try:
            age = time.time() - proc_path.stat().st_mtime
        except FileNotFoundError:
            return "busy"  # raced with another worker that just cleared it
        if age <= stale_seconds:
            return "busy"
        # Stale — take over. Race-safe: at most one worker wins the O_EXCL.
        try:
            proc_path.unlink()
        except FileNotFoundError:
            pass
        try:
            fd = os.open(str(proc_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            print(f"[chunk {ci:06d}] took over stale .processing (age={age:.0f}s)")
            return "claimed"
        except FileExistsError:
            return "busy"


def release_chunk(ci: int, chunks_dir: Path):
    _, _, proc_path = chunk_paths(chunks_dir, ci)
    try:
        proc_path.unlink()
    except FileNotFoundError:
        pass


def build_record(row, parsed, raw: str, valid: bool, err: str | None) -> dict:
    rec = {
        "page_id":            row.page_id,
        "task":               row.task,
        "subtask_short":      getattr(row, 'subtask_short', '') or '',
        "rubric_idx":         int(row.rubric_idx),
        "rubric_name":        row.rubric_name,
        "rubric_description": row.rubric_description,
        "rubric_guidance":    getattr(row, 'rubric_guidance', '') or '',
    }
    if parsed is None:
        rec.update({
            "cls_target": None, "cls_actor": None, "cls_action": None,
            "cls_keep": None, "cls_inputs": [], "cls_reasoning": None,
            "cls_justification": None,
            "cls_ok": False, "cls_error": err or "json_salvage_failed",
            "_raw": raw[:500],
        })
        return rec
    if not valid:
        rec.update({
            "cls_target": parsed.get("target"),
            "cls_actor":  parsed.get("actor"),
            "cls_action": parsed.get("action"),
            "cls_keep":   parsed.get("keep"),
            "cls_inputs": parsed.get("inputs", []),
            "cls_reasoning":     parsed.get("reasoning",""),
            "cls_justification": parsed.get("justification",""),
            "cls_ok": False, "cls_error": f"validation_failed: {err}",
            "_raw": raw[:500],
        })
        return rec
    rec.update({
        "cls_target":          parsed["target"],
        "cls_actor":           parsed["actor"],
        "cls_action":          parsed["action"],
        "cls_inputs":          parsed["inputs"],
        "cls_verifiability":   parsed["verifiability_type"],
        "cls_tractability":    parsed["tractability"],
        "cls_requires_lookup": parsed["requires_lookup"],
        "cls_specificity":     parsed["specificity"],
        "cls_keep":            parsed["keep"],
        "cls_reasoning":       parsed.get("reasoning",""),
        "cls_justification":   parsed.get("justification",""),
        "cls_ok": True, "cls_error": None,
    })
    return rec


def process_chunk(ci: int, chunk_rows, llm, tokenizer, sampling_params, retry_sampling_params,
                  chunks_dir: Path, max_input_tokens: int) -> tuple[int, int, float]:
    """Run pass1 (+ pass2 retry on errors) over the chunk and atomically write its JSONL.
    Returns (n_ok, n_err_after_retry, dt_seconds).

    Pre-tokenizes each prompt and excludes any whose token count exceeds
    max_input_tokens — those get a 'prompt_too_long' error record without
    being sent to vLLM. Without this, a single oversized prompt would crash
    the entire engine (VLLMValidationError → engine death)."""
    chunk_path, tmp_path, _ = chunk_paths(chunks_dir, ci)
    t0 = time.perf_counter()

    prompts = [tokenizer.apply_chat_template(build_messages(r), tokenize=False, add_generation_prompt=True)
               for r in chunk_rows]
    # Pre-check prompt token lengths to filter out over-context rubrics.
    prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    too_long_idx = {i for i, L in enumerate(prompt_lens) if L > max_input_tokens}
    if too_long_idx:
        max_L = max(prompt_lens[i] for i in too_long_idx)
        print(f"[chunk {ci:06d}] skipping {len(too_long_idx)} rubrics with prompt_tokens > {max_input_tokens} (max={max_L})")

    keep_idx = [i for i in range(len(prompts)) if i not in too_long_idx]
    keep_prompts = [prompts[i] for i in keep_idx]
    outs_kept = llm.generate(keep_prompts, sampling_params, use_tqdm=True) if keep_prompts else []

    # Build records array indexed by original row position
    records: list[dict | None] = [None] * len(chunk_rows)
    err_indices: list[int] = []  # only indices of LLM-pass errors (not too_long)

    for i in too_long_idx:
        records[i] = build_record(
            chunk_rows[i], None, "", False,
            f"prompt_too_long: {prompt_lens[i]} > {max_input_tokens}",
        )

    for out_i, orig_i in enumerate(keep_idx):
        row = chunk_rows[orig_i]
        o = outs_kept[out_i]
        raw = o.outputs[0].text if o.outputs else ""
        parsed = salvage_json(raw)
        if parsed is None:
            records[orig_i] = build_record(row, None, raw, False, None)
            err_indices.append(orig_i)
        else:
            valid, why = validate_record(parsed)
            if not valid:
                records[orig_i] = build_record(row, parsed, raw, False, why)
                err_indices.append(orig_i)
            else:
                records[orig_i] = build_record(row, parsed, raw, True, None)

    # Pass 2 retry on errors only (excludes too_long — those can't be retried)
    if err_indices:
        retry_rows = [chunk_rows[i] for i in err_indices]
        retry_prompts = [prompts[i] for i in err_indices]
        retry_outs = llm.generate(retry_prompts, retry_sampling_params, use_tqdm=False)
        for idx, row, o in zip(err_indices, retry_rows, retry_outs):
            raw = o.outputs[0].text if o.outputs else ""
            parsed = salvage_json(raw)
            if parsed is None:
                continue  # keep the pass-1 error record
            valid, why = validate_record(parsed)
            if valid:
                records[idx] = build_record(row, parsed, raw, True, None)
            # else: keep pass-1 record

    # Atomic write: .tmp → fsync → replace
    with tmp_path.open('w') as fout:
        for rec in records:
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        os.fsync(fout.fileno())
    os.replace(tmp_path, chunk_path)

    n_ok = sum(1 for r in records if r.get('cls_ok'))
    n_err = len(records) - n_ok
    return n_ok, n_err, time.perf_counter() - t0


def merge_chunks_to_parquet(chunks_dir: Path, legacy_jsonl: Path | None, parquet_path: Path):
    """Read all chunk_*.jsonl files (and legacy single JSONL if present) into one parquet.
    Dedup by (page_id, rubric_idx), keep last (chunks dir wins over legacy on conflict)."""
    import pandas as pd
    rows = []
    # Legacy first (so chunks-dir entries overwrite on dedup keep='last')
    if legacy_jsonl and legacy_jsonl.exists():
        with legacy_jsonl.open() as f:
            for line in f:
                try: rows.append(json.loads(line))
                except Exception: pass
        print(f"  + legacy JSONL: {legacy_jsonl} ({sum(1 for _ in legacy_jsonl.open()):,} lines)")
    chunk_files = sorted(chunks_dir.glob("chunk_*.jsonl"))
    print(f"  + {len(chunk_files)} chunk JSONLs in {chunks_dir}")
    for cf in chunk_files:
        with cf.open() as f:
            for line in f:
                try: rows.append(json.loads(line))
                except Exception: pass
    if not rows:
        print("no rows to merge")
        return
    df = pd.DataFrame(rows)
    df_total = len(df)
    df = df.drop_duplicates(subset=['page_id','rubric_idx'], keep='last').reset_index(drop=True)
    df.to_parquet(parquet_path)
    print(f"merged {df_total:,} rows ({df_total - len(df):,} dupes deduped) -> {len(df):,} unique -> {parquet_path}")
    valid = df[df['cls_ok']]
    if len(valid):
        print(f"\n=== KEEP/DROP distribution (n_ok={len(valid):,}) ===")
        print(valid['cls_keep'].value_counts().to_string())
        print(f"\n=== KEEP rate by task ===")
        print(valid.groupby('task')['cls_keep'].value_counts().unstack(fill_value=0).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      default=str(ROOT / "notebooks/_explore_cache/rubrics.parquet"))
    ap.add_argument("--output",     default=str(ROOT / "outputs/classifier_llama_vllm_batch.parquet"))
    ap.add_argument("--chunks-dir", default=str(ROOT / "outputs/classifier_chunks"),
                    help="Directory for chunk_NNNNNN.jsonl + .lock files. Multiple workers share this directory.")
    ap.add_argument("--cache-jsonl", default=str(ROOT / "outputs/classifier_llama_vllm_batch.jsonl"),
                    help="LEGACY single-file JSONL (read-only). Successful rows here count as cached.")
    ap.add_argument("--lock-stale-min", type=int, default=30,
                    help="Lock files older than this many minutes can be re-claimed by another worker.")
    ap.add_argument("--task",   default=None, help="Process only rubrics from this task")
    ap.add_argument("--limit",  type=int, default=None, help="Process only the first N rubrics (after filtering + sort)")
    ap.add_argument("--chunk-size", type=int, default=1024,
                    help="Rubrics per chunk. Each chunk becomes one chunk_NNNNNN.jsonl. Smaller = finer-grained "
                         "resume + concurrency, but more per-chunk fixed overhead.")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens",    type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--jsonl-only", action="store_true", help="Skip the final JSONL->parquet merge")
    args = ap.parse_args()

    import pandas as pd
    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stale_seconds = args.lock_stale_min * 60

    df = pd.read_parquet(args.input)
    if args.task:
        df = df[df['task'] == args.task].copy()
        print(f"filtered to task={args.task}: {len(df):,} rubrics")
    # Sort by task — chunk boundaries are deterministic across workers.
    df = df.sort_values(['task','page_id','rubric_idx']).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).copy()
        print(f"limited to first {args.limit:,} rubrics")
    print(f"total rubrics in scope: {len(df):,}")

    # Legacy cache (single monolithic JSONL) — only read for resume completeness.
    legacy_path = Path(args.cache_jsonl)
    legacy_cached: set[str] = load_cached_keys(legacy_path) if legacy_path.exists() else set()
    if legacy_cached:
        print(f"legacy cache ({legacy_path.name}): {len(legacy_cached):,} successfully-classified keys")

    chunk_size = args.chunk_size
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    print(f"chunk plan: {n_chunks} chunks of {chunk_size} → chunks_dir={chunks_dir}")

    # Quick survey of pre-existing chunk files
    pre_existing = sum(1 for ci in range(n_chunks) if chunk_paths(chunks_dir, ci)[0].exists())
    if pre_existing:
        print(f"  pre-existing chunk JSONLs: {pre_existing}/{n_chunks}")

    # If everything is already done in chunks_dir, just merge and exit.
    if pre_existing == n_chunks:
        print("all chunks already exist — skipping model load, merging straight to parquet")
        if not args.jsonl_only:
            merge_chunks_to_parquet(chunks_dir, legacy_path, Path(args.output))
        return

    # Load vLLM (only if we actually have work to do)
    from vllm import LLM, SamplingParams
    print("loading vLLM model (~30-60s)...")
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype="auto",
        tensor_parallel_size=1,
        dtype="auto",
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    retry_sampling_params = SamplingParams(temperature=0.3, max_tokens=args.max_tokens)

    # Visit order: tasks in sort order (alphabetical), chunks within each task
    # are shuffled. Goal: finish whole tasks early (so downstream work can use
    # them) while still avoiding contention between concurrent workers within
    # the same task. Multiple workers will both attack the current task; their
    # per-task shuffles are independent, so collisions are rare and resolved
    # by the .processing markers anyway.
    from collections import OrderedDict
    by_task: "OrderedDict[str, list[int]]" = OrderedDict()
    for ci in range(n_chunks):
        # The chunk's task = the task of its first row (chunks may straddle a
        # task boundary at the ~10 transitions across the 11 tasks; the
        # straddler is grouped with whichever task it starts in).
        task = df.iloc[ci * chunk_size]['task']
        by_task.setdefault(task, []).append(ci)
    chunk_indices: list[int] = []
    for task, cis in by_task.items():
        random.shuffle(cis)
        chunk_indices.extend(cis)
    print("chunk visit order: tasks sequential, within-task shuffled")
    for task, cis in by_task.items():
        print(f"  {task:<28s} {len(cis):>3} chunks")

    n_done = n_skipped_done = n_skipped_busy = n_skipped_legacy = 0
    t_start = time.perf_counter()
    pbar = tqdm(chunk_indices, desc="chunks", unit="chunk", position=0,
                dynamic_ncols=True, mininterval=1.0)
    for ci in pbar:
        chunk_path, _, _ = chunk_paths(chunks_dir, ci)
        if chunk_path.exists():
            n_skipped_done += 1
            pbar.set_postfix(done=n_done, sk_done=n_skipped_done, sk_busy=n_skipped_busy, sk_legacy=n_skipped_legacy)
            continue

        # Materialize this chunk's rows from the deterministic slice
        chunk_rows = list(df.iloc[ci*chunk_size : (ci+1)*chunk_size].itertuples())

        # If every key in this chunk is already in legacy cache, skip without
        # writing a chunk file — merge will pull them from the legacy JSONL.
        if legacy_cached:
            keys = {f"{r.page_id}::{r.rubric_idx}" for r in chunk_rows}
            if keys.issubset(legacy_cached):
                n_skipped_legacy += 1
                pbar.set_postfix(done=n_done, sk_done=n_skipped_done, sk_busy=n_skipped_busy, sk_legacy=n_skipped_legacy)
                continue

        status = try_claim_chunk(ci, chunks_dir, stale_seconds)
        if status == "done":
            n_skipped_done += 1
            pbar.set_postfix(done=n_done, sk_done=n_skipped_done, sk_busy=n_skipped_busy, sk_legacy=n_skipped_legacy)
            continue
        if status == "busy":
            n_skipped_busy += 1
            pbar.set_postfix(done=n_done, sk_done=n_skipped_done, sk_busy=n_skipped_busy, sk_legacy=n_skipped_legacy)
            continue
        # status == "claimed"
        try:
            # Leave headroom for output tokens: input cap = max_model_len - max_tokens
            max_input_tokens = args.max_model_len - args.max_tokens
            n_ok, n_err, dt = process_chunk(
                ci, chunk_rows, llm, tokenizer,
                sampling_params, retry_sampling_params, chunks_dir,
                max_input_tokens=max_input_tokens,
            )
            n_done += 1
            elapsed = time.perf_counter() - t_start
            rate = (n_done * chunk_size) / elapsed if elapsed > 0 else 0
            remaining = n_chunks - (n_done + n_skipped_done + n_skipped_legacy)
            eta_min = (remaining * chunk_size) / rate / 60 if rate > 0 else 0
            # tqdm.write goes ABOVE the progress bar without breaking it
            tqdm.write(f"[chunk {ci:06d}] {len(chunk_rows):>4} rubrics, {dt:>5.1f}s, ok={n_ok} err={n_err}  "
                       f"rate={rate:>5.1f}/s  ETA={eta_min:>5.1f}min")
            pbar.set_postfix(done=n_done, sk_done=n_skipped_done, sk_busy=n_skipped_busy, sk_legacy=n_skipped_legacy)
        finally:
            release_chunk(ci, chunks_dir)
    pbar.close()

    print(f"\nWORKER DONE  processed={n_done}  skipped_done={n_skipped_done}  "
          f"skipped_busy={n_skipped_busy}  skipped_legacy={n_skipped_legacy}")

    if not args.jsonl_only:
        merge_chunks_to_parquet(chunks_dir, legacy_path, Path(args.output))


if __name__ == "__main__":
    main()
