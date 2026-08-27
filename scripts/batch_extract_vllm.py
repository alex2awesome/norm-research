"""
Offline batch extractor for rubrics — runs Llama-3.3-70B-Instruct (BF16) via
vLLM on a single GPU over pages from `datasets/<task>/online-rubrics/raw/`.

Mirrors the architecture of batch_classify_vllm.py:
  - per-task prompts via extract_rubric_features_v5_prompt.build_prompt_for_task
  - chunked processing with append-only JSONL caching after each chunk
  - resume by (page_id) key
  - 2-pass retry on errors at higher temperature

Differences from classifier:
  - Inputs are PAGES (read from disk via load_clean_text) not rubric rows
  - Per-page output is a JSON object with N rubrics_metrics (not 1 classification)
  - Pages can be long; we chunk pages > MAX_PAGE_CHARS into overlapping windows
    and let the model extract from each chunk separately (matching the
    chunked_extract.py logic)

Usage:
  # Trial: 20 random pages, multiple tasks
  CUDA_VISIBLE_DEVICES=5 python3 batch_extract_vllm.py --limit 20

  # Full run (all pages across all 11 tasks, resumable)
  CUDA_VISIBLE_DEVICES=5 python3 batch_extract_vllm.py

  # Restrict to one task
  CUDA_VISIBLE_DEVICES=5 python3 batch_extract_vllm.py --task creative-writing
"""

from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(ROOT / "scripts"))

# Env BEFORE torch/vllm
os.environ.setdefault("HOME", "/lfs/skampere3/0/alexspan")
os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/alexspan/.cache/huggingface")
os.environ.setdefault("TRITON_CACHE_DIR", "/lfs/skampere3/0/alexspan/.cache/triton")
os.environ.setdefault("VLLM_CACHE_ROOT", "/lfs/skampere3/0/alexspan/.cache/vllm")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/lfs/skampere3/0/alexspan/.cache/torchinductor")
os.environ.setdefault("TMPDIR", "/lfs/skampere3/0/alexspan/tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

from extract_rubric_features_v5verbose_prompt import build_prompt_for_task
from extract_rubric_features import load_clean_text, TASKS
# Override DATASETS to be relative to this script's location (the imported one
# is hardcoded to the local-Mac path).
DATASETS = ROOT / "datasets"

# Use BF16 Meta Llama for stable attention (NVIDIA FP8 has uncalibrated scales).
MODEL_PATH = "/lfs/skampere3/0/shared_hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"

# Max page chars to send in a single call. Pages longer than this get chunked
# with overlap (matches scripts/chunked_extract.py).
MAX_PAGE_CHARS = 40_000
CHUNK_OVERLAP = 5_000

# Brief schema reminder appended to the user prompt; v5 SYSTEM_PROMPT covers
# the heavy lifting, but Llama benefits from a final shape reminder.
SCHEMA_HINT = """
You MUST respond with a single JSON object exactly matching this shape:

{
  "reasoning": "1-3 sentences walking through the work-test before extracting.",
  "orientation": one of ["research_article","academic_page","how_to","formal_guideline","blog_post","dataset","tutorial","textbook_excerpt","professional_standard","contest_criteria","stylebook","course_syllabus","wiki","forum_post","news_article","error","other"],
  "intended_audience": "string",
  "subtask_short": "≤8 word label",
  "subtask_description": "1-2 sentences",
  "subtask_keywords": ["3", "to", "7", "lowercase_snake_case"],
  "subtask_breadth": one of ["very_narrow","narrow","moderate","broad","very_broad"],
  "error": null  OR  "short reason string",
  "rubrics_metrics": [
    {"name": "...", "description": "...", "guidance": "...", "inputs": ["≥2 specific noun phrases"]},
    ...
  ]
}

Return ONLY the JSON object. No prose outside it. No markdown fences.
"""


def chunk_text(text: str, chunk_size: int = MAX_PAGE_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries when possible."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ("\n\n", ". ", "\n"):
                idx = text.rfind(sep, start + chunk_size - 2_000, end)
                if idx > start + chunk_size // 2:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_messages_for_page(task: str, page_path: Path, text: str, chunk_idx: int = 0, n_chunks: int = 1) -> list[dict]:
    """Build (system, user) messages for a page (or one chunk of a long page)."""
    sys_prompt = build_prompt_for_task(task)
    user_msg = (
        f"PARENT TASK CONTEXT: This page was collected for the broader task: {task}\n\n"
        f"SOURCE FILE: {page_path.name}\n"
    )
    if n_chunks > 1:
        user_msg += f"CHUNK: {chunk_idx + 1}/{n_chunks} (overlapping windows; some rubrics may appear in adjacent chunks)\n"
    user_msg += f"\nPAGE TEXT:\n{text}\n\n" + SCHEMA_HINT
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_msg},
    ]


def salvage_json(raw: str):
    if not raw or not raw.strip(): return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try: return json.loads(s)
    except Exception: pass
    start = s.find("{")
    if start < 0: return None
    for end in range(len(s), start, -1):
        if s[end-1] == "}":
            try: return json.loads(s[start:end])
            except Exception: continue
    return None


VALID_ORIENTATIONS = {"research_article","academic_page","how_to","formal_guideline","blog_post",
                     "dataset","tutorial","textbook_excerpt","professional_standard","contest_criteria",
                     "stylebook","course_syllabus","wiki","forum_post","news_article","error","other"}
VALID_BREADTHS = {"very_narrow","narrow","moderate","broad","very_broad"}


def validate_record(parsed: dict) -> tuple[bool, str | None]:
    if not isinstance(parsed, dict): return False, "not_a_dict"
    orientation = parsed.get("orientation")
    if orientation not in VALID_ORIENTATIONS:
        return False, f"invalid_orientation={orientation}"
    # Stub/error pages legitimately have no characterizable subtask — allow
    # null subtask_breadth when orientation="error" and rubrics_metrics is empty.
    if orientation == "error":
        rubrics = parsed.get("rubrics_metrics", [])
        if rubrics:
            return False, "error_orientation_with_non_empty_rubrics"
        return True, None
    # Non-error pages must have a valid breadth + properly-shaped rubrics
    if parsed.get("subtask_breadth") not in VALID_BREADTHS:
        return False, f"invalid_subtask_breadth={parsed.get('subtask_breadth')}"
    rubrics = parsed.get("rubrics_metrics")
    if not isinstance(rubrics, list):
        return False, "rubrics_metrics_not_list"
    for i, r in enumerate(rubrics):
        if not isinstance(r, dict):
            return False, f"rubric_{i}_not_dict"
        for k in ("name", "description", "guidance", "inputs"):
            if k not in r:
                return False, f"rubric_{i}_missing_{k}"
        if not isinstance(r["inputs"], list) or len(r["inputs"]) < 2:
            return False, f"rubric_{i}_inputs_too_short"
    return True, None


def discover_pages(task: str, limit: int | None = None) -> list[Path]:
    """List pages from datasets/<task>/online-rubrics/raw/."""
    raw_dir = DATASETS / task / "online-rubrics" / "raw"
    if not raw_dir.exists():
        return []
    pages = []
    for p in sorted(raw_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in (".html", ".pdf", ".md", ".txt"):
            pages.append(p)
        if limit and len(pages) >= limit:
            break
    return pages


def page_key(task: str, page_path: Path, chunk_idx: int, n_chunks: int) -> str:
    return f"{task}::{page_path.name}::{chunk_idx}/{n_chunks}"


def load_cached_keys(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists(): return set()
    last_ok = {}
    with jsonl_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                key = f"{rec['task']}::{rec['source_file']}::{rec.get('chunk_idx',0)}/{rec.get('n_chunks',1)}"
                last_ok[key] = bool(rec.get('ex_ok'))
            except Exception:
                pass
    return {k for k, ok in last_ok.items() if ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="One task name; default = all 11 tasks")
    ap.add_argument("--limit", type=int, default=None, help="First N pages per task (testing)")
    ap.add_argument("--output", default=str(ROOT / "outputs/extractor_llama_vllm.parquet"))
    ap.add_argument("--cache-jsonl", default=str(ROOT / "outputs/extractor_llama_vllm.jsonl"))
    ap.add_argument("--chunk-size", type=int, default=512, help="Pages per llm.generate() call (smaller than classifier because per-page output is heavier)")
    ap.add_argument("--max-model-len", type=int, default=24576, help="Larger than classifier — page text + verbose extraction output")
    ap.add_argument("--max-tokens", type=int, default=4096, help="Output budget per page-chunk. Substantive pages emit 10 rubrics × ~150 tokens + reasoning ≈ 2000-2500 tokens; 4096 leaves comfortable headroom.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    Path(args.cache_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Build the list of (task, page_path, chunk_idx, n_chunks, text) tuples
    tasks_to_run = [args.task] if args.task else list(TASKS)
    print(f"discovering pages across {len(tasks_to_run)} task(s)...")
    work_units = []
    for task in tasks_to_run:
        pages = discover_pages(task, limit=args.limit)
        print(f"  {task}: {len(pages):,} pages")
        for p in pages:
            try:
                text, _kind = load_clean_text(p)
            except Exception as e:
                print(f"    skip (load error): {p.name}: {e}")
                continue
            if not text.strip():
                continue
            chunks = chunk_text(text)
            for i, c in enumerate(chunks):
                work_units.append({"task": task, "page_path": p, "chunk_idx": i,
                                   "n_chunks": len(chunks), "text": c})
    print(f"total work units (page-chunks): {len(work_units):,}")

    # Resume from cache
    cache_path = Path(args.cache_jsonl)
    cached_keys = load_cached_keys(cache_path)
    print(f"cached (already extracted): {len(cached_keys):,}")
    pending = [w for w in work_units
               if page_key(w['task'], w['page_path'], w['chunk_idx'], w['n_chunks']) not in cached_keys]
    print(f"pending: {len(pending):,}")
    if not pending:
        print("nothing to do")
        if cache_path.exists():
            merge_jsonl_to_parquet(cache_path, Path(args.output))
        return

    # Load vLLM
    from vllm import LLM, SamplingParams
    print("loading vLLM model...")
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype="auto",  # BF16 — same reason as classifier
        tensor_parallel_size=1,
        dtype="auto",
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    retry_sampling_params = SamplingParams(temperature=0.3, max_tokens=args.max_tokens)

    def run_one_pass(units, params, label):
        if not units:
            return 0, 0, []
        n_chunks_total = (len(units) + args.chunk_size - 1) // args.chunk_size
        print(f"\n[{label}] processing {len(units):,} page-chunks in {n_chunks_total} chunks of {args.chunk_size}")
        t_start = time.perf_counter()
        ok = err = 0
        err_units = []
        for ci in range(n_chunks_total):
            batch = units[ci*args.chunk_size : (ci+1)*args.chunk_size]
            prompts = []
            for u in batch:
                msgs = build_messages_for_page(u['task'], u['page_path'], u['text'], u['chunk_idx'], u['n_chunks'])
                prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            dt = time.perf_counter() - t0

            chunk_ok = chunk_err = 0
            with cache_path.open('a') as fout:
                for u, out in zip(batch, outputs):
                    raw = out.outputs[0].text if out.outputs else ""
                    rec = {
                        "task":         u['task'],
                        "page_path":    str(u['page_path']),
                        "source_file":  u['page_path'].name,
                        "chunk_idx":    u['chunk_idx'],
                        "n_chunks":     u['n_chunks'],
                    }
                    parsed = salvage_json(raw)
                    if parsed is None:
                        rec.update({"extracted": None, "ex_ok": False,
                                    "ex_error": "json_salvage_failed", "_raw": raw[:500]})
                        chunk_err += 1
                        err_units.append(u)
                    else:
                        valid, why = validate_record(parsed)
                        if not valid:
                            rec.update({"extracted": parsed, "ex_ok": False,
                                        "ex_error": f"validation_failed: {why}", "_raw": raw[:500]})
                            chunk_err += 1
                            err_units.append(u)
                        else:
                            rec.update({"extracted": parsed, "ex_ok": True, "ex_error": None})
                            chunk_ok += 1
                    fout.write(json.dumps(rec) + "\n")
                fout.flush(); os.fsync(fout.fileno())
            ok += chunk_ok; err += chunk_err
            done = min((ci+1) * args.chunk_size, len(units))
            rate = done / (time.perf_counter() - t_start)
            eta_min = (len(units) - done) / rate / 60 if rate > 0 else 0
            print(f"  [{label}] chunk {ci+1:>3}/{n_chunks_total}  ({len(batch):>4} units, {dt:>5.1f}s, ok={chunk_ok} err={chunk_err})  "
                  f"{done:>5,}/{len(units)} ({100*done/len(units):>5.1f}%)  rate={rate:>4.1f}/s  ETA={eta_min:>4.1f}min")
        return ok, err, err_units

    # Pass 1
    p1_ok, p1_err, err_units = run_one_pass(pending, sampling_params, "pass1")

    # Pass 2: retry errors
    p2_ok = p2_err = 0
    if err_units:
        print(f"\n[pass2] retrying {len(err_units)} error units at T=0.3")
        p2_ok, p2_err, _ = run_one_pass(err_units, retry_sampling_params, "pass2")

    print(f"\nDONE  pass1 ok={p1_ok:,} err={p1_err:,}  →  pass2 recovered {p1_err - p2_err:,}  final fail={p2_err:,}")

    merge_jsonl_to_parquet(cache_path, Path(args.output))


def merge_jsonl_to_parquet(jsonl_path: Path, parquet_path: Path):
    import pandas as pd
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            try: rows.append(json.loads(line))
            except: pass
    if not rows: return
    df = pd.DataFrame(rows)
    n_total = len(df)
    df = df.drop_duplicates(subset=['task','source_file','chunk_idx','n_chunks'], keep='last').reset_index(drop=True)
    df.to_parquet(parquet_path)
    print(f"merged {n_total:,} rows ({n_total-len(df):,} dedupes) -> {len(df):,} unique -> {parquet_path}")
    valid = df[df['ex_ok']] if 'ex_ok' in df.columns else df
    if 'extracted' in valid.columns and len(valid):
        n_rubrics = valid['extracted'].apply(lambda x: len((x or {}).get('rubrics_metrics', [])) if x else 0)
        print(f"\nrubrics extracted: total={n_rubrics.sum():,}  mean/page-chunk={n_rubrics.mean():.1f}  median={n_rubrics.median():.0f}")


if __name__ == "__main__":
    main()
