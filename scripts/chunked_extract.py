"""
Chunked rubric extractor for full-book PDFs that would otherwise be truncated.

Splits each large file into ~40K-char chunks with 5K overlap. Runs gpt-5-mini
extraction on each chunk separately. Saves outputs to:
  datasets/<task>/online-rubrics/gpt-parsed/<model>/<source_dir>__<basename>__chunk_<NN>_of_<MM>.json

Resume support: skips chunks that already have outputs.

Usage:
  python chunked_extract.py --task creative-writing --pattern 'manual_*.pdf'
  python chunked_extract.py --task creative-writing --pattern 'manual_*'  --concurrency 8
  python chunked_extract.py --all  --pattern 'manual_*'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Reuse helpers + schema from the main extractor
sys.path.insert(0, str(Path(__file__).parent))
from extract_rubric_features import (
    SYSTEM_PROMPT, JSON_SCHEMA, PRICING, TASKS, DATASETS,
    load_clean_text, _make_client,
)


CHUNK_CHARS = 40_000      # ~10K tokens per chunk
CHUNK_OVERLAP = 5_000     # carry over context across chunk boundaries
MIN_BOOK_CHARS = 80_000   # below this, no chunking — just one extraction


def chunk_text(text: str, chunk_size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks. If short, return [text]."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break on paragraph boundary
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


async def extract_chunk(client, model: str, sem, task: str, source_file: Path, chunk_idx: int, n_chunks: int, chunk_text_: str):
    user_msg = (
        f"PARENT TASK CONTEXT: This page was collected for the broader task: {task}\n\n"
        f"SOURCE FILE: {source_file.name}\n"
        f"CHUNK: {chunk_idx + 1}/{n_chunks} (overlapping windows; some rubrics may appear in adjacent chunks)\n\n"
        f"PAGE TEXT:\n{chunk_text_}"
    )
    t0 = time.perf_counter()
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
            )
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_s": time.perf_counter() - t0,
                    "input_tokens": 0, "output_tokens": 0}
    elapsed = time.perf_counter() - t0
    try:
        extracted = json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"ok": False, "error": f"json_parse: {e}", "elapsed_s": elapsed,
                "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "output_tokens": resp.usage.completion_tokens if resp.usage else 0}
    return {"ok": True, "extracted": extracted, "elapsed_s": elapsed,
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0}


async def process_one_file(client, sem, task: str, fp: Path, model: str, out_dir: Path, skip_head_tail: bool = False):
    text, kind = load_clean_text(fp)
    if not text.strip():
        print(f"  [SKIP empty] {fp.name}")
        return {"chunks": 0, "cost": 0.0, "in": 0, "out": 0, "n_ok": 0, "n_err": 0}
    chunks = chunk_text(text)
    n = len(chunks)
    src_dir = fp.parent.name if fp.parent.name in ("raw", "claude-parsed") else fp.parent.parent.name
    in_p, out_p = PRICING.get(model, (1.0, 4.0))

    # Skip-head-tail mode: for files originally extracted with head+tail truncation,
    # the first chunk overlaps head and the last chunk overlaps tail. Skip those.
    skipped_first_last = 0
    eligible = list(range(n))
    if skip_head_tail and n >= 3:
        eligible = list(range(1, n - 1))
        skipped_first_last = 2
    elif skip_head_tail and n <= 2:
        # File is too short — head+tail already covered everything meaningful
        print(f"  [SKIP short] {fp.name}  chars={len(text):,}  chunks={n}  (head+tail already cover this)")
        return {"chunks": 0, "cost": 0.0, "in": 0, "out": 0, "n_ok": 0, "n_err": 0}

    print(f"  [{fp.name}] chars={len(text):,} chunks={n} skip_first_last={skipped_first_last}")

    # Pre-check which chunks already have outputs
    todo = []
    for i in eligible:
        out_path = out_dir / f"{src_dir}__{fp.name}__chunk_{i:02d}_of_{n:02d}.json"
        if not out_path.exists():
            todo.append((i, out_path))
    if not todo:
        print(f"    all {len(eligible)} eligible chunks already processed")
        return {"chunks": len(eligible), "cost": 0.0, "in": 0, "out": 0, "n_ok": len(eligible), "n_err": 0}
    print(f"    processing {len(todo)} new chunks (of {len(eligible)} eligible)")

    tot_in = tot_out = 0
    n_ok = n_err = 0

    async def run_chunk(i: int, out_path: Path):
        nonlocal tot_in, tot_out, n_ok, n_err
        result = await extract_chunk(client, model, sem, task, fp, i, n, chunks[i])
        record = {
            "path": str(fp),
            "task": task,
            "source_dir": src_dir,
            "source_file": fp.name,
            "chunk_idx": i,
            "n_chunks": n,
            "chunk_chars": len(chunks[i]),
            "model": model,
            "elapsed_s": result.get("elapsed_s", 0.0),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
        }
        if result["ok"]:
            record["extracted"] = result["extracted"]
            n_ok += 1
        else:
            record["error"] = result["error"]
            n_err += 1
        tot_in += record["input_tokens"]
        tot_out += record["output_tokens"]
        out_path.write_text(json.dumps(record, indent=2))

    await asyncio.gather(*(run_chunk(i, op) for i, op in todo))
    cost = tot_in * in_p / 1e6 + tot_out * out_p / 1e6
    print(f"    DONE  ok={n_ok} err={n_err}  in={tot_in:,} out={tot_out:,} cost=${cost:.3f}")
    return {"chunks": n, "cost": cost, "in": tot_in, "out": tot_out, "n_ok": n_ok, "n_err": n_err}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--all", action="store_true")
    p.add_argument("--pattern", default="manual_*", help="filename glob pattern (default manual_*)")
    p.add_argument("--recurse", action="store_true", help="recurse into raw/long_books/ etc.")
    p.add_argument("--filelist", help="path to a text file containing one file path per line")
    p.add_argument("--skip-head-tail", action="store_true",
                   help="for retrofitting truncated extractions: skip the first and last chunk "
                        "(those overlap content already covered by the original head+tail truncated extraction)")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()

    client = _make_client()
    sem = asyncio.Semaphore(args.concurrency)

    # Build the (task, file) list
    targets: list[tuple[str, Path]] = []
    if args.filelist:
        for line in Path(args.filelist).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fp = Path(line)
            if not fp.is_file():
                print(f"  MISSING: {fp}", file=sys.stderr)
                continue
            # infer task from path
            try:
                parts = fp.parts
                t_idx = parts.index("datasets") + 1
                task = parts[t_idx]
            except (ValueError, IndexError):
                print(f"  CANT INFER TASK FROM PATH: {fp}", file=sys.stderr)
                continue
            if task not in TASKS:
                print(f"  UNKNOWN TASK ({task}): {fp}", file=sys.stderr)
                continue
            targets.append((task, fp))
    else:
        tasks = TASKS if args.all else ([args.task] if args.task else [])
        if not tasks:
            print("--task, --all, or --filelist required", file=sys.stderr); sys.exit(2)
        for task in tasks:
            raw_dir = DATASETS / task / "online-rubrics" / "raw"
            if args.recurse:
                files = sorted(raw_dir.rglob(args.pattern))
            else:
                files = sorted(raw_dir.glob(args.pattern))
            for fp in files:
                targets.append((task, fp))

    if not targets:
        print("no targets found", file=sys.stderr); sys.exit(1)
    print(f"=== {len(targets)} target files ===")

    grand = {"chunks": 0, "cost": 0.0, "in": 0, "out": 0, "n_ok": 0, "n_err": 0}
    # Group by task for clearer output
    from collections import defaultdict
    by_task: dict[str, list[Path]] = defaultdict(list)
    for t, fp in targets:
        by_task[t].append(fp)
    for task, files in by_task.items():
        out_dir = DATASETS / task / "online-rubrics" / "gpt-parsed" / args.model
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {task}: {len(files)} files ===")
        for fp in files:
            r = await process_one_file(client, sem, task, fp, args.model, out_dir,
                                       skip_head_tail=args.skip_head_tail)
            for k in grand:
                grand[k] += r[k]
    print(f"\n=== GRAND TOTAL chunks={grand['chunks']} ok={grand['n_ok']} err={grand['n_err']} "
          f"in={grand['in']:,} out={grand['out']:,} cost=${grand['cost']:.2f} ===")


if __name__ == "__main__":
    asyncio.run(main())
