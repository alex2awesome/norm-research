"""Extract publication year per source page.

For each unique (task, source_dir, source_file) in pages_df, ask gpt-5-mini
(flex tier) to return {year: int|null, year_confidence: 1-3, year_evidence: str}
based on filename + wave_tag + subtask metadata + first ~2K chars of source text.

Output: writes one JSONL record per page to outputs/analyses/source_years.jsonl
(append-mode, resumable). At end, joins to pages.parquet and persists with
year/year_confidence/year_evidence columns.

Usage:
  python scripts/extract_source_years.py                # full run, resume from JSONL
  python scripts/extract_source_years.py --limit 50     # smoke test
  python scripts/extract_source_years.py --task patents # one task only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
PAGES_PARQUET = ROOT / "notebooks" / "_explore_cache" / "pages.parquet"
TEXT_CACHE_DIR = ROOT / "notebooks" / "_explore_cache" / "parsed_text"
OUTPUT_DIR = ROOT / "outputs" / "analyses"
OUTPUT_JSONL = OUTPUT_DIR / "source_years.jsonl"
KEY_PATH = Path("/Users/spangher/.openai-salt-lab-key.txt")

MODEL = "gpt-5-mini"
TEXT_SNIPPET_CHARS = 2000      # what we send to the LLM for year extraction
CACHE_MAX_CHARS = 50_000        # how much we save to disk per file (≈12K tokens)
PDF_MAX_PAGES = 5               # cover copyright pages + early body content

YEAR_SCHEMA = {
    "name": "publication_year",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string"},
            "year": {"type": ["integer", "null"]},
            "year_confidence": {"type": "integer", "minimum": 1, "maximum": 3},
            "year_evidence": {"type": "string"},
        },
        "required": ["reasoning", "year", "year_confidence", "year_evidence"],
    },
}

SYSTEM_PROMPT = """You extract a single integer publication year for one source document.

You will see: filename, source folder, wave tag, a short subtask label, a longer subtask description, and the first ~2000 characters of the source text. Your job is to identify the year the source was published — NOT the year it was crawled, NOT a year mentioned in the body about an event.

Use these signals (in order of preference):
1. Explicit copyright / publication date in the source text ("Copyright 2017", "Published Mar 2019")
2. Date in the URL or filename (e.g. /2018/03/, _2020_, "wayback_19980412")
3. Edition / revision year stamps in headers / footers
4. Reference list dates (latest cited year + ~1 year is a fair proxy)
5. Wave tag hints (e.g., wayback waves often pre-2010; recent crawls 2020+)

Output JSON: reasoning (one sentence on what signal you used), year (integer or null), year_confidence (1=guess, 2=likely, 3=high confidence), year_evidence (short quoted snippet from the input that supports your answer; "" if year is null)."""


def _clean_text(text: str) -> str:
    """Collapse whitespace, drop control chars."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_html_full(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        raw = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return _clean_text(soup.get_text(" "))
    except Exception as e:
        return f"[read error: {e}]"


def _read_pdf_full(path: Path) -> str:
    """Three-tier extraction: fitz (fast C) → pdfminer (slower Python) → OCR.
    Mirrors pdf_to_text() in scripts/extract_rubric_features.py."""
    text = ""
    # 1. PyMuPDF (fitz) — 5-10x faster than pdfminer
    try:
        import fitz
        doc = fitz.open(str(path))
        text = "\n\n".join(doc[i].get_text() for i in range(min(PDF_MAX_PAGES, len(doc))))
        doc.close()
    except Exception:
        text = ""
    # 2. Fallback to pdfminer if fitz returned nothing or very little
    if len(text.strip()) < 100:
        try:
            from pdfminer.high_level import extract_text
            text2 = extract_text(str(path), maxpages=PDF_MAX_PAGES) or ""
            if len(text2.strip()) > len(text.strip()):
                text = text2
        except Exception:
            pass
    # 3. OCR fallback — only for pages that still have nothing (image-only PDFs)
    if len(text.strip()) < 100:
        try:
            import pdf2image
            import pytesseract
            images = pdf2image.convert_from_path(str(path), dpi=150, first_page=1, last_page=PDF_MAX_PAGES)
            ocr_text = "\n".join(pytesseract.image_to_string(img) for img in images)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        except Exception as e:
            if not text.strip():
                return f"[ocr error: {str(e)[:80]}]"
    if not text.strip():
        return "[empty pdf — no text and OCR failed]"
    return _clean_text(text)


def _read_md_full(path: Path) -> str:
    try:
        return _clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        return f"[read error: {e}]"


def cache_path_for(task: str, source_dir: str, source_file: str) -> Path:
    """Cache path mirrors the source layout for easy lookup.
    Filename keeps original extension + .txt suffix so the format is
    visible (e.g. abc.pdf.txt = parsed PDF text)."""
    return TEXT_CACHE_DIR / task / f"{source_dir}__{source_file}.txt"


def read_source_text(task: str, source_dir: str, source_file: str) -> str:
    """Return parsed text. Reads from cache if present; otherwise parses and
    writes the cache. Returned text is the FULL parse (capped at CACHE_MAX_CHARS).
    Callers should slice [:N] for whatever they need."""
    cache = cache_path_for(task, source_dir, source_file)
    if cache.exists():
        try:
            return cache.read_text(encoding="utf-8")
        except Exception:
            pass  # fall through to re-parse

    p = ROOT / "datasets" / task / "online-rubrics" / source_dir / source_file
    if not p.exists():
        return f"[file not found: {p.name}]"
    suffix = p.suffix.lower()
    if suffix in {".html", ".htm"}:
        text = _read_html_full(p)
    elif suffix == ".pdf":
        text = _read_pdf_full(p)
    elif suffix in {".md", ".txt", ""}:
        text = _read_md_full(p)
    else:
        text = f"[unhandled suffix {suffix}]"

    text = text[:CACHE_MAX_CHARS]
    # Persist to cache (best-effort; ignore write errors)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    except Exception:
        pass
    return text


def read_source_snippet(task: str, source_dir: str, source_file: str, n_chars: int = TEXT_SNIPPET_CHARS) -> str:
    """Backwards-compatible wrapper returning only the first n_chars."""
    return read_source_text(task, source_dir, source_file)[:n_chars]


def build_user_msg(row: dict, snippet: str) -> str:
    parts = [
        f"task: {row['task']}",
        f"source_dir: {row['source_dir']}",
        f"source_file: {row['source_file']}",
        f"wave_tag: {row.get('wave_tag', '')}",
        f"subtask_short: {row.get('subtask_short', '') or ''}",
        f"subtask_description: {(row.get('subtask_description', '') or '')[:300]}",
        "",
        "FIRST ~2000 CHARS OF SOURCE TEXT:",
        snippet,
    ]
    return "\n".join(parts)


async def call_llm(client: AsyncOpenAI, user: str, sem: asyncio.Semaphore,
                   service_tier: str = "flex", timeout_sec: float = 120.0) -> dict:
    async with sem:
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user},
                        ],
                        response_format={"type": "json_schema", "json_schema": YEAR_SCHEMA},
                        service_tier=service_tier,
                    ),
                    timeout=timeout_sec,
                )
                return json.loads(resp.choices[0].message.content or "{}")
            except asyncio.TimeoutError:
                if attempt == 2:
                    return {"_error": f"timeout after {timeout_sec}s"}
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 2:
                    return {"_error": str(e)[:200]}
                await asyncio.sleep(2 ** attempt)
    return {}


def load_done_ids(jsonl: Path) -> set[str]:
    if not jsonl.exists():
        return set()
    done = set()
    with jsonl.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["page_id"])
            except Exception:
                continue
    return done


def filter_jsonl_keep_non_null(jsonl: Path) -> int:
    """Drop entries with year=null from the JSONL so they get reprocessed.
    Returns count of entries dropped."""
    if not jsonl.exists():
        return 0
    keep = []
    dropped = 0
    with jsonl.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("year") is None:
                    dropped += 1
                else:
                    keep.append(line)
            except Exception:
                continue
    with jsonl.open("w") as f:
        f.writelines(keep)
    return dropped


async def process_one(client, sem, file_sem, row, jsonl_fp):
    # File reads (especially PDF parsing) are CPU-bound and would block the
    # event loop. Run them in a thread pool, gated by a smaller semaphore so
    # we don't oversubscribe the CPU.
    async with file_sem:
        snippet = await asyncio.to_thread(
            read_source_snippet, row["task"], row["source_dir"], row["source_file"]
        )
    user_msg = build_user_msg(row, snippet)
    res = await call_llm(client, user_msg, sem)
    out = {"page_id": row["page_id"], "task": row["task"], **res}
    jsonl_fp.write(json.dumps(out, ensure_ascii=False) + "\n")
    jsonl_fp.flush()
    return out


async def main_async(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PAGES_PARQUET)
    if args.task:
        df = df[df.task == args.task]
    if args.limit:
        df = df.head(args.limit)

    if args.retry_null:
        n_dropped = filter_jsonl_keep_non_null(OUTPUT_JSONL)
        print(f"--retry-null: dropped {n_dropped} null-year entries from JSONL; they'll be reprocessed")

    done = load_done_ids(OUTPUT_JSONL)
    df_todo = df[~df.page_id.isin(done)]
    print(f"loaded {len(df)} pages, {len(done)} already done, {len(df_todo)} to process")

    if df_todo.empty:
        print("nothing to do")
        return

    api_key = KEY_PATH.read_text().strip()
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)
    # File-read semaphore (CPU-bound; default to 16 concurrent threads which
    # roughly matches typical CPU core count without thrashing)
    file_sem = asyncio.Semaphore(args.file_concurrency)

    rows = df_todo.to_dict("records")
    t0 = time.perf_counter()

    with OUTPUT_JSONL.open("a") as jsonl_fp:
        coros = [process_one(client, sem, file_sem, r, jsonl_fp) for r in rows]
        for i, fut in enumerate(asyncio.as_completed(coros)):
            await fut
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (len(rows) - i - 1) / max(rate, 0.001)
                print(f"  done {i+1}/{len(rows)}  rate={rate:.1f}/s  eta={eta:.0f}s")

    print(f"\nfinished in {time.perf_counter() - t0:.0f}s")

    # Summary stats
    results = []
    with OUTPUT_JSONL.open() as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                continue
    n_total = len(results)
    n_year = sum(1 for r in results if r.get("year") is not None)
    n_err = sum(1 for r in results if "_error" in r)
    print(f"\nsummary: {n_total} total, {n_year} with year ({n_year/max(1,n_total)*100:.1f}%), {n_err} errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only first N pages")
    ap.add_argument("--task", default=None, help="filter to one task")
    ap.add_argument("--concurrency", type=int, default=200, help="OpenAI request concurrency")
    ap.add_argument("--file-concurrency", type=int, default=16, help="parallel file reads (CPU-bound)")
    ap.add_argument("--retry-null", action="store_true", help="reprocess JSONL entries where year is null")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
