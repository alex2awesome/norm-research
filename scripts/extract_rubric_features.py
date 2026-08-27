"""
Extract structured features from raw rubric HTML/PDF pages using OpenAI Async API.

For each page, ask the model:
  - rubrics_metrics: list of explicit rubrics/criteria/metrics with surrounding guidance text
  - orientation:    type of page (research_article | academic_page | how_to | formal_guideline | blog_post | dataset | error | other)
  - intended_audience: who the page is written for
  - subtask:        the specific KIND of subtask within the parent task that the page describes
  - error:          set if page is unreadable/captcha/error/empty; orientation=='error' implies this is set

Pilot: --pilot mode runs N files per task on chosen model, dumps outputs to stdout for manual inspection.
Full:  --task <name> --model <name> processes all raw/*.{html,htm,pdf,md} into extracted-<model>/<basename>.json

Usage:
  python extract_rubric_features.py --pilot --task patents --n 5 --model gpt-5-mini
  python extract_rubric_features.py --pilot --task patents --n 5 --model gpt-5
  python extract_rubric_features.py --task patents --model gpt-5-mini --concurrency 32
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

import tiktoken
from bs4 import BeautifulSoup
from openai import AsyncOpenAI


def _load_api_key() -> str:
    if k := os.environ.get("OPENAI_API_KEY"):
        return k
    for cand in [
        Path.home() / ".openai-salt-lab-key.txt",
        Path.home() / ".openai-my-key.txt",
        Path.home() / ".openai-api-key.txt",
    ]:
        if cand.exists():
            return cand.read_text().strip()
    raise RuntimeError("No OpenAI key found (env OPENAI_API_KEY or ~/.openai-*-key.txt)")


def _make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=_load_api_key())

# pdfminer is optional; only required if processing PDFs
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    HAVE_PDF = True
except Exception:
    HAVE_PDF = False


DATASETS = Path("/Users/spangher/Projects/stanford-research/norm-research/datasets")
TASKS = [
    "creative-writing", "peer-review", "math-stackexchange", "news-homepages",
    "press-releases", "code-review", "grant-funding", "humor",
    "legal-outcome-prediction", "notice-and-comment", "patents",
]

# Soft cap on text we send to the model (after cleaning). Pages get truncated past this.
MAX_INPUT_CHARS = 60_000   # ~15K tokens of cleaned text


SYSTEM_PROMPT = """You analyze web pages that describe rubrics, criteria, standards, or guidelines for evaluating a particular kind of work (e.g., short stories, math proofs, peer reviews, press releases, patents, comments on regulations, etc.).

For each page, extract:

1. **rubrics_metrics**: every distinct rubric / criterion / metric / dimension / desideratum / standard / requirement the page mentions for evaluating or producing the target work. Include all surrounding explanation, guidance, examples, anti-patterns, scoring notes, and verbosity. Each entry should be a single coherent criterion. If the page enumerates 10 dimensions, return 10 entries. If the page lists 30 substantive legal/technical standards (e.g., "the invention must be novel", "the comment must be timely"), return 30. Prefer extracting the ORIGINAL WORDING.

2. **orientation**: the genre of the page. Pick the best single label.

3. **intended_audience**: who the page is written for (e.g., "graduate students applying for NIH F31 fellowships", "stand-up comedians at open mics", "first-time patent applicants", "journal reviewers in machine learning").

4. **subtask** (FOUR fields):
   - **subtask_short**: ≤8-word canonical label (e.g. "writing horror short fiction", "NIH R01 review", "appellate brief for ED Tex patent case", "stand-up comedy bombing recovery")
   - **subtask_description**: 1-2 sentence richer description.
   - **subtask_keywords**: 3-7 lowercase snake_case keywords identifying genre/format/audience/jurisdiction/medium facets (e.g. ["horror","short_fiction","online_magazine","adult"] or ["NIH","R01","biomedical","simplified_review_framework_2025"]).
   - **subtask_breadth**: pick from {very_narrow, narrow, moderate, broad, very_broad}. very_narrow = a single contest/program's rules; narrow = one genre+format combo; moderate = one genre OR one format alone; broad = whole field generally; very_broad = cross-field. Be honest about breadth — many pages cover very broad ground; that's fine but flag it.
   These four fields will be used downstream to control for within-task subtask variance, so be precise and consistent across pages.

5. **error**: set this to a short reason string AND set orientation="error" ONLY IF the page is genuinely unusable: captcha/anti-bot interstitial, login wall, 404/empty content, only nav chrome, abstract-only page with no body, non-English without any rubric content, etc. Otherwise leave error null.

# Critical guidance — what counts as a rubric

A "rubric" is broader than a numbered checklist. INCLUDE:

- Substantive legal/regulatory standards ("the application must satisfy 35 U.S.C. §112 written description"; "comments must contain new factual information")
- Field-specific quality criteria embedded in narrative prose ("the prose must avoid passive voice"; "a strong opening sentence introduces conflict")
- Scoring/evaluation rubrics from official forms ("Approach: how rigorous is the methodology?")
- Style guide rules ("never split infinitives"; "use the Oxford comma")
- Editorial standards from publishers ("submissions must be under 5000 words")
- Procedural requirements that imply quality criteria ("the brief must include a Statement of Facts")
- Q&A entries that articulate evaluative principles (a "what makes a good X?" answer; an interview where a practitioner explains their selection criteria)
- Implicit criteria revealed in critique examples ("here's why this comment was ignored — it didn't propose alternatives")

DO NOT mark as "error" just because the page is dense, legalistic, or doesn't say the word "rubric." If the page contains substantive principles, criteria, or standards relevant to evaluating or producing the work, EXTRACT THEM.

DO mark as "error" if the page is:
- A Cloudflare/Just-a-moment interstitial with no actual content
- A 404 / login wall / paywall stub
- An abstract or paper landing page with NO substantive criteria visible (just title + authors + abstract; the rubric is in the unfetchable PDF)
- Pure navigation/chrome with no rubric-bearing prose

# Few-shot examples

EXAMPLE A — dense regulatory Q&A WITH substantive criteria → EXTRACT
Input excerpt: "Patent: An exclusive right granted by the IPOPHL for a product, process, or improvement thereof for a specified period in exchange for full disclosure. To be patentable a technical solution must be (1) new, (2) involve an inventive step, and (3) be industrially applicable. Utility models protect inventions which are new and industrially applicable; they do not need an inventive step..."
Correct output: orientation="professional_standard", error=null, rubrics_metrics includes:
  - {"name": "Patentability requirements", "description": "Must be (1) new, (2) involve an inventive step, (3) industrially applicable", "guidance": "..."}
  - {"name": "Utility model eligibility", "description": "Inventions that are new and industrially applicable; no inventive step required", "guidance": "7-year non-renewable term"}
  - ...
WRONG output: orientation="error" with reason "no rubric here" — this is wrong; the substantive standards ARE the rubric.

EXAMPLE B — abstract-only landing page → MARK AS ERROR
Input excerpt: "Abstract: We test the consistency of post-grant patent quality measures... [paper info, citations, but no actual criteria visible]"
Correct output: orientation="error", error="abstract/landing page only; criteria are in the full paper which was not fetched", rubrics_metrics=[]

EXAMPLE C — Cloudflare interstitial → MARK AS ERROR
Input excerpt: "Just a moment... Verifying you are human."
Correct output: orientation="error", error="Cloudflare anti-bot interstitial", rubrics_metrics=[]

EXAMPLE D — explicit scoring rubric → EXTRACT EACH DIMENSION
Input excerpt: "NIH R01 Review Criteria: 1. Significance — Does the project address an important problem? ... 2. Investigators — Are the PIs well qualified? ... 3. Innovation — Does the project use novel concepts? ... 4. Approach — Are the methods rigorous? ... 5. Environment — Will the institutional setting contribute to success?"
Correct output: orientation="formal_guideline", rubrics_metrics has 5 entries, one per criterion, each with description and any surrounding guidance.

Be EXHAUSTIVE on substantive content. Be CONSERVATIVE only when the page is genuinely empty / blocked / metadata-only.
"""


JSON_SCHEMA = {
    "name": "rubric_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "orientation": {
                "type": "string",
                "enum": [
                    "research_article", "academic_page", "how_to", "formal_guideline",
                    "blog_post", "dataset", "tutorial", "textbook_excerpt",
                    "professional_standard", "contest_criteria", "stylebook",
                    "course_syllabus", "wiki", "forum_post", "news_article",
                    "error", "other",
                ],
            },
            "intended_audience": {"type": "string"},
            "subtask_short": {"type": "string", "description": "≤8-word canonical label for the subtask, e.g. 'writing horror short fiction', 'NIH R01 review', 'patent claim drafting for software inventions'. Use the most specific genre/format/audience axis that's salient on the page."},
            "subtask_description": {"type": "string", "description": "1-2 sentence description of the specific kind of work the rubric is for. Should be richer than subtask_short."},
            "subtask_keywords": {"type": "array", "items": {"type": "string"}, "description": "3-7 keyword facets that identify the subtask (e.g. ['horror','short_fiction','online_magazine','adult_audience'] or ['NIH','R01','biomedical','simplified_review_framework_2025']). Use lowercase snake_case."},
            "subtask_breadth": {
                "type": "string",
                "enum": ["very_narrow", "narrow", "moderate", "broad", "very_broad"],
                "description": "How narrow vs broad is the subtask scope? very_narrow = a single contest's rules; narrow = one genre+format combo; moderate = one genre OR one format; broad = whole field (e.g. 'creative writing' generally); very_broad = cross-field (e.g. 'all academic writing')."
            },
            "error": {"type": ["string", "null"]},
            "rubrics_metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "description": "Short label for the criterion."},
                        "description": {"type": "string", "description": "Verbatim or close-paraphrase description from the page."},
                        "guidance": {"type": "string", "description": "Any surrounding explanation, examples, anti-patterns, scoring guidance, or verbosity. Empty string if none."},
                    },
                    "required": ["name", "description", "guidance"],
                },
            },
        },
        "required": ["orientation", "intended_audience", "subtask_short", "subtask_description", "subtask_keywords", "subtask_breadth", "error", "rubrics_metrics"],
    },
}


def html_to_text(raw_bytes: bytes) -> str:
    """Strip HTML to readable text using BeautifulSoup."""
    try:
        soup = BeautifulSoup(raw_bytes, "lxml")
    except Exception:
        soup = BeautifulSoup(raw_bytes, "html.parser")
    # Drop noise
    for tag in soup(["script", "style", "noscript", "svg", "form", "iframe", "img", "video", "audio"]):
        tag.decompose()
    # Title + body
    parts = []
    if soup.title and soup.title.string:
        parts.append(f"# {soup.title.string.strip()}\n")
    # Use main/article if present, else whole body
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    parts.append(text)
    return "\n".join(parts)


def looks_like_gibberish(text: str, sample_chars: int = 5000) -> bool:
    """Heuristic gibberish detector. PyMuPDF (and even pdfminer) sometimes return
    extracted text whose font encoding got mis-mapped, producing garbled output
    that is not real prose. We flag it and fall back to OCR.

    Checks:
      - too many Unicode replacement / control characters
      - too low letter ratio (real prose ≥ 60% letters)
      - too few real-looking words (mean word length out of 2..15 range, or
        very few words containing common English bigrams)
      - too many "words" (whitespace-separated tokens) longer than 25 chars
        (font-collapse usually fuses tokens together)
    """
    if not text:
        return False
    sample = text[:sample_chars]
    if not sample.strip():
        return False
    n = len(sample)
    # Replacement char + control chars
    bad = sum(1 for c in sample if c == "�" or (ord(c) < 32 and c not in "\t\n\r"))
    if bad / n > 0.05:
        return True
    # Letter ratio
    letters = sum(1 for c in sample if c.isalpha())
    if letters / n < 0.45:
        return True
    # Token analysis
    tokens = sample.split()
    if not tokens:
        return True
    long_tokens = sum(1 for t in tokens if len(t) > 25)
    if long_tokens / len(tokens) > 0.15:
        return True
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    if not (2.0 <= avg_len <= 15.0):
        return True
    # Common-English-bigram presence: real text has lots of "th", "in", "er", "an", "re"
    common_bg = ("th", "in", "er", "an", "re", "on", "at", "en", "nd", "ti")
    bg_hits = sum(sample.lower().count(b) for b in common_bg)
    if bg_hits < n / 200:  # i.e. fewer than 1 bigram-hit per 200 chars
        return True
    return False


def pdf_to_text(path: Path, ocr_max_pages: Optional[int] = None) -> str:
    """Extract PDF text. Tries fitz (fast), then pdfminer, then OCR fallback.
    Falls back to OCR if any earlier step returns gibberish."""
    text = ""
    # 1. PyMuPDF (fitz) — C-based, 5-10x faster than pdfminer
    try:
        import fitz
        doc = fitz.open(str(path))
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception:
        text = ""
    # 2. Fallback to pdfminer if fitz returned little or gibberish
    if HAVE_PDF and (len(text.strip()) < 200 or looks_like_gibberish(text)):
        try:
            text2 = pdf_extract_text(str(path)) or ""
            # Prefer text2 if it's longer AND not gibberish, OR if current text is gibberish
            if len(text2.strip()) > len(text.strip()) and not looks_like_gibberish(text2):
                text = text2
            elif looks_like_gibberish(text) and not looks_like_gibberish(text2) and len(text2.strip()) >= 200:
                text = text2
        except Exception:
            pass
    # 3. OCR fallback for image-only / gibberish-encoded PDFs
    if len(text.strip()) < 200 or looks_like_gibberish(text):
        text_ocr = pdf_ocr_text(path, max_pages=ocr_max_pages)
        if text_ocr and len(text_ocr.strip()) > 200 and not looks_like_gibberish(text_ocr):
            return text_ocr
        # If OCR result is also gibberish but the original wasn't completely empty, return original.
        if text.strip():
            return text
        return text_ocr
    return text


def pdf_ocr_text(path: Path, max_pages: Optional[int] = None, dpi: int = 200) -> str:
    """OCR fallback for image-only PDFs. Uses pdf2image + pytesseract.
    max_pages=None means OCR every page (slow on long books)."""
    try:
        import pdf2image
        import pytesseract
    except Exception:
        return ""
    try:
        images = pdf2image.convert_from_path(str(path), dpi=dpi, first_page=1, last_page=max_pages)
        return "\n\n".join(pytesseract.image_to_string(img) for img in images)
    except Exception as e:
        return f"__OCR_ERROR__: {e}"


def load_clean_text(path: Path) -> tuple[str, str]:
    """Return (clean_text, source_kind). source_kind in {html, pdf, text, empty, error}."""
    try:
        raw = path.read_bytes()
    except Exception as e:
        return f"__READ_ERROR__: {e}", "error"
    if not raw or len(raw) < 50:
        return "", "empty"
    name = path.name.lower()
    head = raw[:512].lstrip().lower()
    if name.endswith(".pdf") or head.startswith(b"%pdf"):
        return pdf_to_text(path), "pdf"
    if name.endswith((".html", ".htm")) or b"<html" in head[:200] or b"<!doctype html" in head[:200]:
        return html_to_text(raw), "html"
    # *_raw.md often wraps html with <!-- URL/HTTP --> headers; treat as html if it contains tags
    if b"<html" in raw[:2000].lower() or b"<body" in raw[:2000].lower():
        return html_to_text(raw), "html"
    # Plain text/markdown
    try:
        return raw.decode("utf-8", errors="replace"), "text"
    except Exception:
        return raw.decode("latin-1", errors="replace"), "text"


def truncate(text: str, limit: int = MAX_INPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.3):]
    return f"{head}\n\n... [TRUNCATED {len(text) - limit} chars] ...\n\n{tail}"


@dataclass
class CallResult:
    path: str
    model: str
    ok: bool
    extracted: Optional[dict]
    error: Optional[str]
    input_tokens: int
    output_tokens: int
    elapsed_s: float


async def extract_one(
    client: AsyncOpenAI,
    path: Path,
    model: str,
    semaphore: asyncio.Semaphore,
) -> CallResult:
    text, kind = load_clean_text(path)
    if kind in {"empty", "error"} or not text.strip():
        return CallResult(
            path=str(path), model=model, ok=True,
            extracted={
                "orientation": "error",
                "intended_audience": "",
                "subtask": "",
                "error": f"unreadable_source ({kind})",
                "rubrics_metrics": [],
            },
            error=None, input_tokens=0, output_tokens=0, elapsed_s=0.0,
        )
    text = truncate(text)
    user_msg = (
        f"PARENT TASK CONTEXT: This page was collected for the broader task: "
        f"{path.parents[2].name}\n\n"
        f"FILE: {path.name}\n\n"
        f"PAGE TEXT:\n{text}"
    )
    t0 = time.perf_counter()
    async with semaphore:
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
            return CallResult(
                path=str(path), model=model, ok=False, extracted=None,
                error=f"{type(e).__name__}: {e}",
                input_tokens=0, output_tokens=0,
                elapsed_s=time.perf_counter() - t0,
            )
    elapsed = time.perf_counter() - t0
    try:
        extracted = json.loads(resp.choices[0].message.content)
    except Exception as e:
        return CallResult(
            path=str(path), model=model, ok=False, extracted=None,
            error=f"json_parse: {e}; raw={resp.choices[0].message.content[:500]}",
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            elapsed_s=elapsed,
        )
    return CallResult(
        path=str(path), model=model, ok=True, extracted=extracted, error=None,
        input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        elapsed_s=elapsed,
    )


def list_raw_files(task: str) -> list[Path]:
    """Return every file we want to extract from: bulk-fetched raw/ + Claude-curated claude-parsed/."""
    base = DATASETS / task / "online-rubrics"
    out: list[Path] = []
    for sub in ("raw", "claude-parsed"):
        d = base / sub
        if d.is_dir():
            out.extend([p for p in d.iterdir() if p.is_file() and p.name != "_checkpoint.json"])
    return sorted(out)


def estimate_input_tokens(path: Path, encoding) -> int:
    """Fast estimate without pdfminer — uses byte-size heuristics."""
    try:
        sz = path.stat().st_size
    except Exception:
        return 0
    name = path.name.lower()
    head = b""
    try:
        with open(path, "rb") as fh:
            head = fh.read(512)
    except Exception:
        pass
    if name.endswith(".pdf") or head.lstrip().lower().startswith(b"%pdf"):
        # PDFs: ~rough 1 token / 5 bytes after extraction; cap at MAX
        est_chars = min(sz // 3, MAX_INPUT_CHARS)
        return min(est_chars // 4, MAX_INPUT_CHARS // 4) + 600
    if name.endswith((".html", ".htm")) or b"<html" in head[:200].lower():
        # HTML: BS strips ~70-90% of bytes; assume 25% remains as text
        est_chars = min(int(sz * 0.25), MAX_INPUT_CHARS)
        return est_chars // 4 + 600
    # Plain text
    return min(sz, MAX_INPUT_CHARS) // 4 + 600


# Approximate USD pricing per 1M tokens (input, output). Update as needed.
PRICING = {
    "gpt-5":      (1.25, 10.00),
    "gpt-5-mini": (0.25,  2.00),
    "gpt-5-nano": (0.05,  0.40),
    "gpt-4o":     (2.50, 10.00),
    "gpt-4o-mini":(0.15,  0.60),
}


async def run_pilot(args):
    client = _make_client()
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        files = []
        for t in tasks:
            f = list_raw_files(t)
            files.extend(f[args.offset : args.offset + args.n])
    else:
        files = list_raw_files(args.task)[args.offset : args.offset + args.n]
    if not files:
        print(f"No raw files for task={args.task}")
        return
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*[extract_one(client, p, args.model, sem) for p in files])
    print(f"\n=== PILOT: task={args.task} model={args.model} n={len(files)} ===")
    for r in results:
        print(f"\n--- {Path(r.path).name}  (in={r.input_tokens} out={r.output_tokens} t={r.elapsed_s:.1f}s ok={r.ok}) ---")
        if r.error:
            print(f"ERROR: {r.error}")
        else:
            print(json.dumps(r.extracted, indent=2)[:3000])
    in_p, out_p = PRICING.get(args.model, (1.0, 4.0))
    tot_in = sum(r.input_tokens for r in results)
    tot_out = sum(r.output_tokens for r in results)
    cost = tot_in * in_p / 1e6 + tot_out * out_p / 1e6
    n_ok = sum(1 for r in results if r.ok)
    print(f"\n--- summary ---")
    print(f"  ok: {n_ok}/{len(results)}")
    print(f"  total input  tokens: {tot_in}")
    print(f"  total output tokens: {tot_out}")
    print(f"  pilot cost @ {args.model}: ${cost:.4f}")
    if n_ok > 0:
        avg_in = tot_in / n_ok
        avg_out = tot_out / n_ok
        print(f"  avg / file: in={avg_in:.0f} out={avg_out:.0f} cost=${cost/n_ok:.4f}")


async def run_estimate(args):
    """Estimate cost across ALL raw files for chosen model without calling API."""
    enc = tiktoken.get_encoding("cl100k_base")
    in_p, out_p = PRICING.get(args.model, (1.0, 4.0))
    grand_in, grand_files = 0, 0
    print(f"=== Cost estimate @ model={args.model}  in=${in_p}/1M  out=${out_p}/1M ===")
    print(f"{'task':30s} {'files':>7s} {'in_tok_total':>14s} {'avg_in':>8s}")
    for task in TASKS:
        files = list_raw_files(task)
        # Sample 30 files for avg estimate, then extrapolate
        sample = files[::max(1, len(files) // 30)][:30]
        if not sample:
            continue
        sample_in = sum(estimate_input_tokens(p, enc) for p in sample)
        avg_in = sample_in / len(sample)
        total_in = int(avg_in * len(files))
        grand_in += total_in
        grand_files += len(files)
        print(f"{task:30s} {len(files):>7d} {total_in:>14d} {avg_in:>8.0f}")
    # Assume avg 800 output tokens / file
    avg_out = 800
    grand_out = grand_files * avg_out
    cost = grand_in * in_p / 1e6 + grand_out * out_p / 1e6
    print(f"{'TOTAL':30s} {grand_files:>7d} {grand_in:>14d}")
    print(f"  est output tokens (@800/file): {grand_out}")
    print(f"  ESTIMATED COST: ${cost:,.2f}")


async def _run_one_task(client: AsyncOpenAI, task: str, model: str, concurrency: int, checkpoint_every: int = 100):
    files = list_raw_files(task)
    out_dir = DATASETS / task / "online-rubrics" / "gpt-parsed" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "_checkpoint.json"
    # Resume support: skip files we've already processed
    todo = [p for p in files if not (out_dir / f"{p.parent.name}__{p.name}.json").exists()]
    done_already = len(files) - len(todo)
    print(f"\n=== task={task} model={model}  total={len(files)} todo={len(todo)} done_already={done_already} ===")
    if not todo:
        return {"task": task, "n_ok": done_already, "n_err": 0, "tot_in": 0, "tot_out": 0, "cost": 0.0, "skipped": True}
    sem = asyncio.Semaphore(concurrency)
    in_p, out_p = PRICING.get(model, (1.0, 4.0))
    n_ok = n_err = 0
    tot_in = tot_out = 0
    t0 = time.perf_counter()

    async def worker(p: Path):
        nonlocal n_ok, n_err, tot_in, tot_out
        r = await extract_one(client, p, model, sem)
        if r.ok:
            n_ok += 1
            (out_dir / f"{p.parent.name}__{p.name}.json").write_text(json.dumps({
                "path": r.path, "model": r.model,
                "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "elapsed_s": r.elapsed_s,
                "extracted": r.extracted,
            }, indent=2))
        else:
            n_err += 1
            (out_dir / f"{p.parent.name}__{p.name}.json").write_text(json.dumps({
                "path": r.path, "model": r.model, "error": r.error,
            }, indent=2))
        tot_in += r.input_tokens
        tot_out += r.output_tokens
        done_count = n_ok + n_err
        if done_count % checkpoint_every == 0:
            cost = tot_in * in_p / 1e6 + tot_out * out_p / 1e6
            elapsed = time.perf_counter() - t0
            rate = done_count / elapsed if elapsed > 0 else 0
            eta_s = (len(todo) - done_count) / rate if rate > 0 else 0
            checkpoint_path.write_text(json.dumps({
                "task": task, "model": model, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_done": done_count, "n_total": len(todo), "n_ok": n_ok, "n_err": n_err,
                "tot_in_tokens": tot_in, "tot_out_tokens": tot_out,
                "cost_so_far_usd": round(cost, 4),
                "elapsed_s": round(elapsed, 1), "rate_files_per_s": round(rate, 2),
                "eta_s": round(eta_s, 1),
            }, indent=2))
            print(f"  [{task}] done={done_count}/{len(todo)} ok={n_ok} err={n_err} cost=${cost:.2f} elapsed={elapsed:.0f}s eta={eta_s:.0f}s")

    await asyncio.gather(*(worker(p) for p in todo))
    cost = tot_in * in_p / 1e6 + tot_out * out_p / 1e6
    elapsed = time.perf_counter() - t0
    summary = {"task": task, "n_ok": n_ok, "n_err": n_err, "tot_in": tot_in, "tot_out": tot_out, "cost": cost, "elapsed_s": elapsed}
    checkpoint_path.write_text(json.dumps({**summary, "model": model, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True}, indent=2))
    print(f"=== DONE [{task}] ok={n_ok} err={n_err}  in={tot_in} out={tot_out} cost=${cost:.2f} elapsed={elapsed:.0f}s ===")
    return summary


async def run_full(args):
    client = _make_client()
    if args.all:
        tasks = TASKS
    else:
        tasks = [args.task]
    summaries = []
    for t in tasks:
        s = await _run_one_task(client, t, args.model, args.concurrency, checkpoint_every=args.checkpoint_every)
        summaries.append(s)
    grand_in = sum(s.get("tot_in", 0) for s in summaries)
    grand_out = sum(s.get("tot_out", 0) for s in summaries)
    grand_cost = sum(s.get("cost", 0.0) for s in summaries)
    grand_ok = sum(s.get("n_ok", 0) for s in summaries)
    grand_err = sum(s.get("n_err", 0) for s in summaries)
    print(f"\n========== GRAND TOTAL ok={grand_ok} err={grand_err} in={grand_in} out={grand_out} cost=${grand_cost:.2f} ==========")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, help="dataset task")
    p.add_argument("--tasks", help="comma-separated tasks for pilot sampling across multiple")
    p.add_argument("--all", action="store_true", help="process every task sequentially")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=100, help="write _checkpoint.json every N completions")
    p.add_argument("--n", type=int, default=5, help="pilot sample size per task")
    p.add_argument("--offset", type=int, default=0, help="pilot file offset")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--estimate", action="store_true", help="estimate cost across all tasks (no API calls)")
    args = p.parse_args()

    if args.estimate:
        asyncio.run(run_estimate(args))
    elif args.pilot:
        if not args.task and not args.tasks:
            print("--task or --tasks required for --pilot", file=sys.stderr); sys.exit(2)
        asyncio.run(run_pilot(args))
    else:
        if not args.task and not args.all:
            print("--task or --all required", file=sys.stderr); sys.exit(2)
        asyncio.run(run_full(args))


if __name__ == "__main__":
    main()
