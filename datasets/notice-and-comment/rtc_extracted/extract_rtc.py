#!/usr/bin/env python3
"""Phase A: extract Response-to-Comments (RTC) sections from final-rule documents.

Source: /Users/spangher/Projects/stanford-research/rfi-research/regulations-demo/data/bulk_downloads/
Output: rtc_sections.parquet + rtc_per_agency.csv

No LLM calls. Pure regex + HTML parsing.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

BULK_ROOT = Path(
    "/Users/spangher/Projects/stanford-research/rfi-research/regulations-demo/data/bulk_downloads"
)
OUT_DIR = Path(
    "/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/rtc_extracted"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_OUT = OUT_DIR / "rtc_sections.parquet"
AGENCY_CSV = OUT_DIR / "rtc_per_agency.csv"
SAMPLE_OUT = OUT_DIR / "samples.txt"

MAX_RTC_CHARS = 50_000
FALLBACK_WINDOW = 5_000

# ---------------------------------------------------------------------------
# Detection regexes (audit-compatible).
# ---------------------------------------------------------------------------
TRIGGER_PATTERNS = [
    r"response to comments?",
    r"we agree with the commenter",
    r"we disagree with the commenter",
    r"the commenter argues",
    r"in response,?\s+we",
    r"after consideration of the comments?",
    r"having reviewed the comments?",
]
TRIGGER_RE = re.compile("|".join(TRIGGER_PATTERNS), re.IGNORECASE)

# Recognise boilerplate "we cannot acknowledge or respond individually" non-RTC sections
# (common in CMS, also a handful of EPA pesticide notices with a one-line "comment" stub).
BOILERPLATE_RE = re.compile(
    r"(?:not\s+able\s+to\s+acknowledge\s+or\s+respond\s+to\s+them\s+individually"
    r"|because\s+of\s+the\s+large\s+number\s+of\s+public\s+comments)",
    re.IGNORECASE,
)

# Header patterns to locate the RTC block start.
HEADER_PATTERNS = [
    r"response\s+to\s+comments?",
    r"summary\s+of\s+(?:the\s+)?comments?\s+and\s+responses?",
    r"public\s+comments?\s+and\s+responses?",
    r"comments?\s+and\s+(?:the\s+)?(?:department'?s?|agency'?s?|epa'?s?|faa'?s?|fcc'?s?|cms'?s?|fws'?s?|sec'?s?)\s*responses?",
    r"comments?\s+received\s+and\s+(?:our|the\s+agency'?s?|epa'?s?)\s+responses?",
    r"discussion\s+of\s+comments?\s+and\s+responses?",
]
HEADER_RE = re.compile("|".join(HEADER_PATTERNS), re.IGNORECASE)

# Next-section break patterns: a new Roman-numeral or letter heading at column 0
# (after whitespace collapsing we look for line-leading markers).
NEXT_SECTION_RE = re.compile(
    r"\n\s*(?:"
    r"(?:[IVX]{1,5})\.\s+[A-Z]"           # Roman numeral
    r"|(?:[A-Z])\.\s+[A-Z][a-z]"          # Capital-letter outline
    r"|Regulatory Impact|Statutory Authority|List of Subjects"
    r"|Paperwork Reduction Act|Effective Date|Final Action"
    r"|Conclusion|References|Authority:?\s"
    r")",
    re.MULTILINE,
)

# Strip excessive whitespace.
WS_RE = re.compile(r"[ \t]+")
NL_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def html_to_text(html: str) -> tuple[str, str | None]:
    """Return (plaintext, title)."""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return html, None
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find(["h1", "h2"])
        if h1:
            title = h1.get_text(" ", strip=True)
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    # collapse
    text = WS_RE.sub(" ", text)
    text = NL_RE.sub("\n\n", text)
    return text.strip(), title


def read_content(rule_dir: Path) -> tuple[str | None, str | None, str]:
    """Return (rule_text, title, source_filename) for a rule directory."""
    # Prefer HTML; fallback to .pdf.txt; then .txt
    for fname in ("content.htm", "content.html"):
        p = rule_dir / fname
        if p.exists():
            try:
                html = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            text, title = html_to_text(html)
            return text, title, fname
    for fname in ("content.pdf.txt", "content.txt"):
        p = rule_dir / fname
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            text = NL_RE.sub("\n\n", WS_RE.sub(" ", text)).strip()
            return text, None, fname
    return None, None, ""


def extract_rtc(text: str) -> tuple[str, str]:
    """Return (rtc_text, extraction_method)."""
    # 1. Header-based extraction. Find the EARLIEST plausible header.
    best_start = None
    for m in HEADER_RE.finditer(text):
        # Heuristic: ignore matches inside the first 500 chars (likely TOC/title).
        # And ensure it's near the start of a line.
        s = m.start()
        line_prefix = text.rfind("\n", max(0, s - 80), s)
        # If the match begins shortly after a newline (or doc start), accept.
        if line_prefix == -1 and s > 800:
            continue
        # take the earliest acceptable
        best_start = s
        break

    if best_start is not None:
        # find end: next section break >= 1000 chars past start
        search_from = best_start + 1000
        next_sec = NEXT_SECTION_RE.search(text, search_from)
        end = next_sec.start() if next_sec else len(text)
        rtc = text[best_start:end].strip()
        if len(rtc) > 500:
            return rtc[:MAX_RTC_CHARS] + (
                f"\n\n[TRUNCATED at {MAX_RTC_CHARS} chars]" if len(rtc) > MAX_RTC_CHARS else ""
            ), "header_match"

    # 2. Fallback: ±FALLBACK_WINDOW around each trigger occurrence, merged.
    spans: list[tuple[int, int]] = []
    for m in TRIGGER_RE.finditer(text):
        s = max(0, m.start() - FALLBACK_WINDOW)
        e = min(len(text), m.end() + FALLBACK_WINDOW)
        if spans and s <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    if not spans:
        return "", ""
    pieces = [text[s:e] for s, e in spans]
    rtc = "\n\n---\n\n".join(pieces).strip()
    if len(rtc) > MAX_RTC_CHARS:
        rtc = rtc[:MAX_RTC_CHARS] + f"\n\n[TRUNCATED at {MAX_RTC_CHARS} chars]"
    return rtc, "fallback"


def load_metadata(rule_dir: Path) -> dict:
    p = rule_dir / "metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def process_rule_dir(rule_dir_str: str) -> dict | None:
    rule_dir = Path(rule_dir_str)
    rule_text, title, source_fname = read_content(rule_dir)
    if not rule_text:
        return None
    # Quick trigger check.
    if not TRIGGER_RE.search(rule_text):
        return {"_skip": True, "_path": str(rule_dir)}

    meta = load_metadata(rule_dir)
    rule_id = meta.get("Document ID") or rule_dir.name
    docket_id = meta.get("Docket ID")
    agency = meta.get("Agency ID")
    if not agency:
        # derive from path: bulk_downloads/<agency>/...
        try:
            parts = rule_dir.relative_to(BULK_ROOT).parts
            agency = parts[0]
        except ValueError:
            agency = "unknown"
    rule_title = meta.get("Title") or title

    rtc_text, method = extract_rtc(rule_text)
    if not rtc_text or len(rtc_text) < 1000:
        # Trigger fired but extraction returned nothing usable.
        return {"_skip": True, "_path": str(rule_dir), "_reason": "rtc_too_short"}
    # Filter the CMS / boilerplate "we cannot respond individually" non-RTC section.
    if BOILERPLATE_RE.search(rtc_text[:1500]) and len(rtc_text) < 2500:
        return {"_skip": True, "_path": str(rule_dir), "_reason": "boilerplate_only"}

    rel_source = str((rule_dir / source_fname).relative_to(BULK_ROOT))
    rule_text_len = len(rule_text)
    return {
        "agency": str(agency),
        "docket_id": str(docket_id) if docket_id else None,
        "rule_id": str(rule_id),
        "rule_title": rule_title,
        "rule_text_len": rule_text_len,
        "rtc_text": rtc_text,
        "rtc_text_len": len(rtc_text),
        "rtc_fraction": len(rtc_text) / max(rule_text_len, 1),
        "extraction_method": method,
        "source_path": rel_source,
    }


def find_rule_dirs() -> list[str]:
    """Find every rule directory across all agencies/years."""
    out: list[str] = []
    for rule_parent in BULK_ROOT.glob("*/*/downloaded_content/rule"):
        if not rule_parent.is_dir():
            continue
        for rule_dir in rule_parent.iterdir():
            if rule_dir.is_dir():
                out.append(str(rule_dir))
    return out


def main() -> None:
    t0 = time.time()
    print("[1/4] enumerating rule directories ...", flush=True)
    rule_dirs = find_rule_dirs()
    print(f"  found {len(rule_dirs):,} rule directories")

    print("[2/4] confirming RTC trigger count (quick scan) ...", flush=True)
    # Re-confirm with same trigger regex on raw text. Do it inside process_rule_dir
    # since that runs in parallel anyway.

    print("[3/4] parsing + extracting in parallel ...", flush=True)
    rows: list[dict] = []
    skipped = 0
    rtc_trigger_hits = 0
    n_workers = max(2, (os.cpu_count() or 4) - 1)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(process_rule_dir, p) for p in rule_dirs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                continue
            if row is None:
                continue
            if row.get("_skip"):
                rtc_trigger_hits += 1
                skipped += 1
                continue
            rows.append(row)
            rtc_trigger_hits += 1
            if i % 1000 == 0:
                print(f"  processed {i:,}/{len(rule_dirs):,} | extracted={len(rows):,}", flush=True)

    print(
        f"  trigger hits: {rtc_trigger_hits:,} | extracted: {len(rows):,} | "
        f"skipped (trigger but no usable section): {skipped:,}",
        flush=True,
    )

    print("[4/4] writing parquet + per-agency CSV ...", flush=True)
    df = pd.DataFrame(rows)
    # enforce dtypes
    if not df.empty:
        df = df.astype(
            {
                "agency": "string",
                "docket_id": "string",
                "rule_id": "string",
                "rule_title": "string",
                "rule_text_len": "int64",
                "rtc_text": "string",
                "rtc_text_len": "int64",
                "rtc_fraction": "float64",
                "extraction_method": "string",
                "source_path": "string",
            }
        )
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"  wrote {PARQUET_OUT} ({len(df):,} rows, {PARQUET_OUT.stat().st_size/1e6:.1f} MB)")

    # per-agency summary
    if not df.empty:
        agg = (
            df.groupby("agency")
            .agg(
                n_rtc=("rule_id", "count"),
                median_rtc_chars=("rtc_text_len", "median"),
                p95_rtc_chars=("rtc_text_len", lambda s: int(s.quantile(0.95))),
                n_header_match=(
                    "extraction_method",
                    lambda s: int((s == "header_match").sum()),
                ),
                n_fallback=("extraction_method", lambda s: int((s == "fallback").sum())),
                n_substantive=("rtc_text_len", lambda s: int((s > 10_000).sum())),
            )
            .reset_index()
            .sort_values("n_rtc", ascending=False)
        )
        agg.to_csv(AGENCY_CSV, index=False)
        print(f"  wrote {AGENCY_CSV}")
        print()
        print("=== per-agency summary ===")
        print(agg.to_string(index=False))
        print()
        print("=== overall stats ===")
        print(f"total extracted          : {len(df):,}")
        print(f"  header_match           : {int((df.extraction_method=='header_match').sum()):,}")
        print(f"  fallback               : {int((df.extraction_method=='fallback').sum()):,}")
        print(f"median rtc chars         : {int(df.rtc_text_len.median()):,}")
        print(f"95th-pct rtc chars       : {int(df.rtc_text_len.quantile(0.95)):,}")
        print(f">10K char RTCs           : {int((df.rtc_text_len > 10_000).sum()):,}")
        print(f">50K (truncated) RTCs    : {int(df.rtc_text.str.contains('TRUNCATED', na=False).sum()):,}")

        # samples
        print()
        print("=== 3 samples across agencies ===")
        sample_text = []
        sampled_agencies = set()
        for _, row in df.sort_values("rtc_text_len", ascending=False).iterrows():
            if row["agency"] in sampled_agencies:
                continue
            sampled_agencies.add(row["agency"])
            snippet = row["rtc_text"][:400].replace("\n", " ")
            block = (
                f"\n--- agency={row['agency']} rule_id={row['rule_id']} "
                f"len={row['rtc_text_len']} method={row['extraction_method']} ---\n"
                f"title: {row['rule_title']}\n"
                f"{snippet}...\n"
            )
            sample_text.append(block)
            print(block)
            if len(sample_text) >= 3:
                break
        SAMPLE_OUT.write_text("\n".join(sample_text))

    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
