#!/usr/bin/env python3
"""V2 backfill: extract verbatim RTC text for docs V2 flagged with responses
but the original strict-match regex missed.

Permissive variant of extract_rtc.py:
  - Adds more header variants (Public Input, Summary of Comments, Discussion of
    Comments, Comments and Responses, Comments and Department/Agency Response).
  - Drops the 1000-char "must have a real header" gate; accepts any header hit.
  - If no header matches, falls back to ±5K-char window around comment-trigger
    phrases capped at MAX_RTC_CHARS = 30K.
  - Returns rows even for `notice` / `proposed_rule` (not just `rule`).
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
PARQUET_OUT = OUT_DIR / "v2_backfill_rtc_sections.parquet"
SAMPLE_OUT = OUT_DIR / "v2_backfill_samples.txt"
INPUT_LIST = Path("/tmp/v2_only_to_extract.parquet")

MAX_RTC_CHARS = 30_000
FALLBACK_WINDOW = 5_000

TRIGGER_PATTERNS = [
    r"response to comments?",
    r"we agree with the commenter",
    r"we disagree with the commenter",
    r"the\s+(?:FAA|EPA|FDA|FCC|CMS|NRC|FWS|NMFS|SEC|FERC|IRS|USDA|HHS|DOT|DOL|FTC|OSHA|MSHA|NHTSA|FRA|PHMSA|FMCSA|BLM|NPS|APHIS|AMS|NOAA|ICE|Department|Agency|Commission|Bureau|Service|Office)\s+(?:agrees?|disagrees?|does\s+not\s+agree|believes?|notes?|recognizes?|acknowledges?|responds?)",
    r"the commenter argues?",
    r"the commenter stated?",
    r"commenter[s]?\s+(?:argued|stated|asked|noted|requested|suggested|recommended)",
    r"in response,?\s+we",
    r"in response to (?:the |these |this )?comments?",
    r"after consideration of the comments?",
    r"having reviewed the comments?",
    r"we received\s+(?:\d+|several|many|numerous|a number of)\s+comments?",
    r"the (?:FAA|EPA|FDA|FCC|CMS|NRC|FWS|NMFS|SEC|FERC|IRS|USDA|HHS|DOT|DOL|FTC|OSHA|MSHA|NHTSA|FRA|PHMSA|FMCSA|BLM|NPS|APHIS|AMS|NOAA|ICE|Department|Agency|Commission|Bureau|Service|Office)'?s? response",
]
TRIGGER_RE = re.compile("|".join(TRIGGER_PATTERNS), re.IGNORECASE)

BOILERPLATE_RE = re.compile(
    r"(?:not\s+able\s+to\s+acknowledge\s+or\s+respond\s+to\s+them\s+individually"
    r"|because\s+of\s+the\s+large\s+number\s+of\s+public\s+comments)",
    re.IGNORECASE,
)

# Expanded header patterns: include Public Input, Discussion of Comments,
# Summary of Comments, plus more agency abbreviations.
HEADER_PATTERNS = [
    r"response\s+to\s+(?:the\s+|public\s+)?comments?",
    r"responses?\s+to\s+(?:the\s+|public\s+)?comments?",
    r"summary\s+of\s+(?:the\s+)?(?:public\s+)?comments?(?:\s+and\s+responses?)?",
    r"public\s+input",
    r"public\s+comments?\s+and\s+(?:our\s+|agency\s+)?responses?",
    r"comments?\s+and\s+(?:the\s+)?(?:department'?s?|agency'?s?|epa'?s?|faa'?s?|fcc'?s?|cms'?s?|fws'?s?|sec'?s?|ftc'?s?|cdc'?s?|fda'?s?|usda'?s?|nrc'?s?|dot'?s?|hhs'?s?|dol'?s?|ed'?s?|noaa'?s?|nmfs'?s?|aphis'?s?|ams'?s?|atf'?s?|ice'?s?|irs'?s?|nps'?s?|blm'?s?|fws'?s?|osha'?s?|msha'?s?|fmcsa'?s?|fra'?s?|nhtsa'?s?|phmsa'?s?|treasury'?s?|coast\s+guard'?s?)\s*responses?",
    r"comments?\s+received\s+and\s+(?:our|the\s+agency'?s?|epa'?s?|department'?s?)\s+responses?",
    r"discussion\s+of\s+(?:public\s+)?comments?(?:\s+and\s+responses?)?",
    r"analysis\s+of\s+(?:public\s+)?comments?",
    r"comments?\s+and\s+responses?",
    r"comments?\s+on\s+the\s+(?:proposed\s+rule|proposal)",
    # Bare "Comments" / "Comments Received" line headers (common in FAA ADs,
    # EPA proposals, NRC). Must be preceded by newline + small indent and
    # followed by newline (heading-like).
    r"(?<=\n)\s{0,4}Comments(?:\s+Received)?\s*\n",
    r"(?<=\n)\s{0,4}Discussion\s*(?:of\s+(?:the\s+)?(?:Final\s+Rule|Public\s+)?Comments?)?\s*\n",
]
HEADER_RE = re.compile("|".join(HEADER_PATTERNS), re.IGNORECASE)

NEXT_SECTION_RE = re.compile(
    r"\n\s*(?:"
    r"(?:[IVX]{1,5})\.\s+[A-Z]"
    r"|(?:[A-Z])\.\s+[A-Z][a-z]"
    r"|Regulatory Impact|Statutory Authority|List of Subjects"
    r"|Paperwork Reduction Act|Effective Date|Final Action"
    r"|Conclusion|References|Authority:?\s"
    r"|Executive Order|Unfunded Mandates|Regulatory Flexibility"
    r")",
    re.MULTILINE,
)

WS_RE = re.compile(r"[ \t]+")
NL_RE = re.compile(r"\n{3,}")


def html_to_text(html: str) -> tuple[str, str | None]:
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
    text = WS_RE.sub(" ", text)
    text = NL_RE.sub("\n\n", text)
    return text.strip(), title


def read_content(rule_dir: Path) -> tuple[str | None, str | None, str]:
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


def extract_rtc_permissive(text: str) -> tuple[str, str]:
    """Return (rtc_text, extraction_method).

    Strategy:
      1) Find earliest header hit (permissive — accept anywhere after first 200 chars).
      2) Extend to next major section break OR end of doc.
      3) If no header or header span too small, use trigger-window fallback.
    """
    best_start = None
    for m in HEADER_RE.finditer(text):
        s = m.start()
        # Skip clear TOC matches in the first ~200 chars.
        if s < 200:
            continue
        best_start = s
        break

    if best_start is not None:
        search_from = best_start + 500
        next_sec = NEXT_SECTION_RE.search(text, search_from)
        end = next_sec.start() if next_sec else len(text)
        rtc = text[best_start:end].strip()
        if len(rtc) > 400:
            if len(rtc) > MAX_RTC_CHARS:
                rtc = rtc[:MAX_RTC_CHARS] + f"\n\n[TRUNCATED at {MAX_RTC_CHARS} chars]"
            return rtc, "header_match"

    # Trigger-window fallback.
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


def process_doc_dir(args: tuple[str, str, str, str]) -> dict | None:
    doc_id, agency_id, docket_id, doc_dir_str = args
    rule_dir = Path(doc_dir_str)
    rule_text, title, source_fname = read_content(rule_dir)
    if not rule_text:
        return None

    meta = load_metadata(rule_dir)
    rule_id = meta.get("Document ID") or doc_id
    docket_id = meta.get("Docket ID") or docket_id
    agency = meta.get("Agency ID") or agency_id
    if not agency:
        try:
            parts = rule_dir.relative_to(BULK_ROOT).parts
            agency = parts[0]
        except ValueError:
            agency = "unknown"
    rule_title = meta.get("Title") or title

    rtc_text, method = extract_rtc_permissive(rule_text)
    if not rtc_text or len(rtc_text) < 400:
        return None
    if BOILERPLATE_RE.search(rtc_text[:1500]) and len(rtc_text) < 2500:
        return None

    try:
        rel_source = str((rule_dir / source_fname).relative_to(BULK_ROOT))
    except ValueError:
        rel_source = str(rule_dir / source_fname)
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


def main() -> None:
    t0 = time.time()
    print("[1/3] loading V2-only doc list ...", flush=True)
    work = pd.read_parquet(INPUT_LIST)
    print(f"  {len(work):,} docs to process")

    args = list(zip(
        work["doc_id"].astype(str).tolist(),
        work["agency_id"].fillna("").astype(str).tolist(),
        work["docket_id"].fillna("").astype(str).tolist(),
        work["doc_dir"].astype(str).tolist(),
    ))

    print("[2/3] parsing + extracting in parallel ...", flush=True)
    rows: list[dict] = []
    n_workers = max(2, (os.cpu_count() or 4) - 1)
    print(f"  using {n_workers} workers")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(process_doc_dir, a) for a in args]
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                row = fut.result()
            except Exception:
                continue
            if row is not None:
                rows.append(row)
            if done % 1000 == 0:
                print(f"  processed {done:,}/{len(args):,} | extracted={len(rows):,}", flush=True)

    print(f"  extracted: {len(rows):,} / {len(args):,}", flush=True)

    print("[3/3] writing parquet ...", flush=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.astype({
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
        })
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"  wrote {PARQUET_OUT} ({len(df):,} rows, {PARQUET_OUT.stat().st_size/1e6:.1f} MB)")

    if not df.empty:
        print()
        print("=== stats ===")
        print(f"total extracted          : {len(df):,}")
        print(f"  header_match           : {int((df.extraction_method=='header_match').sum()):,}")
        print(f"  fallback               : {int((df.extraction_method=='fallback').sum()):,}")
        print(f"median rtc chars         : {int(df.rtc_text_len.median()):,}")
        print(f"95th-pct rtc chars       : {int(df.rtc_text_len.quantile(0.95)):,}")
        print(f">10K char RTCs           : {int((df.rtc_text_len > 10_000).sum()):,}")

        sample_text = []
        for _, row in df.sort_values("rtc_text_len", ascending=False).head(5).iterrows():
            snippet = row["rtc_text"][:400].replace("\n", " ")
            block = (
                f"\n--- agency={row['agency']} rule_id={row['rule_id']} "
                f"len={row['rtc_text_len']} method={row['extraction_method']} ---\n"
                f"title: {row['rule_title']}\n"
                f"{snippet}...\n"
            )
            sample_text.append(block)
            print(block)
        SAMPLE_OUT.write_text("\n".join(sample_text))

    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
