"""Section-aware editorial-code extraction from JS-rendered CF blog HTML.

Input:  datasets/codeforces_delta/rendered_html/blog_{id}.html  (1,428 files;
        v1 old-era + v2 modern passes, shared namespace)
        contest_blog_map.parquet + contest_blog_map_v2.parquet (blog -> contest(s))
Output: datasets/codeforces_delta/editorials_rendered_extracted.parquet
        one row per (blog_id, contest_id, letter) section that has extractable
        code, plus no-code rows for coverage accounting.

Sectioning strategy (priority order per blog):
  1. anchor-heading: heading-ish element (h1-h6, or a <p>/<div> whose text is
     short) containing <a href="/contest/{cid}/problem/{L}"> or
     /problemset/problem/{cid}/{L}.  Gives contest AND letter directly.
  2. text-heading: heading text matching
        "1995B1 - Bouquet", "B — Array Craft", "Problem C", "Div2C", "1006F"
     Letter from the pattern; contest from the blog->contest map (all mapped
     contests get a row — Div1/Div2 shared editorials map the same letter to
     both contests only when the letter pattern carries no contest id).
  3. no sections found: whole-blog fallback, letter=None, ambiguous.

Code extraction within a section: all <pre> blocks (rendered spoilers
included).  Primary code = longest block that looks like a full program
(#include / def / public class / input() / print).  Lang guess from content.

APPEND-ONLY: never modifies existing files.  Safe to re-run (full rewrite of
its own output only).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/codeforces_delta")
HTML_DIR = ROOT / "rendered_html"
OUT_PATH = ROOT / "editorials_rendered_extracted.parquet"

ANCHOR_RE = re.compile(
    r"/(?:contest/(\d+)/problem/([A-Z]\d?)|problemset/problem/(\d+)/([A-Z]\d?))",
    re.I,
)
# "1995B1 - Bouquet", "1006F. Xor-paths", "664A — Complicated GCD"
TXT_CID_RE = re.compile(r"^\s*(\d{2,5})\s*([A-Z]\d?)\s*[\-—–.:]", re.I)
# "B — Array Craft", "A. Theatre Square", "C: something"  (single letter heading)
TXT_LETTER_RE = re.compile(r"^\s*([A-Z]\d?)\s*[\-—–.:]\s+\S")
# "Problem A", "Task B2", "Задача C"
TXT_PROBLEM_RE = re.compile(r"^\s*(?:Problem|Task|Tutorial|Задача)\s+([A-Z]\d?)\b", re.I)
# "Div2C", "Div. 1 B"
TXT_DIV_RE = re.compile(r"^\s*Div(?:ision)?\.?\s*\d\s*[.\-—–:]?\s*([A-Z]\d?)\b", re.I)

FULLPROG_HINTS = (
    "#include", "int main", "def main", "public class", "public static void",
    "input()", "print(", "cin >>", "scanf", "readline", "fn main",
)


def guess_lang(code: str) -> str:
    c = code
    if "#include" in c or "int main" in c or "cin >>" in c or "scanf" in c:
        return "cpp"
    if "public class" in c or "public static void" in c:
        return "java"
    if re.search(r"^\s*def \w+|^\s*import \w+|print\(|input\(\)", c, re.M):
        return "python"
    if re.search(r"^\s*(for|if|while)\b.*:\s*$", c, re.M):
        return "python"
    return "unknown"


def looks_full_program(code: str) -> bool:
    return any(h in code for h in FULLPROG_HINTS)


def text_of(el: Tag) -> str:
    return el.get_text(" ", strip=True)


def is_headingish(el: Tag) -> bool:
    if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return True
    if el.name in ("p", "div", "b", "strong"):
        t = text_of(el)
        return 0 < len(t) <= 160
    return False


def section_key_from_el(el: Tag):
    """Return list of (contest_id|None, letter) if el is a section heading."""
    keys = []
    if not is_headingish(el):
        return keys
    # 1) anchors
    for a in el.find_all("a", href=True):
        m = ANCHOR_RE.search(a["href"])
        if m:
            cid = m.group(1) or m.group(3)
            letter = (m.group(2) or m.group(4)).upper()
            keys.append((cid, letter, "anchor"))
    if keys:
        # only trust anchors in heading-ish position when the element is not a
        # long prose paragraph (filtered by is_headingish text cap already)
        return keys
    t = text_of(el)
    if not t:
        return keys
    m = TXT_CID_RE.match(t)
    if m:
        return [(m.group(1), m.group(2).upper(), "text_cid")]
    m = TXT_DIV_RE.match(t)
    if m:
        return [(None, m.group(1).upper(), "text_div")]
    m = TXT_PROBLEM_RE.match(t)
    if m:
        return [(None, m.group(1).upper(), "text_problem")]
    # bare-letter heading only for real h tags or bold (high precision)
    if el.name in ("h1", "h2", "h3", "h4", "h5", "h6", "b", "strong"):
        m = TXT_LETTER_RE.match(t)
        if m and len(t) <= 80:
            return [(None, m.group(1).upper(), "text_letter")]
    return keys


def flatten_children(content: Tag):
    """Yield document-order top-level-ish elements of the blog body."""
    for child in content.children:
        if isinstance(child, Tag):
            yield child


def extract_blog(blog_id: int, html: str, mapped_contests: list[str]) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("div.ttypography")
    if content is None:
        content = soup.body or soup
    els = list(flatten_children(content))
    # locate section boundaries
    bounds = []  # (idx, [(cid, letter, how), ...])
    for i, el in enumerate(els):
        keys = section_key_from_el(el)
        if keys:
            bounds.append((i, keys))
    rows = []

    def collect_pre(section_els):
        pres = []
        for el in section_els:
            for pre in ([el] if el.name == "pre" else el.find_all("pre")):
                # NO separator: highlight.js wraps every token in <span>;
                # whitespace/newlines live in text nodes, so plain get_text()
                # reconstructs the original source faithfully.
                txt = pre.get_text()
                txt = txt.replace("\r\n", "\n").strip("\n")
                if len(txt.strip()) >= 20:
                    pres.append(txt)
        # dedup identical blocks
        seen, out = set(), []
        for p in pres:
            k = hash(p)
            if k not in seen:
                seen.add(k)
                out.append(p)
        return out

    if not bounds:
        pres = collect_pre(els)
        primary = ""
        full = [p for p in pres if looks_full_program(p)]
        if full:
            primary = max(full, key=len)
        elif pres:
            primary = max(pres, key=len)
        rows.append({
            "blog_id": blog_id,
            "contest_id": mapped_contests[0] if mapped_contests else None,
            "letter": None,
            "section_method": "no_heading_fallback",
            "n_code_blocks": len(pres),
            "extracted_code": primary,
            "all_codes_json": json.dumps(pres[:8]),
            "section_text_len": sum(len(text_of(e)) for e in els),
        })
        return rows

    # consecutive merge: same key appearing twice in a row (e.g. anchor twice)
    for bi, (start, keys) in enumerate(bounds):
        end = bounds[bi + 1][0] if bi + 1 < len(bounds) else len(els)
        section_els = els[start:end]
        pres = collect_pre(section_els)
        full = [p for p in pres if looks_full_program(p)]
        primary = max(full, key=len) if full else (max(pres, key=len) if pres else "")
        # dedupe keys (anchor repeated)
        kseen = set()
        for cid, letter, how in keys:
            if how.startswith("text") and cid is None:
                cids = mapped_contests or [None]
            else:
                cids = [cid]
            for c in cids:
                kk = (c, letter)
                if kk in kseen:
                    continue
                kseen.add(kk)
                rows.append({
                    "blog_id": blog_id,
                    "contest_id": c,
                    "letter": letter,
                    "section_method": how,
                    "n_code_blocks": len(pres),
                    "extracted_code": primary,
                    "all_codes_json": json.dumps(pres[:8]),
                    "section_text_len": sum(len(text_of(e)) for e in section_els),
                })
    return rows


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    # blog -> contests map (both eras)
    m1 = pd.read_parquet(ROOT / "contest_blog_map.parquet")
    m2 = pd.read_parquet(ROOT / "contest_blog_map_v2.parquet")
    blog2contests: dict[int, list[str]] = {}
    for df in (m1, m2):
        sub = df[df["blog_entry_id"].notna()]
        for cid, bid in zip(sub["contest_id"], sub["blog_entry_id"]):
            blog2contests.setdefault(int(bid), []).append(str(cid))
    files = sorted(HTML_DIR.glob("blog_*.html"))
    if limit:
        files = files[:limit]
    all_rows = []
    for k, f in enumerate(files):
        bid = int(f.stem.split("_")[1])
        try:
            html = f.read_text(encoding="utf-8", errors="replace")
            rows = extract_blog(bid, html, blog2contests.get(bid, []))
        except Exception as e:
            rows = [{"blog_id": bid, "contest_id": None, "letter": None,
                     "section_method": f"parse_error:{type(e).__name__}",
                     "n_code_blocks": 0, "extracted_code": "",
                     "all_codes_json": "[]", "section_text_len": 0}]
        all_rows.extend(rows)
        if (k + 1) % 100 == 0:
            print(f"{k+1}/{len(files)} blogs, {len(all_rows)} rows", flush=True)
    df = pd.DataFrame(all_rows)
    df["code_lang"] = df["extracted_code"].map(guess_lang)
    df["has_code"] = df["extracted_code"].str.len() >= 30
    df["canonical_pid"] = None
    ok = df["contest_id"].notna() & df["letter"].notna()
    df.loc[ok, "canonical_pid"] = (
        "cf:" + df.loc[ok, "contest_id"].astype(str) + "_" + df.loc[ok, "letter"].str.lower()
    )
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}: {len(df)} rows, "
          f"{df['has_code'].sum()} with code, "
          f"{df.loc[df['has_code'], 'canonical_pid'].nunique()} unique pids with code")
    print(df["section_method"].value_counts().to_string())


if __name__ == "__main__":
    main()
