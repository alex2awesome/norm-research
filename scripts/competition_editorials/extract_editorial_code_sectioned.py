"""
extract_editorial_code_sectioned.py

Section-aware re-extraction of CF editorial code.

Motivation
----------
`extract_editorial_code.py` (the current pipeline) treats each CF blog row's
`editorial_text` as a single editorial and returns the "best" code block it
finds. But Codeforces editorial blogs are typically multi-problem: a single
blog covers problems A, B, C, D, E of one (or several) contests. The previous
extractor's heuristic of "pick the longest / most full-program-looking block"
silently grabbed code belonging to a sibling problem.

`outputs/v2_analysis/cf_contamination_probe.md` measured this directly via
bge-code-v1 cosine: ~25% of the extracted CF editorial codes were closer to a
SIBLING problem's candidates than to their OWN problem's candidates.

What this script does
---------------------
For each CF editorial row (source `cf_blog`):
  1. Identify per-problem section boundaries in `editorial_text` using
     Codeforces heading patterns. The strongest anchor is "1234A — Title"
     (contest-id + letter + dash) found at line-start. Fallback anchors
     include "Div2C Foo / Div1A Bar" style headings used by some contests.
  2. Locate the section whose heading letter matches the row's
     `canonical_pid` letter. If the row is `cf:<contest>_all`, keep the whole
     text and run extraction over each section, returning a JSON-encoded list
     of {section_letter, code, lang}.
  3. Run the original code-block extractor (`extract_one`) on the chosen
     section only. Assign confidence accordingly.

`open-r1` rows are passed through with `section_confidence="passthrough"` —
they are already per-problem (the upstream scrape sliced them) and contain
virtually no code anyway (3/5,428 had code markers).

AtCoder rows (`platform=ac`) are passed through unchanged; section-splitting
is not relevant to them (a separate AC blog corresponds to one problem).

Output schema
-------------
Same as the original `editorials_code_extracted.parquet` PLUS:
    section_method        one of {own_letter_match, all_letter_aggregate,
                                  passthrough_openr1, single_section,
                                  no_heading_fallback, no_text}
    section_confidence    high | medium | low | passthrough | ambiguous | none
    ambiguous_flag        True if the assigned section is uncertain
    n_sections_detected   int — number of distinct problem sections found
    own_section_present   True if a section header for THIS row's letter was found
    sectioned_text_len    chars in the section actually fed to extractor

NEVER modifies the existing `editorials_code_extracted.parquet`. Writes a
sibling file with the `_sectioned` suffix.

Validation
----------
Run with `--inspect-sample 25 --inspect-out PATH` to dump section boundaries
+ first 300 chars of every detected section across 25 randomly-picked blogs.
Always inspect a sample before running on the full 8.8K.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# Reuse extractor primitives from the original script (it lives next to this
# one). If invoked from a different CWD we still want to find it.
import sys
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from extract_editorial_code import extract_one  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Heading patterns
# ---------------------------------------------------------------------------
# Strongest signal: a CF problem id at line start, e.g. "1234A - Title",
# "664A - Complicated GCD", "1658D2 — Foo". The dash can be -, —, or –.
# Multi-letter suffixes (D1, D2, E1, E2, etc.) are allowed.
RE_PROBLEM_ID_HEADING = re.compile(
    r"(?:^|\n)\s*(?P<contest>\d{2,5})(?P<letter>[A-Z]\d?)\s*[-—–]\s*[^\n]{0,200}",
    re.MULTILINE,
)

# Fallback: "Div2C" / "Div1A" / "Div. 2 B" / "Division 2 A" style headings
# used by some older CF rounds. Often appears as "Div2E ... / Div1C ...".
RE_DIV_HEADING = re.compile(
    r"(?:^|\n)\s*(?:Div(?:ision)?\.?\s*)\s*\d+\s*[.\-—–]?\s*(?P<letter>[A-Z]\d?)\b",
    re.MULTILINE,
)

# Last-resort fallback for some hand-formatted editorials: a line consisting
# of just "Problem A" / "Problem B1" / "Tutorial A".
RE_PROBLEM_X_HEADING = re.compile(
    r"(?:^|\n)\s*(?:Problem|Tutorial|Task)\s+(?P<letter>[A-Z]\d?)\b",
    re.MULTILINE,
)


def _normalize_letter(s: str) -> str:
    """Match how canonical_pid encodes letters. Letters like 'D2' stay 'D2',
    plain 'A' stays 'A'. The CF problem-id format is upper-case; canonical_pid
    is lower-case ('cf:1658_d2'). We return upper-case for internal compare.
    """
    return s.upper()


def _row_letter(canonical_pid: str) -> str:
    """Extract the per-problem letter (or 'ALL') from `cf:<contest>_<letter>`."""
    if not isinstance(canonical_pid, str) or ":" not in canonical_pid:
        return ""
    tail = canonical_pid.split(":", 1)[1]
    if "_" not in tail:
        return ""
    return tail.split("_", 1)[1].upper()


def _row_contest(canonical_pid: str) -> str:
    if not isinstance(canonical_pid, str) or ":" not in canonical_pid:
        return ""
    tail = canonical_pid.split(":", 1)[1]
    return tail.split("_", 1)[0]


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------
def find_sections(text: str) -> list[dict]:
    """Return a list of detected sections.

    Each section dict has keys:
      letter   : problem letter (upper-case, e.g. "A", "D2"). Can be "?" if
                 we found a heading we can't tie to a letter (rare).
      contest  : contest id string if known, else "".
      start    : char position where the heading begins
      heading_end : char position immediately after the heading line
      method   : "contest_id" | "div" | "problem_x"

    Sections are sorted by `start`. Note: section content runs from `start`
    of THIS section until `start` of the NEXT section (or end of text).
    """
    sections: list[dict] = []

    # Phase A: strongest pattern — contest id headings.
    seen_pos: set[int] = set()
    for m in RE_PROBLEM_ID_HEADING.finditer(text):
        contest = m.group("contest")
        letter = _normalize_letter(m.group("letter"))
        # Heuristic guard: contest ids are usually <= 4 digits; "12345A" with
        # 5 digits is unusual but allowed. Skip clearly bogus things like
        # year-style 4-digit numbers that don't start an editorial-like line.
        # We accept everything; later we'll filter by matching the row's
        # contest if available.
        # Anchor at the start of the heading group, not the line break.
        # m.start() includes the leading \n if present.
        start = m.start()
        # advance past leading whitespace/newline
        while start < len(text) and text[start] in "\n\r\t ":
            start += 1
        if start in seen_pos:
            continue
        seen_pos.add(start)
        # The heading line ends at the next newline.
        nl = text.find("\n", m.end())
        heading_end = nl + 1 if nl != -1 else len(text)
        sections.append({
            "letter": letter,
            "contest": contest,
            "start": start,
            "heading_end": heading_end,
            "method": "contest_id",
        })

    if sections:
        sections.sort(key=lambda x: x["start"])
        return sections

    # Phase B: DivXY headings.
    div_hits = []
    for m in RE_DIV_HEADING.finditer(text):
        letter = _normalize_letter(m.group("letter"))
        start = m.start()
        while start < len(text) and text[start] in "\n\r\t ":
            start += 1
        nl = text.find("\n", m.end())
        heading_end = nl + 1 if nl != -1 else len(text)
        div_hits.append({
            "letter": letter,
            "contest": "",
            "start": start,
            "heading_end": heading_end,
            "method": "div",
        })

    if div_hits:
        # CF Div2/Div1 editorials often label the SAME problem twice (e.g.
        # "Div2C Robbers' watch / Div1A Robbers' watch"). If two consecutive
        # Div headings are within ~120 chars of each other, the intervening
        # text is just the slash/alt-name. Merge such pairs: the EARLIER
        # heading keeps both letter aliases, body extends to the next non-
        # adjacent heading.
        div_hits.sort(key=lambda x: x["start"])
        merged = []
        i = 0
        while i < len(div_hits):
            cur = div_hits[i]
            aliases = [cur["letter"]]
            j = i + 1
            while j < len(div_hits) and div_hits[j]["start"] - cur["heading_end"] < 120:
                aliases.append(div_hits[j]["letter"])
                cur = {**cur, "heading_end": div_hits[j]["heading_end"]}
                j += 1
            merged.append({**cur, "letter_aliases": aliases})
            i = j
        # Materialise as sections (use the FIRST alias as primary letter; the
        # row-letter matcher will check all aliases).
        for h in merged:
            h["letter_set"] = set(h["letter_aliases"])
            sections.append(h)
        sections.sort(key=lambda x: x["start"])
        return sections

    # Phase C: bare "Problem X" headings (rare; accept only if >= 2 hits to
    # avoid false positives like a random "Problem A is hard" prose line).
    candidates = []
    for m in RE_PROBLEM_X_HEADING.finditer(text):
        letter = _normalize_letter(m.group("letter"))
        start = m.start()
        while start < len(text) and text[start] in "\n\r\t ":
            start += 1
        nl = text.find("\n", m.end())
        heading_end = nl + 1 if nl != -1 else len(text)
        candidates.append({
            "letter": letter,
            "contest": "",
            "start": start,
            "heading_end": heading_end,
            "method": "problem_x",
        })
    if len(candidates) >= 2:
        candidates.sort(key=lambda x: x["start"])
        return candidates

    return []


def split_text_by_sections(text: str, sections: list[dict]) -> list[dict]:
    """Materialize sections by slicing `text`.

    Returns list of dicts: {letter, contest, method, start, end, body}.
    `body` includes the heading line.
    """
    out = []
    n = len(text)
    for i, s in enumerate(sections):
        start = s["start"]
        end = sections[i + 1]["start"] if i + 1 < len(sections) else n
        out.append({
            **s,
            "end": end,
            "body": text[start:end],
        })
    return out


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------
def process_row(row: dict, allow_passthrough_openr1: bool = True) -> dict:
    """Process a single editorial row, returning a dict of new columns."""
    text = row.get("editorial_text", "") or ""
    platform = row.get("platform", "")
    source = row.get("source", "")
    cp = row.get("canonical_pid", "")
    letter = _row_letter(cp)
    contest = _row_contest(cp)

    base = {
        "section_method": "",
        "section_confidence": "none",
        "ambiguous_flag": False,
        "n_sections_detected": 0,
        "own_section_present": False,
        "sectioned_text_len": 0,
        "all_sections_codes_json": None,
    }

    if not isinstance(text, str) or len(text) == 0:
        base["section_method"] = "no_text"
        return {**base, **extract_one("")}

    # AC and other platforms — pass through (no section splitting).
    if platform != "cf":
        base["section_method"] = "passthrough_non_cf"
        base["section_confidence"] = "passthrough"
        base["sectioned_text_len"] = len(text)
        return {**base, **extract_one(text)}

    # open-r1 CF rows are already per-problem.
    if source == "open-r1":
        base["section_method"] = "passthrough_openr1"
        base["section_confidence"] = "passthrough"
        base["sectioned_text_len"] = len(text)
        return {**base, **extract_one(text)}

    # cf_blog: section-split.
    sections = find_sections(text)
    base["n_sections_detected"] = len(sections)

    if len(sections) == 0:
        # No detectable heading: fall back to whole-text extraction but flag.
        base["section_method"] = "no_heading_fallback"
        base["section_confidence"] = "ambiguous"
        base["ambiguous_flag"] = True
        base["sectioned_text_len"] = len(text)
        return {**base, **extract_one(text)}

    if len(sections) == 1:
        # Single section: whole text is one problem.
        sec_letter = sections[0]["letter"]
        if letter and letter != "ALL" and sec_letter and sec_letter != letter:
            # Heading letter doesn't match the row letter — suspicious.
            base["section_method"] = "single_section_letter_mismatch"
            base["section_confidence"] = "ambiguous"
            base["ambiguous_flag"] = True
        else:
            base["section_method"] = "single_section"
            base["section_confidence"] = "high"
            base["own_section_present"] = True
        base["sectioned_text_len"] = len(text)
        return {**base, **extract_one(text)}

    # Multi-section.
    materialized = split_text_by_sections(text, sections)

    # If row letter is "ALL", emit per-section codes (no single 'extracted_code'
    # is canonical for the aggregate row, so we keep the first section as the
    # primary and store all sections in JSON).
    if letter == "ALL" or letter == "":
        # For each section, run extraction and collect.
        all_codes = []
        for sec in materialized:
            res = extract_one(sec["body"])
            all_codes.append({
                "letter": sec["letter"],
                "contest": sec["contest"],
                "method": sec["method"],
                "code_lang": res.get("code_lang"),
                "n_code_blocks": res.get("n_code_blocks"),
                "extraction_confidence": res.get("extraction_confidence"),
                "extracted_code": res.get("extracted_code"),
                "section_text_len": len(sec["body"]),
            })
        # Pick the longest code as the primary extracted_code (so downstream
        # code that ignores all_sections_codes_json still gets something
        # plausible — but note this is the AGGREGATE row).
        primary = max(all_codes, key=lambda x: len(x["extracted_code"] or "")) if all_codes else None
        primary_code = primary["extracted_code"] if primary else ""
        primary_lang = primary["code_lang"] if primary else "unknown"
        primary_conf = primary["extraction_confidence"] if primary else "none"
        base["section_method"] = "all_letter_aggregate"
        base["section_confidence"] = "high"
        base["sectioned_text_len"] = len(text)
        base["all_sections_codes_json"] = json.dumps(all_codes, ensure_ascii=False)
        # Aggregate rows are intentionally ambiguous in the per-row sense.
        base["ambiguous_flag"] = True
        return {**base,
                "extracted_code": primary_code,
                "code_lang": primary_lang,
                "n_code_blocks": sum(c["n_code_blocks"] for c in all_codes),
                "extraction_confidence": primary_conf}

    # Per-letter row: try to find matching section. A section may carry
    # `letter_set` (Div-merged) with multiple aliases; match against any.
    def _matches_letter(s):
        if s.get("letter_set"):
            return letter in s["letter_set"]
        return s["letter"] == letter
    matches = [s for s in materialized if _matches_letter(s)]
    # If contest is known, prefer same-contest matches.
    if contest and matches:
        same_contest = [s for s in matches if s["contest"] == contest]
        if same_contest:
            matches = same_contest

    if matches:
        # If multiple matches, pick the longest one (more content usually =
        # the actual problem section, not an in-prose reference).
        sec = max(matches, key=lambda s: s["end"] - s["start"])
        base["section_method"] = "own_letter_match"
        # Confidence:
        #   high   -- exactly one matching section AND its contest agrees with
        #             the row's contest (or one of them is unknown).
        #   medium -- multiple sections share this letter (we pick the longest).
        #   low    -- matched on letter but the section's contest disagrees
        #             with the row's contest (data inconsistency at the source).
        contest_ok = (not contest or not sec["contest"]
                      or sec["contest"] == contest)
        if not contest_ok:
            base["section_confidence"] = "low"
            base["ambiguous_flag"] = True
        elif len(matches) == 1:
            base["section_confidence"] = "high"
        else:
            base["section_confidence"] = "medium"
        base["own_section_present"] = True
        base["sectioned_text_len"] = sec["end"] - sec["start"]
        return {**base, **extract_one(sec["body"])}

    # No section with this letter — code may be in pre-heading preamble or
    # absent. Fall back to FIRST section only (often the row's letter section
    # was the first one but the heading was malformed). Flag as ambiguous.
    sec = materialized[0]
    base["section_method"] = "no_own_section_first_fallback"
    base["section_confidence"] = "low"
    base["ambiguous_flag"] = True
    base["sectioned_text_len"] = sec["end"] - sec["start"]
    return {**base, **extract_one(sec["body"])}


# ---------------------------------------------------------------------------
# Inspection dump (for validation before scaling)
# ---------------------------------------------------------------------------
def dump_inspection(df: pd.DataFrame, out_path: Path, n: int = 25, seed: int = 0) -> None:
    sample = df[df.platform == "cf"]
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    lines = []
    for _, row in sample.iterrows():
        lines.append("=" * 100)
        lines.append(f"canonical_pid={row['canonical_pid']} source={row['source']} ed_id={row['editorial_id']}")
        text = row.get("editorial_text", "") or ""
        secs = find_sections(text)
        lines.append(f"text_len={len(text)}  n_sections={len(secs)}")
        if not secs:
            lines.append("  (no section headings detected)")
            lines.append("FIRST 400 CHARS:")
            lines.append(text[:400])
            lines.append("")
            continue
        mats = split_text_by_sections(text, secs)
        for i, s in enumerate(mats):
            preview = s["body"][:300].replace("\n", " ⏎ ")
            lines.append(f"  [{i}] letter={s['letter']} contest={s['contest'] or '?'} method={s['method']} chars={s['end']-s['start']}")
            lines.append(f"        {preview}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"wrote inspection dump to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="Path to editorials.parquet (the raw blog texts)")
    ap.add_argument("--out", dest="out", required=False, default="",
                    help="Output parquet path (e.g. .../editorials_code_extracted_sectioned.parquet); required unless --inspect-sample is set")
    ap.add_argument("--platforms", default="cf",
                    help="Comma-separated platforms to process (default cf only — AC handled by original extractor)")
    ap.add_argument("--inspect-sample", type=int, default=0,
                    help="If >0, write a section-boundary inspection dump and EXIT (no parquet written).")
    ap.add_argument("--inspect-out", default="",
                    help="Path to write the inspection dump (required when --inspect-sample > 0)")
    ap.add_argument("--inspect-seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, only process that many rows total (for dry-run)")
    args = ap.parse_args()

    plats = [p.strip() for p in args.platforms.split(",") if p.strip()]

    print(f"loading {args.inp} …", flush=True)
    df = pd.read_parquet(args.inp)
    print(f"loaded {len(df):,} rows", flush=True)
    sub = df[df["platform"].isin(plats)].copy()
    print(f"processing {len(sub):,} rows on platforms {plats}", flush=True)

    if args.inspect_sample > 0:
        if not args.inspect_out:
            raise SystemExit("--inspect-out required with --inspect-sample")
        dump_inspection(sub, Path(args.inspect_out), n=args.inspect_sample,
                        seed=args.inspect_seed)
        return

    if not args.out:
        raise SystemExit("--out required when not in --inspect-sample mode")

    if args.sample > 0:
        sub = sub.sample(n=min(args.sample, len(sub)), random_state=0)
        print(f"sampled to {len(sub):,} rows for dry-run", flush=True)

    out_rows = []
    counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    n_total = len(sub)
    for i, (_, row) in enumerate(sub.iterrows()):
        if i % 500 == 0:
            print(f"  … {i:,}/{n_total:,}", flush=True)
        res = process_row(row.to_dict())
        method_counts[res["section_method"]] = method_counts.get(res["section_method"], 0) + 1
        counts[res["extraction_confidence"]] = counts.get(res["extraction_confidence"], 0) + 1
        out_rows.append({
            "editorial_id": row["editorial_id"],
            "platform": row["platform"],
            "problem_id": row["problem_id"],
            "canonical_pid": row.get("canonical_pid"),
            "source": row.get("source"),
            **res,
        })

    out_df = pd.DataFrame(out_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"wrote {len(out_df):,} rows to {out_path}", flush=True)

    print("\nsection_method distribution:")
    for k, v in sorted(method_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:42s} {v:6,}  ({100.0*v/n_total:5.1f}%)")
    print("\nextraction_confidence distribution:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {v:6,}  ({100.0*v/n_total:5.1f}%)")


if __name__ == "__main__":
    main()
