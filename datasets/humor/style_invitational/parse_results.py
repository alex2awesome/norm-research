"""Parse Style Invitational weekly text files (nrars.org Book of Weeks)
into style_invitational.jsonl with (week_id, contest_prompt, entry_text, tier).

Each "01 text" file NNNN.txt contains the new contest for that file's week
plus a "Report from Week N" section = the RESULTS (verdicts) of an earlier
week. Tiers: winner / runnerup (First..Sixth Runner-Up) / honorable_mention.

Heuristic plain-text parsing (formats drift across 30 years); raw files are
preserved under raw/01_text/ so parsing can always be redone.
Usage: python3 parse_results.py
"""
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw", "01_text")
INDEX = os.path.join(HERE, "raw", "master_index.json")
OUT = os.path.join(HERE, "style_invitational.jsonl")

RE_REPORT = re.compile(r"(?im)^\s*(?:the\s+)?report from week\s+(\d+)")
RE_END = re.compile(r"(?im)^\s*(next week[:\s]|new contest\b|copyright )")

ORD = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|1st|2nd|3rd|4th|5th|6th)"
MARKERS = re.compile(
    rf"(?im)"
    rf"(?P<ru>\b(?:and\s+)?(?:the\s+)?{ORD}[- ]runner(?:s)?[- ]up[^:\n]{{0,60}}:)"
    rf"|(?P<w>\band\s+(?:the\s+)?winner[^:\n]{{0,120}}:"
    rf"|^\s*(?:and\s+)?(?:the\s+)?winner\s+of[^:\n]{{0,120}}:"
    rf"|^\s*(?:the\s+)?winner\s*:)"
    rf"|(?P<hm>\b(?:and\s+(?:the\s+)?)?honorable\s+mentions?[^:\n]{{0,60}}:"
    rf"|^\s*(?:the\s+)?honorable\s+mentions?\s*$"
    rf"|^\s*(?:and\s+)?(?:the\s+)?(?:other\s+)?honorable\s+mentions\b[^\n]{{0,80}}$)")


def paragraphs(text):
    return [re.sub(r"\s+", " ", p).strip()
            for p in re.split(r"\n\s*\n", text) if p.strip()]


def parse_file(path, prompts):
    txt = open(path, encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
    m = RE_REPORT.search(txt)
    if not m:
        return []
    week = int(m.group(1))
    body = txt[m.end():]
    endm = RE_END.search(body)
    if endm:
        body = body[: endm.start()]

    marks = list(MARKERS.finditer(body))
    rows = []
    for i, mk in enumerate(marks):
        tier = "runnerup" if mk.lastgroup == "ru" else (
            "winner" if mk.lastgroup == "w" else "honorable_mention")
        seg = body[mk.end(): marks[i + 1].start() if i + 1 < len(marks) else len(body)]
        paras = paragraphs(seg)
        if not paras:
            continue
        if tier in ("runnerup", "winner"):
            entries = paras[:1]  # header labels the entry that follows it
        else:
            entries = paras       # every paragraph is an honorable mention
        for e in entries:
            if len(e) < 3:
                continue
            rows.append({
                "week_id": week,
                "contest_prompt": prompts.get(week),
                "entry_text": e,
                "tier": tier,
            })
    return rows


def main():
    prompts = {}
    if os.path.exists(INDEX):
        for r in json.load(open(INDEX)):
            prompts[r["week"]] = r.get("synopsis") or r.get("title")
    all_rows = []
    parsed_files = skipped = 0
    for f in sorted(os.listdir(RAW)):
        if not f.endswith(".txt"):
            continue
        rows = parse_file(os.path.join(RAW, f), prompts)
        if rows:
            parsed_files += 1
            all_rows.extend(rows)
        else:
            skipped += 1
    with open(OUT, "w") as out:
        for r in all_rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    tiers = Counter(r["tier"] for r in all_rows)
    weeks = len(set(r["week_id"] for r in all_rows))
    print(f"files yielding results: {parsed_files}; files without: {skipped}")
    print(f"weeks with results: {weeks}; rows: {len(all_rows)}; tiers: {dict(tiers)}")


if __name__ == "__main__":
    main()
