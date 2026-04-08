#!/usr/bin/env python3
"""Post-process VLLM press release extraction output.

Parses raw_response fields (which are JSON wrapped in markdown fences)
to extract clean text and news_release_found flags.
"""

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_raw_response(raw: str) -> dict:
    """Parse the raw model response to extract fields."""
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        return {
            "news_release_found": parsed.get("news_release_found", True),
            "extracted_text": parsed.get("result", ""),
            "parse_success": True,
        }
    except json.JSONDecodeError:
        pass

    # Fallback 1: extract news_release_found flag
    found_match = re.search(
        r'"news_release_found"\s*:\s*(true|false)', cleaned, re.IGNORECASE
    )
    found = found_match.group(1).lower() == "true" if found_match else True

    # Fallback 2: grab everything after "result": " and clean up the tail
    trunc_match = re.search(r'"result"\s*:\s*"', cleaned)
    if trunc_match:
        text = cleaned[trunc_match.end():]
        # Strip trailing JSON closure: optional trailing quote, whitespace, }, whitespace
        text = re.sub(r'"\s*\}\s*$', "", text)
        # Unescape JSON string escapes
        text = text.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        # Detect if response was truncated (no proper JSON closing)
        truncated = not cleaned.rstrip().endswith("}")
        return {
            "news_release_found": found,
            "extracted_text": text,
            "parse_success": True,
            "truncated": truncated,
        }

    return {
        "news_release_found": found,
        "extracted_text": cleaned,
        "parse_success": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Post-process press release extraction")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "datasets" / "press-releases" / "press_release_extracted.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_name("press_release_clean.jsonl")

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    results = []
    parse_ok = 0
    parse_fail = 0
    found_count = 0
    not_found_count = 0
    empty_text = 0

    for rec in records:
        parsed = parse_raw_response(rec["raw_response"])
        out = {
            "press_release_id": rec["press_release_id"],
            "news_release_found": parsed["news_release_found"],
            "extracted_text": parsed["extracted_text"],
        }
        results.append(out)

        if parsed["parse_success"]:
            parse_ok += 1
        else:
            parse_fail += 1
        if parsed["news_release_found"]:
            found_count += 1
        else:
            not_found_count += 1
        if not parsed["extracted_text"].strip():
            empty_text += 1

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Parse success: {parse_ok} ({100*parse_ok/len(records):.1f}%)")
    print(f"Parse failed:  {parse_fail} ({100*parse_fail/len(records):.1f}%)")
    print(f"PR found:      {found_count}")
    print(f"PR not found:  {not_found_count}")
    print(f"Empty text:    {empty_text}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
