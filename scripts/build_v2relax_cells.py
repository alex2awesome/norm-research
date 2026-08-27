"""Ingest v2relax responses into cells_v1 as a new judge column.

Walks runs/cw_relax_appl/v2relax_responses/shard_{0..5}/, parses each
response file (Qwen-122B no-think JSON output), and writes a parquet at
outputs/v2_db/cells_v1/task=creative_writing/judge=qwen_relaxed_v2_2026_06_01/data.parquet
"""
import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path

import pandas as pd

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASK = "creative_writing"
JUDGE_LABEL = "qwen_relaxed_v2_2026_06_01"
RESPONSE_ROOT = REPO / "runs/cw_relax_appl/v2relax_responses"
OUT_DIR = REPO / f"outputs/v2_db/cells_v1/task={TASK}/judge={JUDGE_LABEL}"


# Parse the response into a {"results": [...]} dict, tolerant of CoT preamble.
def _extract_json(raw: str):
    # Try to find a json fenced block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try to find the last top-level {...} containing "results"
    # by scanning from the end.
    depth = 0
    end = None
    for i in range(len(raw) - 1, -1, -1):
        if raw[i] == "}":
            if depth == 0:
                end = i
            depth += 1
        elif raw[i] == "{":
            depth -= 1
            if depth == 0:
                candidate = raw[i:end + 1]
                if '"results"' in candidate:
                    try:
                        return json.loads(candidate)
                    except Exception:
                        pass
                # Reset search
                end = None
    # Fall back to regex
    m = re.search(r"\{[\s\S]*\"results\"[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def parse_response_file(path: Path, response_dir_label: str):
    raw = path.read_text(errors="ignore")
    parsed = _extract_json(raw)
    file_mtime = dt.datetime.utcfromtimestamp(path.stat().st_mtime).isoformat()
    rows = []
    if not parsed or "results" not in parsed:
        return rows, "parse_failed"
    for tr in parsed.get("results", []):
        if not isinstance(tr, dict):
            continue
        dp_id = tr.get("text_id")
        if not dp_id:
            continue
        scores = tr.get("scores")
        if not isinstance(scores, list):
            continue
        for sc in scores:
            if not isinstance(sc, dict):
                continue
            aid = sc.get("aspect_id")
            if not aid or not isinstance(aid, str) or not aid.startswith("a"):
                continue
            raw_score = sc.get("score")
            try:
                score_val = float(raw_score) if raw_score is not None else None
            except (TypeError, ValueError):
                score_val = None
            rows.append({
                "task": TASK,
                "judge_model": JUDGE_LABEL,
                "aspect_id": aid,
                "datapoint_id": dp_id,
                "paraphrase_idx": 0,
                "level": "R2",
                "score": score_val,
                "applicable": bool(sc.get("applicable", False)),
                "reason": sc.get("reason", "")[:500],
                "date_scored": file_mtime,
                "response_file": path.name,
                "response_key": path.stem,
                "parse_status": "ok",
                "response_dir": response_dir_label,
            })
    return rows, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true",
                    help="Count parses and cells, don't write.")
    args = ap.parse_args()

    all_rows = []
    n_files = 0; n_ok = 0; n_fail = 0
    for shard_dir in sorted(RESPONSE_ROOT.glob("shard_*")):
        for f in sorted(shard_dir.iterdir()):
            if not f.name.startswith(f"{TASK}__"):
                continue
            n_files += 1
            rows, status = parse_response_file(f, shard_dir.name)
            if status == "ok":
                n_ok += 1
            else:
                n_fail += 1
            all_rows.extend(rows)

    print(f"response files scanned: {n_files}")
    print(f"parse ok:   {n_ok}")
    print(f"parse fail: {n_fail}  ({n_fail / max(n_files, 1) * 100:.1f}%)")
    print(f"cells extracted: {len(all_rows)}")
    if all_rows:
        df = pd.DataFrame(all_rows)
        print(f"unique datapoints: {df['datapoint_id'].nunique()}")
        print(f"unique aspects:    {df['aspect_id'].nunique()}")
        print(f"applicability rate: {df['applicable'].mean() * 100:.1f}%")
        if not args.dry_run:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_p = OUT_DIR / "data.parquet"
            df.to_parquet(out_p)
            print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
