"""Parse v2 judge responses into a flat scores JSONL.

For each judge response file, extract (text_id, aspect_id, applicable, score,
reason), join with manifest for paraphrase + bundle metadata, write JSONL.

Output: runs/validity_full/full_v2/judge_scores.jsonl
  {"key": "b0__p0__c0", "bundle_id": "b0", "paraphrase_idx": 0,
   "chunk_idx": 0, "aspect_id": "a0", "datapoint_id": "d00042",
   "applicable": true, "score": 0.5, "reason": "..."}
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_response(raw: str) -> dict | None:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        if s < 0 or e <= s: return None
        try: return json.loads(raw[s:e+1])
        except json.JSONDecodeError: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/validity_full/full_v2")
    ap.add_argument("--response-subdir", default="judge_responses")
    ap.add_argument("--out-name", default="judge_scores.jsonl")
    args = ap.parse_args()

    base = Path(args.run_dir)
    manifest = json.loads((base / "judge_manifest.json").read_text())
    by_key = {m["key"]: m for m in manifest}
    rdir = base / args.response_subdir

    n_files = n_ok = n_parse_fail = n_records = 0
    aspect_set = set()
    out_path = base / args.out_name
    with out_path.open("w") as fh:
        for rp in sorted(rdir.glob("*.json")):
            key = rp.stem
            entry = by_key.get(key)
            if entry is None: continue
            n_files += 1
            try:
                raw = rp.read_text()
            except Exception:
                continue
            obj = _parse_response(raw)
            if obj is None or "results" not in obj:
                n_parse_fail += 1
                continue
            for tr in obj["results"]:
                tid = tr.get("text_id")
                for sc in tr.get("scores", []):
                    aid = sc.get("aspect_id")
                    if not aid or not tid: continue
                    applicable = sc.get("applicable")
                    score = sc.get("score")
                    reason = sc.get("reason", "")
                    # Coerce score to float in [0,1] or None
                    try:
                        score = (float(score) if score is not None
                                  and applicable is not False else None)
                        if score is not None:
                            score = max(0.0, min(1.0, score))
                    except (ValueError, TypeError):
                        score = None
                    fh.write(json.dumps({
                        "key": key,
                        "bundle_id": entry["bundle_id"],
                        "paraphrase_idx": entry["paraphrase_idx"],
                        "chunk_idx": entry["chunk_idx"],
                        "aspect_id": aid,
                        "datapoint_id": tid,
                        "applicable": applicable,
                        "score": score,
                        "reason": reason[:300],
                    }) + "\n")
                    n_records += 1
                    aspect_set.add(aid)
            n_ok += 1

    print(f"parsed {n_ok}/{n_files} response files "
          f"({n_parse_fail} parse failures)")
    print(f"  {n_records:,} score records")
    print(f"  {len(aspect_set)} unique aspects scored")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
