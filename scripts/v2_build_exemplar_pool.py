"""Build per-R2-aspect exemplar candidate pools from v1 judge scores.

For each of 218 R2 aspects, take v1 R2 judge scores, average across 5
paraphrases per datapoint, then dump:
  - top-15 highest-scoring (likely "satisfies")
  - mid-15 around 0.5 (likely "neutral / N/A")
  - bot-15 lowest-scoring (likely "violates")

These are CANDIDATES for the subagents to pick exemplars from.
Stored as `runs/validity_full/full_v2/exemplar_pool/<aspect_id>.json`.

Output schema per file:
  {"aspect_id": "a0", "name": "...", "description": "...",
   "candidates": {
     "satisfier": [{"datapoint_id": "d0042", "v1_judge_score": 0.95,
                    "venue": "iclr25", "label": 1, "text_preview": "first 600 chars"}, ...],
     "neutral":   [...],  # 15
     "violator":  [...]   # 15
   }}
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_score(p):
    raw = p.read_text().strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])


def main():
    v1 = Path("runs/validity_full/full_v1")
    v2 = Path("runs/validity_full/full_v2")
    out_dir = v2 / "exemplar_pool"

    aspects = json.loads((v1 / "r2_aspects.json").read_text())
    datapoints = json.loads((v1 / "datapoints.json").read_text())
    dp_by_id = {d["datapoint_id"]: d for d in datapoints}

    manifest = json.loads((v1 / "judge_manifest.json").read_text())
    rdir = v1 / "judge_responses_llama"

    # Aggregate v1 R2 judge: aspect_id -> dp_id -> [scores across paraphrases]
    print("aggregating v1 judge scores...")
    aspect_dp_scores = defaultdict(lambda: defaultdict(list))
    n_loaded = n_skip = 0
    for entry in manifest:
        rp = rdir / f"{entry['key']}.json"
        if not rp.exists():
            n_skip += 1
            continue
        try:
            obj = parse_score(rp)
            for sc in obj.get("scores", []):
                dp_id = sc.get("id"); s = sc.get("score")
                if dp_id is None or s is None: continue
                aspect_dp_scores[entry["aspect_id"]][dp_id].append(s / 10.0)
            n_loaded += 1
        except Exception:
            n_skip += 1

    print(f"  loaded {n_loaded} response files, skipped {n_skip}")
    print(f"  aspects with data: {len(aspect_dp_scores)}")

    # Per aspect, build candidate pool
    written = 0
    for asp in aspects:
        aid = asp["aspect_id"]
        dp_scores = aspect_dp_scores.get(aid, {})
        # Mean across paraphrases
        means = {dp: statistics.mean(vs) for dp, vs in dp_scores.items() if vs}
        if len(means) < 30:
            print(f"  WARN {aid} {asp['name'][:40]}: only {len(means)} dps with scores; skipping")
            continue
        # Sort
        sorted_dps = sorted(means.items(), key=lambda x: -x[1])
        # Top 15: highest scores
        sat = sorted_dps[:15]
        # Bottom 15: lowest
        vio = sorted_dps[-15:][::-1]  # reverse so most-violating first
        # Mid 15: closest to 0.5
        mid = sorted(means.items(), key=lambda x: abs(x[1] - 0.5))[:15]

        def make_record(dp_id, score):
            d = dp_by_id.get(dp_id, {})
            preview = (d.get("text", "") or "")[:600]
            return {
                "datapoint_id": dp_id,
                "v1_judge_score": round(score, 3),
                "venue": d.get("venue", ""),
                "label": d.get("judgement"),
                "text_preview": preview,
            }

        out = {
            "aspect_id": aid,
            "name": asp["name"],
            "description": asp["description"],
            "n_r1_used": asp["n_r1_used"],
            "candidates": {
                "satisfier": [make_record(dp, s) for dp, s in sat],
                "neutral":   [make_record(dp, s) for dp, s in mid],
                "violator":  [make_record(dp, s) for dp, s in vio],
            },
        }
        (out_dir / f"{aid}.json").write_text(json.dumps(out, indent=1))
        written += 1

    print(f"wrote {written} exemplar pool files to {out_dir}")


if __name__ == "__main__":
    main()
