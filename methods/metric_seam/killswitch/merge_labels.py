"""Merge per-chunk Claude labels into claude_labels.json + sanity summary."""
import collections, json, pathlib

OUT = pathlib.Path(__file__).resolve().parents[3] / "outputs/metric_seam_pilot/killswitch"

merged = {}
for f in sorted((OUT / "claude_labels").glob("chunk_*_labels.json")):
    part = json.load(open(f))
    dup = set(part) & set(merged)
    if dup:
        print(f"{f.name}: {len(dup)} duplicate ids (overwriting)")
    merged.update(part)

bad = {d: v for d, v in merged.items()
       if not (isinstance(v.get("n_quoted_speakers"), int)
               and isinstance(v.get("authenticity_0_10"), int)
               and 0 <= v["authenticity_0_10"] <= 10 and v["n_quoted_speakers"] >= 0)}
if bad:
    print(f"MALFORMED entries: {len(bad)} -> {list(bad)[:5]}")
json.dump(merged, open(OUT / "claude_labels.json", "w"), indent=0)

spk = collections.Counter(v["n_quoted_speakers"] for v in merged.values())
auth = collections.Counter(v["authenticity_0_10"] for v in merged.values())
print(f"{len(merged)} labeled; speakers dist {dict(sorted(spk.items()))}")
print(f"authenticity dist {dict(sorted(auth.items()))}")
