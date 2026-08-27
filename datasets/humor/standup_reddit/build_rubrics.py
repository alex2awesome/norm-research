"""Build the canonical rubric vocabulary for the humor task.

Pulls merged_groups from the two r2_expanded hierarchies:
  - humor_general_r2_expanded.json (broad craft/ethics dimensions)
  - humor_specific_r2_expanded.json (narrower technique-level)

We assign sequential rubric_ids and emit a JSONL row per rubric with the schema
expected by run_openrouter.py:
  {rubric_id, name, description, source, n_leaves, ...}

Outputs to datasets/humor/standup_reddit/rubrics.jsonl
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HIER = "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy"
SRC_FILES = [
    (os.path.join(HIER, "humor_general_r2_expanded.json"), "general_r2_expanded"),
    (os.path.join(HIER, "humor_specific_r2_expanded.json"), "specific_r2_expanded"),
]
OUT = os.path.join(HERE, "rubrics.jsonl")


def main():
    rows = []
    rid = 1
    seen_names = set()
    for fp, tag in SRC_FILES:
        d = json.load(open(fp))
        for g in d.get("merged_groups", []):
            name = g["merged_name"].strip()
            # Dedup by normalized name across the two files
            key = name.lower().replace("'", "").replace("'", "")
            if key in seen_names:
                continue
            seen_names.add(key)
            rows.append({
                "rubric_id": rid,
                "name": name,
                "description": g["merged_description"].strip(),
                "source": tag,
                "n_leaves": g.get("total_leaf_rubrics", 0),
                "source_cluster_ids": g.get("source_r2_cluster_ids", []),
            })
            rid += 1
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote {} rubrics → {}".format(len(rows), OUT))
    # Show category source distribution
    from collections import Counter
    c = Counter(r["source"] for r in rows)
    for k, v in c.items():
        print("  {}: {}".format(k, v))


if __name__ == "__main__":
    main()
