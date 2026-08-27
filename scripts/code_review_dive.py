"""Survey code_review aspects + sample artifacts + cells coverage."""
import json
import pandas as pd
from collections import Counter
from pathlib import Path

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")

arr = json.load(open(REPO / "runs/validity_full/v2/code_review/aspects.json"))
print("total aspects:", len(arr))


def categorize(name, desc):
    t = (name + " " + desc).lower()
    if any(w in t for w in [
        "dry", "kiss", "yagni", "control flow", "cohesion",
        "naming", "indent", "lint", "style", "format",
        "small function", "modular",
    ]):
        return "style/structure"
    if any(w in t for w in ["test", "coverage", "ci", "build", "assert"]):
        return "testing/CI"
    if any(w in t for w in [
        "security", "vulnerab", "auth", "permission", "secret",
    ]):
        return "security"
    if any(w in t for w in [
        "performance", "perf", "memory", "leak", "concurren", "thread", "race",
    ]):
        return "performance"
    if any(w in t for w in [
        "api", "interface", "contract", "deprec", "breaking change", "backward",
    ]):
        return "api/contract"
    if any(w in t for w in ["doc", "comment", "readme", "license", "changelog"]):
        return "docs"
    if any(w in t for w in [
        "communication", "civil", "feedback", "respectful", "tone",
    ]):
        return "review-process"
    if any(w in t for w in [
        "pull request", "title", "description", "rationale", "scope",
        "atomic", "incremental",
    ]):
        return "pr-meta"
    if any(w in t for w in [
        "solv", "problem", "requirement", "use case", "goal",
        "necessity", "value",
    ]):
        return "problem-fit"
    return "other"


buckets = Counter()
for a in arr:
    buckets[categorize(a.get("name", ""), a.get("description", ""))] += 1
print("\nbucket distribution:")
for b, n in buckets.most_common():
    print(f"  {b:<22} {n}")

print("\nexamples per bucket:")
seen_per_bucket = Counter()
for a in arr:
    bk = categorize(a.get("name", ""), a.get("description", ""))
    if seen_per_bucket[bk] < 3:
        seen_per_bucket[bk] += 1
        print(f"  [{bk:<15}] {a['aspect_id']:<5} {a['name'][:65]}")

dps = json.load(open(REPO / "runs/validity_full/v2/code_review/datapoints.json"))
print(f"\ndatapoints: {len(dps)}")
y_counts = Counter(d.get("judgement") for d in dps)
print(f"label distribution: {dict(y_counts)}")

print("\n=== sample datapoints (showing artifact format) ===")
for d in dps[:3]:
    print(f"--- dp={d['datapoint_id']} y={d.get('judgement')} ---")
    txt = d.get("text", "")
    print(f"text length: {len(txt)} chars")
    print("first 800 chars:")
    print(txt[:800])
    print()

# cells
cells = pd.read_parquet(
    REPO / "outputs/v2_db/cells_v1/task=code_review/judge=qwen_thinking_fp8/data.parquet")
print(f"cells: {len(cells):,}, dps: {cells['datapoint_id'].nunique()}, "
      f"aspects: {cells['aspect_id'].nunique()}")
print(f"applicability rate: {cells['applicable'].mean() * 100:.1f}%")
