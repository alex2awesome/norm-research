"""V2 corpus quality + coverage assessment.

Tasks 1-3 from the user's request. Produces:
- basic stats
- coverage vs RTC parquet
- 30-doc stratified sample with verbatim (comment, response) pairs

Writes a markdown report and prints key tables.
"""
from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

random.seed(42)

V2_PATH = Path(
    "/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v2_existing/comment_responses_V2.jsonl"
)
RTC_PARQUET = Path(
    "/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/rtc_extracted/rtc_sections.parquet"
)
REPORT_PATH = Path(
    "/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v2_existing/quality_assessment.md"
)


def iter_v2():
    with V2_PATH.open() as f:
        for line in f:
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Task 1: basic stats + first-pass index
# ---------------------------------------------------------------------------
print("=" * 60)
print("TASK 1: V2 basic stats")
print("=" * 60)

# Index in one pass: document_id -> (agency_id, n_responses, line_offset, doc_type)
# We'll also keep n_responses lists, agency counters, and pull all docs with
# responses into memory by line index so we can subsample later.
n_total = 0
n_with_responses = 0
n_responses_dist = []
per_agency_doc_count = Counter()
per_agency_resp_counts = defaultdict(list)
doc_index = {}  # document_id -> dict(agency_id, n_responses, doc_type, posted_date)
docs_with_resp_by_agency = defaultdict(list)  # agency -> list of (doc_idx_in_file, document_id)

total_pairs = 0
for line_idx, d in enumerate(iter_v2()):
    n_total += 1
    doc_id = d["document_id"]
    agency = d.get("agency_id", "UNK")
    nr = d.get("n_responses", 0)
    per_agency_doc_count[agency] += 1
    per_agency_resp_counts[agency].append(nr)
    doc_index[doc_id] = {
        "agency": agency,
        "n_responses": nr,
        "doc_type": d.get("doc_type"),
        "posted_date": d.get("posted_date"),
        "line_idx": line_idx,
    }
    if nr > 0:
        n_with_responses += 1
        n_responses_dist.append(nr)
        total_pairs += nr
        docs_with_resp_by_agency[agency].append((line_idx, doc_id))

print(f"Total docs: {n_total:,}")
print(f"Docs with n_responses > 0: {n_with_responses:,} ({n_with_responses / n_total * 100:.1f}%)")
print(f"Total (comment, response) pairs: {total_pairs:,}")
if n_responses_dist:
    s = sorted(n_responses_dist)
    print(
        f"n_responses on responsive docs — median: {statistics.median(s)} | "
        f"mean: {statistics.mean(s):.1f} | p90: {s[int(0.9 * len(s))]} | max: {max(s)}"
    )

print("\nTop 15 agencies by doc count:")
print(f"{'agency':<8} {'docs':>8} {'docs_w_resp':>12} {'med_n_resp':>11} {'mean_n_resp':>12}")
top15_agencies = [a for a, _ in per_agency_doc_count.most_common(15)]
for a in top15_agencies:
    counts = per_agency_resp_counts[a]
    resp_counts = [c for c in counts if c > 0]
    med = statistics.median(resp_counts) if resp_counts else 0
    mean = statistics.mean(resp_counts) if resp_counts else 0.0
    print(
        f"{a:<8} {per_agency_doc_count[a]:>8,} {len(resp_counts):>12,} "
        f"{med:>11} {mean:>12.1f}"
    )

# ---------------------------------------------------------------------------
# Task 2: coverage vs RTC parquet
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TASK 2: Coverage vs RTC parquet")
print("=" * 60)

rtc = pd.read_parquet(RTC_PARQUET)
print(f"RTC parquet docs: {len(rtc):,}")

rtc_ids = rtc["rule_id"].astype(str).tolist()
rtc_agency = dict(zip(rtc_ids, rtc["agency"].astype(str)))
rtc_text = dict(zip(rtc_ids, rtc["rtc_text"].astype(str)))

in_v2 = [r for r in rtc_ids if r in doc_index]
in_v2_with_resp = [r for r in in_v2 if doc_index[r]["n_responses"] > 0]
not_in_v2 = [r for r in rtc_ids if r not in doc_index]

print(f"RTC docs present in V2:                {len(in_v2):,} / {len(rtc_ids):,} ({len(in_v2)/len(rtc_ids)*100:.1f}%)")
print(f"  of which V2 has n_responses > 0:    {len(in_v2_with_resp):,} ({len(in_v2_with_resp)/len(rtc_ids)*100:.1f}% of RTC total)")
print(f"RTC docs MISSING from V2:              {len(not_in_v2):,}")

# Per-agency overlap rates (top 10 agencies in RTC)
print("\nPer-agency overlap (top 10 RTC agencies):")
print(f"{'agency':<8} {'rtc_n':>6} {'in_v2':>6} {'%':>6} {'v2_w_resp':>10} {'%_w_resp':>9}")
rtc_agency_counts = Counter(rtc["agency"].astype(str).tolist())
for ag, cnt in rtc_agency_counts.most_common(10):
    ag_ids = [r for r in rtc_ids if rtc_agency[r] == ag]
    ag_in = [r for r in ag_ids if r in doc_index]
    ag_in_resp = [r for r in ag_in if doc_index[r]["n_responses"] > 0]
    p1 = len(ag_in) / len(ag_ids) * 100 if ag_ids else 0
    p2 = len(ag_in_resp) / len(ag_ids) * 100 if ag_ids else 0
    print(f"{ag:<8} {len(ag_ids):>6} {len(ag_in):>6} {p1:>5.1f}% {len(ag_in_resp):>10} {p2:>8.1f}%")

# 5 examples missed by V2
print("\n--- 5 RTC docs MISSED by V2 (with RTC excerpt) ---")
sample_missed = random.sample(not_in_v2, min(5, len(not_in_v2)))
missed_examples = []
for rid in sample_missed:
    excerpt = rtc_text[rid][:400].replace("\n", " ")
    print(f"\n{rid} (agency={rtc_agency[rid]})")
    print(f"  RTC excerpt: {excerpt}...")
    missed_examples.append({"rule_id": rid, "agency": rtc_agency[rid], "excerpt": excerpt})

# 5 V2 docs that the RTC parquet missed entirely (V2 had responses, not in my parquet)
rtc_set = set(rtc_ids)
v2_only_with_resp = [
    did for did, meta in doc_index.items()
    if meta["n_responses"] > 0 and did not in rtc_set
]
print(f"\nV2 docs with responses NOT in my RTC parquet: {len(v2_only_with_resp):,}")
print("\n--- 5 V2-extracted docs MISSED by my RTC parquet ---")
v2_only_sample = random.sample(v2_only_with_resp, min(5, len(v2_only_with_resp)))


def get_doc_by_line(target_line):
    with V2_PATH.open() as f:
        for li, line in enumerate(f):
            if li == target_line:
                return json.loads(line)
    return None


def get_docs_by_lines(target_lines):
    """Batch fetch docs by line offsets — one pass over file."""
    target_set = set(target_lines)
    out = {}
    with V2_PATH.open() as f:
        for li, line in enumerate(f):
            if li in target_set:
                out[li] = json.loads(line)
                if len(out) == len(target_set):
                    break
    return out


v2_only_lines = [doc_index[did]["line_idx"] for did in v2_only_sample]
v2_only_docs = get_docs_by_lines(v2_only_lines)
v2_only_examples = []
for did in v2_only_sample:
    li = doc_index[did]["line_idx"]
    d = v2_only_docs[li]
    print(f"\n{did} (agency={d['agency_id']}, n_resp={d['n_responses']})")
    r = d["responses"][0]
    comment = r.get("content_of_comment", "")[:300]
    resp = r.get("response_to_comment", "")[:300]
    print(f"  COMMENT: {comment}")
    print(f"  RESPONSE: {resp}")
    v2_only_examples.append({
        "doc_id": did, "agency": d["agency_id"],
        "n_responses": d["n_responses"], "comment": comment, "response": resp,
    })


# ---------------------------------------------------------------------------
# Task 3: norm-content sample (30 docs, stratified across top 6 agencies)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TASK 3: 30-sample classification")
print("=" * 60)

# Top 6 agencies by # docs with n_responses >= 2
agency_resp2 = Counter()
for ag, lst in docs_with_resp_by_agency.items():
    n_ge2 = sum(
        1 for li, did in lst if doc_index[did]["n_responses"] >= 2
    )
    if n_ge2:
        agency_resp2[ag] = n_ge2

top6 = [a for a, _ in agency_resp2.most_common(6)]
print(f"Top 6 agencies by # docs with n_responses >= 2: {top6}")

sample_lines = []
sample_meta = []  # (line_idx, doc_id, agency)
for ag in top6:
    pool = [
        (li, did) for li, did in docs_with_resp_by_agency[ag]
        if doc_index[did]["n_responses"] >= 2
    ]
    chosen = random.sample(pool, min(5, len(pool)))
    for li, did in chosen:
        sample_lines.append(li)
        sample_meta.append((li, did, ag))

print(f"Total sampled: {len(sample_lines)}")
sample_docs = get_docs_by_lines(sample_lines)


def classify(resp_text: str, comment_text: str) -> str:
    """Heuristic — not the final classification (user wants manual), but
    used for an initial print + an automated baseline distribution."""
    t = resp_text.lower()
    # PERFUNCTORY signals
    perf_phrases = [
        "no changes were made", "finalized as proposed", "no change",
        "comment was noted", "comment was acknowledged",
        "agency acknowledged", "we acknowledge", "thank the commenter",
        "appreciate the comment", "support for the proposal",
    ]
    # NORMATIVE-RICH signals
    norm_phrases = [
        "section ", "§", "u.s.c", "statute", "required by", "required under",
        "consistent with", "inconsistent with", "balance", "weighed",
        "best available", "we reject", "we disagree", "we decline",
        "policy", "rationale", "principle", "authority", "must ",
        "consistent with the", "we believe", "we conclude", "in light of",
    ]
    # SUBSTANTIVE-DRY signals
    dry_phrases = [
        "we updated", "we revised", "we clarified", "we have clarified",
        "we modified", "we changed", "we added", "we removed",
        "we corrected", "table has been updated",
    ]
    n_norm = sum(1 for p in norm_phrases if p in t)
    n_dry = sum(1 for p in dry_phrases if p in t)
    n_perf = sum(1 for p in perf_phrases if p in t)
    # Short responses bias to perfunctory
    short = len(resp_text) < 200
    if short and n_norm == 0:
        return "PERFUNCTORY"
    if n_norm >= 2 and len(resp_text) > 300:
        return "NORMATIVE-RICH"
    if n_norm >= 1 and n_dry == 0 and n_perf == 0 and len(resp_text) > 250:
        return "NORMATIVE-RICH"
    if n_dry >= 1 and n_norm == 0:
        return "SUBSTANTIVE-DRY"
    if n_perf >= 1 and n_norm == 0:
        return "PERFUNCTORY"
    return "SUBSTANTIVE-DRY"


sample_rows = []
class_counts = Counter()
class_by_agency = defaultdict(Counter)
norm_rich_examples = []
perfunctory_examples = []
dry_examples = []

for li, did, ag in sample_meta:
    d = sample_docs[li]
    print(f"\n--- {ag} | {did} | {d['posted_date']} | n_resp={d['n_responses']} ---")
    # Take first response that has both comment and response
    chosen_pair = None
    for r in d["responses"]:
        if isinstance(r, dict) and r.get("response_to_comment") and r.get("content_of_comment"):
            chosen_pair = r
            break
    if chosen_pair is None:
        continue
    comment = chosen_pair.get("content_of_comment", "")
    response = chosen_pair.get("response_to_comment", "")
    label = classify(response, comment)
    class_counts[label] += 1
    class_by_agency[ag][label] += 1
    print(f"  [{label}]")
    print(f"  COMMENT: {comment[:300]}")
    print(f"  RESPONSE: {response[:500]}")
    row = {
        "agency": ag, "doc_id": did, "posted_date": d.get("posted_date"),
        "n_responses": d["n_responses"], "label": label,
        "comment": comment, "response": response,
    }
    sample_rows.append(row)
    if label == "NORMATIVE-RICH":
        norm_rich_examples.append(row)
    elif label == "PERFUNCTORY":
        perfunctory_examples.append(row)
    else:
        dry_examples.append(row)

print()
print("=" * 60)
print("Classification distribution (heuristic)")
print("=" * 60)
total_lbl = sum(class_counts.values())
for lbl in ["NORMATIVE-RICH", "SUBSTANTIVE-DRY", "PERFUNCTORY"]:
    c = class_counts.get(lbl, 0)
    pct = c / total_lbl * 100 if total_lbl else 0
    print(f"  {lbl}: {c} ({pct:.1f}%)")

print("\nBy agency:")
print(f"{'agency':<8} {'NORM':>6} {'DRY':>6} {'PERF':>6}")
for ag in top6:
    c = class_by_agency[ag]
    print(f"{ag:<8} {c['NORMATIVE-RICH']:>6} {c['SUBSTANTIVE-DRY']:>6} {c['PERFUNCTORY']:>6}")

# Save the sample to JSON for the markdown report
out_data = {
    "basic_stats": {
        "n_total": n_total,
        "n_with_responses": n_with_responses,
        "total_pairs": total_pairs,
        "median_n_resp": statistics.median(n_responses_dist) if n_responses_dist else 0,
        "mean_n_resp": statistics.mean(n_responses_dist) if n_responses_dist else 0,
        "p90_n_resp": sorted(n_responses_dist)[int(0.9 * len(n_responses_dist))] if n_responses_dist else 0,
        "max_n_resp": max(n_responses_dist) if n_responses_dist else 0,
        "top15_agencies": [
            {
                "agency": a, "docs": per_agency_doc_count[a],
                "docs_w_resp": sum(1 for c in per_agency_resp_counts[a] if c > 0),
                "med_n_resp": statistics.median([c for c in per_agency_resp_counts[a] if c > 0]) if any(c > 0 for c in per_agency_resp_counts[a]) else 0,
            }
            for a in top15_agencies
        ],
    },
    "coverage": {
        "rtc_n": len(rtc_ids),
        "in_v2": len(in_v2),
        "in_v2_with_resp": len(in_v2_with_resp),
        "not_in_v2": len(not_in_v2),
        "v2_only_with_resp": len(v2_only_with_resp),
        "per_agency": [
            {
                "agency": ag,
                "rtc_n": len([r for r in rtc_ids if rtc_agency[r] == ag]),
                "in_v2": len([r for r in rtc_ids if rtc_agency[r] == ag and r in doc_index]),
                "in_v2_with_resp": len([
                    r for r in rtc_ids
                    if rtc_agency[r] == ag and r in doc_index and doc_index[r]["n_responses"] > 0
                ]),
            }
            for ag, _ in rtc_agency_counts.most_common(10)
        ],
        "missed_examples": missed_examples,
        "v2_only_examples": v2_only_examples,
    },
    "sample": {
        "rows": sample_rows,
        "class_counts": dict(class_counts),
        "class_by_agency": {ag: dict(c) for ag, c in class_by_agency.items()},
    },
}

import json as _j
with open("/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v2_existing/analysis_output.json", "w") as f:
    _j.dump(out_data, f, indent=2, default=str)

print("\nSaved structured output to analysis_output.json")
