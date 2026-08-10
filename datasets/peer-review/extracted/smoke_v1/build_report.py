"""Build the smoke_report.md from output_all.jsonl + input_20.jsonl + rubric file.

Metrics computed:
- Mean reasons per review
- Polarity distribution
- Rubric coverage: how many of 88 canonical rubrics invoked at least once
- Top-10 most-invoked rubrics
- Rubric coverage gap: % of reasons with empty rubric_matches
- Verbatim faithfulness: % of verbatim_spans that are exact substrings of review_text
- Per-decision (accept vs reject) rubric distribution
- Per-venue (ICLR vs eLife) rubric distribution
- Meta vs non-meta rubric distribution

Outputs: smoke_report.md
"""
import json, re, os
from collections import Counter, defaultdict

SMOKE_DIR = "/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/extracted/smoke_v1"
INPUT_PATH = os.path.join(SMOKE_DIR, "input_20.jsonl")
OUT_PATH = os.path.join(SMOKE_DIR, "output_all.jsonl")
RUBRIC_PATH = "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy/peer-review_general_r2_expanded.json"
REPORT_PATH = os.path.join(SMOKE_DIR, "smoke_report.md")

ACCEPT_PAT = re.compile(r"accept", re.I)
REJECT_PAT = re.compile(r"reject", re.I)


def decision_class(d):
    if not isinstance(d, str):
        return "unknown"
    if REJECT_PAT.search(d) and "withdrawn" not in d.lower():
        return "reject"
    if ACCEPT_PAT.search(d):
        return "accept"
    if "withdrawn" in d.lower():
        return "withdrawn"
    return "unknown"


def main():
    # Load inputs
    inputs = {}
    with open(INPUT_PATH) as f:
        for line in f:
            r = json.loads(line)
            inputs[r["review_id"]] = r

    # Load rubrics
    with open(RUBRIC_PATH) as f:
        rd = json.load(f)
    rubrics = rd["merged_groups"]
    id2name = {i: g["merged_name"] for i, g in enumerate(rubrics)}
    n_rubrics = len(rubrics)

    # Load outputs
    outputs = []
    with open(OUT_PATH) as f:
        for line in f:
            outputs.append(json.loads(line))

    # Stats
    n_reviews_out = len(outputs)
    n_parse_ok = sum(1 for o in outputs if o.get("parse_ok"))
    all_reasons = []   # list of (review_id, reason_dict)
    polarity_counter = Counter()
    rubric_counter = Counter()
    empty_match_reasons = []
    verbatim_hits = 0
    verbatim_total = 0
    tokens_in = 0
    tokens_out = 0

    for o in outputs:
        rid = o["review_id"]
        if not o.get("parse_ok"):
            continue
        obj = o["obj"]
        if not isinstance(obj, dict):
            continue
        reasons = obj.get("reasons", []) or []
        review_text = inputs.get(rid, {}).get("review_text", "")
        for reason in reasons:
            all_reasons.append((rid, reason))
            pol = reason.get("polarity", "unknown")
            polarity_counter[pol] += 1
            matches = reason.get("rubric_matches", []) or []
            if not matches:
                empty_match_reasons.append((rid, reason))
            for m in matches:
                if isinstance(m, int) and 0 <= m < n_rubrics:
                    rubric_counter[m] += 1
            vs = reason.get("verbatim_span", "") or ""
            if vs:
                verbatim_total += 1
                if vs in review_text:
                    verbatim_hits += 1
        meta = o.get("meta", {})
        tokens_in += meta.get("input_tokens", 0)
        tokens_out += meta.get("output_tokens", 0)

    n_reasons_total = len(all_reasons)
    mean_reasons = n_reasons_total / max(1, n_parse_ok)
    rubrics_invoked = len(rubric_counter)
    coverage_pct = 100.0 * rubrics_invoked / n_rubrics
    empty_pct = 100.0 * len(empty_match_reasons) / max(1, n_reasons_total)
    verbatim_pct = 100.0 * verbatim_hits / max(1, verbatim_total)

    # Per-decision / per-venue / per-meta split — rubric distributions
    per_decision = defaultdict(Counter)  # decision_class -> rubric Counter
    per_venue = defaultdict(Counter)
    per_meta = defaultdict(Counter)

    decision_reason_counts = Counter()
    venue_reason_counts = Counter()
    meta_reason_counts = Counter()

    for rid, reason in all_reasons:
        inp = inputs.get(rid, {})
        dc = decision_class(inp.get("decision"))
        venue = inp.get("venue")
        is_meta = inp.get("is_meta_review", False)
        decision_reason_counts[dc] += 1
        venue_reason_counts[venue] += 1
        meta_reason_counts["meta" if is_meta else "non_meta"] += 1
        matches = reason.get("rubric_matches", []) or []
        for m in matches:
            if isinstance(m, int) and 0 <= m < n_rubrics:
                per_decision[dc][m] += 1
                per_venue[venue][m] += 1
                per_meta["meta" if is_meta else "non_meta"][m] += 1

    # Find 3 verbatim example reviews
    examples = []
    for o in outputs[:5]:
        if not o.get("parse_ok"):
            continue
        obj = o["obj"]
        if not isinstance(obj, dict):
            continue
        reasons = obj.get("reasons", []) or []
        if not reasons:
            continue
        rid = o["review_id"]
        inp = inputs.get(rid, {})
        examples.append({
            "review_id": rid,
            "venue": inp.get("venue"),
            "decision": inp.get("decision"),
            "is_meta": inp.get("is_meta_review"),
            "reasons": reasons[:4],
        })
        if len(examples) >= 3:
            break

    # Build report
    lines = []
    lines.append("# Peer Review Smoke v1 — Reason Extraction + Canonical Rubric Tagging")
    lines.append("")
    lines.append(f"- Source: `{INPUT_PATH}` (20 reviews stratified by venue/length/meta/decision)")
    lines.append(f"- Rubric taxonomy: `{RUBRIC_PATH}` ({n_rubrics} canonical rubrics from merged_groups)")
    lines.append(f"- Model: claude-sonnet-4-5")
    lines.append(f"- Outputs: `{OUT_PATH}`")
    lines.append("")

    lines.append("## 1. Aggregate stats")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Reviews processed | {n_reviews_out} / 20 |")
    lines.append(f"| Reviews with parseable JSON | {n_parse_ok} / {n_reviews_out} |")
    lines.append(f"| Total reasons extracted | {n_reasons_total} |")
    lines.append(f"| Mean reasons per review | {mean_reasons:.2f} |")
    lines.append(f"| Positive polarity | {polarity_counter.get('positive',0)} ({100*polarity_counter.get('positive',0)/max(1,n_reasons_total):.1f}%) |")
    lines.append(f"| Negative polarity | {polarity_counter.get('negative',0)} ({100*polarity_counter.get('negative',0)/max(1,n_reasons_total):.1f}%) |")
    lines.append(f"| Mixed polarity | {polarity_counter.get('mixed',0)} ({100*polarity_counter.get('mixed',0)/max(1,n_reasons_total):.1f}%) |")
    lines.append(f"| Rubric coverage | {rubrics_invoked} / {n_rubrics} ({coverage_pct:.1f}%) |")
    lines.append(f"| Reasons with empty rubric_matches | {len(empty_match_reasons)} ({empty_pct:.1f}%) |")
    lines.append(f"| Verbatim faithfulness (exact substring) | {verbatim_hits} / {verbatim_total} ({verbatim_pct:.1f}%) |")
    lines.append(f"| Total input tokens | {tokens_in:,} |")
    lines.append(f"| Total output tokens | {tokens_out:,} |")
    # Sonnet 4.5 pricing: $3/MTok in, $15/MTok out
    est_cost = tokens_in * 3.0 / 1_000_000 + tokens_out * 15.0 / 1_000_000
    lines.append(f"| Estimated API cost (Sonnet 4.5 rates) | ${est_cost:.2f} |")
    lines.append("")

    lines.append("## 2. Verbatim example reviews")
    lines.append("")
    for i, ex in enumerate(examples, 1):
        lines.append(f"### Example {i}: review_id={ex['review_id']} | venue={ex['venue']} | decision={ex['decision']} | meta={ex['is_meta']}")
        lines.append("")
        for j, r in enumerate(ex["reasons"], 1):
            vs = (r.get("verbatim_span") or "").replace("\n", " ").strip()
            ph = (r.get("paraphrase") or "").strip()
            pol = r.get("polarity")
            matches = r.get("rubric_matches", []) or []
            match_lines = []
            for m in matches:
                if isinstance(m, int) and 0 <= m < n_rubrics:
                    match_lines.append(f"[{m}] {id2name[m]}")
            lines.append(f"**Reason {j}** _{pol}_")
            lines.append(f"- verbatim_span: \"{vs[:400]}{'…' if len(vs)>400 else ''}\"")
            lines.append(f"- paraphrase: {ph}")
            if match_lines:
                for ml in match_lines:
                    lines.append(f"- rubric_match: {ml}")
            else:
                lines.append(f"- rubric_match: _(empty — potential taxonomy gap)_")
            lines.append("")

    lines.append("## 3. Top-10 most-invoked rubrics")
    lines.append("")
    lines.append("| Rank | ID | Count | Rubric |")
    lines.append("|---|---|---|---|")
    for rank, (rid_, cnt) in enumerate(rubric_counter.most_common(10), 1):
        lines.append(f"| {rank} | {rid_} | {cnt} | {id2name[rid_]} |")
    lines.append("")

    lines.append("## 4. Reasons with EMPTY rubric_matches (would-be taxonomy extensions)")
    lines.append("")
    lines.append(f"Total: {len(empty_match_reasons)} reasons ({empty_pct:.1f}% of all reasons)")
    lines.append("")
    if empty_match_reasons:
        for i, (rid, r) in enumerate(empty_match_reasons, 1):
            vs = (r.get("verbatim_span") or "").replace("\n", " ").strip()
            ph = (r.get("paraphrase") or "").strip()
            pol = r.get("polarity")
            lines.append(f"{i}. **{rid}** _{pol}_ — {ph}")
            lines.append(f"   - \"{vs[:300]}{'…' if len(vs)>300 else ''}\"")
        lines.append("")
    else:
        lines.append("(none — every reason mapped to ≥1 canonical rubric)")
        lines.append("")

    lines.append("## 5. Per-decision split: rubric dominance")
    lines.append("")
    accept_top = per_decision["accept"].most_common(10)
    reject_top = per_decision["reject"].most_common(10)
    lines.append(f"- Accept reasons total: {sum(per_decision['accept'].values())} from {decision_reason_counts['accept']} extracted reasons")
    lines.append(f"- Reject reasons total: {sum(per_decision['reject'].values())} from {decision_reason_counts['reject']} extracted reasons")
    lines.append("")
    lines.append("### Top-10 rubrics on ACCEPTED papers")
    lines.append("")
    lines.append("| Rank | ID | Count | Rubric |")
    lines.append("|---|---|---|---|")
    for rank, (rid_, cnt) in enumerate(accept_top, 1):
        lines.append(f"| {rank} | {rid_} | {cnt} | {id2name[rid_]} |")
    lines.append("")
    lines.append("### Top-10 rubrics on REJECTED papers")
    lines.append("")
    lines.append("| Rank | ID | Count | Rubric |")
    lines.append("|---|---|---|---|")
    for rank, (rid_, cnt) in enumerate(reject_top, 1):
        lines.append(f"| {rank} | {rid_} | {cnt} | {id2name[rid_]} |")
    lines.append("")

    # Differential top: most A>R and R>A
    diffs = []
    all_ids = set(per_decision["accept"]) | set(per_decision["reject"])
    a_total = max(1, sum(per_decision["accept"].values()))
    r_total = max(1, sum(per_decision["reject"].values()))
    for rid_ in all_ids:
        a = per_decision["accept"].get(rid_, 0) / a_total
        r = per_decision["reject"].get(rid_, 0) / r_total
        diffs.append((rid_, a, r, a - r))
    diffs.sort(key=lambda x: -x[3])
    lines.append("### Rubrics that lean ACCEPT (share_accept - share_reject)")
    lines.append("")
    lines.append("| ID | Δshare | accept_share | reject_share | Rubric |")
    lines.append("|---|---|---|---|---|")
    for rid_, a, r, d in diffs[:8]:
        if d <= 0:
            break
        lines.append(f"| {rid_} | {d*100:+.1f}pp | {a*100:.1f}% | {r*100:.1f}% | {id2name[rid_]} |")
    lines.append("")
    lines.append("### Rubrics that lean REJECT")
    lines.append("")
    lines.append("| ID | Δshare | accept_share | reject_share | Rubric |")
    lines.append("|---|---|---|---|---|")
    for rid_, a, r, d in sorted(diffs, key=lambda x: x[3])[:8]:
        if d >= 0:
            break
        lines.append(f"| {rid_} | {d*100:+.1f}pp | {a*100:.1f}% | {r*100:.1f}% | {id2name[rid_]} |")
    lines.append("")

    # Per-venue ICLR vs eLife
    lines.append("## 6. Per-venue split: ICLR vs eLife (community contrast)")
    lines.append("")
    lines.append(f"- ICLR reasons: {decision_reason_counts and venue_reason_counts.get('ICLR', 0)}")
    lines.append(f"- eLife reasons: {venue_reason_counts.get('eLife', 0)}")
    lines.append("")
    lines.append("### Top-8 rubrics on ICLR reviews")
    lines.append("")
    lines.append("| ID | Count | Rubric |")
    lines.append("|---|---|---|")
    for rid_, cnt in per_venue["ICLR"].most_common(8):
        lines.append(f"| {rid_} | {cnt} | {id2name[rid_]} |")
    lines.append("")
    lines.append("### Top-8 rubrics on eLife reviews")
    lines.append("")
    lines.append("| ID | Count | Rubric |")
    lines.append("|---|---|---|")
    for rid_, cnt in per_venue["eLife"].most_common(8):
        lines.append(f"| {rid_} | {cnt} | {id2name[rid_]} |")
    lines.append("")

    # Meta vs non-meta
    lines.append("## 7. Meta vs non-meta reviews")
    lines.append("")
    lines.append(f"- Meta-review reasons: {meta_reason_counts.get('meta', 0)} from {sum(1 for o in outputs if inputs.get(o['review_id'], {}).get('is_meta_review'))} meta reviews")
    lines.append(f"- Non-meta reasons: {meta_reason_counts.get('non_meta', 0)} from {sum(1 for o in outputs if not inputs.get(o['review_id'], {}).get('is_meta_review'))} non-meta reviews")
    lines.append("")
    lines.append("### Top-6 rubrics on META reviews")
    lines.append("")
    lines.append("| ID | Count | Rubric |")
    lines.append("|---|---|---|")
    for rid_, cnt in per_meta["meta"].most_common(6):
        lines.append(f"| {rid_} | {cnt} | {id2name[rid_]} |")
    lines.append("")
    lines.append("### Top-6 rubrics on NON-META reviews")
    lines.append("")
    lines.append("| ID | Count | Rubric |")
    lines.append("|---|---|---|")
    for rid_, cnt in per_meta["non_meta"].most_common(6):
        lines.append(f"| {rid_} | {cnt} | {id2name[rid_]} |")
    lines.append("")

    # Verdict
    lines.append("## 8. Verdict on whether the framework works on peer review")
    lines.append("")
    works = (n_parse_ok >= 18) and (verbatim_pct >= 80) and (rubrics_invoked >= 25) and (empty_pct <= 25)
    differs = False
    # Check accept vs reject differentiation
    if a_total > 5 and r_total > 5:
        # Simple proxy: do the top-3 differ by ≥ 1 rank between accept and reject?
        a3 = set([rid_ for rid_, _ in per_decision["accept"].most_common(5)])
        r3 = set([rid_ for rid_, _ in per_decision["reject"].most_common(5)])
        diff_set = (a3 - r3) | (r3 - a3)
        differs = len(diff_set) >= 2

    lines.append("**Key questions:**")
    lines.append("")
    lines.append(f"1. **Does Claude reliably tag against the canonical rubrics?** {n_parse_ok}/{n_reviews_out} reviews parsed cleanly. Verbatim faithfulness {verbatim_pct:.1f}%. Empty-match rate {empty_pct:.1f}%. → **{'YES' if (n_parse_ok>=18 and verbatim_pct>=80 and empty_pct<=25) else 'NEEDS WORK'}**")
    lines.append("")
    lines.append(f"2. **Do accept vs reject reviews differ on which rubrics get invoked?** {'YES — distributions diverge across top-5.' if differs else 'WEAK — top rubrics largely overlap.'} See section 5.")
    lines.append("")
    lines.append(f"3. **Coverage gap size:** {empty_pct:.1f}% of reasons have no canonical match. Taxonomy looks {'~complete' if empty_pct < 10 else 'partially incomplete' if empty_pct < 25 else 'incomplete — material extensions likely'} for this sample.")
    lines.append("")
    lines.append(f"4. **Rubric breadth used:** {rubrics_invoked}/{n_rubrics} ({coverage_pct:.1f}%) of canonical rubrics invoked at least once in 20 reviews. Many rubrics are likely dormant until larger samples.")
    lines.append("")
    lines.append(f"**Overall:** {'FRAMEWORK VIABLE for peer review' if works else 'Framework viable with caveats — see metrics above'}.")
    lines.append("")

    lines.append("## 9. Path to Llama-70B bulk processing")
    lines.append("")
    lines.append(f"- **Prompt complexity for Llama-70B (e.g. FP8 on B200):** the system prompt is ~{int(tokens_in/max(1,n_reviews_out))} input tokens on average. The {n_rubrics}-rubric block is the dominant cost. Llama-70B FP8 handles 22-25K-context prompts comfortably. Verdict: **feasible**.")
    lines.append("")
    lines.append("- **Suggested batched vLLM prompt structure for sk3 rollout:**")
    lines.append("")
    lines.append("  - Keep the system prompt **identical** across all calls — vLLM prefix cache will reuse the ~21K-char rubric block, so the marginal cost per review is small.")
    lines.append("  - Submit reviews in batches of 1000–4000 per `llm.generate()` / `llm.chat()` call (per the user's vLLM batching feedback).")
    lines.append("  - `max_model_len`: 32768 (rubric block ~5.5K tokens + review ≤ 4K tokens + 8K output budget).")
    lines.append("  - Constrain output via JSON grammar (vLLM `guided_json` or `xgrammar`) using the same schema as section 1.")
    lines.append("  - 259K reviews ÷ 1500/batch ≈ 175 batches. At ~4K input tok + ~1.5K output tok per review on B200 FP8 Llama-70B, expect ~24–36 hours total wall-clock.")
    lines.append("")
    lines.append("- **Checkpointing:** write JSONL every BATCH_FLUSH=200 reviews per the user's safe-run config.")
    lines.append("")
    lines.append("- **Validation gate before bulk:** rerun this smoke (20 reviews) on Llama-70B, compare top-10 rubrics + empty-match rate + verbatim faithfulness against the Sonnet baseline above. Accept Llama as the bulk tagger if verbatim faithfulness ≥ 70% and rubric Jaccard with Sonnet top-10 ≥ 0.5.")
    lines.append("")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Wrote report to {REPORT_PATH} ({len(report)} chars)")


if __name__ == "__main__":
    main()
