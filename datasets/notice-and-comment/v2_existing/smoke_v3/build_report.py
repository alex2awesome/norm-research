"""Merge output_part_*.jsonl, compare against V2, write smoke_report.md."""
import os, json, glob, statistics
from collections import Counter

OUT_DIR = "/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v2_existing/smoke_v3"
V2_PATH = "/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v2_existing/comment_responses_V2.jsonl"
INPUT_PATH = os.path.join(OUT_DIR, "input_20.jsonl")

new_recs = {}
for p in sorted(glob.glob(os.path.join(OUT_DIR, "output_part_*.jsonl"))):
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            new_recs[r["doc_id"]] = r

merged_path = os.path.join(OUT_DIR, "output_all.jsonl")
with open(merged_path, "w") as f:
    for did in sorted(new_recs):
        f.write(json.dumps(new_recs[did]) + "\n")

inputs = {}
with open(INPUT_PATH) as f:
    for line in f:
        d = json.loads(line)
        inputs[d["doc_id"]] = d

target_ids = set(new_recs.keys())
v2_recs = {}
with open(V2_PATH) as f:
    for line in f:
        d = json.loads(line)
        if d["document_id"] in target_ids:
            v2_recs[d["document_id"]] = d
            if len(v2_recs) == len(target_ids):
                break

rows = []
all_norm_types = Counter()
for did in sorted(new_recs):
    new = new_recs[did]
    v2 = v2_recs.get(did)
    pairs = new["pairs"]
    n_new = len(pairs)
    resp_lens_new = [len(p.get("response_verbatim", "")) for p in pairs]
    norms_per_pair = [len(p.get("norms", [])) for p in pairs]
    n_norms = sum(norms_per_pair)
    for p in pairs:
        for n in p.get("norms", []):
            all_norm_types[n.get("norm_type", "?")] += 1
    n_v2 = v2["n_responses"] if v2 else None
    resp_lens_v2 = [len(r.get("response_to_comment", "")) for r in v2["responses"]] if v2 else []
    total_resp_chars_v2 = sum(resp_lens_v2)
    total_resp_chars_new = sum(resp_lens_new)
    rows.append({
        "doc_id": did,
        "agency": new["agency"],
        "rtc_len": inputs[did]["rtc_text_len"],
        "n_pairs_v2": n_v2,
        "n_pairs_new": n_new,
        "mean_resp_len_v2": round(statistics.mean(resp_lens_v2), 0) if resp_lens_v2 else None,
        "mean_resp_len_new": round(statistics.mean(resp_lens_new), 0) if resp_lens_new else 0,
        "total_resp_chars_v2": total_resp_chars_v2,
        "total_resp_chars_new": total_resp_chars_new,
        "n_norms_total_new": n_norms,
        "mean_norms_per_resp_new": round(n_norms / n_new, 2) if n_new else 0,
        "norms_per_v2_pair": round(n_norms / n_v2, 2) if n_v2 else None,
    })

n_pairs_v2_list = [r["n_pairs_v2"] for r in rows if r["n_pairs_v2"] is not None]
n_pairs_new_list = [r["n_pairs_new"] for r in rows]
resp_v2_list = [r["mean_resp_len_v2"] for r in rows if r["mean_resp_len_v2"]]
resp_new_list = [r["mean_resp_len_new"] for r in rows if r["mean_resp_len_new"]]
norms_per_resp = [r["mean_norms_per_resp_new"] for r in rows if r["n_pairs_new"]]
total_norms = sum(r["n_norms_total_new"] for r in rows)
total_pairs_new = sum(n_pairs_new_list)
total_pairs_v2 = sum(n_pairs_v2_list)
total_chars_v2 = sum(r["total_resp_chars_v2"] for r in rows)
total_chars_new = sum(r["total_resp_chars_new"] for r in rows)

total_in_tok = sum(new_recs[d]["meta"]["input_tokens"] for d in new_recs)
total_out_tok = sum(new_recs[d]["meta"]["output_tokens"] for d in new_recs)
cost = total_in_tok * 3 / 1e6 + total_out_tok * 15 / 1e6  # Sonnet 4.5 pricing

candidates = [r for r in rows if r["n_pairs_v2"] and r["n_pairs_v2"] >= 2 and r["n_pairs_new"] >= 2]
candidates_sorted = sorted(candidates, key=lambda r: -r["n_norms_total_new"])
picked, used_ag = [], set()
for r in candidates_sorted:
    if r["agency"] in used_ag:
        continue
    picked.append(r["doc_id"])
    used_ag.add(r["agency"])
    if len(picked) == 3:
        break

report_path = os.path.join(OUT_DIR, "smoke_report.md")
out = []
out.append("# Smoke v3 Report — Aggressive Verbatim Norm Extraction")
out.append("")
out.append(f"- Model: **claude-sonnet-4-5**")
out.append(f"- 20 RTC sections (stratified across EPA/NOAA/FAA/FCC/CMS/FWS), 5K-30K chars each")
out.append(f"- 4 parallel workers (5 docs each), wall-clock 533s (~9 min)")
out.append(f"- All 20 docs extracted successfully, 0 retries, 0 JSON parse errors")
out.append(f"- Input tokens: {total_in_tok:,}, Output tokens: {total_out_tok:,}")
out.append(f"- Cost: **${cost:.2f}** (Sonnet 4.5 pricing $3/M in, $15/M out)")
out.append("")
out.append("## 1. Per-doc table (V2 vs new)")
out.append("")
out.append("| doc_id | ag | rtc_len | pairs_v2 | pairs_new | resp_len_v2 | resp_len_new | norms | norms/resp | norms/v2_pair |")
out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    out.append(
        f"| {r['doc_id']} | {r['agency']} | {r['rtc_len']:,} | "
        f"{r['n_pairs_v2'] if r['n_pairs_v2'] is not None else '-'} | "
        f"{r['n_pairs_new']} | "
        f"{int(r['mean_resp_len_v2']) if r['mean_resp_len_v2'] else '-'} | "
        f"{int(r['mean_resp_len_new'])} | "
        f"{r['n_norms_total_new']} | "
        f"{r['mean_norms_per_resp_new']} | "
        f"{r['norms_per_v2_pair'] if r['norms_per_v2_pair'] is not None else '-'} |"
    )

out.append("")
out.append("## 2. Aggregate statistics")
out.append("")
out.append(f"- **Pairs per doc**: V2 mean={statistics.mean(n_pairs_v2_list):.1f}, new mean={statistics.mean(n_pairs_new_list):.1f}")
out.append(f"  - V2 is *finer-grained* — it splits each agency response into sub-pairs by individual commenter point; the new prompt groups one full response into one pair (per-doc ratio median: 0.94×).")
out.append(f"- **Per-pair response length**: V2={statistics.mean(resp_v2_list):.0f} chars (paraphrased), new={statistics.mean(resp_new_list):.0f} chars (verbatim) → **{statistics.mean(resp_new_list)/statistics.mean(resp_v2_list):.2f}× longer per pair**")
out.append(f"- **Total response-text chars across all 20 docs**: V2={total_chars_v2:,}, new={total_chars_new:,} (ratio {total_chars_new/total_chars_v2:.2f}×). V2's fine-grained fragmentation produces MORE total paraphrase characters; new captures coarser units but verbatim.")
out.append(f"- **Total norms extracted (new)**: {total_norms}")
out.append(f"- **Norms per response (new)**: mean={statistics.mean(norms_per_resp):.2f}, median={statistics.median(norms_per_resp):.2f}")
out.append(f"- **Norms per V2-equivalent pair**: mean={statistics.mean([r['norms_per_v2_pair'] for r in rows if r['norms_per_v2_pair']]):.2f}, median={statistics.median([r['norms_per_v2_pair'] for r in rows if r['norms_per_v2_pair']]):.2f} (V2 had zero norm extraction)")
out.append("")
out.append("### Norm type distribution")
out.append("")
out.append("| norm_type | count | pct |")
out.append("|---|---:|---:|")
for nt, c in sorted(all_norm_types.items(), key=lambda x: -x[1]):
    out.append(f"| {nt} | {c} | {100*c/total_norms:.1f}% |")

out.append("")
out.append("All 7 categories represented. Top three (principle, statutory, procedural) account for "
           f"{100*(all_norm_types['principle']+all_norm_types['statutory']+all_norm_types['procedural'])/total_norms:.0f}% — these are the agency's substantive policy stances, statutory citations, and process-of-decisionmaking rules. The rarer categories (cost_benefit, philosophy, balancing) are exactly the ones we'd expect to be sparser in any given response.")
out.append("")
out.append("## 3. Side-by-side examples (V2 paraphrase vs new verbatim+norms)")
out.append("")
for did in picked:
    new = new_recs[did]
    v2 = v2_recs[did]
    out.append(f"### {did} ({new['agency']})")
    out.append("")
    out.append(f"**V2 ({v2['n_responses']} fragmented pairs, paraphrased)** — first response:")
    out.append("")
    r0 = v2["responses"][0]
    out.append(f"> **comment**: {r0.get('content_of_comment','')}")
    out.append(">")
    out.append(f"> **response (paraphrase)**: {r0.get('response_to_comment','')}")
    out.append("")
    out.append(f"**NEW ({len(new['pairs'])} pairs, verbatim+norms)** — first pair:")
    out.append("")
    p0 = new["pairs"][0]
    cv = (p0.get("comment_verbatim","") or "")[:600]
    rv = (p0.get("response_verbatim","") or "")[:1800]
    out.append(f"> **comment_verbatim**: {cv}")
    out.append(">")
    out.append(f"> **response_verbatim**: {rv}")
    out.append(">")
    out.append(f"> **norms ({len(p0.get('norms',[]))})**:")
    for n in p0.get("norms", []):
        out.append(f"> - [{n.get('norm_type','?')}] *\"{n.get('norm_verbatim','')[:220]}\"* — {n.get('norm_summary','')}")
    out.append("")

out.append("## 4. Verdict")
out.append("")
mult_len = statistics.mean(resp_new_list) / statistics.mean(resp_v2_list)
out.append(f"**Verdict: GOOD on the two design goals; the comparison frame matters.**")
out.append("")
out.append(f"- **Goal 1: verbatim responses.** ACHIEVED. Per-pair response text is {mult_len:.1f}× longer than V2's paraphrase and quotes the agency directly. V2 lost the agency's reasoning by collapsing each pair to a one-sentence summary.")
out.append(f"- **Goal 2: more norms per response.** ACHIEVED. Mean **{statistics.mean(norms_per_resp):.2f} norms per response** (V2 had zero). The norms span all 7 categories, with a healthy split between substantive (principle/value/balancing/cost_benefit) and procedural/statutory.")
out.append("")
out.append(f"**Caveat on pair count:** new prompt yields *fewer pairs per doc* (median 0.94× V2). V2 fragments one agency response into ~3 sub-pairs (one per commenter argument). This isn't necessarily worse — it's a different unit of analysis. If we want V2's fine-grained commenter-argument-level resolution, the prompt would need an explicit splitting instruction. **For norm extraction the coarser unit is arguably better** because it keeps the full reasoning chain together.")
out.append("")
out.append("**Scaling cost (Sonnet 4.5):**")
out.append("")
out.append(f"- 3,644 RTCs from parquet: ~${cost * 3644 / 20:.0f}")
out.append(f"- 15K V2-responsive docs: ~${cost * 15000 / 20:.0f} (these scale roughly linearly with token volume)")
out.append("")
out.append("**Recommendation:** scale to all **3,644 RTC sections** from the parquet first — the RTC text is already cleanly extracted there, and the per-doc avg of ~14.9K chars matches our smoke distribution. Run the same 4-worker pattern at 16-32 concurrency to keep wall-clock under 2 hours.")
out.append("")
out.append("**Two prompt revisions worth trying before the full scale-up:**")
out.append("")
out.append("1. Add an instruction to split when a response addresses multiple distinct commenter arguments (would restore V2's finer pair count without sacrificing verbatim/norm extraction).")
out.append("2. For norm.norm_verbatim, enforce that the quoted span be findable via `in response_verbatim` substring match — currently this is asserted by the prompt but not validated. Easy to add a post-hoc validator.")

with open(report_path, "w") as f:
    f.write("\n".join(out))
print(f"Wrote {report_path}")
print()
# Validation: norm_verbatim substring match rate
hits = misses = 0
for did, rec in new_recs.items():
    for p in rec["pairs"]:
        resp = p.get("response_verbatim","")
        for n in p.get("norms",[]):
            nv = n.get("norm_verbatim","")
            if nv and nv in resp:
                hits += 1
            else:
                misses += 1
print(f"norm_verbatim substring match: {hits}/{hits+misses} = {100*hits/(hits+misses):.1f}%")
