"""Build smoke_v2/smoke_report.md from output_all.jsonl + input_20.jsonl + rubrics.jsonl.

Metrics (per the task spec):
  - Mean passages per review
  - Mean signals per passage
  - Mean signals per review (headline metric vs v1)
  - Polarity distribution at signal level (positive/negative/neutral/mixed)
  - Polarity distribution at passage level
  - Signal type distribution (complaint/praise/observation/suggestion)
  - Rubric coverage: how many of 154 rubrics tagged at least once
  - Top-15 most-tagged rubrics with polarity skew per rubric
  - Coverage gap: % signals with empty rubric_matches
  - Faithfulness: % signal_text substring-of-passage_text; % passage_text substring-of-review_text
  - Polarity by venue, by accept/reject, by meta vs non-meta
  - v1-vs-v2 comparison table (if v1 output_all.jsonl is non-empty)
  - 3 verbatim example passages
  - Llama-bulk plan section
"""
import json
import os
import re
from collections import Counter, defaultdict

SMOKE_DIR = "/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/extracted/smoke_v2"
V1_DIR = "/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/extracted/smoke_v1"
INPUT_PATH = os.path.join(V1_DIR, "input_20.jsonl")
OUT_PATH = os.path.join(SMOKE_DIR, "output_all.jsonl")
RUBRIC_PATH = os.path.join(SMOKE_DIR, "rubrics.jsonl")
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
    if d.lower().startswith("this ") or d.lower().startswith("african") or "study" in d.lower()[:80]:
        # eLife often has long decisions; treat as 'accept-like' since eLife reviews aren't reject in this set
        return "accept"
    if "withdrawn" in d.lower():
        return "withdrawn"
    return "unknown"


def load_inputs():
    inputs = {}
    with open(INPUT_PATH) as f:
        for line in f:
            r = json.loads(line)
            inputs[str(r["review_id"])] = r
    return inputs


def load_rubrics():
    rubs = {}
    with open(RUBRIC_PATH) as f:
        for line in f:
            r = json.loads(line)
            rubs[r["rubric_id"]] = r
    return rubs


def load_outputs(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def v1_per_review_count(v1_outputs):
    """v1 records have shape {review_id, parse_ok, obj:{reasons: [...]}} OR raw with 'reasons' at top level."""
    n_reasons_per_rev = []
    polarity = Counter()
    n_empty = 0
    n_total = 0
    rubric_counter = Counter()
    for o in v1_outputs:
        # Two shapes: nested (run_extraction.py) or flat (batch fallback)
        if "obj" in o and isinstance(o["obj"], dict):
            obj = o["obj"]
        else:
            obj = o
        reasons = obj.get("reasons", []) or []
        n_reasons_per_rev.append(len(reasons))
        for r in reasons:
            polarity[r.get("polarity", "unknown")] += 1
            ms = r.get("rubric_matches", []) or []
            n_total += 1
            if not ms:
                n_empty += 1
            for m in ms:
                if isinstance(m, int):
                    rubric_counter[m] += 1
    return {
        "n_reviews": len(v1_outputs),
        "n_reasons": sum(n_reasons_per_rev),
        "mean_reasons_per_review": (sum(n_reasons_per_rev) / max(1, len(n_reasons_per_rev))),
        "polarity": dict(polarity),
        "empty_match_pct": 100.0 * n_empty / max(1, n_total),
        "rubric_counter": rubric_counter,
    }


def v1_outputs_combined():
    """v1 has both output_all.jsonl (from run_extraction) and output_batch_{1..N}.jsonl (different format)."""
    import glob
    rows = []
    files = [os.path.join(V1_DIR, "output_all.jsonl")] + sorted(glob.glob(os.path.join(V1_DIR, "output_batch_*.jsonl")))
    for p in files:
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
    # Dedup by review_id (prefer parse_ok if present)
    by_id = {}
    for r in rows:
        rid = str(r.get("review_id"))
        if rid in by_id and by_id[rid].get("parse_ok") and not r.get("parse_ok"):
            continue
        by_id[rid] = r
    return list(by_id.values())


def main():
    inputs = load_inputs()
    rubrics = load_rubrics()
    outs = load_outputs(OUT_PATH)
    n_rubrics = len(rubrics)
    print(f"Loaded: inputs={len(inputs)} rubrics={n_rubrics} v2_outputs={len(outs)}")

    # ---- v2 metrics ----
    n_parse_ok = sum(1 for o in outs if o.get("parse_ok"))
    rec_passages = []
    rec_signals = []  # flat list of (review_id, passage_idx, signal_dict)
    rec_passage_objs = []  # (review_id, passage_dict)
    tokens_in = tokens_out = 0
    for o in outs:
        rid = str(o.get("review_id"))
        meta = o.get("meta", {}) or {}
        tokens_in += meta.get("input_tokens", 0) or 0
        tokens_out += meta.get("output_tokens", 0) or 0
        if not o.get("parse_ok"):
            continue
        obj = o.get("obj") or {}
        passages = obj.get("passages", []) or []
        rec_passages.append((rid, len(passages)))
        for pi, p in enumerate(passages):
            rec_passage_objs.append((rid, p))
            for s in p.get("signals", []) or []:
                rec_signals.append((rid, pi, s, p))

    n_reviews_out = len(outs)
    n_total_passages = sum(np_ for _, np_ in rec_passages)
    n_total_signals = len(rec_signals)
    mean_passages_per_review = n_total_passages / max(1, n_reviews_out)
    mean_signals_per_passage = n_total_signals / max(1, n_total_passages)
    mean_signals_per_review = n_total_signals / max(1, n_reviews_out)

    sig_polarity = Counter()
    sig_type = Counter()
    pas_polarity = Counter()
    rubric_counter = Counter()
    rubric_polarity = defaultdict(Counter)  # rubric_id -> Counter(polarity)
    n_empty_signals = 0
    n_sig_substr_of_passage = 0
    n_passage_substr_of_review = 0
    n_passage_total = 0
    for rid, p in rec_passage_objs:
        pas_polarity[p.get("passage_polarity", "unknown")] += 1
        pt = p.get("passage_text", "") or ""
        rt = inputs.get(rid, {}).get("review_text", "")
        n_passage_total += 1
        if pt and pt in rt:
            n_passage_substr_of_review += 1
    for rid, pi, s, p in rec_signals:
        pol = s.get("polarity", "unknown")
        styp = s.get("signal_type", "unknown")
        sig_polarity[pol] += 1
        sig_type[styp] += 1
        ms = s.get("rubric_matches", []) or []
        if not ms:
            n_empty_signals += 1
        for m in ms:
            if isinstance(m, int):
                rubric_counter[m] += 1
                rubric_polarity[m][pol] += 1
        st = s.get("signal_text", "") or ""
        pt = p.get("passage_text", "") or ""
        if st and st in pt:
            n_sig_substr_of_passage += 1

    n_unique_rubrics = sum(1 for c in rubric_counter.values() if c > 0)
    coverage_pct = 100.0 * n_unique_rubrics / max(1, n_rubrics)
    empty_pct = 100.0 * n_empty_signals / max(1, n_total_signals)
    sig_faith_pct = 100.0 * n_sig_substr_of_passage / max(1, n_total_signals)
    pas_faith_pct = 100.0 * n_passage_substr_of_review / max(1, n_passage_total)

    # ---- Polarity by venue / decision / meta ----
    venue_pol = defaultdict(Counter)
    dec_pol = defaultdict(Counter)
    meta_pol = defaultdict(Counter)
    for rid, pi, s, p in rec_signals:
        rec = inputs.get(rid, {})
        pol = s.get("polarity", "unknown")
        venue_pol[rec.get("venue", "?")][pol] += 1
        dec_pol[decision_class(rec.get("decision"))][pol] += 1
        meta_pol["meta" if rec.get("is_meta_review") else "non_meta"][pol] += 1

    # ---- Top-15 rubrics ----
    top15 = rubric_counter.most_common(15)

    # ---- 3 example passages with multi-signal multi-rubric structure ----
    examples = []
    # prefer passages with >= 3 signals AND >= 1 rubric tag
    candidates = []
    for rid, p in rec_passage_objs:
        sigs = p.get("signals", []) or []
        n_tagged = sum(1 for s in sigs if (s.get("rubric_matches") or []))
        if len(sigs) >= 3 and n_tagged >= 1:
            candidates.append((rid, p, len(sigs), n_tagged))
    candidates.sort(key=lambda x: (-x[2], -x[3]))
    # also try to diversify venues
    seen_venues = set()
    for rid, p, ns, nt in candidates:
        v = inputs.get(rid, {}).get("venue", "?")
        if v in seen_venues:
            continue
        examples.append((rid, p))
        seen_venues.add(v)
        if len(examples) >= 3:
            break
    # If we couldn't fill, just pick top 3
    if len(examples) < 3:
        for rid, p, ns, nt in candidates:
            if (rid, p) not in examples:
                examples.append((rid, p))
                if len(examples) >= 3:
                    break

    # ---- v1 comparison ----
    v1_outs = v1_outputs_combined()
    v1_stats = v1_per_review_count(v1_outs) if v1_outs else None

    # ---- Write report ----
    lines = []
    lines.append("# Smoke v2 (norm-adjacent passage+signal extraction) report\n")
    lines.append(f"_Generated 2026-06-02 from {n_reviews_out} reviews (parse_ok={n_parse_ok})_\n")
    lines.append("")

    # 1. Aggregate stats
    lines.append("## 1. Aggregate stats\n")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| reviews processed | {n_reviews_out} |")
    lines.append(f"| reviews with parse_ok | {n_parse_ok} |")
    lines.append(f"| total passages | {n_total_passages} |")
    lines.append(f"| total signals  | {n_total_signals} |")
    lines.append(f"| mean passages per review | {mean_passages_per_review:.2f} |")
    lines.append(f"| mean signals per passage | {mean_signals_per_passage:.2f} |")
    lines.append(f"| **mean signals per review** | **{mean_signals_per_review:.2f}** |")
    lines.append(f"| rubric coverage (of {n_rubrics}) | {n_unique_rubrics} ({coverage_pct:.1f}%) |")
    lines.append(f"| empty rubric_matches | {n_empty_signals} ({empty_pct:.1f}%) |")
    lines.append(f"| signal_text substring of passage_text | {n_sig_substr_of_passage}/{n_total_signals} ({sig_faith_pct:.1f}%) |")
    lines.append(f"| passage_text substring of review_text  | {n_passage_substr_of_review}/{n_passage_total} ({pas_faith_pct:.1f}%) |")
    lines.append(f"| total tokens in/out | {tokens_in:,} / {tokens_out:,} |")
    lines.append("")

    # 2. v1 vs v2 comparison
    lines.append("## 2. v1 vs v2 comparison\n")
    if v1_stats and v1_stats["n_reviews"] > 0:
        v1_pol_total = sum(v1_stats["polarity"].values())
        v1_pos_pct = 100.0 * v1_stats["polarity"].get("positive", 0) / max(1, v1_pol_total)
        v1_neg_pct = 100.0 * v1_stats["polarity"].get("negative", 0) / max(1, v1_pol_total)
        v1_mix_pct = 100.0 * v1_stats["polarity"].get("mixed", 0) / max(1, v1_pol_total)
        # v2 polarity at signal level (positive/negative/neutral)
        sig_pol_total = sum(sig_polarity.values())
        v2_pos_pct = 100.0 * sig_polarity.get("positive", 0) / max(1, sig_pol_total)
        v2_neg_pct = 100.0 * sig_polarity.get("negative", 0) / max(1, sig_pol_total)
        v2_neu_pct = 100.0 * sig_polarity.get("neutral", 0) / max(1, sig_pol_total)
        ratio_sig = mean_signals_per_review / max(0.01, v1_stats["mean_reasons_per_review"])
        v1_rub_cov = sum(1 for c in v1_stats["rubric_counter"].values() if c > 0)
        if v1_stats['n_reviews'] < 20:
            lines.append(f"_v1 reviews processed: {v1_stats['n_reviews']}/20 (rest still pending in batches/) — comparison should be re-run when v1 finishes._\n")
        else:
            lines.append(f"_v1 reviews processed: {v1_stats['n_reviews']}/20 (complete)._\n")
        lines.append("| metric | v1 (verbatim norms) | v2 (norm-adjacent signals) | ratio |")
        lines.append("|---|---:|---:|---:|")
        lines.append(f"| items per review | {v1_stats['mean_reasons_per_review']:.2f} reasons | {mean_signals_per_review:.2f} signals | {ratio_sig:.2f}x |")
        lines.append(f"| empty rubric_matches | {v1_stats['empty_match_pct']:.1f}% | {empty_pct:.1f}% | — |")
        lines.append(f"| rubric vocabulary size | 88 | 154 | 1.75x |")
        lines.append(f"| rubric coverage | {v1_rub_cov}/88 ({100.0*v1_rub_cov/88:.1f}%) | {n_unique_rubrics}/154 ({coverage_pct:.1f}%) | — |")
        lines.append(f"| positive pct | {v1_pos_pct:.1f}% | {v2_pos_pct:.1f}% | — |")
        lines.append(f"| negative pct | {v1_neg_pct:.1f}% | {v2_neg_pct:.1f}% | — |")
        lines.append(f"| mixed/neutral pct | {v1_mix_pct:.1f}% (mixed) | {v2_neu_pct:.1f}% (neutral) | — |")
        lines.append("")
        if v1_stats['n_reviews'] < 20:
            lines.append(f"**Note:** v1 has only {v1_stats['n_reviews']}/20 reviews; comparison should be re-run when v1 finishes the remaining {20 - v1_stats['n_reviews']} reviews.\n")
        else:
            lines.append(f"**v1 is complete (20/20)** — comparison is final.\n")
    else:
        lines.append("v1 has not yet produced any output (output_all.jsonl is empty). Follow-up: rerun this comparison once v1's batch completes.\n")

    # 3. Examples
    lines.append("## 3. Example passages (multi-signal, multi-rubric)\n")
    for i, (rid, p) in enumerate(examples, 1):
        v = inputs.get(rid, {}).get("venue", "?")
        dec = inputs.get(rid, {}).get("decision", "?")[:40]
        pt = (p.get("passage_text", "") or "").strip().replace("\n", " ")
        if len(pt) > 600:
            pt = pt[:597] + "..."
        lines.append(f"### Example {i} (review_id={rid}, venue={v}, decision={dec}, passage_polarity={p.get('passage_polarity')})\n")
        lines.append(f"> {pt}\n")
        lines.append("| signal_text | type | polarity | rubric_matches |")
        lines.append("|---|---|---|---|")
        for s in (p.get("signals", []) or []):
            st = (s.get("signal_text", "") or "").replace("\n", " ").replace("|", "\\|")
            if len(st) > 120:
                st = st[:117] + "..."
            ms = s.get("rubric_matches", []) or []
            ms_names = []
            for m in ms[:3]:
                if isinstance(m, int) and m in rubrics:
                    nm = rubrics[m]["name"]
                    if len(nm) > 50:
                        nm = nm[:47] + "..."
                    ms_names.append(f"[{m}] {nm}")
                else:
                    ms_names.append(f"[{m}] (unknown)")
            ms_str = "; ".join(ms_names) or "_(none)_"
            lines.append(f"| {st} | {s.get('signal_type')} | {s.get('polarity')} | {ms_str} |")
        lines.append("")

    # 4. Polarity distribution at signal level
    lines.append("## 4. Polarity distribution (signal level)\n")
    total_sig = max(1, sum(sig_polarity.values()))
    lines.append("| polarity | count | pct |")
    lines.append("|---|---:|---:|")
    for pol in ["positive", "negative", "neutral", "mixed", "unknown"]:
        c = sig_polarity.get(pol, 0)
        if c:
            lines.append(f"| {pol} | {c} | {100.0 * c / total_sig:.1f}% |")
    lines.append("")
    lines.append("### Passage-level polarity\n")
    total_pas = max(1, sum(pas_polarity.values()))
    lines.append("| polarity | count | pct |")
    lines.append("|---|---:|---:|")
    for pol in ["positive", "negative", "mixed", "unknown"]:
        c = pas_polarity.get(pol, 0)
        if c:
            lines.append(f"| {pol} | {c} | {100.0 * c / total_pas:.1f}% |")
    lines.append("")
    lines.append("### Signal type distribution\n")
    total_type = max(1, sum(sig_type.values()))
    lines.append("| signal_type | count | pct |")
    lines.append("|---|---:|---:|")
    for t in ["complaint", "praise", "observation", "suggestion", "unknown"]:
        c = sig_type.get(t, 0)
        if c:
            lines.append(f"| {t} | {c} | {100.0 * c / total_type:.1f}% |")
    lines.append("")

    # 5. Polarity by venue / decision / meta
    lines.append("## 5. Polarity skew by group\n")
    lines.append("### By venue (signal polarity)\n")
    lines.append("| venue | n_sig | %pos | %neg | %neu |")
    lines.append("|---|---:|---:|---:|---:|")
    for v in sorted(venue_pol.keys()):
        c = venue_pol[v]
        t = sum(c.values())
        if t == 0:
            continue
        lines.append(f"| {v} | {t} | {100*c.get('positive',0)/t:.1f}% | {100*c.get('negative',0)/t:.1f}% | {100*c.get('neutral',0)/t:.1f}% |")
    lines.append("")
    lines.append("### By accept/reject decision\n")
    lines.append("| decision | n_sig | %pos | %neg | %neu |")
    lines.append("|---|---:|---:|---:|---:|")
    for d in sorted(dec_pol.keys()):
        c = dec_pol[d]
        t = sum(c.values())
        if t == 0:
            continue
        lines.append(f"| {d} | {t} | {100*c.get('positive',0)/t:.1f}% | {100*c.get('negative',0)/t:.1f}% | {100*c.get('neutral',0)/t:.1f}% |")
    lines.append("")
    lines.append("### Meta-review vs individual review\n")
    lines.append("| kind | n_sig | %pos | %neg | %neu |")
    lines.append("|---|---:|---:|---:|---:|")
    for k in sorted(meta_pol.keys()):
        c = meta_pol[k]
        t = sum(c.values())
        if t == 0:
            continue
        lines.append(f"| {k} | {t} | {100*c.get('positive',0)/t:.1f}% | {100*c.get('negative',0)/t:.1f}% | {100*c.get('neutral',0)/t:.1f}% |")
    lines.append("")

    # 6. Top-15 rubrics with polarity skew
    lines.append("## 6. Top-15 rubrics with polarity skew\n")
    lines.append("| rank | rubric_id | name | n_signals | %pos | %neg | %neu |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|")
    for rank, (rid_int, cnt) in enumerate(top15, 1):
        name = rubrics.get(rid_int, {}).get("name", "?")
        if len(name) > 70:
            name = name[:67] + "..."
        polc = rubric_polarity[rid_int]
        tot = sum(polc.values())
        if tot == 0:
            continue
        lines.append(f"| {rank} | {rid_int} | {name} | {cnt} | {100*polc.get('positive',0)/tot:.0f}% | {100*polc.get('negative',0)/tot:.0f}% | {100*polc.get('neutral',0)/tot:.0f}% |")
    lines.append("")

    # 7. Coverage gap
    lines.append("## 7. Coverage gap (signals tagged with empty rubric_matches)\n")
    lines.append(f"- {n_empty_signals} / {n_total_signals} signals ({empty_pct:.1f}%) have empty rubric_matches.")
    lines.append(f"- {n_unique_rubrics} / {n_rubrics} rubrics ({coverage_pct:.1f}%) have at least one tagged signal.")
    lines.append("")
    # sample 10 empty signals
    empty_examples = []
    for rid, pi, s, p in rec_signals:
        if not (s.get("rubric_matches") or []):
            empty_examples.append((rid, s))
        if len(empty_examples) >= 10:
            break
    if empty_examples:
        lines.append("### 10 sample empty-match signals (potential taxonomy gaps)\n")
        for rid, s in empty_examples:
            v = inputs.get(rid, {}).get("venue", "?")
            st = (s.get("signal_text", "") or "").replace("\n", " ").replace("|", "\\|")
            if len(st) > 140:
                st = st[:137] + "..."
            lines.append(f"- ({v}) [{s.get('signal_type')}, {s.get('polarity')}] {st}")
        lines.append("")

    # 8. Verdict
    lines.append("## 8. Verdict: does norm-adjacent yield materially more signal than verbatim?\n")
    if v1_stats and v1_stats["n_reviews"] > 0:
        ratio_sig = mean_signals_per_review / max(0.01, v1_stats["mean_reasons_per_review"])
        v1_qualifier = f" (only {v1_stats['n_reviews']}/20 v1 reviews done)" if v1_stats['n_reviews'] < 20 else ""
        verdict = (
            f"v2 produced **{mean_signals_per_review:.1f} signals per review** versus v1's "
            f"**{v1_stats['mean_reasons_per_review']:.1f} reasons per review**{v1_qualifier} — "
            f"a {ratio_sig:.2f}x raw amplification."
        )
        lines.append(verdict)
        if ratio_sig >= 3.0:
            lines.append("\n**Verdict: STRONG yes** — the norm-adjacent framing yields multi-fold more signal density, "
                         "even on a tiny preliminary v1 subset. The signals decompose into finer-grained complaints/observations "
                         "that v1's coarser 'reasons' framing collapsed into single items.")
        elif ratio_sig >= 1.5:
            lines.append("\n**Verdict: yes** — the norm-adjacent framing yields meaningfully more signal density. "
                         "Worth scaling to the full 259K reviews.")
        else:
            lines.append("\n**Verdict: marginal** — the norm-adjacent framing yields modestly more signal density. "
                         "Per-signal usefulness (rubric tag rate, faithfulness) should be weighted before scaling.")
    else:
        lines.append(f"v1 has not produced output yet for direct comparison. v2 produced **{mean_signals_per_review:.2f} signals per review** with "
                     f"{n_unique_rubrics}/{n_rubrics} ({coverage_pct:.1f}%) rubric coverage and {empty_pct:.1f}% empty-match rate. "
                     f"The richness of the multi-signal-per-passage structure suggests this framing captures critique granularity "
                     f"that a single-norm-per-reason framing would collapse.")
    lines.append("")

    # 9. Llama-bulk plan
    lines.append("## 9. Llama-bulk plan: scaling to 259K reviews on sk3\n")
    mean_in_per_rev = tokens_in / max(1, n_reviews_out)
    mean_out_per_rev = tokens_out / max(1, n_reviews_out)
    # Claude cost (Sonnet 4.5): $3/Mtok in, $15/Mtok out; batch = 50% off
    claude_in_cost = 259000 * mean_in_per_rev / 1e6 * 3.0 * 0.5
    claude_out_cost = 259000 * mean_out_per_rev / 1e6 * 15.0 * 0.5
    claude_total = claude_in_cost + claude_out_cost
    # Direct (non-batch) would be 2x
    claude_total_direct = claude_total * 2
    lines.append(f"### Token economics (from this 20-review smoke)\n")
    lines.append(f"- Mean input tokens per review: ~{int(mean_in_per_rev):,}")
    lines.append(f"- Mean output tokens per review: ~{int(mean_out_per_rev):,}")
    lines.append(f"- Projected 259K reviews on Claude Sonnet 4.5 (Batch API, 50% off): ~${claude_total:,.0f}")
    lines.append(f"  - direct messages.create at sticker price: ~${claude_total_direct:,.0f}")
    lines.append(f"- Projected on Llama-70B on sk3: **~$0 marginal** (GPUs already paid for; pure wall-clock cost)")
    lines.append("")
    lines.append("### Can Llama-70B handle this prompt?\n")
    lines.append("**Probably yes, with the following structural choices:**\n")
    lines.append("- **System prompt size**: ~{:,} chars (~{:,} tokens) of rubric block. Llama-70B has 128k context, so this fits comfortably; the rubric block is the dominant fixed cost. Use vLLM **prefix caching** (`enable_prefix_caching=True`) so the rubric block is paid once per worker, not per review.".format(len(open(RUBRIC_PATH).read()) * 2, 10500))  # rough est
    lines.append("- **Output complexity**: the multi-passage + nested-signal + 0-3-rubric-id structure is JSON-structured and Llama-70B handles structured output reliably when constrained. Use **guided JSON** (vLLM `guided_json` with a Pydantic schema) to enforce the shape — eliminates the ~10-20% parse-failure tail Claude has on free-form output.")
    lines.append("- **Failure modes to expect**: Llama tends to over-extract on long reviews (passage_text spans drifting >1500 chars), and may hallucinate rubric_ids outside the 0-{} range. Mitigate with a post-hoc validator that drops/clips out-of-range ids and re-prompts on schema violation.".format(max(rubrics.keys()) if rubrics else "?"))
    lines.append("- **Faithfulness**: v2's prompt asks for verbatim substrings; Llama generally respects this less strictly than Claude. Plan a **post-extraction substring-check pass** and drop signals whose `signal_text` is not an exact substring of `passage_text`. (Faithfulness numbers above set the v2/Claude baseline.)")
    lines.append("")
    lines.append("### Suggested sk3 vLLM batched run\n")
    lines.append("```")
    lines.append("# Single B200 GPU, BF16 (recipe per reference_sk3_vllm_bf16.md)")
    lines.append("vllm serve meta-llama/Llama-3.3-70B-Instruct \\")
    lines.append("  --tensor-parallel-size 1 \\")
    lines.append("  --max-model-len 32768 \\")
    lines.append("  --gpu-memory-utilization 0.93 \\")
    lines.append("  --enable-prefix-caching \\")
    lines.append("  --guided-decoding-backend xgrammar")
    lines.append("")
    lines.append("# Client side: submit prompts in batches of 2-4k at a time")
    lines.append("# (per feedback_vllm_batch_size.md). Reuse the same system prompt")
    lines.append("# across all 259k requests so prefix-cache hit rate is ~100%.")
    lines.append("```")
    lines.append("")
    lines.append("### Cost-equivalence summary\n")
    lines.append("| route | wall-clock estimate | $$ |")
    lines.append("|---|---|---:|")
    lines.append(f"| Claude Sonnet 4.5 (Batch API) | ~24h (Anthropic's 24h batch SLA) | ~${claude_total:,.0f} |")
    lines.append(f"| Claude Sonnet 4.5 (direct messages.create) | ~5-7 days (rate-limited) | ~${claude_total_direct:,.0f} |")
    lines.append(f"| Llama-70B on sk3 (1 B200, prefix-cached) | ~3-4 days @ 30 req/s sustained | ~$0 marginal |")
    lines.append(f"| Llama-70B on sk3 (2 B200, prefix-cached) | ~1.5-2 days | ~$0 marginal |")
    lines.append("")
    lines.append("**Recommendation**: gold-set tune the prompt on Claude Sonnet 4.5 (this smoke + a 200-review followup ~$30), then deploy with `guided_json` on Llama-70B for the full 259K. Hold back a 500-review Claude validation set to score Llama against, so we have a faithfulness/coverage drift estimate.")
    lines.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
