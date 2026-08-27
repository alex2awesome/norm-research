"""Compare Claude baseline vs Qwen-122B extraction on the same 5 code-review PRs.

Outputs a markdown table to stdout summarizing:
- Per-PR: n_passages, n_signals, rubric Jaccard, verbatim-snap kind distribution, polarity dist
- Overall means and a few qualitative examples.
"""
import json, os, sys, collections, statistics
from pathlib import Path

RUN_DIR = Path(__file__).parent
QWEN = RUN_DIR / "results.jsonl"
CLAUDE = RUN_DIR / "claude_baseline.jsonl"
INPUTS = RUN_DIR / "input_units.jsonl"


def load_inputs():
    return {json.loads(l)["unit_id"]: json.loads(l) for l in open(INPUTS)}


def load_qwen():
    out = {}
    for l in open(QWEN):
        r = json.loads(l)
        out[str(r["unit_id"])] = r
    return out


def load_claude():
    out = {}
    for l in open(CLAUDE):
        r = json.loads(l)
        out[r["unit_id"]] = r
    return out


def summarize_unit(uid, parsed, source_text):
    """Return dict with: n_pass, n_sig, rubric_ids (multiset), polarity_dist, types, verbatim_ok_rate."""
    if not isinstance(parsed, dict):
        return None
    passages = parsed.get("passages", [])
    rubrics = []
    pols = collections.Counter()
    types = collections.Counter()
    n_sig = 0
    verbatim_pass_ok = 0
    verbatim_sig_ok = 0
    for p in passages:
        pt = p.get("passage_text", "")
        if pt in source_text:
            verbatim_pass_ok += 1
        for s in p.get("signals", []):
            n_sig += 1
            st = s.get("signal_text", "")
            if st in pt:
                verbatim_sig_ok += 1
            for rid in s.get("rubric_matches", []) or []:
                rubrics.append(int(rid))
            pols[s.get("polarity", "?")] += 1
            types[s.get("signal_type", "?")] += 1
    return {
        "n_pass": len(passages),
        "n_sig": n_sig,
        "rubric_ids": rubrics,
        "polarity": dict(pols),
        "types": dict(types),
        "verbatim_pass_rate": verbatim_pass_ok / max(1, len(passages)),
        "verbatim_sig_rate": verbatim_sig_ok / max(1, n_sig),
    }


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    inputs = load_inputs()
    qwen = load_qwen()
    claude = load_claude()

    rows = []
    for uid in sorted(claude.keys()):
        src = inputs[uid]["text"]
        c_sum = summarize_unit(uid, claude[uid], src)
        q_obj = qwen.get(uid)
        q_parsed = q_obj.get("parsed") if q_obj else None
        q_sum = summarize_unit(uid, q_parsed, src) if q_parsed else None
        rows.append({"uid": uid, "claude": c_sum, "qwen": q_sum, "qwen_ok": q_obj.get("ok") if q_obj else False, "qwen_error": q_obj.get("error") if q_obj else None})

    # Per-PR table
    print("# Code-review smoke v1 — Claude vs Qwen-122B (n=5)\n")
    print("## Per-PR summary\n")
    print("| PR | Claude pass/sig | Qwen pass/sig | Rubric Jaccard | Qwen ok | Qwen verbatim sig | Qwen polarity |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["claude"]
        q = r["qwen"]
        if q is None:
            print(f"| {r['uid']} | {c['n_pass']}/{c['n_sig']} | PARSE FAIL ({r['qwen_error']}) | – | {r['qwen_ok']} | – | – |")
            continue
        jac = jaccard(c["rubric_ids"], q["rubric_ids"])
        pol = ",".join(f"{k}={v}" for k, v in sorted(q["polarity"].items()))
        print(f"| {r['uid']} | {c['n_pass']}/{c['n_sig']} | {q['n_pass']}/{q['n_sig']} | {jac:.2f} | {r['qwen_ok']} | {q['verbatim_sig_rate']:.2f} | {pol} |")

    # Aggregate stats
    print("\n## Aggregates\n")
    c_pass = [r["claude"]["n_pass"] for r in rows]
    c_sig = [r["claude"]["n_sig"] for r in rows]
    q_pass = [r["qwen"]["n_pass"] for r in rows if r["qwen"]]
    q_sig = [r["qwen"]["n_sig"] for r in rows if r["qwen"]]
    q_verb_pass = [r["qwen"]["verbatim_pass_rate"] for r in rows if r["qwen"]]
    q_verb_sig = [r["qwen"]["verbatim_sig_rate"] for r in rows if r["qwen"]]

    def stats(name, xs):
        if not xs:
            print(f"- {name}: n=0")
            return
        print(f"- {name}: mean={statistics.mean(xs):.2f}, min={min(xs):.2f}, max={max(xs):.2f}, n={len(xs)}")

    stats("Claude passages/PR", c_pass)
    stats("Claude signals/PR", c_sig)
    stats("Qwen passages/PR (parsed)", q_pass)
    stats("Qwen signals/PR (parsed)", q_sig)
    stats("Qwen verbatim-pass rate", q_verb_pass)
    stats("Qwen verbatim-signal rate", q_verb_sig)

    n_ok = sum(1 for r in rows if r["qwen_ok"])
    n_parsed = sum(1 for r in rows if r["qwen"])
    print(f"\n- Qwen validation: {n_ok}/{len(rows)} fully ok, {n_parsed}/{len(rows)} parsed")

    # Polarity comparison
    all_c_pol = collections.Counter()
    all_q_pol = collections.Counter()
    for r in rows:
        for k, v in r["claude"]["polarity"].items():
            all_c_pol[k] += v
        if r["qwen"]:
            for k, v in r["qwen"]["polarity"].items():
                all_q_pol[k] += v
    print("\n## Polarity distribution (signals)\n")
    print("| polarity | Claude | Qwen |")
    print("|---|---|---|")
    for k in ("positive", "negative", "neutral"):
        print(f"| {k} | {all_c_pol[k]} | {all_q_pol[k]} |")

    # Rubric usage diversity
    all_c_rub = collections.Counter()
    all_q_rub = collections.Counter()
    for r in rows:
        for rid in r["claude"]["rubric_ids"]:
            all_c_rub[rid] += 1
        if r["qwen"]:
            for rid in r["qwen"]["rubric_ids"]:
                all_q_rub[rid] += 1
    print(f"\n## Rubric diversity\n")
    print(f"- Claude: {len(all_c_rub)} unique rubric ids used, {sum(all_c_rub.values())} total tags")
    print(f"- Qwen:   {len(all_q_rub)} unique rubric ids used, {sum(all_q_rub.values())} total tags")
    print(f"- Top 5 Claude: {all_c_rub.most_common(5)}")
    print(f"- Top 5 Qwen:   {all_q_rub.most_common(5)}")
    print(f"- Overall rubric Jaccard (set-of-ids used): {jaccard(all_c_rub.keys(), all_q_rub.keys()):.2f}")


if __name__ == "__main__":
    main()
