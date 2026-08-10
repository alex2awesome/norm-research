"""AGENTIC-COMPILE iteration harness (seam note §8) — TRAIN split only.

Runs a candidate program against the TRAIN items of one criterion and prints:
train Spearman rho (vs judge), the h0 reference rho on the same items, and the 15
worst residual items with doc snippets — the agent's reflection signal.

The candidate module must define score(text, extracted, ops) like every fleet program.
`extracted` is served from the EXISTING Gemma field extraction (f_orig): candidates may
reuse any field name already extracted for this criterion. New field names simply
arrive as missing (None) during iteration — declare them in LLM_FIELDS for the final
post-hoc extraction, but do not rely on them for the train signal you see here.

TEST SPLIT IS NEVER TOUCHED HERE (dpid contract). HARNESS v2 (2026-07-10, post external
review): candidates are now EXECUTED on train texts only — v1 executed them on all items
(no held-out information was ever surfaced, but the seal was procedural; now structural).

Usage: python3 agentic_run.py <task> <aid> <candidate.py>
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, run_prog, BASE, ROOT  # noqa: E402
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402


def main():
    task, aid, cand = sys.argv[1], sys.argv[2], sys.argv[3]
    ctx = load_ctx(task)
    train = sorted(ctx["train"])
    judge = ctx["judge"].get(aid, {})
    fmap = ctx["f_orig"].get(aid, {})

    mod = load_mod(pathlib.Path(cand))
    train_texts = {d: ctx["items"][d] for d in train if d in ctx["items"]}
    col = run_prog(mod.score, train_texts, fmap, ctx["ops"])
    ref = load_mod(ctx["hyb"] / f"{aid}_h0.py")
    col0 = run_prog(ref.score, train_texts, fmap, ctx["ops"])

    sel = [d for d in train if d in judge and col.get(d) is not None]
    if len(sel) < 40:
        print(f"ERROR: only {len(sel)} scoreable train items")
        return
    xs = [col[d] for d in sel]
    ys = [judge[d] for d in sel]
    rho = spearman(xs, ys)
    sel0 = [d for d in sel if col0.get(d) is not None]
    rho0 = spearman([col0[d] for d in sel0], [judge[d] for d in sel0])
    print(f"TRAIN rho candidate = {rho:.4f}   (h0 reference = {rho0:.4f}, n={len(sel)})")

    # worst residuals: rank-space disagreement
    import statistics
    rx = {d: sorted(xs).index(col[d]) / len(xs) for d in sel}
    ry = {d: sorted(ys).index(judge[d]) / len(ys) for d in sel}
    worst = sorted(sel, key=lambda d: -abs(rx[d] - ry[d]))[:15]
    print("== worst residuals (candidate rank vs judge rank) ==")
    for d in worst:
        snippet = (ctx["items"][d][:160].replace("\n", " "))
        print(f"{d} cand={col[d]:.3f}(r{rx[d]:.2f}) judge={judge[d]:.2f}(r{ry[d]:.2f}) "
              f"fields={fmap.get(d)} | {snippet}")


if __name__ == "__main__":
    main()
