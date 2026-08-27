"""GEPA-H2H round-N ingest (seam note Sec 9, arm G).

Reads the Gemma results for round <round> (gemma_score_v1.py output: channel/aspect_id/
datapoint_id/raw/score), computes each criterion's dev Spearman rho vs its OWN judge verdict
(ctx["judge"], TRAIN dev subset only -- never test), and builds the rank-residual feedback
(10 worst |model_rank - judge_rank| dev items, doc snippet + "scored too HIGH/LOW relative
to peers" -- no raw judge numbers) that propose.py will hand to the GLM proposer.

Does NOT touch state's "round"/"prompt" fields -- that's propose.py's job, so this script is
safe to re-run (e.g. after a partial/failed scoring job) without disturbing the loop position.

Usage: python3 ingest_round.py <round> <results.jsonl>
-> updates state.json (appends one history[] entry per criterion at this round)
"""
import json, sys

from common import CRITERIA, crit_key, load_ctx, load_state, save_state, ROOT

sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman, ranks  # noqa: E402

N_WORST = 10
SNIPPET_CHARS = 300


def load_results(path, rd):
    """-> {aspect_id: {datapoint_id: int_score}}, keeping only this round's rows with a
    parsed int score (gemma_score_v1.py writes score=None/"NA" for unparseable replies)."""
    out = {}
    suffix = f".g{rd}"
    n_seen = n_kept = 0
    for line in open(path):
        r = json.loads(line)
        n_seen += 1
        if not r.get("aspect_id", "").endswith(suffix):
            continue
        sc = r.get("score")
        if not isinstance(sc, int):
            continue
        out.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = sc
        n_kept += 1
    print(f"loaded {path}: {n_seen} rows, {n_kept} usable int scores for round {rd}")
    return out


def rank_residual_feedback(sel, col, judge_aid, items):
    """sel: list of datapoint_ids (dev, judge+model both present). -> top-N_WORST worst
    mismatches as {datapoint_id, snippet, direction, rank_gap}, no raw judge numbers."""
    model_vals = [col[d] for d in sel]
    judge_vals = [judge_aid[d] for d in sel]
    r_model = ranks(model_vals)
    r_judge = ranks(judge_vals)
    rows = []
    for d, rm, rj in zip(sel, r_model, r_judge):
        gap = rm - rj
        rows.append((abs(gap), d, gap))
    rows.sort(reverse=True)
    worst = []
    for absgap, d, gap in rows[:N_WORST]:
        snippet = (items.get(d, "") or "")[:SNIPPET_CHARS]
        direction = "scored too HIGH relative to peers" if gap > 0 else \
                    "scored too LOW relative to peers"
        worst.append({"datapoint_id": d, "snippet": snippet, "direction": direction,
                      "rank_gap": round(absgap, 1)})
    return worst


def main():
    if len(sys.argv) != 3:
        print("usage: python3 ingest_round.py <round> <results.jsonl>"); sys.exit(1)
    rd = int(sys.argv[1])
    results_path = sys.argv[2]
    state = load_state()
    scored = load_results(results_path, rd)

    ctxs = {}
    rows_report = []
    for task, aid in CRITERIA:
        key = crit_key(task, aid)
        c = state["criteria"][key]
        if c["round"] != rd:
            rows_report.append((key, "SKIP", f"state round={c['round']} != {rd}"))
            continue
        if task not in ctxs:
            ctxs[task] = load_ctx(task)
        ctx = ctxs[task]
        judge_aid = ctx["judge"].get(aid, {})
        col = scored.get(f"{key}.g{rd}", {})
        sel = [d for d in c["dev_ids"] if d in col and d in judge_aid]
        if len(sel) < 10:
            rows_report.append((key, "SKIP", f"only {len(sel)} scored dev items (<10)"))
            continue
        rho = spearman([col[d] for d in sel], [judge_aid[d] for d in sel])
        worst = rank_residual_feedback(sel, col, judge_aid, ctx["items"])
        entry = {"round": rd, "rho_dev": round(rho, 4) if rho == rho else None,
                 "n": len(sel), "prompt_used": c["prompt"], "worst_items": worst}
        c["history"].append(entry)
        if entry["rho_dev"] is not None and (
                c["best"] is None or entry["rho_dev"] > c["best"]["rho_dev"]):
            c["best"] = {"round": rd, "rho_dev": entry["rho_dev"], "prompt": c["prompt"]}
        rows_report.append((key, f"{entry['rho_dev']}", f"n={len(sel)} best={c['best']}"))

    save_state(state)
    print(f"\n{'criterion':<24}{'status':<40}{'detail'}")
    for key, status, detail in rows_report:
        print(f"{key:<24}{status:<40}{detail}")
    print(f"\n-> updated state.json (round {rd} ingested)")


if __name__ == "__main__":
    main()
