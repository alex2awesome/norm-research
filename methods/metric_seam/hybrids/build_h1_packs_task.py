"""Round-1 (h1) feedback packs for near-gate hybrid aspects — task-generic.

Protocol = the kill-switch p901/p907 h1 protocol (which itself = the v1 a80-h1 protocol):
original improver pack (criterion + contract) + h0 source + TRAIN-only residual diagnostics
(top-12 |judge - h0| divergent cells with the h0 LLM-field values + 6 well-fit anchors) +
explicit anti-overfit warning. The held-out gate decides h0 vs h1; test items NEVER appear here.

Usage: python3 build_h1_packs_task.py <task> <progdir> <aid1,aid2,...> [math]
-> outputs/metric_seam_pilot/tasks/<task>/h1_packs/<aid>.json
"""
import importlib.util, json, pathlib, signal, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman, attenuation_ceiling            # noqa: E402
from eval_hybrids_task import split_task_ids, load_judge, load_fields  # noqa: E402
from ops import Ops                                               # noqa: E402


def _alarm(s, f):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _alarm)


def main():
    task, progdir, aids = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    use_math = len(sys.argv) > 4 and sys.argv[4] == "math"
    OUT = ROOT / "outputs/metric_seam_pilot/tasks" / task
    HYB = pathlib.Path(__file__).parent / progdir

    items_l = json.load(open(OUT / "items.json"))
    items = {x["datapoint_id"]: x["ctext"] for x in items_l}
    train, _ = split_task_ids(items_l)
    judge, rel = load_judge(OUT / "results.jsonl")
    fields = load_fields(OUT / "field_results.jsonl")
    gate_rep = json.load(open(OUT / "hybrid_gate_report.json"))
    if use_math:
        from ops_math import MathOps
        ops = MathOps(corpus_path=str(OUT / "items.json"))
    else:
        ops = Ops(corpus_path=str(OUT / "items.json"))

    packs = OUT / "h1_packs"
    packs.mkdir(exist_ok=True)
    for aid in aids:
        prog = HYB / f"{aid}_h0.py"
        orig = json.load(open(OUT / "improver_packs" / f"{aid}.json"))
        spec = importlib.util.spec_from_file_location(prog.stem, prog)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fl = fields.get(aid, {})
        rows = []
        for d in sorted(train):
            if d not in judge.get(aid, {}):
                continue
            try:
                signal.alarm(20)
                h0 = float(mod.score(items[d], fl.get(d, {}), ops))
            except Exception:
                h0 = None
            finally:
                signal.alarm(0)
            if h0 is not None:
                rows.append((d, judge[aid][d], h0, abs(judge[aid][d] - h0)))
        tr_rho = spearman([r[2] for r in rows], [r[1] for r in rows])
        rows.sort(key=lambda r: -r[3])
        worst, best = rows[:12], rows[-6:]
        rep = gate_rep.get(aid, {})
        r1 = max(0.0, min(1.0, rel.get(aid, 0.0)))

        def cell(d, j, h, head, tail=0):
            c = {"datapoint_id": d, "judge_score_0_1": round(j, 2),
                 "h0_score": round(h, 3), "h0_field_values": fl.get(d, {}),
                 "text_excerpt_head": items[d][:head]}
            if tail:
                c["text_excerpt_tail"] = items[d][-tail:]
            return c

        pack = {
            "aspect_id": aid,
            "criterion_name": orig["criterion_name"],
            "criterion_description": orig["criterion_description"],
            "contract": orig["contract"],
            "h0_source": prog.read_text(),
            "h0_train_rho": round(tr_rho, 3),
            "baseline_flavor": rep.get("baseline_flavor"),
            "baseline_train_rho": rep.get("baseline_train_rho"),
            "judge_reliability": round(r1, 3),
            "attenuation_ceiling": round(attenuation_ceiling(r1, 2), 3),
            "target_note": (
                "h0 beats the frozen code baseline but does not clear the certification gate "
                "(held-out rho >= max(baseline+0.10, 0.60), paired bootstrap). Your job: write "
                "h1 = a targeted revision of h0. The residual cells below are TRAIN items only."),
            "anti_overfit_warning": (
                "A previous h1 improved 12/12 train residual cells and then DROPPED 0.13 on the "
                "held-out gate (train-selected special-casing did not transfer). Do NOT write "
                "narrow rules keyed to these specific excerpts; fix the GENERAL failure mode "
                "they illustrate. The held-out gate decides whether h1 replaces h0."),
            "worst_cells": [cell(d, j, h, 2200, 1200) for d, j, h, _ in worst],
            "well_fit_anchors": [cell(d, j, h, 1200) for d, j, h, _ in best],
        }
        json.dump(pack, open(packs / f"{aid}.json", "w"), indent=1)
        print(f"{aid}: h0 train rho={tr_rho:.3f} (n={len(rows)}); "
              f"{len(worst)} worst + {len(best)} anchors -> {packs / (aid + '.json')}")


if __name__ == "__main__":
    main()
