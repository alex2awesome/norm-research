"""Round-1 feedback packs for the kill-switch near-misses (p901, p907, Arm S).

Protocol = the v1 a80-h1 protocol: h0 source + the train cells where h0 diverges most from
the channel (top-12 by |residual| + 6 well-fit anchors for contrast), explicit anti-overfit
warning (the a80 lesson: 12/12 train cells improved, test dropped .13 — held-out gate
decides h0 vs h1; train-selection test-gated).
"""
import importlib.util, json, pathlib, signal, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman, attenuation_ceiling  # noqa: E402
from harness import split_ids                            # noqa: E402
from ops import Ops                                      # noqa: E402
from eval_killswitch import load_two_pass                # noqa: E402

OUT = ROOT / "outputs/metric_seam_pilot/killswitch"
HYB = pathlib.Path(__file__).parent / "programs_ks"


def _alarm(s, f):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _alarm)


def main():
    items = {x["datapoint_id"]: x["ctext"]
             for x in json.load(open(ROOT / "outputs/metric_seam_pilot/v1/items_v1.json"))}
    train, _ = split_ids()
    chan, rel = load_two_pass(OUT / "channels_synth.jsonl")
    ops = Ops(corpus_path=str(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))
    packs = OUT / "h1_packs"
    packs.mkdir(exist_ok=True)

    for pid in ["p901", "p907"]:
        prog = HYB / f"{pid}_h0.py"
        spec = importlib.util.spec_from_file_location(prog.stem, prog)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ch = chan[pid]
        rows = []
        for d in train:
            if d not in ch:
                continue
            try:
                signal.alarm(20)
                h0 = float(mod.score(items[d], {}, ops))
            except Exception:
                h0 = None
            finally:
                signal.alarm(0)
            if h0 is not None:
                rows.append((d, ch[d], h0, abs(ch[d] - h0)))
        tr_rho = spearman([r[2] for r in rows], [r[1] for r in rows])
        rows.sort(key=lambda r: -r[3])
        worst, best = rows[:12], rows[-6:]
        r1 = max(0.0, min(1.0, rel[pid]))
        pack = {
            "aspect_id": pid,
            "h0_source": prog.read_text(),
            "h0_train_rho": round(tr_rho, 3),
            "judge_reliability": round(rel[pid], 3),
            "attenuation_ceiling": round(attenuation_ceiling(r1, 2), 3),
            "target_note": ("h0 is close but not at the certification bar (85% of the "
                            "attenuation ceiling on held-out). The residual below is "
                            "computed on TRAIN items only."),
            "worst_cells": [{"datapoint_id": d, "judge_score_0_1": round(j, 2),
                             "h0_score": round(h, 3),
                             "text_excerpt_head": items[d][:2200],
                             "text_excerpt_tail": items[d][-1200:]}
                            for d, j, h, _ in worst],
            "well_fit_anchors": [{"datapoint_id": d, "judge_score_0_1": round(j, 2),
                                  "h0_score": round(h, 3),
                                  "text_excerpt_head": items[d][:1200]}
                                 for d, j, h, _ in best]}
        json.dump(pack, open(packs / f"{pid}.json", "w"), indent=1)
        print(f"{pid}: h0 train rho={tr_rho:.3f}; pack with {len(worst)} worst + "
              f"{len(best)} anchors -> {packs / (pid + '.json')}")


if __name__ == "__main__":
    main()
