"""Shared context loader for retrieval-battery evals (E1/E2/E5).

load_ctx(task) -> dict with items, test ids, judge, certified gemma field maps, ops,
and the hybrid program dir — PR (v2 machinery) and fleet tasks unified.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
sys.path.insert(0, str(ROOT / "methods/metric_seam/pilot"))

from eval_hybrids_task import (split_task_ids, load_mod, load_judge,   # noqa: E402
                               load_fields, run_prog)                  # noqa: F401
from ops import Ops                                                    # noqa: E402

PROGDIR = {"press_releases": "programs_v2", "creative_writing": "programs_cw",
           "math": "programs_math", "humor": "programs_humor",
           "legal_title_vii": "programs_legal", "peer_review": "programs_peer",
           "legal_ss_disability": "programs_ssdis", "humor_units": "programs_units"}


def load_ctx(task):
    hyb = ROOT / "methods/metric_seam/hybrids" / PROGDIR[task]
    if task == "press_releases":
        from harness import split_ids
        from analyze_v2 import load_judge_v2, OUT, V1
        items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(V1 / "items_v1.json"))}
        train, test = split_ids()
        judge, _, _ = load_judge_v2()
        f_orig = load_fields(OUT / "field_results_v2.jsonl")
        ops = Ops(corpus_path=str(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))
        outdir = OUT
    else:
        outdir = BASE / "tasks" / task
        items_l = json.load(open(outdir / "items.json"))
        items = {x["datapoint_id"]: x["ctext"] for x in items_l}
        train, test = split_task_ids(items_l)
        judge, _ = load_judge(outdir / "results.jsonl")
        f_orig = load_fields(outdir / "field_results.jsonl")
        if task == "math":
            from ops_math import MathOps
            ops = MathOps(corpus_path=str(outdir / "items.json"))
        else:
            ops = Ops(corpus_path=str(outdir / "items.json"))
    return {"items": items, "train": train, "test": test, "judge": judge,
            "f_orig": f_orig, "ops": ops, "hyb": hyb, "outdir": outdir}
