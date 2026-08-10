"""Mixedness measurement: ablate each medium of a hybrid channel and attribute test-rho.

Conditions (2x2): {LLM fields on/off} x {tool ops full/null}.
  code core        = fields OFF, ops NULL   (pure predicate on raw text)
  +tools           = fields OFF, ops FULL   (computation/evidence ops added)
  +LLM             = fields ON,  ops NULL   (thick extraction added, no tools)
  full hybrid      = fields ON,  ops FULL
Also: per-item LLM-touched share (fraction of test items where fields move the score > 0.02),
tool-touched share, and static medium shares (code LOC, field-instruction tokens, ops call sites).
"""
import json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from harness import OUT, ROOT, load_judge, load_scope, split_ids, spearman, load_hybrid, \
    load_fields, run_hybrid
from ops import Ops

HEAD = {"a80": "a80_h0.py", "a86": "a86_h0.py", "a110": "a110_h0.py", "a105": "a105_h0.py"}


class NullOps(Ops):
    """Ablated ops: every tool is inert."""
    def __init__(self):
        super().__init__(corpus_path=None)
    @staticmethod
    def normalize(text):
        return text
    @staticmethod
    def extract_dates(text):
        return []
    @staticmethod
    def sent_stats(text):
        return 0, 0.0, 0.0
    def retrieve_similar(self, text, k=5, exclude_id=None):
        return []


def main():
    judge, _, _ = load_judge()
    _, test = split_ids()
    in_scope, _ = load_scope()
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(OUT / "items_v1.json"))}
    ops_full = Ops(corpus_path=str(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))
    ops_null = NullOps()
    prog_dir = pathlib.Path(__file__).parent / "programs"

    print(f"{'aspect':6} {'core':>6} {'+tools':>7} {'+LLM':>6} {'full':>6} "
          f"{'ΔLLM':>6} {'Δtool':>6} {'LLMtouch':>8} {'tooltouch':>9}")
    report = {}
    for aid, fname in HEAD.items():
        mod = load_hybrid(prog_dir / fname)
        fields = load_fields(aid)
        conds = {
            "core":  run_hybrid(mod, items, {}, ops_null),
            "tools": run_hybrid(mod, items, {}, ops_full),
            "llm":   run_hybrid(mod, items, fields, ops_null),
            "full":  run_hybrid(mod, items, fields, ops_full),
        }
        rho = {}
        for name, col in conds.items():
            sel = [d for d in test if d in judge.get(aid, {}) and col.get(d) is not None]
            rho[name] = round(spearman([col[d] for d in sel],
                                       [judge[aid][d] for d in sel]), 3)
        touched = lambda a, b: sum(  # noqa: E731
            1 for d in test
            if conds[a].get(d) is not None and conds[b].get(d) is not None
            and abs(conds[a][d] - conds[b][d]) > 0.02) / max(
                1, sum(1 for d in test if conds[a].get(d) is not None))
        llm_touch = touched("full", "tools")
        tool_touch = touched("full", "llm")

        src = (prog_dir / fname).read_text()
        loc = sum(1 for l in src.splitlines() if l.strip() and not l.strip().startswith("#"))
        field_toks = sum(len(v.split()) for v in (getattr(mod, "LLM_FIELDS", {}) or {}).values())
        ops_calls = len(re.findall(r"\bops\.\w+\(", src))

        d_llm = round(rho["full"] - rho["tools"], 3)
        d_tool = round(rho["full"] - rho["llm"], 3)
        print(f"{aid:6} {rho['core']:6.2f} {rho['tools']:7.2f} {rho['llm']:6.2f} "
              f"{rho['full']:6.2f} {d_llm:+6.2f} {d_tool:+6.2f} "
              f"{llm_touch:8.2f} {tool_touch:9.2f}")
        report[aid] = {"rho": rho, "delta_llm": d_llm, "delta_tool": d_tool,
                       "llm_touched_share": round(llm_touch, 3),
                       "tool_touched_share": round(tool_touch, 3),
                       "static": {"code_loc": loc, "field_instruction_tokens": field_toks,
                                  "ops_call_sites": ops_calls}}
    json.dump(report, open(OUT / "mixedness_report.json", "w"), indent=1)
    print("\nstatic medium shares:")
    for aid, r in report.items():
        s = r["static"]
        print(f"  {aid}: {s['code_loc']} code LOC | {s['field_instruction_tokens']} "
              f"LLM-instruction tokens | {s['ops_call_sites']} ops call sites")


if __name__ == "__main__":
    main()
