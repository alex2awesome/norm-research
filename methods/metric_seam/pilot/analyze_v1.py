"""Robust seam table v1: same-input channels, 2-pass judge reliability, scoped subset, CF probe."""
import json, pathlib, statistics as st
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "hybrids"))
from harness import load_judge, load_scope, spearman, OUT

ASPECTS = ["a79", "a80", "a110", "a100", "a101", "a86", "a105", "a118", "a117", "a73"]
FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]
NAMES = {"a79": "Wire-ready format", "a80": "Media contacts", "a110": "Boilerplate/metadata",
         "a100": "Lede 5Ws", "a101": "Inverted pyramid", "a86": "Quote quality",
         "a105": "Plain language", "a118": "Timeliness", "a117": "Newsworthiness",
         "a73": "Empathy"}


def main():
    judge, p1, p2 = load_judge()
    in_scope, scope_scores = load_scope()
    code = json.load(open(OUT / "code_scores_v1.json"))
    print(f"scope: {len(in_scope)}/250 items are press releases (scope>=7); "
          f"scope score dist: mean={st.mean(scope_scores.values()):.1f}")

    table = []
    hdr = (f"{'aspect':6} {'name':20} {'rel':>5} {'n_sc':>4} "
           f"{'rho v0':>7} {'rho v1':>7} {'rho v2':>7} {'best(scoped)':>12}")
    print(hdr)
    for aid in ASPECTS:
        both = [d for d in p1.get(aid, {}) if d in p2.get(aid, {})]
        rel = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
        row = {"aspect": aid, "reliability": round(rel, 3)}
        cells_all, cells_sc = {}, {}
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}")
            if col is None:
                cells_all[fl] = None
                continue
            sel = [d for d in judge.get(aid, {}) if col.get(d) is not None]
            cells_all[fl] = round(spearman([col[d] for d in sel],
                                           [judge[aid][d] for d in sel]), 3)
            sel_sc = [d for d in sel if d in in_scope]
            cells_sc[fl] = round(spearman([col[d] for d in sel_sc],
                                          [judge[aid][d] for d in sel_sc]), 3)
        row["rho_all"] = cells_all
        row["rho_scoped"] = cells_sc
        row["n_scoped"] = len([d for d in judge.get(aid, {}) if d in in_scope])
        best_sc = max((v for v in cells_sc.values() if v == v), default=float("nan"))
        def s(fl):
            v = cells_all.get(fl)
            return f"{v:+.2f}" if isinstance(v, float) else "  brk"
        print(f"{aid:6} {NAMES[aid][:20]:20} {rel:5.2f} {row['n_scoped']:4} "
              f"{s('v0_keyword'):>7} {s('v1_structure'):>7} {s('v2_holistic'):>7} "
              f"{best_sc:12.2f}")
        table.append(row)

    # CF probe: judge delta on quote-injected items vs their original pass1 a86 score
    cf_j = {}
    for line in open(OUT / "results_v1.jsonl"):
        r = json.loads(line)
        if r["channel"] == "cf_a86" and isinstance(r["score"], int):
            cf_j[r["datapoint_id"]] = r["score"] / 10.0
    orig = {d: p1["a86"][d] / 10.0 for d in cf_j if d in p1.get("a86", {})}
    cf_code = json.load(open(OUT / "code_scores_cf_a86.json"))["a86_v0_keyword"]
    code_orig = code["a86_v0_keyword"]
    dj = [cf_j[d] - orig[d] for d in orig]
    dc = [cf_code[d] - code_orig[d] for d in orig
          if cf_code.get(d) is not None and code_orig.get(d) is not None]
    print(f"\nCF probe (inject generic excited-CEO quote, n={len(orig)}):")
    print(f"  judge  delta: mean {st.mean(dj):+.3f}  (quality judge should stay ~flat)")
    print(f"  v0 code delta: mean {st.mean(dc):+.3f}  (presence proxy should jump)")
    json.dump({"table": table,
               "cf": {"judge_delta": st.mean(dj), "code_delta": st.mean(dc),
                      "n": len(orig)},
               "scope_n": len(in_scope)},
              open(OUT / "seam_table_v1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
