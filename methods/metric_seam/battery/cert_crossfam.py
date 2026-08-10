"""Cross-family gate recertification for the CW + humor certified sets.

PR's 12 certified gates were swap-tested in the transport eval (3/12 fail). This
closes the fleet: re-run the G1 gate (hybrid vs frozen train-best codegen flavor,
paired bootstrap) for every CW/humor certified criterion with the hybrid's
LLM_FIELDS swapped to the Llama-70B and Qwen-122B extractions (frozen programs,
frozen splits, same judge target). Legal has no swap extractions (known thin spot).

Usage: python3 cert_crossfam.py
-> outputs/metric_seam_pilot/battery/crossfam_cert.json
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, load_fields, run_prog, BASE, ROOT  # noqa: E402
from eval_hybrids_task import paired_boot, FLAVORS  # noqa: E402
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402

CERTIFIED = {"creative_writing": ["a144", "a72", "a99", "a90", "a342"],
             "humor": ["a351", "a135", "a153", "a81"],
             "legal_title_vii": ["a44", "a46", "a39", "a15", "a0", "a23", "a18",
                                 "a36", "a5"]}


def main():
    report = {}
    for task, aids in CERTIFIED.items():
        ctx = load_ctx(task)
        td = ctx["outdir"]
        codes = json.load(open(td / "code_scores.json"))
        fam = {"gemma": ctx["f_orig"],
               "llama": load_fields(td / "field_results_llama.jsonl"),
               "qwen": load_fields(td / "field_results_qwen.jsonl")}
        for aid in aids:
            judge = ctx["judge"].get(aid, {})
            pm = load_mod(ctx["hyb"] / f"{aid}_h0.py")
            te = [d for d in sorted(ctx["test"]) if d in judge]
            # frozen baseline flavor: train-best, as in the original gates
            train = sorted(ctx["train"])
            best_fl, best_tr, base_col = None, -2, None
            for fl in FLAVORS:
                col = codes.get(f"{aid}_{fl}") or {}
                s = [d for d in train if d in judge and col.get(d) is not None]
                if len(s) < 30:
                    continue
                r = spearman([col[d] for d in s], [judge[d] for d in s])
                if r == r and r > best_tr:
                    best_fl, best_tr, base_col = fl, r, col
            row = {"baseline_flavor": best_fl}
            for name, fmap_all in fam.items():
                fmap = fmap_all.get(aid, {})
                col = run_prog(pm.score, ctx["items"], fmap, ctx["ops"])
                s = [d for d in te if col.get(d) is not None]
                rho = spearman([col[d] for d in s], [judge[d] for d in s]) \
                    if len(s) >= 20 else float("nan")
                ent = {"rho_test": round(rho, 4) if rho == rho else None,
                       "n": len(s)}
                if base_col is not None:
                    sg = [d for d in s if base_col.get(d) is not None]
                    pg, _, _ = paired_boot(sg, col, base_col, judge)
                    ent["P_gate"] = pg
                row[name] = ent
            report[f"{task}.{aid}"] = row
            print(f"{task}.{aid} (base={best_fl}): " + "  ".join(
                f"{n} rho={row[n]['rho_test']} P_gate={row[n].get('P_gate')}"
                for n in ("gemma", "llama", "qwen")))
    out = BASE / "battery/crossfam_cert.json"
    json.dump(report, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
