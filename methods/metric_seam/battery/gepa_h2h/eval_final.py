"""GEPA-H2H held-out certification (seam note Sec 9) -- arm G (GEPA-optimized single-call
Gemma scoring prompt) vs arm H (frozen certified hybrid: code + <=2 Gemma field extractions).

THE ONLY SCRIPT IN THIS HARNESS THAT READS ctx["test"]. Everything upstream (init/build/
ingest/propose) touches TRAIN dev items exclusively.

Two subcommands:

  python3 eval_final.py build
      Freezes each criterion's best-DEV-rho prompt (state.json "best", falling back to the
      current in-flight prompt with a warning if a criterion was never ingested), renders it
      against the TEST split, writes gepa_final_prompts.jsonl. Prints the queue2 submit hint;
      does not call any LLM API itself.

  python3 eval_final.py eval <final_results.jsonl>
      Loads arm G's test scores from <final_results.jsonl> (gemma_score_v1.py output),
      recomputes arm H's test column the same way cert_agentic.py does (runs the certified
      agentic program AND its h0 baseline, takes whichever has the higher rho_test already on
      record in battery/agentic_cert.json -- "the certified arm"), computes arm G's test rho,
      and runs the SAME paired bootstrap (eval_hybrids_task.paired_boot, B=2000) used
      everywhere else in this battery to get P(G > H) on the identical test items. Adds a
      cost column (1 Gemma call/doc for G vs code + k Gemma field calls/doc for H).
      -> gepa_h2h_final.json
"""
import json, sys

from common import (AGENTIC_DIR, AGENTIC_FNAME, BASE, CRITERIA, HERE, ROOT,
                    build_doc_prompt, crit_key, load_ctx, load_state)  # noqa: F401 (load_state used by cmd_build)

sys.path.insert(0, str(ROOT / "methods/metric_seam/battery"))
from battery_common import load_mod, run_prog                       # noqa: E402
from eval_hybrids_task import paired_boot, FLAVORS                  # noqa: E402,F401
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman                                    # noqa: E402

AGENTIC_CERT_PATH = BASE / "battery/agentic_cert.json"


def cmd_build():
    state = load_state()
    ctxs = {}
    out_path = HERE / "gepa_final_prompts.jsonl"
    n = 0
    frozen = {}
    with open(out_path, "w") as f:
        for task, aid in CRITERIA:
            key = crit_key(task, aid)
            c = state["criteria"][key]
            if c["best"] is not None:
                body = c["best"]["prompt"]
                src = f"best round {c['best']['round']} (dev rho={c['best']['rho_dev']})"
            else:
                body = c["prompt"]
                src = f"WARNING: never ingested -- using in-flight round {c['round']} prompt"
            frozen[key] = {"prompt": body, "source": src}
            if task not in ctxs:
                ctxs[task] = load_ctx(task)
            items, test_ids = ctxs[task]["items"], sorted(ctxs[task]["test"])
            for dpid in test_ids:
                row = {"channel": "field", "aspect_id": f"{key}.final",
                      "datapoint_id": dpid, "prompt": build_doc_prompt(body, items.get(dpid, ""))}
                f.write(json.dumps(row) + "\n")
                n += 1
            print(f"{key}: {src} -> {len(test_ids)} test rows")

    json.dump(frozen, open(HERE / "gepa_final_frozen_prompts.json", "w"), indent=1)
    print(f"\nwrote {n} rows -> {out_path}")
    print(f"frozen prompts recorded -> {HERE / 'gepa_final_frozen_prompts.json'}")
    print()
    print("orchestrator queue2 submit hint (sk3, gemma4 env, GPU N):")
    out_results = HERE / "gepa_final_results.jsonl"
    print(f"  echo '{out_path} {out_results}' > $QDIR2/900_gepa_final.job")
    print(f"  # after it completes: python3 eval_final.py eval {out_results}")


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    if len(s) < 20:
        return float("nan"), 0
    return spearman([col[d] for d in s], [judge[d] for d in s]), len(s)


def load_arm_g(results_path):
    """-> {crit_key: {datapoint_id: int_score}} from gepa_final_prompts.jsonl scoring."""
    out = {}
    for line in open(results_path):
        r = json.loads(line)
        aid_field = r.get("aspect_id", "")
        if not aid_field.endswith(".final"):
            continue
        key = aid_field[: -len(".final")]
        sc = r.get("score")
        if isinstance(sc, int):
            out.setdefault(key, {})[r["datapoint_id"]] = sc
    return out


def cmd_eval(results_path):
    cert = json.load(open(AGENTIC_CERT_PATH))
    col_g_by_key = load_arm_g(results_path)

    ctxs = {}
    report = {}
    for task, aid in CRITERIA:
        key = crit_key(task, aid)
        if task not in ctxs:
            ctxs[task] = load_ctx(task)
        ctx = ctxs[task]
        judge = ctx["judge"].get(aid, {})
        col_g = col_g_by_key.get(key, {})

        # arm H: recompute the raw test column for whichever of {cand, h0} is certified
        # (higher rho_test already on record in agentic_cert.json), same recipe as
        # cert_agentic.py so the item selection and program execution are identical.
        fmap = ctx["f_orig"].get(aid, {})
        cand_mod = load_mod(AGENTIC_DIR / AGENTIC_FNAME[(task, aid)])
        h0_mod = load_mod(ctx["hyb"] / f"{aid}_h0.py")
        col_cand = run_prog(cand_mod.score, ctx["items"], fmap, ctx["ops"])
        col_h0 = run_prog(h0_mod.score, ctx["items"], fmap, ctx["ops"])

        cert_row = cert.get(key, {})
        rho_test_cert = cert_row.get("rho_test", {})
        use_cand = (rho_test_cert.get("cand", -2) >= rho_test_cert.get("h0", -2))
        col_h = col_cand if use_cand else col_h0
        h_mod = cand_mod if use_cand else h0_mod
        h_label = "cand" if use_cand else "h0"
        n_fields_h = len(getattr(h_mod, "LLM_FIELDS", {}) or {})

        test_ids = sorted(ctx["test"])
        sel = [d for d in test_ids if d in judge and col_g.get(d) is not None
               and col_h.get(d) is not None]
        r_g, n_g = rho_on(sel, col_g, judge)
        r_h, n_h = rho_on(sel, col_h, judge)
        p_gate, p_beat, used = paired_boot(sel, col_g, col_h, judge, gate_floor=-2, margin=-2)

        row = {"n_test": len(sel), "arm_h_source": h_label,
               "rho_test": {"G": round(r_g, 4) if r_g == r_g else None,
                            "H": round(r_h, 4) if r_h == r_h else None},
               "delta_test_G_minus_H": (round(r_g - r_h, 4) if r_g == r_g and r_h == r_h
                                        else None),
               "P_G_gt_H": p_beat, "boot_used": used,
               "cost": {"G": "1 Gemma call/doc", "H": f"code + {n_fields_h} Gemma call(s)/doc"}}
        report[key] = row
        print(f"{key}: test G {r_g:+.3f} vs H[{h_label}] {r_h:+.3f}  "
              f"(n={len(sel)})  P(G>H)={p_beat}  cost G=1 call vs H={n_fields_h} call(s)+code")

    out = HERE / "gepa_h2h_final.json"
    json.dump(report, open(out, "w"), indent=1)
    print(f"\n-> {out}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "eval"):
        print("usage: python3 eval_final.py build | eval <final_results.jsonl>")
        sys.exit(1)
    if sys.argv[1] == "build":
        cmd_build()
    else:
        if len(sys.argv) != 3:
            print("usage: python3 eval_final.py eval <final_results.jsonl>"); sys.exit(1)
        cmd_eval(sys.argv[2])


if __name__ == "__main__":
    main()
