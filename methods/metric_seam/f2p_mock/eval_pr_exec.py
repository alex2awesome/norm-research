"""Certified eval for the pr_exec F2P-mock experiment (run after the Gemma job lands).

Per exec-op aspect (a67 correctness, a104 tests-presence, a128 test-adequacy):
  channels: best description-compiled code flavor | hybrid with the MOCKED transplant op
            (ExecOps) | same hybrid with the op nulled (NullExecOps)
  readouts: held-out rho vs judge + bootstrap gate vs code baseline
            + gate(hybrid_full vs hybrid_noexec) = the heavy machinery's CERTIFIED MARGINAL
Anchors (judgement accept/reject, days_open), Simpson-guarded:
  pooled AND within-batch weighted Spearman for every channel + exec-label alone.

Usage (laptop, after pulling results):  python3 eval_pr_exec.py
Needs: outputs/metric_seam_pilot/tasks/pr_exec/{items,exec_features,aspects_used}.json,
       results.jsonl, code_scores.json (run_code_flavors_task.py pr_exec via the
       runs/validity_full/v2/pr_exec -> code_review symlink).
"""
import importlib.util, json, pathlib, random, re, statistics as st, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/tasks/pr_exec"
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman, attenuation_ceiling, bootstrap_gate
from ops_exec import ExecOps, NullExecOps

FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]
HYB_ASPECTS = ["a67", "a104", "a128"]


def clean(raw):
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    line = re.sub(r"^(answer|reply)\s*[:\-]\s*", "", line, flags=re.I).strip()
    return "" if line.upper().startswith("NONE") else line[:200]


def load_all():
    p1, p2, scope, fields = {}, {}, {}, {}
    for line in open(OUT / "results.jsonl"):
        r = json.loads(line)
        if r["channel"] == "field":
            fields.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = clean(r.get("raw", ""))
            continue
        if not isinstance(r["score"], int):
            continue
        d = {"pass1": p1, "pass2": p2}.get(r["channel"])
        if d is not None:
            d.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
        elif r["channel"] == "scope":
            scope[r["datapoint_id"]] = r["score"]
    judge = {}
    for aid in set(p1) | set(p2):
        for dp in set(p1.get(aid, {})) | set(p2.get(aid, {})):
            vals = [m[aid][dp] for m in (p1, p2) if dp in m.get(aid, {})]
            judge.setdefault(aid, {})[dp] = sum(vals) / len(vals) / 10.0
    rel1 = {aid: spearman([p1[aid][d] for d in p1.get(aid, {}) if d in p2.get(aid, {})],
                          [p2[aid][d] for d in p1.get(aid, {}) if d in p2.get(aid, {})])
            for aid in set(p1) & set(p2)}
    return judge, rel1, scope, fields


def within_batch_spearman(x, y, batch, min_n=8):
    """Simpson guard: weighted mean of within-batch Spearman (only batches with variance)."""
    by = {}
    for d in x:
        if d in y and d in batch:
            by.setdefault(batch[d], []).append(d)
    num = den = 0.0
    used = []
    for b, ds in by.items():
        if len(ds) < min_n:
            continue
        r = spearman([x[d] for d in ds], [y[d] for d in ds])
        if r == r:
            num += r * len(ds); den += len(ds); used.append((b, len(ds), round(r, 3)))
    return (num / den if den else float("nan")), used


def main():
    judge, rel1, scope, fields = load_all()
    items = json.load(open(OUT / "items.json"))
    texts = {it["datapoint_id"]: it["ctext"] for it in items}
    batch = {it["datapoint_id"]: it["batch"] for it in items}
    accept = {it["datapoint_id"]: it["judgement"] for it in items
              if it.get("judgement") is not None}
    days = {it["datapoint_id"]: it["days_open"] for it in items
            if it.get("days_open") is not None}
    feats = json.load(open(OUT / "exec_features.json"))
    code = json.load(open(OUT / "code_scores.json"))
    ops_full = ExecOps(OUT / "exec_features.json")
    ops_null = NullExecOps(OUT / "exec_features.json")

    ids = sorted(texts)
    rng = random.Random(7)
    test = set(rng.sample(ids, int(0.4 * len(ids))))
    train = [d for d in ids if d not in test]

    # exec-label as its own raw channel (for anchors)
    lab_score = {d: {"pinned": 1.0, "partial_pinned": 0.75, "vacuous": 0.25,
                     "none": 0.35, "indeterminate": 0.5}.get(f.get("label"), 0.5)
                 for d, f in feats.items()}

    report = {}
    for aid in HYB_ASPECTS:
        prog = pathlib.Path(__file__).parent / "programs" / f"{aid}_h0.py"
        spec = importlib.util.spec_from_file_location(prog.stem, prog)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fmap = {d: {f.split("__", 1)[1]: fields[f].get(d, "")
                    for f in fields if f.startswith(aid + "__")} for d in ids}
        hyb_full = {d: mod.score(texts[d], fmap[d], ops_full, dpid=d) for d in ids}
        hyb_null = {d: mod.score(texts[d], fmap[d], ops_null, dpid=d) for d in ids}

        j = judge.get(aid, {})
        tsel = [d for d in test if d in j]
        best = (None, float("-inf"), None)
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}")
            if not col:
                continue
            sel = [d for d in tsel if col.get(d) is not None]
            if len(sel) < 40:
                continue
            r = spearman([col[d] for d in sel], [j[d] for d in sel])
            if r == r and r > best[1]:
                best = (fl, r, col)
        fl, base_rho, base_col = best
        if base_col is None:
            base_col, fl, base_rho = {d: 0.5 for d in ids}, "none(constant)", 0.0

        g_code = bootstrap_gate(hyb_full, base_col, j, tsel, B=2000)
        g_op = bootstrap_gate(hyb_full, hyb_null, j, tsel, margin=0.0, floor=0.0, B=2000)
        r1 = rel1.get(aid, float("nan"))
        row = {
            "rel1": round(r1, 3) if r1 == r1 else None,
            "ceiling": round(attenuation_ceiling(min(max(r1, 0), 1), 2), 3) if r1 == r1 else None,
            "baseline": {"flavor": fl, "rho_test": round(base_rho, 3)},
            "hybrid_full": g_code,
            "rho_noexec": round(spearman([hyb_null[d] for d in tsel],
                                         [j[d] for d in tsel]), 3),
            "op_marginal_gate": {"P_beats_noexec": g_op["P_beats_baseline"],
                                 "rho_full": g_op["rho_mean"]},
        }
        # anchors, every channel, pooled + within-batch
        for label_name, lab in (("accept", accept), ("days_open", days)):
            anch = {}
            for ch_name, ch in (("judge", j), ("code", base_col), ("hybrid", hyb_full),
                                ("hybrid_noexec", hyb_null), ("exec_label", lab_score)):
                sel = [d for d in ids if d in lab and ch.get(d) is not None]
                if len(sel) < 60:
                    continue
                pooled = spearman([ch[d] for d in sel], [lab[d] for d in sel])
                wb, used = within_batch_spearman({d: ch[d] for d in sel},
                                                 {d: lab[d] for d in sel}, batch)
                anch[ch_name] = {"pooled": round(pooled, 3),
                                 "within_batch": round(wb, 3) if wb == wb else None,
                                 "n": len(sel), "batches_used": len(used)}
            row[f"anchor_{label_name}"] = anch
        report[aid] = row
        print(f"{aid}: base={base_rho:.3f}({fl}) hybrid={g_code['rho_mean']} "
              f"CI{g_code['rho_ci']} P(gate)={g_code['P_gate']} "
              f"P(>base)={g_code['P_beats_baseline']} | no-exec ρ={row['rho_noexec']} "
              f"P(op helps)={g_op['P_beats_baseline']}")
        for ln in ("accept", "days_open"):
            a = row.get(f"anchor_{ln}", {})
            print(f"   {ln}: " + "  ".join(
                f"{k} {v['pooled']}/{v['within_batch']}" for k, v in a.items()))
    json.dump(report, open(OUT / "pr_exec_eval.json", "w"), indent=1)
    print("\nwrote", OUT / "pr_exec_eval.json",
          "\n(anchor cells are pooled/within-batch — trust the within-batch number)")


if __name__ == "__main__":
    main()
