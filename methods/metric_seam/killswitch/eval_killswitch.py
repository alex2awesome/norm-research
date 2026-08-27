"""Kill-switch evaluation, code-rung arm (hybrids wired in after the improver round).

Per (arm, plant): rel1 + S1 ceiling, per-flavor train/test rho (harness seed-7 split),
best-rung-by-train verdict with Rung-3 bootstrap (B=2000) of test rho, P(>= 85% ceiling),
P(>= 60% floor); oracle rungs (the generators, run by US, never by agents) for
p901/p902/p903/p907 — p907's oracle-vs-ceiling gap is the S1 formula check; op-type readout
(strong-executor-without-tool = the Arm-J Gemma channel) for p902/p903.

Outputs killswitch_report.json + printed table. Uses CLEAN-ROOM code scores only.
"""
import json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman, attenuation_ceiling  # noqa: E402
from harness import split_ids                            # noqa: E402
from plants import (TRUTH_TYPE, map_p904, truth_p901_raw, truth_p902,  # noqa: E402
                    truth_p907_raw)

OUT = ROOT / "outputs/metric_seam_pilot/killswitch"
PIDS = ["p901", "p902", "p903", "p904", "p905", "p906", "p907"]
FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]
B = 2000


def load_two_pass(path):
    p1, p2 = {}, {}
    for line in open(path):
        r = json.loads(line)
        if not isinstance(r["score"], int):
            continue
        d = p1 if r["channel"] == "pass1" else p2 if r["channel"] == "pass2" else None
        if d is not None:
            d.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    comb, rel = {}, {}
    for aid in set(p1) | set(p2):
        both = [d for d in p1.get(aid, {}) if d in p2.get(aid, {})]
        rel[aid] = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both]) \
            if len(both) >= 10 else float("nan")
        for dpid in set(p1.get(aid, {})) | set(p2.get(aid, {})):
            vals = [d[aid][dpid] for d in (p1, p2) if dpid in d.get(aid, {})]
            comb.setdefault(aid, {})[dpid] = sum(vals) / len(vals) / 10.0
    return comb, rel


def boot_rho(xs, ys, n_boot=B, seed=13):
    rng = random.Random(seed)
    n = len(xs)
    out = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(spearman([xs[i] for i in idx], [ys[i] for i in idx]))
    out = [v for v in out if v == v]
    out.sort()
    return out


def oracle_scores(items):
    """The generating functions run as programs (by us). p903 needs the evidence op."""
    from ops import Ops
    ops = Ops(corpus_path=str(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))
    labels = json.load(open(OUT / "claude_labels.json"))
    o = {"p901": {d: truth_p901_raw(t) for d, t in items.items()},
         "p902": {d: truth_p902(t) for d, t in items.items()},
         "p903": {d: -(lambda top: top[0][0] if top else 0.0)
                  (ops.retrieve_similar(t, k=1, exclude_id=d)) for d, t in items.items()},
         "p904": {d: map_p904(int(labels[d]["n_quoted_speakers"]))
                  for d in items if d in labels},
         "p907": {d: truth_p907_raw(t) for d, t in items.items()}}
    return o


def main():
    items = {x["datapoint_id"]: x["ctext"]
             for x in json.load(open(ROOT / "outputs/metric_seam_pilot/v1/items_v1.json"))}
    train, test = split_ids()
    cs = json.load(open(OUT / "code_scores_ks.json"))["scores"]
    arms = {"S": load_two_pass(OUT / "channels_synth.jsonl"),
            "J": load_two_pass(OUT / "results_judge.jsonl")}
    oracle = oracle_scores(items)

    report = {}
    hdr = ("arm plant type              rel1  ceil  best_rung     rho_tr rho_te "
           "P>=.85c P>=.60c oracle/ceil")
    print(hdr)
    for arm, (chan, rel) in arms.items():
        for pid in PIDS:
            ch = chan.get(pid)
            if not ch:
                continue
            r1 = rel[pid]
            ceil = attenuation_ceiling(max(0.0, min(1.0, r1)), 2) if r1 == r1 else None
            row = {"rel1": round(r1, 3) if r1 == r1 else None,
                   "ceiling": round(ceil, 3) if ceil else None, "flavors": {}}
            best = (None, -2, None)
            for fl in FLAVORS:
                col = cs.get(f"{pid}_{fl}") or {}
                str_ = {}
                for name, idset in [("train", train), ("test", test)]:
                    sel = [d for d in idset if d in ch and col.get(d) is not None]
                    str_[name] = (spearman([col[d] for d in sel], [ch[d] for d in sel]),
                                  len(sel))
                row["flavors"][fl] = {k: (round(v[0], 3) if v[0] == v[0] else None,
                                          v[1]) for k, v in str_.items()}
                if str_["train"][0] == str_["train"][0] and str_["train"][0] > best[1]:
                    best = (fl, str_["train"][0], col)
            if best[0]:
                col = best[2]
                sel = [d for d in test if d in ch and col.get(d) is not None]
                xs, ys = [col[d] for d in sel], [ch[d] for d in sel]
                bs = boot_rho(xs, ys)
                te = spearman(xs, ys)
                p85 = (sum(1 for v in bs if ceil and v >= 0.85 * ceil) / len(bs)
                       if bs and ceil else None)
                p60 = sum(1 for v in bs if v >= 0.60) / len(bs) if bs else None
                row["best"] = {"flavor": best[0], "rho_train": round(best[1], 3),
                               "rho_test": round(te, 3),
                               "ci": [round(bs[int(.025 * len(bs))], 3),
                                      round(bs[int(.975 * len(bs))], 3)],
                               "P_ge_85ceil": p85, "P_ge_60": p60, "n_test": len(sel)}
            orc = None
            if pid in oracle:
                sel = [d for d in ch if d in oracle[pid]]
                orho = spearman([oracle[pid][d] for d in sel], [ch[d] for d in sel])
                orc = {"rho": round(orho, 3),
                       "over_ceiling": round(orho / ceil, 3) if ceil else None}
                row["oracle"] = orc
            report[f"{arm}:{pid}"] = row
            b = row.get("best", {})
            print(f"{arm}   {pid} {TRUTH_TYPE[pid]:17s} "
                  f"{row['rel1'] if row['rel1'] is not None else float('nan'):.3f} "
                  f"{row['ceiling'] if row['ceiling'] else float('nan'):.3f} "
                  f"{b.get('flavor','-'):13s} "
                  f"{b.get('rho_train', float('nan')):+.3f} "
                  f"{b.get('rho_test', float('nan')):+.3f} "
                  f"{b.get('P_ge_85ceil') if b.get('P_ge_85ceil') is not None else -1:7.2f} "
                  f"{b.get('P_ge_60') if b.get('P_ge_60') is not None else -1:7.2f} "
                  f"{orc['over_ceiling'] if orc and orc['over_ceiling'] else float('nan'):+.3f}")

    # op-type readout: strong executor WITHOUT tool = Arm-J judge channel vs truth
    tr = json.load(open(OUT / "truth.json"))["truths"]
    ops_read = {}
    for pid in ["p902", "p903"]:
        chJ, relJ = arms["J"]
        ch = chJ.get(pid, {})
        sel = [d for d in ch if tr.get(pid, {}).get(d) is not None]
        rho = spearman([ch[d] for d in sel], [tr[pid][d] for d in sel])
        ceilJ = attenuation_ceiling(max(0.0, min(1.0, relJ[pid])), 2)
        frac = rho / ceilJ if ceilJ else None
        ops_read[pid] = {"rho_judge_truth": round(rho, 3),
                         "frac_of_judge_ceiling": round(frac, 3),
                         "verdict": "computation" if frac and frac >= 0.5 else "evidence"}
        print(f"op-type {pid}: judge~truth={rho:+.3f} ({frac:.2f} of judge ceiling) "
              f"-> {ops_read[pid]['verdict']} (designed: {TRUTH_TYPE[pid]})")
    report["op_type_readout"] = ops_read

    json.dump(report, open(OUT / "killswitch_report.json", "w"), indent=1)
    print(f"-> {OUT/'killswitch_report.json'}")


if __name__ == "__main__":
    main()
