"""TVD-MI <-> Spearman bridge calibration slice (roadmap-v2 R2, lemma-note Gap 9).

The seam gates/ceilings run on Spearman (Lemma A2 stack); the PO machinery runs on TVD-MI
(Lemma A1 stack). No inequality connects them (lemma note Gap 9), so cross-stack sentences are
directional prose. This script builds the EMPIRICAL correspondence: for every surveyed
(aspect x channel) pair with a 2-pass Gemma judge target, compute Spearman rho, Pearson r, and
TVD-MI (vinfo.tvd_mi, rank-median split, permutation-debiased) between the channel column and
the 2-pass judge mean, plus the judge's own pass-pair reliabilities in both currencies
(rel1 Spearman vs TVD-MI(pass1, pass2)).

Also settles Gap 3's empirical half: the Spearman-vs-Pearson slack distribution at survey scale
(T5 measured ~.01-.03 on one task; here it's every channel of every survey).

Sources: PR v1/v2/v3 (incl. saved per-item hybrid columns v1+v2) + 7 task surveys (code flavors).
-> outputs/metric_seam_pilot/bridge_calibration.json
"""
import json, math, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_implementer"))
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from vinfo import tvd_mi            # noqa: E402
from certificates import spearman   # noqa: E402

BASE = ROOT / "outputs/metric_seam_pilot"
MIN_N = 30


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    ca = [x - ma for x in a]; cb = [y - mb for y in b]
    va = sum(x * x for x in ca); vb = sum(y * y for y in cb)
    if va <= 0 or vb <= 0:
        return float("nan")
    return sum(x * y for x, y in zip(ca, cb)) / math.sqrt(va * vb)


def load_judge(path):
    """-> {aspect: (p1 dict, p2 dict)} for int-scored pass1/pass2 rows."""
    p1, p2 = {}, {}
    for line in open(path):
        r = json.loads(line)
        if not isinstance(r.get("score"), int) or r.get("channel") not in ("pass1", "pass2"):
            continue
        (p1 if r["channel"] == "pass1" else p2).setdefault(
            r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    return {a: (p1[a], p2[a]) for a in set(p1) & set(p2)}


def sources():
    yield "pr_v1", BASE / "v1/results_v1.jsonl", BASE / "v1/code_scores_v1.json", \
        sorted((BASE / "v1").glob("hybrid_scores_*.json"))
    yield "pr_v2", BASE / "v2/results_v2.jsonl", BASE / "v2/code_scores_v2.json", \
        sorted((BASE / "v2").glob("hybrid_scores_*.json"))
    yield "pr_v3", BASE / "v3/results_v3.jsonl", BASE / "v3/code_scores_v3.json", []
    for t in sorted(p.name for p in (BASE / "tasks").iterdir() if p.is_dir()):
        d = BASE / "tasks" / t
        if (d / "results.jsonl").exists() and (d / "code_scores.json").exists():
            yield t, d / "results.jsonl", d / "code_scores.json", []


def main():
    rows = []
    for src, res_path, code_path, hyb_files in sources():
        judge = load_judge(res_path)
        cols = {k: v for k, v in json.load(open(code_path)).items()}
        for hf in hyb_files:
            aid_h = hf.stem.replace("hybrid_scores_", "")           # e.g. a86_h0
            cols[aid_h + "__hybrid"] = json.load(open(hf))
        for key, col in cols.items():
            aid = key.split("_")[0]
            if aid not in judge or not isinstance(col, dict):
                continue                                            # broken-codegen column
            p1, p2 = judge[aid]
            both = sorted(set(p1) & set(p2) & {d for d, v in col.items() if v is not None})
            if len(both) < MIN_N:
                continue
            v1 = [float(p1[d]) for d in both]
            v2 = [float(p2[d]) for d in both]
            jm = [(a + b) / 2.0 for a, b in zip(v1, v2)]
            cv = [float(col[d]) for d in both]
            if max(cv) - min(cv) < 1e-12:
                continue                                            # constant channel
            rel1_s = spearman(v1, v2)
            if rel1_s != rel1_s or rel1_s <= 0:
                continue                                            # unusable judge target
            rel2 = 2 * rel1_s / (1 + rel1_s)                        # Spearman-Brown K=2
            rho_s = spearman(cv, jm)
            r_p = pearson(cv, jm)
            rows.append({
                "source": src, "channel": key, "n": len(both),
                "kind": "hybrid" if key.endswith("__hybrid") else "code",
                "rel1_spearman": round(rel1_s, 4),
                "rel_tvd": round(tvd_mi(v1, v2), 4),
                "rho_spearman": round(rho_s, 4),
                "r_pearson": round(r_p, 4) if r_p == r_p else None,
                "rho_tilde": round(rho_s / math.sqrt(rel2), 4),
                "tvd_mi": round(tvd_mi(cv, jm), 4),
                "tvd_mi_5bin": round(tvd_mi(cv, jm, n_bins=5), 4),
            })
        print(f"{src}: {sum(r['source'] == src for r in rows)} pairs")

    # ---- summaries -------------------------------------------------------
    ok = [r for r in rows if r["r_pearson"] is not None]
    xs = [abs(r["rho_spearman"]) for r in ok]
    ys = [r["tvd_mi"] for r in ok]
    bridge_mono = spearman(xs, ys)
    slack = sorted(abs(r["rho_spearman"] - r["r_pearson"]) for r in ok)
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
    hi = [abs(r["rho_spearman"] - r["r_pearson"]) for r in ok if abs(r["rho_spearman"]) >= .5]

    dec = {}
    for r in ok:
        b = min(9, int(abs(r["rho_spearman"]) * 10))
        dec.setdefault(b, []).append(r["tvd_mi"])
    decile_tab = {f"{b/10:.1f}-{(b+1)/10:.1f}": {
        "n": len(v), "tvd_mean": round(sum(v) / len(v), 3),
        "tvd_min": round(min(v), 3), "tvd_max": round(max(v), 3)}
        for b, v in sorted(dec.items())}

    summary = {
        "n_pairs": len(ok),
        "bridge_monotonicity_spearman_absrho_vs_tvd": round(bridge_mono, 4),
        "gap3_slack_abs_rhoS_minus_rP": {
            "p50": round(q(slack, .50), 4), "p90": round(q(slack, .90), 4),
            "p99": round(q(slack, .99), 4), "max": round(slack[-1], 4),
            "p90_when_absrho_ge_0.5": round(q(sorted(hi), .90), 4) if hi else None,
            "max_when_absrho_ge_0.5": round(max(hi), 4) if hi else None,
        },
        "tvd_by_absrho_decile": decile_tab,
    }
    out = {"summary": summary, "pairs": rows}
    json.dump(out, open(BASE / "bridge_calibration.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"-> {BASE / 'bridge_calibration.json'}")


if __name__ == "__main__":
    main()
