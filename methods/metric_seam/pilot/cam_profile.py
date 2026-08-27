"""Certified Articulable Mass (CAM) — historical per-task label (roadmap-v2, 2026-07-04).

Terminology note (reconstruction-v2, 2026-07-12): this statistic measures certified
agreement of materialized code/hybrid programs with a frozen LLM reference.  It is therefore
a *reference-reconstruction* mass, not prompt-based articulability in the v2 vocabulary.
The CAM key/name is retained so historical JSON and figures remain reproducible.

Per criterion, the one-sided certificate is the ceiling-normalized certified floor
r~ = clip[0,1](rho_test / attenuation_ceiling(rel1, K=2)) of its best MATERIALIZED
implementation (description-compiled baseline = lower arm; evolved hybrid = certified arm).
Per task, the survival curve of r~ over judge-measurable criteria; scalars:

  CAM        = mean r~  (= area under the survival curve)
  frac>=.5   = criteria certified above half their ceiling
  frac>=.8   = deep-codability mass

Semantics (lemma-note discipline): r~ is a LOWER bound object — monotone under more search,
never overstates tacitness; 1 - CAM is UNCERTIFIED residual (tacitness + search shortfall +
executor limits mixed), not proven-tacit. Kill-switch plants calibrate the search: one blind
improver round reaches ~76-90% of a criterion's true ceiling (h1 closes to 86-90%), so
r~ / 0.8-0.9 is a calibrated point-estimate band for the reachable ceiling share.
Gap-3 rule: r~ is Spearman-based; near-ceiling reads need the Pearson companion.

-> outputs/metric_seam_pilot/cam_profile.json
"""
import json, math, pathlib, statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"


def ceiling(rel1, k=2):
    r = max(0.0, min(1.0, rel1))
    relk = k * r / (1 + (k - 1) * r)
    return math.sqrt(relk) if relk > 0 else float("nan")


def clip01(x):
    return max(0.0, min(1.0, x))


def profile(pairs):
    """pairs: [(aspect, rho_hyb, rho_base, ceil)] -> per-task profile dict."""
    rows = [{"aspect": a, "r_hyb": round(clip01(h / c), 3), "r_base": round(clip01(b / c), 3)}
            for a, h, b, c in pairs if c == c and c > 0.3]
    rh = [r["r_hyb"] for r in rows]
    rb = [r["r_base"] for r in rows]
    f = lambda v, t: round(sum(x >= t for x in v) / len(v), 3)
    return {
        "n": len(rows),
        "CAM_certified": round(st.mean(rh), 3), "CAM_baseline": round(st.mean(rb), 3),
        "median_certified": round(st.median(rh), 3),
        "frac_ge_.5": f(rh, .5), "frac_ge_.5_baseline": f(rb, .5),
        "frac_ge_.8": f(rh, .8), "frac_ge_.8_baseline": f(rb, .8),
        "per_criterion": sorted(rows, key=lambda r: -r["r_hyb"]),
    }


def main():
    out = {}
    d = json.load(open(BASE / "v2/hybrid_eval_v2.json"))
    out["press_releases"] = profile([
        (a, v["gate"]["rho_mean"], v["baseline"]["rho_test"], v["ceiling"])
        for a, v in d.items() if v.get("gate") and v.get("ceiling")])
    for task in ["creative_writing", "math", "humor", "legal_title_vii",
                 "peer_review", "legal_ss_disability", "humor_units"]:
        p = BASE / "tasks" / task / "hybrid_gate_report.json"
        if not p.exists():
            continue
        r = json.load(open(p))
        out[task] = profile([
            (a, v["full"]["rho_hybrid"], v["full"]["rho_baseline"], ceiling(v["judge_rel1"]))
            for a, v in r.items()
            if isinstance(v.get("full"), dict) and v["full"].get("rho_hybrid") is not None
            and v["full"].get("rho_baseline") is not None
            and v.get("judge_rel1") == v.get("judge_rel1")])
    json.dump(out, open(BASE / "cam_profile.json", "w"), indent=1)
    for t, p in out.items():
        print(f"{t:18s} n={p['n']:2d} CAM {p['CAM_baseline']:.3f}->{p['CAM_certified']:.3f} "
              f"frac>=.5 {p['frac_ge_.5']:.2f} frac>=.8 {p['frac_ge_.8']:.2f}")
    print(f"-> {BASE / 'cam_profile.json'}")


if __name__ == "__main__":
    main()
