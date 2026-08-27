"""R1 money figure — cross-task codability brackets + CAM survival curves (roadmap-v2).

Panel A: per-criterion bracket [description-compiled floor -> evolved-certified upper], both
ceiling-normalized (r~ = clip01(rho/attenuation ceiling)), for the three tasks with both arms
(PR held-out gates, CW/math held-out gates). Marker color = gate status.
Panel B: CAM survival curves (certified solid, baseline dashed) + recon-R annotations.
Panel C: lower-arm-only floors for the survey-only corpora (survey-grade full-sample, flagged),
annotated with dominant op-type diagnosis.

-> outputs/metric_seam_pilot/figures/money_bracket.png (+ .pdf)
"""
import json, math, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)


def ceiling(rel1, k=2):
    r = max(0.0, min(1.0, rel1))
    relk = k * r / (1 + (k - 1) * r)
    return math.sqrt(relk) if relk > 0 else float("nan")


def clip01(x):
    return max(0.0, min(1.0, x))


def spearman(a, b):
    n = len(a)
    if n < 3:
        return float("nan")
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


# ---------------------------------------------------------------- fleet tasks (both arms)
def fleet_rows():
    rows = {}
    d = json.load(open(BASE / "v2/hybrid_eval_v2.json"))
    rows["press_releases"] = [
        {"aspect": a, "base": clip01(v["baseline"]["rho_test"] / v["ceiling"]),
         "hyb": clip01(v["gate"]["rho_mean"] / v["ceiling"]),
         "P_gate": v["gate"]["P_gate"], "P_beats": v["gate"]["P_beats_baseline"]}
        for a, v in d.items() if v.get("gate") and v.get("ceiling")]
    for task in ["math", "creative_writing", "humor", "legal_title_vii"]:
        p = BASE / "tasks" / task / "hybrid_gate_report.json"
        if not p.exists():
            continue
        r = json.load(open(p))
        rows[task] = [
            {"aspect": a, "base": clip01(v["full"]["rho_baseline"] / c),
             "hyb": clip01(v["full"]["rho_hybrid"] / c),
             "P_gate": v["full"]["P_gate"] or 0.0, "P_beats": v["full"]["P_beats_baseline"] or 0.0}
            for a, v in r.items()
            if isinstance(v.get("full"), dict) and v["full"].get("rho_hybrid") is not None
            and v["full"].get("rho_baseline") is not None
            for c in [ceiling(v["judge_rel1"])] if c == c and c > 0.3]
    return rows


# ---------------------------------------------------------------- survey-only floors
def survey_floor(task):
    """Best-flavor full-sample r~ per judge-measurable aspect (survey grade, no held-out)."""
    p1, p2 = {}, {}
    for line in open(BASE / "tasks" / task / "results.jsonl"):
        r = json.loads(line)
        if isinstance(r.get("score"), int) and r.get("channel") in ("pass1", "pass2"):
            (p1 if r["channel"] == "pass1" else p2).setdefault(
                r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    code = json.load(open(BASE / "tasks" / task / "code_scores.json"))
    by_aspect = {}
    for key, col in code.items():
        if not isinstance(col, dict):
            continue
        aid = key.split("_")[0]
        if aid not in p1 or aid not in p2:
            continue
        both = sorted(set(p1[aid]) & set(p2[aid]) & {d for d, v in col.items() if v is not None})
        if len(both) < 30:
            continue
        v1 = [float(p1[aid][d]) for d in both]
        v2 = [float(p2[aid][d]) for d in both]
        rel1 = spearman(v1, v2)
        if rel1 != rel1 or rel1 <= 0.1:
            continue
        c = ceiling(rel1)
        if c != c or c <= 0.3:
            continue
        jm = [(a + b) / 2 for a, b in zip(v1, v2)]
        cv = [float(col[d]) for d in both]
        if max(cv) - min(cv) < 1e-12:
            continue
        rho = spearman(cv, jm)
        if rho == rho:
            by_aspect[aid] = max(by_aspect.get(aid, 0.0), clip01(rho / c))
    return sorted(by_aspect.values(), reverse=True)


SURVEY_TASKS = {          # dominant op-type diagnosis (results note, cross-task op-map)
    "code_review": "evidence-starved (no diff in X)",
    "code_review_diffs": "evidence present; OOD codegen",
    "code_competition": "computation (exec anchor)",
    "patents": "evidence-dominant (prior art)",
    "pr_exec": "evidence op (exec mock)",
}
RECON_R = {"press_releases": .471, "math": .318,          # recon sweep medians (results note
           "code_review": .240, "patents": .197}          # §RECONSTRUCTION, GLM-5.2)


def main():
    rows = fleet_rows()
    cam = json.load(open(BASE / "cam_profile.json"))
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.34, wspace=0.22)

    # ---- Panel A: brackets ------------------------------------------------
    axA = fig.add_subplot(gs[0, :])
    xoff, xticks, xlabels = 0, [], []
    order = [t for t in ["press_releases", "legal_title_vii", "math", "creative_writing", "humor"] if t in rows]
    for task in order:
        rr = sorted(rows[task], key=lambda r: -r["hyb"])
        for i, r in enumerate(rr):
            x = xoff + i
            color = ("#1a7f37" if r["P_gate"] >= .5
                     else "#d29922" if r["P_beats"] >= .95 else "#cf222e")
            axA.plot([x, x], [r["base"], r["hyb"]], color="#9a9a9a", lw=1.1, zorder=1)
            axA.plot(x, r["base"], marker="_", color="#57606a", ms=7, mew=1.6, zorder=2)
            axA.plot(x, r["hyb"], marker="o", color=color, ms=4.5, zorder=3)
        xticks.append(xoff + len(rr) / 2 - .5)
        n_cert = sum(r["P_gate"] >= .5 for r in rr)
        xlabels.append(f"{task}\n(n={len(rr)}, {n_cert} certified)")
        xoff += len(rr) + 4
    axA.set_ylim(-0.02, 1.02)
    axA.set_xticks(xticks)
    axA.set_xticklabels(xlabels, fontsize=9)
    axA.set_ylabel("r̃ = ρ / attenuation ceiling (held-out)")
    axA.set_title("Codability brackets: description-compiled floor (–) → evolved-certified (●), "
                  "per criterion; green = gate-certified (P≥.5), amber = beats baseline (P≥.95), red = neither")
    axA.axhline(1.0, color="#eee", lw=.8)
    axA.grid(axis="y", alpha=.25)

    # ---- Panel B: CAM survival curves ------------------------------------
    axB = fig.add_subplot(gs[1, 0])
    colors = {"press_releases": "#0969da", "legal_title_vii": "#1a7f37", "math": "#8250df", "creative_writing": "#bf3989",
              "humor": "#bc4c00"}
    for task in order:
        if task not in cam:
            continue
        p = cam[task]
        for key, style in [("r_hyb", "-"), ("r_base", "--")]:
            vals = sorted((r[key] for r in p["per_criterion"]), reverse=True)
            xs = [i / len(vals) for i in range(len(vals) + 1)]
            ys = [1.0] + vals if False else vals + [vals[-1]]
            axB.step([i / len(vals) for i in range(1, len(vals) + 1)], vals,
                     style, color=colors[task], lw=1.8 if style == "-" else 1.1,
                     alpha=1.0 if style == "-" else .55,
                     label=f"{task} (CAM {p['CAM_certified']:.2f})" if style == "-" else None)
        if task in RECON_R:
            axB.annotate(f"recon-R {RECON_R[task]:.2f}", xy=(.99, RECON_R[task]),
                         xycoords=("axes fraction", "data"), fontsize=7,
                         color=colors[task], ha="right", va="bottom")
    axB.set_xlabel("fraction of judge-measurable criteria")
    axB.set_ylabel("certified r̃ (survival)")
    axB.set_title("CAM survival curves (solid = evolved, dashed = floor)")
    axB.legend(fontsize=8, loc="upper right")
    axB.set_ylim(0, 1.02)
    axB.grid(alpha=.25)

    # ---- Panel C: survey-only floors --------------------------------------
    axC = fig.add_subplot(gs[1, 1])
    y = 0
    for task, diag in SURVEY_TASKS.items():
        vals = survey_floor(task)
        if not vals:
            continue
        axC.boxplot([vals], positions=[y], vert=False, widths=.6, showfliers=False)
        med = sorted(vals)[len(vals) // 2]
        lab = f"{task} (n={len(vals)}, med {med:.2f})"
        if task in RECON_R:
            lab += f", recon-R {RECON_R[task]:.2f}"
        axC.text(1.02, y, diag, fontsize=7, va="center", color="#57606a")
        axC.text(-0.02, y - .48, lab, fontsize=8, va="top")
        y += 1
    axC.set_xlim(-0.02, 1.45)
    axC.set_ylim(-0.85, y - 0.35)
    axC.set_yticks([])
    axC.set_xlabel("floor r̃ (survey-grade, full-sample) — upper arm held/blocked")
    axC.set_title("Lower-arm-only corpora (op-type diagnosis right)")
    axC.axvline(1.0, color="#eee", lw=.8)
    axC.grid(axis="x", alpha=.25)

    fig.suptitle("Metric-seam frontier: how much of each frozen LLM reference is certified "
                 "reconstructable (historical CAM; scoped-gate rule 2026-07-04)",
                 fontsize=11, y=.995)
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"money_bracket.{ext}", dpi=180, bbox_inches="tight")
    print(f"-> {FIG / 'money_bracket.png'}")


if __name__ == "__main__":
    main()
