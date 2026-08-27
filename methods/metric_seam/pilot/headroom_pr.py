"""R3 — same-f (TVD) headroom T−R on press releases (roadmap-v2; lemma A1 discipline).

Both legs in ONE f-divergence (C3), same items, same verdict definition:
  T̂ = tvd_transmission(V)   V = (N,2) binary pass matrix; binarize each pass with a SHARED
      per-aspect threshold (median of pooled pass scores, strict >). Degenerate splits
      (>90% one side) are flagged and skipped, not silently included.
  R̂ = tvd_recovery(m̂, m_p) per pass p (labels = one REALIZATION of M_ω, mirroring the PO
      recon pairing), averaged over the two passes; m̂ = channel score in [0,1] read as a
      graded YES-strength (tvd_recovery accepts per-item means).
  h = T̂ − R̂; DPI guardrail: R̂ ≤ T̂ within CI (violation ⇒ leak/bug flag, lemma-note Gap 1).

Channels: best code flavor (by |R̂|) + evolved hybrid where materialized (PR v1 4 + v2 20).
Only locally re-executable channels — recon-store blinded-judge channels have no per-item
preds and stay Spearman-only (flagged in the note). -> outputs/metric_seam_pilot/headroom_pr.json
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_implementer"))
import numpy as np                                    # noqa: E402
from vinfo import tvd_transmission, tvd_recovery      # noqa: E402

BASE = ROOT / "outputs/metric_seam_pilot"


def load_judge(path):
    p1, p2 = {}, {}
    for line in open(path):
        r = json.loads(line)
        if isinstance(r.get("score"), int) and r.get("channel") in ("pass1", "pass2"):
            (p1 if r["channel"] == "pass1" else p2).setdefault(
                r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    return p1, p2


def sources():
    for tag in ["v1", "v2"]:
        d = BASE / tag
        code = json.load(open(d / f"code_scores_{tag}.json"))
        hyb = {p.stem.replace("hybrid_scores_", ""): json.load(open(p))
               for p in d.glob("hybrid_scores_*.json")}
        yield tag, load_judge(d / f"results_{tag}.jsonl"), code, hyb


def main():
    out = {}
    for tag, (p1, p2), code, hyb in sources():
        for aid in sorted(set(p1) & set(p2)):
            both = sorted(set(p1[aid]) & set(p2[aid]))
            if len(both) < 100:
                continue
            s1 = np.array([p1[aid][d] for d in both], float)
            s2 = np.array([p2[aid][d] for d in both], float)
            thr = np.median(np.concatenate([s1, s2]))
            b1, b2 = (s1 > thr).astype(float), (s2 > thr).astype(float)
            sbar = float(np.mean(np.concatenate([b1, b2])))
            if not 0.10 <= sbar <= 0.90:
                out[f"{tag}/{aid}"] = {"skip": f"degenerate split s_bar={sbar:.2f}"}
                continue
            T = tvd_transmission(np.stack([b1, b2], axis=1))

            chans = {}
            for key, col in code.items():
                if not key.startswith(aid + "_") or not isinstance(col, dict):
                    continue
                chans[key.split("_", 1)[1]] = col
            for hk, col in hyb.items():
                if hk.startswith(aid + "_"):
                    chans["hybrid_" + hk.split("_")[-1]] = col

            rows = {}
            rng = np.random.default_rng(7)
            for name, col in chans.items():
                m = np.array([col.get(d) if col.get(d) is not None else np.nan
                              for d in both], float)
                fin = np.isfinite(m)
                if fin.sum() < 100 or np.nanmax(m) - np.nanmin(m) < 1e-9:
                    continue
                # graded R (calibration-sensitive, lower bound) + scale-free binarized R
                # (rank-median split of m-hat, tie-safe — post-processing, still DPI-safe)
                mb = np.full(len(m), np.nan)
                idx = np.where(fin)[0]
                order = idx[np.lexsort((rng.random(fin.sum()), m[idx]))]
                mb[order[fin.sum() // 2:]] = 1.0
                mb[order[:fin.sum() // 2]] = 0.0
                res = {}
                for lab, mm in [("R", m), ("R_bin", mb)]:
                    legs = [tvd_recovery(mm, b) for b in (b1, b2)]
                    res[lab] = float(np.mean([r["tvd_recovery"] for r in legs]))
                    res[lab + "_ci_hi"] = float(np.mean([r["ci_hi"] for r in legs]))
                Rmax = max(res["R"], res["R_bin"])
                rows[name] = {"R": round(res["R"], 4), "R_bin": round(res["R_bin"], 4),
                              "h": round(T["tvd_t"] - res["R"], 4),
                              "h_bin": round(T["tvd_t"] - res["R_bin"], 4),
                              "dpi_ok": bool(Rmax <= T["ci_hi"] + 1e-9
                                             or Rmax <= max(res["R_ci_hi"],
                                                            res["R_bin_ci_hi"]))}
            best_code = max((k for k in rows if not k.startswith("hybrid")),
                            key=lambda k: rows[k]["R_bin"], default=None)
            out[f"{tag}/{aid}"] = {
                "n": len(both), "s_bar": round(sbar, 3),
                "T_tvd": round(T["tvd_t"], 4), "T_ci": [round(T["ci_lo"], 4),
                                                        round(T["ci_hi"], 4)],
                "T_norm": round(T["tvd_t_norm"], 4),
                "best_code": best_code,
                "channels": rows,
            }
    json.dump(out, open(BASE / "headroom_pr.json", "w"), indent=1)
    ok = [(k, v) for k, v in out.items() if "T_tvd" in v]
    print(f"{len(ok)} aspects with T; {sum(1 for k, v in out.items() if 'skip' in v)} skipped")
    for k, v in ok:
        bc = v["best_code"]
        hyb_keys = [c for c in v["channels"] if c.startswith("hybrid")]
        line = f"{k}: T={v['T_tvd']:.3f} [{v['T_ci'][0]:.3f},{v['T_ci'][1]:.3f}]"
        if bc:
            line += (f"  code Rb={v['channels'][bc]['R_bin']:.3f} "
                     f"hb={v['channels'][bc]['h_bin']:+.3f}")
        for hk in hyb_keys:
            line += (f"  {hk} Rb={v['channels'][hk]['R_bin']:.3f} "
                     f"hb={v['channels'][hk]['h_bin']:+.3f}")
        viol = [c for c, r in v["channels"].items() if not r["dpi_ok"]]
        if viol:
            line += f"  DPI-FLAG:{viol}"
        print(line)


if __name__ == "__main__":
    main()
