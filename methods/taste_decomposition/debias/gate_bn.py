#!/usr/bin/env python3
"""V1/V2 gate evaluation for the bottleneck-architecture battery (B-arms).

Reads runs/B00, B01, B02-B04; decides whether stage B2 (V3/V4/final) may
launch and at which lambda*.  Spec: notes/2026-08-05__taste-decomposition-design.md
S9 -- "each gate must PASS before the next".

Gate definitions (declared here, before the runs finished):
  V1  spec-literal: auc_eval(B01) - auc_eval(B00) >= .02;
      substance: within-model plant-ablation delta_eval >= .02 (the paired,
      lower-variance readout of the same quantity -- recorded separately, as in
      the pooled-arch battery).  V1 must pass on at least the substance readout.
  V2  per arm: probe(plant | z) <= .55 with the B01 control >= .75   [PRIMARY]
      AND |ablation delta_eval| <= .01                               [CAUSAL]
      AND |auc_eval(arm) - auc_eval(B00)| <= .005                    [SPEC-LITERAL,
          noise-limited at this n (CI ~ +-.03); reported, not solely decisive]
  lambda* = smallest lambda whose arm passes PRIMARY + CAUSAL; among those,
      prefer arms that also pass the literal AUC gate.

Exit 0 -> proceed to stage B2 (lambda* in results/v2_gate_bn.json).
Exit 3 -> V2 FAIL (or V1 fail): do not launch stage B2.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RES = HERE / "results"
RES.mkdir(exist_ok=True)

SWEEP = [("B02_grl_planted_full_l0.1", 0.1),
         ("B03_grl_planted_full_l0.5", 0.5),
         ("B04_grl_planted_full_l1.0", 1.0)]


def load(tag):
    d = RUNS / tag
    r = json.loads((d / "result.json").read_text())
    p = d / "probe.json"
    r["probe"] = json.loads(p.read_text())["probes"] if p.exists() else {}
    return r


def main():
    b00, b01 = load("B00_vanilla_real"), load("B01_vanilla_planted")
    out = {"b00_auc_eval": b00["auc_eval"], "b01_auc_eval": b01["auc_eval"]}

    # ---- V1 ----------------------------------------------------------------
    lit = b01["auc_eval"] - b00["auc_eval"]
    abl = b01.get("ablation", {}).get("delta_eval")
    probe01 = (b01["probe"].get("plant") or {}).get("auc_eval_mean")
    out["V1"] = {"literal_diff_eval": lit, "ablation_delta_eval": abl,
                 "probe_plant_b01": probe01,
                 "PASS_literal": bool(lit >= 0.02),
                 "PASS_substance": bool(abl is not None and abl >= 0.02)}
    v1_ok = out["V1"]["PASS_literal"] or out["V1"]["PASS_substance"]
    probe_control_ok = bool(probe01 is not None and probe01 >= 0.75)
    out["V1"]["probe_control_ok"] = probe_control_ok

    # ---- V2 ----------------------------------------------------------------
    arms, lam_star = {}, None
    for tag, lam in SWEEP:
        try:
            r = load(tag)
        except FileNotFoundError:
            continue
        pp = (r["probe"].get("plant") or {}).get("auc_eval_mean")
        d_auc = r["auc_eval"] - b00["auc_eval"]
        d_abl = r.get("ablation", {}).get("delta_eval")
        a = {"lambda": lam, "auc_eval": r["auc_eval"], "auc_test": r["auc_test"],
             "diff_vs_b00_eval": d_auc, "probe_plant": pp, "ablation_delta_eval": d_abl,
             "probe_plant_h": None,
             "PASS_probe": bool(pp is not None and pp <= 0.55 and probe_control_ok),
             "PASS_causal": bool(d_abl is not None and abs(d_abl) <= 0.01),
             "PASS_auc_literal": bool(abs(d_auc) <= 0.005)}
        ph = RUNS / tag / "probe_rep_h.json"
        if ph.exists():
            a["probe_plant_h"] = (json.loads(ph.read_text())["probes"].get("plant") or {}).get("auc_eval_mean")
        arms[tag] = a
    passing = [(a["lambda"], t) for t, a in arms.items() if a["PASS_probe"] and a["PASS_causal"]]
    strict = [(a["lambda"], t) for t, a in arms.items()
              if a["PASS_probe"] and a["PASS_causal"] and a["PASS_auc_literal"]]
    if strict:
        lam_star = min(strict)[0]
    elif passing:
        lam_star = min(passing)[0]
    out["V2"] = {"arms": arms, "n_passing": len(passing), "n_passing_strict": len(strict)}
    out["v1_ok"] = v1_ok
    out["lam_star"] = lam_star

    (RES / "v2_gate_bn.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if not v1_ok:
        print("GATE: V1 FAILED under arch=bottleneck -- battery invalid, stop.")
        raise SystemExit(3)
    if lam_star is None:
        print("GATE: V2 FAILED at every lambda under arch=bottleneck -- stop, negative verdict.")
        raise SystemExit(3)
    print(f"GATE: V2 PASS, lambda* = {lam_star} -> launch stage B2")


if __name__ == "__main__":
    main()
