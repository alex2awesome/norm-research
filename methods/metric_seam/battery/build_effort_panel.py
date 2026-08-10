"""Freeze the compiler-effort-ladder panel (WS1, 2026-07-10 runbook).

Pre-registered selection rule (BEFORE any ladder data): from cam_profile.json per_criterion
(r_hyb = ceiling-normed certified hybrid, r_base = code floor), per core fleet task pick
  2 CONTROL   r_hyb >= .70 (top-2 by r_hyb)          — certified; ladder must not degrade them
  4 MID       .15 <= r_hyb < .55 (evenly spaced)     — headroom where movement is detectable
  2 FLOOR     r_hyb < .15 (lowest 2)                 — current tacit candidates
Ties break by aspect id (stable). Plants p901/p903/p905/p907 ride every rung on PR items.
-> outputs/metric_seam_pilot/battery/effort_ladder/panel_freeze.json
"""
import hashlib, json, pathlib

BASE = pathlib.Path(__file__).resolve().parents[3] / "outputs/metric_seam_pilot"
TASKS = ["press_releases", "creative_writing", "math", "humor", "legal_title_vii"]
PLANTS = ["p901", "p903", "p905", "p907"]


def pick(pc):
    pc = sorted(pc, key=lambda r: (r["r_hyb"], r["aspect"]))
    ctl = [r for r in pc if r["r_hyb"] >= .70][-2:]
    mid = [r for r in pc if .15 <= r["r_hyb"] < .55]
    if len(mid) > 4:  # evenly spaced across the mid range
        idx = [round(i * (len(mid) - 1) / 3) for i in range(4)]
        mid = [mid[i] for i in sorted(set(idx))]
    floor = [r for r in pc if r["r_hyb"] < .15][:2]
    return {"control": ctl, "mid": mid, "floor": floor}


def main():
    raw = open(BASE / "cam_profile.json", "rb").read()
    cam = json.loads(raw)
    panel = {t: pick(cam[t]["per_criterion"]) for t in TASKS}
    out = {"rule": __doc__.strip().splitlines()[2:9], "cam_sha1": hashlib.sha1(raw).hexdigest(),
           "plants": PLANTS, "panel": panel,
           "n_total": sum(len(v) for p in panel.values() for v in p.values())}
    o = BASE / "battery/effort_ladder/panel_freeze.json"
    o.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(o, "w"), indent=1)
    for t, p in panel.items():
        print(t, {k: [r["aspect"] for r in v] for k, v in p.items()})
    print(f"n={out['n_total']} + {len(PLANTS)} plants  sha1={out['cam_sha1'][:12]} -> {o}")


if __name__ == "__main__":
    main()
