#!/usr/bin/env python3
"""F2: recompute ONLY the Z bound / verdict / judge-family of an existing E-value block.

Why this exists rather than re-running f2_evalue.py: X, Y and RR are MASS-INDEPENDENT
(X comes from the adversarial sweep against the conditioning block; Y from the found
channels). A corrected missing mass moves only Z, the verdict, and the provenance. So
the expensive sweep is never repeated -- the stored X is reused verbatim.

Corrected strict-B resolver (2026-08-11): a TOP-LEVEL `b_merge` block is the strict
merge certificate. Campaigns write it inconsistently -- `strict: true` on
jokes/cap_finalist, only `n_merge_edges_strict` on mathse_vote/mathse_accepted -- so the
previous inner-flag test mislabelled already-strict masses as tau-era. Two further
strict shapes are now read: nc_responded's `round<N>_species_b.json` two-judge merge and
cw_community's `round<N>_species.json` blind partition (both give Good-Turing as f1/N).

Every Z output now records the JUDGE FAMILY of the mass it consumed: the strict-B corpus
is heterogeneous, so cross-cell Z comparisons need it documented.

Idempotent. CPU only, seconds. Usage: python3 f2_rez.py [--cell X ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


EV = _mod(HERE / "f2_evalue.py", "f2_evalue_rez")


def redo(cell):
    p = RESULTS / f"f2_deconf_{cell}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    found = [c["alone_auc"] for c in d.get("top_nuisance_channels", [])]
    mass = EV.find_mass(cell)
    out = []
    for key in ("evalue_analog", "evalue_analog_matched"):
        blk = d.get(key)
        if not blk:
            continue
        prev_Z, prev_verdict = blk.get("Z"), blk.get("verdict")
        prev_mass = (blk.get("missing_mass") or {})
        blk["missing_mass"] = mass
        blk["Z_recomputed_2026_08_11"] = {
            "reason": ("strict-B resolver corrected: a top-level b_merge block is the "
                       "strict merge certificate; two further strict shapes "
                       "(species_b two-judge merge, blind partition) are now read"),
            "previous_M_hat": prev_mass.get("M_hat"),
            "previous_strict_flag": prev_mass.get("strict_marker_present"),
            "previous_Z": prev_Z, "previous_verdict": prev_verdict,
            "X_unchanged": True,
            "X_note": "X, Y and RR are mass-independent and are reused verbatim",
        }
        X = blk.get("X")
        if X is None and blk.get("X_lower_bound_exceeds") is not None:
            X = blk["X_lower_bound_exceeds"]
        if blk.get("verdict") == "n/a" or X is None:
            blk["Z"] = None
            blk["verdict"] = blk.get("verdict", "n/a")
            out.append((key, blk["verdict"], None, None))
            continue
        if not mass.get("available"):
            blk["Z"] = None
            blk["verdict"] = "Z_UNAVAILABLE"
            blk["verdict_reason"] = ("no Track-B Good-Turing missing mass on disk for "
                                     "this cell; X and RR stand, the M-hat-coupled "
                                     "bound does not")
            out.append((key, blk["verdict"], None, None))
            continue
        zb = EV.z_bound(found, mass["M_hat"],
                        mass.get("S_obs_reported") or len(found))
        blk["Z_detail"] = zb
        blk["Z"] = zb["Z"]
        blk["verdict"] = "ROBUST" if X > zb["Z"] else "ABSORBABLE-IN-PRINCIPLE"
        blk["verdict_reason"] = (f"X {'>' if X > zb['Z'] else '<='} Z "
                                 f"({X:.4f} vs {zb['Z']:.4f})")
        blk["Z_judge_family"] = {
            "judge_family": mass.get("judge_family"),
            "judge_labelled": mass.get("judge_labelled"),
            "judges_raw": mass.get("judges_raw"),
            "merge_certificate": mass.get("merge_certificate"),
            "source": mass.get("source"),
            "caveat": ("the strict-B corpus is heterogeneous across cells (Sonnet on "
                       "jokes/mathse_vote, GPT-5+GLM on the caption cells, unlabelled "
                       "elsewhere); cross-cell Z comparisons must carry the judge family"),
        }
        if mass.get("TAU_ERA_MASS"):
            blk["Z_FLAG"] = ("TAU_ERA_MASS -- this cell's B-side has no strict merge "
                             "certificate; Z is provisional pending the sol+luna certB "
                             "re-survey")
        else:
            blk.pop("Z_FLAG", None)
        out.append((key, blk["verdict"], prev_Z, zb["Z"]))
    p.write_text(json.dumps(d, indent=2, default=str))
    return {"mass": mass, "blocks": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", default=None)
    args = ap.parse_args()
    cells = args.cell or sorted(q.stem.replace("f2_deconf_", "")
                                for q in RESULTS.glob("f2_deconf_*.json"))
    for c in cells:
        r = redo(c)
        if not r:
            print(f"  [{c}] no results file")
            continue
        m = r["mass"]
        tag = ("STRICT" if m.get("strict_marker_present") else
               ("tau-era" if m.get("available") else "UNAVAILABLE"))
        jf = m.get("judge_family", "—")
        for key, verdict, pz, nz in r["blocks"]:
            pzs = "—" if pz is None else f"{pz:.4f}"
            nzs = "—" if nz is None else f"{nz:.4f}"
            print(f"  [{c}/{key.replace('evalue_analog','EV')}] mass {tag} "
                  f"M={m.get('M_hat')} judges={jf} | Z {pzs} -> {nzs} | {verdict}")


if __name__ == "__main__":
    main()
