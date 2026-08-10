#!/usr/bin/env python3
"""Emit the per-run JSON configs for the debias-pilot chain (stage 1 = V1/V2
sweep; stage 2 = V3/V4/final, which need the lambda picked from stage 1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SK3 = "/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment/debias_pilot"
D = f"{SK3}/build"
OUT = f"{SK3}/runs"
PLANT, REALTOK = "⟦QX7⟧", "⟦RS4⟧"


ARCH = None  # set from --arch; "bn" -> tags R->B + arch=bottleneck in every config


def cfg(tag, corpus, channels, lam, ablate=None, subset=None, note=""):
    if ARCH == "bn":
        tag = "B" + tag[1:]
    c = {"tag": tag, "corpus": f"{D}/{corpus}", "nuisance": f"{D}/nuisance.npz",
         "channels": channels, "lambda_adv": lam, "out_dir": OUT, "note": note}
    if ARCH == "bn":
        c["arch"] = "bottleneck"
        c["note"] = ("[arch=bottleneck: task head + adversary both read z=proj(h); "
                     "GRL shapes proj; V2-failure fix] " + note)
    if ablate:
        c["ablate_token"] = ablate
    if subset:
        c["subset_ids_from"] = f"{D}/{subset}"
    return c


def stage1():
    return [
        cfg("R00_vanilla_real", "population.csv", [], 0.0,
            note="unplanted vanilla baseline; V1 reference, V4 reference, final reference"),
        cfg("R01_vanilla_planted", "corpus_planted.csv", [], 0.0, ablate=PLANT,
            note="V1 EXPLOIT: vanilla on the planted corpus"),
        cfg("R02_grl_planted_full_l0.1", "corpus_planted.csv", ["standard", "plant"], 0.1, ablate=PLANT,
            note="V2 REMOVAL literal spec (standard nuisances + plant indicator), lambda .1"),
        cfg("R03_grl_planted_full_l0.5", "corpus_planted.csv", ["standard", "plant"], 0.5, ablate=PLANT,
            note="V2 REMOVAL, lambda .5"),
        cfg("R04_grl_planted_full_l1.0", "corpus_planted.csv", ["standard", "plant"], 1.0, ablate=PLANT,
            note="V2 REMOVAL, lambda 1.0"),
    ]


def stage2(lam):
    return [
        cfg("R05_grl_planted_plantonly", "corpus_planted.csv", ["plant"], lam, ablate=PLANT,
            note="V2 DIAGNOSTIC: plant indicator named ALONE, so the AUC gate is not "
                 "confounded by legitimate removal of the standard nuisances"),
        cfg("R06_vanilla_realtok", "corpus_realtok.csv", [], 0.0, ablate=REALTOK,
            note="V3a reference: vanilla on the real-signal-token corpus"),
        cfg("R07_grl_realtok_standard", "corpus_realtok.csv", ["standard"], lam, ablate=REALTOK,
            note="V3a SPECIFICITY: debias the STANDARD nuisances only; the unnamed "
                 "real-signal token must survive"),
        cfg("R08_vanilla_v3b", "population.csv", [], 0.0, subset="v3b_population.csv",
            note="V3b reference: vanilla on the year-balanced subsample (date-alone AUC .500)"),
        cfg("R09_grl_v3b_date", "population.csv", ["date"], lam, subset="v3b_population.csv",
            note="V3b SPECIFICITY: debias a channel the model verifiably cannot use for y"),
        cfg("R10_grl_length_real", "population.csv", ["length"], lam,
            note="V4 CONSISTENCY: length channel alone, real unplanted corpus"),
        cfg("R11_grl_full_real", "population.csv", ["standard"], lam,
            note="FINAL: full standard nuisance vector, real corpus"),
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--arch", choices=["pooled", "bn"], default="pooled")
    a = ap.parse_args()
    ARCH = "bn" if a.arch == "bn" else None
    d = HERE / "configs"
    d.mkdir(exist_ok=True)
    runs = stage1() if a.stage == 1 else stage2(a.lam)
    for c in runs:
        (d / f"{c['tag']}.json").write_text(json.dumps(c, indent=2))
        print(c["tag"], c["channels"], c["lambda_adv"])
    pre = "stageB" if a.arch == "bn" else "stage"
    (d / f"{pre}{a.stage}_order.txt").write_text("\n".join(c["tag"] for c in runs) + "\n")
