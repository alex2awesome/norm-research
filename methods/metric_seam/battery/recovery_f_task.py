"""Generalized recovery-objective seam: V/A in recovered BITS for any battery task.

Same channel as recovery_f_pilot.py (canonical recon_channel precomputed-behavior MCQ,
GLM-5.2 reconstructor, identity channel I(J;Jhat) in bits), generalized:

  V column source per task:
    programs   : certified h0 run with EMPTY LLM-field map (pure code+ops)   [legal, CW]
    flavors    : the gate report's train-selected codegen flavor column      [code_review]
  G column     : the task's seed-G per-criterion results (seed_g_extend/)

Usage: python3 recovery_f_task.py <task> [--limit N] [--model glm-5.2]
Writes seed_g_extend/recovery_f_<task>.json
"""
import argparse, json, pathlib, sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/battery"))
sys.path.insert(0, str(ROOT))
import battery_common as bc
from battery_common import load_mod, run_prog

from methods.metric_implementer.backends import LLMBackend
from methods.metric_implementer.config import ImplementerConfig
from methods.metric_implementer.recon_channel import (
    mcq_value_from_precomputed_behavior, mcq_identity_channel)

BASE = ROOT / "outputs/metric_seam_pilot"
OUT = HERE / "seed_g_extend"

CFG = {
    "legal_title_vii": dict(
        noun="U.S. Title VII employment-discrimination court opinion",
        v_source="programs", progdir=ROOT / "methods/metric_seam/hybrids/programs_legal"),
    "creative_writing": dict(
        noun="creative-writing story or story excerpt",
        v_source="programs", progdir=ROOT / "methods/metric_seam/hybrids/programs_cw"),
    "code_review": dict(
        noun="GitHub pull request with its code-review discussion",
        v_source="flavors", progdir=None),
}
bc.PROGDIR.update({"legal_title_vii": "programs_legal", "creative_writing": "programs_cw",
                   "code_review": "programs_code_review"})


def binarize(col, ids):
    """Balanced-cut binarization: threshold at the observed value whose > cut is closest
    to a 50/50 split (label-free, tier-symmetric; robust to compressed distributions
    where the median equals the modal top value)."""
    vals = [col.get(d) for d in ids]
    if any(v is None or not isinstance(v, (int, float)) for v in vals):
        keep = [d for d in ids if isinstance(col.get(d), (int, float))]
        if len(keep) < 0.9 * len(ids):
            return None
        med_fill = float(np.median([col[d] for d in keep]))
        vals = [col[d] if isinstance(col.get(d), (int, float)) else med_fill for d in ids]
    arr = np.asarray(vals, dtype=float)
    cuts = sorted(set(arr))[:-1]  # thresholds between observed values
    if not cuts:
        return None
    best = min(cuts, key=lambda t: abs(float((arr > t).mean()) - 0.5))
    b = (arr > best).astype(float)
    minority = min(int(b.sum()), len(b) - int(b.sum()))
    return b if minority >= 5 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--distractor", default="hard", choices=["hard", "random"],
                    help="hard = highest same-tier behavioral agreement (clones excluded); "
                         "random = seeded uniform draw from non-clone same-tier columns")
    a = ap.parse_args()
    cfg = CFG[a.task]

    ctx = bc.load_ctx(a.task)
    test_ids = sorted(ctx["test"])
    texts = [ctx["items"][d] for d in test_ids]

    # criteria = the post-guard rows of the task's seed-G final json
    final = json.load(open(OUT / f"seed_g_{a.task}_final.json"))
    aids = [r["aid"] for r in final["rows"]]

    cols = {"V": {}, "G": {}}
    if cfg["v_source"] == "programs":
        for aid in aids:
            p = cfg["progdir"] / f"{aid}_h0.py"
            if not p.exists():
                continue
            mod = load_mod(p)
            cols["V"][aid] = run_prog(mod.score, ctx["items"], {}, ctx["ops"])
    else:  # flavors: gate report names the train-selected flavor; columns in code_scores.json
        gate = json.load(open(BASE / f"tasks/{a.task}/hybrid_gate_report.json"))
        cs = json.load(open(BASE / f"tasks/{a.task}/code_scores.json"))
        for aid in aids:
            fl = gate[aid]["full"]["flavor"]
            cols["V"][aid] = cs.get(f"{aid}_{fl}", {})
    for line in open(OUT / f"seed_g_{a.task}_results.jsonl"):
        r = json.loads(line)
        if r.get("aspect_id", "").endswith(".final") and isinstance(r.get("score"), int):
            cols["G"].setdefault(r["aspect_id"].split(".")[1], {})[r["datapoint_id"]] = r["score"]

    packs = {aid: json.load(open(BASE / f"tasks/{a.task}/improver_packs/{aid}.json"))
             for aid in aids}
    desc = {aid: f"{packs[aid]['criterion_name']} — {packs[aid]['criterion_description'][:400]}"
            for aid in aids}

    panel = {t: {} for t in cols}
    for t in cols:
        for aid in aids:
            b = binarize(cols[t].get(aid, {}), test_ids)
            if b is not None:
                panel[t][aid] = b
    print({t: f"{len(panel[t])}/{len(aids)} usable binarized columns" for t in panel})

    icfg = ImplementerConfig()
    icfg.backend = "zai_anthropic"
    icfg.other_temperature = 0.0
    icfg.request_timeout_s = 180
    recon = LLMBackend(model=a.model, role="reconstructor", cfg=icfg)

    rng = np.random.default_rng(13)
    design = np.sort(rng.choice(len(test_ids), size=24, replace=False))

    results = {t: [] for t in panel}
    aid_list = aids[: a.limit] if a.limit else aids
    for tier in panel:
        for aid in aid_list:
            if aid not in panel[tier]:
                continue
            target = panel[tier][aid]
            sims = []
            for other, ocol in panel[tier].items():
                if other == aid:
                    continue
                agree = float((target == ocol).mean())
                if agree >= 0.95:
                    continue
                sims.append((agree, other))
            sims.sort(reverse=True)
            if len(sims) < 3:
                print(f"SKIP {tier}/{aid}: <3 non-clone distractors"); continue
            if a.distractor == "random":
                import hashlib as _hl
                seed = int(_hl.sha256(f"{a.task}|{tier}|{aid}".encode()).hexdigest()[:8], 16)
                drng = np.random.default_rng(seed)
                pick = list(drng.choice([o for _, o in sims], size=3, replace=False))
            else:
                pick = [o for _, o in sims[:3]]
            distractors = [dict(metric_id=o, description=desc[o], scores=panel[tier][o])
                           for o in pick]
            try:
                row = mcq_value_from_precomputed_behavior(
                    recon, noun=cfg["noun"], candidate_prompt_text=f"{tier}:{aid}",
                    target_metric_id=aid, target_description=desc[aid],
                    target_scores=target, probe_texts=texts, distractors=distractors,
                    design_indices=design, codebook_frozen_before_prompt_search=True,
                    n_examples=8, n_reconstruction_draws=4)
                results[tier].append(row)
                print(f"{tier}/{aid}: raw_p={row['raw_target_option_probability']:.2f} "
                      f"value={row['value_mark']:.2f}", flush=True)
            except Exception as e:
                print(f"FAIL {tier}/{aid}: {type(e).__name__}: {str(e)[:120]}", flush=True)

    summary = {}
    for tier, rows in results.items():
        ch = mcq_identity_channel(rows)
        summary[tier] = dict(
            n_rows=len(rows),
            mean_raw_p=float(np.mean([r["raw_target_option_probability"] for r in rows]))
            if rows else None,
            mean_value=float(np.mean([r["value_mark"] for r in rows])) if rows else None,
            identity_bits=ch.get("mutual_information_bits") if ch.get("valid") else None,
            identity_norm=ch.get("normalized_mutual_information") if ch.get("valid") else None,
            channel_valid=ch.get("valid"), channel_error=ch.get("error"))
        print(f"TIER {tier}: {summary[tier]}")
    bV, bG = summary.get("V", {}).get("identity_bits"), summary.get("G", {}).get("identity_bits")
    if bV is not None and bG is not None and bG > 0:
        summary["recovered_bits_seam"] = round((bG - bV) / bG, 3)
        print(f"recovered-bits seam (G-V)/G = {summary['recovered_bits_seam']}")
    suffix = "" if a.distractor == "hard" else f"_{a.distractor}"
    json.dump({"task": a.task, "distractor": a.distractor, "aids": aid_list, "summary": summary,
               "rows": {t: rows for t, rows in results.items()}},
              open(OUT / f"recovery_f_{a.task}{suffix}.json", "w"), indent=1, default=float)
    print(f"-> seed_g_extend/recovery_f_{a.task}{suffix}.json")


if __name__ == "__main__":
    main()
