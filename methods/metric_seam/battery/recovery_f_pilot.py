"""S2 pilot — recovery-objective seam width: V/A in recovered BITS, not rho-vs-judge.

Uses the canonical reconstruction-MCQ channel (methods/metric_implementer/recon_channel.py,
precomputed-behavior API — arbitrary per-item score columns, judge-free at value time).

Per criterion (legal_title_vii, the 17 post-degeneracy-guard aids), two candidate tiers:
  V : the certified h0 program run with an EMPTY LLM-field map (pure code+ops) on TEST
  G : the seed-G single-prompt column (Gemma-4-31B) on TEST
Each tier's column is median-binarized (identical rule across tiers). The GLM-5.2
reconstructor sees 8 (text, score) teaching examples drawn from the tier's own column and
must identify WHICH criterion (4-option codebook: target + 3 hard same-tier distractors,
clones excluded) produced it. Controls (no-demo, shuffled) run per call.

Aggregate per tier with mcq_identity_channel -> I(J; Jhat) in bits over the criterion set:
  bits_V = how much criterion-identity the code tier's behavior carries
  bits_G = same for the language tier;   recovered-bits seam = (bits_G - bits_V) / bits_G

Usage: python3 recovery_f_pilot.py [--limit N]   (GLM API only; no GPU)
Writes seed_g_extend/recovery_f_legal_pilot.json
"""
import argparse, json, pathlib, sys
from collections import Counter

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

bc.PROGDIR.setdefault("legal_title_vii", "programs_legal")
TASK = "legal_title_vii"
NOUN = "U.S. Title VII employment-discrimination court opinion"
BASE = ROOT / "outputs/metric_seam_pilot"
OUT = HERE / "seed_g_extend"

# the 17 post-guard criteria (seed_g_legal_title_vii_final.json rows)
AIDS = [r["aid"] for r in json.load(open(OUT / "seed_g_legal_title_vii_final.json"))["rows"]]


def binarize(col, ids):
    vals = [col.get(d) for d in ids]
    if any(v is None for v in vals):
        return None
    med = float(np.median(vals))
    b = np.array([1.0 if v > med else 0.0 for v in vals])
    minority = min(int(b.sum()), len(b) - int(b.sum()))
    return b if minority >= 5 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="limit #criteria (debug)")
    ap.add_argument("--model", default="glm-5.2")
    a = ap.parse_args()

    ctx = bc.load_ctx(TASK)
    test_ids = sorted(ctx["test"])
    texts = [ctx["items"][d] for d in test_ids]

    # --- tier columns ---
    pdir = ROOT / "methods/metric_seam/hybrids/programs_legal"
    cols = {"V": {}, "G": {}}
    for aid in AIDS:
        mod = load_mod(pdir / f"{aid}_h0.py")
        cols["V"][aid] = run_prog(mod.score, ctx["items"], {}, ctx["ops"])
    for line in open(OUT / "seed_g_legal_title_vii_results.jsonl"):
        r = json.loads(line)
        if r.get("aspect_id", "").endswith(".final") and isinstance(r.get("score"), int):
            cols["G"].setdefault(r["aspect_id"].split(".")[1], {})[r["datapoint_id"]] = r["score"]

    packs = {aid: json.load(open(BASE / f"tasks/{TASK}/improver_packs/{aid}.json"))
             for aid in AIDS}
    desc = {aid: f"{packs[aid]['criterion_name']} — {packs[aid]['criterion_description'][:400]}"
            for aid in AIDS}

    # binarized panels per tier (identical rule)
    panel = {t: {} for t in cols}
    for t in cols:
        for aid in AIDS:
            b = binarize(cols[t].get(aid, {}), test_ids)
            if b is not None:
                panel[t][aid] = b
    print({t: f"{len(panel[t])}/{len(AIDS)} usable binarized columns" for t in panel})

    cfg = ImplementerConfig()
    cfg.backend = "zai_anthropic"
    cfg.other_temperature = 0.0
    cfg.request_timeout_s = 180
    recon = LLMBackend(model=a.model, role="reconstructor", cfg=cfg)

    rng = np.random.default_rng(13)
    design = np.sort(rng.choice(len(test_ids), size=24, replace=False))

    results = {t: [] for t in panel}
    aid_list = AIDS[: a.limit] if a.limit else AIDS
    for tier in panel:
        for aid in aid_list:
            if aid not in panel[tier]:
                continue
            target = panel[tier][aid]
            # hard distractors: same-tier columns, highest agreement, clones (>=.95) excluded
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
            distractors = [dict(metric_id=o, description=desc[o], scores=panel[tier][o])
                           for _, o in sims[:3]]
            try:
                row = mcq_value_from_precomputed_behavior(
                    recon, noun=NOUN, candidate_prompt_text=f"{tier}:{aid}",
                    target_metric_id=aid, target_description=desc[aid],
                    target_scores=target, probe_texts=texts, distractors=distractors,
                    design_indices=design, codebook_frozen_before_prompt_search=True,
                    n_examples=8, n_reconstruction_draws=4)
                results[tier].append(row)
                print(f"{tier}/{aid}: raw_p={row['raw_target_option_probability']:.2f} "
                      f"value={row['value_mark']:.2f}")
            except Exception as e:
                print(f"FAIL {tier}/{aid}: {type(e).__name__}: {str(e)[:120]}")

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
    json.dump({"task": TASK, "aids": aid_list, "summary": summary,
               "rows": {t: rows for t, rows in results.items()}},
              open(OUT / "recovery_f_legal_pilot.json", "w"), indent=1, default=float)
    print("-> seed_g_extend/recovery_f_legal_pilot.json")


if __name__ == "__main__":
    main()
