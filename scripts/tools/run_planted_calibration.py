#!/usr/bin/env python3
"""Planted-bank whole-pipeline calibration (MCC §7): on REAL texts with a SYNTHETIC label
generated from known code-computable features, the pipeline must
  (a) RECOVER V -> its known value when given rubric versions of the planted features,
  (b) FIND the withheld planted feature through the generator arms + full gate
      (confirm stage included) — the planted-positive control at real-LLM fidelity,
  (c) STAY SILENT afterwards — the residual is provably pure Bernoulli noise, so any further
      acceptance is a measured false-positive rate of the full protocol.
Real labels are never used; y is synthetic by construction.

  python scripts/tools/run_planted_calibration.py --task creative-writing \
      --n 500 --max-rounds 20 --acceptance-eval cv --judge-backend vllm_offline \
      --judge-model meta-llama/Llama-3.3-70B-Instruct
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "methods")

import numpy as np
import pandas as pd

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.feature_gen import make_proposer
from metrics_tree_infilling.flux import flux_from_ledgers
from metrics_tree_infilling.generators import (
    autometrics_iterative_generator, label_contrast_generator, residual_generator)
from metrics_tree_infilling.global_infill import run_global_infill
from metrics_tree_infilling.io_metrics import (
    REPO_ROOT, MetricSpec, load_rubric_metrics, load_rubric_metrics_from_dir,
    make_vllm_judge_scorer, materialize, three_way_split)
from metrics_tree_infilling.run import DATASET_CONFIGS
from sklearn.metrics import roc_auc_score

# planted features: code-computable ground truth, content-level, reliably LLM-judgeable
PLANTED = {
    "dialogue_present": {
        "fn": lambda t: int(('"' in t) or ("“" in t) or ("said" in t.lower())),
        "beta": 1.2,
        "rubric": "YES if the text contains spoken dialogue (quoted speech or an explicit "
                  "speech verb like 'said'); NO otherwise.",
    },
    "first_person_narration": {
        "fn": lambda t: int(len(re.findall(r"\bI\b", t[:1500])) >= 2),
        "beta": 1.0,
        "rubric": "YES if the text is narrated in the first person (the narrator refers to "
                  "themselves as 'I' repeatedly); NO otherwise.",
    },
    "question_posed": {
        "fn": lambda t: int("?" in t),
        "beta": 0.8,
        "rubric": "YES if the text poses at least one direct question (contains a question "
                  "mark); NO otherwise.",
    },
    # the WITHHELD one — arms must rediscover it
    "numeral_specificity": {
        "fn": lambda t: int(bool(re.search(r"\d", t))),
        "beta": 1.4,
        "rubric": "YES if the text mentions at least one specific number, date, or quantity "
                  "written in digits; NO otherwise.",
    },
}
WITHHELD = "numeral_specificity"


def _h(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def exact_mi(F: np.ndarray, probs: np.ndarray) -> float:
    """I(Y; F) in bits, exact given the generative model: cells of F x known p(y=1|cell)."""
    df = pd.DataFrame(F)
    mi = _h(probs.mean())
    for _, idx in df.groupby(list(df.columns)).groups.items():
        idx = np.asarray(idx)
        mi -= (len(idx) / len(F)) * _h(probs[idx].mean())
    return float(mi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative-writing", choices=sorted(DATASET_CONFIGS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-distractors", type=int, default=10)
    ap.add_argument("--max-rounds", type=int, default=20)
    ap.add_argument("--arms", default="residual,label_contrast,autometrics_iterative")
    ap.add_argument("--min-auc-gain", type=float, default=0.02)
    ap.add_argument("--min-bits-gain", type=float, default=0.01)
    ap.add_argument("--acceptance-eval", default="cv", choices=["guard", "cv"])
    ap.add_argument("--confirm-repeats", type=int, default=5)
    ap.add_argument("--judge-backend", default="anthropic",
                    choices=["anthropic", "vllm_offline", "openai_compatible"])
    ap.add_argument("--judge-model", default="glm-5.2")
    ap.add_argument("--proposer-backend", default="anthropic",
                    choices=["anthropic", "vllm_offline"])
    ap.add_argument("--proposer-model", default="glm-5.2")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="outputs/ctree/planted_calibration")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dcfg = DATASET_CONFIGS[args.task]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    # -- corpus: real texts, synthetic y ------------------------------------------------
    df = pd.read_csv(REPO_ROOT / dcfg["split"], low_memory=False).dropna(subset=[dcfg["text"]])
    df = df.sample(min(args.n + args.n // 2, len(df)), random_state=args.seed).reset_index(drop=True)
    texts = df[dcfg["text"]].astype(str).tolist()
    names = list(PLANTED)
    # apples-to-apples: ground truth must live in the JUDGE'S VIEW of the text — the scorer
    # truncates to max_text_tokens*4 chars, so a feature beyond that boundary would be truth
    # the judge provably cannot see (caught on the 2026-07-05 first run: full-text truth vs
    # truncated judge view depressed fidelity)
    judge_view = 700 * 4                       # keep in lockstep with cfg.max_text_tokens below
    F = np.column_stack([[PLANTED[k]["fn"](t[:judge_view]) for t in texts] for k in names])
    betas = np.array([PLANTED[k]["beta"] for k in names])
    logit = F @ betas - (F.mean(0) @ betas)          # center -> base rate ~ 0.5
    probs = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=len(texts)) < probs).astype(int)
    df = pd.DataFrame({"id": np.arange(len(texts)).astype(str), "text": texts, "judgement": y})
    # carry ground truth INSIDE the frame: three_way_split resets indices, so any
    # position-based F lookup after the split is silently misaligned (bit us 2026-07-05:
    # fidelity ~0.5 on question-mark detection was row misalignment, not judge failure)
    for j, k in enumerate(names):
        df[f"_truth_{k}"] = F[:, j]

    # eyeball guard (validate-before-scaling): feature rates + a sample per feature
    rates = {k: float(F[:, j].mean()) for j, k in enumerate(names)}
    print(f"feature rates: {rates}  base rate: {y.mean():.3f}", flush=True)
    for j, k in enumerate(names):
        i = int(np.argmax(F[:, j]))
        print(f"  sample [{k}=1]: {texts[i][:110]!r}", flush=True)
    v_star_all = exact_mi(F, probs)
    v_star_bank = exact_mi(F[:, [j for j, k in enumerate(names) if k != WITHHELD]], probs)
    print(f"V* (all 4 planted) = {v_star_all:.4f} bits;  V* (bank of 3) = {v_star_bank:.4f}",
          flush=True)

    cfg = InfillConfig(
        random_seed=args.seed, proposer_backend=args.proposer_backend,
        proposer_model=args.proposer_model,
        materialize_backend=args.judge_backend, materialize_model=args.judge_model,
        llm_concurrency=args.concurrency, max_text_tokens=700, verbose=False,
        min_auc_gain=args.min_auc_gain, min_bits_gain=args.min_bits_gain,
        acceptance_eval=args.acceptance_eval,
        confirm_n_repeats=args.confirm_repeats, gate_alpha=0.05,
        gate_bonferroni_m=len(arms) * args.max_rounds,
        id_column="id", text_column="text", label_column="judgement",
        output_dir=str(out), cache_dir=str(out / "judge_cache"),
        curated_z_only=True, include_text_length_in_z=False)

    # -- bank: 3 planted rubrics (withhold one) + real-bank distractors ------------------
    bank = [MetricSpec(metric_id=f"planted_{k}", name=k, description=PLANTED[k]["rubric"],
                       kind="judge", guidance=PLANTED[k]["rubric"])
            for k in names if k != WITHHELD]
    medoid_dir = REPO_ROOT / f"datasets/{args.task}/medoid-bank"
    distractors = (load_rubric_metrics_from_dir(medoid_dir) if medoid_dir.exists()
                   else load_rubric_metrics(args.task, limit=args.n_distractors))
    bank = bank + distractors[: args.n_distractors]

    judge = make_vllm_judge_scorer(cfg)
    if args.proposer_backend == "vllm_offline":
        from metrics_tree_infilling.io_metrics import make_offline_vllm_proposer
        proposer = make_offline_vllm_proposer(cfg)
    else:
        proposer = make_proposer(cfg)
    df_d, df_g, df_t = three_way_split(df, cfg)
    print(f"rows d/g/t = {len(df_d)}/{len(df_g)}/{len(df_t)}", flush=True)
    sm_d = materialize(bank, df_d, cfg, judge)
    sm_g = materialize(bank, df_g, cfg, judge)
    y_d = df_d["judgement"].to_numpy(); y_g = df_g["judgement"].to_numpy()
    texts_d = df_d["text"].astype(str).tolist()

    # (a0) judge fidelity on the planted rubrics: judge levels vs code truth (built-in anchors)
    fidelity = {}
    for j, m in enumerate(bank[:len(names) - 1]):
        truth = df_d[f"_truth_{m.name}"].to_numpy()
        lv = sm_d.levels[:, j]; ap_ = sm_d.applicable[:, j] & np.isfinite(lv)
        fidelity[m.name] = (float(roc_auc_score(truth[ap_], lv[ap_]))
                            if ap_.sum() > 20 and len(np.unique(truth[ap_])) > 1 else None)
    print(f"judge fidelity (AUC vs code truth): {fidelity}", flush=True)

    arm_factories = {
        "residual": lambda: residual_generator(),
        "label_contrast": lambda: label_contrast_generator(texts_d, y_d, seed=args.seed),
        "autometrics_iterative": lambda: autometrics_iterative_generator(
            "judging short texts against an unknown editorial standard", k=4),
    }
    truth_withheld_d = df_d[f"_truth_{WITHHELD}"].to_numpy()

    results = {}
    for arm in arms:
        print(f"\n=== ARM {arm} ===", flush=True)
        res = run_global_infill(
            sm_d, df_d, y_d, sm_g, df_g, y_g, list(bank), cfg,
            judge_scorer=judge, proposer=proposer,
            max_rounds=args.max_rounds, patience=args.max_rounds,
            measure_reconstruction=False, proposal_fn=arm_factories[arm]())
        (out / arm).mkdir(exist_ok=True, parents=True)
        res.save(out / arm)
        kept = [l for l in res.ledgers if l.status == "kept"]
        # did any accepted metric rediscover the withheld feature? score it on discover truth
        found, false_pos = [], []
        for l in kept:
            spec = next((m for m in res.metrics if m.name == l.name), None)
            if spec is None:
                continue
            from metrics_tree_infilling.loop import _score_one
            lv, ap_ = _score_one(spec, texts_d, judge)
            mmask = ap_ & np.isfinite(lv)
            auc = (float(roc_auc_score(truth_withheld_d[mmask], lv[mmask]))
                   if mmask.sum() > 20 and len(np.unique(truth_withheld_d[mmask])) > 1
                   else float("nan"))
            (found if np.isfinite(auc) and max(auc, 1 - auc) >= 0.80 else false_pos).append(
                {"name": l.name, "auc_vs_withheld": auc, "bits_gain": l.bits_gain})
        results[arm] = {
            "proposals": len(res.ledgers), "kept": len(kept),
            "found_withheld": found, "false_positives": false_pos,
            "bits_trajectory": res.guard_bits_trajectory,
            "statuses": [l.status for l in res.ledgers],
        }
        print(json.dumps(results[arm], indent=2, default=float), flush=True)

    ledger_paths = [out / a / "global_infill_ledger.json" for a in arms
                    if (out / a / "global_infill_ledger.json").exists()]
    fx = None
    if ledger_paths:
        try:
            fx = flux_from_ledgers(ledger_paths, base_rate=float(y.mean()))
        except Exception as e:
            print(f"flux read failed: {e}", flush=True)

    report = {
        "task_texts": args.task, "n": len(df), "base_rate": float(y.mean()),
        "planted_feature_rates": rates,
        "v_star_all_bits": v_star_all, "v_star_bank_bits": v_star_bank,
        "judge_fidelity_auc": fidelity,
        "bank_v_bits_start": results[arms[0]]["bits_trajectory"][0] if arms else None,
        "arms": results,
        "withheld": WITHHELD,
        "positive_control_passed": any(r["found_withheld"] for r in results.values()),
        "false_acceptances_total": sum(len(r["false_positives"]) for r in results.values()),
        "flux": fx,
    }
    with open(out / "planted_calibration.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n== CALIBRATION VERDICT ==\n"
          f"V recovery: bank V_bits {report['bank_v_bits_start']} vs V* {v_star_bank:.4f}\n"
          f"positive control (found {WITHHELD}): {report['positive_control_passed']}\n"
          f"false acceptances: {report['false_acceptances_total']}\n"
          f"-> {out/'planted_calibration.json'}", flush=True)


if __name__ == "__main__":
    main()
